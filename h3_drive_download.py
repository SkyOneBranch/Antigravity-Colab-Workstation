"""Noninteractive Google Drive download for the H3 production path.

TASK-006 P0-B (Issue #44): the paid path must not depend on interactive
``colab drivemount``. This module downloads pinned model files (and the
ComfyUI source archive) from Drive with a short-lived, read-only access
token, using only the standard library so the remote Colab worker can
import it with no extra dependency.

v0.2.0 F1 upgrades:
- Zero-copy ``os.truncate`` + lock-free multi-threaded ``os.pwrite`` into
  a single ``.part`` file (peak disk <= 1.0x, zero shard merge).
- High-concurrency default (``DEFAULT_CONCURRENT_WORKERS = 12``,
  configurable via ``H3_DRIVE_CONCURRENT_WORKERS`` in ``1..16``) with a
  global connection semaphore cap to prevent Drive HTTP 429 bursts.
- Streaming SHA-256 during chunk reception via an ordered sliding-window
  block hasher (verifying full-file SHA-256 without re-reading ``.part``
  from disk).
- ``.part.bitmap.json`` sidecar for Range-block resume on interruption
  (with legacy linear ``.part`` Range fallback) and cooperative
  ``abort_event`` cancellation for dual-track workers.

Security rules (hard):
- This module never holds, mints, or stores a refresh token, ADC, or
  any long-lived credential. It only ever uses an access token handed
  to it through a token provider.
- A token value is NEVER written to a log, manifest, audit record,
  exception message, or return value. Only ``token_id`` (the first 16
  hex chars of SHA-256(token)) may appear in diagnostics; it is a
  non-sensitive identifier for correlating mint/renewal events.
- A token file is read and then deleted immediately by the reader
  (``read_token_file``), except for the explicit ``delete=False`` probe
  path, which is documented at its call site.
- Tokens are never passed on the command line (``/proc/<pid>/cmdline``
  is world-readable on the VM).

Download rules:
- Pinned identity: Drive file id + expected byte count + expected
  SHA-256 (and MD5 when the server reports one). The server metadata
  is re-checked before AND after the transfer; a file that changed
  mid-transfer fails instead of being silently accepted.
- HTTP Range resume with a ``.part`` file (and ``.part.bitmap.json``
  sidecar for concurrent block resume), atomic replace on success,
  full SHA-256 verification before the file is ever handed to a model
  loader. A ``.part`` file is never a verified model.
- Error classification: 429/5xx are transient (bounded backoff);
  401 triggers exactly one token refresh, then fails closed;
  403 fails closed immediately; other 4xx fail immediately.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Rebindable for tests: the fake transport patches h3_drive_download.urlopen.
urlopen = urllib.request.urlopen

DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"
READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"

# Token lifetime margin: ask for a renewal while this much life remains.
TOKEN_REFRESH_MARGIN_SEC = 600
# Hard cap on renewals per run; the controller enforces the same cap.
MAX_TOKEN_RENEWALS = 6

DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB streaming writes (TASK-007)
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BACKOFF_BASE_SEC = 2.0
DEFAULT_TIMEOUT_SEC = 300
# Files at or above this size get a concurrent pwrite Range download
# (TASK-007 / v0.2.0 F1): a 40 GiB restore over one TCP stream is the
# single biggest cold-start risk against a ~60-minute session budget.
CONCURRENT_MIN_SIZE = 256 * 1024 * 1024
MIN_CONCURRENT_WORKERS = 1
MAX_CONCURRENT_WORKERS = 16
DEFAULT_CONCURRENT_WORKERS = 12
ENV_CONCURRENT_WORKERS = "H3_DRIVE_CONCURRENT_WORKERS"

DEFAULT_MAX_GLOBAL_CONNECTIONS = 16
ENV_MAX_GLOBAL_CONNECTIONS = "H3_DRIVE_MAX_GLOBAL_CONNECTIONS"

# Fixed block size for concurrent sliding-window Range requests and
# streaming SHA-256 aggregation (32 MiB balances HTTP setup overhead
# against reorder-buffer memory).
DEFAULT_RANGE_BLOCK_SIZE = 32 * 1024 * 1024
# Maximum RAM reserved for out-of-order blocks waiting on the SHA-256
# frontier before spilling to selective os.pread.
DEFAULT_MAX_HASH_BUFFER_BYTES = 256 * 1024 * 1024

# A slow-but-alive download is not killed; the read timeout only bounds
# a single stalled read inside urlopen.
TRANSIENT_STATUSES = (429, 500, 502, 503, 504)

_BITMAP_VERSION = 1

_GLOBAL_CONN_LOCK = threading.Lock()
_GLOBAL_CONN_SEMAPHORE: threading.BoundedSemaphore | None = None
_GLOBAL_CONN_LIMIT: int | None = None


class DriveDownloadError(RuntimeError):
    """Base failure for a Drive download. Never carries a token value."""


class DriveAuthError(DriveDownloadError):
    """401 after a refresh, or 403: fail closed, never retry the token."""


class DriveChangedError(DriveDownloadError):
    """Server metadata changed between the pre/post checks: the remote
    file is not the pinned file any more. Never accept the bytes."""


class DriveRetryExhaustedError(DriveDownloadError):
    """Transient failures outlived the bounded retry budget."""


class DriveDownloadAbortedError(DriveDownloadError):
    """Download stopped promptly because abort_event was set."""


class _NoRangeSupport(Exception):
    """Internal: a ranged request was answered 200, so the server does
    not honour Range for this file. Signals the caller to fall back
    to the single-stream path (never an error by itself)."""


class _ProgressAborted(Exception):
    """Internal: a worker stopped early because the progress callback
    raised (e.g. the worker's budget projection). The recorded
    callback error is what surfaces, never this marker."""


class TokenUnavailable(RuntimeError):
    """The token provider has no usable access token right now."""


def resolve_concurrent_workers(explicit: int | None = None) -> int:
    """Resolve concurrent Range worker count in ``1..16``.

    When ``explicit`` is provided, validates ``1 <= explicit <= 16``.
    Otherwise reads ``H3_DRIVE_CONCURRENT_WORKERS`` (clamped to
    ``1..16``) and falls back to ``DEFAULT_CONCURRENT_WORKERS`` (12).
    """
    if explicit is not None:
        val = int(explicit)
        if val < MIN_CONCURRENT_WORKERS or val > MAX_CONCURRENT_WORKERS:
            raise ValueError(
                f"concurrent_workers must be between {MIN_CONCURRENT_WORKERS} "
                f"and {MAX_CONCURRENT_WORKERS}, got {val}")
        return val
    raw = os.environ.get(ENV_CONCURRENT_WORKERS, "").strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            return DEFAULT_CONCURRENT_WORKERS
        return max(MIN_CONCURRENT_WORKERS, min(MAX_CONCURRENT_WORKERS, parsed))
    return DEFAULT_CONCURRENT_WORKERS


def resolve_max_global_connections(explicit: int | None = None) -> int:
    """Resolve global concurrent Drive connection cap (>= 1)."""
    if explicit is not None:
        val = int(explicit)
        if val < 1:
            raise ValueError("max_global_connections must be >= 1")
        return val
    raw = os.environ.get(ENV_MAX_GLOBAL_CONNECTIONS, "").strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            return DEFAULT_MAX_GLOBAL_CONNECTIONS
        return max(1, parsed)
    return DEFAULT_MAX_GLOBAL_CONNECTIONS


def get_global_connection_semaphore(
        limit: int | None = None) -> threading.BoundedSemaphore:
    """Return the process-wide Drive connection semaphore.

    Shared across ``DriveDownloader`` instances (e.g. F3 dual-track
    ``ensure_comfy`` + ``restore_weights``) so total in-flight media
    connections stay bounded and avoid Drive 429 throttling.
    """
    global _GLOBAL_CONN_SEMAPHORE, _GLOBAL_CONN_LIMIT
    target = resolve_max_global_connections(limit)
    with _GLOBAL_CONN_LOCK:
        if _GLOBAL_CONN_SEMAPHORE is None or (
                limit is not None and _GLOBAL_CONN_LIMIT != target):
            _GLOBAL_CONN_SEMAPHORE = threading.BoundedSemaphore(target)
            _GLOBAL_CONN_LIMIT = target
        return _GLOBAL_CONN_SEMAPHORE


def token_id_for(token: str) -> str:
    """Non-sensitive identifier for a token value (audit correlation
    only). The id cannot be used to reconstruct or use the token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _parse_utc(value: str) -> float:
    """Parse an ISO-8601 UTC timestamp to epoch seconds."""
    from datetime import datetime, timezone

    text = (value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).astimezone(timezone.utc).timestamp()


def validate_token_doc(doc: dict) -> dict:
    """Shape-check a token document. Returns the doc unchanged.
    Raises ValueError on a bad shape. Never logs the token value."""
    if not isinstance(doc, dict):
        raise ValueError("token document is not an object")
    token = doc.get("access_token")
    if not isinstance(token, str) or not token:
        raise ValueError("token document has no access_token")
    expires = doc.get("expires_at_utc")
    if not isinstance(expires, str) or not expires:
        raise ValueError("token document has no expires_at_utc")
    _parse_utc(expires)  # raises ValueError on a bad timestamp
    token_id = doc.get("token_id")
    if not isinstance(token_id, str) or not token_id:
        raise ValueError("token document has no token_id")
    if token_id != token_id_for(token):
        raise ValueError("token document token_id does not match its token")
    return doc


def read_token_file(path, delete: bool = True) -> dict:
    """Read a 0600 token file and delete it immediately after a
    successful read (default). The caller owns the lifetime decision:
    the download probe passes delete=False so the worker's own token
    file survives the probe; the worker always reads with delete=True.
    Raises FileNotFoundError when no token file is present (the
    controller has not delivered one yet) and ValueError on a bad
    shape. Never logs the token value."""
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    validate_token_doc(doc)
    if delete:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    return doc


class FileTokenProvider:
    """Thread-safe access-token holder for the remote worker.

    The token is loaded from a 0600 token file delivered by the
    controller (``colab upload``). ``get_token()`` returns the current
    token while it has more than ``margin_sec`` of life left; when it
    is about to expire it tries to pick up a freshly delivered token
    file at the same path (the controller uploads one after the worker
    requests renewal through its job state). ``refresh()`` forces a
    re-read of the token file (used exactly once after a 401), and is
    single-flight/idempotent when ``failed_token_id`` is passed so
    concurrent 401s on the same expired token share one file read.
    The provider never logs a token value.
    """

    def __init__(self, token_path, margin_sec: int = TOKEN_REFRESH_MARGIN_SEC):
        self.token_path = Path(token_path)
        self.margin_sec = margin_sec
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires_at: float | None = None
        self._token_id: str | None = None

    @property
    def token_id(self) -> str | None:
        with self._lock:
            return self._token_id

    def _load_file_locked(self) -> bool:
        try:
            doc = read_token_file(self.token_path, delete=True)
        except FileNotFoundError:
            return False
        self._token = doc["access_token"]
        self._expires_at = _parse_utc(doc["expires_at_utc"])
        self._token_id = doc["token_id"]
        return True

    def load_initial(self) -> dict:
        """Load the first token file. Raises TokenUnavailable when no
        token file has been delivered yet."""
        with self._lock:
            if not self._load_file_locked():
                raise TokenUnavailable(
                    "no Drive token file delivered at " + str(self.token_path))
            return {"token_id": self._token_id, "expires_at": self._expires_at}

    def get_token(self) -> str:
        """Current usable token, or raise TokenUnavailable. Picks up a
        freshly delivered token file when the current one is inside the
        renewal margin."""
        with self._lock:
            now = time.time()
            if self._token is not None and self._expires_at is not None:
                if self._expires_at - now > self.margin_sec:
                    return self._token
            # Inside the margin (or no token yet): try a fresh file.
            if self._load_file_locked():
                return self._token
            if self._token is not None and self._expires_at is not None \
                    and self._expires_at > now:
                # No fresh file, but the old token is still alive: use it
                # once more rather than failing the in-flight chunk.
                return self._token
            raise TokenUnavailable("no usable Drive access token")

    def refresh(self, failed_token_id: str | None = None) -> str:
        """Force a re-read of the token file (after a 401).

        When ``failed_token_id`` is supplied and another thread has
        already rotated past that token to a still-valid token, returns
        the already-refreshed token without re-reading the consumed
        file (single-flight safety for concurrent dual-track 401s).
        Raises TokenUnavailable when no fresh token file is present.
        """
        with self._lock:
            now = time.time()
            if (failed_token_id is not None
                    and self._token is not None
                    and self._token_id is not None
                    and self._token_id != failed_token_id
                    and (self._expires_at is None or self._expires_at > now)):
                return self._token
            if self._load_file_locked():
                return self._token
            raise TokenUnavailable("no fresh Drive token file for refresh")

    def seconds_remaining(self) -> float | None:
        with self._lock:
            if self._expires_at is None:
                return None
            return self._expires_at - time.time()


def _default_sleep(seconds: float) -> None:
    time.sleep(seconds)


# Rebindable for tests: unit tests patch h3_drive_download._sleep.
_sleep = _default_sleep


def _backoff_delay(attempt: int, base: float) -> float:
    # Exponential backoff with jitter; the attempt number is 1-based.
    return base * (2 ** (attempt - 1)) + random.uniform(0, 1.0)


def _check_abort(abort_event, file_id: str) -> None:
    if abort_event is not None and abort_event.is_set():
        raise DriveDownloadAbortedError(
            f"Drive download aborted for {file_id}")


def _sleep_or_abort(seconds: float, abort_event, file_id: str) -> None:
    _check_abort(abort_event, file_id)
    if seconds <= 0:
        return
    if abort_event is not None and _sleep is _default_sleep:
        if abort_event.wait(timeout=seconds):
            raise DriveDownloadAbortedError(
                f"Drive download aborted for {file_id}")
        return
    _sleep(seconds)
    _check_abort(abort_event, file_id)


class _CompositeAbortEvent:
    """Read-only Event-like view that is set when any parent event is set."""

    def __init__(self, first: threading.Event, second: threading.Event):
        self._first = first
        self._second = second

    def is_set(self) -> bool:
        return self._first.is_set() or self._second.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        if self.is_set():
            return True
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self.is_set()
                step = min(0.05, remaining)
            else:
                step = 0.05
            self._first.wait(timeout=step)
            if self.is_set():
                return True
        return True


def _pwrite_all(fd: int, data: bytes | bytearray | memoryview,
                offset: int) -> None:
    """Write ``data`` completely at ``offset`` via lock-free ``os.pwrite``."""
    mv = memoryview(data)
    total = len(mv)
    written = 0
    while written < total:
        n = os.pwrite(fd, mv[written:], offset + written)
        if n <= 0:
            raise OSError("os.pwrite returned 0 bytes")
        written += n


def _pread_into_hasher(fd: int, offset: int, length: int, hasher,
                       chunk_size: int = DEFAULT_CHUNK_SIZE) -> int:
    """Feed ``length`` bytes from ``fd`` at ``offset`` into ``hasher``."""
    remaining = length
    pos = offset
    read_total = 0
    while remaining > 0:
        to_read = min(chunk_size, remaining)
        chunk = os.pread(fd, to_read, pos)
        if not chunk:
            raise OSError(f"os.pread unexpected EOF at offset {pos}")
        hasher.update(chunk)
        got = len(chunk)
        pos += got
        remaining -= got
        read_total += got
    return read_total


def _bitmap_path(part: Path) -> Path:
    return Path(str(part) + ".bitmap.json")


def _load_bitmap(bitmap_path: Path, file_id: str, expected_size: int,
                 expected_sha256: str) -> dict | None:
    """Load and validate a ``.part.bitmap.json`` sidecar, or return None."""
    if not bitmap_path.is_file():
        return None
    try:
        doc = json.loads(bitmap_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(doc, dict):
        return None
    if doc.get("version") != _BITMAP_VERSION:
        return None
    if doc.get("file_id") != file_id:
        return None
    if doc.get("expected_size") != expected_size:
        return None
    if (doc.get("expected_sha256") or "").lower() != expected_sha256.lower():
        return None
    block_size = doc.get("block_size")
    if not isinstance(block_size, int) or block_size <= 0:
        return None
    expected_blocks = (expected_size + block_size - 1) // block_size
    if doc.get("total_blocks") != expected_blocks:
        return None
    completed = doc.get("completed")
    if not isinstance(completed, list):
        return None
    valid_completed = []
    for idx in completed:
        if not isinstance(idx, int) or idx < 0 or idx >= expected_blocks:
            return None
        valid_completed.append(idx)
    return {
        "version": _BITMAP_VERSION,
        "file_id": file_id,
        "expected_size": expected_size,
        "expected_sha256": expected_sha256.lower(),
        "block_size": block_size,
        "total_blocks": expected_blocks,
        "completed": sorted(set(valid_completed)),
    }


def _save_bitmap(bitmap_path: Path, doc: dict) -> None:
    """Atomically write the ``.part.bitmap.json`` sidecar."""
    tmp = Path(str(bitmap_path) + ".tmp")
    tmp.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, bitmap_path)


class _StreamingBlockHasher:
    """Ordered frontier SHA-256 hasher for concurrent Range blocks.

    SHA-256 is sequential, so blocks must be fed in index order ``0..M-1``.
    Workers download blocks in ascending order and submit completed block
    bytes directly from memory; out-of-order completions buffer up to
    ``max_buffer_bytes`` and spill to ``None`` (selective ``os.pread``
    only when the frontier catches up). Resumed blocks are marked as
    ``None`` so only previously-on-disk blocks are read back.
    """

    def __init__(self, ranges: list[tuple[int, int]], fd: int,
                 max_buffer_bytes: int = DEFAULT_MAX_HASH_BUFFER_BYTES,
                 pread_chunk_size: int = DEFAULT_CHUNK_SIZE):
        self._ranges = ranges
        self._total_blocks = len(ranges)
        self._fd = fd
        self._max_buffer_bytes = max(0, int(max_buffer_bytes))
        self._pread_chunk_size = max(1, int(pread_chunk_size))
        self._lock = threading.Lock()
        self._hasher = hashlib.sha256()
        self._next_idx = 0
        self._pending: dict[int, bytes | None] = {}
        self._buffered_bytes = 0
        self.re_read_bytes = 0

    def _hash_from_fd_locked(self, idx: int) -> None:
        start, end = self._ranges[idx]
        length = end - start + 1
        self.re_read_bytes += _pread_into_hasher(
            self._fd, start, length, self._hasher, self._pread_chunk_size)

    def _drain_locked(self) -> None:
        while self._next_idx in self._pending:
            buf = self._pending.pop(self._next_idx)
            if buf is not None:
                self._buffered_bytes -= len(buf)
                self._hasher.update(buf)
            else:
                self._hash_from_fd_locked(self._next_idx)
            self._next_idx += 1

    def mark_resumed(self, idx: int) -> None:
        """Record a block that already exists on disk from a prior run."""
        with self._lock:
            if idx < self._next_idx or idx in self._pending:
                return
            self._pending[idx] = None
            self._drain_locked()

    def submit_block(self, idx: int, data: bytes | None) -> None:
        """Submit completed block bytes (or None if already spilled)."""
        with self._lock:
            if idx < self._next_idx:
                return
            if idx == self._next_idx:
                if data is not None:
                    self._hasher.update(data)
                else:
                    self._hash_from_fd_locked(idx)
                self._next_idx += 1
                self._drain_locked()
                return
            if data is not None and (
                    self._buffered_bytes + len(data) <= self._max_buffer_bytes):
                self._pending[idx] = data
                self._buffered_bytes += len(data)
            else:
                self._pending[idx] = None

    def hexdigest(self) -> str:
        with self._lock:
            self._drain_locked()
            if self._next_idx != self._total_blocks:
                raise DriveDownloadError(
                    f"Streaming hasher incomplete: {self._next_idx}/"
                    f"{self._total_blocks} blocks")
            return self._hasher.hexdigest()


class DriveDownloader:
    """Range-resuming Drive file downloader over a token provider.

    ``token_provider`` must expose ``get_token() -> str`` and
    ``refresh() -> str`` (see FileTokenProvider). ``on_progress`` is
    called as ``on_progress(bytes_done, total)`` after each chunk; the
    worker uses it to request a token renewal while life remains.
    ``on_token_expiring`` is called as ``on_token_expiring(token_id,
    seconds_remaining)`` when the provider reports less than the
    renewal margin during a download.
    ``concurrent_workers`` bounds the zero-copy ``pwrite`` Range pool
    for large files (``>= CONCURRENT_MIN_SIZE``, default 12 or
    ``H3_DRIVE_CONCURRENT_WORKERS`` in ``1..16``); ``max_workers``
    bounds multi-file ``download_all``; ``connection_semaphore`` (or
    ``max_global_connections``) caps total simultaneous Drive media
    connections across all files and threads.
    """

    def __init__(self, token_provider, chunk_size: int = DEFAULT_CHUNK_SIZE,
                 max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                 backoff_base_sec: float = DEFAULT_BACKOFF_BASE_SEC,
                 max_workers: int = 1,
                 concurrent_workers: int | None = None,
                 timeout_sec: int = DEFAULT_TIMEOUT_SEC,
                 max_global_connections: int | None = None,
                 connection_semaphore: threading.Semaphore | None = None,
                 abort_event: threading.Event | None = None,
                 range_block_size: int = DEFAULT_RANGE_BLOCK_SIZE,
                 max_hash_buffer_bytes: int = DEFAULT_MAX_HASH_BUFFER_BYTES):
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if range_block_size < 1:
            raise ValueError("range_block_size must be >= 1")
        if max_hash_buffer_bytes < 0:
            raise ValueError("max_hash_buffer_bytes must be >= 0")
        resolved_workers = resolve_concurrent_workers(concurrent_workers)
        if connection_semaphore is not None:
            sem = connection_semaphore
        elif max_global_connections is not None:
            sem = threading.BoundedSemaphore(
                resolve_max_global_connections(max_global_connections))
        else:
            sem = get_global_connection_semaphore()
        self.token_provider = token_provider
        self.chunk_size = chunk_size
        self.max_attempts = max_attempts
        self.backoff_base_sec = backoff_base_sec
        self.max_workers = max_workers
        self.concurrent_workers = resolved_workers
        self.timeout_sec = timeout_sec
        self._conn_sem = sem
        self.abort_event = abort_event
        self.range_block_size = int(range_block_size)
        self.max_hash_buffer_bytes = int(max_hash_buffer_bytes)
        self.last_re_read_bytes = 0

    def _effective_abort(self, abort_event: threading.Event | None):
        if self.abort_event is None:
            return abort_event
        if abort_event is None or abort_event is self.abort_event:
            return self.abort_event
        return _CompositeAbortEvent(self.abort_event, abort_event)

    @contextlib.contextmanager
    def _acquire_connection(self, abort_event, file_id: str):
        acquired = False
        try:
            while not acquired:
                _check_abort(abort_event, file_id)
                acquired = self._conn_sem.acquire(timeout=0.05)
            _check_abort(abort_event, file_id)
            yield
        finally:
            if acquired:
                self._conn_sem.release()

    def _refresh_token(self, failed_token: str | None = None) -> str:
        failed_tid = token_id_for(failed_token) if failed_token else None
        if failed_tid is not None:
            try:
                return self.token_provider.refresh(failed_token_id=failed_tid)
            except TypeError:
                pass
        return self.token_provider.refresh()

    # -- HTTP plumbing (stdlib only; urlopen is module-level for tests) --

    def _request(self, url: str, headers: dict | None = None,
                 method: str = "GET"):
        token = self.token_provider.get_token()
        req_headers = {"Authorization": "Bearer " + token}
        if headers:
            req_headers.update(headers)
        req = urllib.request.Request(url, headers=req_headers, method=method)
        return urlopen(req, timeout=self.timeout_sec)

    def get_metadata(self, file_id: str) -> dict:
        """Server-side metadata for one Drive file. Raises
        DriveDownloadError on any non-200 answer; a 404 may mean
        permissions rather than absence."""
        url = (f"{DRIVE_API_BASE}/files/{file_id}"
               "?fields=id,name,size,md5Checksum,sha256Checksum,trashed"
               "&supportsAllDrives=true")
        try:
            with self._request(url) as resp:
                status = getattr(resp, "status", 200)
                if status != 200:
                    raise DriveDownloadError(
                        f"Drive metadata HTTP {status} for file {file_id}")
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise DriveDownloadError(
                f"Drive metadata HTTP {e.code} for file {file_id}") from e
        except urllib.error.URLError as e:
            raise DriveDownloadError(
                f"Drive metadata transport error for file {file_id}: "
                f"{type(e).__name__}") from e

    def _check_metadata(self, meta: dict, file_id: str, expected_size: int,
                        expected_sha256: str, expected_md5: str | None) -> None:
        if meta.get("trashed"):
            raise DriveChangedError(f"Drive file is trashed: {file_id}")
        size = meta.get("size")
        if size is None or int(size) != int(expected_size):
            raise DriveChangedError(
                f"Drive file size mismatch for {file_id}: server "
                f"{size!r} != pinned {expected_size}")
        server_sha = (meta.get("sha256Checksum") or "").lower()
        if server_sha and server_sha != expected_sha256.lower():
            raise DriveChangedError(
                f"Drive server SHA-256 mismatch for {file_id}")
        if expected_md5:
            server_md5 = (meta.get("md5Checksum") or "").lower()
            if server_md5 and server_md5 != expected_md5.lower():
                raise DriveChangedError(
                    f"Drive server MD5 mismatch for {file_id}")

    def _signal_token_expiring(self, on_token_expiring) -> None:
        if on_token_expiring is None:
            return
        remaining = None
        seconds_remaining = getattr(
            self.token_provider, "seconds_remaining", None)
        if callable(seconds_remaining):
            remaining = seconds_remaining()
        if remaining is not None and remaining < TOKEN_REFRESH_MARGIN_SEC:
            token_id = getattr(self.token_provider, "token_id", None)
            on_token_expiring(token_id, remaining)

    # -- single file --

    def download_file(self, file_id: str, expected_size: int,
                      expected_sha256: str, dest,
                      expected_md5: str | None = None,
                      on_progress=None, on_token_expiring=None,
                      abort_event: threading.Event | None = None) -> dict:
        """Download one pinned Drive file to ``dest``.

        Returns ``{"drive_file_id", "size", "sha256", "path"}``. The
        destination is atomically replaced only after the full SHA-256
        verifies; a ``.part`` file (and ``.part.bitmap.json`` for
        concurrent downloads) is left behind on interruption for resume.
        Raises DriveDownloadError (or a subclass, including
        DriveDownloadAbortedError) on any failure; the exception message
        never contains a token value.
        """
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or "", re.I):
            raise ValueError("Pinned SHA-256 required for " + str(file_id))
        if not expected_size or int(expected_size) <= 0:
            raise ValueError("Pinned byte count required for " + str(file_id))
        expected_size = int(expected_size)
        expected_sha256_l = expected_sha256.lower()
        eff_abort = self._effective_abort(abort_event)
        _check_abort(eff_abort, file_id)

        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = Path(str(dest) + ".part")
        bpath = _bitmap_path(part)

        # Pre-check: the server must agree this is the pinned file.
        meta = self.get_metadata(file_id)
        self._check_metadata(meta, file_id, expected_size, expected_sha256_l,
                             expected_md5)
        _check_abort(eff_abort, file_id)

        # Reuse a complete, already-verified local copy.
        if dest.is_file() and dest.stat().st_size == expected_size:
            from h3_contract import sha256_file

            if sha256_file(dest) == expected_sha256_l:
                bpath.unlink(missing_ok=True)
                return {"drive_file_id": file_id, "size": expected_size,
                        "sha256": expected_sha256_l, "path": str(dest)}
            dest.unlink()  # stale or corrupt local copy; never trust it

        # Check for a valid concurrent-resume bitmap sidecar before
        # touching .part. A preallocated .part WITH a bitmap has size ==
        # expected_size even while incomplete, so it must NOT be treated
        # as a complete linear .part.
        resume_bitmap = None
        if bpath.is_file():
            resume_bitmap = _load_bitmap(
                bpath, file_id, expected_size, expected_sha256_l)
            if resume_bitmap is None or not part.is_file() \
                    or part.stat().st_size != expected_size:
                # Corrupt or mismatched sidecar: clean reset.
                bpath.unlink(missing_ok=True)
                part.unlink(missing_ok=True)
                resume_bitmap = None

        start = 0
        if resume_bitmap is None:
            start = part.stat().st_size if part.is_file() else 0
            if start > expected_size:
                part.unlink()
                start = 0
            if start == expected_size:
                from h3_contract import sha256_file

                if sha256_file(part) == expected_sha256_l:
                    os.replace(part, dest)
                    return {"drive_file_id": file_id, "size": expected_size,
                            "sha256": expected_sha256_l, "path": str(dest)}
                part.unlink()
                start = 0

        last_error: Exception | None = None
        refreshed_once = False
        computed_sha256: str | None = None
        url = f"{DRIVE_API_BASE}/files/{file_id}?alt=media&supportsAllDrives=true"

        # v0.2.0 F1: large fresh downloads (or bitmap-resumed downloads)
        # use single-file zero-copy pwrite + streaming SHA-256.
        use_concurrent = (
            resume_bitmap is not None
            or (start == 0 and expected_size >= CONCURRENT_MIN_SIZE
                and self.concurrent_workers > 1)
        )
        if use_concurrent:
            ok, computed_sha256 = self._download_concurrent(
                file_id, expected_size, expected_sha256_l, part, bpath,
                on_progress, on_token_expiring, eff_abort, resume_bitmap)
            if ok:
                start = expected_size
            else:
                # Concurrent path fell back (HTTP 200 or shard retries
                # exhausted): .part and bitmap were cleaned up.
                start = 0
                computed_sha256 = None

        stream_hasher = None
        hashed_offset = 0

        for attempt in range(1, self.max_attempts + 1):
            if start >= expected_size:
                break
            _check_abort(eff_abort, file_id)
            self._signal_token_expiring(on_token_expiring)
            headers = {}
            if start > 0:
                headers["Range"] = f"bytes={start}-"
            used_token: str | None = None
            try:
                req = urllib.request.Request(url, headers=headers, method="GET")
                used_token = self.token_provider.get_token()
                req.add_header("Authorization", "Bearer " + used_token)
                with self._acquire_connection(eff_abort, file_id):
                    with urlopen(req, timeout=self.timeout_sec) as resp:
                        status = getattr(resp, "status", 200)
                        if status == 401:
                            if not refreshed_once:
                                refreshed_once = True
                                try:
                                    self._refresh_token(used_token)
                                except TokenUnavailable as e:
                                    raise DriveAuthError(
                                        "Drive download 401 and no fresh "
                                        f"token available for {file_id}") from e
                                continue
                            raise DriveAuthError(
                                "Drive download 401 after token refresh for "
                                + str(file_id))
                        if status == 403:
                            raise DriveAuthError(
                                "Drive download 403 (revoked or insufficient "
                                f"permission) for {file_id}")
                        if status == 416:
                            from h3_contract import sha256_file

                            if part.is_file() \
                                    and part.stat().st_size == expected_size \
                                    and sha256_file(part) == expected_sha256_l:
                                bpath.unlink(missing_ok=True)
                                os.replace(part, dest)
                                return {"drive_file_id": file_id,
                                        "size": expected_size,
                                        "sha256": expected_sha256_l,
                                        "path": str(dest)}
                            part.unlink(missing_ok=True)
                            start = 0
                            stream_hasher = None
                            hashed_offset = 0
                            last_error = DriveDownloadError(
                                "HTTP 416 with incomplete part for "
                                + str(file_id))
                            continue
                        if status == 200 and start > 0:
                            # Server ignored Range; restart from zero, do
                            # not append to a stale prefix.
                            part.unlink(missing_ok=True)
                            start = 0
                            stream_hasher = None
                            hashed_offset = 0
                            last_error = DriveDownloadError(
                                "Server ignored Range for " + str(file_id))
                            continue
                        if status not in (200, 206):
                            if status in TRANSIENT_STATUSES:
                                last_error = DriveDownloadError(
                                    f"Drive download HTTP {status} for {file_id}")
                                if attempt < self.max_attempts:
                                    # Release connection before sleeping.
                                    pass
                                else:
                                    raise DriveRetryExhaustedError(
                                        "Drive download retries exhausted for "
                                        f"{file_id}: {last_error}") from last_error
                            else:
                                raise DriveDownloadError(
                                    f"Drive download HTTP {status} for {file_id}")
                        else:
                            if status == 206:
                                cr = resp.headers.get("Content-Range", "")
                                m = re.fullmatch(
                                    r"bytes (\d+)-(\d+)/(\d+)", cr or "")
                                if not m or int(m.group(1)) != start \
                                        or int(m.group(3)) != expected_size:
                                    raise DriveChangedError(
                                        "Unexpected Content-Range for "
                                        + str(file_id))
                            # Prepare streaming SHA-256 hasher aligned to start.
                            if start == 0:
                                stream_hasher = hashlib.sha256()
                                hashed_offset = 0
                            elif stream_hasher is None or hashed_offset != start:
                                stream_hasher = hashlib.sha256()
                                with part.open("rb") as existing:
                                    rem = start
                                    while rem > 0:
                                        buf = existing.read(
                                            min(self.chunk_size, rem))
                                        if not buf:
                                            break
                                        stream_hasher.update(buf)
                                        rem -= len(buf)
                                hashed_offset = start - rem
                            mode = "ab" if start > 0 else "wb"
                            written = start
                            with part.open(mode) as f:
                                while True:
                                    _check_abort(eff_abort, file_id)
                                    chunk = resp.read(self.chunk_size)
                                    if not chunk:
                                        break
                                    f.write(chunk)
                                    if stream_hasher is not None \
                                            and hashed_offset == written:
                                        stream_hasher.update(chunk)
                                        hashed_offset += len(chunk)
                                    written += len(chunk)
                                    if written > expected_size:
                                        raise DriveDownloadError(
                                            "Drive download overran pinned "
                                            "size for " + str(file_id))
                                    if on_progress is not None:
                                        on_progress(written, expected_size)
                            start = written
                            if written == expected_size:
                                if stream_hasher is not None \
                                        and hashed_offset == expected_size:
                                    computed_sha256 = stream_hasher.hexdigest()
                                break
                            last_error = DriveDownloadError(
                                "Drive download short read for " + str(file_id))
            except (DriveDownloadError, DriveAuthError, DriveChangedError,
                    DriveDownloadAbortedError, ValueError):
                raise
            except urllib.error.HTTPError as e:
                if e.code == 401 and not refreshed_once:
                    refreshed_once = True
                    try:
                        self._refresh_token(used_token)
                    except TokenUnavailable as exc:
                        raise DriveAuthError(
                            "Drive download 401 and no fresh token "
                            f"available for {file_id}") from exc
                    continue
                if e.code == 401:
                    raise DriveAuthError(
                        "Drive download 401 after token refresh for "
                        + str(file_id)) from e
                if e.code == 403:
                    raise DriveAuthError(
                        "Drive download 403 (revoked or insufficient "
                        f"permission) for {file_id}") from e
                if e.code in TRANSIENT_STATUSES:
                    last_error = DriveDownloadError(
                        f"Drive download HTTP {e.code} for {file_id}")
                    if attempt >= self.max_attempts:
                        raise DriveRetryExhaustedError(
                            "Drive download retries exhausted for "
                            f"{file_id}: {last_error}") from e
                else:
                    raise DriveDownloadError(
                        f"Drive download HTTP {e.code} for {file_id}") from e
            except Exception as e:  # transport errors are transient
                last_error = e
            if attempt < self.max_attempts:
                _sleep_or_abort(
                    _backoff_delay(attempt, self.backoff_base_sec),
                    eff_abort, file_id)
                start = part.stat().st_size if part.is_file() else 0

        _check_abort(eff_abort, file_id)

        # Final verification: size + streaming (or fallback) SHA-256,
        # post-download metadata check, then atomic replace.
        if part.is_file() and part.stat().st_size == expected_size:
            if computed_sha256 is None:
                from h3_contract import sha256_file

                computed_sha256 = sha256_file(part)
            if computed_sha256 == expected_sha256_l:
                # Post-check: the server file must still be the pinned one;
                # a file that changed mid-transfer is refused even if the
                # downloaded bytes happen to hash correctly.
                post = self.get_metadata(file_id)
                self._check_metadata(post, file_id, expected_size,
                                     expected_sha256_l, expected_md5)
                _check_abort(eff_abort, file_id)
                bpath.unlink(missing_ok=True)
                os.replace(part, dest)
                return {"drive_file_id": file_id, "size": expected_size,
                        "sha256": expected_sha256_l, "path": str(dest)}
        if part.is_file():
            part.unlink(missing_ok=True)
        bpath.unlink(missing_ok=True)
        detail = (str(last_error)[:120] if last_error else "unknown")
        raise DriveRetryExhaustedError(
            "Drive download failed checksum/size for " + str(file_id)
            + ": " + detail)

    # -- zero-copy pwrite concurrent download + streaming SHA-256 (v0.2.0 F1) --

    def _download_block(self, url: str, file_id: str, start: int, end: int,
                        expected_size: int, fd: int, report,
                        stop_flag: threading.Event, abort_event) -> bytes:
        """Download one byte range ``[start, end]`` directly into ``fd``
        at ``offset=start`` via lock-free ``os.pwrite``, returning the
        block bytes for streaming SHA-256 aggregation."""
        length = end - start + 1
        last_error: Exception | None = None
        refreshed_once = False
        for attempt in range(1, self.max_attempts + 1):
            _check_abort(abort_event, file_id)
            if stop_flag.is_set():
                raise _ProgressAborted()
            used_token: str | None = None
            try:
                used_token = self.token_provider.get_token()
                req = urllib.request.Request(
                    url, headers={"Authorization": "Bearer " + used_token,
                                  "Range": f"bytes={start}-{end}"},
                    method="GET")
                with self._acquire_connection(abort_event, file_id):
                    with urlopen(req, timeout=self.timeout_sec) as resp:
                        status = getattr(resp, "status", 200)
                        if status == 200:
                            raise _NoRangeSupport(
                                "server answered 200 to a Range request for "
                                + str(file_id))
                        if status == 401:
                            if not refreshed_once:
                                refreshed_once = True
                                try:
                                    self._refresh_token(used_token)
                                except TokenUnavailable as e:
                                    raise DriveAuthError(
                                        "Drive download 401 and no fresh "
                                        f"token available for {file_id}") from e
                                continue
                            raise DriveAuthError(
                                "Drive download 401 after token refresh for "
                                + str(file_id))
                        if status == 403:
                            raise DriveAuthError(
                                "Drive download 403 (revoked or insufficient "
                                f"permission) for {file_id}")
                        if status != 206:
                            if status in TRANSIENT_STATUSES:
                                last_error = DriveDownloadError(
                                    f"Drive download HTTP {status} for {file_id}")
                                if attempt >= self.max_attempts:
                                    raise DriveRetryExhaustedError(
                                        "Drive download retries exhausted for "
                                        f"{file_id}: {last_error}") from last_error
                            else:
                                raise DriveDownloadError(
                                    f"Drive download HTTP {status} for {file_id}")
                        else:
                            cr = resp.headers.get("Content-Range", "")
                            m = re.fullmatch(
                                r"bytes (\d+)-(\d+)/(\d+)", cr or "")
                            if not m or int(m.group(1)) != start \
                                    or int(m.group(2)) != end \
                                    or int(m.group(3)) != expected_size:
                                raise DriveChangedError(
                                    "Unexpected Content-Range for "
                                    + str(file_id))
                            written = 0
                            chunks: list[bytes] = []
                            while True:
                                _check_abort(abort_event, file_id)
                                if stop_flag.is_set():
                                    raise _ProgressAborted()
                                chunk = resp.read(self.chunk_size)
                                if not chunk:
                                    break
                                if written + len(chunk) > length:
                                    raise DriveDownloadError(
                                        "Drive block overran its range for "
                                        + str(file_id))
                                _pwrite_all(fd, chunk, start + written)
                                chunks.append(chunk)
                                written += len(chunk)
                                report(len(chunk))
                            if written == length:
                                return (chunks[0] if len(chunks) == 1
                                        else b"".join(chunks))
                            last_error = DriveDownloadError(
                                "Drive block short read for " + str(file_id))
            except (DriveAuthError, DriveChangedError,
                    DriveDownloadAbortedError, _NoRangeSupport,
                    _ProgressAborted):
                raise
            except urllib.error.HTTPError as e:
                if e.code == 401 and not refreshed_once:
                    refreshed_once = True
                    try:
                        self._refresh_token(used_token)
                    except TokenUnavailable as exc:
                        raise DriveAuthError(
                            "Drive download 401 and no fresh token "
                            f"available for {file_id}") from exc
                    continue
                if e.code == 401:
                    raise DriveAuthError(
                        "Drive download 401 after token refresh for "
                        + str(file_id)) from e
                if e.code == 403:
                    raise DriveAuthError(
                        "Drive download 403 (revoked or insufficient "
                        f"permission) for {file_id}") from e
                if e.code in TRANSIENT_STATUSES:
                    last_error = DriveDownloadError(
                        f"Drive download HTTP {e.code} for {file_id}")
                    if attempt >= self.max_attempts:
                        raise DriveRetryExhaustedError(
                            "Drive download retries exhausted for "
                            f"{file_id}: {last_error}") from e
                else:
                    raise DriveDownloadError(
                        f"Drive download HTTP {e.code} for {file_id}") from e
            except Exception as e:  # transport errors are transient
                last_error = e
            if attempt < self.max_attempts:
                _sleep_or_abort(
                    _backoff_delay(attempt, self.backoff_base_sec),
                    abort_event, file_id)
        raise DriveRetryExhaustedError(
            "Drive block retries exhausted for " + str(file_id)
            + ": " + (str(last_error)[:120] if last_error else "unknown"))

    def _download_concurrent(
            self, file_id: str, expected_size: int, expected_sha256: str,
            part: Path, bpath: Path, on_progress, on_token_expiring,
            abort_event, resume_bitmap: dict | None) -> tuple[bool, str | None]:
        """Download a large file into a single preallocated ``.part`` via
        lock-free multi-threaded ``os.pwrite`` and streaming SHA-256.

        Returns ``(True, computed_sha256)`` when all blocks complete.
        Returns ``(False, None)`` to fall back to single-stream (server
        ignored Range or retries exhausted), cleaning up ``.part`` and
        ``.part.bitmap.json``. Fatal errors (``DriveAuthError``,
        ``DriveChangedError``, ``DriveDownloadAbortedError``, or an
        ``on_progress`` exception) are re-raised with no fallback; on
        abort/progress failure the bitmap sidecar is persisted so a
        subsequent call can resume without re-downloading completed
        blocks.
        """
        _check_abort(abort_event, file_id)
        self._signal_token_expiring(on_token_expiring)

        url = f"{DRIVE_API_BASE}/files/{file_id}?alt=media&supportsAllDrives=true"
        n = self.concurrent_workers
        if resume_bitmap is not None:
            block_size = int(resume_bitmap["block_size"])
            completed_set: set[int] = set(resume_bitmap["completed"])
        else:
            block_size = max(
                1, min(self.range_block_size, (expected_size + n - 1) // n))
            completed_set = set()

        total_blocks = (expected_size + block_size - 1) // block_size
        ranges = [
            (i * block_size, min((i + 1) * block_size, expected_size) - 1)
            for i in range(total_blocks)
        ]

        if resume_bitmap is not None and part.is_file():
            fd = os.open(str(part), os.O_RDWR, 0o644)
            if os.fstat(fd).st_size != expected_size:
                os.truncate(fd, expected_size)
                completed_set.clear()
        else:
            fd = os.open(
                str(part), os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
            os.truncate(fd, expected_size)

        resumed_bytes = sum(
            ranges[i][1] - ranges[i][0] + 1 for i in completed_set)
        done = [resumed_bytes]
        progress_lock = threading.Lock()
        bitmap_lock = threading.Lock()
        progress_error: list = []
        stop_flag = threading.Event()

        def _persist_bitmap_locked() -> None:
            _save_bitmap(bpath, {
                "version": _BITMAP_VERSION,
                "file_id": file_id,
                "expected_size": expected_size,
                "expected_sha256": expected_sha256,
                "block_size": block_size,
                "total_blocks": total_blocks,
                "completed": sorted(completed_set),
            })

        def _report(nbytes: int) -> None:
            # Held across the callback so concurrent workers report a
            # monotonic cumulative total.
            with progress_lock:
                done[0] = min(expected_size, done[0] + nbytes)
                total = done[0]
                if on_progress is not None:
                    try:
                        on_progress(total, expected_size)
                    except Exception as exc:  # recorded; raised after pool
                        if not progress_error:
                            progress_error.append(exc)
                        stop_flag.set()

        hasher = _StreamingBlockHasher(
            ranges, fd, max_buffer_bytes=self.max_hash_buffer_bytes,
            pread_chunk_size=self.chunk_size)
        for idx in sorted(completed_set):
            hasher.mark_resumed(idx)

        pending_indices = [
            i for i in range(total_blocks) if i not in completed_set]
        cursor = [0]
        cursor_lock = threading.Lock()

        def _worker() -> None:
            while True:
                _check_abort(abort_event, file_id)
                if stop_flag.is_set():
                    return
                with cursor_lock:
                    if cursor[0] >= len(pending_indices):
                        return
                    idx = pending_indices[cursor[0]]
                    cursor[0] += 1
                s, e = ranges[idx]
                block_bytes = self._download_block(
                    url, file_id, s, e, expected_size, fd,
                    _report, stop_flag, abort_event)
                hasher.submit_block(idx, block_bytes)
                with bitmap_lock:
                    completed_set.add(idx)
                    _persist_bitmap_locked()

        errors: list[Exception] = []
        pool_workers = max(1, min(n, len(pending_indices) or 1))
        preserve_for_resume = False
        try:
            if pending_indices:
                with ThreadPoolExecutor(max_workers=pool_workers) as pool:
                    futs = [pool.submit(_worker) for _ in range(pool_workers)]
                    for fut in futs:
                        try:
                            fut.result()
                        except Exception as exc:
                            errors.append(exc)
                            if isinstance(
                                    exc, (DriveAuthError, DriveChangedError,
                                          DriveDownloadAbortedError,
                                          _NoRangeSupport)):
                                stop_flag.set()

            if abort_event is not None and abort_event.is_set():
                preserve_for_resume = True
                raise DriveDownloadAbortedError(
                    f"Drive download aborted for {file_id}")
            for exc in errors:
                if isinstance(exc, DriveDownloadAbortedError):
                    preserve_for_resume = True
                    raise exc
            if progress_error:
                preserve_for_resume = True
                raise progress_error[0]
            for exc in errors:
                if isinstance(exc, (DriveAuthError, DriveChangedError)):
                    preserve_for_resume = bool(completed_set)
                    raise exc
            if errors:
                # No Range support, or a block gave up: clean up the
                # preallocated .part so single-stream starts cleanly.
                return False, None

            computed = hasher.hexdigest()
            self.last_re_read_bytes = hasher.re_read_bytes
            return True, computed
        finally:
            try:
                if preserve_for_resume and completed_set:
                    with bitmap_lock:
                        _persist_bitmap_locked()
            finally:
                os.close(fd)
                if not preserve_for_resume and errors:
                    part.unlink(missing_ok=True)
                    bpath.unlink(missing_ok=True)

    def probe_transport(self, file_id: str,
                        probe_bytes: int = 1024 * 1024) -> dict:
        """Cheap auth + transport probe: metadata GET, then a single
        Range GET of the first ``probe_bytes`` bytes. Validates the
        206 status, the Content-Range header, and the exact byte
        count. Proves the token works and the server honours Range;
        it does NOT prove full-file integrity (the real restore does
        that with a full SHA-256). Raises DriveDownloadError on any
        failure. Never returns token material."""
        meta = self.get_metadata(file_id)
        total = meta.get("size")
        if total is None or int(total) <= 0:
            raise DriveDownloadError(
                f"Drive probe metadata missing size for {file_id}")
        want = min(int(probe_bytes), int(total))
        url = f"{DRIVE_API_BASE}/files/{file_id}?alt=media&supportsAllDrives=true"
        token = self.token_provider.get_token()
        req = urllib.request.Request(
            url, headers={"Authorization": "Bearer " + token,
                          "Range": f"bytes=0-{want - 1}"}, method="GET")
        try:
            with urlopen(req, timeout=self.timeout_sec) as resp:
                status = getattr(resp, "status", 200)
                if status != 206:
                    raise DriveDownloadError(
                        f"Drive probe expected HTTP 206, got {status} "
                        f"for {file_id}")
                cr = resp.headers.get("Content-Range", "")
                m = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", cr or "")
                if not m or int(m.group(1)) != 0 \
                        or int(m.group(3)) != int(total):
                    raise DriveChangedError(
                        "Drive probe unexpected Content-Range for "
                        + str(file_id))
                got = 0
                while True:
                    chunk = resp.read(self.chunk_size)
                    if not chunk:
                        break
                    got += len(chunk)
                if got != want:
                    raise DriveDownloadError(
                        f"Drive probe short read for {file_id}: got "
                        f"{got}, wanted {want}")
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise DriveAuthError(
                    "Drive probe 401 for " + str(file_id)) from e
            if e.code == 403:
                raise DriveAuthError(
                    "Drive probe 403 for " + str(file_id)) from e
            raise DriveDownloadError(
                f"Drive probe HTTP {e.code} for {file_id}") from e
        return {"drive_file_id": file_id, "size": int(total),
                "probe_bytes": want}

    # -- many files --

    def download_all(self, files, dest_dir, on_progress=None,
                     on_token_expiring=None,
                     abort_event: threading.Event | None = None) -> list:
        """Download a list of pinned files.

        Each entry: ``{"file_id", "path", "size", "sha256",
        "md5" (optional), "on_progress" (optional)}``. ``path`` is
        relative to ``dest_dir``. Files download with bounded
        concurrency (``max_workers``; default 1 = sequential) while
        sharing the global connection semaphore cap. Returns per-file
        result dicts in input order. On failure the first error is
        raised; files that already verified stay on disk and are reused
        on retry.
        """
        dest_dir = Path(dest_dir)
        eff_abort = self._effective_abort(abort_event)

        def _one(entry):
            rel = entry["path"]
            cb = entry.get("on_progress", on_progress)
            return self.download_file(
                entry["file_id"], int(entry["size"]), entry["sha256"],
                dest_dir / rel, expected_md5=entry.get("md5"),
                on_progress=cb,
                on_token_expiring=on_token_expiring,
                abort_event=eff_abort)

        if self.max_workers == 1:
            return [_one(entry) for entry in files]

        entries = list(files)
        results: list = [None] * len(entries)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futs = [(idx, pool.submit(_one, entry))
                    for idx, entry in enumerate(entries)]
            first_exc: Exception | None = None
            for idx, fut in futs:
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    if first_exc is None:
                        first_exc = exc
                        for _, pending_fut in futs:
                            pending_fut.cancel()
            if first_exc is not None:
                raise first_exc
        return results
