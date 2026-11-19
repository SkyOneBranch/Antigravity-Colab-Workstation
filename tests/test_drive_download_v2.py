"""v0.2.0 F1 unit tests for zero-copy pwrite concurrent downloads,
streaming SHA-256, .part.bitmap.json resume, global connection cap,
abort_event cancellation, and single-flight token refresh.

All tests run 100% offline (0 CU) against FakeDrive.
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h3_contract
import h3_drive_download as dd
from h3_drive_download import (
    DriveDownloadAbortedError,
    DriveDownloader,
    DriveRetryExhaustedError,
    FileTokenProvider,
    token_id_for,
)
from tests.test_drive_download import FakeDrive, FakeProvider, FakeResp, sha


def _write_tok(path: Path, tok: str, expires_utc: str = "2030-01-01T10:00:00Z"):
    doc = {
        "access_token": tok,
        "expires_at_utc": expires_utc,
        "token_id": token_id_for(tok),
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    os.chmod(path, 0o600)


class ConfigResolutionTests(unittest.TestCase):
    def test_default_and_env_concurrent_workers(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(dd.ENV_CONCURRENT_WORKERS, None)
            self.assertEqual(dd.resolve_concurrent_workers(), 12)

        with patch.dict(os.environ, {dd.ENV_CONCURRENT_WORKERS: "8"}):
            self.assertEqual(dd.resolve_concurrent_workers(), 8)
            self.assertEqual(
                DriveDownloader(FakeProvider()).concurrent_workers, 8)

        with patch.dict(os.environ, {dd.ENV_CONCURRENT_WORKERS: "0"}):
            self.assertEqual(dd.resolve_concurrent_workers(), 1)

        with patch.dict(os.environ, {dd.ENV_CONCURRENT_WORKERS: "64"}):
            self.assertEqual(dd.resolve_concurrent_workers(), 16)

        with patch.dict(os.environ, {dd.ENV_CONCURRENT_WORKERS: "not-a-num"}):
            self.assertEqual(dd.resolve_concurrent_workers(), 12)

    def test_explicit_concurrent_workers_validation(self):
        self.assertEqual(dd.resolve_concurrent_workers(1), 1)
        self.assertEqual(dd.resolve_concurrent_workers(16), 16)
        with self.assertRaises(ValueError):
            dd.resolve_concurrent_workers(0)
        with self.assertRaises(ValueError):
            dd.resolve_concurrent_workers(17)
        with self.assertRaises(ValueError):
            DriveDownloader(FakeProvider(), concurrent_workers=17)

    def test_max_global_connections_resolution(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(dd.ENV_MAX_GLOBAL_CONNECTIONS, None)
            self.assertEqual(dd.resolve_max_global_connections(), 16)
        with patch.dict(os.environ, {dd.ENV_MAX_GLOBAL_CONNECTIONS: "10"}):
            self.assertEqual(dd.resolve_max_global_connections(), 10)
        with self.assertRaises(ValueError):
            dd.resolve_max_global_connections(0)


class _V2Base(unittest.TestCase):
    FILE_ID = "fid-v2-big"
    CONTENT = bytes((i * 37 + 13) % 256 for i in range(8192))

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.td, ignore_errors=True))
        self.drive = FakeDrive()
        self._register_file(self.FILE_ID, self.CONTENT)
        self.provider = FakeProvider()
        self.p_urlopen = patch.object(dd, "urlopen", self.drive)
        self.p_sleep = patch.object(dd, "_sleep", lambda s: None)
        self.p_min = patch.object(dd, "CONCURRENT_MIN_SIZE", 1024)
        self.p_urlopen.start()
        self.p_sleep.start()
        self.p_min.start()
        self.addCleanup(self.p_urlopen.stop)
        self.addCleanup(self.p_sleep.stop)
        self.addCleanup(self.p_min.stop)

    def _register_file(self, fid: str, data: bytes):
        self.drive.metadata[fid] = {
            "size": len(data),
            "md5": hashlib.md5(data).hexdigest(),
            "sha256": sha(data),
        }
        self.drive.media_content[fid] = data

    def _dest(self, name="model.bin") -> Path:
        return Path(self.td) / name


class ZeroCopyPwriteTests(_V2Base):
    def test_partial_pwrite_loop_and_peak_disk(self):
        real_pwrite = os.pwrite
        peak_bytes = [0]

        def _short_pwrite(fd, data, offset):
            # Force partial writes of at most 113 bytes per call.
            chunk = memoryview(data)[:113]
            n = real_pwrite(fd, chunk, offset)
            total_on_disk = sum(
                p.stat().st_size for p in Path(self.td).iterdir()
                if p.is_file() and not p.name.endswith(".bitmap.json")
                and not p.name.endswith(".bitmap.json.tmp"))
            if total_on_disk > peak_bytes[0]:
                peak_bytes[0] = total_on_disk
            return n

        dl = DriveDownloader(
            self.provider, concurrent_workers=4, range_block_size=1024,
            backoff_base_sec=0)
        with patch.object(os, "pwrite", _short_pwrite):
            out = dl.download_file(
                self.FILE_ID, len(self.CONTENT), sha(self.CONTENT),
                self._dest())
        self.assertEqual(Path(out["path"]).read_bytes(), self.CONTENT)
        self.assertEqual(peak_bytes[0], len(self.CONTENT))
        self.assertEqual(list(Path(self.td).glob("*.shard*")), [])
        self.assertFalse(Path(str(self._dest()) + ".part").exists())
        self.assertFalse(Path(str(self._dest()) + ".part.bitmap.json").exists())


class StreamingSha256Tests(_V2Base):
    def test_concurrent_fresh_does_not_reread_part(self):
        def _forbid_sha256_file(_path):
            raise AssertionError("sha256_file must not be called on fresh download")

        dl = DriveDownloader(
            self.provider, concurrent_workers=4, range_block_size=1024,
            backoff_base_sec=0)
        with patch.object(h3_contract, "sha256_file", _forbid_sha256_file):
            out = dl.download_file(
                self.FILE_ID, len(self.CONTENT), sha(self.CONTENT),
                self._dest())
        self.assertEqual(Path(out["path"]).read_bytes(), self.CONTENT)
        self.assertEqual(dl.last_re_read_bytes, 0)

    def test_single_stream_fresh_does_not_reread_part(self):
        small_id = "fid-small"
        small_data = b"hello-streaming-sha256"
        self._register_file(small_id, small_data)

        def _forbid_sha256_file(_path):
            raise AssertionError("sha256_file must not be called on fresh single-stream")

        dl = DriveDownloader(self.provider, concurrent_workers=4, backoff_base_sec=0)
        with patch.object(h3_contract, "sha256_file", _forbid_sha256_file):
            out = dl.download_file(
                small_id, len(small_data), sha(small_data), self._dest("s.bin"))
        self.assertEqual(Path(out["path"]).read_bytes(), small_data)

    def test_out_of_order_blocks_and_buffer_spillback(self):
        orig_call = self.drive.__call__
        block0_release = threading.Event()
        later_blocks_done = threading.Event()
        completed_later = [0]
        lock = threading.Lock()

        def _delayed_block0(request, timeout=None):
            headers = {k.lower(): v for k, v in dict(getattr(request, "headers", {}) or {}).items()}
            rng = headers.get("range", "")
            if "alt=media" in request.full_url and rng.startswith("bytes=0-"):
                later_blocks_done.wait(timeout=2.0)
                block0_release.set()
            resp = orig_call(request, timeout)
            if "alt=media" in request.full_url and rng and not rng.startswith("bytes=0-"):
                with lock:
                    completed_later[0] += 1
                    if completed_later[0] >= 3:
                        later_blocks_done.set()
            return resp

        # Tiny max_hash_buffer_bytes forces out-of-order blocks to spill
        # and verify via selective os.pread when block 0 completes.
        dl = DriveDownloader(
            self.provider, concurrent_workers=4, range_block_size=2048,
            max_hash_buffer_bytes=1024, backoff_base_sec=0)
        with patch.object(dd, "urlopen", _delayed_block0):
            out = dl.download_file(
                self.FILE_ID, len(self.CONTENT), sha(self.CONTENT),
                self._dest())
        self.assertTrue(block0_release.is_set())
        self.assertEqual(Path(out["path"]).read_bytes(), self.CONTENT)
        self.assertGreater(dl.last_re_read_bytes, 0)

    def test_corrupt_block_fails_and_cleans_up(self):
        bad_content = bytearray(self.CONTENT)
        bad_content[4096] ^= 0xFF
        self.drive.media_content[self.FILE_ID] = bytes(bad_content)
        dl = DriveDownloader(
            self.provider, concurrent_workers=4, range_block_size=2048,
            backoff_base_sec=0)
        with self.assertRaises(DriveRetryExhaustedError):
            dl.download_file(
                self.FILE_ID, len(self.CONTENT), sha(self.CONTENT),
                self._dest())
        self.assertFalse(self._dest().exists())
        self.assertFalse(Path(str(self._dest()) + ".part").exists())
        self.assertFalse(Path(str(self._dest()) + ".part.bitmap.json").exists())


class BitmapResumeTests(_V2Base):
    def test_resume_from_bitmap_skips_completed_blocks(self):
        dest = self._dest()
        part = Path(str(dest) + ".part")
        bpath = Path(str(part) + ".bitmap.json")
        abort = threading.Event()
        # Use 1 worker (still concurrent via bitmap? no, use 2 workers with
        # deterministic abort after 2 blocks complete).
        dl1 = DriveDownloader(
            self.provider, concurrent_workers=2, range_block_size=2048,
            backoff_base_sec=0)

        def _stop_halfway(done, total):
            if done >= 4096:
                abort.set()

        with self.assertRaises(DriveDownloadAbortedError):
            dl1.download_file(
                self.FILE_ID, len(self.CONTENT), sha(self.CONTENT), dest,
                on_progress=_stop_halfway, abort_event=abort)

        self.assertFalse(dest.exists())
        self.assertTrue(part.is_file())
        self.assertTrue(bpath.is_file())
        saved = json.loads(bpath.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(saved["completed"]), 1)
        self.assertLess(len(saved["completed"]), saved["total_blocks"])

        # Clear recorded calls, then resume without abort.
        self.drive.calls.clear()
        dl2 = DriveDownloader(
            self.provider, concurrent_workers=2, range_block_size=2048,
            backoff_base_sec=0)
        out = dl2.download_file(
            self.FILE_ID, len(self.CONTENT), sha(self.CONTENT), dest)
        self.assertEqual(Path(out["path"]).read_bytes(), self.CONTENT)
        self.assertFalse(part.exists())
        self.assertFalse(bpath.exists())

        # Already-completed blocks must NOT have been re-requested.
        resumed_ranges = [
            h.get("range") for u, h in self.drive.calls if "alt=media" in u]
        expected_remaining = saved["total_blocks"] - len(saved["completed"])
        self.assertEqual(len(resumed_ranges), expected_remaining)
        self.assertGreater(dl2.last_re_read_bytes, 0)

    def test_corrupt_bitmap_resets_cleanly(self):
        dest = self._dest()
        part = Path(str(dest) + ".part")
        bpath = Path(str(part) + ".bitmap.json")
        part.write_bytes(b"\x00" * len(self.CONTENT))
        bpath.write_text("{corrupt-json", encoding="utf-8")

        dl = DriveDownloader(
            self.provider, concurrent_workers=4, range_block_size=2048,
            backoff_base_sec=0)
        out = dl.download_file(
            self.FILE_ID, len(self.CONTENT), sha(self.CONTENT), dest)
        self.assertEqual(Path(out["path"]).read_bytes(), self.CONTENT)
        self.assertFalse(bpath.exists())

    def test_mismatched_bitmap_resets_cleanly(self):
        dest = self._dest()
        part = Path(str(dest) + ".part")
        bpath = Path(str(part) + ".bitmap.json")
        part.write_bytes(b"\x00" * len(self.CONTENT))
        bpath.write_text(json.dumps({
            "version": 1, "file_id": "other-file",
            "expected_size": len(self.CONTENT),
            "expected_sha256": sha(self.CONTENT),
            "block_size": 2048, "total_blocks": 4, "completed": [0, 1],
        }), encoding="utf-8")

        dl = DriveDownloader(
            self.provider, concurrent_workers=4, range_block_size=2048,
            backoff_base_sec=0)
        out = dl.download_file(
            self.FILE_ID, len(self.CONTENT), sha(self.CONTENT), dest)
        self.assertEqual(Path(out["path"]).read_bytes(), self.CONTENT)
        self.assertFalse(bpath.exists())


class GlobalConnectionCapTests(_V2Base):
    def test_download_all_respects_global_connection_cap(self):
        files = []
        for idx in range(3):
            fid = f"fid-cap-{idx}"
            data = bytes(((i + idx * 7) * 19) % 256 for i in range(4096))
            self._register_file(fid, data)
            files.append({
                "file_id": fid, "path": f"sub/file_{idx}.bin",
                "size": len(data), "sha256": sha(data),
            })

        orig_call = self.drive.__call__
        lock = threading.Lock()
        in_flight = [0]
        max_in_flight = [0]

        def _tracked(request, timeout=None):
            if "alt=media" in request.full_url:
                with lock:
                    in_flight[0] += 1
                    if in_flight[0] > max_in_flight[0]:
                        max_in_flight[0] = in_flight[0]
                try:
                    time.sleep(0.01)
                    return orig_call(request, timeout)
                finally:
                    with lock:
                        in_flight[0] -= 1
            return orig_call(request, timeout)

        dl = DriveDownloader(
            self.provider, max_workers=3, concurrent_workers=6,
            max_global_connections=4, range_block_size=1024,
            backoff_base_sec=0)
        with patch.object(dd, "urlopen", _tracked):
            results = dl.download_all(files, Path(self.td) / "out")

        self.assertEqual(len(results), 3)
        self.assertLessEqual(max_in_flight[0], 4)
        self.assertGreaterEqual(max_in_flight[0], 2)
        for entry, res in zip(files, results):
            self.assertEqual(
                Path(res["path"]).read_bytes(),
                self.drive.media_content[entry["file_id"]])

    def test_shared_semaphore_caps_across_two_downloaders(self):
        fid_a, fid_b = "fid-dual-a", "fid-dual-b"
        data_a = bytes((i * 11) % 256 for i in range(4096))
        data_b = bytes((i * 17) % 256 for i in range(4096))
        self._register_file(fid_a, data_a)
        self._register_file(fid_b, data_b)

        orig_call = self.drive.__call__
        lock = threading.Lock()
        in_flight = [0]
        max_in_flight = [0]

        def _tracked(request, timeout=None):
            if "alt=media" in request.full_url:
                with lock:
                    in_flight[0] += 1
                    if in_flight[0] > max_in_flight[0]:
                        max_in_flight[0] = in_flight[0]
                try:
                    time.sleep(0.01)
                    return orig_call(request, timeout)
                finally:
                    with lock:
                        in_flight[0] -= 1
            return orig_call(request, timeout)

        shared_sem = threading.BoundedSemaphore(3)
        dl_a = DriveDownloader(
            self.provider, concurrent_workers=4, connection_semaphore=shared_sem,
            range_block_size=1024, backoff_base_sec=0)
        dl_b = DriveDownloader(
            self.provider, concurrent_workers=4, connection_semaphore=shared_sem,
            range_block_size=1024, backoff_base_sec=0)

        errs = []

        def _run(dl, fid, data, name):
            try:
                dl.download_file(fid, len(data), sha(data), self._dest(name))
            except Exception as exc:
                errs.append(exc)

        with patch.object(dd, "urlopen", _tracked):
            t1 = threading.Thread(target=_run, args=(dl_a, fid_a, data_a, "a.bin"))
            t2 = threading.Thread(target=_run, args=(dl_b, fid_b, data_b, "b.bin"))
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

        self.assertEqual(errs, [])
        self.assertLessEqual(max_in_flight[0], 3)
        self.assertEqual(self._dest("a.bin").read_bytes(), data_a)
        self.assertEqual(self._dest("b.bin").read_bytes(), data_b)


class AbortEventTests(_V2Base):
    def test_preset_abort_stops_before_network(self):
        abort = threading.Event()
        abort.set()
        dl = DriveDownloader(self.provider, concurrent_workers=4, backoff_base_sec=0)
        with self.assertRaises(DriveDownloadAbortedError):
            dl.download_file(
                self.FILE_ID, len(self.CONTENT), sha(self.CONTENT),
                self._dest(), abort_event=abort)
        self.assertEqual(self.drive.calls, [])

    def test_single_stream_mid_chunk_abort(self):
        small_id = "fid-abort-small"
        small_data = b"abcdefghij" * 10
        self._register_file(small_id, small_data)
        abort = threading.Event()

        def _abort_on_progress(done, total):
            abort.set()

        dl = DriveDownloader(
            self.provider, chunk_size=10, concurrent_workers=1,
            backoff_base_sec=0)
        with self.assertRaises(DriveDownloadAbortedError):
            dl.download_file(
                small_id, len(small_data), sha(small_data), self._dest("s.bin"),
                on_progress=_abort_on_progress, abort_event=abort)
        self.assertFalse(self._dest("s.bin").exists())

    def test_abort_wakes_backoff_sleep_immediately(self):
        small_id = "fid-abort-sleep"
        small_data = b"0123456789"
        self._register_file(small_id, small_data)
        self.drive.media_plan[small_id] = [
            ("resp", 503, b"", {}),
            ("resp", 200, small_data, {}),
        ]
        abort = threading.Event()

        def _trigger():
            time.sleep(0.05)
            abort.set()

        t = threading.Thread(target=_trigger)
        dl = DriveDownloader(
            self.provider, max_attempts=3, backoff_base_sec=30.0)
        # Restore real _default_sleep so abort_event.wait() is exercised.
        t0 = time.monotonic()
        t.start()
        try:
            with patch.object(dd, "_sleep", dd._default_sleep):
                with self.assertRaises(DriveDownloadAbortedError):
                    dl.download_file(
                        small_id, len(small_data), sha(small_data),
                        self._dest("w.bin"), abort_event=abort)
        finally:
            t.join(timeout=2)
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 2.0)

    def test_instance_abort_event_aborts_download(self):
        abort = threading.Event()
        abort.set()
        dl = DriveDownloader(self.provider, abort_event=abort)
        with self.assertRaises(DriveDownloadAbortedError):
            dl.download_file(
                self.FILE_ID, len(self.CONTENT), sha(self.CONTENT),
                self._dest())


class SingleFlightTokenRefreshTests(unittest.TestCase):
    def test_concurrent_refresh_same_failed_token_id_is_single_flight(self):
        td = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        tok_path = Path(td) / "drive_tok.json"
        _write_tok(tok_path, "tok-a")
        provider = FileTokenProvider(tok_path)
        provider.load_initial()
        failed_tid = provider.token_id
        self.assertEqual(failed_tid, token_id_for("tok-a"))

        # Controller delivers ONE fresh token file.
        _write_tok(tok_path, "tok-b")

        barrier = threading.Barrier(2)
        results = []
        errors = []

        def _do_refresh():
            try:
                barrier.wait(timeout=2)
                results.append(provider.refresh(failed_token_id=failed_tid))
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=_do_refresh)
        t2 = threading.Thread(target=_do_refresh)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        self.assertEqual(errors, [])
        self.assertEqual(results, ["tok-b", "tok-b"])
        self.assertEqual(provider.token_id, token_id_for("tok-b"))
        self.assertFalse(tok_path.exists())


if __name__ == "__main__":
    unittest.main()
