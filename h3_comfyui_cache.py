"""Build and publish the pinned ComfyUI source & offline wheelhouse cache to Google Drive.

The cache lives OUTSIDE the model depot, in a dedicated folder
``Muse-H3-ComfyUI-Source`` in the same authorized Google account. It holds:

  comfyui-<commit>.tar.gz                  git archive of the pinned commit
                                           (prefix comfyui/, plus a
                                           COMFYUI_COMMIT.txt marker)
  comfyui-<commit>.tar.gz.sha256           hex SHA-256 of the source archive
  comfyui-source-manifest.json             source cache metadata

And optionally one or more per-Python-version offline wheelhouse bundles:

  comfyui-wheelhouse-<commit>-<pytag>.tar.gz        deterministic wheelhouse
                                                    (prefix wheelhouse/ with
                                                    *.whl + WHEELHOUSE_META.json)
  comfyui-wheelhouse-<commit>-<pytag>.tar.gz.sha256 hex SHA-256 of wheelhouse
  comfyui-wheelhouse-<commit>-<pytag>.manifest.json wheelhouse metadata

The remote worker restores the source archive instead of cloning from GitHub
whenever the cache is present, and installs incremental ComfyUI dependencies
offline from the matching wheelhouse (falling back to PyPI if absent or
mismatched). No model file and no READY manifest is touched: the folder is
separate from Muse-H3-Model-Depot.

All Drive SDK imports are lazy so the pure-stdlib helpers in this module can
be imported on the Colab worker or in minimal CI environments without Drive
credentials or googleapiclient installed.

Usage (from the project root):
  python3 h3_comfyui_cache.py build --repo comfyui-source --commit <sha>
  python3 h3_comfyui_cache.py publish --repo comfyui-source --commit <sha> \
      --colab-auth adc --allow-drive-write
  python3 h3_comfyui_cache.py build-wheelhouse --repo comfyui-source \
      --commit <sha> --python-tag py312
  python3 h3_comfyui_cache.py publish-wheelhouse --repo comfyui-source \
      --commit <sha> --python-tag py312 --colab-auth adc --allow-drive-write
  python3 h3_comfyui_cache.py verify --colab-auth adc
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COMFYUI_REPO_URL = "https://github.com/Comfy-Org/ComfyUI.git"
CACHE_FOLDER_NAME = "Muse-H3-ComfyUI-Source"
CACHE_DRIVE_DIR = "/content/drive/MyDrive/Muse-H3-ComfyUI-Source"
LOCAL_MANIFEST = ROOT / "model_cache" / "comfyui_source_manifest.json"
STAGING = ROOT / "model_cache" / "comfyui_source"

DEFAULT_EXCLUDED_PACKAGES: tuple[str, ...] = (
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
)
DEFAULT_EXCLUDED_PREFIXES: tuple[str, ...] = ("nvidia-",)
DEFAULT_PLATFORMS: tuple[str, ...] = (
    "manylinux2014_x86_64",
    "manylinux_2_17_x86_64",
    "linux_x86_64",
)
WHEELHOUSE_META_FILENAME = "WHEELHOUSE_META.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PKG_NAME_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")


def _file_digests(path: Path) -> dict:
    """Pure-stdlib size + SHA-256 + MD5 digest helper (matches drive_api.file_digests)."""
    h_sha256 = hashlib.sha256()
    h_md5 = hashlib.md5()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            size += len(chunk)
            h_sha256.update(chunk)
            h_md5.update(chunk)
    return {"size": size, "sha256": h_sha256.hexdigest(), "md5": h_md5.hexdigest()}


def _atomic_json_local(path: Path, payload: dict) -> None:
    """Pure-stdlib atomic JSON write with parent dir creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Pure-stdlib wheelhouse helpers (safe to import on Colab worker / CI)
# ---------------------------------------------------------------------------

def normalize_pkg_name(name: str) -> str:
    """PEP 503 package name normalization."""
    return re.sub(r"[-_.]+", "-", str(name).strip()).lower()


def normalize_python_tag(tag: str | None = None) -> str:
    """Normalize Python version / tag into canonical ``py<major><minor>`` form.

    Accepts ``None`` (current interpreter), ``"3.12"``, ``"py3.12"``,
    ``"py312"``, ``"cp312"``, or ``"312"``. Raises ``ValueError`` otherwise.
    """
    if tag is None:
        return f"py{sys.version_info.major}{sys.version_info.minor}"
    raw = str(tag).strip().lower()
    m = re.match(r"^(?:py|cp)?(\d+)\.(\d+)$", raw)
    if m:
        major, minor = int(m.group(1)), int(m.group(2))
        if major == 3 and 8 <= minor <= 20:
            return f"py{major}{minor}"
    m = re.match(r"^(?:py|cp)?(3)(\d{1,2})$", raw)
    if m:
        major, minor = int(m.group(1)), int(m.group(2))
        if 8 <= minor <= 20:
            return f"py{major}{minor}"
    raise ValueError(f"Unsupported Python tag: {tag!r}")


def _python_tag_to_version_abi(python_tag: str) -> tuple[str, str]:
    """Convert canonical ``py312`` -> ``("3.12", "cp312")``."""
    pytag = normalize_python_tag(python_tag)
    major = pytag[2]
    minor = pytag[3:]
    return f"{major}.{minor}", f"cp{major}{minor}"


def archive_name(commit: str) -> str:
    return f"comfyui-{commit}.tar.gz"


def wheelhouse_archive_name(commit: str, python_tag: str) -> str:
    pytag = normalize_python_tag(python_tag)
    return f"comfyui-wheelhouse-{commit}-{pytag}.tar.gz"


def wheelhouse_manifest_name(commit: str, python_tag: str) -> str:
    pytag = normalize_python_tag(python_tag)
    return f"comfyui-wheelhouse-{commit}-{pytag}.manifest.json"


def _extract_requirement_pkg_name(line: str) -> str | None:
    """Extract normalized package name from a requirements.txt line, or None."""
    s = line.strip()
    if not s or s.startswith("#") or s.startswith("-"):
        return None
    # Strip inline comments preceded by whitespace.
    s = re.split(r"\s+#", s, maxsplit=1)[0].strip()
    if not s:
        return None
    # Strip environment markers.
    s = s.split(";", 1)[0].strip()
    # Handle direct URL syntax: pkg @ https://...
    if "@" in s:
        s = s.split("@", 1)[0].strip()
    m = _PKG_NAME_RE.match(s)
    if not m:
        return None
    return normalize_pkg_name(m.group(1))


def _is_excluded_package(
    pkg_name: str,
    excluded_packages: Iterable[str] = DEFAULT_EXCLUDED_PACKAGES,
    excluded_prefixes: Iterable[str] = DEFAULT_EXCLUDED_PREFIXES,
) -> bool:
    norm = normalize_pkg_name(pkg_name)
    excl_set = {normalize_pkg_name(p) for p in excluded_packages if str(p).strip()}
    excl_pref = tuple(normalize_pkg_name(p) for p in excluded_prefixes if str(p).strip())
    if norm in excl_set:
        return True
    return any(norm.startswith(pref) for pref in excl_pref)


def _wheel_dist_name(wheel_filename: str) -> str:
    """Extract PEP 503 normalized distribution name from a .whl filename."""
    base = Path(wheel_filename).name
    dist = base.split("-", 1)[0]
    return normalize_pkg_name(dist)


def filter_requirements_lines(
    lines: Iterable[str],
    excluded_packages: Iterable[str] = DEFAULT_EXCLUDED_PACKAGES,
    excluded_prefixes: Iterable[str] = DEFAULT_EXCLUDED_PREFIXES,
) -> list[str]:
    """Return requirement lines excluding preinstalled heavy packages."""
    kept: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        pkg = _extract_requirement_pkg_name(line)
        if pkg is not None and _is_excluded_package(pkg, excluded_packages, excluded_prefixes):
            continue
        kept.append(line)
    return kept


def filter_requirements_file(
    src: Path,
    dst: Path,
    excluded_packages: Iterable[str] = DEFAULT_EXCLUDED_PACKAGES,
    excluded_prefixes: Iterable[str] = DEFAULT_EXCLUDED_PREFIXES,
) -> dict:
    """Filter ``src`` requirements.txt into ``dst``, returning summary + SHA-256."""
    raw_lines = Path(src).read_text(encoding="utf-8").splitlines()
    kept_lines: list[str] = []
    kept_reqs: list[str] = []
    excluded_reqs: list[str] = []
    for raw in raw_lines:
        line = raw.rstrip("\r\n")
        pkg = _extract_requirement_pkg_name(line)
        if pkg is not None and _is_excluded_package(pkg, excluded_packages, excluded_prefixes):
            excluded_reqs.append(line.strip())
            continue
        kept_lines.append(line)
        if pkg is not None:
            kept_reqs.append(line.strip())
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    content = ("\n".join(kept_lines) + "\n") if kept_lines else ""
    dst.write_text(content, encoding="utf-8")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return {
        "kept": kept_reqs,
        "excluded": excluded_reqs,
        "sha256": digest,
        "output_path": str(dst),
    }


def validate_wheelhouse_entry(
    entry: dict | None,
    expected_commit: str,
    expected_python_tag: str | None = None,
) -> tuple[bool, str]:
    """Validate a wheelhouse manifest entry against expected commit & Python tag."""
    if not isinstance(entry, dict):
        return False, "entry_not_dict"
    archive = entry.get("archive_name") or entry.get("filename") or entry.get("archive")
    if not isinstance(archive, str) or not archive.strip():
        return False, "missing_archive_name"
    sha = str(entry.get("sha256", "")).strip().lower()
    if not _SHA256_RE.match(sha):
        return False, "invalid_sha256"
    raw_size = entry.get("size_bytes", entry.get("size"))
    try:
        size_val = int(raw_size)
    except (TypeError, ValueError):
        return False, "invalid_size_bytes"
    if size_val <= 0:
        return False, "non_positive_size_bytes"
    entry_commit = str(entry.get("commit", "")).strip()
    if not expected_commit or entry_commit != expected_commit:
        return False, f"commit_mismatch:{entry_commit}!={expected_commit}"
    raw_pytag = entry.get("python_tag")
    try:
        entry_pytag = normalize_python_tag(raw_pytag)
    except ValueError:
        return False, f"invalid_python_tag:{raw_pytag!r}"
    if expected_python_tag is not None:
        try:
            exp_pytag = normalize_python_tag(expected_python_tag)
        except ValueError:
            return False, f"invalid_expected_python_tag:{expected_python_tag!r}"
        if entry_pytag != exp_pytag:
            return False, f"python_tag_mismatch:{entry_pytag}!={exp_pytag}"
    return True, "ok"


def select_wheelhouse_entry(
    manifest: dict | None,
    commit: str,
    python_tag: str | None = None,
) -> dict | None:
    """Return validated wheelhouse entry for ``(commit, python_tag)`` or ``None``."""
    if not isinstance(manifest, dict):
        return None
    wheelhouses = manifest.get("wheelhouses")
    if not isinstance(wheelhouses, dict):
        return None
    try:
        pytag = normalize_python_tag(python_tag)
    except ValueError:
        return None
    entry = wheelhouses.get(pytag)
    ok, _ = validate_wheelhouse_entry(entry, commit, pytag)
    return entry if ok else None


def build_wheelhouse_task_descriptor(
    manifest: dict | None,
    commit: str | None = None,
) -> dict[str, dict] | None:
    """Build ``pytag -> descriptor`` map for ``drive_download['wheelhouses']``.

    Returns ``None`` when no valid wheelhouse entry with a Drive ``file_id``
    matches ``commit`` (or ``manifest['commit']``).
    """
    if not isinstance(manifest, dict):
        return None
    target_commit = (commit or manifest.get("commit") or "").strip()
    if not target_commit:
        return None
    wheelhouses = manifest.get("wheelhouses")
    if not isinstance(wheelhouses, dict):
        return None
    descriptors: dict[str, dict] = {}
    for raw_tag, entry in sorted(wheelhouses.items()):
        if not isinstance(entry, dict):
            continue
        try:
            pytag = normalize_python_tag(entry.get("python_tag") or raw_tag)
        except ValueError:
            continue
        ok, _ = validate_wheelhouse_entry(entry, target_commit, pytag)
        if not ok:
            continue
        file_id = str(entry.get("file_id", "")).strip()
        if not file_id:
            continue
        archive = entry.get("archive_name") or entry.get("filename") or entry.get("archive")
        size_val = int(entry.get("size_bytes", entry.get("size")))
        descriptors[pytag] = {
            "file_id": file_id,
            "size": size_val,
            "sha256": str(entry["sha256"]).strip().lower(),
            "archive": str(archive),
            "commit": target_commit,
            "python_tag": pytag,
            "excluded_packages": list(entry.get("excluded_packages", DEFAULT_EXCLUDED_PACKAGES)),
            "excluded_prefixes": list(entry.get("excluded_prefixes", DEFAULT_EXCLUDED_PREFIXES)),
            "server_md5": str(entry.get("server_md5", "")),
        }
    return descriptors or None


# ---------------------------------------------------------------------------
# ComfyUI source archive build / verify / publish
# ---------------------------------------------------------------------------

def build_archive(repo_dir: Path, commit: str, out_path: Path) -> str:
    """git archive of <commit> (+ COMFYUI_COMMIT.txt marker), gzipped
    deterministically (mtime=0). Returns the SHA-256 of the .tar.gz."""
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), "archive", commit, "--prefix=comfyui/"],
        capture_output=True, check=True, timeout=600)
    raw_tar = proc.stdout
    if not raw_tar:
        raise RuntimeError("git archive produced no output for " + commit)
    with tempfile.TemporaryDirectory() as d:
        tar_path = Path(d) / "comfyui.tar"
        tar_path.write_bytes(raw_tar)
        with tarfile.open(tar_path, "a") as tf:
            data = (commit + "\n").encode()
            info = tarfile.TarInfo("comfyui/COMFYUI_COMMIT.txt")
            info.size = len(data)
            info.mtime = 0
            tf.addfile(info, io.BytesIO(data))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with tar_path.open("rb") as fi, out_path.open("wb") as fo:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=fo,
                compresslevel=9, mtime=0,
            ) as gz:
                for chunk in iter(lambda: fi.read(4 * 1024 * 1024), b""):
                    gz.write(chunk)
    return _file_digests(out_path)["sha256"]


def build_manifest(repo_url: str, commit: str, name: str, sha256: str,
                   size: int) -> dict:
    return {
        "repo": repo_url,
        "commit": commit,
        "archive_name": name,
        "sha256": sha256,
        "size_bytes": size,
        "created_utc": datetime.datetime.now(datetime.timezone.utc)
                       .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "git archive of the pinned ComfyUI commit, "
                  "plus COMFYUI_COMMIT.txt marker",
        "cache_folder": CACHE_FOLDER_NAME,
        "cache_drive_dir": CACHE_DRIVE_DIR,
    }


def verify_local_archive(repo_dir: Path, commit: str, archive_path: Path,
                         expected_sha256: str) -> dict:
    """Independent local check: SHA-256, extract, marker commit, and the
    extracted tree's files match the pinned commit's tree."""
    digests = _file_digests(archive_path)
    if digests["sha256"] != expected_sha256:
        raise RuntimeError("local archive SHA-256 mismatch")
    with tempfile.TemporaryDirectory() as d:
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(d, filter="data")
        marker = Path(d) / "comfyui" / "COMFYUI_COMMIT.txt"
        if not marker.is_file() or marker.read_text().strip() != commit:
            raise RuntimeError("local archive commit marker mismatch")
        for name in ("main.py", "requirements.txt"):
            if not (Path(d) / "comfyui" / name).is_file():
                raise RuntimeError("local archive missing " + name)
        # Tree identity: the extracted tree must equal the pinned commit's.
        proc = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", commit + "^{tree}"],
            capture_output=True, text=True, check=True, timeout=60)
        tree = proc.stdout.strip()
        listed = subprocess.run(
            ["git", "-C", str(repo_dir), "ls-tree", "-r", "--format=%(objectname) %(path)",
             commit],
            capture_output=True, text=True, check=True, timeout=60)
        expected = {}
        for line in listed.stdout.splitlines():
            oid, path = line.split(" ", 1)
            expected[path] = oid
        for path, oid in expected.items():
            data = (Path(d) / "comfyui" / path).read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if blob != oid:
                raise RuntimeError("extracted file differs from pinned tree: " + path)
    return {"archive_sha256": digests["sha256"], "size_bytes": digests["size"],
            "tree": tree, "files_checked": len(expected)}


# ---------------------------------------------------------------------------
# Offline wheelhouse build / verify / publish
# ---------------------------------------------------------------------------

def build_wheelhouse_manifest(
    commit: str,
    python_tag: str,
    archive: str,
    sha256: str,
    size_bytes: int,
    *,
    abi: str,
    platforms: Sequence[str],
    wheel_count: int,
    excluded_packages: Sequence[str],
    excluded_prefixes: Sequence[str],
    requirements_sha256: str,
) -> dict:
    return {
        "commit": commit,
        "python_tag": normalize_python_tag(python_tag),
        "abi": abi,
        "platforms": list(platforms),
        "archive_name": archive,
        "sha256": sha256,
        "size_bytes": int(size_bytes),
        "wheel_count": int(wheel_count),
        "excluded_packages": [normalize_pkg_name(p) for p in excluded_packages],
        "excluded_prefixes": [normalize_pkg_name(p) for p in excluded_prefixes],
        "requirements_sha256": requirements_sha256,
        "created_utc": datetime.datetime.now(datetime.timezone.utc)
                       .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cache_folder": CACHE_FOLDER_NAME,
        "cache_drive_dir": CACHE_DRIVE_DIR,
    }


def _pack_wheelhouse_tar_gz(
    wheel_paths: Sequence[Path],
    meta: dict,
    out_path: Path,
) -> str:
    """Create a deterministic ``wheelhouse/`` .tar.gz archive (mtime=0)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_bytes = (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode("utf-8")
    sorted_wheels = sorted(wheel_paths, key=lambda p: p.name)
    with tempfile.TemporaryDirectory() as d:
        tar_path = Path(d) / "wheelhouse.tar"
        with tarfile.open(tar_path, "w") as tf:
            meta_info = tarfile.TarInfo(f"wheelhouse/{WHEELHOUSE_META_FILENAME}")
            meta_info.size = len(meta_bytes)
            meta_info.mtime = 0
            meta_info.mode = 0o644
            meta_info.uid = 0
            meta_info.gid = 0
            meta_info.uname = ""
            meta_info.gname = ""
            tf.addfile(meta_info, io.BytesIO(meta_bytes))
            for whl in sorted_wheels:
                data = whl.read_bytes()
                info = tarfile.TarInfo(f"wheelhouse/{whl.name}")
                info.size = len(data)
                info.mtime = 0
                info.mode = 0o644
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                tf.addfile(info, io.BytesIO(data))
        with tar_path.open("rb") as fi, out_path.open("wb") as fo:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=fo,
                compresslevel=9, mtime=0,
            ) as gz:
                for chunk in iter(lambda: fi.read(4 * 1024 * 1024), b""):
                    gz.write(chunk)
    return _file_digests(out_path)["sha256"]


def verify_local_wheelhouse(
    archive_path: Path,
    expected_sha256: str,
    expected_commit: str,
    expected_python_tag: str,
) -> dict:
    """Verify wheelhouse archive SHA-256, metadata, wheel list, and exclusion invariants."""
    digests = _file_digests(archive_path)
    if digests["sha256"] != expected_sha256:
        raise RuntimeError("local wheelhouse SHA-256 mismatch")
    exp_pytag = normalize_python_tag(expected_python_tag)
    with tempfile.TemporaryDirectory() as d:
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(d, filter="data")
        wh_dir = Path(d) / "wheelhouse"
        meta_path = wh_dir / WHEELHOUSE_META_FILENAME
        if not meta_path.is_file():
            raise RuntimeError("local wheelhouse missing " + WHEELHOUSE_META_FILENAME)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("commit") != expected_commit:
            raise RuntimeError(
                f"local wheelhouse commit mismatch: {meta.get('commit')} != {expected_commit}"
            )
        actual_pytag = normalize_python_tag(meta.get("python_tag"))
        if actual_pytag != exp_pytag:
            raise RuntimeError(
                f"local wheelhouse python_tag mismatch: {actual_pytag} != {exp_pytag}"
            )
        excluded_pkgs = meta.get("excluded_packages", list(DEFAULT_EXCLUDED_PACKAGES))
        excluded_prefs = meta.get("excluded_prefixes", list(DEFAULT_EXCLUDED_PREFIXES))
        extracted_wheels = sorted(p.name for p in wh_dir.glob("*.whl"))
        if not extracted_wheels:
            raise RuntimeError("local wheelhouse contains no .whl files")
        if len(extracted_wheels) != int(meta.get("wheel_count", -1)):
            raise RuntimeError("local wheelhouse wheel_count mismatch")
        if extracted_wheels != sorted(meta.get("wheels", [])):
            raise RuntimeError("local wheelhouse wheel list mismatch")
        for whl_name in extracted_wheels:
            dist = _wheel_dist_name(whl_name)
            if _is_excluded_package(dist, excluded_pkgs, excluded_prefs):
                raise RuntimeError(f"local wheelhouse contains excluded wheel: {whl_name}")
    return {
        "archive_sha256": digests["sha256"],
        "size_bytes": digests["size"],
        "commit": expected_commit,
        "python_tag": exp_pytag,
        "wheel_count": len(extracted_wheels),
    }


def build_wheelhouse(
    commit: str,
    *,
    requirements_path: Path | None = None,
    repo_dir: Path | None = None,
    python_tag: str = "py312",
    out_dir: Path = STAGING,
    platforms: Sequence[str] = DEFAULT_PLATFORMS,
    excluded_packages: Sequence[str] = DEFAULT_EXCLUDED_PACKAGES,
    excluded_prefixes: Sequence[str] = DEFAULT_EXCLUDED_PREFIXES,
    extra_index_url: str | None = None,
    wheels_dir: Path | None = None,
    pip_runner=None,
) -> dict:
    """Build deterministic offline ComfyUI wheelhouse tarball + .sha256 + .manifest.json."""
    if not commit or not str(commit).strip():
        raise ValueError("commit is required")
    commit = str(commit).strip()
    pytag = normalize_python_tag(python_tag)
    py_ver, abi = _python_tag_to_version_abi(pytag)

    if requirements_path is None:
        if repo_dir is None:
            raise ValueError("Either requirements_path or repo_dir must be provided")
        repo_dir = Path(repo_dir)
        head = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=60,
        ).stdout.strip()
        if head != commit:
            raise RuntimeError("repo HEAD " + head + " != pinned commit " + commit)
        requirements_path = repo_dir / "requirements.txt"
    requirements_path = Path(requirements_path)
    if not requirements_path.is_file():
        raise FileNotFoundError(f"requirements.txt not found: {requirements_path}")

    norm_excl_pkgs = tuple(normalize_pkg_name(p) for p in excluded_packages if str(p).strip())
    norm_excl_prefs = tuple(normalize_pkg_name(p) for p in excluded_prefixes if str(p).strip())
    plat_list = tuple(str(p).strip() for p in platforms if str(p).strip()) or DEFAULT_PLATFORMS

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        filtered_req = tmp_path / "requirements.filtered.txt"
        filter_info = filter_requirements_file(
            requirements_path,
            filtered_req,
            excluded_packages=norm_excl_pkgs,
            excluded_prefixes=norm_excl_prefs,
        )
        if wheels_dir is not None:
            src_wheels_dir = Path(wheels_dir)
            if not src_wheels_dir.is_dir():
                raise FileNotFoundError(f"wheels_dir not found: {src_wheels_dir}")
        else:
            src_wheels_dir = tmp_path / "wheels"
            src_wheels_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, "-m", "pip", "download",
                "--only-binary=:all:",
                "--python-version", py_ver,
                "--implementation", "cp",
                "--abi", abi,
            ]
            for plat in plat_list:
                cmd.extend(["--platform", plat])
            if extra_index_url:
                cmd.extend(["--extra-index-url", extra_index_url])
            cmd.extend(["-r", str(filtered_req), "-d", str(src_wheels_dir)])
            if pip_runner is not None:
                pip_runner(cmd)
            else:
                subprocess.run(cmd, check=True, timeout=1800)

        all_wheels = sorted(src_wheels_dir.glob("*.whl"), key=lambda p: p.name)
        kept_wheels: list[Path] = []
        pruned_wheels: list[str] = []
        for whl in all_wheels:
            dist = _wheel_dist_name(whl.name)
            if _is_excluded_package(dist, norm_excl_pkgs, norm_excl_prefs):
                pruned_wheels.append(whl.name)
            else:
                kept_wheels.append(whl)
        if not kept_wheels:
            raise RuntimeError("No non-excluded .whl files collected for wheelhouse")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = wheelhouse_archive_name(commit, pytag)
        archive_path = out_dir / name
        meta = {
            "commit": commit,
            "python_tag": pytag,
            "abi": abi,
            "platforms": list(plat_list),
            "excluded_packages": list(norm_excl_pkgs),
            "excluded_prefixes": list(norm_excl_prefs),
            "requirements_sha256": filter_info["sha256"],
            "wheel_count": len(kept_wheels),
            "wheels": [p.name for p in kept_wheels],
        }
        sha256 = _pack_wheelhouse_tar_gz(kept_wheels, meta, archive_path)
        digests = _file_digests(archive_path)
        verify_local_wheelhouse(archive_path, sha256, commit, pytag)

        sha_path = out_dir / f"{name}.sha256"
        sha_path.write_text(f"{sha256}  {name}\n", encoding="utf-8")

        wh_manifest = build_wheelhouse_manifest(
            commit,
            pytag,
            name,
            sha256,
            digests["size"],
            abi=abi,
            platforms=plat_list,
            wheel_count=len(kept_wheels),
            excluded_packages=norm_excl_pkgs,
            excluded_prefixes=norm_excl_prefs,
            requirements_sha256=filter_info["sha256"],
        )
        m_name = wheelhouse_manifest_name(commit, pytag)
        manifest_path = out_dir / m_name
        manifest_path.write_text(
            json.dumps(wh_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return {
        "manifest": wh_manifest,
        "archive_name": name,
        "manifest_name": m_name,
        "archive_path": archive_path,
        "sha256_path": sha_path,
        "manifest_path": manifest_path,
        "pruned_wheels": pruned_wheels,
        "excluded_requirements": filter_info["excluded"],
    }


def publish(repo_dir: Path, commit: str, colab_auth: str) -> dict:
    """Build the archive locally, create the Drive folder, upload the three
    files, re-verify each from server metadata, and write the local
    manifest with Drive ids. Never touches the model depot or READY."""
    from drive_api import verify_uploaded_file
    from drive_depot import (
        account_matches,
        atomic_json,
        drive_service,
        ensure_folder,
        upload_one,
    )

    if not (repo_dir / ".git").exists():
        raise RuntimeError("repo dir has no .git: " + str(repo_dir))
    head = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True,
                          timeout=60).stdout.strip()
    if head != commit:
        raise RuntimeError("repo HEAD " + head + " != pinned commit " + commit)
    service = drive_service()
    if not account_matches(service, colab_auth):
        raise RuntimeError("ACCOUNT_MISMATCH")
    STAGING.mkdir(parents=True, exist_ok=True)
    name = archive_name(commit)
    archive_path = STAGING / name
    sha256 = build_archive(repo_dir, commit, archive_path)
    digests = _file_digests(archive_path)
    verify_local_archive(repo_dir, commit, archive_path, sha256)
    sha_path = STAGING / (name + ".sha256")
    sha_path.write_text(sha256 + "  " + name + "\n")
    manifest = build_manifest(COMFYUI_REPO_URL, commit, name, sha256,
                              digests["size"])
    manifest_path = STAGING / "comfyui-source-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    folder_id = ensure_folder(service, None, CACHE_FOLDER_NAME,
                              allow_write=True)
    entries = {}
    for local_path, rel in ((archive_path, name),
                            (sha_path, name + ".sha256"),
                            (manifest_path, "comfyui-source-manifest.json")):
        entry = upload_one(service, local_path, folder_id, rel, rel,
                           _file_digests(local_path)["sha256"], commit)
        entries[rel] = entry
    # Independent re-verification from server metadata for every file.
    for rel, entry in entries.items():
        verify_uploaded_file(service, entry["drive_file_id"],
                             STAGING / rel)
    # Preserve any existing wheelhouse metadata in local manifest if same commit.
    existing_wheelhouses = None
    existing_drive_files: dict = {}
    if LOCAL_MANIFEST.is_file():
        try:
            prev = json.loads(LOCAL_MANIFEST.read_text(encoding="utf-8"))
            if prev.get("commit") == commit and isinstance(prev.get("wheelhouses"), dict):
                existing_wheelhouses = prev["wheelhouses"]
                for k, v in (prev.get("drive_files") or {}).items():
                    if k.startswith("comfyui-wheelhouse-"):
                        existing_drive_files[k] = v
        except Exception:
            pass
    manifest["drive_folder_id"] = folder_id
    manifest["drive_files"] = {
        **existing_drive_files,
        **{
            rel: {"id": e["drive_file_id"], "size": e["size"],
                  "sha256": e["sha256"], "server_md5": e["server_md5"]}
            for rel, e in entries.items()
        },
    }
    if existing_wheelhouses:
        manifest["wheelhouses"] = existing_wheelhouses
    atomic_json(LOCAL_MANIFEST, manifest)
    os.chmod(LOCAL_MANIFEST, 0o600)
    return manifest


def publish_wheelhouse(
    commit: str,
    colab_auth: str,
    *,
    requirements_path: Path | None = None,
    repo_dir: Path | None = None,
    python_tag: str = "py312",
    platforms: Sequence[str] = DEFAULT_PLATFORMS,
    excluded_packages: Sequence[str] = DEFAULT_EXCLUDED_PACKAGES,
    excluded_prefixes: Sequence[str] = DEFAULT_EXCLUDED_PREFIXES,
    extra_index_url: str | None = None,
    wheels_dir: Path | None = None,
    pip_runner=None,
) -> dict:
    """Build wheelhouse locally, upload 3 immutable files to Muse-H3-ComfyUI-Source,
    re-verify from Drive metadata, and merge into local comfyui_source_manifest.json."""
    from drive_api import verify_uploaded_file
    from drive_depot import (
        account_matches,
        atomic_json,
        drive_service,
        ensure_folder,
        upload_one,
    )

    service = drive_service()
    if not account_matches(service, colab_auth):
        raise RuntimeError("ACCOUNT_MISMATCH")

    built = build_wheelhouse(
        commit,
        requirements_path=requirements_path,
        repo_dir=repo_dir,
        python_tag=python_tag,
        out_dir=STAGING,
        platforms=platforms,
        excluded_packages=excluded_packages,
        excluded_prefixes=excluded_prefixes,
        extra_index_url=extra_index_url,
        wheels_dir=wheels_dir,
        pip_runner=pip_runner,
    )
    wh_manifest = built["manifest"]
    name = built["archive_name"]
    m_name = built["manifest_name"]
    sha_name = f"{name}.sha256"
    pytag = wh_manifest["python_tag"]

    folder_id = ensure_folder(service, None, CACHE_FOLDER_NAME, allow_write=True)
    entries = {}
    for local_path, rel in (
        (built["archive_path"], name),
        (built["sha256_path"], sha_name),
        (built["manifest_path"], m_name),
    ):
        entry = upload_one(
            service, local_path, folder_id, rel, rel,
            _file_digests(local_path)["sha256"], commit,
        )
        entries[rel] = entry

    for rel, entry in entries.items():
        verify_uploaded_file(service, entry["drive_file_id"], STAGING / rel)

    if LOCAL_MANIFEST.is_file():
        local_doc = json.loads(LOCAL_MANIFEST.read_text(encoding="utf-8"))
    else:
        local_doc = {
            "repo": COMFYUI_REPO_URL,
            "commit": commit,
            "cache_folder": CACHE_FOLDER_NAME,
            "cache_drive_dir": CACHE_DRIVE_DIR,
        }

    local_doc["drive_folder_id"] = folder_id
    drive_files = dict(local_doc.get("drive_files") or {})
    for rel, e in entries.items():
        drive_files[rel] = {
            "id": e["drive_file_id"],
            "size": e["size"],
            "sha256": e["sha256"],
            "server_md5": e["server_md5"],
        }
    local_doc["drive_files"] = drive_files

    wheelhouses = dict(local_doc.get("wheelhouses") or {})
    wheelhouses[pytag] = {
        **wh_manifest,
        "file_id": entries[name]["drive_file_id"],
        "server_md5": entries[name]["server_md5"],
        "sha256_file_id": entries[sha_name]["drive_file_id"],
        "manifest_file_id": entries[m_name]["drive_file_id"],
    }
    local_doc["wheelhouses"] = wheelhouses

    atomic_json(LOCAL_MANIFEST, local_doc)
    os.chmod(LOCAL_MANIFEST, 0o600)
    return local_doc


def verify(colab_auth: str) -> dict:
    """Read-only: re-check the Drive cache files against the local manifest."""
    from drive_api import verify_uploaded_file
    from drive_depot import account_matches, drive_service

    if not LOCAL_MANIFEST.exists():
        raise RuntimeError("local manifest missing: " + str(LOCAL_MANIFEST))
    manifest = json.loads(LOCAL_MANIFEST.read_text())
    service = drive_service()
    if not account_matches(service, colab_auth):
        raise RuntimeError("ACCOUNT_MISMATCH")
    result = {"folder_id": manifest.get("drive_folder_id"), "files": {}}
    for rel, info in manifest.get("drive_files", {}).items():
        proof = verify_uploaded_file(
            service, info["id"], STAGING / rel)
        result["files"][rel] = {
            "drive_file_id": info["id"], "size": proof["size"],
            "sha256": proof["sha256"], "server_verified": True}
    return result


def _add_wheelhouse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--commit", required=True)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", help="Local ComfyUI git repo directory (uses <repo>/requirements.txt)")
    src.add_argument("--requirements", help="Explicit path to requirements.txt")
    parser.add_argument("--python-tag", default="py312", help="Target Colab Python tag (e.g. py311, py312)")
    parser.add_argument(
        "--platform",
        action="append",
        dest="platforms",
        default=None,
        help="Target wheel platform tag (repeatable; default manylinux2014_x86_64/manylinux_2_17_x86_64/linux_x86_64)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        dest="excludes",
        default=None,
        help="Additional package name to exclude (repeatable)",
    )
    parser.add_argument("--extra-index-url", default=None)
    parser.add_argument("--wheels-dir", default=None, help="Pre-downloaded wheel directory (skips pip download)")


def cli(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pinned ComfyUI source & wheelhouse cache")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--repo", required=True)
    b.add_argument("--commit", required=True)

    pub = sub.add_parser("publish")
    pub.add_argument("--repo", required=True)
    pub.add_argument("--commit", required=True)
    pub.add_argument("--colab-auth", required=True)
    pub.add_argument("--allow-drive-write", action="store_true")

    bw = sub.add_parser("build-wheelhouse")
    _add_wheelhouse_args(bw)
    bw.add_argument("--out-dir", default=str(STAGING))

    pw = sub.add_parser("publish-wheelhouse")
    _add_wheelhouse_args(pw)
    pw.add_argument("--colab-auth", required=True)
    pw.add_argument("--allow-drive-write", action="store_true")

    v = sub.add_parser("verify")
    v.add_argument("--colab-auth", required=True)

    args = p.parse_args(argv)
    if args.cmd == "build":
        repo = Path(args.repo)
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True,
                              timeout=60).stdout.strip()
        if head != args.commit:
            raise RuntimeError("repo HEAD " + head + " != pinned commit " + args.commit)
        STAGING.mkdir(parents=True, exist_ok=True)
        name = archive_name(args.commit)
        archive_path = STAGING / name
        sha256 = build_archive(repo, args.commit, archive_path)
        digests = _file_digests(archive_path)
        checked = verify_local_archive(repo, args.commit, archive_path, sha256)
        (STAGING / (name + ".sha256")).write_text(sha256 + "  " + name + "\n")
        manifest = build_manifest(COMFYUI_REPO_URL, args.commit, name,
                                  sha256, digests["size"])
        (STAGING / "comfyui-source-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"archive": name, "sha256": sha256,
                          "size_bytes": digests["size"],
                          "tree": checked["tree"],
                          "files_checked": checked["files_checked"]},
                         indent=2))
        return 0

    if args.cmd == "publish":
        if not args.allow_drive_write:
            p.error("Mutating Drive requires --allow-drive-write")
        manifest = publish(Path(args.repo), args.commit, args.colab_auth)
        print(json.dumps({"folder": manifest["cache_folder"],
                          "drive_folder_id": manifest["drive_folder_id"],
                          "archive": manifest["archive_name"],
                          "sha256": manifest["sha256"],
                          "size_bytes": manifest["size_bytes"],
                          "local_manifest": str(LOCAL_MANIFEST)},
                         indent=2))
        return 0

    if args.cmd == "build-wheelhouse":
        excl = tuple(DEFAULT_EXCLUDED_PACKAGES) + tuple(args.excludes or ())
        plats = tuple(args.platforms) if args.platforms else DEFAULT_PLATFORMS
        built = build_wheelhouse(
            args.commit,
            requirements_path=Path(args.requirements) if args.requirements else None,
            repo_dir=Path(args.repo) if args.repo else None,
            python_tag=args.python_tag,
            out_dir=Path(args.out_dir),
            platforms=plats,
            excluded_packages=excl,
            extra_index_url=args.extra_index_url,
            wheels_dir=Path(args.wheels_dir) if args.wheels_dir else None,
        )
        m = built["manifest"]
        print(json.dumps({
            "archive": m["archive_name"],
            "python_tag": m["python_tag"],
            "commit": m["commit"],
            "sha256": m["sha256"],
            "size_bytes": m["size_bytes"],
            "wheel_count": m["wheel_count"],
            "pruned_wheels": built["pruned_wheels"],
            "out_dir": str(args.out_dir),
        }, indent=2))
        return 0

    if args.cmd == "publish-wheelhouse":
        if not args.allow_drive_write:
            p.error("Mutating Drive requires --allow-drive-write")
        excl = tuple(DEFAULT_EXCLUDED_PACKAGES) + tuple(args.excludes or ())
        plats = tuple(args.platforms) if args.platforms else DEFAULT_PLATFORMS
        local_doc = publish_wheelhouse(
            args.commit,
            args.colab_auth,
            requirements_path=Path(args.requirements) if args.requirements else None,
            repo_dir=Path(args.repo) if args.repo else None,
            python_tag=args.python_tag,
            platforms=plats,
            excluded_packages=excl,
            extra_index_url=args.extra_index_url,
            wheels_dir=Path(args.wheels_dir) if args.wheels_dir else None,
        )
        pytag = normalize_python_tag(args.python_tag)
        wh = local_doc["wheelhouses"][pytag]
        print(json.dumps({
            "folder": CACHE_FOLDER_NAME,
            "drive_folder_id": local_doc["drive_folder_id"],
            "archive": wh["archive_name"],
            "python_tag": wh["python_tag"],
            "file_id": wh["file_id"],
            "sha256": wh["sha256"],
            "size_bytes": wh["size_bytes"],
            "wheel_count": wh["wheel_count"],
            "local_manifest": str(LOCAL_MANIFEST),
        }, indent=2))
        return 0

    if args.cmd == "verify":
        result = verify(args.colab_auth)
        print(json.dumps(result, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(cli())
