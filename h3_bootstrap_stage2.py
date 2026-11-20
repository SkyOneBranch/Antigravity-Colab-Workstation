"""Remote Stage-2 bundled bootstrap extractor & worker launcher (v0.3.0 M2).

Executed via a single `colab exec -f` call after Stage 1 passes and
`/content/muse_h3/bootstrap_stage2.tar.gz` has been uploaded (also used for
M4 warm-session lightweight dispatch).

Security invariants:
- Rejects absolute paths, `..` traversal, symlinks, hardlinks, device nodes,
  and any member resolving outside `/content/muse_h3`.
- Enforces mode `0o600` on every extracted `*token*.json` file.
- Unlinks `/content/muse_h3/bootstrap_stage2.tar.gz` in `finally` even on error.
- Invokes `h3_worker_launcher.main()` which prints `MUSE_H3_LAUNCH_JSON:...`.
"""

from __future__ import annotations

import os
import sys
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path("/content/muse_h3")
DEFAULT_TAR_PATH = ROOT / "bootstrap_stage2.tar.gz"


def _is_token_file(name: str) -> bool:
    lower = name.lower()
    return "token" in lower and lower.endswith(".json")


def safe_extract_tar(tar_path: Path, dest_dir: Path) -> list[str]:
    """Safely extract regular files and directories from ``tar_path`` into
    ``dest_dir``, rejecting any path traversal or link member."""
    dest_resolved = dest_dir.resolve()
    dest_resolved.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            raw_name = member.name or ""
            posix = PurePosixPath(raw_name)
            if not raw_name or raw_name.startswith("/") or posix.is_absolute():
                raise ValueError(f"Unsafe tar member (absolute path): {raw_name!r}")
            if any(part == ".." for part in posix.parts):
                raise ValueError(f"Unsafe tar member (path traversal): {raw_name!r}")
            if member.issym() or member.islnk():
                raise ValueError(f"Unsafe tar member (symlink/hardlink): {raw_name!r}")
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"Unsafe tar member (unsupported type): {raw_name!r}")
            target = (dest_resolved / Path(*posix.parts)).resolve()
            if not target.is_relative_to(dest_resolved):
                raise ValueError(f"Unsafe tar member (escapes destination): {raw_name!r}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                raise ValueError(f"Cannot read tar member: {raw_name!r}")
            data = src.read()
            if _is_token_file(target.name):
                fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                try:
                    os.write(fd, data)
                finally:
                    os.close(fd)
                os.chmod(target, 0o600)
            else:
                target.write_bytes(data)
            extracted.append(str(posix))
    return extracted


def main(root: Path | None = None, tar_path: Path | None = None) -> None:
    target_root = Path(root) if root is not None else ROOT
    bundle_path = Path(tar_path) if tar_path is not None else (target_root / "bootstrap_stage2.tar.gz")
    target_root.mkdir(parents=True, exist_ok=True)
    (target_root / "output").mkdir(parents=True, exist_ok=True)
    try:
        if not bundle_path.is_file():
            raise FileNotFoundError(f"Stage 2 bundle missing: {bundle_path}")
        safe_extract_tar(bundle_path, target_root)
    finally:
        try:
            if bundle_path.exists():
                bundle_path.unlink()
        except OSError:
            pass
    if str(target_root) not in sys.path:
        sys.path.insert(0, str(target_root))
    import h3_worker_launcher

    h3_worker_launcher.ROOT = target_root
    h3_worker_launcher.main()


if __name__ == "__main__":
    main()
