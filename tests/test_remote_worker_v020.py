"""Offline 0-CU unit tests for v0.2.0 fast cold-start features in h3_remote_worker.py.

Covers:
- F1: Controlled multi-file weight restore into independent NVMe dir + abort_event
- F2: Offline wheelhouse pip install (--no-index --find-links) + filtered requirements + PyPI fallback
- F3: Dual-track concurrency (ensure_comfy || restore_weights), fast abort propagation,
      accurate failing stage in WORKER_FAILED_JSON, thread-safe token refresh throttling
- F4: Atomic link_models_into_comfy() symlinks with boundary/readability checks +
      GPU VRAM detection & --highvram --disable-mmap flag resolution / env override
- F5: COLDSTART_METRICS_JSON locked schema & field validation
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import h3_remote_worker as w


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_wheelhouse_tarball(out_path: Path, commit: str, python_tag: str, include_wheel: bool = True) -> tuple[str, int]:
    """Create a deterministic wheelhouse .tar.gz with wheelhouse/*.whl + WHEELHOUSE_META.json."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tf:
        if include_wheel:
            whl_bytes = b"PK\x03\x04fake-wheel-content"
            info = tarfile.TarInfo("wheelhouse/dummy_pkg-0.1.0-py3-none-any.whl")
            info.size = len(whl_bytes)
            info.mtime = 0
            tf.addfile(info, io.BytesIO(whl_bytes))
        meta = json.dumps({"commit": commit, "python_tag": python_tag}).encode("utf-8")
        minfo = tarfile.TarInfo("wheelhouse/WHEELHOUSE_META.json")
        minfo.size = len(meta)
        minfo.mtime = 0
        tf.addfile(minfo, io.BytesIO(meta))
    raw = out_path.read_bytes()
    return _sha256_bytes(raw), len(raw)


def _sample_manifest(content: bytes) -> tuple[dict, list[str]]:
    paths = [
        "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "vae/minimax_h3_video_vae_int8_convrot.safetensors",
        "vae/minimax_h3_audio_vae_fp32.safetensors",
    ]
    digest = _sha256_bytes(content)
    manifest = {
        "status": "READY",
        "verification_method": "drive_v3_server_api",
        "account_verified": True,
        "verified_at_utc": "2026-11-10T22:36:00Z",
        "files": [
            {
                "path": p,
                "size": len(content),
                "sha256": digest,
                "server_md5": hashlib.md5(content).hexdigest(),
                "drive_file_id": f"fid-{i}",
                "server_verified": True,
                "hf_revision": "54472221d27c695fca9621b0c3ed52f06c4af707",
            }
            for i, p in enumerate(paths)
        ],
    }
    return manifest, paths


class _DummyTokenProvider:
    def __init__(self, tid: str = "tok-id-0001"):
        self.token_id = tid

    def get_token(self) -> str:
        return "short"


class TestWheelhouseF2(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.td, ignore_errors=True))
        self.base = Path(self.td)
        self.commit = "61e524bdba7a4e193b45c2697f1af83c1beb86d6"
        self.pytag = w._current_python_tag()
        self.comfy = self.base / "ComfyUI"
        self.comfy.mkdir(parents=True, exist_ok=True)
        (self.comfy / "COMFYUI_COMMIT.txt").write_text(self.commit + "\n", encoding="utf-8")
        (self.comfy / "main.py").write_text("print('ok')\n", encoding="utf-8")
        (self.comfy / "requirements.txt").write_text(
            "# comment\ntorch>=2.4.0\ntorchvision\ntorchaudio\ntriton>=3.0\n"
            "nvidia-cublas-cu12==12.1.3.1\neinops>=0.7.0\nsafetensors>=0.4.2\n",
            encoding="utf-8",
        )
        self.old_root = w.ROOT
        w.ROOT = self.base / "worker_root"
        w.ROOT.mkdir(parents=True, exist_ok=True)
        self.addCleanup(self._restore_root)

    def _restore_root(self):
        w.ROOT = self.old_root

    def test_filter_requirements_excludes_preinstalled_and_prefixes(self):
        lines = [
            "# header",
            "",
            "torch>=2.5.0",
            "Torch[cuda]==2.5.1 ; platform_system == 'Linux'",
            "torchvision>=0.20",
            "torchaudio",
            "triton==3.1.0",
            "nvidia-cublas-cu12>=12.1",
            "nvidia_nccl_cu12==2.20.5",
            "einops>=0.7.0",
            "safetensors>=0.4.2",
            "aiohttp>=3.9",
        ]
        kept = w.filter_requirements_lines(lines)
        self.assertEqual(kept, ["einops>=0.7.0", "safetensors>=0.4.2", "aiohttp>=3.9"])

    def test_wheelhouse_happy_path_offline_pip(self):
        tar_src = self.base / f"comfyui-wheelhouse-{self.commit}-{self.pytag}.tar.gz"
        sha, size = _make_wheelhouse_tarball(tar_src, self.commit, self.pytag)
        tar_bytes = tar_src.read_bytes()

        class _FakeDownloader:
            def __init__(self, tp, **kw):
                pass

            def download_file(self, file_id, expected_size, expected_sha256, dest, **kw):
                dest = Path(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(tar_bytes)
                return {"drive_file_id": file_id, "size": expected_size, "sha256": expected_sha256, "path": str(dest)}

        drive_dl = {
            "wheelhouses": {
                self.pytag: {
                    "file_id": "fid-whl-1",
                    "size": size,
                    "sha256": sha,
                    "archive": tar_src.name,
                    "commit": self.commit,
                    "python_tag": self.pytag,
                    "excluded_packages": ["torch", "torchvision", "torchaudio", "triton"],
                    "excluded_prefixes": ["nvidia-"],
                }
            }
        }
        cmds = []
        metrics = {}
        buf = io.StringIO()
        with mock.patch("h3_drive_download.DriveDownloader", _FakeDownloader), \
                mock.patch.object(w, "verify_comfyui_commit", return_value=True), \
                mock.patch.object(w, "pip_install_needed", return_value=True), \
                mock.patch.object(w, "run_cmd", side_effect=lambda args, **kw: cmds.append(list(args))):
            (self.comfy / ".git").mkdir(exist_ok=True)
            with redirect_stdout(buf):
                w.ensure_comfy(self.comfy, self.commit, drive_download=drive_dl,
                               token_provider=_DummyTokenProvider(), metrics=metrics)
        self.assertEqual(len(cmds), 1)
        self.assertIn("--no-index", cmds[0])
        self.assertTrue(any(a.startswith("--find-links=") for a in cmds[0]))
        filtered_req = Path(cmds[0][-1]).read_text(encoding="utf-8").splitlines()
        self.assertEqual(filtered_req, ["einops>=0.7.0", "safetensors>=0.4.2"])
        self.assertTrue(metrics["wheelhouse_used"])
        self.assertIsNone(metrics["wheelhouse_fallback_reason"])
        self.assertIn("WHEELHOUSE_INSTALL_OK", buf.getvalue())
        self.assertNotIn("WHEELHOUSE_FALLBACK_PYPI", buf.getvalue())

    def test_wheelhouse_single_entry_compat(self):
        tar_src = self.base / f"comfyui-wheelhouse-{self.commit}-{self.pytag}.tar.gz"
        sha, size = _make_wheelhouse_tarball(tar_src, self.commit, self.pytag)
        tar_bytes = tar_src.read_bytes()

        class _FakeDownloader:
            def __init__(self, tp, **kw):
                pass

            def download_file(self, file_id, expected_size, expected_sha256, dest, **kw):
                Path(dest).parent.mkdir(parents=True, exist_ok=True)
                Path(dest).write_bytes(tar_bytes)
                return {"drive_file_id": file_id, "size": expected_size, "sha256": expected_sha256, "path": str(dest)}

        drive_dl = {
            "wheelhouse": {
                "file_id": "fid-whl-single",
                "size": size,
                "sha256": sha,
                "archive": tar_src.name,
                "commit": self.commit,
                "python_tag": self.pytag,
            }
        }
        cmds = []
        metrics = {}
        with mock.patch("h3_drive_download.DriveDownloader", _FakeDownloader), \
                mock.patch.object(w, "verify_comfyui_commit", return_value=True), \
                mock.patch.object(w, "pip_install_needed", return_value=True), \
                mock.patch.object(w, "run_cmd", side_effect=lambda args, **kw: cmds.append(list(args))):
            (self.comfy / ".git").mkdir(exist_ok=True)
            w.ensure_comfy(self.comfy, self.commit, drive_download=drive_dl,
                           token_provider=_DummyTokenProvider(), metrics=metrics)
        self.assertTrue(metrics["wheelhouse_used"])
        self.assertIn("--no-index", cmds[0])

    def test_wheelhouse_missing_falls_back_to_pypi(self):
        cmds = []
        metrics = {}
        buf = io.StringIO()
        with mock.patch.object(w, "verify_comfyui_commit", return_value=True), \
                mock.patch.object(w, "pip_install_needed", return_value=True), \
                mock.patch.object(w, "run_cmd", side_effect=lambda args, **kw: cmds.append(list(args))):
            (self.comfy / ".git").mkdir(exist_ok=True)
            with redirect_stdout(buf):
                w.ensure_comfy(self.comfy, self.commit, drive_download={},
                               token_provider=_DummyTokenProvider(), metrics=metrics)
        self.assertFalse(metrics["wheelhouse_used"])
        self.assertEqual(metrics["wheelhouse_fallback_reason"], "wheelhouse_missing")
        self.assertIn("WHEELHOUSE_FALLBACK_PYPI:", buf.getvalue())
        self.assertEqual(len(cmds), 1)
        self.assertNotIn("--no-index", cmds[0])

    def test_wheelhouse_python_tag_mismatch_falls_back(self):
        drive_dl = {
            "wheelhouses": {
                "py399": {
                    "file_id": "fid-whl-other",
                    "size": 123,
                    "sha256": "a" * 64,
                    "archive": "comfyui-wheelhouse-other.tar.gz",
                    "commit": self.commit,
                    "python_tag": "py399",
                }
            }
        }
        cmds = []
        metrics = {}
        buf = io.StringIO()
        with mock.patch.object(w, "verify_comfyui_commit", return_value=True), \
                mock.patch.object(w, "pip_install_needed", return_value=True), \
                mock.patch.object(w, "run_cmd", side_effect=lambda args, **kw: cmds.append(list(args))):
            (self.comfy / ".git").mkdir(exist_ok=True)
            with redirect_stdout(buf):
                w.ensure_comfy(self.comfy, self.commit, drive_download=drive_dl,
                               token_provider=_DummyTokenProvider(), metrics=metrics)
        self.assertFalse(metrics["wheelhouse_used"])
        self.assertIn("python_tag_mismatch", metrics["wheelhouse_fallback_reason"])
        self.assertIn("WHEELHOUSE_FALLBACK_PYPI:", buf.getvalue())
        self.assertNotIn("--no-index", cmds[0])

    def test_wheelhouse_offline_pip_failure_falls_back(self):
        tar_src = self.base / f"comfyui-wheelhouse-{self.commit}-{self.pytag}.tar.gz"
        sha, size = _make_wheelhouse_tarball(tar_src, self.commit, self.pytag)
        tar_bytes = tar_src.read_bytes()

        class _FakeDownloader:
            def __init__(self, tp, **kw):
                pass

            def download_file(self, file_id, expected_size, expected_sha256, dest, **kw):
                Path(dest).parent.mkdir(parents=True, exist_ok=True)
                Path(dest).write_bytes(tar_bytes)
                return {"drive_file_id": file_id, "size": expected_size, "sha256": expected_sha256, "path": str(dest)}

        drive_dl = {
            "wheelhouses": {
                self.pytag: {
                    "file_id": "fid-whl-1",
                    "size": size,
                    "sha256": sha,
                    "archive": tar_src.name,
                    "commit": self.commit,
                    "python_tag": self.pytag,
                }
            }
        }
        cmds = []

        def _fake_run_cmd(args, **kw):
            cmds.append(list(args))
            if "--no-index" in args:
                raise subprocess.CalledProcessError(1, args, "", "No matching distribution found")

        metrics = {}
        buf = io.StringIO()
        with mock.patch("h3_drive_download.DriveDownloader", _FakeDownloader), \
                mock.patch.object(w, "verify_comfyui_commit", return_value=True), \
                mock.patch.object(w, "pip_install_needed", return_value=True), \
                mock.patch.object(w, "run_cmd", side_effect=_fake_run_cmd):
            (self.comfy / ".git").mkdir(exist_ok=True)
            with redirect_stdout(buf):
                w.ensure_comfy(self.comfy, self.commit, drive_download=drive_dl,
                               token_provider=_DummyTokenProvider(), metrics=metrics)
        self.assertEqual(len(cmds), 2)
        self.assertIn("--no-index", cmds[0])
        self.assertNotIn("--no-index", cmds[1])
        self.assertFalse(metrics["wheelhouse_used"])
        self.assertIn("CalledProcessError", metrics["wheelhouse_fallback_reason"])
        self.assertIn("WHEELHOUSE_FALLBACK_PYPI:", buf.getvalue())


class TestDualTrackConcurrencyF3(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.td, ignore_errors=True))
        self.base = Path(self.td)
        self.comfy = self.base / "ComfyUI"
        self.nvme = self.base / "nvme_models" / "h3"
        self.manifest, self.paths = _sample_manifest(b"model-bytes-1234")

    def test_dual_track_runs_concurrently_and_links_after_join(self):
        barrier = threading.Barrier(2, timeout=5)
        order = []

        def _fake_comfy(comfy, git_ref, cache=None, **kw):
            barrier.wait()
            time.sleep(0.05)
            order.append("comfy_done")
            kw["metrics"]["comfy_source_sec"] = 0.02
            kw["metrics"]["pip_install_sec"] = 0.03
            kw["metrics"]["wheelhouse_used"] = True
            kw["metrics"]["wheelhouse_fallback_reason"] = None

        def _fake_weights(comfy, cache, manifest, **kw):
            barrier.wait()
            time.sleep(0.05)
            for p in self.paths:
                f = Path(kw["models_dir"]) / p
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(b"model-bytes-1234")
            order.append("weights_done")
            kw["metrics"]["weights_download_sec"] = 0.05
            kw["metrics"]["effective_download_mbps"] = 123.4
            kw["metrics"]["download_workers"] = 12

        real_link = w.link_models_into_comfy

        def _wrapped_link(nvme_root, comfy_models_dir, manifest):
            order.append("link_called")
            return real_link(nvme_root, comfy_models_dir, manifest)

        with mock.patch.object(w, "ensure_comfy", side_effect=_fake_comfy), \
                mock.patch.object(w, "restore_weights", side_effect=_fake_weights), \
                mock.patch.object(w, "link_models_into_comfy", side_effect=_wrapped_link):
            res = w.run_dual_track_setup(
                self.comfy, "a" * 40, None, self.manifest, self.nvme,
                drive_download={"files": []}, token_provider=_DummyTokenProvider())

        # link_models_into_comfy must run strictly AFTER both tracks joined.
        self.assertEqual(order[-1], "link_called")
        self.assertEqual(set(order[:2]), {"comfy_done", "weights_done"})
        self.assertGreater(res["parallel_overlap_sec"], 0.0)
        self.assertTrue(res["wheelhouse_used"])
        self.assertEqual(res["download_workers"], 12)
        for p in self.paths:
            dst = self.comfy / "models" / p
            self.assertTrue(dst.is_symlink())
            self.assertEqual(dst.read_bytes(), b"model-bytes-1234")

    def test_track_a_failure_aborts_track_b_and_reports_ensure_comfy_stage(self):
        track_b_saw_abort = threading.Event()

        def _failing_comfy(*a, **kw):
            time.sleep(0.03)
            raise RuntimeError("pip install exploded")

        def _waiting_weights(*a, **kw):
            ev = kw["abort_event"]
            if ev.wait(timeout=5.0):
                track_b_saw_abort.set()
                raise w.WorkerAbortError("aborted")
            raise AssertionError("Track B never received abort_event")

        with mock.patch.object(w, "ensure_comfy", side_effect=_failing_comfy), \
                mock.patch.object(w, "restore_weights", side_effect=_waiting_weights):
            with self.assertRaises(RuntimeError) as ctx:
                w.run_dual_track_setup(
                    self.comfy, "a" * 40, None, self.manifest, self.nvme,
                    drive_download={"files": []}, token_provider=_DummyTokenProvider())
        self.assertIn("pip install exploded", str(ctx.exception))
        self.assertTrue(track_b_saw_abort.is_set())
        info = w.worker_failure_info(w._LAST_STAGE, ctx.exception)
        self.assertEqual(info["stage"], "ensure_comfy")

    def test_track_b_failure_aborts_track_a_and_reports_restore_weights_stage(self):
        track_a_saw_abort = threading.Event()

        def _waiting_comfy(*a, **kw):
            ev = kw["abort_event"]
            if ev.wait(timeout=5.0):
                track_a_saw_abort.set()
                raise w.WorkerAbortError("aborted")
            raise AssertionError("Track A never received abort_event")

        def _failing_weights(*a, **kw):
            time.sleep(0.03)
            raise RuntimeError("drive 403 denied")

        with mock.patch.object(w, "ensure_comfy", side_effect=_waiting_comfy), \
                mock.patch.object(w, "restore_weights", side_effect=_failing_weights):
            with self.assertRaises(RuntimeError) as ctx:
                w.run_dual_track_setup(
                    self.comfy, "a" * 40, None, self.manifest, self.nvme,
                    drive_download={"files": []}, token_provider=_DummyTokenProvider())
        self.assertIn("drive 403 denied", str(ctx.exception))
        self.assertTrue(track_a_saw_abort.is_set())
        info = w.worker_failure_info(w._LAST_STAGE, ctx.exception)
        self.assertEqual(info["stage"], "restore_weights")

    def test_thread_safe_request_token_refresh_throttles_concurrent_callers(self):
        writes = []
        old_run, old_var, old_last = w._JOB_RUN_ID, w._JOB_VARIANT, w._LAST_REFRESH_REQUEST_TOKEN_ID
        w._JOB_RUN_ID, w._JOB_VARIANT, w._LAST_REFRESH_REQUEST_TOKEN_ID = "run-mt", "A", None
        try:
            barrier = threading.Barrier(8, timeout=5)
            provider = _DummyTokenProvider("same-tok-id")

            def _fake_write(*a, **kw):
                writes.append(kw.get("extra"))

            def _worker():
                barrier.wait()
                w._request_token_refresh(provider)

            with mock.patch.object(w, "write_job_state", side_effect=_fake_write):
                threads = [threading.Thread(target=_worker) for _ in range(8)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=5)
            self.assertEqual(len(writes), 1)
        finally:
            w._JOB_RUN_ID, w._JOB_VARIANT, w._LAST_REFRESH_REQUEST_TOKEN_ID = old_run, old_var, old_last


class TestNvmeSymlinkAndVramF4(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.td, ignore_errors=True))
        self.base = Path(self.td)
        self.nvme = self.base / "nvme_models" / "h3"
        self.models = self.base / "ComfyUI" / "models"
        self.manifest, self.paths = _sample_manifest(b"weight-payload-xyz")

    def test_link_models_into_comfy_creates_atomic_symlinks_and_replaces_existing(self):
        for p in self.paths:
            src = self.nvme / p
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_bytes(b"weight-payload-xyz")
        # Pre-create a stale regular file at one target to verify atomic replacement.
        stale = self.models / self.paths[0]
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_bytes(b"stale-old-data")

        linked = w.link_models_into_comfy(self.nvme, self.models, self.manifest)
        self.assertEqual(len(linked), len(self.paths))
        for p in self.paths:
            dst = self.models / p
            self.assertTrue(dst.is_symlink())
            self.assertEqual(dst.resolve(), (self.nvme / p).resolve())
            self.assertEqual(dst.read_bytes(), b"weight-payload-xyz")

        # Calling a second time over existing symlinks must remain idempotent.
        linked2 = w.link_models_into_comfy(self.nvme, self.models, self.manifest)
        self.assertEqual(len(linked2), len(self.paths))

    def test_link_models_rejects_path_traversal(self):
        bad_manifest = {"files": [{"path": "../escape.safetensors"}]}
        with self.assertRaises(ValueError):
            w.link_models_into_comfy(self.nvme, self.models, bad_manifest)

    def test_link_models_rejects_missing_source(self):
        with self.assertRaises(RuntimeError) as ctx:
            w.link_models_into_comfy(self.nvme, self.models, self.manifest)
        self.assertIn("missing or unreadable", str(ctx.exception))

    def test_resolve_comfy_vram_flags_thresholds_and_overrides(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("H3_COMFY_VRAM_FLAGS", None)
            self.assertEqual(w.resolve_comfy_vram_flags(97871), ["--highvram", "--disable-mmap"])
            self.assertEqual(w.resolve_comfy_vram_flags(81920), ["--highvram", "--disable-mmap"])
            self.assertEqual(w.resolve_comfy_vram_flags(76800), ["--highvram", "--disable-mmap"])
            self.assertEqual(w.resolve_comfy_vram_flags(76799), [])
            self.assertEqual(w.resolve_comfy_vram_flags(40960), [])
            self.assertEqual(w.resolve_comfy_vram_flags(None), [])

        with mock.patch.dict(os.environ, {"H3_COMFY_VRAM_FLAGS": "--lowvram --fp16-vae"}):
            self.assertEqual(w.resolve_comfy_vram_flags(97871), ["--lowvram", "--fp16-vae"])

        with mock.patch.dict(os.environ, {"H3_COMFY_VRAM_FLAGS": ""}):
            self.assertEqual(w.resolve_comfy_vram_flags(97871), [])

    def test_detect_gpu_vram_mib(self):
        ok_proc = mock.Mock(returncode=0, stdout="97871\n")
        with mock.patch.object(w.subprocess, "run", return_value=ok_proc):
            self.assertEqual(w.detect_gpu_vram_mib(), 97871)

        fail_proc = mock.Mock(returncode=1, stdout="")
        with mock.patch.object(w.subprocess, "run", return_value=fail_proc):
            self.assertIsNone(w.detect_gpu_vram_mib())

        with mock.patch.object(w.subprocess, "run", side_effect=FileNotFoundError("no nvidia-smi")):
            self.assertIsNone(w.detect_gpu_vram_mib())


class TestMultiFileRestoreAndTelemetryF1F5(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.td, ignore_errors=True))
        self.base = Path(self.td)
        self.comfy = self.base / "ComfyUI"
        self.nvme = self.base / "nvme_models" / "h3"
        self.content = b"0123456789abcdef"
        self.manifest, self.paths = _sample_manifest(self.content)
        self.drive_dl = {
            "files": [
                {"file_id": f"fid-{i}", "path": p, "size": len(self.content), "sha256": _sha256_bytes(self.content)}
                for i, p in enumerate(self.paths)
            ],
            "dest_dir": str(self.base / "dl"),
        }

    def test_restore_weights_to_independent_nvme_dir_concurrently(self):
        seen_abort_events = []

        class _WritingDownloader:
            def __init__(self, tp, abort_event=None, **kw):
                self._abort = abort_event

            def download_file(self, file_id, size, sha256, dest, abort_event=None, **kw):
                seen_abort_events.append(abort_event)
                dest = Path(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"0123456789abcdef")
                return {"drive_file_id": file_id, "size": size, "sha256": sha256, "path": str(dest)}

        usage = type("usage", (), {"total": 200 * 1024 ** 3, "used": 10 * 1024 ** 3, "free": 190 * 1024 ** 3})()
        abort_ev = threading.Event()
        metrics = {}
        with mock.patch("h3_drive_download.DriveDownloader", _WritingDownloader), \
                mock.patch("shutil.disk_usage", return_value=usage), \
                mock.patch.dict(os.environ, {"H3_MODEL_FILE_CONCURRENCY": "2"}):
            w.restore_weights(
                self.comfy, Path("/nonexistent"), self.manifest,
                drive_download=self.drive_dl, token_provider=_DummyTokenProvider(),
                models_dir=self.nvme, abort_event=abort_ev, metrics=metrics)

        self.assertEqual(len(seen_abort_events), len(self.paths))
        self.assertTrue(all(ev is abort_ev for ev in seen_abort_events))
        for p in self.paths:
            self.assertEqual((self.nvme / p).read_bytes(), self.content)
            self.assertFalse((self.comfy / "models" / p).exists())
        self.assertGreaterEqual(metrics["weights_download_sec"], 0.0)
        self.assertGreaterEqual(metrics["effective_download_mbps"], 0.0)
        self.assertGreaterEqual(metrics["download_workers"], 1)

    def test_restore_weights_preset_abort_event_stops_before_download(self):
        called = []

        class _NeverDownloader:
            def __init__(self, *a, **kw):
                pass

            def download_file(self, *a, **kw):
                called.append(True)

        abort_ev = threading.Event()
        abort_ev.set()
        with mock.patch("h3_drive_download.DriveDownloader", _NeverDownloader):
            with self.assertRaises(w.WorkerAbortError):
                w.restore_weights(
                    self.comfy, Path("/nonexistent"), self.manifest,
                    drive_download=self.drive_dl, token_provider=_DummyTokenProvider(),
                    models_dir=self.nvme, abort_event=abort_ev)
        self.assertEqual(called, [])

    def test_build_coldstart_metrics_schema_matches_contract(self):
        setup = {
            "comfy_source_sec": 3.21,
            "pip_install_sec": 4.56,
            "weights_download_sec": 118.4,
            "effective_download_mbps": 338.5,
            "wheelhouse_used": True,
            "wheelhouse_fallback_reason": None,
            "parallel_overlap_sec": 7.77,
            "download_workers": 12,
        }
        payload = w.build_coldstart_metrics(
            setup,
            comfy_boot_sec=14.2,
            total_coldstart_sec=136.1,
            vram_mib=97871,
            vram_flags=["--highvram", "--disable-mmap"],
        )
        expected_keys = {
            "schema_version",
            "comfy_source_sec",
            "pip_install_sec",
            "weights_download_sec",
            "effective_download_mbps",
            "comfy_boot_sec",
            "total_coldstart_sec",
            "wheelhouse_used",
            "wheelhouse_fallback_reason",
            "parallel_overlap_sec",
            "download_workers",
            "vram_mib",
            "highvram_enabled",
        }
        self.assertEqual(set(payload.keys()), expected_keys)
        self.assertEqual(payload["schema_version"], 1)
        self.assertTrue(payload["wheelhouse_used"])
        self.assertIsNone(payload["wheelhouse_fallback_reason"])
        self.assertTrue(payload["highvram_enabled"])
        self.assertEqual(payload["vram_mib"], 97871)
        line = "COLDSTART_METRICS_JSON:" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self.assertNotIn("\n", line)
        parsed = json.loads(line.split("COLDSTART_METRICS_JSON:", 1)[1])
        self.assertEqual(parsed, payload)


if __name__ == "__main__":
    unittest.main()
