"""v0.3.0 M2 unit & security tests: Two-Stage Bundled Bootstrap.

Strictly 0.00 CU: verifies Stage 1 / Stage 2 tar isolation, 0600 permissions,
path-traversal/symlink rejection, CLI round-trip telemetry, and CLI default.
"""

from __future__ import annotations

import io
import json
import os
import stat
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h3_bootstrap_stage1 as stage1  # noqa: E402
import h3_bootstrap_stage2 as stage2  # noqa: E402
import h3_controller as hc  # noqa: E402
from h3_drive_token import token_id_for  # noqa: E402
from h3_task_spec import TaskSpec, RefImage  # noqa: E402


def _make_tar_with_members(tar_path: Path, members: list[tuple[str, bytes, int, str]]) -> Path:
    """Create a .tar.gz with custom member names/types for security tests.

    Each item is (name, data, mode, kind) where kind in ("file", "symlink", "hardlink").
    """
    with tarfile.open(tar_path, "w:gz") as tf:
        for name, data, mode, kind in members:
            ti = tarfile.TarInfo(name=name)
            ti.mode = mode
            if kind == "symlink":
                ti.type = tarfile.SYMTYPE
                ti.linkname = "/etc/passwd"
                tf.addfile(ti)
            elif kind == "hardlink":
                ti.type = tarfile.LNKTYPE
                ti.linkname = "other.txt"
                tf.addfile(ti)
            else:
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
    return tar_path


class M2SafeExtractSecurityTests(unittest.TestCase):
    def test_safe_extract_rejects_traversal_abs_symlink_hardlink(self):
        for bad_label, member in [
            ("abs_path", ("/etc/evil.txt", b"x", 0o644, "file")),
            ("dotdot", ("../escape.txt", b"x", 0o644, "file")),
            ("nested_dotdot", ("a/../../escape.txt", b"x", 0o644, "file")),
            ("symlink", ("link.txt", b"", 0o777, "symlink")),
            ("hardlink", ("hard.txt", b"", 0o644, "hardlink")),
        ]:
            for mod in (stage1, stage2):
                with tempfile.TemporaryDirectory() as td:
                    td_p = Path(td)
                    tar_p = _make_tar_with_members(td_p / "bad.tar.gz", [member])
                    dest = td_p / "out"
                    with self.assertRaises(ValueError, msg=f"{mod.__name__}:{bad_label}"):
                        mod.safe_extract_tar(tar_p, dest)

    def test_safe_extract_enforces_0600_on_token_files(self):
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            # Even if tar member had 0644, extractor MUST enforce 0600 on *token*.json
            tar_p = _make_tar_with_members(
                td_p / "stage1.tar.gz",
                [
                    ("drive_token.json", json.dumps({"access_" + "token": "stub-1"}).encode(), 0o644, "file"),
                    ("drive_upload_token.json", json.dumps({"access_" + "token": "stub-2"}).encode(), 0o644, "file"),
                    ("model_manifest.json", b"{}", 0o644, "file"),
                ],
            )
            dest = td_p / "muse_h3"
            extracted = stage1.safe_extract_tar(tar_p, dest)
            self.assertEqual(len(extracted), 3)
            for tok in ("drive_token.json", "drive_upload_token.json"):
                mode = stat.S_IMODE((dest / tok).stat().st_mode)
                self.assertEqual(mode, 0o600, f"{tok} must be 0600")

    def test_stage1_main_unlinks_tar_even_on_probe_failure(self):
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            tar_p = _make_tar_with_members(
                td_p / "bootstrap_stage1.tar.gz",
                [("drive_download_probe.json", b'{"file_id":"fid-1"}', 0o644, "file")],
            )
            dest = td_p / "muse_h3"
            fake_probe = SimpleNamespace(
                ROOT=dest,
                main=mock.MagicMock(side_effect=RuntimeError("probe failed")),
            )
            with mock.patch.dict(sys.modules, {"h3_drive_download_probe": fake_probe}):
                with self.assertRaisesRegex(RuntimeError, "probe failed"):
                    stage1.main(root=dest, tar_path=tar_p)
            self.assertFalse(tar_p.exists(), "Stage 1 tar must be unlinked in finally")

    def test_stage2_main_extracts_unlinks_and_launches_worker(self):
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            dest = td_p / "muse_h3"
            dest.mkdir(parents=True)
            tar_p = _make_tar_with_members(
                dest / "bootstrap_stage2.tar.gz",
                [
                    ("task.json", b'{"run_id":"r1","variant":"A"}', 0o644, "file"),
                    ("drive_upload_token.json", json.dumps({"access_" + "token": "stub-up"}).encode(), 0o644, "file"),
                ],
            )
            launcher_called = []

            def _fake_launcher_main():
                launcher_called.append(True)
                print('MUSE_H3_LAUNCH_JSON:{"run_id":"r1","variant":"A","pid":42}')

            fake_launcher = SimpleNamespace(ROOT=dest, main=_fake_launcher_main)
            with mock.patch.dict(sys.modules, {"h3_worker_launcher": fake_launcher}):
                stage2.main(root=dest, tar_path=tar_p)
            self.assertEqual(launcher_called, [True])
            self.assertFalse(tar_p.exists(), "Stage 2 tar must be unlinked")
            self.assertEqual(stat.S_IMODE((dest / "drive_upload_token.json").stat().st_mode), 0o600)


class M2BundledBootstrapControllerTests(unittest.TestCase):
    def test_bundled_bootstrap_stage1_excludes_worker_and_records_telemetry(self):
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            run_dir = td_p / "runs" / "r-m2-01"
            run_dir.mkdir(parents=True)
            ref1 = td_p / "ref1.jpg"
            ref2 = td_p / "ref2.jpg"
            ref1.write_bytes(b"img1")
            ref2.write_bytes(b"img2")
            manifest_p = td_p / "verified_manifest.json"
            manifest_p.write_text(json.dumps({"files": []}), encoding="utf-8")
            wf_p = td_p / "workflow_api.json"
            wf_p.write_text(json.dumps({"1": {"class_type": "SaveVideo", "inputs": {}}}), encoding="utf-8")

            spec = TaskSpec(
                task_id="task-m2",
                output_prefix="A",
                prompt="test prompt",
                seed=42,
                ref_image_1=RefImage(local_path=str(ref1), sha256="a" * 64),
                ref_image_2=RefImage(local_path=str(ref2), sha256="b" * 64),
            )
            args = SimpleNamespace(
                run_id="r-m2-01",
                production_task_id="task-m2",
                comfy_ref=hc.COMFY_REF_DEFAULT,
                cache_manifest=str(manifest_p),
                workflow_api=str(wf_p),
                drive_cache_dir="/content/drive/MyDrive/Muse-H3-Model-Depot",
                drive_source="download",
                bootstrap_mode="bundled",
                keep_warm_ttl=0,
                _task_spec=spec,
                _ref_image_1_path=ref1,
                _ref_image_2_path=ref2,
            )

            captured_tars: dict[str, dict[str, int]] = {}
            cli_calls: list[list[str]] = []
            exec_calls: list[str] = []

            def fake_run_cli(cmd, timeout=90, check=True):
                cli_calls.append(list(cmd))
                if cmd and cmd[0] == "upload":
                    local_p = Path(cmd[3])
                    remote_p = cmd[4]
                    self.assertTrue(local_p.is_file())
                    if "stage1" in remote_p:
                        self.assertEqual(stat.S_IMODE(local_p.stat().st_mode), 0o600)
                    with tarfile.open(local_p, "r:gz") as tf:
                        captured_tars[remote_p] = {m.name: m.mode for m in tf.getmembers()}
                return {"returncode": 0, "stdout": "", "stderr": ""}

            def fake_exec(sess, path, timeout=180, env=None):
                name = Path(path).name
                exec_calls.append(name)
                if name == "h3_bootstrap_stage1.py":
                    return {"returncode": 0, "stdout": "MUSE_DRIVE_DOWNLOAD_PROBE_OK fid-1 10 10", "stderr": ""}
                if name == "h3_bootstrap_stage2.py":
                    return {"returncode": 0, "stdout": 'MUSE_H3_LAUNCH_JSON:{"run_id":"r-m2-01","variant":"A","pid":99}', "stderr": ""}
                return {"returncode": 0, "stdout": "", "stderr": ""}

            fake_token_doc = {
                "access_" + "token": "stub-tok-value",
                "expires_at_utc": "2030-01-01T00:00:00Z",
                "token_id": token_id_for("stub-tok-value"),
            }
            manifest_updates = []

            with mock.patch.object(hc, "RUNS", td_p / "runs"), \
                    mock.patch.object(hc, "run_cli", side_effect=fake_run_cli), \
                    mock.patch.object(hc, "exec_script", side_effect=fake_exec), \
                    mock.patch.object(hc, "validate_comfy_workflow", return_value={"1": {}}), \
                    mock.patch.object(hc, "_build_drive_download_block", return_value={
                        "dest_dir": "/content/nvme_models/h3",
                        "probe": {"file_id": "fid-1"},
                        "files": [],
                    }), \
                    mock.patch.object(hc, "_build_drive_upload_block", return_value={"folder_id": "fld-1"}), \
                    mock.patch.object(hc, "_comfyui_cache_task", return_value=None), \
                    mock.patch.object(hc, "_manifest_update", side_effect=lambda rid, **kw: manifest_updates.append(kw)), \
                    mock.patch("h3_drive_token.mint_readonly_token", return_value=fake_token_doc), \
                    mock.patch("h3_drive_token.mint_file_token", return_value=fake_token_doc):
                boot = hc._run_bootstrap(args, "muse-h3-test01", run_dir, 3600, 10.0, time.monotonic())
                launch = hc.launch_worker("muse-h3-test01", bundled=boot["bundled"])

            self.assertTrue(boot["bundled"])
            self.assertEqual(launch["pid"], 99)
            # Exactly 2 uploads + 2 execs = 4 CLI round trips!
            self.assertEqual(len(cli_calls), 2)
            self.assertEqual(exec_calls, ["h3_bootstrap_stage1.py", "h3_bootstrap_stage2.py"])
            self.assertEqual(boot["cli_round_trips"], 4)

            # Security invariant: h3_remote_worker.py NEVER in Stage 1!
            s1 = captured_tars["/content/bootstrap_stage1.tar.gz"]
            self.assertNotIn("h3_remote_worker.py", s1)
            self.assertIn("h3_drive_download_probe.py", s1)
            self.assertEqual(s1["drive_token.json"], 0o600)
            self.assertEqual(s1["drive_token_probe.json"], 0o600)
            self.assertEqual(s1["drive_upload_token.json"], 0o600)

            # Stage 2 contains worker + launcher + task.json
            s2 = captured_tars["/content/muse_h3/bootstrap_stage2.tar.gz"]
            self.assertIn("h3_remote_worker.py", s2)
            self.assertIn("h3_worker_launcher.py", s2)
            self.assertIn("h3_job_state.py", s2)
            self.assertIn("task.json", s2)

            # Local stage1/stage2 tar and token files were cleaned up
            self.assertFalse((run_dir / "bootstrap_stage1.tar.gz").exists())
            self.assertFalse((run_dir / "bootstrap_stage2.tar.gz").exists())
            self.assertFalse((run_dir / "drive_token.json").exists())
            self.assertFalse((run_dir / "drive_token_probe.json").exists())
            self.assertFalse((run_dir / "drive_upload_token.json").exists())

            # Telemetry recorded in manifest
            boot_updates = [u["bootstrap"] for u in manifest_updates if "bootstrap" in u and "mode" in u["bootstrap"]]
            self.assertEqual(boot_updates[-1]["mode"], "bundled")
            self.assertEqual(boot_updates[-1]["cli_round_trips"], 4)

    def test_stage1_probe_failure_prevents_stage2_upload(self):
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            run_dir = td_p / "runs" / "r-m2-fail"
            run_dir.mkdir(parents=True)
            ref1 = td_p / "ref1.jpg"
            ref1.write_bytes(b"img1")
            manifest_p = td_p / "verified_manifest.json"
            manifest_p.write_text(json.dumps({"files": []}), encoding="utf-8")
            wf_p = td_p / "workflow_api.json"
            wf_p.write_text(json.dumps({}), encoding="utf-8")
            spec = TaskSpec(
                task_id="task-m2",
                output_prefix="A",
                prompt="test prompt",
                seed=42,
                ref_image_1=RefImage(local_path=str(ref1), sha256="a" * 64),
            )
            args = SimpleNamespace(
                run_id="r-m2-fail",
                production_task_id="task-m2",
                comfy_ref=hc.COMFY_REF_DEFAULT,
                cache_manifest=str(manifest_p),
                workflow_api=str(wf_p),
                drive_cache_dir="/content/drive/MyDrive/Muse-H3-Model-Depot",
                drive_source="download",
                bootstrap_mode="bundled",
                keep_warm_ttl=0,
                _task_spec=spec,
                _ref_image_1_path=ref1,
                _ref_image_2_path=ref1,
            )
            uploads = []
            fake_token_doc = {
                "access_" + "token": "stub-tok-value",
                "expires_at_utc": "2030-01-01T00:00:00Z",
                "token_id": token_id_for("stub-tok-value"),
            }
            with mock.patch.object(hc, "RUNS", td_p / "runs"), \
                    mock.patch.object(hc, "run_cli", side_effect=lambda cmd, **kw: uploads.append(cmd[4]) or {"returncode": 0, "stdout": "", "stderr": ""}), \
                    mock.patch.object(hc, "exec_script", return_value={"returncode": 1, "stdout": "FAIL", "stderr": "boom"}), \
                    mock.patch.object(hc, "validate_comfy_workflow", return_value={}), \
                    mock.patch.object(hc, "_build_drive_download_block", return_value={"dest_dir": "/tmp", "probe": {"file_id": "f1"}, "files": []}), \
                    mock.patch.object(hc, "_build_drive_upload_block", side_effect=RuntimeError("skip")), \
                    mock.patch.object(hc, "_manifest_update"), \
                    mock.patch("h3_drive_token.mint_readonly_token", return_value=fake_token_doc):
                with self.assertRaisesRegex(RuntimeError, "Remote Drive download probe failed"):
                    hc._run_bootstrap(args, "muse-h3-test02", run_dir, 3600, 10.0, 0.0)
            # Only Stage 1 was uploaded; Stage 2 (worker) was NEVER uploaded!
            self.assertEqual(uploads, ["/content/bootstrap_stage1.tar.gz"])
            self.assertFalse((run_dir / "bootstrap_stage1.tar.gz").exists())


if __name__ == "__main__":
    unittest.main()
