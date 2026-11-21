"""Comprehensive offline regression tests for Antigravity-Colab-Workstation v0.3.1:
- P0-1: Pre-allocation manifest persistence & owner-verified reconciliation (Cases A/B/C)
- P1-3: ComfyUI /prompt three-state idempotent reconciliation (found/absent/unknown)
- P1-4: Completed delivery tamper checks & 0-CU atomic local restore
- P1-5: Metric F lease-registry cross-check + cold/warm stratification + Wilson 95% CI
- P1-6: Local Drive ID override resolution + Drive ACL private audit + secret_scan gate

All tests are 100% offline and spend 0.00 CU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import drive_depot as dd  # noqa: E402
import h3_contract as hct  # noqa: E402
import h3_controller as hc  # noqa: E402
import h3_gpu_lease_registry as lr  # noqa: E402
import h3_reliability as rel  # noqa: E402
import h3_remote_worker as hrw  # noqa: E402
import h3_run_manifest as rm  # noqa: E402
import h3_session_watchdog as sw  # noqa: E402
import scripts.reliability_report as rep  # noqa: E402
import scripts.secret_scan as sec  # noqa: E402
from h3_budget import ledger_path_for_task  # noqa: E402


def _write_ledger(root: Path, task_id: str, *, deadline_utc: str = "2026-12-01T12:00:00Z",
                  spent: float = 2.0, budget: float = 30.0) -> Path:
    path = ledger_path_for_task(root, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "production_task_id": task_id,
        "project_budget_cu": budget,
        "initial_balance_cu": 188.01,
        "hard_stop_balance_cu": round(188.01 - budget, 2),
        "max_observed_spent_cu": spent,
        "authorized": True,
        "deadline_utc": deadline_utc,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class PendingAllocationReconciliationTests(unittest.TestCase):
    """P0-1: eliminate colab new allocation-window orphan blind spot."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        (self.root / "model_cache").mkdir(parents=True)
        self.lease_path = self.root / "model_cache" / "h3_gpu_leases.json"
        self.watch_path = self.root / "model_cache" / "h3_managed_sessions.json"
        self.owner_id = lr.owner_id_for("owner@example.test")
        self.other_owner_id = lr.owner_id_for("other@example.test")
        self.registry = lr.LeaseRegistry(path=self.lease_path, mirror=None)
        self.events: list[dict] = []
        self.patches = [
            mock.patch.object(hc, "ROOT", self.root),
            mock.patch.object(hc, "RUNS", self.root / "runs"),
            mock.patch.object(hc, "OUTPUT_ROOT", self.root / "output"),
            mock.patch.object(hc, "record", lambda d: self.events.append(d)),
            mock.patch.object(hc, "verify_environment", lambda: {"ok": True}),
            mock.patch.object(hc, "_current_source_commit", lambda: "abc1234"),
            mock.patch.object(hc, "_lease_registry", lambda: self.registry),
            mock.patch.object(hc.time, "sleep", lambda s: None),
            mock.patch.object(sw, "STATE_PATH", self.watch_path),
        ]
        for p in self.patches:
            p.start()
        self.addCleanup(mock.patch.stopall)
        self.addCleanup(self.td.cleanup)

    def _create_pending_run(self, task_id: str, run_id: str, pending_sid: str,
                            *, register_lease: bool = True,
                            lease_owner: str | None = None,
                            manifest_owner: str | None = None):
        _write_ledger(self.root, task_id)
        rm.create_run_manifest(self.root, task_id, run_id, variant="A", code_commit="abc1234")
        rm.update_run(
            self.root,
            run_id,
            pending_session_id=pending_sid,
            session_owner_id=manifest_owner if manifest_owner is not None else self.owner_id,
            session_state="ALLOCATING",
        )
        sw.record_session(
            pending_sid, "G4", run_id, "2026-12-01T12:00:00+00:00",
            path=self.watch_path, production_task_id=task_id, status="allocating",
        )
        if register_lease:
            self.registry.register_lease(
                pending_sid, run_id, task_id, "G4",
                "2026-12-01T12:00:00+00:00", 25.0,
                lease_owner if lease_owner is not None else self.owner_id,
                max_wall_sec=3600,
            )

    def test_case_a_remote_absent_cleans_up_without_self_lock(self):
        task_id = "task-p01-a"
        run_id = "run-p01-a"
        sid = "muse-h3-pending-a"
        self._create_pending_run(task_id, run_id, sid)

        with mock.patch.object(sw, "query_session_state", return_value=sw.SessionState.STOPPED):
            res = hc._resume_run_inner(argparse.Namespace(
                run_id=run_id, production_task_id=task_id, max_wall_sec=3600, poll_interval_sec=5,
            ))

        self.assertTrue(res["resumed"])
        self.assertEqual(res["outcome"], "pending_allocation_absent_cleaned")
        manifest = rm.load_run_manifest(self.root, run_id)
        self.assertEqual(manifest["task_state"], "FAILED")
        self.assertEqual(manifest["session_state"], "STOPPED")
        self.assertIsNone(manifest["pending_session_id"])
        self.assertEqual(manifest["session_id"], "")
        # Lease and watchdog record must also be cleaned up.
        lease = self.registry.get_lease(sid)
        self.assertEqual(lease["last_confirmed_state"], "STOPPED")
        rec = next(s for s in sw.load_managed_sessions(self.watch_path) if s["session_id"] == sid)
        self.assertEqual(rec["status"], "stopped")

    def test_case_b_remote_active_owner_verified_promotes_and_resumes_bootstrap(self):
        task_id = "task-p01-b"
        run_id = "run-p01-b"
        sid = "muse-h3-pending-b"
        self._create_pending_run(task_id, run_id, sid)

        promoted_calls = []

        def fake_resume_bootstrap(args, manifest):
            promoted_calls.append(dict(manifest))
            return {"resumed": True, "outcome": "bootstrapped_promoted", "session_id": manifest["session_id"]}

        with mock.patch.object(sw, "query_session_state", return_value=sw.SessionState.ACTIVE), \
                mock.patch.object(lr, "current_owner_id", return_value=self.owner_id), \
                mock.patch.object(hc, "poll_worker", side_effect=hc.WorkerPollError("not started yet")), \
                mock.patch.object(hc, "_resume_bootstrap", side_effect=fake_resume_bootstrap):
            res = hc._resume_run_inner(argparse.Namespace(
                run_id=run_id, production_task_id=task_id, max_wall_sec=3600, poll_interval_sec=5,
            ))

        self.assertTrue(res["resumed"])
        self.assertEqual(res["outcome"], "bootstrapped_promoted")
        self.assertEqual(len(promoted_calls), 1)
        self.assertEqual(promoted_calls[0]["session_id"], sid)
        manifest = rm.load_run_manifest(self.root, run_id)
        self.assertEqual(manifest["session_id"], sid)
        self.assertEqual(manifest["session_state"], "ACTIVE")
        self.assertEqual(manifest["task_state"], "GPU_ALLOCATED")
        self.assertIsNone(manifest["pending_session_id"])
        rec = next(s for s in sw.load_managed_sessions(self.watch_path) if s["session_id"] == sid)
        self.assertEqual(rec["status"], "active")

    def test_case_b_unhealthy_promoted_session_stops_cleanly(self):
        task_id = "task-p01-b2"
        run_id = "run-p01-b2"
        sid = "muse-h3-pending-b2"
        self._create_pending_run(task_id, run_id, sid)

        with mock.patch.object(sw, "query_session_state", return_value=sw.SessionState.ACTIVE), \
                mock.patch.object(lr, "current_owner_id", return_value=self.owner_id), \
                mock.patch.object(hc, "poll_worker", side_effect=hc.WorkerPollError("no worker")), \
                mock.patch.object(hc, "_resume_bootstrap", side_effect=RuntimeError("remote exec dead")), \
                mock.patch.object(hc, "stop_and_verify_session", return_value={"session": sid, "stopped_confirmed": True, "state": "STOPPED"}) as mock_stop:
            res = hc._resume_run_inner(argparse.Namespace(
                run_id=run_id, production_task_id=task_id, max_wall_sec=3600, poll_interval_sec=5,
            ))

        mock_stop.assert_called_once_with(sid)
        self.assertTrue(res["resumed"])
        self.assertTrue(res["stop_verified"])
        manifest = rm.load_run_manifest(self.root, run_id)
        self.assertEqual(manifest["task_state"], "FAILED")
        self.assertEqual(manifest["session_state"], "STOPPED")

    def test_case_c_unverified_owner_fails_closed_with_critical_orphan(self):
        task_id = "task-p01-c"
        run_id = "run-p01-c"
        sid = "muse-h3-pending-c"
        # Lease registered under a DIFFERENT owner_id -> must fail closed!
        self._create_pending_run(task_id, run_id, sid, lease_owner=self.other_owner_id, manifest_owner="")

        with mock.patch.object(sw, "query_session_state", return_value=sw.SessionState.ACTIVE), \
                mock.patch.object(lr, "current_owner_id", return_value=self.owner_id):
            with self.assertRaises(RuntimeError) as ctx:
                hc._resume_run_inner(argparse.Namespace(
                    run_id=run_id, production_task_id=task_id, max_wall_sec=3600, poll_interval_sec=5,
                ))
        self.assertIn("CRITICAL_ORPHAN_GPU_SESSION", str(ctx.exception))
        # Manifest must NOT have adopted the unverified session.
        manifest = rm.load_run_manifest(self.root, run_id)
        self.assertEqual(manifest["session_id"], "")
        self.assertEqual(manifest["pending_session_id"], sid)

    def test_case_c_missing_lease_fails_closed_with_critical_orphan(self):
        task_id = "task-p01-c2"
        run_id = "run-p01-c2"
        sid = "muse-h3-pending-c2"
        self._create_pending_run(task_id, run_id, sid, register_lease=False)

        with mock.patch.object(sw, "query_session_state", return_value=sw.SessionState.ACTIVE), \
                mock.patch.object(lr, "current_owner_id", return_value=self.owner_id):
            with self.assertRaises(RuntimeError) as ctx:
                hc._resume_run_inner(argparse.Namespace(
                    run_id=run_id, production_task_id=task_id, max_wall_sec=3600, poll_interval_sec=5,
                ))
        self.assertIn("CRITICAL_ORPHAN_GPU_SESSION", str(ctx.exception))

    def test_manifest_backward_compat_v1_to_v3(self):
        # Legacy manifest without pending_session_id / session_owner_id loads cleanly.
        m = rm.create_run_manifest(self.root, "task-compat", "run-compat", variant="A")
        p = rm.manifest_dir(self.root) / "run-compat.json"
        del m["pending_session_id"]
        del m["session_owner_id"]
        p.write_text(json.dumps(m), encoding="utf-8")
        view = rm.recovery_view(self.root, "run-compat")
        self.assertIsNone(view["pending_session_id"])
        self.assertIsNone(view["session_owner_id"])

    def test_reconcile_on_startup_cleans_sibling_allocating_lease(self):
        sid = "muse-h3-alloc-dead"
        sw.record_session(
            sid, "G4", "run-dead", "2026-12-01T12:00:00+00:00",
            path=self.watch_path, production_task_id="task-dead", status="allocating",
        )
        self.registry.register_lease(
            sid, "run-dead", "task-dead", "G4",
            "2026-12-01T12:00:00+00:00", 25.0, self.owner_id, max_wall_sec=3600,
        )
        with mock.patch.object(sw, "query_session_state", return_value=sw.SessionState.STOPPED):
            res = sw.reconcile_on_startup("oauth2", path=self.watch_path)
        self.assertEqual(res["failed_allocations"], [sid])
        self.assertEqual(res["managed_active_sessions"], [])
        self.assertEqual(self.registry.get_lease(sid)["last_confirmed_state"], "STOPPED")

    def test_pre_allocation_manifest_persisted_before_colab_new_executes(self):
        """Verify pending_session_id + session_owner_id + ALLOCATING are
        persisted to run_manifest.json strictly BEFORE run_cli(['new', ...]) runs."""
        task_id = "task-pre-alloc"
        run_id = "run-pre-alloc"
        (self.root / "runs").mkdir(parents=True, exist_ok=True)
        (self.root / "output").mkdir(parents=True, exist_ok=True)
        ref_file = self.root / "ref.jpg"
        ref_file.write_bytes(b"fake-ref-jpg")
        ref_sha = hashlib.sha256(b"fake-ref-jpg").hexdigest()
        spec_path = self.root / "spec.json"
        spec_path.write_text(json.dumps({
            "task_id": task_id,
            "output_prefix": "A",
            "ref_image_1": {"local_path": str(ref_file), "sha256": ref_sha},
            "prompt": "test prompt",
            "seed": 42,
        }), encoding="utf-8")
        wf_path = self.root / "workflow_api.json"
        wf_path.write_text((ROOT / "workflow_api.json").read_text(encoding="utf-8"), encoding="utf-8")
        cm_path = self.root / "model_cache" / "verified_manifest.json"
        cm_path.write_text((ROOT / "config" / "model_manifest.example.json").read_text(encoding="utf-8"), encoding="utf-8")

        observed_during_new: list[dict] = []

        def fake_run_cli(cmd, timeout=120, check=True):
            if cmd and cmd[0] == "new":
                m = rm.load_run_manifest(self.root, run_id)
                observed_during_new.append(dict(m))
                raise RuntimeError("simulated crash during colab new (AUTH_FATAL: 401)")
            return {"returncode": 0, "stdout": "", "stderr": ""}

        args = argparse.Namespace(
            action="run",
            production_task_id=task_id,
            run_id=run_id,
            task_spec=str(spec_path),
            variant=None,
            execute=True,
            confirm_gpu=True,
            authorize_task_gpu=True,
            allow_g4=True,
            allow_a100_highmem=True,
            fresh_inference=True,
            project_budget_cu=30.0,
            per_session_budget_cu=25.0,
            gpu_policy="G4_THEN_A100_HIGHMEM",
            comfy_ref="61e524bdba7a4e193b45c2697f1af83c1beb86d6",
            workflow_api=str(wf_path),
            cache_manifest=str(cm_path),
            drive_source="download",
            drive_cache_dir="/content/drive/MyDrive/Muse-H3-Model-Depot",
            approve_interactive_drive_mount=False,
            keep_warm_ttl=0,
            allocation_rounds=1,
            max_wall_sec=3600,
            max_rate_cu_h=14.0,
            poll_interval_sec=5,
            authorize_new_task=True,
            prior_unbilled_reserve_cu=0.0,
            task_deadline_utc="",
            sleep_fn=lambda s: None,
        )
        with mock.patch("drive_depot.drive_service", return_value=object()), \
                mock.patch("drive_depot.account_matches", return_value=True), \
                mock.patch("drive_depot.verify_ready", return_value={"ready": True}), \
                mock.patch.object(hc, "read_live_usage", return_value={"balance": 188.01, "rate": 0.0}), \
                mock.patch.object(hc, "inspect", return_value={"cu_balance": 188.01, "rate": 0.0, "sessions": []}), \
                mock.patch.object(hc, "verify_drive_download_preflight", return_value={"ok": True}), \
                mock.patch.object(sw, "reconcile_on_startup", return_value={"managed_active_sessions": []}), \
                mock.patch.object(sw, "record_session", lambda *a, **kw: {"session_id": a[0], "status": kw.get("status", "allocating")}), \
                mock.patch.object(sw, "mark_session_stopped", lambda *a, **kw: True), \
                mock.patch.object(lr, "current_owner_id", return_value=self.owner_id), \
                mock.patch.object(hc, "run_cli", side_effect=fake_run_cli), \
                mock.patch.object(hc, "stop_and_verify_session", return_value={"session": "s", "stopped_confirmed": True, "state": "STOPPED"}):
            with self.assertRaises(RuntimeError):
                hc._run_variant_inner(args)

        self.assertEqual(len(observed_during_new), 1)
        snap = observed_during_new[0]
        self.assertEqual(snap["session_id"], "")
        self.assertTrue(snap["pending_session_id"])
        self.assertEqual(snap["session_owner_id"], self.owner_id)
        self.assertEqual(snap["session_state"], "ALLOCATING")


class PromptSubmitTristateTests(unittest.TestCase):
    """P1-3: ComfyUI /prompt read-timeout three-state reconciliation."""

    def setUp(self):
        self.post_json = mock.patch.object(hrw, "post_json").start()
        self.get_json = mock.patch.object(hrw, "get_json").start()
        self.sleep = mock.patch.object(hrw.time, "sleep").start()
        self.addCleanup(mock.patch.stopall)

    def _queued(self, pid: str, client_id: str = "cid-p13"):
        return {"queue_running": [[0, pid, {}, {"client_id": client_id}, []]], "queue_pending": []}

    def test_find_prompt_three_states(self):
        self.get_json.side_effect = [self._queued("pid-1")]
        self.assertEqual(hrw._find_prompt_by_client_id("http://127.0.0.1:8188", "cid-p13"), ("found", "pid-1"))

        self.get_json.side_effect = [{"queue_running": [], "queue_pending": []}, {}]
        self.assertEqual(hrw._find_prompt_by_client_id("http://127.0.0.1:8188", "cid-p13"), ("absent", None))

        self.get_json.side_effect = [TimeoutError("queue stalled"), {}]
        status, detail = hrw._find_prompt_by_client_id("http://127.0.0.1:8188", "cid-p13")
        self.assertEqual(status, "unknown")
        self.assertIn("/queue error", detail)

    def test_unknown_twice_then_found_reuses_without_second_post(self):
        self.post_json.side_effect = urllib.error.URLError(TimeoutError("read timed out"))
        # Probe 1: /queue timeout, /history timeout -> unknown
        # Probe 2: /queue timeout, /history timeout -> unknown
        # Probe 3: /queue returns queued prompt -> found!
        self.get_json.side_effect = [
            TimeoutError("busy 1"), TimeoutError("busy 1h"),
            TimeoutError("busy 2"), TimeoutError("busy 2h"),
            self._queued("pid-recovered-42"),
        ]
        res = hrw._submit_prompt(
            "http://127.0.0.1:8188", {"1": {}}, "cid-p13", reconcile_delays=(1, 2, 4),
        )
        self.assertEqual(res, {"prompt_id": "pid-recovered-42"})
        self.assertEqual(self.post_json.call_count, 1)
        self.assertEqual(self.sleep.call_count, 2)

    def test_absent_resubmits_once_safely(self):
        self.post_json.side_effect = [
            urllib.error.URLError(TimeoutError("read timed out")),
            {"prompt_id": "pid-second"},
        ]
        self.get_json.side_effect = [{"queue_running": [], "queue_pending": []}, {}]
        res = hrw._submit_prompt(
            "http://127.0.0.1:8188", {"1": {}}, "cid-p13", reconcile_delays=(1, 2),
        )
        self.assertEqual(res, {"prompt_id": "pid-second"})
        self.assertEqual(self.post_json.call_count, 2)

    def test_persistent_unknown_never_resubmits_and_raises_clear_error(self):
        self.post_json.side_effect = urllib.error.URLError(TimeoutError("read timed out"))
        self.get_json.side_effect = TimeoutError("comfyui wedged")
        with self.assertRaises(RuntimeError) as ctx:
            hrw._submit_prompt(
                "http://127.0.0.1:8188", {"1": {}}, "cid-p13", reconcile_delays=(1, 2, 4),
            )
        self.assertIn("cid-p13", str(ctx.exception))
        self.assertIn("unknown", str(ctx.exception))
        self.assertIn("refusing blind resubmit", str(ctx.exception))
        # CRITICAL: never issued a second POST /prompt while unknown!
        self.assertEqual(self.post_json.call_count, 1)
        self.assertEqual(self.sleep.call_count, 3)


class CompletedDeliveryTamperAndRestoreTests(unittest.TestCase):
    """P1-4: tamper-evident Drive verification & 0-CU atomic local restore."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.output_root = self.root / "output"
        self.payload = b"fake-validated-mp4-bytes-v031" * 100
        self.size = len(self.payload)
        self.md5 = hashlib.md5(self.payload).hexdigest()
        self.sha256 = hashlib.sha256(self.payload).hexdigest()
        self.manifest = {
            "run_id": "run-p14-1",
            "variant": "A",
            "task_state": "TASK_COMPLETED",
            "artifacts": {
                "drive_file_id": "drive-file-p14",
                "drive_output_folder_id": "drive-folder-p14",
                "local_frames": 274,
                "local_sha256": self.sha256,
                "output_size": self.size,
                "output_md5": self.md5,
                "output_sha256": self.sha256,
            },
        }
        self.patches = [
            mock.patch.object(hc, "ROOT", self.root),
            mock.patch.object(hc, "OUTPUT_ROOT", self.output_root),
            mock.patch.object(hc, "record", lambda d: None),
        ]
        for p in self.patches:
            p.start()
        self.addCleanup(mock.patch.stopall)
        self.addCleanup(self.td.cleanup)

    def _meta(self, **over):
        m = {
            "id": "drive-file-p14",
            "name": "A.mp4",
            "size": str(self.size),
            "md5Checksum": self.md5,
            "trashed": False,
        }
        m.update(over)
        return m

    def test_missing_local_mp4_restores_0cu_and_validates(self):
        def fake_download(file_id, dest_path, drive=None):
            Path(dest_path).write_bytes(self.payload)

        def fake_validate(path, **kw):
            return {"frame_count": 274, "sha256": self.sha256, "size": self.size, "technical_pass": True}

        with mock.patch("drive_depot.drive_service", return_value=object()), \
                mock.patch("drive_depot.get_file_evidence", return_value=self._meta()), \
                mock.patch.object(hc, "_download_drive_file", side_effect=fake_download), \
                mock.patch("h3_video_qa.validate_final_video", side_effect=fake_validate) as mock_val:
            status, evidence = hc._verify_completed_delivery(self.manifest, restore_local=True)

        self.assertEqual(status, "verified")
        self.assertTrue(evidence["local_video_validated"])
        self.assertTrue(evidence["restored_from_drive"])
        self.assertEqual(evidence["output_md5"], self.md5)
        self.assertEqual(evidence["output_sha256"], self.sha256)
        mock_val.assert_called_once()
        restored = self.output_root / "run-p14-1" / "A.mp4"
        self.assertTrue(restored.is_file())
        self.assertEqual(restored.read_bytes(), self.payload)

    def test_missing_local_mp4_readonly_probe_reports_local_validated_false(self):
        with mock.patch("drive_depot.drive_service", return_value=object()), \
                mock.patch("drive_depot.get_file_evidence", return_value=self._meta()):
            status, evidence = hc._verify_completed_delivery(self.manifest, restore_local=False)

        self.assertEqual(status, "verified")
        self.assertFalse(evidence["local_video_validated"])
        self.assertFalse(evidence["restored_from_drive"])
        self.assertFalse((self.output_root / "run-p14-1" / "A.mp4").exists())

    def test_remote_md5_tamper_rejected(self):
        tampered_meta = self._meta(md5Checksum="0" * 32)
        with mock.patch("drive_depot.drive_service", return_value=object()), \
                mock.patch("drive_depot.get_file_evidence", return_value=tampered_meta):
            status, reason = hc._verify_completed_delivery(self.manifest, restore_local=True)

        self.assertEqual(status, "lost")
        self.assertIn("md5Checksum mismatch", reason)

    def test_remote_size_tamper_rejected(self):
        tampered_meta = self._meta(size=str(self.size - 10))
        with mock.patch("drive_depot.drive_service", return_value=object()), \
                mock.patch("drive_depot.get_file_evidence", return_value=tampered_meta):
            status, reason = hc._verify_completed_delivery(self.manifest, restore_local=True)

        self.assertEqual(status, "lost")
        self.assertIn("size mismatch", reason)


class ReliabilityV031Tests(unittest.TestCase):
    """P1-5: Metric F lease cross-check, cold/warm stratification, Wilson CI."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        (self.root / "model_cache").mkdir(parents=True)
        self.addCleanup(self.td.cleanup)

    def _add_completed_run(self, task_id: str, run_id: str, *, warm: bool = False,
                           session_id: str = "sess-ok", cold_sec: float = 120.0):
        rm.create_run_manifest(self.root, task_id, run_id, variant="A")
        rm.update_run(self.root, run_id, task_state="GPU_ALLOCATED", session_id=session_id, session_state="ACTIVE")
        rm.update_run(self.root, run_id, task_state="MODELS_READY", last_verified_stage="MODELS_READY")
        rm.update_run(
            self.root, run_id, task_state="VIDEO_VALIDATED", last_verified_stage="VIDEO_VALIDATED",
            artifacts={"local_sha256": "a" * 64, "local_frames": 274, "delivery_state": "COMPLETE_LOCAL"},
        )
        rm.update_run(self.root, run_id, task_state="ARTIFACT_DURABLE")
        rm.update_run(self.root, run_id, task_state="GPU_STOP_CONFIRMED", session_state="STOPPED")
        rm.update_run(
            self.root, run_id, task_state="DRIVE_VERIFIED", last_verified_stage="DRIVE_VERIFIED",
            artifacts={
                "drive_file_id": "df-1",
                "delivery_state": "DELIVERED",
                "warm_session_reused": warm,
            },
            coldstart_metrics={"total_coldstart_sec": cold_sec, "warm_session_reused": warm, "wheelhouse_used": True},
        )
        rm.update_run(self.root, run_id, task_state="TASK_COMPLETED", session_state="STOPPED")

    def test_metric_f_catches_unreleased_lease_even_when_manifest_session_id_empty(self):
        task_id = "task-lease-orphan"
        _write_ledger(self.root, task_id, deadline_utc="2099-01-01T00:00:00Z", spent=5.0)
        # Attempt 1 died during allocation: session_id="" but lease still ACTIVE!
        rm.create_run_manifest(self.root, task_id, "run-alloc-crash", variant="A")
        rm.update_run(self.root, "run-alloc-crash", task_state="FAILED", session_state="UNKNOWN")
        reg = lr.LeaseRegistry(path=self.root / "model_cache" / "h3_gpu_leases.json", mirror=None)
        reg.register_lease(
            "sess-orphan-alloc", "run-alloc-crash", task_id, "G4",
            "2099-01-01T00:00:00+00:00", 25.0, "owner123", max_wall_sec=3600,
        )
        # Attempt 2 succeeded cleanly.
        self._add_completed_run(task_id, "run-ok-2", session_id="sess-ok-2")
        reg.register_lease(
            "sess-ok-2", "run-ok-2", task_id, "G4",
            "2099-01-01T00:00:00+00:00", 25.0, "owner123", max_wall_sec=3600,
        )
        reg.confirm_stopped("sess-ok-2", "STOPPED")

        report = rel.compute_reliability(self.root, now_utc="2026-11-21T00:00:00Z")
        f_metric = report["metrics"]["deadline_bound_e2e_success_rate"]
        self.assertEqual(f_metric["denominator"], 1)
        self.assertEqual(f_metric["numerator"], 0)
        self.assertEqual(f_metric["verdicts"][0]["verdict"], "failure")
        self.assertTrue(any("sess-orphan-alloc" in c for c in f_metric["verdicts"][0]["failed_conditions"]))

        # Once the orphan lease is confirmed STOPPED, Metric F passes.
        reg.confirm_stopped("sess-orphan-alloc", "STOPPED")
        report_after = rel.compute_reliability(self.root, now_utc="2026-11-21T00:00:00Z")
        self.assertEqual(report_after["metrics"]["deadline_bound_e2e_success_rate"]["numerator"], 1)

    def test_cold_vs_warm_stratification_and_wilson_ci(self):
        _write_ledger(self.root, "task-cold", deadline_utc="2099-01-01T00:00:00Z")
        _write_ledger(self.root, "task-warm", deadline_utc="2099-01-01T00:00:00Z")
        self._add_completed_run("task-cold", "run-cold", warm=False, cold_sec=185.0)
        self._add_completed_run("task-warm", "run-warm", warm=True, cold_sec=4.2)

        report = rep.build_report(self.root)
        self.assertIn("summary_comparison", report)
        self.assertEqual(report["summary_comparison"]["min_zero_failure_sample_for_95_lower_bound"], 73)
        self.assertIn("wilson_95_ci", report["metrics"]["first_attempt_success_rate"])
        self.assertIn("lower", report["metrics"]["first_attempt_success_rate"]["wilson_95_ci"])
        self.assertIn("upper", report["metrics"]["first_attempt_success_rate"]["wilson_95_ci"])

        breakdown = report["startup_mode_breakdown"]
        self.assertEqual(breakdown["coldstart_runs"]["runs_total"], 1)
        self.assertEqual(breakdown["coldstart_runs"]["startup_wall_sec"]["p50"], 185.0)
        self.assertEqual(breakdown["warm_reused_runs"]["runs_total"], 1)
        self.assertEqual(breakdown["warm_reused_runs"]["startup_wall_sec"]["p50"], 4.2)
        self.assertEqual(report["coldstart_metrics"]["coldstart_runs"]["runs"], 1)
        self.assertEqual(report["coldstart_metrics"]["warm_reused_runs"]["runs"], 1)


class PrivacyAndDriveAclTests(unittest.TestCase):
    """P1-6: local Drive ID overrides, Drive ACL 0-CU audit, secret_scan rules."""

    def test_resolve_drive_id_local_override_and_env(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "model_cache").mkdir()
            (root / "model_cache" / "local_drive_ids.json").write_text(
                json.dumps({"EXAMPLE_POSTER_DRIVE_ID": "local-real-id-123"}),
                encoding="utf-8",
            )
            self.assertEqual(hct.resolve_drive_id("EXAMPLE_POSTER_DRIVE_ID", root=root), "local-real-id-123")
            self.assertEqual(hct.resolve_drive_id("concrete-id-xyz", root=root), "concrete-id-xyz")
            with mock.patch.dict(os.environ, {"H3_DRIVE_ID_EXAMPLE_POSTER_DRIVE_ID": "env-override-999"}):
                self.assertEqual(hct.resolve_drive_id("EXAMPLE_POSTER_DRIVE_ID", root=root), "env-override-999")

    def test_audit_drive_acl_private_verified(self):
        svc = mock.MagicMock()
        svc.files().get().execute.return_value = {
            "id": "folder-1",
            "name": "Muse-H3-Model-Depot",
            "shared": False,
            "permissions": [{"id": "p1", "type": "user", "role": "owner"}],
        }
        res = dd.audit_drive_acl(svc, [("model_depot_root", "folder-1")])
        self.assertEqual(res["status"], "ACL_PRIVATE_VERIFIED")
        self.assertEqual(res["checked_count"], 1)
        self.assertEqual(res["public_violations"], [])

    def test_audit_drive_acl_blocks_anyone_permission(self):
        svc = mock.MagicMock()
        svc.files().get().execute.return_value = {
            "id": "folder-pub",
            "name": "Muse-H3-Model-Depot",
            "shared": True,
            "permissions": [
                {"id": "p1", "type": "user", "role": "owner"},
                {"id": "anyoneWithLink", "type": "anyone", "role": "reader"},
            ],
        }
        with self.assertRaises(RuntimeError) as ctx:
            dd.audit_drive_acl(svc, [("model_depot_root", "folder-pub")])
        self.assertIn("DRIVE_ACL_PUBLIC_EXPOSURE", str(ctx.exception))

    def test_audit_drive_acl_falls_back_to_permissions_list(self):
        svc = mock.MagicMock()
        svc.files().get().execute.return_value = {"id": "f2", "name": "weights", "shared": True}
        svc.permissions().list().execute.return_value = {
            "permissions": [{"id": "anyoneWithLink", "type": "anyone", "role": "reader"}]
        }
        with self.assertRaises(RuntimeError) as ctx:
            dd.audit_drive_acl(svc, [("model:w", "f2")])
        self.assertIn("DRIVE_ACL_PUBLIC_EXPOSURE", str(ctx.exception))

    def test_secret_scan_drive_id_classifier(self):
        self.assertFalse(sec._is_high_entropy_drive_id("EXAMPLE_POSTER_DRIVE_ID"))
        self.assertFalse(sec._is_high_entropy_drive_id("1" + "a" * 32))  # pure hex
        self.assertTrue(sec._is_high_entropy_drive_id("1AbCdEfGhIjKlMnOpQrStUvWxYz012345"))


if __name__ == "__main__":
    unittest.main()
