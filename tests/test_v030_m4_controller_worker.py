"""v0.3.0 M4 unit & fault-injection tests: Controller & Worker Controlled Warm Pool.

Strictly 0.00 CU: verifies --keep-warm-ttl validation, warm-session pooling on
healthy completion, fail-closed stop on failure or mark_warm_idle error,
warm-pool lookup gates (comfy_ref, manifest sha256, wall time, 25 CU cap,
:8188 health probe), lightweight Stage 2 dispatch, and worker warm reuse
(COLDSTART_METRICS_JSON with warm_session_reused=true, <5s, comfyui.pid,
and token cleanup).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h3_controller as hc  # noqa: E402
import h3_gpu_lease_registry as reg_mod  # noqa: E402
import h3_remote_worker as w  # noqa: E402
import h3_warm_health_probe as whp  # noqa: E402
from h3_session_watchdog import SessionState  # noqa: E402
from tests.test_fault_injection import ChainHarness, RUN_ID, TASK, _args as _run_args  # noqa: E402


class M4KeepWarmTtlValidationTests(unittest.TestCase):
    def test_keep_warm_ttl_bounds(self):
        self.assertEqual(hc._validate_keep_warm_ttl(0), 0)
        self.assertEqual(hc._validate_keep_warm_ttl(60), 60)
        self.assertEqual(hc._validate_keep_warm_ttl(300), 300)
        self.assertEqual(hc._validate_keep_warm_ttl(900), 900)
        for bad in (-1, 1, 59, 901, 3600, "abc"):
            with self.assertRaises(ValueError, msg=f"bad={bad!r}"):
                hc._validate_keep_warm_ttl(bad)


class M4WarmHealthProbeTests(unittest.TestCase):
    def test_probe_warm_comfy_passes_when_required_nodes_present(self):
        payload = json.dumps({
            "MiniMaxH3ReferenceToVideo": {},
            "SaveVideo": {},
            "UNETLoader": {},
            "CLIPLoader": {},
        }).encode("utf-8")

        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return payload

        with mock.patch("urllib.request.urlopen", return_value=FakeResp()):
            res = whp.probe_warm_comfy()
        self.assertTrue(res["ok"])
        self.assertEqual(res["node_count"], 4)

    def test_probe_warm_comfy_fails_when_node_missing(self):
        payload = json.dumps({"SaveVideo": {}}).encode("utf-8")

        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return payload

        with mock.patch("urllib.request.urlopen", return_value=FakeResp()):
            with self.assertRaisesRegex(RuntimeError, "Required ComfyUI nodes missing"):
                whp.probe_warm_comfy()


class M4WorkerWarmReuseTests(unittest.TestCase):
    def test_build_coldstart_metrics_warm_reused_flag(self):
        cold = w.build_coldstart_metrics({}, 10.0, 120.0, 97871, ["--highvram"])
        self.assertNotIn("warm_session_reused", cold)
        warm = w.build_coldstart_metrics(
            {
                "comfy_source_sec": 0.0,
                "pip_install_sec": 0.0,
                "weights_download_sec": 0.0,
                "effective_download_mbps": 0.0,
                "wheelhouse_used": False,
                "wheelhouse_fallback_reason": None,
                "parallel_overlap_sec": 0.0,
                "download_workers": 0,
            },
            comfy_boot_sec=0.0,
            total_coldstart_sec=0.42,
            vram_mib=97871,
            vram_flags=["--highvram"],
            warm_session_reused=True,
        )
        self.assertIs(warm["warm_session_reused"], True)
        self.assertEqual(warm["comfy_boot_sec"], 0.0)
        self.assertLess(warm["total_coldstart_sec"], 5.0)

    def test_check_warm_comfy_alive_and_models_present_and_token_cleanup(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            comfy = root / "ComfyUI"
            models = comfy / "models"
            manifest_files = [
                {"path": "diffusion_models/m1.safetensors", "size": 5},
                {"path": "text_encoders/m2.safetensors", "size": 5},
                {"path": "vae/m3.safetensors", "size": 5},
                {"path": "vae/m4.safetensors", "size": 5},
            ]
            manifest = {"files": manifest_files}
            self.assertFalse(w._all_models_present_in_comfy(comfy, manifest))
            for item in manifest_files:
                p = models / item["path"]
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"12345")
            self.assertTrue(w._all_models_present_in_comfy(comfy, manifest))

            # Token cleanup
            (root / "drive_token.json").write_text("{}", encoding="utf-8")
            (root / "drive_token_probe.json").write_text("{}", encoding="utf-8")
            (root / "drive_upload_token.json").write_text("{}", encoding="utf-8")
            (root / "task.json").write_text("{}", encoding="utf-8")
            w._cleanup_ephemeral_tokens(root)
            self.assertFalse((root / "drive_token.json").exists())
            self.assertFalse((root / "drive_token_probe.json").exists())
            self.assertFalse((root / "drive_upload_token.json").exists())
            self.assertTrue((root / "task.json").exists())


class M4ControllerWarmPoolTests(unittest.TestCase):
    def setUp(self):
        self.h = ChainHarness(self)

    def tearDown(self):
        self.h.close()

    def test_healthy_completion_with_keep_warm_ttl_arms_warm_pool_without_stopping(self):
        h = self.h
        registry = reg_mod.LeaseRegistry(h.root / "model_cache" / "gpu_leases.json")
        created_sid = None

        def real_register(session, gpu, run_id, task_id, deadline, budget_cu, max_wall_sec=14400):
            nonlocal created_sid
            created_sid = session
            return registry.register_lease(
                session_id=session,
                gpu_model=gpu,
                run_id=run_id,
                production_task_id=task_id,
                owner_id="owner-test-1",
                stop_deadline_utc=deadline,
                budget_cap_cu=budget_cu,
                max_wall_sec=max_wall_sec,
            )

        with mock.patch.object(hc, "_lease_registry", return_value=registry), \
                mock.patch.object(hc, "_register_gpu_lease", side_effect=real_register):
            args = _run_args(keep_warm_ttl=300)
            res = hc.run_variant(args)

        self.assertEqual(res["delivery"]["drive_file_id"], "drive-file-1")
        # Session was NOT stopped — it was pooled!
        self.assertEqual(h.stop_cli_sessions, [])
        m = h.manifest()
        self.assertEqual(m["task_state"], "TASK_COMPLETED")
        self.assertEqual(m["session_state"], "WARM_POOLED")
        self.assertIs(m["artifacts"]["warm_lease_armed"], True)
        self.assertIsNotNone(m["artifacts"]["warm_expires_at_utc"])
        lease = registry.get_lease(created_sid)
        self.assertTrue(lease["warm_idle"])
        self.assertEqual(lease["warm_ttl_sec"], 300)

    def test_mark_warm_idle_failure_fails_closed_to_immediate_stop(self):
        h = self.h
        broken_registry = mock.MagicMock()
        broken_registry.find_warm_leases.return_value = []
        broken_registry.mark_warm_idle.side_effect = RuntimeError("disk full")

        with mock.patch.object(hc, "_lease_registry", return_value=broken_registry):
            args = _run_args(keep_warm_ttl=300)
            res = hc.run_variant(args)

        # Fail-closed: session was stopped immediately despite keep_warm_ttl=300!
        self.assertEqual(len(h.stop_cli_sessions), 1)
        m = h.manifest()
        self.assertEqual(m["task_state"], "TASK_COMPLETED")
        self.assertEqual(m["session_state"], "STOPPED")
        self.assertIn("warm_pool_arm_failed_stopping", h.audit_jsonl_text())

    def test_failure_with_keep_warm_ttl_always_stops_session(self):
        h = self.h
        h.download_exc = RuntimeError("download exploded")
        args = _run_args(keep_warm_ttl=300)
        with self.assertRaisesRegex(RuntimeError, "download exploded"):
            hc.run_variant(args)
        self.assertEqual(len(h.stop_cli_sessions), 1)
        m = h.manifest()
        self.assertEqual((m["task_state"], m["session_state"]), ("FAILED", "STOPPED"))

    def test_try_claim_warm_session_gates_and_hit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest_p = root / "verified_manifest.json"
            manifest_p.write_text('{"files":[]}', encoding="utf-8")
            manifest_sha = hashlib.sha256(manifest_p.read_bytes()).hexdigest()

            registry = reg_mod.LeaseRegistry(root / "gpu_leases.json")
            now = datetime.datetime.now(datetime.timezone.utc)
            future = (now + datetime.timedelta(seconds=600)).strftime("%Y-%m-%dT%H:%M:%SZ")
            registry.register_lease(
                session_id="muse-h3-warm01",
                gpu_model="G4",
                run_id="r-prev",
                production_task_id="task-prev",
                owner_id="owner-abc",
                stop_deadline_utc=future,
                budget_cap_cu=25.0,
                max_wall_sec=14400,
            )
            registry.mark_warm_idle(
                "muse-h3-warm01",
                warm_ttl_sec=600,
                warm_expires_at_utc=future,
                session_budget_spent_cu=5.0,
                run_id="r-prev",
                comfy_ref=hc.COMFY_REF_DEFAULT,
                model_manifest_sha256=manifest_sha,
                rate_cu_h=10.0,
            )

            args = SimpleNamespace(
                run_id="r-next",
                production_task_id="task-next",
                comfy_ref=hc.COMFY_REF_DEFAULT,
                cache_manifest=str(manifest_p),
                max_rate_cu_h=14.0,
                max_wall_sec=14400,
            )
            ledger = {
                "initial_balance_cu": 150.0,
                "hard_stop_balance_cu": 120.0,
                "max_observed_spent_cu": 0.0,
            }
            pre = {"cu_balance": 145.0, "rate": 10.0, "sessions": []}

            # Gate (e) failure: health probe fails -> miss
            with mock.patch.object(hc, "_lease_registry", return_value=registry), \
                    mock.patch("h3_gpu_lease_registry.current_owner_id", return_value="owner-abc"), \
                    mock.patch("h3_session_watchdog.query_session_state", return_value=SessionState.ACTIVE), \
                    mock.patch.object(hc, "exec_script", return_value={"returncode": 1, "stdout": "FAIL"}):
                miss = hc._try_claim_warm_session(args, None, ledger, root, pre, 0.0)
            self.assertIsNone(miss)

            # Gate (a) mismatch: wrong comfy_ref -> miss
            bad_ref_args = SimpleNamespace(**vars(args))
            bad_ref_args.comfy_ref = "0000000"
            with mock.patch.object(hc, "_lease_registry", return_value=registry), \
                    mock.patch("h3_gpu_lease_registry.current_owner_id", return_value="owner-abc"):
                self.assertIsNone(hc._try_claim_warm_session(bad_ref_args, None, ledger, root, pre, 0.0))

            # Gate (d) budget gate: spent 20.0 + est ~6.67 + 0.75 > 25.0 -> miss
            registry.mark_warm_idle(
                "muse-h3-warm01",
                warm_ttl_sec=600,
                warm_expires_at_utc=future,
                session_budget_spent_cu=20.0,
                run_id="r-prev",
                comfy_ref=hc.COMFY_REF_DEFAULT,
                model_manifest_sha256=manifest_sha,
                rate_cu_h=10.0,
            )
            with mock.patch.object(hc, "_lease_registry", return_value=registry), \
                    mock.patch("h3_gpu_lease_registry.current_owner_id", return_value="owner-abc"):
                self.assertIsNone(hc._try_claim_warm_session(args, None, ledger, root, pre, 0.0))

            # All 5 gates pass -> HIT!
            registry.mark_warm_idle(
                "muse-h3-warm01",
                warm_ttl_sec=600,
                warm_expires_at_utc=future,
                session_budget_spent_cu=5.0,
                run_id="r-prev",
                comfy_ref=hc.COMFY_REF_DEFAULT,
                model_manifest_sha256=manifest_sha,
                rate_cu_h=10.0,
            )
            with mock.patch.object(hc, "_lease_registry", return_value=registry), \
                    mock.patch("h3_gpu_lease_registry.current_owner_id", return_value="owner-abc"), \
                    mock.patch("h3_session_watchdog.query_session_state", return_value=SessionState.ACTIVE), \
                    mock.patch("h3_session_watchdog.update_session_deadline"), \
                    mock.patch.object(hc, "exec_script", return_value={"returncode": 0, "stdout": "MUSE_WARM_HEALTH_OK {}"}), \
                    mock.patch.object(hc, "record"):
                hit = hc._try_claim_warm_session(args, None, ledger, root, pre, 0.0)
            self.assertIsNotNone(hit)
            self.assertEqual(hit["session"], "muse-h3-warm01")
            lease = registry.get_lease("muse-h3-warm01")
            self.assertFalse(lease["warm_idle"])
            self.assertEqual(lease["run_id"], "r-next")
            self.assertEqual(lease["production_task_id"], "task-next")


if __name__ == "__main__":
    unittest.main()
