"""v0.3.0 M1 unit & fault-injection tests: GPU 503 elastic backoff & error classification.

Strictly 0.00 CU: all Colab CLI, Drive, and sleep calls are mocked/injected.
"""

from __future__ import annotations

import datetime
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h3_budget as hb  # noqa: E402
import h3_controller as hc  # noqa: E402
import h3_run_manifest as rm  # noqa: E402
from tests.test_fault_injection import ChainHarness, RUN_ID, TASK, _args as _run_args  # noqa: E402


class M1ErrorClassificationTests(unittest.TestCase):
    def test_classify_allocation_error_all_categories(self):
        # AUTH_FATAL
        for msg in (
            "HTTP 401 Unauthorized",
            "403 Forbidden",
            "Invalid credentials for account",
            "Permission denied on project",
        ):
            self.assertEqual(hb.classify_allocation_error(msg), "AUTH_FATAL", msg)
            self.assertEqual(hc._classify_allocation_error(msg), "AUTH_FATAL", msg)

        # QUOTA_EXCEEDED
        for msg in (
            "Quota exceeded for compute units",
            "Usage limit reached",
            "Compute units exhausted",
            "Insufficient CU balance",
            "Billing disabled for account",
        ):
            self.assertEqual(hb.classify_allocation_error(msg), "QUOTA_EXCEEDED", msg)

        # CAPACITY_TRANSIENT
        for msg in (
            "HTTP 503 Service Unavailable",
            "HTTP 429 Too Many Requests",
            "502 Bad Gateway",
            "504 Gateway Timeout",
            "GPU stockout in region",
            "Resource exhausted",
            "No backend available",
            "status 412 precondition failed",
            "capacity limit reached",
            "Too many assignments",
        ):
            self.assertEqual(hb.classify_allocation_error(msg), "CAPACITY_TRANSIENT", msg)

        # TRANSPORT_TRANSIENT
        for msg in (
            "ProxyError: tunnel failed",
            "RemoteDisconnected: closed connection",
            "SSLEOFError: EOF occurred in violation of protocol",
            "TLS handshake failed",
            "HTTP 411 Length Required",
            "Read timed out after 30s",
            "Connection reset by peer",
        ):
            self.assertEqual(hb.classify_allocation_error(msg), "TRANSPORT_TRANSIENT", msg)

        # UNKNOWN_FATAL
        for msg in ("SyntaxError in CLI", "Unexpected JSON payload", ""):
            self.assertEqual(hb.classify_allocation_error(msg), "UNKNOWN_FATAL", msg)

    def test_compute_backoff_delay_schedule_and_jitter(self):
        self.assertEqual(hb.ALLOCATION_BACKOFF_BASE_SEC, (15.0, 45.0, 90.0))
        self.assertAlmostEqual(hb.ALLOCATION_BACKOFF_JITTER, 0.20)
        self.assertEqual(hb.compute_backoff_delay(0, jitter_fn=lambda: 1.0), 15.0)
        self.assertEqual(hb.compute_backoff_delay(1, jitter_fn=lambda: 1.0), 45.0)
        self.assertEqual(hb.compute_backoff_delay(2, jitter_fn=lambda: 1.0), 90.0)
        # Clamped to last base entry
        self.assertEqual(hb.compute_backoff_delay(4, jitter_fn=lambda: 1.0), 90.0)
        # ±20% jitter bounds
        for _ in range(50):
            d = hb.compute_backoff_delay(0)
            self.assertGreaterEqual(d, 15.0 * 0.8)
            self.assertLessEqual(d, 15.0 * 1.2)

    def test_resolve_allocation_rounds_defaults_and_bounds(self):
        self.assertEqual(hc._resolve_allocation_rounds(SimpleNamespace(allocation_rounds=None, action="run")), 1)
        self.assertEqual(hc._resolve_allocation_rounds(SimpleNamespace(allocation_rounds=None, action="deliver")), 3)
        self.assertEqual(hc._resolve_allocation_rounds(SimpleNamespace(allocation_rounds=None, auto_recover=True)), 3)
        self.assertEqual(hc._resolve_allocation_rounds(SimpleNamespace(allocation_rounds=5)), 5)
        with self.assertRaises(ValueError):
            hc._resolve_allocation_rounds(SimpleNamespace(allocation_rounds=0))
        with self.assertRaises(ValueError):
            hc._resolve_allocation_rounds(SimpleNamespace(allocation_rounds=6))


class M1ElasticBackoffExecutionTests(unittest.TestCase):
    def setUp(self):
        self.h = ChainHarness(self)

    def tearDown(self):
        self.h.close()

    def test_round1_503_exhaustion_sleeps_and_succeeds_in_round2(self):
        h = self.h
        # Round 1: G4 (503), G4 (503), A100 (503) -> backoff 15.0s -> Round 2: G4 succeeds!
        h.new_failures = [
            RuntimeError("HTTP 503 Service Unavailable"),
            RuntimeError("HTTP 503 Service Unavailable"),
            RuntimeError("HTTP 503 Service Unavailable"),
        ]
        sleeps = []
        args = _run_args(
            allocation_rounds=3,
            sleep_fn=lambda s: sleeps.append(s),
            jitter_fn=lambda: 1.0,
        )
        res = hc.run_variant(args)
        self.assertEqual(h.manifest_state()[0], "TASK_COMPLETED")
        self.assertEqual(res["delivery"]["drive_file_id"], "drive-file-1")
        # 3 failed attempts + 1 succeeded attempt = 4 allocations attempted
        self.assertEqual(len(h.allocate_ids), 4)
        # Backoff delay of 15.0s recorded (alongside 5s late-alloc checks)
        self.assertIn(15.0, sleeps)
        # GPU_ALLOCATION_BACKOFF recorded in both h3_audit.jsonl and run manifest audit log
        self.assertIn("GPU_ALLOCATION_BACKOFF", h.audit_jsonl_text())
        audit_path = rm._audit_path(h.root, RUN_ID)
        audit_events = [
            json.loads(line)
            for line in audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        backoff_events = [
            e for e in audit_events
            if e.get("event") == "GPU_ALLOCATION_BACKOFF"
        ]
        self.assertEqual(len(backoff_events), 1)
        self.assertEqual(backoff_events[0]["detail"]["round"], 1)
        self.assertEqual(backoff_events[0]["detail"]["sleep_sec"], 15.0)

    def test_auth_fatal_and_quota_fail_fast_without_backoff(self):
        self.h.close()
        for err_msg in ("HTTP 401 Unauthorized", "Quota exceeded for project"):
            h = ChainHarness(self)
            try:
                h.new_failures = [RuntimeError(err_msg)]
                sleeps = []
                args = _run_args(
                    allocation_rounds=3,
                    sleep_fn=lambda s: sleeps.append(s),
                    jitter_fn=lambda: 1.0,
                )
                with self.assertRaisesRegex(RuntimeError, "GPU allocation failed"):
                    hc.run_variant(args)
                # Only 1 attempt: no A100 fallback, no round 2 backoff
                self.assertEqual(len(h.allocate_ids), 1)
                self.assertNotIn(15.0, sleeps)
            finally:
                h.close()

    def test_deadline_prevents_sleeping_into_next_round(self):
        h = self.h
        h.new_failures = [
            RuntimeError("HTTP 503 Service Unavailable"),
            RuntimeError("HTTP 503 Service Unavailable"),
            RuntimeError("HTTP 503 Service Unavailable"),
        ]
        # Deadline only 100s from now (< MIN_ATTEMPT_WALL_SEC + DEADLINE_MARGIN_SEC = 2700s)
        soon = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=100)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        sleeps = []
        args = _run_args(
            allocation_rounds=3,
            task_deadline_utc=soon,
            sleep_fn=lambda s: sleeps.append(s),
            jitter_fn=lambda: 1.0,
        )
        with self.assertRaisesRegex(RuntimeError, "task deadline"):
            hc.run_variant(args)
        # Aborted immediately without sleeping 15.0s
        self.assertNotIn(15.0, sleeps)
        self.assertEqual(len(h.allocate_ids), 3)

    def test_undersized_vram_still_stops_session_immediately(self):
        h = self.h
        h.probe_output = "RTX PRO 6000 Blackwell, 40960"
        args = _run_args(allocation_rounds=3, sleep_fn=lambda s: None)
        with self.assertRaisesRegex(RuntimeError, "VRAM insufficient"):
            hc.run_variant(args)
        self.assertEqual(len(h.allocate_ids), 1)
        self.assertEqual(len(h.stop_cli_sessions), 1)

    def test_stale_historical_ledger_deadline_ignored_for_backoff(self):
        h = self.h
        # Ledger carries a historical deadline already in the past; args.task_deadline_utc is unset.
        h.new_failures = [
            RuntimeError("HTTP 503 Service Unavailable"),
            RuntimeError("HTTP 503 Service Unavailable"),
            RuntimeError("HTTP 503 Service Unavailable"),
        ]
        sleeps = []
        stale_ledger = {
            "max_observed_spent_cu": 0.0,
            "production_task_id": TASK,
            "project_budget_cu": 30.0,
            "hard_stop_balance_cu": 132.77,
            "deadline_utc": "2020-01-01T00:00:00Z",
        }
        with mock.patch.object(hc, "initialize_project_budget", return_value=stale_ledger):
            args = _run_args(
                allocation_rounds=3,
                sleep_fn=lambda s: sleeps.append(s),
                jitter_fn=lambda: 1.0,
            )
            res = hc.run_variant(args)
        self.assertEqual(h.manifest_state()[0], "TASK_COMPLETED")
        self.assertEqual(res["delivery"]["drive_file_id"], "drive-file-1")
        self.assertIn(15.0, sleeps)
        self.assertIn("stale_ledger_deadline_ignored", h.audit_jsonl_text())

    def test_failed_allocation_attempts_mark_leases_stopped_immediately(self):
        import h3_gpu_lease_registry as reg_mod

        h = self.h
        registry = reg_mod.LeaseRegistry(h.root / "model_cache" / "gpu_leases.json")
        registered_sids = []

        def real_register(session, gpu, run_id, task_id, deadline, budget_cu, max_wall_sec=14400):
            registered_sids.append(session)
            return registry.register_lease(
                session_id=session,
                gpu_model=gpu,
                run_id=run_id,
                production_task_id=task_id,
                owner_id="owner-m1-test",
                stop_deadline_utc=deadline,
                budget_cap_cu=budget_cu,
                max_wall_sec=max_wall_sec,
            )

        def real_confirm_stopped(session):
            registry.confirm_stopped(session, "STOPPED")

        # Full Round 1 (G4, G4, A100) fails with 503, then Round 2 G4 succeeds
        h.new_failures = [
            RuntimeError("HTTP 503 Service Unavailable"),
            RuntimeError("HTTP 503 Service Unavailable"),
            RuntimeError("HTTP 503 Service Unavailable"),
        ]
        with mock.patch.object(hc, "_lease_registry", return_value=registry), \
                mock.patch.object(hc, "_register_gpu_lease", side_effect=real_register), \
                mock.patch.object(hc, "_confirm_gpu_lease_stopped", side_effect=real_confirm_stopped):
            args = _run_args(allocation_rounds=2, sleep_fn=lambda s: None, jitter_fn=lambda: 1.0)
            hc.run_variant(args)

        self.assertEqual(len(registered_sids), 4)
        # All 4 leases (3 failed attempts + 1 completed run) are STOPPED — zero phantom active/not_yet_due leases!
        for sid in registered_sids:
            lease = registry.get_lease(sid)
            self.assertEqual(lease["last_confirmed_state"], "STOPPED", f"lease {sid} still active/not_yet_due")
        non_stopped = [l for l in registry.list_leases() if l.get("last_confirmed_state") != "STOPPED"]
        self.assertEqual(non_stopped, [])


if __name__ == "__main__":
    unittest.main()
