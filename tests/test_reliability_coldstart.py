"""Offline unit tests for coldstart P50/P90 telemetry summary in
scripts/reliability_report.py (0 CU).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import h3_run_manifest as rm
import reliability_report as rr


def _add_run_with_coldstart(root: Path, run_id: str, task_id: str,
                            total_sec: float | None,
                            wh_used: bool | None = True,
                            mbps: float = 400.0) -> None:
    rm.create_run_manifest(root, task_id, run_id, variant="A")
    if total_sec is not None:
        metrics = {
            "schema_version": 1,
            "comfy_source_sec": 5.0,
            "pip_install_sec": 10.0 if wh_used else 65.0,
            "weights_download_sec": max(1.0, total_sec - 25.0),
            "effective_download_mbps": mbps,
            "comfy_boot_sec": 10.0,
            "total_coldstart_sec": total_sec,
            "wheelhouse_used": wh_used,
            "wheelhouse_fallback_reason": None if wh_used else "missing",
            "parallel_overlap_sec": 12.0,
            "download_workers": 12,
            "vram_mib": 81920,
            "highvram_enabled": True,
        }
        rm.update_run(root, run_id, coldstart_metrics=metrics)


class ReliabilityColdstartTest(unittest.TestCase):
    def test_percentile_math(self):
        self.assertIsNone(rr.percentile([], 50))
        self.assertEqual(rr.percentile([42.0], 50), 42.0)
        self.assertEqual(rr.percentile([42.0], 90), 42.0)
        # [100, 120, 140, 160, 200] -> P50 = 140.0, P90 = 160 + 0.6*(200-160) = 184.0
        vals = [200.0, 100.0, 160.0, 120.0, 140.0]
        self.assertEqual(rr.percentile(vals, 50), 140.0)
        self.assertEqual(rr.percentile(vals, 90), 184.0)

    def test_empty_root_coldstart_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            summary = rr.compute_coldstart_summary(td)
        self.assertEqual(summary["sample_size"], 0)
        self.assertEqual(summary["status"], "UNKNOWN")
        self.assertIsNone(summary["total_coldstart_sec_p50"])
        self.assertIsNone(summary["total_coldstart_sec_p90"])
        self.assertIsNone(summary["wheelhouse_hit_rate"])

    def test_multi_run_summary_and_task_filter(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Task 1: 4 runs with coldstart metrics + 1 run without metrics.
            _add_run_with_coldstart(root, "r1", "task-1", 100.0, wh_used=True, mbps=500.0)
            _add_run_with_coldstart(root, "r2", "task-1", 120.0, wh_used=True, mbps=450.0)
            _add_run_with_coldstart(root, "r3", "task-1", 140.0, wh_used=False, mbps=350.0)
            _add_run_with_coldstart(root, "r4", "task-1", 200.0, wh_used=True, mbps=300.0)
            _add_run_with_coldstart(root, "r5", "task-1", None)
            # Task 2: outlier run excluded when --task task-1 is passed.
            _add_run_with_coldstart(root, "r6", "task-2", 900.0, wh_used=False, mbps=50.0)

            s_all = rr.compute_coldstart_summary(root)
            self.assertEqual(s_all["sample_size"], 5)
            self.assertEqual(s_all["runs_in_scope"], 6)
            self.assertEqual(s_all["status"], "INSUFFICIENT_SAMPLE")
            self.assertEqual(s_all["wheelhouse_hits"], 3)
            self.assertEqual(s_all["wheelhouse_samples"], 5)
            self.assertEqual(s_all["wheelhouse_hit_rate"], 0.6)

            s_t1 = rr.compute_coldstart_summary(root, production_task_id="task-1")
            self.assertEqual(s_t1["sample_size"], 4)
            self.assertEqual(s_t1["runs_in_scope"], 5)
            self.assertEqual(s_t1["total_coldstart_sec_p50"], 130.0)
            self.assertEqual(s_t1["total_coldstart_sec_p90"], 182.0)
            self.assertEqual(s_t1["wheelhouse_hit_rate"], 0.75)

    def test_cli_json_preserves_existing_reliability_keys_and_adds_coldstart(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _add_run_with_coldstart(root, "r1", "task-cli", 115.0, wh_used=True)
            proc = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "reliability_report.py"), "--root", str(root)],
                capture_output=True, text=True, check=True,
            )
            payload = json.loads(proc.stdout)
            self.assertIn("metrics", payload)
            self.assertIn("first_attempt_success_rate", payload["metrics"])
            self.assertIn("deadline_bound_e2e_success_rate", payload["metrics"])
            self.assertIn("coldstart_metrics", payload)
            self.assertEqual(payload["coldstart_metrics"]["sample_size"], 1)
            self.assertEqual(payload["coldstart_metrics"]["total_coldstart_sec_p50"], 115.0)
            self.assertEqual(payload["coldstart_metrics"]["total_coldstart_sec_p90"], 115.0)


if __name__ == "__main__":
    unittest.main()
