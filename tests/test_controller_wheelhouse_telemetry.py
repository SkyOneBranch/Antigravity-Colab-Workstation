"""Offline unit tests for F2 controller wheelhouse wiring and F5 coldstart
telemetry parsing + durable manifest persistence (0 CU, no live Drive/GPU).
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h3_controller as ctrl
import h3_run_manifest as rm
from h3_contract import MODEL_FILES


def _write_verified_model_manifest(path: Path) -> Path:
    files = []
    for idx, rel in enumerate(sorted(MODEL_FILES)):
        files.append({
            "path": rel,
            "drive_file_id": f"file-id-{idx}",
            "size": 1000 + idx * 100,
            "sha256": f"{idx:x}" * 64,
            "server_md5": f"{idx:x}" * 32,
        })
    path.write_text(json.dumps({"files": files}), encoding="utf-8")
    return path


def _sample_coldstart_metrics(total_sec: float = 112.5, wh_used: bool = True) -> dict:
    return {
        "schema_version": 1,
        "comfy_source_sec": 4.2,
        "pip_install_sec": 12.1,
        "weights_download_sec": 85.4,
        "effective_download_mbps": 469.2,
        "comfy_boot_sec": 10.8,
        "total_coldstart_sec": total_sec,
        "wheelhouse_used": wh_used,
        "wheelhouse_fallback_reason": None if wh_used else "missing_entry",
        "parallel_overlap_sec": 16.0,
        "download_workers": 12,
        "vram_mib": 81920,
        "highvram_enabled": True,
    }


class ControllerWheelhouseWiringTest(unittest.TestCase):
    def test_wheelhouse_task_absent_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "comfyui_source_manifest.json"
            self.assertIsNone(ctrl._comfyui_wheelhouse_task(manifest_path=missing))

            no_wh = Path(td) / "no_wh.json"
            no_wh.write_text(json.dumps({
                "commit": "a" * 40,
                "archive_name": "comfyui-a.tar.gz",
            }), encoding="utf-8")
            self.assertIsNone(ctrl._comfyui_wheelhouse_task(manifest_path=no_wh))

    def test_wheelhouse_task_extracts_matching_entries(self):
        with tempfile.TemporaryDirectory() as td:
            commit = "61e524bdba7a4e193b45c2697f1af83c1beb86d6"
            manifest_path = Path(td) / "comfyui_source_manifest.json"
            manifest_path.write_text(json.dumps({
                "commit": commit,
                "archive_name": f"comfyui-{commit}.tar.gz",
                "wheelhouses": {
                    "py312": {
                        "archive_name": f"comfyui-wheelhouse-{commit}-py312.tar.gz",
                        "python_tag": "py312",
                        "commit": commit,
                        "size_bytes": 123456,
                        "sha256": "b" * 64,
                        "file_id": "wh-file-312",
                        "server_md5": "c" * 32,
                        "excluded_packages": ["torch", "torchvision", "torchaudio", "triton"],
                        "excluded_prefixes": ["nvidia-"],
                    },
                    "py311": {
                        "archive_name": "comfyui-wheelhouse-other-py311.tar.gz",
                        "python_tag": "py311",
                        "commit": "0" * 40,
                        "size_bytes": 99999,
                        "sha256": "d" * 64,
                        "file_id": "wh-file-311",
                    },
                },
            }), encoding="utf-8")
            desc = ctrl._comfyui_wheelhouse_task(manifest_path=manifest_path, commit=commit)
            self.assertIsInstance(desc, dict)
            self.assertEqual(list(desc.keys()), ["py312"])
            entry = desc["py312"]
            self.assertEqual(entry["file_id"], "wh-file-312")
            self.assertEqual(entry["size"], 123456)
            self.assertEqual(entry["sha256"], "b" * 64)
            self.assertEqual(entry["archive"], f"comfyui-wheelhouse-{commit}-py312.tar.gz")
            self.assertEqual(entry["python_tag"], "py312")
            self.assertIn("torch", entry["excluded_packages"])

    def test_build_drive_download_block_backwards_compat_and_wired(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            model_manifest = _write_verified_model_manifest(td_path / "verified_manifest.json")
            with mock.patch.object(ctrl, "_comfyui_download_task", return_value=None), \
                 mock.patch.object(ctrl, "_comfyui_wheelhouse_task", return_value=None):
                block = ctrl._build_drive_download_block(model_manifest)
            self.assertNotIn("wheelhouses", block)
            self.assertEqual(len(block["files"]), 4)

            wh_desc = {
                "py312": {
                    "file_id": "wh-1",
                    "size": 555,
                    "sha256": "e" * 64,
                    "archive": "comfyui-wheelhouse-abc-py312.tar.gz",
                    "commit": "a" * 40,
                    "python_tag": "py312",
                    "excluded_packages": ["torch"],
                    "excluded_prefixes": ["nvidia-"],
                    "server_md5": "f" * 32,
                }
            }
            with mock.patch.object(ctrl, "_comfyui_download_task", return_value=None), \
                 mock.patch.object(ctrl, "_comfyui_wheelhouse_task", return_value=wh_desc):
                block_wh = ctrl._build_drive_download_block(model_manifest, comfy_ref="a" * 40)
            self.assertEqual(block_wh["wheelhouses"], wh_desc)


class ColdstartTelemetryTest(unittest.TestCase):
    def test_parse_coldstart_metrics_valid_multiline_and_malformed(self):
        m1 = _sample_coldstart_metrics(total_sec=200.0, wh_used=False)
        m2 = _sample_coldstart_metrics(total_sec=98.4, wh_used=True)
        log = "\n".join([
            "WORKER_STAGE ensure_comfy",
            "COLDSTART_METRICS_JSON: {not-valid-json",
            "COLDSTART_METRICS_JSON: [1, 2, 3]",
            "COLDSTART_METRICS_JSON:" + json.dumps(m1),
            "COMFYUI_HEALTH_OK",
            "prefix COLDSTART_METRICS_JSON:" + json.dumps(m2),
        ])
        parsed = ctrl.parse_coldstart_metrics(log)
        self.assertEqual(parsed, m2)
        self.assertIsNone(ctrl.parse_coldstart_metrics("no marker here\n"))
        self.assertIsNone(ctrl.parse_coldstart_metrics(""))

    def test_run_manifest_persists_coldstart_metrics_idempotently(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rm.create_run_manifest(root, "task-cs-1", "run-cs-1", variant="A")
            metrics = _sample_coldstart_metrics(total_sec=105.2, wh_used=True)

            manifest, changed, anomaly = rm.update_run(
                root, "run-cs-1", coldstart_metrics=metrics
            )
            self.assertTrue(changed)
            self.assertIsNone(anomaly)
            self.assertEqual(manifest["coldstart_metrics"], metrics)

            # Second identical update must be a no-op (idempotent).
            _, changed_again, anomaly_again = rm.update_run(
                root, "run-cs-1", coldstart_metrics=metrics
            )
            self.assertFalse(changed_again)
            self.assertIsNone(anomaly_again)

            # Exposed in load_run_manifest and recovery_view.
            loaded = rm.load_run_manifest(root, "run-cs-1")
            self.assertEqual(loaded["coldstart_metrics"]["total_coldstart_sec"], 105.2)
            view = rm.recovery_view(root, "run-cs-1")
            self.assertEqual(view["coldstart_metrics"], metrics)

            # Audit log records COLDSTART_METRICS event once.
            events = [e for e in rm.read_audit_log(root, "run-cs-1") if e["event"] == "COLDSTART_METRICS"]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["detail"]["total_coldstart_sec"], 105.2)

    def test_record_coldstart_metrics_from_state_and_log(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rm.create_run_manifest(root, "task-cs-2", "run-from-log", variant="A")
            rm.create_run_manifest(root, "task-cs-2", "run-from-state", variant="A")

            m_log = _sample_coldstart_metrics(total_sec=120.0, wh_used=False)
            log_file = root / "worker_remote.log"
            log_file.write_text("booting\nCOLDSTART_METRICS_JSON:" + json.dumps(m_log) + "\n", encoding="utf-8")

            m_state = _sample_coldstart_metrics(total_sec=88.8, wh_used=True)
            with mock.patch.object(ctrl, "ROOT", root), \
                 mock.patch.object(ctrl, "RUNS", root / "logs"):
                res_log = ctrl._record_coldstart_metrics("run-from-log", worker_state={}, worker_log_path=log_file)
                res_state = ctrl._record_coldstart_metrics(
                    "run-from-state",
                    worker_state={"stage": "MODELS_READY", "coldstart_metrics": m_state},
                    worker_log_path=None,
                )

            self.assertEqual(res_log, m_log)
            self.assertEqual(res_state, m_state)
            self.assertEqual(rm.load_run_manifest(root, "run-from-log")["coldstart_metrics"], m_log)
            self.assertEqual(rm.load_run_manifest(root, "run-from-state")["coldstart_metrics"], m_state)

    def test_download_remote_worker_log_best_effort(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            def _fake_download(cmd, timeout=120):
                dest = Path(cmd[-1])
                dest.write_text("COLDSTART_METRICS_JSON:{}\n", encoding="utf-8")
                return {"returncode": 0, "stdout": "", "stderr": ""}

            with mock.patch.object(ctrl, "run_cli", side_effect=_fake_download):
                path = ctrl.download_remote_worker_log("sess-1", run_dir)
            self.assertEqual(path, run_dir / "worker_remote.log")

            with mock.patch.object(ctrl, "run_cli", side_effect=RuntimeError("offline")):
                self.assertIsNone(ctrl.download_remote_worker_log("sess-1", run_dir / "sub"))

    def test_manifest_worker_state_backfills_models_ready_monotonically(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Case 1: poll jumps from WORKER_STARTED -> INFERENCE_RUNNING
            rm.create_run_manifest(root, "task-bf", "run-bf-1", variant="A")
            rm.update_run(root, "run-bf-1", task_state="GPU_ALLOCATED",
                          session_id="sess-1", session_state="ACTIVE")
            rm.update_run(root, "run-bf-1", task_state="WORKER_STARTED")
            with mock.patch.object(ctrl, "ROOT", root), \
                 mock.patch.object(ctrl, "RUNS", root / "logs"):
                # RESTORING_MODELS must NOT backfill MODELS_READY
                ctrl._manifest_worker_state("run-bf-1", {"stage": "RESTORING_MODELS"})
                self.assertEqual(rm.load_run_manifest(root, "run-bf-1")["task_state"], "WORKER_STARTED")
                # INFERENCE_RUNNING backfills MODELS_READY
                ctrl._manifest_worker_state("run-bf-1", {"stage": "INFERENCE_RUNNING"})
                self.assertEqual(rm.load_run_manifest(root, "run-bf-1")["task_state"], "MODELS_READY")
                # Repeated INFERENCE_RUNNING does not create anomalies
                ctrl._manifest_worker_state("run-bf-1", {"stage": "INFERENCE_RUNNING"})
            anomalies = [e for e in rm.read_audit_log(root, "run-bf-1") if e["event"] == "ANOMALY"]
            self.assertEqual(anomalies, [])

            # Case 2: poll jumps directly from WORKER_STARTED -> INFERENCE_COMPLETED
            rm.create_run_manifest(root, "task-bf", "run-bf-2", variant="A")
            rm.update_run(root, "run-bf-2", task_state="GPU_ALLOCATED",
                          session_id="sess-2", session_state="ACTIVE")
            rm.update_run(root, "run-bf-2", task_state="WORKER_STARTED")
            with mock.patch.object(ctrl, "ROOT", root), \
                 mock.patch.object(ctrl, "RUNS", root / "logs"):
                ctrl._manifest_worker_state("run-bf-2", {"stage": "INFERENCE_COMPLETED"})
                # A subsequent poll with coldstart_metrics must NOT try backward move
                ctrl._manifest_worker_state("run-bf-2", {
                    "stage": "VIDEO_READY",
                    "coldstart_metrics": _sample_coldstart_metrics(),
                })
            stages = [e["detail"]["to"] for e in rm.read_audit_log(root, "run-bf-2") if e["event"] == "STAGE"]
            self.assertIn("MODELS_READY", stages)
            self.assertIn("INFERENCE_COMPLETED", stages)
            self.assertLess(stages.index("MODELS_READY"), stages.index("INFERENCE_COMPLETED"))
            anomalies2 = [e for e in rm.read_audit_log(root, "run-bf-2") if e["event"] == "ANOMALY"]
            self.assertEqual(anomalies2, [])


if __name__ == "__main__":
    unittest.main()
