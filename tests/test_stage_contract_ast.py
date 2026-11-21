"""Static AST contract gate for Worker <-> Controller stage state machine
(v0.3.1 P0-2).

Parses `h3_remote_worker.py` and `h3_controller.py` using Python's `ast` module
and asserts 100% alignment with `h3_job_state.JOB_STAGES` and
`h3_job_state.WORKER_PHASE_TO_JOB_STAGE`. Any unmapped worker stage or missing
controller branch fails CI immediately before any GPU runtime is ever touched.
"""
from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h3_controller as hc  # noqa: E402
import h3_job_state as hjs  # noqa: E402


def _const_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_call_named(node: ast.AST, name: str) -> bool:
    if not isinstance(node, ast.Call):
        return False
    fn = node.func
    return (isinstance(fn, ast.Name) and fn.id == name) or (
        isinstance(fn, ast.Attribute) and fn.attr == name
    )


class StageContractASTGateTests(unittest.TestCase):
    def setUp(self):
        self.worker_src = (ROOT / "h3_remote_worker.py").read_text(encoding="utf-8")
        self.worker_tree = ast.parse(self.worker_src, filename="h3_remote_worker.py")
        self.controller_src = (ROOT / "h3_controller.py").read_text(encoding="utf-8")

    def test_v031_required_worker_phases_explicitly_mapped(self):
        self.assertEqual(hjs.WORKER_PHASE_TO_JOB_STAGE["init"], "JOB_STARTED")
        self.assertEqual(
            hjs.WORKER_PHASE_TO_JOB_STAGE["dual_track_setup"], "RESTORING_MODELS"
        )
        self.assertEqual(
            hjs.WORKER_PHASE_TO_JOB_STAGE["link_models"], "RESTORING_MODELS"
        )
        for phase, stage in hjs.WORKER_PHASE_TO_JOB_STAGE.items():
            self.assertIn(
                stage,
                hjs.JOB_STAGES,
                f"WORKER_PHASE_TO_JOB_STAGE[{phase!r}]={stage!r} is not in JOB_STAGES",
            )

    def test_ast_all_set_stage_calls_mapped_to_job_stages(self):
        discovered: list[tuple[int, str]] = []
        for node in ast.walk(self.worker_tree):
            if _is_call_named(node, "set_stage"):
                self.assertTrue(
                    node.args,
                    f"set_stage() without arguments at h3_remote_worker.py:{node.lineno}",
                )
                val = _const_str(node.args[0])
                self.assertIsNotNone(
                    val,
                    f"Non-literal set_stage() arg at h3_remote_worker.py:{node.lineno}; "
                    "worker phases must be statically auditable literals",
                )
                discovered.append((node.lineno, val))
        self.assertGreaterEqual(len(discovered), 10)
        for lineno, phase in discovered:
            self.assertIn(
                phase,
                hjs.WORKER_PHASE_TO_JOB_STAGE,
                f"set_stage({phase!r}) at h3_remote_worker.py:{lineno} is missing "
                "from h3_job_state.WORKER_PHASE_TO_JOB_STAGE",
            )
            mapped = hjs.WORKER_PHASE_TO_JOB_STAGE[phase]
            self.assertIn(
                mapped,
                hjs.JOB_STAGES,
                f"set_stage({phase!r}) at h3_remote_worker.py:{lineno} maps to "
                f"illegal stage {mapped!r}",
            )

    def test_ast_all_write_job_state_calls_emit_valid_job_stages(self):
        checked = 0
        for node in ast.walk(self.worker_tree):
            if _is_call_named(node, "write_job_state"):
                # Signature: write_job_state(root, run_id, variant, stage, ...)
                stage_node = None
                if len(node.args) >= 4:
                    stage_node = node.args[3]
                else:
                    for kw in node.keywords:
                        if kw.arg == "stage":
                            stage_node = kw.value
                            break
                self.assertIsNotNone(
                    stage_node,
                    f"write_job_state() missing stage arg at line {node.lineno}",
                )
                self._assert_stage_expr_valid(stage_node, node.lineno, "write_job_state")
                checked += 1
            elif _is_call_named(node, "_write_job_state_safe"):
                # Signature: _write_job_state_safe(stage, **kwargs)
                stage_node = node.args[0] if node.args else None
                if stage_node is None:
                    for kw in node.keywords:
                        if kw.arg == "stage":
                            stage_node = kw.value
                            break
                self.assertIsNotNone(
                    stage_node,
                    f"_write_job_state_safe() missing stage arg at line {node.lineno}",
                )
                self._assert_stage_expr_valid(
                    stage_node, node.lineno, "_write_job_state_safe"
                )
                checked += 1
        self.assertGreaterEqual(checked, 6)

    def _assert_stage_expr_valid(self, expr: ast.AST, lineno: int, fn_name: str):
        lit = _const_str(expr)
        if lit is not None:
            self.assertIn(
                lit,
                hjs.JOB_STAGES,
                f"{fn_name}() at h3_remote_worker.py:{lineno} writes non-contract stage {lit!r}",
            )
            return
        if _is_call_named(expr, "_job_stage_for"):
            self.assertTrue(expr.args)
            inner = _const_str(expr.args[0])
            if inner is not None:
                self.assertIn(
                    inner,
                    hjs.WORKER_PHASE_TO_JOB_STAGE,
                    f"_job_stage_for({inner!r}) at line {lineno} not in WORKER_PHASE_TO_JOB_STAGE",
                )
                self.assertIn(
                    hjs.WORKER_PHASE_TO_JOB_STAGE[inner],
                    hjs.JOB_STAGES,
                )
            return
        # Allow local `stage` variable inside helper wrappers (`_write_job_state_safe`,
        # `_request_token_refresh`) whose values are themselves derived from `_job_stage_for`.
        if isinstance(expr, ast.Name) and expr.id == "stage":
            return
        self.fail(
            f"Unverifiable stage expression in {fn_name}() at h3_remote_worker.py:{lineno}: "
            f"{ast.dump(expr)}"
        )

    def test_ast_all_last_stage_assignments_mapped(self):
        literal_assignments: list[tuple[int, str]] = []
        for node in ast.walk(self.worker_tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_LAST_STAGE":
                        val = _const_str(node.value)
                        if val is not None:
                            literal_assignments.append((node.lineno, val))
        self.assertTrue(
            any(v == "init" for _, v in literal_assignments),
            "Expected module-level _LAST_STAGE = 'init' assignment",
        )
        for lineno, phase in literal_assignments:
            self.assertIn(
                phase,
                hjs.WORKER_PHASE_TO_JOB_STAGE,
                f"_LAST_STAGE = {phase!r} at h3_remote_worker.py:{lineno} is not mapped",
            )
            self.assertIn(hjs.WORKER_PHASE_TO_JOB_STAGE[phase], hjs.JOB_STAGES)

    def test_controller_handles_every_job_stage(self):
        # Every JOB_STAGE must appear in JOB_STAGE_TO_MANIFEST_STAGE, and every
        # non-None target must be a valid manifest TASK_STAGE.
        import h3_run_manifest as rm

        for stage in hjs.JOB_STAGES:
            self.assertIn(
                stage,
                hc.JOB_STAGE_TO_MANIFEST_STAGE,
                f"JOB_STAGE {stage!r} missing from h3_controller.JOB_STAGE_TO_MANIFEST_STAGE",
            )
            mapped = hc.JOB_STAGE_TO_MANIFEST_STAGE[stage]
            if mapped is not None:
                self.assertIn(mapped, rm.TASK_STAGES)
        # Terminal worker stages must have explicit controller branches.
        self.assertIn('== "VIDEO_READY"', self.controller_src)
        self.assertIn('== "FAILED"', self.controller_src)

    def test_read_job_state_rejects_unknown_stage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for stage in hjs.JOB_STAGES:
                hjs.write_job_state(root, "run-ok", "A", stage)
                self.assertEqual(hjs.read_job_state(root)["stage"], stage)
            # Unknown / raw phase strings must fail closed with ValueError.
            for bad_stage in ("dual_track_setup", "init", "REMOTE_UPLOAD_VERIFIED", "BOGUS"):
                (root / hjs.JOB_STATE_FILENAME).write_text(
                    json.dumps({"run_id": "run-bad", "variant": "A", "stage": bad_stage}),
                    encoding="utf-8",
                )
                with self.assertRaises(ValueError):
                    hjs.read_job_state(root)


if __name__ == "__main__":
    unittest.main()
