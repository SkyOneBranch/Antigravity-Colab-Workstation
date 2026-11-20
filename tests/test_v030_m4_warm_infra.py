"""v0.3.0 M4 Warm-Pool Infrastructure unit & fault-injection tests.

Covers:
- h3_gpu_lease_registry.py: backward-compatible warm fields, mark_warm_idle,
  claim_warm_lease, find_warm_leases, confirm_stopped clearing warm state,
  fail-closed local write & best-effort mirror sync.
- h3_shutdown_executor.py & scripts/run_shutdown_guardian.py: cross-task
  reclamation of expired warm_idle leases, --stop-warm manual reclamation,
  preserved exit-2 refusal for normal cross-task stop, owner_id / claim /
  3-state STOPPED enforcement, and best-effort manifest STOPPED backfill.
- h3_run_manifest.py: WARM_POOLED session_state, armed warm-lease gate for
  TASK_COMPLETED, idempotent record_warm_session_stopped.
- h3_reliability.py & scripts/reliability_report.py: warm_session_hits,
  warm_pooled_active_runs, and strictly honest Metric D & Metric F shutdown
  accounting (WARM_POOLED != verified stopped until reclaimed).

Strictly 0.00 CU: never touches real Colab CLI, Drive, or credentials.
"""
from __future__ import annotations

import datetime
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import h3_gpu_lease_registry as lr
import h3_reliability as rel
import h3_run_manifest as rm
import h3_shutdown_executor as se
import reliability_report as rr
import run_shutdown_guardian as guardian
from h3_contract import TOTAL_FRAMES
from h3_gpu_lease_registry import (
    LeaseRegistry,
    MirrorUnavailableError,
    owner_id_for,
)

OWNER = owner_id_for("warm-owner@example.com")
OTHER_OWNER = owner_id_for("other-owner@example.com")
T0 = datetime.datetime(2026, 11, 20, 10, 0, 0, tzinfo=datetime.timezone.utc)


def _iso(dt: datetime.datetime) -> str:
    return dt.isoformat()


class FakeMirror:
    def __init__(self):
        self.doc = None
        self.fail_writes = 0
        self.writes = 0

    def read_mirror(self):
        return json.loads(json.dumps(self.doc)) if self.doc else None

    def write_mirror(self, doc, expected_version):
        if self.fail_writes > 0:
            self.fail_writes -= 1
            raise MirrorUnavailableError("mirror offline")
        self.writes += 1
        written = json.loads(json.dumps(doc))
        cur = (self.doc or {}).get("version") or 0
        written["version"] = max(int(doc.get("version") or 0), int(cur)) + 1
        written["mirror_file_id"] = "fake-mirror-1"
        self.doc = written
        return {"version": written["version"], "mirror_file_id": "fake-mirror-1"}


def _write_authorized_ledger(root: Path, task_id: str, deadline_utc: str,
                             spent_cu: float = 1.5, budget_cu: float = 30.0):
    cache = root / "model_cache"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"h3_production_budget_{task_id}.json"
    path.write_text(json.dumps({
        "production_task_id": task_id,
        "authorized": True,
        "project_budget_cu": budget_cu,
        "max_observed_spent_cu": spent_cu,
        "deadline_utc": deadline_utc,
    }), encoding="utf-8")


def _create_warm_completed_run(root: Path, task_id: str, run_id: str,
                               session_id: str, *, warm_reused: bool = False):
    rm.create_run_manifest(root, task_id, run_id, variant="A")
    rm.update_run(root, run_id, task_state="GPU_ALLOCATED",
                  session_id=session_id, session_state="ACTIVE")
    rm.update_run(root, run_id, task_state="WORKER_STARTED")
    rm.update_run(root, run_id, task_state="MODELS_READY")
    rm.update_run(root, run_id, task_state="INFERENCE_COMPLETED")
    rm.update_run(
        root, run_id, task_state="VIDEO_VALIDATED",
        artifacts={
            "local_sha256": "a" * 64,
            "local_frames": TOTAL_FRAMES,
            "delivery_state": "COMPLETE_LOCAL",
        },
        coldstart_metrics={
            "total_coldstart_sec": 3.2 if warm_reused else 120.0,
            "warm_session_reused": warm_reused,
            "wheelhouse_used": True,
        },
    )
    rm.update_run(root, run_id, task_state="ARTIFACT_DURABLE")
    rm.update_run(
        root, run_id,
        task_state="DRIVE_VERIFIED",
        session_state="WARM_POOLED",
        artifacts={
            "delivery_state": "DELIVERED",
            "drive_file_id": f"drive-{run_id}",
            "warm_lease_armed": True,
            "warm_expires_at_utc": _iso(T0 + datetime.timedelta(minutes=10)),
            **({"warm_session_reused": True} if warm_reused else {}),
        },
    )
    m, changed, anomaly = rm.update_run(root, run_id, task_state="TASK_COMPLETED")
    return m, changed, anomaly


class WarmLeaseRegistryTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.td, ignore_errors=True))
        self.mirror = FakeMirror()
        self.reg = LeaseRegistry(path=Path(self.td) / "leases.json",
                                 mirror=self.mirror)

    def _register(self, sid="sess-1", run_id="run-1", task_id="task-a",
                  owner=OWNER, started=T0):
        return self.reg.register_lease(
            session_id=sid,
            run_id=run_id,
            production_task_id=task_id,
            gpu_model="G4",
            stop_deadline_utc=_iso(started + datetime.timedelta(hours=1)),
            budget_cap_cu=25.0,
            owner_id=owner,
            started_at_utc=_iso(started),
        )

    def test_backward_compat_defaults_on_legacy_lease(self):
        legacy = {
            "leases": [{
                "session_id": "legacy-1",
                "run_id": "run-old",
                "production_task_id": "task-old",
                "owner_id": OWNER,
                "gpu_model": "G4",
                "started_at_utc": _iso(T0),
                "stop_deadline_utc": _iso(T0 + datetime.timedelta(hours=1)),
                "deadline_final": True,
                "budget_cap_cu": 25.0,
                "last_confirmed_state": "ACTIVE",
                "version": 1,
            }],
            "version": 1,
        }
        self.reg.path.write_text(json.dumps(legacy), encoding="utf-8")
        lease = self.reg.get_lease("legacy-1")
        self.assertIsNotNone(lease)
        self.assertFalse(lease["warm_idle"])
        self.assertIsNone(lease["warm_expires_at_utc"])
        self.assertEqual(lease["warm_ttl_sec"], 0)
        self.assertEqual(lease["session_budget_spent_cu"], 0.0)
        self.assertEqual(lease["warm_run_ids"], [])
        self.assertIsNone(lease["comfy_ref"])
        self.assertIsNone(lease["model_manifest_sha256"])
        self.assertIsNone(lease["rate_cu_h"])

    def test_mark_warm_idle_and_find_warm_leases(self):
        self._register("sess-old", "run-1", "task-a", started=T0)
        self._register("sess-new", "run-2", "task-b",
                       started=T0 + datetime.timedelta(minutes=5))
        self._register("sess-other-owner", "run-3", "task-c",
                       owner=OTHER_OWNER,
                       started=T0 + datetime.timedelta(minutes=6))

        exp1 = _iso(T0 + datetime.timedelta(minutes=15))
        exp2 = _iso(T0 + datetime.timedelta(minutes=20))
        l1 = self.reg.mark_warm_idle(
            "sess-old",
            warm_ttl_sec=600,
            warm_expires_at_utc=exp1,
            session_budget_spent_cu=1.25,
            run_id="run-1",
            comfy_ref="abc123",
            model_manifest_sha256="deadbeef",
            rate_cu_h=5.26,
        )
        self.assertTrue(l1["warm_idle"])
        self.assertTrue(l1["deadline_final"])
        self.assertEqual(l1["stop_deadline_utc"], exp1)
        self.assertEqual(l1["warm_expires_at_utc"], exp1)
        self.assertEqual(l1["warm_ttl_sec"], 600)
        self.assertEqual(l1["session_budget_spent_cu"], 1.25)
        self.assertEqual(l1["warm_run_ids"], ["run-1"])
        self.assertEqual(l1["comfy_ref"], "abc123")
        self.assertEqual(l1["model_manifest_sha256"], "deadbeef")
        self.assertEqual(l1["rate_cu_h"], 5.26)
        self.assertEqual(l1["version"], 2)
        self.assertIsNotNone(self.mirror.doc)

        self.reg.mark_warm_idle(
            "sess-new",
            warm_ttl_sec=600,
            warm_expires_at_utc=exp2,
            session_budget_spent_cu=0.9,
            run_id="run-2",
        )
        self.reg.mark_warm_idle(
            "sess-other-owner",
            warm_ttl_sec=600,
            warm_expires_at_utc=exp2,
            session_budget_spent_cu=0.5,
            run_id="run-3",
        )

        # At T0 + 10m: both owner leases are live, newest first
        found = self.reg.find_warm_leases(
            now=T0 + datetime.timedelta(minutes=10), owner_id=OWNER)
        self.assertEqual([x["session_id"] for x in found],
                         ["sess-new", "sess-old"])

        # At T0 + 16m: sess-old expired, only sess-new remains
        found_later = self.reg.find_warm_leases(
            now=T0 + datetime.timedelta(minutes=16), owner_id=OWNER)
        self.assertEqual([x["session_id"] for x in found_later], ["sess-new"])

    def test_claim_warm_lease_and_confirm_stopped_clears_warm_state(self):
        self._register("sess-1", "run-1", "task-a")
        exp = _iso(T0 + datetime.timedelta(minutes=15))
        self.reg.mark_warm_idle(
            "sess-1", warm_ttl_sec=600, warm_expires_at_utc=exp,
            session_budget_spent_cu=1.1, run_id="run-1")

        new_dl = _iso(T0 + datetime.timedelta(hours=2))
        claimed = self.reg.claim_warm_lease("sess-1", "run-2", "task-b", new_dl)
        self.assertIsNotNone(claimed)
        self.assertFalse(claimed["warm_idle"])
        self.assertIsNone(claimed["warm_expires_at_utc"])
        self.assertEqual(claimed["run_id"], "run-2")
        self.assertEqual(claimed["production_task_id"], "task-b")
        self.assertEqual(claimed["stop_deadline_utc"], new_dl)
        self.assertTrue(claimed["deadline_final"])
        self.assertEqual(claimed["warm_run_ids"], ["run-1", "run-2"])

        # Double-claim fails while no longer warm_idle
        self.assertIsNone(
            self.reg.claim_warm_lease("sess-1", "run-3", "task-c", new_dl))

        # Re-pool then confirm_stopped clears warm_idle & warm_expires_at_utc
        self.reg.mark_warm_idle(
            "sess-1", warm_ttl_sec=300, warm_expires_at_utc=exp,
            session_budget_spent_cu=2.2, run_id="run-2")
        self.assertTrue(self.reg.confirm_stopped("sess-1", "STOPPED"))
        stopped = self.reg.get_lease("sess-1")
        self.assertEqual(stopped["last_confirmed_state"], "STOPPED")
        self.assertFalse(stopped["warm_idle"])
        self.assertIsNone(stopped["warm_expires_at_utc"])
        # Cannot mark_warm_idle on a STOPPED lease
        self.assertIsNone(
            self.reg.mark_warm_idle(
                "sess-1", warm_ttl_sec=300, warm_expires_at_utc=exp,
                session_budget_spent_cu=2.2))

    def test_mark_warm_idle_local_failure_raises_and_mirror_failure_is_best_effort(self):
        self._register("sess-1", "run-1", "task-a")
        exp = _iso(T0 + datetime.timedelta(minutes=10))
        with patch.object(lr, "_atomic_write_json",
                          side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.reg.mark_warm_idle(
                    "sess-1", warm_ttl_sec=300, warm_expires_at_utc=exp,
                    session_budget_spent_cu=1.0)

        self.mirror.fail_writes = 99
        updated = self.reg.mark_warm_idle(
            "sess-1", warm_ttl_sec=300, warm_expires_at_utc=exp,
            session_budget_spent_cu=1.0)
        self.assertTrue(updated["warm_idle"])
        self.assertTrue(self.reg.mirror_pending())


class WarmShutdownExecutorAndGuardianTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.td, ignore_errors=True))
        self.root = Path(self.td)
        self.reg = LeaseRegistry(path=self.root / "leases.json", mirror=None)

    def _add_warm(self, sid, run_id, task_id, expires_dt, owner=OWNER):
        self.reg.register_lease(
            session_id=sid, run_id=run_id, production_task_id=task_id,
            gpu_model="G4",
            stop_deadline_utc=_iso(T0 + datetime.timedelta(hours=1)),
            budget_cap_cu=25.0, owner_id=owner, started_at_utc=_iso(T0))
        self.reg.mark_warm_idle(
            sid, warm_ttl_sec=300, warm_expires_at_utc=_iso(expires_dt),
            session_budget_spent_cu=1.0, run_id=run_id)

    def _add_normal_overdue(self, sid, run_id, task_id, deadline_dt, owner=OWNER):
        self.reg.register_lease(
            session_id=sid, run_id=run_id, production_task_id=task_id,
            gpu_model="G4",
            stop_deadline_utc=_iso(deadline_dt),
            budget_cap_cu=25.0, owner_id=owner, started_at_utc=_iso(T0))
        self.reg.update_deadline(sid, _iso(deadline_dt), final=True)

    def test_audit_bypasses_task_filter_for_expired_and_forced_warm(self):
        self._add_warm("warm-expired", "run-w1", "task-alpha",
                       T0 + datetime.timedelta(minutes=5))
        self._add_warm("warm-live", "run-w2", "task-beta",
                       T0 + datetime.timedelta(minutes=30))
        self._add_normal_overdue("norm-other", "run-n1", "task-gamma",
                                 T0 + datetime.timedelta(minutes=5))

        now = T0 + datetime.timedelta(minutes=10)
        # Task-scoped audit for task-unrelated STILL surfaces warm-expired
        rep = se.audit_leases(self.reg, now=now, owner_id=OWNER,
                              production_task_id="task-unrelated")
        self.assertEqual([x["session_id"] for x in rep["overdue"]],
                         ["warm-expired"])
        self.assertEqual(rep["overdue"][0]["reason"], "warm_ttl_expired")
        self.assertTrue(rep["overdue"][0]["warm_idle"])

        # force_stop_warm=True surfaces BOTH warm-expired and warm-live
        rep_force = se.audit_leases(self.reg, now=now, owner_id=OWNER,
                                    production_task_id="task-unrelated",
                                    force_stop_warm=True)
        self.assertEqual(sorted(x["session_id"] for x in rep_force["overdue"]),
                         ["warm-expired", "warm-live"])
        self.assertTrue(all(x["reason"] == "manual_stop_warm"
                            for x in rep_force["overdue"]))

    def test_normal_cross_task_stop_still_refused_exit_2(self):
        self._add_normal_overdue("norm-1", "run-n1", "task-a",
                                 T0 + datetime.timedelta(minutes=5))
        stopped = []
        rep = se.run_executor(
            self.reg, confirm_stop=True, production_task_id=None,
            now=T0 + datetime.timedelta(minutes=10),
            stop_fn=lambda sid: (stopped.append(sid), {"state": "STOPPED"})[1],
            owner_id_fn=lambda: OWNER,
        )
        self.assertEqual(rep["exit_hint"], 2)
        self.assertEqual(stopped, [])

    def test_stop_warm_only_cross_task_and_backfills_manifests(self):
        # Two expired warm leases across DIFFERENT tasks + 1 normal overdue + 1 wrong owner
        self._add_warm("warm-a", "run-wa", "task-alpha",
                       T0 + datetime.timedelta(minutes=5))
        self._add_warm("warm-b", "run-wb", "task-beta",
                       T0 + datetime.timedelta(minutes=6))
        self._add_warm("warm-foreign", "run-wf", "task-delta",
                       T0 + datetime.timedelta(minutes=5), owner=OTHER_OWNER)
        self._add_normal_overdue("norm-c", "run-nc", "task-gamma",
                                 T0 + datetime.timedelta(minutes=5))

        _create_warm_completed_run(self.root, "task-alpha", "run-wa", "warm-a")
        _create_warm_completed_run(self.root, "task-beta", "run-wb", "warm-b")

        stopped_calls = []

        def fake_stop(sid):
            stopped_calls.append(sid)
            return {"state": "STOPPED", "attempts": 1}

        rep = se.run_executor(
            self.reg,
            confirm_stop=True,
            max_stops=5,
            now=T0 + datetime.timedelta(minutes=10),
            stop_fn=fake_stop,
            owner_id_fn=lambda: OWNER,
            production_task_id=None,
            stop_warm_only=True,
            manifest_root=self.root,
        )
        self.assertEqual(rep["exit_hint"], 0)
        self.assertEqual(sorted(stopped_calls), ["warm-a", "warm-b"])
        # Foreign owner refused, normal overdue NOT touched
        self.assertEqual([x["session_id"] for x in rep["refused"]],
                         ["warm-foreign"])
        self.assertEqual(self.reg.get_lease("norm-c")["last_confirmed_state"],
                         "ACTIVE")
        # Manifests backfilled to STOPPED + GPU_STOP_CONFIRMED
        for rid in ("run-wa", "run-wb"):
            m = rm.load_run_manifest(self.root, rid)
            self.assertEqual(m["session_state"], "STOPPED")
            self.assertFalse(m["artifacts"]["warm_lease_armed"])
            stages = [e["detail"]["to"] for e in rm.read_audit_log(self.root, rid)
                      if e.get("event") == "STAGE"]
            self.assertIn("GPU_STOP_CONFIRMED", stages)

    def test_stop_warm_unconfirmed_stays_pending_and_respects_claim(self):
        self._add_warm("warm-unconf", "run-u1", "task-a",
                       T0 + datetime.timedelta(minutes=5))
        self._add_warm("warm-claimed", "run-u2", "task-b",
                       T0 + datetime.timedelta(minutes=5))
        self.reg.claim_lease("warm-claimed", "other-executor")

        rep = se.run_executor(
            self.reg,
            confirm_stop=True,
            now=T0 + datetime.timedelta(minutes=10),
            stop_fn=lambda sid: {"state": "UNKNOWN", "error": "timeout"},
            owner_id_fn=lambda: OWNER,
            stop_warm_only=True,
            manifest_root=self.root,
        )
        self.assertEqual(rep["exit_hint"], 1)
        self.assertEqual(rep["stopped"], [])
        reasons = {x["session_id"]: x["reason"] for x in rep["pending"]}
        self.assertEqual(reasons["warm-unconf"], "stop_unconfirmed")
        self.assertEqual(reasons["warm-claimed"], "claimed_by_other_executor")
        self.assertEqual(
            self.reg.get_lease("warm-unconf")["last_confirmed_state"], "ACTIVE")

    def test_cli_stop_warm_flag_and_exit_2_without_it(self):
        self._add_warm("warm-live", "run-wl", "task-z",
                       T0 + datetime.timedelta(minutes=30))

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc_refused = se.main([
                "stop", "--confirm-stop",
                "--registry", str(self.reg.path), "--no-mirror",
            ])
        self.assertEqual(rc_refused, 2)

        stopped = []
        buf2 = io.StringIO()
        with patch.object(se, "current_owner_id", return_value=OWNER), \
             patch.object(se, "_default_stop",
                          side_effect=lambda sid: (stopped.append(sid),
                                                   {"state": "STOPPED", "attempts": 1})[1]), \
             patch.object(se, "_write_report", return_value=self.root / "r.json"), \
             redirect_stdout(buf2):
            rc_warm = se.main([
                "stop", "--confirm-stop", "--stop-warm",
                "--registry", str(self.reg.path), "--no-mirror",
            ])
        self.assertEqual(rc_warm, 0)
        self.assertEqual(stopped, ["warm-live"])
        self.assertEqual(
            self.reg.get_lease("warm-live")["last_confirmed_state"], "STOPPED")

    def test_guardian_unconditionally_stops_expired_warm_across_tasks(self):
        self._add_warm("warm-t1", "run-wt1", "task-one",
                       T0 + datetime.timedelta(minutes=5))
        self._add_warm("warm-t2", "run-wt2", "task-two",
                       T0 + datetime.timedelta(minutes=6))
        self._add_warm("warm-not-expired", "run-wt3", "task-three",
                       T0 + datetime.timedelta(minutes=30))
        self._add_normal_overdue("norm-t4", "run-nt4", "task-four",
                                 T0 + datetime.timedelta(minutes=5))

        stopped = []

        def fake_stop(sid):
            stopped.append(sid)
            return {"state": "STOPPED", "attempts": 1}

        rc, summary = guardian.run_guardian_pass(
            self.reg,
            now=T0 + datetime.timedelta(minutes=10),
            stop_fn=fake_stop,
            owner_id_fn=lambda: OWNER,
            manifest_root=self.root,
            write_reports=False,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(summary["action"], "guardian_stop_pass")
        self.assertEqual(sorted(summary["warm_leases_targeted"]),
                         ["warm-t1", "warm-t2"])
        self.assertEqual(summary["tasks_targeted"], ["task-four"])
        self.assertEqual(sorted(stopped), ["norm-t4", "warm-t1", "warm-t2"])
        self.assertEqual(
            self.reg.get_lease("warm-not-expired")["last_confirmed_state"],
            "ACTIVE")


class WarmManifestAndReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.td, ignore_errors=True))
        self.root = Path(self.td)

    def test_manifest_warm_pooled_gate_and_idempotent_stop(self):
        rm.create_run_manifest(self.root, "task-1", "run-1", variant="A")
        rm.update_run(self.root, "run-1", task_state="DRIVE_VERIFIED",
                      session_id="sess-1", session_state="WARM_POOLED",
                      artifacts={"delivery_state": "DELIVERED",
                                 "drive_file_id": "fid-1"})

        # Unarmed WARM_POOLED must be REFUSED from TASK_COMPLETED
        m, changed, anomaly = rm.update_run(self.root, "run-1",
                                            task_state="TASK_COMPLETED")
        self.assertFalse(changed)
        self.assertIsNotNone(anomaly)
        self.assertEqual(m["task_state"], "DRIVE_VERIFIED")

        # Armed WARM_POOLED is ALLOWED to reach TASK_COMPLETED
        m, changed, anomaly = rm.update_run(
            self.root, "run-1", task_state="TASK_COMPLETED",
            artifacts={"warm_lease_armed": True})
        self.assertTrue(changed)
        self.assertIsNone(anomaly)
        self.assertEqual(m["task_state"], "TASK_COMPLETED")
        self.assertEqual(m["session_state"], "WARM_POOLED")

        # Wrong session_id returns None
        self.assertIsNone(
            rm.record_warm_session_stopped(self.root, "run-1",
                                           session_id="sess-other"))

        # Reclaim transitions WARM_POOLED -> STOPPED + GPU_STOP_CONFIRMED
        updated = rm.record_warm_session_stopped(self.root, "run-1",
                                                 session_id="sess-1")
        self.assertIsNotNone(updated)
        self.assertEqual(updated["session_state"], "STOPPED")
        self.assertFalse(updated["artifacts"]["warm_lease_armed"])
        stages = [e for e in rm.read_audit_log(self.root, "run-1")
                  if e.get("event") == "STAGE"
                  and e["detail"].get("to") == "GPU_STOP_CONFIRMED"]
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0]["detail"]["from"], "WARM_POOLED")

        # Idempotent second call does NOT duplicate audit event
        again = rm.record_warm_session_stopped(self.root, "run-1",
                                               session_id="sess-1")
        self.assertEqual(again["session_state"], "STOPPED")
        stages2 = [e for e in rm.read_audit_log(self.root, "run-1")
                   if e.get("event") == "STAGE"
                   and e["detail"].get("to") == "GPU_STOP_CONFIRMED"]
        self.assertEqual(len(stages2), 1)

    def test_reliability_honest_shutdown_and_warm_session_hits(self):
        deadline = _iso(T0 + datetime.timedelta(hours=2))
        _write_authorized_ledger(self.root, "task-warm", deadline, spent_cu=2.0)

        # Run 1 allocates and stays WARM_POOLED; Run 2 reuses it and stays WARM_POOLED
        _create_warm_completed_run(self.root, "task-warm", "run-1", "sess-w1",
                                   warm_reused=False)
        _create_warm_completed_run(self.root, "task-warm", "run-2", "sess-w1",
                                   warm_reused=True)

        # Before reclamation: WARM_POOLED must NOT count as verified stopped!
        rep_before = rel.compute_reliability(
            self.root, now_utc=_iso(T0 + datetime.timedelta(hours=3)))
        self.assertEqual(rep_before["warm_session_hits"], 1)
        self.assertEqual(rep_before["warm_pooled_active_runs"], 2)
        self.assertEqual(
            rep_before["metrics"]["verified_shutdown_rate"]["numerator"], 0)
        self.assertEqual(
            rep_before["metrics"]["verified_shutdown_rate"]["denominator"], 2)
        self.assertEqual(
            rep_before["metrics"]["deadline_bound_e2e_success_rate"]["numerator"], 0)
        self.assertEqual(
            rep_before["metrics"]["deadline_bound_e2e_success_rate"]["denominator"], 1)

        # Also check reliability_report.py enrichment
        full_before = rr.build_report(self.root)
        self.assertEqual(full_before["warm_session_hits"], 1)
        self.assertEqual(full_before["warm_pooled_active_runs"], 2)
        self.assertEqual(full_before["coldstart_metrics"]["warm_session_hits"], 1)
        self.assertEqual(full_before["coldstart_metrics"]["warm_session_hit_rate"], 0.5)

        # After reclamation: both runs transition to STOPPED + GPU_STOP_CONFIRMED
        rm.record_warm_session_stopped(self.root, "run-1", session_id="sess-w1")
        rm.record_warm_session_stopped(self.root, "run-2", session_id="sess-w1")

        # Ensure TASK_COMPLETED timestamp is before deadline (tests run at current wall clock)
        far_deadline = "2099-01-01T00:00:00+00:00"
        _write_authorized_ledger(self.root, "task-warm", far_deadline, spent_cu=2.0)

        rep_after = rel.compute_reliability(
            self.root, now_utc="2099-01-02T00:00:00+00:00")
        self.assertEqual(rep_after["warm_session_hits"], 1)
        self.assertEqual(rep_after["warm_pooled_active_runs"], 0)
        self.assertEqual(
            rep_after["metrics"]["verified_shutdown_rate"]["numerator"], 2)
        self.assertEqual(
            rep_after["metrics"]["verified_shutdown_rate"]["denominator"], 2)
        self.assertEqual(
            rep_after["metrics"]["verified_shutdown_rate"]["rate"], 1.0)
        self.assertEqual(
            rep_after["metrics"]["deadline_bound_e2e_success_rate"]["numerator"], 1)
        self.assertEqual(
            rep_after["metrics"]["deadline_bound_e2e_success_rate"]["denominator"], 1)


if __name__ == "__main__":
    unittest.main()
