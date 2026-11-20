#!/usr/bin/env python3
"""macOS launchd guardian wrapper for independent overdue GPU lease shutdown.

Runs every 5 minutes via `com.colabworkstation.shutdown-guardian`:
1. Enforces Project 3 isolated credentials (never touches global main-account ADC).
2. Exits 0 quietly if isolated credentials have not been initialized yet.
3. Runs an initial read-only `audit` pass across the LeaseRegistry (Drive mirror
   first, local fallback).
4. For each unique `production_task_id` with overdue leases, executes task-scoped
   `run_executor(..., confirm_stop=True, max_stops=3, production_task_id=task_id)`
   so an orphaned GPU is genuinely stopped and confirmed STOPPED even if
   Antigravity or `h3_controller.py` exited unexpectedly.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _apply_isolated_env() -> None:
    os.environ.setdefault("COLAB_AUTH_MODE", "oauth2")
    os.environ.setdefault(
        "GOOGLE_APPLICATION_CREDENTIALS",
        str(ROOT / ".secrets" / "colab_drive_token.json"),
    )
    os.environ.setdefault("CLOUDSDK_CONFIG", str(ROOT / ".gcloud-colab"))
    os.environ.setdefault("H3_REQUIRE_ISOLATED_ADC", "1")


def run_guardian_pass(
    registry,
    *,
    now=None,
    stop_fn=None,
    owner_id_fn=None,
    query_fn=None,
    manifest_root=None,
    write_reports: bool = True,
) -> tuple[int, dict]:
    """Execute one guardian audit + stop pass.

    1. Runs a read-only audit across the LeaseRegistry.
    2. Whenever any expired `warm_idle` lease is found, unconditionally runs
       `run_executor(registry, confirm_stop=True, stop_warm_only=True)`
       regardless of `production_task_id` (cross-task warm-pool reclamation).
    3. For each unique `production_task_id` with normal overdue leases,
       runs task-scoped `run_executor(..., confirm_stop=True,
       production_task_id=task_id)`.
    """
    from h3_shutdown_executor import _write_report, run_executor

    audit_report = run_executor(
        registry,
        confirm_stop=False,
        now=now,
        owner_id_fn=owner_id_fn,
        query_fn=query_fn,
    )
    if write_reports:
        try:
            _write_report(audit_report)
        except Exception:  # noqa: BLE001
            pass

    overdue = audit_report.get("overdue") or []
    expired_warm = [item for item in overdue if item.get("warm_idle")]
    task_ids = sorted(
        {
            str(item.get("production_task_id") or "").strip()
            for item in overdue
            if not item.get("warm_idle")
            and str(item.get("production_task_id") or "").strip()
        }
    )
    if not expired_warm and not task_ids:
        rc = 2 if audit_report.get("owner_error") else 0
        return rc, audit_report

    exit_code = 0
    stop_reports = []

    if expired_warm:
        warm_stop_report = run_executor(
            registry,
            confirm_stop=True,
            max_stops=max(3, len(expired_warm)),
            now=now,
            stop_fn=stop_fn,
            owner_id_fn=owner_id_fn,
            query_fn=query_fn,
            stop_warm_only=True,
            manifest_root=manifest_root,
        )
        if write_reports:
            try:
                _write_report(warm_stop_report)
            except Exception:  # noqa: BLE001
                pass
        stop_reports.append(warm_stop_report)
        if warm_stop_report.get("owner_error") or int(warm_stop_report.get("exit_hint", 0)) == 2:
            exit_code = max(exit_code, 2)
        elif int(warm_stop_report.get("exit_hint", 0)) != 0:
            exit_code = max(exit_code, 1)

    for task_id in task_ids:
        stop_report = run_executor(
            registry,
            confirm_stop=True,
            max_stops=3,
            now=now,
            stop_fn=stop_fn,
            owner_id_fn=owner_id_fn,
            production_task_id=task_id,
            query_fn=query_fn,
            manifest_root=manifest_root,
        )
        if write_reports:
            try:
                _write_report(stop_report)
            except Exception:  # noqa: BLE001
                pass
        stop_reports.append(stop_report)
        if stop_report.get("owner_error") or int(stop_report.get("exit_hint", 0)) == 2:
            exit_code = max(exit_code, 2)
        elif int(stop_report.get("exit_hint", 0)) != 0:
            exit_code = max(exit_code, 1)

    summary = {
        "action": "guardian_stop_pass",
        "tasks_targeted": task_ids,
        "warm_leases_targeted": [item.get("session_id") for item in expired_warm],
        "reports": stop_reports,
    }
    return exit_code, summary


def main() -> int:
    _apply_isolated_env()
    from auth_support import audit_colab_oauth_token_file, audit_user_adc_file

    try:
        audit_user_adc_file()
        audit_colab_oauth_token_file()
    except FileNotFoundError:
        print(
            json.dumps(
                {
                    "action": "guardian_idle",
                    "reason": "awaiting_isolated_account_setup",
                }
            )
        )
        return 0

    from h3_gpu_lease_registry import DriveMirrorBackend, LeaseRegistry

    registry = LeaseRegistry(mirror=DriveMirrorBackend())
    exit_code, payload = run_guardian_pass(registry, write_reports=True)
    print(json.dumps(payload, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
