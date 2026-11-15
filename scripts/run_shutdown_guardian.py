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
sys.path.insert(0, str(ROOT))

os.environ.setdefault("COLAB_AUTH_MODE", "oauth2")
os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS",
    str(ROOT / ".secrets" / "colab_drive_token.json"),
)
os.environ.setdefault("CLOUDSDK_CONFIG", str(ROOT / ".gcloud-colab"))
os.environ.setdefault("H3_REQUIRE_ISOLATED_ADC", "1")


def main() -> int:
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
    from h3_shutdown_executor import _write_report, run_executor

    registry = LeaseRegistry(mirror=DriveMirrorBackend())
    audit_report = run_executor(registry, confirm_stop=False)
    try:
        _write_report(audit_report)
    except Exception:  # noqa: BLE001
        pass

    overdue = audit_report.get("overdue") or []
    task_ids = sorted(
        {
            str(item.get("production_task_id") or "").strip()
            for item in overdue
            if str(item.get("production_task_id") or "").strip()
        }
    )
    if not task_ids:
        print(json.dumps(audit_report, sort_keys=True))
        if audit_report.get("owner_error"):
            return 2
        return 0

    exit_code = 0
    stop_reports = []
    for task_id in task_ids:
        stop_report = run_executor(
            registry,
            confirm_stop=True,
            max_stops=3,
            production_task_id=task_id,
        )
        try:
            _write_report(stop_report)
        except Exception:  # noqa: BLE001
            pass
        stop_reports.append(stop_report)
        if stop_report.get("owner_error") or int(stop_report.get("exit_hint", 0)) == 2:
            exit_code = max(exit_code, 2)
        elif int(stop_report.get("exit_hint", 0)) != 0:
            exit_code = max(exit_code, 1)

    print(
        json.dumps(
            {
                "action": "guardian_stop_pass",
                "tasks_targeted": task_ids,
                "reports": stop_reports,
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
