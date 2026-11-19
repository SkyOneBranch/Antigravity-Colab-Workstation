#!/usr/bin/env python3
"""Print the TASK-005 §12 reliability report + v0.2.0 coldstart telemetry
summary for a run-manifest root.

Reads only: run manifests, audit logs, and budget ledgers under --root.
Prints the JSON report from h3_reliability.compute_reliability enriched
with coldstart P50/P90 and wheelhouse hit-rate metrics.

Example:
    python3 scripts/reliability_report.py --root ~/workspace/Muse_Colab_H3_Drive_v3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h3_run_manifest as rm
from h3_reliability import MIN_SAMPLE_FOR_MEASURED, compute_reliability

COLDSTART_NUMERIC_FIELDS = (
    "total_coldstart_sec",
    "comfy_source_sec",
    "pip_install_sec",
    "weights_download_sec",
    "effective_download_mbps",
    "comfy_boot_sec",
    "parallel_overlap_sec",
)


def percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolation percentile (pct in [0, 100]), rounded to 3 decimals.
    Returns None for an empty sequence."""
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    pos = (len(ordered) - 1) * (float(pct) / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return round(ordered[lo], 3)
    frac = pos - lo
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * frac, 3)


def _filtered_manifests(root: str | Path, production_task_id: str | None = None,
                        since: str | None = None, until: str | None = None) -> list[dict]:
    out = []
    for manifest in rm.list_runs(Path(root)):
        if not isinstance(manifest, dict) or not manifest.get("run_id"):
            continue
        if production_task_id and manifest.get("production_task_id") != production_task_id:
            continue
        created = manifest.get("created_at_utc") or ""
        if since and created < since:
            continue
        if until and created > until:
            continue
        out.append(manifest)
    return out


def compute_coldstart_summary(root: str | Path,
                              production_task_id: str | None = None,
                              since: str | None = None,
                              until: str | None = None) -> dict:
    """Aggregate coldstart P50/P90 and wheelhouse hit-rate from persisted
    run manifests under `root`."""
    manifests = _filtered_manifests(root, production_task_id, since, until)
    samples = [
        m["coldstart_metrics"] for m in manifests
        if isinstance(m.get("coldstart_metrics"), dict) and m["coldstart_metrics"]
    ]
    n = len(samples)
    if n == 0:
        status = "UNKNOWN"
    elif n < MIN_SAMPLE_FOR_MEASURED:
        status = "INSUFFICIENT_SAMPLE"
    else:
        status = "MEASURED"

    percentiles_map = {}
    for field in COLDSTART_NUMERIC_FIELDS:
        vals = [
            float(s[field]) for s in samples
            if isinstance(s.get(field), (int, float))
            and not isinstance(s.get(field), bool)
            and math.isfinite(float(s[field]))
        ]
        if vals:
            percentiles_map[field] = {
                "count": len(vals),
                "p50": percentile(vals, 50),
                "p90": percentile(vals, 90),
                "min": round(min(vals), 3),
                "max": round(max(vals), 3),
            }
        else:
            percentiles_map[field] = {
                "count": 0,
                "p50": None,
                "p90": None,
                "min": None,
                "max": None,
            }

    wh_bools = [s["wheelhouse_used"] for s in samples if isinstance(s.get("wheelhouse_used"), bool)]
    wh_hits = sum(1 for b in wh_bools if b)
    wh_rate = round(wh_hits / len(wh_bools), 4) if wh_bools else None

    total_stats = percentiles_map["total_coldstart_sec"]
    return {
        "sample_size": n,
        "runs_in_scope": len(manifests),
        "status": status,
        "total_coldstart_sec_p50": total_stats["p50"],
        "total_coldstart_sec_p90": total_stats["p90"],
        "wheelhouse_hits": wh_hits,
        "wheelhouse_samples": len(wh_bools),
        "wheelhouse_hit_rate": wh_rate,
        "percentiles": percentiles_map,
    }


def build_report(root: str | Path, production_task_id: str | None = None,
                 since: str | None = None, until: str | None = None,
                 output_root: str | Path | None = None) -> dict:
    report = compute_reliability(
        root, production_task_id=production_task_id,
        since=since, until=until,
        output_root=output_root)
    report["coldstart_metrics"] = compute_coldstart_summary(
        root, production_task_id=production_task_id,
        since=since, until=until)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True,
                        help="production root holding model_cache/run_manifests/")
    parser.add_argument("--task", default=None,
                        help="only runs for this production_task_id")
    parser.add_argument("--since", default=None,
                        help="only runs created at or after this UTC timestamp")
    parser.add_argument("--until", default=None,
                        help="only runs created at or before this UTC timestamp")
    parser.add_argument("--output-root", default=None,
                        help="directory holding <run_id>/<variant>.mp4 files "
                             "(metric C also verifies the local file)")
    args = parser.parse_args()
    report = build_report(
        args.root, production_task_id=args.task,
        since=args.since, until=args.until,
        output_root=args.output_root)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
