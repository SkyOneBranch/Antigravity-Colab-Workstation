# CHANGELOG

## v0.3.1 — 2026-11-21 — 0.00 CU Self-Healing Closure, Idempotent Tamper-Proof Delivery & Privacy Hardening (P0-1 ~ P1-6)

Zero-CU reliability and security hardening release closing the remaining six
self-healing, protocol, and public-repository governance gaps discovered during
the post-v0.3.0 deep architecture audit. Verified end-to-end at **0.00 CU**
(zero Colab sessions created).

### Added / Fixed

- **P0-1 Allocation-Window Orphan Closure (`h3_controller.py`, `h3_run_manifest.py`, `h3_gpu_lease_registry.py`, `h3_session_watchdog.py`)**:
  - Atomically persists `pending_session_id`, `session_owner_id`, and `session_state="ALLOCATING"` into `run_manifest.json` **before** calling blocking `colab new` (fully backward-compatible with v1–v3 manifests).
  - Added owner-verified reconciliation (`_reconcile_pending_allocation`, `LeaseRegistry.verify_session_ownership`) on `--deliver` / `resume` / `reconcile_on_startup`: cleans up absent `allocating` records and sibling leases automatically; safely promotes verified `run_id + owner_id` active sessions to `GPU_ALLOCATED` (or stops cleanly on handshake failure); strictly fails closed with `CRITICAL_ORPHAN_GPU_SESSION` for any unverified or foreign session.
- **P0-2 Worker↔Controller Stage State Machine Contract & Static AST Gate (`h3_job_state.py`, `h3_remote_worker.py`, `tests/test_stage_contract_ast.py`)**:
  - Explicitly mapped `"init": "JOB_STARTED"`, `"dual_track_setup": "RESTORING_MODELS"`, and `"link_models": "RESTORING_MODELS"` in `WORKER_PHASE_TO_JOB_STAGE`, and aligned remote-upload state writes to `_job_stage_for(...)` (`"INFERENCE_COMPLETED"`).
  - Tightened `read_job_state()` to reject unknown stages outside `JOB_STAGES` with `ValueError`.
  - Added static Python `ast` CI gate (`tests/test_stage_contract_ast.py`) scanning every `set_stage`, `write_job_state`, `_write_job_state_safe`, and `_LAST_STAGE` assignment in `h3_remote_worker.py` against `JOB_STAGES` and `h3_controller.JOB_STAGE_TO_MANIFEST_STAGE`.
- **P1-3 ComfyUI `/prompt` Read-Timeout Three-State Reconciliation (`h3_remote_worker.py`)**:
  - Refactored `_find_prompt_by_client_id()` to return explicit three-state tuples: `("found", prompt_id)`, `("absent", None)`, or `("unknown", error_detail)`.
  - Updated `_submit_prompt()` with bounded exponential backoff polling during `"unknown"` states (e.g. during 40GB model loading stalls), strictly forbidding blind duplicate `POST /prompt` submissions while `"unknown"`.
- **P1-4 Tamper-Evident Completed Delivery & 0-CU Local Restore (`h3_controller.py`)**:
  - Persists `output_size`, `output_md5` (32-char lowercase hex), and `output_sha256` into `run_manifest.json["artifacts"]` (backward-compatible with legacy `remote_upload` / `local_sha256` fields).
  - Hardened `_verify_completed_delivery()` to always verify remote Drive `not trashed`, `size`, and normalized `md5Checksum` (failing closed as `"lost"` on any tamper). When `local_path` is absent, `--deliver` performs a 0-CU atomic stream download from Drive + digest check + `validate_final_video()` before reporting `local_video_validated: True` (and read-only probes with `restore_local=False` honestly report `local_video_validated: False`).
- **P1-5 Metric F Lease Cross-Check, Cold/Warm Stratification & Wilson 95% CI (`h3_reliability.py`, `scripts/reliability_report.py`)**:
  - Closed the Metric F orphan-GPU check gap by inspecting `pending_session_id` and cross-checking `model_cache/h3_gpu_leases.json` by `production_task_id` / `run_id` even when `manifest.session_id` is empty.
  - Added `wilson_95_ci: {"lower", "upper"}` to all rate metrics, side-by-side `summary_comparison` (Metric A vs Metric F with `73/73` zero-failure sample note), and `startup_mode_breakdown` (`coldstart_runs` vs `warm_reused_runs`) while keeping full top-level backward compatibility.
- **P1-6 Public Drive ID Desensitization, 0-CU Drive ACL Audit & `LICENSE` (`h3_contract.py`, `h3_task_spec.py`, `h3_controller.py`, `drive_depot.py`, `scripts/secret_scan.py`, `scripts/verify_drive.sh`, `LICENSE`)**:
  - Scrubbed all real Google Drive File/Folder IDs from tracked files in favor of synthetic `EXAMPLE_*` placeholders resolved locally via git-ignored `model_cache/local_drive_ids.json` (`0600`) or `H3_DRIVE_ID_*` env overrides, guarded by one-way SHA-256 and high-entropy Drive ID rules in `scripts/secret_scan.py`.
  - Added read-only 0-CU Drive ACL audit (`drive_depot.py audit-acl`, wired into `scripts/verify_drive.sh` and `preflight`) that fails closed on any `type == "anyone"` public link exposure and reports `ACL_PRIVATE_VERIFIED` when strictly private.
  - Added standard `Apache-2.0` root [`LICENSE`](LICENSE) (`Copyright 2026 SkyOneBranch`).

## v0.3.0 — 2026-11-19 — Self-Healing & Controlled Warm Pool (M1–M4)

Resilience, bootstrap round-trip reduction, host-disconnect self-healing, and
budget-bounded warm session reuse. All frozen budget (`30.0 CU` task /
`25.0 CU` session / `0.75 CU` cleanup reserve / `14400s` max wall) and
three-layer account-isolation invariants remain strictly enforced.

### Added / Changed

- **M1 GPU 503 Elastic Backoff & Error Classification (`h3_budget.py`, `h3_controller.py`, `scripts/run_h3_task.sh`)**:
  - Added `classify_allocation_error()` (`AUTH_FATAL`, `QUOTA_EXCEEDED`,
    `CAPACITY_TRANSIENT`, `TRANSPORT_TRANSIENT`, `UNKNOWN_FATAL`) and
    `compute_backoff_delay()` (`15s / 45s / 90s ±20%` jitter).
  - Added `--allocation-rounds <1..5>` (default `3` for `deliver` /
    `--auto-recover`, `1` for raw `run`) wrapping `("G4", "G4", "A100_HIGH_MEM")`
    with deadline-margin gating, fail-fast on fatal/quota errors, and
    `GPU_ALLOCATION_BACKOFF` audit events.
- **M2 Two-Stage Bundled Bootstrap (`h3_bootstrap_stage1.py`, `h3_bootstrap_stage2.py`, `h3_controller.py`, `scripts/run_h3_task.sh`)**:
  - Added `--bootstrap-mode bundled|legacy` (default `bundled`).
  - Stage 1 bundles `0600` downscoped tokens and Drive probe scripts (never
    `h3_remote_worker.py`), extracts with strict path-traversal rejection, and
    must emit `MUSE_DRIVE_DOWNLOAD_PROBE_OK` before Stage 2 (`h3_remote_worker.py`
    + `task.json`) is built or uploaded.
  - Persists `bootstrap` telemetry (`mode`, `bootstrap_wall_sec`,
    `cli_round_trips`) in run manifests.
- **M3 Host Disconnect Self-Healing & Drive 0-GPU Recovery (`h3_controller.py`, `h3_drive_token.py`)**:
  - Added `_call_with_transient_retry()` (`5x`: `5/10/20/30s`) across host
    polling, inspection, delivery verification, and resume paths.
  - Added `ensure_token_freshness(min_remaining_sec=900)` in `h3_drive_token.py`
    and wired `token_freshness` into `0-CU` preflight.
  - Added `recover_from_drive_archive()` so `--deliver` / `--auto-recover`
    recovers archived MP4s from `Muse-H3-Video-Outputs` at **0 CU** when the
    host disconnects after the remote GPU has already stopped.
- **M4 Controlled Warm Session Pool (`h3_gpu_lease_registry.py`, `h3_shutdown_executor.py`, `scripts/run_shutdown_guardian.py`, `h3_run_manifest.py`, `h3_reliability.py`, `h3_remote_worker.py`, `h3_warm_health_probe.py`, `h3_controller.py`, `scripts/run_h3_task.sh`, `scripts/verify_shutdown.sh`, `.agents/pre_tool_guard.py`)**:
  - Added `--keep-warm-ttl <sec>` (`0` or `60..900`; default `0`): keeps healthy
    ComfyUI `:8188` sessions warm within `MAX_SESSION_SECONDS` (`14400s`) and
    `PER_SESSION_BUDGET_CU` (`25.0 CU`) caps, enabling `< 5s` warm reuse
    (`warm_session_reused: true`) after `MUSE_WARM_HEALTH_OK` verification.
  - Extended `LeaseRegistry` with `mark_warm_idle()`, `claim_warm_lease()`, and
    `find_warm_leases()`; extended `RunManifest` with `WARM_POOLED` +
    `record_warm_session_stopped()` and `ReliabilityReport` with
    `warm_session_hits` / `warm_pooled_active_runs`.
  - Updated `h3_shutdown_executor.py` (`--stop-warm`),
    `scripts/run_shutdown_guardian.py` (unconditional cross-task reclamation of
    expired warm leases), `scripts/verify_shutdown.sh` (read-only by default;
    `--stop-warm` reclaims warm leases then asserts `sessions == []` and
    `rate == 0.0`), and `.agents/pre_tool_guard.py` (permits
    `./scripts/verify_shutdown.sh --stop-warm` and
    `scripts/verify_shutdown.sh --stop-warm` while keeping all credential and
    unmanaged-`colab` blocks enforced).

### Safety Gates & Offline CI Verification (0.00 CU)

- All v0.3.0 modules, wrappers, and guards are covered by offline unit tests
  (`tests/test_v030_m1_backoff.py`, `tests/test_v030_m2_bundled_bootstrap.py`,
  `tests/test_v030_m3_self_healing.py`, `tests/test_v030_m4_controller_worker.py`,
  `tests/test_v030_m4_warm_infra.py`, `tests/test_v030_scripts_guard.py`),
  `scripts/secret_scan.py`, `scripts/check_docs_index.py`, and
  `tests/run_ci_tests.py` with **0.00 CU** spent.

## v0.2.1 — 2026-11-19 — Zero-CU Polish (P1–P3)

Targeted post-benchmark polish based on the `v0.2.0` live A100 run (`489s ->
230.9s` coldstart, `4.44 CU -> 3.70 CU`). All changes verified at **0.00 CU**.

### Fixed / Improved

- **P1 Drive Upload MD5 Hex Alignment (`h3_remote_worker.py`, `h3_drive_upload.py`)**:
  - Replaced `_md5_b64` with `_md5_hex` (32-char lowercase hex) in
    `validate_assembled_mp4()` to match Google Drive API v3 `md5Checksum`
    directly.
  - Added defensive `_normalize_md5_hex()` normalization in
    `DriveUploader.verify_uploaded()` (accepts 32-char hex case-insensitively
    or 24-char Base64 16-byte digests decoded to hex; fails closed on mismatch
    or missing server `md5Checksum`).
- **P2 ComfyUI Headless Optional Exclusions & `py313` Wheelhouse (`h3_comfyui_cache.py`, `h3_remote_worker.py`)**:
  - Source-audited pinned ComfyUI commit
    `61e524bdba7a4e193b45c2697f1af83c1beb86d6` (`server.py` +
    `app/frontend_management.py`): excluded 100%-optional headless UI asset
    packages (`comfyui-workflow-templates`, `comfyui-workflow-templates-*`,
    `comfyui-embedded-docs`) while retaining `comfyui-frontend-package` (which
    calls `sys.exit(-1)` on `ImportError`).
  - Added `manylinux_2_28_x86_64` to `DEFAULT_PLATFORMS` (required by
    `av>=17.0.0`, supported by Colab glibc 2.35), enhanced
    `h3_comfyui_cache.py verify` for remote-only synced Drive entries, and
    built/published a `py313` headless wheelhouse (`243.6 MB`, `72` wheels —
    69.4% smaller than `py312`'s `834.7 MB`) alongside `py312`.
- **P3 Monotonic `MODELS_READY` Backfill & SLA Calibration (`h3_controller.py`, `h3_reliability.py`)**:
  - Monotonically backfills `MODELS_READY` in `h3_controller.py` when remote
    polling observes `INFERENCE_RUNNING`/`INFERENCE_COMPLETED`/`VIDEO_READY` or
    `coldstart_metrics` before a separate `MODELS_READY` poll arrived.
  - Calibrated `h3_reliability.py::_evaluate_run_f()` so condition (2) is
    satisfied by `MODELS_READY`, audit `COLDSTART_METRICS`, or monotonic
    `WORKER_STARTED -> INFERENCE_COMPLETED -> VIDEO_VALIDATED` completion while
    keeping incomplete runs failing closed.

## v0.2.0 — 2026-11-19 — Fast Cold-Start Engine (F1–F5)

End-to-end cold-start acceleration across the Drive downloader, ComfyUI
dependency cache, remote worker setup pipeline, in-cloud depot seeder, and
cold-start telemetry reporting. All existing `TaskSpec`, budget (`30.0 CU`
task / `25.0 CU` session), and three-layer account-isolation invariants remain
unchanged and 100% backwards-compatible.

### Added / Changed

- **F1 Zero-Copy High-Concurrency Drive Download (`h3_drive_download.py`)**:
  - Replaced `.part.shard0..N` temp files and single-threaded merge with a
    preallocated single `.part` file (`os.truncate` + lock-free multi-threaded
    `os.pwrite`, peak disk `<= 1.0x`).
  - Raised `DEFAULT_CONCURRENT_WORKERS` from `4` to `12` (configurable via
    `H3_DRIVE_CONCURRENT_WORKERS` in `1..16`) and added a process-wide Drive
    connection semaphore (`DEFAULT_MAX_GLOBAL_CONNECTIONS = 16`, configurable
    via `H3_DRIVE_MAX_GLOBAL_CONNECTIONS`).
  - Added ordered in-flight streaming SHA-256 (`_StreamingBlockHasher`) with
    bounded memory buffering and selective `os.pread` spillback/resume, atomic
    `.part.bitmap.json` block-level resume, `abort_event` /
    `DriveDownloadAbortedError` cancellation, and single-flight
    `FileTokenProvider.refresh(failed_token_id=None)`.
- **F2 Offline ComfyUI Wheelhouse Cache (`h3_comfyui_cache.py`, `scripts/setup_isolated_account.py`, `h3_controller.py`)**:
  - Made Drive SDK imports lazy in `h3_comfyui_cache.py` so pure-stdlib
    wheelhouse helpers (`normalize_pkg_name`, `normalize_python_tag`,
    `filter_requirements_lines`, `filter_requirements_file`,
    `validate_wheelhouse_entry`, `select_wheelhouse_entry`,
    `build_wheelhouse_task_descriptor`, `verify_local_wheelhouse`) work without
    Drive dependencies.
  - Added `build-wheelhouse` and `publish-wheelhouse` (gated by
    `--allow-drive-write`) packaging deterministic
    `comfyui-wheelhouse-<commit>-<pytag>.tar.gz` (`wheelhouse/*.whl` +
    `wheelhouse/WHEELHOUSE_META.json`) for Colab Linux x86_64 (`py311`/`py312`)
    while excluding preinstalled `torch`, `torchvision`, `torchaudio`,
    `triton`, and `nvidia-*` packages.
  - Updated `scripts/setup_isolated_account.py` (`--sync-shared-caches`) to
    discover and verify commit-matching wheelhouse bundles in
    `Muse-H3-ComfyUI-Source` alongside the source tarball, and wired optional
    `drive_download["wheelhouses"]` through `h3_controller.py`.
- **F3 Dual-Track Parallel Setup (`h3_remote_worker.py`)**:
  - Runs `ensure_comfy()` and `restore_weights()` concurrently via
    `ThreadPoolExecutor(max_workers=2)` with a shared `abort_event`,
    thread-safe token refresh, thread-local command failure tracking, and
    accurate failing-stage attribution in `WORKER_FAILED_JSON`.
  - Prefers offline wheelhouse install (`pip install --no-index --find-links`)
    and emits structured `WHEELHOUSE_FALLBACK_PYPI` when falling back to PyPI.
- **F4 Independent NVMe Model Store & High-VRAM Flags (`h3_remote_worker.py`)**:
  - Stores downloaded weights under `/content/nvme_models/h3` (overridable via
    `H3_NVME_MODELS_DIR`) and atomically symlinks verified weights into
    `/content/ComfyUI/models/` via `link_models_into_comfy()` with
    `require_inside` and readability checks.
  - Auto-detects GPU VRAM (`detect_gpu_vram_mib`) and appends
    `--highvram --disable-mmap` for VRAM `>= 76800` MiB (overridable via
    `H3_COMFY_VRAM_FLAGS`).
- **F5 In-Cloud Remote Seeding & Cold-Start Telemetry (`drive_depot.py`, `h3_controller.py`, `h3_run_manifest.py`, `scripts/reliability_report.py`)**:
  - Added `drive_depot.py remote-seed` gated by `--allow-drive-write`,
    `--allow-large-transfer`, and `--confirm-remote-seed`: launches a
    watchdog-protected temporary Colab session with a short-lived `drive.file`
    token, downloads locked `config/pinned_hf_files.json` weights concurrently
    in-cloud (`aria2c` or multi-threaded Range), verifies SHA-256 remotely,
    uploads to Drive Depot, and enforces `colab stop` + tristate `STOPPED`
    verification in `finally`.
  - Worker emits `COLDSTART_METRICS_JSON` (also persisted in
    `MUSE_H3_JOB_STATE.json`); `h3_controller.py` persists it into
    `model_cache/run_manifests/<run_id>.json` via `h3_run_manifest.py`, and
    `scripts/reliability_report.py` aggregates cold-start P50/P90 latencies and
    `wheelhouse_hit_rate`.

### New Offline Unit Test Suites (0 CU)

- `tests/test_drive_download_v2.py` (F1 zero-copy `pwrite`, streaming SHA-256,
  bitmap resume, global cap, `abort_event`, single-flight token refresh)
- `tests/test_comfyui_wheelhouse.py` (F2 wheelhouse build/verify/publish,
  requirements filtering, cross-account sync, lazy Drive imports)
- `tests/test_remote_worker_v2.py` (F1–F5 worker dual-track, wheelhouse
  offline/fallback, NVMe symlinks, VRAM flags, `COLDSTART_METRICS_JSON`)
- `tests/test_controller_wheelhouse_telemetry.py` (F2/F5 controller wiring,
  log parsing, run manifest persistence)
- `tests/test_drive_depot_remote_seed.py` (F5 triple-flag gate, watchdog &
  `finally` shutdown guarantees, remote Range/SHA-256 helpers)
- `tests/test_reliability_coldstart.py` (F5 P50/P90 & `wheelhouse_hit_rate`
  aggregation)

## v0.1.0 — 2026-11-12 — initial migration (TASK-001, local repo)

Initial private migration of the verified Muse Colab H3 production
baseline from `~/workspace/Muse_Colab_H3_Drive_v3/`. No production logic
was rewritten; all production files are verbatim copies (SHA-256
recorded in `SOURCE_MANIFEST.json`).

### Migrated

- 21 flat-root production modules + `workflow_api.json` +
  `requirements-local.txt` (verbatim).
- `reference_v2/colab_manager.py` **and** `reference_v2/generate_video.py`
  (verbatim). Correction found by the test run: an early grep that
  excluded `reference_v2/` lines wrongly concluded `generate_video.py`
  was unimported; full discovery failed with
  `ModuleNotFoundError: No module named 'generate_video'`, proving the
  transitive import (`colab_manager.py:39` imports 10 names from it).
  The file was then migrated verbatim and the suite re-run.
  `reference_v2/config.yaml` stays excluded: no production code path
  loads it (the loader takes an explicit path and is never called by the
  H3 controller or tests), and it holds a hardcoded Mac user path plus
  SVD settings unrelated to this job.
- 8 migrated test suites (173 tests) + live-only
  `tests/live_drive_resume_check.py` (naturally excluded from discovery).
- Project docs verbatim: v3 README/progress, deployment status, both
  MUSE prompt docs, storyboard references, Drive remote-download design
  doc, P0 archive prompt docs, pinned HF files, ComfyUI source manifest,
  depot config example, UI-format source workflow.
- Real CLI patch diffs (`patches/google-colab-cli-0.7.5/`), generated
  against the pristine 0.7.5 wheel and verified with `git apply --check`
  plus a byte-identical round-trip.

### Added in this migration

- Configs mirroring code constants (`config/*.yaml`, `*.example.json`),
  asserted by new `tests/test_config_contract.py` (4 tests).
- New regression tests `tests/test_savevideo_history.py` (8 tests) for
  `find_rendered_video` (PreviewVideo `{images, animated}` list/bool,
  legacy `{videos}`, bare `{animated}`, images-without-animated,
  suffix/size gates, missing file, documented no-ffprobe gap).
- `jobs/minimax_h3_ab/` (job.yaml, `video_contract.py` facade — imports
  verified — README with the MiniMax_H3_Clean non-presence note).
- `scripts/` wrappers (all verified against real controller/CLI
  behaviour; `run_h3_ab.sh` refuses without `--confirm-live`) plus
  `verify_manifest.py`, `verify_patch.py`, `check_workflow_schema.py`,
  `check_docs_index.py`.
- `docs/`: architecture, 19-stage execution flow (with one UNVERIFIED
  stage — see below), module inventory, code provenance, 4 runbooks,
  24 incident records, 5 ADRs, 9 issue drafts, audits/research notes.
- Offline-only CI (`.github/workflows/ci.yml`): py_compile, test suite,
  workflow schema check, patch verification, secret scan, docs-index
  check. No GPU, no Drive, no credentials.
- `requirements.lock` (72 pins from the verified source venv,
  Python 3.12.3) and `pyproject.toml`.

### Test results (2026-11-12, source venv python, new repo root)

- Source baseline: `python -m unittest discover -s tests -q` in the
  source project → **173 tests, OK**.
- New repo full discovery: **185 tests ran, 183 passed, 2 errors**.
  The 2 errors are the asset-gated poster tests, failing verbatim with
  `ValueError: Original poster missing or SHA256 mismatch:
  <repo>/assets/original_poster.jpg`:
  - `test_contract.ContractTest.test_poster_hash`
  - `test_h3_production.ProductionTest.test_poster_sha256_required`
  Both pass in the source project only because the poster binary is
  present there; the binary is intentionally excluded from this repo.
  This is recorded, not hidden.
- CI entry point `python tests/run_ci_tests.py`: **183 ran, 0 failures,
  0 errors, 2 asset-gated skips** (printed).
- `python h3_controller.py plan` verified (prints `no_gpu_created: true`,
  project budget 30.0, per-session 25.0). `--help` verified.
- `scripts/verify_patch.py`: OK. `scripts/check_workflow_schema.py`: OK.
  `python -m py_compile` over all repo Python files: OK.
  `jobs/minimax_h3_ab/video_contract.py` import smoke test: OK
  (736/1312/24, 226/274 frames, seeds 42/43, Turbo off).

### Secret scan (2026-11-12, new repo)

- Pattern scan (Google/GitHub token shapes, private keys,
  refresh_token/client_secret with values): **clean** after tuning —
  the only hits were (a) the redaction regexes in
  `reference_v2/colab_manager.py` (the scrubber itself, allow-listed in
  CI), (b) the CI workflow's own search-pattern literals (excluded by
  filename, documented in the workflow), and (c) a loose JWT-shape
  pattern false-positiving on a dotted model filename in
  `config/pinned_hf_files.json` (pattern removed from CI, documented).
- Broad word scan (`token|secret|password|api_key`, file:line only):
  all hits inspected — code words only (`name_tokens`, `token_hex`
  run-id suffix, `psnr_avg` token parsing, `atomic_secret_json`,
  comments stating tokens are never logged). No credential values found.
- No model weights, poster binary, or MP4 in the repo.

### Unverified / PLANNED (not claimed as done)

- **DRIVE_OUTPUT_VERIFIED stage is UNVERIFIED**: `archive_variant_to_drive`
  is defined but has no call site in `run_variant` (verified by grep).
  Drive delivery is not verified by the current live flow.
- **Variant B pending re-run** (run `h3-20261112T093207Z-7ab515`
  interrupted by VM restart; orphan A100 stopped; no MP4).
- `workstation/` package layout is PLANNED only (no implementation).
- `prompts_ab.py` / `workflow_builder.py` (MiniMax_H3_Clean) are NOT in
  this repo — the files were never obtained (P2 evaluation issue).
- Drivemount 600s fix: mock-verified; live mounts so far stayed inside
  the old 120s authorization window (incident 07).
- `find_rendered_video` has no ffprobe gate and no RUN_ID-prefix
  fallback (P0 issue; gap documented by a passing test).
- VM-restart orphan recovery is manual (incident 12).

### Safety

- No Colab GPU created, no CU spent, no Drive reads/writes, no `colab`
  commands run by this migration. No remote set, nothing pushed.
- GitHub repo creation is NOT part of this local build (parent agent
  verifies identity/permissions and creates the private repo).
