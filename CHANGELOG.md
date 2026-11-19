# CHANGELOG

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
