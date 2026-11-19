# Antigravity-Colab-Workstation

An **Antigravity IDE + macOS-hardened Google Colab GPU workstation** for **MiniMax-H3 / ComfyUI** video generation with 3-layer Google account isolation, hard Compute Unit (CU) budget enforcement, autonomous remote worker telemetry, Google Drive delivery verification, and independent shutdown guardians.

---

## Verified Live Benchmark (2026-11-15)

Live end-to-end production run on `A100-SXM4-80GB` via `--task-spec config/tasks/shao_nian_v1_A.json`:

| Field | Verified Result |
| :--- | :--- |
| **Task ID / Run ID** | `shao-nian-agy-001` / `h3-20261115T064923Z-5675ec` |
| **State / QA** | `DELIVERY_COMPLETED` · `technical_pass: true` |
| **Video Spec** | `1080x1920`, `24.0 fps`, `274` frames (`11.4167 s`), video + audio streams, `7,334,314` bytes |
| **Drive Archival** | Uploaded & server-verified (`drive_upload_verified: true`) under `Muse-H3-Video-Outputs` |
| **Actual CU Cost** | **`4.44 CU`** (`200.00 CU -> 195.56 CU`, under `30.0 CU` task / `25.0 CU` session caps) |
| **Post-Run Shutdown** | `GPU_STOP_VERIFIED: no active sessions.` (`rate: 0.0`, `sessions: []`) |

---

## Key Capabilities Added for Antigravity & macOS

- **Three-Layer Credential Isolation ([`scripts/_colab_env.sh`](scripts/_colab_env.sh), [`scripts/setup_isolated_account.py`](scripts/setup_isolated_account.py))**:
  - Dedicated Drive data plane (`.secrets/colab_drive_token.json` + `.gcloud-colab`) and dedicated Colab compute plane (`~/.config/colab-cli/token.json`) verified to match (`same_google_account == True`) without ever reading or modifying your host global `~/.config/gcloud/application_default_credentials.json`.
  - Supports `--browser-login` local loopback OAuth and `--sync-shared-caches` for shared Drive model/ComfyUI depots.
- **Antigravity Native Guardrails (`.agents/`)**:
  - Skill workflow in [`.agents/skills/colab-h3-video/SKILL.md`](.agents/skills/colab-h3-video/SKILL.md).
  - Deterministic `PreToolUse` guard in [`.agents/hooks.json`](.agents/hooks.json) + [`.agents/pre_tool_guard.py`](.agents/pre_tool_guard.py) that blocks secret inspection and unguarded `colab` commands while permitting audited wrapper scripts.
- **Cross-Project Callable Entrypoints**:
  - [`scripts/preflight.sh`](scripts/preflight.sh), [`scripts/run_h3_task.sh`](scripts/run_h3_task.sh), and [`scripts/check_outputs.sh`](scripts/check_outputs.sh) resolve relative `--task-spec` and MP4 arguments against the caller's working directory (`CALLER_PWD`), and `check_outputs.sh` supports `--task-spec` geometry validation.
- **Portable macOS Sleep & Shutdown Hardening**:
  - Live runs use `/usr/bin/caffeinate -dimsu ./scripts/run_h3_task.sh ...` to survive display sleep, backed by a 300-second `launchd` shutdown guardian via [`scripts/install_macos_guardian.sh`](scripts/install_macos_guardian.sh).

---

## Fast Cold-Start Engine (v0.2.0)

- **F1 Zero-Copy High-Concurrency Drive Download ([`h3_drive_download.py`](h3_drive_download.py))**:
  - Preallocates a single `.part` file (`os.truncate` + lock-free multi-threaded `os.pwrite`, peak disk `<= 1.0x`, zero `.shard*` temp files or merge pass) with `12` default concurrent workers (`H3_DRIVE_CONCURRENT_WORKERS` in `1..16`) and a process-wide Drive connection cap (`H3_DRIVE_MAX_GLOBAL_CONNECTIONS`, default `16`).
  - Computes ordered in-flight streaming SHA-256 without re-reading completed files from disk, persists `.part.bitmap.json` for block-level resume, supports `abort_event` cancellation (`DriveDownloadAbortedError`), and provides single-flight `FileTokenProvider.refresh` under concurrent 401s.
- **F2 Offline ComfyUI Wheelhouse Cache ([`h3_comfyui_cache.py`](h3_comfyui_cache.py), [`scripts/setup_isolated_account.py`](scripts/setup_isolated_account.py))**:
  - `build-wheelhouse` and `publish-wheelhouse` package deterministic SHA-256-locked `comfyui-wheelhouse-<commit>-<pytag>.tar.gz` bundles targeting Colab Linux x86_64 (`py311`/`py312`) while excluding preinstalled heavy packages (`torch`, `torchvision`, `torchaudio`, `triton`, `nvidia-*`).
  - `scripts/setup_isolated_account.py --sync-shared-caches` discovers and verifies commit-matching wheelhouse bundles across accounts alongside the source archive. The remote worker prefers offline `pip install --no-index --find-links=...` and emits `WHEELHOUSE_FALLBACK_PYPI` when falling back to PyPI.
- **F3 Dual-Track Parallel Setup ([`h3_remote_worker.py`](h3_remote_worker.py))**:
  - Runs `ensure_comfy()` and `restore_weights()` concurrently via `ThreadPoolExecutor(max_workers=2)` with a shared `abort_event`, thread-safe token refresh, and thread-safe state/failure recording so an error in either track cancels the other promptly and reports the exact failing stage.
- **F4 Independent NVMe Model Store & High-VRAM Launch ([`h3_remote_worker.py`](h3_remote_worker.py))**:
  - Downloads weights into `/content/nvme_models/h3` (overridable via `H3_NVME_MODELS_DIR`) and atomically symlinks verified weights into `/content/ComfyUI/models/` (`link_models_into_comfy`) after both tracks join.
  - Detects GPU VRAM and appends `--highvram --disable-mmap` automatically when VRAM `>= 76800` MiB (A100 80GB / G4 96GB), overridable via `H3_COMFY_VRAM_FLAGS`.
- **F5 In-Cloud Remote Seeding & Cold-Start Telemetry ([`drive_depot.py`](drive_depot.py), [`h3_controller.py`](h3_controller.py), [`h3_run_manifest.py`](h3_run_manifest.py), [`scripts/reliability_report.py`](scripts/reliability_report.py))**:
  - `drive_depot.py remote-seed` (strictly gated by `--allow-drive-write --allow-large-transfer --confirm-remote-seed`) downloads locked Hugging Face weights in-cloud using a short-lived `drive.file` token under watchdog supervision, verifies SHA-256 remotely, and unconditionally enforces `colab stop` + tristate `STOPPED` verification in `finally`.
  - Worker emits structured `COLDSTART_METRICS_JSON` (persisted in `model_cache/run_manifests/<run_id>.json`), summarized with cold-start P50/P90 phase latencies and `wheelhouse_hit_rate` by `scripts/reliability_report.py`.

---

## Quickstart

### 1. Prerequisites & Environment

- Python `>= 3.11`, `ffmpeg`, `ffprobe`, and Node/macOS or Linux shell.
- Install pinned `google-colab-cli==0.7.5`, control-plane libraries (`google-api-python-client`, `google-auth`, `google-auth-oauthlib`, `requests`, `certifi`, `pyyaml==6.0.1`), and both required production patches (`patches/google-colab-cli-0.7.5/`) into `.venv-colab-cli`:

```bash
./scripts/install_colab_cli.sh
```

### 2. Bind Isolated Google Account (`0 CU`)

Place a Desktop OAuth client JSON (Drive API enabled) at `.secrets/client_secret.json`, then run:

```bash
.venv-colab-cli/bin/python scripts/setup_isolated_account.py --browser-login
.venv-colab-cli/bin/python scripts/setup_isolated_account.py --check-only
./scripts/preflight.sh
```

Optional on macOS: install the background `launchd` shutdown guardian:

```bash
./scripts/install_macos_guardian.sh --install
./scripts/install_macos_guardian.sh --status
```

### 3. Run a Task (`TaskSpec`)

```bash
./scripts/preflight.sh --task-spec config/tasks/shao_nian_v1_A.json

# macOS: keep host awake during remote execution + Drive archival
/usr/bin/caffeinate -dimsu ./scripts/run_h3_task.sh \
  --confirm-live \
  --task-spec config/tasks/shao_nian_v1_A.json \
  --production-task-id shao-nian-agy-001 \
  --authorize-new-task \
  --deliver \
  --auto-recover

./scripts/check_outputs.sh output/<run_id>/shao_nian_v1.mp4 \
  --task-spec config/tasks/shao_nian_v1_A.json

./scripts/verify_shutdown.sh
./scripts/check_budget.sh
```

To invoke this workstation from **another project directory or Antigravity workspace**, see [`docs/runbooks/antigravity-cross-project-integration.md`](docs/runbooks/antigravity-cross-project-integration.md).

---

## Pinned Architecture & Contracts

- **Compute & Storage**: Colab Enterprise/Pro A100-SXM4-80GB (`config/gpu_policy.yaml`), ~37 GB locked MiniMax-H3 safetensors from Drive Model Depot (`drive_depot.py`), reproducible ComfyUI install via pinned source + offline wheelhouse (`h3_comfyui_cache.py`).
- **Hard Budget Caps (`config/budget_policy.yaml`)**: `30.0 CU` per production task (`MAX_PRODUCTION_TASK_CU = 30.0`) and `25.0 CU` per single runtime session (`SESSION_STOP_CU = 25.0`).
- **CLI Patches (`patches/google-colab-cli-0.7.5/`)**:
  - `proxy_post_body.patch` — preserves empty POST bodies across HTTP 411/400 Colab assign retry paths.
  - `drivemount_timeout.patch` — honors longer timeout budgets during Drive mount initialization.
- **Repository Invariants**:
  - Flat `.py` layout at repo root (ADR-003, [`docs/decisions/ADR-003.md`](docs/decisions/ADR-003.md)).
  - Secrets, OAuth tokens, `.venv*`, runtime caches (`model_cache/` except `verified_manifest.example.json`), and video outputs are gitignored and enforced by `scripts/secret_scan.py` and `gitleaks`.

---

## Offline Verification & CI

All checks below spend **0 CU** and match [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

```bash
.venv-colab-cli/bin/python -m py_compile *.py scripts/*.py tests/*.py
.venv-colab-cli/bin/python tests/run_ci_tests.py
.venv-colab-cli/bin/python scripts/check_workflow_schema.py
.venv-colab-cli/bin/python scripts/verify_patch.py
.venv-colab-cli/bin/python scripts/verify_manifest.py
.venv-colab-cli/bin/python scripts/secret_scan.py
.venv-colab-cli/bin/python scripts/check_docs_index.py
```

---

## Key Documentation

- [`docs/runbooks/antigravity-cross-project-integration.md`](docs/runbooks/antigravity-cross-project-integration.md) — cross-project & multi-workspace calling guide
- [`docs/runbooks/generic-task-spec.md`](docs/runbooks/generic-task-spec.md) — `TaskSpec` JSON specification & `shots[]` timeline rules
- [`docs/runbooks/production-run.md`](docs/runbooks/production-run.md) — production run procedures
- [`docs/runbooks/production-acceptance.md`](docs/runbooks/production-acceptance.md) — output QA & acceptance criteria
- [`docs/runbooks/shutdown-verification.md`](docs/runbooks/shutdown-verification.md) — post-run shutdown proof
- [`docs/architecture.md`](docs/architecture.md) · [`docs/execution_flow.md`](docs/execution_flow.md) · [`docs/module_inventory.md`](docs/module_inventory.md)
