# Antigravity Cross-Project & Multi-Workspace Integration Runbook

`Antigravity-Colab-Workstation` is designed to operate as an **isolated, reusable GPU compute & video-rendering workstation** that any other local repository or Antigravity IDE workspace can invoke without copying secrets, colliding with your main Google ADC credentials, or duplicating `google-colab-cli` patches.

---

## 1. Architecture Overview

When another application (web backend, content pipeline, or standalone Antigravity agent workspace) needs MiniMax-H3 video rendering on Colab GPUs, it delegates to this workstation through three clean boundaries:

1. **Wrapper Entrypoints (`scripts/*.sh`)**:
   - `scripts/preflight.sh`, `scripts/run_h3_task.sh`, and `scripts/check_outputs.sh` capture `CALLER_PWD` before changing directories into the workstation root, so external callers can pass relative `--task-spec` and output MP4 paths from their own repository directory.
2. **3-Layer Account Isolation (`scripts/_colab_env.sh`)**:
   - Prevents cross-account contamination on machines where your host default `gcloud` ADC belongs to another Google account.
3. **Antigravity Skill + Pre-Tool Safety Hook (`.agents/`)**:
   - `.agents/skills/colab-h3-video/SKILL.md` defines the mandatory preflight -> live execution -> QA -> shutdown verification sequence.
   - `.agents/hooks.json` + `.agents/pre_tool_guard.py` auto-approve vetted wrapper commands while blocking direct inspection of `.secrets/` and unguarded `colab` CLI calls that could leak active GPU sessions.

---

## 2. Three-Layer Google Account Isolation Contract

| Layer | Purpose | Storage Location | Isolation Guarantee |
| :--- | :--- | :--- | :--- |
| **Layer 1: Host Global ADC** | Your existing general projects & daily account | `~/.config/gcloud/application_default_credentials.json` | **Never read, overwritten, or revoked.** `H3_REQUIRE_ISOLATED_ADC=1` fails closed if `GOOGLE_APPLICATION_CREDENTIALS` points here. |
| **Layer 2: Workstation Data Plane** | Dedicated Drive API access for assets, model depot, & MP4 archival | `<workstation>/.secrets/client_secret.json` + `<workstation>/.secrets/colab_drive_token.json` (`0600`) | Loaded exclusively via `CLOUDSDK_CONFIG=<workstation>/.gcloud-colab` and explicit `GOOGLE_APPLICATION_CREDENTIALS`. Downscoped (`drive.readonly` + `drive.file`) before shipping to Colab workers. |
| **Layer 3: Workstation Compute Plane** | Dedicated Colab runtime lifecycle & CU billing | `~/.config/colab-cli/token.json` (`0600`) | Accessed with `COLAB_AUTH_MODE=oauth2`. Setup & preflight verify `same_google_account == True` against Layer 2 at `0 CU`. |

`scripts/_colab_env.sh` sets these environment variables whenever any wrapper script runs, and `scripts/run_shutdown_guardian.py` sets the same defaults inside background launchd runs.

---

## 3. New Machine Bootstrap (One-Time Setup)

Run once inside `<workstation>` (`Antigravity-Colab-Workstation`):

```bash
git clone https://github.com/SkyOneBranch/Antigravity-Colab-Workstation.git
cd Antigravity-Colab-Workstation

# 1. Install pinned google-colab-cli 0.7.5 + production patches into .venv-colab-cli
./scripts/install_colab_cli.sh
```

Place your Desktop OAuth client JSON (from a GCP project with Google Drive API enabled) at:

```bash
.secrets/client_secret.json
```

Bind both the Drive data plane and Colab compute plane cleanly using local loopback OAuth (`0 CU`):

```bash
.venv-colab-cli/bin/python scripts/setup_isolated_account.py --browser-login
```

If your Google Drive already hosts shared `Muse-H3-Model-Depot` or `Muse-H3-ComfyUI-Source` folders (instead of owning them in `My Drive`), sync `model_cache/` manifests explicitly at `0 CU`:

```bash
.venv-colab-cli/bin/python scripts/setup_isolated_account.py \
  --sync-shared-caches \
  --depot-folder-id <MODEL_DEPOT_FOLDER_ID> \
  --comfyui-folder-id <COMFYUI_SOURCE_FOLDER_ID>
```

Verify isolated credentials and full offline preflight (`0 CU`):

```bash
.venv-colab-cli/bin/python scripts/setup_isolated_account.py --check-only
./scripts/preflight.sh
```

Optional on macOS hosts: install the 300-second background `launchd` shutdown guardian:

```bash
./scripts/install_macos_guardian.sh --install
./scripts/install_macos_guardian.sh --status
```

---

## 4. Calling From Another Local Project (CLI / Scripts)

Suppose your caller project lives at `~/projects/my-video-app` and the workstation lives at `~/Agent/Antigravity-Colab-Workstation`:

```bash
COLAB_WS="$HOME/Agent/Antigravity-Colab-Workstation"
SPEC="specs/episode_01.json"
TASK_ID="episode-01-prod"
```

### Step 1: 0-CU Preflight With External Spec

```bash
"$COLAB_WS/scripts/preflight.sh" --task-spec "$SPEC"
```

`preflight.sh` resolves relative `$SPEC` against `~/projects/my-video-app`, validates `TaskSpec`, verifies credential isolation and account match, checks Drive depot visibility, and checks CU budget without allocating any GPU.

### Step 2: Live Render + Delivery + Auto-Recovery (v0.3.0)

On macOS, always wrap live execution with `/usr/bin/caffeinate -dimsu` so display/idle sleep never interrupts heartbeat telemetry or Drive archival. Use `--deliver` (or `run --auto-recover` — pass at most one, they are mutually exclusive):

```bash
/usr/bin/caffeinate -dimsu "$COLAB_WS/scripts/run_h3_task.sh" \
  --confirm-live \
  --task-spec "$SPEC" \
  --production-task-id "$TASK_ID" \
  --authorize-new-task \
  --deliver \
  --allocation-rounds 3 \
  --bootstrap-mode bundled
```

On Linux, omit `/usr/bin/caffeinate -dimsu`.

**v0.3.0 Resilience & Batch Warm-Pool Flags**:
- **`--allocation-rounds <1..5>`**: controls outer elastic backoff rounds (`15s / 45s / 90s ±20%`) around `("G4", "G4", "A100_HIGH_MEM")` on transient HTTP 503/412 capacity errors (default `3` for `--deliver` / `--auto-recover`, `1` for raw `run`).
- **`--bootstrap-mode <bundled|legacy>`**: defaults to `bundled` (2-stage tar upload/exec with path-traversal protection); pass `legacy` only for diagnostic fallback.
- **`--keep-warm-ttl <60..900>`**: optional controlled warm pool for back-to-back cross-project renders. Keeps a healthy ComfyUI `:8188` runtime alive for `<sec>` after delivery (bounded by `25.0 CU` per session and `14400s` max wall time) so the next cross-project render reuses it in `< 5s`. Any error forces an immediate fail-closed stop.
- **Idempotent Drive 0-GPU Recovery**: if the caller host disconnects or sleeps while the remote worker finishes and uploads to `Muse-H3-Video-Outputs`, re-running the exact same `--deliver` (or `--auto-recover`) command detects the stopped GPU and executes `recover_from_drive_archive()` at **0 CU** to download, validate, and mark the run `TASK_COMPLETED` without allocating a new GPU.

### Step 3: Output QA & Delivery Inspection

Find the delivered file under `$COLAB_WS/output/<run_id>/<output_prefix>.mp4` (and run ledger/receipts under `$COLAB_WS/runs/<run_id>/`), then verify exact `TaskSpec` frame count, fps, resolution, and audio stream:

```bash
"$COLAB_WS/scripts/check_outputs.sh" \
  "$COLAB_WS/output/<run_id>/<output_prefix>.mp4" \
  --task-spec "$SPEC"
```

### Step 4: Mandatory Shutdown & Balance Verification

```bash
"$COLAB_WS/scripts/verify_shutdown.sh"
"$COLAB_WS/scripts/check_budget.sh"
```

Confirm `"rate": 0.0`, `"sessions": []`, and `GPU_STOP_VERIFIED: no active sessions.`.

If your batch used `--keep-warm-ttl <60..900>` and no more renders are queued, reclaim the warm session immediately and verify zero billing:

```bash
"$COLAB_WS/scripts/verify_shutdown.sh" --stop-warm
"$COLAB_WS/scripts/check_budget.sh"
```

---

## 5. Invoking From Another Antigravity IDE Workspace

When orchestrating from a separate Antigravity workspace:

1. Point the caller agent to the workstation path (e.g. `Antigravity-Colab-Workstation`) and your `TaskSpec` file (prefer `drive_file_id` + `sha256`, or an **absolute** `local_path` + `sha256` for reference images — see Section 6).
2. Follow the workflow in [`.agents/skills/colab-h3-video/SKILL.md`](../../.agents/skills/colab-h3-video/SKILL.md):
   - Always run `preflight.sh --task-spec <spec>` first (`0 CU`).
   - State the CU cap (`30.0 CU` per task / `25.0 CU` per session) and request explicit confirmation before launching live execution.
   - Always run `verify_shutdown.sh` and `check_budget.sh` after delivery or failure.
3. Keep `col` / `run_h3_task.sh` inside approved wrapper entrypoints rather than raw `colab new` / `colab run`, because `.agents/pre_tool_guard.py` enforces budget gates and protects `.secrets/` when operating inside the workstation workspace.

---

## 6. `TaskSpec` Cross-Project Input & Output Contract

Schema enforced by `h3_task_spec.py` (see [`config/task_spec.example.json`](../../config/task_spec.example.json), [`config/task_spec.with_tail.example.json`](../../config/task_spec.with_tail.example.json), and [`config/tasks/shao_nian_v1_A.json`](../../config/tasks/shao_nian_v1_A.json)):

- **Required identifiers**:
  - `task_id`: matches `[A-Za-z0-9][A-Za-z0-9_-]{0,63}`
  - `output_prefix`: output filename stem (`<output_prefix>.mp4`, same regex)
- **Reference images (`ref_image_1` required, `ref_image_2` optional)**:
  - Requires 64-char hex `sha256` and **exactly one** of `drive_file_id` or `local_path`.
  - **Cross-project note**: `h3_controller.py` runs with working directory set to the workstation root (`<workstation>`). Therefore external callers should pass either **`drive_file_id`** (recommended) or an **absolute** `local_path` inside `TaskSpec`. Relative `local_path` values resolve inside `<workstation>`.
- **Prompt / Storyboard**:
  - Exactly one of `prompt` (string) or `shots` (list of `{text, label?, start_s?, end_s?}`).
- **Generation & QA geometry**:
  - `width`, `height`, `fps`, `animation_frames`, `tail_card_frames`, `steps`, `seed`, `output_width`, `output_height`.
- **Outputs & Receipts**:
  - Local MP4: `<workstation>/output/<run_id>/<output_prefix>.mp4`
  - Controller ledger & receipts: `<workstation>/runs/<run_id>/`
  - Remote archival: uploaded and server-verified under `Muse-H3-Video-Outputs` on Google Drive (`drive_upload_verified: true`).
