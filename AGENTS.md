# Antigravity-Colab-Workstation — Agent Guide

This repository is a standalone, fail-closed **Google Colab GPU workstation** driven by **Antigravity IDE** (`.agents/`) and standalone shell wrappers (`scripts/`), built atop the `colab-workstation` control plane.

## 1. Repository Layout & Architecture (ADR-003)

- Production `.py` modules live at **repo root** (flat imports, `sys.path.insert(0, str(ROOT))`); **do NOT** relocate root `.py` modules into a `src/` package without preserving imports and updating `SOURCE_MANIFEST.json`.
- `.agents/` houses the Antigravity skill and tool guard:
  - `.agents/skills/colab-h3-video/SKILL.md` — mandatory preflight -> confirm -> caffeinate live run -> QA -> verify_shutdown -> budget workflow.
  - `.agents/hooks.json` & `.agents/pre_tool_guard.py` — blocks `.secrets/`, `.gcloud-colab/`, `~/.config/colab-cli/`, `~/.config/gcloud/`, and unguarded `colab` CLI calls while permitting audited wrapper commands.
- `scripts/` provides shell entrypoints (`install_colab_cli.sh`, `setup_isolated_account.py`, `preflight.sh`, `run_h3_task.sh`, `check_outputs.sh`, `verify_shutdown.sh`, `check_budget.sh`, `install_macos_guardian.sh`, CI validators).
- `config/` contains pinned YAML policy contracts (`budget_policy.yaml`, `gpu_policy.yaml`, `video_profiles.yaml`), manifest examples, and task specs (`config/tasks/`).

## 2. Three-Layer Account & Secret Isolation

- Never read, print, edit, or revoke the host's global ADC (`~/.config/gcloud/application_default_credentials.json`).
- All wrapper scripts load [`scripts/_colab_env.sh`](scripts/_colab_env.sh) (and `scripts/run_shutdown_guardian.py` sets identical defaults), pinning:
  - `COLAB_AUTH_MODE=oauth2`
  - `GOOGLE_APPLICATION_CREDENTIALS=<repo>/.secrets/colab_drive_token.json`
  - `CLOUDSDK_CONFIG=<repo>/.gcloud-colab`
  - `H3_REQUIRE_ISOLATED_ADC=1`
- Never commit `.secrets/`, `.gcloud-colab/`, `~/.config/colab-cli/`, live `model_cache/` runtime manifests, `runs/`, `output/`, `logs/`, binary weights, or generated `.mp4`/`.jpg` media.
- When documenting credentials or OAuth flows in docs or tests, never embed token prefixes (`ya29.`, `ghp_`, `github_pat_`) or `refresh_token`/`client_secret` assignment strings that trip `scripts/secret_scan.py` or `gitleaks`.

## 3. Budget & Live Execution Safety (ADR-001)

- Hard budget caps: **`30.0 CU` per production task** and **`25.0 CU` per single Colab runtime session**. Do NOT modify `config/*.yaml` or `h3_budget.py` / `h3_contract.py` constants unless updating contract tests intentionally.
- Never allocate a live Colab runtime (`colab new` or `scripts/run_h3_task.sh --confirm-live ...`) during documentation, refactoring, CI, or review work unless explicitly approved for a production run.
- Always run `./scripts/preflight.sh` (`0 CU`) before any live task, wrap macOS live runs with `/usr/bin/caffeinate -dimsu`, and run `./scripts/verify_shutdown.sh` + `./scripts/check_budget.sh` afterward.
- Note on cross-project callers: `scripts/preflight.sh`, `scripts/run_h3_task.sh`, and `scripts/check_outputs.sh` resolve relative `--task-spec` / MP4 paths against `CALLER_PWD` before `cd`ing to repo root, whereas relative `ref_image_*.local_path` inside a `TaskSpec` still resolves from repo root (cross-project specs should use `drive_file_id` or an absolute `local_path`).

## 4. Verification & Manifest Discipline

All offline checks spend **0 CU**:

```bash
.venv-colab-cli/bin/python -m py_compile *.py scripts/*.py tests/*.py
.venv-colab-cli/bin/python tests/run_ci_tests.py
.venv-colab-cli/bin/python scripts/check_workflow_schema.py
.venv-colab-cli/bin/python scripts/verify_patch.py
.venv-colab-cli/bin/python scripts/verify_manifest.py
.venv-colab-cli/bin/python scripts/secret_scan.py
.venv-colab-cli/bin/python scripts/check_docs_index.py
```

- **Important test runner distinction**: Two poster-specific integration tests in `tests/` fail under raw `unittest discover` when binary poster assets are absent; always use `tests/run_ci_tests.py`, which explicitly handles asset-gated skips cleanly.
- If you add, rename, delete, or modify tracked files, regenerate [`SOURCE_MANIFEST.json`](SOURCE_MANIFEST.json) with `.venv-colab-cli/bin/python scripts/regenerate_manifest.py` before committing. (`regenerate_manifest.py` and `verify_manifest.py` skip `.DS_Store`, `__pycache__`, `logs/`, `runs/`, `output/`, `outputs/`, `artifacts/`, and non-example `model_cache/` files.)
- Also run `grep` / `secret_scan.py` checks before pushing to the public remote to confirm zero personal usernames, emails, host paths, or credentials appear in tracked files.
