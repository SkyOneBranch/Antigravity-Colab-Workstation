---
name: colab-h3-video
description: Standard macOS Antigravity execution workflow for colab-workstation H3 video generation tasks with >=95% reliability, Project 3 isolated dual-channel Google auth, and 30 CU per-task budget enforcement.
---

# `colab-h3-video` — macOS Antigravity 执行规范（Project 3 多账号物理隔离版）

## 1. 多账号物理隔离与安全红线（防串号）
- **全局主账号零触碰**：宿主机全局 `~/.config/gcloud/application_default_credentials.json` 永久归属日常主账号（项目 1），本项目通过 `H3_REQUIRE_ISOLATED_ADC=1` 强制物理阻断全局回退，**严禁读取、修改或覆盖全局 ADC**。
- **双通道独立凭据（项目 3）**：
  - **数据面（新 Colab 账号 Google Drive）**：`GOOGLE_APPLICATION_CREDENTIALS=$REPO/.secrets/colab_drive_token.json`（权限 `0600`）。
  - **算力面（新 Colab 账号消费级 GPU Runtime）**：`COLAB_AUTH_MODE=oauth2` 读取 `~/.config/colab-cli/token.json`（权限 `0600`）。
  - 所有封装脚本（`scripts/*.sh`）自动加载 `scripts/_colab_env.sh` 并在运行前执行 `same_google_account` 一致性断言。
- **严禁读取或打印凭据明文**：永远不要读取 `.secrets/`、`~/.config/gcloud/`、`~/.config/colab-cli/token.json` 或打印任何 token/secret。仅可通过 `.venv-colab-cli/bin/python scripts/setup_isolated_account.py --check-only` 检查元数据就绪状态。
- **严禁裸调 `colab new` / `colab run`**：所有 GPU 分配必须通过 `./scripts/run_h3_task.sh` 受管执行，确保写入单任务 30 CU 账本与 Lease 停机注册表。

## 2. 标准执行链路
1. **隔离账号状态自检（0 CU）**：
   ```bash
   .venv-colab-cli/bin/python scripts/setup_isolated_account.py --check-only
   ```
2. **TaskSpec 校验 + 实况预检（0 CU）**：
   ```bash
   ./scripts/preflight.sh --task-spec config/tasks/<task_id>.json
   ```
   确认 `adc_valid`、`colab_oauth_token_valid`、`same_account_and_drive_ready`、`drive_download_preflight`、`drive_file_token_archive` 等全项全绿才允许进入实跑。
3. **防休眠受管交付（`caffeinate` + `--deliver`，支持 v0.3.0 自愈与温池）**：
   ```bash
   /usr/bin/caffeinate -dimsu ./scripts/run_h3_task.sh \
     --confirm-live --deliver \
     --production-task-id <task_id> \
     --authorize-new-task \
     --task-spec config/tasks/<task_id>.json \
     --max-task-attempts 2 \
     --allocation-rounds 3 \
     --bootstrap-mode bundled \
     --task-deadline-utc <ISO8601_UTC>
   ```
   - **批量连跑温池复用（可选 `--keep-warm-ttl <60..900>`）**：仅在连续提交多个任务时追加 `--keep-warm-ttl 300`（`0` 关闭，合法区间 `60..900` 秒）。任务成功后在 `25.0 CU` 单会话预算与 `14400s` 生命周期内保留 `:8188` ComfyUI 温态供下一任务秒级复用；任何失败均立即 fail-closed 停机。
   - **宿主机断连 0-GPU 自愈**：若宿主机中断而远端 GPU 已停机且成片已归档至 Google Drive，再次执行 `--deliver`（或 `--auto-recover`）会自动走 `recover_from_drive_archive()` 以 **0 CU** 拉回并验收成片。
4. **异步等待与验收 / 温池清理**：使用后台异步等待（Reactive Wakeup，不在循环中轮询），完成后运行：
   ```bash
   ./scripts/check_outputs.sh <path/to/final.mp4>
   ./scripts/verify_shutdown.sh
   ```
   确认成片 MD5 持久化、本地 QA 通过、且 GPU 会话确认为 `sessions: []` 且 `rate: 0.0`。
   - 若本批次启用了 `--keep-warm-ttl` 且所有连跑任务已结束，**必须立即显式回收温池会话并验证停机**：
     ```bash
     ./scripts/verify_shutdown.sh --stop-warm
     ```
     （默认 `./scripts/verify_shutdown.sh` 保持 100% 只读；`--stop-warm` 会先跨任务回收全部 `warm_idle` 租约，再执行只读 `check` 断言 `sessions == []` 且 `rate == 0.0`。）
