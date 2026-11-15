#!/usr/bin/env bash
# preflight.sh — offline + live read-only preflight. NEVER creates a GPU.
# Usage: ./scripts/preflight.sh [--run-id ID] [--task-spec SPEC]
# Runs: <resolved python> h3_controller.py preflight
# Interpreter resolution (TASK-007.1, fix 1): prefer the production venv
# python (.venv-colab-cli, then .venv) so the Drive checks have their
# google libs; fall back to system python3 with a warning when no venv
# is present (offline checks still run, Drive checks will fail closed).
# Preflight checks (h3_controller.preflight): local sources, CLI post-body
# and drivemount patches, PTY availability, budget ledger, live `colab
# usage` / `colab sessions` (read-only), GPU policy, and restore capacity.
# It does NOT allocate a session, mount Drive, or spend CU.
set -euo pipefail
CALLER_PWD="$(pwd -P)"
cd "$(dirname "$0")/.."
REPO_REAL="$(pwd -P)"
. "./scripts/_colab_env.sh"

_resolve_caller_path() {
  local p="$1"
  case "$p" in
    ""|/*) printf '%s' "$p"; return ;;
  esac
  if [ "$CALLER_PWD" = "$REPO_REAL" ]; then
    printf '%s' "$p"
  elif [ -e "$CALLER_PWD/$p" ]; then
    printf '%s' "$CALLER_PWD/$p"
  elif [ -e "$REPO/$p" ]; then
    printf '%s' "$REPO/$p"
  else
    printf '%s' "$CALLER_PWD/$p"
  fi
}

ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --task-spec)
      if [ $# -lt 2 ]; then
        echo "--task-spec requires a path argument" >&2
        exit 2
      fi
      ARGS+=("--task-spec" "$(_resolve_caller_path "$2")")
      shift 2
      ;;
    --task-spec=*)
      ARGS+=("--task-spec=$(_resolve_caller_path "${1#--task-spec=}")")
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [ ${#ARGS[@]} -gt 0 ]; then
  exec "$PY" h3_controller.py preflight "${ARGS[@]}"
else
  exec "$PY" h3_controller.py preflight
fi
