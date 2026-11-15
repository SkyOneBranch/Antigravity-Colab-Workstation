#!/usr/bin/env bash
# check_sessions.sh — READ-ONLY. Prints live CU balance and active sessions.
# Usage: ./scripts/check_sessions.sh
# Runs: python3 h3_controller.py check
# (h3_controller.inspect runs `colab usage` and `colab sessions` via run_cli
# and parses them with the strict parsers; it never creates/stops anything.)
# If a `colab` binary is on PATH it also prints the raw `colab sessions`
# output for comparison (same command the watchdog uses: [exe, --auth=adc,
# sessions] in h3_session_watchdog._run_colab).
set -euo pipefail
cd "$(dirname "$0")/.."
. "./scripts/_colab_env.sh"
"$PY" h3_controller.py check
if [ -x "$COLAB_CLI_PATH" ] || command -v "$COLAB_CLI_PATH" >/dev/null 2>&1; then
  echo "--- raw: colab --auth=$COLAB_AUTH_MODE sessions ---"
  "$COLAB_CLI_PATH" --auth="$COLAB_AUTH_MODE" sessions || true
fi
