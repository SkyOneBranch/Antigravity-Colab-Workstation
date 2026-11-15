#!/usr/bin/env bash
# verify_shutdown.sh — READ-ONLY. Confirms no GPU session is still active.
# Usage: ./scripts/verify_shutdown.sh [SESSION_ID]
# Runs `python3 h3_controller.py check` (read-only `colab usage`/`sessions`)
# and fails unless the sessions list is empty. If a session id is given, it
# is additionally checked against raw `colab --auth=adc sessions` output
# (the watchdog's session_still_active check). This script never stops
# anything; stopping is h3_session_watchdog.stop_session (3 attempts) and
# h3_controller.stop_and_verify_session, run explicitly by an operator.
set -euo pipefail
cd "$(dirname "$0")/.."
. "./scripts/_colab_env.sh"
OUT="$("$PY" h3_controller.py check)"
echo "$OUT"
if ! "$PY" -c "import json,sys; d=json.loads(sys.argv[1]); sys.exit(0 if d.get('sessions')==[] else 1)" "$OUT"; then
  echo "SHUTDOWN NOT VERIFIED: active sessions remain (see above)." >&2
  exit 1
fi
if [ $# -ge 1 ] && { [ -x "$COLAB_CLI_PATH" ] || command -v "$COLAB_CLI_PATH" >/dev/null 2>&1; }; then
  if "$COLAB_CLI_PATH" --auth="$COLAB_AUTH_MODE" sessions 2>&1 | grep -q "$1"; then
    echo "SHUTDOWN NOT VERIFIED: session $1 still appears in raw sessions output." >&2
    exit 1
  fi
fi
echo "GPU_STOP_VERIFIED: no active sessions."
