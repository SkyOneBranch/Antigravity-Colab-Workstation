#!/usr/bin/env bash
# _colab_env.sh — Shared Project 3 isolated credential & venv environment.
# Ensures colab-workstation NEVER reads or overwrites the host's global
# ~/.config/gcloud/application_default_credentials.json (main account).

_SELF_SRC="${BASH_SOURCE[0]:-${(%):-%x}}"
REPO="$(cd "$(dirname "$_SELF_SRC")/.." && pwd)"
unset _SELF_SRC

if [ -x "$REPO/.venv-colab-cli/bin/python" ]; then
  PY="$REPO/.venv-colab-cli/bin/python"
  export COLAB_CLI_PATH="${COLAB_CLI_PATH:-$REPO/.venv-colab-cli/bin/colab}"
elif [ -x "$REPO/.venv/bin/python" ]; then
  PY="$REPO/.venv/bin/python"
  export COLAB_CLI_PATH="${COLAB_CLI_PATH:-$REPO/.venv/bin/colab}"
else
  echo "WARNING: no production venv python found; falling back to system python3" >&2
  PY="python3"
  export COLAB_CLI_PATH="${COLAB_CLI_PATH:-colab}"
fi

# Project 3 isolated dual-channel auth defaults:
# 1) Compute plane: colab-cli first-party OAuth2 (~/.config/colab-cli/token.json)
# 2) Data plane: project-local Drive token ($REPO/.secrets/colab_drive_token.json)
export COLAB_AUTH_MODE="${COLAB_AUTH_MODE:-oauth2}"
export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-$REPO/.secrets/colab_drive_token.json}"
export CLOUDSDK_CONFIG="${CLOUDSDK_CONFIG:-$REPO/.gcloud-colab}"
export H3_REQUIRE_ISOLATED_ADC="${H3_REQUIRE_ISOLATED_ADC:-1}"
