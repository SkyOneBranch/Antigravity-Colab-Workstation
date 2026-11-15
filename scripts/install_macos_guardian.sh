#!/usr/bin/env bash
# install_macos_guardian.sh — manage the macOS launchd shutdown guardian
# (`com.colabworkstation.shutdown-guardian`) that runs
# scripts/run_shutdown_guardian.py every 300 seconds with isolated credentials.
#
# Usage:
#   ./scripts/install_macos_guardian.sh --install
#   ./scripts/install_macos_guardian.sh --status
#   ./scripts/install_macos_guardian.sh --uninstall
set -euo pipefail

cd "$(dirname "$0")/.."
. "./scripts/_colab_env.sh"

LABEL="com.colabworkstation.shutdown-guardian"
PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST_PATH="$PLIST_DIR/$LABEL.plist"
LOG_DIR="$HOME/Library/Logs"
STDOUT_LOG="$LOG_DIR/colab-workstation-shutdown-guardian.log"
STDERR_LOG="$LOG_DIR/colab-workstation-shutdown-guardian.err.log"
GUARDIAN_SCRIPT="$REPO/scripts/run_shutdown_guardian.py"

ACTION=""
while [ $# -gt 0 ]; do
  case "$1" in
    --install|--status|--uninstall)
      if [ -n "$ACTION" ]; then
        echo "Specify only one of --install, --status, or --uninstall" >&2
        exit 2
      fi
      ACTION="$1"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--install | --status | --uninstall]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--install | --status | --uninstall]" >&2
      exit 2
      ;;
  esac
done

ACTION="${ACTION:---status}"

_xml_escape() {
  local s="$1"
  s="${s//&/&amp;}"
  s="${s//</&lt;}"
  s="${s//>/&gt;}"
  printf '%s' "$s"
}

_require_macos() {
  if [ "$(uname -s)" != "Darwin" ]; then
    echo "macOS LaunchAgent management is only supported on Darwin (got $(uname -s))." >&2
    exit 1
  fi
}

case "$ACTION" in
  --status)
    echo "label: $LABEL"
    echo "repo: $REPO"
    echo "plist_path: $PLIST_PATH"
    if [ -f "$PLIST_PATH" ]; then
      echo "plist_installed: true"
    else
      echo "plist_installed: false"
    fi
    if [ "$(uname -s)" = "Darwin" ] && command -v launchctl >/dev/null 2>&1; then
      LAUNCHD_ROW="$(launchctl list 2>/dev/null | grep -F "$LABEL" || true)"
      if [ -n "$LAUNCHD_ROW" ]; then
        echo "launchd_loaded: true ($LAUNCHD_ROW)"
      else
        echo "launchd_loaded: false"
      fi
    else
      echo "launchd_loaded: unavailable (non-macOS host)"
    fi
    ;;

  --install)
    _require_macos
    if [ "$PY" = "python3" ] || [ ! -x "$PY" ]; then
      echo "Refusing to install LaunchAgent without a repo venv Python (.venv-colab-cli or .venv)." >&2
      exit 1
    fi
    if [ ! -f "$GUARDIAN_SCRIPT" ]; then
      echo "Missing guardian script: $GUARDIAN_SCRIPT" >&2
      exit 1
    fi

    mkdir -p "$PLIST_DIR" "$LOG_DIR"
    TMP_PLIST="$(mktemp "${PLIST_PATH}.tmp.XXXXXX")"
    trap 'rm -f "$TMP_PLIST"' EXIT

    cat >"$TMP_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$(_xml_escape "$LABEL")</string>
  <key>ProgramArguments</key>
  <array>
    <string>$(_xml_escape "$PY")</string>
    <string>$(_xml_escape "$GUARDIAN_SCRIPT")</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$(_xml_escape "$REPO")</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>COLAB_CLI_PATH</key>
    <string>$(_xml_escape "$COLAB_CLI_PATH")</string>
    <key>COLAB_AUTH_MODE</key>
    <string>$(_xml_escape "$COLAB_AUTH_MODE")</string>
    <key>GOOGLE_APPLICATION_CREDENTIALS</key>
    <string>$(_xml_escape "$GOOGLE_APPLICATION_CREDENTIALS")</string>
    <key>CLOUDSDK_CONFIG</key>
    <string>$(_xml_escape "$CLOUDSDK_CONFIG")</string>
    <key>H3_REQUIRE_ISOLATED_ADC</key>
    <string>$(_xml_escape "$H3_REQUIRE_ISOLATED_ADC")</string>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
  </dict>
  <key>StartInterval</key>
  <integer>300</integer>
  <key>RunAtLoad</key>
  <true/>
  <key>StandardOutPath</key>
  <string>$(_xml_escape "$STDOUT_LOG")</string>
  <key>StandardErrorPath</key>
  <string>$(_xml_escape "$STDERR_LOG")</string>
</dict>
</plist>
PLIST

    if command -v plutil >/dev/null 2>&1; then
      plutil -lint "$TMP_PLIST" >/dev/null
    fi

    DOMAIN="gui/$(id -u)"
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || launchctl unload "$PLIST_PATH" 2>/dev/null || true
    mv "$TMP_PLIST" "$PLIST_PATH"
    trap - EXIT
    launchctl bootstrap "$DOMAIN" "$PLIST_PATH" 2>/dev/null || launchctl load -w "$PLIST_PATH"
    echo "Installed and loaded $LABEL -> $PLIST_PATH"
    ;;

  --uninstall)
    _require_macos
    DOMAIN="gui/$(id -u)"
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || launchctl unload "$PLIST_PATH" 2>/dev/null || true
    rm -f "$PLIST_PATH"
    echo "Uninstalled $LABEL"
    ;;
esac
