#!/usr/bin/env python3
"""Antigravity PreToolUse guard for colab-workstation.

Auto-allows controlled operational scripts and normal workspace file reads
while hard-blocking:
1. Reading/printing global ADC, Colab OAuth2 keyring, or .secrets/ credential files.
2. Unmanaged `colab new` / `colab run` calls outside h3_controller.py.
"""
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[1])

SECRET_PATH_PATTERNS = (
    r"\.config/gcloud/",
    r"application_default_credentials\.json",
    r"\.config/colab-cli/token\.json",
    r"(^|[\s/'\"~])\.secrets(/|$)",
    r"(^|[\s/'\"~])\.gcloud-colab(/|$)",
    r"(^|[\s/'\"~])private/",
    r"drive_token\.json",
    r"colab_drive_token\.json",
    r"client_secret[^/\s'\"]*\.json",
    r"desktop_oauth_client\.json",
)

UNMANAGED_COLAB_PATTERNS = (
    r"(^|[;&|\s])(?:\S*/)?colab\b[^\n]*\b(new|run)\b",
)

ALLOW_PREFIXES = (
    "./scripts/preflight.sh",
    "./scripts/check_sessions.sh",
    "./scripts/check_budget.sh",
    "./scripts/verify_drive.sh",
    "./scripts/verify_shutdown.sh",
    "scripts/verify_shutdown.sh",
    "./scripts/check_outputs.sh",
    "./scripts/run_h3_task.sh",
    "./scripts/install_macos_guardian.sh --status",
    "/usr/bin/caffeinate -dimsu ./scripts/run_h3_task.sh",
    ".venv-colab-cli/bin/python scripts/setup_isolated_account.py --check-only",
    ".venv-colab-cli/bin/python scripts/run_shutdown_guardian.py",
    ".venv-colab-cli/bin/python h3_shutdown_executor.py",
    ".venv-colab-cli/bin/python drive_depot.py account-check",
    ".venv-colab-cli/bin/python drive_depot.py verify",
    ".venv-colab-cli/bin/python tests/run_ci_tests.py",
    ".venv-colab-cli/bin/python scripts/check_workflow_schema.py",
    ".venv-colab-cli/bin/python scripts/verify_patch.py",
    ".venv-colab-cli/bin/python scripts/secret_scan.py",
    ".venv-colab-cli/bin/python scripts/verify_manifest.py",
    ".venv-colab-cli/bin/python scripts/check_docs_index.py",
)

_CD_PREFIX_RE = re.compile(
    r"^cd\s+(?:'[^']+'|\"[^\"]+\"|\S+)\s*&&\s*"
)


def _reply(decision: str, reason: str = "") -> int:
    payload = {"decision": decision}
    if reason:
        payload["reason"] = reason
    sys.stdout.write(json.dumps(payload))
    return 0


def _is_safe_workspace_path(raw_path: str) -> bool:
    if not raw_path:
        return False
    try:
        resolved = str(Path(raw_path).expanduser().resolve())
    except Exception:
        return False
    return resolved == REPO_ROOT or resolved.startswith(REPO_ROOT + os.sep)


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return _reply("ask", "Failed to parse hook input")

    tool_call = data.get("toolCall") or {}
    tool_name = tool_call.get("name", "")
    args = tool_call.get("args") or {}

    if tool_name == "read_file":
        path = str(args.get("Path", ""))
        for pat in SECRET_PATH_PATTERNS:
            if re.search(pat, path):
                return _reply("deny", f"Blocked reading credential path: {path}")
        if _is_safe_workspace_path(path):
            return _reply("allow", "Non-secret colab-workstation file")
        return _reply("ask")

    if tool_name == "run_command":
        cmd = str(args.get("CommandLine", "")).strip()
        # Note: allow passing `--check-only` to setup_isolated_account.py even though
        # it does not reference secret filenames on the command line; check secret
        # path patterns first so cat/jq/python reading .secrets/ is always denied.
        for pat in SECRET_PATH_PATTERNS:
            if re.search(pat, cmd):
                return _reply("deny", "Blocked shell access to credential/OAuth file.")
        for pat in UNMANAGED_COLAB_PATTERNS:
            if re.search(pat, cmd):
                return _reply("deny", "Blocked unmanaged colab new/run; use ./scripts/run_h3_task.sh.")
        normalized = _CD_PREFIX_RE.sub("", cmd)
        for prefix in ALLOW_PREFIXES:
            if normalized == prefix or normalized.startswith(prefix + " "):
                return _reply("allow", f"Approved colab-workstation entrypoint: {prefix}")
        return _reply("ask")

    return _reply("ask")


if __name__ == "__main__":
    raise SystemExit(main())
