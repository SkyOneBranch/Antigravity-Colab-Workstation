"""Secret scan for the worktree (stdlib only, no network, no credentials).

Run from the repo root:

    python3 scripts/secret_scan.py

Checks every tracked text file for credential-shaped content and reports
only file:line, rule id and severity — never the matched value.

Rules (same shapes as the previous CI grep step):
  ya29-access-token   Google OAuth access token prefix
  github-pat          ghp_ / github_pat_ token prefixes
  private-key-block   -----BEGIN ... PRIVATE KEY-----
  credential-assign   refresh_token / client_secret / etc. assigned a value

Narrow, documented exception (TASK-002 §3.5): the redaction-regex table
in reference_v2/colab_manager.py MUST contain those shapes — the patterns
are the scrubber itself, not values. Only lines between the markers
`# secret-scan: scrubber-patterns-begin` / `...-end` in that one file are
exempt, and the exemption is reported in the output so it stays visible.
No other file, and no other range in that file, is allow-listed.

Also fails on any tracked file whose name looks like a live credential /
state file (credentials, token, oauth, secret, ADC JSON).

This is the second layer: CI additionally runs gitleaks 8.27.2 (pinned
release, SHA-256 verified) over the worktree and the git history.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# SHA-256 digests of historical real Google Drive file/folder IDs scrubbed in
# v0.3.1 (P1-6). Stored as one-way hashes so the scanner blocks re-introduction
# without embedding any real ID in the public repository.
HISTORICAL_DRIVE_ID_SHA256 = {
    "07dfdd430773e102158380b8768c0b79604acbec827c1f14a94b720e98cd1afc",
    "0861c56550dde991b272f85547cae137788031ecb3a43393dceddf192ae1a9c2",
    "1fb429fc7757c14dbbd33d1a716309804b1df2a9b777b03ce4c6d79289b385ad",
    "2b07890a81b1f2bdfa03fa01b33a6aa7fdbc388ef08eddec4ad7bc33b9cade71",
    "3a90e7dfb70ed60f8a7943cf33d483673a0afff04e9142f8a863b7ec92eb2bec",
    "4c5ebbb9cf2a862a8a29f636d7b3a1983daa980bc2766e5e3ffa2a5333aac5b4",
    "71b044ef460ec0398d770f6acc6774c0bdbc192ba29c31f6f99101cba0db9c23",
    "80d8b211ea33b4a58b0fa50a9d0fb19ca3cf7e54449eb5376438e0afb85abd94",
    "80ef435ceb3701474f025ca013f770404582e3976e3aa1bc9e4b13fd15d96cac",
    "878ef0dd999fdf8bcc04a39bd9797e42ca843ac7757358d2568c6850807cefff",
    "8f2dc300e1c4ce527f14fde205468aa75e51acc92b5bdb827128ad269bb33930",
    "8f5c271c36b4043cab1776e645826200cb4666388a7b0f9f9d16e2bfa09a3dbb",
    "991ff3c3ac483713298a6e8441b491140f232cc4411a077c0e37ff800548f069",
    "a2b52d26b3dc69932ee591497d0a336e2da0c6be00f7aff91a16687c397dedbb",
    "a87a523e657c4dcf94d03ae86e3cedd326e9e6916b53908e3a5ca68893b93996",
    "b280edf67dfbdb261f89b341ab49e46604a825c5aa7026511c9bcee6ad271f19",
    "c9db2943334470e5905fff439cf3162d696e091533ee5f7db44daaf8c8e70f00",
    "ca748e1db86073c4b2319b244070fd4d915338c9bada0f60768abfe000ad9f80",
    "e023f3db6cf3f4c0866fe6bd4cb69eafc12026bd269e2614fbe2e99fda4fef19",
    "e329bf56662d51f3c2986f49bf8080cdfce56a1981ec74cd0b21f6be2897b83f",
}

DRIVE_ID_TOKEN_RE = re.compile(r"\b1[A-Za-z0-9_-]{31,34}\b")
DRIVE_ID_ASSIGN_RE = re.compile(
    r"(?i)(?:drive_file_id|drive_root_folder_id|drive_output_folder_id|"
    r"drive_folder_id|cache_drive_folder_id|poster_drive_id|story_drive_id|"
    r"ready_manifest_drive_id|depot_root_drive_id|a_mp4_drive_id|"
    r"folder_drive_id|parent_id)"
    r"""["']?\s*[:=]\s*["']?(1[A-Za-z0-9_-]{31,34})["']?"""
)
_SYNTHETIC_DRIVE_PREFIXES = ("EXAMPLE_", "PLACEHOLDER_", "SYNTHETIC_", "FAKE_", "TEST_")


def _is_high_entropy_drive_id(token: str) -> bool:
    if any(token.upper().startswith(p) for p in _SYNTHETIC_DRIVE_PREFIXES):
        return False
    # Exclude pure lowercase/uppercase hex hashes (MD5/SHA).
    if re.fullmatch(r"[0-9a-fA-F]{32,}", token):
        return False
    has_upper = any(c.isupper() for c in token)
    has_lower = any(c.islower() for c in token)
    has_digit = any(c.isdigit() for c in token)
    return has_upper and has_lower and has_digit

RULES = [
    ("ya29-access-token", "HIGH", re.compile(r"ya29\.[A-Za-z0-9_\-\.]{10,}")),
    ("github-pat", "HIGH", re.compile(r"(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("private-key-block", "HIGH", re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |PGP )?PRIVATE KEY-----")),
    (
        "credential-assign",
        "HIGH",
        re.compile(
            r"(?i)(?:refresh_token|client_secret|access_token|authorization_code|id_token|private_key)"
            r"""\s*["']?\s*[:=]\s*["'][A-Za-z0-9._\-/]{10,}"""
        ),
    ),
]

TEXT_SUFFIXES = {".py", ".sh", ".md", ".json", ".yaml", ".yml", ".toml", ".txt", ".cfg", ".ini"}
# A tracked file is credential-like only when its BASENAME starts with a
# secret-material word (or is a key/cert/env file). Incident docs *about*
# credentials (docs/incidents/08-browser-oauth-pty-credential-...) and
# this scanner itself (scripts/secret_scan.py) are not credential files:
# the former is evidence prose, the latter is the scanner — both are
# content-scanned above like every other text file.
CREDENTIAL_BASENAME = re.compile(
    r"(?i)^(credentials?|tokens?|oauth|application_default_credentials)([._-]|$)"
)
CREDENTIAL_SUFFIX = {".pem", ".key", ".p12", ".env"}
SELF = "scripts/secret_scan.py"
EXEMPT_FILE = "reference_v2/colab_manager.py"
EXEMPT_BEGIN = "# secret-scan: scrubber-patterns-begin"
EXEMPT_END = "# secret-scan: scrubber-patterns-end"


def tracked_files() -> list[str]:
    # Tracked files plus untracked, non-ignored files (so the scan also
    # covers new files before they are committed — CI sees them tracked).
    r = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return [line for line in r.stdout.splitlines() if line.strip()]


def main() -> int:
    findings = []
    exemptions = 0
    for rel in tracked_files():
        p = ROOT / rel
        if not p.is_file():
            continue
        if rel != SELF and (CREDENTIAL_BASENAME.match(p.name) or p.suffix in CREDENTIAL_SUFFIX):
            findings.append((rel, 0, "credential-filename", "HIGH"))
            continue
        if p.suffix not in TEXT_SUFFIXES and rel not in (".github/workflows/ci.yml",):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, ValueError):
            continue
        in_exempt_block = False
        for i, line in enumerate(text.splitlines(), start=1):
            if rel == EXEMPT_FILE and EXEMPT_BEGIN in line:
                in_exempt_block = True
                continue
            if rel == EXEMPT_FILE and EXEMPT_END in line:
                in_exempt_block = False
                continue
            if in_exempt_block:
                exemptions += 1
                continue
            for rule_id, severity, pattern in RULES:
                if pattern.search(line):
                    findings.append((rel, i, rule_id, severity))
                    break
            else:
                if rel != SELF:
                    drive_hit = False
                    for m in DRIVE_ID_TOKEN_RE.finditer(line):
                        tok = m.group(0)
                        if hashlib.sha256(tok.encode("utf-8")).hexdigest() in HISTORICAL_DRIVE_ID_SHA256:
                            findings.append((rel, i, "leaked-historical-drive-id", "HIGH"))
                            drive_hit = True
                            break
                    if not drive_hit:
                        for m in DRIVE_ID_ASSIGN_RE.finditer(line):
                            tok = m.group(1)
                            if _is_high_entropy_drive_id(tok):
                                findings.append((rel, i, "real-drive-id-literal", "HIGH"))
                                break
    if findings:
        print("SECRET SCAN FAILED:")
        for rel, line, rule_id, severity in findings:
            print(f"  {{\"file\": \"{rel}\", \"line\": {line}, \"type\": \"{rule_id}\", "
                  f"\"severity\": \"{severity}\", \"verified_secret\": false}}")
        return 1
    print(f"SECRET SCAN OK: no credential-shaped content in tracked text files "
          f"(scrubber-pattern exemption lines skipped in {EXEMPT_FILE}: {exemptions}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
