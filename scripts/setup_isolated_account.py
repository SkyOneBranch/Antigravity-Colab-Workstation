#!/usr/bin/env python3
"""Interactive / Browser-Click setup for Project 3 isolated Colab + Drive account.

Binds colab-workstation to a dedicated "New Colab Account" across two planes
WITHOUT ever reading, modifying, or revoking the host's global main-account ADC
(~/.config/gcloud/application_default_credentials.json):

  1. Data Plane (Google Drive API):
     Uses Desktop OAuth client config (.secrets/client_secret.json) via local
     loopback callback (`http://localhost:<port>/`, `prompt="select_account consent"`)
     and writes an isolated 0600 credential file to .secrets/colab_drive_token.json
     with drive + drive.readonly + drive.file + userinfo.email scopes.
  2. Compute Plane (Consumer Colab RuntimeService):
     Uses official `colab_cli/oauth_config.json` via local loopback callback
     (`--browser-login`) or terminal prompt, writing ~/.config/colab-cli/token.json
     with mode 0600.
  3. Cross-Plane Consistency & Zero-CU Verification:
     Proves both planes belong to the exact same Google account
     (`same_google_account == True`), verifies downscoped worker token minting
     (`drive.readonly` + `drive.file`), verifies `colab --auth=oauth2 usage`,
     optionally discovers an existing `Muse-H3-Model-Depot` on that Drive, and
     verifies global ADC metadata was untouched.

Security invariant: NEVER prints, logs, or commits any auth code, token, secret, or email.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import resources
import io
import json
import os
import subprocess
import sys
import time
import urllib.parse
import wsgiref.simple_server
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auth_support import (  # noqa: E402
    audit_colab_oauth_token_file,
    audit_user_adc_file,
    global_gcloud_adc_path,
    resolve_colab_executable,
)

SECRETS_DIR = ROOT / ".secrets"
DEFAULT_CLIENT_CFG = SECRETS_DIR / "client_secret.json"
DEFAULT_DRIVE_CRED = SECRETS_DIR / "colab_drive_token.json"
COLAB_TOKEN_PATH = Path.home() / ".config" / "colab-cli" / "token.json"

REQUIRED_DRIVE_SCOPES = (
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
)


def _stat_fingerprint(path: Path) -> tuple[bool, int | None, int | None]:
    """Metadata-only snapshot (never reads file bytes)."""
    if not path.exists():
        return (False, None, None)
    st = path.stat()
    return (True, st.st_mtime_ns, st.st_size)


class _SilentWSGIHandler(wsgiref.simple_server.WSGIRequestHandler):
    """Never log callback query strings (which contain one-time OAuth codes)."""

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return


def _run_loopback_browser_oauth(flow, *, step_title: str, html_body: str,
                                open_browser: bool, timeout_sec: int = 600):
    """Start ephemeral localhost callback, open macOS browser, and exchange token
    without ever logging the callback URI or authorization code."""
    captured: dict[str, str] = {}

    def _app(environ, start_response):
        qs = environ.get("QUERY_STRING", "")
        params = urllib.parse.parse_qs(qs)
        path_info = environ.get("PATH_INFO", "/")
        if "code" in params or "error" in params:
            req_uri = wsgiref.util.request_uri(environ)
            captured["uri"] = req_uri
            if "error" in params:
                err_code = params.get("error", ["unknown"])[0]
                page = (
                    "<html><head><meta charset='utf-8'><title>OAuth Error</title></head>"
                    "<body style='font-family:-apple-system,sans-serif;padding:36px;'>"
                    f"<h2>授权未完成 ({err_code})</h2>"
                    "<p>请回到 Antigravity 对话框查看处理提示。</p></body></html>"
                )
            else:
                page = (
                    "<html><head><meta charset='utf-8'><title>OAuth Complete</title></head>"
                    "<body style='font-family:-apple-system,sans-serif;padding:36px;'>"
                    f"{html_body}</body></html>"
                )
            start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
            return [page.encode("utf-8")]
        # Ignore /favicon.ico or stray browser preflights without consuming the flow.
        start_response("204 No Content", [("Content-Type", "text/plain; charset=utf-8")])
        return [b""]

    server = wsgiref.simple_server.make_server(
        "localhost", 0, _app, handler_class=_SilentWSGIHandler
    )
    try:
        port = server.server_port
        flow.redirect_uri = f"http://localhost:{port}/"
        auth_url, _ = flow.authorization_url(
            prompt="select_account consent",
            access_type="offline",
        )
        print(f"\n=== {step_title} ===", flush=True)
        print(f"AUTH_URL: {auth_url}", flush=True)
        if open_browser:
            subprocess.run(["/usr/bin/open", auth_url], check=False)

        deadline = time.monotonic() + timeout_sec
        while "uri" not in captured:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting {timeout_sec}s for browser OAuth click ({step_title})")
            server.timeout = min(5.0, max(0.5, remaining))
            server.handle_request()

        callback_uri = captured["uri"]
        parsed_qs = urllib.parse.parse_qs(urllib.parse.urlsplit(callback_uri).query)
        if "error" in parsed_qs:
            err_name = parsed_qs["error"][0]
            raise RuntimeError(f"OAuth callback returned error={err_name} during {step_title}")

        # oauthlib requires https scheme check unless relaxed via https replacement (same as google_auth_oauthlib).
        https_uri = "https" + callback_uri[4:] if callback_uri.startswith("http:") else callback_uri
        flow.fetch_token(authorization_response=https_uri)
        return flow.credentials
    finally:
        server.server_close()


def _check_only(drive_cred_path: Path) -> int:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(drive_cred_path)
    os.environ["H3_REQUIRE_ISOLATED_ADC"] = "1"
    os.environ["COLAB_AUTH_MODE"] = "oauth2"

    report: dict = {
        "global_main_adc_path": str(global_gcloud_adc_path()),
        "isolated_drive_cred_path": str(drive_cred_path),
        "colab_oauth_token_path": str(COLAB_TOKEN_PATH),
        "drive_cred_ready": False,
        "colab_oauth_ready": False,
        "same_google_account": None,
        "worker_tokens_mintable": False,
    }
    try:
        report["drive_cred_audit"] = audit_user_adc_file()
        report["drive_cred_ready"] = True
    except Exception as exc:  # noqa: BLE001
        report["drive_cred_error"] = str(exc)

    try:
        report["colab_oauth_audit"] = audit_colab_oauth_token_file()
        report["colab_oauth_ready"] = True
    except Exception as exc:  # noqa: BLE001
        report["colab_oauth_error"] = str(exc)

    if report["drive_cred_ready"] and report["colab_oauth_ready"]:
        try:
            from drive_depot import account_matches, drive_service
            from h3_budget import read_live_usage
            from h3_drive_token import mint_file_token, mint_readonly_token
            from h3_gpu_lease_registry import current_owner_id

            svc = drive_service()
            report["same_google_account"] = bool(account_matches(svc, "oauth2"))
            ro = mint_readonly_token()
            rw = mint_file_token()
            usage = read_live_usage("oauth2")
            report["worker_tokens_mintable"] = True
            report["readonly_token_id"] = ro["token_id"]
            report["file_token_id"] = rw["token_id"]
            report["owner_id"] = current_owner_id("oauth2")
            report["colab_cu_balance"] = usage.get("balance")
            report["colab_active_assignments"] = usage.get("active_assignments")
        except Exception as exc:  # noqa: BLE001
            report["verification_error"] = f"{type(exc).__name__}: {str(exc)[:240]}"

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    ok = bool(
        report["drive_cred_ready"]
        and report["colab_oauth_ready"]
        and report["same_google_account"] is True
        and report["worker_tokens_mintable"]
    )
    return 0 if ok else 1


def _authorize_drive(client_cfg_path: Path, drive_cred_path: Path,
                     quota_project: str | None, *, open_browser: bool,
                     timeout_sec: int) -> None:
    from drive_resumable import atomic_secret_json
    from google_auth_oauthlib.flow import InstalledAppFlow

    if not client_cfg_path.is_file():
        raise FileNotFoundError(f"Desktop OAuth client JSON not found at {client_cfg_path}.")
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(SECRETS_DIR, 0o700)
    if client_cfg_path.parent.resolve() == SECRETS_DIR.resolve():
        os.chmod(client_cfg_path, 0o600)

    os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
    flow = InstalledAppFlow.from_client_secrets_file(
        str(client_cfg_path),
        scopes=list(REQUIRED_DRIVE_SCOPES),
    )
    creds = _run_loopback_browser_oauth(
        flow,
        step_title="[Step 1/2] 新 Colab 账号 Google Drive 隔离凭据授权",
        html_body=(
            "<h2>✅ [1/2 完成] 新 Colab 账号 Google Drive 隔离凭据已写入！</h2>"
            "<p>正在自动为您弹出 <b>第 [2/2] 个窗口（Colab 算力通道授权）</b>，"
            "请在下一个页面继续选择<b>同一个新 Colab 账号</b>点击允许即可。</p>"
        ),
        open_browser=open_browser,
        timeout_sec=timeout_sec,
    )
    if not getattr(creds, "refresh_token", None):
        raise RuntimeError("Google OAuth response did not include a refresh token for Drive.")
    authorized_doc = {
        "type": "authorized_user",
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "refresh_token": creds.refresh_token,
    }
    if quota_project:
        authorized_doc["quota_project_id"] = quota_project
    atomic_secret_json(drive_cred_path, authorized_doc)


def _authorize_colab_runtime(force_colab: bool, *, browser_login: bool,
                             timeout_sec: int) -> None:
    from colab_cli.auth import PUBLIC_SCOPES
    from drive_resumable import atomic_secret_json
    from google_auth_oauthlib.flow import InstalledAppFlow

    COLAB_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(COLAB_TOKEN_PATH.parent, 0o700)
    if force_colab and COLAB_TOKEN_PATH.exists():
        COLAB_TOKEN_PATH.unlink()

    if not browser_login:
        exe = resolve_colab_executable()
        proc = subprocess.run([exe, "--auth=oauth2", "usage"], check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"colab --auth=oauth2 usage exited with code {proc.returncode}")
        if not COLAB_TOKEN_PATH.is_file():
            raise FileNotFoundError(f"Expected Colab OAuth token at {COLAB_TOKEN_PATH} after login")
        os.chmod(COLAB_TOKEN_PATH, 0o600)
        return

    os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
    colab_cfg = json.loads(resources.files("colab_cli").joinpath("oauth_config.json").read_text())
    flow = InstalledAppFlow.from_client_config(colab_cfg, scopes=list(PUBLIC_SCOPES))
    creds = _run_loopback_browser_oauth(
        flow,
        step_title="[Step 2/2] 新 Colab 账号 Colab GPU 算力通道授权",
        html_body=(
            "<h2>🎉 [2/2 全部完成] 新 Colab 账号双通道（Drive + Colab 算力）已绑定成功！</h2>"
            "<p>后台正在自动完成账号一致性核验与预检，您可以直接关闭此标签页回到 Antigravity。</p>"
        ),
        open_browser=True,
        timeout_sec=timeout_sec,
    )
    if not getattr(creds, "refresh_token", None):
        raise RuntimeError("Colab OAuth response did not include a refresh token.")
    token_doc = json.loads(creds.to_json())
    atomic_secret_json(COLAB_TOKEN_PATH, token_doc)


def _ensure_isolated_env(drive_cred_path: Path = DEFAULT_DRIVE_CRED) -> None:
    """Bind isolated env vars so this process never falls back to global ADC."""
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(drive_cred_path)
    os.environ["H3_REQUIRE_ISOLATED_ADC"] = "1"
    os.environ["COLAB_AUTH_MODE"] = "oauth2"


def _download_drive_bytes(svc, file_id: str) -> bytes:
    """Download a small Drive file directly into memory using requests transport."""
    from googleapiclient.http import MediaIoBaseDownload

    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, svc.files().get_media(fileId=file_id, supportsAllDrives=True))
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()


def _discover_existing_depot(
    depot_folder_id: str | None = None,
    drive_cred_path: Path = DEFAULT_DRIVE_CRED,
) -> dict:
    """Read-only check: if Muse-H3-Model-Depot + READY manifest already exist on
    this account's Drive (My Drive or Shared with me), sync local
    model_cache/{depot_config,verified_manifest}.json at 0 GB transfer and 0 CU."""
    from drive_depot import CFG, FOLDER_MIME, HF_REPO, READY, STATE, atomic_json, drive_service, verify_ready
    from h3_contract import MODEL_HF_REVISION

    _ensure_isolated_env(drive_cred_path)

    if depot_folder_id is None and CFG.is_file() and READY.is_file():
        svc = drive_service()
        verified = verify_ready(svc, True, READY)
        return {"existing_local_manifest_verified": True, **verified}

    svc = drive_service()
    if depot_folder_id:
        folder_meta = svc.files().get(
            fileId=depot_folder_id,
            fields="id,name,mimeType,trashed",
            supportsAllDrives=True,
        ).execute()
        if folder_meta.get("trashed") or folder_meta.get("mimeType") != FOLDER_MIME:
            raise RuntimeError(f"Depot folder {depot_folder_id} is trashed or not a Drive folder")
        root_id = folder_meta["id"]
    else:
        res = svc.files().list(
            q=f"name = 'Muse-H3-Model-Depot' and mimeType = '{FOLDER_MIME}' and trashed = false",
            pageSize=10,
            fields="files(id,name,mimeType,parents,shared)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        folders = res.get("files", [])
        if not folders:
            return {
                "existing_depot_found": False,
                "next_step": (
                    "Run `drive_depot.py init --allow-drive-write` then "
                    "`drive_depot.py seed --allow-drive-write --allow-large-transfer "
                    f"--hf-revision {MODEL_HF_REVISION}` to seed the model depot."
                ),
            }
        if len(folders) > 1:
            raise RuntimeError("Multiple Muse-H3-Model-Depot folders found across Drive; pass --depot-folder-id")
        root_id = folders[0]["id"]

    pinned_name = f"READY_{MODEL_HF_REVISION}.json"
    q = (
        f"'{root_id}' in parents and trashed = false and ("
        f"name = '{pinned_name}' or "
        f"(name contains 'READY_' and name contains '.json') or "
        f"name = 'verified_manifest.json')"
    )
    manifests = svc.files().list(
        q=q,
        pageSize=10,
        fields="files(id,name,size,md5Checksum,parents,trashed)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute().get("files", [])
    if not manifests:
        return {
            "existing_depot_found": True,
            "depot_root_id": root_id,
            "remote_manifest_found": False,
            "next_step": f"Depot folder {root_id} exists without READY manifest; run drive_depot.py seed.",
        }

    exact = [m for m in manifests if m.get("name") == pinned_name]
    ready_prefixed = [m for m in manifests if str(m.get("name", "")).startswith("READY_")]
    candidates = exact or ready_prefixed or manifests
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple READY manifests found in Muse-H3-Model-Depot {root_id}; refusing ambiguity")
    remote_meta = candidates[0]

    raw_bytes = _download_drive_bytes(svc, remote_meta["id"])
    manifest_doc = json.loads(raw_bytes.decode("utf-8"))

    # Persist the EXACT remote raw bytes so READY size & MD5 match remote_manifest identically.
    READY.parent.mkdir(parents=True, exist_ok=True)
    READY.write_bytes(raw_bytes)
    os.chmod(READY, 0o600)

    cfg_doc = {
        "drive_root_folder_id": root_id,
        "scope": "drive",
        "hf_repo": HF_REPO,
        "remote_manifest": {
            "drive_file_id": remote_meta["id"],
            "name": remote_meta["name"],
            "md5": remote_meta.get("md5Checksum", ""),
            "size": int(remote_meta.get("size", len(raw_bytes))),
        },
    }
    atomic_json(CFG, cfg_doc)
    os.chmod(CFG, 0o600)

    verified = verify_ready(svc, True, READY)
    atomic_json(
        STATE,
        {
            "status": "READY",
            "hf_revision": manifest_doc.get("hf_revision", ""),
            "verified_files": len(manifest_doc.get("files", [])),
        },
    )
    os.chmod(STATE, 0o600)
    return {
        "existing_depot_found": True,
        "depot_root_id": root_id,
        "remote_manifest_id": remote_meta["id"],
        "remote_manifest_name": remote_meta["name"],
        "synced_from_drive": True,
        **verified,
    }


def _discover_comfyui_cache(
    comfyui_folder_id: str | None = None,
    drive_cred_path: Path = DEFAULT_DRIVE_CRED,
) -> dict:
    """Read-only sync of Muse-H3-ComfyUI-Source metadata (never downloads the tarball)."""
    from drive_depot import FOLDER_MIME, atomic_json, drive_service
    from h3_comfyui_cache import CACHE_FOLDER_NAME, LOCAL_MANIFEST as COMFY_LOCAL_MANIFEST

    _ensure_isolated_env(drive_cred_path)
    svc = drive_service()

    if comfyui_folder_id:
        folder_meta = svc.files().get(
            fileId=comfyui_folder_id,
            fields="id,name,mimeType,trashed",
            supportsAllDrives=True,
        ).execute()
        if folder_meta.get("trashed") or folder_meta.get("mimeType") != FOLDER_MIME:
            raise RuntimeError(f"ComfyUI cache folder {comfyui_folder_id} is trashed or not a Drive folder")
        folder_id = folder_meta["id"]
    else:
        res = svc.files().list(
            q=f"name = '{CACHE_FOLDER_NAME}' and mimeType = '{FOLDER_MIME}' and trashed = false",
            pageSize=10,
            fields="files(id,name,mimeType,parents,shared)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        folders = res.get("files", [])
        if not folders:
            return {"comfyui_cache_found": False}
        if len(folders) > 1:
            raise RuntimeError("Multiple Muse-H3-ComfyUI-Source folders found; pass --comfyui-folder-id")
        folder_id = folders[0]["id"]

    children = svc.files().list(
        q=f"'{folder_id}' in parents and trashed = false",
        pageSize=20,
        fields="files(id,name,size,md5Checksum)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute().get("files", [])
    by_name = {item["name"]: item for item in children}

    manifest_meta = by_name.get("comfyui-source-manifest.json")
    if not manifest_meta:
        return {"comfyui_cache_found": False, "drive_folder_id": folder_id, "reason": "missing comfyui-source-manifest.json"}

    manifest_bytes = _download_drive_bytes(svc, manifest_meta["id"])
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    archive_name = manifest["archive_name"]
    sha_name = f"{archive_name}.sha256"

    tar_meta = by_name.get(archive_name)
    sha_meta = by_name.get(sha_name)
    if not tar_meta or not sha_meta:
        return {
            "comfyui_cache_found": False,
            "drive_folder_id": folder_id,
            "reason": f"missing {archive_name} or {sha_name}",
        }

    sha_bytes = _download_drive_bytes(svc, sha_meta["id"])
    sha_hex = sha_bytes.decode("utf-8").strip().split()[0].lower()
    if sha_hex != manifest["sha256"].lower():
        raise RuntimeError(f"ComfyUI cache sha256 mismatch between manifest and {sha_name}")
    if int(tar_meta["size"]) != int(manifest["size_bytes"]):
        raise RuntimeError(f"ComfyUI cache tarball size mismatch for {archive_name}")

    manifest["drive_folder_id"] = folder_id
    manifest["drive_files"] = {
        archive_name: {
            "id": tar_meta["id"],
            "size": int(tar_meta["size"]),
            "sha256": manifest["sha256"],
            "server_md5": tar_meta.get("md5Checksum", ""),
        },
        sha_name: {
            "id": sha_meta["id"],
            "size": int(sha_meta["size"]),
            "sha256": hashlib.sha256(sha_bytes).hexdigest(),
            "server_md5": sha_meta.get("md5Checksum", ""),
        },
        "comfyui-source-manifest.json": {
            "id": manifest_meta["id"],
            "size": int(manifest_meta["size"]),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "server_md5": manifest_meta.get("md5Checksum", ""),
        },
    }
    atomic_json(COMFY_LOCAL_MANIFEST, manifest)
    os.chmod(COMFY_LOCAL_MANIFEST, 0o600)
    return {
        "comfyui_cache_found": True,
        "drive_folder_id": folder_id,
        "archive_file_id": tar_meta["id"],
        "commit": manifest["commit"],
        "size_bytes": int(tar_meta["size"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bind colab-workstation to an isolated Colab + Drive Google account (0 CU)."
    )
    parser.add_argument(
        "--client-secret",
        default=str(DEFAULT_CLIENT_CFG),
        help="Path to Desktop OAuth client JSON (default: .secrets/client_secret.json).",
    )
    parser.add_argument(
        "--drive-token-out",
        default=str(DEFAULT_DRIVE_CRED),
        help="Isolated output path for Drive authorized_user JSON (default: .secrets/colab_drive_token.json).",
    )
    parser.add_argument(
        "--quota-project",
        default=None,
        help="Optional GCP quota_project_id (omit when reusing External OAuth client across accounts).",
    )
    parser.add_argument(
        "--browser-login",
        action="store_true",
        help="Auto-open macOS browser windows for both Drive and Colab OAuth via localhost callback (zero copy-paste).",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=600,
        help="Seconds to wait per browser OAuth window (default: 600).",
    )
    parser.add_argument("--force-drive", action="store_true", help="Re-run Step 1 Drive OAuth even if token file exists.")
    parser.add_argument("--force-colab", action="store_true", help="Re-run Step 2 Colab OAuth even if token file exists.")
    parser.add_argument("--skip-depot-discovery", action="store_true", help="Skip read-only Drive Model Depot discovery.")
    parser.add_argument("--check-only", action="store_true", help="Non-interactive status check (0 CU, never reads secrets).")
    parser.add_argument(
        "--sync-shared-caches",
        action="store_true",
        help="Non-interactive 0-CU sync of Model Depot & ComfyUI Source Cache manifests using existing credentials.",
    )
    parser.add_argument(
        "--depot-folder-id",
        default=None,
        help="Optional explicit Drive folder ID for Muse-H3-Model-Depot (supports shared folders).",
    )
    parser.add_argument(
        "--comfyui-folder-id",
        default=None,
        help="Optional explicit Drive folder ID for Muse-H3-ComfyUI-Source (supports shared folders).",
    )
    args = parser.parse_args()

    drive_cred_path = Path(args.drive_token_out).expanduser().resolve()
    client_cfg_path = Path(args.client_secret).expanduser().resolve()

    if drive_cred_path == global_gcloud_adc_path().resolve():
        raise SystemExit("REFUSED: --drive-token-out must never point to global ~/.config/gcloud/application_default_credentials.json")

    if args.check_only:
        return _check_only(drive_cred_path)

    global_adc = global_gcloud_adc_path()
    before_global_adc = _stat_fingerprint(global_adc)

    if args.sync_shared_caches:
        rc = _check_only(drive_cred_path)
        if rc != 0:
            print("BLOCKED: Cannot sync shared caches before both Drive and Colab OAuth channels pass verification.", file=sys.stderr)
            return rc
        depot_info = _discover_existing_depot(args.depot_folder_id, drive_cred_path=drive_cred_path)
        comfyui_info = _discover_comfyui_cache(args.comfyui_folder_id, drive_cred_path=drive_cred_path)
        after_global_adc = _stat_fingerprint(global_adc)
        if before_global_adc != after_global_adc:
            raise RuntimeError("CRITICAL: Global ~/.config/gcloud/application_default_credentials.json metadata changed unexpectedly!")
        print(
            json.dumps(
                {
                    "depot_discovery": depot_info,
                    "comfyui_cache_discovery": comfyui_info,
                    "global_main_adc_untouched": True,
                    "no_gpu_created": True,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 0

    if not args.browser_login and not sys.stdin.isatty():
        print("REFUSED: Pass --browser-login when launching from non-TTY agent runner.", file=sys.stderr)
        return 2

    if args.force_drive or not drive_cred_path.is_file():
        _authorize_drive(
            client_cfg_path,
            drive_cred_path,
            args.quota_project,
            open_browser=args.browser_login,
            timeout_sec=args.timeout_sec,
        )
    else:
        os.chmod(drive_cred_path, 0o600)
        print(f"[Step 1/2] Reusing existing isolated Drive token at {drive_cred_path}", flush=True)

    if args.force_colab or not COLAB_TOKEN_PATH.is_file():
        _authorize_colab_runtime(
            args.force_colab,
            browser_login=args.browser_login,
            timeout_sec=args.timeout_sec,
        )
    else:
        os.chmod(COLAB_TOKEN_PATH, 0o600)
        print(f"[Step 2/2] Reusing existing Colab OAuth token at {COLAB_TOKEN_PATH}", flush=True)

    print("\n[Verification] Checking cross-plane account consistency & worker token minting (0 CU)...", flush=True)
    rc = _check_only(drive_cred_path)
    if rc != 0:
        print(
            "\nBLOCKED: Isolated account verification failed (check above whether Drive and Colab "
            "were signed into two different Google accounts; re-run with --force-drive / --force-colab if needed).",
            file=sys.stderr,
        )
        return rc

    if not args.skip_depot_discovery:
        try:
            depot_info = _discover_existing_depot(args.depot_folder_id, drive_cred_path=drive_cred_path)
            print(json.dumps({"depot_discovery": depot_info}, ensure_ascii=False, indent=2), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"depot_discovery_warning": f"{type(exc).__name__}: {str(exc)[:240]}"}, ensure_ascii=False, indent=2), flush=True)
        try:
            comfyui_info = _discover_comfyui_cache(args.comfyui_folder_id, drive_cred_path=drive_cred_path)
            print(json.dumps({"comfyui_cache_discovery": comfyui_info}, ensure_ascii=False, indent=2), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"comfyui_cache_warning": f"{type(exc).__name__}: {str(exc)[:240]}"}, ensure_ascii=False, indent=2), flush=True)

    after_global_adc = _stat_fingerprint(global_adc)
    if before_global_adc != after_global_adc:
        raise RuntimeError("CRITICAL: Global ~/.config/gcloud/application_default_credentials.json metadata changed unexpectedly!")
    print(
        json.dumps(
            {
                "global_main_adc_untouched": True,
                "isolated_account_ready": True,
                "no_gpu_created": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
