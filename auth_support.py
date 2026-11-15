"""Muse-local executable and credential *metadata* guards (never reads OAuth secrets)."""
from __future__ import annotations
import os, shutil, stat, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VALID_COLAB_AUTH_MODES = ('adc', 'oauth2')


def default_colab_auth() -> str:
    """Resolve default Colab CLI auth mode from COLAB_AUTH_MODE ('adc' when unset)."""
    mode = (os.environ.get('COLAB_AUTH_MODE') or 'adc').strip().lower()
    if mode not in VALID_COLAB_AUTH_MODES:
        raise ValueError(f'Unsupported COLAB_AUTH_MODE {mode!r}; expected one of {VALID_COLAB_AUTH_MODES}')
    return mode


def global_gcloud_adc_path() -> Path:
    return Path.home() / '.config/gcloud/application_default_credentials.json'


def ensure_adc_isolation() -> None:
    """Fail-closed anti-cross-account guard when H3_REQUIRE_ISOLATED_ADC=1.

    Protects the host's global ~/.config/gcloud/application_default_credentials.json
    (Project 1 main account) from ever being read or used by colab-workstation
    (Project 3 isolated Colab account).
    """
    if os.environ.get('H3_REQUIRE_ISOLATED_ADC', '').strip() != '1':
        return
    raw = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '').strip()
    if not raw:
        raise RuntimeError(
            'BLOCKED: H3_REQUIRE_ISOLATED_ADC=1 requires GOOGLE_APPLICATION_CREDENTIALS '
            'to point to an isolated project credential file (refusing global ADC fallback)'
        )
    candidate = Path(raw).expanduser().resolve()
    global_adc = global_gcloud_adc_path().resolve()
    if candidate == global_adc:
        raise RuntimeError(
            'BLOCKED: GOOGLE_APPLICATION_CREDENTIALS points to global '
            '~/.config/gcloud/application_default_credentials.json; refusing cross-account collision'
        )


def resolve_adc_credential_path() -> Path:
    """Return the active ADC / authorized-user credential path without reading its contents."""
    ensure_adc_isolation()
    raw = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '').strip()
    if raw:
        return Path(raw).expanduser()
    return global_gcloud_adc_path()


def resolve_colab_executable() -> str:
    """Prefer the active project's venv, then the explicit override and PATH.

    Never guess a macOS installation path; supports `python script.py` and
    `.venv/bin/python script.py` launching in a minimal Muse shell.
    """
    override = os.environ.get('COLAB_CLI_PATH')
    candidates = ([Path(override).expanduser()] if override else []) + [
        Path(sys.executable).parent / 'colab',
        ROOT / '.venv-colab-cli' / 'bin' / 'colab',
        ROOT / '.venv' / 'bin' / 'colab',
    ]
    path = shutil.which('colab')
    if path:
        candidates.append(Path(path))
    candidates.append(Path.home() / '.local/bin/colab')
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    raise FileNotFoundError('Colab CLI missing; activate project venv or install official google-colab-cli')


def audit_user_adc_file() -> dict:
    """Metadata-only: no token JSON, stdout of gcloud, or secret is read."""
    p = resolve_adc_credential_path()
    isolated_required = os.environ.get('H3_REQUIRE_ISOLATED_ADC', '').strip() == '1'
    if not p.exists():
        if isolated_required:
            raise FileNotFoundError(
                f'Isolated Drive credential file missing at {p}; '
                'run scripts/setup_isolated_account.py in Terminal.app'
            )
        return {'present': False, 'path': str(p), 'credentials_read': False}
    if p.is_symlink() or not p.is_file():
        raise RuntimeError('ADC path is symlink/not regular file')
    mode = stat.S_IMODE(p.stat().st_mode)
    if mode & 0o077:
        raise RuntimeError('ADC credential permissions are not private (expected mode 0600)')
    return {
        'present': True,
        'mode': oct(mode),
        'isolated': bool(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '').strip()),
        'credentials_read': False,
    }


def audit_colab_oauth_token_file() -> dict:
    """Metadata-only check for colab-cli's first-party OAuth2 token (~/.config/colab-cli/token.json)."""
    p = Path.home() / '.config/colab-cli/token.json'
    if not p.exists():
        raise FileNotFoundError(
            f'Colab OAuth2 token missing at {p}; '
            'run scripts/setup_isolated_account.py in Terminal.app'
        )
    if p.is_symlink() or not p.is_file():
        raise RuntimeError('Colab OAuth2 token path is symlink/not regular file')
    mode = stat.S_IMODE(p.stat().st_mode)
    if mode & 0o077:
        raise RuntimeError('Colab OAuth2 token permissions are not private (expected mode 0600)')
    return {'present': True, 'mode': oct(mode), 'credentials_read': False}
