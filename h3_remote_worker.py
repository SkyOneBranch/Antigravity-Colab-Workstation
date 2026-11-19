"""Remote Colab single-variant H3 execution. Invoked by `colab exec -f`.
Requires a previously authorized/mounted Drive (mount path) or a
delivered short-lived Drive token plus a drive_download task block
(noninteractive download path, TASK-006 P0-B), and a server-verified
cache manifest.
Must NOT be considered production-verified until ComfyUI API workflow is exported/validated.
"""
from __future__ import annotations
import base64,hashlib,inspect,json,os,re,shlex,shutil,subprocess,sys,tarfile,tempfile,threading,time,urllib.request,urllib.error,uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT=Path('/content/muse_h3')
sys.path.insert(0,str(ROOT))
from h3_contract import MODEL_FILES,validate_model_cache_manifest,sha256_file,patch_api_workflow_for_spec,validate_ref_image,WIDTH,HEIGHT,ANIMATION_FRAMES,FPS,TOTAL_FRAMES
from h3_job_state import WORKER_PHASE_TO_JOB_STAGE, write_job_state
from assemble import run as assemble_video,ffprobe

# ---- v0.2.0 / v0.2.1 Fast Cold-Start constants (F1/F2/F3/F4/F5) ----
DEFAULT_NVME_MODELS_DIR = Path('/content/nvme_models/h3')
HEADLESS_EXCLUDED_PACKAGES = ("comfyui-workflow-templates", "comfyui-embedded-docs")
HEADLESS_EXCLUDED_PREFIXES = ("comfyui-workflow-templates-",)
DEFAULT_EXCLUDED_PACKAGES = (
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "comfyui-workflow-templates",
    "comfyui-embedded-docs",
)
DEFAULT_EXCLUDED_PREFIXES = ("nvidia-", "comfyui-workflow-templates-")
HIGHVRAM_THRESHOLD_MIB = 76800
HIGHVRAM_DEFAULT_FLAGS = ("--highvram", "--disable-mmap")
DEFAULT_MODEL_FILE_CONCURRENCY = 2
MAX_MODEL_FILE_CONCURRENCY = 4
COLDSTART_METRICS_SCHEMA_VERSION = 1


class WorkerAbortError(RuntimeError):
    """Raised when a sibling dual-track stage fails and triggers abort_event."""


def _spec_from_task(task):
    """Resolve the TaskSpec for this worker invocation (TASK-007).

    The controller always writes task['task_spec'] (a TaskSpec dict).
    A legacy task.json with only task['variant'] in A/B falls back to
    the built-in fixture adapter bound to the uploaded poster.jpg.
    Anything else is a hard failure: the worker never guesses a prompt
    or a frame count.
    """
    from h3_task_spec import TaskSpec, task_spec_for_variant

    spec = task.get('task_spec')
    if isinstance(spec, dict):
        return TaskSpec.from_dict(spec)
    version = task.get('variant')
    if version in ('A', 'B'):
        return task_spec_for_variant(version, task.get('production_task_id', 'legacy'), ROOT / 'poster.jpg')
    raise ValueError('task.json carries neither task_spec nor a known variant')

# ---- Worker stage tracking (user requirement 2026-11-11 09:12) ----
# Every phase prints a WORKER_STAGE marker, so the controller's persisted
# exec output always shows the last successful stage. On any failure the
# worker prints one structured WORKER_FAILED_JSON line with the last stage
# and the exception, then re-raises (the controller stops the GPU in its
# finally block). This is observability only: it does not claim the
# 2026-11-11 worker failure itself is fixed.
#
# TASK-005 §4.2 (Issue #27): every phase ALSO writes the durable job
# state file (MUSE_H3_JOB_STATE.json, atomic write) on the VM, so the
# controller can poll the worker with short exec calls instead of holding
# one long blocking exec. The state write is best-effort and never raises:
# a failed bookkeeping write must not kill an inference. The state file
# says where the worker is; the MP4 is still independently ffprobe-verified
# before any success is reported.
_LAST_STAGE = "init"
_JOB_RUN_ID = ""
_JOB_VARIANT = ""
# Token id of the last renewal request sent to the controller, so a
# long download asks once per token instead of on every chunk.
_LAST_REFRESH_REQUEST_TOKEN_ID = None
# Lock protecting shared worker state & MUSE_H3_JOB_STATE.json writes
# across F3 dual-track threads (ensure_comfy + restore_weights).
_STATE_LOCK = threading.Lock()
_CMD_FAILURE_LOCAL = threading.local()


def _job_stage_for(name: str) -> str:
    mapped = WORKER_PHASE_TO_JOB_STAGE.get(name)
    if mapped is not None:
        return mapped
    if name in ("dual_track_setup", "link_models"):
        return "RESTORING_MODELS"
    return name


def _write_job_state_safe(stage: str, **kwargs) -> None:
    if not (_JOB_RUN_ID and _JOB_VARIANT):
        return
    with _STATE_LOCK:
        try:
            write_job_state(ROOT, _JOB_RUN_ID, _JOB_VARIANT, stage, **kwargs)
        except Exception:
            pass


def _request_token_refresh(provider, scope="drive.readonly") -> None:
    """Ask the controller (through the durable job state) for a fresh
    short-lived Drive token. Best-effort, thread-safe, and throttled:
    one request per token id. The controller uploads a new 0600 token
    file; the provider picks it up on its next get_token() call. Never
    carries a token value, only the token id."""
    global _LAST_REFRESH_REQUEST_TOKEN_ID
    try:
        token_id = provider.token_id if provider is not None else None
        with _STATE_LOCK:
            if token_id is not None and token_id == _LAST_REFRESH_REQUEST_TOKEN_ID:
                return
            _LAST_REFRESH_REQUEST_TOKEN_ID = token_id
            stage = _job_stage_for(_LAST_STAGE)
            if _JOB_RUN_ID and _JOB_VARIANT:
                write_job_state(ROOT, _JOB_RUN_ID, _JOB_VARIANT, stage,
                                extra={"token_refresh_requested": {
                                    "token_id": token_id,
                                    "scope": scope,
                                    "requested_at_utc": time.strftime(
                                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}})
                print("TOKEN_REFRESH_REQUESTED", token_id, scope, flush=True)
    except Exception:
        pass


def set_stage(name: str) -> None:
    global _LAST_STAGE
    with _STATE_LOCK:
        _LAST_STAGE = name
        print("WORKER_STAGE", name, flush=True)
        if _JOB_RUN_ID and _JOB_VARIANT:
            try:
                write_job_state(ROOT, _JOB_RUN_ID, _JOB_VARIANT,
                                _job_stage_for(name))
            except Exception:
                pass


def _sanitize_text(text):
    """Redact any embedded user:password in a URL before it reaches a log.
    Git commands here use a public URL and carry no credentials, but the
    rule is applied uniformly so a secret can never leak via a failure
    record."""
    if not text:
        return text
    return re.sub(r'(://[^/\s:]+:)[^/\s@]+@', r'\1***@', text)


# Detail of the most recent run_cmd failure, if any. Cleared at the start
# of every run_cmd call, so it only ever describes the failure that is
# currently propagating. worker_failure_info merges it into the structured
# WORKER_FAILED_JSON line, so a Git failure is recorded with its sanitized
# stdout/stderr, exit code, command stage and exception type in one place.
_LAST_CMD_FAILURE = None


def worker_failure_info(stage: str, exc: BaseException, cmd_failure=None) -> dict:
    info = {"stage": stage, "error_type": type(exc).__name__, "error": str(exc)[:1500]}
    detail = cmd_failure if cmd_failure is not None else getattr(_CMD_FAILURE_LOCAL, "value", None)
    if detail is None:
        detail = _LAST_CMD_FAILURE
    if detail:
        info["cmd_failure"] = detail
    return info

def require_inside(path:Path, root:Path):
    if not path.resolve().is_relative_to(root.resolve()):raise ValueError('unsafe path')

# 2026-11-11 (item 8): Drive -> Colab 40GB restore capacity detection.
# The four weights total 40,073,842,159 bytes. The restore copies them from
# the mounted Drive cache to local NVMe before inference, so a short disk or
# a slow copy can burn the whole session budget. Two gates, both fail-closed:
#  (a) disk gate before any copy (free >= total + 5 GiB reserve);
#  (b) throughput/cost projection after the first 256MB copied, using the
#      measured MB/s and the controller-supplied live rate + session budget.
# No throughput has ever been measured (the mount never succeeded), so the
# first real run measures it here; the local preflight only reports the
# requirement at an assumed 50-150 MB/s range, labelled as an assumption.
RESTORE_RESERVE_BYTES = 5 * 1024 ** 3

def estimate_restore(total_bytes, free_bytes=None, measured_mbps=None, rate_cu_h=None, budget_cu=None, reserve_bytes=RESTORE_RESERVE_BYTES):
    """Pure estimator. free_bytes=None skips the disk gate; measured_mbps=None
    skips the time/cost projection. Returns a dict with ok + reason."""
    total = int(total_bytes)
    required = total + int(reserve_bytes)
    result = {"total_bytes": total, "required_bytes": required, "free_bytes": free_bytes,
              "measured_mbps": measured_mbps, "eta_sec": None, "estimated_cost_cu": None,
              "ok": True, "reason": None}
    if free_bytes is not None and int(free_bytes) < required:
        result["ok"] = False
        result["reason"] = (f"insufficient disk for 40GB restore: need {required} bytes "
                            f"(models {total} + reserve {reserve_bytes}), have {int(free_bytes)} "
                            f"(shortfall {required - int(free_bytes)})")
        return result
    if measured_mbps:
        eta = total / (float(measured_mbps) * 1_000_000)
        result["eta_sec"] = round(eta, 1)
        if rate_cu_h:
            cost = eta / 3600.0 * float(rate_cu_h)
            result["estimated_cost_cu"] = round(cost, 3)
            if budget_cu is not None and cost > float(budget_cu):
                result["ok"] = False
                result["reason"] = (f"restore cost {cost:.2f} CU exceeds session budget "
                                    f"{float(budget_cu):.2f} CU at {float(measured_mbps):.1f} MB/s")
    return result

def check_restore_capacity(comfy, total_bytes, rate_cu_h=None, budget_cu=None, target_dir=None):
    """Remote disk gate, run before the first copy. Raises on failure."""
    models = Path(target_dir) if target_dir is not None else (Path(comfy) / "models")
    models.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(models).free
    est = estimate_restore(total_bytes, free, rate_cu_h=rate_cu_h, budget_cu=budget_cu)
    print("RESTORE_CAPACITY", json.dumps(est), flush=True)
    if not est["ok"]:
        raise RuntimeError("Restore capacity check failed: " + str(est["reason"]))
    return est

def project_restore_throughput(total_bytes, copied_bytes, elapsed_sec, rate_cu_h=None, budget_cu=None):
    """Throughput/cost projection from the first measured chunk. Raises on failure."""
    mbps = (int(copied_bytes) / 1_000_000) / elapsed_sec if elapsed_sec and elapsed_sec > 0 else None
    est = estimate_restore(total_bytes, None, measured_mbps=mbps, rate_cu_h=rate_cu_h, budget_cu=budget_cu)
    print("RESTORE_PROJECTION", json.dumps(est), flush=True)
    if not est["ok"]:
        raise RuntimeError("Restore projection blocked: " + str(est["reason"]))
    return est

def run_cmd(args,timeout=1200,cwd=None):
    """Run a shell command, capturing its output. On failure the captured
    stdout/stderr are printed BEFORE re-raising: 2026-11-12 the git clone
    of ComfyUI failed with exit 128 and git's own fatal message was lost
    (inherited stdio was not captured by the exec channel), leaving only
    the exit code to diagnose from. The failure detail is also stashed in
    _CMD_FAILURE_LOCAL / _LAST_CMD_FAILURE so worker_failure_info carries
    it in the structured WORKER_FAILED_JSON line (sanitized, truncated)."""
    global _LAST_CMD_FAILURE
    _CMD_FAILURE_LOCAL.value = None
    _LAST_CMD_FAILURE = None
    print('STAGE_CMD',args[0],flush=True)
    try:
        proc = subprocess.run(args,cwd=cwd,check=True,timeout=timeout,
                              stdin=subprocess.DEVNULL,capture_output=True,text=True)
        if proc.stdout:
            print(proc.stdout,flush=True)
        if proc.stderr:
            print(proc.stderr,flush=True)
        return proc
    except subprocess.CalledProcessError as e:
        detail = {
            "cmd": _sanitize_text(" ".join(str(a) for a in args)),
            "returncode": e.returncode,
            "stdout": _sanitize_text(e.stdout or "")[-2000:],
            "stderr": _sanitize_text(e.stderr or "")[-2000:],
        }
        _CMD_FAILURE_LOCAL.value = detail
        _LAST_CMD_FAILURE = detail
        print('CMD_FAILED',args[0],'exit',e.returncode,flush=True)
        if e.stdout:
            print(_sanitize_text(e.stdout),flush=True)
        if e.stderr:
            print(_sanitize_text(e.stderr),flush=True)
        raise


def run_cmd_with_retry(args,timeout=1200,cwd=None,attempts=2,delay_sec=20):
    """Bounded retry for one idempotent shell command (git clone/fetch).
    A single transient network failure must not kill a whole GPU session.
    Raises the last CalledProcessError after the final attempt."""
    last = None
    for attempt in range(1, attempts + 1):
        try:
            return run_cmd(args,timeout=timeout,cwd=cwd)
        except subprocess.CalledProcessError as e:
            last = e
            if attempt < attempts:
                print('CMD_RETRY',args[0],'attempt',attempt,'failed; retrying in',delay_sec,'s',flush=True)
                time.sleep(delay_sec)
    raise last

def _spec_satisfied(version: str, spec: str) -> bool:
    """Minimal PEP 440 subset (==, >=, <=, >, <, ~=, comma-separated).
    Returns False on anything unparsable: the caller treats that as
    'not satisfied' and pip runs, which is the previous behavior."""
    def _parts(v):
        out = []
        for piece in str(v).strip().split('.'):
            digits = re.match(r'^(\d+)', piece)
            if digits is None:
                return None
            out.append(int(digits.group(1)))
        return out
    ver = _parts(version)
    if ver is None:
        return False
    for clause in spec.split(','):
        clause = clause.strip()
        m = re.match(r'^(==|>=|<=|>|<|~=)\s*([0-9][0-9A-Za-z.]*)$', clause)
        if not m:
            return False
        op, want = m.group(1), _parts(m.group(2))
        if want is None:
            return False
        want_raw = list(want)
        want = want + [0] * (len(ver) - len(want)) if len(want) < len(ver) else want
        ver_p = ver + [0] * (len(want) - len(ver)) if len(ver) < len(want) else ver
        if op == '==' and not ver_p == want:
            return False
        if op == '>=' and not ver_p >= want:
            return False
        if op == '<=' and not ver_p <= want:
            return False
        if op == '>' and not ver_p > want:
            return False
        if op == '<' and not ver_p < want:
            return False
        if op == '~=':
            # ~= X.Y means >= X.Y and == X.* — the prefix is all but the
            # last component of the ORIGINAL spec, before padding.
            if not ver_p >= want:
                return False
            prefix = want_raw[:-1]
            if ver_p[:len(prefix)] != prefix:
                return False
    return True


def log_requirements_status(
    requirements_path,
    excluded_packages=HEADLESS_EXCLUDED_PACKAGES,
    excluded_prefixes=HEADLESS_EXCLUDED_PREFIXES,
):
    """Best-effort, no-network diagnostic: which requirements of the pinned
    ComfyUI tree are already satisfied in this runtime. Never blocks the
    install; on any doubt the entry is 'unknown' and pip runs as before.
    Headless-optional packages (verified optional in ComfyUI API mode) are
    recorded under 'excluded' rather than 'missing'.
    Returns a dict {satisfied, missing, unknown, excluded}."""
    from importlib import metadata as _md
    satisfied, missing, unknown, excluded = [], [], [], []
    excl_exact = {normalize_pkg_name(p) for p in (excluded_packages or ())}
    excl_pref = tuple(normalize_pkg_name(p) for p in (excluded_prefixes or ()) if str(p).strip())
    try:
        for line in Path(requirements_path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            head = line.split('#', 1)[0].split(';', 1)[0].strip()
            m_name = re.match(r'^([A-Za-z0-9][A-Za-z0-9_.\-]*)', head)
            if m_name:
                norm = normalize_pkg_name(m_name.group(1))
                if norm in excl_exact or any(norm.startswith(pref) for pref in excl_pref):
                    excluded.append(line)
                    continue
            m = re.match(r'^([A-Za-z0-9_.\-]+)\s*(.*)$', line)
            if not m:
                unknown.append(line)
                continue
            name, spec = m.group(1), m.group(2).strip()
            try:
                installed = _md.version(name)
            except Exception:
                missing.append(line)
                continue
            if not spec or _spec_satisfied(installed, spec):
                satisfied.append(line)
            else:
                missing.append(line + ' (installed ' + installed + ')')
    except Exception as e:
        print('REQUIREMENTS_CHECK_UNAVAILABLE', type(e).__name__, str(e)[:200], flush=True)
        return {"satisfied": [], "missing": [], "unknown": ["check failed"], "excluded": []}
    status = {"satisfied": satisfied, "missing": missing, "unknown": unknown, "excluded": excluded}
    print('REQUIREMENTS_STATUS', json.dumps(status, ensure_ascii=False), flush=True)
    return status


def pip_install_needed(
    requirements_path,
    excluded_packages=HEADLESS_EXCLUDED_PACKAGES,
    excluded_prefixes=HEADLESS_EXCLUDED_PREFIXES,
):
    """True unless the requirements check ran cleanly (no unknowns) and
    nothing is missing (excluding headless-optional packages). Skipping pip
    then is equivalent to running it: pip would install nothing needed.
    Any doubt -> run pip as before."""
    status = log_requirements_status(
        requirements_path,
        excluded_packages=excluded_packages,
        excluded_prefixes=excluded_prefixes,
    )
    return bool(status["missing"] or status["unknown"])


def verify_comfyui_commit(comfy: Path, git_ref: str) -> bool:
    """True iff <comfy> is a git checkout whose HEAD is exactly git_ref."""
    try:
        proc = subprocess.run(['git', '-C', str(comfy), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, timeout=60)
    except Exception:
        return False
    return proc.returncode == 0 and proc.stdout.strip() == git_ref


def comfyui_commit(comfy: Path):
    """Commit id of an installed ComfyUI tree: from .git when present,
    else from the COMFYUI_COMMIT.txt marker written into a cache-restored
    tree (git archive carries no .git). None when neither is available."""
    if (comfy / '.git').exists():
        proc = subprocess.run(['git', '-C', str(comfy), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            return proc.stdout.strip()
    marker = comfy / 'COMFYUI_COMMIT.txt'
    if marker.is_file():
        return marker.read_text().strip()
    return None


def clone_comfyui_atomic(comfy: Path, git_ref: str, repo_url='https://github.com/Comfy-Org/ComfyUI.git',
                         attempts=2, delay_sec=20):
    """Clone the pinned ComfyUI commit with an atomic install.

    2026-11-12 root-cause follow-up: the previous retry cloned straight
    into /content/ComfyUI, but git leaves a partial target directory
    behind when a clone fails, and a second `git clone` into a non-empty
    directory fails at once (exit 128) — so the retry could never work.
    Each attempt now clones into a fresh temp sibling directory, verifies
    the commit, and only then renames it into place (same filesystem, so
    the rename is atomic). A failed attempt's temp dir is removed.
    Returns 'present' when the target is already the pinned commit,
    'cloned' on a fresh install."""
    if (comfy / '.git').exists() and verify_comfyui_commit(comfy, git_ref):
        print('COMFYUI_ALREADY_PRESENT', git_ref, flush=True)
        return 'present'
    if not git_ref:
        raise RuntimeError('No pinned ComfyUI commit/ref in manifest; refuse moving main')
    parent = comfy.parent
    last_exc = None
    for attempt in range(1, attempts + 1):
        tmp = Path(tempfile.mkdtemp(prefix='.comfyui-install-', dir=str(parent)))
        try:
            run_cmd(['git', 'clone', repo_url, str(tmp)], timeout=600)
            run_cmd(['git', '-C', str(tmp), 'fetch', '--depth', '1', 'origin', git_ref], timeout=600)
            run_cmd(['git', '-C', str(tmp), 'checkout', '--detach', 'FETCH_HEAD'], timeout=120)
            if not verify_comfyui_commit(tmp, git_ref):
                raise RuntimeError('cloned ComfyUI HEAD does not match pinned commit ' + git_ref)
            if comfy.exists():
                shutil.rmtree(comfy, ignore_errors=True)  # partial/stale install
            os.rename(tmp, comfy)
            print('COMFYUI_CLONED', git_ref, flush=True)
            return 'cloned'
        except Exception as e:
            last_exc = e
            shutil.rmtree(tmp, ignore_errors=True)
            if attempt < attempts:
                print('COMFYUI_CLONE_RETRY', attempt, 'failed; retrying in', delay_sec, 's', flush=True)
                time.sleep(delay_sec)
    raise last_exc


def restore_comfyui_from_cache(cache_dir, comfy: Path, expected_sha256: str,
                               expected_commit: str, archive_name=None):
    """Restore the pinned ComfyUI source from the Drive source cache.
    Verifies the archive SHA-256, then the embedded commit id, then that
    main.py/requirements.txt exist; installs atomically. Raises
    RuntimeError with the reason on any mismatch — the caller logs
    COMFYUI_CACHE_FAILED and falls back to git clone. Returns 'restored'."""
    if not cache_dir:
        raise RuntimeError('no cache dir configured')
    archive_name = archive_name or ('comfyui-' + expected_commit + '.tar.gz')
    archive = Path(cache_dir) / archive_name
    if not archive.is_file():
        raise RuntimeError('cache archive missing: ' + archive_name)
    digest = hashlib.sha256()
    with archive.open('rb') as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError('cache archive SHA-256 mismatch')
    parent = comfy.parent
    tmp = Path(tempfile.mkdtemp(prefix='.comfyui-cache-', dir=str(parent)))
    try:
        with tarfile.open(archive, 'r:gz') as tf:
            tf.extractall(tmp, filter='data')
        extracted = tmp / 'comfyui'
        marker = extracted / 'COMFYUI_COMMIT.txt'
        if not marker.is_file():
            raise RuntimeError('cache archive has no COMFYUI_COMMIT.txt')
        if marker.read_text().strip() != expected_commit:
            raise RuntimeError('cache archive commit mismatch')
        if not (extracted / 'main.py').is_file() or not (extracted / 'requirements.txt').is_file():
            raise RuntimeError('cache archive missing main.py/requirements.txt')
        if comfy.exists():
            shutil.rmtree(comfy, ignore_errors=True)  # partial/stale install
        os.rename(extracted, comfy)
        print('COMFYUI_CACHE_RESTORED', expected_commit, flush=True)
        return 'restored'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def get_json(url,timeout=10):
    with urllib.request.urlopen(url,timeout=timeout) as r:return json.loads(r.read().decode())

# Transient local ComfyUI API failures (TASK-008): a single slow
# /history read, a dropped connection, or a 502/503/504 from the local
# server is NOT proof the inference died. Only these are retried; any
# other HTTP status or a malformed response is a real failure.
HISTORY_TRANSIENT_MAX_CONSECUTIVE = 8
HISTORY_POLL_TIMEOUT_SEC = 30


def _is_transient_api_error(exc) -> bool:
    """True for a transient local ComfyUI API failure: timeouts,
    connection errors, HTTP 502/503/504 (raw, or as re-raised by
    post_json as 'POST ... failed HTTP 502/503/504')."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in (502, 503, 504)
    if isinstance(exc, RuntimeError):
        text = str(exc)
        return any(f"failed HTTP {code}" in text for code in (502, 503, 504))
    return isinstance(exc, (TimeoutError, urllib.error.URLError, ConnectionError))


def _find_prompt_by_client_id(server, client_id, timeout=10):
    """Look for a prompt submitted under client_id in /queue (running +
    pending) and /history. Returns the prompt_id, or None when the
    submit never landed. A lookup failure returns None — the caller
    treats 'not found' as 'not submitted', never as proof of failure."""
    try:
        queue = get_json(server + '/queue', timeout=timeout)
    except Exception:
        queue = {}
    for key in ('queue_running', 'queue_pending'):
        for item in (queue or {}).get(key) or []:
            try:
                if len(item) >= 4 and isinstance(item[3], dict) \
                        and item[3].get('client_id') == client_id:
                    return item[1]
            except Exception:
                continue
    try:
        history = get_json(server + '/history', timeout=timeout)
    except Exception:
        return None
    for pid, event in (history or {}).items():
        try:
            prompt = (event or {}).get('prompt') or []
            extra = prompt[3] if len(prompt) >= 4 else {}
            if isinstance(extra, dict) and extra.get('client_id') == client_id:
                return pid
        except Exception:
            continue
    return None


def _submit_prompt(server, api, client_id):
    """Submit once with a fixed client_id; on a read timeout the request
    may already be queued, so check /queue + /history for this client_id
    before any resubmit. At most one checked resubmit; a second timeout
    fails closed naming the client_id so the operator can see whether
    the prompt landed. A deterministic rejection (node_errors / no
    prompt_id) raises immediately — a rejected workflow never queues."""
    payload = {'prompt': api, 'client_id': client_id}
    for attempt in (1, 2):
        try:
            result = post_json(server + '/prompt', payload)
        except Exception as e:  # noqa: BLE001 - transient vs fatal split below
            if not _is_transient_api_error(e):
                raise
            found = _find_prompt_by_client_id(server, client_id)
            if found:
                print('H3_PROMPT_RECOVERED', found,
                      'submit timed out but the prompt is already queued/running', flush=True)
                return {'prompt_id': found}
            if attempt == 2:
                raise RuntimeError(
                    f'ComfyUI /prompt submit timed out twice and no queued/running '
                    f'prompt was found for client_id {client_id}; refusing a third '
                    f'submit. Last error: {type(e).__name__}: {str(e)[:300]}') from e
            print('H3_PROMPT_RESUBMIT',
                  f'submit attempt {attempt} timed out and no queued prompt was found '
                  f'for this client_id; one checked resubmit', flush=True)
            continue
        if result.get('node_errors') or not result.get('prompt_id'):
            raise RuntimeError('ComfyUI rejected workflow: ' + json.dumps(result, ensure_ascii=False)[:1800])
        return result
    raise RuntimeError('unreachable: _submit_prompt loop exhausted')

def post_json(url,data,timeout=30):
    req=urllib.request.Request(url,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        # The server's rejection reason lives in the response body (e.g.
        # /prompt 400 validation errors). Without it the failure is
        # undiagnosable: 2026-11-12 a bare "HTTP Error 400: Bad Request"
        # cost a GPU session to chase. Surface the body.
        try:body=e.read().decode(errors='replace')
        except Exception:body=''
        raise RuntimeError(f'POST {url} failed HTTP {e.code}: {body[:2000]}') from e

def _current_python_tag() -> str:
    return f"py{sys.version_info.major}{sys.version_info.minor}"


def normalize_pkg_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", str(name).strip()).lower()


def filter_requirements_lines(
    lines,
    excluded_packages=DEFAULT_EXCLUDED_PACKAGES,
    excluded_prefixes=DEFAULT_EXCLUDED_PREFIXES,
) -> list[str]:
    """Filter requirements.txt lines in pure stdlib, excluding Colab-preinstalled
    heavy packages (torch/torchvision/torchaudio/triton, nvidia-*) so offline
    wheelhouse installs do not fail on missing preinstalled wheels."""
    excl_exact = {normalize_pkg_name(p) for p in (excluded_packages or ())}
    excl_pref = tuple(normalize_pkg_name(p) for p in (excluded_prefixes or ()) if str(p).strip())
    kept: list[str] = []
    for raw in lines:
        line = str(raw).strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            kept.append(line)
            continue
        # Strip inline comments and environment markers for name extraction.
        head = line.split("#", 1)[0].split(";", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9_.\-]*)", head)
        if m:
            norm = normalize_pkg_name(m.group(1))
            if norm in excl_exact or any(norm.startswith(pref) for pref in excl_pref):
                continue
        kept.append(line)
    return kept


def filter_requirements_file(
    src: Path,
    dst: Path,
    excluded_packages=DEFAULT_EXCLUDED_PACKAGES,
    excluded_prefixes=DEFAULT_EXCLUDED_PREFIXES,
) -> dict:
    """Write a filtered requirements file excluding preinstalled packages."""
    raw_lines = Path(src).read_text(encoding="utf-8").splitlines()
    kept = filter_requirements_lines(
        raw_lines,
        excluded_packages=excluded_packages,
        excluded_prefixes=excluded_prefixes,
    )
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return {"kept_count": len(kept), "path": str(dst)}


def _emit_wheelhouse_fallback(reason: str, python_tag: str) -> None:
    payload = {"reason": str(reason)[:500], "python_tag": python_tag}
    print("WHEELHOUSE_FALLBACK_PYPI:" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def _resolve_wheelhouse_entry(drive_download, cache, expected_commit: str, python_tag: str):
    """Resolve a matching wheelhouse descriptor from drive_download or cache.
    Returns (entry_dict | None, source_kind | None, fallback_reason | None)."""
    def _pick(container, kind):
        if not isinstance(container, dict):
            return None, None, None
        wheelhouses = container.get("wheelhouses")
        single = container.get("wheelhouse")
        entry = None
        if isinstance(wheelhouses, dict) and wheelhouses:
            entry = wheelhouses.get(python_tag)
            if entry is None:
                return None, None, f"python_tag_mismatch:{python_tag}"
        elif isinstance(single, dict) and single:
            entry_tag = single.get("python_tag")
            if entry_tag and str(entry_tag) != python_tag:
                return None, None, f"python_tag_mismatch:{python_tag}!=manifest:{entry_tag}"
            entry = single
        else:
            return None, None, None
        if not isinstance(entry, dict):
            return None, None, "invalid_wheelhouse_entry"
        entry_commit = entry.get("commit")
        if entry_commit and expected_commit and str(entry_commit) != str(expected_commit):
            return None, None, "commit_mismatch"
        archive = entry.get("archive") or entry.get("archive_name") or entry.get("filename")
        sha256 = entry.get("sha256")
        if kind == "drive_download":
            size = entry.get("size") if entry.get("size") is not None else entry.get("size_bytes")
            if not (entry.get("file_id") and archive and sha256 and size is not None):
                return None, None, "incomplete_wheelhouse_descriptor"
        else:
            if not (container.get("dir") and archive and sha256):
                return None, None, "incomplete_wheelhouse_cache_descriptor"
        return entry, kind, None

    entry, kind, reason = _pick(drive_download, "drive_download")
    if entry is not None or reason is not None:
        return entry, kind, reason
    entry, kind, reason = _pick(cache, "cache")
    if entry is not None or reason is not None:
        return entry, kind, reason
    return None, None, "wheelhouse_missing"


def extract_wheelhouse_archive(
    archive_path: Path,
    expected_sha256: str,
    dest_dir: Path,
    expected_commit: str | None = None,
    expected_python_tag: str | None = None,
) -> Path:
    """Verify SHA-256 of a wheelhouse tar.gz and atomically extract it.
    Returns the directory containing the extracted .whl files."""
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise RuntimeError("wheelhouse archive missing: " + str(archive_path.name))
    digest = hashlib.sha256()
    with archive_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest().lower() != str(expected_sha256).lower():
        raise RuntimeError("wheelhouse archive SHA-256 mismatch")
    dest_dir = Path(dest_dir)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".wheelhouse-", dir=str(dest_dir.parent)))
    try:
        with tarfile.open(archive_path, "r:gz") as tf:
            try:
                tf.extractall(tmp, filter="data")
            except TypeError:
                tf.extractall(tmp)
        whl_dir = tmp / "wheelhouse" if (tmp / "wheelhouse").is_dir() else tmp
        meta_path = whl_dir / "WHEELHOUSE_META.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if expected_commit and meta.get("commit") and str(meta["commit"]) != str(expected_commit):
                raise RuntimeError("wheelhouse metadata commit mismatch")
            if expected_python_tag and meta.get("python_tag") and str(meta["python_tag"]) != str(expected_python_tag):
                raise RuntimeError("wheelhouse metadata python_tag mismatch")
        if not list(whl_dir.glob("*.whl")):
            raise RuntimeError("wheelhouse archive contains no .whl files")
        if dest_dir.exists():
            shutil.rmtree(dest_dir, ignore_errors=True)
        os.rename(whl_dir, dest_dir)
        return dest_dir
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _construct_downloader(downloader_cls, token_provider, abort_event=None):
    if abort_event is not None:
        try:
            sig = inspect.signature(downloader_cls.__init__)
            params = sig.parameters
            if "abort_event" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                return downloader_cls(token_provider, abort_event=abort_event)
        except (TypeError, ValueError):
            pass
    return downloader_cls(token_provider)


def _download_file_with_abort(downloader, file_id, size, sha256, dest, abort_event=None, **kwargs):
    if abort_event is not None and abort_event.is_set():
        raise WorkerAbortError("download aborted before start")
    call_kwargs = dict(kwargs)
    if abort_event is not None:
        try:
            sig = inspect.signature(downloader.download_file)
            params = sig.parameters
            if "abort_event" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                call_kwargs["abort_event"] = abort_event
        except (TypeError, ValueError):
            pass
    return downloader.download_file(file_id, size, sha256, dest, **call_kwargs)


def _check_abort(abort_event, stage_name: str) -> None:
    if abort_event is not None and abort_event.is_set():
        raise WorkerAbortError(f"{stage_name} aborted by sibling track failure")


def ensure_comfy(
    comfy: Path,
    git_ref: str,
    cache=None,
    drive_download=None,
    token_provider=None,
    abort_event=None,
    metrics: dict | None = None,
):
    """Install the pinned ComfyUI source, then its Python dependencies.

    Order: (1) already-installed pinned commit -> reuse; (2) Drive source
    cache (archive SHA-256 + embedded commit id verified, atomic install)
    -> restore; (3) GitHub clone (atomic, bounded retry) -> clone. A
    cache failure is logged as COMFYUI_CACHE_FAILED with its reason and
    falls through to the clone; the original error stays on the record.
    Dependency install: a no-network requirements check first — when
    every requirement is already satisfied, pip is skipped; otherwise
    prefer the offline Wheelhouse (F2) with --no-index --find-links,
    falling back to online PyPI with WHEELHOUSE_FALLBACK_PYPI on missing,
    mismatched, or failed offline install."""
    if not git_ref:
        raise RuntimeError('No pinned ComfyUI commit/ref in manifest; refuse moving main')
    _check_abort(abort_event, "ensure_comfy")
    src_start = time.monotonic()
    if (comfy/'.git').exists() and verify_comfyui_commit(comfy, git_ref):
        print('COMFYUI_ALREADY_PRESENT', git_ref, flush=True)
    else:
        restored = False
        if drive_download and drive_download.get('comfyui') and token_provider is not None:
            try:
                from h3_drive_download import DriveDownloader
                entry = drive_download['comfyui']
                dl_dir = ROOT / 'drive_downloads'
                downloader = _construct_downloader(DriveDownloader, token_provider, abort_event=abort_event)
                _download_file_with_abort(
                    downloader,
                    entry['file_id'], int(entry['size']), entry['sha256'],
                    dl_dir / entry['archive'],
                    abort_event=abort_event,
                    on_token_expiring=lambda tid, rem: _request_token_refresh(token_provider))
                _check_abort(abort_event, "ensure_comfy")
                restore_comfyui_from_cache(
                    dl_dir, comfy, entry['sha256'], entry['commit'],
                    entry['archive'])
                restored = True
            except Exception as e:
                if abort_event is not None and abort_event.is_set():
                    raise WorkerAbortError("ensure_comfy aborted during source download") from e
                print('COMFYUI_CACHE_FAILED', type(e).__name__, str(e)[:500], flush=True)
        if not restored and cache and cache.get('dir') and cache.get('sha256') and cache.get('commit'):
            _check_abort(abort_event, "ensure_comfy")
            try:
                restore_comfyui_from_cache(Path(cache['dir']), comfy,
                                           cache['sha256'], cache['commit'],
                                           cache.get('archive'))
                restored = True
            except Exception as e:
                # Keep the original error on the record; fall through to git.
                print('COMFYUI_CACHE_FAILED', type(e).__name__, str(e)[:500], flush=True)
        if not restored:
            _check_abort(abort_event, "ensure_comfy")
            clone_comfyui_atomic(comfy, git_ref)
    installed = comfyui_commit(comfy)
    if installed != git_ref:
        raise RuntimeError('installed ComfyUI commit ' + str(installed) + ' != pinned ' + git_ref)
    print('COMFY_REF', installed, flush=True)
    comfy_source_sec = round(time.monotonic() - src_start, 3)

    _check_abort(abort_event, "ensure_comfy")
    pip_start = time.monotonic()
    wheelhouse_used = False
    wheelhouse_fallback_reason = None
    requirements = comfy / 'requirements.txt'
    if pip_install_needed(requirements):
        pytag = _current_python_tag()
        entry, source_kind, reason = _resolve_wheelhouse_entry(drive_download, cache, git_ref, pytag)
        if entry is None:
            wheelhouse_fallback_reason = reason or "wheelhouse_missing"
            _emit_wheelhouse_fallback(wheelhouse_fallback_reason, pytag)
            _check_abort(abort_event, "ensure_comfy")
            run_cmd([sys.executable, '-m', 'pip', 'install', '-r', str(requirements)], timeout=1200)
        else:
            try:
                archive_name = entry.get('archive') or entry.get('archive_name') or entry.get('filename')
                if source_kind == "drive_download":
                    if token_provider is None:
                        raise RuntimeError("no token provider for wheelhouse download")
                    from h3_drive_download import DriveDownloader
                    dl_dir = ROOT / 'drive_downloads'
                    dl_dir.mkdir(parents=True, exist_ok=True)
                    archive_path = dl_dir / archive_name
                    size = int(entry['size'] if entry.get('size') is not None else entry['size_bytes'])
                    downloader = _construct_downloader(DriveDownloader, token_provider, abort_event=abort_event)
                    _download_file_with_abort(
                        downloader,
                        entry['file_id'], size, entry['sha256'], archive_path,
                        abort_event=abort_event,
                        expected_md5=entry.get('server_md5'),
                        on_token_expiring=lambda tid, rem: _request_token_refresh(token_provider))
                else:
                    archive_path = Path(cache['dir']) / archive_name
                _check_abort(abort_event, "ensure_comfy")
                whl_dest = ROOT / 'wheelhouse'
                whl_dir = extract_wheelhouse_archive(
                    archive_path, entry['sha256'], whl_dest,
                    expected_commit=git_ref, expected_python_tag=pytag)
                filtered_req = ROOT / 'wheelhouse_requirements.txt'
                excl_pkgs = tuple(dict.fromkeys(
                    list(entry.get('excluded_packages') or DEFAULT_EXCLUDED_PACKAGES)
                    + list(HEADLESS_EXCLUDED_PACKAGES)))
                excl_prefs = tuple(dict.fromkeys(
                    list(entry.get('excluded_prefixes') or DEFAULT_EXCLUDED_PREFIXES)
                    + list(HEADLESS_EXCLUDED_PREFIXES)))
                filt = filter_requirements_file(
                    requirements, filtered_req,
                    excluded_packages=excl_pkgs, excluded_prefixes=excl_prefs)
                _check_abort(abort_event, "ensure_comfy")
                if filt['kept_count'] > 0:
                    run_cmd(
                        [sys.executable, '-m', 'pip', 'install', '--no-index',
                         f'--find-links={whl_dir}', '-r', str(filtered_req)],
                        timeout=1200)
                wheelhouse_used = True
                print('WHEELHOUSE_INSTALL_OK', pytag, str(whl_dir), flush=True)
            except Exception as e:
                if abort_event is not None and abort_event.is_set():
                    raise WorkerAbortError("ensure_comfy aborted during wheelhouse install") from e
                wheelhouse_fallback_reason = f"{type(e).__name__}: {str(e)[:300]}"
                _emit_wheelhouse_fallback(wheelhouse_fallback_reason, pytag)
                _check_abort(abort_event, "ensure_comfy")
                run_cmd([sys.executable, '-m', 'pip', 'install', '-r', str(requirements)], timeout=1200)
    else:
        print('PIP_INSTALL_SKIPPED', 'all requirements already satisfied', flush=True)
    pip_install_sec = round(time.monotonic() - pip_start, 3)
    if metrics is not None:
        metrics["comfy_source_sec"] = comfy_source_sec
        metrics["pip_install_sec"] = pip_install_sec
        metrics["wheelhouse_used"] = wheelhouse_used
        metrics["wheelhouse_fallback_reason"] = wheelhouse_fallback_reason


def _resolve_model_file_concurrency(num_files: int) -> int:
    if num_files <= 1:
        return 1
    raw = os.environ.get("H3_MODEL_FILE_CONCURRENCY")
    if raw:
        try:
            val = int(raw)
        except ValueError:
            val = DEFAULT_MODEL_FILE_CONCURRENCY
    else:
        val = DEFAULT_MODEL_FILE_CONCURRENCY
    val = max(1, min(MAX_MODEL_FILE_CONCURRENCY, val))
    return min(val, num_files)


def _resolve_download_workers_count() -> int:
    try:
        import h3_drive_download as dd
        resolver = getattr(dd, "resolve_concurrent_workers", None)
        if callable(resolver):
            return int(resolver())
        return int(getattr(dd, "DEFAULT_CONCURRENT_WORKERS", 12))
    except Exception:
        return 12


def restore_weights(
    comfy: Path,
    cache: Path,
    manifest: dict,
    rate_cu_h=None,
    budget_cu=None,
    drive_download=None,
    token_provider=None,
    models_dir: Path | None = None,
    abort_event=None,
    metrics: dict | None = None,
):
    validate_model_cache_manifest(manifest)
    _check_abort(abort_event, "restore_weights")
    target_root = Path(models_dir) if models_dir is not None else (Path(comfy) / 'models')
    dl_start = time.monotonic()
    downloaded_bytes = 0
    # TASK-006 P0-B + v0.2.0 F1/F4: noninteractive download path. Each model
    # file is downloaded straight to target_root (NVMe independent dir when
    # models_dir is passed) using pwrite/streaming-hash with controlled
    # multi-file concurrency and shared abort_event.
    if drive_download is not None:
        if token_provider is None:
            raise RuntimeError('Drive download requested but no token provider supplied')
        from h3_drive_download import DriveDownloader
        total = int(sum(int(x['size']) for x in manifest['files']))
        check_restore_capacity(comfy, total, rate_cu_h=rate_cu_h, budget_cu=budget_cu, target_dir=target_root)
        files = drive_download.get('files') or []
        by_path = {entry['path']: entry for entry in files}
        pending = []
        for item in manifest['files']:
            relative = Path(item['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('unsafe path')
            dst = target_root / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            require_inside(dst.parent, target_root)
            if dst.exists() and dst.stat().st_size == item['size'] and sha256_file(dst) == item['sha256']:
                print('CACHE_REUSE', relative, flush=True)
                continue
            entry = by_path.get(item['path'])
            if not entry or not entry.get('file_id'):
                raise RuntimeError('Drive download entry missing for ' + str(relative))
            pending.append((item, relative, dst, entry))

        local_abort = abort_event if abort_event is not None else threading.Event()
        downloader = _construct_downloader(DriveDownloader, token_provider, abort_event=local_abort)
        projected = [False]
        progress_lock = threading.Lock()

        def _download_one(spec_tuple):
            item, relative, dst, entry = spec_tuple
            _check_abort(local_abort, "restore_weights")
            copy_start = time.monotonic()
            next_milestone = [256 * 1024 * 1024]
            _download_file_with_abort(
                downloader,
                entry['file_id'], int(item['size']), item['sha256'], dst,
                abort_event=local_abort,
                expected_md5=item.get('server_md5'),
                on_progress=lambda done, total_bytes, rel=relative, cs=copy_start, nm=next_milestone: _download_progress(
                    rel, done, total, cs, projected, rate_cu_h, budget_cu, nm, lock=progress_lock),
                on_token_expiring=lambda tid, rem: _request_token_refresh(token_provider))
            print('CACHE_VERIFIED_NVME', relative, flush=True)
            return int(item['size'])

        file_workers = _resolve_model_file_concurrency(len(pending))
        if file_workers <= 1:
            for spec_tuple in pending:
                downloaded_bytes += _download_one(spec_tuple)
        else:
            first_exc = None
            with ThreadPoolExecutor(max_workers=file_workers) as pool:
                futures = [pool.submit(_download_one, sp) for sp in pending]
                for fut in as_completed(futures):
                    try:
                        downloaded_bytes += fut.result()
                    except Exception as e:
                        if first_exc is None:
                            first_exc = e
                            local_abort.set()
            if first_exc is not None:
                raise first_exc

        elapsed = time.monotonic() - dl_start
        mbps = round((downloaded_bytes / 1_000_000) / elapsed, 2) if elapsed > 0 and downloaded_bytes > 0 else 0.0
        if metrics is not None:
            metrics["weights_download_sec"] = round(elapsed, 3)
            metrics["weights_downloaded_bytes"] = downloaded_bytes
            metrics["effective_download_mbps"] = mbps
            metrics["download_workers"] = _resolve_download_workers_count()
        return

    if not cache.is_dir():
        raise RuntimeError('Drive cache not mounted. OAuth/mount MUST be handled interactively before worker.')
    total = int(sum(int(x['size']) for x in manifest['files']))
    check_restore_capacity(comfy, total, rate_cu_h=rate_cu_h, budget_cu=budget_cu, target_dir=target_root)
    projected = False
    for item in manifest['files']:
        _check_abort(abort_event, "restore_weights")
        relative = Path(item['path'])
        src = cache / relative
        require_inside(src, cache)
        if not src.is_file() or src.stat().st_size != item['size']:
            raise RuntimeError('Drive cache absent/incomplete: ' + str(src))
        dst = target_root / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size == item['size'] and sha256_file(dst) == item['sha256']:
            print('CACHE_REUSE', relative, flush=True)
            continue
        tmp = dst.with_name(dst.name + '.part')
        try:
            with src.open('rb') as fi, tmp.open('wb') as fo:
                copied = 0
                copy_start = time.monotonic()
                while True:
                    _check_abort(abort_event, "restore_weights")
                    chunk = fi.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    fo.write(chunk)
                    copied += len(chunk)
                    if copied % (256 * 1024 * 1024) == 0:
                        print('MODEL_COPY_PROGRESS', relative, copied, flush=True)
                        if not projected:
                            projected = True
                            project_restore_throughput(total, copied, time.monotonic() - copy_start, rate_cu_h=rate_cu_h, budget_cu=budget_cu)
            if tmp.stat().st_size != item['size'] or sha256_file(tmp) != item['sha256']:
                raise RuntimeError('Model checksum mismatch: ' + str(relative))
            tmp.replace(dst)
            downloaded_bytes += int(item['size'])
        finally:
            if tmp.exists():
                tmp.unlink()
        print('CACHE_VERIFIED_NVME', relative, flush=True)
    elapsed = time.monotonic() - dl_start
    mbps = round((downloaded_bytes / 1_000_000) / elapsed, 2) if elapsed > 0 and downloaded_bytes > 0 else 0.0
    if metrics is not None:
        metrics["weights_download_sec"] = round(elapsed, 3)
        metrics["weights_downloaded_bytes"] = downloaded_bytes
        metrics["effective_download_mbps"] = mbps
        metrics["download_workers"] = 1


def link_models_into_comfy(nvme_root: Path, comfy_models_dir: Path, manifest: dict) -> list[dict]:
    """Atomically symlink verified NVMe model weights into ComfyUI/models/.
    Enforces require_inside boundary checks on both source and destination
    and verifies source readability before creating each symlink."""
    nvme_root = Path(nvme_root)
    comfy_models_dir = Path(comfy_models_dir)
    files = (manifest or {}).get("files") or []
    if not files:
        raise ValueError("manifest has no model files to link")
    comfy_models_dir.mkdir(parents=True, exist_ok=True)
    linked: list[dict] = []
    for item in files:
        rel_str = item.get("path") if isinstance(item, dict) else None
        if not rel_str:
            raise ValueError("manifest file entry missing path")
        rel = Path(rel_str)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe path: {rel_str}")
        src = nvme_root / rel
        require_inside(src, nvme_root)
        if not src.is_file() or not os.access(src, os.R_OK):
            raise RuntimeError(f"NVMe model source missing or unreadable: {src}")
        try:
            with src.open("rb") as fh:
                fh.read(1)
        except OSError as e:
            raise RuntimeError(f"NVMe model source unreadable: {src}: {e}") from e
        dst = comfy_models_dir / rel
        # Check destination parent boundary without following a final symlink at dst.
        require_inside(dst.parent, comfy_models_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        require_inside(dst.parent, comfy_models_dir)
        tmp_link = dst.parent / f".{dst.name}.link.{uuid.uuid4().hex[:8]}"
        try:
            os.symlink(str(src.resolve()), str(tmp_link))
            os.replace(str(tmp_link), str(dst))
        finally:
            if tmp_link.is_symlink() or tmp_link.exists():
                try:
                    tmp_link.unlink()
                except OSError:
                    pass
        if not dst.is_symlink() or dst.resolve() != src.resolve():
            raise RuntimeError(f"symlink verification failed for {dst}")
        linked.append({"path": rel_str, "src": str(src.resolve()), "dst": str(dst)})
    print("MODELS_LINKED_INTO_COMFY", len(linked), str(nvme_root), "->", str(comfy_models_dir), flush=True)
    return linked


def detect_gpu_vram_mib() -> int | None:
    """Best-effort GPU VRAM detection in MiB via nvidia-smi. Returns None
    when nvidia-smi is unavailable or unparsable."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        if proc.returncode != 0:
            return None
        for line in (proc.stdout or "").splitlines():
            text = line.strip()
            if text:
                return int(float(text.split()[0]))
    except Exception:
        return None
    return None


def resolve_comfy_vram_flags(vram_mib: int | None, env_override: str | None = None) -> list[str]:
    """Return extra ComfyUI VRAM flags. H3_COMFY_VRAM_FLAGS (including empty
    string to disable) overrides auto-detection; otherwise GPUs with
    >= 76800 MiB (G4 96GB / A100 80GB) get --highvram --disable-mmap."""
    raw = env_override if env_override is not None else os.environ.get("H3_COMFY_VRAM_FLAGS")
    if raw is not None:
        return shlex.split(raw)
    if vram_mib is not None and int(vram_mib) >= HIGHVRAM_THRESHOLD_MIB:
        return list(HIGHVRAM_DEFAULT_FLAGS)
    return []


def run_dual_track_setup(
    comfy: Path,
    git_ref: str,
    cache: Path | None,
    manifest: dict,
    nvme_models_dir: Path,
    comfyui_cache=None,
    drive_download=None,
    token_provider=None,
    rate_cu_h=None,
    budget_cu=None,
) -> dict:
    """F3 dual-track concurrency: run ensure_comfy() and restore_weights()
    concurrently with a shared abort_event, then join and atomically symlink
    weights into ComfyUI/models/. On any track failure, abort_event is set
    immediately and _LAST_STAGE reflects the exact failing track."""
    global _LAST_STAGE, _LAST_CMD_FAILURE
    abort_event = threading.Event()
    comfy_metrics: dict = {}
    weights_metrics: dict = {}
    timings: dict[str, list[float]] = {"ensure_comfy": [0.0, 0.0], "restore_weights": [0.0, 0.0]}

    def _track_comfy():
        timings["ensure_comfy"][0] = time.monotonic()
        try:
            ensure_comfy(
                comfy, git_ref, comfyui_cache,
                drive_download=drive_download, token_provider=token_provider,
                abort_event=abort_event, metrics=comfy_metrics)
        except Exception as e:
            cmd_fail = getattr(_CMD_FAILURE_LOCAL, "value", None)
            abort_event.set()
            return ("ensure_comfy", e, cmd_fail)
        finally:
            timings["ensure_comfy"][1] = time.monotonic()
        return None

    def _track_weights():
        timings["restore_weights"][0] = time.monotonic()
        try:
            restore_weights(
                comfy, cache, manifest,
                rate_cu_h=rate_cu_h, budget_cu=budget_cu,
                drive_download=drive_download, token_provider=token_provider,
                models_dir=nvme_models_dir, abort_event=abort_event,
                metrics=weights_metrics)
        except Exception as e:
            cmd_fail = getattr(_CMD_FAILURE_LOCAL, "value", None)
            abort_event.set()
            return ("restore_weights", e, cmd_fail)
        finally:
            timings["restore_weights"][1] = time.monotonic()
        return None

    set_stage("dual_track_setup")
    primary_failure = None
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_track_comfy), pool.submit(_track_weights)]
        for fut in as_completed(futures):
            outcome = fut.result()
            if outcome is not None and primary_failure is None:
                primary_failure = outcome

    if primary_failure is not None:
        fail_stage, fail_exc, fail_cmd = primary_failure
        with _STATE_LOCK:
            _LAST_STAGE = fail_stage
            _CMD_FAILURE_LOCAL.value = fail_cmd
            _LAST_CMD_FAILURE = fail_cmd
        raise fail_exc

    overlap = max(
        0.0,
        min(timings["ensure_comfy"][1], timings["restore_weights"][1])
        - max(timings["ensure_comfy"][0], timings["restore_weights"][0]),
    )
    set_stage("link_models")
    link_models_into_comfy(nvme_models_dir, comfy / "models", manifest)
    return {
        "comfy_source_sec": comfy_metrics.get("comfy_source_sec", 0.0),
        "pip_install_sec": comfy_metrics.get("pip_install_sec", 0.0),
        "wheelhouse_used": bool(comfy_metrics.get("wheelhouse_used", False)),
        "wheelhouse_fallback_reason": comfy_metrics.get("wheelhouse_fallback_reason"),
        "weights_download_sec": weights_metrics.get("weights_download_sec", 0.0),
        "effective_download_mbps": weights_metrics.get("effective_download_mbps", 0.0),
        "download_workers": int(weights_metrics.get("download_workers", _resolve_download_workers_count())),
        "parallel_overlap_sec": round(overlap, 3),
    }


def build_coldstart_metrics(
    setup_metrics: dict,
    comfy_boot_sec: float,
    total_coldstart_sec: float,
    vram_mib: int | None,
    vram_flags: list[str],
    *,
    warm_session_reused: bool = False,
) -> dict:
    """Build the locked v0.2.0 COLDSTART_METRICS_JSON payload.

    When ``warm_session_reused`` is True (v0.3.0 M4), includes
    ``"warm_session_reused": True``; when False, keeps the exact v0.2.0 key
    set so existing schema-contract tests remain unchanged.
    """
    payload = {
        "schema_version": COLDSTART_METRICS_SCHEMA_VERSION,
        "comfy_source_sec": round(float(setup_metrics.get("comfy_source_sec", 0.0)), 3),
        "pip_install_sec": round(float(setup_metrics.get("pip_install_sec", 0.0)), 3),
        "weights_download_sec": round(float(setup_metrics.get("weights_download_sec", 0.0)), 3),
        "effective_download_mbps": round(float(setup_metrics.get("effective_download_mbps", 0.0)), 2),
        "comfy_boot_sec": round(float(comfy_boot_sec), 3),
        "total_coldstart_sec": round(float(total_coldstart_sec), 3),
        "wheelhouse_used": bool(setup_metrics.get("wheelhouse_used", False)),
        "wheelhouse_fallback_reason": setup_metrics.get("wheelhouse_fallback_reason"),
        "parallel_overlap_sec": round(float(setup_metrics.get("parallel_overlap_sec", 0.0)), 3),
        "download_workers": int(setup_metrics.get("download_workers", 1)),
        "vram_mib": int(vram_mib) if vram_mib is not None else None,
        "highvram_enabled": "--highvram" in (vram_flags or []),
    }
    if warm_session_reused:
        payload["warm_session_reused"] = True
    return payload


def check_warm_comfy_alive(server: str, timeout: float = 5.0) -> dict | None:
    """Check whether a resident ComfyUI server on ``server`` is alive and
    exposes all required MiniMax-H3 nodes (v0.3.0 M4). Returns the parsed
    ``/object_info`` dict when healthy, or ``None`` otherwise."""
    try:
        info = get_json(server.rstrip("/") + "/object_info", timeout=timeout)
    except Exception:
        return None
    if not isinstance(info, dict):
        return None
    for typ in ("MiniMaxH3ReferenceToVideo", "SaveVideo", "UNETLoader", "CLIPLoader"):
        if typ not in info:
            return None
    return info


def _all_models_present_in_comfy(comfy: Path, manifest: dict) -> bool:
    """True when all pinned model files from ``manifest`` are present and
    match their expected byte size under ``comfy / 'models'``."""
    files = manifest.get("files") or []
    if len(files) < 4:
        return False
    models_dir = comfy / "models"
    for item in files:
        rel = item.get("path")
        expected_size = item.get("size")
        if not rel or expected_size is None:
            return False
        target = models_dir / rel
        try:
            if not target.is_file() or target.stat().st_size != int(expected_size):
                return False
        except OSError:
            return False
    return True


def _cleanup_ephemeral_tokens(root: Path) -> None:
    """Best-effort delete ephemeral Drive token files after a warm run."""
    for pattern in ("drive_token*.json", "drive_upload_token.json"):
        for tok in root.glob(pattern):
            try:
                tok.unlink()
            except OSError:
                pass


def _terminate_comfy_pid(root: Path) -> None:
    """Terminate a persisted ComfyUI PID from ``root / 'comfyui.pid'``."""
    import signal

    pid_file = root / "comfyui.pid"
    if not pid_file.is_file():
        return
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        pid = None
    if pid is not None and pid > 1:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    try:
        pid_file.unlink()
    except OSError:
        pass


def _download_progress(relative, done, grand_total, copy_start, projected, rate_cu_h, budget_cu, next_milestone=None, lock=None):
    """Progress hook for the noninteractive download path: periodic
    throughput projection (same once-only rule as the mounted-copy
    path: grand total projected from the current file's measured rate)
    without ever touching token material. Thread-safe when lock is supplied."""
    if next_milestone is None:
        next_milestone = [256 * 1024 * 1024]
    should_project = False
    if lock is not None:
        with lock:
            if done >= next_milestone[0]:
                print('MODEL_COPY_PROGRESS', relative, done, flush=True)
                next_milestone[0] = (done // (256 * 1024 * 1024) + 1) * (256 * 1024 * 1024)
                if not projected[0]:
                    projected[0] = True
                    should_project = True
    else:
        if done >= next_milestone[0]:
            print('MODEL_COPY_PROGRESS', relative, done, flush=True)
            next_milestone[0] = (done // (256 * 1024 * 1024) + 1) * (256 * 1024 * 1024)
            if not projected[0]:
                projected[0] = True
                should_project = True
    if should_project:
        project_restore_throughput(grand_total, done, time.monotonic() - copy_start,
                                   rate_cu_h=rate_cu_h, budget_cu=budget_cu)

def wait_health(server, proc,deadline):
    while time.monotonic()<deadline:
        if proc.poll() is not None:raise RuntimeError('ComfyUI server exited '+str(proc.returncode))
        try:
            info=get_json(server+'/object_info',timeout=5)
            for typ in ('MiniMaxH3ReferenceToVideo','SaveVideo','UNETLoader','CLIPLoader'):
                if typ not in info:raise RuntimeError('Required Comfy node missing '+typ)
            return info
        except (urllib.error.URLError,TimeoutError,ConnectionError):pass
        time.sleep(4)
    raise TimeoutError('ComfyUI /object_info start timeout')

# Control-after-generate modes, serialized by the frontend as extra input
# keys on primitive nodes (e.g. PrimitiveInt {"value": 20, "fixed": "fixed"}).
# The live /object_info does NOT list them, and the server ignores unknown
# prompt keys (execution.py: validate_inputs only iterates schema inputs;
# get_input_data drops keys whose input_category is None), so the validator
# must tolerate them. Closed set from the pinned ComfyUI enum
# ControlAfterGenerate (comfy_api/latest/_io.py). 2026-11-12: the old
# check rejected the official template's nodes 143/144 for exactly this.
CONTROL_AFTER_GENERATE_MODES = {"fixed", "increment", "decrement", "randomize"}


def validate_api_against_object_info(api:dict,info:dict)->None:
    """Real /object_info schema check. Reject unknown class_types, missing
    required inputs, and a sampler chain not connected to SaveVideo.
    Static node-name presence alone is not acceptance.

    Autogrow (dynamic-group) inputs: the live schema lists the group name
    (e.g. "values", "ref_images") while the prompt addresses entries with
    dotted keys ("values.a", "ref_images.ref_image_0") — the server nests
    them into the group via dynamic_paths before execution. A required
    group is therefore satisfied by any dotted entry under it, and dotted
    keys whose root is a schema input are allowed. 2026-11-12: the old
    literal check rejected the official template's node 131
    (ComfyMathExpression, "values.a") and would have rejected node 136
    (MiniMaxH3ReferenceToVideo, "ref_images.ref_image_*"); the workflow
    was correct, the validator was not. The server remains the final
    authority on prompt validity."""
    if 'nodes' in api:
        raise ValueError('Expected ComfyUI API format, not UI-format workflow (nodes/links)')
    for key,node in api.items():
        typ=node.get('class_type') if isinstance(node,dict) else None
        if typ not in info:
            raise RuntimeError(f'ComfyUI /object_info has no node type {typ!r} for workflow node {key}')
        schema=(info[typ].get('input') or {})
        required=schema.get('required') or {}
        inputs=node.get('inputs') or {}
        for field in required:
            if field not in inputs and not any(
                    f.startswith(field + '.') for f in inputs):
                raise RuntimeError(f'Workflow node {key} ({typ}) missing required input {field!r}')
        # Unknown input fields are rejected: they indicate a stale template.
        # Tolerated: dotted autogrow entries (root is a schema input) and
        # the frontend-serialized control-after-generate mode keys.
        allowed=set(required) | set(schema.get('optional') or {})
        for field in inputs:
            if (field not in allowed and field.split('.', 1)[0] not in allowed
                    and field not in CONTROL_AFTER_GENERATE_MODES):
                raise RuntimeError(f'Workflow node {key} ({typ}) has unknown input {field!r} not in /object_info schema')
    # SaveVideo must be reachable: at least one node in the graph must feed it.
    # The API format links inputs by [node_id, slot]; check SaveVideo inputs reference other nodes.
    save_nodes=[n for n in api.values() if isinstance(n,dict) and n.get('class_type')=='SaveVideo']
    if not save_nodes:
        raise RuntimeError('No SaveVideo node in submitted workflow')
    for node in save_nodes:
        refs=[v for v in (node.get('inputs') or {}).values() if isinstance(v,list) and v and isinstance(v[0],(str,int))]
        if not refs:
            raise RuntimeError('SaveVideo node has no connected sampler/video input')

def execute_prompt(server,api:dict,timeout_sec:float,proc=None):
    """Submit api to ComfyUI and poll /history until the prompt succeeds.

    TASK-008 debounce: the /history poll uses a 30 s single-request
    timeout, and transient failures (timeouts, connection errors,
    HTTP 502/503/504) are logged as H3_HISTORY_POLL_TRANSIENT and
    retried with backoff — up to 8 consecutive — while ComfyUI is
    still alive and the overall inference deadline has not passed. A
    transient failure never kills a running GPU inference. Only a
    dead ComfyUI process (proc.poll() is not None, when a proc handle
    is supplied), a non-transient error, 9 consecutive transient
    failures, or the overall deadline ends the poll. The submit itself
    is idempotent: a fixed client_id is used for the whole call, and
    a read timeout on /prompt is resolved by checking /queue +
    /history before any resubmit (see _submit_prompt)."""
    client_id='muse-h3-'+uuid.uuid4().hex
    result=_submit_prompt(server,api,client_id)
    pid=result['prompt_id'];end=time.monotonic()+timeout_sec
    print('INFERENCE_SUBMITTED',pid,flush=True)
    transient=0
    while time.monotonic()<end:
        try:
            data=get_json(server+'/history/'+pid,timeout=HISTORY_POLL_TIMEOUT_SEC)
        except Exception as e:  # noqa: BLE001 - transient vs fatal split below
            if not _is_transient_api_error(e):
                raise
            transient+=1
            print('H3_HISTORY_POLL_TRANSIENT',transient,type(e).__name__,str(e)[:300],flush=True)
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(
                    f'ComfyUI process exited (returncode {proc.poll()}) during '
                    f'/history poll after {transient} transient failures; '
                    f'last poll error: {type(e).__name__}: {str(e)[:300]}')
            if transient>HISTORY_TRANSIENT_MAX_CONSECUTIVE:
                raise RuntimeError(
                    f'/history poll failed {transient} consecutive times '
                    f'(transient threshold {HISTORY_TRANSIENT_MAX_CONSECUTIVE}); '
                    f'last error: {type(e).__name__}: {str(e)[:300]}')
            time.sleep(min(5*transient,30))
            continue
        transient=0
        if pid in data:
            event=data[pid]
            if event.get('status',{}).get('status_str')=='error':
                raise RuntimeError('ComfyUI inference error: '+str(event.get('status'))[:1500])
            if event.get('status',{}).get('messages'):
                for msg in event['status']['messages']:
                    if msg and msg[0] in ('execution_error','execution_interrupted'):
                        raise RuntimeError('ComfyUI execution error: '+str(msg)[:1500])
            outputs=event.get('outputs',{})
            # Only a completed (success) history entry with outputs counts.
            # HTTP 200 / non-empty history alone is not success.
            if outputs and event.get('status',{}).get('status_str')=='success':
                return outputs
        print('H3_INFERENCE_HEARTBEAT',int(end-time.monotonic()),flush=True)
        time.sleep(8)
    raise TimeoutError('H3 inference deadline exceeded')

VIDEO_SUFFIXES = ('.mp4', '.webm', '.mov')

# Raw animation the worker must locate: the TaskSpec's animation shape
# (animation frames at fps and width x height). The ffprobe gate below
# requires the container to decode fully (-count_frames), a video
# stream, exactly that many frames at that fps and size. Those are the
# spec values assemble.py already enforces downstream; the gate fails
# fast here, with the observed value in the error, instead of letting a
# wrong file reach assembly. With no spec the frozen A/B constants
# apply. If a future live run shows ComfyUI writing a different
# container fps, the gate fails closed naming it — correct the spec
# with that evidence, never by weakening the gate to suffix+size.


def probe_animation(path, spec=None):
    """ffprobe decode gate for a raw animation file (full decode via
    assemble.ffprobe -count_frames). Returns probe evidence or raises
    RuntimeError naming the mismatch. A non-video blob fails here: ffprobe
    cannot parse it, so there is no video stream to measure."""
    expected_frames = spec.animation_frames if spec is not None else ANIMATION_FRAMES
    expected_fps = spec.fps if spec is not None else FPS
    expected_width = spec.width if spec is not None else WIDTH
    expected_height = spec.height if spec is not None else HEIGHT
    try:
        data = ffprobe(Path(path))
    except Exception as e:  # noqa: BLE001 - ffprobe errors are the verdict
        raise RuntimeError(f"ffprobe failed on {Path(path).name}: {type(e).__name__}: {str(e)[:200]}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"No video stream in {Path(path).name}")
    frames = int(video.get("nb_read_frames") or video.get("nb_frames") or 0)
    rate = video.get("r_frame_rate", "0/1")
    num, den = (rate.split("/") + ["1"])[:2]
    fps = float(num) / float(den) if float(den) else 0.0
    info = {"path": str(path), "frames": frames, "fps": round(fps, 3),
            "width": int(video.get("width", 0)), "height": int(video.get("height", 0)),
            "codec": video.get("codec_name")}
    problems = []
    if frames != expected_frames:
        problems.append(f"frames={frames}, expected {expected_frames}")
    if abs(fps - expected_fps) > 0.5:
        problems.append(f"fps={fps}, expected {expected_fps}")
    if info["width"] != expected_width or info["height"] != expected_height:
        problems.append(f"size={info['width']}x{info['height']}, expected {expected_width}x{expected_height}")
    if problems:
        raise RuntimeError(f"Animation probe failed for {Path(path).name}: " + "; ".join(problems))
    return info


def _history_candidates(comfy, outputs, run_id="", variant=""):
    """Filenames named by /history outputs, in history order, de-duplicated.
    Entries without a filename can never name a file. When run_id/variant
    are given, only files whose name contains both are this run's — a
    previous run's file in the same output dir must never be picked up."""
    candidates = []
    for output in outputs.values():
        # SaveVideo's UI output in the pinned ComfyUI (0.44.0) is
        # PreviewVideo, whose as_dict() is {"images": [...], "animated":
        # (True,)} — NOT {"videos": [...]}. The old 'videos'/'gifs' keys
        # are kept for older formats; 'images' is read only when that
        # output entry has a truthy 'animated' value.
        keys = ['videos', 'gifs']
        if output.get('animated'):
            keys.append('images')
        for key in keys:
            for x in output.get(key, []) or []:
                name = x.get('filename')
                if not name:
                    continue
                if run_id and run_id not in name:
                    continue
                if variant and variant not in name:
                    continue
                p = comfy / 'output' / (x.get('subfolder') or '') / name
                if p not in candidates:
                    candidates.append(p)
    return candidates


def _validate_candidate(p, comfy, probe):
    """Suffix + size + ffprobe gates for one history candidate.
    Returns (probe_info, None) on pass or (None, reason) on failure."""
    try:
        require_inside(p, comfy / 'output')
    except ValueError:
        return None, 'path escapes ComfyUI/output'
    if not p.is_file():
        return None, 'not on disk'
    if p.suffix.lower() not in VIDEO_SUFFIXES:
        return None, f'unsupported suffix {p.suffix}'
    if p.stat().st_size <= 10000:
        return None, 'too small'
    try:
        return probe(p), None
    except RuntimeError as e:
        return None, str(e)[:300]


def find_rendered_video(comfy, outputs, run_id="", variant="", probe=None, spec=None):
    """Locate this run's rendered animation from /history outputs.

    Priority: history file records (videos/gifs, plus images when animated
    is truthy). Every candidate must pass suffix, size and the ffprobe
    decode gate (probe_animation by default, judged against the spec;
    inject a fake probe in tests to exercise selection logic without
    encoding video). Exactly one valid candidate wins; several valid
    candidates is AMBIGUITY, never "take the first" — a stale file from
    another run must not be silently used.
    Raises RuntimeError when nothing valid is found; the caller then runs
    rescue_rendered_video() before giving up.
    """
    probe = probe or (lambda p: probe_animation(p, spec))
    candidates = _history_candidates(comfy, outputs, run_id, variant)
    valid = []
    rejected = []
    for p in candidates:
        info, reason = _validate_candidate(p, comfy, probe)
        if info is None:
            rejected.append(f"{p.name}: {reason}")
        else:
            valid.append(p)
    if len(valid) == 1:
        return valid[0]
    if len(valid) > 1:
        raise RuntimeError('Ambiguous rendered video: multiple valid candidates for run '
                           + (run_id or '?') + ': ' + ', '.join(str(p) for p in valid))
    raise RuntimeError('ComfyUI history did not contain verifiable saved video; DO NOT assume successful render. '
                       'History output keys: ' + json.dumps({k: sorted(v.keys()) for k, v in outputs.items()}, ensure_ascii=False)[:800]
                       + ('; rejected: ' + '; '.join(rejected) if rejected else ''))


def rescue_rendered_video(comfy, run_id, variant, remaining_budget_seconds, probe=None, spec=None):
    """Bounded rescue when history cannot locate the video (TASK-002 §4.4).

    Read-only scan of ComfyUI/output/ for files whose name contains this
    run's RUN_ID (and variant). Each name match still passes suffix, size
    and the ffprobe decode gate, and its SHA-256/path/size are recorded.
    Returns evidence dict for the single valid file. Raises when there is
    no valid file, when several valid files make ownership ambiguous, or
    when remaining_budget_seconds runs out — never to keep the GPU alive
    past the session budget for a rescue. The caller's outer exec timeout
    remains the hard bound; this is the inner one.
    """
    probe = probe or (lambda p: probe_animation(p, spec))
    if not run_id:
        raise RuntimeError('Rescue needs a run_id; refusing to guess which output belongs to this run')
    if remaining_budget_seconds is None or float(remaining_budget_seconds) <= 0:
        raise RuntimeError('Rescue time budget exhausted before scan')
    deadline = time.monotonic() + float(remaining_budget_seconds)
    out_root = comfy / 'output'
    if not out_root.is_dir():
        raise RuntimeError('Rescue: ComfyUI/output missing')
    matches = []
    for p in sorted(out_root.rglob('*'), key=lambda q: q.stat().st_mtime if q.is_file() else 0, reverse=True):
        if time.monotonic() > deadline:
            raise RuntimeError('Rescue time budget exhausted during scan')
        if not p.is_file() or p.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        if run_id not in p.name or (variant and variant not in p.name):
            continue
        if p.stat().st_size <= 10000:
            continue
        matches.append(p)
        if len(matches) > 20:
            break
    valid = []
    for p in matches:
        if time.monotonic() > deadline:
            raise RuntimeError('Rescue time budget exhausted during probe')
        try:
            info = probe(p)
        except RuntimeError:
            continue
        valid.append((p, info))
    if len(valid) == 1:
        p, info = valid[0]
        return {"path": str(p), "size_bytes": p.stat().st_size, "sha256": sha256_file(p),
                "frames": info["frames"], "fps": info["fps"],
                "width": info["width"], "height": info["height"], "codec": info.get("codec")}
    if len(valid) > 1:
        raise RuntimeError('Rescue ambiguous: multiple valid videos for run ' + run_id + ': '
                           + ', '.join(str(p) for p, _ in valid))
    raise RuntimeError(f'Rescue found no verifiable video for run {run_id} variant {variant} '
                       f'(scanned {len(matches)} name-matching files)')

def _md5_hex(path) -> str:
    """Return 32-char lowercase hex MD5 matching Google Drive API v3 md5Checksum."""
    h = hashlib.md5()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def validate_assembled_mp4(final, info, spec=None) -> dict:
    """Independent worker-side validation of the assembled MP4 before
    upload (TASK-006 P1-A): exact frames/fps/dimensions + one decoded
    frame + SHA-256 agreement with the assemble info. Raises on any
    mismatch; a failed validation is never uploaded. Expectations come
    from the TaskSpec when given; without one the frozen A/B shape
    (274 frames, 24 fps, 1080x1920) is expected, as before."""
    expected_frames = spec.total_frames if spec is not None else TOTAL_FRAMES
    expected_fps = spec.fps if spec is not None else FPS
    expected_width = spec.output_width if spec is not None else 1080
    expected_height = spec.output_height if spec is not None else 1920
    final = Path(final)
    if not final.is_file():
        raise ValueError(f"assembled MP4 missing: {final}")
    probe = ffprobe(final)
    streams = probe.get("streams", [])
    vid = next((x for x in streams if x.get("codec_type") == "video"), None)
    if not vid:
        raise ValueError("assembled MP4 has no video stream")
    frames = int(vid.get("nb_read_frames") or vid.get("nb_frames") or 0)
    if frames != expected_frames:
        raise ValueError(f"assembled MP4 has {frames} frames, expected {expected_frames}")
    fps_raw = str(vid.get("r_frame_rate") or vid.get("avg_frame_rate") or "")
    try:
        num, den = (int(x) for x in fps_raw.split("/"))
        fps_val = num / den if den else 0
    except (ValueError, ZeroDivisionError):
        fps_val = 0
    if abs(fps_val - expected_fps) > 0.01:
        raise ValueError(f"assembled MP4 fps {fps_raw}, expected {expected_fps}")
    # The final video size is the spec's output size (1080x1920 for the
    # frozen A/B shape; WIDTH/HEIGHT in h3_contract are the 736x1312
    # model resolution, not the assembled output).
    if int(vid.get("width") or 0) != expected_width or int(vid.get("height") or 0) != expected_height:
        raise ValueError(f"assembled MP4 is {vid.get('width')}x{vid.get('height')}, expected {expected_width}x{expected_height}")
    # Decodable: one frame through the decoder.
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(final),
                    "-frames:v", "1", "-f", "null", "-"],
                   check=True, timeout=120, capture_output=True)
    sha = sha256_file(final)
    if info.get("sha256") and sha.lower() != str(info["sha256"]).lower():
        raise ValueError("assembled MP4 SHA-256 does not match assemble info")
    return {"sha256": sha, "md5": _md5_hex(final), "frames": frames}


def upload_remote_archive(final, info, task, run_id, version, spec=None):
    """Validate + resumably upload the assembled MP4 from the worker
    (TASK-006 P1-A). Returns the evidence dict, or None when no
    drive_upload block was delivered (mount path / legacy runs).
    Raises on validation, upload or verification failure — the caller
    records REMOTE_UPLOAD_FAILED and continues with the controller-side
    fallback; inference is never re-run because an upload failed."""
    block = task.get("drive_upload")
    if not isinstance(block, dict):
        return None
    folder_id = block.get("folder_id")
    file_name = block.get("file_name") or f"{version}.mp4"
    if not folder_id:
        raise ValueError("drive_upload block has no folder_id")
    from h3_drive_upload import DriveUploader, FileTokenProvider

    validated = validate_assembled_mp4(final, info, spec)
    upload_provider = FileTokenProvider(ROOT / "drive_upload_token.json")
    upload_provider.load_initial()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    uploader = DriveUploader(upload_provider)
    evidence = uploader.upload_file(
        final, folder_id, file_name, run_id, version,
        validated["sha256"],
        sidecar_path=ROOT / "output" / f"{version}.upload_state.json",
        on_token_expiring=lambda tid, rem: _request_token_refresh(
            upload_provider, scope="drive.file"))
    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    verified = uploader.verify_uploaded(
        evidence["drive_file_id"], evidence["size"], evidence["sha256"],
        expected_md5=validated["md5"], parent_id=folder_id)
    return {
        "drive_file_id": evidence["drive_file_id"],
        "size": evidence["size"],
        "sha256": evidence["sha256"],
        "md5": validated["md5"],
        "uploaded_by": "worker",
        "verified_by": "worker_readback",
        "upload_started_at_utc": started_at,
        "upload_finished_at_utc": finished_at,
        "reused": bool(evidence.get("reused")),
        "reused_via": evidence.get("reused_via"),
        "drive_output_folder_id": folder_id,
        "read_back_bytes": verified.get("read_back_bytes"),
    }


def main():
    task=json.loads((ROOT/'task.json').read_text())
    # TASK-007: the task is a TaskSpec now. The naming token for every
    # output file is spec.output_prefix (the old task['variant']); the
    # A/B fixture adapter supplies it for legacy task.json files.
    spec=_spec_from_task(task)
    version=spec.output_prefix
    run_id=task.get('run_id','')
    global _JOB_RUN_ID, _JOB_VARIANT
    _JOB_RUN_ID = run_id
    _JOB_VARIANT = version
    manifest=json.loads((ROOT/'model_manifest.json').read_text())
    # TASK-006 P0-B: noninteractive download path. The token provider is
    # built once here and shared by the ComfyUI restore and the model
    # restore; load_initial fails closed before any inference when no
    # token file was delivered. The mount path leaves drive_download
    # unset and keeps the previous behaviour exactly.
    drive_download=task.get('drive_download')
    token_provider=None
    if drive_download is not None:
        from h3_drive_download import FileTokenProvider
        token_provider=FileTokenProvider(ROOT/'drive_token.json')
        token_provider.load_initial()
    if task.get('drive_cache_dir'):
        cache=Path(task['drive_cache_dir'])
    elif drive_download is not None:
        cache=Path(drive_download['dest_dir'])
    else:
        cache=None
    comfy=Path('/content/ComfyUI')
    # Reference images, verified before any expensive stage. Legacy
    # task.json (no task_spec): the only reference is the uploaded
    # poster.jpg. Spec task: the controller uploaded ref_image_1.jpg /
    # ref_image_2.jpg; re-verify the received bytes against the spec
    # SHA-256s here, so a corrupted upload can never reach inference.
    legacy = task.get('task_spec') is None
    if legacy:
        ref1 = ref2 = ROOT/'poster.jpg'
        if not ref1.is_file():raise FileNotFoundError('poster missing')
    else:
        ref1 = ROOT/'ref_image_1.jpg'
        ref2 = ROOT/'ref_image_2.jpg'
        validate_ref_image(ref1, spec.ref_image_1.sha256)
        validate_ref_image(ref2, spec.effective_ref_image_2.sha256)
    server='http://127.0.0.1:8188'
    logfile=(ROOT/'comfyui.log').open('a')
    proc=None
    object_info=None
    api=None
    coldstart_metrics=None
    run_succeeded=False
    keep_warm_ttl_sec=int(task.get('keep_warm_ttl_sec') or 0)
    nvme_models_dir=Path(os.environ.get('H3_NVME_MODELS_DIR') or str(DEFAULT_NVME_MODELS_DIR))
    try:
        # The whole body is inside the try now: 2026-11-12 the worker died
        # in ensure_comfy (git clone exit 128) BEFORE the old try block,
        # so no WORKER_FAILED_JSON was printed and the stage had to be
        # inferred from the persisted traceback instead.
        coldstart_start=time.monotonic()
        # v0.3.0 M4: check whether a warm resident ComfyUI server is already
        # running with the exact pinned commit and all 4 model files intact.
        warm_object_info = check_warm_comfy_alive(server)
        warm_reused = (
            warm_object_info is not None
            and comfyui_commit(comfy) == task['comfy_ref']
            and _all_models_present_in_comfy(comfy, manifest)
        )
        if warm_reused:
            print('WARM_SESSION_REUSED', task['comfy_ref'], flush=True)
            (comfy/'input').mkdir(exist_ok=True)
            shutil.copy2(ref1,comfy/'input'/'muse_ref_1.jpg')
            shutil.copy2(ref2,comfy/'input'/'muse_ref_2.jpg')
            set_stage('object_info_validation')
            object_info=warm_object_info
            info=object_info
            vram_mib=detect_gpu_vram_mib()
            vram_flags=resolve_comfy_vram_flags(vram_mib)
            boot_end=time.monotonic()
            print('COMFYUI_HEALTH_OK',flush=True)
            coldstart_metrics=build_coldstart_metrics(
                {
                    "comfy_source_sec": 0.0,
                    "pip_install_sec": 0.0,
                    "weights_download_sec": 0.0,
                    "effective_download_mbps": 0.0,
                    "wheelhouse_used": False,
                    "wheelhouse_fallback_reason": None,
                    "parallel_overlap_sec": 0.0,
                    "download_workers": 0,
                },
                comfy_boot_sec=0.0,
                total_coldstart_sec=boot_end-coldstart_start,
                vram_mib=vram_mib,
                vram_flags=vram_flags,
                warm_session_reused=True,
            )
            print('COLDSTART_METRICS_JSON:'+json.dumps(coldstart_metrics,ensure_ascii=False,separators=(',',':')),flush=True)
        else:
            setup_metrics=run_dual_track_setup(
                comfy,
                task['comfy_ref'],
                cache,
                manifest,
                nvme_models_dir,
                comfyui_cache=task.get('comfyui_cache'),
                drive_download=drive_download,
                token_provider=token_provider,
                rate_cu_h=task.get('rate_cu_h'),
                budget_cu=task.get('session_budget_cu'),
            )
            (comfy/'input').mkdir(exist_ok=True)
            shutil.copy2(ref1,comfy/'input'/'muse_ref_1.jpg')
            shutil.copy2(ref2,comfy/'input'/'muse_ref_2.jpg')
            set_stage('comfyui_start')
            vram_mib=detect_gpu_vram_mib()
            vram_flags=resolve_comfy_vram_flags(vram_mib)
            print('COMFYUI_VRAM_CONFIG',json.dumps({'vram_mib':vram_mib,'flags':vram_flags},separators=(',',':')),flush=True)
            comfy_cmd=[sys.executable,'main.py','--listen','127.0.0.1','--port','8188',*vram_flags]
            boot_start=time.monotonic()
            proc=subprocess.Popen(comfy_cmd,cwd=comfy,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                (ROOT/'comfyui.pid').write_text(str(proc.pid),encoding='utf-8')
            except OSError:
                pass
            set_stage('object_info_validation')
            object_info=wait_health(server,proc,time.monotonic()+180)
            boot_end=time.monotonic()
            info=object_info
            print('COMFYUI_HEALTH_OK',flush=True)
            coldstart_metrics=build_coldstart_metrics(
                setup_metrics,
                comfy_boot_sec=boot_end-boot_start,
                total_coldstart_sec=boot_end-coldstart_start,
                vram_mib=vram_mib,
                vram_flags=vram_flags,
            )
            print('COLDSTART_METRICS_JSON:'+json.dumps(coldstart_metrics,ensure_ascii=False,separators=(',',':')),flush=True)
        raw=json.loads((ROOT/'workflow_api.json').read_text())
        api=patch_api_workflow_for_spec(raw,spec,'muse_ref_1.jpg','muse_ref_2.jpg',run_id=run_id)
        validate_api_against_object_info(api,info)
        (ROOT/'workflow_submitted.json').write_text(json.dumps(api,ensure_ascii=False,indent=2))
        print('COMFY_OBJECT_INFO_VALIDATED',flush=True)
        # Explicit MODELS_READY: only after object_info validation passed,
        # never at the start of the validation stage (TASK-005 §4.2).
        _write_job_state_safe('MODELS_READY',extra={'coldstart_metrics':coldstart_metrics})
        set_stage('inference_submit')
        outputs=execute_prompt(server,api,timeout_sec=task['inference_timeout_sec'],proc=proc)
        set_stage('find_rendered_video')
        try:
            source = find_rendered_video(comfy, outputs, run_id, version, spec=spec)
        except RuntimeError:
            # History could not locate a verifiable video (wrong shape,
            # stale name, or failed probe). Bounded rescue scans
            # ComfyUI/output/ for this RUN_ID before the GPU is released;
            # it never uses another run's file and never extends the
            # session budget — the controller's exec timeout still bounds
            # the whole worker run.
            set_stage('rescue_rendered_video')
            rescued = rescue_rendered_video(comfy, run_id, version, remaining_budget_seconds=300, spec=spec)
            source = Path(rescued['path'])
            print('VIDEO_RESCUED_JSON:' + json.dumps(rescued, ensure_ascii=False), flush=True)
        # Explicit INFERENCE_COMPLETED: a verifiable rendered video exists.
        try:write_job_state(ROOT,run_id,version,'INFERENCE_COMPLETED')
        except Exception:pass
        set_stage('assemble_mp4')
        final=ROOT/'output'/f'shao_nian_h3_{version}.mp4'
        # The tail card is the spec's ref_image_1 held for
        # tail_card_frames; with tail_card_frames=0 there is no tail
        # and the animation is delivered alone.
        tail_image = ref1 if spec.tail_card_frames > 0 else None
        info=assemble_video(source,tail_image,final,
                            animation_frames=spec.animation_frames,
                            tail_card_frames=spec.tail_card_frames,
                            tail_sha256=spec.ref_image_1.sha256 if spec.tail_card_frames > 0 else None,
                            fps=spec.fps, output_width=spec.output_width,
                            output_height=spec.output_height)
        # assemble_video returns 'frames' (see assemble.py); the old
        # 'frame_count' key printed None here (TASK-003 §3).
        print('MP4_WRITTEN',str(final),final.stat().st_size,info.get('frames'),flush=True)
        (ROOT/'output'/f'{version}.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
        # TASK-006 P1-A: worker-side durable upload. A failure here never
        # fails the run: the MP4 is valid locally and the controller falls
        # back to its own download + idempotent delivery. Inference is
        # never re-run because an upload failed.
        remote_upload=None
        if task.get('drive_upload') is not None:
            set_stage('remote_upload')
            try:
                remote_upload=upload_remote_archive(final,info,task,run_id,version,spec)
            except Exception as e:
                try:write_job_state(ROOT,run_id,version,'REMOTE_UPLOAD_FAILED',extra={'remote_upload_failed':{'error_type':type(e).__name__,'error':str(e)[:300]}})
                except Exception:pass
                print('REMOTE_UPLOAD_FAILED_JSON:'+json.dumps({'error_type':type(e).__name__,'error':str(e)[:300]},ensure_ascii=False),flush=True)
        if remote_upload is not None:
            try:write_job_state(ROOT,run_id,version,'REMOTE_UPLOAD_VERIFIED',extra={'remote_upload':remote_upload})
            except Exception:pass
            print('REMOTE_UPLOAD_JSON:'+json.dumps(remote_upload,ensure_ascii=False),flush=True)
        set_stage('done')
        # VIDEO_READY carries the same done info as MUSE_H3_DONE_JSON, so
        # a resumed controller can complete the run from the state file
        # without re-reading the worker's stdout (TASK-005 §4.2). The
        # remote-upload evidence and coldstart_metrics travel with it
        # so the controller can record them without parsing stdout.
        ready_extra={}
        if remote_upload is not None:
            ready_extra['remote_upload']=remote_upload
        if coldstart_metrics is not None:
            ready_extra['coldstart_metrics']=coldstart_metrics
        _write_job_state_safe('VIDEO_READY',output=info,extra=ready_extra or None)
        print('MUSE_H3_DONE_JSON:'+json.dumps(info,ensure_ascii=False),flush=True)
        run_succeeded=True
    except Exception as e:
        fail_extra={'coldstart_metrics':coldstart_metrics} if coldstart_metrics is not None else None
        _write_job_state_safe('FAILED',error=worker_failure_info(_LAST_STAGE,e),extra=fail_extra)
        print('WORKER_FAILED_JSON:'+json.dumps(worker_failure_info(_LAST_STAGE,e),ensure_ascii=False),flush=True)
        # On a validation failure, persist the live schemas for the node
        # types in this workflow (compact, filtered): the next diagnosis
        # then runs offline instead of costing another GPU session.
        if _LAST_STAGE=='object_info_validation' and object_info and api:
            try:
                types={n.get('class_type') for n in api.values() if isinstance(n,dict)}
                filtered={t:object_info[t] for t in sorted(types) if t in object_info}
                print('OBJECT_INFO_WORKFLOW_JSON:'+json.dumps(filtered,ensure_ascii=False),flush=True)
            except Exception:
                pass
        raise
    finally:
        if run_succeeded and keep_warm_ttl_sec > 0:
            # v0.3.0 M4: keep resident ComfyUI :8188 alive for the warm TTL
            # window, but scrub ephemeral Drive tokens from disk.
            _cleanup_ephemeral_tokens(ROOT)
        else:
            if proc is not None:
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill()
            _terminate_comfy_pid(ROOT)
        logfile.close()
if __name__=='__main__':main()
