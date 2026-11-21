"""Muse-host Google Drive Model Depot for MiniMax-H3 Ref2VA.

Default is read-only. The only write actions require explicit flags. No Colab
GPU is created here. An ADC user credential with Drive scopes is required.

Typical use:
    python drive_depot.py account-check --colab-auth adc
    python drive_depot.py init --allow-drive-write --colab-auth adc
    python drive_depot.py seed --allow-drive-write --allow-large-transfer --hf-revision <40-hex>
    python drive_depot.py verify --colab-auth adc

Uses *application-owned* Drive folders (drive.file permission). It deliberately
DOES NOT attempt to read the historic other-app H3 cache, which may be invisible
under drive.file scope even for the same human Google account.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, os, re, shutil, subprocess, sys, tempfile, time, uuid
from auth_support import ensure_adc_isolation, resolve_colab_executable
from pathlib import Path
from h3_contract import MODEL_FILES, resolve_drive_id, sha256_file
from drive_api import file_digests, get_file_evidence, verify_uploaded_file

ROOT=Path(__file__).resolve().parent
DEPOT_DIR=ROOT/'model_cache'
CFG=DEPOT_DIR/'depot_config.json'
READY=DEPOT_DIR/'verified_manifest.json'
STATE=DEPOT_DIR/'depot_state.json'
HF_REPO='Comfy-Org/MiniMax-H3'
HF_RESOLVE_BASE='https://huggingface.co'
PIN_FILE=DEPOT_DIR/'pinned_hf_files.json'
CONFIG_PIN_FILE=ROOT/'config'/'pinned_hf_files.json'
REMOTE_SEED_DIR='/content/muse_remote_seed'
REMOTE_SEED_MARKER='REMOTE_SEED_RESULT_JSON:'
REMOTE_SEED_DEFAULT_WALL_SEC=1800
DIRECT_CHUNK=1024*1024  # 1 MiB streaming writes
DIRECT_MAX_ATTEMPTS=3
DIRECT_CONNECT_TIMEOUT=30  # seconds
DIRECT_READ_TIMEOUT=120  # seconds between bytes; a slow-but-alive download is not killed
DIRECT_BACKOFF_SEC=(2,4)
# User has explicitly approved full Drive scope for this dedicated project OAuth client.
# The code itself still writes ONLY under the chosen depot folder.
DRIVE_SCOPE='https://www.googleapis.com/auth/drive'
CHUNK=8*1024*1024
REQUIRED=tuple(sorted(MODEL_FILES))
FOLDER_MIME='application/vnd.google-apps.folder'
# Synthetic placeholder keys for historical VAE reuse (v0.3.1 P1-6).
# Real Drive IDs for local runs live in git-ignored `model_cache/local_drive_ids.json`.
LEGACY_VAE_IDS={
    'vae/minimax_h3_video_vae_int8_convrot.safetensors':'EXAMPLE_LEGACY_VIDEO_VAE_ID',
    'vae/minimax_h3_audio_vae_fp32.safetensors':'EXAMPLE_LEGACY_AUDIO_VAE_ID',
}


def atomic_json(path:Path,obj:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,sort_keys=True,indent=2)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def read_json(path:Path)->dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _is_transient_drive_error(exc: BaseException) -> bool:
    """Transient transport failures are retryable; verification failures are not.
    The VM egresses via a TLS-intercepting proxy; the TLS connection is
    intermittently torn down mid-request (ssl.SSLEOFError: EOF occurred in
    violation of protocol) during Drive API calls. The Drive calls are
    idempotent (reuse existing files, resume uploads, re-verify), so a retry
    with a fresh transport is safe. Checksum mismatches, duplicates and
    wrong-parent errors must still fail immediately.
    """
    import http.client, ssl
    return isinstance(exc, (ssl.SSLError, ConnectionError,
                            http.client.HTTPException, TimeoutError))


def drive_service():
    # Google Cloud ADC created by USER, not Colab CLI's private OAuth keyring.
    # User chose broad Drive scope; never hand the long-lived ADC file to Colab.
    # Transport (TASK-005 PR-7, Issue #30): requests via h3_drive_transport.
    # The stock httplib2 transport failed on this VM with
    # SSL WRONG_VERSION_NUMBER on every Drive API call (measured §6/§7:
    # same ADC, same scopes — requests files.get OK, httplib2 fails), and
    # build(credentials=...) does NOT escape httplib2 (google-api-python-
    # client 2.202.0 routes credentials through AuthorizedHttp). The
    # requests adapter is the only transport that works here. No request,
    # scope or verification logic is changed by this.
    import google.auth
    from h3_drive_transport import build_requests_drive_service

    ensure_adc_isolation()
    creds,_=google.auth.default(scopes=[DRIVE_SCOPE,'https://www.googleapis.com/auth/userinfo.email'])
    return build_requests_drive_service(creds)


def account_matches(service, colab_auth:str)->bool:
    """Compare accounts in memory without emitting email or tokens to logs."""
    exe=resolve_colab_executable()
    proc=subprocess.run([exe,'--auth='+colab_auth,'whoami'],input='',capture_output=True,
                        text=True,timeout=65,check=False)
    # Hidden official developer command; fail closed if unavailable.
    if proc.returncode:raise RuntimeError('CLI whoami unavailable or not authorized: check CLI version and interactive OAuth')
    mat=re.search(r'^Email:\s*([^\s]+@[^\s]+)\s*$',proc.stdout,re.I|re.M)
    if not mat:raise RuntimeError('CLI whoami output lacks email; refuse account assertion')
    about=service.about().get(fields='user(emailAddress)').execute()
    drive_email=(about.get('user') or {}).get('emailAddress','').strip().casefold()
    cli_email=mat.group(1).strip().casefold()
    if not drive_email:raise RuntimeError('Drive API did not disclose signed-in user')
    return drive_email==cli_email


def list_named(service,parent:str,name:str):
    safe=name.replace('\\','\\\\').replace("'","\\'")
    q=f"'{parent}' in parents and name = '{safe}' and trashed = false"
    result=service.files().list(q=q,spaces='drive',pageSize=100,
         fields='nextPageToken,files(id,name,mimeType,size,md5Checksum,parents,appProperties)',
         supportsAllDrives=True,includeItemsFromAllDrives=True).execute()
    if result.get('nextPageToken'):raise RuntimeError('Ambiguous Drive listing, >100 items; refuse overwrite')
    return result.get('files',[])


def ensure_folder(service,parent:str|None,name:str,allow_write:bool):
    # Prevent duplicate root folders if VM crashed after remote creation but
    # before local depot_config.json was written.
    lookup_parent=parent or 'root'
    found=[f for f in list_named(service,lookup_parent,name) if f.get('mimeType')==FOLDER_MIME]
    if len(found)>1:raise RuntimeError('Duplicate folder name; refuse ambiguity')
    if found:return found[0]['id']
    if not allow_write:raise RuntimeError('Missing Drive folder '+name)
    body={'name':name,'mimeType':FOLDER_MIME}
    if parent:body['parents']=[parent]
    out=service.files().create(body=body,fields='id,name,mimeType',supportsAllDrives=True).execute()
    return out['id']


def verify_drive_entry(service,record:dict)->dict:
    """Refetch provider metadata, not a FUSE write acknowledgment or READY string."""
    meta=get_file_evidence(service,record['drive_file_id'])
    if meta.get('trashed') or meta.get('mimeType')==FOLDER_MIME:raise RuntimeError('Trashed/folder entry')
    if int(meta.get('size',-1))!=int(record['size']):raise RuntimeError('Server size mismatch for '+record['path'])
    expected_md5=record.get('server_md5','').lower()
    if len(expected_md5)!=32 or meta.get('md5Checksum','').lower()!=expected_md5:
        raise RuntimeError('Drive service MD5 mismatch for '+record['path'])
    if record.get('parent_id') not in meta.get('parents',[]):
        raise RuntimeError('Parent folder mismatch for '+record['path'])
    sha=record.get('sha256','')
    if len(sha)!=64 or not re.fullmatch('[0-9a-f]{64}',sha,re.I):
        raise RuntimeError('Missing local/HF SHA-256 evidence for '+record['path'])
    if meta.get('sha256Checksum') and meta['sha256Checksum'].lower()!=sha.lower():
        raise RuntimeError('Drive service SHA-256 mismatch for '+record['path'])
    return {'size':int(meta['size']),'file_id':meta['id'],'server_md5':meta['md5Checksum'],
            'sha256_source':'local_file_and_pinned_HF','server_sha256_available':bool(meta.get('sha256Checksum'))}


def audit_drive_acl(service, targets:list[tuple[str,str]])->dict:
    """Read-only 0-CU Drive ACL audit (v0.3.1 P1-6).

    Inspects Drive `permissions` for each `(label, file_or_folder_id)` target.
    Fails closed (`RuntimeError('DRIVE_ACL_PUBLIC_EXPOSURE: ...')`) if ANY
    target exposes a `type == 'anyone'` permission (anyoneWithLink / public).
    When all targets are strictly private, returns `status='ACL_PRIVATE_VERIFIED'`.
    """
    if not targets:
        raise RuntimeError('DRIVE_ACL_AUDIT_EMPTY: no managed Drive targets to audit')
    checked=[]
    violations=[]
    seen_ids:set[str]=set()
    for label,target_id in targets:
        if not target_id or target_id in seen_ids:
            continue
        seen_ids.add(target_id)
        meta=service.files().get(
            fileId=target_id,
            fields='id,name,shared,permissions(id,type,role)',
            supportsAllDrives=True,
        ).execute()
        perms=meta.get('permissions')
        if perms is None:
            perm_res=service.permissions().list(
                fileId=target_id,
                fields='permissions(id,type,role)',
                supportsAllDrives=True,
            ).execute()
            perms=perm_res.get('permissions',[])
        public_perms=[
            {'id':p.get('id'),'type':p.get('type'),'role':p.get('role')}
            for p in (perms or [])
            if str(p.get('type','')).lower()=='anyone'
        ]
        entry={
            'label':label,
            'name':meta.get('name') or label,
            'shared':bool(meta.get('shared',False)),
            'permission_types':sorted({str(p.get('type','')).lower() for p in (perms or []) if p.get('type')}),
            'public_anyone':bool(public_perms),
        }
        checked.append(entry)
        if public_perms:
            violations.append({'label':label,'name':entry['name'],'permissions':public_perms})
    if violations:
        names=', '.join(f"{v['label']}({v['name']})" for v in violations)
        raise RuntimeError(
            f"DRIVE_ACL_PUBLIC_EXPOSURE: public 'anyone' permission detected on {names}; "
            'revoke public link sharing before running'
        )
    return {
        'status':'ACL_PRIVATE_VERIFIED',
        'checked_count':len(checked),
        'checked':checked,
        'public_violations':[],
        'no_gpu_created':True,
    }


def audit_local_drive_acls(service, root:Path=ROOT)->dict:
    """Discover locally configured managed Drive folders/files under `root/model_cache`
    (model depot root + 4 weights, ComfyUI source cache folder, and video outputs
    archive folder) and run read-only `audit_drive_acl` (0 CU)."""
    cache_dir=Path(root)/'model_cache'
    targets:list[tuple[str,str]]=[]
    cfg_path=cache_dir/'depot_config.json'
    if cfg_path.is_file():
        cfg=read_json(cfg_path)
        depot_root_id=cfg.get('drive_root_folder_id')
        if depot_root_id:
            targets.append(('model_depot_root',depot_root_id))
    ready_path=cache_dir/'verified_manifest.json'
    if ready_path.is_file():
        manifest=read_json(ready_path)
        if manifest.get('depot_root_id'):
            targets.append(('model_depot_root',manifest['depot_root_id']))
        for item in manifest.get('files',[]) or []:
            fid=item.get('drive_file_id')
            if fid:
                targets.append((f"model:{item.get('path','weight')}",fid))
    comfy_cache_path=cache_dir/'comfyui_source_manifest.json'
    if comfy_cache_path.is_file():
        comfy_manifest=read_json(comfy_cache_path)
        src_folder_id=comfy_manifest.get('drive_folder_id') or comfy_manifest.get('folder_id')
        if src_folder_id:
            targets.append(('comfyui_source_cache_folder',src_folder_id))
    # Also audit the video outputs archive root if it exists on Drive (read-only).
    try:
        out_folders=[
            f for f in list_named(service,'root','Muse-H3-Video-Outputs')
            if f.get('mimeType')==FOLDER_MIME
        ]
        for idx,folder in enumerate(out_folders):
            label='video_outputs_folder' if idx==0 else f'video_outputs_folder_{idx+1}'
            targets.append((label,folder['id']))
    except Exception:
        # Output folder lookup is best-effort if not yet created; depot targets still gate.
        pass
    return audit_drive_acl(service,targets)


def verify_ready(service, account_ok:bool,ready_path:Path=READY)->dict:
    if not account_ok:raise RuntimeError('Colab account != Drive account')
    manifest=read_json(ready_path)
    if manifest.get('status')!='READY' or manifest.get('verification_method')!='drive_v3_server_api':
        raise RuntimeError('Missing/invalid verified manifest; ignore historical CACHE_READY.json')
    files=manifest.get('files',[])
    if {x.get('path') for x in files}!=set(REQUIRED) or len(files)!=4:
        raise RuntimeError('Four model files required, no duplicates')
    root_id=read_json(CFG)['drive_root_folder_id']
    if manifest.get('depot_root_id')!=root_id:raise RuntimeError('Wrong depot folder')
    # Verify provider-side READY payload, not just a persisted local flag.
    remote_ready=read_json(CFG).get('remote_manifest')
    if not isinstance(remote_ready,dict) or not remote_ready.get('drive_file_id'):
        raise RuntimeError('No server-verified READY document on Drive')
    proof=get_file_evidence(service,remote_ready['drive_file_id'])
    if proof.get('trashed') or root_id not in proof.get('parents',[]):
        raise RuntimeError('Drive READY document missing/wrong parent')
    if int(proof.get('size',-1))!=ready_path.stat().st_size or int(proof.get('size',-1))!=remote_ready.get('size'):
        raise RuntimeError('Drive READY document size mismatch')
    ready_md5=hashlib.md5(ready_path.read_bytes()).hexdigest()
    if proof.get('md5Checksum')!=ready_md5 or ready_md5!=remote_ready.get('md5'):
        raise RuntimeError('Drive READY document checksum mismatch')
    for x in files:
        if x.get('hf_revision')!=manifest.get('hf_revision'):raise RuntimeError('Mixed HF revision')
        verify_drive_entry(service,x)
    return {'ready':True,'files':len(files),'bytes':sum(x['size'] for x in files),
            'account_verified':True,'server_metadata_rechecked':True}


def upload_one(service,source:Path,folder_id:str,name:str,rel:str,sha:str,hf_rev:str)->dict:
    """Upload atomically from the Drive API's perspective; never overwrite wrong data."""
    local=file_digests(source)
    existing=list_named(service,folder_id,name)
    if len(existing)>1:raise RuntimeError('Duplicate Drive filenames for '+rel)
    if existing:
        meta=get_file_evidence(service,existing[0]['id'])
        if int(meta.get('size',-1))!=local['size'] or meta.get('md5Checksum','')!=local['md5']:
            raise RuntimeError('Existing Drive file differs; DO NOT overwrite: '+rel)
        file_id=meta['id']
    else:
        from google.auth.transport.requests import AuthorizedSession
        import google.auth
        from drive_resumable import upload_with_persistent_session
        ensure_adc_isolation()
        creds,_=google.auth.default(scopes=[DRIVE_SCOPE])
        auth_session=AuthorizedSession(creds)
        session_key=hashlib.sha256((folder_id+'/'+name+'/'+hf_rev).encode()).hexdigest()[:32]
        sidecar=DEPOT_DIR/'.upload_sessions'/(session_key+'.json')
        response=upload_with_persistent_session(auth_session,source,name,folder_id,rel,hf_rev,sidecar,sha)
        file_id=response['id']
    proof=verify_uploaded_file(service,file_id,source)
    entry={'path':rel,'drive_file_id':file_id,'parent_id':folder_id,
           'size':local['size'],'sha256':sha,'server_md5':proof['server_md5'],
           'server_verified':True,'hf_revision':hf_rev}
    verify_drive_entry(service,entry)
    return entry



def publish_manifest_to_drive(service,root_id:str,manifest:dict)->dict:
    """Durably publish a versioned READY file *on Drive* after four-file checks.

    A local READY is never written before provider-side metadata for this
    exact JSON has been checked. This is an auditable small final commit.
    """
    from googleapiclient.http import MediaFileUpload
    revision=manifest['hf_revision']
    name='READY_'+revision+'.json'
    local=DEPOT_DIR/name
    atomic_json(local,manifest)
    existing=list_named(service,root_id,name)
    if len(existing)>1:raise RuntimeError('Multiple Drive READY documents')
    if existing:
        proof=verify_uploaded_file(service,existing[0]['id'],local)
        file_id=existing[0]['id']
    else:
        media=MediaFileUpload(str(local),mimetype='application/json',resumable=False)
        obj=service.files().create(body={'name':name,'parents':[root_id]},media_body=media,
            fields='id,name,size,md5Checksum,parents',supportsAllDrives=True).execute()
        file_id=obj['id'];proof=verify_uploaded_file(service,file_id,local)
    server=get_file_evidence(service,file_id)
    if root_id not in server.get('parents',[]):
        raise RuntimeError('READY document not in model depot root')
    return {'drive_file_id':file_id,'name':name,'md5':proof['server_md5'],
            'size':proof['size']}



def maybe_copy_verified_legacy_vae(service, source:Path, target_folder:str, name:str, rel:str)->bool:
    """Prefer server-side copy of a historic VAE ONLY when bytes match HF.

    This is an optimization, not trust in the old CACHE_READY.json. Full-Drive
    scope is necessary for access to historical other-app-created files.
    """
    legacy_raw=LEGACY_VAE_IDS.get(rel)
    legacy_id=resolve_drive_id(legacy_raw,root=ROOT) if legacy_raw else None
    if not legacy_id or legacy_id.startswith('EXAMPLE_') or list_named(service,target_folder,name):return False
    try:
        old=get_file_evidence(service,legacy_id)
    except Exception as e:
        status=getattr(getattr(e,'resp',None),'status',None)
        if status in (403,404):
            print('LEGACY_VAE_INACCESSIBLE_NOT_MISSING',rel,flush=True)
            return False
        raise
    digest=file_digests(source)  # source already checked against pinned HF LFS
    if old.get('trashed') or old.get('name')!=name or int(old.get('size',-1))!=digest['size']:
        return False
    if old.get('md5Checksum','').lower()!=digest['md5'].lower():
        return False
    if old.get('sha256Checksum') and old['sha256Checksum'].lower()!=digest['sha256'].lower():
        return False
    res=service.files().copy(fileId=legacy_id,
        body={'name':name,'parents':[target_folder]},
        fields='id,name,size,md5Checksum,parents',supportsAllDrives=True).execute()
    verify_uploaded_file(service,res['id'],source)
    if target_folder not in get_file_evidence(service,res['id']).get('parents',[]):
        raise RuntimeError('Copied historic VAE into wrong depot folder')
    print('LEGACY_VAE_SERVER_COPIED_VERIFIED',rel,flush=True)
    return True


def fetch_pinned_metadata(revision:str)->dict:
    """One HF API call to pin sizes + official LFS SHA-256 for this revision.

    The result is written to PIN_FILE so every later direct download uses
    pinned values and needs no further HF API call. Never prints URLs.
    """
    from huggingface_hub import HfApi
    if not re.fullmatch('[0-9a-f]{40}',revision):
        raise ValueError('HF revision must be a pinned 40-char commit SHA')
    info=HfApi().model_info(repo_id=HF_REPO,revision=revision,files_metadata=True)
    if info.sha!=revision:raise RuntimeError('Resolved HF revision differs')
    siblings={x.rfilename:x for x in info.siblings}
    missing=set(REQUIRED)-set(siblings)
    if missing:raise RuntimeError('HF files missing: '+repr(sorted(missing)))
    files={}
    for rel in sorted(REQUIRED):
        sib=siblings[rel]
        size=getattr(sib,'size',None)
        lfs=getattr(sib,'lfs',None)
        sha=(lfs or {}).get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)
        if not size or not sha or len(str(sha))!=64:
            raise RuntimeError('Pinned HF metadata incomplete for '+rel)
        files[rel]={'size':int(size),'sha256':str(sha).lower(),'lfs':bool(lfs)}
    others=[]
    for rel in sorted(siblings):
        if rel in files:continue
        sib=siblings[rel]
        lfs=getattr(sib,'lfs',None)
        lfs_sha=(lfs or {}).get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)
        blob_sha=getattr(sib,'sha256',None)
        sha=str(lfs_sha or blob_sha or '').lower()
        others.append({'path':rel,'size':int(getattr(sib,'size',0) or 0),
                       'sha256':sha,'lfs':bool(lfs)})
    pin={'repo':HF_REPO,'revision':revision,'files':files,'other_files':others}
    atomic_json(PIN_FILE,pin)
    return pin


def load_pinned_metadata(revision:str)->dict:
    """Reuse a previously fetched pin; fetch once (one HF API call) if absent."""
    if PIN_FILE.exists():
        pin=read_json(PIN_FILE)
        if pin.get('repo')==HF_REPO and pin.get('revision')==revision and set(pin.get('files',{}))==set(REQUIRED):
            return pin
    return fetch_pinned_metadata(revision)


def download_pinned_model(rel:str,expected_size:int,expected_sha256:str,dest_dir:Path,revision:str)->Path:
    """Direct sequential download of one pinned model file. No concurrency.

    Streaming write to `<file>.part` with HTTP Range resume, bounded retries
    and connect/read timeouts. Verifies exact byte count + official LFS
    SHA-256 before the file is ever handed to Drive upload; a mismatch
    deletes the partial file and raises. Never prints URLs (a CDN redirect
    may carry a signature) or credentials. A `.part` file is never READY.
    """
    import requests
    if not re.fullmatch('[0-9a-f]{64}',expected_sha256 or '',re.I):
        raise ValueError('Pinned SHA-256 required for '+rel)
    if not expected_size or expected_size<=0:
        raise ValueError('Pinned byte count required for '+rel)
    dest_dir.mkdir(parents=True,exist_ok=True)
    dest=dest_dir/rel
    dest.parent.mkdir(parents=True,exist_ok=True)
    part=Path(str(dest)+'.part')
    url=f'{HF_RESOLVE_BASE}/{HF_REPO}/resolve/{revision}/{rel}'
    session=requests.Session()
    # Reuse a complete, already-verified local copy instead of re-downloading.
    if dest.is_file() and dest.stat().st_size==expected_size and sha256_file(dest)==expected_sha256.lower():
        return dest
    if dest.is_file():
        dest.unlink()  # stale or corrupt local copy; never upload it
    start=part.stat().st_size if part.is_file() else 0
    if start>expected_size:
        part.unlink();start=0
    if start==expected_size and sha256_file(part)==expected_sha256.lower():
        os.replace(part,dest);return dest
    last_error=None
    for attempt in range(1,DIRECT_MAX_ATTEMPTS+1):
        headers={}
        if start>0:headers['Range']=f'bytes={start}-'
        try:
            with session.get(url,headers=headers,stream=True,
                             timeout=(DIRECT_CONNECT_TIMEOUT,DIRECT_READ_TIMEOUT),
                             allow_redirects=True) as resp:
                status=resp.status_code
                if status==416:
                    # Server says the range is unsatisfiable: the part may
                    # already be complete; verify before deciding.
                    if part.is_file() and part.stat().st_size==expected_size and sha256_file(part)==expected_sha256.lower():
                        os.replace(part,dest);return dest
                    part.unlink(missing_ok=True);start=0;last_error=RuntimeError('HTTP 416 with incomplete part for '+rel)
                    continue
                if status==200 and start>0:
                    # Server ignored Range; restart from zero, do not append.
                    part.unlink(missing_ok=True);start=0;last_error=RuntimeError('Server ignored Range for '+rel)
                    continue
                if status not in (200,206):
                    if status in (429,500,502,503,504):
                        last_error=RuntimeError('Direct download HTTP '+str(status)+' for '+rel)
                        if attempt<DIRECT_MAX_ATTEMPTS:time.sleep(DIRECT_BACKOFF_SEC[min(attempt-1,len(DIRECT_BACKOFF_SEC)-1)]);continue
                    raise RuntimeError('Direct download HTTP '+str(status)+' for '+rel)
                if status==206:
                    cr=resp.headers.get('Content-Range','')
                    m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',cr)
                    if not m or int(m.group(1))!=start or int(m.group(3))!=expected_size:
                        raise RuntimeError('Unexpected Content-Range for '+rel)
                mode='ab' if start>0 else 'wb'
                written=start
                with part.open(mode) as f:
                    for chunk in resp.iter_content(chunk_size=DIRECT_CHUNK):
                        if not chunk:continue
                        f.write(chunk);written+=len(chunk)
                        if written>expected_size:
                            raise RuntimeError('Direct download overran pinned size for '+rel)
                start=written
                if written==expected_size:break
                last_error=RuntimeError('Direct download short read for '+rel)
        except (RuntimeError,ValueError):
            raise
        except Exception as e:
            last_error=e
        if attempt<DIRECT_MAX_ATTEMPTS:time.sleep(DIRECT_BACKOFF_SEC[min(attempt-1,len(DIRECT_BACKOFF_SEC)-1)]);start=part.stat().st_size if part.is_file() else 0
    if part.is_file() and part.stat().st_size==expected_size and sha256_file(part)==expected_sha256.lower():
        os.replace(part,dest)
        print('DIRECT_DOWNLOAD_OK',rel,'bytes',expected_size,flush=True)
        return dest
    if part.is_file():part.unlink(missing_ok=True)
    raise RuntimeError('Direct download failed checksum/size for '+rel+': '+(str(last_error)[:120] if last_error else 'unknown'))


def seed(service,root_id:str,revision:str,source:str='hub'):
    if source not in ('hub','direct'):
        raise ValueError("seed source must be 'hub' or 'direct'")
    if not re.fullmatch('[0-9a-f]{40}',revision):
        raise ValueError('HF revision must be a pinned 40-char commit SHA')
    if source=='hub':
        from huggingface_hub import HfApi,hf_hub_download
        api=HfApi(); info=api.model_info(repo_id=HF_REPO,revision=revision,files_metadata=True)
        if info.sha!=revision:raise RuntimeError('Resolved HF revision differs')
        siblings={x.rfilename:x for x in info.siblings}
        missing=set(REQUIRED)-set(siblings)
        if missing:raise RuntimeError('HF files missing: '+repr(sorted(missing)))
        pin={}
        for rel in REQUIRED:
            lfs=getattr(siblings[rel],'lfs',None)
            hf_sha=(lfs or {}).get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)
            pin[rel]={'size':int(getattr(siblings[rel],'size',0) or 0),'sha256':str(hf_sha or '').lower()}
        def acquire(rel):
            return Path(hf_hub_download(repo_id=HF_REPO,filename=rel,revision=revision,
                                        local_dir=str(DEPOT_DIR/'hf_stage')))
    else:
        pin_data=load_pinned_metadata(revision)
        pin=pin_data['files']
        def acquire(rel):
            return download_pinned_model(rel,pin[rel]['size'],pin[rel]['sha256'],
                                         DEPOT_DIR/'hf_stage',revision)
    required_bytes=sum(pin[rel]['size'] for rel in REQUIRED)
    if required_bytes and shutil.disk_usage(DEPOT_DIR).free<required_bytes*1.3:
        raise RuntimeError('Insufficient Muse home storage; cannot stage all weights')
    entries=[]
    atomic_json(STATE,{'status':'UPLOADING','hf_revision':revision,'verified_files':[]})
    for rel in REQUIRED:
        entry = None
        for attempt in range(1, 4):
            try:
                sub, name = rel.split('/', 1)
                folder_id = ensure_folder(service, root_id, sub, allow_write=True)
                local = acquire(rel)
                sha = sha256_file(local)
                hf_sha = pin[rel]['sha256']
                if not hf_sha or sha.lower() != hf_sha.lower():
                    raise RuntimeError('Hugging Face LFS checksum mismatch for ' + rel)
                if rel in LEGACY_VAE_IDS:
                    try:
                        maybe_copy_verified_legacy_vae(service, local, folder_id, name, rel)
                    except Exception as e:
                        # The legacy copy is an optimization, not a requirement.
                        # If it fails transiently (egress TLS EOF), fall back to
                        # the normal resumable upload instead of killing the file.
                        # A copy that succeeded server-side but lost its response
                        # is still safe: upload_one reuses the existing file after
                        # verifying size + MD5. Non-transient errors still raise.
                        if not _is_transient_drive_error(e):
                            raise
                        print('LEGACY_COPY_UNAVAILABLE_FALLBACK_UPLOAD', rel,
                              type(e).__name__, flush=True)
                entry = upload_one(service, local, folder_id, name, rel, sha, revision)
                break
            except Exception as e:
                if not _is_transient_drive_error(e) or attempt >= 3:
                    raise
                print('TRANSIENT_DRIVE_ERROR_RETRY', rel, 'attempt', attempt,
                      type(e).__name__, flush=True)
                time.sleep(5 * attempt)
                service = drive_service()  # fresh transport; discard stale TLS connection
        entries.append(entry)
        atomic_json(STATE, {'status': 'VERIFYING', 'hf_revision': revision, 'verified_files': entries})
        print('FILE_VERIFIED', rel, 'size', entry['size'], '(no secret)', flush=True)
    # Manifest commit happens only AFTER all 4 remote provider checks.
    manifest = {'status': 'READY', 'verification_method': 'drive_v3_server_api',
                'account_verified': True, 'verified_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'depot_root_id': root_id, 'hf_repo': HF_REPO, 'hf_revision': revision, 'files': entries}
    for attempt in range(1, 4):
        try:
            for e in entries: verify_drive_entry(service, e)
            remote_ready = publish_manifest_to_drive(service, root_id, manifest)
            break
        except Exception as e:
            if not _is_transient_drive_error(e) or attempt >= 3:
                raise
            print('TRANSIENT_DRIVE_ERROR_RETRY manifest attempt', attempt,
                  type(e).__name__, flush=True)
            time.sleep(5 * attempt)
            service = drive_service()
    config=read_json(CFG)
    config['remote_manifest']=remote_ready
    atomic_json(CFG,config)
    atomic_json(READY,manifest)
    atomic_json(STATE,{'status':'READY','hf_revision':revision,'verified_files':len(entries)})
    return {'ready':True,'files':len(entries),'bytes':sum(e['size'] for e in entries)}


def smoke(service,root_id:str)->dict:
    """Idempotent, real Drive API upload/readback canary, zero Colab GPU."""
    folder_id=ensure_folder(service,root_id,'_checks',allow_write=True)
    source=DEPOT_DIR/'smoke_test.bin'
    source.parent.mkdir(parents=True,exist_ok=True)
    payload=b'Muse H3 Drive depot canary v1\n'
    if source.exists() and source.read_bytes()!=payload:
        raise RuntimeError('Unexpected existing smoke source; do not overwrite')
    if not source.exists():source.write_bytes(payload)
    expected_sha=sha256_file(source)
    entry=upload_one(service,source,folder_id,'smoke_test.bin',
                     '_checks/smoke_test.bin',expected_sha,'0'*40)
    verify_drive_entry(service,entry)
    return {'drive_canary_verified':True,'file_id':entry['drive_file_id'],
            'size':entry['size'],'server_md5_verified':True,'no_gpu_created':True}


def load_remote_seed_pin(pin_path:Path=CONFIG_PIN_FILE,revision:str='')->dict:
    """Load and validate locked HF model files for remote-seed (v0.2.0 F5).

    Validates repo, 40-hex commit revision (and exact match with --hf-revision
    when provided), and all 4 required model files with positive byte sizes and
    64-hex SHA-256 digests.
    """
    path=Path(pin_path)
    if not path.is_file():
        raise FileNotFoundError(f'Pinned HF manifest not found: {path}')
    pin=read_json(path)
    if not isinstance(pin,dict):
        raise ValueError('Pinned HF manifest must be a JSON object')
    if pin.get('repo')!=HF_REPO:
        raise ValueError(f"Pinned HF repo mismatch: expected {HF_REPO!r}, got {pin.get('repo')!r}")
    pin_rev=str(pin.get('revision') or '')
    if not re.fullmatch('[0-9a-f]{40}',pin_rev):
        raise ValueError('Pinned HF revision must be a 40-char lowercase hex commit SHA')
    if revision and revision!=pin_rev:
        raise ValueError(f'--hf-revision {revision} does not match pinned revision {pin_rev}')
    files=pin.get('files')
    if not isinstance(files,dict) or set(REQUIRED)-set(files):
        missing=sorted(set(REQUIRED)-set(files or {}))
        raise ValueError('Pinned HF manifest missing required files: '+repr(missing))
    validated={}
    for rel in REQUIRED:
        meta=files[rel]
        if not isinstance(meta,dict):
            raise ValueError('Invalid pinned file entry for '+rel)
        size=meta.get('size')
        sha=str(meta.get('sha256') or '').lower()
        if not isinstance(size,int) or isinstance(size,bool) or size<=0:
            raise ValueError('Invalid pinned size for '+rel)
        if not re.fullmatch('[0-9a-f]{64}',sha):
            raise ValueError('Invalid pinned SHA-256 for '+rel)
        validated[rel]={'size':size,'sha256':sha,'lfs':bool(meta.get('lfs',True))}
    return {'repo':HF_REPO,'revision':pin_rev,'files':validated}


def parse_remote_seed_result(text:str)->dict:
    """Extract REMOTE_SEED_RESULT_JSON from remote worker stdout."""
    for line in (text or '').splitlines():
        idx=line.find(REMOTE_SEED_MARKER)
        if idx>=0:
            raw=line[idx+len(REMOTE_SEED_MARKER):].strip()
            obj=json.loads(raw)
            if isinstance(obj,dict):
                return obj
    raise RuntimeError('Remote seed worker did not emit REMOTE_SEED_RESULT_JSON')


def build_remote_seed_script()->str:
    """Return a self-contained stdlib-only Python script executed inside the
    temporary Colab session for in-cloud HF -> Drive Depot seeding (v0.2.0 F5).

    Security & integrity invariants:
    - Reads short-lived drive.file token from REMOTE_SEED_DIR/drive_token.json
      and unlinks the token file immediately.
    - Never prints tokens or signed CDN URLs.
    - Downloads concurrently via aria2c (when selected/available) or built-in
      multi-threaded HTTP Range + os.pwrite (falling back to single-stream on
      HTTP 200).
    - Verifies exact byte size + SHA-256 before uploading to Drive.
    - Reuses an existing Drive file only on exact size + checksum match; never
      overwrites mismatched data.
    - Deletes staged files after upload to keep Colab disk bounded.
    """
    return '''#!/usr/bin/env python3
from __future__ import annotations
import concurrent.futures, hashlib, json, math, os, shutil, subprocess, sys, time, urllib.parse, urllib.request
from pathlib import Path

WORK = Path("/content/muse_remote_seed")
SPEC_PATH = WORK / "seed_spec.json"
TOKEN_PATH = WORK / "drive_token.json"
DRIVE_API = "https://www.googleapis.com/drive/v3/files"
DRIVE_UPLOAD = "https://www.googleapis.com/upload/drive/v3/files"
UPLOAD_CHUNK = 16 * 1024 * 1024
RANGE_WORKERS = 12
RANGE_BLOCK = 32 * 1024 * 1024


def _load_and_unlink_token() -> str:
    doc = json.loads(TOKEN_PATH.read_text(encoding="utf-8"))
    try:
        TOKEN_PATH.unlink()
    except OSError:
        pass
    tok = doc.get("access_token") or ""
    if not tok:
        raise RuntimeError("Missing short-lived token in remote seed session")
    return tok


def _drive_json(url: str, token: str, method: str = "GET", body: bytes | None = None,
                headers: dict | None = None, timeout: int = 120) -> tuple[int, dict, dict]:
    hdrs = {"Authorization": f"Bearer {token}"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return resp.status, dict(resp.headers.items()), json.loads(raw.decode("utf-8")) if raw else {}


def _list_existing(token: str, folder_id: str, name: str) -> list[dict]:
    safe = name.replace("\\\\", "\\\\\\\\").replace("'", "\\\\'")
    q = f"'{folder_id}' in parents and name = '{safe}' and trashed = false"
    qs = urllib.parse.urlencode({
        "q": q, "spaces": "drive", "pageSize": "10",
        "fields": "files(id,name,size,md5Checksum,sha256Checksum,parents)",
        "supportsAllDrives": "true", "includeItemsFromAllDrives": "true",
    })
    _, _, data = _drive_json(f"{DRIVE_API}?{qs}", token)
    return data.get("files", [])


def _file_digests(path: Path) -> tuple[int, str, str]:
    h_sha = hashlib.sha256()
    h_md5 = hashlib.md5()
    total = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            h_sha.update(chunk)
            h_md5.update(chunk)
    return total, h_sha.hexdigest(), h_md5.hexdigest()


def _pwrite_all(fd: int, data: bytes, offset: int) -> None:
    view = memoryview(data)
    written = 0
    while written < len(view):
        n = os.pwrite(fd, view[written:], offset + written)
        if n <= 0:
            raise OSError("short pwrite")
        written += n


def _download_range_concurrent(url: str, dest: Path, expected_size: int) -> str:
    part = Path(str(dest) + ".part")
    fd = os.open(str(part), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        os.truncate(fd, expected_size)
        block = min(RANGE_BLOCK, max(1024 * 1024, math.ceil(expected_size / RANGE_WORKERS)))
        ranges = []
        off = 0
        while off < expected_size:
            end = min(expected_size - 1, off + block - 1)
            ranges.append((off, end))
            off = end + 1

        def _fetch(span: tuple[int, int]) -> None:
            start, end = span
            for attempt in range(3):
                try:
                    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        if resp.status == 200 and start == 0:
                            raise ValueError("NO_RANGE_SUPPORT")
                        if resp.status != 206:
                            raise RuntimeError(f"HTTP {resp.status} for range")
                        pos = start
                        while True:
                            buf = resp.read(1024 * 1024)
                            if not buf:
                                break
                            _pwrite_all(fd, buf, pos)
                            pos += len(buf)
                        if pos != end + 1:
                            raise RuntimeError("Short range read")
                        return
                except ValueError:
                    raise
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(RANGE_WORKERS, len(ranges))) as ex:
                list(ex.map(_fetch, ranges))
            os.fsync(fd)
        except ValueError:
            os.close(fd)
            fd = -1
            part.unlink(missing_ok=True)
            with urllib.request.urlopen(url, timeout=300) as resp, part.open("wb") as out:
                while True:
                    buf = resp.read(1024 * 1024)
                    if not buf:
                        break
                    out.write(buf)
            os.replace(part, dest)
            return "single_stream"
    finally:
        if fd >= 0:
            os.close(fd)
    os.replace(part, dest)
    return "range_pwrite"


def _download_hf(url: str, dest: Path, expected_size: int, strategy: str) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    use_aria2 = strategy == "aria2c" or (strategy == "auto" and shutil.which("aria2c") is not None)
    if use_aria2:
        cmd = [
            "aria2c", "-x", "16", "-s", "16", "-k", "1M",
            "--file-allocation=none", "--allow-overwrite=true",
            "--console-log-level=warn", "--summary-interval=0",
            "-d", str(dest.parent), "-o", dest.name, url,
        ]
        subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL, timeout=1800)
        return "aria2c"
    return _download_range_concurrent(url, dest, expected_size)


def _upload_resumable(token: str, source: Path, folder_id: str, name: str, rel: str, revision: str, sha: str) -> dict:
    size = source.stat().st_size
    meta = {
        "name": name,
        "parents": [folder_id],
        "appProperties": {"muse_rel": rel, "hf_revision": revision, "sha256": sha},
    }
    init_qs = urllib.parse.urlencode({
        "uploadType": "resumable",
        "supportsAllDrives": "true",
        "fields": "id,name,size,md5Checksum,sha256Checksum,parents",
    })
    init_req = urllib.request.Request(
        f"{DRIVE_UPLOAD}?{init_qs}",
        data=json.dumps(meta).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Type": "application/octet-stream",
            "X-Upload-Content-Length": str(size),
        },
        method="POST",
    )
    with urllib.request.urlopen(init_req, timeout=120) as resp:
        upload_url = resp.headers.get("Location") or resp.headers.get("location")
    if not upload_url:
        raise RuntimeError("Drive resumable upload init returned no Location")
    offset = 0
    with source.open("rb") as f:
        while offset < size:
            chunk = f.read(UPLOAD_CHUNK)
            if not chunk:
                break
            end = offset + len(chunk) - 1
            put_req = urllib.request.Request(
                upload_url,
                data=chunk,
                headers={
                    "Content-Length": str(len(chunk)),
                    "Content-Range": f"bytes {offset}-{end}/{size}",
                    "Content-Type": "application/octet-stream",
                },
                method="PUT",
            )
            try:
                with urllib.request.urlopen(put_req, timeout=300) as resp:
                    if resp.status in (200, 201):
                        return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code == 308:
                    offset = end + 1
                    continue
                raise
            offset = end + 1
    raise RuntimeError("Resumable upload ended without Drive response for " + rel)


def main() -> int:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    token = _load_and_unlink_token()
    repo = spec["repo"]
    revision = spec["revision"]
    strategy = spec.get("downloader", "auto")
    stage_dir = WORK / "stage"
    stage_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for item in spec["files"]:
        rel = item["path"]
        name = item["name"]
        folder_id = item["folder_id"]
        expected_size = int(item["size"])
        expected_sha = str(item["sha256"]).lower()
        existing = _list_existing(token, folder_id, name)
        if len(existing) > 1:
            raise RuntimeError("Duplicate Drive filenames for " + rel)
        if existing:
            meta = existing[0]
            srv_sha = (meta.get("sha256Checksum") or "").lower()
            if int(meta.get("size", -1)) != expected_size or (srv_sha and srv_sha != expected_sha):
                raise RuntimeError("Existing Drive file differs; DO NOT overwrite: " + rel)
            entries.append({
                "path": rel, "drive_file_id": meta["id"], "parent_id": folder_id,
                "size": expected_size, "sha256": expected_sha,
                "server_md5": meta.get("md5Checksum", ""),
                "server_verified": True, "hf_revision": revision, "method": "reused_drive",
            })
            continue
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{rel}"
        local = stage_dir / name
        method = _download_hf(url, local, expected_size, strategy)
        actual_size, actual_sha, actual_md5 = _file_digests(local)
        if actual_size != expected_size or actual_sha.lower() != expected_sha:
            local.unlink(missing_ok=True)
            raise RuntimeError("Remote HF checksum/size mismatch for " + rel)
        uploaded = _upload_resumable(token, local, folder_id, name, rel, revision, expected_sha)
        local.unlink(missing_ok=True)
        if int(uploaded.get("size", -1)) != expected_size or (uploaded.get("md5Checksum") or "").lower() != actual_md5.lower():
            raise RuntimeError("Uploaded Drive checksum/size mismatch for " + rel)
        entries.append({
            "path": rel, "drive_file_id": uploaded["id"], "parent_id": folder_id,
            "size": expected_size, "sha256": expected_sha,
            "server_md5": uploaded["md5Checksum"],
            "server_verified": True, "hf_revision": revision, "method": method,
        })
    print("REMOTE_SEED_RESULT_JSON:" + json.dumps({"revision": revision, "files": entries}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _run_remote_seed_cli(cmd:list[str],colab_auth:str,timeout:int=180,check:bool=True)->dict:
    exe=resolve_colab_executable()
    env={**os.environ,'REQUEST_TIMEOUT':'120','COLAB_CLI_REQUEST_TIMEOUT':'180'}
    proc=subprocess.run([exe,'--auth='+colab_auth,*cmd],capture_output=True,text=True,
                        stdin=subprocess.DEVNULL,timeout=timeout,check=False,env=env)
    res={'args':cmd,'returncode':proc.returncode,'stdout':proc.stdout or '','stderr':proc.stderr or ''}
    if check and proc.returncode!=0:
        raise RuntimeError('Colab CLI failed: '+str(cmd)+' rc='+str(proc.returncode))
    return res


def _wait_remote_seed_watchdog(proc,ready_path:Path,session:str,timeout_sec:float=15.0)->dict:
    deadline=time.monotonic()+timeout_sec
    while time.monotonic()<deadline:
        if proc.poll() is not None:
            raise RuntimeError(f'WATCHDOG_NOT_RUNNING: watchdog exited ({proc.returncode}) for {session}')
        if ready_path.exists():
            try:
                ready=json.loads(ready_path.read_text(encoding='utf-8'))
            except Exception:
                pass
            else:
                if ready.get('session_id')==session:
                    return ready
                raise RuntimeError(f'WATCHDOG_NOT_RUNNING: readiness mismatch for {session}')
        time.sleep(0.25)
    raise RuntimeError(f'WATCHDOG_NOT_RUNNING: watchdog did not signal readiness for {session}')


def _stop_and_verify_remote_seed_session(session:str,colab_auth:str,run_cli_fn=None,
                                         query_state_fn=None,mark_stopped_fn=None)->dict:
    """Stop temporary remote-seed Colab session and verify STOPPED tristate."""
    from h3_session_watchdog import SessionState,mark_session_stopped,query_session_state

    cli_fn=run_cli_fn or _run_remote_seed_cli
    q_fn=query_state_fn or query_session_state
    mark_fn=mark_stopped_fn or mark_session_stopped
    stop_ok=False
    last_state=SessionState.UNKNOWN
    for _ in range(3):
        try:
            state=q_fn(session,colab_auth=colab_auth)
            last_state=state
            if state is SessionState.STOPPED:
                stop_ok=True
                break
            if state is SessionState.ACTIVE:
                cli_fn(['stop','-s',session],colab_auth,timeout=65,check=False)
                state=q_fn(session,colab_auth=colab_auth)
                last_state=state
                if state is SessionState.STOPPED:
                    stop_ok=True
                    break
        except Exception:
            pass
        time.sleep(2)
    record_marked=False
    if stop_ok:
        try:
            record_marked=bool(mark_fn(session,SessionState.STOPPED))
        except Exception:
            record_marked=False
    return {'session':session,'stopped_confirmed':stop_ok,'state':last_state.value,
            'record_marked':record_marked}


def remote_seed(service,root_id:str,pin_path:Path=CONFIG_PIN_FILE,revision:str='',
                colab_auth:str='oauth2',max_wall_sec:int=REMOTE_SEED_DEFAULT_WALL_SEC,
                downloader:str='auto',gpu_type:str='',
                run_cli_fn=None,mint_file_token_fn=None,
                record_session_fn=None,confirm_active_fn=None,
                start_watchdog_fn=None,stop_verify_fn=None)->dict:
    """In-cloud HF -> Drive Depot seeding via a watchdog-protected temporary
    Colab session (v0.2.0 F5).

    Always enforces colab stop + tristate shutdown verification in a finally
    block. Terminates the watchdog only after a confirmed STOPPED verdict.
    Never passes long-lived ADC credentials to Colab (only a short-lived
    drive.file token).
    """
    import h3_drive_token
    from h3_session_watchdog import confirm_session_active,record_session

    if downloader not in ('auto','aria2c','range'):
        raise ValueError("downloader must be 'auto', 'aria2c', or 'range'")
    if max_wall_sec<120 or max_wall_sec>7200:
        raise ValueError('max_wall_sec must be between 120 and 7200')
    pin=load_remote_seed_pin(pin_path,revision=revision)
    rev=pin['revision']
    # Pre-create target depot subfolders locally with full-scope Drive service
    # so the remote short-lived drive.file token writes only into app-owned folders.
    file_specs=[]
    for rel in REQUIRED:
        sub,name=rel.split('/',1)
        folder_id=ensure_folder(service,root_id,sub,allow_write=True)
        file_specs.append({
            'path':rel,'name':name,'folder_id':folder_id,
            'size':pin['files'][rel]['size'],'sha256':pin['files'][rel]['sha256'],
        })
    mint_fn=mint_file_token_fn or h3_drive_token.mint_file_token
    token_doc=mint_fn()
    if not isinstance(token_doc,dict) or not token_doc.get('access_token'):
        raise RuntimeError('Failed to mint short-lived drive.file token for remote-seed')

    cli_fn=run_cli_fn or _run_remote_seed_cli
    rec_fn=record_session_fn or record_session
    confirm_fn=confirm_active_fn or confirm_session_active
    stop_fn=stop_verify_fn or (lambda s: _stop_and_verify_remote_seed_session(s,colab_auth,run_cli_fn=cli_fn))

    session='muse-seed-'+uuid.uuid4().hex[:10]
    deadline_utc=(dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=max_wall_sec)).isoformat()
    allocation_attempted=False
    watchdog_proc=None
    stop_result={'session':session,'stopped_confirmed':False,'state':'UNKNOWN','record_marked':False}
    remote_entries=None

    atomic_json(STATE,{'status':'REMOTE_SEEDING','hf_revision':rev,'session':session,'verified_files':[]})
    with tempfile.TemporaryDirectory(prefix='muse_remote_seed_') as td:
        work=Path(td)
        ready_path=work/'watchdog_ready.json'
        spec_path=work/'seed_spec.json'
        spec_path.write_text(json.dumps({
            'repo':HF_REPO,'revision':rev,'downloader':downloader,'files':file_specs,
        },ensure_ascii=False,indent=2),encoding='utf-8')
        token_path=h3_drive_token.write_token_file(work/'drive_token.json',token_doc)
        worker_path=work/'remote_seed_worker.py'
        worker_path.write_text(build_remote_seed_script(),encoding='utf-8')

        try:
            rec_fn(session,gpu_type or 'CPU',session,deadline_utc,
                   production_task_id='remote-seed',status='allocating')
            new_cmd=['new','-s',session]
            if gpu_type:
                new_cmd+=['--gpu',gpu_type]
            allocation_attempted=True
            cli_fn(new_cmd,colab_auth,timeout=420,check=True)
            if confirm_fn(session) is None:
                raise RuntimeError(f'Pre-registered remote-seed session {session} vanished')
            if start_watchdog_fn is not None:
                watchdog_proc=start_watchdog_fn(session,ready_path)
            else:
                log_fp=open(work/'watchdog.log','a',encoding='utf-8')
                try:
                    watchdog_proc=subprocess.Popen(
                        [sys.executable,str(ROOT/'h3_session_watchdog.py'),'watch',
                         '--session',session,'--colab-auth',colab_auth,
                         '--production-task-id','remote-seed','--ready-file',str(ready_path)],
                        stdout=log_fp,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                finally:
                    log_fp.close()
                _wait_remote_seed_watchdog(watchdog_proc,ready_path,session)

            cli_fn(['exec','-s',session,'-c',f'import os; os.makedirs({REMOTE_SEED_DIR!r}, exist_ok=True)'],
                   colab_auth,timeout=120,check=True)
            for local_f,remote_name in (
                (spec_path,'seed_spec.json'),
                (token_path,'drive_token.json'),
                (worker_path,'remote_seed_worker.py'),
            ):
                cli_fn(['upload','-s',session,str(local_f),f'{REMOTE_SEED_DIR}/{remote_name}'],
                       colab_auth,timeout=120,check=True)
            exec_res=cli_fn(['exec','-s',session,'-f',str(worker_path),'--timeout',str(max_wall_sec)],
                            colab_auth,timeout=max_wall_sec+120,check=True)
            parsed=parse_remote_seed_result(exec_res.get('stdout',''))
            remote_entries=parsed.get('files')
            if not isinstance(remote_entries,list) or {e.get('path') for e in remote_entries}!=set(REQUIRED):
                raise RuntimeError('Remote seed did not return all four required model entries')
        finally:
            if allocation_attempted:
                stop_result=stop_fn(session)
            if watchdog_proc is not None and stop_result.get('stopped_confirmed'):
                try:
                    if watchdog_proc.poll() is None:
                        watchdog_proc.terminate()
                except Exception:
                    pass

    if not stop_result.get('stopped_confirmed'):
        raise RuntimeError(f'Remote seed session {session} stop not verified; refusing READY publish')

    entries=[]
    for e in remote_entries:
        entry={
            'path':e['path'],'drive_file_id':e['drive_file_id'],'parent_id':e['parent_id'],
            'size':int(e['size']),'sha256':str(e['sha256']).lower(),
            'server_md5':e['server_md5'],'server_verified':True,'hf_revision':rev,
        }
        verify_drive_entry(service,entry)
        entries.append(entry)

    manifest={
        'status':'READY','verification_method':'drive_v3_server_api',
        'account_verified':True,'verified_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'depot_root_id':root_id,'hf_repo':HF_REPO,'hf_revision':rev,'files':entries,
    }
    remote_ready=publish_manifest_to_drive(service,root_id,manifest)
    config=read_json(CFG)
    config['remote_manifest']=remote_ready
    atomic_json(CFG,config)
    atomic_json(READY,manifest)
    atomic_json(STATE,{'status':'READY','hf_revision':rev,'verified_files':len(entries)})
    return {
        'ready':True,'remote_seeded':True,'session':session,'stop_verified':True,
        'files':len(entries),'bytes':sum(e['size'] for e in entries),
    }


def cli():
    p=argparse.ArgumentParser(description='Muse independent Drive H3 model depot; remote-seed explicitly allocates a temporary watchdog-protected Colab session')
    p.add_argument('action',choices=('account-check','init','smoke','seed','remote-seed','verify','audit-acl'))
    p.add_argument('--colab-auth',choices=('oauth2','adc'),default='oauth2')
    p.add_argument('--allow-drive-write',action='store_true')
    p.add_argument('--allow-large-transfer',action='store_true')
    p.add_argument('--confirm-remote-seed',action='store_true',
                   help='Explicit confirmation to allocate a temporary Colab session for remote-seed')
    p.add_argument('--hf-revision',default='')
    p.add_argument('--pin-file',default=str(CONFIG_PIN_FILE),
                   help='Locked HF files manifest for remote-seed (default: config/pinned_hf_files.json)')
    p.add_argument('--max-wall-sec',type=int,default=REMOTE_SEED_DEFAULT_WALL_SEC,
                   help='Watchdog wall-time cap in seconds for remote-seed (120..7200)')
    p.add_argument('--downloader',choices=('auto','aria2c','range'),default='auto',
                   help='In-cloud HF download strategy for remote-seed')
    p.add_argument('--gpu-type',default='',
                   help='Optional Colab runtime accelerator for remote-seed (default: CPU)')
    p.add_argument('--source',choices=('hub','direct'),default='hub',
                   help='model download strategy: hub (huggingface_hub) or direct (pinned resolve URL + Range resume)')
    a=p.parse_args()
    if a.action in ('init','smoke','seed','remote-seed') and not a.allow_drive_write:
        p.error('Mutating Drive requires --allow-drive-write')
    if a.action in ('seed','remote-seed') and not a.allow_large_transfer:
        p.error('37GB-class transfer requires --allow-large-transfer')
    if a.action=='remote-seed' and not a.confirm_remote_seed:
        p.error('Remote Colab seeding requires --confirm-remote-seed')
    try:
        s=drive_service()
        ok=account_matches(s,a.colab_auth)
        if not ok:raise RuntimeError('BLOCKED: Drive / Colab Google account mismatch')
        if a.action=='account-check':result={'same_google_account':True,'no_gpu_created':True}
        elif a.action=='init':
            if CFG.exists():raise RuntimeError('Depot already initialized; refuse duplicate creation')
            folder=ensure_folder(s,None,'Muse-H3-Model-Depot',allow_write=True)
            atomic_json(CFG,{'drive_root_folder_id':folder,'scope':'drive','hf_repo':HF_REPO})
            result={'created_new_app_folder':True,'root_id':folder,'no_gpu_created':True}
        elif a.action=='smoke':
            config=read_json(CFG)
            result=smoke(s,config['drive_root_folder_id'])
        elif a.action=='seed':
            config=read_json(CFG)
            result=seed(s,config['drive_root_folder_id'],a.hf_revision,source=a.source)
        elif a.action=='remote-seed':
            config=read_json(CFG)
            result=remote_seed(s,config['drive_root_folder_id'],pin_path=Path(a.pin_file),
                               revision=a.hf_revision,colab_auth=a.colab_auth,
                               max_wall_sec=a.max_wall_sec,downloader=a.downloader,
                               gpu_type=a.gpu_type)
        elif a.action=='audit-acl':
            result=audit_local_drive_acls(s,root=ROOT)
        else:result=verify_ready(s,ok)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except Exception as e:
        # Never log Google OAuth tokens, URLs or raw Google API exceptions:
        # they may contain authorization data in unusual errors.
        print('BLOCKED:',type(e).__name__,str(e)[:180] if isinstance(e,(ValueError,RuntimeError,FileNotFoundError)) else 'See local protected diagnostics',file=sys.stderr)
        return 2
    return 0

if __name__=='__main__':raise SystemExit(cli())
