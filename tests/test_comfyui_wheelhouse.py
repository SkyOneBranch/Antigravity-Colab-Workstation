"""Offline unit tests for F2 ComfyUI wheelhouse cache and cross-account sync.

Covers:
- Python tag & PEP 503 package name normalization
- requirements.txt filtering (excluding Colab preinstalled torch/triton/nvidia-* packages)
- Deterministic wheelhouse tar.gz build, transitive wheel pruning, and local verification
- Manifest entry selection, validation, and controller task descriptor building
- Mocked Drive publish-wheelhouse + verify + CLI --allow-drive-write gate
- scripts/setup_isolated_account.py::_discover_comfyui_cache wheelhouse sync & tamper rejection
- Lazy Drive SDK import invariant (helpers usable on Colab worker without drive_api/drive_depot)
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h3_comfyui_cache as cache_mod
from h3_comfyui_cache import (
    DEFAULT_EXCLUDED_PACKAGES,
    DEFAULT_EXCLUDED_PREFIXES,
    WHEELHOUSE_META_FILENAME,
    build_wheelhouse,
    build_wheelhouse_task_descriptor,
    filter_requirements_file,
    filter_requirements_lines,
    normalize_pkg_name,
    normalize_python_tag,
    publish_wheelhouse,
    select_wheelhouse_entry,
    validate_wheelhouse_entry,
    verify,
    verify_local_wheelhouse,
    wheelhouse_archive_name,
    wheelhouse_manifest_name,
)

SAMPLE_COMMIT = "61e524bdba7a4e193b45c2697f1af83c1beb86d6"
OTHER_COMMIT = "1111111111111111111111111111111111111111"


def _make_fake_wheel(directory: Path, filename: str, payload: bytes = b"wheel-bytes") -> Path:
    p = directory / filename
    p.write_bytes(payload)
    return p


class TestNormalizationAndFiltering(unittest.TestCase):
    def test_normalize_pkg_name(self):
        self.assertEqual(normalize_pkg_name("Torch_Vision"), "torch-vision")
        self.assertEqual(normalize_pkg_name("nvidia.cublas_cu12"), "nvidia-cublas-cu12")
        self.assertEqual(normalize_pkg_name("  safetensors  "), "safetensors")

    def test_normalize_python_tag_variants(self):
        self.assertEqual(normalize_python_tag("3.11"), "py311")
        self.assertEqual(normalize_python_tag("py3.12"), "py312")
        self.assertEqual(normalize_python_tag("py312"), "py312")
        self.assertEqual(normalize_python_tag("cp311"), "py311")
        self.assertEqual(normalize_python_tag("312"), "py312")
        self.assertEqual(normalize_python_tag("3.13"), "py313")
        self.assertEqual(normalize_python_tag("cp313"), "py313")
        expected_cur = f"py{sys.version_info.major}{sys.version_info.minor}"
        self.assertEqual(normalize_python_tag(None), expected_cur)
        self.assertIn("manylinux_2_28_x86_64", cache_mod.DEFAULT_PLATFORMS)

    def test_normalize_python_tag_invalid(self):
        for bad in ("", "py27", "py37", "py999", "not-a-tag"):
            with self.assertRaises(ValueError):
                normalize_python_tag(bad)

    def test_filter_requirements_lines_excludes_preinstalled(self):
        lines = [
            "# ComfyUI deps",
            "",
            "--extra-index-url https://example.invalid/simple",
            "torch>=2.4.0",
            "torch[cuda]==2.4.1 ; sys_platform == 'linux'",
            "TorchVision>=0.19",
            "torchaudio",
            "triton>=3.0.0  # preinstalled",
            "nvidia_cublas_cu12==12.1.3.1",
            "nvidia-cuda-runtime-cu12>=12.0",
            "comfyui-workflow-templates==0.11.103",
            "comfyui-workflow-templates-media-api==0.3.84",
            "comfyui-embedded-docs==0.5.16",
            "comfyui-frontend-package==1.58.5",
            "torchsde>=0.2.6",
            "safetensors>=0.4.2",
            "einops==0.8.0",
            "kornia>=0.7.1 ; python_version >= '3.10'",
        ]
        kept = filter_requirements_lines(lines)
        self.assertIn("# ComfyUI deps", kept)
        self.assertIn("--extra-index-url https://example.invalid/simple", kept)
        self.assertIn("comfyui-frontend-package==1.58.5", kept)
        self.assertIn("torchsde>=0.2.6", kept)
        self.assertIn("safetensors>=0.4.2", kept)
        self.assertIn("einops==0.8.0", kept)
        self.assertIn("kornia>=0.7.1 ; python_version >= '3.10'", kept)
        for excluded_prefix in (
            "torch>=", "torch[cuda]", "TorchVision", "torchaudio", "triton",
            "nvidia_cublas", "nvidia-cuda",
            "comfyui-workflow-templates", "comfyui-embedded-docs",
        ):
            self.assertFalse(any(line.startswith(excluded_prefix) for line in kept))

    def test_filter_requirements_file_deterministic_hash(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            src = base / "requirements.txt"
            src.write_text("torch>=2.4\ntriton\nsafetensors>=0.4\neinops\n", encoding="utf-8")
            dst1 = base / "f1.txt"
            dst2 = base / "f2.txt"
            r1 = filter_requirements_file(src, dst1)
            r2 = filter_requirements_file(src, dst2)
            self.assertEqual(r1["sha256"], r2["sha256"])
            self.assertEqual(r1["kept"], ["safetensors>=0.4", "einops"])
            self.assertEqual(r1["excluded"], ["torch>=2.4", "triton"])
            self.assertEqual(dst1.read_text(encoding="utf-8"), "safetensors>=0.4\neinops\n")


class TestBuildAndVerifyWheelhouse(unittest.TestCase):
    def test_build_wheelhouse_deterministic_and_prunes_excluded_wheels(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            req = base / "requirements.txt"
            req.write_text("torch>=2.4\nsafetensors>=0.4\neinops>=0.8\n", encoding="utf-8")
            wheels_dir = base / "wheels"
            wheels_dir.mkdir()
            _make_fake_wheel(wheels_dir, "safetensors-0.4.5-cp312-cp312-manylinux_2_17_x86_64.whl", b"safe")
            _make_fake_wheel(wheels_dir, "einops-0.8.0-py3-none-any.whl", b"einops")
            # Simulate transitive excluded wheels pulled in by pip download.
            _make_fake_wheel(wheels_dir, "torch-2.5.0-cp312-cp312-manylinux1_x86_64.whl", b"huge-torch")
            _make_fake_wheel(wheels_dir, "nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl", b"huge-cuda")

            out1 = base / "out1"
            out2 = base / "out2"
            b1 = build_wheelhouse(SAMPLE_COMMIT, requirements_path=req, python_tag="3.12", out_dir=out1, wheels_dir=wheels_dir)
            b2 = build_wheelhouse(SAMPLE_COMMIT, requirements_path=req, python_tag="cp312", out_dir=out2, wheels_dir=wheels_dir)

            self.assertEqual(b1["manifest"]["sha256"], b2["manifest"]["sha256"])
            self.assertEqual(b1["manifest"]["wheel_count"], 2)
            self.assertEqual(b1["manifest"]["python_tag"], "py312")
            self.assertEqual(b1["manifest"]["abi"], "cp312")
            self.assertEqual(
                sorted(b1["pruned_wheels"]),
                [
                    "nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl",
                    "torch-2.5.0-cp312-cp312-manylinux1_x86_64.whl",
                ],
            )
            checked = verify_local_wheelhouse(
                b1["archive_path"], b1["manifest"]["sha256"], SAMPLE_COMMIT, "py312"
            )
            self.assertEqual(checked["wheel_count"], 2)

    def test_build_wheelhouse_invokes_pip_download_with_target_platform_flags(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            req = base / "requirements.txt"
            req.write_text("torch\neinops>=0.8\n", encoding="utf-8")
            seen_cmds: list[list[str]] = []

            def fake_pip(cmd: list[str]) -> None:
                seen_cmds.append(list(cmd))
                d_idx = cmd.index("-d") + 1
                dl_dir = Path(cmd[d_idx])
                _make_fake_wheel(dl_dir, "einops-0.8.0-py3-none-any.whl", b"einops")

            built = build_wheelhouse(
                SAMPLE_COMMIT,
                requirements_path=req,
                python_tag="py311",
                out_dir=base / "out",
                platforms=("manylinux2014_x86_64",),
                extra_index_url="https://example.invalid/whl",
                pip_runner=fake_pip,
            )
            self.assertEqual(len(seen_cmds), 1)
            cmd = seen_cmds[0]
            self.assertIn("--only-binary=:all:", cmd)
            self.assertIn("--python-version", cmd)
            self.assertEqual(cmd[cmd.index("--python-version") + 1], "3.11")
            self.assertEqual(cmd[cmd.index("--abi") + 1], "cp311")
            self.assertEqual(cmd[cmd.index("--platform") + 1], "manylinux2014_x86_64")
            self.assertEqual(cmd[cmd.index("--extra-index-url") + 1], "https://example.invalid/whl")
            self.assertEqual(built["manifest"]["python_tag"], "py311")

    def test_verify_local_wheelhouse_rejects_mismatches_and_smuggled_excluded_wheel(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            req = base / "requirements.txt"
            req.write_text("einops>=0.8\n", encoding="utf-8")
            wheels_dir = base / "wheels"
            wheels_dir.mkdir()
            _make_fake_wheel(wheels_dir, "einops-0.8.0-py3-none-any.whl", b"einops")
            built = build_wheelhouse(SAMPLE_COMMIT, requirements_path=req, python_tag="py312", out_dir=base / "out", wheels_dir=wheels_dir)
            archive = built["archive_path"]
            sha = built["manifest"]["sha256"]

            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                verify_local_wheelhouse(archive, "0" * 64, SAMPLE_COMMIT, "py312")
            with self.assertRaisesRegex(RuntimeError, "commit mismatch"):
                verify_local_wheelhouse(archive, sha, OTHER_COMMIT, "py312")
            with self.assertRaisesRegex(RuntimeError, "python_tag mismatch"):
                verify_local_wheelhouse(archive, sha, SAMPLE_COMMIT, "py311")

            # Craft an archive smuggling torch-*.whl despite metadata listing it.
            smuggled = base / "smuggled.tar.gz"
            meta = {
                "commit": SAMPLE_COMMIT,
                "python_tag": "py312",
                "excluded_packages": list(DEFAULT_EXCLUDED_PACKAGES),
                "excluded_prefixes": list(DEFAULT_EXCLUDED_PREFIXES),
                "wheel_count": 1,
                "wheels": ["torch-2.5.0-cp312-cp312-manylinux1_x86_64.whl"],
            }
            meta_bytes = json.dumps(meta).encode("utf-8")
            with tarfile.open(smuggled, "w:gz") as tf:
                mi = tarfile.TarInfo(f"wheelhouse/{WHEELHOUSE_META_FILENAME}")
                mi.size = len(meta_bytes)
                tf.addfile(mi, io.BytesIO(meta_bytes))
                wi = tarfile.TarInfo("wheelhouse/torch-2.5.0-cp312-cp312-manylinux1_x86_64.whl")
                wi.size = 4
                tf.addfile(wi, io.BytesIO(b"evil"))
            smuggled_sha = hashlib.sha256(smuggled.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError, "excluded wheel"):
                verify_local_wheelhouse(smuggled, smuggled_sha, SAMPLE_COMMIT, "py312")


class TestManifestHelpers(unittest.TestCase):
    def _sample_manifest(self) -> dict:
        return {
            "commit": SAMPLE_COMMIT,
            "archive_name": f"comfyui-{SAMPLE_COMMIT}.tar.gz",
            "wheelhouses": {
                "py311": {
                    "archive_name": wheelhouse_archive_name(SAMPLE_COMMIT, "py311"),
                    "python_tag": "py311",
                    "commit": SAMPLE_COMMIT,
                    "size_bytes": 12345,
                    "sha256": "a" * 64,
                    "file_id": "drive-wh-311",
                    "server_md5": "md5-311",
                    "excluded_packages": list(DEFAULT_EXCLUDED_PACKAGES),
                    "excluded_prefixes": list(DEFAULT_EXCLUDED_PREFIXES),
                },
                "py312": {
                    "archive_name": wheelhouse_archive_name(SAMPLE_COMMIT, "py312"),
                    "python_tag": "py312",
                    "commit": SAMPLE_COMMIT,
                    "size_bytes": 23456,
                    "sha256": "b" * 64,
                    "file_id": "drive-wh-312",
                    "server_md5": "md5-312",
                    "excluded_packages": list(DEFAULT_EXCLUDED_PACKAGES),
                    "excluded_prefixes": list(DEFAULT_EXCLUDED_PREFIXES),
                },
            },
        }

    def test_select_and_validate_wheelhouse_entry(self):
        m = self._sample_manifest()
        e312 = select_wheelhouse_entry(m, SAMPLE_COMMIT, "3.12")
        self.assertIsNotNone(e312)
        self.assertEqual(e312["file_id"], "drive-wh-312")
        self.assertIsNone(select_wheelhouse_entry(m, SAMPLE_COMMIT, "py310"))
        self.assertIsNone(select_wheelhouse_entry(m, OTHER_COMMIT, "py312"))
        self.assertIsNone(select_wheelhouse_entry(None, SAMPLE_COMMIT, "py312"))

        ok, reason = validate_wheelhouse_entry(e312, SAMPLE_COMMIT, "py312")
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

        bad = dict(e312, sha256="not-hex")
        ok2, reason2 = validate_wheelhouse_entry(bad, SAMPLE_COMMIT, "py312")
        self.assertFalse(ok2)
        self.assertEqual(reason2, "invalid_sha256")

    def test_build_wheelhouse_task_descriptor(self):
        m = self._sample_manifest()
        desc = build_wheelhouse_task_descriptor(m, SAMPLE_COMMIT)
        self.assertIsNotNone(desc)
        self.assertEqual(sorted(desc.keys()), ["py311", "py312"])
        self.assertEqual(desc["py312"]["file_id"], "drive-wh-312")
        self.assertEqual(desc["py312"]["size"], 23456)
        self.assertEqual(desc["py312"]["archive"], wheelhouse_archive_name(SAMPLE_COMMIT, "py312"))
        self.assertEqual(desc["py312"]["excluded_packages"], list(DEFAULT_EXCLUDED_PACKAGES))

        # Entry without file_id is omitted; empty result returns None.
        m_no_id = {"commit": SAMPLE_COMMIT, "wheelhouses": {"py312": dict(m["wheelhouses"]["py312"], file_id="")}}
        self.assertIsNone(build_wheelhouse_task_descriptor(m_no_id, SAMPLE_COMMIT))
        self.assertIsNone(build_wheelhouse_task_descriptor({"commit": SAMPLE_COMMIT}, SAMPLE_COMMIT))


class TestPublishAndCli(unittest.TestCase):
    def test_cli_publish_wheelhouse_requires_allow_drive_write(self):
        with tempfile.TemporaryDirectory() as d:
            req = Path(d) / "requirements.txt"
            req.write_text("einops>=0.8\n", encoding="utf-8")
            with self.assertRaises(SystemExit) as cm:
                cache_mod.cli([
                    "publish-wheelhouse",
                    "--commit", SAMPLE_COMMIT,
                    "--requirements", str(req),
                    "--colab-auth", "oauth2",
                ])
            self.assertEqual(cm.exception.code, 2)

    def test_publish_wheelhouse_merges_without_clobbering_source_or_other_tags(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            staging = base / "staging"
            local_manifest = base / "comfyui_source_manifest.json"
            # Pre-existing source manifest with py311 already published.
            local_manifest.write_text(json.dumps({
                "repo": cache_mod.COMFYUI_REPO_URL,
                "commit": SAMPLE_COMMIT,
                "archive_name": f"comfyui-{SAMPLE_COMMIT}.tar.gz",
                "sha256": "c" * 64,
                "size_bytes": 999,
                "drive_folder_id": "folder-1",
                "drive_files": {
                    f"comfyui-{SAMPLE_COMMIT}.tar.gz": {
                        "id": "src-tar-id",
                        "size": 999,
                        "sha256": "c" * 64,
                        "server_md5": "src-md5",
                    }
                },
                "wheelhouses": {
                    "py311": {
                        "archive_name": wheelhouse_archive_name(SAMPLE_COMMIT, "py311"),
                        "python_tag": "py311",
                        "commit": SAMPLE_COMMIT,
                        "size_bytes": 555,
                        "sha256": "d" * 64,
                        "file_id": "wh-311-id",
                    }
                },
            }), encoding="utf-8")

            req = base / "requirements.txt"
            req.write_text("einops>=0.8\n", encoding="utf-8")
            wheels_dir = base / "wheels"
            wheels_dir.mkdir()
            _make_fake_wheel(wheels_dir, "einops-0.8.0-py3-none-any.whl", b"einops")

            uploaded: list[str] = []

            def fake_upload_one(service, local_path, folder_id, name, rel, sha, rev):
                uploaded.append(rel)
                dig = cache_mod._file_digests(local_path)
                return {
                    "drive_file_id": f"id-{rel}",
                    "size": dig["size"],
                    "sha256": dig["sha256"],
                    "server_md5": dig["md5"],
                }

            def fake_verify_uploaded(service, file_id, local_path):
                dig = cache_mod._file_digests(local_path)
                return {"size": dig["size"], "sha256": dig["sha256"], "md5": dig["md5"]}

            with mock.patch.object(cache_mod, "STAGING", staging), \
                 mock.patch.object(cache_mod, "LOCAL_MANIFEST", local_manifest), \
                 mock.patch("drive_depot.drive_service", return_value=object()), \
                 mock.patch("drive_depot.account_matches", return_value=True), \
                 mock.patch("drive_depot.ensure_folder", return_value="folder-1"), \
                 mock.patch("drive_depot.upload_one", side_effect=fake_upload_one), \
                 mock.patch("drive_api.verify_uploaded_file", side_effect=fake_verify_uploaded):
                merged = publish_wheelhouse(
                    SAMPLE_COMMIT,
                    "oauth2",
                    requirements_path=req,
                    python_tag="py312",
                    wheels_dir=wheels_dir,
                )
                # Source fields and py311 preserved alongside new py312.
                self.assertEqual(merged["archive_name"], f"comfyui-{SAMPLE_COMMIT}.tar.gz")
                self.assertIn("py311", merged["wheelhouses"])
                self.assertIn("py312", merged["wheelhouses"])
                wh312 = merged["wheelhouses"]["py312"]
                self.assertEqual(wh312["file_id"], f"id-{wh312['archive_name']}")
                self.assertEqual(len(uploaded), 3)

                # Verify read-only verify() checks both staged files and remote-only synced files.
                fake_meta = {
                    "id": "src-tar-id",
                    "size": "999",
                    "md5Checksum": "src-md5",
                    "sha256Checksum": "c" * 64,
                    "trashed": False,
                }
                with mock.patch("drive_api.get_file_evidence", return_value=fake_meta):
                    v_res = verify("oauth2")
                self.assertIn(f"comfyui-{SAMPLE_COMMIT}.tar.gz", v_res["files"])
                self.assertTrue(v_res["files"][f"comfyui-{SAMPLE_COMMIT}.tar.gz"]["server_verified"])
                self.assertIn(wh312["archive_name"], v_res["files"])
                self.assertTrue(v_res["files"][wh312["archive_name"]]["server_verified"])


class TestSetupIsolatedAccountWheelhouseSync(unittest.TestCase):
    def _build_fake_drive_svc(self, children: list[dict], content_by_id: dict[str, bytes]):
        class _Exec:
            def __init__(self, value):
                self._value = value

            def execute(self):
                return self._value

        class _Files:
            def get(self, fileId, **kwargs):
                return _Exec({
                    "id": fileId,
                    "name": cache_mod.CACHE_FOLDER_NAME,
                    "mimeType": "application/vnd.google-apps.folder",
                    "trashed": False,
                })

            def list(self, q, **kwargs):
                if "mimeType" in q:
                    return _Exec({"files": [{"id": "folder-cache", "name": cache_mod.CACHE_FOLDER_NAME}]})
                return _Exec({"files": children})

        class _Svc:
            def files(self):
                return _Files()

        return _Svc(), lambda svc, fid: content_by_id[fid]

    def _source_fixture(self):
        src_name = f"comfyui-{SAMPLE_COMMIT}.tar.gz"
        src_sha = "1" * 64
        src_manifest = {
            "repo": cache_mod.COMFYUI_REPO_URL,
            "commit": SAMPLE_COMMIT,
            "archive_name": src_name,
            "sha256": src_sha,
            "size_bytes": 4096,
        }
        src_manifest_bytes = json.dumps(src_manifest).encode("utf-8")
        src_sha_bytes = f"{src_sha}  {src_name}\n".encode("utf-8")
        children = [
            {"id": "f-src-man", "name": "comfyui-source-manifest.json", "size": str(len(src_manifest_bytes)), "md5Checksum": "m1"},
            {"id": "f-src-tar", "name": src_name, "size": "4096", "md5Checksum": "m2"},
            {"id": "f-src-sha", "name": f"{src_name}.sha256", "size": str(len(src_sha_bytes)), "md5Checksum": "m3"},
        ]
        blobs = {
            "f-src-man": src_manifest_bytes,
            "f-src-sha": src_sha_bytes,
        }
        return children, blobs

    def test_discover_comfyui_cache_source_only_and_multi_wheelhouse(self):
        import scripts.setup_isolated_account as setup_mod

        # 1) Source-only backwards compatibility
        children, blobs = self._source_fixture()
        svc, dl = self._build_fake_drive_svc(children, blobs)
        with tempfile.TemporaryDirectory() as d:
            out_manifest = Path(d) / "comfyui_source_manifest.json"
            with mock.patch.object(cache_mod, "LOCAL_MANIFEST", out_manifest), \
                 mock.patch.object(setup_mod, "_ensure_isolated_env"), \
                 mock.patch.object(setup_mod, "_download_drive_bytes", side_effect=dl), \
                 mock.patch("drive_depot.drive_service", return_value=svc):
                res = setup_mod._discover_comfyui_cache()
            self.assertTrue(res["comfyui_cache_found"])
            self.assertEqual(res["wheelhouses"], {})
            saved = json.loads(out_manifest.read_text(encoding="utf-8"))
            self.assertNotIn("wheelhouses", saved)

        # 2) Source + py311 + py312 + stale other-commit wheelhouse (ignored)
        children, blobs = self._source_fixture()
        for pytag, commit, sha_char, sz in (
            ("py311", SAMPLE_COMMIT, "a", 8192),
            ("py312", SAMPLE_COMMIT, "b", 9216),
            ("py312", OTHER_COMMIT, "f", 1111),
        ):
            wh_arch = wheelhouse_archive_name(commit, pytag)
            wh_man_name = wheelhouse_manifest_name(commit, pytag)
            wh_sha = sha_char * 64
            wh_man = {
                "commit": commit,
                "python_tag": pytag,
                "archive_name": wh_arch,
                "sha256": wh_sha,
                "size_bytes": sz,
                "excluded_packages": list(DEFAULT_EXCLUDED_PACKAGES),
                "excluded_prefixes": list(DEFAULT_EXCLUDED_PREFIXES),
            }
            man_b = json.dumps(wh_man).encode("utf-8")
            sha_b = f"{wh_sha}  {wh_arch}\n".encode("utf-8")
            tag_id = f"{commit[:6]}-{pytag}"
            children.extend([
                {"id": f"m-{tag_id}", "name": wh_man_name, "size": str(len(man_b)), "md5Checksum": f"mm-{tag_id}"},
                {"id": f"t-{tag_id}", "name": wh_arch, "size": str(sz), "md5Checksum": f"mt-{tag_id}"},
                {"id": f"s-{tag_id}", "name": f"{wh_arch}.sha256", "size": str(len(sha_b)), "md5Checksum": f"ms-{tag_id}"},
            ])
            blobs[f"m-{tag_id}"] = man_b
            blobs[f"s-{tag_id}"] = sha_b

        svc2, dl2 = self._build_fake_drive_svc(children, blobs)
        with tempfile.TemporaryDirectory() as d:
            out_manifest = Path(d) / "comfyui_source_manifest.json"
            with mock.patch.object(cache_mod, "LOCAL_MANIFEST", out_manifest), \
                 mock.patch.object(setup_mod, "_ensure_isolated_env"), \
                 mock.patch.object(setup_mod, "_download_drive_bytes", side_effect=dl2), \
                 mock.patch("drive_depot.drive_service", return_value=svc2):
                res2 = setup_mod._discover_comfyui_cache()
            self.assertEqual(sorted(res2["wheelhouses"].keys()), ["py311", "py312"])
            saved2 = json.loads(out_manifest.read_text(encoding="utf-8"))
            self.assertEqual(sorted(saved2["wheelhouses"].keys()), ["py311", "py312"])
            self.assertEqual(saved2["wheelhouses"]["py312"]["file_id"], f"t-{SAMPLE_COMMIT[:6]}-py312")
            self.assertEqual(saved2["wheelhouses"]["py311"]["size_bytes"], 8192)
            # Task descriptor helper consumes synced manifest cleanly.
            desc = build_wheelhouse_task_descriptor(saved2, SAMPLE_COMMIT)
            self.assertEqual(sorted(desc.keys()), ["py311", "py312"])

    def test_discover_comfyui_cache_rejects_tampered_wheelhouse(self):
        import scripts.setup_isolated_account as setup_mod

        children, blobs = self._source_fixture()
        wh_arch = wheelhouse_archive_name(SAMPLE_COMMIT, "py312")
        wh_man_name = wheelhouse_manifest_name(SAMPLE_COMMIT, "py312")
        wh_man = {
            "commit": SAMPLE_COMMIT,
            "python_tag": "py312",
            "archive_name": wh_arch,
            "sha256": "b" * 64,
            "size_bytes": 5000,
        }
        man_b = json.dumps(wh_man).encode("utf-8")
        bad_sha_b = f"{'0' * 64}  {wh_arch}\n".encode("utf-8")
        children.extend([
            {"id": "wm", "name": wh_man_name, "size": str(len(man_b)), "md5Checksum": "m1"},
            {"id": "wt", "name": wh_arch, "size": "5000", "md5Checksum": "m2"},
            {"id": "ws", "name": f"{wh_arch}.sha256", "size": str(len(bad_sha_b)), "md5Checksum": "m3"},
        ])
        blobs["wm"] = man_b
        blobs["ws"] = bad_sha_b

        svc, dl = self._build_fake_drive_svc(children, blobs)
        with tempfile.TemporaryDirectory() as d:
            out_manifest = Path(d) / "comfyui_source_manifest.json"
            with mock.patch.object(cache_mod, "LOCAL_MANIFEST", out_manifest), \
                 mock.patch.object(setup_mod, "_ensure_isolated_env"), \
                 mock.patch.object(setup_mod, "_download_drive_bytes", side_effect=dl), \
                 mock.patch("drive_depot.drive_service", return_value=svc):
                with self.assertRaisesRegex(RuntimeError, "Wheelhouse cache sha256 mismatch"):
                    setup_mod._discover_comfyui_cache()


class TestLazyDriveImports(unittest.TestCase):
    def test_helpers_work_without_drive_modules(self):
        code = (
            "import sys; "
            "sys.modules['drive_api'] = None; "
            "sys.modules['drive_depot'] = None; "
            "import h3_comfyui_cache as m; "
            "assert m.normalize_python_tag('3.12') == 'py312'; "
            "assert m.filter_requirements_lines(['torch>=2.0', 'einops']) == ['einops']; "
            "print('OK')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(proc.stdout.strip(), "OK")


if __name__ == "__main__":
    unittest.main()
