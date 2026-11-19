"""Offline unit tests for drive_depot.py remote-seed (v0.2.0 F5).

0 CU, zero live Colab sessions, zero live Drive calls.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import drive_depot as d
from h3_session_watchdog import SessionState


def _make_pin_file(path: Path, revision: str = "a" * 40) -> Path:
    files = {}
    for idx, rel in enumerate(d.REQUIRED):
        files[rel] = {
            "lfs": True,
            "size": 1024 * (idx + 1),
            "sha256": f"{idx + 1:x}" * 64,
        }
    path.write_text(json.dumps({
        "repo": d.HF_REPO,
        "revision": revision,
        "files": files,
    }), encoding="utf-8")
    return path


def _fake_remote_entries(revision: str = "a" * 40) -> list[dict]:
    entries = []
    for idx, rel in enumerate(d.REQUIRED):
        entries.append({
            "path": rel,
            "drive_file_id": f"drv-{idx}",
            "parent_id": f"folder-{rel.split('/')[0]}",
            "size": 1024 * (idx + 1),
            "sha256": f"{idx + 1:x}" * 64,
            "server_md5": f"{idx + 1:x}" * 32,
            "server_verified": True,
            "hf_revision": revision,
            "method": "range_pwrite",
        })
    return entries


def _short_token_doc() -> dict:
    # Keep token fixture short to satisfy secret_scan credential-assign rule.
    import h3_drive_token

    tok = "tok1"
    return {
        "access_token": tok,
        "token_id": h3_drive_token.token_id_for(tok),
        "scope": h3_drive_token.WRITE_FILE_SCOPE,
        "expires_at_utc": "2099-01-01T00:00:00Z",
    }


class RemoteSeedGateAndPinTest(unittest.TestCase):
    def test_cli_requires_all_three_gate_flags(self):
        cases = [
            ["drive_depot.py", "remote-seed"],
            ["drive_depot.py", "remote-seed", "--allow-drive-write"],
            ["drive_depot.py", "remote-seed", "--allow-drive-write", "--allow-large-transfer"],
            ["drive_depot.py", "remote-seed", "--allow-drive-write", "--confirm-remote-seed"],
            ["drive_depot.py", "remote-seed", "--allow-large-transfer", "--confirm-remote-seed"],
        ]
        for argv in cases:
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit) as cm:
                    d.cli()
            self.assertEqual(cm.exception.code, 2, f"Expected exit 2 for {argv}")

    def test_load_remote_seed_pin_validates_repo_config_file(self):
        pin = d.load_remote_seed_pin(d.CONFIG_PIN_FILE)
        self.assertEqual(pin["repo"], d.HF_REPO)
        self.assertEqual(len(pin["revision"]), 40)
        self.assertEqual(set(pin["files"].keys()), set(d.REQUIRED))

    def test_load_remote_seed_pin_rejects_bad_revision_or_missing_files(self):
        with tempfile.TemporaryDirectory() as td:
            p = _make_pin_file(Path(td) / "pin.json", revision="b" * 40)
            with self.assertRaisesRegex(ValueError, "does not match"):
                d.load_remote_seed_pin(p, revision="c" * 40)

            bad = Path(td) / "bad.json"
            bad.write_text(json.dumps({"repo": d.HF_REPO, "revision": "short", "files": {}}))
            with self.assertRaisesRegex(ValueError, "40-char"):
                d.load_remote_seed_pin(bad)

    def test_build_remote_seed_script_compiles_cleanly(self):
        src = d.build_remote_seed_script()
        compile(src, "remote_seed_worker.py", "exec")
        self.assertIn("aria2c", src)
        self.assertIn("os.pwrite", src)
        self.assertIn("REMOTE_SEED_RESULT_JSON:", src)


class RemoteSeedLifecycleTest(unittest.TestCase):
    def _run_remote_seed_mocked(self, exec_side_effect=None, stop_confirmed: bool = True):
        rev = "a" * 40
        entries = _fake_remote_entries(rev)
        cli_calls = []
        stop_calls = []
        fake_watchdog = mock.MagicMock()
        fake_watchdog.poll.return_value = None

        def _fake_cli(cmd, auth, timeout=180, check=True):
            cli_calls.append(cmd)
            if cmd[0] == "exec" and "-f" in cmd:
                if exec_side_effect is not None:
                    raise exec_side_effect
                return {
                    "returncode": 0,
                    "stdout": "REMOTE_SEED_RESULT_JSON:" + json.dumps({"revision": rev, "files": entries}),
                    "stderr": "",
                }
            return {"returncode": 0, "stdout": "", "stderr": ""}

        def _fake_stop(sess):
            stop_calls.append(sess)
            return {
                "session": sess,
                "stopped_confirmed": stop_confirmed,
                "state": "STOPPED" if stop_confirmed else "UNKNOWN",
                "record_marked": stop_confirmed,
            }

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            pin_path = _make_pin_file(td_path / "pinned.json", revision=rev)
            cfg_path = td_path / "depot_config.json"
            cfg_path.write_text(json.dumps({"drive_root_folder_id": "root-1"}), encoding="utf-8")
            ready_path = td_path / "verified_manifest.json"
            state_path = td_path / "depot_state.json"

            with (mock.patch.object(d, "DEPOT_DIR", td_path),
                  mock.patch.object(d, "CFG", cfg_path),
                  mock.patch.object(d, "READY", ready_path),
                  mock.patch.object(d, "STATE", state_path),
                  mock.patch.object(d, "ensure_folder", side_effect=lambda svc, r, sub, allow_write: f"folder-{sub}"),
                  mock.patch.object(d, "verify_drive_entry", return_value={"size": 1024}),
                  mock.patch.object(d, "publish_manifest_to_drive", return_value={"drive_file_id": "ready-1"})):
                result = d.remote_seed(
                    mock.MagicMock(),
                    "root-1",
                    pin_path=pin_path,
                    revision=rev,
                    colab_auth="oauth2",
                    max_wall_sec=600,
                    downloader="range",
                    run_cli_fn=_fake_cli,
                    mint_file_token_fn=_short_token_doc,
                    record_session_fn=mock.MagicMock(),
                    confirm_active_fn=mock.MagicMock(return_value={"status": "active"}),
                    start_watchdog_fn=lambda sess, rp: fake_watchdog,
                    stop_verify_fn=_fake_stop,
                )
            return result, cli_calls, stop_calls, fake_watchdog, ready_path.exists()

    def test_remote_seed_happy_path_stops_session_and_publishes_ready(self):
        result, cli_calls, stop_calls, watchdog, ready_exists = self._run_remote_seed_mocked()
        self.assertTrue(result["ready"])
        self.assertTrue(result["remote_seeded"])
        self.assertTrue(result["stop_verified"])
        self.assertEqual(result["files"], 4)
        self.assertEqual(len(stop_calls), 1)
        self.assertTrue(any(c[0] == "new" for c in cli_calls))
        watchdog.terminate.assert_called_once()
        self.assertTrue(ready_exists)

    def test_remote_seed_failure_still_enforces_finally_stop(self):
        stop_calls = []
        fake_watchdog = mock.MagicMock()
        fake_watchdog.poll.return_value = None
        rev = "a" * 40

        def _failing_cli(cmd, auth, timeout=180, check=True):
            if cmd[0] == "exec" and "-f" in cmd:
                raise RuntimeError("remote download boom")
            return {"returncode": 0, "stdout": "", "stderr": ""}

        def _fake_stop(sess):
            stop_calls.append(sess)
            return {"session": sess, "stopped_confirmed": True, "state": "STOPPED", "record_marked": True}

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            pin_path = _make_pin_file(td_path / "pinned.json", revision=rev)
            cfg_path = td_path / "depot_config.json"
            cfg_path.write_text(json.dumps({"drive_root_folder_id": "root-1"}), encoding="utf-8")
            with (mock.patch.object(d, "DEPOT_DIR", td_path),
                  mock.patch.object(d, "CFG", cfg_path),
                  mock.patch.object(d, "READY", td_path / "verified_manifest.json"),
                  mock.patch.object(d, "STATE", td_path / "depot_state.json"),
                  mock.patch.object(d, "ensure_folder", side_effect=lambda svc, r, sub, allow_write: f"folder-{sub}")):
                with self.assertRaisesRegex(RuntimeError, "remote download boom"):
                    d.remote_seed(
                        mock.MagicMock(),
                        "root-1",
                        pin_path=pin_path,
                        run_cli_fn=_failing_cli,
                        mint_file_token_fn=_short_token_doc,
                        record_session_fn=mock.MagicMock(),
                        confirm_active_fn=mock.MagicMock(return_value={"status": "active"}),
                        start_watchdog_fn=lambda sess, rp: fake_watchdog,
                        stop_verify_fn=_fake_stop,
                    )
        self.assertEqual(len(stop_calls), 1)
        fake_watchdog.terminate.assert_called_once()

    def test_remote_seed_unconfirmed_stop_keeps_watchdog_alive_and_refuses_ready(self):
        rev = "a" * 40
        entries = _fake_remote_entries(rev)
        fake_watchdog = mock.MagicMock()
        fake_watchdog.poll.return_value = None

        def _ok_cli(cmd, auth, timeout=180, check=True):
            if cmd[0] == "exec" and "-f" in cmd:
                return {
                    "returncode": 0,
                    "stdout": "REMOTE_SEED_RESULT_JSON:" + json.dumps({"revision": rev, "files": entries}),
                    "stderr": "",
                }
            return {"returncode": 0, "stdout": "", "stderr": ""}

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            pin_path = _make_pin_file(td_path / "pinned.json", revision=rev)
            cfg_path = td_path / "depot_config.json"
            cfg_path.write_text(json.dumps({"drive_root_folder_id": "root-1"}), encoding="utf-8")
            ready_path = td_path / "verified_manifest.json"
            with (mock.patch.object(d, "DEPOT_DIR", td_path),
                  mock.patch.object(d, "CFG", cfg_path),
                  mock.patch.object(d, "READY", ready_path),
                  mock.patch.object(d, "STATE", td_path / "depot_state.json"),
                  mock.patch.object(d, "ensure_folder", side_effect=lambda svc, r, sub, allow_write: f"folder-{sub}")):
                with self.assertRaisesRegex(RuntimeError, "stop not verified"):
                    d.remote_seed(
                        mock.MagicMock(),
                        "root-1",
                        pin_path=pin_path,
                        run_cli_fn=_ok_cli,
                        mint_file_token_fn=_short_token_doc,
                        record_session_fn=mock.MagicMock(),
                        confirm_active_fn=mock.MagicMock(return_value={"status": "active"}),
                        start_watchdog_fn=lambda sess, rp: fake_watchdog,
                        stop_verify_fn=lambda s: {
                            "session": s, "stopped_confirmed": False,
                            "state": "UNKNOWN", "record_marked": False,
                        },
                    )
            fake_watchdog.terminate.assert_not_called()
            self.assertFalse(ready_path.exists())

    def test_stop_and_verify_remote_seed_session_tristate(self):
        states = iter([SessionState.ACTIVE, SessionState.STOPPED])
        cli_calls = []
        marked = []
        with mock.patch.object(d.time, "sleep", return_value=None):
            res = d._stop_and_verify_remote_seed_session(
                "muse-seed-1234",
                "oauth2",
                run_cli_fn=lambda cmd, auth, timeout=65, check=False: cli_calls.append(cmd) or {"returncode": 0},
                query_state_fn=lambda sess, colab_auth: next(states),
                mark_stopped_fn=lambda sess, st: marked.append((sess, st)) or True,
            )
        self.assertTrue(res["stopped_confirmed"])
        self.assertEqual(res["state"], "STOPPED")
        self.assertEqual(cli_calls, [["stop", "-s", "muse-seed-1234"]])
        self.assertEqual(marked, [("muse-seed-1234", SessionState.STOPPED)])


if __name__ == "__main__":
    unittest.main()
