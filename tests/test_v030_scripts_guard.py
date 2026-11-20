"""Offline unit tests for v0.3.0 shell wrappers and PreToolUse guard (0.00 CU)."""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GUARD_SCRIPT = REPO_ROOT / ".agents" / "pre_tool_guard.py"
RUN_H3_TASK_SH = REPO_ROOT / "scripts" / "run_h3_task.sh"
VERIFY_SHUTDOWN_SH = REPO_ROOT / "scripts" / "verify_shutdown.sh"
COLAB_ENV_SH = REPO_ROOT / "scripts" / "_colab_env.sh"


def _run_guard(tool_name: str, args: dict) -> dict:
    payload = {"toolCall": {"name": tool_name, "args": args}}
    proc = subprocess.run(
        [sys.executable, str(GUARD_SCRIPT)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(proc.stdout)


def _make_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class PreToolGuardV030Tests(unittest.TestCase):
    def test_verify_shutdown_stop_warm_allowed(self):
        allowed_cmds = [
            "./scripts/verify_shutdown.sh",
            "./scripts/verify_shutdown.sh --stop-warm",
            "scripts/verify_shutdown.sh --stop-warm",
            "cd /tmp/ws && ./scripts/verify_shutdown.sh --stop-warm",
            "cd /tmp/ws && scripts/verify_shutdown.sh --stop-warm",
            "./scripts/run_h3_task.sh --confirm-live --deliver --production-task-id t1 "
            "--task-spec config/tasks/a.json --allocation-rounds 3 "
            "--bootstrap-mode bundled --keep-warm-ttl 300",
        ]
        for cmd in allowed_cmds:
            with self.subTest(cmd=cmd):
                res = _run_guard("run_command", {"CommandLine": cmd})
                self.assertEqual(res["decision"], "allow", msg=f"unexpected for {cmd}: {res}")

    def test_secret_and_adc_blocks_still_enforced(self):
        blocked_cmds = [
            "cat ~/.config/gcloud/application_default_credentials.json",
            "./scripts/verify_shutdown.sh --stop-warm && cat .secrets/colab_drive_token.json",
            "scripts/verify_shutdown.sh --stop-warm && cat ~/.config/colab-cli/token.json",
            "./scripts/verify_shutdown.sh --stop-warm; ls .gcloud-colab",
        ]
        for cmd in blocked_cmds:
            with self.subTest(cmd=cmd):
                res = _run_guard("run_command", {"CommandLine": cmd})
                self.assertEqual(res["decision"], "deny", msg=f"expected deny for {cmd}: {res}")

        for secret_path in (
            "~/.config/gcloud/application_default_credentials.json",
            str(REPO_ROOT / ".secrets" / "colab_drive_token.json"),
            "~/.config/colab-cli/token.json",
        ):
            with self.subTest(read_path=secret_path):
                res = _run_guard("read_file", {"Path": secret_path})
                self.assertEqual(res["decision"], "deny")

    def test_unmanaged_colab_blocks_still_enforced(self):
        blocked_cmds = [
            "colab new --gpu A100",
            "./scripts/verify_shutdown.sh --stop-warm && colab new --gpu G4",
            "scripts/verify_shutdown.sh --stop-warm && .venv-colab-cli/bin/colab run -s s1 foo.py",
        ]
        for cmd in blocked_cmds:
            with self.subTest(cmd=cmd):
                res = _run_guard("run_command", {"CommandLine": cmd})
                self.assertEqual(res["decision"], "deny", msg=f"expected deny for {cmd}: {res}")


class RunH3TaskScriptV030Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v030-run-h3-"))
        self.repo = self.tmp / "repo"
        self.scripts = self.repo / "scripts"
        self.scripts.mkdir(parents=True)
        shutil.copy2(RUN_H3_TASK_SH, self.scripts / "run_h3_task.sh")
        self.argv_log = self.tmp / "argv.json"

        # Fake venv python: handles the import check (-c ...) and logs exec args.
        fake_py = (
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "if len(sys.argv) >= 2 and sys.argv[1] == '-c':\n"
            "    sys.exit(0)\n"
            "with open(os.environ['FAKE_ARGV_LOG'], 'w', encoding='utf-8') as f:\n"
            "    json.dump(sys.argv[1:], f)\n"
            "sys.exit(0)\n"
        )
        _make_executable(self.repo / ".venv-colab-cli" / "bin" / "python", fake_py)
        _make_executable(self.repo / ".venv-colab-cli" / "bin" / "colab", "#!/usr/bin/env bash\nexit 0\n")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _invoke(self, args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env["FAKE_ARGV_LOG"] = str(self.argv_log)
        return subprocess.run(
            ["bash", str(self.scripts / "run_h3_task.sh"), *args],
            cwd=str(cwd or self.repo),
            env=env,
            text=True,
            capture_output=True,
        )

    def test_usage_includes_v030_flags(self):
        proc = self._invoke([])
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--allocation-rounds", proc.stderr)
        self.assertIn("--bootstrap-mode", proc.stderr)
        self.assertIn("--keep-warm-ttl", proc.stderr)

    def test_forwards_v030_flags_in_deliver_mode_and_preserves_caller_pwd(self):
        caller = self.tmp / "caller_project"
        caller.mkdir()
        spec = caller / "my_spec.json"
        spec.write_text("{}", encoding="utf-8")

        proc = self._invoke(
            [
                "--confirm-live",
                "--deliver",
                "--production-task-id", "task-v030",
                "--authorize-new-task",
                "--task-spec", "my_spec.json",
                "--max-task-attempts", "2",
                "--allocation-rounds", "3",
                "--bootstrap-mode", "bundled",
                "--keep-warm-ttl", "300",
            ],
            cwd=caller,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        forwarded = json.loads(self.argv_log.read_text(encoding="utf-8"))
        self.assertEqual(forwarded[:2], ["h3_controller.py", "deliver"])
        self.assertIn("--task-spec", forwarded)
        spec_idx = forwarded.index("--task-spec")
        self.assertEqual(Path(forwarded[spec_idx + 1]).resolve(), spec.resolve())
        for flag, val in (
            ("--keep-warm-ttl", "300"),
            ("--allocation-rounds", "3"),
            ("--bootstrap-mode", "bundled"),
            ("--max-task-attempts", "2"),
        ):
            idx = forwarded.index(flag)
            self.assertEqual(forwarded[idx + 1], val)

    def test_forwards_v030_flags_in_run_mode_with_legacy_and_zero_ttl(self):
        proc = self._invoke(
            [
                "--confirm-live",
                "--production-task-id", "task-v030-run",
                "--variant", "A",
                "--run-id", "h3-20261119T000000Z-abcdef",
                "--allocation-rounds", "1",
                "--bootstrap-mode", "legacy",
                "--keep-warm-ttl", "0",
            ]
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        forwarded = json.loads(self.argv_log.read_text(encoding="utf-8"))
        self.assertEqual(forwarded[:2], ["h3_controller.py", "run"])
        self.assertEqual(forwarded[forwarded.index("--keep-warm-ttl") + 1], "0")
        self.assertEqual(forwarded[forwarded.index("--allocation-rounds") + 1], "1")
        self.assertEqual(forwarded[forwarded.index("--bootstrap-mode") + 1], "legacy")

    def test_rejects_invalid_v030_flag_values(self):
        base = [
            "--confirm-live",
            "--deliver",
            "--production-task-id", "task-v030",
            "--variant", "A",
        ]
        bad_cases = [
            ["--keep-warm-ttl", "30"],
            ["--keep-warm-ttl", "901"],
            ["--keep-warm-ttl", "abc"],
            ["--keep-warm-ttl"],
            ["--allocation-rounds", "0"],
            ["--allocation-rounds", "6"],
            ["--allocation-rounds"],
            ["--bootstrap-mode", "fast"],
            ["--bootstrap-mode"],
        ]
        for extra in bad_cases:
            with self.subTest(extra=extra):
                proc = self._invoke(base + extra)
                self.assertEqual(proc.returncode, 2, msg=f"expected exit 2 for {extra}: {proc.stderr}")


class VerifyShutdownScriptV030Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v030-verify-shutdown-"))
        self.repo = self.tmp / "repo"
        self.scripts = self.repo / "scripts"
        self.scripts.mkdir(parents=True)
        shutil.copy2(VERIFY_SHUTDOWN_SH, self.scripts / "verify_shutdown.sh")
        shutil.copy2(COLAB_ENV_SH, self.scripts / "_colab_env.sh")
        self.calls_log = self.tmp / "calls.jsonl"

        fake_py = (
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "args = sys.argv[1:]\n"
            "if args and args[0] == '-c':\n"
            "    code = args[1]\n"
            "    sys.argv = ['-c'] + args[2:]\n"
            "    exec(compile(code, '<inline>', 'exec'), {'__name__': '__main__'})\n"
            "    sys.exit(0)\n"
            "with open(os.environ['FAKE_CALLS_LOG'], 'a', encoding='utf-8') as f:\n"
            "    f.write(json.dumps(args) + '\\n')\n"
            "if args[:2] == ['h3_controller.py', 'check']:\n"
            "    print(os.environ['FAKE_CHECK_JSON'])\n"
            "    sys.exit(0)\n"
            "if args[:2] == ['h3_shutdown_executor.py', 'stop']:\n"
            "    print(json.dumps({'stopped_warm': True}))\n"
            "    sys.exit(int(os.environ.get('FAKE_STOP_RC', '0')))\n"
            "sys.exit(99)\n"
        )
        _make_executable(self.repo / ".venv-colab-cli" / "bin" / "python", fake_py)
        _make_executable(
            self.repo / ".venv-colab-cli" / "bin" / "colab",
            "#!/usr/bin/env bash\necho ''\n",
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _invoke(
        self,
        args: list[str],
        check_json: dict | None = None,
        stop_rc: int = 0,
    ) -> tuple[subprocess.CompletedProcess, list[list[str]]]:
        if self.calls_log.exists():
            self.calls_log.unlink()
        env = os.environ.copy()
        env["FAKE_CALLS_LOG"] = str(self.calls_log)
        env["FAKE_CHECK_JSON"] = json.dumps(
            check_json if check_json is not None else {"cu_balance": 195.0, "rate": 0.0, "sessions": []}
        )
        env["FAKE_STOP_RC"] = str(stop_rc)
        proc = subprocess.run(
            ["bash", str(self.scripts / "verify_shutdown.sh"), *args],
            cwd=str(self.repo),
            env=env,
            text=True,
            capture_output=True,
        )
        calls = []
        if self.calls_log.exists():
            calls = [json.loads(line) for line in self.calls_log.read_text(encoding="utf-8").splitlines() if line]
        return proc, calls

    def test_default_is_strictly_read_only_and_checks_sessions_and_rate(self):
        proc, calls = self._invoke([])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertEqual(calls, [["h3_controller.py", "check"]])
        self.assertIn("GPU_STOP_VERIFIED", proc.stdout)

    def test_default_fails_when_rate_nonzero_or_sessions_present(self):
        proc_rate, _ = self._invoke([], check_json={"cu_balance": 195.0, "rate": 1.25, "sessions": []})
        self.assertEqual(proc_rate.returncode, 1)
        self.assertIn("SHUTDOWN NOT VERIFIED", proc_rate.stderr)

        proc_sess, _ = self._invoke(
            [],
            check_json={"cu_balance": 195.0, "rate": 0.0, "sessions": [{"name": "s1", "gpu": "G4"}]},
        )
        self.assertEqual(proc_sess.returncode, 1)

    def test_stop_warm_invokes_executor_then_read_only_check(self):
        proc, calls = self._invoke(["--stop-warm"])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertEqual(
            calls,
            [
                ["h3_shutdown_executor.py", "stop", "--confirm-stop", "--stop-warm"],
                ["h3_controller.py", "check"],
            ],
        )
        self.assertIn("GPU_STOP_VERIFIED", proc.stdout)

    def test_stop_warm_fails_if_post_check_not_clean(self):
        proc, calls = self._invoke(
            ["--stop-warm"],
            check_json={"cu_balance": 195.0, "rate": 2.0, "sessions": []},
        )
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(len(calls), 2)

    def test_unknown_flag_refused(self):
        proc, calls = self._invoke(["--unknown-flag"])
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
