from __future__ import annotations

import io
from pathlib import Path

from kagglebot.agents import codex_runner
from kagglebot.agents.codex_runner import _format_command_for_log


def test_format_command_for_log_truncates_to_two_lines() -> None:
    command = '/bin/bash -lc "' + "verylongtoken " * 40 + '"'
    first, second = _format_command_for_log(command)

    assert first
    assert second
    assert second.endswith("...")


def test_format_command_for_log_keeps_short_command_on_one_line() -> None:
    first, second = _format_command_for_log("uv run pytest -q")

    assert first == "uv run pytest -q"
    assert second == ""


def test_codex_timeout_uses_environment_override(monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_CODEX_TIMEOUT_SEC", "600")

    assert codex_runner._codex_timeout_seconds() == 600.0  # noqa: SLF001


def test_run_codex_retries_without_sandbox_on_bootstrap_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLEBOT_AGENT_SANDBOX_MODE", "fallback")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("fix it", encoding="utf-8")
    captured_args: list[list[str]] = []
    calls = {"count": 0}

    class DummyProcess:
        def __init__(self, args: list[str], *, returncode: int, stderr_text: str, last_message: str) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO(stderr_text)
            self._returncode = returncode
            captured_args.append(args)
            last_message_path = Path(args[args.index("--output-last-message") + 1])
            last_message_path.parent.mkdir(parents=True, exist_ok=True)
            last_message_path.write_text(last_message, encoding="utf-8")

        def wait(self) -> int:
            return self._returncode

    def fake_popen(args: list[str], **kwargs) -> DummyProcess:  # noqa: ARG001
        calls["count"] += 1
        if calls["count"] == 1:
            return DummyProcess(
                args,
                returncode=1,
                stderr_text="bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted\n",
                last_message="sandbox failed\n",
            )
        return DummyProcess(args, returncode=0, stderr_text="", last_message="fixed\n")

    monkeypatch.setattr(
        codex_runner,
        "_supported_flags",
        lambda: {"--full-auto", "--sandbox", "--dangerously-bypass-approvals-and-sandbox"},
    )
    monkeypatch.setattr(codex_runner.subprocess, "Popen", fake_popen)

    result = codex_runner.run_codex(prompt_path, tmp_path / "out")

    assert result.returncode == 0
    assert result.used_sandbox_fallback is True
    assert result.sandbox_failure_excerpt is not None
    assert "bwrap:" in result.sandbox_failure_excerpt
    assert calls["count"] == 2
    assert "--sandbox" in captured_args[0]
    assert "workspace-write" in captured_args[0]
    assert "--dangerously-bypass-approvals-and-sandbox" in captured_args[1]
    assert "--full-auto" not in captured_args[1]


def test_run_codex_falls_back_to_danger_full_access_when_dangerous_flag_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLEBOT_AGENT_SANDBOX_MODE", "fallback")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("fix it", encoding="utf-8")
    captured_args: list[list[str]] = []
    calls = {"count": 0}

    class DummyProcess:
        def __init__(self, args: list[str], *, returncode: int, stderr_text: str, last_message: str) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO(stderr_text)
            self._returncode = returncode
            captured_args.append(args)
            last_message_path = Path(args[args.index("--output-last-message") + 1])
            last_message_path.parent.mkdir(parents=True, exist_ok=True)
            last_message_path.write_text(last_message, encoding="utf-8")

        def wait(self) -> int:
            return self._returncode

    def fake_popen(args: list[str], **kwargs) -> DummyProcess:  # noqa: ARG001
        calls["count"] += 1
        if calls["count"] == 1:
            return DummyProcess(
                args,
                returncode=1,
                stderr_text="bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted\n",
                last_message="sandbox failed\n",
            )
        return DummyProcess(args, returncode=0, stderr_text="", last_message="fixed\n")

    monkeypatch.setattr(codex_runner, "_supported_flags", lambda: {"--full-auto", "--sandbox"})
    monkeypatch.setattr(codex_runner.subprocess, "Popen", fake_popen)

    result = codex_runner.run_codex(prompt_path, tmp_path / "out")

    assert result.returncode == 0
    assert result.used_sandbox_fallback is True
    assert calls["count"] == 2
    assert "--dangerously-bypass-approvals-and-sandbox" not in captured_args[1]
    assert "--sandbox" in captured_args[1]
    assert "danger-full-access" in captured_args[1]
    assert "--full-auto" not in captured_args[1]


def test_run_codex_does_not_retry_non_sandbox_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLEBOT_AGENT_SANDBOX_MODE", "fallback")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("fix it", encoding="utf-8")
    calls = {"count": 0}

    class DummyProcess:
        def __init__(self, args: list[str], *, returncode: int, stderr_text: str, last_message: str) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO(stderr_text)
            self._returncode = returncode
            last_message_path = Path(args[args.index("--output-last-message") + 1])
            last_message_path.parent.mkdir(parents=True, exist_ok=True)
            last_message_path.write_text(last_message, encoding="utf-8")

        def wait(self) -> int:
            return self._returncode

    def fake_popen(args: list[str], **kwargs) -> DummyProcess:  # noqa: ARG001
        calls["count"] += 1
        return DummyProcess(
            args,
            returncode=1,
            stderr_text="RuntimeError: model failed\n",
            last_message="model failed\n",
        )

    monkeypatch.setattr(
        codex_runner,
        "_supported_flags",
        lambda: {"--full-auto", "--sandbox", "--dangerously-bypass-approvals-and-sandbox"},
    )
    monkeypatch.setattr(codex_runner.subprocess, "Popen", fake_popen)

    result = codex_runner.run_codex(prompt_path, tmp_path / "out")

    assert result.returncode == 1
    assert result.used_sandbox_fallback is False
    assert result.sandbox_failure_excerpt is None
    assert calls["count"] == 1


def test_run_codex_uses_permissive_mode_by_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KAGGLEBOT_AGENT_SANDBOX_MODE", raising=False)
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("fix it", encoding="utf-8")
    captured_args: list[list[str]] = []

    class DummyProcess:
        def __init__(self, args: list[str], *, last_message: str) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("")
            self._returncode = 0
            captured_args.append(args)
            last_message_path = Path(args[args.index("--output-last-message") + 1])
            last_message_path.parent.mkdir(parents=True, exist_ok=True)
            last_message_path.write_text(last_message, encoding="utf-8")

        def wait(self) -> int:
            return self._returncode

    monkeypatch.setattr(
        codex_runner,
        "_supported_flags",
        lambda: {"--full-auto", "--sandbox", "--dangerously-bypass-approvals-and-sandbox"},
    )
    monkeypatch.setattr(
        codex_runner.subprocess,
        "Popen",
        lambda args, **kwargs: DummyProcess(args, last_message="ok\n"),
    )

    result = codex_runner.run_codex(prompt_path, tmp_path / "out")

    assert result.returncode == 0
    assert result.sandbox_policy_mode == "permissive"
    assert result.used_sandbox_fallback is False
    assert len(captured_args) == 1
    assert "--dangerously-bypass-approvals-and-sandbox" in captured_args[0]
    assert "--sandbox" not in captured_args[0]


def test_run_codex_removes_stale_outputs_before_start(monkeypatch, tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    prompt_path.write_text("implement it", encoding="utf-8")
    transcript_path = output_dir / "codex_exec.jsonl"
    last_message_path = output_dir / "codex_last_message.txt"
    transcript_path.write_text('{"type":"turn.completed","stale":true}\n', encoding="utf-8")
    last_message_path.write_text("stale completion\n", encoding="utf-8")

    class DummyProcess:
        def __init__(self, args: list[str]) -> None:
            assert not transcript_path.exists()
            assert not last_message_path.exists()
            self.stdin = io.StringIO()
            self.stdout = io.StringIO('{"type":"turn.started"}\n')
            self.stderr = io.StringIO("")
            self._returncode = 0
            result_path = Path(args[args.index("--output-last-message") + 1])
            result_path.write_text("current completion\n", encoding="utf-8")

        def wait(self) -> int:
            return self._returncode

    monkeypatch.setattr(codex_runner, "_supported_flags", lambda: set())
    monkeypatch.setattr(codex_runner.subprocess, "Popen", lambda args, **kwargs: DummyProcess(args))

    result = codex_runner.run_codex(prompt_path, output_dir)

    assert result.stdout == '{"type":"turn.started"}\n'
    assert transcript_path.read_text(encoding="utf-8") == result.stdout
    assert last_message_path.read_text(encoding="utf-8") == "current completion\n"


def test_run_codex_workspace_write_mode_skips_permissive_retry(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLEBOT_AGENT_SANDBOX_MODE", "workspace-write")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("fix it", encoding="utf-8")
    calls = {"count": 0}

    class DummyProcess:
        def __init__(self, args: list[str], *, returncode: int, stderr_text: str, last_message: str) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO(stderr_text)
            self._returncode = returncode
            last_message_path = Path(args[args.index("--output-last-message") + 1])
            last_message_path.parent.mkdir(parents=True, exist_ok=True)
            last_message_path.write_text(last_message, encoding="utf-8")

        def wait(self) -> int:
            return self._returncode

    def fake_popen(args: list[str], **kwargs) -> DummyProcess:  # noqa: ARG001
        calls["count"] += 1
        return DummyProcess(
            args,
            returncode=1,
            stderr_text="bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted\n",
            last_message="sandbox failed\n",
        )

    monkeypatch.setattr(
        codex_runner,
        "_supported_flags",
        lambda: {"--full-auto", "--sandbox", "--dangerously-bypass-approvals-and-sandbox"},
    )
    monkeypatch.setattr(codex_runner.subprocess, "Popen", fake_popen)

    result = codex_runner.run_codex(prompt_path, tmp_path / "out")

    assert result.returncode == 1
    assert result.sandbox_policy_mode == "workspace-write"
    assert result.used_sandbox_fallback is False
    assert calls["count"] == 1


def test_run_codex_passes_profile_model_effort_and_workdir(monkeypatch, tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("implement the Oracle brief", encoding="utf-8")
    workdir = tmp_path / "repo"
    workdir.mkdir()
    captured: dict[str, object] = {}

    class DummyProcess:
        def __init__(self, args: list[str], **kwargs) -> None:
            captured["args"] = args
            captured["cwd"] = kwargs.get("cwd")
            self.stdin = io.StringIO()
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("")
            self._returncode = 0
            last_message_path = Path(args[args.index("--output-last-message") + 1])
            last_message_path.write_text("done\n", encoding="utf-8")

        def wait(self) -> int:
            return self._returncode

    monkeypatch.setattr(codex_runner, "_supported_flags", lambda: {"--full-auto"})
    monkeypatch.setattr(codex_runner.subprocess, "Popen", DummyProcess)

    result = codex_runner.run_codex(
        prompt_path,
        tmp_path / "out",
        model="gpt-5.6-sol",
        reasoning_effort="xhigh",
        reasoning_profile="xhigh",
        cli_profile="sol-xhigh",
        cwd=workdir,
    )

    args = captured["args"]
    assert isinstance(args, list)
    assert args[args.index("--profile") + 1] == "sol-xhigh"
    assert args[args.index("-m") + 1] == "gpt-5.6-sol"
    assert args[args.index("-C") + 1] == str(workdir.resolve())
    assert 'model_reasoning_effort="xhigh"' in args
    assert captured["cwd"] == workdir
    assert result.reasoning_profile == "xhigh"
