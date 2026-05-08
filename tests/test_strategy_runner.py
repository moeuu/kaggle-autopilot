from __future__ import annotations

from pathlib import Path

from kagglebot.agents import strategy_runner
from kagglebot.exec_utils import CommandResult


def test_run_strategy_uses_full_auto_and_sandbox_flags_when_supported(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLEBOT_AGENT_SANDBOX_MODE", "workspace-write")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    captured_args: list[str] = []

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:
        nonlocal captured_args
        captured_args = args
        last_message_path = Path(args[args.index("--output-last-message") + 1])
        last_message_path.write_text("final strategy\n", encoding="utf-8")
        return CommandResult(args=args, returncode=0, stdout="full transcript\n", stderr="", duration_sec=0.01)

    monkeypatch.setattr(strategy_runner, "_supported_flags", lambda: {"--full-auto", "--sandbox"})
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False)

    assert "--full-auto" in captured_args
    assert "--sandbox" in captured_args
    assert "-a" not in captured_args
    assert result.stdout == "final strategy"
    assert (tmp_path / "strategy_exec.txt").read_text(encoding="utf-8") == "full transcript\n"


def test_run_strategy_uses_full_auto_when_short_approval_flag_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLEBOT_AGENT_SANDBOX_MODE", "workspace-write")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    captured_args: list[str] = []

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:
        nonlocal captured_args
        captured_args = args
        return CommandResult(args=args, returncode=0, stdout="fallback output\n", stderr="", duration_sec=0.01)

    monkeypatch.setattr(strategy_runner, "_supported_flags", lambda: {"--full-auto"})
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False)

    assert "--full-auto" in captured_args
    assert "-a" not in captured_args
    assert "--sandbox" not in captured_args
    assert result.stdout == "fallback output"


def test_run_strategy_retries_without_sandbox_on_bootstrap_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLEBOT_AGENT_SANDBOX_MODE", "fallback")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    captured_args: list[list[str]] = []
    calls = {"count": 0}

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:  # noqa: ARG001
        calls["count"] += 1
        captured_args.append(args)
        last_message_path = Path(args[args.index("--output-last-message") + 1])
        last_message_path.parent.mkdir(parents=True, exist_ok=True)
        if calls["count"] == 1:
            last_message_path.write_text("sandbox failed\n", encoding="utf-8")
            return CommandResult(
                args=args,
                returncode=1,
                stdout="",
                stderr="bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted\n",
                duration_sec=0.01,
            )
        last_message_path.write_text("final strategy\n", encoding="utf-8")
        return CommandResult(args=args, returncode=0, stdout="fallback output\n", stderr="", duration_sec=0.01)

    monkeypatch.setattr(
        strategy_runner,
        "_supported_flags",
        lambda: {"--full-auto", "--sandbox", "--dangerously-bypass-approvals-and-sandbox"},
    )
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False)

    assert result.returncode == 0
    assert result.used_sandbox_fallback is True
    assert result.sandbox_failure_excerpt is not None
    assert "bwrap:" in result.sandbox_failure_excerpt
    assert calls["count"] == 2
    assert "--sandbox" in captured_args[0]
    assert "--dangerously-bypass-approvals-and-sandbox" in captured_args[1]
    assert "--full-auto" not in captured_args[1]


def test_run_strategy_falls_back_to_danger_full_access_when_dangerous_flag_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLEBOT_AGENT_SANDBOX_MODE", "fallback")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    captured_args: list[list[str]] = []
    calls = {"count": 0}

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:  # noqa: ARG001
        calls["count"] += 1
        captured_args.append(args)
        last_message_path = Path(args[args.index("--output-last-message") + 1])
        last_message_path.parent.mkdir(parents=True, exist_ok=True)
        if calls["count"] == 1:
            last_message_path.write_text("sandbox failed\n", encoding="utf-8")
            return CommandResult(
                args=args,
                returncode=1,
                stdout="",
                stderr="bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted\n",
                duration_sec=0.01,
            )
        last_message_path.write_text("final strategy\n", encoding="utf-8")
        return CommandResult(args=args, returncode=0, stdout="fallback output\n", stderr="", duration_sec=0.01)

    monkeypatch.setattr(strategy_runner, "_supported_flags", lambda: {"--full-auto", "--sandbox"})
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False)

    assert result.returncode == 0
    assert result.used_sandbox_fallback is True
    assert calls["count"] == 2
    assert "--dangerously-bypass-approvals-and-sandbox" not in captured_args[1]
    assert "--sandbox" in captured_args[1]
    assert "danger-full-access" in captured_args[1]


def test_run_strategy_does_not_retry_non_sandbox_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLEBOT_AGENT_SANDBOX_MODE", "fallback")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    calls = {"count": 0}

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:  # noqa: ARG001
        calls["count"] += 1
        last_message_path = Path(args[args.index("--output-last-message") + 1])
        last_message_path.parent.mkdir(parents=True, exist_ok=True)
        last_message_path.write_text("ordinary failure\n", encoding="utf-8")
        return CommandResult(
            args=args,
            returncode=1,
            stdout="",
            stderr="RuntimeError: ordinary failure\n",
            duration_sec=0.01,
        )

    monkeypatch.setattr(
        strategy_runner,
        "_supported_flags",
        lambda: {"--full-auto", "--sandbox", "--dangerously-bypass-approvals-and-sandbox"},
    )
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False)

    assert result.returncode == 1
    assert result.used_sandbox_fallback is False
    assert result.sandbox_failure_excerpt is None
    assert calls["count"] == 1


def test_run_strategy_uses_permissive_mode_by_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KAGGLEBOT_AGENT_SANDBOX_MODE", raising=False)
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    captured_args: list[str] = []

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:
        nonlocal captured_args
        captured_args = args
        last_message_path = Path(args[args.index("--output-last-message") + 1])
        last_message_path.write_text("final strategy\n", encoding="utf-8")
        return CommandResult(args=args, returncode=0, stdout="permissive output\n", stderr="", duration_sec=0.01)

    monkeypatch.setattr(
        strategy_runner,
        "_supported_flags",
        lambda: {"--full-auto", "--sandbox", "--dangerously-bypass-approvals-and-sandbox"},
    )
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False)

    assert result.returncode == 0
    assert result.sandbox_policy_mode == "permissive"
    assert result.used_sandbox_fallback is False
    assert "--dangerously-bypass-approvals-and-sandbox" in captured_args
    assert "--sandbox" not in captured_args


def test_run_strategy_workspace_write_mode_skips_permissive_retry(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLEBOT_AGENT_SANDBOX_MODE", "workspace-write")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    calls = {"count": 0}

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:  # noqa: ARG001
        calls["count"] += 1
        last_message_path = Path(args[args.index("--output-last-message") + 1])
        last_message_path.parent.mkdir(parents=True, exist_ok=True)
        last_message_path.write_text("sandbox failed\n", encoding="utf-8")
        return CommandResult(
            args=args,
            returncode=1,
            stdout="",
            stderr="bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted\n",
            duration_sec=0.01,
        )

    monkeypatch.setattr(
        strategy_runner,
        "_supported_flags",
        lambda: {"--full-auto", "--sandbox", "--dangerously-bypass-approvals-and-sandbox"},
    )
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False)

    assert result.returncode == 1
    assert result.sandbox_policy_mode == "workspace-write"
    assert result.used_sandbox_fallback is False
    assert calls["count"] == 1
