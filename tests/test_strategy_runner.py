from __future__ import annotations

from pathlib import Path

from kagglebot.agents import strategy_runner
from kagglebot.exec_utils import CommandResult


def test_run_strategy_uses_full_auto_and_sandbox_flags_when_supported(monkeypatch, tmp_path: Path) -> None:
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
