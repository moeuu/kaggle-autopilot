from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kagglebot.agent_strategy import StrategyPromptRunConfig, run_strategy_prompt


@dataclass(frozen=True)
class DummyStrategyResult:
    last_message_path: Path
    returncode: int = 0


def test_run_strategy_prompt_writes_prompt_and_returns_response(tmp_path: Path) -> None:
    calls: list[tuple[Path, Path, bool]] = []

    def fake_run_strategy(prompt_path: Path, output_dir: Path, *, dry_run: bool) -> DummyStrategyResult:
        calls.append((prompt_path, output_dir, dry_run))
        last_message = output_dir / "strategy_last_message.txt"
        last_message.write_text("  concrete strategy  \n", encoding="utf-8")
        return DummyStrategyResult(last_message)

    text = run_strategy_prompt(
        prompt_text="prompt body",
        output_dir=tmp_path / "strategy",
        dry_run=True,
        config=StrategyPromptRunConfig(
            prompt_filename="gpt_strategy_prompt.md",
            start_message="[cyan]stage[/cyan]: starting",
            failure_message="failed",
            empty_message="empty",
        ),
        run_strategy_func=fake_run_strategy,
    )

    prompt_path = tmp_path / "strategy" / "gpt_strategy_prompt.md"
    assert prompt_path.read_text(encoding="utf-8") == "prompt body"
    assert calls == [(prompt_path, tmp_path / "strategy", True)]
    assert text == "concrete strategy"


def test_run_strategy_prompt_returns_empty_for_failed_runner(tmp_path: Path) -> None:
    last_message = tmp_path / "last.txt"
    last_message.write_text("ignored response", encoding="utf-8")

    def fake_run_strategy(prompt_path: Path, output_dir: Path, *, dry_run: bool) -> DummyStrategyResult:  # noqa: ARG001
        return DummyStrategyResult(last_message, returncode=2)

    text = run_strategy_prompt(
        prompt_text="prompt body",
        output_dir=tmp_path / "strategy",
        dry_run=False,
        config=StrategyPromptRunConfig(
            prompt_filename="prompt.md",
            start_message="starting",
            failure_message="failed",
            empty_message="empty",
        ),
        run_strategy_func=fake_run_strategy,
    )

    assert text == ""


def test_run_strategy_prompt_returns_empty_for_blank_response(tmp_path: Path) -> None:
    last_message = tmp_path / "last.txt"
    last_message.write_text("   \n", encoding="utf-8")

    def fake_run_strategy(prompt_path: Path, output_dir: Path, *, dry_run: bool) -> DummyStrategyResult:  # noqa: ARG001
        return DummyStrategyResult(last_message)

    text = run_strategy_prompt(
        prompt_text="prompt body",
        output_dir=tmp_path / "strategy",
        dry_run=False,
        config=StrategyPromptRunConfig(
            prompt_filename="prompt.md",
            start_message="starting",
            failure_message="failed",
            empty_message="empty",
        ),
        run_strategy_func=fake_run_strategy,
    )

    assert text == ""
