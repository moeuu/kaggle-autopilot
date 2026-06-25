from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich import print

from kagglebot.agent_io import read_agent_response
from kagglebot.agents.strategy_runner import StrategyResult, run_strategy


@dataclass(frozen=True)
class StrategyPromptRunConfig:
    prompt_filename: str
    start_message: str
    failure_message: str
    empty_message: str
    detail_message: str = ""


def run_strategy_prompt(
    *,
    prompt_text: str,
    output_dir: Path,
    dry_run: bool,
    config: StrategyPromptRunConfig,
    run_strategy_func: Callable[[Path, Path, bool], StrategyResult] = run_strategy,
    read_response_func: Callable[[Path], str] = read_agent_response,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / config.prompt_filename
    prompt_path.write_text(prompt_text, encoding="utf-8")
    print(config.start_message)
    if config.detail_message:
        print(config.detail_message)

    result = run_strategy_func(prompt_path, output_dir, dry_run=dry_run)
    strategy_text = read_response_func(result.last_message_path).strip()
    if result.returncode != 0:
        print(config.failure_message)
        return ""
    if not strategy_text:
        print(config.empty_message)
        return ""
    return strategy_text
