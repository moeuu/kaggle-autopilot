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


def run_improvement_strategy_prompt(
    *,
    prompt_text: str,
    output_dir: Path,
    dry_run: bool,
    implementation_agent_alias: str,
    run_strategy_func: Callable[[Path, Path, bool], StrategyResult] = run_strategy,
) -> str:
    return run_strategy_prompt(
        prompt_text=prompt_text,
        output_dir=output_dir,
        dry_run=dry_run,
        config=StrategyPromptRunConfig(
            prompt_filename="gpt_improvement_prompt.md",
            start_message="[cyan]improve[/cyan]: gpt drafting improvement prompt",
            failure_message=(
                "[yellow]improve[/yellow]: gpt improvement strategy failed, "
                f"falling back to direct {implementation_agent_alias} prompt"
            ),
            empty_message=(
                "[yellow]improve[/yellow]: gpt improvement strategy empty, "
                f"falling back to direct {implementation_agent_alias} prompt"
            ),
        ),
        run_strategy_func=run_strategy_func,
    )


def run_error_strategy_prompt(
    *,
    prompt_text: str,
    output_dir: Path,
    dry_run: bool,
    stage_label: str,
    implementation_agent_alias: str,
    strategy_model: str,
    reasoning_effort: str,
    run_strategy_func: Callable[[Path, Path, bool], StrategyResult] = run_strategy,
) -> str:
    return run_strategy_prompt(
        prompt_text=prompt_text,
        output_dir=output_dir,
        dry_run=dry_run,
        config=StrategyPromptRunConfig(
            prompt_filename="gpt_strategy_prompt.md",
            start_message=f"[cyan]{stage_label}[/cyan]: gpt analyzing error",
            detail_message=(
                f"[cyan]{stage_label}[/cyan]: strategy model={strategy_model} reasoning={reasoning_effort}"
            ),
            failure_message=(
                f"[yellow]{stage_label}[/yellow]: gpt strategy failed, "
                f"continuing with direct {implementation_agent_alias} fix"
            ),
            empty_message=(
                f"[yellow]{stage_label}[/yellow]: gpt strategy empty, "
                f"continuing with direct {implementation_agent_alias} fix"
            ),
        ),
        run_strategy_func=run_strategy_func,
    )
