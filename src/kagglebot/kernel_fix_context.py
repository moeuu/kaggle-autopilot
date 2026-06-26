from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from kagglebot import agent_prompts as _agent_prompts
from kagglebot import json_utils as _json_utils
from kagglebot import kernel_quality as _kernel_quality
from kagglebot import runtime_fixes as _runtime_fixes
from kagglebot.agents.identity import render_prompt_identity


class KernelFixContextConfig(Protocol):
    slug: str
    compute: str
    accelerator: str
    paths: object


@dataclass(frozen=True)
class KernelFixPromptPlan:
    prompt_path: Path
    attempt_path: Path
    prompt_text: str
    strategy_skip_reason: str | None
    strategy_prompt: str | None
    strategy_dir: Path


def prepare_lightweight_kernel_fix(
    *,
    config: KernelFixContextConfig,
    iter_dir: Path,
    attempt: int,
    error_text: str,
) -> _runtime_fixes.LightweightRuntimeFixResult | None:
    note_path = iter_dir / "agent" / f"kernel_fix_note-{attempt:02d}.txt"
    return _runtime_fixes.apply_lightweight_runtime_fix(
        config=config,
        error_text=error_text,
        note_path=note_path,
    )


def build_kernel_fix_prompt_plan(
    *,
    config: KernelFixContextConfig,
    run_id: str,
    iteration: int,
    iter_dir: Path,
    agent_dir: Path,
    error_message: str,
    attempt: int,
    prompt_prefix: str,
    use_gpt_strategy: bool,
    prompt_identity_args: dict[str, object],
    hardware_constraints: str,
) -> KernelFixPromptPlan:
    prompt_template = render_prompt_identity(config.paths.codex_kernel_fix_template.read_text(encoding="utf-8"))
    prompt_path = agent_dir / "kernel_fix_prompt.md"
    missing_module = _runtime_fixes.extract_missing_module(error_message)
    blocked_modules = _runtime_fixes.load_blocked_modules(config.paths.context_dir)
    if missing_module:
        blocked_modules = [name for name in blocked_modules if name != missing_module]
        _runtime_fixes.save_blocked_modules(config.paths.context_dir, blocked_modules)
    blocked_text = "\n".join(f"- {name}" for name in blocked_modules) if blocked_modules else "None"
    prompt_text = prompt_template.format(
        **prompt_identity_args,
        slug=config.slug,
        run_id=run_id,
        iteration=iteration,
        compute=config.compute,
        accelerator=config.accelerator,
        error_message=error_message,
        blocked_modules=blocked_text,
        logs_dir=str(iter_dir / "logs"),
        kernel_main=str(config.paths.kernel_source_dir / "kernel.py"),
        kernel_script=str(config.paths.kernel_run_dir(run_id) / "kernel.py"),
        rules_url=str(config.paths.rules_url_path),
        rules_md=str(config.paths.rules_md_path),
        overview_md=str(config.paths.overview_md_path),
        data_md=str(config.paths.data_md_path),
        submission_format=str(config.paths.submission_format_md_path),
        dataset_profile=str(config.paths.dataset_profile_path),
        sample_submission=str(config.paths.sample_submission_path),
    )
    subgroup_metrics_path = iter_dir / "output" / "metrics.json"
    subgroup_payload = _json_utils.load_json_object(subgroup_metrics_path) if subgroup_metrics_path.exists() else {}
    subgroup_collapse_signal = _kernel_quality.detect_subgroup_collapse_signal(
        kernel_metrics_payload=subgroup_payload if isinstance(subgroup_payload, dict) else None,
        direction="minimize",
    )
    if subgroup_collapse_signal is not None:
        prompt_text = (
            "Subgroup repair target:\n"
            f"- {subgroup_collapse_signal['note']}\n"
            "- Prefer subgroup-aware fixes over global retuning.\n"
            "- If selection or fallback logic is coarse, refine it to (model_id,node_type) granularity.\n\n"
            + prompt_text
        )
    if missing_module:
        prompt_text = (
            f"Missing dependency detected: {missing_module}\n"
            "Guard/disable only this missing package path. Keep actively using other available "
            "repo dependencies (torch/timm/torchvision/opencv/xgboost/lightgbm/catboost/"
            "transformers/tabicl/ultralytics/sklearn) and avoid silent capacity downgrades. "
            "If this package is required, add it via `uv add <package>` and update `pyproject.toml` "
            "+ `uv.lock`.\n\n" + prompt_text
        )
    if prompt_prefix.strip():
        prompt_text = f"{prompt_prefix.strip()}\n\n{prompt_text}"

    if not use_gpt_strategy:
        strategy_skip_reason = "metric_fix_policy"
    else:
        strategy_skip_reason = _runtime_fixes.error_strategy_skip_reason(stage="kernel_fix", error_text=error_message)
    strategy_prompt = None
    strategy_dir = agent_dir / f"kernel_fix_strategy-{attempt:02d}"
    if not strategy_skip_reason:
        strategy_prompt = _agent_prompts.build_error_strategy_prompt(
            stage="kernel_fix",
            slug=config.slug,
            run_id=run_id,
            attempt=attempt,
            compute=config.compute,
            accelerator=config.accelerator,
            hardware_constraints=hardware_constraints,
            error_text=error_message,
            codex_prompt=prompt_text,
        )
    return KernelFixPromptPlan(
        prompt_path=prompt_path,
        attempt_path=agent_dir / f"kernel_fix_prompt-{attempt:02d}.md",
        prompt_text=prompt_text,
        strategy_skip_reason=strategy_skip_reason,
        strategy_prompt=strategy_prompt,
        strategy_dir=strategy_dir,
    )


def append_kernel_fix_strategy(
    *,
    prompt_text: str,
    strategy_text: str,
    strategy_agent_display_name: str,
) -> str:
    if not strategy_text:
        return prompt_text
    return (
        prompt_text + f"\n\n## {strategy_agent_display_name} Extra-High Error-Fix Strategy\n"
        "Use the strategy below as guidance, then apply minimal targeted edits.\n\n"
        f"{strategy_text}\n"
    )
