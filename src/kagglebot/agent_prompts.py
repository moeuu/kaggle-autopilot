from __future__ import annotations

from pathlib import Path

from kagglebot import code_reference
from kagglebot.agents.identity import IMPLEMENTATION_AGENT, STRATEGY_AGENT


def build_improvement_strategy_prompt(
    *,
    slug: str,
    run_id: str,
    iteration: int,
    metric: str,
    direction: str,
    current_score: float,
    current_score_source: str,
    target_score: float,
    top1_score: float | None,
    top1_source: str,
    top1_gap: float | None,
    delta_offline: float | None,
    improvement_mode: str,
    hardware_constraints: str,
    codex_prompt: str,
    problem_type_knowledge: str,
) -> str:
    return f"""\
# Kagglebot Improvement Strategy

You are {STRATEGY_AGENT.display_name} in extra-high reasoning mode.
Design a concrete improvement prompt for {IMPLEMENTATION_AGENT.display_name} (extra-high), which will implement changes.

Competition: {slug}
Run ID: {run_id}
Iteration: {iteration}
Metric: {metric} ({direction})
Current score: {current_score:.6f} (source: {current_score_source})
Target score: {target_score:.6f}
Top1 score: {"unavailable" if top1_score is None else f"{top1_score:.6f}"}
Top1 source: {top1_source}
Top1 gap: {"unavailable" if top1_gap is None else f"{top1_gap:.6f}"}
Delta vs previous best: {"unavailable" if delta_offline is None else f"{delta_offline:.6f}"}
Improvement mode: {improvement_mode}

Hardware execution budget:
{hardware_constraints}

## Existing {IMPLEMENTATION_AGENT.display_name} Improvement Prompt Draft

```
{codex_prompt}
```

## Problem-Type Knowledge (Past Causes and Fixes)

{problem_type_knowledge}

## Required Output

Return concise, actionable implementation instructions for {IMPLEMENTATION_AGENT.display_name}:
1) What to change and why (root-cause hypothesis of current gap).
2) Exact file-level edits and model/training changes.
3) Validation checks after edits (what metrics/logs to confirm).
4) Fallback if the first plan fails.

Do not include chain-of-thought.
"""


def build_error_strategy_prompt(
    *,
    stage: str,
    slug: str,
    run_id: str,
    attempt: int,
    compute: str,
    accelerator: str,
    hardware_constraints: str,
    error_text: str,
    codex_prompt: str,
) -> str:
    return f"""\
# Kagglebot {STRATEGY_AGENT.display_name} Error Strategy

You are {STRATEGY_AGENT.display_name} in xhigh reasoning mode.
Think through the failure and produce a concrete fix strategy for
{IMPLEMENTATION_AGENT.display_name} (xhigh), which will apply edits.

Stage: {stage}
Competition: {slug}
Run ID: {run_id}
Attempt: {attempt}
Compute: {compute} ({accelerator})
Hardware execution budget:
{hardware_constraints}

## Error

```
{error_text}
```

## {IMPLEMENTATION_AGENT.display_name} Fix Prompt (current)

```
{codex_prompt}
```

## Required Output

Return concise, actionable instructions for {IMPLEMENTATION_AGENT.display_name}:
1) Root cause hypothesis.
2) Minimal file edits (paths + what to change).
3) Safety checks to run after edits.
4) Fallback if the first fix does not work.
"""


def build_code_reference_repair_prompt(
    *,
    base_prompt_text: str,
    reference: code_reference.CodeReferenceNotebook,
    issues: list[str],
    kernel_path: Path,
) -> str:
    issues_text = ", ".join(issues) if issues else "unknown"
    tabicl_required = code_reference.reference_requires_tabicl(reference)
    tabicl_line = (
        "- This reference appears to be TabICL-based. You MUST include a real TabICL path in kernel.py."
        if tabicl_required
        else "- TabICL path is optional for this reference notebook."
    )
    return (
        f"# {IMPLEMENTATION_AGENT.display_name} Improvement Repair: Mandatory Code Reference Implementation\n\n"
        "The previous change did not satisfy mandatory code-reference implementation requirements.\n\n"
        f"- Failed checks: {issues_text}\n"
        f"- Required notebook: {reference.kernel_id} ({reference.title})\n"
        f"- Kernel path: {kernel_path}\n"
        f"- Required marker: `{code_reference.code_reference_marker(reference)}`\n"
        f"{tabicl_line}\n\n"
        "Make minimal edits to kernel.py so all checks pass.\n"
        "Do not weaken the model by collapsing to tiny conservative feature subsets that reduce offline score.\n\n"
        "## Original Improvement Context\n\n"
        f"{base_prompt_text}\n"
    )
