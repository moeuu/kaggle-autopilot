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

## Required Output: Improvement Contract

Read the frozen iteration evidence bundle in the existing prompt and attachments before answering.
Return concise, actionable implementation instructions for {IMPLEMENTATION_AGENT.display_name} with these headings:
1) **Evidence diagnosis**: cite exact evidence paths/fields and separate model quality from execution,
   submission, and measurement defects.
2) **Falsifiable hypothesis**: state one primary root-cause hypothesis, why it outranks alternatives,
   and what observation would disprove it.
3) **Material delta**: exact file-level edits and how they differ from the current method and any
   previously unsupported/no-op transition. Reusing proven components is allowed; repeating a failed approach
   unchanged is not.
4) **Validation and attribution**: checks after edits, including the metric, score source, split/data scope,
   runtime/output contract, and the comparable before/after baseline.
5) **Expected observation**: predicted metric/log/artifact changes if the hypothesis is correct.
6) **Stop or rollback criteria**: when to revert, stop spending compute, or repair evaluation before tuning.
7) **Fallback**: the next materially different action if the first plan fails.

Never describe a delta between different metrics, directions, score sources, or untrusted evaluations as an improvement.

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


def build_autofix_prompt(
    *,
    slug: str,
    run_id: str,
    attempt: int,
    compute: str,
    accelerator: str,
    error_text: str,
    error_path: Path,
    repo_root: Path,
    run_dir: Path,
    kernel_dir: Path,
    context_dir: Path,
    data_dir: Path,
    prompts_dir: Path,
    autopilot_path: Path,
    allowed_prefixes: list[Path],
    denied_prefixes: list[Path],
    submit_context: str = "",
) -> str:
    allowed_list = "\n".join(f"- {path}" for path in allowed_prefixes)
    denied_list = "\n".join(f"- {path}" for path in denied_prefixes)
    submit_context_block = ""
    if submit_context:
        submit_context_block = (
            "\n## Submit Context\n\n"
            "This is a submit-stage failure. Use `repair_target` to decide whether to fix the submission artifact, "
            "submit mode/kernel path, or a platform/manual blocker.\n"
            "This run must fix the submission contract before further model changes.\n"
            "If the error is competition-specific, edit only authoritative `kernel.py`.\n"
            "Do not leave iter2 with the same Kaggle row/column/evaluation exception.\n\n"
            "```\n"
            f"{submit_context}\n"
            "```\n"
        )
    return f"""\
# Kagglebot {IMPLEMENTATION_AGENT.display_name}: Auto-Fix

## Context

Competition: {slug}
Run ID: {run_id}
Attempt: {attempt}
Compute: {compute} ({accelerator})

## Error

```
{error_text}
```

Error log file: {error_path}
{submit_context_block}

## Relevant Paths

- repo_root: {repo_root}
- run_dir: {run_dir}
- kernel_dir: {kernel_dir}
- context_dir: {context_dir}
- data_dir: {data_dir}
- prompts_dir: {prompts_dir}
- autopilot: {autopilot_path}

## Allowed Edit Scope

{allowed_list}

## Forbidden Edit Scope

{denied_list or "- None"}

## Task

1) Identify root cause of the failure.
2) Apply minimal, targeted fixes so autopilot can continue.
3) Do not touch datasets or credentials.
   Prefer already-installed dependencies; add new dependencies only with clear justification.
   If a dependency must be added, use `uv add <package>` and keep `pyproject.toml` + `uv.lock` consistent.
4) Explain what you changed in your response.
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
