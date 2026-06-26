from __future__ import annotations

from collections.abc import Callable


def run_metric_only_competition_metric_fix(
    *,
    mismatch_reason: str,
    attempt: int,
    run_kernel_fix: Callable[..., None],
    codex_model: str,
    codex_reasoning_effort: str,
    max_codex_passes: int,
) -> None:
    """Apply a metric-only kernel fix using the implementation agent without strategy mediation."""
    policy_prefix = (
        "Metric-only repair policy:\n"
        "- Edit ONLY competition metric selection/reporting logic in kernel outputs.\n"
        "- Do NOT change model architecture, features, training schedule, folds, seeds, or ensembling.\n"
        "- Ensure metrics.json reports the official competition metric exactly.\n"
        "- Ensure submission.csv format stays unchanged.\n"
    )
    metric_fix_error = (
        "Competition metric mismatch detected in strict mode.\n"
        f"Details: {mismatch_reason}\n"
        "Apply a minimal metric-only fix and stop."
    )
    run_kernel_fix(
        error_message=metric_fix_error,
        attempt=attempt,
        use_gpt_strategy=False,
        codex_model=codex_model,
        codex_reasoning_effort=codex_reasoning_effort,
        prompt_prefix=policy_prefix,
        max_codex_passes=max_codex_passes,
    )
