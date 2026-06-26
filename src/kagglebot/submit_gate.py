from __future__ import annotations

from dataclasses import dataclass

from kagglebot.score_utils import should_update_best_score


@dataclass(frozen=True)
class FallbackSubmitGateDecision:
    allow_submit: bool
    message: str


@dataclass(frozen=True)
class IterationSubmitImprovementGateDecision:
    submit_improvement_allowed: bool
    submit_non_improving: bool
    forced_submit_reason: str | None
    message: str


def decide_fallback_submit_gate(
    *,
    submit_improved_only: bool,
    force_submit: bool,
    require_submit_improvement: bool,
    best_submittable_score: float | None,
    best_submitted_score: float | None,
    direction: str,
    min_improvement: float,
    final_iteration_reached: bool,
) -> FallbackSubmitGateDecision:
    if submit_improved_only and not force_submit and best_submitted_score is None:
        return FallbackSubmitGateDecision(allow_submit=False, message="")

    if (
        (require_submit_improvement or submit_improved_only)
        and not force_submit
        and best_submittable_score is not None
        and best_submitted_score is not None
    ):
        allow_submit = should_update_best_score(
            best_submitted_score, best_submittable_score, direction, min_improvement
        )
        if (not allow_submit) and final_iteration_reached and not submit_improved_only:
            return FallbackSubmitGateDecision(
                allow_submit=True,
                message=(
                    "[yellow]submit override[/yellow]: final iteration reached; "
                    "allowing fallback submit even though offline metric did not improve."
                ),
            )
        return FallbackSubmitGateDecision(allow_submit=allow_submit, message="")

    return FallbackSubmitGateDecision(allow_submit=True, message="")


def decide_iteration_submit_improvement_gate(
    *,
    submit_improved_only: bool,
    force_submit: bool,
    require_submit_improvement: bool,
    best_submitted_score: float | None,
    current_score: float,
    direction: str,
    min_improvement: float,
    final_iteration: bool,
    submit_enabled: bool,
    quality_allows_submit: bool,
    spare_daily_submission_slot: bool,
    submission_limit_per_day: int | None,
    forced_submit_reason: str | None,
    spare_submit_reason: str,
) -> IterationSubmitImprovementGateDecision:
    if submit_improved_only and not force_submit and best_submitted_score is None:
        if (
            submit_enabled
            and quality_allows_submit
            and (spare_daily_submission_slot or submission_limit_per_day is None)
        ):
            slot_reason = (
                "spare daily submission slots remain"
                if spare_daily_submission_slot
                else "no numeric daily submission limit is known"
            )
            return IterationSubmitImprovementGateDecision(
                submit_improvement_allowed=True,
                submit_non_improving=False,
                forced_submit_reason=forced_submit_reason or spare_submit_reason,
                message=(
                    f"[yellow]submit override[/yellow]: {slot_reason}; allowing submit without a prior "
                    "submitted checkpoint."
                ),
            )
        return IterationSubmitImprovementGateDecision(
            submit_improvement_allowed=False,
            submit_non_improving=True,
            forced_submit_reason=forced_submit_reason,
            message=("[yellow]submit deferred[/yellow]: submit_policy=improved requires a prior submitted checkpoint."),
        )

    if (require_submit_improvement or submit_improved_only) and not force_submit and best_submitted_score is not None:
        allowed = should_update_best_score(best_submitted_score, current_score, direction, min_improvement)
        if allowed:
            return IterationSubmitImprovementGateDecision(True, False, forced_submit_reason, "")
        if final_iteration and not submit_improved_only:
            return IterationSubmitImprovementGateDecision(
                submit_improvement_allowed=True,
                submit_non_improving=False,
                forced_submit_reason=forced_submit_reason,
                message=(
                    "[yellow]submit override[/yellow]: final iteration reached; "
                    "allowing submit even though score did not improve over submitted checkpoints."
                ),
            )
        if submit_enabled and spare_daily_submission_slot and quality_allows_submit:
            return IterationSubmitImprovementGateDecision(
                submit_improvement_allowed=True,
                submit_non_improving=False,
                forced_submit_reason=forced_submit_reason or spare_submit_reason,
                message=(
                    "[yellow]submit override[/yellow]: spare daily submission slots remain; "
                    "allowing non-improving checkpoint submit."
                ),
            )
        return IterationSubmitImprovementGateDecision(
            submit_improvement_allowed=False,
            submit_non_improving=True,
            forced_submit_reason=forced_submit_reason,
            message="[yellow]submit deferred[/yellow]: score did not improve over previous submitted checkpoint.",
        )

    return IterationSubmitImprovementGateDecision(True, False, forced_submit_reason, "")
