from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoopStopDecision:
    should_stop: bool
    reason: str


@dataclass(frozen=True)
class NoImproveMajorOverhaulDecision:
    force_major_overhaul: bool
    reason: str
    skip_message: str


def decide_stagnation_stop(
    *,
    stop_allowed: bool,
    no_improve_streak: int,
    no_improve_patience: int,
    stop_min_delta: float,
    track_label: str,
    same_config_streak: int,
    same_config_patience: int,
) -> LoopStopDecision:
    """Return a deterministic stop reason for bug-like loop stagnation."""

    if not stop_allowed:
        return LoopStopDecision(should_stop=False, reason="")

    if no_improve_patience > 0 and no_improve_streak >= no_improve_patience:
        return LoopStopDecision(
            should_stop=True,
            reason=(
                f"{track_label} did not improve by >= {stop_min_delta:.6f} "
                f"for {no_improve_streak} consecutive iterations"
            ),
        )

    if same_config_patience > 0 and same_config_streak >= same_config_patience:
        return LoopStopDecision(
            should_stop=True,
            reason=f"model/pipeline config hash unchanged for {same_config_streak} consecutive iterations",
        )

    return LoopStopDecision(should_stop=False, reason="")


def decide_no_improve_major_overhaul(
    *,
    force_enabled: bool,
    improved: bool,
    high_potential_improved: bool,
    best_score_guarded: bool,
    metric_name: str,
    current_score: float,
    previous_best_score: float | None,
) -> NoImproveMajorOverhaulDecision:
    if not force_enabled or improved or high_potential_improved:
        return NoImproveMajorOverhaulDecision(force_major_overhaul=False, reason="", skip_message="")

    if best_score_guarded:
        return NoImproveMajorOverhaulDecision(
            force_major_overhaul=False,
            reason="",
            skip_message=(
                "[yellow]improve guard[/yellow]: "
                "skipping no-improve major-overhaul override because previous best "
                "was clipped as an outlier."
            ),
        )

    if previous_best_score is None:
        reason = f"Offline {metric_name} did not improve."
    else:
        reason = f"Offline {metric_name} did not improve (current={current_score:.6f}, best={previous_best_score:.6f})."
    return NoImproveMajorOverhaulDecision(force_major_overhaul=True, reason=reason, skip_message="")


def append_policy_reason(existing: str | None, addition: str) -> str | None:
    normalized = str(addition or "").strip()
    if not normalized:
        return existing
    if existing:
        return f"{existing} {normalized}".strip()
    return normalized
