from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class LoopStopDecision:
    should_stop: bool
    reason: str


@dataclass(frozen=True)
class LoopTerminalDecision:
    should_stop: bool
    status: str
    stop_reason: str
    message: str


@dataclass(frozen=True)
class NoImproveMajorOverhaulDecision:
    force_major_overhaul: bool
    reason: str
    skip_message: str


@dataclass(frozen=True)
class IterationScoreUpdateDecision:
    delta_offline: float | None
    improved: bool
    best_score: float | None
    best_submission: object | None
    no_improve_streak: int
    capture_best_snapshot: bool
    restore_regression_snapshot: bool


@dataclass(frozen=True)
class ConfigStreakState:
    same_config_streak: int
    last_config_hash: str


@dataclass(frozen=True)
class StagnationTrack:
    no_improve_streak: int
    label: str


def update_same_config_streak(
    *,
    current_config_hash: str,
    last_config_hash: str | None,
    same_config_streak: int,
) -> ConfigStreakState:
    if current_config_hash == last_config_hash:
        return ConfigStreakState(
            same_config_streak=same_config_streak + 1,
            last_config_hash=current_config_hash,
        )
    return ConfigStreakState(same_config_streak=0, last_config_hash=current_config_hash)


def select_stagnation_track(
    *,
    best_high_potential_score: float | None,
    no_improve_streak: int,
    frontier_no_improve_streak: int,
) -> StagnationTrack:
    if best_high_potential_score is not None:
        return StagnationTrack(no_improve_streak=frontier_no_improve_streak, label="accuracy frontier")
    return StagnationTrack(no_improve_streak=no_improve_streak, label="offline metric")


def decide_iteration_score_update(
    *,
    metric_mismatch_detected: bool,
    non_generalizable_eval_detected: bool,
    previous_best_score: float | None,
    current_score: float,
    submission_path: object,
    no_improve_streak: int,
    stop_min_delta: float,
    conservative_regression_detected: bool,
    delta_from_best: Callable[[float | None, float], float | None],
    should_update_best: Callable[[float | None, float, float], bool],
) -> IterationScoreUpdateDecision:
    if metric_mismatch_detected or non_generalizable_eval_detected:
        return IterationScoreUpdateDecision(
            delta_offline=None,
            improved=False,
            best_score=previous_best_score,
            best_submission=None,
            no_improve_streak=no_improve_streak + 1,
            capture_best_snapshot=False,
            restore_regression_snapshot=bool(conservative_regression_detected),
        )

    delta_offline = delta_from_best(previous_best_score, current_score)
    improved = should_update_best(previous_best_score, current_score, stop_min_delta)
    if improved:
        return IterationScoreUpdateDecision(
            delta_offline=delta_offline,
            improved=True,
            best_score=current_score,
            best_submission=submission_path,
            no_improve_streak=0,
            capture_best_snapshot=True,
            restore_regression_snapshot=False,
        )
    return IterationScoreUpdateDecision(
        delta_offline=delta_offline,
        improved=False,
        best_score=previous_best_score,
        best_submission=None,
        no_improve_streak=no_improve_streak + 1,
        capture_best_snapshot=False,
        restore_regression_snapshot=bool(conservative_regression_detected),
    )


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


def decide_terminal_iteration_stop(
    *,
    confirmed_first_place: bool,
    iteration: int,
    max_iterations: int,
    submitted: bool,
    allow_max_iteration_stop: bool = True,
) -> LoopTerminalDecision:
    completed_status = "submitted" if submitted else "completed"
    if confirmed_first_place:
        return LoopTerminalDecision(
            should_stop=True,
            status=completed_status,
            stop_reason="submission_rank_1",
            message="[green]stop[/green]: submission rank reached #1",
        )
    if allow_max_iteration_stop and iteration >= max_iterations:
        return LoopTerminalDecision(
            should_stop=True,
            status=completed_status,
            stop_reason="",
            message="",
        )
    return LoopTerminalDecision(should_stop=False, status="", stop_reason="", message="")


def decide_max_total_time_stop(
    *,
    elapsed_total_min: float,
    max_total_min: float | None,
) -> LoopTerminalDecision:
    if max_total_min is None or max_total_min <= 0:
        return LoopTerminalDecision(should_stop=False, status="", stop_reason="", message="")
    if elapsed_total_min < max_total_min:
        return LoopTerminalDecision(should_stop=False, status="", stop_reason="", message="")
    reason = f"max_total_min reached: elapsed={elapsed_total_min:.1f}m limit={max_total_min:.1f}m"
    return LoopTerminalDecision(
        should_stop=True,
        status="stopped",
        stop_reason=reason,
        message=f"[yellow]stop[/yellow]: {reason}",
    )


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
