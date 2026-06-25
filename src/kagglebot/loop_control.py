from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoopStopDecision:
    should_stop: bool
    reason: str


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
