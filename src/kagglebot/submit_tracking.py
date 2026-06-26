from __future__ import annotations

import math
from dataclasses import dataclass

from kagglebot.scalar_utils import parse_finite_float
from kagglebot.score_utils import should_update_best_score


@dataclass(frozen=True)
class SubmittedTrackingScoreDecision:
    online_score: float | None
    tracking_score: float | None
    tracking_source: str
    update_best_submitted_score: bool
    best_submitted_score: float | None


def submission_score_for_tracking(*, offline_score: float, online_score: float | None) -> tuple[float, str]:
    if isinstance(online_score, (int, float)):
        value = float(online_score)
        if math.isfinite(value):
            return value, "submission_public_score"
    return float(offline_score), "offline"


def decide_submitted_tracking_score_update(
    *,
    submission_result: dict[str, object],
    offline_score: float | None,
    previous_best_score: float | None,
    direction: str,
    min_improvement: float = 0.0,
) -> SubmittedTrackingScoreDecision:
    if offline_score is None:
        return SubmittedTrackingScoreDecision(
            online_score=None,
            tracking_score=None,
            tracking_source="unavailable",
            update_best_submitted_score=False,
            best_submitted_score=previous_best_score,
        )

    online_score: float | None = None
    outcome_payload = submission_result.get("outcome")
    if isinstance(outcome_payload, dict):
        online_score = parse_finite_float(outcome_payload.get("score"))
    tracking_score, tracking_source = submission_score_for_tracking(
        offline_score=offline_score,
        online_score=online_score,
    )
    should_update = should_update_best_score(previous_best_score, tracking_score, direction, min_improvement)
    return SubmittedTrackingScoreDecision(
        online_score=online_score,
        tracking_score=tracking_score,
        tracking_source=tracking_source,
        update_best_submitted_score=should_update,
        best_submitted_score=tracking_score if should_update else previous_best_score,
    )
