from __future__ import annotations

from kagglebot.json_utils import load_json_object
from kagglebot.paths import CompetitionPaths
from kagglebot.scalar_utils import parse_finite_float
from kagglebot.score_utils import should_update_best_score


def resume_best_online_submission_score(
    *,
    paths: CompetitionPaths,
    run_id: str,
    direction: str,
    max_iterations: int,
) -> float | None:
    best: float | None = None
    for iteration in range(1, max_iterations + 1):
        metrics_path = paths.iter_dir(run_id, iteration) / "metrics.json"
        if not metrics_path.exists():
            continue
        payload = load_json_object(metrics_path)
        if payload is None:
            continue
        score = parse_finite_float(payload.get("submission_score"), allow_commas=True)
        if score is None:
            continue
        if should_update_best_score(best, score, direction, 0.0):
            best = score
    return best


def should_force_major_overhaul_by_rank(
    *,
    rank: int | None,
    total_teams: int | None,
    max_percentile: float,
    min_teams: int,
) -> bool:
    if rank is None or total_teams is None:
        return False
    if total_teams < max(1, min_teams):
        return False
    if rank <= 0:
        return False
    return (rank / total_teams) > max_percentile


def meets_rank_percentile_target(
    *,
    rank_percentile: float | None,
    estimated_rank_percentile: float | None,
    target_rank_percentile: float | None,
) -> bool:
    if target_rank_percentile is None:
        return False
    for candidate in (rank_percentile, estimated_rank_percentile):
        if candidate is not None and candidate <= target_rank_percentile:
            return True
    return False


def build_medal_target_reason(
    *,
    target_medal: str | None,
    target_rank_percentile: float | None,
    rank_percentile: float | None,
    estimated_rank_percentile: float | None,
) -> str | None:
    if target_rank_percentile is None:
        return None
    target_text = f"top {target_rank_percentile * 100:.2f}%"
    if target_medal:
        target_text = f"{target_medal} target ({target_text})"
    observed = rank_percentile if rank_percentile is not None else estimated_rank_percentile
    if observed is None:
        return (
            "Medal-aware search policy is active. "
            f"Keep exploration broader until the competition reaches the {target_text} band."
        )
    return (
        "Medal-aware search policy is active. "
        f"Current leaderboard percentile is {(observed * 100):.2f}%, so keep exploring until {target_text} is met."
    )
