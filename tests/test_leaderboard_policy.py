from __future__ import annotations

import json

import pytest

from kagglebot.leaderboard_policy import (
    build_medal_target_reason,
    meets_rank_percentile_target,
    resume_best_online_submission_score,
    should_force_major_overhaul_by_rank,
    update_best_online_submission_score,
)
from kagglebot.paths import CompetitionPaths


def test_resume_best_online_submission_score_ignores_invalid_and_non_finite_scores(tmp_path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    run_id = "run-1"
    for iteration, payload in (
        (1, {"submission_score": "nan"}),
        (2, {"submission_score": "0.72"}),
        (3, {"submission_score": "0.68"}),
    ):
        iter_dir = paths.iter_dir(run_id, iteration)
        iter_dir.mkdir(parents=True)
        (iter_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    assert resume_best_online_submission_score(
        paths=paths,
        run_id=run_id,
        direction="minimize",
        max_iterations=3,
    ) == pytest.approx(0.68)


def test_update_best_online_submission_score_respects_direction_and_invalid_values() -> None:
    assert update_best_online_submission_score(
        current_best_score=0.72,
        candidate_score="0.68",
        direction="minimize",
    ) == pytest.approx(0.68)
    assert update_best_online_submission_score(
        current_best_score=0.72,
        candidate_score="0.80",
        direction="minimize",
    ) == pytest.approx(0.72)
    assert update_best_online_submission_score(
        current_best_score=0.72,
        candidate_score="0.80",
        direction="maximize",
    ) == pytest.approx(0.80)
    assert update_best_online_submission_score(
        current_best_score=0.72,
        candidate_score="nan",
        direction="maximize",
    ) == pytest.approx(0.72)


def test_rank_policy_forces_major_overhaul_only_for_poor_large_competitions() -> None:
    assert should_force_major_overhaul_by_rank(rank=1300, total_teams=2700, max_percentile=0.35, min_teams=200)
    assert not should_force_major_overhaul_by_rank(rank=200, total_teams=2700, max_percentile=0.35, min_teams=200)
    assert not should_force_major_overhaul_by_rank(rank=50, total_teams=120, max_percentile=0.35, min_teams=200)


def test_medal_target_policy_uses_observed_or_estimated_percentile() -> None:
    assert meets_rank_percentile_target(
        rank_percentile=None,
        estimated_rank_percentile=0.005,
        target_rank_percentile=0.01,
    )
    assert not meets_rank_percentile_target(
        rank_percentile=0.02,
        estimated_rank_percentile=None,
        target_rank_percentile=0.01,
    )

    reason = build_medal_target_reason(
        target_medal="gold",
        target_rank_percentile=0.01,
        rank_percentile=0.02,
        estimated_rank_percentile=None,
    )

    assert reason is not None
    assert "gold target" in reason
    assert "2.00%" in reason
