from __future__ import annotations

from kagglebot.leaderboard_anomaly import assess_leaderboard_anomaly


def test_observed_bottom_decile_requires_implementation_audit() -> None:
    assessment = assess_leaderboard_anomaly(
        direction="maximize",
        online_score=0.42,
        offline_score=0.81,
        top1_score=0.90,
        rank=96,
        total_teams=100,
    )

    assert assessment is not None
    assert assessment.severity == "high"
    assert assessment.confidence == "high"
    assert "observed_bottom_decile" in assessment.signals
    assert assessment.to_payload()["implementation_audit_required"] is True


def test_observed_bottom_two_percent_is_critical() -> None:
    assessment = assess_leaderboard_anomaly(
        direction="maximize",
        online_score=0.0,
        offline_score=86.5,
        top1_score=0.86,
        rank="99",
        total_teams="100",
    )

    assert assessment is not None
    assert assessment.severity == "critical"
    assert assessment.signals == (
        "observed_bottom_two_percent",
        "online_score_collapse_vs_top1",
        "offline_online_scale_or_output_collapse",
    )


def test_estimated_bottom_rank_needs_independent_score_evidence() -> None:
    assert (
        assess_leaderboard_anomaly(
            direction="maximize",
            online_score=0.65,
            top1_score=0.90,
            estimated_rank=990,
            estimated_total_teams=1000,
        )
        is None
    )

    assessment = assess_leaderboard_anomaly(
        direction="maximize",
        online_score=0.0,
        top1_score=0.90,
        estimated_rank=990,
        estimated_total_teams=1000,
    )
    assert assessment is not None
    assert assessment.confidence == "high"
    assert "estimated_bottom_two_percent" in assessment.signals


def test_small_leaderboard_does_not_trigger_rank_anomaly() -> None:
    assert (
        assess_leaderboard_anomaly(
            direction="maximize",
            online_score=0.65,
            top1_score=0.90,
            rank=9,
            total_teams=10,
        )
        is None
    )


def test_minimize_metric_does_not_treat_zero_as_bad() -> None:
    assert (
        assess_leaderboard_anomaly(
            direction="minimize",
            online_score=0.0,
            offline_score=0.1,
            top1_score=0.01,
            rank=1,
            total_teams=100,
        )
        is None
    )
