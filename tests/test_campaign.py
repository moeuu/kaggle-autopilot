from __future__ import annotations

import json
from pathlib import Path

from kagglebot.campaign import (
    CampaignCandidate,
    allocate_submission,
    build_campaign_candidate,
    campaign_state_path,
    candidate_registry_path,
    classify_against_campaign_baseline,
    list_candidates,
    recommend_blend_pairs,
    update_campaign_state,
    upsert_candidate,
)


def test_campaign_state_integrates_history_top1_latest_and_registry(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    registry_path = candidate_registry_path(context_dir)
    candidate = CampaignCandidate(
        candidate_id="run-1-i001-strong_single",
        category="strong_single",
        run_id="run-1",
        iteration=1,
        direction="maximize",
        offline_score=0.78,
        public_score=0.77,
        submitted=True,
    )
    upsert_candidate(registry_path, candidate)

    state = update_campaign_state(
        state_path=campaign_state_path(context_dir),
        registry_path=registry_path,
        slug="demo",
        run_id="run-1",
        mode="top1",
        direction="maximize",
        top1_info={"score": 0.91, "timestamp": "2026-01-01T00:00:00Z"},
        submission_history={"best_score": 0.82, "latest_score": 0.80},
        remaining_daily_slots=2,
    )

    assert state["historical_best_score"] == 0.82
    assert state["latest_submission_score"] == 0.80
    assert state["champion_score"] == 0.82
    assert state["top1_score"] == 0.91
    assert round(float(state["top1_gap"]), 6) == -0.09
    assert state["remaining_daily_slots"] == 2
    assert state["candidates"] == ["run-1-i001-strong_single"]
    assert json.loads(campaign_state_path(context_dir).read_text(encoding="utf-8"))["mode"] == "top1"


def test_campaign_regression_detection_handles_minimize_and_maximize() -> None:
    assert (
        classify_against_campaign_baseline(
            candidate_score=10.2,
            historical_best_score=9.6,
            champion_score=None,
            direction="minimize",
        )
        == "regression"
    )
    assert (
        classify_against_campaign_baseline(
            candidate_score=0.79,
            historical_best_score=0.81,
            champion_score=None,
            direction="maximize",
        )
        == "regression"
    )
    assert (
        classify_against_campaign_baseline(
            candidate_score=9.4,
            historical_best_score=9.6,
            champion_score=None,
            direction="minimize",
        )
        == "improvement"
    )


def test_submit_allocator_obeys_slots_duplicates_and_calibration_exception() -> None:
    candidate = CampaignCandidate(
        candidate_id="run-1-i002-strong_single",
        category="strong_single",
        run_id="run-1",
        iteration=2,
        direction="minimize",
        offline_score=10.1,
    )
    state = {
        "direction": "minimize",
        "historical_best_score": 9.6,
        "champion_score": 9.6,
        "remaining_daily_slots": 1,
    }

    assert allocate_submission(candidate=candidate, campaign_state=state, remaining_daily_slots=0).reason == (
        "daily_submission_limit_reached"
    )
    assert allocate_submission(candidate=candidate, campaign_state=state, is_duplicate=True).reason == (
        "duplicate_submission"
    )
    assert allocate_submission(candidate=candidate, campaign_state=state).reason == "below_campaign_baseline"

    calibration = CampaignCandidate(
        candidate_id="run-1-i003-calibration",
        category="calibration",
        run_id="run-1",
        iteration=3,
        direction="minimize",
        offline_score=10.1,
    )
    decision = allocate_submission(
        candidate=calibration,
        campaign_state=state,
        calibration_exception=True,
        validation_trust=0.9,
        novelty=0.9,
    )
    assert decision.allow_submit is True
    assert decision.reason == "calibration_exception"


def test_candidate_registry_records_metadata_and_blend_pairs(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    submission = tmp_path / "submission.csv"
    submission.write_text("id,target\n1,0\n", encoding="utf-8")
    first = build_campaign_candidate(
        run_id="run-1",
        iteration=1,
        direction="maximize",
        category="strong_single",
        offline_score=0.81,
        offline_std=0.01,
        score_source="cv",
        submission_path=submission,
        metrics_path=tmp_path / "metrics.json",
        oof_path=tmp_path / "oof.npy",
        prediction_path=tmp_path / "test.npy",
        model_family="lgbm",
        feature_set="base",
        method_id="tabular-gbdt-portfolio",
        validation_profile_id="group_or_proxy_cv",
        prediction_correlation={"run-1-i002-feature_variant": 0.72},
    )
    second = CampaignCandidate(
        candidate_id="run-1-i002-feature_variant",
        category="feature_variant",
        run_id="run-1",
        iteration=2,
        direction="maximize",
        offline_score=0.80,
        model_family="catboost",
    )

    registry_path = candidate_registry_path(context_dir)
    upsert_candidate(registry_path, first)
    upsert_candidate(registry_path, second)
    candidates = list_candidates(registry_path)

    assert candidates[0].model_family == "lgbm"
    assert candidates[0].method_id == "tabular-gbdt-portfolio"
    assert candidates[0].validation_profile_id == "group_or_proxy_cv"
    assert candidates[0].oof_path and candidates[0].prediction_path
    assert candidates[0].submission_sha256
    assert candidates[0].prediction_correlation["run-1-i002-feature_variant"] == 0.72
    assert recommend_blend_pairs(candidates, direction="maximize") == [
        ("run-1-i001-strong_single", "run-1-i002-feature_variant")
    ]


def test_submit_allocator_records_information_value_for_validation_candidate() -> None:
    candidate = CampaignCandidate(
        candidate_id="run-1-i002-validation_variant",
        category="validation_variant",
        run_id="run-1",
        iteration=2,
        direction="maximize",
        offline_score=0.83,
    )
    decision = allocate_submission(
        candidate=candidate,
        campaign_state={
            "direction": "maximize",
            "historical_best_score": 0.82,
            "champion_score": 0.82,
            "offline_online_correlation": 0.1,
        },
        novelty=0.7,
    )

    assert decision.information_value is not None
    assert decision.information_value > 0.5
