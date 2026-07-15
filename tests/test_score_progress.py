from __future__ import annotations

import pytest

from kagglebot.score_progress import (
    IterationPhase,
    classify_improvement_mode,
    effective_best_score_for_progress,
    is_confirmed_first_place,
    is_conservative_feature_collapse,
    is_severe_regression_vs_best,
    normalize_code_reference_score_for_comparison,
    regression_drop_threshold,
    resolve_explicit_official_metric_override,
    score_delta_vs_reference,
    should_update_best_accuracy_candidate,
)


def test_iteration_phase_delta_and_best_update_policy() -> None:
    minimize = IterationPhase(metric_direction="minimize")
    maximize = IterationPhase(metric_direction="maximize")

    assert minimize.delta_from_best(None, 0.4) is None
    assert minimize.delta_from_best(0.5, 0.4) == pytest.approx(0.1)
    assert maximize.delta_from_best(0.5, 0.6) == pytest.approx(0.1)
    assert minimize.should_update_best(0.5, 0.4, 0.1)
    assert not minimize.should_update_best(0.5, 0.41, 0.1)
    assert maximize.should_update_best(0.5, 0.6, 0.1)
    assert not maximize.should_update_best(0.5, 0.59, 0.1)


def test_resolve_explicit_official_metric_override_accepts_generic_plan_fallback() -> None:
    official_metric = "Geometric mean of corpus BLEU and chrF++ (micro-averaged sufficient statistics)"
    payload = {"official_metric": official_metric, "metric": official_metric}

    override = resolve_explicit_official_metric_override(
        payload,
        target_metric="accuracy",
        evaluation_metric=official_metric,
    )

    assert override == official_metric


def test_resolve_explicit_official_metric_override_rejects_non_generic_target() -> None:
    official_metric = "Geometric mean of corpus BLEU and chrF++ (micro-averaged sufficient statistics)"
    payload = {"official_metric": official_metric, "metric": official_metric}

    override = resolve_explicit_official_metric_override(
        payload,
        target_metric="rmse_on_log_target",
        evaluation_metric=official_metric,
    )

    assert override is None


def test_score_progress_classifies_modes_and_deltas() -> None:
    mode, gap = classify_improvement_mode(0.70, 0.78, "maximize")
    assert mode == "major_overhaul"
    assert gap == pytest.approx(0.08)
    mode, gap = classify_improvement_mode(0.765, 0.78, "maximize")
    assert mode == "moderate_update"
    assert gap == pytest.approx(0.015)
    mode, gap = classify_improvement_mode(0.775, 0.78, "maximize")
    assert mode == "minor_tuning"
    assert gap == pytest.approx(0.005)
    assert score_delta_vs_reference(0.70, 0.74, "maximize") == pytest.approx(-0.04)
    assert score_delta_vs_reference(0.70, 0.74, "minimize") == pytest.approx(0.04)


def test_normalize_code_reference_percentage_for_bounded_metric() -> None:
    assert normalize_code_reference_score_for_comparison(current=0.86, reference=87.5, metric="accuracy") == 0.875
    assert normalize_code_reference_score_for_comparison(
        current=0.86,
        reference=948.0,
        metric="macro-averaged ROC-AUC skipping classes with no positive labels",
    ) == pytest.approx(0.948)
    assert normalize_code_reference_score_for_comparison(current=12.0, reference=87.5, metric="rmse") == 87.5


def test_regression_and_conservative_collapse_guards() -> None:
    assert regression_drop_threshold(metric="auc", direction="maximize") == pytest.approx(0.03)
    assert regression_drop_threshold(metric="rmse", direction="minimize") == pytest.approx(0.10)
    assert is_severe_regression_vs_best(metric="auc", direction="maximize", best_score=0.91, current_score=0.87)
    assert not is_severe_regression_vs_best(metric="auc", direction="maximize", best_score=0.91, current_score=0.89)
    assert is_conservative_feature_collapse(
        {"selected_feature_count": 4, "robust_subset_report": {"selected_features": ["a", "b", "c", "d"]}}
    )
    assert not is_conservative_feature_collapse(
        {"selected_feature_count": 8, "robust_subset_report": {"selected_features": list("abcdefgh")}}
    )


def test_effective_best_score_clips_implausible_previous_best() -> None:
    effective_best, guard = effective_best_score_for_progress(
        prev_best=0.999511,
        current_score=0.799651,
        top1_score=0.78,
        direction="maximize",
    )

    assert effective_best is not None
    assert effective_best < 0.999511
    assert guard is not None
    assert guard["reason"] == "clip_prev_best_above_top1_band"


def test_rank_and_best_accuracy_candidate_policy() -> None:
    assert is_confirmed_first_place(1, None)
    assert not is_confirmed_first_place(1, "leaderboard_score_estimate")
    assert should_update_best_accuracy_candidate(
        current_potential={"frontier_priority": 2},
        best_potential={"frontier_priority": 1},
        current_score=0.70,
        best_score=0.75,
        direction="maximize",
    )
    assert not should_update_best_accuracy_candidate(
        current_potential={"frontier_priority": 1},
        best_potential={"frontier_priority": 2},
        current_score=0.90,
        best_score=0.75,
        direction="maximize",
    )
    assert should_update_best_accuracy_candidate(
        current_potential={"frontier_priority": "2.9"},
        best_potential={"frontier_priority": "2"},
        current_score=0.80,
        best_score=0.75,
        direction="maximize",
    )
