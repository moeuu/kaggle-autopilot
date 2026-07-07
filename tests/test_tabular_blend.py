from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from kagglebot.kernel_runtime.tabular_blend import (
    build_hill_climb_candidates,
    make_component_blend_result,
    make_logit_blend_result,
    select_blend_candidate_pool,
    select_top_blend_components,
    should_select_blend_candidate,
)
from kagglebot.kernel_runtime.tabular_ensemble import PipelineResult


def _result(
    name: str,
    *,
    cv_score: float,
    family: str,
    oof_preds: list[float],
    test_preds: list[float],
    seed_scores: list[dict[str, float]] | None = None,
) -> PipelineResult:
    return PipelineResult(
        name=name,
        oof_preds=np.asarray(oof_preds, dtype=np.float64),
        test_preds=np.asarray(test_preds, dtype=np.float64),
        cv_score=cv_score,
        fold_scores=[],
        feature_manifest={"final_feature_count": 3},
        metadata={
            "kind": "single",
            "model_family": family,
            "suite_name": "suite_a" if family != "lightgbm" else "suite_b",
            "model_seeds": [42, 2024],
            "seed_scores": seed_scores or [{"seed": 42, "auc": cv_score}, {"seed": 2024, "auc": cv_score}],
            "seed_oof_preds": {
                42: np.asarray(oof_preds, dtype=np.float64),
                2024: np.asarray(oof_preds, dtype=np.float64),
            },
            "seed_test_preds": {
                42: np.asarray(test_preds, dtype=np.float64),
                2024: np.asarray(test_preds, dtype=np.float64),
            },
        },
        test_predictions_by_fold={"fold_1": np.asarray([test_preds[0]]), "fold_2": np.asarray([test_preds[1]])},
        oof_predictions_by_fold={"fold_1": np.asarray([oof_preds[0]]), "fold_2": np.asarray([oof_preds[1]])},
        valid_indices_by_fold={"fold_1": np.asarray([0]), "fold_2": np.asarray([1])},
    )


def test_select_top_blend_components_prefers_distinct_families() -> None:
    selected = select_top_blend_components(
        [
            _result("xgb_best", cv_score=0.92, family="xgb", oof_preds=[0.1, 0.9], test_preds=[0.2, 0.8]),
            _result("xgb_second", cv_score=0.919, family="xgb", oof_preds=[0.2, 0.8], test_preds=[0.3, 0.7]),
            _result("cat_best", cv_score=0.918, family="catboost", oof_preds=[0.15, 0.85], test_preds=[0.25, 0.75]),
        ]
    )

    assert [result.name for result in selected] == ["xgb_best", "cat_best"]


def test_should_select_blend_candidate_requires_positive_margin_and_seed_support() -> None:
    best_single = _result(
        "xgb_best",
        cv_score=0.92,
        family="xgb",
        oof_preds=[0.1, 0.9],
        test_preds=[0.2, 0.8],
        seed_scores=[{"seed": 42, "auc": 0.92}, {"seed": 2024, "auc": 0.919}],
    )
    accepted_blend = _result(
        "blend_ok",
        cv_score=0.921,
        family="blend",
        oof_preds=[0.12, 0.88],
        test_preds=[0.22, 0.78],
        seed_scores=[{"seed": 42, "auc": 0.921}, {"seed": 2024, "auc": 0.9195}],
    )
    rejected_blend = _result(
        "blend_bad",
        cv_score=0.921,
        family="blend",
        oof_preds=[0.12, 0.88],
        test_preds=[0.22, 0.78],
        seed_scores=[{"seed": 42, "auc": 0.918}, {"seed": 2024, "auc": 0.9185}],
    )

    assert should_select_blend_candidate(accepted_blend, best_single, min_margin=0.00005) is True
    assert should_select_blend_candidate(rejected_blend, best_single, min_margin=0.00005) is False


def test_should_select_blend_candidate_accepts_regression_seed_scores() -> None:
    best_single = _result(
        "ridge_best",
        cv_score=-2.0,
        family="ridge",
        oof_preds=[10.0, 20.0],
        test_preds=[12.0, 22.0],
        seed_scores=[{"seed": 42, "score": -2.0, "metric": "rmse"}, {"seed": 2024, "score": -2.2, "metric": "rmse"}],
    )
    accepted_blend = _result(
        "blend_ok",
        cv_score=-1.8,
        family="blend",
        oof_preds=[11.0, 19.0],
        test_preds=[13.0, 21.0],
        seed_scores=[{"seed": 42, "score": -1.7, "metric": "rmse"}, {"seed": 2024, "score": -2.2, "metric": "rmse"}],
    )
    rejected_blend = _result(
        "blend_bad",
        cv_score=-1.8,
        family="blend",
        oof_preds=[11.0, 19.0],
        test_preds=[13.0, 21.0],
        seed_scores=[{"seed": 42, "score": -2.4, "metric": "rmse"}, {"seed": 2024, "score": -2.3, "metric": "rmse"}],
    )

    assert should_select_blend_candidate(accepted_blend, best_single, min_margin=0.00005) is True
    assert should_select_blend_candidate(rejected_blend, best_single, min_margin=0.00005) is False


def test_make_logit_blend_result_builds_expected_metadata() -> None:
    bundle = SimpleNamespace(target_values=np.asarray([0, 1], dtype=np.int8))
    artifacts = SimpleNamespace(suite_name="comp_only", train_mode="competition_only", feature_recipe="full")
    first = _result("xgb_a", cv_score=0.91, family="xgb", oof_preds=[0.2, 0.8], test_preds=[0.25, 0.75])
    second = _result(
        "cat_b",
        cv_score=0.909,
        family="catboost",
        oof_preds=[0.3, 0.7],
        test_preds=[0.35, 0.65],
    )

    result = make_logit_blend_result(
        bundle=bundle,
        artifacts=artifacts,
        results_by_name={"xgb_a": first, "cat_b": second},
        first_name="xgb_a",
        second_name="cat_b",
        first_weight=0.5,
        outer_folds=2,
    )

    assert result.name.startswith("logit_blend_xgb_a_cat_b_w50")
    assert result.metadata["method"] == "logit"
    assert result.metadata["kind"] == "logit_blend"
    assert result.metadata["blend_components"] == ["xgb_a", "cat_b"]
    assert np.all(result.oof_preds > 0.0)
    assert np.all(result.oof_preds < 1.0)


def test_make_component_blend_result_preserves_regression_predictions() -> None:
    bundle = SimpleNamespace(
        target_values=np.asarray([10.0, 30.0], dtype=np.float64),
        prediction_kind="regression",
        target_col="target",
    )
    artifacts = SimpleNamespace(suite_name="regression_suite", train_mode="competition_only", feature_recipe="full")
    first = _result(
        "ridge_a",
        cv_score=-4.0,
        family="ridge",
        oof_preds=[10.0, 20.0],
        test_preds=[10.0, 40.0],
    )
    second = _result(
        "ridge_b",
        cv_score=-5.0,
        family="ridge",
        oof_preds=[20.0, 40.0],
        test_preds=[20.0, 60.0],
    )

    result = make_component_blend_result(
        bundle=bundle,
        artifacts=artifacts,
        results_by_name={"ridge_a": first, "ridge_b": second},
        component_weights={"ridge_a": 0.5, "ridge_b": 0.5},
        method="weighted",
        outer_folds=2,
    )

    assert result.metadata["metric"] == "rmse"
    assert result.metadata["prediction_kind"] == "regression"
    assert result.metadata["prediction_range"] == [15.0, 50.0]
    assert np.allclose(result.oof_preds, [15.0, 30.0])
    assert np.allclose(result.test_preds, [15.0, 50.0])
    assert result.cv_score == -float(np.sqrt(((10.0 - 15.0) ** 2 + (30.0 - 30.0) ** 2) / 2.0))


def test_make_component_blend_result_clips_count_regression_predictions() -> None:
    bundle = SimpleNamespace(
        target_values=np.asarray([0.0, 20.0], dtype=np.float64),
        prediction_kind="count_regression",
        target_col="count",
    )
    artifacts = SimpleNamespace(suite_name="count_suite", train_mode="competition_only", feature_recipe="full")
    first = _result(
        "count_a",
        cv_score=-1.0,
        family="ridge",
        oof_preds=[-10.0, 12.0],
        test_preds=[-5.0, 16.0],
    )
    second = _result(
        "count_b",
        cv_score=-2.0,
        family="ridge",
        oof_preds=[-4.0, 24.0],
        test_preds=[-3.0, 28.0],
    )

    result = make_component_blend_result(
        bundle=bundle,
        artifacts=artifacts,
        results_by_name={"count_a": first, "count_b": second},
        component_weights={"count_a": 0.5, "count_b": 0.5},
        method="weighted",
        outer_folds=2,
    )

    assert result.metadata["metric"] == "rmsle"
    assert result.metadata["prediction_range"] == [0.0, 22.0]
    assert np.allclose(result.oof_preds, [0.0, 18.0])
    assert np.allclose(result.test_preds, [0.0, 22.0])


def test_select_blend_candidate_pool_keeps_diverse_families_and_suites() -> None:
    selected = select_blend_candidate_pool(
        [
            _result("xgb_best", cv_score=0.921, family="xgb", oof_preds=[0.1, 0.9], test_preds=[0.2, 0.8]),
            _result("xgb_second", cv_score=0.920, family="xgb", oof_preds=[0.12, 0.88], test_preds=[0.22, 0.78]),
            _result("cat_best", cv_score=0.919, family="catboost", oof_preds=[0.18, 0.82], test_preds=[0.28, 0.72]),
            _result("lgbm_best", cv_score=0.918, family="lightgbm", oof_preds=[0.3, 0.7], test_preds=[0.35, 0.65]),
        ],
        max_candidates=3,
    )

    assert [result.name for result in selected] == ["xgb_best", "cat_best", "lgbm_best"]


def test_build_hill_climb_candidates_emits_three_component_candidate() -> None:
    bundle = SimpleNamespace(target_values=np.asarray([0, 0, 1, 1], dtype=np.int8))
    artifacts = SimpleNamespace(
        suite_name="comp_plus_orig",
        train_mode="competition_plus_original",
        feature_recipe="full",
        original_row_weight=0.425,
    )
    results = [
        _result("xgb_a", cv_score=0.910, family="xgb", oof_preds=[0.10, 0.35, 0.72, 0.85], test_preds=[0.2, 0.8]),
        _result(
            "cat_b",
            cv_score=0.909,
            family="catboost",
            oof_preds=[0.18, 0.28, 0.68, 0.88],
            test_preds=[0.3, 0.7],
        ),
        _result(
            "lgbm_c", cv_score=0.907, family="lightgbm", oof_preds=[0.14, 0.40, 0.60, 0.92], test_preds=[0.25, 0.75]
        ),
    ]

    generated = build_hill_climb_candidates(
        bundle=bundle,
        artifacts=artifacts,
        results_by_name={result.name: result for result in results},
        candidate_results=results,
        outer_folds=2,
    )

    assert generated
    assert any(len(result.metadata["blend_components"]) >= 3 for result in generated)


def test_build_hill_climb_candidates_uses_weighted_only_for_regression() -> None:
    bundle = SimpleNamespace(
        target_values=np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float64),
        prediction_kind="regression",
        target_col="target",
    )
    artifacts = SimpleNamespace(
        suite_name="regression_suite",
        train_mode="competition_only",
        feature_recipe="full",
        original_row_weight=None,
    )
    results = [
        _result("ridge_a", cv_score=-2.0, family="ridge", oof_preds=[9.0, 19.0, 31.0, 41.0], test_preds=[12.0, 44.0]),
        _result("ridge_b", cv_score=-3.0, family="ridge", oof_preds=[12.0, 18.0, 28.0, 42.0], test_preds=[15.0, 48.0]),
        _result("ridge_c", cv_score=-4.0, family="ridge", oof_preds=[8.0, 22.0, 33.0, 39.0], test_preds=[10.0, 50.0]),
    ]

    generated = build_hill_climb_candidates(
        bundle=bundle,
        artifacts=artifacts,
        results_by_name={result.name: result for result in results},
        candidate_results=results,
        outer_folds=2,
    )

    assert generated
    assert {result.metadata["method"] for result in generated} == {"weighted"}
    assert all(result.metadata["metric"] == "rmse" for result in generated)
    assert all(result.metadata["prediction_range"][1] > 1.0 for result in generated)
