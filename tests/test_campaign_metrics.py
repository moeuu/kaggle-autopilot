from __future__ import annotations

from pathlib import Path

from kagglebot.campaign_metrics import (
    extract_campaign_artifact_path,
    extract_campaign_fold_scores,
    extract_campaign_method_id,
    extract_campaign_prediction_correlation,
    extract_campaign_validation_profile_id,
    infer_campaign_candidate_category,
    infer_campaign_feature_set,
    infer_campaign_model_family,
)


def test_infer_campaign_candidate_category_prefers_payload_then_reasons() -> None:
    assert (
        infer_campaign_candidate_category(
            iteration=2,
            kernel_metrics_payload={"candidate_category": "Blend"},
            quality_reasons=["validation drift"],
        )
        == "blend"
    )
    assert (
        infer_campaign_candidate_category(
            iteration=2,
            kernel_metrics_payload={},
            quality_reasons=["needs validation redesign"],
        )
        == "validation_variant"
    )
    assert infer_campaign_candidate_category(iteration=1, kernel_metrics_payload={}, quality_reasons=[]) == (
        "reference_reproduction"
    )
    assert infer_campaign_candidate_category(iteration=2, kernel_metrics_payload={}, quality_reasons=[]) == (
        "strong_single"
    )


def test_infer_campaign_model_family_and_feature_set_prefer_model_summary() -> None:
    model_summary = {"model_family": "lightgbm", "feature_set": "v2"}
    metrics = {"model_name": "catboost", "features": "v1"}

    assert infer_campaign_model_family(model_summary, metrics) == "lightgbm"
    assert infer_campaign_feature_set(model_summary, metrics) == "v2"


def test_extract_campaign_fold_scores_and_prediction_correlations() -> None:
    payload = {
        "fold_scores": ["0.71", "bad", 0.73],
        "prediction_correlation": {"a": "0.2", "b": None, 3: 0.4},
    }

    assert extract_campaign_fold_scores(payload) == [0.71, 0.73]
    assert extract_campaign_prediction_correlation(payload) == {"a": 0.2, "3": 0.4}


def test_extract_campaign_artifact_paths_from_direct_and_nested_payloads() -> None:
    assert extract_campaign_artifact_path({"oof_predictions_path": " oof.npy "}, "oof") == Path("oof.npy")
    assert extract_campaign_artifact_path({"artifacts": {"prediction_path": "pred.npy"}}, "prediction") == Path(
        "pred.npy"
    )
    assert extract_campaign_artifact_path({}, "prediction") is None


def test_extract_campaign_method_and_validation_profile_ids() -> None:
    assert extract_campaign_method_id({"active_method_id": " method-a "}) == "method-a"
    assert extract_campaign_method_id({}) is None
    assert extract_campaign_validation_profile_id({"split_profile_id": " split-a "}) == "split-a"
    assert extract_campaign_validation_profile_id({"readiness": {"split_strategy": "group-kfold"}}) == "group-kfold"
    assert extract_campaign_validation_profile_id({}) is None
