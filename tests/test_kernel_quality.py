from __future__ import annotations

from kagglebot.kernel_quality import (
    build_accuracy_potential,
    build_baseline_quality_signal,
    build_code_reference_quality_signal,
    build_validation_metric_alignment,
    detect_candidate_selection_mismatch,
    detect_external_test_label_transfer_signal,
    detect_prediction_distribution_collapse,
    detect_step_bucket_collapse_signal,
    detect_subgroup_collapse_signal,
    extract_competition_faithfulness,
    extract_cv_breakdown_by_model_node,
    infer_capacity_tier,
    infer_data_tier,
    is_significantly_worse,
)


def test_is_significantly_worse_respects_direction_and_margins() -> None:
    assert is_significantly_worse(
        current=1.13,
        reference=1.0,
        direction="minimize",
        rel_margin=0.1,
        abs_margin=0.01,
    )
    assert not is_significantly_worse(
        current=1.05,
        reference=1.0,
        direction="minimize",
        rel_margin=0.1,
        abs_margin=0.01,
    )
    assert is_significantly_worse(
        current=0.84,
        reference=1.0,
        direction="maximize",
        rel_margin=0.1,
        abs_margin=0.01,
    )


def test_extract_cv_breakdown_by_model_node_filters_invalid_entries() -> None:
    payload = {
        "cv_breakdown_by_model_node": {
            "model_2_node_type_1": "0.42",
            "bad": 0.1,
            "model_3_node_type_4": True,
        }
    }

    assert extract_cv_breakdown_by_model_node(payload) == {(2, 1): 0.42}


def test_detect_subgroup_collapse_signal_reports_worst_model_node() -> None:
    payload = {
        "cv_breakdown_by_model_node": {
            "model_1_node_type_1": 0.015,
            "model_1_node_type_2": 0.010,
            "model_2_node_type_1": 0.220,
            "model_2_node_type_2": 0.080,
        },
        "cv_step_buckets": {
            "000-011": 0.04,
            "144-155": 0.11,
        },
    }

    signal = detect_subgroup_collapse_signal(kernel_metrics_payload=payload, direction="minimize")

    assert signal is not None
    assert signal["model_id"] == 2
    assert signal["node_type"] == 1
    assert signal["worst_step_bucket"] == "144-155"
    assert "Subgroup collapse detected" in str(signal["note"])


def test_detect_step_bucket_collapse_signal_reports_count_and_collapse() -> None:
    stable = detect_step_bucket_collapse_signal({"cv_step_buckets": {"a": 0.4, "b": 0.41, "c": 0.39, "d": 0.42}})
    collapsed = detect_step_bucket_collapse_signal({"cv_step_buckets": {"a": 0.4, "b": 0.41, "c": 0.39, "d": 1.6}})

    assert stable["count"] == 4
    assert stable["collapse_detected"] is False
    assert collapsed["count"] == 4
    assert collapsed["collapse_detected"] is True
    assert collapsed["worst_score"] == 1.6


def test_build_validation_metric_alignment_flags_minimize_mismatch() -> None:
    signal = build_validation_metric_alignment(
        current_value=1.3,
        validation_scores=[0.4, 0.42, 0.45],
        direction="minimize",
    )

    assert signal == {
        "best_validation_score": 0.4,
        "validation_score_count": 3,
        "severe_mismatch": True,
    }


def test_build_validation_metric_alignment_flags_maximize_mismatch() -> None:
    signal = build_validation_metric_alignment(
        current_value=0.62,
        validation_scores=[0.9, 0.91],
        direction="maximize",
    )

    assert signal == {
        "best_validation_score": 0.91,
        "validation_score_count": 2,
        "severe_mismatch": True,
    }


def test_build_validation_metric_alignment_handles_missing_scores() -> None:
    signal = build_validation_metric_alignment(
        current_value=0.62,
        validation_scores=[],
        direction="maximize",
    )

    assert signal == {
        "best_validation_score": None,
        "validation_score_count": 0,
        "severe_mismatch": False,
    }


def test_build_baseline_quality_signal_uses_best_minimize_baseline() -> None:
    signal = build_baseline_quality_signal(
        current_value=0.54,
        baseline_candidates=[("mean", 0.60), ("ridge", 0.50)],
        direction="minimize",
    )

    assert signal == {
        "best_source": "ridge",
        "best_score": 0.50,
        "candidate_count": 2,
        "selected_worse_than_baseline": True,
    }


def test_build_baseline_quality_signal_uses_best_maximize_baseline() -> None:
    signal = build_baseline_quality_signal(
        current_value=0.79,
        baseline_candidates=[("mean", 0.62), ("gbm", 0.82)],
        direction="maximize",
    )

    assert signal == {
        "best_source": "gbm",
        "best_score": 0.82,
        "candidate_count": 2,
        "selected_worse_than_baseline": True,
    }


def test_build_baseline_quality_signal_handles_no_candidates() -> None:
    signal = build_baseline_quality_signal(
        current_value=0.79,
        baseline_candidates=[],
        direction="maximize",
    )

    assert signal == {
        "best_source": None,
        "best_score": None,
        "candidate_count": 0,
        "selected_worse_than_baseline": False,
    }


def test_build_code_reference_quality_signal_flags_below_reference() -> None:
    signal = build_code_reference_quality_signal(
        current_value=0.70,
        metric="accuracy",
        code_reference_score=0.76,
        code_reference_source="code_index:user/ref",
        direction="maximize",
    )

    assert signal["score"] == 0.76
    assert signal["comparison_score"] == 0.76
    assert signal["source"] == "code_index:user/ref"
    assert round(float(signal["delta_vs_current"]), 6) == -0.06
    assert signal["below_reference"] is True
    assert "code_reference_score=0.760000" in str(signal["warning"])


def test_build_code_reference_quality_signal_normalizes_percent_accuracy_reference() -> None:
    signal = build_code_reference_quality_signal(
        current_value=0.989,
        metric="accuracy",
        code_reference_score=98.9,
        code_reference_source="code_index:user/ref",
        direction="maximize",
    )

    assert signal["score"] == 98.9
    assert round(float(signal["comparison_score"]), 6) == 0.989
    assert round(float(signal["delta_vs_current"]), 6) == 0.0
    assert signal["below_reference"] is False
    assert signal["warning"] is None


def test_detect_external_test_label_transfer_signal_requires_specific_leak_pattern() -> None:
    benign = {
        "external_data_allowed": True,
        "submission_rows": 100,
        "test_selected_row_count": 100,
    }
    assert detect_external_test_label_transfer_signal(benign) is None

    payload = {
        "final_selected_method": "official_multiview_overlap_mapping",
        "submission_rows": 6872,
        "submission_audit": {
            "external_overlap_trusted": True,
            "exact_coverage_pass": True,
            "external_root_path": "/kaggle/input/some-public-overlap",
            "test_selected_row_count": 6872,
            "uncovered_test_row_count": 0,
            "test_exact_sha1_matched_image_count": 926,
            "official_overlap_audit": {
                "match_type_counts": {
                    "test_selected_rows": {"exact_sha1": 6872},
                },
            },
        },
    }

    signal = detect_external_test_label_transfer_signal(payload)

    assert signal is not None
    assert signal["test_selected_row_count"] == 6872
    assert signal["final_selected_method"] == "official_multiview_overlap_mapping"


def test_detect_candidate_selection_mismatch_reports_better_holdout_candidate() -> None:
    payload = {
        "chosen_pipeline": "cv_only_sparse",
        "pipelines": [
            {
                "name": "public_like_reference",
                "cv_score": 0.0497,
                "holdout_score": 0.0384,
            },
            {
                "name": "cv_only_sparse",
                "cv_score": 0.0752,
                "holdout_score": 0.0200,
            },
        ],
    }

    mismatch = detect_candidate_selection_mismatch(payload=payload, direction="maximize")

    assert mismatch is not None
    assert mismatch["selected"] == "cv_only_sparse"
    assert mismatch["best_secondary_candidate"] == "public_like_reference"


def test_detect_prediction_distribution_collapse_compares_candidate_means() -> None:
    payload = {
        "selected_pipeline": "sparse",
        "pipelines": [
            {
                "name": "dense",
                "prediction_count_summary": {"test": {"mean": 10.0}},
            },
            {
                "name": "sparse",
                "prediction_count_summary": {"test": {"mean": 2.0}},
            },
        ],
    }

    signal = detect_prediction_distribution_collapse(payload)

    assert signal is not None
    assert signal["selected"] == "sparse"
    assert signal["largest_mean_candidate"] == "dense"


def test_extract_competition_faithfulness_prefers_metric_name_over_numeric_metric() -> None:
    faithfulness = extract_competition_faithfulness(
        evaluation_metric="rmse",
        evaluation_score_source="cv",
        kernel_metrics_payload={
            "metric": 0.123,
            "metric_name": "standardized_rmse",
            "score_source": "cv",
            "split_strategy": "timeseries_split",
            "full_dataset_resolved": True,
            "competition_faithful": True,
        },
        evaluation_report_split_strategy=None,
        evaluation_contract={
            "expected_metric": "standardized_rmse",
            "expected_split_strategy": "timeseries_split",
            "accepted_score_sources": ["cv", "holdout"],
            "require_metric_match": True,
            "require_split_match": True,
            "require_trusted_score_source": True,
            "require_competition_faithful": True,
            "require_full_dataset": True,
        },
    )

    assert faithfulness["actual_metric"] == "standardized_rmse"
    assert faithfulness["metric_match"] is True
    assert faithfulness["reasons"] == []


def test_extract_competition_faithfulness_flags_sample_score_source_and_data_mode() -> None:
    faithfulness = extract_competition_faithfulness(
        evaluation_metric="logloss",
        evaluation_score_source="sample_cv",
        kernel_metrics_payload={"dataset_mode": "sample"},
        evaluation_report_split_strategy="kfold",
        evaluation_contract={
            "expected_metric": "logloss",
            "expected_split_strategy": "stratified_kfold",
            "accepted_score_sources": ["cv", "holdout"],
            "require_metric_match": True,
            "require_split_match": True,
            "require_trusted_score_source": True,
            "require_competition_faithful": True,
            "require_full_dataset": True,
        },
    )

    assert "competition_score_source_mismatch" in faithfulness["reasons"]
    assert "competition_split_mismatch" in faithfulness["reasons"]
    assert "competition_evaluation_unfaithful" in faithfulness["reasons"]
    assert "missing_competitive_data" in faithfulness["reasons"]


def test_infer_capacity_tier_uses_pipeline_and_model_summary_hints() -> None:
    assert (
        infer_capacity_tier(
            kernel_metrics_payload={"selected_pipeline": "qwen llm blend"},
            model_summary=None,
        )
        == "extreme"
    )
    assert (
        infer_capacity_tier(
            kernel_metrics_payload={"pipelines": [{"name": "graph transformer"}]},
            model_summary=None,
        )
        == "high"
    )
    assert (
        infer_capacity_tier(
            kernel_metrics_payload={},
            model_summary={"model_name": "ridge baseline"},
        )
        == "low"
    )


def test_infer_data_tier_uses_faithfulness_and_full_dataset_contract() -> None:
    assert (
        infer_data_tier(
            competition_faithfulness={"faithful": True},
            evaluation_contract={"require_full_dataset": True},
        )
        == "high_accuracy_data"
    )
    assert (
        infer_data_tier(
            competition_faithfulness={"metric_match": True, "split_match": True},
            evaluation_contract={},
        )
        == "trusted_eval_data"
    )
    assert (
        infer_data_tier(
            competition_faithfulness={"full_dataset_resolved": False},
            evaluation_contract={"require_full_dataset": True},
        )
        == "minimum_submit_data"
    )


def test_build_accuracy_potential_marks_high_capacity_unfaithful_candidate_as_frontier() -> None:
    potential = build_accuracy_potential(
        score_source="sample",
        kernel_metrics_payload={"selected_pipeline": "transformer ensemble"},
        model_summary=None,
        quality_guard={
            "competition_faithfulness": {
                "faithful": False,
                "metric_match": True,
                "split_match": True,
                "full_dataset_resolved": False,
            },
            "reasons": ["missing_competitive_data"],
        },
        evaluation_contract={"require_full_dataset": True},
    )

    assert potential["status"] == "frontier"
    assert potential["eligible"] is True
    assert potential["capacity_tier"] == "high"
    assert potential["data_tier"] == "minimum_submit_data"


def test_build_accuracy_potential_blocks_hard_policy_reasons() -> None:
    potential = build_accuracy_potential(
        score_source="cv",
        kernel_metrics_payload={"selected_pipeline": "foundation transformer"},
        model_summary=None,
        quality_guard={
            "competition_faithfulness": {"faithful": True},
            "reasons": ["external_test_label_transfer_detected"],
        },
        evaluation_contract=None,
    )

    assert potential["status"] == "blocked"
    assert potential["eligible"] is False
