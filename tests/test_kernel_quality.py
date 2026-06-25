from __future__ import annotations

from kagglebot.kernel_quality import (
    detect_candidate_selection_mismatch,
    detect_external_test_label_transfer_signal,
    detect_prediction_distribution_collapse,
    detect_subgroup_collapse_signal,
    extract_cv_breakdown_by_model_node,
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
