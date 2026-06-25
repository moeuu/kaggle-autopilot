from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from kagglebot.kernel_metrics import (
    collect_kernel_log_text,
    extract_baseline_candidates_from_metrics_payload,
    extract_baseline_scores_from_log_text,
    extract_kernel_metric,
    extract_numeric_list,
    extract_trusted_cv_value_from_metrics_payload,
    extract_validation_scores_from_log_text,
    load_kernel_metrics,
    persist_metric_recheck_payload,
    pick_oof_prediction_column,
    pick_oof_target_column,
)


def test_extract_trusted_cv_value_prefers_named_cv_score() -> None:
    payload = {
        "score": 0.7,
        "cv_mean": "0.42",
        "fold_scores": [0.3, 0.4],
    }

    assert extract_trusted_cv_value_from_metrics_payload(payload) == 0.42


def test_extract_trusted_cv_value_averages_numeric_fold_scores() -> None:
    payload = {"fold_scores": [0.8, "ignored", 1.0, True, None, "0.6", "nan", "inf"]}

    assert extract_trusted_cv_value_from_metrics_payload(payload) == pytest.approx(0.8)


def test_pick_oof_columns_prefers_probability_for_proba_metrics() -> None:
    frame = pd.DataFrame(
        {
            "row_id": [1, 2],
            "target": [0, 1],
            "prediction": [0, 1],
            "oof_proba": [0.2, 0.8],
        }
    )

    assert pick_oof_target_column(frame) == "target"
    assert pick_oof_prediction_column(frame, metric="roc_auc") == "oof_proba"
    assert pick_oof_prediction_column(frame, metric="rmse") == "prediction"


def test_extract_numeric_list_keeps_numeric_items_only() -> None:
    assert extract_numeric_list([0.1, "0.2", 3, True, None]) == [0.1, 3.0, 1.0]
    assert extract_numeric_list(["0.1", None]) is None
    assert extract_numeric_list("0.1") is None


def test_persist_metric_recheck_payload_writes_canonical_metric_paths(tmp_path: Path) -> None:
    iter_dir = tmp_path / "iter-1"
    resolved_path = iter_dir / "custom" / "metrics.json"
    payload = {"metric": "auc", "offline_value": 0.9}

    persist_metric_recheck_payload(
        iter_dir=iter_dir,
        resolved_metrics_path=resolved_path,
        payload=payload,
    )

    for path in (resolved_path, iter_dir / "metrics.json", iter_dir / "output" / "metrics.json"):
        assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_persist_metric_recheck_payload_dedupes_resolved_iter_metrics_path(tmp_path: Path) -> None:
    iter_dir = tmp_path / "iter-1"
    resolved_path = iter_dir / "metrics.json"

    persist_metric_recheck_payload(
        iter_dir=iter_dir,
        resolved_metrics_path=resolved_path,
        payload={"metric": "rmse"},
    )

    assert json.loads(resolved_path.read_text(encoding="utf-8")) == {"metric": "rmse"}
    assert json.loads((iter_dir / "output" / "metrics.json").read_text(encoding="utf-8")) == {"metric": "rmse"}


def test_extract_kernel_metric_ignores_non_finite_direct_values() -> None:
    payload = {
        "metric": "rmse",
        "score": float("inf"),
        "rmse": "0.42",
    }

    assert extract_kernel_metric(payload, "rmse") == ("rmse", 0.42)


def test_extract_kernel_metric_from_selected_combined_score_schema() -> None:
    payload = {
        "primary_metric": "0.5*mAP@[0.5:0.95] + 0.5*F1",
        "selected": {
            "name": "yolo11m_kfold_wbf_geom_rp",
            "mean_map": 0.6698263357562932,
            "oof_f1": 0.6666666666666666,
            "combined_score": 0.66824650121148,
        },
    }

    metric, value = extract_kernel_metric(payload, "0.5*mAP@[0.5:0.95] + 0.5*F1")

    assert metric == "0.5*mAP@[0.5:0.95] + 0.5*F1"
    assert value == 0.66824650121148


def test_extract_kernel_metric_prefers_map_when_primary_metric_is_map() -> None:
    payload = {
        "primary_metric": "mAP@[0.5:0.95]",
        "selected": {
            "mean_map": 0.669,
            "oof_f1": 0.123,
            "combined_score": 0.456,
        },
    }

    metric, value = extract_kernel_metric(payload, "mAP@[0.5:0.95]")

    assert metric == "mAP@[0.5:0.95]"
    assert value == 0.669


def test_extract_kernel_metric_from_selected_cv_mean_schema() -> None:
    payload = {
        "metric": "rmse_on_log_target",
        "selected_cv_mean": 0.11915219856213632,
        "selected_cv_std": 0.016552210168151973,
        "selected_pipeline": "demo_pipeline",
        "leaderboard": [
            {"pipeline": "demo_pipeline", "cv_mean": 0.11915219856213632, "cv_std": 0.016552210168151973},
            {"pipeline": "other_pipeline", "cv_mean": 0.12005931755814533, "cv_std": 0.026762972223791422},
        ],
        "target_direction": "minimize",
    }

    metric, value = extract_kernel_metric(payload, "rmse")

    assert metric == "rmse_on_log_target"
    assert value == 0.11915219856213632


def test_load_kernel_metrics_supports_selected_cv_mean_schema(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "leaderboard": [
                    {
                        "cv_mean": 0.11915219856213632,
                        "cv_std": 0.016552210168151973,
                        "pipeline": "demo_pipeline",
                    }
                ],
                "metric": "rmse_on_log_target",
                "selected_cv_mean": 0.11915219856213632,
                "selected_cv_std": 0.016552210168151973,
                "selected_pipeline": "demo_pipeline",
                "target_direction": "minimize",
            }
        ),
        encoding="utf-8",
    )

    evaluation = load_kernel_metrics(metrics_path, direction="maximize", target_metric="rmse")

    assert evaluation is not None
    assert evaluation.metric == "rmse_on_log_target"
    assert evaluation.direction == "minimize"
    assert evaluation.value == 0.11915219856213632
    assert evaluation.std == 0.016552210168151973


def test_load_kernel_metrics_falls_back_to_cv_for_untrusted_score_source(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "metric": "brier_score",
                "direction": "minimize",
                "score_source": "oracle",
                "brier_score": 0.0,
                "cv_brier": 0.17321,
            }
        ),
        encoding="utf-8",
    )

    evaluation = load_kernel_metrics(metrics_path, direction="minimize", target_metric="brier_score")

    assert evaluation is not None
    assert evaluation.score_source == "cv"
    assert evaluation.value == pytest.approx(0.17321)


def test_extract_kernel_metric_supports_pipelines_cv_mean_schema() -> None:
    payload = {
        "metric": "rmse",
        "pipelines": [
            {"name": "p1", "cv_mean": "0.12", "cv_std": "0.02"},
            {"name": "p2", "cv_mean": "0.11", "cv_std": "0.01"},
        ],
        "selected": {"name": "p2"},
        "target_direction": "minimize",
    }

    metric, value = extract_kernel_metric(payload, "rmse")

    assert metric == "rmse"
    assert value == 0.11


def test_extract_kernel_metric_from_direct_brier_score_key() -> None:
    payload = {
        "metric": "brier_score",
        "brier_score": 0.18,
        "direction": "minimize",
    }

    metric, value = extract_kernel_metric(payload, "brier_score")

    assert metric == "brier_score"
    assert value == 0.18


def test_extract_kernel_metric_from_oof_dict_respects_selected_key() -> None:
    payload = {
        "oof_rmse": {
            "lgb": 8.75,
            "catboost": 8.79,
            "xgboost": 8.84,
            "stacked": 8.76,
            "average": 8.77,
            "selected": 8.76,
        },
        "selection": "selected",
    }

    metric, value = extract_kernel_metric(payload, "rmse")

    assert metric == "rmse"
    assert value == 8.76


def test_extract_baseline_candidates_from_metrics_payload() -> None:
    payload = {
        "pipelines": [
            {"name": "candidate", "cv_mean": 0.5},
            {"name": "class_prior_baseline", "cv_mean": 0.33},
        ],
        "persistence_scores": [0.4, {"score": 0.38}],
        "baseline_nested": {"mean": 0.37},
    }

    candidates = extract_baseline_candidates_from_metrics_payload(payload)

    assert candidates == [
        ("pipelines:class_prior_baseline", 0.33),
        ("metrics:persistence_scores[0]", 0.4),
        ("metrics:persistence_scores[1]", 0.38),
        ("metrics:baseline_nested.mean", 0.37),
    ]


def test_collect_kernel_log_text_reads_only_kernel_like_logs(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "kernel.log").write_text("kernel text", encoding="utf-8")
    (logs_dir / "stdout.log").write_text("stdout text", encoding="utf-8")
    (logs_dir / "notes.log").write_text("ignored", encoding="utf-8")
    (logs_dir / "kernel.txt").write_text("ignored extension", encoding="utf-8")

    text = collect_kernel_log_text(logs_dir)

    assert "kernel text" in text
    assert "stdout text" in text
    assert "ignored" not in text


def test_extract_validation_scores_from_log_text_filters_metric_name() -> None:
    log_text = "\n".join(
        [
            "fold 0 val_rmse = 0.45",
            "fold 0 val_mae = 0.25",
            "fold 1 val_rmse=4.2e-1",
        ]
    )

    assert extract_validation_scores_from_log_text(log_text, "rmse") == [0.45, 0.42]


def test_extract_baseline_scores_from_log_text_skips_fold_lines() -> None:
    log_text = "\n".join(
        [
            "baseline_rmse=0.50",
            "fold=1 baseline_rmse=0.48",
            "persistence_auc = 0.61",
            "candidate_rmse=0.44",
        ]
    )

    assert extract_baseline_scores_from_log_text(log_text) == [0.5, 0.61]
