from __future__ import annotations

import numpy as np

from kagglebot.eval import EvaluationReport
from kagglebot.iteration_metrics import (
    append_run_evaluation_report,
    build_eval_data_cache_fallback,
    build_metrics_payload,
    build_split_index_fingerprints,
    evaluation_to_payload,
    extract_fold_scores_for_report,
    iteration_metrics_allow_submit,
    resume_best_readiness_score,
    resume_noise_guard_state,
)
from kagglebot.solver.evaluate import EvaluationResult


def test_evaluation_to_payload_preserves_optional_scores() -> None:
    evaluation = EvaluationResult(
        score_source="cv",
        metric="rmse",
        direction="minimize",
        value=0.42,
        std=0.03,
        train_score=0.2,
        val_score=0.43,
        fold_scores=[0.4, 0.44],
    )

    assert evaluation_to_payload(evaluation) == {
        "score_source": "cv",
        "metric": "rmse",
        "direction": "minimize",
        "value": 0.42,
        "std": 0.03,
        "fold_scores": [0.4, 0.44],
        "train_score": 0.2,
        "val_score": 0.43,
    }


def test_build_metrics_payload_includes_readiness_and_offline_sources() -> None:
    evaluation = EvaluationResult(
        score_source="cv",
        metric="auc",
        direction="maximize",
        value=0.81,
        std=0.02,
        train_score=None,
        val_score=None,
        fold_scores=[0.8, 0.82],
    )
    holdout = EvaluationResult(
        score_source="holdout",
        metric="auc",
        direction="maximize",
        value=0.79,
        std=None,
        train_score=None,
        val_score=0.79,
        fold_scores=None,
    )
    report = EvaluationReport(
        metric_name="auc",
        direction="maximize",
        split_strategy="stratified_kfold",
        n_splits=5,
        seeds=[42, 2024],
        repeats=2,
        per_fold_scores=[0.8, 0.82],
        mean=0.81,
        std=0.02,
        ci_low=0.78,
        ci_high=0.84,
        drift_auc=0.55,
        readiness_score=0.77,
    )

    payload = build_metrics_payload(
        run_id="run-1",
        iteration=3,
        evaluation=evaluation,
        target_score=0.9,
        met_target=False,
        top1_info={"score": 0.95, "timestamp": "2026-01-01T00:00:00Z"},
        compute="local_gpu",
        accelerator="gpu",
        holdout_frac=0.2,
        cv_folds=5,
        seed=123,
        evaluation_by_source={"holdout": holdout},
        evaluation_report=report,
        readiness_target=0.85,
        evaluation_contract={"faithful": True},
        competition_faithfulness={"faithful": True},
        accuracy_potential={"status": "frontier"},
        timestamp=999,
    )

    assert payload["timestamp"] == 999
    assert payload["folds"] == 5
    assert payload["holdout_frac"] is None
    assert payload["offline_by_source"] == {
        "holdout": {
            "score_source": "holdout",
            "metric": "auc",
            "direction": "maximize",
            "value": 0.79,
            "std": None,
            "val_score": 0.79,
        }
    }
    assert payload["readiness"] == {
        "score": 0.77,
        "mean": 0.81,
        "std": 0.02,
        "ci_low": 0.78,
        "ci_high": 0.84,
        "target": 0.85,
        "split_strategy": "stratified_kfold",
        "n_splits": 5,
        "seeds": [42, 2024],
        "repeats": 2,
        "drift_auc": 0.55,
    }
    assert payload["evaluation_contract"] == {"faithful": True}
    assert payload["competition_faithfulness"] == {"faithful": True}
    assert payload["accuracy_potential"] == {"status": "frontier"}


def test_iteration_metrics_allow_submit_prefers_quality_guard(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text('{"quality_guard": {"allow_submit": false}}\n', encoding="utf-8")
    evaluation = EvaluationResult(
        score_source="cv",
        metric="auc",
        direction="maximize",
        value=0.8,
        std=None,
        train_score=None,
        val_score=None,
        fold_scores=None,
    )

    assert iteration_metrics_allow_submit(path, evaluation) is False


def test_iteration_metrics_allow_submit_uses_faithfulness_when_quality_missing(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text('{"competition_faithfulness": {"faithful": true}}\n', encoding="utf-8")
    evaluation = EvaluationResult(
        score_source="sample",
        metric="auc",
        direction="maximize",
        value=0.8,
        std=None,
        train_score=None,
        val_score=None,
        fold_scores=None,
    )

    assert iteration_metrics_allow_submit(path, evaluation) is True


def test_iteration_metrics_allow_submit_blocks_external_label_transfer(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text('{"external_test_label_transfer": true}\n', encoding="utf-8")
    evaluation = EvaluationResult(
        score_source="cv",
        metric="auc",
        direction="maximize",
        value=0.8,
        std=None,
        train_score=None,
        val_score=None,
        fold_scores=None,
    )

    assert iteration_metrics_allow_submit(path, evaluation) is False


def test_build_eval_data_cache_fallback_normalizes_split() -> None:
    fallback = build_eval_data_cache_fallback(split_strategy="timeseries_split", cv_folds=5)

    assert fallback == {
        "split_strategy": "timeseries_split",
        "n_splits": 5,
        "split_index_fingerprints": [],
        "drift_train_x": None,
        "drift_test_x": None,
    }


def test_extract_fold_scores_for_report_prefers_cv_fold_scores() -> None:
    evaluation = EvaluationResult(
        score_source="holdout",
        metric="auc",
        direction="maximize",
        value=0.7,
        std=None,
        train_score=None,
        val_score=0.7,
        fold_scores=None,
    )
    cv = EvaluationResult(
        score_source="cv",
        metric="auc",
        direction="maximize",
        value=0.8,
        std=0.01,
        train_score=None,
        val_score=None,
        fold_scores=[0.79, 0.81],
    )

    assert extract_fold_scores_for_report(evaluation=evaluation, evaluation_by_source={"cv": cv}) == [0.79, 0.81]


def test_build_split_index_fingerprints_are_stable() -> None:
    class DemoSplitter:
        name = "kfold"

        @property
        def splitter(self):
            return self

        def split(self, x, y):  # noqa: ANN001
            del x, y
            yield np.array([0, 1]), np.array([2])
            yield np.array([2]), np.array([0, 1])

    fingerprints = build_split_index_fingerprints(split=DemoSplitter(), y=np.array([0, 1, 0]), seed=123)

    assert fingerprints == [
        {
            "seed": 123,
            "fold": 0,
            "train_size": 2,
            "valid_size": 1,
            "train_hash": "9d34149fbd1fe777",
            "valid_hash": "d86e8112f3c4c444",
        },
        {
            "seed": 123,
            "fold": 1,
            "train_size": 1,
            "valid_size": 2,
            "train_hash": "d86e8112f3c4c444",
            "valid_hash": "9d34149fbd1fe777",
        },
    ]


def test_append_run_evaluation_report_recovers_invalid_existing_report(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report_path = run_dir / "evaluation_report.json"
    report_path.write_text("{", encoding="utf-8")

    append_run_evaluation_report(
        run_dir=run_dir,
        iteration=1,
        payload={"iteration": 1, "readiness_score": 0.42},
    )

    assert report_path.read_text(encoding="utf-8").strip().startswith("{")
    best = resume_best_readiness_score(run_dir=run_dir, direction="maximize", max_iterations=1)
    assert best == 0.42


def test_resume_best_readiness_score_honors_direction_and_max_iteration(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    append_run_evaluation_report(run_dir=run_dir, iteration=1, payload={"iteration": 1, "readiness_score": 0.50})
    append_run_evaluation_report(run_dir=run_dir, iteration=2, payload={"iteration": 2, "readiness_score": 0.40})
    append_run_evaluation_report(run_dir=run_dir, iteration=3, payload={"iteration": 3, "readiness_score": 0.30})

    assert resume_best_readiness_score(run_dir=run_dir, direction="minimize", max_iterations=2) == 0.40
    assert resume_best_readiness_score(run_dir=run_dir, direction="maximize", max_iterations=2) == 0.50


def test_resume_noise_guard_state_counts_small_changes(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    append_run_evaluation_report(
        run_dir=run_dir,
        iteration=1,
        payload={"iteration": 1, "readiness_score": 0.500, "std": 0.02},
    )
    append_run_evaluation_report(
        run_dir=run_dir,
        iteration=2,
        payload={"iteration": 2, "readiness_score": 0.505, "std": 0.02},
    )
    append_run_evaluation_report(
        run_dir=run_dir,
        iteration=3,
        payload={"iteration": 3, "readiness_score": 0.508, "std": 0.02},
    )

    assert resume_noise_guard_state(run_dir=run_dir, max_iterations=3) == (0.508, 2)
