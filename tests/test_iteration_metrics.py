from __future__ import annotations

from kagglebot.eval import EvaluationReport
from kagglebot.iteration_metrics import build_metrics_payload, evaluation_to_payload
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
