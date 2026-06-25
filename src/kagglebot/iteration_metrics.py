from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kagglebot.eval import EvaluationReport
    from kagglebot.solver.evaluate import EvaluationResult


def evaluation_to_payload(evaluation: EvaluationResult) -> dict[str, object]:
    payload: dict[str, object] = {
        "score_source": evaluation.score_source,
        "metric": evaluation.metric,
        "direction": evaluation.direction,
        "value": evaluation.value,
        "std": evaluation.std,
    }
    if evaluation.fold_scores is not None:
        payload["fold_scores"] = list(evaluation.fold_scores)
    if evaluation.train_score is not None:
        payload["train_score"] = evaluation.train_score
    if evaluation.val_score is not None:
        payload["val_score"] = evaluation.val_score
    return payload


def build_metrics_payload(
    *,
    run_id: str,
    iteration: int,
    evaluation: EvaluationResult,
    target_score: float,
    met_target: bool,
    top1_info: dict[str, object],
    compute: str,
    accelerator: str,
    holdout_frac: float,
    cv_folds: int,
    seed: int,
    evaluation_by_source: dict[str, EvaluationResult] | None = None,
    evaluation_report: EvaluationReport | None = None,
    readiness_target: float | None = None,
    evaluation_contract: dict[str, object] | None = None,
    competition_faithfulness: dict[str, object] | None = None,
    accuracy_potential: dict[str, object] | None = None,
    timestamp: int | None = None,
) -> dict[str, object]:
    payload = {
        "run_id": run_id,
        "iter": iteration,
        "metric": evaluation.metric,
        "direction": evaluation.direction,
        "score_source": evaluation.score_source,
        "offline_value": evaluation.value,
        "offline_std": evaluation.std,
        "target_score": target_score,
        "met_target": met_target,
        "top1_public_score": top1_info.get("score"),
        "top1_public_timestamp": top1_info.get("timestamp"),
        "compute": compute,
        "accelerator": accelerator,
        "timestamp": int(datetime.now(UTC).timestamp()) if timestamp is None else int(timestamp),
        "folds": cv_folds if evaluation.score_source in {"cv", "consensus"} else None,
        "holdout_frac": holdout_frac if evaluation.score_source in {"holdout", "consensus"} else None,
        "seed": seed,
    }
    if evaluation_by_source:
        payload["offline_by_source"] = {
            source: evaluation_to_payload(result) for source, result in evaluation_by_source.items()
        }
    if evaluation_report is not None:
        payload["readiness"] = {
            "score": evaluation_report.readiness_score,
            "mean": evaluation_report.mean,
            "std": evaluation_report.std,
            "ci_low": evaluation_report.ci_low,
            "ci_high": evaluation_report.ci_high,
            "target": readiness_target,
            "split_strategy": evaluation_report.split_strategy,
            "n_splits": evaluation_report.n_splits,
            "seeds": evaluation_report.seeds,
            "repeats": evaluation_report.repeats,
            "drift_auc": evaluation_report.drift_auc,
        }
    if evaluation_contract:
        payload["evaluation_contract"] = evaluation_contract
    if competition_faithfulness:
        payload["competition_faithfulness"] = competition_faithfulness
    if accuracy_potential:
        payload["accuracy_potential"] = accuracy_potential
    return payload
