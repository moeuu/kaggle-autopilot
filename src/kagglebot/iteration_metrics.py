from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from kagglebot.autopilot_helpers import _to_float, _to_int, _update_best_score
from kagglebot.json_utils import load_json_object, write_json_object

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


def append_run_evaluation_report(*, run_dir: Path, iteration: int, payload: dict[str, object]) -> None:
    path = run_dir / "evaluation_report.json"
    state: dict[str, object] = {"latest_iteration": iteration, "latest": payload, "history": [payload]}
    existing = load_json_object(path)
    if existing is not None:
        history_raw = existing.get("history", [])
        if isinstance(history_raw, list):
            history = [item for item in history_raw if isinstance(item, dict)]
        else:
            history = []
        history = [item for item in history if item.get("iteration") != iteration]
        history.append(payload)
        history.sort(key=lambda item: int(item.get("iteration", 0)))
        state["history"] = history
    state["latest_iteration"] = iteration
    state["latest"] = payload
    write_json_object(path, state)


def resume_best_readiness_score(*, run_dir: Path, direction: str, max_iterations: int) -> float | None:
    payload = load_json_object(run_dir / "evaluation_report.json")
    if payload is None:
        return None
    history = _evaluation_history(payload)
    best: float | None = None
    for item in history:
        iteration = _to_int(item.get("iteration"))
        if iteration is not None and iteration > max_iterations:
            continue
        score = _to_float(item.get("readiness_score"))
        if score is None:
            continue
        if _update_best_score(best, score, direction, 0.0):
            best = score
    return best


def resume_noise_guard_state(*, run_dir: Path, max_iterations: int) -> tuple[float | None, int]:
    payload = load_json_object(run_dir / "evaluation_report.json")
    if payload is None:
        return None, 0
    rows: list[dict[str, object]] = []
    for item in _evaluation_history(payload):
        iteration = _to_int(item.get("iteration"))
        if iteration is None or iteration > max_iterations:
            continue
        rows.append(item)
    if not rows:
        return None, 0
    rows.sort(key=lambda item: int(_to_int(item.get("iteration")) or 0))
    streak = 0
    prev_score: float | None = None
    for item in rows:
        score = _to_float(item.get("readiness_score"))
        std = _to_float(item.get("std"))
        if score is None:
            continue
        if prev_score is not None and std is not None:
            threshold = 0.5 * max(std, 0.0)
            if abs(score - prev_score) < threshold:
                streak += 1
            else:
                streak = 0
        prev_score = score
    return prev_score, streak


def _evaluation_history(payload: dict[str, object]) -> list[dict[str, object]]:
    history = payload.get("history")
    if not isinstance(history, list):
        latest = payload.get("latest")
        history = [latest] if isinstance(latest, dict) else []
    return [item for item in history if isinstance(item, dict)]
