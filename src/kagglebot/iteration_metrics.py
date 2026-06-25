from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from kagglebot import plan_policy
from kagglebot.autopilot_helpers import _to_float, _to_int, _update_best_score
from kagglebot.eval import (
    DriftChecker,
    EvaluationReport,
    SplitStrategyFactory,
    SubmissionReadinessScorer,
    UncertaintyEstimator,
)
from kagglebot.json_utils import load_json_object, write_json_object
from kagglebot.solver.io import load_competition_data

if TYPE_CHECKING:
    from kagglebot.solver.evaluate import EvaluationResult

DEFAULT_EVAL_SEEDS = [42, 2024, 777]
DEFAULT_EVAL_REPEATS = 2
EVAL_REPEAT_SEED_OFFSET = 1009


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


def build_iteration_evaluation_report(
    *,
    config: object,
    run_id: str,
    iteration: int,
    evaluation: EvaluationResult,
    evaluation_by_source: dict[str, EvaluationResult],
    metric_direction: str,
    cv_folds: int,
    split_strategy: str | None,
    seed: int,
    eval_seeds: list[int],
    eval_repeats: int,
    score_source: str,
    ci_method: str,
    ci_alpha: float,
    readiness_method: str,
    readiness_k: float,
    drift_check_enabled: bool,
    drift_weight: float,
    eval_data_cache: dict[str, object] | None,
) -> tuple[EvaluationReport, dict[str, object], dict[str, object] | None]:
    cache = ensure_eval_data_cache(
        config=config,
        cv_folds=cv_folds,
        split_strategy=split_strategy,
        seed=seed,
        eval_seeds=eval_seeds,
        eval_repeats=eval_repeats,
        score_source=score_source,
        eval_data_cache=eval_data_cache,
    )
    fold_scores = extract_fold_scores_for_report(evaluation=evaluation, evaluation_by_source=evaluation_by_source)

    ci_method_value = "bootstrap" if str(ci_method).lower() == "bootstrap" else "normal"
    uncertainty = UncertaintyEstimator.estimate(
        fold_scores,
        method=ci_method_value,  # type: ignore[arg-type]
        alpha=max(1e-6, min(float(ci_alpha), 0.5)),
        random_state=eval_seeds[0],
    )

    drift_auc = DriftChecker.adversarial_auc(
        cache.get("drift_train_x"),  # type: ignore[arg-type]
        cache.get("drift_test_x"),  # type: ignore[arg-type]
        enabled=bool(drift_check_enabled),
        random_state=eval_seeds[0],
        n_splits=max(2, min(cv_folds, 5)),
    )
    readiness_method_value = "mean_std" if str(readiness_method).lower() == "mean_std" else "ci_bound"
    readiness_score = SubmissionReadinessScorer.compute(
        direction=metric_direction,  # type: ignore[arg-type]
        mean_score=uncertainty.mean,
        std_score=uncertainty.std,
        ci_low=uncertainty.ci_low,
        ci_high=uncertainty.ci_high,
        method=readiness_method_value,  # type: ignore[arg-type]
        k=float(readiness_k),
        drift_auc=drift_auc,
        drift_enabled=bool(drift_check_enabled),
        drift_weight=float(drift_weight),
    )
    report = EvaluationReport(
        metric_name=evaluation.metric,
        direction=metric_direction,  # type: ignore[arg-type]
        split_strategy=str(cache.get("split_strategy") or "kfold"),  # type: ignore[arg-type]
        n_splits=int(cache.get("n_splits") or max(2, cv_folds)),
        seeds=list(eval_seeds),
        repeats=int(eval_repeats),
        per_fold_scores=[float(item) for item in fold_scores],
        mean=uncertainty.mean,
        std=uncertainty.std,
        ci_low=uncertainty.ci_low,
        ci_high=uncertainty.ci_high,
        drift_auc=drift_auc,
        readiness_score=readiness_score,
    )
    payload = report.to_dict()
    payload.update(
        {
            "run_id": run_id,
            "iteration": iteration,
            "metric_value": evaluation.value,
            "score_source": evaluation.score_source,
            "split_index_fingerprints": cache.get("split_index_fingerprints", []),
        }
    )
    return report, payload, cache


def build_eval_data_cache_fallback(*, split_strategy: str | None, cv_folds: int) -> dict[str, object]:
    normalized_split = plan_policy.normalize_split_strategy_name(split_strategy) or "kfold"
    return {
        "split_strategy": normalized_split,
        "n_splits": max(2, int(cv_folds)),
        "split_index_fingerprints": [],
        "drift_train_x": None,
        "drift_test_x": None,
    }


def extract_fold_scores_for_report(
    *,
    evaluation: EvaluationResult,
    evaluation_by_source: dict[str, EvaluationResult],
) -> list[float]:
    cv_eval = evaluation_by_source.get("cv")
    if cv_eval is not None and cv_eval.fold_scores:
        return [float(value) for value in cv_eval.fold_scores]
    if evaluation.fold_scores:
        return [float(value) for value in evaluation.fold_scores]
    if evaluation_by_source:
        return [float(item.value) for item in evaluation_by_source.values()]
    return [float(evaluation.value)]


def ensure_eval_data_cache(
    *,
    config: object,
    cv_folds: int,
    split_strategy: str | None,
    seed: int,
    eval_seeds: list[int],
    eval_repeats: int,
    score_source: str,
    eval_data_cache: dict[str, object] | None,
) -> dict[str, object]:
    if eval_data_cache is not None:
        return eval_data_cache
    fallback = build_eval_data_cache_fallback(split_strategy=split_strategy, cv_folds=cv_folds)
    if score_source == "holdout":
        return fallback
    try:
        data_dir = getattr(getattr(config, "paths"), "data_dir")
        data = load_competition_data(data_dir)
    except Exception:  # noqa: BLE001
        return fallback
    try:
        y = np.asarray(data.train[data.target_column])
        expanded_seeds = plan_policy.expanded_eval_seeds(
            base_seeds=eval_seeds,
            repeats=eval_repeats,
            default_seeds=DEFAULT_EVAL_SEEDS,
            default_repeats=DEFAULT_EVAL_REPEATS,
            repeat_seed_offset=EVAL_REPEAT_SEED_OFFSET,
        )
        split = SplitStrategyFactory.create(y=y, strategy=split_strategy, n_splits=cv_folds, seed=seed)
        fingerprints: list[dict[str, object]] = []
        for expanded_seed in expanded_seeds:
            split_for_seed = SplitStrategyFactory.create(
                y=y,
                strategy=split_strategy,
                n_splits=cv_folds,
                seed=expanded_seed,
            )
            fingerprints.extend(build_split_index_fingerprints(split=split_for_seed, y=y, seed=expanded_seed))
    except Exception:  # noqa: BLE001
        return fallback
    drift_cols_raw = list(data.feature_columns or [])
    drift_cols: list[str] = []
    seen_drift_cols: set[str] = set()
    for col in drift_cols_raw:
        if col in seen_drift_cols:
            continue
        seen_drift_cols.add(col)
        drift_cols.append(col)
    try:
        common_cols = set(data.train.columns).intersection(set(data.test.columns))
        drift_cols = [col for col in drift_cols if col in common_cols]
        drift_train_x = data.train.reindex(columns=drift_cols).copy() if drift_cols else None
        drift_test_x = data.test.reindex(columns=drift_cols).copy() if drift_cols else None
    except Exception:  # noqa: BLE001
        drift_train_x = None
        drift_test_x = None
    return {
        "split_strategy": split.name,
        "n_splits": split.n_splits,
        "split_index_fingerprints": fingerprints,
        "drift_train_x": drift_train_x,
        "drift_test_x": drift_test_x,
    }


def build_split_index_fingerprints(*, split: object, y: np.ndarray, seed: int) -> list[dict[str, object]]:
    split_strategy = split
    name = getattr(split_strategy, "name", "kfold")
    splitter = getattr(split_strategy, "splitter", None)
    if splitter is None:
        return []
    groups = None
    x_dummy = np.zeros((len(y), 1), dtype=np.float64)
    records: list[dict[str, object]] = []
    split_iter = iter_split_indices(name=name, splitter=splitter, x=x_dummy, y=y, groups=groups)
    for fold_idx, (train_idx, valid_idx) in enumerate(split_iter):
        train_arr = np.asarray(train_idx, dtype=np.int64)
        valid_arr = np.asarray(valid_idx, dtype=np.int64)
        records.append(
            {
                "seed": seed,
                "fold": fold_idx,
                "train_size": int(train_arr.size),
                "valid_size": int(valid_arr.size),
                "train_hash": hashlib.sha256(train_arr.tobytes()).hexdigest()[:16],
                "valid_hash": hashlib.sha256(valid_arr.tobytes()).hexdigest()[:16],
            }
        )
    return records


def iter_split_indices(*, name: str, splitter: object, x: np.ndarray, y: np.ndarray, groups: object):
    if name == "timeseries_split":
        yield from splitter.split(x)  # type: ignore[union-attr]
        return
    if name == "group_kfold" and groups is not None:
        yield from splitter.split(x, y, groups=groups)  # type: ignore[union-attr]
        return
    if name == "stratified_kfold":
        yield from splitter.split(x, y)  # type: ignore[union-attr]
        return
    yield from splitter.split(x, y)  # type: ignore[union-attr]


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
