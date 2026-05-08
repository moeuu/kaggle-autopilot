from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from catboost import CatBoostClassifier
except ImportError as exc:  # pragma: no cover - dependency is expected in kernel runtimes
    raise RuntimeError("catboost is required for shared tabular ensemble helpers.") from exc

try:
    from lightgbm import LGBMClassifier
except ImportError as exc:  # pragma: no cover - dependency is expected in kernel runtimes
    raise RuntimeError("lightgbm is required for shared tabular ensemble helpers.") from exc

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - dependency is expected in kernel runtimes
    raise RuntimeError("xgboost is required for shared tabular ensemble helpers.") from exc


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


FAST_DEV = env_flag("KAGGLEBOT_FAST_DEV", False)
OUTER_FOLDS = 2 if FAST_DEV else 5
DEFAULT_PSEUDO_THRESHOLD = float(os.getenv("KAGGLEBOT_PSEUDO_THRESHOLD", "0.995"))
PREFER_CUDA = not env_flag("KAGGLEBOT_DISABLE_CUDA", False)
DEFAULT_XGB_N_ESTIMATORS = int(os.getenv("KAGGLEBOT_XGB_ESTIMATORS", "50000"))
CPU_FALLBACK_N_ESTIMATORS = int(os.getenv("KAGGLEBOT_XGB_CPU_FALLBACK_ESTIMATORS", "300" if FAST_DEV else "8000"))
FAST_DEV_N_ESTIMATORS = int(os.getenv("KAGGLEBOT_FAST_DEV_XGB_ESTIMATORS", "150"))
DEFAULT_CB_ITERATIONS = int(os.getenv("KAGGLEBOT_CB_ITERATIONS", "8000"))
DEFAULT_LGBM_ESTIMATORS = int(os.getenv("KAGGLEBOT_LGBM_ESTIMATORS", "12000"))


@dataclass
class PipelineResult:
    name: str
    oof_preds: np.ndarray
    test_preds: np.ndarray
    cv_score: float
    fold_scores: list[dict[str, Any]]
    feature_manifest: dict[str, Any]
    metadata: dict[str, Any]
    test_predictions_by_fold: dict[str, np.ndarray]
    oof_predictions_by_fold: dict[str, np.ndarray]
    valid_indices_by_fold: dict[str, np.ndarray]


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    model_family: str
    model_seeds: list[int]
    params_override: dict[str, Any]
    enable_pseudo: bool = False
    pseudo_threshold: float = DEFAULT_PSEUDO_THRESHOLD


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def clip_predictions(preds: np.ndarray) -> np.ndarray:
    arr = np.asarray(preds, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(arr, 1e-6, 1 - 1e-6)


def build_prediction_range(preds: np.ndarray) -> list[float]:
    clipped = clip_predictions(preds)
    return [float(clipped.min()), float(clipped.max())]


def resolve_component_models(result: PipelineResult) -> list[str]:
    components = result.metadata.get("blend_components")
    if isinstance(components, list):
        values = [str(item).strip() for item in components if str(item).strip()]
        if values:
            return values
    return [result.name]


def build_prediction_correlation_summary(results: list[PipelineResult]) -> dict[str, float | None]:
    single_model_results = [result for result in results if result.metadata.get("kind", "single") == "single"]
    corr_values: list[float] = []
    for idx, first in enumerate(single_model_results[:-1]):
        first_preds = np.asarray(first.oof_preds, dtype=np.float64)
        if first_preds.size <= 1:
            continue
        for second in single_model_results[idx + 1 :]:
            second_preds = np.asarray(second.oof_preds, dtype=np.float64)
            if second_preds.shape != first_preds.shape:
                continue
            corr = np.corrcoef(first_preds, second_preds)[0, 1]
            if np.isfinite(corr):
                corr_values.append(float(abs(corr)))
    if not corr_values:
        return {
            "pair_count": 0,
            "mean_abs_corr": None,
            "max_abs_corr": None,
            "min_abs_corr": None,
        }
    return {
        "pair_count": len(corr_values),
        "mean_abs_corr": float(np.mean(corr_values)),
        "max_abs_corr": float(np.max(corr_values)),
        "min_abs_corr": float(np.min(corr_values)),
    }


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


def mirror_json(filename: str, payload: Any, output_dirs: list[Path]) -> None:
    for output_dir in output_dirs:
        write_json(output_dir / filename, payload)


def mirror_df(filename: str, df: pd.DataFrame, output_dirs: list[Path]) -> None:
    for output_dir in output_dirs:
        df.to_csv(output_dir / filename, index=False)


def mirror_npy(filename: str, arr: np.ndarray, output_dirs: list[Path]) -> None:
    for output_dir in output_dirs:
        np.save(output_dir / filename, arr)


def train_xgb_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    model_seed: int,
    params_override: dict[str, Any],
    sample_weight: np.ndarray | None = None,
) -> tuple[XGBClassifier, dict[str, Any]]:
    params = {
        "n_estimators": FAST_DEV_N_ESTIMATORS if FAST_DEV else DEFAULT_XGB_N_ESTIMATORS,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "gamma": 0.05,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "enable_categorical": True,
        "tree_method": "hist",
        "device": "cuda" if PREFER_CUDA else "cpu",
        "early_stopping_rounds": 50 if FAST_DEV else 500,
        "random_state": model_seed,
        "verbosity": 0,
    }
    params.update(params_override)
    params["random_state"] = model_seed
    if FAST_DEV:
        params["early_stopping_rounds"] = min(int(params.get("early_stopping_rounds", 50)), 50)
    try:
        model = XGBClassifier(**params)
        fit_kwargs: dict[str, Any] = {
            "eval_set": [(x_valid, y_valid)],
            "verbose": False,
        }
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(x_train, y_train, **fit_kwargs)
        return model, {
            "device": params["device"],
            "best_iteration": getattr(model, "best_iteration", None),
            "params": dict(params),
        }
    except Exception as exc:
        if params["device"] == "cpu":
            raise
        params["device"] = "cpu"
        params["n_estimators"] = min(int(params["n_estimators"]), CPU_FALLBACK_N_ESTIMATORS)
        model = XGBClassifier(**params)
        fit_kwargs = {
            "eval_set": [(x_valid, y_valid)],
            "verbose": False,
        }
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(x_train, y_train, **fit_kwargs)
        return model, {
            "device": "cpu",
            "best_iteration": getattr(model, "best_iteration", None),
            "fallback_reason": str(exc),
            "params": dict(params),
        }


def train_catboost_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    model_seed: int,
    params_override: dict[str, Any],
    cat_features: list[str],
    sample_weight: np.ndarray | None = None,
) -> tuple[CatBoostClassifier, dict[str, Any]]:
    params = {
        "iterations": 150 if FAST_DEV else DEFAULT_CB_ITERATIONS,
        "learning_rate": 0.01 if not FAST_DEV else 0.05,
        "depth": 8,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "task_type": "GPU" if PREFER_CUDA else "CPU",
        "random_seed": model_seed,
        "verbose": False,
        "allow_writing_files": False,
        "od_type": "Iter",
        "od_wait": 100 if FAST_DEV else 300,
    }
    params.update(params_override)
    params["random_seed"] = model_seed
    try:
        model = CatBoostClassifier(**params)
        fit_kwargs: dict[str, Any] = {
            "eval_set": (x_valid, y_valid),
            "cat_features": cat_features,
            "verbose": False,
        }
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(x_train, y_train, **fit_kwargs)
        return model, {
            "device": str(params["task_type"]).lower(),
            "best_iteration": getattr(model, "get_best_iteration", lambda: None)(),
            "params": dict(params),
        }
    except Exception as exc:
        if str(params["task_type"]).upper() == "CPU":
            raise
        print(f"[kernel] CatBoost GPU failed; retrying on CPU: {type(exc).__name__}: {exc}", flush=True)
        params["task_type"] = "CPU"
        model = CatBoostClassifier(**params)
        fit_kwargs = {
            "eval_set": (x_valid, y_valid),
            "cat_features": cat_features,
            "verbose": False,
        }
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(x_train, y_train, **fit_kwargs)
        return model, {
            "device": "cpu",
            "best_iteration": getattr(model, "get_best_iteration", lambda: None)(),
            "fallback_reason": str(exc),
            "params": dict(params),
        }


def train_lgbm_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    model_seed: int,
    params_override: dict[str, Any],
    sample_weight: np.ndarray | None = None,
) -> tuple[LGBMClassifier, dict[str, Any]]:
    params = {
        "n_estimators": 200 if FAST_DEV else DEFAULT_LGBM_ESTIMATORS,
        "learning_rate": 0.02,
        "num_leaves": 63,
        "min_child_samples": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_lambda": 1.0,
        "objective": "binary",
        "metric": "auc",
        "random_state": model_seed,
        "verbosity": -1,
    }
    params.update(params_override)
    params["random_state"] = model_seed
    model = LGBMClassifier(**params)
    fit_kwargs: dict[str, Any] = {
        "eval_set": [(x_valid, y_valid)],
        "eval_metric": "auc",
        "callbacks": [],
    }
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(x_train, y_train, **fit_kwargs)
    return model, {
        "device": "cpu",
        "best_iteration": getattr(model, "best_iteration_", None),
        "params": dict(params),
    }


def maybe_apply_pseudo_labels(
    model: XGBClassifier,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    x_test: pd.DataFrame,
    model_seed: int,
    threshold: float,
    enabled: bool,
    params_override: dict[str, Any],
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, float]]:
    base_valid = clip_predictions(model.predict_proba(x_valid)[:, 1])
    base_test = clip_predictions(model.predict_proba(x_test)[:, 1])
    base_auc = safe_auc(y_valid, base_valid)
    if not enabled:
        return (
            base_valid,
            base_test,
            {
                "status": "skipped_disabled",
                "threshold": threshold,
                "candidate_count": 0,
                "base_auc": base_auc,
                "pl_auc": base_auc,
            },
            {},
        )
    mask = (base_test > threshold) | (base_test < (1.0 - threshold))
    pl_log: dict[str, Any] = {
        "status": "rejected",
        "threshold": threshold,
        "candidate_count": int(mask.sum()),
        "base_auc": base_auc,
        "pl_auc": base_auc,
    }
    if mask.sum() == 0:
        pl_log["status"] = "skipped_no_candidates"
        return base_valid, base_test, pl_log, {}

    x_train_pl = pd.concat([x_train, x_test.loc[mask].reset_index(drop=True)], axis=0, ignore_index=True)
    y_train_pl = np.concatenate([y_train, (base_test[mask] > 0.5).astype(np.int8)])
    sample_weight_pl = None
    if sample_weight is not None:
        pseudo_weight = np.ones(int(mask.sum()), dtype=np.float32)
        sample_weight_pl = np.concatenate([sample_weight, pseudo_weight])
    pl_model, pl_meta = train_xgb_model(
        x_train_pl,
        y_train_pl,
        x_valid,
        y_valid,
        model_seed,
        params_override=params_override,
        sample_weight=sample_weight_pl,
    )
    pl_valid = clip_predictions(pl_model.predict_proba(x_valid)[:, 1])
    pl_test = clip_predictions(pl_model.predict_proba(x_test)[:, 1])
    pl_auc = safe_auc(y_valid, pl_valid)
    pl_log["pl_auc"] = pl_auc
    if pl_auc > base_auc:
        pl_log["status"] = "accepted"
        return (
            pl_valid,
            pl_test,
            pl_log,
            {"pseudo_model_device": pl_meta.get("device"), "pseudo_best_iteration": pl_meta.get("best_iteration")},
        )
    return (
        base_valid,
        base_test,
        pl_log,
        {"pseudo_model_device": pl_meta.get("device"), "pseudo_best_iteration": pl_meta.get("best_iteration")},
    )


def validate_submission(
    sample_submission: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str,
    target_col: str,
    preds: np.ndarray,
) -> pd.DataFrame:
    submission = sample_submission.copy()
    preds = clip_predictions(preds)
    assert list(submission.columns) == [id_col, target_col], (
        "Sample submission columns differ from expected id,target format."
    )
    assert len(submission) == len(test_df), "Submission row count must match test row count."
    test_ids = test_df[id_col].reset_index(drop=True)
    sample_ids = submission[id_col].reset_index(drop=True)
    assert sample_ids.equals(test_ids), "Sample submission ids must align exactly with test ids."
    submission[target_col] = preds
    assert list(submission.columns) == list(sample_submission.columns), "Submission columns changed unexpectedly."
    return submission


def write_submission_manifest(
    *,
    output_dirs: list[Path],
    final_result: PipelineResult,
    summary: dict[str, Any],
) -> None:
    for output_dir in output_dirs:
        submission_path = output_dir / "submission.csv"
        submission_sha = None
        if submission_path.exists():
            submission_sha = sha256(submission_path.read_bytes()).hexdigest()
        payload = {
            "artifact_class": "tabular",
            "submission_path": "submission.csv",
            "staging_dir": None,
            "members": [],
            "selected_pipeline": final_result.name,
            "selected_kind": final_result.metadata.get("kind", "single"),
            "component_models": resolve_component_models(final_result),
            "metrics_summary_path": "metrics_summary.json",
            "cv_results_path": "cv_results.json",
            "sha256": submission_sha,
            "prediction_range": summary.get("prediction_range"),
        }
        write_json(output_dir / "submission_manifest.json", payload)
