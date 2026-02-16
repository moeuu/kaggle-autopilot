from __future__ import annotations

import importlib
import importlib.util
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

Direction = Literal["minimize", "maximize"]


def canonical_metric(metric: str) -> str:
    metric_raw = metric.strip()
    if metric_raw.lower().startswith("custom:"):
        return metric_raw
    metric_lower = metric_raw.lower()
    metric_key = re.sub(r"[^a-z0-9]+", "", metric_lower)
    aliases = {
        "auc": "auc",
        "aucroc": "auc",
        "rocauc": "auc",
        "logloss": "logloss",
        "crossentropy": "logloss",
        "accuracy": "accuracy",
        "acc": "accuracy",
        "f1": "f1",
        "f1score": "f1",
        "precision": "precision",
        "recall": "recall",
        "prauc": "average_precision",
        "averageprecision": "average_precision",
        "ap": "average_precision",
        "map": "average_precision",
        "rmse": "rmse",
        "rmsle": "rmsle",
        "mae": "mae",
        "mape": "mape",
        "mse": "mse",
        "r2": "r2",
        "rsquared": "r2",
    }
    return aliases.get(metric_key, metric_lower)


def infer_direction(metric: str, explicit: str | None = None) -> Direction:
    if explicit and explicit != "auto":
        return explicit  # type: ignore[return-value]

    metric_name = canonical_metric(metric)
    if metric_name in {"rmse", "rmsle", "mae", "mape", "mse", "logloss"}:
        return "minimize"
    if metric_name in {"auc", "accuracy", "f1", "precision", "recall", "average_precision", "r2", "r_squared"}:
        return "maximize"

    metric_lower = metric.lower()
    if any(key in metric_lower for key in ["loss", "error"]):
        return "minimize"
    if any(key in metric_lower for key in ["auc", "accuracy", "f1", "precision", "recall", "ap", "r2", "r_squared"]):
        return "maximize"
    return "minimize"


def metric_requires_proba(metric: str) -> bool:
    return canonical_metric(metric) in {"logloss", "auc", "average_precision"}


def compute_metric(metric: str, y_true, y_pred) -> float:
    metric_name = canonical_metric(metric)
    if metric_name.startswith("custom:"):
        return _compute_custom_metric(metric_name, y_true, y_pred)
    if metric_name == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if metric_name == "rmsle":
        y_true_clip = np.clip(np.asarray(y_true, dtype=float), 0, None)
        y_pred_clip = np.clip(np.asarray(y_pred, dtype=float), 0, None)
        return float(np.sqrt(mean_squared_error(np.log1p(y_true_clip), np.log1p(y_pred_clip))))
    if metric_name == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if metric_name == "mape":
        return float(mean_absolute_percentage_error(y_true, y_pred))
    if metric_name == "mse":
        return float(mean_squared_error(y_true, y_pred))
    if metric_name == "r2":
        return float(r2_score(y_true, y_pred))
    if metric_name == "logloss":
        return float(log_loss(y_true, y_pred))
    if metric_name == "auc":
        y_pred_arr = np.asarray(y_pred)
        if y_pred_arr.ndim == 2 and y_pred_arr.shape[1] > 2:
            return float(roc_auc_score(y_true, y_pred_arr, multi_class="ovr"))
        return float(roc_auc_score(y_true, y_pred_arr))
    if metric_name == "average_precision":
        y_pred_arr = np.asarray(y_pred)
        if y_pred_arr.ndim == 2 and y_pred_arr.shape[1] > 1:
            y_pred_arr = y_pred_arr[:, -1]
        return float(average_precision_score(y_true, y_pred_arr))
    if metric_name == "accuracy":
        return float(accuracy_score(y_true, _as_label_predictions(y_pred)))
    if metric_name == "f1":
        average = _classification_average(y_true)
        return float(f1_score(y_true, _as_label_predictions(y_pred), average=average))
    if metric_name == "precision":
        average = _classification_average(y_true)
        return float(precision_score(y_true, _as_label_predictions(y_pred), average=average, zero_division=0))
    if metric_name == "recall":
        average = _classification_average(y_true)
        return float(recall_score(y_true, _as_label_predictions(y_pred), average=average, zero_division=0))
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _as_label_predictions(y_pred) -> np.ndarray:
    arr = np.asarray(y_pred)
    if arr.ndim == 2:
        if arr.shape[1] == 1:
            return (arr[:, 0] >= 0.5).astype(int)
        return np.asarray(arr.argmax(axis=1), dtype=int)
    if arr.ndim == 1 and np.issubdtype(arr.dtype, np.floating):
        if np.all((arr >= 0.0) & (arr <= 1.0)):
            return (arr >= 0.5).astype(int)
    return arr


def _classification_average(y_true) -> str:
    classes = np.unique(np.asarray(y_true))
    return "binary" if len(classes) <= 2 else "macro"


def _compute_custom_metric(metric_name: str, y_true, y_pred) -> float:
    spec = metric_name.split("custom:", 1)[1].strip()
    if not spec:
        raise ValueError("Custom metric spec is empty. Use custom:<module_or_path>:<function>.")
    func = _load_custom_metric_function(spec)
    try:
        value = func(y_true, y_pred, metric_name)
    except TypeError:
        value = func(y_true, y_pred)
    return float(value)


@lru_cache(maxsize=64)
def _load_custom_metric_function(spec: str):
    if ":" in spec:
        module_spec, func_name = spec.rsplit(":", 1)
    else:
        module_spec, func_name = spec, "compute_metric"
    module_spec = module_spec.strip()
    func_name = func_name.strip() or "compute_metric"

    module = _load_module_from_spec(module_spec)
    func = getattr(module, func_name, None)
    if func is None or not callable(func):
        raise ValueError(f"Custom metric function not found or not callable: {module_spec}:{func_name}")
    return func


def _load_module_from_spec(module_spec: str):
    path = Path(module_spec)
    if path.suffix == ".py" and path.exists():
        module_name = f"kagglebot_custom_metric_{abs(hash(str(path.resolve())))}"
        loader = importlib.util.spec_from_file_location(module_name, path)
        if loader is None or loader.loader is None:
            raise ValueError(f"Unable to load custom metric module: {module_spec}")
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)  # type: ignore[union-attr]
        return module
    return importlib.import_module(module_spec)
