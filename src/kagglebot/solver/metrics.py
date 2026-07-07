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
    brier_score_loss,
    cohen_kappa_score,
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


def normalize_direction(value: object, *, default: Direction | None = None) -> Direction | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"minimize", "maximize"}:
        return normalized  # type: ignore[return-value]
    return default


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
        "brier": "brier_score",
        "brierloss": "brier_score",
        "brierscore": "brier_score",
        "brierscoreloss": "brier_score",
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
        "ndcg": "ndcg",
        "normalizeddiscountedcumulativegain": "ndcg",
        "rmse": "rmse",
        "mcrmse": "mcrmse",
        "meancolumnwisermse": "mcrmse",
        "columnwisermse": "mcrmse",
        "rmsle": "rmsle",
        "mae": "mae",
        "mape": "mape",
        "smape": "smape",
        "pinball": "pinball_loss",
        "pinballloss": "pinball_loss",
        "quantileloss": "pinball_loss",
        "intervalscore": "interval_score",
        "predictionintervalscore": "interval_score",
        "mse": "mse",
        "r2": "r2",
        "rsquared": "r2",
        "cindex": "concordance_index",
        "concordance": "concordance_index",
        "concordanceindex": "concordance_index",
        "pearson": "pearson",
        "pearsonr": "pearson",
        "spearman": "spearman",
        "spearmanr": "spearman",
        "qwk": "quadratic_weighted_kappa",
        "quadraticweightedkappa": "quadratic_weighted_kappa",
        "weightedkappa": "quadratic_weighted_kappa",
        "cohenkappa": "quadratic_weighted_kappa",
    }
    return aliases.get(metric_key, metric_lower)


def infer_direction(metric: str, explicit: str | None = None) -> Direction:
    normalized_explicit = normalize_direction(explicit)
    if normalized_explicit is not None:
        return normalized_explicit

    metric_name = canonical_metric(metric)
    if metric_name in {
        "rmse",
        "mcrmse",
        "rmsle",
        "mae",
        "mape",
        "smape",
        "pinball_loss",
        "interval_score",
        "mse",
        "logloss",
        "brier_score",
    }:
        return "minimize"
    if metric_name in {
        "auc",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "average_precision",
        "ndcg",
        "r2",
        "r_squared",
        "concordance_index",
        "pearson",
        "spearman",
        "quadratic_weighted_kappa",
    }:
        return "maximize"

    metric_lower = metric.lower()
    if any(key in metric_lower for key in ["loss", "error"]):
        return "minimize"
    if any(key in metric_lower for key in ["auc", "accuracy", "f1", "precision", "recall", "ap", "r2", "r_squared"]):
        return "maximize"
    return "minimize"


def metric_requires_proba(metric: str) -> bool:
    return canonical_metric(metric) in {"logloss", "auc", "average_precision", "brier_score"}


def compute_metric(metric: str, y_true, y_pred, sample_weight=None) -> float:
    metric_name = canonical_metric(metric)
    if metric_name.startswith("custom:"):
        return _compute_custom_metric(metric_name, y_true, y_pred, sample_weight=sample_weight)
    if metric_name == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight)))
    if metric_name == "mcrmse":
        return _mcrmse(y_true, y_pred, sample_weight=sample_weight)
    if metric_name == "rmsle":
        y_true_clip = np.clip(np.asarray(y_true, dtype=float), 0, None)
        y_pred_clip = np.clip(np.asarray(y_pred, dtype=float), 0, None)
        return float(
            np.sqrt(mean_squared_error(np.log1p(y_true_clip), np.log1p(y_pred_clip), sample_weight=sample_weight))
        )
    if metric_name == "mae":
        return float(mean_absolute_error(y_true, y_pred, sample_weight=sample_weight))
    if metric_name == "mape":
        return float(mean_absolute_percentage_error(y_true, y_pred, sample_weight=sample_weight))
    if metric_name == "smape":
        return _smape(y_true, y_pred, sample_weight=sample_weight)
    if metric_name == "pinball_loss":
        return _pinball_loss(y_true, y_pred, sample_weight=sample_weight)
    if metric_name == "interval_score":
        return _interval_score(y_true, y_pred, sample_weight=sample_weight)
    if metric_name == "mse":
        return float(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))
    if metric_name == "r2":
        return float(r2_score(y_true, y_pred, sample_weight=sample_weight))
    if metric_name == "pearson":
        return _weighted_pearson(y_true, y_pred, sample_weight=sample_weight)
    if metric_name == "spearman":
        return _weighted_spearman(y_true, y_pred, sample_weight=sample_weight)
    if metric_name == "ndcg":
        return _ndcg(y_true, y_pred, sample_weight=sample_weight)
    if metric_name == "concordance_index":
        return _concordance_index(y_true, y_pred, sample_weight=sample_weight)
    if metric_name == "quadratic_weighted_kappa":
        return float(
            cohen_kappa_score(
                _as_ordinal_labels(y_true),
                _as_ordinal_predictions(y_true, y_pred),
                weights="quadratic",
                sample_weight=sample_weight,
            )
        )
    if metric_name == "logloss":
        return float(log_loss(y_true, y_pred, sample_weight=sample_weight))
    if metric_name == "brier_score":
        y_pred_arr = np.asarray(y_pred, dtype=float)
        if y_pred_arr.ndim == 2:
            if y_pred_arr.shape[1] == 1:
                y_pred_arr = y_pred_arr[:, 0]
            else:
                y_pred_arr = y_pred_arr[:, -1]
        y_pred_arr = np.clip(y_pred_arr, 0.0, 1.0)
        return float(brier_score_loss(y_true, y_pred_arr, sample_weight=sample_weight))
    if metric_name == "auc":
        y_pred_arr = np.asarray(y_pred)
        if y_pred_arr.ndim == 2 and y_pred_arr.shape[1] > 2:
            return float(roc_auc_score(y_true, y_pred_arr, multi_class="ovr", sample_weight=sample_weight))
        return float(roc_auc_score(y_true, y_pred_arr, sample_weight=sample_weight))
    if metric_name == "average_precision":
        y_pred_arr = np.asarray(y_pred)
        if y_pred_arr.ndim == 2 and y_pred_arr.shape[1] > 1:
            y_pred_arr = y_pred_arr[:, -1]
        return float(average_precision_score(y_true, y_pred_arr, sample_weight=sample_weight))
    if metric_name == "accuracy":
        return float(accuracy_score(y_true, _as_label_predictions(y_pred), sample_weight=sample_weight))
    if metric_name == "f1":
        average = _classification_average(y_true)
        return float(f1_score(y_true, _as_label_predictions(y_pred), average=average, sample_weight=sample_weight))
    if metric_name == "precision":
        average = _classification_average(y_true)
        return float(
            precision_score(
                y_true,
                _as_label_predictions(y_pred),
                average=average,
                zero_division=0,
                sample_weight=sample_weight,
            )
        )
    if metric_name == "recall":
        average = _classification_average(y_true)
        return float(
            recall_score(
                y_true,
                _as_label_predictions(y_pred),
                average=average,
                zero_division=0,
                sample_weight=sample_weight,
            )
        )
    return float(np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight)))


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


def _as_ordinal_labels(values) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.floating):
        return np.rint(arr).astype(int)
    return arr


def _as_ordinal_predictions(y_true, y_pred) -> np.ndarray:
    arr = np.asarray(y_pred)
    if arr.ndim == 2:
        return np.asarray(arr.argmax(axis=1), dtype=int)
    if np.issubdtype(arr.dtype, np.number):
        labels = _as_ordinal_labels(y_true)
        numeric_labels = _to_numeric_array(labels)
        rounded = np.rint(arr.astype(float))
        if numeric_labels.size:
            rounded = np.clip(rounded, float(np.nanmin(numeric_labels)), float(np.nanmax(numeric_labels)))
        return rounded.astype(int)
    return arr


def _to_numeric_array(values) -> np.ndarray:
    arr = np.asarray(values)
    try:
        numeric = arr.astype(float)
    except (TypeError, ValueError):
        return np.array([], dtype=float)
    return numeric[np.isfinite(numeric)]


def _smape(y_true, y_pred, *, sample_weight=None) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true_arr) + np.abs(y_pred_arr)
    values = np.zeros_like(denom, dtype=float)
    np.divide(2.0 * np.abs(y_true_arr - y_pred_arr), denom, out=values, where=denom > 1e-15)
    return _weighted_mean(values, sample_weight=sample_weight)


def _mcrmse(y_true, y_pred, *, sample_weight=None) -> float:
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    if true_arr.ndim == 1:
        return float(np.sqrt(mean_squared_error(true_arr, pred_arr, sample_weight=sample_weight)))
    if pred_arr.ndim == 1:
        pred_arr = pred_arr.reshape(-1, 1)
    scores = [
        float(np.sqrt(mean_squared_error(true_arr[:, index], pred_arr[:, index], sample_weight=sample_weight)))
        for index in range(true_arr.shape[1])
    ]
    return float(np.mean(scores))


def _pinball_loss(y_true, y_pred, *, sample_weight=None) -> float:
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    if pred_arr.ndim == 1:
        error = true_arr - pred_arr
        values = np.maximum(0.5 * error, -0.5 * error)
        return _weighted_mean(values, sample_weight=sample_weight)
    quantiles = _default_quantiles(pred_arr.shape[1])
    values_by_quantile = []
    for index, quantile in enumerate(quantiles):
        error = true_arr - pred_arr[:, index]
        values_by_quantile.append(np.maximum(quantile * error, (quantile - 1.0) * error))
    return _weighted_mean(np.mean(np.vstack(values_by_quantile), axis=0), sample_weight=sample_weight)


def _default_quantiles(count: int) -> np.ndarray:
    if count == 3:
        return np.array([0.1, 0.5, 0.9], dtype=float)
    if count == 9:
        return np.arange(0.1, 1.0, 0.1, dtype=float)
    return np.linspace(1.0 / (count + 1), count / (count + 1), count, dtype=float)


def _interval_score(y_true, y_pred, *, sample_weight=None, alpha: float = 0.1) -> float:
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    if pred_arr.ndim == 1:
        return float(mean_absolute_error(true_arr, pred_arr, sample_weight=sample_weight))
    lower = pred_arr[:, 0]
    upper = pred_arr[:, -1]
    below = true_arr < lower
    above = true_arr > upper
    values = (upper - lower).astype(float)
    values = values + np.where(below, (2.0 / alpha) * (lower - true_arr), 0.0)
    values = values + np.where(above, (2.0 / alpha) * (true_arr - upper), 0.0)
    return _weighted_mean(values, sample_weight=sample_weight)


def _weighted_pearson(y_true, y_pred, *, sample_weight=None) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    weights = _normalized_weights(sample_weight, len(y_true_arr))
    if weights is None:
        if np.std(y_true_arr) <= 0.0 or np.std(y_pred_arr) <= 0.0:
            return 0.0
        return float(np.corrcoef(y_true_arr, y_pred_arr)[0, 1])
    mean_true = float(np.sum(weights * y_true_arr))
    mean_pred = float(np.sum(weights * y_pred_arr))
    centered_true = y_true_arr - mean_true
    centered_pred = y_pred_arr - mean_pred
    cov = float(np.sum(weights * centered_true * centered_pred))
    var_true = float(np.sum(weights * centered_true * centered_true))
    var_pred = float(np.sum(weights * centered_pred * centered_pred))
    denom = np.sqrt(var_true * var_pred)
    return 0.0 if denom <= 0.0 else float(cov / denom)


def _weighted_spearman(y_true, y_pred, *, sample_weight=None) -> float:
    true_ranks = _average_ranks(np.asarray(y_true, dtype=float))
    pred_ranks = _average_ranks(np.asarray(y_pred, dtype=float))
    return _weighted_pearson(true_ranks, pred_ranks, sample_weight=sample_weight)


def _ndcg(y_true, y_pred, *, sample_weight=None) -> float:
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    if true_arr.ndim > 1:
        true_arr = true_arr.reshape(-1)
    if pred_arr.ndim > 1:
        pred_arr = pred_arr.reshape(-1)
    weights = _normalized_weights(sample_weight, len(true_arr))
    if weights is not None:
        true_arr = true_arr * weights * len(true_arr)
    order = np.argsort(-pred_arr, kind="mergesort")
    ideal = np.argsort(-true_arr, kind="mergesort")
    dcg = _discounted_gain(true_arr[order])
    ideal_dcg = _discounted_gain(true_arr[ideal])
    return 0.0 if ideal_dcg <= 0.0 else float(dcg / ideal_dcg)


def _discounted_gain(relevance: np.ndarray) -> float:
    gains = np.power(2.0, relevance.astype(float)) - 1.0
    discounts = 1.0 / np.log2(np.arange(2, len(relevance) + 2, dtype=float))
    return float(np.sum(gains * discounts))


def _concordance_index(y_true, y_pred, *, sample_weight=None) -> float:
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    if pred_arr.ndim > 1:
        pred_arr = pred_arr.reshape(-1)
    weights = _normalized_weights(sample_weight, len(true_arr))
    concordant = 0.0
    comparable = 0.0
    for left in range(len(true_arr)):
        for right in range(left + 1, len(true_arr)):
            if true_arr[left] == true_arr[right]:
                continue
            pair_weight = 1.0 if weights is None else float(weights[left] * weights[right])
            comparable += pair_weight
            truth_order = np.sign(true_arr[left] - true_arr[right])
            pred_order = np.sign(pred_arr[left] - pred_arr[right])
            if pred_order == 0:
                concordant += 0.5 * pair_weight
            elif pred_order == truth_order:
                concordant += pair_weight
    return 0.0 if comparable <= 0.0 else float(concordant / comparable)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _weighted_mean(values: np.ndarray, *, sample_weight=None) -> float:
    weights = _normalized_weights(sample_weight, len(values))
    if weights is None:
        return float(np.mean(values))
    return float(np.sum(weights * values))


def _normalized_weights(sample_weight, length: int) -> np.ndarray | None:
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=float)
    if weights.shape[0] != length:
        return None
    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        return None
    return weights / total


def _classification_average(y_true) -> str:
    classes = np.unique(np.asarray(y_true))
    return "binary" if len(classes) <= 2 else "macro"


def _compute_custom_metric(metric_name: str, y_true, y_pred, *, sample_weight=None) -> float:
    spec = metric_name.split("custom:", 1)[1].strip()
    if not spec:
        raise ValueError("Custom metric spec is empty. Use custom:<module_or_path>:<function>.")
    func = _load_custom_metric_function(spec)
    try:
        value = func(y_true, y_pred, metric_name, sample_weight=sample_weight)
    except TypeError:
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
