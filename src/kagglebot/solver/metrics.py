from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score

Direction = Literal["minimize", "maximize"]


def infer_direction(metric: str, explicit: str | None = None) -> Direction:
    if explicit and explicit != "auto":
        return explicit  # type: ignore[return-value]

    metric_lower = metric.lower()
    if any(key in metric_lower for key in ["rmse", "rmsle", "mae", "mape", "logloss", "loss", "error"]):
        return "minimize"
    if any(key in metric_lower for key in ["auc", "accuracy", "f1", "precision", "recall"]):
        return "maximize"
    return "minimize"


def metric_requires_proba(metric: str) -> bool:
    metric_lower = metric.lower()
    return any(key in metric_lower for key in ["logloss", "auc"])


def compute_metric(metric: str, y_true, y_pred) -> float:
    metric_lower = metric.lower()
    if metric_lower in {"rmse"}:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if metric_lower in {"rmsle"}:
        y_true_clip = np.clip(np.asarray(y_true, dtype=float), 0, None)
        y_pred_clip = np.clip(np.asarray(y_pred, dtype=float), 0, None)
        return float(np.sqrt(mean_squared_error(np.log1p(y_true_clip), np.log1p(y_pred_clip))))
    if metric_lower in {"logloss", "log_loss"}:
        return float(log_loss(y_true, y_pred))
    if metric_lower in {"auc"}:
        return float(roc_auc_score(y_true, y_pred))
    if metric_lower in {"accuracy"}:
        return float(accuracy_score(y_true, y_pred))
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
