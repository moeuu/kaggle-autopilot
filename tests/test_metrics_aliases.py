from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

from kagglebot.solver.metrics import compute_metric, infer_direction, metric_requires_proba


def test_auc_roc_alias_requires_probability() -> None:
    assert metric_requires_proba("AUC-ROC") is True


def test_auc_roc_alias_computes_auc() -> None:
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0.1, 0.8, 0.3, 0.7, 0.9])
    expected = roc_auc_score(y_true, y_pred)
    assert compute_metric("AUC-ROC", y_true, y_pred) == expected


def test_auc_roc_alias_direction() -> None:
    assert infer_direction("AUC-ROC") == "maximize"


def test_average_precision_alias_requires_probability() -> None:
    assert metric_requires_proba("PR-AUC") is True


def test_average_precision_alias_computes_score() -> None:
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0.1, 0.8, 0.3, 0.7, 0.9])
    expected = average_precision_score(y_true, y_pred)
    assert compute_metric("PR-AUC", y_true, y_pred) == expected


def test_f1_from_probability_predictions() -> None:
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0.1, 0.8, 0.3, 0.7, 0.9])
    expected = f1_score(y_true, (y_pred >= 0.5).astype(int))
    assert compute_metric("f1", y_true, y_pred) == expected


def test_mape_direction_is_minimize() -> None:
    assert infer_direction("MAPE") == "minimize"


def test_brier_score_alias_requires_probability() -> None:
    assert metric_requires_proba("Brier Score") is True


def test_brier_score_alias_computes_score() -> None:
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0.1, 0.8, 0.3, 0.7, 0.9])
    expected = brier_score_loss(y_true, y_pred)
    assert compute_metric("Brier Score", y_true, y_pred) == expected


def test_brier_score_direction_is_minimize() -> None:
    assert infer_direction("Brier Score") == "minimize"
