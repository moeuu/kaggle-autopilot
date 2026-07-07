from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    mean_squared_error,
    roc_auc_score,
)

from kagglebot.solver.metrics import compute_metric, infer_direction, metric_requires_proba, normalize_direction


def test_auc_roc_alias_requires_probability() -> None:
    assert metric_requires_proba("AUC-ROC") is True


def test_auc_roc_alias_computes_auc() -> None:
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0.1, 0.8, 0.3, 0.7, 0.9])
    expected = roc_auc_score(y_true, y_pred)
    assert compute_metric("AUC-ROC", y_true, y_pred) == expected


def test_compute_metric_accepts_sample_weight() -> None:
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([0.0, 1.5, 4.0])
    sample_weight = np.array([1.0, 1.0, 4.0])

    expected = np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))

    assert compute_metric("rmse", y_true, y_pred, sample_weight=sample_weight) == expected


def test_quadratic_weighted_kappa_alias_computes_score() -> None:
    y_true = np.array([0, 1, 2, 3, 3])
    y_pred = np.array([0.2, 1.1, 1.8, 2.9, 2.0])

    expected = cohen_kappa_score(y_true, np.array([0, 1, 2, 3, 2]), weights="quadratic")

    assert compute_metric("quadratic_weighted_kappa", y_true, y_pred) == expected
    assert infer_direction("QWK") == "maximize"


def test_smape_alias_computes_score() -> None:
    y_true = np.array([100.0, 200.0, 0.0])
    y_pred = np.array([110.0, 180.0, 0.0])
    expected = np.mean(np.array([2 * 10 / 210, 2 * 20 / 380, 0.0]))

    assert compute_metric("SMAPE", y_true, y_pred) == expected
    assert infer_direction("SMAPE") == "minimize"


def test_correlation_metric_aliases_compute_scores() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 4.0, 6.0, 8.0])

    assert np.isclose(compute_metric("pearson", y_true, y_pred), 1.0)
    assert np.isclose(compute_metric("spearman", y_true, y_pred), 1.0)
    assert infer_direction("pearsonr") == "maximize"
    assert infer_direction("spearmanr") == "maximize"


def test_quantile_and_interval_metrics_compute_scores() -> None:
    y_true = np.array([0.0, 2.0])
    median_pred = np.array([1.0, 1.0])
    interval_true = np.array([5.0, 12.0, 20.0])
    interval_pred = np.array([[4.0, 6.0], [10.0, 11.0], [21.0, 25.0]])

    assert compute_metric("pinball_loss", y_true, median_pred) == 0.5
    assert compute_metric("interval_score", interval_true, interval_pred) == np.mean([2.0, 21.0, 24.0])
    assert infer_direction("pinball_loss") == "minimize"
    assert infer_direction("interval_score") == "minimize"


def test_ranking_survival_and_multioutput_metrics_compute_scores() -> None:
    relevance = np.array([3.0, 2.0, 0.0, 1.0])
    ranking_pred = np.array([0.9, 0.8, 0.1, 0.2])
    survival_true = np.array([1.0, 2.0, 3.0])
    survival_pred = np.array([0.1, 0.4, 0.2])
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.0, 3.0], [5.0, 4.0]])

    expected_mcrmse = np.mean([np.sqrt(2.0), np.sqrt(0.5)])

    assert compute_metric("ndcg", relevance, ranking_pred) == 1.0
    assert compute_metric("concordance_index", survival_true, survival_pred) == 2 / 3
    assert compute_metric("mcrmse", y_true, y_pred) == expected_mcrmse
    assert infer_direction("ndcg") == "maximize"
    assert infer_direction("concordance_index") == "maximize"
    assert infer_direction("mcrmse") == "minimize"


def test_auc_roc_alias_direction() -> None:
    assert infer_direction("AUC-ROC") == "maximize"


def test_normalize_direction_accepts_only_canonical_directions() -> None:
    assert normalize_direction(" MAXIMIZE ") == "maximize"
    assert normalize_direction("minimize") == "minimize"
    assert normalize_direction("auto", default="minimize") == "minimize"
    assert normalize_direction("bad") is None


def test_infer_direction_ignores_invalid_explicit_direction() -> None:
    assert infer_direction("AUC", explicit="bad") == "maximize"


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
