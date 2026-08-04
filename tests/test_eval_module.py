from __future__ import annotations

import numpy as np
import pytest

from kagglebot.eval import (
    MetricRegistry,
    RepeatedCVRunner,
    SplitStrategyFactory,
    SubmissionReadinessScorer,
    UncertaintyEstimator,
)


def test_metric_registry_basic_metrics() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    y_pred = np.array([0, 0, 1, 1])

    assert MetricRegistry.score("auc", y_true, y_prob) == pytest.approx(1.0)
    assert MetricRegistry.score("accuracy", y_true, y_pred) == pytest.approx(1.0)
    assert MetricRegistry.score("f1", y_true, y_pred) == pytest.approx(1.0)
    assert MetricRegistry.score("logloss", y_true, y_prob) < 0.3
    assert MetricRegistry.score("brier_score", y_true, y_prob) == pytest.approx(0.025)


def test_metric_registry_aurc_orders_risk_by_descending_confidence() -> None:
    risks = np.array([0.0, 1.0])

    assert MetricRegistry.direction("aurc") == "minimize"
    assert MetricRegistry.score("aurc", risks, np.array([1.0, 0.5])) == pytest.approx(0.125)
    assert MetricRegistry.score("aurc", risks, np.array([0.5, 1.0])) == pytest.approx(0.875)


def test_metric_registry_regression_and_correlations() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])

    assert MetricRegistry.score("rmse", y_true, y_pred) == pytest.approx(0.0)
    assert MetricRegistry.score("mae", y_true, y_pred) == pytest.approx(0.0)
    assert MetricRegistry.score("rmsle", y_true, y_pred) == pytest.approx(0.0)
    assert MetricRegistry.score("mape", y_true, y_pred) == pytest.approx(0.0)
    assert MetricRegistry.score("smape", y_true, y_pred) == pytest.approx(0.0)
    assert MetricRegistry.score("pearson", y_true, y_pred) == pytest.approx(1.0)
    assert MetricRegistry.score("spearman", y_true, y_pred) == pytest.approx(1.0)


def test_uncertainty_bootstrap_is_deterministic() -> None:
    scores = [0.71, 0.73, 0.74, 0.72, 0.75, 0.76]
    first = UncertaintyEstimator.estimate(
        scores,
        method="bootstrap",
        alpha=0.05,
        bootstrap_iterations=400,
        random_state=123,
    )
    second = UncertaintyEstimator.estimate(
        scores,
        method="bootstrap",
        alpha=0.05,
        bootstrap_iterations=400,
        random_state=123,
    )
    assert first.mean == pytest.approx(second.mean)
    assert first.std == pytest.approx(second.std)
    assert first.ci_low == pytest.approx(second.ci_low)
    assert first.ci_high == pytest.approx(second.ci_high)


def test_split_strategy_binary_defaults_to_stratified() -> None:
    y = np.array([0, 1] * 20)
    split = SplitStrategyFactory.create(y, strategy=None, n_splits=5, seed=42)
    assert split.name == "stratified_kfold"
    assert split.n_splits == 5


def test_readiness_score_maximize_and_minimize() -> None:
    max_score = SubmissionReadinessScorer.compute(
        direction="maximize",
        mean_score=0.80,
        std_score=0.05,
        method="mean_std",
        k=1.0,
    )
    min_score = SubmissionReadinessScorer.compute(
        direction="minimize",
        mean_score=0.80,
        std_score=0.05,
        method="mean_std",
        k=1.0,
    )
    assert max_score == pytest.approx(0.75)
    assert min_score == pytest.approx(0.85)

    ci_based = SubmissionReadinessScorer.compute(
        direction="maximize",
        mean_score=0.80,
        std_score=0.05,
        ci_low=0.77,
        ci_high=0.83,
        method="ci_bound",
    )
    assert ci_based == pytest.approx(0.77)


def test_repeated_cv_runner_collects_fold_scores() -> None:
    y = np.array([0, 1] * 12)
    runner = RepeatedCVRunner()

    def predict_fn(train_idx, valid_idx, repeat_idx, seed, fold_idx):  # noqa: ARG001
        return y[valid_idx]

    result = runner.run(
        y=y,
        predict_fn=predict_fn,
        metric_name="accuracy",
        n_splits=4,
        seeds=[7, 11],
        repeats=2,
    )

    assert result.direction == "maximize"
    assert result.split_strategy == "stratified_kfold"
    assert len(result.per_fold_scores) == 16
    assert all(score == pytest.approx(1.0) for score in result.per_fold_scores)
