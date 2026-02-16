from __future__ import annotations

import pytest

from kagglebot.autopilot import _build_diagnostics
from kagglebot.solver.evaluate import EvaluationResult


def test_build_diagnostics_handles_unserializable_objects() -> None:
    sklearn_compose = pytest.importorskip("sklearn.compose")
    column_transformer_cls = sklearn_compose.ColumnTransformer

    evaluation = EvaluationResult(
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        value=0.123,
        std=None,
        train_score=None,
        val_score=None,
        fold_scores=None,
    )
    model_summary = {"preprocessing": column_transformer_cls(transformers=[])}

    diagnostics = _build_diagnostics(
        evaluation=evaluation,
        model_summary=model_summary,
        best_score=None,
        target_score=0.1,
        dataset_profile={},
        top1_score=None,
        top1_tier=False,
        diff_summary="",
    )

    assert "Pipeline summary:" in diagnostics
    assert "ColumnTransformer" in diagnostics


def test_build_diagnostics_reports_loop_decision_signal() -> None:
    evaluation = EvaluationResult(
        score_source="cv",
        metric="auc",
        direction="maximize",
        value=0.941,
        std=0.01,
        train_score=None,
        val_score=None,
        fold_scores=None,
    )

    diagnostics = _build_diagnostics(
        evaluation=evaluation,
        model_summary={},
        best_score=0.95,
        target_score=0.96,
        dataset_profile={},
        top1_score=0.97,
        top1_tier=False,
        diff_summary="",
        loop_decision_score=0.95323,
        loop_decision_source="submission",
    )

    assert "Loop decision: source=submission score=0.953230" in diagnostics
    assert "Score vs target: 0.953230 vs 0.960000" in diagnostics
