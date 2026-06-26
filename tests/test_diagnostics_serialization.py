from __future__ import annotations

import pytest

from kagglebot.diagnostics import (
    build_diagnostics,
    diagnostics_path_for_iteration,
    load_iteration_diagnostics_text,
    pipeline_config_hash,
    write_iteration_diagnostics,
)
from kagglebot.solver.evaluate import EvaluationResult


def test_iteration_diagnostics_helpers_round_trip_text(tmp_path) -> None:
    iter_dir = tmp_path / "iter-1"
    iter_dir.mkdir()

    path = write_iteration_diagnostics(iter_dir=iter_dir, diagnostics="# Diagnostics\n")

    assert path == diagnostics_path_for_iteration(iter_dir)
    assert path.name == "diagnostics.md"
    assert load_iteration_diagnostics_text(iter_dir) == "# Diagnostics\n"
    assert load_iteration_diagnostics_text(tmp_path / "missing") == ""


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

    diagnostics = build_diagnostics(
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

    diagnostics = build_diagnostics(
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


def test_pipeline_config_hash_ignores_runtime_metadata() -> None:
    base = pipeline_config_hash(
        model_summary={"family": "lgbm", "timing": {"seconds": 10}, "evaluation_by_source": {"cv": 0.8}},
        metric="auc",
        accelerator="gpu",
    )
    changed_runtime = pipeline_config_hash(
        model_summary={"family": "lgbm", "timing": {"seconds": 20}, "duration": 99},
        metric="auc",
        accelerator="gpu",
    )
    changed_model = pipeline_config_hash(
        model_summary={"family": "catboost"},
        metric="auc",
        accelerator="gpu",
    )

    assert base == changed_runtime
    assert base != changed_model
