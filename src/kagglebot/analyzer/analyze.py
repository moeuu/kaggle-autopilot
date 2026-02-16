from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from kagglebot.analyzer.schema import infer_schema, load_schema_frames
from kagglebot.analyzer.strategy import build_strategy
from kagglebot.analyzer.types import CompetitionMetadata
from kagglebot.paths import CompetitionPaths
from kagglebot.solver.io import find_competition_files


class UnsupportedCompetitionError(RuntimeError):
    pass


@dataclass(frozen=True)
class AnalysisResult:
    metadata: CompetitionMetadata
    analysis_path: Path


def analyze_competition(
    *,
    slug: str,
    paths: CompetitionPaths,
    time_budget_minutes: int,
    cv_folds: int,
    models: list[str] | None,
    use_stacking: bool,
) -> AnalysisResult:
    train_path, test_path, sample_path = _find_required(paths)
    frames = load_schema_frames(train_path, test_path, sample_path)
    schema = infer_schema(
        frames=frames,
        train_path=train_path,
        test_path=test_path,
        sample_path=sample_path,
    )

    target_col = schema.target_columns[0]
    target_series = frames.train[target_col]

    task = _infer_task(target_series)
    prediction_kind = _infer_prediction_kind(task, frames.sample[target_col])
    metric, metric_direction = _default_metric(task, prediction_kind=prediction_kind)

    strategy = build_strategy(
        task,
        time_budget_minutes=time_budget_minutes,
        cv_folds=cv_folds,
        models=models,
        use_stacking=use_stacking,
    )

    assumptions = ["tabular competition inferred from local train/test/sample files"]
    if len(schema.target_columns) == 1:
        assumptions.append("single target column inferred from sample submission")
    else:
        assumptions.append(
            f"multi-target submission detected ({len(schema.target_columns)} targets); "
            f"primary target for baseline is '{target_col}'"
        )
    if schema.id_column is None:
        assumptions.append("id column not clearly detected; submission alignment defaults to row order")
    if prediction_kind == "probability":
        assumptions.append("probability submission inferred from sample_submission target dtype")

    metadata = CompetitionMetadata(
        slug=slug,
        competition_type="tabular",
        task=task,
        metric=metric,
        metric_direction=metric_direction,
        prediction_kind=prediction_kind,
        schema=schema,
        strategy=strategy,
        assumptions=assumptions,
    )

    analysis_path = paths.analysis_path
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    payload = metadata.to_dict()
    payload["generated_at"] = datetime.now(UTC).isoformat()
    analysis_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return AnalysisResult(metadata=metadata, analysis_path=analysis_path)


def _find_required(paths: CompetitionPaths) -> tuple[Path, Path, Path]:
    try:
        return find_competition_files(paths.data_raw)
    except FileNotFoundError as exc:
        raise UnsupportedCompetitionError(str(exc)) from exc


def _infer_task(target: pd.Series) -> str:
    if pd.api.types.is_object_dtype(target) or isinstance(target.dtype, pd.CategoricalDtype):
        return "classification"
    if not pd.api.types.is_numeric_dtype(target):
        return "classification"
    nunique = target.nunique(dropna=True)
    if nunique <= 20:
        return "classification"
    if nunique / max(len(target), 1) <= 0.05:
        return "classification"
    return "regression"


def _default_metric(task: str, *, prediction_kind: str) -> tuple[str, str]:
    if task == "classification":
        if prediction_kind == "probability":
            return "logloss", "minimize"
        return "accuracy", "maximize"
    return "rmse", "minimize"


def _infer_prediction_kind(task: str, sample_target: pd.Series) -> str:
    if task != "classification":
        return "continuous"
    if pd.api.types.is_float_dtype(sample_target):
        return "probability"
    return "class"
