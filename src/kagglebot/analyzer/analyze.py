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

    if len(schema.target_columns) != 1:
        raise UnsupportedCompetitionError("Multi-target competitions are not supported yet.")

    target_col = schema.target_columns[0]
    target_series = frames.train[target_col]

    task = _infer_task(target_series)
    metric, metric_direction = _default_metric(task)
    prediction_kind = _infer_prediction_kind(task, frames.sample[target_col])

    strategy = build_strategy(
        task,
        time_budget_minutes=time_budget_minutes,
        cv_folds=cv_folds,
        models=models,
        use_stacking=use_stacking,
    )

    assumptions = [
        "tabular competition (train/test/sample_submission CSVs)",
        "single target column inferred from sample_submission",
    ]
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
    raw = paths.data_raw
    sample = raw / "sample_submission.csv"
    train = raw / "train.csv"
    test = raw / "test.csv"
    missing = [path for path in (sample, train, test) if not path.exists()]
    if missing:
        raise UnsupportedCompetitionError(
            "Missing required CSVs in data directory: " + ", ".join(str(p) for p in missing)
        )
    return train, test, sample


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


def _default_metric(task: str) -> tuple[str, str]:
    if task == "classification":
        return "accuracy", "maximize"
    return "rmse", "minimize"


def _infer_prediction_kind(task: str, sample_target: pd.Series) -> str:
    if task != "classification":
        return "continuous"
    if pd.api.types.is_float_dtype(sample_target):
        return "probability"
    return "class"
