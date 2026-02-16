from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from kagglebot.analyzer.types import CompetitionSchema
from kagglebot.solver.io import infer_submission_layout


@dataclass(frozen=True)
class SchemaFrames:
    train: pd.DataFrame
    test: pd.DataFrame
    sample: pd.DataFrame


def load_schema_frames(train_path: Path, test_path: Path, sample_path: Path) -> SchemaFrames:
    return SchemaFrames(
        train=pd.read_csv(train_path),
        test=pd.read_csv(test_path),
        sample=pd.read_csv(sample_path),
    )


def infer_schema(
    *,
    frames: SchemaFrames,
    train_path: Path,
    test_path: Path,
    sample_path: Path,
) -> CompetitionSchema:
    sample = frames.sample
    if sample.shape[1] < 1:
        raise ValueError("sample submission must contain at least one column.")

    id_column, target_columns, feature_columns = infer_submission_layout(
        train=frames.train,
        test=frames.test,
        sample=sample,
    )
    if not target_columns:
        raise ValueError("Unable to infer target columns from train/test/sample files.")

    train_columns = list(frames.train.columns)
    missing_targets = [col for col in target_columns if col not in train_columns]
    if missing_targets:
        raise ValueError(f"Target columns missing from train.csv: {missing_targets}")

    if not feature_columns:
        raise ValueError("No feature columns detected after removing target and id columns.")

    missing_features = [col for col in feature_columns if col not in frames.test.columns]
    if missing_features:
        raise ValueError(f"Feature columns missing from test.csv: {missing_features}")

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    datetime_columns: list[str] = []

    for col in feature_columns:
        series = frames.train[col]
        if _looks_datetime(series):
            datetime_columns.append(col)
            continue
        if (
            pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            categorical_columns.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(col)
        else:
            categorical_columns.append(col)

    return CompetitionSchema(
        train_path=train_path,
        test_path=test_path,
        sample_submission_path=sample_path,
        id_column=id_column,
        target_columns=target_columns,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
    )


def _looks_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    if series.empty:
        return False
    sample = series.dropna()
    if sample.empty:
        return False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce", utc=True)
    valid_ratio = parsed.notna().mean()
    return valid_ratio >= 0.8
