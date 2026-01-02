from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CompetitionData:
    train: pd.DataFrame
    test: pd.DataFrame
    sample: pd.DataFrame
    id_column: str
    target_column: str
    feature_columns: list[str]
    task: str
    prediction_kind: str


def find_competition_files(data_dir: Path) -> tuple[Path, Path, Path]:
    csvs = [p for p in data_dir.rglob("*.csv") if p.is_file()]
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {data_dir}.")

    def score_sample(path: Path) -> int:
        name = path.name.lower()
        if "sample_submission" in name:
            return 3
        if "sample" in name and "submission" in name:
            return 2
        if "submission" in name:
            return 1
        return 0

    sample_candidates = sorted(csvs, key=score_sample, reverse=True)
    sample_path = sample_candidates[0] if score_sample(sample_candidates[0]) > 0 else None

    train_path = None
    test_path = None
    for path in csvs:
        name = path.name.lower()
        if "train" in name and train_path is None:
            train_path = path
        if "test" in name and test_path is None:
            test_path = path

    if train_path is None or test_path is None:
        raise FileNotFoundError("Unable to locate train.csv or test.csv in competition data.")
    if sample_path is None:
        raise FileNotFoundError("Unable to locate sample_submission.csv in competition data.")

    return train_path, test_path, sample_path


def infer_target(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> tuple[str, str, list[str]]:
    id_col = sample.columns[0]
    sample_targets = list(sample.columns[1:])
    candidates = [c for c in train.columns if c not in test.columns and c in sample.columns]
    target_cols = candidates or sample_targets
    if len(target_cols) != 1:
        raise ValueError("This baseline only supports single-target competitions.")
    target_col = target_cols[0]
    if target_col not in train.columns:
        raise ValueError(f"Target column '{target_col}' not found in train.csv.")
    feature_cols = [c for c in train.columns if c not in target_cols]
    if id_col in feature_cols:
        feature_cols.remove(id_col)
    return id_col, target_col, feature_cols


def infer_task(y: pd.Series) -> str:
    if y.dtype == "object":
        return "classification"
    nunique = y.nunique(dropna=True)
    if nunique <= 20:
        return "classification"
    if nunique / max(len(y), 1) <= 0.05:
        return "classification"
    return "regression"


def load_competition_data(data_dir: Path, *, target_column_override: str | None = None) -> CompetitionData:
    train_path, test_path, sample_path = find_competition_files(data_dir)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)

    if target_column_override and target_column_override in train.columns:
        id_col = sample.columns[0]
        target_col = target_column_override
        feature_cols = [c for c in train.columns if c != target_col]
        if id_col in feature_cols:
            feature_cols.remove(id_col)
    else:
        id_col, target_col, feature_cols = infer_target(train, test, sample)
    task = infer_task(train[target_col])
    prediction_kind = "probability" if sample[target_col].dtype.kind in {"f", "c"} else "class"

    return CompetitionData(
        train=train,
        test=test,
        sample=sample,
        id_column=id_col,
        target_column=target_col,
        feature_columns=feature_cols,
        task=task,
        prediction_kind=prediction_kind,
    )


def write_submission(
    sample: pd.DataFrame,
    test: pd.DataFrame,
    preds,
    *,
    id_column: str,
    target_column: str,
    output_path: Path,
) -> Path:
    submission = sample.copy()
    if id_column in test.columns:
        if test[id_column].duplicated().any():
            raise ValueError(f"Duplicate ids detected in test column '{id_column}'.")
        pred_map = pd.Series(preds, index=test[id_column])
        submission[target_column] = submission[id_column].map(pred_map)
        if submission[target_column].isna().any():
            raise ValueError("Missing predictions after aligning by id column.")
    else:
        if len(preds) != len(submission):
            raise ValueError("Prediction length does not match sample_submission rows.")
        submission[target_column] = preds
    submission[target_column] = _coerce_prediction_dtype(
        sample[target_column],
        submission[target_column],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return output_path


def _coerce_prediction_dtype(sample_series: pd.Series, pred_series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(sample_series.dtype):
        if pd.api.types.is_bool_dtype(pred_series.dtype):
            return pred_series
        if pd.api.types.is_numeric_dtype(pred_series.dtype):
            values = pred_series.dropna().to_numpy()
            if values.size == 0:
                return pred_series
            binary_mask = np.isclose(values, 0.0) | np.isclose(values, 1.0)
            if binary_mask.all():
                return pred_series.astype(bool)
            return pred_series
        lowered = pred_series.astype(str).str.lower()
        if set(lowered.dropna().unique()).issubset({"true", "false"}):
            return lowered == "true"
    return pred_series
