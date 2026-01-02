from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from kagglebot.solver.metrics import Direction


@dataclass(frozen=True)
class LabeledTest:
    frame: pd.DataFrame
    target: pd.Series


@dataclass(frozen=True)
class ScoreSelection:
    source: str
    labeled_test: LabeledTest | None


@dataclass(frozen=True)
class EvaluationResult:
    score_source: str
    metric: str
    direction: Direction
    value: float
    std: float | None
    train_score: float | None
    val_score: float | None
    fold_scores: list[float] | None


LABEL_FILENAMES = ("test_labels.csv", "labels.csv", "y_test.csv")


def select_score_source(
    *,
    score_source: str,
    plan_score_source: str | None,
    data_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    id_col: str,
) -> ScoreSelection:
    labeled = find_labeled_test(
        data_dir=data_dir,
        test=test,
        target_col=target_col,
        id_col=id_col,
    )

    if score_source == "test":
        if labeled is None:
            raise ValueError("score-source=test requested but labeled test data not found.")
        return ScoreSelection(source="test", labeled_test=labeled)

    if score_source == "auto":
        if labeled is not None:
            return ScoreSelection(source="test", labeled_test=labeled)
        if plan_score_source == "cv":
            return ScoreSelection(source="cv", labeled_test=None)
        return ScoreSelection(source="holdout", labeled_test=None)

    if score_source in {"holdout", "cv"}:
        return ScoreSelection(source=score_source, labeled_test=None)

    raise ValueError(f"Unknown score source: {score_source}")


def find_labeled_test(
    *,
    data_dir: Path,
    test: pd.DataFrame,
    target_col: str,
    id_col: str,
) -> LabeledTest | None:
    if target_col in test.columns:
        return LabeledTest(frame=test, target=test[target_col])

    label_path = _find_label_file(data_dir)
    if label_path is None:
        return None

    labels = pd.read_csv(label_path)
    if target_col in labels.columns and id_col in labels.columns:
        merged = test.merge(labels[[id_col, target_col]], on=id_col, how="inner")
        if merged.empty:
            return None
        return LabeledTest(frame=merged, target=merged[target_col])

    if target_col in labels.columns and len(labels) == len(test):
        labeled = test.copy()
        labeled[target_col] = labels[target_col].values
        return LabeledTest(frame=labeled, target=labeled[target_col])

    if len(labels.columns) == 1 and len(labels) == len(test):
        labeled = test.copy()
        labeled[target_col] = labels.iloc[:, 0].values
        return LabeledTest(frame=labeled, target=labeled[target_col])

    return None


def _find_label_file(data_dir: Path) -> Path | None:
    for name in LABEL_FILENAMES:
        candidate = data_dir / name
        if candidate.exists():
            return candidate
    for path in data_dir.rglob("*.csv"):
        if path.name.lower() in LABEL_FILENAMES:
            return path
    return None
