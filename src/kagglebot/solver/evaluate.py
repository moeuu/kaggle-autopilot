from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from kagglebot.solver.metrics import Direction


@dataclass(frozen=True)
class ScoreSelection:
    source: str


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


def select_score_source(
    *,
    score_source: str,
    plan_score_source: str | None,
    data_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    id_col: str | None,
) -> ScoreSelection:
    """Return score source selection restricted to generalizable offline modes."""
    del plan_score_source, data_dir, train, test, target_col, id_col
    normalized = str(score_source or "").strip().lower()
    if normalized in {"auto", "test"}:
        raise ValueError("score-source auto/test is removed; use holdout or cv.")
    if normalized in {"holdout", "cv"}:
        return ScoreSelection(source=normalized)
    raise ValueError(f"Unknown score source: {score_source}. Allowed values: holdout, cv.")
