"""Tests for score-source selection logic."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kagglebot.solver.evaluate import select_score_source


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.write_text(df.to_csv(index=False), encoding="utf-8")


def test_score_source_auto_prefers_labeled_test(tmp_path: Path) -> None:
    train = pd.DataFrame({"id": [1, 2], "feat": [0.1, 0.2], "target": [0, 1]})
    test = pd.DataFrame({"id": [3, 4], "feat": [0.3, 0.4], "target": [0, 1]})
    selection = select_score_source(
        score_source="auto",
        plan_score_source=None,
        data_dir=tmp_path,
        train=train,
        test=test,
        target_col="target",
        id_col="id",
    )
    assert selection.source == "test"
    assert selection.labeled_test is not None


def test_score_source_auto_uses_label_file(tmp_path: Path) -> None:
    train = pd.DataFrame({"id": [1, 2], "feat": [0.1, 0.2], "target": [0, 1]})
    test = pd.DataFrame({"id": [3, 4], "feat": [0.3, 0.4]})
    labels = pd.DataFrame({"id": [3, 4], "target": [1, 0]})
    _write_csv(tmp_path / "labels.csv", labels)
    selection = select_score_source(
        score_source="auto",
        plan_score_source=None,
        data_dir=tmp_path,
        train=train,
        test=test,
        target_col="target",
        id_col="id",
    )
    assert selection.source == "test"
    assert selection.labeled_test is not None


def test_score_source_auto_falls_back_to_holdout(tmp_path: Path) -> None:
    train = pd.DataFrame({"id": [1, 2], "feat": [0.1, 0.2], "target": [0, 1]})
    test = pd.DataFrame({"id": [3, 4], "feat": [0.3, 0.4]})
    selection = select_score_source(
        score_source="auto",
        plan_score_source=None,
        data_dir=tmp_path,
        train=train,
        test=test,
        target_col="target",
        id_col="id",
    )
    assert selection.source == "holdout"
    assert selection.labeled_test is None


def test_score_source_test_requires_labels(tmp_path: Path) -> None:
    train = pd.DataFrame({"id": [1, 2], "feat": [0.1, 0.2], "target": [0, 1]})
    test = pd.DataFrame({"id": [3, 4], "feat": [0.3, 0.4]})
    with pytest.raises(ValueError, match="labeled test data"):
        select_score_source(
            score_source="test",
            plan_score_source=None,
            data_dir=tmp_path,
            train=train,
            test=test,
            target_col="target",
            id_col="id",
        )
