"""Tests for score-source selection logic."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kagglebot.solver.evaluate import select_score_source


def test_score_source_cv_selected(tmp_path: Path) -> None:
    train = pd.DataFrame({"id": [1, 2], "feat": [0.1, 0.2], "target": [0, 1]})
    test = pd.DataFrame({"id": [3, 4], "feat": [0.3, 0.4], "target": [0, 1]})
    selection = select_score_source(
        score_source="cv",
        plan_score_source=None,
        data_dir=tmp_path,
        train=train,
        test=test,
        target_col="target",
        id_col="id",
    )
    assert selection.source == "cv"


def test_score_source_holdout_selected(tmp_path: Path) -> None:
    train = pd.DataFrame({"id": [1, 2], "feat": [0.1, 0.2], "target": [0, 1]})
    test = pd.DataFrame({"id": [3, 4], "feat": [0.3, 0.4]})
    selection = select_score_source(
        score_source="holdout",
        plan_score_source=None,
        data_dir=tmp_path,
        train=train,
        test=test,
        target_col="target",
        id_col="id",
    )
    assert selection.source == "holdout"


def test_score_source_auto_is_removed(tmp_path: Path) -> None:
    train = pd.DataFrame({"id": [1, 2], "feat": [0.1, 0.2], "target": [0, 1]})
    test = pd.DataFrame({"id": [3, 4], "feat": [0.3, 0.4]})
    with pytest.raises(ValueError, match="removed"):
        select_score_source(
            score_source="auto",
            plan_score_source=None,
            data_dir=tmp_path,
            train=train,
            test=test,
            target_col="target",
            id_col="id",
        )


def test_score_source_test_is_disabled_for_integrity(tmp_path: Path) -> None:
    train = pd.DataFrame({"id": [1, 2], "feat": [0.1, 0.2], "target": [0, 1]})
    test = pd.DataFrame({"id": [3, 4], "feat": [0.3, 0.4]})
    with pytest.raises(ValueError, match="removed"):
        select_score_source(
            score_source="test",
            plan_score_source=None,
            data_dir=tmp_path,
            train=train,
            test=test,
            target_col="target",
            id_col="id",
        )
