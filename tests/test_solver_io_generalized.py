from __future__ import annotations

import numpy as np
import pandas as pd

from kagglebot.solver.io import infer_submission_layout, write_submission


def test_infer_submission_layout_supports_multi_target() -> None:
    train = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "f1": [10, 20, 30],
            "target_a": [0, 1, 0],
            "target_b": [0.2, 0.4, 0.6],
        }
    )
    test = pd.DataFrame({"id": [4, 5], "f1": [40, 50]})
    sample = pd.DataFrame({"id": [4, 5], "target_a": [0, 0], "target_b": [0.0, 0.0]})

    id_col, target_cols, feature_cols = infer_submission_layout(train=train, test=test, sample=sample)

    assert id_col == "id"
    assert target_cols == ["target_a", "target_b"]
    assert feature_cols == ["f1"]


def test_write_submission_multitarget_aligns_by_id(tmp_path) -> None:
    sample = pd.DataFrame({"id": [11, 12], "target_a": [0, 0], "target_b": [0.0, 0.0]})
    test = pd.DataFrame({"id": [12, 11], "f": [1, 2]})
    preds = {"target_a": np.array([1, 0]), "target_b": np.array([0.9, 0.1])}

    out = write_submission(
        sample=sample,
        test=test,
        preds=preds,
        id_column="id",
        target_columns=["target_a", "target_b"],
        output_path=tmp_path / "submission.csv",
    )
    frame = pd.read_csv(out)
    assert frame["target_a"].tolist() == [0, 1]
    assert frame["target_b"].tolist() == [0.1, 0.9]


def test_write_submission_without_id_uses_row_order(tmp_path) -> None:
    sample = pd.DataFrame({"prediction": [0.0, 0.0, 0.0]})
    test = pd.DataFrame({"f": [1, 2, 3]})
    preds = np.array([0.2, 0.4, 0.6])

    out = write_submission(
        sample=sample,
        test=test,
        preds=preds,
        id_column=None,
        target_column="prediction",
        output_path=tmp_path / "submission.csv",
    )
    frame = pd.read_csv(out)
    assert frame["prediction"].tolist() == [0.2, 0.4, 0.6]


def test_write_submission_accepts_2d_prediction_matrix(tmp_path) -> None:
    sample = pd.DataFrame({"row_id": [1, 2], "y1": [0.0, 0.0], "y2": [0.0, 0.0]})
    test = pd.DataFrame({"row_id": [1, 2], "f": [5, 6]})
    preds = np.array([[0.3, 0.7], [0.4, 0.6]])

    out = write_submission(
        sample=sample,
        test=test,
        preds=preds,
        id_column="row_id",
        target_columns=["y1", "y2"],
        output_path=tmp_path / "submission.csv",
    )
    frame = pd.read_csv(out)
    assert frame["y1"].tolist() == [0.3, 0.4]
    assert frame["y2"].tolist() == [0.7, 0.6]
