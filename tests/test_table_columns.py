from __future__ import annotations

import numpy as np
import pandas as pd

from kagglebot.table_columns import frame_with_normalized_table_columns, normalize_table_column_names


def test_normalize_table_column_names_stabilizes_missing_generated_and_duplicate_names() -> None:
    columns = ["id", "", None, np.nan, "Unnamed: 4", "score", "score"]

    assert normalize_table_column_names(columns) == [
        "id",
        "column_2",
        "column_3",
        "column_4",
        "column_5",
        "score",
        "score_1",
    ]


def test_normalize_table_column_names_flattens_multiindex_generated_missing_parts() -> None:
    columns = pd.MultiIndex.from_tuples([("fold", "a"), ("Unnamed: 1_level_0", None), ("x", "x")])

    assert normalize_table_column_names(columns) == ["fold_a", "column_2", "x_x"]


def test_frame_with_normalized_table_columns_copies_only_when_needed() -> None:
    frame = pd.DataFrame([[1, 2, 3]], columns=["id", "", "id"])

    normalized = frame_with_normalized_table_columns(frame)

    assert normalized is not frame
    assert list(normalized.columns) == ["id", "column_2", "id_1"]
    assert list(frame.columns) == ["id", "", "id"]

    already_normalized = pd.DataFrame({"id": [1], "target": [0.1]})

    assert frame_with_normalized_table_columns(already_normalized) is already_normalized
