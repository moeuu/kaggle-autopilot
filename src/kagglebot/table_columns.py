from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd


def normalize_table_column_names(columns: Sequence[object]) -> list[str]:
    return dedupe_table_column_names(
        [stable_table_column_name(column, position) for position, column in enumerate(columns)]
    )


def frame_with_normalized_table_columns(frame):
    normalized = normalize_table_column_names(frame.columns)
    if list(frame.columns) == normalized:
        return frame
    copied = frame.copy()
    copied.columns = normalized
    return copied


def stable_table_column_name(column: object, position: int) -> str:
    fallback = f"column_{position + 1}"
    if isinstance(column, tuple):
        parts = [
            str(part).strip()
            for part in column
            if not table_column_part_is_missing(part) and not table_column_name_is_generated_missing(str(part).strip())
        ]
        return "_".join(part for part in parts if part) or fallback
    if table_column_part_is_missing(column):
        return fallback
    name = str(column)
    stripped = name.strip()
    if not stripped or table_column_name_is_generated_missing(stripped):
        return fallback
    return name


def table_column_name_is_generated_missing(name: str) -> bool:
    return bool(re.fullmatch(r"Unnamed:\s*\d+(?:_level_\d+)?", str(name).strip(), flags=re.IGNORECASE))


def table_column_part_is_missing(value: object) -> bool:
    if value is None:
        return True
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def dedupe_table_column_names(columns: Sequence[str]) -> list[str]:
    counts: dict[str, int] = {}
    deduped: list[str] = []
    for column in columns:
        base = str(column)
        count = counts.get(base, 0)
        deduped.append(base if count == 0 else f"{base}_{count}")
        counts[base] = count + 1
    return deduped
