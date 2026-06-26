from __future__ import annotations

import json

import pandas as pd


def safe_nunique(series: pd.Series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        normalized = series.dropna().map(profile_hashable_value)
        return int(normalized.nunique(dropna=True))


def profile_hashable_value(value: object) -> object:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return value
