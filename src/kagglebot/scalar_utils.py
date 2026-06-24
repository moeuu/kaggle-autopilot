from __future__ import annotations

import math


def optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def finite_float(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def non_nan_float(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


def optional_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
