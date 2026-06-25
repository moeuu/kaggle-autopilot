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


def parse_finite_float(value: object, *, allow_commas: bool = False) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if allow_commas:
        text = text.replace(",", "")
    if not text:
        return None
    return finite_float(text)


def parse_int(
    value: object,
    *,
    allow_commas: bool = False,
    allow_float: bool = False,
    require_integral_float: bool = True,
) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if allow_commas:
        text = text.replace(",", "")
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        if not allow_float:
            return None
    parsed = finite_float(text)
    if parsed is None:
        return None
    if require_integral_float and not parsed.is_integer():
        return None
    return int(parsed)
