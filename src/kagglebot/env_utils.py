from __future__ import annotations

import math
import os

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def env_flag(name: str, *, default: bool) -> bool:
    """Parse a boolean environment flag with an explicit default fallback."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return default


def env_int(name: str, *, default: int, min_value: int = 0) -> int:
    raw = os.environ.get(name)
    parsed = parse_int_value(raw)
    if parsed is None:
        return default
    return max(min_value, parsed)


def env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in _TRUE_VALUES


def parse_int_value(raw: object, *, allow_float: bool = False) -> int | None:
    if raw is None or isinstance(raw, bool):
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        if allow_float:
            value = float(text)
            return int(value) if math.isfinite(value) and value.is_integer() else None
        return int(text)
    except ValueError:
        return None


def parse_float_value(raw: object) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None
