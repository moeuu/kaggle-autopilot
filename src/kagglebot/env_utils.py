from __future__ import annotations

import os

from kagglebot.scalar_utils import parse_finite_float, parse_int

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


def env_flag(name: str, *, default: bool) -> bool:
    """Parse a boolean environment flag with an explicit default fallback."""
    raw = os.environ.get(name)
    return parse_bool_value(raw, default=default)


def env_int(name: str, *, default: int, min_value: int = 0) -> int:
    raw = os.environ.get(name)
    parsed = parse_int_value(raw)
    if parsed is None:
        return default
    return max(min_value, parsed)


def env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return parse_bool_value(value, default=False)


def parse_bool_value(raw: object, *, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    normalized = str(raw).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return default


def parse_int_value(raw: object, *, allow_float: bool = False) -> int | None:
    return parse_int(raw, allow_float=allow_float)


def parse_float_value(raw: object) -> float | None:
    return parse_finite_float(raw)
