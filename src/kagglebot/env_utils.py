from __future__ import annotations

import os
from pathlib import Path

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


def env_optional_int(name: str, *, allow_float: bool = False) -> int | None:
    return parse_int_value(os.environ.get(name), allow_float=allow_float)


def env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return parse_bool_value(value, default=False)


def read_env_or_file(env_name: str, file_env_name: str) -> str | None:
    direct = os.environ.get(env_name)
    if direct and direct.strip():
        return direct.strip()
    file_value = os.environ.get(file_env_name)
    if not file_value:
        return None
    try:
        return Path(file_value).expanduser().read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


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
