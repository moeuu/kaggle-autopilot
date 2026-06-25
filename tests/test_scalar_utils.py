from __future__ import annotations

from kagglebot.scalar_utils import (
    finite_float,
    non_nan_float,
    non_negative_finite_float,
    optional_int,
    optional_str,
    parse_finite_float,
    parse_int,
)


def test_optional_str_strips_and_drops_empty_values() -> None:
    assert optional_str(None) is None
    assert optional_str("") is None
    assert optional_str("  value  ") == "value"
    assert optional_str(42) == "42"


def test_finite_float_rejects_empty_non_numeric_and_non_finite_values() -> None:
    assert finite_float(None) is None
    assert finite_float("x") is None
    assert finite_float("nan") is None
    assert finite_float("inf") is None
    assert finite_float("1.25") == 1.25


def test_non_nan_float_preserves_infinity_but_rejects_nan() -> None:
    assert non_nan_float(None) is None
    assert non_nan_float("x") is None
    assert non_nan_float("nan") is None
    assert non_nan_float("inf") == float("inf")
    assert non_nan_float("1.25") == 1.25


def test_optional_int_matches_plain_int_conversion() -> None:
    assert optional_int(None) is None
    assert optional_int("x") is None
    assert optional_int("4") == 4
    assert optional_int(3.0) == 3


def test_parse_finite_float_rejects_bool_blank_and_non_finite_values() -> None:
    assert parse_finite_float(True) is None
    assert parse_finite_float("") is None
    assert parse_finite_float("nan") is None
    assert parse_finite_float("inf") is None
    assert parse_finite_float("1,234.5", allow_commas=True) == 1234.5
    assert parse_finite_float("1,234.5") is None


def test_parse_int_controls_float_and_comma_handling() -> None:
    assert parse_int(True) is None
    assert parse_int("") is None
    assert parse_int("1,234", allow_commas=True) == 1234
    assert parse_int("1,234") is None
    assert parse_int("12.0", allow_float=True) == 12
    assert parse_int("12.9", allow_float=True) is None
    assert parse_int("12.9", allow_float=True, require_integral_float=False) == 12
    assert parse_int("inf", allow_float=True, require_integral_float=False) is None


def test_non_negative_finite_float_clamps_and_defaults_values() -> None:
    assert non_negative_finite_float(4) == 4.0
    assert non_negative_finite_float(-2.5) == 0.0
    assert non_negative_finite_float(True, default=3.0) == 3.0
    assert non_negative_finite_float(float("nan"), default=2.0) == 2.0
    assert non_negative_finite_float(float("inf"), default=-1.0) == 0.0
    assert non_negative_finite_float("5", default=1.0) == 1.0
    assert non_negative_finite_float("5", default=1.0, allow_strings=True) == 5.0
