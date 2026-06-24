from __future__ import annotations

from kagglebot.scalar_utils import finite_float, non_nan_float, optional_int, optional_str


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
