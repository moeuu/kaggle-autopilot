from __future__ import annotations

from kagglebot.scalar_utils import optional_str


def test_optional_str_strips_and_drops_empty_values() -> None:
    assert optional_str(None) is None
    assert optional_str("") is None
    assert optional_str("  value  ") == "value"
    assert optional_str(42) == "42"
