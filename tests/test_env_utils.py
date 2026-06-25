from __future__ import annotations

from kagglebot.env_utils import env_flag, env_int, env_truthy, parse_bool_value, parse_float_value, parse_int_value


def test_env_flag_parses_boolean_values(monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_TEST_FLAG", "yes")
    assert env_flag("KAGGLEBOT_TEST_FLAG", default=False) is True

    monkeypatch.setenv("KAGGLEBOT_TEST_FLAG", "off")
    assert env_flag("KAGGLEBOT_TEST_FLAG", default=True) is False

    monkeypatch.setenv("KAGGLEBOT_TEST_FLAG", "maybe")
    assert env_flag("KAGGLEBOT_TEST_FLAG", default=True) is True

    monkeypatch.delenv("KAGGLEBOT_TEST_FLAG", raising=False)
    assert env_flag("KAGGLEBOT_TEST_FLAG", default=False) is False


def test_parse_bool_value_handles_env_and_policy_values() -> None:
    assert parse_bool_value("y", default=False) is True
    assert parse_bool_value("n", default=True) is False
    assert parse_bool_value(1, default=False) is True
    assert parse_bool_value(0, default=True) is False
    assert parse_bool_value(True, default=False) is True
    assert parse_bool_value("maybe", default=True) is True


def test_env_int_parses_with_minimum_and_default(monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_TEST_INT", "7")
    assert env_int("KAGGLEBOT_TEST_INT", default=3) == 7

    monkeypatch.setenv("KAGGLEBOT_TEST_INT", "-4")
    assert env_int("KAGGLEBOT_TEST_INT", default=3) == 0
    assert env_int("KAGGLEBOT_TEST_INT", default=3, min_value=-10) == -4

    monkeypatch.setenv("KAGGLEBOT_TEST_INT", "bad")
    assert env_int("KAGGLEBOT_TEST_INT", default=3) == 3


def test_env_truthy_checks_true_values(monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_TEST_TRUTHY", "on")
    assert env_truthy("KAGGLEBOT_TEST_TRUTHY") is True

    monkeypatch.setenv("KAGGLEBOT_TEST_TRUTHY", "0")
    assert env_truthy("KAGGLEBOT_TEST_TRUTHY") is False

    monkeypatch.delenv("KAGGLEBOT_TEST_TRUTHY", raising=False)
    assert env_truthy("KAGGLEBOT_TEST_TRUTHY") is False


def test_parse_int_value_supports_explicit_float_mode() -> None:
    assert parse_int_value("7") == 7
    assert parse_int_value("7.0") is None
    assert parse_int_value("7.0", allow_float=True) == 7
    assert parse_int_value("7.5", allow_float=True) is None
    assert parse_int_value(True) is None
    assert parse_int_value("bad") is None


def test_parse_float_value_rejects_blank_bool_and_non_finite() -> None:
    assert parse_float_value("1.5") == 1.5
    assert parse_float_value("") is None
    assert parse_float_value(False) is None
    assert parse_float_value("nan") is None
    assert parse_float_value("bad") is None
