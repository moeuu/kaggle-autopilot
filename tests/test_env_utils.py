from __future__ import annotations

from kagglebot.env_utils import env_flag, env_int, env_truthy


def test_env_flag_parses_boolean_values(monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_TEST_FLAG", "yes")
    assert env_flag("KAGGLEBOT_TEST_FLAG", default=False) is True

    monkeypatch.setenv("KAGGLEBOT_TEST_FLAG", "off")
    assert env_flag("KAGGLEBOT_TEST_FLAG", default=True) is False

    monkeypatch.setenv("KAGGLEBOT_TEST_FLAG", "maybe")
    assert env_flag("KAGGLEBOT_TEST_FLAG", default=True) is True

    monkeypatch.delenv("KAGGLEBOT_TEST_FLAG", raising=False)
    assert env_flag("KAGGLEBOT_TEST_FLAG", default=False) is False


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
