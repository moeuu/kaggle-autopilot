"""Smoke test for CLI import."""

from kagglebot import cli


def test_cli_import() -> None:
    assert cli.app is not None
