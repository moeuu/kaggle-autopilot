"""Smoke test for CLI import."""

from kagglebot import cli


def test_cli_import() -> None:
    assert cli.app is not None


def test_watch_help() -> None:
    from typer.testing import CliRunner

    result = CliRunner().invoke(cli.app, ["watch", "--help"])

    assert result.exit_code == 0
    assert "--submit-policy" in result.stdout
