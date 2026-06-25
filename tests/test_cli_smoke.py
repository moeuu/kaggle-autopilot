"""Smoke test for CLI import."""

from pathlib import Path

from kagglebot import cli


def test_cli_import() -> None:
    assert cli.app is not None


def test_watch_help() -> None:
    from typer.testing import CliRunner

    result = CliRunner().invoke(cli.app, ["watch", "--help"])

    assert result.exit_code == 0
    assert "--submit-policy" in result.stdout


def test_cli_run_verify_uses_shared_verify_helper(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run_verify(cmd: str, **kwargs: object) -> None:
        captured["cmd"] = cmd
        captured.update(kwargs)

    monkeypatch.setattr(cli, "run_verify", fake_run_verify)

    cli._run_verify("uv run pytest -q", dry_run=False, artifacts_dir=tmp_path / "artifacts")

    assert captured["cmd"] == "uv run pytest -q"
    assert captured["dry_run"] is False
    assert captured["artifacts_dir"] == tmp_path / "artifacts"
    assert captured["repo_root"] == Path.cwd()
