"""Smoke test for CLI import."""

from pathlib import Path

import pytest
import typer

from kagglebot import cli


def test_cli_import() -> None:
    assert cli.app is not None


def test_watch_help() -> None:
    from typer.testing import CliRunner

    result = CliRunner().invoke(cli.app, ["watch", "--help"])

    assert result.exit_code == 0
    assert "--submit-policy" in result.stdout


def test_cli_implement_uses_shared_verify_helper(monkeypatch, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    captured: dict[str, object] = {}

    def fake_run_repo_verify(cmd: str, **kwargs: object) -> None:
        captured["cmd"] = cmd
        captured.update(kwargs)

    def fake_bootstrap_competition(**kwargs: object) -> Path:
        paths = kwargs["paths"]
        meta_path = paths.context_dir / "metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text("{}", encoding="utf-8")
        return meta_path

    monkeypatch.setattr(cli, "run_repo_verify", fake_run_repo_verify)
    monkeypatch.setattr(cli, "bootstrap_competition", fake_bootstrap_competition)
    monkeypatch.setattr(cli, "run_codex", lambda *args, **kwargs: None)

    result = CliRunner().invoke(
        cli.app,
        [
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--workdir",
            str(tmp_path),
            "implement",
            "demo",
            "--verify-cmd",
            "uv run pytest -q",
        ],
    )

    assert result.exit_code == 0
    assert captured["cmd"] == "uv run pytest -q"
    assert captured["dry_run"] is False
    assert captured["artifacts_dir"] == tmp_path / "artifacts"


def test_cli_resolve_accelerator_converts_policy_error(monkeypatch) -> None:
    monkeypatch.setattr(cli, "resolve_accelerator", lambda *_args: (_ for _ in ()).throw(ValueError("bad accel")))

    with pytest.raises(typer.BadParameter, match="bad accel"):
        cli._resolve_accelerator("local_gpu", "tpu")
