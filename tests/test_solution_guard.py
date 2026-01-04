from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from kagglebot.cli import app
from kagglebot.solution_guard import ensure_solution_path_allowed


def test_import_kagglebot() -> None:
    import kagglebot  # noqa: F401


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_solution_guard_allows_artifacts(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    allowed = artifacts_dir / "demo" / "kernel" / "kernel.py"
    allowed.parent.mkdir(parents=True, exist_ok=True)
    ensure_solution_path_allowed(allowed, artifacts_dir=artifacts_dir, slug="demo")


def test_solution_guard_blocks_src(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    artifacts_dir = repo_root / "artifacts"
    forbidden = repo_root / "src" / "kagglebot" / "demo" / "kernel" / "kernel.py"
    forbidden.parent.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="must not live under"):
        ensure_solution_path_allowed(forbidden, artifacts_dir=artifacts_dir, slug="demo")


def test_solution_guard_blocks_non_solution_root(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    forbidden = artifacts_dir / "demo" / "scratch" / "script.py"
    forbidden.parent.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="Solution code must live under"):
        ensure_solution_path_allowed(forbidden, artifacts_dir=artifacts_dir, slug="demo")
