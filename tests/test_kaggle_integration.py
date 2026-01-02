"""Optional integration tests against Kaggle CLI (network + credentials required)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from kagglebot import kaggle_api


def _integration_enabled() -> bool:
    return os.getenv("KAGGLE_INTEGRATION") == "1"


def _has_kaggle_cli() -> bool:
    return shutil.which("kaggle") is not None


def _has_kaggle_creds() -> bool:
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


@pytest.mark.integration
def test_kaggle_titanic_download_and_leaderboard(tmp_path: Path) -> None:
    if not _integration_enabled():
        pytest.skip("Set KAGGLE_INTEGRATION=1 to run Kaggle integration tests.")
    if not _has_kaggle_cli():
        pytest.skip("Kaggle CLI not found on PATH.")
    if not _has_kaggle_creds():
        pytest.skip("Kaggle credentials not configured.")

    data_dir = tmp_path / "data"
    context_dir = tmp_path / "context"
    try:
        kaggle_api.download_competition("titanic", data_dir, force=True, quiet=True)
        result = kaggle_api.leaderboard_top1("titanic", context_dir)
        assert "score" in result
        assert any(data_dir.glob("*.zip"))
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)
        shutil.rmtree(context_dir, ignore_errors=True)
