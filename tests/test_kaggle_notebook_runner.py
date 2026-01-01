"""Tests for Kaggle notebook runner helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagglebot.bootstrap import bootstrap_competition
from kagglebot.paths import CompetitionPaths
from kagglebot.runners.base import RunContext
from kagglebot.runners.kaggle_notebook import (
    KaggleNotebookRunner,
    build_kernel_metadata,
    find_submission_file,
    sanitize_kernel_slug,
)


def test_sanitize_kernel_slug() -> None:
    assert sanitize_kernel_slug("KaggleBot Titan! 2024") == "kagglebot-titan-2024"


def test_build_kernel_metadata_accelerators() -> None:
    base = dict(
        kaggle_username="user",
        kernel_slug="kernel-slug",
        title="Kagglebot demo",
        competition_slug="titanic",
        enable_internet=False,
    )
    meta_gpu = build_kernel_metadata(**base, accelerator="gpu")
    assert meta_gpu["enable_gpu"] is True
    assert meta_gpu["enable_tpu"] is False

    meta_tpu = build_kernel_metadata(**base, accelerator="tpu")
    assert meta_tpu["enable_gpu"] is False
    assert meta_tpu["enable_tpu"] is True

    meta_none = build_kernel_metadata(**base, accelerator="none")
    assert meta_none["enable_gpu"] is False
    assert meta_none["enable_tpu"] is False


def test_find_submission_file(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    nested = output_dir / "nested"
    nested.mkdir()
    submission = nested / "submission.csv"
    submission.write_text("id,target\n1,0.1\n", encoding="utf-8")
    assert find_submission_file(output_dir) == submission

    extra = output_dir / "submission.csv"
    extra.write_text("id,target\n2,0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Multiple submission.csv"):
        find_submission_file(output_dir)


def test_kaggle_notebook_runner_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    slug = "demo"
    bootstrap_competition(slug=slug, root=tmp_path)
    paths = CompetitionPaths(slug=slug, repo_root=tmp_path)

    monkeypatch.setenv("KAGGLE_USERNAME", "demo-user")
    monkeypatch.setattr(
        "kagglebot.kaggle_cli.kernels_push",
        lambda *args, **kwargs: pytest.fail("kernels_push should not be called in dry-run"),
    )
    monkeypatch.setattr(
        "kagglebot.kaggle_cli.kernels_status",
        lambda *args, **kwargs: pytest.fail("kernels_status should not be called in dry-run"),
    )
    monkeypatch.setattr(
        "kagglebot.kaggle_cli.kernels_output",
        lambda *args, **kwargs: pytest.fail("kernels_output should not be called in dry-run"),
    )
    monkeypatch.setattr(
        "kagglebot.kaggle_cli.competitions_files",
        lambda *args, **kwargs: pytest.fail("competitions_files should not be called in dry-run"),
    )

    runner = KaggleNotebookRunner()
    context = RunContext(
        competition=slug,
        slug=slug,
        run_id="run-1",
        paths=paths,
        workdir=tmp_path,
        dry_run=True,
        submit=False,
        force=False,
        force_submit=False,
        message="",
        time_budget_minutes=1,
        cv_folds=2,
        model_names=None,
        use_stacking=False,
        compute="kaggle_gpu",
        accelerator="gpu",
        enable_internet=False,
        kaggle_username=None,
        strict_accelerator=False,
    )

    result = runner.run(context)
    kernel_dir = paths.artifacts / "run-1" / "kernel"
    assert result.submission_path is None
    assert (kernel_dir / "kernel-metadata.json").exists()
    assert (kernel_dir / "main.py").exists()
    summary = json.loads((paths.artifacts / "run-1" / "summary.json").read_text(encoding="utf-8"))
    assert summary["runner"] == "kaggle_notebook"
