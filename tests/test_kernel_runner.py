"""Tests for kernel runner helpers."""

from __future__ import annotations

import json
from pathlib import Path

from kagglebot.kernel_runner import find_submission_file, run_kernel, sanitize_kernel_slug


def test_sanitize_kernel_slug() -> None:
    assert sanitize_kernel_slug("KaggleBot Titan! 2024") == "kagglebot-titan-2024"


def test_run_kernel_dry_run(tmp_path: Path) -> None:
    # Ensure dry-run avoids Kaggle CLI calls.
    from kagglebot import kernel_runner

    kernel_runner.kernels_init = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should not run"))
    run_kernel(
        slug="demo",
        run_id="run-1",
        iteration=1,
        base_dir=tmp_path,
        kaggle_username="user",
        kernel_name=None,
        accelerator="gpu",
        enable_internet=False,
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        seed=42,
        dry_run=True,
        timeout_minutes=1,
    )
    meta_path = tmp_path / "demo" / "runs" / "run-1" / "kernel" / "kernel-metadata.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["enable_gpu"] is True
    assert payload["enable_tpu"] is False
    assert (tmp_path / "demo" / "runs" / "run-1" / "kernel" / "kernel.py").exists()


def test_find_submission_file(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    nested = output_dir / "nested"
    nested.mkdir()
    submission = nested / "submission.csv"
    submission.write_text("id,target\n1,0.1\n", encoding="utf-8")
    assert find_submission_file(output_dir) == submission


def test_kernel_metadata_tpu(tmp_path: Path) -> None:
    run_kernel(
        slug="demo",
        run_id="run-2",
        iteration=1,
        base_dir=tmp_path,
        kaggle_username="user",
        kernel_name=None,
        accelerator="tpu",
        enable_internet=False,
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        seed=42,
        dry_run=True,
        timeout_minutes=1,
    )
    meta_path = tmp_path / "demo" / "runs" / "run-2" / "kernel" / "kernel-metadata.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["enable_tpu"] is True
    assert payload["enable_gpu"] is False
