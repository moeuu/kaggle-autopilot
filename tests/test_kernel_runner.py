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
    kernel_path = tmp_path / "demo" / "kernel" / "kernel.py"
    kernel_path.parent.mkdir(parents=True, exist_ok=True)
    kernel_path.write_text(
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/train.csv'",
                "OUT1 = '/kaggle/working/submission.csv'",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
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
        timeout_minutes=None,
    )
    meta_path = tmp_path / "demo" / "kernels" / "run-1" / "kernel-metadata.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["enable_gpu"] is True
    assert payload["enable_tpu"] is False
    assert (tmp_path / "demo" / "kernels" / "run-1" / "kernel.py").exists()
    assert (tmp_path / "demo" / "kernels" / "run-1" / "kernel.py").exists()


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
        timeout_minutes=None,
    )
    meta_path = tmp_path / "demo" / "kernels" / "run-2" / "kernel-metadata.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["enable_tpu"] is True
    assert payload["enable_gpu"] is False


def test_kernel_bootstrap_preserves_future_import(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python",
                "# -*- coding: utf-8 -*-",
                '"""docstring"""',
                "from __future__ import annotations",
                "",
                "print('ok')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    kernel_runner._ensure_kernel_import_path(kernel_dir)
    lines = kernel_path.read_text(encoding="utf-8").splitlines()
    future_idx = next(i for i, line in enumerate(lines) if "from __future__ import annotations" in line)
    marker_idx = next(i for i, line in enumerate(lines) if "kagglebot:kernel_sys_path" in line)
    assert marker_idx > future_idx


def test_run_kernel_uses_custom_kernel(tmp_path: Path) -> None:
    custom_kernel = tmp_path / "demo" / "kernel" / "kernel.py"
    custom_kernel.parent.mkdir(parents=True, exist_ok=True)
    custom_kernel.write_text(
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/train.csv'",
                "OUT1 = '/kaggle/working/submission.csv'",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_kernel(
        slug="demo",
        run_id="run-3",
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
        timeout_minutes=None,
    )
    kernel_path = tmp_path / "demo" / "kernels" / "run-3" / "kernel.py"
    content = kernel_path.read_text(encoding="utf-8")
    assert "/kaggle/input/" in content
    assert "submission.csv" in content
    assert "metrics.json" in content
