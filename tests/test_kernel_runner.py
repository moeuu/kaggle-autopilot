"""Tests for kernel runner helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagglebot.exceptions import KernelFailedError
from kagglebot.kernel_runner import (
    _append_local_kernel_duration_history,
    _build_local_kernel_progress_tracker,
    _estimate_local_kernel_duration_seconds,
    _extract_training_stage_from_line,
    _resolve_fold_current,
    _resolve_seed_current,
    find_submission_file,
    run_kernel,
    run_kernel_local,
    sanitize_kernel_slug,
)


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
    plan_path = tmp_path / "demo" / "plan.json"
    plan_path.write_text(json.dumps({"toggles": {"USE_MODEL": True}}, indent=2), encoding="utf-8")
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
    staged_plan = tmp_path / "demo" / "kernels" / "run-1" / "plan.json"
    assert staged_plan.exists()
    assert json.loads(staged_plan.read_text(encoding="utf-8")) == {"toggles": {"USE_MODEL": True}}


def test_find_submission_file(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    nested = output_dir / "nested"
    nested.mkdir()
    submission = nested / "submission.csv"
    submission.write_text("id,target\n1,0.1\n", encoding="utf-8")
    assert find_submission_file(output_dir) == submission


def test_kernel_metadata_tpu(tmp_path: Path) -> None:
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


def test_inject_column_fill_shim(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text("print('ok')\n", encoding="utf-8")
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    payload = {"files": {"test.csv": ["A", "B"]}}
    (context_dir / "column_fill.json").write_text(json.dumps(payload), encoding="utf-8")

    kernel_runner._inject_column_fill_shim(kernel_dir, context_dir)

    site_path = kernel_dir / "sitecustomize.py"
    assert site_path.exists()
    text = site_path.read_text(encoding="utf-8")
    assert "column-fill-shim" in text
    assert "column_fill.json" in text
    assert (kernel_dir / "column_fill.json").exists()


def test_inject_object_coerce_shim(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text("print('ok')\n", encoding="utf-8")
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    payload = {"enabled": True}
    (context_dir / "object_coerce.json").write_text(json.dumps(payload), encoding="utf-8")

    kernel_runner._inject_object_coerce_shim(kernel_dir, context_dir)

    site_path = kernel_dir / "sitecustomize.py"
    assert site_path.exists()
    text = site_path.read_text(encoding="utf-8")
    assert "object-coerce-shim" in text
    assert "object_coerce.json" in text
    assert (kernel_dir / "object_coerce.json").exists()


def test_inject_device_coerce_shim(tmp_path: Path) -> None:
    from kagglebot import kernel_runner

    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text("print('ok')\n", encoding="utf-8")
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    payload = {"enabled": True}
    (context_dir / "device_coerce.json").write_text(json.dumps(payload), encoding="utf-8")

    kernel_runner._inject_device_coerce_shim(kernel_dir, context_dir)

    site_path = kernel_dir / "sitecustomize.py"
    assert site_path.exists()
    text = site_path.read_text(encoding="utf-8")
    assert "device-coerce-shim" in text
    assert "device_coerce.json" in text
    assert (kernel_dir / "device_coerce.json").exists()


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


def test_run_kernel_requires_authoritative_kernel(tmp_path: Path) -> None:
    with pytest.raises(KernelFailedError, match="Authoritative kernel entrypoint is missing"):
        run_kernel(
            slug="demo",
            run_id="run-4",
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


def test_run_kernel_local_executes_staged_copy(tmp_path: Path) -> None:
    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    source_kernel_path = source_kernel_dir / "kernel.py"
    source_text = "\n".join(
        [
            "from pathlib import Path",
            "",
            "Path('submission.csv').write_text('id,target\\n1,0.1\\n', encoding='utf-8')",
            "Path('metrics.json').write_text('{\"metric\":\"rmse\",\"offline_value\":0.1}', encoding='utf-8')",
        ]
    )
    source_kernel_path.write_text(source_text + "\n", encoding="utf-8")
    plan_path = tmp_path / "demo" / "plan.json"
    plan_path.write_text(json.dumps({"toggles": {"USE_MODEL": True}}, indent=2), encoding="utf-8")

    result = run_kernel_local(
        slug="demo",
        run_id="run-5",
        iteration=1,
        base_dir=tmp_path,
        accelerator="gpu",
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        seed=42,
        dry_run=False,
        timeout_minutes=1,
        strict_accelerator=False,
    )

    assert result.submission_path is not None and result.submission_path.exists()
    assert result.metrics_path is not None and result.metrics_path.exists()
    assert source_kernel_path.read_text(encoding="utf-8") == source_text + "\n"
    staged_kernel = tmp_path / "demo" / "kernels" / "run-5" / "local-iter-1" / "kernel.py"
    assert staged_kernel.exists()
    staged_plan_local = tmp_path / "demo" / "kernels" / "run-5" / "local-iter-1" / "plan.json"
    staged_plan_parent = tmp_path / "demo" / "kernels" / "run-5" / "plan.json"
    assert staged_plan_local.exists()
    assert staged_plan_parent.exists()
    assert json.loads(staged_plan_local.read_text(encoding="utf-8")) == {"toggles": {"USE_MODEL": True}}
    assert json.loads(staged_plan_parent.read_text(encoding="utf-8")) == {"toggles": {"USE_MODEL": True}}


def test_run_kernel_local_mirrors_context_sample_submission(tmp_path: Path) -> None:
    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    source_kernel_path = source_kernel_dir / "kernel.py"
    source_kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "data_root = Path(__file__).resolve().parents[3] / 'data'",
                "sample = data_root / 'sample_submission.csv'",
                "if not sample.exists():",
                "    raise FileNotFoundError(f'sample missing at {sample}')",
                "Path('submission.csv').write_text('id,target\\n1,0.1\\n', encoding='utf-8')",
                "Path('metrics.json').write_text('{\"metric\":\"rmse\",\"offline_value\":0.1}', encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    context_sample = tmp_path / "demo" / "context" / "sample_submission.csv"
    context_sample.parent.mkdir(parents=True, exist_ok=True)
    context_sample.write_text("id,target\n1,0.0\n", encoding="utf-8")

    result = run_kernel_local(
        slug="demo",
        run_id="run-6",
        iteration=1,
        base_dir=tmp_path,
        accelerator="gpu",
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        seed=42,
        dry_run=False,
        timeout_minutes=1,
        strict_accelerator=False,
    )

    mirrored = tmp_path / "demo" / "data" / "sample_submission.csv"
    assert mirrored.exists()
    assert mirrored.read_text(encoding="utf-8") == "id,target\n1,0.0\n"
    assert result.submission_path is not None and result.submission_path.exists()
    assert result.metrics_path is not None and result.metrics_path.exists()


def test_local_kernel_duration_history_estimate_uses_recent_median(tmp_path: Path) -> None:
    for idx, duration in enumerate([100.0, 120.0, 80.0, 110.0], start=1):
        _append_local_kernel_duration_history(
            base_dir=tmp_path,
            slug="demo",
            run_id="run-a",
            iteration=idx,
            duration_sec=duration,
        )

    estimate, samples = _estimate_local_kernel_duration_seconds(base_dir=tmp_path, slug="demo")
    assert samples == 4
    assert estimate == 105.0


def test_run_kernel_local_records_duration_history(tmp_path: Path) -> None:
    source_kernel_dir = tmp_path / "demo" / "kernel"
    source_kernel_dir.mkdir(parents=True, exist_ok=True)
    source_kernel_path = source_kernel_dir / "kernel.py"
    source_kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "Path('submission.csv').write_text('id,target\\n1,0.1\\n', encoding='utf-8')",
                "Path('metrics.json').write_text('{\"metric\":\"rmse\",\"offline_value\":0.1}', encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    context_sample = tmp_path / "demo" / "context" / "sample_submission.csv"
    context_sample.parent.mkdir(parents=True, exist_ok=True)
    context_sample.write_text("id,target\n1,0.0\n", encoding="utf-8")

    _ = run_kernel_local(
        slug="demo",
        run_id="run-7",
        iteration=1,
        base_dir=tmp_path,
        accelerator="gpu",
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        seed=42,
        dry_run=False,
        timeout_minutes=1,
        strict_accelerator=False,
    )
    estimate, samples = _estimate_local_kernel_duration_seconds(base_dir=tmp_path, slug="demo")
    assert samples == 1
    assert estimate is not None and estimate > 0.0


def test_extract_training_stage_from_line() -> None:
    inline = "[kernel] yolo_ensemble_wbf_geometry: seed=2024 fold=0 imgsz=768 epochs=250"
    assert _extract_training_stage_from_line(inline) == ("yolo_ensemble_wbf_geometry", 2024, 0)

    path_line = "/tmp/runs/yolo_ensemble_wbf_geometry_seed2024_fold1/weights/best.pt saved"
    assert _extract_training_stage_from_line(path_line) == ("yolo_ensemble_wbf_geometry", 2024, 1)


def test_progress_helpers() -> None:
    assert _resolve_seed_current(seed=2024, expected_seeds=[42, 2024, 777]) == 2
    assert _resolve_seed_current(seed=999, expected_seeds=[42, 2024, 777]) is None
    assert _resolve_fold_current(fold_raw=0, expected_folds=5, zero_based=True) == 1
    assert _resolve_fold_current(fold_raw=2, expected_folds=5, zero_based=True) == 3
    assert _resolve_fold_current(fold_raw=2, expected_folds=5, zero_based=False) == 2


def test_build_local_kernel_progress_tracker_reads_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "demo" / "plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps({"cv_folds": 5, "eval_seeds": [42, 2024, 777]}, indent=2),
        encoding="utf-8",
    )

    tracker = _build_local_kernel_progress_tracker(base_dir=tmp_path, slug="demo")
    assert tracker.expected_folds == 5
    assert tracker.expected_seeds == [42, 2024, 777]
