from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from kagglebot.local_kernel_progress import (
    build_local_kernel_progress_tracker,
    format_local_gpu_activity_suffix,
    format_local_kernel_activity_suffix,
    print_local_kernel_progress,
)


def test_build_local_kernel_progress_tracker_reads_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "demo" / "plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps({"cv_folds": 5, "eval_seeds": [42, 2024, 777]}, indent=2), encoding="utf-8")

    tracker = build_local_kernel_progress_tracker(base_dir=tmp_path, slug="demo")

    assert tracker.expected_folds == 5
    assert tracker.expected_seeds == [42, 2024, 777]


def test_build_local_kernel_progress_tracker_ignores_invalid_or_non_object_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "demo" / "plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)

    plan_path.write_text("{", encoding="utf-8")
    invalid_tracker = build_local_kernel_progress_tracker(base_dir=tmp_path, slug="demo")
    assert invalid_tracker.expected_folds is None
    assert invalid_tracker.expected_seeds == []

    plan_path.write_text("[]", encoding="utf-8")
    array_tracker = build_local_kernel_progress_tracker(base_dir=tmp_path, slug="demo")
    assert array_tracker.expected_folds is None
    assert array_tracker.expected_seeds == []


def test_progress_tracker_reports_generic_activity(tmp_path: Path) -> None:
    plan_path = tmp_path / "demo" / "plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps({"cv_folds": 3, "eval_seeds": [42]}, indent=2), encoding="utf-8")
    tracker = build_local_kernel_progress_tracker(base_dir=tmp_path, slug="demo")

    tracker.observe_line("[kernel] Running pipeline: tri_blend_stack")
    tracker.observe_line("[kernel] Pipeline tri_blend_stack: CV=0.123 method=weighted_mean_log")

    snapshot = tracker.snapshot()
    assert snapshot["lines_seen"] == 2
    assert snapshot["current_pipeline"] == "tri_blend_stack"
    assert snapshot["completed_pipeline_count"] == 1
    assert isinstance(snapshot["last_log_age_sec"], (int, float))
    assert "artifact_count" in snapshot
    assert "last_artifact_age_sec" in snapshot

    suffix = format_local_kernel_activity_suffix(tracker)
    assert "logs=2" in suffix
    assert "pipeline=tri_blend_stack" in suffix
    assert "pipelines_done=1" in suffix
    assert "artifacts=" in suffix
    assert "last_artifact=" in suffix


def test_progress_tracker_reports_runtime_pipeline_suite_and_model(tmp_path: Path) -> None:
    tracker = build_local_kernel_progress_tracker(base_dir=tmp_path, slug="demo")

    tracker.observe_line("Training pipeline: catboost_origstats_multiseed_fast__origw0800")
    tracker.observe_line("Suite: comp_plus_orig")
    tracker.observe_line("[kernel] train start: model=catboost.CatBoostClassifier rows=123 cols=45")
    tracker.observe_line("[kernel] CatBoost GPU failed; retrying on CPU: RuntimeError: CUDA out of memory")

    snapshot = tracker.snapshot()
    assert snapshot["current_pipeline"] == "catboost_origstats_multiseed_fast__origw0800"
    assert snapshot["current_suite"] == "comp_plus_orig"
    assert snapshot["current_model"] == "catboost.CatBoostClassifier"
    assert snapshot["last_fallback_reason"] == "RuntimeError: CUDA out of memory"

    suffix = format_local_kernel_activity_suffix(tracker)
    assert "pipeline=catboost_origstats_multiseed_fast__origw0800" in suffix
    assert "suite=comp_plus_orig" in suffix
    assert "model=catboost.CatBoostClassifier" in suffix
    assert "fallback=RuntimeError: CUDA out of memory" in suffix


def test_progress_tracker_reports_artifact_activity(tmp_path: Path) -> None:
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir(parents=True, exist_ok=True)
    artifact = watch_dir / "metrics.json"
    artifact.write_text('{"ok":true}\n', encoding="utf-8")
    tracker = build_local_kernel_progress_tracker(base_dir=tmp_path, slug="demo", watch_dirs=[watch_dir])

    snapshot = tracker.snapshot()
    assert int(snapshot["artifact_count"]) >= 1
    assert isinstance(snapshot["last_artifact_age_sec"], (int, float))

    suffix = format_local_kernel_activity_suffix(tracker)
    assert "artifacts=" in suffix
    assert "last_artifact=" in suffix


def test_progress_tracker_ignores_stale_artifacts_then_counts_new_activity(tmp_path: Path) -> None:
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir(parents=True, exist_ok=True)
    stale_artifact = watch_dir / "submission.csv"
    stale_artifact.write_text("id,target\n1,0.0\n", encoding="utf-8")
    stale_mtime = time.time() - 60.0
    os.utime(stale_artifact, (stale_mtime, stale_mtime))

    tracker = build_local_kernel_progress_tracker(base_dir=tmp_path, slug="demo", watch_dirs=[watch_dir])
    stale_snapshot = tracker.snapshot()
    assert stale_snapshot["artifact_count"] == 0
    assert stale_snapshot["last_artifact_age_sec"] is None

    fresh_artifact = watch_dir / "metrics.json"
    fresh_artifact.write_text('{"ok":true}\n', encoding="utf-8")

    fresh_snapshot = tracker.snapshot()
    assert int(fresh_snapshot["artifact_count"]) >= 1
    assert isinstance(fresh_snapshot["last_artifact_age_sec"], (int, float))


def test_format_local_gpu_activity_suffix_handles_missing_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert format_local_gpu_activity_suffix(accelerator="gpu") == ""


def test_progress_does_not_report_zero_eta_after_historical_estimate_expires(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_local_kernel_progress(
        elapsed_sec=600.0,
        timeout_sec=1440 * 60,
        eta_total_sec=120.0,
        eta_samples=2,
        progress_tracker=None,
        accelerator="cpu",
    )

    output = capsys.readouterr().out
    assert "eta=unknown" in output
    assert "eta~0s" not in output
    assert "historical median~120s exceeded" in output
    assert "timeout in <= 85800s" in output
