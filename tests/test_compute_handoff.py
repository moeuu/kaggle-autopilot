from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from kagglebot import compute_handoff
from kagglebot.exceptions import KernelFailedError, KernelTimeoutError


def test_should_handoff_only_after_repeated_resource_limited_local_failures() -> None:
    timeout = KernelTimeoutError("Local kernel timed out")
    assert not compute_handoff.should_handoff_local_failure(timeout, consecutive_failures=2, enabled=True)
    assert compute_handoff.should_handoff_local_failure(timeout, consecutive_failures=3, enabled=True)
    assert not compute_handoff.should_handoff_local_failure(
        KernelFailedError("Local kernel exceeded host memory guard at 24 GiB"),
        consecutive_failures=1,
        enabled=True,
    )
    assert compute_handoff.should_handoff_local_failure(
        KernelFailedError("Local kernel execution failed with CUDA OOM, then failed again after disabling LLM."),
        consecutive_failures=3,
        enabled=True,
    )
    assert compute_handoff.should_handoff_local_failure(
        KernelFailedError("No local GPU detected"),
        consecutive_failures=1,
        enabled=True,
    )
    assert not compute_handoff.should_handoff_local_failure(
        KernelFailedError("ValueError: submission shape mismatch"),
        enabled=True,
    )
    assert not compute_handoff.should_handoff_local_failure(
        KernelTimeoutError("Local kernel timed out"),
        consecutive_failures=3,
        enabled=False,
    )


def test_handoff_quota_requires_fresh_sufficient_capacity(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLEBOT_KAGGLE_GPU_HANDOFF_MIN_AVAILABLE_MINUTES", "900")
    quota_path = tmp_path / "_watch" / "kaggle_gpu" / "quota.json"
    quota_path.parent.mkdir(parents=True)
    quota_path.write_text(
        json.dumps(
            {
                "available_minutes": 899,
                "updated_at": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    low = compute_handoff.evaluate_kaggle_gpu_handoff_quota(
        artifact_root=tmp_path,
        time_budget_minutes=300,
    )
    assert low.allowed is False
    assert low.reason == "quota_low"
    assert low.available_minutes == 899

    monkeypatch.setenv("KAGGLEBOT_KAGGLE_GPU_AVAILABLE_MINUTES", "1200")
    enough = compute_handoff.evaluate_kaggle_gpu_handoff_quota(
        artifact_root=tmp_path,
        time_budget_minutes=300,
    )
    assert enough.allowed is True
    assert enough.required_minutes == 900
    assert enough.available_minutes == 1200


def test_handoff_state_is_persisted_for_resume(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    iter_dir = run_dir / "iter-2"

    payload = compute_handoff.begin_handoff(
        run_dir=run_dir,
        iter_dir=iter_dir,
        run_id="run-1",
        iteration=2,
        error_text="CUDA out of memory",
        to_hardware_profile="kaggle_p100",
    )

    resumed = compute_handoff.load_committed_handoff(run_dir)
    assert resumed is None
    assert payload["status"] == "kaggle_gpu_preparing"
    assert payload["destination_committed"] is False
    assert (iter_dir / compute_handoff.HANDOFF_FILENAME).exists()

    running = compute_handoff.finish_handoff(
        run_dir=run_dir,
        iter_dir=iter_dir,
        payload=payload,
        status="kaggle_gpu_running",
        kernel_id="user/demo-run-1-i2",
    )
    resumed = compute_handoff.load_committed_handoff(run_dir)
    assert resumed is not None
    assert resumed["status"] == "kaggle_gpu_running"

    compute_handoff.finish_handoff(
        run_dir=run_dir,
        iter_dir=iter_dir,
        payload=running,
        status="completed",
        kernel_id="user/demo-run-1-i2",
    )

    resumed = compute_handoff.load_committed_handoff(run_dir)
    assert resumed is not None
    assert resumed["status"] == "completed"
    assert resumed["kernel_id"] == "user/demo-run-1-i2"
