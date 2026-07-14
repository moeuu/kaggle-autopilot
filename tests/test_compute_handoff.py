from __future__ import annotations

from pathlib import Path

from kagglebot import compute_handoff
from kagglebot.exceptions import KernelFailedError, KernelTimeoutError


def test_should_handoff_only_resource_limited_local_failures() -> None:
    assert compute_handoff.should_handoff_local_failure(KernelTimeoutError("Local kernel timed out"), enabled=True)
    assert compute_handoff.should_handoff_local_failure(
        KernelFailedError("Local kernel exceeded host memory guard at 24 GiB"),
        enabled=True,
    )
    assert compute_handoff.should_handoff_local_failure(
        KernelFailedError("Local kernel execution failed with CUDA OOM, then failed again after disabling LLM."),
        enabled=True,
    )
    assert not compute_handoff.should_handoff_local_failure(
        KernelFailedError("ValueError: submission shape mismatch"),
        enabled=True,
    )
    assert not compute_handoff.should_handoff_local_failure(
        KernelTimeoutError("Local kernel timed out"),
        enabled=False,
    )


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
    assert resumed is not None
    assert resumed["status"] == "kaggle_gpu_running"
    assert resumed["to_compute"] == "kaggle_gpu"
    assert (iter_dir / compute_handoff.HANDOFF_FILENAME).exists()

    compute_handoff.finish_handoff(
        run_dir=run_dir,
        iter_dir=iter_dir,
        payload=payload,
        status="completed",
        kernel_id="user/demo-run-1-i2",
    )

    resumed = compute_handoff.load_committed_handoff(run_dir)
    assert resumed is not None
    assert resumed["status"] == "completed"
    assert resumed["kernel_id"] == "user/demo-run-1-i2"
