from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.env_utils import env_flag
from kagglebot.exceptions import KernelTimeoutError
from kagglebot.json_utils import load_json_object, write_json_object

HANDOFF_FILENAME = "compute_handoff.json"
_DEFAULT_KAGGLE_GPU_PROFILE = "kaggle_p100"
_RESOURCE_FAILURE_MARKERS = (
    "cuda out of memory",
    "cuda oom",
    "out of memory, then failed again",
    "exceeded host memory guard",
    "host memory guard",
    "peak rss=",
    "local kernel stalled",
    "stalled after cuda oom retry",
    "cannot allocate memory",
    "no local gpu detected",
    "exit code 137",
    "exit code -9",
)


def local_to_kaggle_gpu_enabled() -> bool:
    return env_flag("KAGGLEBOT_LOCAL_TO_KAGGLE_GPU", default=True)


def kaggle_gpu_handoff_profile() -> str:
    return os.environ.get("KAGGLEBOT_KAGGLE_GPU_HANDOFF_PROFILE", _DEFAULT_KAGGLE_GPU_PROFILE).strip() or (
        _DEFAULT_KAGGLE_GPU_PROFILE
    )


def should_handoff_local_failure(error: BaseException, *, enabled: bool | None = None) -> bool:
    if enabled is None:
        enabled = local_to_kaggle_gpu_enabled()
    if not enabled:
        return False
    if isinstance(error, KernelTimeoutError):
        return True
    lowered = str(error).lower()
    return any(marker in lowered for marker in _RESOURCE_FAILURE_MARKERS)


def load_committed_handoff(run_dir: Path) -> dict[str, object] | None:
    payload = load_json_object(run_dir / HANDOFF_FILENAME)
    if payload is None or payload.get("destination_committed") is not True:
        return None
    if payload.get("to_compute") != "kaggle_gpu":
        return None
    return payload


def begin_handoff(
    *,
    run_dir: Path,
    iter_dir: Path,
    run_id: str,
    iteration: int,
    error_text: str,
    to_hardware_profile: str,
) -> dict[str, object]:
    now = datetime.now(UTC).isoformat()
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "iteration": iteration,
        "status": "kaggle_gpu_running",
        "destination_committed": True,
        "from_compute": "local_gpu",
        "to_compute": "kaggle_gpu",
        "to_accelerator": "gpu",
        "to_hardware_profile": to_hardware_profile,
        "reason": "local_resource_limit",
        "local_error": error_text[-16000:],
        "started_at": now,
        "updated_at": now,
    }
    _write_handoff(run_dir=run_dir, iter_dir=iter_dir, payload=payload)
    return payload


def finish_handoff(
    *,
    run_dir: Path,
    iter_dir: Path,
    payload: dict[str, object],
    status: str,
    kernel_id: str | None = None,
    error_text: str | None = None,
) -> dict[str, object]:
    updated = dict(payload)
    updated["status"] = status
    updated["updated_at"] = datetime.now(UTC).isoformat()
    if kernel_id:
        updated["kernel_id"] = kernel_id
    if error_text:
        updated["kaggle_error"] = error_text[-16000:]
    _write_handoff(run_dir=run_dir, iter_dir=iter_dir, payload=updated)
    return updated


def _write_handoff(*, run_dir: Path, iter_dir: Path, payload: dict[str, object]) -> None:
    for path in (run_dir / HANDOFF_FILENAME, iter_dir / HANDOFF_FILENAME):
        temporary = path.with_name(f".{path.name}.tmp")
        write_json_object(temporary, payload, sort_keys=True)
        temporary.replace(path)
