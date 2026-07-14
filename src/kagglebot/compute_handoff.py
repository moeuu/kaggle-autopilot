from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.env_utils import env_flag, env_int, env_optional_int
from kagglebot.exceptions import KernelTimeoutError
from kagglebot.json_utils import load_json_object, write_json_object
from kagglebot.kaggle_gpu_quota import (
    KaggleGpuQuotaStatus,
    parse_kaggle_gpu_quota_text,
    read_kaggle_gpu_quota_file,
)

HANDOFF_FILENAME = "compute_handoff.json"
_DEFAULT_KAGGLE_GPU_PROFILE = "kaggle_p100"
_DEFAULT_MIN_RESOURCE_FAILURES = 3
_DEFAULT_MIN_AVAILABLE_GPU_MINUTES = 15 * 60
_RESOURCE_FAILURE_MARKERS = (
    ("no_local_gpu", ("no local gpu detected",)),
    (
        "cuda_oom",
        (
            "cuda out of memory",
            "cuda oom",
            "out of memory, then failed again",
            "stalled after cuda oom retry",
        ),
    ),
    (
        "host_memory",
        (
            "exceeded host memory guard",
            "host memory guard",
            "peak rss=",
            "cannot allocate memory",
            "exit code 137",
            "exit code -9",
        ),
    ),
    ("stalled", ("local kernel stalled",)),
)


@dataclass(frozen=True)
class HandoffQuotaDecision:
    allowed: bool
    reason: str
    required_minutes: int
    available_minutes: int | None = None
    source: str | None = None


def local_to_kaggle_gpu_enabled() -> bool:
    return env_flag("KAGGLEBOT_LOCAL_TO_KAGGLE_GPU", default=True)


def kaggle_gpu_handoff_profile() -> str:
    return os.environ.get("KAGGLEBOT_KAGGLE_GPU_HANDOFF_PROFILE", _DEFAULT_KAGGLE_GPU_PROFILE).strip() or (
        _DEFAULT_KAGGLE_GPU_PROFILE
    )


def local_resource_failure_kind(error: BaseException) -> str | None:
    if isinstance(error, KernelTimeoutError):
        return "timeout"
    lowered = str(error).lower()
    for kind, markers in _RESOURCE_FAILURE_MARKERS:
        if any(marker in lowered for marker in markers):
            return kind
    return None


def should_handoff_local_failure(
    error: BaseException,
    *,
    consecutive_failures: int = 1,
    enabled: bool | None = None,
) -> bool:
    if enabled is None:
        enabled = local_to_kaggle_gpu_enabled()
    if not enabled:
        return False
    kind = local_resource_failure_kind(error)
    if kind is None:
        return False
    required_failures = (
        1
        if kind == "no_local_gpu"
        else env_int(
            "KAGGLEBOT_LOCAL_TO_KAGGLE_GPU_MIN_RESOURCE_FAILURES",
            default=_DEFAULT_MIN_RESOURCE_FAILURES,
            min_value=1,
        )
    )
    return consecutive_failures >= required_failures


def evaluate_kaggle_gpu_handoff_quota(
    *,
    artifact_root: Path,
    time_budget_minutes: int | None,
) -> HandoffQuotaDecision:
    configured_minimum = env_int(
        "KAGGLEBOT_KAGGLE_GPU_HANDOFF_MIN_AVAILABLE_MINUTES",
        default=_DEFAULT_MIN_AVAILABLE_GPU_MINUTES,
        min_value=1,
    )
    required_minutes = max(configured_minimum, int(time_budget_minutes or 0))
    quota = _resolve_kaggle_gpu_quota(artifact_root)
    if quota is None or quota.available_minutes is None:
        return HandoffQuotaDecision(
            allowed=False,
            reason="quota_unavailable",
            required_minutes=required_minutes,
        )
    if quota.available_minutes < required_minutes:
        return HandoffQuotaDecision(
            allowed=False,
            reason="quota_low",
            required_minutes=required_minutes,
            available_minutes=quota.available_minutes,
            source=quota.source,
        )
    return HandoffQuotaDecision(
        allowed=True,
        reason="quota_sufficient",
        required_minutes=required_minutes,
        available_minutes=quota.available_minutes,
        source=quota.source,
    )


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
        "status": "kaggle_gpu_preparing",
        "destination_committed": False,
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
    if status in {"kaggle_gpu_running", "completed"}:
        updated["destination_committed"] = True
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


def _resolve_kaggle_gpu_quota(artifact_root: Path) -> KaggleGpuQuotaStatus | None:
    explicit_minutes = env_optional_int("KAGGLEBOT_KAGGLE_GPU_AVAILABLE_MINUTES", allow_float=True)
    if explicit_minutes is not None:
        return KaggleGpuQuotaStatus(
            available_minutes=explicit_minutes,
            total_minutes=env_optional_int("KAGGLEBOT_KAGGLE_GPU_TOTAL_MINUTES", allow_float=True),
            source="env:KAGGLEBOT_KAGGLE_GPU_AVAILABLE_MINUTES",
        )

    for name in ("KAGGLEBOT_KAGGLE_GPU_QUOTA_TEXT", "KAGGLE_GPU_QUOTA_TEXT"):
        quota = parse_kaggle_gpu_quota_text(os.environ.get(name), source=f"env:{name}")
        if quota is not None:
            return quota

    candidates: list[Path] = []
    if configured_path := os.environ.get("KAGGLEBOT_KAGGLE_GPU_QUOTA_FILE"):
        candidates.append(Path(configured_path).expanduser())
    candidates.extend(
        (
            artifact_root / "_watch" / "kaggle_gpu" / "quota.json",
            artifact_root / "_watch" / "kaggle_gpu_quota.json",
        )
    )
    for path in dict.fromkeys(candidates):
        quota = read_kaggle_gpu_quota_file(path)
        if quota is not None:
            return quota
    return None
