from __future__ import annotations

import time
from pathlib import Path

from kagglebot.json_utils import append_jsonl_record, load_jsonl_records

LOCAL_KERNEL_DURATION_HISTORY_LIMIT = 20
LOCAL_KERNEL_PREFLIGHT_MIN_SAMPLES = 2


def local_kernel_history_path(*, base_dir: Path, slug: str) -> Path:
    return base_dir / slug / "context" / "local_kernel_duration_history.jsonl"


def estimate_local_kernel_duration_seconds(
    *,
    base_dir: Path,
    slug: str,
    kernel_fingerprint: str | None = None,
) -> tuple[float | None, int]:
    path = local_kernel_history_path(base_dir=base_dir, slug=slug)
    if not path.exists():
        return None, 0
    durations: list[float] = []
    for payload in load_jsonl_records(path, errors="ignore", reverse=True):
        if kernel_fingerprint is not None and payload.get("kernel_fingerprint") != kernel_fingerprint:
            continue
        if payload.get("outcome") not in (None, "completed"):
            continue
        value = payload.get("duration_sec")
        if isinstance(value, (int, float)) and value > 0:
            durations.append(float(value))
        if len(durations) >= LOCAL_KERNEL_DURATION_HISTORY_LIMIT:
            break
    if not durations:
        return None, 0
    durations_sorted = sorted(durations)
    mid = len(durations_sorted) // 2
    if len(durations_sorted) % 2 == 1:
        median = durations_sorted[mid]
    else:
        median = (durations_sorted[mid - 1] + durations_sorted[mid]) / 2.0
    return median, len(durations_sorted)


def append_local_kernel_duration_history(
    *,
    base_dir: Path,
    slug: str,
    run_id: str,
    iteration: int,
    duration_sec: float,
    kernel_fingerprint: str | None = None,
    outcome: str = "completed",
) -> None:
    path = local_kernel_history_path(base_dir=base_dir, slug=slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "iteration": int(iteration),
        "duration_sec": float(duration_sec),
        "outcome": outcome,
        "recorded_at": int(time.time()),
    }
    if kernel_fingerprint is not None:
        payload["kernel_fingerprint"] = kernel_fingerprint
    append_jsonl_record(path, payload, ensure_ascii=True)


def exact_source_exceeds_timeout(
    *,
    estimated_duration_sec: float | None,
    sample_count: int,
    timeout_sec: int | None,
    timeout_count: int = 0,
) -> bool:
    return bool(
        timeout_sec is not None
        and (
            timeout_count >= LOCAL_KERNEL_PREFLIGHT_MIN_SAMPLES
            or (
                sample_count >= LOCAL_KERNEL_PREFLIGHT_MIN_SAMPLES
                and estimated_duration_sec is not None
                and estimated_duration_sec > timeout_sec
            )
        )
    )


def exact_source_timeout_count(*, base_dir: Path, slug: str, kernel_fingerprint: str) -> int:
    path = local_kernel_history_path(base_dir=base_dir, slug=slug)
    if not path.exists():
        return 0
    count = 0
    for payload in load_jsonl_records(path, errors="ignore", reverse=True):
        if payload.get("kernel_fingerprint") != kernel_fingerprint:
            continue
        if payload.get("outcome") == "timeout":
            count += 1
        if count >= LOCAL_KERNEL_PREFLIGHT_MIN_SAMPLES:
            break
    return count
