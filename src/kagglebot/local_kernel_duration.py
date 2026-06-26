from __future__ import annotations

import time
from pathlib import Path

from kagglebot.json_utils import append_jsonl_record, load_jsonl_records

LOCAL_KERNEL_DURATION_HISTORY_LIMIT = 20


def local_kernel_history_path(*, base_dir: Path, slug: str) -> Path:
    return base_dir / slug / "context" / "local_kernel_duration_history.jsonl"


def estimate_local_kernel_duration_seconds(*, base_dir: Path, slug: str) -> tuple[float | None, int]:
    path = local_kernel_history_path(base_dir=base_dir, slug=slug)
    if not path.exists():
        return None, 0
    durations: list[float] = []
    for payload in load_jsonl_records(path, errors="ignore", reverse=True):
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
) -> None:
    path = local_kernel_history_path(base_dir=base_dir, slug=slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "iteration": int(iteration),
        "duration_sec": float(duration_sec),
        "recorded_at": int(time.time()),
    }
    append_jsonl_record(path, payload, ensure_ascii=True)
