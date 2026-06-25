from __future__ import annotations

import os
from collections.abc import Callable

HEAVY_DEEP_LEARNING_MODALITIES = frozenset({"image", "video", "audio", "text", "rna_structure"})


def is_local_gpu_compute(compute: object) -> bool:
    return str(compute or "").strip().lower() == "local_gpu"


def is_heavy_deep_learning_modality(modality: object) -> bool:
    return str(modality or "").strip().lower() in HEAVY_DEEP_LEARNING_MODALITIES


def local_gpu_time_budget_limit_min(
    *,
    getenv: Callable[[str], str | None] = os.environ.get,
    minimum_minutes: int = 60,
) -> int | None:
    raw = getenv("KAGGLEBOT_LOCAL_GPU_TIME_BUDGET_MIN")
    if raw is None or not raw.strip():
        return None
    try:
        parsed = int(float(raw.strip()))
    except ValueError:
        return None
    if parsed <= 0:
        return None
    return max(minimum_minutes, parsed)
