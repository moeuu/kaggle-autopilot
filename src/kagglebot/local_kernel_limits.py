from __future__ import annotations

import psutil

from kagglebot.env_utils import parse_float_value, parse_int_value
from kagglebot.exceptions import KernelFailedError

MEMORY_CAP_ENV = "KAGGLEBOT_LOCAL_KERNEL_MAX_RSS_MB"
STALL_ENV = "KAGGLEBOT_LOCAL_KERNEL_STALL_SEC"
DEFAULT_STALL_SEC = 900.0
MEMORY_CAP_RATIO = 0.80


def resolve_memory_cap_bytes(env: dict[str, str]) -> int | None:
    override_raw = env.get(MEMORY_CAP_ENV)
    if override_raw is not None and str(override_raw).strip():
        override_mb = parse_int_value(override_raw)
        if override_mb is None:
            raise KernelFailedError(f"{MEMORY_CAP_ENV} must be a positive integer number of MiB.")
        if override_mb <= 0:
            raise KernelFailedError(f"{MEMORY_CAP_ENV} must be a positive integer number of MiB.")
        return override_mb * 1024 * 1024

    available_bytes = int(psutil.virtual_memory().available)
    if available_bytes <= 0:
        return None
    return max(512 * 1024 * 1024, int(available_bytes * MEMORY_CAP_RATIO))


def resolve_stall_timeout_sec(env: dict[str, str]) -> float | None:
    raw = str(env.get(STALL_ENV, str(int(DEFAULT_STALL_SEC)))).strip()
    if not raw:
        return DEFAULT_STALL_SEC
    value = parse_float_value(raw)
    if value is None:
        raise KernelFailedError(f"{STALL_ENV} must be a positive number of seconds.")
    if value <= 0:
        return None
    return max(5.0, value)


def process_tree_rss_bytes(pid: int) -> int:
    try:
        root = psutil.Process(pid)
    except psutil.Error:
        return 0

    total = 0
    seen: set[int] = set()
    processes = [root]
    try:
        processes.extend(root.children(recursive=True))
    except psutil.Error:
        pass

    for proc in processes:
        if proc.pid in seen:
            continue
        seen.add(proc.pid)
        try:
            total += int(proc.memory_info().rss)
        except psutil.Error:
            continue
    return total
