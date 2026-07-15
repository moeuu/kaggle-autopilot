from __future__ import annotations

import os
import re
import time
from pathlib import Path

from kagglebot.exceptions import KernelCapacityError
from kagglebot.json_utils import load_json_object, write_json_object

PENDING_REMOTE_KERNEL_FILENAME = "remote_kernel_pending.json"
REMOTE_KERNEL_SOURCE_FILENAME = "remote_kernel_source.json"
REMOTE_KERNEL_QUEUED_TIMEOUT_ENV = "KAGGLEBOT_KERNEL_QUEUED_TIMEOUT_SEC"
REMOTE_KERNEL_DEFAULT_QUEUED_TIMEOUT_SEC = 1800.0

_KERNEL_URL_RE = re.compile(
    r"https?://(?:www\.)?kaggle\.com/(?:code/)?(?P<user>[A-Za-z0-9_-]+)/(?P<slug>[A-Za-z0-9_.-]+)"
)
_KERNEL_ID_RE = re.compile(r"(?P<user>[A-Za-z0-9_-]+)/(?P<slug>[A-Za-z0-9_.-]+)")


def pending_remote_kernel_path(logs_dir: Path) -> Path:
    return logs_dir / PENDING_REMOTE_KERNEL_FILENAME


def remote_kernel_source_path(logs_dir: Path) -> Path:
    return logs_dir / REMOTE_KERNEL_SOURCE_FILENAME


def write_remote_kernel_source(
    logs_dir: Path,
    *,
    kernel_id: str,
    source_fingerprint: str,
) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    write_json_object(
        remote_kernel_source_path(logs_dir),
        {
            "kernel_id": kernel_id,
            "source_fingerprint": source_fingerprint,
            "recorded_at_unix": time.time(),
        },
    )


def remote_kernel_source_matches(
    logs_dir: Path,
    *,
    kernel_id: str,
    source_fingerprint: str,
) -> bool:
    payload = load_json_object(remote_kernel_source_path(logs_dir))
    if not isinstance(payload, dict):
        return False
    return (
        str(payload.get("kernel_id") or "").strip() == kernel_id
        and str(payload.get("source_fingerprint") or "").strip() == source_fingerprint
    )


def write_pending_remote_kernel(logs_dir: Path, *, kernel_id: str, kernel_slug: str) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "kernel_id": kernel_id,
        "kernel_slug": kernel_slug,
        "recorded_at_unix": time.time(),
    }
    write_json_object(pending_remote_kernel_path(logs_dir), payload)


def clear_pending_remote_kernel(logs_dir: Path) -> None:
    try:
        pending_remote_kernel_path(logs_dir).unlink()
    except FileNotFoundError:
        return


def read_pending_remote_kernel_id(logs_dir: Path) -> str | None:
    payload = load_json_object(pending_remote_kernel_path(logs_dir))
    kernel_id = payload.get("kernel_id") if isinstance(payload, dict) else None
    if kernel_id is None:
        return None
    return str(kernel_id).strip() or None


def extract_kernel_id_from_push(output: str) -> str | None:
    if not output:
        return None
    match = _KERNEL_URL_RE.search(output)
    if match:
        return f"{match.group('user')}/{match.group('slug')}"
    for line in output.splitlines():
        if "kernel" not in line.lower():
            continue
        match = _KERNEL_ID_RE.search(line)
        if match:
            return f"{match.group('user')}/{match.group('slug')}"
    return None


def last_pushed_kernel_id(logs_dir: Path, default_kernel_id: str) -> str | None:
    for path in sorted(logs_dir.glob("kernel_push-*.txt"), reverse=True):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        pushed_kernel_id = extract_kernel_id_from_push(text)
        if pushed_kernel_id:
            return pushed_kernel_id
        if "successfully pushed" in text.lower():
            return default_kernel_id
    return None


def remote_kernel_queued_timeout_sec() -> float | None:
    raw = os.getenv(REMOTE_KERNEL_QUEUED_TIMEOUT_ENV)
    if raw is None or raw.strip() == "":
        return REMOTE_KERNEL_DEFAULT_QUEUED_TIMEOUT_SEC
    try:
        value = float(raw)
    except ValueError:
        return REMOTE_KERNEL_DEFAULT_QUEUED_TIMEOUT_SEC
    if value <= 0:
        return None
    return value


def raise_kernel_queued_timeout(kernel_id: str, elapsed_sec: float, timeout_sec: float) -> None:
    raise KernelCapacityError(
        f"Kaggle kernel {kernel_id} stayed queued for {int(elapsed_sec)}s "
        f"(queue timeout {int(timeout_sec)}s). Kaggle workers are not starting this run.",
        output=f"KernelWorkerStatus.QUEUED elapsed={int(elapsed_sec)} timeout={int(timeout_sec)}",
    )


def last_kernel_push_wall_time(logs_dir: Path) -> float | None:
    latest: float | None = None
    for path in logs_dir.glob("kernel_push-*.txt"):
        try:
            timestamp = path.stat().st_mtime
        except OSError:
            continue
        if latest is None or timestamp > latest:
            latest = timestamp
    return latest


def queued_since_from_push_logs(logs_dir: Path) -> float | None:
    pushed_at = last_kernel_push_wall_time(logs_dir)
    if pushed_at is None:
        return None
    elapsed = max(0.0, time.time() - pushed_at)
    return time.monotonic() - elapsed


def is_remote_kernel_queue_stale(queued_since: float | None, now: float | None = None) -> bool:
    timeout_sec = remote_kernel_queued_timeout_sec()
    if queued_since is None or timeout_sec is None:
        return False
    current = time.monotonic() if now is None else now
    return current - queued_since >= timeout_sec
