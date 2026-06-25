from __future__ import annotations

import os
from pathlib import Path

import pytest

from kagglebot.exceptions import KernelCapacityError
from kagglebot.remote_kernel_state import (
    PENDING_REMOTE_KERNEL_FILENAME,
    clear_pending_remote_kernel,
    extract_kernel_id_from_push,
    is_remote_kernel_queue_stale,
    last_pushed_kernel_id,
    queued_since_from_push_logs,
    raise_kernel_queued_timeout,
    read_pending_remote_kernel_id,
    remote_kernel_queued_timeout_sec,
    write_pending_remote_kernel,
)


def test_read_pending_remote_kernel_id_ignores_invalid_or_non_object_payload(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    path = logs_dir / PENDING_REMOTE_KERNEL_FILENAME

    path.write_text("{", encoding="utf-8")
    assert read_pending_remote_kernel_id(logs_dir) is None

    path.write_text("[]", encoding="utf-8")
    assert read_pending_remote_kernel_id(logs_dir) is None

    write_pending_remote_kernel(logs_dir, kernel_id="user/kernel", kernel_slug="kernel")
    assert read_pending_remote_kernel_id(logs_dir) == "user/kernel"

    clear_pending_remote_kernel(logs_dir)
    assert read_pending_remote_kernel_id(logs_dir) is None


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("Kernel pushed: https://www.kaggle.com/code/user/demo-kernel", "user/demo-kernel"),
        ("Created kernel user/demo_kernel successfully", "user/demo_kernel"),
        ("Kernel version 1 successfully pushed.", None),
        ("", None),
    ],
)
def test_extract_kernel_id_from_push(output: str, expected: str | None) -> None:
    assert extract_kernel_id_from_push(output) == expected


def test_last_pushed_kernel_id_prefers_newest_push_log(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    older = logs_dir / "kernel_push-01.txt"
    newer = logs_dir / "kernel_push-02.txt"
    older.write_text("Kernel pushed: https://www.kaggle.com/code/user/old-kernel\n", encoding="utf-8")
    newer.write_text("Kernel version 2 successfully pushed.\n", encoding="utf-8")
    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))

    assert last_pushed_kernel_id(logs_dir, "user/default-kernel") == "user/default-kernel"


def test_remote_kernel_queue_timeout_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLEBOT_KERNEL_QUEUED_TIMEOUT_SEC", raising=False)
    assert remote_kernel_queued_timeout_sec() == 1800.0

    monkeypatch.setenv("KAGGLEBOT_KERNEL_QUEUED_TIMEOUT_SEC", "30")
    assert remote_kernel_queued_timeout_sec() == 30.0

    monkeypatch.setenv("KAGGLEBOT_KERNEL_QUEUED_TIMEOUT_SEC", "0")
    assert remote_kernel_queued_timeout_sec() is None

    monkeypatch.setenv("KAGGLEBOT_KERNEL_QUEUED_TIMEOUT_SEC", "bad")
    assert remote_kernel_queued_timeout_sec() == 1800.0


def test_queue_stale_and_since_from_push_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    push_log = logs_dir / "kernel_push-01.txt"
    push_log.write_text("Kernel version 1 successfully pushed.\n", encoding="utf-8")
    os.utime(push_log, (100.0, 100.0))

    monkeypatch.setattr("kagglebot.remote_kernel_state.time.time", lambda: 160.0)
    monkeypatch.setattr("kagglebot.remote_kernel_state.time.monotonic", lambda: 1000.0)
    assert queued_since_from_push_logs(logs_dir) == 940.0

    monkeypatch.setenv("KAGGLEBOT_KERNEL_QUEUED_TIMEOUT_SEC", "30")
    assert is_remote_kernel_queue_stale(queued_since=900.0, now=940.0) is True
    assert is_remote_kernel_queue_stale(queued_since=920.0, now=940.0) is False


def test_raise_kernel_queued_timeout() -> None:
    with pytest.raises(KernelCapacityError, match="stayed queued") as exc_info:
        raise_kernel_queued_timeout("user/kernel", elapsed_sec=31.2, timeout_sec=30.0)
    assert "KernelWorkerStatus.QUEUED" in str(exc_info.value.output)
