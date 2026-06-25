from __future__ import annotations

import os

import pytest

from kagglebot.exceptions import KernelFailedError
from kagglebot.local_kernel_limits import (
    MEMORY_CAP_ENV,
    STALL_ENV,
    resolve_memory_cap_bytes,
    resolve_stall_timeout_sec,
)


def test_local_kernel_memory_cap_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MEMORY_CAP_ENV, "64")
    cap_bytes = resolve_memory_cap_bytes(dict(os.environ))
    assert cap_bytes == 64 * 1024 * 1024


def test_local_kernel_memory_cap_env_rejects_invalid_override() -> None:
    with pytest.raises(KernelFailedError, match="positive integer number of MiB"):
        resolve_memory_cap_bytes({MEMORY_CAP_ENV: "64.5"})


def test_local_kernel_stall_timeout_env_uses_minimum_and_disable() -> None:
    assert resolve_stall_timeout_sec({STALL_ENV: "1"}) == 5.0
    assert resolve_stall_timeout_sec({STALL_ENV: "0"}) is None


def test_local_kernel_stall_timeout_env_rejects_invalid_override() -> None:
    with pytest.raises(KernelFailedError, match="positive number of seconds"):
        resolve_stall_timeout_sec({STALL_ENV: "nan"})
