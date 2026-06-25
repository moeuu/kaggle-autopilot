from __future__ import annotations

import pytest

from kagglebot.kernel_status import (
    is_kernel_status_complete,
    is_kernel_status_failed,
    is_kernel_status_queued,
    is_kernel_status_running,
    parse_kernel_status,
)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ('owner/kernel has status "KernelWorkerStatus.RUNNING"', "running"),
        ('owner/kernel has status "KernelWorkerStatus.QUEUED"', "queued"),
        ('owner/kernel has status "KernelWorkerStatus.PENDING"', "queued"),
        ('owner/kernel has status "KernelWorkerStatus.COMPLETE"', "complete"),
        ('owner/kernel has status "KernelWorkerStatus.ERROR"\nFailure message: "Your notebook failed"', "failed"),
        ('owner/kernel has status "KernelWorkerStatus.FAILED"', "failed"),
        ("Kernel version 1 successfully pushed.", "complete"),
        ("unrecognized output", "unknown"),
    ],
)
def test_parse_kernel_status(output: str, expected: str) -> None:
    assert parse_kernel_status(output) == expected


def test_kernel_status_predicates() -> None:
    assert is_kernel_status_running("running") is True
    assert is_kernel_status_running("queued") is True
    assert is_kernel_status_running("complete") is False
    assert is_kernel_status_queued("queued") is True
    assert is_kernel_status_complete("complete") is True
    assert is_kernel_status_failed("failed") is True
