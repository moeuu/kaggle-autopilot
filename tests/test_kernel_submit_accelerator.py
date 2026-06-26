from __future__ import annotations

import pytest

from kagglebot.kernel_submit_accelerator import (
    SUBMIT_KERNEL_ACCELERATOR_ENV,
    resolve_submit_kernel_accelerator,
)


def test_resolve_submit_kernel_accelerator_defaults_to_cpu_without_override() -> None:
    assert resolve_submit_kernel_accelerator("gpu", env_get=lambda _name: None) == "cpu"


def test_resolve_submit_kernel_accelerator_honors_valid_override() -> None:
    assert (
        resolve_submit_kernel_accelerator(
            "cpu",
            env_get=lambda name: "gpu" if name == SUBMIT_KERNEL_ACCELERATOR_ENV else None,
        )
        == "gpu"
    )


def test_resolve_submit_kernel_accelerator_maps_falsey_override_to_cpu() -> None:
    assert (
        resolve_submit_kernel_accelerator(
            "gpu",
            env_get=lambda _name: "false",
        )
        == "cpu"
    )


def test_resolve_submit_kernel_accelerator_rejects_invalid_override() -> None:
    with pytest.raises(ValueError, match=SUBMIT_KERNEL_ACCELERATOR_ENV):
        resolve_submit_kernel_accelerator("gpu", env_get=lambda _name: "invalid")
