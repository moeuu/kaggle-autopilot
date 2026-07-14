from __future__ import annotations

from collections.abc import Callable

SUBMIT_KERNEL_ACCELERATOR_ENV = "KAGGLEBOT_SUBMIT_KERNEL_ACCELERATOR"
_VALID_ACCELERATORS = {"cpu", "gpu", "tpu"}
_CPU_ALIASES = {"none", "no", "false", "0"}


def resolve_submit_kernel_accelerator(
    requested: str,
    *,
    env_get: Callable[[str], str | None],
) -> str:
    override = env_get(SUBMIT_KERNEL_ACCELERATOR_ENV)
    value = str(override if override is not None else requested or "cpu").strip().lower()
    if value in _CPU_ALIASES:
        return "cpu"
    if value in _VALID_ACCELERATORS:
        return value
    if override is not None:
        raise ValueError(f"{SUBMIT_KERNEL_ACCELERATOR_ENV} must be one of cpu, gpu, or tpu; got {override!r}.")
    return "cpu"
