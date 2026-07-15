from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

SUBMIT_KERNEL_ACCELERATOR_ENV = "KAGGLEBOT_SUBMIT_KERNEL_ACCELERATOR"
SUBMIT_KERNEL_MACHINE_SHAPE_ENV = "KAGGLEBOT_SUBMIT_KERNEL_MACHINE_SHAPE"
ARC_AGI_3_COMPETITION_SLUG = "arc-prize-2026-arc-agi-3"
ARC_AGI_3_RTX_MACHINE_SHAPE = "NvidiaRtxPro6000"
_VALID_ACCELERATORS = {"cpu", "gpu", "tpu"}
_CPU_ALIASES = {"none", "no", "false", "0"}
_VALID_MACHINE_SHAPES = {
    "NvidiaH100",
    "NvidiaL4",
    "NvidiaL4X1",
    "NvidiaRtxPro6000",
    "NvidiaTeslaA100",
    "NvidiaTeslaP100",
    "NvidiaTeslaT4",
    "NvidiaTeslaT4Highmem",
}
_CANONICAL_MACHINE_SHAPE_BY_CASEFOLD = {shape.casefold(): shape for shape in _VALID_MACHINE_SHAPES}
_MACHINE_SHAPE_DISABLED_ALIASES = {"default", "none", "off", "false", "0"}
_HARDWARE_PROFILE_MACHINE_SHAPES = {
    "kaggle_rtx_pro_6000": "NvidiaRtxPro6000",
    "rtx_pro_6000": "NvidiaRtxPro6000",
    "rtxpro6000": "NvidiaRtxPro6000",
    "kaggle_p100": "NvidiaTeslaP100",
    "p100": "NvidiaTeslaP100",
    "kaggle_t4": "NvidiaTeslaT4",
    "kaggle_t4x2": "NvidiaTeslaT4",
    "t4": "NvidiaTeslaT4",
    "t4x2": "NvidiaTeslaT4",
}
_MACHINE_SHAPE_HARDWARE_PROFILES = {
    "NvidiaRtxPro6000": "kaggle_rtx_pro_6000",
    "NvidiaTeslaP100": "kaggle_p100",
    "NvidiaTeslaT4": "kaggle_t4",
    "NvidiaTeslaT4Highmem": "kaggle_t4",
}
_DEFAULT_GPU_MACHINE_SHAPE = "NvidiaTeslaT4"


@dataclass(frozen=True)
class SubmitKernelMachineShapeDecision:
    machine_shape: str | None
    source: str


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


def resolve_submit_kernel_machine_shape(
    *,
    env_get: Callable[[str], str | None],
    accelerator: str = "gpu",
    hardware_profile: str | None = None,
    plan: Mapping[str, object] | None = None,
    competition_slug: str | None = None,
) -> str | None:
    return resolve_submit_kernel_machine_shape_decision(
        env_get=env_get,
        accelerator=accelerator,
        hardware_profile=hardware_profile,
        plan=plan,
        competition_slug=competition_slug,
    ).machine_shape


def resolve_submit_kernel_machine_shape_decision(
    *,
    env_get: Callable[[str], str | None],
    accelerator: str = "gpu",
    hardware_profile: str | None = None,
    plan: Mapping[str, object] | None = None,
    competition_slug: str | None = None,
) -> SubmitKernelMachineShapeDecision:
    """Resolve a Kaggle CLI accelerator ID, never notebook metadata casing."""
    accelerator_value = str(accelerator or "cpu").strip().lower()
    if accelerator_value != "gpu":
        return SubmitKernelMachineShapeDecision(machine_shape=None, source=f"accelerator:{accelerator_value}")

    override = str(env_get(SUBMIT_KERNEL_MACHINE_SHAPE_ENV) or "").strip()
    if override and override.casefold() != "auto":
        if override.casefold() in _MACHINE_SHAPE_DISABLED_ALIASES:
            return SubmitKernelMachineShapeDecision(machine_shape=None, source=f"env:{SUBMIT_KERNEL_MACHINE_SHAPE_ENV}")
        return SubmitKernelMachineShapeDecision(
            machine_shape=_canonical_machine_shape(override, source=SUBMIT_KERNEL_MACHINE_SHAPE_ENV),
            source=f"env:{SUBMIT_KERNEL_MACHINE_SHAPE_ENV}",
        )

    plan_shape = _plan_machine_shape(plan)
    if plan_shape and plan_shape.casefold() != "auto":
        if plan_shape.casefold() in _MACHINE_SHAPE_DISABLED_ALIASES:
            return SubmitKernelMachineShapeDecision(machine_shape=None, source="plan:submit_machine_shape")
        return SubmitKernelMachineShapeDecision(
            machine_shape=_canonical_machine_shape(plan_shape, source="plan submit_machine_shape"),
            source="plan:submit_machine_shape",
        )

    resolved_profile = _resolve_hardware_profile(hardware_profile, plan)
    profile_shape = _HARDWARE_PROFILE_MACHINE_SHAPES.get(resolved_profile)
    if profile_shape is not None:
        return SubmitKernelMachineShapeDecision(
            machine_shape=profile_shape,
            source=f"hardware_profile:{resolved_profile}",
        )
    if _normalized_competition_slug(competition_slug) == ARC_AGI_3_COMPETITION_SLUG:
        return SubmitKernelMachineShapeDecision(
            machine_shape=ARC_AGI_3_RTX_MACHINE_SHAPE,
            source=f"competition_policy:{ARC_AGI_3_COMPETITION_SLUG}",
        )
    return SubmitKernelMachineShapeDecision(machine_shape=_DEFAULT_GPU_MACHINE_SHAPE, source="gpu_default")


def submit_hardware_profile_for_machine_shape(machine_shape: str | None) -> str | None:
    if machine_shape is None:
        return None
    return _MACHINE_SHAPE_HARDWARE_PROFILES.get(machine_shape)


def machine_shape_requires_offline(
    machine_shape: str | None,
    *,
    competition_slug: str | None = None,
) -> bool:
    """Return the competition-specific offline requirement for an accelerator.

    Kaggle exposes ``NvidiaRtxPro6000`` as a machine ID and may grant it to
    more than one competition.  The no-internet rule captured here belongs to
    ARC-AGI-3's reserved RTX pool, not to every RTX PRO 6000 session globally.
    """
    return (
        machine_shape == ARC_AGI_3_RTX_MACHINE_SHAPE
        and _normalized_competition_slug(competition_slug) == ARC_AGI_3_COMPETITION_SLUG
    )


def _normalized_competition_slug(value: str | None) -> str:
    return str(value or "").strip().lower()


def _canonical_machine_shape(value: str, *, source: str) -> str:
    canonical = _CANONICAL_MACHINE_SHAPE_BY_CASEFOLD.get(str(value).strip().casefold())
    if canonical is not None:
        return canonical
    allowed = ", ".join(sorted(_VALID_MACHINE_SHAPES))
    raise ValueError(f"{source} must be one of {allowed}; got {value!r}.")


def _plan_machine_shape(plan: Mapping[str, object] | None) -> str:
    if not isinstance(plan, Mapping):
        return ""
    for key in ("submit_machine_shape", "kaggle_submit_machine_shape"):
        value = str(plan.get(key) or "").strip()
        if value:
            return value
    runtime_budget = plan.get("runtime_budget")
    if isinstance(runtime_budget, Mapping):
        for key in ("submit_machine_shape", "kaggle_submit_machine_shape"):
            value = str(runtime_budget.get(key) or "").strip()
            if value:
                return value
    return ""


def _resolve_hardware_profile(hardware_profile: str | None, plan: Mapping[str, object] | None) -> str:
    explicit = str(hardware_profile or "").strip().lower()
    if explicit and explicit != "auto":
        return explicit
    if not isinstance(plan, Mapping):
        return ""
    for value in (
        plan.get("submit_hardware_profile"),
        plan.get("hardware_profile"),
    ):
        normalized = str(value or "").strip().lower()
        if normalized and normalized != "auto":
            return normalized
    runtime_budget = plan.get("runtime_budget")
    if isinstance(runtime_budget, Mapping):
        normalized = str(runtime_budget.get("hardware_profile") or "").strip().lower()
        if normalized and normalized != "auto":
            return normalized
    return ""
