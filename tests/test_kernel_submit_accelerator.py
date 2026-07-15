from __future__ import annotations

import pytest

from kagglebot.kernel_submit_accelerator import (
    ARC_AGI_3_COMPETITION_SLUG,
    ARC_AGI_3_RTX_MACHINE_SHAPE,
    SUBMIT_KERNEL_ACCELERATOR_ENV,
    SUBMIT_KERNEL_MACHINE_SHAPE_ENV,
    machine_shape_requires_offline,
    resolve_submit_kernel_accelerator,
    resolve_submit_kernel_machine_shape,
    resolve_submit_kernel_machine_shape_decision,
    submit_hardware_profile_for_machine_shape,
)


def test_resolve_submit_kernel_accelerator_honors_requested_gpu_without_override() -> None:
    assert resolve_submit_kernel_accelerator("gpu", env_get=lambda _name: None) == "gpu"


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


def test_resolve_submit_kernel_machine_shape_defaults_gpu_submissions_to_official_t4_id() -> None:
    decision = resolve_submit_kernel_machine_shape_decision(env_get=lambda _name: None)

    assert decision.machine_shape == "NvidiaTeslaT4"
    assert decision.source == "gpu_default"
    assert (
        resolve_submit_kernel_machine_shape(
            env_get=lambda name: "NvidiaTeslaT4" if name == SUBMIT_KERNEL_MACHINE_SHAPE_ENV else None
        )
        == "NvidiaTeslaT4"
    )


def test_resolve_submit_kernel_machine_shape_canonicalizes_notebook_metadata_casing() -> None:
    assert (
        resolve_submit_kernel_machine_shape(
            env_get=lambda name: "nvidiaTeslaT4" if name == SUBMIT_KERNEL_MACHINE_SHAPE_ENV else None
        )
        == "NvidiaTeslaT4"
    )


def test_resolve_submit_kernel_machine_shape_uses_plan_and_hardware_profile() -> None:
    assert (
        resolve_submit_kernel_machine_shape(
            env_get=lambda _name: None,
            plan={"runtime_budget": {"submit_machine_shape": "NvidiaTeslaA100"}},
        )
        == "NvidiaTeslaA100"
    )
    assert (
        resolve_submit_kernel_machine_shape(
            env_get=lambda _name: None,
            hardware_profile="kaggle_p100",
        )
        == "NvidiaTeslaP100"
    )


def test_resolve_submit_kernel_machine_shape_omits_gpu_shape_for_cpu() -> None:
    assert resolve_submit_kernel_machine_shape(env_get=lambda _name: None, accelerator="cpu") is None
    assert (
        resolve_submit_kernel_machine_shape(
            env_get=lambda name: "NvidiaTeslaT4" if name == SUBMIT_KERNEL_MACHINE_SHAPE_ENV else None,
            accelerator="cpu",
        )
        is None
    )
    assert submit_hardware_profile_for_machine_shape("NvidiaTeslaT4") == "kaggle_t4"
    assert submit_hardware_profile_for_machine_shape(None) is None


def test_resolve_submit_kernel_machine_shape_rejects_unknown_shape() -> None:
    with pytest.raises(ValueError, match=SUBMIT_KERNEL_MACHINE_SHAPE_ENV):
        resolve_submit_kernel_machine_shape(env_get=lambda _name: "gpu-mystery")


def test_arc_agi_3_defaults_to_competition_rtx_and_requires_offline() -> None:
    decision = resolve_submit_kernel_machine_shape_decision(
        env_get=lambda _name: None,
        competition_slug=ARC_AGI_3_COMPETITION_SLUG,
        plan={"runtime_budget": {"hardware_profile": "rtx3060"}},
    )

    assert decision.machine_shape == ARC_AGI_3_RTX_MACHINE_SHAPE
    assert decision.source == f"competition_policy:{ARC_AGI_3_COMPETITION_SLUG}"
    assert (
        machine_shape_requires_offline(
            decision.machine_shape,
            competition_slug=ARC_AGI_3_COMPETITION_SLUG,
        )
        is True
    )


def test_rtx_pro_6000_can_be_explicitly_selected_outside_arc_agi_3() -> None:
    machine_shape = resolve_submit_kernel_machine_shape(
        env_get=lambda name: ARC_AGI_3_RTX_MACHINE_SHAPE if name == SUBMIT_KERNEL_MACHINE_SHAPE_ENV else None,
        competition_slug="another-rtx-enabled-competition",
    )

    assert machine_shape == ARC_AGI_3_RTX_MACHINE_SHAPE
    assert machine_shape_requires_offline(machine_shape, competition_slug="another-rtx-enabled-competition") is False


def test_arc_agi_3_allows_explicit_non_restricted_gpu_override() -> None:
    decision = resolve_submit_kernel_machine_shape_decision(
        env_get=lambda name: "NvidiaTeslaT4" if name == SUBMIT_KERNEL_MACHINE_SHAPE_ENV else None,
        competition_slug=ARC_AGI_3_COMPETITION_SLUG,
    )

    assert decision.machine_shape == "NvidiaTeslaT4"
