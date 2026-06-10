from __future__ import annotations

import subprocess

import pytest

from kagglebot.hardware import hardware_env, render_hardware_constraints, resolve_hardware_profile


def test_resolve_explicit_rtx3060_profile() -> None:
    profile = resolve_hardware_profile("rtx3060", compute="local_gpu")

    assert profile.key == "rtx3060"
    assert profile.vram_gb == 12
    assert hardware_env(profile)["KAGGLEBOT_GPU_VRAM_GB"] == "12"
    assert hardware_env(profile)["KAGGLEBOT_ACCURACY_FIRST"] == "1"
    rendered = render_hardware_constraints(profile, compute="local_gpu")
    assert "Planning time budget: unlimited per kernel iteration" in rendered
    assert "RTX3060-class accuracy-first rule" in rendered


def test_resolve_explicit_rtx5090_profile() -> None:
    profile = resolve_hardware_profile("rtx5090", compute="local_gpu")

    assert profile.key == "rtx5090"
    assert profile.vram_gb == 32


def test_resolve_explicit_kaggle_profile_with_underscore() -> None:
    profile = resolve_hardware_profile("kaggle_p100", compute="kaggle_gpu")

    assert profile.key == "kaggle_p100"
    assert profile.vram_gb == 16


def test_resolve_unknown_profile_rejects() -> None:
    with pytest.raises(ValueError, match="Unknown hardware profile"):
        resolve_hardware_profile("mystery_gpu", compute="local_gpu")


def test_auto_detects_rtx3060_from_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    class Completed:
        returncode = 0
        stdout = "NVIDIA GeForce RTX 3060, 12288\n"

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: Completed())  # noqa: ARG005

    profile = resolve_hardware_profile("auto", compute="local_gpu")

    assert profile.key == "rtx3060"
