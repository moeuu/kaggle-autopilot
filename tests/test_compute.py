"""Tests for compute mapping and GPU detection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kagglebot.compute import Compute, compute_to_runner_and_accelerator, detect_local_gpu


@pytest.mark.parametrize(
    "compute,runner,accelerator",
    [
        (Compute.local_cpu, "local", "none"),
        (Compute.local_gpu, "local", "gpu"),
        (Compute.kaggle_gpu, "kaggle_notebook", "gpu"),
        (Compute.kaggle_tpu, "kaggle_notebook", "tpu"),
    ],
)
def test_compute_mapping(compute: Compute, runner: str, accelerator: str) -> None:
    selection = compute_to_runner_and_accelerator(compute)
    assert selection.runner == runner
    assert selection.accelerator == accelerator


def test_detect_local_gpu_with_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setattr("kagglebot.compute.torch", fake_torch)
    monkeypatch.setattr("kagglebot.compute.run_command", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    availability = detect_local_gpu()
    assert availability.cuda is True
    assert availability.mps is False


def test_detect_local_gpu_with_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setattr("kagglebot.compute.torch", fake_torch)
    monkeypatch.setattr("kagglebot.compute.run_command", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    availability = detect_local_gpu()
    assert availability.cuda is False
    assert availability.mps is True
