from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from kagglebot.exec_utils import run_command


class Compute(str, Enum):
    local_gpu = "local_gpu"
    kaggle_gpu = "kaggle_gpu"
    kaggle_tpu = "kaggle_tpu"


@dataclass(frozen=True)
class RunnerSelection:
    runner: str
    accelerator: str


def compute_to_runner_and_accelerator(compute: Compute) -> RunnerSelection:
    if compute == Compute.local_gpu:
        return RunnerSelection(runner="local_kernel", accelerator="gpu")
    if compute == Compute.kaggle_gpu:
        return RunnerSelection(runner="kaggle_notebook", accelerator="gpu")
    if compute == Compute.kaggle_tpu:
        return RunnerSelection(runner="kaggle_notebook", accelerator="tpu")
    raise ValueError(f"Unknown compute option: {compute}")


@dataclass(frozen=True)
class GpuAvailability:
    cuda: bool
    mps: bool

    @property
    def any(self) -> bool:
        return self.cuda or self.mps


try:  # pragma: no cover - exercised via monkeypatch in tests
    import torch
except Exception:  # noqa: BLE001
    torch = None


def detect_local_gpu() -> GpuAvailability:
    cuda = False
    mps = False
    if torch is not None:
        cuda = bool(torch.cuda.is_available())
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if not cuda:
        try:
            result = run_command(["nvidia-smi"])
            cuda = result.returncode == 0
        except FileNotFoundError:
            cuda = False
    return GpuAvailability(cuda=cuda, mps=mps)
