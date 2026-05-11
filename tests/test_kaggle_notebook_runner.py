from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagglebot.kernel_sources import KernelSourceConfig
from kagglebot.runners import kaggle_notebook
from kagglebot.runners.kaggle_notebook import _parse_kernel_status, _wait_for_kernel, build_kernel_metadata


def test_build_kernel_metadata_uses_plan_driven_sources() -> None:
    metadata = build_kernel_metadata(
        kaggle_username="user",
        kernel_slug="demo-kernel",
        title="demo kernel",
        competition_slug="demo",
        accelerator="gpu",
        enable_internet=False,
        source_config=KernelSourceConfig(
            dataset_sources=("alice/demo-dataset",),
            kernel_sources=("bob/demo-kernel",),
            model_sources=("carol/demo-model/PyTorch/default/1",),
        ),
    )

    assert metadata["competition_sources"] == ["demo"]
    assert metadata["dataset_sources"] == ["alice/demo-dataset"]
    assert metadata["kernel_sources"] == ["bob/demo-kernel"]
    assert metadata["model_sources"] == ["carol/demo-model/PyTorch/default/1"]


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ('owner/kernel has status "KernelWorkerStatus.RUNNING"', "running"),
        ('owner/kernel has status "KernelWorkerStatus.COMPLETE"', "complete"),
        ('owner/kernel has status "KernelWorkerStatus.ERROR"\nFailure message: "Your notebook failed"', "failed"),
        ('owner/kernel has status "KernelWorkerStatus.FAILED"', "failed"),
    ],
)
def test_parse_kernel_status(output: str, expected: str) -> None:
    assert _parse_kernel_status(output) == expected


def test_wait_for_kernel_pushes_cpu_stop_marker_after_failed_gpu_run(monkeypatch, tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    logs_dir = tmp_path / "logs"
    kernel_dir.mkdir()
    logs_dir.mkdir()
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "owner/kernel",
                "title": "kernel",
                "code_file": "main.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": True,
                "competition_sources": ["demo"],
                "dataset_sources": [],
                "kernel_sources": [],
                "model_sources": [],
                "keywords": [],
            }
        ),
        encoding="utf-8",
    )
    pushed: list[Path] = []
    monkeypatch.setattr(
        kaggle_notebook.kaggle_cli,
        "kernels_status",
        lambda *_args,
        **_kwargs: 'owner/kernel has status "KernelWorkerStatus.ERROR"\nFailure message: "Your notebook failed"',
    )
    monkeypatch.setattr(
        kaggle_notebook.kaggle_cli,
        "kernels_push",
        lambda kernel_path, **_kwargs: pushed.append(Path(kernel_path)) or "pushed",
    )

    with pytest.raises(RuntimeError, match="Kernel run failed"):
        _wait_for_kernel("owner/kernel", logs_dir=logs_dir, slug="demo", kernel_dir=kernel_dir)

    assert pushed == [tmp_path / "kernel-stop"]
    metadata = json.loads((tmp_path / "kernel-stop" / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["enable_gpu"] is False
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert (logs_dir / "kernel_stop.log").exists()
