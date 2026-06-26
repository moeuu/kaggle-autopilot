from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from kagglebot.kernel_sources import KernelSourceConfig
from kagglebot.runners import kaggle_notebook
from kagglebot.runners.kaggle_notebook import (
    KERNEL_TEMPLATE,
    _wait_for_kernel,
    build_kernel_metadata,
    find_submission_file,
)


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


def test_generated_kernel_expands_tiny_public_sample_to_test_ids(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(20),
            "feature": [float(idx) for idx in range(20)],
            "target": [idx % 2 for idx in range(20)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102, 103, 104], "feature": [1.0, 2.0, 3.0, 4.0, 5.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101, 102], "target": [0, 0, 0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    submission = pd.read_csv(working_dir / "submission.csv")
    assert submission["id"].tolist() == [100, 101, 102, 103, 104]
    assert list(submission.columns) == ["id", "target"]


def test_generated_kernel_regression_uses_rmse_without_squared_argument(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")

    input_root = tmp_path / "input"
    data_dir = input_root / "demo"
    working_dir = tmp_path / "working"
    data_dir.mkdir(parents=True)
    working_dir.mkdir()
    pd.DataFrame(
        {
            "id": range(30),
            "feature": [float(idx) for idx in range(30)],
            "target": [float(idx) * 1.5 for idx in range(30)],
        }
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [100, 101, 102], "feature": [1.0, 2.0, 3.0]}).to_csv(
        data_dir / "test.csv",
        index=False,
    )
    pd.DataFrame({"id": [100, 101, 102], "target": [0.0, 0.0, 0.0]}).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )

    script = (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", "demo")
        .replace("__ACCELERATOR__", "none")
        .replace('Path("/kaggle/input")', f'Path("{input_root}")')
        .replace('Path("/kaggle/working")', f'Path("{working_dir}")')
    )
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(script, encoding="utf-8")

    runpy.run_path(str(kernel_path), run_name="__main__")

    metrics = json.loads((working_dir / "metrics.json").read_text(encoding="utf-8"))
    submission = pd.read_csv(working_dir / "submission.csv")
    assert metrics["metric"] == "rmse"
    assert submission["id"].tolist() == [100, 101, 102]


def test_runner_submission_discovery_uses_fold_intermediate_when_final_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    fold1 = output_dir / "submission_model_fold1.csv"
    fold2 = output_dir / "nested" / "submission_model_fold2.csv"
    fold2.parent.mkdir()
    fold1.write_text("id,target\n1,0.1\n", encoding="utf-8")
    fold2.write_text("id,target\n1,0.2\n", encoding="utf-8")

    assert find_submission_file(output_dir) == fold2
