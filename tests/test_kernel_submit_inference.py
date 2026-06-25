from __future__ import annotations

import pytest

from kagglebot.exceptions import KernelFailedError
from kagglebot.kernel_submit_inference import (
    sanitize_submit_inference_output_roots,
    validate_inference_submit_kernel,
)


def test_sanitize_submit_inference_output_roots_rewrites_staged_outputs(tmp_path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "KERNEL_DIR = Path('/kaggle/src')",
                "OUT = KERNEL_DIR / 'outputs'",
                "MET = KERNEL_DIR.joinpath('output')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    sanitize_submit_inference_output_roots(kernel_dir)

    text = kernel_path.read_text(encoding="utf-8")
    assert "Path('/kaggle/working')" in text
    assert "KERNEL_DIR / 'outputs'" not in text
    assert "KERNEL_DIR.joinpath('output')" not in text


def test_validate_inference_submit_kernel_rejects_missing_working_output(tmp_path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "kernel.py").write_text("print('no output path')\n", encoding="utf-8")

    with pytest.raises(KernelFailedError, match="/kaggle/working"):
        validate_inference_submit_kernel(kernel_dir)
