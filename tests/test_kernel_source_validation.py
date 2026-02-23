from __future__ import annotations

from pathlib import Path

from kagglebot.validators import ensure_kernel_sources_valid, validate_kernel_sources


def _write_kernel(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_kernel_source_validation_flags_missing_kernel(tmp_path: Path) -> None:
    issues = validate_kernel_sources(tmp_path / "kernel")
    assert issues


def test_kernel_source_validation_requires_outputs(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(kernel_dir / "kernel.py", "print('hello')\n")
    issues = validate_kernel_sources(kernel_dir)
    assert any("submission.csv" in issue for issue in issues)
    assert any("metrics.json" in issue for issue in issues)


def test_kernel_source_validation_passes_basic(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(
        kernel_dir / "kernel.py",
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/train.csv'",
                "OUT1 = '/kaggle/working/submission.csv'",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
    )
    ensure_kernel_sources_valid(kernel_dir)


def test_kernel_source_validation_accepts_kaggle_input_without_trailing_slash(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(
        kernel_dir / "kernel.py",
        "\n".join(
            [
                "DATA_ROOT = '/kaggle/input'",
                "DATA = DATA_ROOT + '/demo/train.csv'",
                "OUT1 = '/kaggle/working/submission.csv'",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
    )
    ensure_kernel_sources_valid(kernel_dir)


def test_kernel_source_validation_flags_prott5_automodel(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(
        kernel_dir / "kernel.py",
        "from transformers import AutoModel\\n"
        "MODEL = 'Rostlab/prot_t5_xl_uniref50'\\n"
        "m = AutoModel.from_pretrained(MODEL)\\n"
        "DATA = '/kaggle/input/demo/train.csv'\\n"
        "OUT1 = '/kaggle/working/submission.csv'\\n"
        "OUT2 = '/kaggle/working/metrics.json'\\n",
    )
    issues = validate_kernel_sources(kernel_dir)
    assert any("T5/ProtT5" in issue for issue in issues)
