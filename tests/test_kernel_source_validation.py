from __future__ import annotations

from pathlib import Path

import pytest

from kagglebot.validators import ensure_kernel_sources_valid, kernel_source_preflight_error, validate_kernel_sources


def _write_kernel(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_kernel_source_validation_flags_missing_kernel(tmp_path: Path) -> None:
    issues = validate_kernel_sources(tmp_path / "kernel")
    assert issues


def test_kernel_source_preflight_error_reports_missing_kernel(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"

    error = kernel_source_preflight_error(kernel_dir)

    assert error is not None
    assert "requires kernel.py" in error
    assert str(kernel_dir / "kernel.py") in error


def test_kernel_source_preflight_error_uses_formatter(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(kernel_dir / "kernel.py", "OUT1 = '/kaggle/working/submission.csv'\n")

    error = kernel_source_preflight_error(
        kernel_dir,
        require_kaggle_input=False,
        format_error=lambda exc: f"formatted: {exc}",
    )

    assert error is not None
    assert error.startswith("formatted: Kernel source validation failed:")
    assert "metrics.json" in error


def test_kernel_source_validation_requires_outputs(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(kernel_dir / "kernel.py", "print('hello')\n")
    issues = validate_kernel_sources(kernel_dir)
    assert any("submission output" in issue for issue in issues)
    assert any("metrics.json" in issue for issue in issues)


def test_kernel_source_validation_accepts_non_csv_submission_output(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(
        kernel_dir / "kernel.py",
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/train.tsv'",
                "OUT1 = '/kaggle/working/submission.tsv'",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
    )

    ensure_kernel_sources_valid(kernel_dir)


def test_kernel_source_validation_accepts_compressed_submission_output(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(
        kernel_dir / "kernel.py",
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/train.jsonl.gz'",
                "OUT1 = '/kaggle/working/submission.jsonl.gz'",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
    )

    ensure_kernel_sources_valid(kernel_dir)


def test_kernel_source_validation_accepts_excel_submission_output(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(
        kernel_dir / "kernel.py",
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/train.xlsx'",
                "OUT1 = '/kaggle/working/submission.xlsx'",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
    )

    ensure_kernel_sources_valid(kernel_dir)


def test_kernel_source_validation_accepts_zstd_pickle_submission_output(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(
        kernel_dir / "kernel.py",
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/train.pkl.zst'",
                "OUT1 = '/kaggle/working/submission.pkl.zst'",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
    )

    ensure_kernel_sources_valid(kernel_dir)


def test_kernel_source_validation_accepts_archive_submission_output(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(
        kernel_dir / "kernel.py",
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/train.csv'",
                "OUT1 = '/kaggle/working/submission.tar.gz'",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
    )

    ensure_kernel_sources_valid(kernel_dir)


def test_kernel_source_validation_accepts_directory_submission_output(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(
        kernel_dir / "kernel.py",
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/train.zarr'",
                "OUT1 = '/kaggle/working/submission.zarr'",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
    )

    ensure_kernel_sources_valid(kernel_dir)


def test_kernel_source_validation_accepts_plain_tar_submission_output(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(
        kernel_dir / "kernel.py",
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/train.csv'",
                "OUT1 = '/kaggle/working/submission.tar'",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
    )

    ensure_kernel_sources_valid(kernel_dir)


@pytest.mark.parametrize("suffix", [".tar.xz", ".tar.zst"])
def test_kernel_source_validation_accepts_tar_submission_output(tmp_path: Path, suffix: str) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(
        kernel_dir / "kernel.py",
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/train.csv'",
                f"OUT1 = '/kaggle/working/submission{suffix}'",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
    )

    ensure_kernel_sources_valid(kernel_dir)


def test_kernel_source_validation_accepts_tbz2_submission_output(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(
        kernel_dir / "kernel.py",
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/train.csv'",
                "OUT1 = '/kaggle/working/submission.tbz2'",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
    )

    ensure_kernel_sources_valid(kernel_dir)


def test_kernel_source_validation_does_not_count_sample_submission_as_output(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(
        kernel_dir / "kernel.py",
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/train.csv'",
                "SAMPLE = '/kaggle/input/demo/sample_submission.csv.gz'",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
    )

    issues = validate_kernel_sources(kernel_dir)
    assert any("submission output" in issue for issue in issues)


def test_kernel_source_validation_accepts_manifest_submission_output(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(
        kernel_dir / "kernel.py",
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/train.csv'",
                "MANIFEST = '/kaggle/working/submission_manifest.json'",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
    )

    ensure_kernel_sources_valid(kernel_dir)


def test_kernel_source_validation_accepts_configured_submission_filename(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(
        kernel_dir / "kernel.py",
        "\n".join(
            [
                "import os",
                "DATA = '/kaggle/input/demo/train.parquet'",
                "OUT1 = os.getenv('KAGGLEBOT_SUBMISSION_FILENAME', 'submission.parquet')",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
    )

    ensure_kernel_sources_valid(kernel_dir)


def test_kernel_source_validation_accepts_submission_output_name_contract_marker(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(
        kernel_dir / "kernel.py",
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/train.parquet'",
                "SUBMISSION_OUTPUT_NAME = 'predictions.bin'",
                "OUT1 = '/kaggle/working/' + SUBMISSION_OUTPUT_NAME",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
    )

    ensure_kernel_sources_valid(kernel_dir)


@pytest.mark.parametrize("name", ["solution.csv", "outputs.parquet"])
def test_kernel_source_validation_accepts_shared_generic_submission_aliases(tmp_path: Path, name: str) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(
        kernel_dir / "kernel.py",
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/train.csv'",
                f"OUT1 = '/kaggle/working/{name}'",
                "Path(OUT1).write_text('id,target\\n1,0.1\\n', encoding='utf-8')",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
    )

    ensure_kernel_sources_valid(kernel_dir)


def test_kernel_source_validation_accepts_medical_single_file_submission_output(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(
        kernel_dir / "kernel.py",
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/train.svs'",
                "OUT1 = '/kaggle/working/mask.svs'",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
    )

    ensure_kernel_sources_valid(kernel_dir)


def test_kernel_source_validation_accepts_array_single_file_submission_output(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(
        kernel_dir / "kernel.py",
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/train.npy'",
                "OUT1 = '/kaggle/working/submission.npy'",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
    )

    ensure_kernel_sources_valid(kernel_dir)


def test_kernel_source_validation_accepts_model_single_file_submission_output(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(
        kernel_dir / "kernel.py",
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/train.csv'",
                "OUT1 = '/kaggle/working/submission.safetensors'",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
    )

    ensure_kernel_sources_valid(kernel_dir)


def test_kernel_source_validation_accepts_generic_non_tabular_output_name(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(
        kernel_dir / "kernel.py",
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/train.npy'",
                "OUT1 = '/kaggle/working/predictions.npy'",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
    )

    ensure_kernel_sources_valid(kernel_dir)


def test_kernel_source_validation_does_not_count_input_generic_prediction_file_as_output(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(
        kernel_dir / "kernel.py",
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/predictions.npy'",
                "OUT2 = '/kaggle/working/metrics.json'",
            ]
        )
        + "\n",
    )

    issues = validate_kernel_sources(kernel_dir)

    assert any("submission output" in issue for issue in issues)


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


def test_kernel_source_validation_flags_oracle_override_patterns(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(
        kernel_dir / "kernel.py",
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/train.csv'",
                "OUT1 = '/kaggle/working/submission.csv'",
                "OUT2 = '/kaggle/working/metrics.json'",
                "MODE = 'auto'",
                "def build_oracle_game_map():",
                "    return {}",
                "def apply_oracle_override(submission):",
                "    return submission",
                "ORACLE_MODE = 'KAGGLEBOT_ORACLE_MODE'",
            ]
        )
        + "\n",
    )
    issues = validate_kernel_sources(kernel_dir)
    assert any("oracle" in issue.lower() for issue in issues)


def test_kernel_source_validation_flags_lb_proxy_score_source(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel(
        kernel_dir / "kernel.py",
        "\n".join(
            [
                "DATA = '/kaggle/input/demo/train.csv'",
                "OUT1 = '/kaggle/working/submission.csv'",
                "OUT2 = '/kaggle/working/metrics.json'",
                "metrics_payload = {'score_source': 'lb_proxy'}",
            ]
        )
        + "\n",
    )
    issues = validate_kernel_sources(kernel_dir)
    assert any("lb_proxy" in issue for issue in issues)
