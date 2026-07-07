from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kagglebot.kernel_runner import _local_submission_filename_from_sample


def test_local_submission_filename_uses_tabular_sample_suffix(tmp_path: Path) -> None:
    sample = tmp_path / "demo" / "context" / "sample_submission.jsonl"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text('{"id":1,"target":0.0}\n', encoding="utf-8")

    assert _local_submission_filename_from_sample(base_dir=tmp_path, slug="demo") == "submission.jsonl"


@pytest.mark.parametrize("suffix", [".orc", ".hdf", ".hdf5"])
def test_local_submission_filename_uses_binary_tabular_sample_suffix(tmp_path: Path, suffix: str) -> None:
    sample = tmp_path / "demo" / "context" / f"sample_submission{suffix}"
    sample.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"id": [1], "target": [0.0]})
    if suffix == ".orc":
        frame.to_orc(sample, index=False)
    else:
        frame.to_hdf(sample, key="submission", mode="w", format="table", index=False)

    assert _local_submission_filename_from_sample(base_dir=tmp_path, slug="demo") == f"submission{suffix}"


def test_local_submission_filename_ignores_archive_sample_suffix_without_format_hint(tmp_path: Path) -> None:
    sample = tmp_path / "demo" / "context" / "sample_submission.zip"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_bytes(b"archive")

    assert _local_submission_filename_from_sample(base_dir=tmp_path, slug="demo") is None


@pytest.mark.parametrize("suffix", [".tar.xz", ".tar.zst"])
def test_local_submission_filename_uses_submission_format_archive_suffix(tmp_path: Path, suffix: str) -> None:
    context_dir = tmp_path / "demo" / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        f"## Submission Format\nSubmit a submission{suffix} archive containing model weights and inference code.\n",
        encoding="utf-8",
    )

    assert _local_submission_filename_from_sample(base_dir=tmp_path, slug="demo") == f"submission{suffix}"


@pytest.mark.parametrize(
    ("description", "filename"),
    [
        ("Submit a zstd-compressed NDJSON file with columns row_id,target.", "submission.ndjson.zst"),
        ("Submit a bzip2-compressed HTML file with columns row_id,target.", "submission.html.bz2"),
    ],
)
def test_local_submission_filename_uses_submission_format_compressed_tabular_keywords(
    tmp_path: Path,
    description: str,
    filename: str,
) -> None:
    context_dir = tmp_path / "demo" / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(f"## Submission Format\n{description}\n", encoding="utf-8")

    assert _local_submission_filename_from_sample(base_dir=tmp_path, slug="demo") == filename


def test_local_submission_filename_uses_submission_format_external_archive_suffix(tmp_path: Path) -> None:
    context_dir = tmp_path / "demo" / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload a single `submission.7z` archive for scoring.\n",
        encoding="utf-8",
    )

    assert _local_submission_filename_from_sample(base_dir=tmp_path, slug="demo") == "submission.7z"


def test_local_submission_filename_uses_submission_format_single_file_suffix(tmp_path: Path) -> None:
    context_dir = tmp_path / "demo" / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nSubmit a single ONNX file named `submission.onnx`.\n",
        encoding="utf-8",
    )

    assert _local_submission_filename_from_sample(base_dir=tmp_path, slug="demo") == "submission.onnx"


def test_local_submission_filename_uses_submission_format_sqlite_suffix(tmp_path: Path) -> None:
    context_dir = tmp_path / "demo" / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload a SQLite database named `predictions.sqlite` for scoring.\n",
        encoding="utf-8",
    )

    assert _local_submission_filename_from_sample(base_dir=tmp_path, slug="demo") == "predictions.sqlite"
