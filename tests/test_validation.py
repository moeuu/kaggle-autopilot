"""Tests for submission validation helpers."""

from __future__ import annotations

import bz2
import gzip
import json
import lzma
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import zstandard as zstd

from kagglebot.exceptions import SubmissionRateLimitError
from kagglebot.history import SubmissionLedger
from kagglebot.solver.io import write_table
from kagglebot.validation import ensure_submission_rate_limit, validate_submission


def test_validate_submission_success():
    """Test validation passes for matching submissions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_path = Path(tmpdir) / "sample.csv"
        submission_path = Path(tmpdir) / "submission.csv"

        # Create matching sample and submission
        df = pd.DataFrame({"id": [1, 2, 3], "target": [0.5, 0.7, 0.3]})
        df.to_csv(sample_path, index=False)
        df.to_csv(submission_path, index=False)

        # Should not raise
        validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_jsonl_success(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.jsonl"
    submission_path = tmp_path / "submission.jsonl"
    sample_path.write_text(
        '{"id":1,"target":0.0}\n{"id":2,"target":0.0}\n',
        encoding="utf-8",
    )
    submission_path.write_text(
        '{"id":1,"target":0.1}\n{"id":2,"target":0.2}\n',
        encoding="utf-8",
    )

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_wrapped_json_success(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.json"
    submission_path = tmp_path / "submission.json"
    sample_path.write_text(
        '{"records":[{"id":1,"target":0.0},{"id":2,"target":0.0}]}',
        encoding="utf-8",
    )
    submission_path.write_text(
        '{"rows":[{"id":1,"target":0.1},{"id":2,"target":0.2}]}',
        encoding="utf-8",
    )

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_csv_gz_success(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv.gz"
    submission_path = tmp_path / "submission.csv.gz"

    with gzip.open(sample_path, "wt", encoding="utf-8") as handle:
        handle.write("id,target\n1,0.0\n2,0.0\n")
    with gzip.open(submission_path, "wt", encoding="utf-8") as handle:
        handle.write("id,target\n1,0.1\n2,0.2\n")

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_tsv_gz_success(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.tsv.gz"
    submission_path = tmp_path / "submission.tsv.gz"

    with gzip.open(sample_path, "wt", encoding="utf-8") as handle:
        handle.write("id\ttarget\n001\t0.0\n002\t0.0\n")
    with gzip.open(submission_path, "wt", encoding="utf-8") as handle:
        handle.write("id\ttarget\n001\t0.1\n002\t0.2\n")

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_tab_delimited_txt_gz_success(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.txt.gz"
    submission_path = tmp_path / "submission.txt.gz"

    with gzip.open(sample_path, "wt", encoding="utf-8") as handle:
        handle.write("id\ttarget\n001\t0.0\n002\t0.0\n")
    with gzip.open(submission_path, "wt", encoding="utf-8") as handle:
        handle.write("id\ttarget\n001\t0.1\n002\t0.2\n")

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_csv_zst_success(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv.zst"
    submission_path = tmp_path / "submission.csv.zst"
    compressor = zstd.ZstdCompressor()

    sample_path.write_bytes(compressor.compress(b"id,target\n001,0.0\n002,0.0\n"))
    submission_path.write_bytes(compressor.compress(b"id,target\n001,0.1\n002,0.2\n"))

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_html_success(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.html"
    submission_path = tmp_path / "submission.html"
    pd.DataFrame({"id": ["001", "002"], "target": [0.0, 0.0]}).to_html(sample_path, index=False)
    pd.DataFrame({"id": ["001", "002"], "target": [0.1, 0.2]}).to_html(submission_path, index=False)

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_html_zst_success(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.html.zst"
    submission_path = tmp_path / "submission.html.zst"
    compressor = zstd.ZstdCompressor()
    sample_html = pd.DataFrame({"id": ["001", "002"], "target": [0.0, 0.0]}).to_html(index=False)
    submission_html = pd.DataFrame({"id": ["001", "002"], "target": [0.1, 0.2]}).to_html(index=False)

    sample_path.write_bytes(compressor.compress(sample_html.encode("utf-8")))
    submission_path.write_bytes(compressor.compress(submission_html.encode("utf-8")))

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_jsonl_gz_success(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.jsonl.gz"
    submission_path = tmp_path / "submission.jsonl.gz"

    with gzip.open(sample_path, "wt", encoding="utf-8") as handle:
        handle.write('{"id":1,"target":0.0}\n{"id":2,"target":0.0}\n')
    with gzip.open(submission_path, "wt", encoding="utf-8") as handle:
        handle.write('{"id":1,"target":0.1}\n{"id":2,"target":0.2}\n')

    validate_submission(str(sample_path), str(submission_path))


@pytest.mark.parametrize(
    "suffix",
    [
        ".jsonl.bz2",
        ".jsonl.xz",
        ".jsonlines",
        ".jsonlines.bz2",
        ".jsonlines.gz",
        ".jsonlines.xz",
        ".jsonlines.zst",
        ".ndjson",
        ".ndjson.bz2",
        ".ndjson.gz",
        ".ndjson.xz",
        ".ndjson.zst",
    ],
)
def test_validate_submission_json_lines_variants_success(tmp_path: Path, suffix: str) -> None:
    sample_path = tmp_path / f"sample_submission{suffix}"
    submission_path = tmp_path / f"submission{suffix}"
    sample_payload = b'{"id":1,"target":0.0}\n{"id":2,"target":0.0}\n'
    submission_payload = b'{"id":1,"target":0.1}\n{"id":2,"target":0.2}\n'
    if suffix.endswith(".gz"):
        with gzip.open(sample_path, "wb") as handle:
            handle.write(sample_payload)
        with gzip.open(submission_path, "wb") as handle:
            handle.write(submission_payload)
    elif suffix.endswith(".bz2"):
        with bz2.open(sample_path, "wb") as handle:
            handle.write(sample_payload)
        with bz2.open(submission_path, "wb") as handle:
            handle.write(submission_payload)
    elif suffix.endswith(".xz"):
        with lzma.open(sample_path, "wb") as handle:
            handle.write(sample_payload)
        with lzma.open(submission_path, "wb") as handle:
            handle.write(submission_payload)
    elif suffix.endswith(".zst"):
        compressor = zstd.ZstdCompressor()
        sample_path.write_bytes(compressor.compress(sample_payload))
        submission_path.write_bytes(compressor.compress(submission_payload))
    else:
        sample_path.write_bytes(sample_payload)
        submission_path.write_bytes(submission_payload)

    validate_submission(str(sample_path), str(submission_path))


@pytest.mark.parametrize("suffix", [".yaml.xz", ".xml.bz2", ".html.bz2", ".psv.xz", ".tab.zst"])
def test_validate_submission_compressed_structured_tabular_success(tmp_path: Path, suffix: str) -> None:
    sample_path = tmp_path / f"sample_submission{suffix}"
    submission_path = tmp_path / f"submission{suffix}"

    write_table(pd.DataFrame({"id": ["001", "002"], "target": [0.0, 0.0]}), sample_path)
    write_table(pd.DataFrame({"id": ["001", "002"], "target": [0.1, 0.2]}), submission_path)

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_xlsx_success(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.xlsx"
    submission_path = tmp_path / "submission.xlsx"
    pd.DataFrame({"id": ["001", "002"], "target": [0.0, 0.0]}).to_excel(sample_path, index=False)
    pd.DataFrame({"id": ["001", "002"], "target": [0.1, 0.2]}).to_excel(submission_path, index=False)

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_orc_success(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.orc"
    submission_path = tmp_path / "submission.orc"
    pd.DataFrame({"id": ["001", "002"], "target": [0.0, 0.0]}).to_orc(sample_path, index=False)
    pd.DataFrame({"id": ["001", "002"], "target": [0.1, 0.2]}).to_orc(submission_path, index=False)

    validate_submission(str(sample_path), str(submission_path))


@pytest.mark.parametrize("suffix", [".hdf", ".hdf5"])
def test_validate_submission_hdf_success(tmp_path: Path, suffix: str) -> None:
    sample_path = tmp_path / f"sample_submission{suffix}"
    submission_path = tmp_path / f"submission{suffix}"
    pd.DataFrame({"id": ["001", "002"], "target": [0.0, 0.0]}).to_hdf(
        sample_path,
        key="sample_submission",
        mode="w",
        format="table",
        index=False,
    )
    pd.DataFrame({"id": ["001", "002"], "target": [0.1, 0.2]}).to_hdf(
        submission_path,
        key="submission",
        mode="w",
        format="table",
        index=False,
    )

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_stata_success(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.dta"
    submission_path = tmp_path / "submission.dta"
    pd.DataFrame({"id": ["001", "002"], "target": [0.0, 0.0]}).to_stata(sample_path, write_index=False)
    pd.DataFrame({"id": ["001", "002"], "target": [0.1, 0.2]}).to_stata(submission_path, write_index=False)

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_column_mismatch():
    """Test validation fails when columns don't match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_path = Path(tmpdir) / "sample.csv"
        submission_path = Path(tmpdir) / "submission.csv"

        pd.DataFrame({"id": [1, 2, 3], "target": [0.5, 0.7, 0.3]}).to_csv(sample_path, index=False)
        pd.DataFrame({"id": [1, 2, 3], "score": [0.5, 0.7, 0.3]}).to_csv(submission_path, index=False)

        with pytest.raises(ValueError, match="columns mismatch"):
            validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_row_count_mismatch():
    """Test validation fails when row counts don't match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_path = Path(tmpdir) / "sample.csv"
        submission_path = Path(tmpdir) / "submission.csv"

        pd.DataFrame({"id": [1, 2, 3], "target": [0.5, 0.7, 0.3]}).to_csv(sample_path, index=False)
        pd.DataFrame({"id": [1, 2], "target": [0.5, 0.7]}).to_csv(submission_path, index=False)

        with pytest.raises(ValueError, match="row count mismatch"):
            validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_allows_declared_variable_instance_rows(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    context_dir = tmp_path / "context"
    data_dir.mkdir()
    context_dir.mkdir()
    sample_path = data_dir / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    sample_path.write_text(
        "image_id,segmentation_rle\nimg_a,0\nimg_b,0\n",
        encoding="utf-8",
    )
    submission_path.write_text(
        "image_id,segmentation_rle\nimg_a_1,rle-a\nimg_a_custom,rle-b\nimg_b_1,rle-c\n",
        encoding="utf-8",
    )
    (context_dir / "submission_format.md").write_text(
        "Each row corresponds to one predicted instance. ID suffixes make the rows unique.\n",
        encoding="utf-8",
    )

    validate_submission(str(sample_path), str(submission_path), data_dir=data_dir)


def test_validate_submission_does_not_infer_variable_instance_rows_without_context(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    sample_path.write_text("image_id,target\nimg_a,0\nimg_b,0\n", encoding="utf-8")
    submission_path.write_text(
        "image_id,target\nimg_a_1,0.1\nimg_a_2,0.2\nimg_b_1,0.3\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="row count mismatch"):
        validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_rejects_unknown_variable_instance_base(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    context_dir = tmp_path / "context"
    data_dir.mkdir()
    context_dir.mkdir()
    sample_path = data_dir / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    sample_path.write_text("image_id,target\nimg_a,0\nimg_b,0\n", encoding="utf-8")
    submission_path.write_text(
        "image_id,target\nimg_a_1,0.1\nimg_unknown_1,0.2\n",
        encoding="utf-8",
    )
    (context_dir / "submission_format.md").write_text(
        "Each row corresponds to one predicted object and the ID suffix makes rows unique.\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="id values mismatch"):
        validate_submission(str(sample_path), str(submission_path), data_dir=data_dir)


def test_validate_submission_checks_declared_compressed_coco_rle(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    context_dir = tmp_path / "context"
    data_dir.mkdir()
    context_dir.mkdir()
    sample_path = data_dir / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    sample_path.write_text("filament_id,segmentation_rle\nimg_a,0\n", encoding="utf-8")
    submission_path.write_text(
        "filament_id,segmentation_rle\nimg_a_1,01?\n",
        encoding="utf-8",
    )
    (context_dir / "submission_format.md").write_text(
        "Each row corresponds to one predicted filament. Use pycocotools RLE counts. "
        "The fixed Size is 4 X 4 pixels. ID suffixes make rows unique.\n",
        encoding="utf-8",
    )

    validate_submission(str(sample_path), str(submission_path), data_dir=data_dir)


def test_validate_submission_rejects_invalid_declared_compressed_coco_rle(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    context_dir = tmp_path / "context"
    data_dir.mkdir()
    context_dir.mkdir()
    sample_path = data_dir / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    sample_path.write_text("filament_id,segmentation_rle\nimg_a,0\n", encoding="utf-8")
    submission_path.write_text("filament_id,segmentation_rle\nimg_a_1,1\n", encoding="utf-8")
    (context_dir / "submission_format.md").write_text(
        "Each row corresponds to one predicted filament. Use compressed COCO-RLE counts. "
        "The fixed Size is 4 X 4 pixels. ID suffixes make rows unique.\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid compressed COCO RLE"):
        validate_submission(str(sample_path), str(submission_path), data_dir=data_dir)


def test_validate_submission_missing_id():
    """Test validation fails when id column has missing values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_path = Path(tmpdir) / "sample.csv"
        submission_path = Path(tmpdir) / "submission.csv"

        pd.DataFrame({"id": [1, 2, 3], "target": [0.5, 0.7, 0.3]}).to_csv(sample_path, index=False)
        pd.DataFrame({"id": [1, None, 3], "target": [0.5, 0.7, 0.3]}).to_csv(submission_path, index=False)

        with pytest.raises(ValueError, match="id column 'id' contains NaN"):
            validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_all_nan_target():
    """Test validation fails when all target values are NaN."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_path = Path(tmpdir) / "sample.csv"
        submission_path = Path(tmpdir) / "submission.csv"

        pd.DataFrame({"id": [1, 2, 3], "target": [0.5, 0.7, 0.3]}).to_csv(sample_path, index=False)
        pd.DataFrame({"id": [1, 2, 3], "target": [None, None, None]}).to_csv(submission_path, index=False)

        with pytest.raises(ValueError, match="prediction column 'target' contains NaN/non-numeric values"):
            validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_rejects_non_numeric_target_for_numeric_pickle_sample(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.pkl"
    submission_path = tmp_path / "submission.pkl"
    pd.DataFrame({"id": ["001", "002"], "target": [0.0, 0.0]}).to_pickle(sample_path)
    pd.DataFrame({"id": ["001", "002"], "target": [0.1, "bad"]}).to_pickle(submission_path)

    with pytest.raises(ValueError, match="NaN/non-numeric"):
        validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_rejects_infinite_target_for_numeric_sample(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_csv(sample_path, index=False)
    pd.DataFrame({"id": [1, 2], "target": [0.1, float("inf")]}).to_csv(submission_path, index=False)

    with pytest.raises(ValueError, match="\\+/-inf"):
        validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_allows_empty_text_prediction_values(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    sample_path.write_text("query_id,predicted_citations\ntest_001,0.0\ntest_002,0.0\n", encoding="utf-8")
    submission_path.write_text(
        "query_id,predicted_citations\ntest_001,Art. 1 OR;Art. 2 ZGB\ntest_002,\n",
        encoding="utf-8",
    )

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_allows_rle_empty_mask_marker(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    sample_path.write_text("id,EncodedPixels\ntest_001,\ntest_002,\n", encoding="utf-8")
    submission_path.write_text("id,EncodedPixels\ntest_001,-\ntest_002,-\n", encoding="utf-8")

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_id_mismatch():
    """Test validation fails when ids do not match sample submission."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_path = Path(tmpdir) / "sample.csv"
        submission_path = Path(tmpdir) / "submission.csv"

        pd.DataFrame({"id": [1, 2, 3], "target": [0.5, 0.7, 0.3]}).to_csv(sample_path, index=False)
        pd.DataFrame({"id": [1, 2, 4], "target": [0.5, 0.7, 0.3]}).to_csv(submission_path, index=False)

        with pytest.raises(ValueError, match="id values mismatch"):
            validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_duplicate_id():
    """Test validation fails when ids are duplicated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_path = Path(tmpdir) / "sample.csv"
        submission_path = Path(tmpdir) / "submission.csv"

        pd.DataFrame({"id": [1, 2, 3], "target": [0.5, 0.7, 0.3]}).to_csv(sample_path, index=False)
        pd.DataFrame({"id": [1, 1, 3], "target": [0.5, 0.7, 0.3]}).to_csv(submission_path, index=False)

        with pytest.raises(ValueError, match="duplicate values"):
            validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_long_format_allows_row_mismatch_and_duplicates(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.tsv"
    submission_path = tmp_path / "submission.tsv"

    sample_path.write_text(
        "id\tterm\tscore\nP1\tGO:0000001\t0.123\nP1\tGO:0000002\t0.456\n",
        encoding="utf-8",
    )
    submission_path.write_text(
        "P1\tGO:0000001\t0.999\nP1\tGO:0000003\t0.888\nP2\tGO:0000002\t0.777\n",
        encoding="utf-8",
    )

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_header_only_sample_allows_row_mismatch(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"

    sample_path.write_text("id,target\n", encoding="utf-8")
    pd.DataFrame({"id": [1, 2, 3], "target": [0.1, 0.2, 0.3]}).to_csv(submission_path, index=False)

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_compat_wrapper_passes_data_dir_context(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    test_dir = data_dir / "images" / "test"
    test_dir.mkdir(parents=True)
    (test_dir / "img_001.png").write_bytes(b"png")
    (test_dir / "img_002.png").write_bytes(b"png")
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    sample_path.write_text("id,target\n", encoding="utf-8")
    pd.DataFrame({"id": ["img_001.png", "img_002.png"], "target": [0.1, 0.2]}).to_csv(
        submission_path,
        index=False,
    )

    validate_submission(str(sample_path), str(submission_path), data_dir=data_dir)


def test_validate_submission_file_passes_data_dir_context(tmp_path: Path) -> None:
    from kagglebot.solver.validate import validate_submission_file

    data_dir = tmp_path / "data"
    test_dir = data_dir / "audio" / "test"
    test_dir.mkdir(parents=True)
    (test_dir / "clip_001.wav").write_bytes(b"audio")
    (test_dir / "clip_002.wav").write_bytes(b"audio")
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    sample_path.write_text("id,target\n", encoding="utf-8")
    pd.DataFrame({"id": ["clip_001.wav", "clip_002.wav"], "target": [0.1, 0.2]}).to_csv(
        submission_path,
        index=False,
    )

    validate_submission_file(sample_path, submission_path, data_dir=data_dir)


def test_validate_submission_handles_irregular_tsv(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.tsv"
    submission_path = tmp_path / "submission.csv"
    format_path = tmp_path / "submission_format.md"

    format_path.write_text("## Submission Format\n\nid,term,score\n", encoding="utf-8")
    sample_path.write_text(
        "A0A0C5B5G6\tGO:0000001\t0.123\nA0A0C5B5G6\tText\t0.456\tExtra text column\n",
        encoding="utf-8",
    )
    submission_path.write_text("id,term,score\nA0A0C5B5G6,GO:0000001,0.999\n", encoding="utf-8")

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_handles_tabbed_csv_with_commas(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"

    sample_path.write_text(
        "A0A0C5B5G6\tGO:0000001\t0.123\nA0A0C5B5G6\tText\t0.456\tInhibits, something\n",
        encoding="utf-8",
    )
    submission_path.write_text("A0A0C5B5G6\tGO:0000001\t0.999\n", encoding="utf-8")

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_keeps_header_delimiter_when_values_contain_semicolons(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"

    sample_path.write_text(
        "query_id,predicted_citations\ntest_001,0.0\ntest_002,0.0\n",
        encoding="utf-8",
    )
    submission_path.write_text(
        "query_id,predicted_citations\n"
        "test_001,Art. 1 OR;Art. 2 ZGB;Art. 3 BV\n"
        "test_002,Art. 4 OR;Art. 5 ZGB;Art. 6 BV\n",
        encoding="utf-8",
    )

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_handles_semicolon_delimited_csv(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"

    sample_path.write_text("id;target\n1;0.0\n2;0.0\n", encoding="utf-8")
    submission_path.write_text("id;target\n1;0.7\n2;0.2\n", encoding="utf-8")

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_uses_pipe_default_for_psv(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.psv"
    submission_path = tmp_path / "submission.psv"

    sample_path.write_text("id|target\n1|0.0\n2|0.0\n", encoding="utf-8")
    submission_path.write_text("id|target\n1|0.7\n2|0.2\n", encoding="utf-8")

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_fallback_sniffer_handles_pipe_delimited_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"

    sample_path.write_text("id|target\n1|0.0\n2|0.0\n", encoding="utf-8")
    submission_path.write_text("id|target\n1|0.7\n2|0.2\n", encoding="utf-8")

    def fail_sniff(*args: object, **kwargs: object) -> str:
        raise RuntimeError("sniff failed")

    monkeypatch.setattr("kagglebot.submission.validate.sniff_tabular_text_delimiter", fail_sniff)

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_allows_citation_text_when_sample_has_numeric_placeholders(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    format_path = tmp_path / "submission_format.md"

    format_path.write_text(
        "## Submission Format\n\nUse semicolon-separated citations in `predicted_citations`; empty string allowed.\n",
        encoding="utf-8",
    )
    sample_path.write_text(
        "query_id,predicted_citations\ntest_001,0.0\ntest_002,0.0\n",
        encoding="utf-8",
    )
    submission_path.write_text(
        "query_id,predicted_citations\ntest_001,Art. 1 OR;Art. 2 ZGB\ntest_002,\n",
        encoding="utf-8",
    )

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_ignores_noisy_format_hint(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    format_path = tmp_path / "submission_format.md"

    format_path.write_text(
        "## Ahoy, welcome to Kaggle! You're in the right place. "
        "This is the legendary Titanic ML competition -- the best, first challenge. "
        "PassengerId,Survived\n",
        encoding="utf-8",
    )
    sample_path.write_text("PassengerId,Survived\n1,0\n", encoding="utf-8")
    submission_path.write_text("PassengerId,Survived\n1,0\n", encoding="utf-8")

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_rejects_markdown_sample_without_columns(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"

    sample_path.write_text("# Sample submission\n\nDownload the real sample from Kaggle.\n", encoding="utf-8")
    submission_path.write_text("id,target\n1,0.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sample_submission has no columns"):
        validate_submission(str(sample_path), str(submission_path))


def test_submission_rate_limit(tmp_path):
    ledger = SubmissionLedger(tmp_path / "ledger.jsonl")
    now = datetime.now(UTC)
    entries = [
        {
            "ts": (now - timedelta(hours=1)).isoformat(),
            "sha256": "a",
            "fingerprint": "f1",
            "slug": "demo",
            "submission_path": "sub.csv",
            "message": "m1",
            "run_id": "r1",
        },
        {
            "ts": (now - timedelta(minutes=10)).isoformat(),
            "sha256": "b",
            "fingerprint": "f2",
            "slug": "demo",
            "submission_path": "sub2.csv",
            "message": "m2",
            "run_id": "r2",
        },
    ]
    ledger.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.ledger_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    with pytest.raises(SubmissionRateLimitError, match="cooldown"):
        ensure_submission_rate_limit(ledger, max_submissions_per_day=5, min_hours_between=1.0)


def test_submission_ledger_detects_duplicate_directory_submission(tmp_path: Path) -> None:
    ledger = SubmissionLedger(tmp_path / "ledger.jsonl")
    first = tmp_path / "submission-a.zarr"
    second = tmp_path / "submission-b.zarr"
    for root in (first, second):
        (root / "arrays").mkdir(parents=True)
        (root / ".zgroup").write_text("{}", encoding="utf-8")
        (root / "arrays" / "0").write_bytes(b"chunk")

    ledger.record(slug="demo", message="first", submission_path=first, run_id="run-1")

    assert ledger.is_duplicate(slug="demo", message="different message", submission_path=second)


def test_submission_rate_limit_default_cooldown_is_five_minutes(tmp_path):
    ledger = SubmissionLedger(tmp_path / "ledger.jsonl")
    now = datetime.now(UTC)
    entries = [
        {
            "ts": (now - timedelta(minutes=4)).isoformat(),
            "sha256": "a",
            "fingerprint": "f1",
            "slug": "demo",
            "submission_path": "sub.csv",
            "message": "m1",
            "run_id": "r1",
        }
    ]
    ledger.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.ledger_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    with pytest.raises(SubmissionRateLimitError, match="cooldown"):
        ensure_submission_rate_limit(ledger)


def test_submission_rate_limit_allows_env_cooldown_override(monkeypatch, tmp_path):
    monkeypatch.setenv("KAGGLEBOT_SUBMISSION_MIN_HOURS_BETWEEN", "0")
    ledger = SubmissionLedger(tmp_path / "ledger.jsonl")
    now = datetime.now(UTC)
    entries = [
        {
            "ts": (now - timedelta(minutes=1)).isoformat(),
            "sha256": "a",
            "fingerprint": "f1",
            "slug": "demo",
            "submission_path": "sub.csv",
            "message": "m1",
            "run_id": "r1",
        }
    ]
    ledger.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.ledger_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    ensure_submission_rate_limit(ledger)


def test_submission_rate_limit_ignores_outcome_events(tmp_path):
    ledger = SubmissionLedger(tmp_path / "ledger.jsonl")
    now = datetime.now(UTC)
    entries = [
        {
            "ts": (now - timedelta(hours=20)).isoformat(),
            "event": "submit",
            "sha256": "a",
            "fingerprint": "f1",
            "slug": "demo",
            "submission_path": "sub.csv",
            "message": "m1",
            "run_id": "r1",
        },
        {
            "ts": (now - timedelta(hours=20) + timedelta(seconds=30)).isoformat(),
            "event": "outcome",
            "sha256": "a",
            "slug": "demo",
            "submission_path": "sub.csv",
            "message": "m1",
            "run_id": "r1",
            "outcome": {"status": "complete"},
        },
        {
            "ts": (now - timedelta(hours=12)).isoformat(),
            "event": "submit",
            "sha256": "b",
            "fingerprint": "f2",
            "slug": "demo",
            "submission_path": "sub2.csv",
            "message": "m2",
            "run_id": "r2",
        },
        {
            "ts": (now - timedelta(hours=12) + timedelta(seconds=30)).isoformat(),
            "event": "outcome",
            "sha256": "b",
            "slug": "demo",
            "submission_path": "sub2.csv",
            "message": "m2",
            "run_id": "r2",
            "outcome": {"status": "complete"},
        },
        {
            "ts": (now - timedelta(hours=6)).isoformat(),
            "event": "submit",
            "sha256": "c",
            "fingerprint": "f3",
            "slug": "demo",
            "submission_path": "sub3.csv",
            "message": "m3",
            "run_id": "r3",
        },
        {
            "ts": (now - timedelta(hours=6) + timedelta(seconds=30)).isoformat(),
            "event": "outcome",
            "sha256": "c",
            "slug": "demo",
            "submission_path": "sub3.csv",
            "message": "m3",
            "run_id": "r3",
            "outcome": {"status": "complete"},
        },
        {
            "ts": (now - timedelta(minutes=6)).isoformat(),
            "event": "submit",
            "sha256": "d",
            "fingerprint": "f4",
            "slug": "demo",
            "submission_path": "sub4.csv",
            "message": "m4",
            "run_id": "r4",
        },
        {
            "ts": (now - timedelta(minutes=5, seconds=30)).isoformat(),
            "event": "outcome",
            "sha256": "d",
            "slug": "demo",
            "submission_path": "sub4.csv",
            "message": "m4",
            "run_id": "r4",
            "outcome": {"status": "complete"},
        },
    ]
    ledger.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.ledger_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    ensure_submission_rate_limit(ledger)
