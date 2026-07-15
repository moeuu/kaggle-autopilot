from __future__ import annotations

import gzip
import io
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import zstandard as zstd

from kagglebot.exceptions import SubmissionCliError, SubmissionValidationError
from kagglebot.submission.guard import (
    classify_submit_error,
    compute_error_fingerprint,
    normalize_error_text,
    run_kaggle_submit,
    run_kaggle_submit_kernel,
)
from kagglebot.submission.validate import validate_submission


def _write_sample_and_submission(tmp_path: Path) -> tuple[Path, Path]:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2, 3], "target": [0.0, 0.0, 0.0]}).to_csv(sample, index=False)
    pd.DataFrame({"id": [1, 2, 3], "target": [0.1, 0.2, 0.3]}).to_csv(submission, index=False)
    return sample, submission


def test_validate_submission_columns_mismatch(tmp_path: Path) -> None:
    sample, submission = _write_sample_and_submission(tmp_path)
    pd.DataFrame({"id": [1, 2, 3], "score": [0.1, 0.2, 0.3]}).to_csv(submission, index=False)
    with pytest.raises(SubmissionValidationError, match="columns mismatch"):
        validate_submission(str(submission), str(sample))


def test_validate_submission_rejects_markdown_sample_without_columns(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    sample.write_text("# Sample submission\n\nDownload the real sample from Kaggle.\n", encoding="utf-8")
    submission.write_text("id,target\n1,0.5\n", encoding="utf-8")

    with pytest.raises(SubmissionValidationError, match="sample_submission has no columns"):
        validate_submission(str(submission), str(sample))


def test_validate_submission_accepts_csv_gz_sample_and_submission(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv.gz"
    submission = tmp_path / "submission.csv.gz"
    with gzip.open(sample, "wt", encoding="utf-8") as handle:
        handle.write("id,target\n001,0.0\n002,0.0\n")
    with gzip.open(submission, "wt", encoding="utf-8") as handle:
        handle.write("id,target\n001,0.1\n002,0.2\n")

    validate_submission(str(submission), str(sample))


def test_validate_submission_accepts_csv_zst_sample_and_submission(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv.zst"
    submission = tmp_path / "submission.csv.zst"
    compressor = zstd.ZstdCompressor()

    sample.write_bytes(compressor.compress(b"id,target\n001,0.0\n002,0.0\n"))
    submission.write_bytes(compressor.compress(b"id,target\n001,0.1\n002,0.2\n"))

    validate_submission(str(submission), str(sample))


def test_validate_submission_normalizes_blank_csv_header_columns(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    sample.write_text("id,,target\n1,-,0\n2,-,0\n", encoding="utf-8")
    submission.write_text("id,column_2,target\n1,-,0.1\n2,-,0.2\n", encoding="utf-8")

    validate_submission(str(submission), str(sample))


def test_validate_submission_accepts_html_sample_and_submission(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.html"
    submission = tmp_path / "submission.html"
    pd.DataFrame({"id": ["001", "002"], "target": [0.0, 0.0]}).to_html(sample, index=False)
    pd.DataFrame({"id": ["001", "002"], "target": [0.1, 0.2]}).to_html(submission, index=False)

    validate_submission(str(submission), str(sample))


def test_validate_submission_accepts_html_zst_sample_and_submission(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.html.zst"
    submission = tmp_path / "submission.html.zst"
    compressor = zstd.ZstdCompressor()
    sample_html = pd.DataFrame({"id": ["001", "002"], "target": [0.0, 0.0]}).to_html(index=False)
    submission_html = pd.DataFrame({"id": ["001", "002"], "target": [0.1, 0.2]}).to_html(index=False)

    sample.write_bytes(compressor.compress(sample_html.encode("utf-8")))
    submission.write_bytes(compressor.compress(submission_html.encode("utf-8")))

    validate_submission(str(submission), str(sample))


def test_validate_submission_accepts_ndjson_sample_and_submission(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.ndjson"
    submission = tmp_path / "submission.ndjson"
    sample.write_text('{"id":"001","target":0.0}\n{"id":"002","target":0.0}\n', encoding="utf-8")
    submission.write_text('{"id":"001","target":0.1}\n{"id":"002","target":0.2}\n', encoding="utf-8")

    validate_submission(str(submission), str(sample))


def test_validate_submission_accepts_wrapped_json_sample_and_submission(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.json"
    submission = tmp_path / "submission.json"
    sample.write_text(
        '{"records":[{"id":"001","target":0.0},{"id":"002","target":0.0}]}',
        encoding="utf-8",
    )
    submission.write_text(
        '{"rows":[{"id":"001","target":0.1},{"id":"002","target":0.2}]}',
        encoding="utf-8",
    )

    validate_submission(str(submission), str(sample))


def test_validate_submission_accepts_compressed_wrapped_json_sample_and_submission(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.json.zst"
    submission = tmp_path / "submission.json.zst"
    compressor = zstd.ZstdCompressor()
    sample.write_bytes(
        compressor.compress(b'{"data":[{"id":"001","target":0.0},{"id":"002","target":0.0}]}'),
    )
    submission.write_bytes(
        compressor.compress(b'{"items":[{"id":"001","target":0.1},{"id":"002","target":0.2}]}'),
    )

    validate_submission(str(submission), str(sample))


def test_validate_submission_accepts_rle_empty_mask_marker(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    sample.write_text("id,EncodedPixels\ntest_001,\ntest_002,\n", encoding="utf-8")
    submission.write_text("id,EncodedPixels\ntest_001,-\ntest_002,-\n", encoding="utf-8")

    validate_submission(str(submission), str(sample))


def test_validate_submission_accepts_pickle_sample_and_submission(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.pkl"
    submission = tmp_path / "submission.pkl"
    pd.DataFrame({"id": ["001", "002"], "target": [0.0, 0.0]}).to_pickle(sample)
    pd.DataFrame({"id": ["001", "002"], "target": [0.1, 0.2]}).to_pickle(submission)

    validate_submission(str(submission), str(sample))


def test_validate_submission_accepts_zstd_pickle_sample_and_submission(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.pkl.zst"
    submission = tmp_path / "submission.pkl.zst"
    pd.DataFrame({"id": ["001", "002"], "target": [0.0, 0.0]}).to_pickle(sample)
    pd.DataFrame({"id": ["001", "002"], "target": [0.1, 0.2]}).to_pickle(submission)

    validate_submission(str(submission), str(sample))


def test_validate_submission_accepts_stata_sample_and_submission(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.dta"
    submission = tmp_path / "submission.dta"
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_stata(sample, write_index=False)
    pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}).to_stata(submission, write_index=False)

    validate_submission(str(submission), str(sample))


def test_validate_submission_accepts_xml_sample_and_submission(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.xml"
    submission = tmp_path / "submission.xml"
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_xml(sample, index=False, parser="etree")
    pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}).to_xml(submission, index=False, parser="etree")

    validate_submission(str(submission), str(sample))


def test_validate_submission_uses_nifti_test_asset_ids_for_header_only_sample(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    test_dir = data_dir / "scans" / "test"
    test_dir.mkdir(parents=True)
    (test_dir / "case_001.nii.gz").write_bytes(b"scan-1")
    (test_dir / "case_002.nii.gz").write_bytes(b"scan-2")
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    sample.write_text("id,target\n", encoding="utf-8")
    pd.DataFrame(
        {
            "id": ["case_001.nii.gz", "case_002.nii.gz"],
            "target": [0.1, 0.2],
        }
    ).to_csv(submission, index=False)

    validate_submission(str(submission), str(sample), data_dir=data_dir)


def test_validate_submission_uses_audio_test_asset_ids_for_header_only_sample(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    test_dir = data_dir / "audio" / "test"
    test_dir.mkdir(parents=True)
    (test_dir / "clip_001.wav").write_bytes(b"audio-1")
    (test_dir / "clip_002.wav").write_bytes(b"audio-2")
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    sample.write_text("id,target\n", encoding="utf-8")
    pd.DataFrame(
        {
            "id": ["clip_001.wav", "clip_002.wav"],
            "target": [0.1, 0.2],
        }
    ).to_csv(submission, index=False)

    validate_submission(str(submission), str(sample), data_dir=data_dir)


def test_validate_submission_uses_model_artifact_test_asset_ids_for_header_only_sample(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    test_dir = data_dir / "models" / "test"
    test_dir.mkdir(parents=True)
    (test_dir / "fold_001.onnx").write_bytes(b"model-1")
    (test_dir / "fold_002.onnx").write_bytes(b"model-2")
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    sample.write_text("id,target\n", encoding="utf-8")
    pd.DataFrame(
        {
            "id": ["fold_001.onnx", "fold_002.onnx"],
            "target": [0.1, 0.2],
        }
    ).to_csv(submission, index=False)

    validate_submission(str(submission), str(sample), data_dir=data_dir)


def test_validate_submission_uses_zarr_test_asset_ids_for_header_only_sample(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    test_dir = data_dir / "arrays" / "test"
    test_dir.mkdir(parents=True)
    for name in ("cell_001.zarr", "cell_002.zarr"):
        store = test_dir / name
        store.mkdir()
        (store / ".zarray").write_text("{}", encoding="utf-8")
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    sample.write_text("id,target\n", encoding="utf-8")
    pd.DataFrame(
        {
            "id": ["cell_001.zarr", "cell_002.zarr"],
            "target": [0.1, 0.2],
        }
    ).to_csv(submission, index=False)

    validate_submission(str(submission), str(sample), data_dir=data_dir)


def test_validate_submission_row_count_mismatch(tmp_path: Path) -> None:
    sample, submission = _write_sample_and_submission(tmp_path)
    pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}).to_csv(submission, index=False)
    with pytest.raises(SubmissionValidationError, match="row count mismatch"):
        validate_submission(str(submission), str(sample))


def test_validate_submission_target_only_sample_uses_test_row_count(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.svmlight").write_text("0 1:0.0\n1 1:1.0\n", encoding="utf-8")
    (data_dir / "test.svmlight").write_text(
        "0 1:0.1\n0 1:0.2\n0 1:0.3\n",
        encoding="utf-8",
    )
    sample = data_dir / "sample_submission.csv"
    sample.write_text("target\n0.0\n0.0\n", encoding="utf-8")
    submission = tmp_path / "submission.csv"
    submission.write_text("target\n0.1\n0.2\n0.3\n", encoding="utf-8")

    validate_submission(str(submission), str(sample), data_dir=data_dir)


def test_validate_submission_target_only_sample_rejects_test_row_count_mismatch(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.svmlight").write_text("0 1:0.0\n1 1:1.0\n", encoding="utf-8")
    (data_dir / "test.svmlight").write_text(
        "0 1:0.1\n0 1:0.2\n0 1:0.3\n",
        encoding="utf-8",
    )
    sample = data_dir / "sample_submission.csv"
    sample.write_text("target\n0.0\n0.0\n", encoding="utf-8")
    submission = tmp_path / "submission.csv"
    submission.write_text("target\n0.1\n", encoding="utf-8")

    with pytest.raises(SubmissionValidationError, match="row count mismatch"):
        validate_submission(str(submission), str(sample), data_dir=data_dir)


def test_validate_submission_id_nan(tmp_path: Path) -> None:
    sample, submission = _write_sample_and_submission(tmp_path)
    pd.DataFrame({"id": [1, None, 3], "target": [0.1, 0.2, 0.3]}).to_csv(submission, index=False)
    with pytest.raises(SubmissionValidationError, match="id column 'id' contains NaN"):
        validate_submission(str(submission), str(sample))


def test_validate_submission_id_duplicate(tmp_path: Path) -> None:
    sample, submission = _write_sample_and_submission(tmp_path)
    pd.DataFrame({"id": [1, 1, 3], "target": [0.1, 0.2, 0.3]}).to_csv(submission, index=False)
    with pytest.raises(SubmissionValidationError, match="duplicate values"):
        validate_submission(str(submission), str(sample))


def test_validate_submission_rejects_parquet_id_set_mismatch(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.parquet"
    submission = tmp_path / "submission.parquet"
    pd.DataFrame({"id": ["001", "002", "003"], "target": [0.0, 0.0, 0.0]}).to_parquet(sample, index=False)
    pd.DataFrame({"id": ["001", "002", "999"], "target": [0.1, 0.2, 0.3]}).to_parquet(
        submission,
        index=False,
    )

    with pytest.raises(SubmissionValidationError) as exc:
        validate_submission(str(submission), str(sample))

    message = str(exc.value)
    assert "id values mismatch (sample submission ids)" in message
    assert "003" in message
    assert "999" in message


def test_validate_submission_rejects_file_path_id_set_mismatch(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    pd.DataFrame(
        {
            "file_path": ["test/a.png", "test/b.png", "test/c.png"],
            "target": [0.0, 0.0, 0.0],
        }
    ).to_csv(sample, index=False)
    pd.DataFrame(
        {
            "file_path": ["test/a.png", "test/b.png", "test/z.png"],
            "target": [0.1, 0.2, 0.3],
        }
    ).to_csv(submission, index=False)

    with pytest.raises(SubmissionValidationError) as exc:
        validate_submission(str(submission), str(sample), data_dir=tmp_path / "data")

    message = str(exc.value)
    assert "id values mismatch (sample submission ids)" in message
    assert "test/c.png" in message
    assert "test/z.png" in message


def test_validate_submission_accepts_zip_wrapped_parquet_sample(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.parquet.zip"
    payload = io.BytesIO()
    pd.DataFrame({"id": ["001", "002", "003"], "target": [0.0, 0.0, 0.0]}).to_parquet(payload, index=False)
    with zipfile.ZipFile(sample, "w") as archive:
        archive.writestr("nested/sample_submission.parquet", payload.getvalue())
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": ["001", "002", "003"], "target": [0.1, 0.2, 0.3]}).to_csv(submission, index=False)

    validate_submission(str(submission), str(sample))


def test_validate_submission_allows_jsonl_id_reordering_when_ids_match(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.jsonl"
    submission = tmp_path / "submission.jsonl"
    sample.write_text(
        '{"id":"001","target":0.0}\n{"id":"002","target":0.0}\n{"id":"003","target":0.0}\n',
        encoding="utf-8",
    )
    submission.write_text(
        '{"id":"003","target":0.3}\n{"id":"001","target":0.1}\n{"id":"002","target":0.2}\n',
        encoding="utf-8",
    )

    validate_submission(str(submission), str(sample))


def test_validate_submission_pred_nan_or_non_numeric(tmp_path: Path) -> None:
    sample, submission = _write_sample_and_submission(tmp_path)
    pd.DataFrame({"id": [1, 2, 3], "target": [0.1, "abc", 0.3]}).to_csv(submission, index=False)
    with pytest.raises(SubmissionValidationError, match="NaN/non-numeric"):
        validate_submission(str(submission), str(sample))


def test_validate_submission_pred_inf(tmp_path: Path) -> None:
    sample, submission = _write_sample_and_submission(tmp_path)
    pd.DataFrame({"id": [1, 2, 3], "target": [0.1, float("inf"), 0.3]}).to_csv(submission, index=False)
    with pytest.raises(SubmissionValidationError, match="contains \\+/-inf"):
        validate_submission(str(submission), str(sample))


def test_validate_submission_categorical_target_passes(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2, 3], "target": ["Absence", "Presence", "Absence"]}).to_csv(sample, index=False)
    pd.DataFrame({"id": [1, 2, 3], "target": ["Presence", "Absence", "Presence"]}).to_csv(submission, index=False)

    validate_submission(str(submission), str(sample))


def test_validate_submission_categorical_target_allows_unknown_values(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2, 3], "target": ["Absence", "Presence", "Absence"]}).to_csv(sample, index=False)
    pd.DataFrame({"id": [1, 2, 3], "target": [0, 1, 0]}).to_csv(submission, index=False)

    validate_submission(str(submission), str(sample))


def test_validate_submission_allows_text_prediction_columns_with_empty_sample_values(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "translation": ["", ""]}).to_csv(sample, index=False)
    pd.DataFrame({"id": [1, 2], "translation": ["alpha one", "beta two"]}).to_csv(submission, index=False)

    validate_submission(str(submission), str(sample))


def test_validate_submission_requires_rna_anchor_columns_to_match_sample(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    pd.DataFrame(
        {
            "ID": ["RNA1_1", "RNA1_2"],
            "resname": ["A", "C"],
            "resid": [1, 2],
            "x_1": [0.0, 0.0],
            "y_1": [0.0, 0.0],
            "z_1": [0.0, 0.0],
        }
    ).to_csv(sample, index=False)
    pd.DataFrame(
        {
            "ID": ["RNA1_1", "RNA1_2"],
            "resname": ["G", "C"],
            "resid": [1, 2],
            "x_1": [0.1, 0.2],
            "y_1": [0.3, 0.4],
            "z_1": [0.5, 0.6],
        }
    ).to_csv(submission, index=False)

    with pytest.raises(SubmissionValidationError, match="anchor column 'resname'"):
        validate_submission(str(submission), str(sample))


def test_validate_submission_reports_actual_sample_name_for_structured_anchor_mismatch(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.jsonl"
    submission = tmp_path / "submission.jsonl"
    sample.write_text(
        '{"ID":"RNA1_1","resname":"A","resid":1,"x_1":0.0,"y_1":0.0,"z_1":0.0}\n',
        encoding="utf-8",
    )
    submission.write_text(
        '{"ID":"RNA1_1","resname":"G","resid":1,"x_1":0.1,"y_1":0.2,"z_1":0.3}\n',
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError) as exc:
        validate_submission(str(submission), str(sample))

    assert "anchor column 'resname' must match sample_submission.jsonl exactly" in str(exc.value)


def test_validate_submission_reports_multiple_problems(tmp_path: Path) -> None:
    sample, submission = _write_sample_and_submission(tmp_path)
    pd.DataFrame({"id": [1, None], "target": [0.1, "bad"]}).to_csv(submission, index=False)
    with pytest.raises(SubmissionValidationError) as exc:
        validate_submission(str(submission), str(sample))
    message = str(exc.value)
    assert "Submission validation failed:" in message
    assert "- row count mismatch:" in message
    assert "- id column 'id' contains NaN values:" in message
    assert "- prediction column 'target' contains NaN/non-numeric values:" in message


def test_validate_submission_rejects_tiny_static_hidden_test_notebook_submission(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    sample = context_dir / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    pd.DataFrame(
        {
            "id": [136060, 211333, 1233961],
            "winner_model_a": [1 / 3, 1 / 3, 1 / 3],
            "winner_model_b": [1 / 3, 1 / 3, 1 / 3],
            "winner_tie": [1 / 3, 1 / 3, 1 / 3],
        }
    ).to_csv(sample, index=False)
    pd.DataFrame(
        {
            "id": [136060, 211333, 1233961],
            "winner_model_a": [0.2, 0.3, 0.4],
            "winner_model_b": [0.3, 0.4, 0.3],
            "winner_tie": [0.5, 0.3, 0.3],
        }
    ).to_csv(submission, index=False)
    (context_dir / "overview.md").write_text(
        "This is a Code Competition. The public test set is dummy data and hidden/full test runs in Kaggle.\n",
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="tiny static submission"):
        validate_submission(str(submission), str(sample), data_dir=tmp_path / "data")


def test_validate_submission_rejects_prerelease_sample_as_full_submission(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    sample = context_dir / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    rows = {"id": [f"sample-{index}" for index in range(13)], "label": [0.1] * 13}
    pd.DataFrame(rows).to_csv(sample, index=False)
    pd.DataFrame(rows).to_csv(submission, index=False)
    (context_dir / "data.md").write_text(
        "Currently, only a sample of the training dataset has been released. "
        "The full dataset, including the public training and testing sets, is expected to be released later.\n",
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="tiny static submission"):
        validate_submission(str(submission), str(sample), data_dir=tmp_path / "data")


def test_validate_submission_uses_overview_hint_when_sample_is_header_only(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    context_dir = data_dir / "context"
    cache_dir = data_dir / ".kagglebot_cache"
    context_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sample = cache_dir / "sample_submission_synth.csv"
    sample.write_text("id,target\n", encoding="utf-8")
    (context_dir / "overview.md").write_text(
        "## Submission Format\n\n```csv\nfilename,right_place,prediction_string\n```\n",
        encoding="utf-8",
    )
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"filename": ["0.jpg"], "right_place": [0], "prediction_string": ["-"]}).to_csv(
        submission, index=False
    )

    validate_submission(str(submission), str(sample))


def test_validate_submission_uses_data_md_hint_when_sample_is_header_only(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    context_dir = data_dir / "context"
    cache_dir = data_dir / ".kagglebot_cache"
    context_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sample = cache_dir / "sample_submission_synth.csv"
    sample.write_text("id,prediction\n", encoding="utf-8")
    (context_dir / "data.md").write_text(
        "## Submission Format\n\n"
        "A CSV file with the following columns:\n"
        "* `Id`: The filename\n"
        "* `Category`: The predicted class\n",
        encoding="utf-8",
    )
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"Id": ["x"], "Category": ["Health"]}).to_csv(submission, index=False)

    validate_submission(str(submission), str(sample))


def test_validate_submission_header_only_sample_validates_against_icpr_evaluation_ids(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    eval_root = data_dir / "ICPR02" / "kaggle" / "evaluation"
    for sample_id in ("a0", "b1", "c2"):
        sample_dir = eval_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "B2.tif").write_bytes(b"TIFF")

    sample = tmp_path / "sample_submission.csv"
    sample.write_text("id,prediction\n", encoding="utf-8")
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": ["a0", "b1"], "prediction": [0.1, 0.2]}).to_csv(submission, index=False)

    with pytest.raises(SubmissionValidationError) as exc:
        validate_submission(str(submission), str(sample), data_dir=data_dir)
    message = str(exc.value)
    assert "row count mismatch" in message
    assert "id values mismatch (header-only sample detected; validated against evaluation directory ids)" in message


def test_validate_submission_header_only_sample_accepts_icpr_evaluation_id_set(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    eval_root = data_dir / "ICPR02" / "kaggle" / "evaluation"
    for sample_id in ("a0", "b1", "c2"):
        sample_dir = eval_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "B2.tif").write_bytes(b"TIFF")

    sample = tmp_path / "sample_submission.csv"
    sample.write_text("id,prediction\n", encoding="utf-8")
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": ["c2", "a0", "b1"], "prediction": [0.3, 0.1, 0.2]}).to_csv(submission, index=False)

    validate_submission(str(submission), str(sample), data_dir=data_dir)


def test_validate_submission_header_only_sample_validates_against_jsonl_test_ids(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.jsonl").write_text(
        '{"id": "a", "feature": 1, "target": 0}\n{"id": "b", "feature": 2, "target": 1}\n',
        encoding="utf-8",
    )
    (data_dir / "test.jsonl").write_text(
        '{"id": "a", "feature": 10}\n{"id": "b", "feature": 20}\n{"id": "c", "feature": 30}\n',
        encoding="utf-8",
    )
    sample = data_dir / "sample_submission.csv"
    sample.write_text("id,prediction\n", encoding="utf-8")
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": ["a", "b"], "prediction": [0.1, 0.2]}).to_csv(submission, index=False)

    with pytest.raises(SubmissionValidationError) as exc:
        validate_submission(str(submission), str(sample), data_dir=data_dir)

    message = str(exc.value)
    assert "row count mismatch" in message
    assert "id values mismatch (header-only sample detected; validated against evaluation directory ids)" not in message
    assert "id values mismatch (header-only sample detected; validated against test data ids)" in message


def test_validate_submission_rejects_missing_required_id_suffix_when_inferred(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    data_dir = tmp_path / "data"
    (data_dir / "Kaggle_Prepared" / "val" / "MS").mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)
    sample = context_dir / "sample_submission.csv"
    sample.write_text("id,prediction\n", encoding="utf-8")
    (context_dir / "data.md").write_text(
        "## Submission Format\n\n"
        "A CSV file with the following columns:\n"
        "* `Id`: The filename (e.g., `val_a1b2c3d4.tif`)\n"
        "* `Category`: The predicted class\n",
        encoding="utf-8",
    )
    for stem in ("val_0001", "val_0002"):
        (data_dir / "Kaggle_Prepared" / "val" / "MS" / f"{stem}.tif").write_bytes(b"TIFF")

    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": ["val_0001", "val_0002"], "prediction": ["Health", "Rust"]}).to_csv(submission, index=False)

    with pytest.raises(SubmissionValidationError, match="require '\\.tif' suffix"):
        validate_submission(str(submission), str(sample), data_dir=data_dir)


def test_validate_submission_accepts_inferred_required_id_suffix_when_present(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    data_dir = tmp_path / "data"
    (data_dir / "Kaggle_Prepared" / "val" / "MS").mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)
    sample = context_dir / "sample_submission.csv"
    sample.write_text("id,prediction\n", encoding="utf-8")
    (context_dir / "data.md").write_text(
        "## Submission Format\n\n"
        "A CSV file with the following columns:\n"
        "* `Id`: The filename (e.g., `val_a1b2c3d4.tif`)\n"
        "* `Category`: The predicted class\n",
        encoding="utf-8",
    )
    for stem in ("val_0001", "val_0002"):
        (data_dir / "Kaggle_Prepared" / "val" / "MS" / f"{stem}.tif").write_bytes(b"TIFF")

    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": ["val_0001.tif", "val_0002.tif"], "prediction": ["Health", "Rust"]}).to_csv(
        submission, index=False
    )

    validate_submission(str(submission), str(sample), data_dir=data_dir)


def test_validate_submission_rejects_missing_compound_required_id_suffix_when_inferred(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "scans" / "test").mkdir(parents=True, exist_ok=True)
    for stem in ("case_001", "case_002"):
        (data_dir / "scans" / "test" / f"{stem}.nii.gz").write_bytes(b"scan")

    sample = tmp_path / "sample_submission.csv"
    sample.write_text("id,target\n", encoding="utf-8")
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": ["case_001", "case_002"], "target": [0.1, 0.2]}).to_csv(submission, index=False)

    with pytest.raises(SubmissionValidationError, match="require '\\.nii\\.gz' suffix"):
        validate_submission(str(submission), str(sample), data_dir=data_dir)


def test_validate_submission_rejects_partial_compound_required_id_suffix_when_inferred(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "scans" / "test").mkdir(parents=True, exist_ok=True)
    for stem in ("case_001", "case_002"):
        (data_dir / "scans" / "test" / f"{stem}.nii.gz").write_bytes(b"scan")

    sample = tmp_path / "sample_submission.csv"
    sample.write_text("id,target\n", encoding="utf-8")
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": ["case_001.nii", "case_002.nii"], "target": [0.1, 0.2]}).to_csv(
        submission,
        index=False,
    )

    with pytest.raises(SubmissionValidationError, match="require '\\.nii\\.gz' suffix"):
        validate_submission(str(submission), str(sample), data_dir=data_dir)


def test_validate_submission_accepts_inferred_compound_required_id_suffix_when_present(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "scans" / "test").mkdir(parents=True, exist_ok=True)
    for stem in ("case_001", "case_002"):
        (data_dir / "scans" / "test" / f"{stem}.nii.gz").write_bytes(b"scan")

    sample = tmp_path / "sample_submission.csv"
    sample.write_text("id,target\n", encoding="utf-8")
    submission = tmp_path / "submission.csv"
    pd.DataFrame(
        {
            "id": ["case_001.nii.gz", "case_002.nii.gz"],
            "target": [0.1, 0.2],
        }
    ).to_csv(submission, index=False)

    validate_submission(str(submission), str(sample), data_dir=data_dir)


def test_validate_submission_prefers_context_compound_id_suffix_when_ambiguous(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    data_dir = tmp_path / "data"
    (data_dir / "scans" / "test").mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("case_001", "case_002"):
        (data_dir / "scans" / "test" / f"{stem}.nii.gz").write_bytes(b"scan")
        (data_dir / "scans" / "test" / f"{stem}.nrrd").write_bytes(b"scan")

    sample = context_dir / "sample_submission.csv"
    sample.write_text("id,target\n", encoding="utf-8")
    (context_dir / "data.md").write_text(
        "## Submission Format\n\nThe id column must contain the full scan filename, for example `case_001.nii.gz`.\n",
        encoding="utf-8",
    )
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": ["case_001", "case_002"], "target": [0.1, 0.2]}).to_csv(submission, index=False)

    with pytest.raises(SubmissionValidationError, match="require '\\.nii\\.gz' suffix"):
        validate_submission(str(submission), str(sample), data_dir=data_dir)


@pytest.mark.parametrize("suffix", [".csv", ".csv.gz", ".jsonl.zst", ".parquet", ".xlsx", ".sqlite3", ".py"])
def test_validate_submission_does_not_infer_required_id_suffix_from_non_asset_files(
    tmp_path: Path,
    suffix: str,
) -> None:
    data_dir = tmp_path / "data"
    test_dir = data_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("case_001", "case_002"):
        (test_dir / f"{stem}{suffix}").write_text("not an asset\n", encoding="utf-8")

    sample = tmp_path / "sample_submission.csv"
    sample.write_text("id,target\n", encoding="utf-8")
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": ["case_001", "case_002"], "target": [0.1, 0.2]}).to_csv(submission, index=False)

    validate_submission(str(submission), str(sample), data_dir=data_dir)


def test_validate_submission_does_not_infer_suffix_when_real_sample_ids_are_suffixless(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "test_set").mkdir(parents=True, exist_ok=True)
    for stem in ("0", "1", "2"):
        (data_dir / "test_set" / f"{stem}.png").write_bytes(b"PNG")

    sample = tmp_path / "sample_submission.csv"
    sample.write_text(
        "id,image_id,prediction_string\n0,0,0.9 1 2 3 4\n1,1, \n2,2,0.8 5 6 7 8\n",
        encoding="utf-8",
    )
    submission = tmp_path / "submission.csv"
    submission.write_text(
        "id,image_id,prediction_string\n0,0,0.9 1 2 3 4\n1,1, \n2,2,0.8 5 6 7 8\n",
        encoding="utf-8",
    )

    validate_submission(str(submission), str(sample), data_dir=data_dir)


def test_validate_submission_does_not_infer_suffix_when_jsonl_sample_ids_are_suffixless(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "test_set").mkdir(parents=True, exist_ok=True)
    for stem in ("0", "1", "2"):
        (data_dir / "test_set" / f"{stem}.png").write_bytes(b"PNG")

    sample = tmp_path / "sample_submission.jsonl"
    sample.write_text(
        '{"id":"0","image_id":"0","prediction_string":"0.9 1 2 3 4"}\n'
        '{"id":"1","image_id":"1","prediction_string":" "}\n'
        '{"id":"2","image_id":"2","prediction_string":"0.8 5 6 7 8"}\n',
        encoding="utf-8",
    )
    submission = tmp_path / "submission.jsonl"
    submission.write_text(
        '{"id":"0","image_id":"0","prediction_string":"0.9 1 2 3 4"}\n'
        '{"id":"1","image_id":"1","prediction_string":" "}\n'
        '{"id":"2","image_id":"2","prediction_string":"0.8 5 6 7 8"}\n',
        encoding="utf-8",
    )

    validate_submission(str(submission), str(sample), data_dir=data_dir)


def test_validate_submission_checks_overview_hint_not_only_sample(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    context_dir = data_dir / "context"
    cache_dir = data_dir / ".kagglebot_cache"
    context_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sample = cache_dir / "sample_submission_synth.csv"
    sample.write_text("id,target\n", encoding="utf-8")
    (context_dir / "overview.md").write_text(
        "## Submission Format\n\n```csv\nfilename,right_place,prediction_string\n```\n",
        encoding="utf-8",
    )
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1], "target": [0.5]}).to_csv(submission, index=False)

    with pytest.raises(SubmissionValidationError, match="expected \\(submission_format/overview hint\\)"):
        validate_submission(str(submission), str(sample))


def test_validate_submission_ignores_overview_rules_text_with_commas(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    sample = context_dir / "sample_submission.csv"
    sample.write_text("id,prediction\n", encoding="utf-8")
    (context_dir / "overview.md").write_text(
        "## Submission Code Requirements\n"
        "a. Private Code Sharing. Unless otherwise specifically permitted under the Competition Website or "
        "Competition Specific Rules above, during the Competition Period, you are not allowed to privately share "
        "source or executable code developed in connection with or based upon the Competition Data.</h5>\n",
        encoding="utf-8",
    )
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": ["X"], "prediction": [0.1]}).to_csv(submission, index=False)

    validate_submission(str(submission), str(sample))


def test_validate_submission_ignores_submission_format_rules_text_with_commas(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    sample = context_dir / "sample_submission.csv"
    sample.write_text("id,prediction\n", encoding="utf-8")
    (context_dir / "submission_format.md").write_text(
        "#### 6. SUBMISSION CODE REQUIREMENTS\n"
        "a. Private Code Sharing, during the Competition Period, you are not allowed to privately share "
        "source or executable code developed in connection with or based upon the Competition Data.\n",
        encoding="utf-8",
    )
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": ["X"], "prediction": [0.1]}).to_csv(submission, index=False)

    validate_submission(str(submission), str(sample))


def test_validate_submission_prefers_real_submission_section_over_rules_heading(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    context_dir = data_dir / "context"
    cache_dir = data_dir / ".kagglebot_cache"
    context_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sample = cache_dir / "sample_submission_synth.csv"
    sample.write_text("id,target\n", encoding="utf-8")
    (context_dir / "overview.md").write_text(
        "#### 6. SUBMISSION CODE REQUIREMENTS\n"
        "a. Private Code Sharing. Unless otherwise specifically permitted under the Competition Website or "
        "Competition Specific Rules above, during the Competition Period, you are not allowed to privately share "
        "source or executable code developed in connection with or based upon the Competition Data.</h5>\n\n"
        "## Submission\n\n"
        "```csv\n"
        "id,prediction\n"
        "```\n",
        encoding="utf-8",
    )
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": ["X"], "prediction": [0.1]}).to_csv(submission, index=False)

    validate_submission(str(submission), str(sample))


def test_validate_submission_allows_documented_targets_with_leading_anchor_for_placeholder_sample(
    tmp_path: Path,
) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    sample = context_dir / "sample_submission.csv"
    sample.write_text("id,prediction\n", encoding="utf-8")
    (context_dir / "overview.md").write_text(
        "## Submission Format\n\n"
        "Participants should submit their files in CSV format. "
        "Each submission must include the columns KEEP, ASSOCIATION, and DIFF.\n",
        encoding="utf-8",
    )
    submission = tmp_path / "submission.csv"
    pd.DataFrame(
        {
            "Condition": ["Epistaxis", "Intracranial Pressure"],
            "KEEP": ["R04", "H47"],
            "ASSOCIATION": ["Not Applicable", "G93"],
            "DIFF": ["Not Applicable", "I10"],
        }
    ).to_csv(submission, index=False)

    validate_submission(str(submission), str(sample))


def test_validate_submission_allows_cohortx_targets_with_pmcids_anchor_for_placeholder_sample(
    tmp_path: Path,
) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    sample = context_dir / "sample_submission.csv"
    sample.write_text("id,prediction\n", encoding="utf-8")
    (context_dir / "submission_format.md").write_text(
        "## Submission\n\n"
        "For each row, use the pmcids value and populate the following fields:\n\n"
        "- `conditions` – a list of diseases\n"
        "- `study_type` – a string describing the study type\n"
        "- `sex` – a string indicating the sex of participants\n"
        "- `minimum_age` – a string representing the minimum age\n"
        "- `maximum_age` – a string representing the maximum age\n"
        "- `eligibility_criteria` – a text field containing the eligibility criteria\n",
        encoding="utf-8",
    )
    submission = tmp_path / "submission.csv"
    pd.DataFrame(
        {
            "pmcids": ["11452962", "11731389"],
            "conditions": ["Neurodevelopment", "Lymphoma"],
            "study_type": ["INTERVENTIONAL", "INTERVENTIONAL"],
            "sex": ["ALL", "ALL"],
            "minimum_age": ["2 Years", "18 Years"],
            "maximum_age": ["12 Years", "85 Years"],
            "eligibility_criteria": ["Eligible children", "Eligible adults"],
        }
    ).to_csv(submission, index=False)

    validate_submission(str(submission), str(sample))


def test_validate_submission_rejects_placeholder_wrapper_when_context_requires_wide_targets(
    tmp_path: Path,
) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    sample = context_dir / "sample_submission.csv"
    sample.write_text("id,prediction\n", encoding="utf-8")
    (context_dir / "overview.md").write_text(
        "## Submission Format\n\nEach submission must include the columns KEEP, ASSOCIATION, and DIFF.\n",
        encoding="utf-8",
    )
    submission = tmp_path / "submission.csv"
    pd.DataFrame({"id": [0], "prediction": ["R04 | Not Applicable | Not Applicable"]}).to_csv(
        submission,
        index=False,
    )

    with pytest.raises(SubmissionValidationError, match="expected \\(submission_format/overview hint\\)"):
        validate_submission(str(submission), str(sample))


def test_validate_submission_sniffs_tab_delimiter_and_flags_missing_header(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"

    pd.DataFrame({"col0": ["P1", "P2"], "col1": ["T1", "T2"], "col2": [0.0, 0.0]}).to_csv(sample, index=False)
    submission.write_text("P1\tT1\t0.9\nP2\tT2\t0.8\n", encoding="utf-8")

    with pytest.raises(SubmissionValidationError) as exc:
        validate_submission(str(submission), str(sample))
    message = str(exc.value)
    assert "columns mismatch" in message
    assert "missing a header row" in message


@pytest.mark.parametrize(
    ("stderr_text", "expected_kind", "expected_reason"),
    [
        (
            "Submission not allowed: This competition only accepts Submissions from Notebooks.",
            "permanent",
            "notebook_only_submission_required",
        ),
        (
            "Code competition submissions require both the output file name and the version label.",
            "permanent",
            "notebook_submit_argument_missing",
        ),
        (
            "kernel must be specified as <owner>/<notebook>",
            "permanent",
            "notebook_submit_argument_missing",
        ),
        ("400 Client Error: Bad Request for url: https://www.kaggle.com/api/v1/...", "permanent", "bad_request"),
        ("You must accept the rules before submitting", "permanent", "rules_not_accepted"),
        ("No Kaggle API credentials found", "permanent", "authentication"),
        ("Unauthorized (401)", "permanent", "authentication"),
        ("Kernel push error: Notebook not found", "permanent", "kernel_push_failed"),
        ("Kaggle kernel not found after push; aborting.", "permanent", "kernel_push_failed"),
        ("Competition is not accepting submissions", "permanent", "competition_unavailable"),
        ("Submission limit reached: maximum number of submissions", "permanent", "submission_limit"),
        (
            "Submission not allowed: Your team has used its daily Submission allowance (10) today.",
            "permanent",
            "submission_limit",
        ),
        ("ConnectionError: temporarily unavailable (503)", "transient", "network_or_timeout"),
        ("Bad Gateway (502)", "transient", "network_or_timeout"),
        ("Gateway Timeout 504", "transient", "network_or_timeout"),
    ],
)
def test_classify_submit_error_examples(stderr_text: str, expected_kind: str, expected_reason: str) -> None:
    classified = classify_submit_error("", stderr_text, 1)
    assert classified["kind"] == expected_kind
    assert classified["reason"] == expected_reason
    if expected_kind == "transient":
        assert classified["retry_after_seconds"] == 2
    else:
        assert classified["retry_after_seconds"] is None


def test_classify_submit_error_unknown() -> None:
    classified = classify_submit_error("", "some uncategorized cli message", 3)
    assert classified["kind"] == "unknown"
    assert classified["reason"] == "unclassified_submit_error"
    assert classified["retry_after_seconds"] is None


def test_classify_submit_error_ambiguous_notebook_bad_request() -> None:
    classified = classify_submit_error(
        "",
        (
            "400 Client Error: Bad Request for url: "
            "https://www.kaggle.com/api/v1/competitions/submissions/submit-notebook/"
            "deep-past-initiative-machine-translation"
        ),
        1,
    )
    assert classified["kind"] == "unknown"
    assert classified["reason"] == "ambiguous_notebook_bad_request"
    assert classified["retry_after_seconds"] == 3


def test_normalize_and_fingerprint_are_stable() -> None:
    a = (
        "Error at /home/user/repo/artifacts/demo/runs/20260101T000000Z-abcd1234: "
        "timeout 2026-02-15T12:00:00Z on 2026-02-15"
    )
    b = (
        "Error at /home/other/repo/artifacts/demo/runs/20260101T000000Z-efef2222: "
        "timeout 2026-02-16T12:00:00Z on 2026-02-16"
    )
    na = normalize_error_text(a)
    normalize_error_text(b)
    assert "<PATH>" in na or "<ARTIFACT_PATH>" in na
    assert "<DATETIME>" in na
    assert "<DATE>" in na
    assert compute_error_fingerprint(a, "") == compute_error_fingerprint(b, "")


def test_run_kaggle_submit_captures_stdout_stderr(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_subprocess_run(*args, **kwargs):  # noqa: ARG001
        captured.update(kwargs)
        return subprocess.CompletedProcess(
            args=["kaggle", "competitions", "submit"],
            returncode=0,
            stdout="submit ok",
            stderr="warning line",
        )

    monkeypatch.setattr("kagglebot.submission.guard.subprocess.run", fake_subprocess_run)
    result = run_kaggle_submit(slug="demo", submission_file=Path("submission.csv"), message="m")
    assert result.returncode == 0
    assert result.stdout == "submit ok"
    assert result.stderr == "warning line"
    assert result.command[:3] == ["kaggle", "competitions", "submit"]
    assert "-q" in result.command
    assert result.duration_sec >= 0.0
    assert captured["timeout"] == 300.0


def test_run_kaggle_submit_timeout_env_uses_shared_number_parsing(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_subprocess_run(*args, **kwargs):  # noqa: ARG001
        captured.update(kwargs)
        return subprocess.CompletedProcess(
            args=["kaggle", "competitions", "submit"],
            returncode=0,
            stdout="submit ok",
            stderr="",
        )

    monkeypatch.setattr("kagglebot.submission.guard.subprocess.run", fake_subprocess_run)

    monkeypatch.setenv("KAGGLEBOT_SUBMIT_TIMEOUT_SEC", "0")
    run_kaggle_submit(slug="demo", submission_file=Path("submission.csv"), message="m")
    assert captured["timeout"] == 1.0

    monkeypatch.setenv("KAGGLEBOT_SUBMIT_TIMEOUT_SEC", "nan")
    run_kaggle_submit(slug="demo", submission_file=Path("submission.csv"), message="m")
    assert captured["timeout"] == 300.0


def test_run_kaggle_submit_failure_includes_tails_and_returncode(monkeypatch) -> None:
    long_stdout = "X" * 8000
    long_stderr = "\n".join(f"stderr line {idx}" for idx in range(300))

    def fake_subprocess_run(*args, **kwargs):  # noqa: ARG001
        return subprocess.CompletedProcess(
            args=["kaggle", "competitions", "submit"],
            returncode=2,
            stdout=long_stdout,
            stderr=long_stderr,
        )

    monkeypatch.setattr("kagglebot.submission.guard.subprocess.run", fake_subprocess_run)
    with pytest.raises(SubmissionCliError) as exc:
        run_kaggle_submit(slug="demo", submission_file=Path("submission.csv"), message="m")
    err = exc.value
    assert err.exit_code == 2
    assert err.command[:3] == ["kaggle", "competitions", "submit"]
    assert "-q" in err.command
    assert len(err.stdout) <= 6000
    assert "stderr line 299" in err.stderr
    assert "stderr line 0" not in err.stderr


def test_run_kaggle_submit_timeout_raises_cli_error(monkeypatch) -> None:
    def fake_subprocess_run(*args, **kwargs):  # noqa: ARG001
        raise subprocess.TimeoutExpired(
            cmd=["kaggle", "competitions", "submit"],
            timeout=1.5,
            output="partial stdout",
            stderr="partial stderr",
        )

    monkeypatch.setattr("kagglebot.submission.guard.subprocess.run", fake_subprocess_run)
    with pytest.raises(SubmissionCliError) as exc:
        run_kaggle_submit(slug="demo", submission_file=Path("submission.csv"), message="m")

    assert exc.value.exit_code == 124
    assert "timed out after 1.5s" in str(exc.value)
    assert "partial stdout" in exc.value.stdout
    assert "partial stderr" in exc.value.stderr


def test_run_kaggle_submit_kernel_uses_kernel_flag(monkeypatch) -> None:
    def fake_subprocess_run(*args, **kwargs):  # noqa: ARG001
        return subprocess.CompletedProcess(
            args=["kaggle", "competitions", "submit"],
            returncode=0,
            stdout="submit ok",
            stderr="",
        )

    monkeypatch.setattr("kagglebot.submission.guard.subprocess.run", fake_subprocess_run)
    result = run_kaggle_submit_kernel(
        slug="demo",
        kernel="user/demo-kernel",
        message="m",
        output_file="submission.csv",
        version="1",
    )
    assert result.returncode == 0
    assert result.command[:3] == ["kaggle", "competitions", "submit"]
    assert "-k" in result.command
    assert "user/demo-kernel" in result.command


def test_run_kaggle_submit_kernel_supports_output_and_version(monkeypatch) -> None:
    def fake_subprocess_run(*args, **kwargs):  # noqa: ARG001
        return subprocess.CompletedProcess(
            args=["kaggle", "competitions", "submit"],
            returncode=0,
            stdout="submit ok",
            stderr="",
        )

    monkeypatch.setattr("kagglebot.submission.guard.subprocess.run", fake_subprocess_run)
    result = run_kaggle_submit_kernel(
        slug="demo",
        kernel="user/demo-kernel",
        message="m",
        output_file="submission.csv",
        version="3",
    )
    assert "-k" in result.command
    assert "user/demo-kernel" in result.command
    assert "-f" in result.command
    assert "submission.csv" in result.command
    assert "-v" in result.command
    assert "3" in result.command


@pytest.mark.parametrize(
    "output_file",
    [
        "/data/run/submission.parquet",
        "nested/submission.parquet",
        "test_array_mask.npy",
        "candidate_mask.npy",
        "oof_predictions.npy",
        "test_preds_model.npy",
    ],
)
def test_run_kaggle_submit_kernel_rejects_invalid_code_output_before_cli(
    monkeypatch,
    output_file: str,
) -> None:
    invoked = False

    def fake_subprocess_run(*args, **kwargs):  # noqa: ARG001
        nonlocal invoked
        invoked = True
        raise AssertionError("Kaggle CLI must not be invoked")

    monkeypatch.setattr("kagglebot.submission.guard.subprocess.run", fake_subprocess_run)
    with pytest.raises(SubmissionCliError, match="local code-output validation") as exc:
        run_kaggle_submit_kernel(
            slug="demo",
            kernel="user/demo-kernel",
            message="m",
            output_file=output_file,
            expected_output_file="submission.parquet",
            version="2",
        )

    assert invoked is False
    classification = classify_submit_error(exc.value.stdout, exc.value.stderr, exc.value.exit_code)
    assert classification["reason"] == "invalid_code_submission_output"
    assert classification["kind"] == "validation"


def test_run_kaggle_submit_kernel_requires_exact_expected_output(monkeypatch) -> None:
    monkeypatch.setattr(
        "kagglebot.submission.guard.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("CLI must not run")),
    )

    with pytest.raises(SubmissionCliError, match="local code-output validation"):
        run_kaggle_submit_kernel(
            slug="demo",
            kernel="user/demo-kernel",
            message="m",
            output_file="submission.csv",
            expected_output_file="submission.parquet",
            version="2",
        )
