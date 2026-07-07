from __future__ import annotations

import bz2
import gzip
import io
import json
import lzma
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import zstandard as zstd

from kagglebot.exceptions import SubmissionValidationError
from kagglebot.solver.io import read_table
from kagglebot.submission.validate import validate_submission
from kagglebot.submission_format import SubmissionFormatHint
from kagglebot.submission_service import SubmissionConfig, SubmissionService


def _build_service(tmp_path: Path) -> tuple[SubmissionService, Path]:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    sample_path = context_dir / "sample_submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_csv(sample_path, index=False)
    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=tmp_path / "data",
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    return service, context_dir


def _write_payload_for_suffix(path: Path, payload: bytes = b"single-file-payload") -> None:
    if path.name.endswith(".gz"):
        path.write_bytes(gzip.compress(payload))
    elif path.name.endswith(".bz2"):
        path.write_bytes(bz2.compress(payload))
    elif path.name.endswith(".xz"):
        path.write_bytes(lzma.compress(payload))
    elif path.name.endswith(".zst"):
        path.write_bytes(zstd.ZstdCompressor().compress(payload))
    else:
        path.write_bytes(payload)


def test_validate_and_prepare_submission_converts_csv_to_tsv_when_required(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nSubmit a TSV file.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]}).to_csv(submission_path, index=False)

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.suffix == ".tsv"
    validate_submission(str(prepared), str(context_dir / "sample_submission.csv"))


def test_write_tabular_submission_stabilizes_problematic_columns(tmp_path: Path) -> None:
    service, _context_dir = _build_service(tmp_path)
    destination = tmp_path / "submission.tsv"
    frame = pd.DataFrame([[1, "-", 0.1, 0.2]], columns=["id", "", "score", "score"])

    service._write_tabular_submission(
        frame=frame,
        destination=destination,
        target_suffix=".tsv",
        format_hint=None,
    )

    loaded = pd.read_csv(destination, sep="\t")

    assert list(loaded.columns) == ["id", "column_2", "score", "score_1"]
    assert list(frame.columns) == ["id", "", "score", "score"]


def test_write_txt_submission_uses_format_hint_delimiter(tmp_path: Path) -> None:
    service, _context_dir = _build_service(tmp_path)
    destination = tmp_path / "submission.txt"
    frame = pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]})

    service._write_tabular_submission(
        frame=frame,
        destination=destination,
        target_suffix=".txt",
        format_hint=SubmissionFormatHint(columns=None, delimiter=";", expected_suffixes=None),
    )

    assert destination.read_text(encoding="utf-8").splitlines() == ["id;target", "1;0.25", "2;0.75"]


def test_validate_and_prepare_submission_rejects_manifest_tabular_fallback_for_non_tabular_requested_output(
    tmp_path: Path,
) -> None:
    service, _context_dir = _build_service(tmp_path)
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]}).to_csv(submission_path, index=False)
    (tmp_path / "submission_manifest.json").write_text(
        json.dumps(
            {
                "artifact_class": "tabular",
                "submission_path": "submission.csv",
                "requested_output_path": "answers.nii.gz",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="tabular fallback for a non-tabular requested output"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_uses_run_specific_manifest_for_stored_artifact(
    tmp_path: Path,
) -> None:
    service, _context_dir = _build_service(tmp_path)
    submission_path = tmp_path / "run-123_submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]}).to_csv(submission_path, index=False)
    (tmp_path / "run-123_submission_manifest.json").write_text(
        json.dumps(
            {
                "artifact_class": "tabular",
                "submission_path": "run-123_submission.csv",
                "requested_output_path": "answers.nii.gz",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="tabular fallback for a non-tabular requested output"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_uses_run_specific_manifest_metadata_without_submission_path(
    tmp_path: Path,
) -> None:
    service, _context_dir = _build_service(tmp_path)
    submission_path = tmp_path / "run-123_submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]}).to_csv(submission_path, index=False)
    (tmp_path / "run-123_submission_manifest.json").write_text(
        json.dumps(
            {
                "artifact_class": "tabular",
                "requested_output_path": "answers.nii.gz",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="tabular fallback for a non-tabular requested output"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_infers_class_for_run_specific_manifest_without_artifact_class(
    tmp_path: Path,
) -> None:
    service, _context_dir = _build_service(tmp_path)
    submission_path = tmp_path / "run-123_submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]}).to_csv(submission_path, index=False)
    (tmp_path / "run-123_submission_manifest.json").write_text(
        json.dumps(
            {
                "requested_output_path": "answers.nii.gz",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="tabular fallback for a non-tabular requested output"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_tabular_file_even_when_manifest_claims_single_file(
    tmp_path: Path,
) -> None:
    service, _context_dir = _build_service(tmp_path)
    submission_path = tmp_path / "run-123_submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]}).to_csv(submission_path, index=False)
    (tmp_path / "run-123_submission_manifest.json").write_text(
        json.dumps(
            {
                "artifact_class": "single_file",
                "requested_output_path": "answers.nii.gz",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="tabular fallback for a non-tabular requested output"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_run_specific_manifest_for_other_file(
    tmp_path: Path,
) -> None:
    service, _context_dir = _build_service(tmp_path)
    submission_path = tmp_path / "run-123_submission.csv"
    other_path = tmp_path / "other.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]}).to_csv(submission_path, index=False)
    pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}).to_csv(other_path, index=False)
    (tmp_path / "run-123_submission_manifest.json").write_text(
        json.dumps(
            {
                "artifact_class": "tabular",
                "submission_path": "other.csv",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="run-specific manifest does not reference submitted file"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_invalid_run_specific_manifest(
    tmp_path: Path,
) -> None:
    service, _context_dir = _build_service(tmp_path)
    submission_path = tmp_path / "run-123_submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]}).to_csv(submission_path, index=False)
    (tmp_path / "run-123_submission_manifest.json").write_text("{", encoding="utf-8")

    with pytest.raises(SubmissionValidationError, match="invalid submission manifest"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_uses_run_specific_manifest_for_compound_suffix_artifact(
    tmp_path: Path,
) -> None:
    service, _context_dir = _build_service(tmp_path)
    submission_path = tmp_path / "run-123_submission.csv.gz"
    with gzip.open(submission_path, "wt", encoding="utf-8") as handle:
        handle.write("id,target\n1,0.25\n2,0.75\n")
    (tmp_path / "run-123_submission_manifest.json").write_text(
        json.dumps(
            {
                "artifact_class": "tabular",
                "submission_path": "run-123_submission.csv.gz",
                "requested_output_path": "answers.nii.gz",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="tabular fallback for a non-tabular requested output"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_ignores_nested_unrelated_manifest_for_file(
    tmp_path: Path,
) -> None:
    service, context_dir = _build_service(tmp_path)
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]}).to_csv(submission_path, index=False)
    nested = tmp_path / "other" / "submission_manifest.json"
    nested.parent.mkdir()
    nested.write_text(
        json.dumps(
            {
                "artifact_class": "tabular",
                "submission_path": "other.csv",
                "requested_output_path": "answers.nii.gz",
            }
        ),
        encoding="utf-8",
    )

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == submission_path
    validate_submission(str(prepared), str(context_dir / "sample_submission.csv"))


def test_validate_and_prepare_submission_ignores_same_directory_manifest_for_other_file(
    tmp_path: Path,
) -> None:
    service, context_dir = _build_service(tmp_path)
    submission_path = tmp_path / "submission.csv"
    other_path = tmp_path / "other.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]}).to_csv(submission_path, index=False)
    pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}).to_csv(other_path, index=False)
    (tmp_path / "submission_manifest.json").write_text(
        json.dumps(
            {
                "artifact_class": "tabular",
                "submission_path": "other.csv",
                "requested_output_path": "answers.nii.gz",
            }
        ),
        encoding="utf-8",
    )

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == submission_path
    validate_submission(str(prepared), str(context_dir / "sample_submission.csv"))


def test_validate_and_prepare_submission_accepts_compressed_csv(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    submission_path = tmp_path / "submission.csv.gz"
    with gzip.open(submission_path, "wt", encoding="utf-8") as handle:
        handle.write("id,target\n1,0.25\n2,0.75\n")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == submission_path
    validate_submission(str(prepared), str(context_dir / "sample_submission.csv"))


def test_validate_and_prepare_submission_accepts_zstd_compressed_csv(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    submission_path = tmp_path / "submission.csv.zst"
    submission_path.write_bytes(zstd.ZstdCompressor().compress(b"id,target\n1,0.25\n2,0.75\n"))

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == submission_path
    validate_submission(str(prepared), str(context_dir / "sample_submission.csv"))


def test_validate_and_prepare_submission_validates_excel_as_tabular(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    submission_path = tmp_path / "submission.xlsx"
    pd.DataFrame({"id": [1], "target": [0.25]}).to_excel(submission_path, index=False)

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == "submission.autofixed.xlsx"
    validate_submission(str(prepared), str(context_dir / "sample_submission.csv"))
    frame = pd.read_excel(prepared)
    assert frame.to_dict("list") == {"id": [1, 2], "target": [0.25, 0.0]}


def test_validate_and_prepare_submission_renames_tab_delimited_compressed_file_to_tsv_gz(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    sample_path = context_dir / "sample_submission.tsv.gz"
    with gzip.open(sample_path, "wt", encoding="utf-8") as handle:
        handle.write("id\ttarget\n1\t0.0\n2\t0.0\n")
    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=tmp_path / "data",
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    submission_path = tmp_path / "submission.csv.gz"
    with gzip.open(submission_path, "wt", encoding="utf-8") as handle:
        handle.write("id\ttarget\n1\t0.25\n2\t0.75\n")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == "submission.tsv.gz"
    validate_submission(str(prepared), str(sample_path))


def test_validate_and_prepare_submission_renames_pipe_delimited_compressed_file_to_psv_gz(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    sample_path = context_dir / "sample_submission.psv.gz"
    with gzip.open(sample_path, "wt", encoding="utf-8") as handle:
        handle.write("id|target\n1|0.0\n2|0.0\n")
    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=tmp_path / "data",
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    submission_path = tmp_path / "submission.csv.gz"
    with gzip.open(submission_path, "wt", encoding="utf-8") as handle:
        handle.write("id|target\n1|0.25\n2|0.75\n")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == "submission.psv.gz"
    validate_submission(str(prepared), str(sample_path))


def test_validate_and_prepare_submission_preserves_tab_suffix_for_tab_delimited_file(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    sample_path = context_dir / "sample_submission.tab"
    sample_path.write_text("id\ttarget\n1\t0.0\n2\t0.0\n", encoding="utf-8")
    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=tmp_path / "data",
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    submission_path = tmp_path / "submission.tab"
    submission_path.write_text("id\ttarget\n1\t0.25\n2\t0.75\n", encoding="utf-8")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == submission_path
    validate_submission(str(prepared), str(sample_path))


def test_validate_and_prepare_submission_converts_csv_to_compressed_csv_when_required(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.csv.gz` with columns id,target.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]}).to_csv(submission_path, index=False)

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == "submission.csv.gz"
    with gzip.open(prepared, "rt", encoding="utf-8") as handle:
        assert handle.readline().strip() == "id,target"
    validate_submission(str(prepared), str(context_dir / "sample_submission.csv"))


def test_validate_and_prepare_submission_converts_csv_to_zstd_csv_when_required(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.csv.zst` with columns id,target.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]}).to_csv(submission_path, index=False)

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == "submission.csv.zst"
    assert pd.read_csv(prepared).to_dict("list") == {"id": [1, 2], "target": [0.25, 0.75]}
    validate_submission(str(prepared), str(context_dir / "sample_submission.csv"))


def test_validate_and_prepare_submission_converts_csv_to_compressed_xml_when_required(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.xml.zst` with columns id,target.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]}).to_csv(submission_path, index=False)

    prepared = service.validate_and_prepare_submission(submission_path)
    loaded = pd.read_xml(
        io.BytesIO(zstd.ZstdDecompressor().decompress(prepared.read_bytes())),
        parser="etree",
    )

    assert prepared.name == "submission.xml.zst"
    assert loaded.to_dict("list") == {"id": [1, 2], "target": [0.25, 0.75]}
    validate_submission(str(prepared), str(context_dir / "sample_submission.csv"))


@pytest.mark.parametrize(
    "suffix",
    [
        ".json.gz",
        ".json.xz",
        ".json.zst",
        ".jsonl.bz2",
        ".jsonl.gz",
        ".jsonl.xz",
        ".jsonl.zst",
        ".jsonlines.bz2",
        ".jsonlines.gz",
        ".jsonlines.xz",
        ".jsonlines.zst",
        ".ndjson.bz2",
        ".ndjson.gz",
        ".ndjson.xz",
        ".ndjson.zst",
    ],
)
def test_validate_and_prepare_submission_converts_csv_to_compressed_json_when_required(
    tmp_path: Path,
    suffix: str,
) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        f"## Submission Format\nUpload `submission{suffix}` with columns id,target.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]}).to_csv(submission_path, index=False)

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == f"submission{suffix}"
    loaded = read_table(prepared)
    assert loaded.to_dict("list") == {"id": [1, 2], "target": [0.25, 0.75]}
    validate_submission(str(prepared), str(context_dir / "sample_submission.csv"))


def test_validate_and_prepare_submission_converts_csv_to_zstd_pickle_when_required(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.pkl.zst` with columns id,target.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]}).to_csv(submission_path, index=False)

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == "submission.pkl.zst"
    assert pd.read_pickle(prepared).to_dict("list") == {"id": [1, 2], "target": [0.25, 0.75]}
    validate_submission(str(prepared), str(context_dir / "sample_submission.csv"))


@pytest.mark.parametrize("suffix", [".pkl", ".pkl.zst"])
def test_validate_and_prepare_submission_accepts_expected_model_pickle_single_file(
    tmp_path: Path,
    suffix: str,
) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        f"## Submission Format\nUpload `submission{suffix}` as a pickle model file for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / f"submission{suffix}"
    if suffix.endswith(".zst"):
        submission_path.write_bytes(zstd.ZstdCompressor().compress(b"not-a-dataframe-pickle-model"))
    else:
        submission_path.write_bytes(b"not-a-dataframe-pickle-model")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == submission_path


@pytest.mark.parametrize("suffix", [".hdf", ".hdf5"])
def test_validate_and_prepare_submission_accepts_expected_hdf_model_single_file(
    tmp_path: Path,
    suffix: str,
) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        f"## Submission Format\nUpload `submission{suffix}` as an HDF5 model file for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / f"submission{suffix}"
    submission_path.write_bytes(b"not-a-dataframe-hdf5-model")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == submission_path


@pytest.mark.parametrize("suffix", [".parq", ".pq"])
def test_validate_and_prepare_submission_converts_csv_to_parquet_alias_when_required(
    tmp_path: Path,
    suffix: str,
) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        f"## Submission Format\nUpload `submission{suffix}` with columns id,target.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]}).to_csv(submission_path, index=False)

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == f"submission{suffix}"
    assert pd.read_parquet(prepared).to_dict("list") == {"id": [1, 2], "target": [0.25, 0.75]}
    validate_submission(str(prepared), str(context_dir / "sample_submission.csv"))


@pytest.mark.parametrize(
    ("suffix", "separator"),
    [
        (".tab", "\t"),
        (".psv", "|"),
    ],
)
def test_validate_and_prepare_submission_converts_csv_to_delimited_text_alias_when_required(
    tmp_path: Path,
    suffix: str,
    separator: str,
) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        f"## Submission Format\nUpload `submission{suffix}` with columns id,target.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]}).to_csv(submission_path, index=False)

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == f"submission{suffix}"
    assert prepared.read_text(encoding="utf-8").splitlines()[0] == separator.join(["id", "target"])
    assert pd.read_csv(prepared, sep=separator).to_dict("list") == {"id": [1, 2], "target": [0.25, 0.75]}
    validate_submission(str(prepared), str(context_dir / "sample_submission.csv"))


@pytest.mark.parametrize("suffix", [".feather", ".ftr", ".arrow", ".ipc"])
def test_validate_and_prepare_submission_converts_csv_to_arrow_ipc_when_required(
    tmp_path: Path,
    suffix: str,
) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        f"## Submission Format\nUpload `submission{suffix}` with columns id,target.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]}).to_csv(submission_path, index=False)

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == f"submission{suffix}"
    assert pd.read_feather(prepared).to_dict("list") == {"id": [1, 2], "target": [0.25, 0.75]}
    validate_submission(str(prepared), str(context_dir / "sample_submission.csv"))


def test_validate_and_prepare_submission_converts_csv_to_orc_when_required(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.orc` with columns id,target.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]}).to_csv(submission_path, index=False)

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == "submission.orc"
    assert pd.read_orc(prepared).to_dict("list") == {"id": [1, 2], "target": [0.25, 0.75]}
    validate_submission(str(prepared), str(context_dir / "sample_submission.csv"))


def test_validate_and_prepare_submission_converts_csv_to_hdf5_when_required(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.hdf5` with columns id,target.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]}).to_csv(submission_path, index=False)

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == "submission.hdf5"
    assert pd.read_hdf(prepared).to_dict("list") == {"id": [1, 2], "target": [0.25, 0.75]}
    validate_submission(str(prepared), str(context_dir / "sample_submission.csv"))


@pytest.mark.parametrize("suffix", [".yaml.xz", ".xml.bz2", ".html.bz2", ".psv.xz", ".tab.zst"])
def test_validate_and_prepare_submission_converts_csv_to_compressed_structured_tabular_when_required(
    tmp_path: Path,
    suffix: str,
) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        f"## Submission Format\nUpload `submission{suffix}` with columns id,target.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]}).to_csv(submission_path, index=False)

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == f"submission{suffix}"
    assert read_table(prepared).to_dict("list") == {"id": [1, 2], "target": [0.25, 0.75]}
    validate_submission(str(prepared), str(context_dir / "sample_submission.csv"))


@pytest.mark.parametrize("suffix", [".yaml.xz", ".yml.zst"])
def test_validate_and_prepare_submission_writes_compressed_yaml_payload_when_required(
    tmp_path: Path,
    suffix: str,
) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        f"## Submission Format\nUpload `submission{suffix}` with columns id,target.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]}).to_csv(submission_path, index=False)

    prepared = service.validate_and_prepare_submission(submission_path)

    if suffix.endswith(".xz"):
        payload = lzma.open(prepared, "rt", encoding="utf-8").read()
    else:
        payload = zstd.open(prepared, "rt", encoding="utf-8").read()
    assert payload.startswith("- id: 1\n")
    assert not payload.lstrip().startswith("[")
    assert read_table(prepared).to_dict("list") == {"id": [1, 2], "target": [0.25, 0.75]}
    validate_submission(str(prepared), str(context_dir / "sample_submission.csv"))


@pytest.mark.parametrize(
    ("description", "suffix"),
    [
        ("Upload an xz-compressed YAML file with columns id,target.", ".yaml.xz"),
        ("Upload a bzip2-compressed HTML file with columns id,target.", ".html.bz2"),
        ("Upload a zstd-compressed NDJSON file with columns id,target.", ".ndjson.zst"),
        ("Upload an xz-compressed PSV file with columns id,target.", ".psv.xz"),
        ("Upload a zstd-compressed TAB file with columns id,target.", ".tab.zst"),
    ],
)
def test_validate_and_prepare_submission_converts_csv_to_compressed_structured_tabular_from_keywords(
    tmp_path: Path,
    description: str,
    suffix: str,
) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(f"## Submission Format\n{description}\n", encoding="utf-8")
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]}).to_csv(submission_path, index=False)

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == f"submission{suffix}"
    assert read_table(prepared).to_dict("list") == {"id": [1, 2], "target": [0.25, 0.75]}
    validate_submission(str(prepared), str(context_dir / "sample_submission.csv"))


@pytest.mark.parametrize("suffix", [".html", ".html.zst"])
def test_validate_and_prepare_submission_converts_csv_to_html_when_required(
    tmp_path: Path,
    suffix: str,
) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        f"## Submission Format\nUpload `submission{suffix}` with columns id,target.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]}).to_csv(submission_path, index=False)

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == f"submission{suffix}"
    if suffix.endswith(".zst"):
        html = zstd.ZstdDecompressor().decompress(prepared.read_bytes()).decode("utf-8")
        loaded = pd.read_html(io.StringIO(html))[0]
    else:
        loaded = pd.read_html(prepared)[0]
    assert loaded.to_dict("list") == {"id": [1, 2], "target": [0.25, 0.75]}
    validate_submission(str(prepared), str(context_dir / "sample_submission.csv"))


@pytest.mark.parametrize("suffix", [".xls", ".xlsm", ".xlsx", ".ods"])
def test_validate_and_prepare_submission_converts_csv_to_excel_when_required(tmp_path: Path, suffix: str) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        f"## Submission Format\nUpload `submission{suffix}` with columns id,target.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.25, 0.75]}).to_csv(submission_path, index=False)

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == f"submission{suffix}"
    assert pd.read_excel(prepared).to_dict("list") == {"id": [1, 2], "target": [0.25, 0.75]}
    validate_submission(str(prepared), str(context_dir / "sample_submission.csv"))


@pytest.mark.parametrize("suffix", [".db", ".sqlite", ".sqlite3"])
def test_validate_and_prepare_submission_accepts_expected_sqlite_single_file(
    tmp_path: Path,
    suffix: str,
) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        f"## Submission Format\nUpload `submission{suffix}` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / f"submission{suffix}"
    with sqlite3.connect(submission_path) as conn:
        conn.execute("CREATE TABLE predictions (id INTEGER, target REAL)")
        conn.executemany("INSERT INTO predictions VALUES (?, ?)", [(1, 0.25), (2, 0.75)])

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == submission_path


@pytest.mark.parametrize(
    ("suffix", "setup", "message"),
    [
        (".sqlite", "corrupt", "unable to read SQLite submission file"),
        (".sqlite3", "schema_only", "SQLite submission has no user tables or views"),
        (".db", "empty_table", "SQLite submission has no data rows"),
    ],
)
def test_validate_and_prepare_submission_rejects_invalid_sqlite_single_file(
    tmp_path: Path,
    suffix: str,
    setup: str,
    message: str,
) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        f"## Submission Format\nUpload `submission{suffix}` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / f"submission{suffix}"
    if setup == "corrupt":
        submission_path.write_bytes(b"not a sqlite database")
    elif setup == "schema_only":
        with sqlite3.connect(submission_path) as conn:
            conn.execute("PRAGMA user_version = 1")
    else:
        with sqlite3.connect(submission_path) as conn:
            conn.execute("CREATE TABLE predictions (id INTEGER, target REAL)")

    with pytest.raises(SubmissionValidationError, match=message):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_raises_when_expected_format_cannot_be_converted(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nYou must submit a CSV file.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.bin"
    submission_path.write_bytes(b"\x00\x01")

    with pytest.raises(SubmissionValidationError, match="submission file format mismatch"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_accepts_expected_non_tabular_single_file_suffix(tmp_path: Path) -> None:
    suffixes = [
        ".nii.gz",
        ".npy",
        ".npz",
        ".fasta.gz",
        ".graphml.bz2",
        ".mat",
        ".onnx",
        ".ply.zst",
        ".safetensors",
        ".safetensors.index.json",
        ".wav",
        ".ply",
        ".bst",
        ".cbm",
        ".ipynb",
        ".jl",
        ".py",
        ".r",
        ".mlmodel",
        ".pmml",
        ".skops",
        ".ubj",
        ".xgb",
        ".ccp4.zst",
        ".fif.gz",
        ".gff3.zst",
        ".hgt.xz",
        ".mrc.gz",
        ".vcf.gz",
        ".vhdr.zst",
        ".vtp.gz",
    ]
    for index, suffix in enumerate(suffixes):
        case_dir = tmp_path / f"case-{index}"
        service, context_dir = _build_service(case_dir)
        (context_dir / "submission_format.md").write_text(
            f"## Submission Format\nYou must upload `submission{suffix}` for scoring.\n",
            encoding="utf-8",
        )
        submission_path = case_dir / f"submission{suffix}"
        _write_payload_for_suffix(submission_path)

        prepared = service.validate_and_prepare_submission(submission_path)

        assert prepared == submission_path, suffix


@pytest.mark.parametrize(
    ("suffix", "payload"),
    [
        (".fasta.gz", gzip.compress(b"", mtime=0)),
        (".graphml.bz2", bz2.compress(b"")),
        (".nii.xz", lzma.compress(b"")),
        (".ply.zst", zstd.ZstdCompressor().compress(b"")),
    ],
    ids=["gzip", "bzip2", "xz", "zstd"],
)
def test_validate_and_prepare_submission_rejects_empty_compressed_non_tabular_payload(
    tmp_path: Path,
    suffix: str,
    payload: bytes,
) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        f"## Submission Format\nYou must upload `submission{suffix}` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / f"submission{suffix}"
    submission_path.write_bytes(payload)

    with pytest.raises(SubmissionValidationError, match="compressed submission payload is empty"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_corrupt_compressed_non_tabular_payload(
    tmp_path: Path,
) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nYou must upload `submission.ply.zst` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.ply.zst"
    submission_path.write_bytes(b"not-a-zstd-stream")

    with pytest.raises(SubmissionValidationError, match="unable to read compressed submission file"):
        service.validate_and_prepare_submission(submission_path)


@pytest.mark.parametrize(
    ("suffix", "description"),
    [
        (".zarr", "Zarr"),
        (".ome.zarr", "OME-Zarr"),
        (".n5", "N5"),
    ],
)
def test_validate_and_prepare_submission_archives_expected_array_store_directory(
    tmp_path: Path,
    suffix: str,
    description: str,
) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        f"## Submission Format\nUpload `submission{suffix}` for scoring as a {description} store.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / f"submission{suffix}"
    (submission_path / "arrays").mkdir(parents=True)
    (submission_path / "empty_group").mkdir()
    (submission_path / ".zgroup").write_text("{}", encoding="utf-8")
    (submission_path / "arrays" / "0").write_bytes(b"chunk")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / f"submission{suffix}.zip"
    with zipfile.ZipFile(prepared) as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
        dirs = sorted(info.filename for info in archive.infolist() if info.is_dir())
        assert archive.read("arrays/0") == b"chunk"
    assert members == [".zgroup", "arrays/0"]
    assert "empty_group/" in dirs


def test_validate_and_prepare_submission_archives_saved_model_directory(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload a TensorFlow SavedModel directory for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "saved_model"
    variables_dir = submission_path / "variables"
    assets_dir = submission_path / "assets"
    variables_dir.mkdir(parents=True)
    assets_dir.mkdir()
    (submission_path / "saved_model.pb").write_bytes(b"saved-model")
    (variables_dir / "variables.index").write_bytes(b"index")
    (variables_dir / "variables.data-00000-of-00001").write_bytes(b"weights")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "saved_model.zip"
    with zipfile.ZipFile(prepared) as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
        dirs = sorted(info.filename for info in archive.infolist() if info.is_dir())
        assert archive.read("saved_model.pb") == b"saved-model"
    assert members == [
        "saved_model.pb",
        "variables/variables.data-00000-of-00001",
        "variables/variables.index",
    ]
    assert "assets/" in dirs


def test_validate_and_prepare_submission_archives_tensorflow_checkpoint_directory(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload a TensorFlow checkpoint directory for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "checkpoint_model"
    submission_path.mkdir()
    (submission_path / "model.ckpt.index").write_bytes(b"index")
    (submission_path / "model.ckpt.data-00000-of-00001").write_bytes(b"weights")
    (submission_path / "checkpoint").write_text('model_checkpoint_path: "model.ckpt"\n', encoding="utf-8")
    (submission_path / "empty_assets").mkdir()

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "checkpoint_model.zip"
    with zipfile.ZipFile(prepared) as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
        dirs = sorted(info.filename for info in archive.infolist() if info.is_dir())
        assert archive.read("model.ckpt.data-00000-of-00001") == b"weights"
    assert members == [
        "checkpoint",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
    ]
    assert "empty_assets/" in dirs


def test_validate_and_prepare_submission_archives_huggingface_model_directory(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload a Hugging Face model directory for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "model"
    submission_path.mkdir()
    (submission_path / "config.json").write_text('{"architectures": ["DemoModel"]}\n', encoding="utf-8")
    (submission_path / "model.safetensors").write_bytes(b"weights")
    (submission_path / "tokenizer.json").write_text('{"version": "1.0"}\n', encoding="utf-8")
    (submission_path / "empty_cache").mkdir()

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "model.zip"
    with zipfile.ZipFile(prepared) as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
        dirs = sorted(info.filename for info in archive.infolist() if info.is_dir())
        assert archive.read("model.safetensors") == b"weights"
    assert members == ["config.json", "model.safetensors", "tokenizer.json"]
    assert "empty_cache/" in dirs


def test_validate_and_prepare_submission_archives_mlflow_model_directory(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload an MLflow model directory for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "mlflow_model"
    data_dir = submission_path / "data"
    data_dir.mkdir(parents=True)
    (submission_path / "MLmodel").write_text(
        "flavors:\n  python_function:\n    data: data/model.skops\n",
        encoding="utf-8",
    )
    (data_dir / "model.skops").write_bytes(b"skops-model")
    (submission_path / "conda.yaml").write_text("name: demo\n", encoding="utf-8")
    (submission_path / "empty_artifacts").mkdir()

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "mlflow_model.zip"
    with zipfile.ZipFile(prepared) as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
        dirs = sorted(info.filename for info in archive.infolist() if info.is_dir())
        assert archive.read("data/model.skops") == b"skops-model"
    assert members == ["MLmodel", "conda.yaml", "data/model.skops"]
    assert "empty_artifacts/" in dirs


def test_validate_and_prepare_submission_archives_coreml_package_directory(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload a CoreML `model.mlpackage` directory for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "model.mlpackage"
    data_dir = submission_path / "Data" / "com.apple.CoreML"
    data_dir.mkdir(parents=True)
    (submission_path / "Manifest.json").write_text('{"fileFormatVersion": "1.0.0"}\n', encoding="utf-8")
    (data_dir / "model.mlmodel").write_bytes(b"coreml-model")
    (submission_path / "empty_assets").mkdir()

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "model.mlpackage.zip"
    with zipfile.ZipFile(prepared) as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
        dirs = sorted(info.filename for info in archive.infolist() if info.is_dir())
        assert archive.read("Data/com.apple.CoreML/model.mlmodel") == b"coreml-model"
    assert members == ["Data/com.apple.CoreML/model.mlmodel", "Manifest.json"]
    assert "empty_assets/" in dirs


def test_validate_and_prepare_submission_archives_coreml_compiled_package_directory(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload a compiled CoreML `model.mlmodelc` directory for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "model.mlmodelc"
    payload_dir = submission_path / "com.apple.CoreML"
    payload_dir.mkdir(parents=True)
    (payload_dir / "model.mil").write_bytes(b"compiled-coreml")
    (submission_path / "empty_assets").mkdir()

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "model.mlmodelc.zip"
    with zipfile.ZipFile(prepared) as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
        dirs = sorted(info.filename for info in archive.infolist() if info.is_dir())
        assert archive.read("com.apple.CoreML/model.mil") == b"compiled-coreml"
    assert members == ["com.apple.CoreML/model.mil"]
    assert "empty_assets/" in dirs


def test_validate_and_prepare_submission_archives_shapefile_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload a Shapefile named `submission.shp` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.shp"
    submission_path.write_bytes(b"shape")
    (tmp_path / "submission.shx").write_bytes(b"index")
    (tmp_path / "submission.dbf").write_bytes(b"attributes")
    (tmp_path / "submission.prj").write_text("EPSG:4326\n", encoding="utf-8")
    (tmp_path / "submission.cpg").write_text("UTF-8\n", encoding="ascii")
    (tmp_path / "submission.qix").write_bytes(b"qix")
    (tmp_path / "submission.shp.aux.xml").write_text("<PAMDataset />\n", encoding="utf-8")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == [
            "submission.cpg",
            "submission.dbf",
            "submission.prj",
            "submission.qix",
            "submission.shp",
            "submission.shp.aux.xml",
            "submission.shx",
        ]
        assert archive.read("submission.dbf") == b"attributes"
        assert archive.read("submission.qix") == b"qix"


def test_validate_and_prepare_submission_rejects_incomplete_shapefile_bundle(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.shp` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.shp"
    submission_path.write_bytes(b"shape")

    with pytest.raises(SubmissionValidationError, match="missing required sidecar"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_archives_mapinfo_tab_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload a MapInfo TAB bundle named `submission.tab` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.tab"
    submission_path.write_text("!table\n!version 300\n", encoding="utf-8")
    (tmp_path / "submission.dat").write_bytes(b"data")
    (tmp_path / "submission.id").write_bytes(b"ids")
    (tmp_path / "submission.map").write_bytes(b"map")
    (tmp_path / "submission.ind").write_bytes(b"index")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == [
            "submission.dat",
            "submission.id",
            "submission.ind",
            "submission.map",
            "submission.tab",
        ]
        assert archive.read("submission.map") == b"map"


def test_validate_and_prepare_submission_rejects_incomplete_mapinfo_tab_bundle(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\n"
        "Upload a MapInfo TAB file named `submission.tab` with `.dat`, `.map`, and `.id` files.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.tab"
    submission_path.write_text("!table\n!version 300\n", encoding="utf-8")
    (tmp_path / "submission.dat").write_bytes(b"data")

    with pytest.raises(SubmissionValidationError, match="MapInfo TAB submission is missing required sidecar"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_keeps_plain_tabular_tab_without_mapinfo_sidecars(tmp_path: Path) -> None:
    _service, context_dir = _build_service(tmp_path)
    sample_path = context_dir / "sample_submission.tab"
    sample_path.write_text("id\ttarget\n1\t0.0\n2\t0.0\n", encoding="utf-8")
    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=tmp_path / "data",
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nSubmit a tab-delimited `.tab` file with columns `id` and `target`.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.tab"
    submission_path.write_text("id\ttarget\n1\t0.25\n2\t0.75\n", encoding="utf-8")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == submission_path
    validate_submission(str(prepared), str(sample_path))


def test_validate_and_prepare_submission_archives_mapinfo_mif_mid_sidecar(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload a MapInfo MIF/MID pair named `submission.mif` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.mif"
    submission_path.write_text("Version 300\nColumns 1\n  Name Char(20)\nData\n", encoding="utf-8")
    (tmp_path / "submission.mid").write_text('"parcel-a"\n', encoding="utf-8")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == ["submission.mid", "submission.mif"]
        assert archive.read("submission.mid") == b'"parcel-a"\n'


def test_validate_and_prepare_submission_rejects_incomplete_mapinfo_mif_mid_pair(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload a MapInfo MIF/MID pair named `submission.mif` and `submission.mid`.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.mif"
    submission_path.write_text("Version 300\nColumns 1\n  Name Char(20)\nData\n", encoding="utf-8")

    with pytest.raises(SubmissionValidationError, match="MapInfo MIF/MID submission is missing required sidecar"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_archives_georeferenced_raster_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload a georeferenced raster named `submission.tif` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.tif"
    submission_path.write_bytes(b"raster")
    (tmp_path / "submission.tfw").write_text("1\n0\n0\n-1\n100\n200\n", encoding="ascii")
    (tmp_path / "submission.prj").write_text("EPSG:4326\n", encoding="utf-8")
    (tmp_path / "submission.tif.aux.xml").write_text("<PAMDataset />\n", encoding="utf-8")
    (tmp_path / "submission.tif.ovr").write_bytes(b"overview")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == [
            "submission.prj",
            "submission.tfw",
            "submission.tif",
            "submission.tif.aux.xml",
            "submission.tif.ovr",
        ]
        assert archive.read("submission.tfw") == b"1\n0\n0\n-1\n100\n200\n"
        assert archive.read("submission.tif.aux.xml") == b"<PAMDataset />\n"


def test_validate_and_prepare_submission_keeps_plain_image_without_geospatial_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.png` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.png"
    submission_path.write_bytes(b"image")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == submission_path


def test_validate_and_prepare_submission_archives_vrt_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload a GDAL VRT named `submission.vrt` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.vrt"
    submission_path.write_text(
        """
        <VRTDataset rasterXSize="2" rasterYSize="2">
          <VRTRasterBand dataType="Byte" band="1">
            <SimpleSource>
              <SourceFilename relativeToVRT="1">rasters/source.tif</SourceFilename>
            </SimpleSource>
          </VRTRasterBand>
        </VRTDataset>
        """,
        encoding="utf-8",
    )
    (tmp_path / "rasters").mkdir()
    (tmp_path / "rasters" / "source.tif").write_bytes(b"raster")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == ["rasters/source.tif", "submission.vrt"]
        assert archive.read("rasters/source.tif") == b"raster"


def test_validate_and_prepare_submission_preserves_nested_vrt_layout(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload a GDAL VRT named `submission.vrt` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "layers" / "submission.vrt"
    submission_path.parent.mkdir()
    (tmp_path / "rasters").mkdir()
    submission_path.write_text(
        """
        <VRTDataset rasterXSize="2" rasterYSize="2">
          <VRTRasterBand dataType="Byte" band="1">
            <SimpleSource>
              <SourceFilename relativeToVRT="1">../rasters/source.tif</SourceFilename>
            </SimpleSource>
          </VRTRasterBand>
        </VRTDataset>
        """,
        encoding="utf-8",
    )
    (tmp_path / "rasters" / "source.tif").write_bytes(b"raster")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "layers" / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == ["layers/submission.vrt", "rasters/source.tif"]
        assert archive.read("rasters/source.tif") == b"raster"


def test_validate_and_prepare_submission_rejects_vrt_with_missing_sidecar(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload a GDAL VRT named `submission.vrt` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.vrt"
    submission_path.write_text(
        """
        <VRTDataset rasterXSize="2" rasterYSize="2">
          <VRTRasterBand dataType="Byte" band="1">
            <SimpleSource><SourceFilename relativeToVRT="1">rasters/source.tif</SourceFilename></SimpleSource>
          </VRTRasterBand>
        </VRTDataset>
        """,
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="GDAL VRT submission is missing referenced source"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_vrt_with_unsafe_sidecar(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload a GDAL VRT named `submission.vrt` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.vrt"
    submission_path.write_text(
        """
        <VRTDataset rasterXSize="2" rasterYSize="2">
          <VRTRasterBand dataType="Byte" band="1">
            <SimpleSource><SourceFilename relativeToVRT="1">https://example.com/source.tif</SourceFilename></SimpleSource>
          </VRTRasterBand>
        </VRTDataset>
        """,
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="GDAL VRT submission references unsafe source"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_archives_kml_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.kml` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.kml"
    submission_path.write_text(
        """
        <kml xmlns="http://www.opengis.net/kml/2.2">
          <Document>
            <Icon><href>icons/pin.png</href></Icon>
            <GroundOverlay><Icon><href>overlays/ground.png</href></Icon></GroundOverlay>
          </Document>
        </kml>
        """,
        encoding="utf-8",
    )
    (tmp_path / "icons").mkdir()
    (tmp_path / "overlays").mkdir()
    (tmp_path / "icons" / "pin.png").write_bytes(b"pin")
    (tmp_path / "overlays" / "ground.png").write_bytes(b"ground")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == [
            "icons/pin.png",
            "overlays/ground.png",
            "submission.kml",
        ]
        assert archive.read("icons/pin.png") == b"pin"
        assert archive.read("overlays/ground.png") == b"ground"


def test_validate_and_prepare_submission_decodes_kml_percent_encoded_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.kml` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.kml"
    submission_path.write_text(
        """
        <kml xmlns="http://www.opengis.net/kml/2.2">
          <Document><Icon><href>icons/pin%20blue.png</href></Icon></Document>
        </kml>
        """,
        encoding="utf-8",
    )
    (tmp_path / "icons").mkdir()
    (tmp_path / "icons" / "pin blue.png").write_bytes(b"pin")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == ["icons/pin blue.png", "submission.kml"]
        assert archive.read("icons/pin blue.png") == b"pin"


def test_validate_and_prepare_submission_preserves_nested_kml_layout(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.kml` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "layers" / "submission.kml"
    submission_path.parent.mkdir()
    (tmp_path / "icons").mkdir()
    submission_path.write_text(
        """
        <kml xmlns="http://www.opengis.net/kml/2.2">
          <Document><Icon><href>../icons/pin.png</href></Icon></Document>
        </kml>
        """,
        encoding="utf-8",
    )
    (tmp_path / "icons" / "pin.png").write_bytes(b"pin")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "layers" / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == ["icons/pin.png", "layers/submission.kml"]
        assert archive.read("icons/pin.png") == b"pin"


def test_validate_and_prepare_submission_rejects_kml_with_missing_sidecar(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.kml` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.kml"
    submission_path.write_text(
        """
        <kml xmlns="http://www.opengis.net/kml/2.2">
          <Document><Icon><href>icons/pin.png</href></Icon></Document>
        </kml>
        """,
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="missing referenced href"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_kml_with_external_url(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.kml` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.kml"
    submission_path.write_text(
        """
        <kml xmlns="http://www.opengis.net/kml/2.2">
          <Document><Icon><href>https://example.com/pin.png</href></Icon></Document>
        </kml>
        """,
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="unsafe href"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_kml_with_encoded_path_traversal(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.kml` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.kml"
    submission_path.write_text(
        """
        <kml xmlns="http://www.opengis.net/kml/2.2">
          <Document><Icon><href>%2e%2e/secret.png</href></Icon></Document>
        </kml>
        """,
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="unsafe href"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_archives_envi_header_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload an ENVI raster header named `submission.hdr` with its binary data file.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.hdr"
    submission_path.write_text("ENVI\nsamples = 2\nlines = 2\nbands = 1\n", encoding="utf-8")
    (tmp_path / "submission.dat").write_bytes(b"raster")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == ["submission.dat", "submission.hdr"]
        assert archive.read("submission.dat") == b"raster"


def test_validate_and_prepare_submission_rejects_envi_header_with_missing_sidecar(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload an ENVI raster header named `submission.hdr` with its binary data file.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.hdr"
    submission_path.write_text("ENVI\nsamples = 2\nlines = 2\nbands = 1\n", encoding="utf-8")

    with pytest.raises(SubmissionValidationError, match="ENVI header submission is missing referenced data"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_envi_header_with_unsafe_sidecar(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload an ENVI raster header named `submission.hdr` with its binary data file.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.hdr"
    submission_path.write_text("ENVI\ndata file = ../secret.dat\nsamples = 2\nlines = 2\nbands = 1\n", encoding="utf-8")

    with pytest.raises(SubmissionValidationError, match="unsafe data file"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_archives_metaimage_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.mhd` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.mhd"
    submission_path.write_text("ObjectType = Image\nElementDataFile = raw/volume.raw\n", encoding="utf-8")
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "volume.raw").write_bytes(b"voxels")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == ["raw/volume.raw", "submission.mhd"]
        assert archive.read("raw/volume.raw") == b"voxels"


def test_validate_and_prepare_submission_rejects_metaimage_with_missing_sidecar(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.mhd` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.mhd"
    submission_path.write_text("ObjectType = Image\nElementDataFile = volume.raw\n", encoding="utf-8")

    with pytest.raises(SubmissionValidationError, match="missing referenced ElementDataFile"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_metaimage_with_unsafe_sidecar_path(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.mhd` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.mhd"
    submission_path.write_text("ObjectType = Image\nElementDataFile = ../volume.raw\n", encoding="utf-8")

    with pytest.raises(SubmissionValidationError, match="unsafe ElementDataFile"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_archives_detached_nrrd_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.nhdr` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.nhdr"
    submission_path.write_text("NRRD0005\nsizes: 4 5 6\ndata file: raw/volume.raw\n", encoding="utf-8")
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "volume.raw").write_bytes(b"voxels")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == ["raw/volume.raw", "submission.nhdr"]
        assert archive.read("raw/volume.raw") == b"voxels"


def test_validate_and_prepare_submission_archives_detached_nrrd_list_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.nhdr` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.nhdr"
    submission_path.write_text(
        "NRRD0005\nsizes: 4 5 6\ndata file: LIST\nraw/slice0.raw\nraw/slice1.raw\n\n",
        encoding="utf-8",
    )
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "slice0.raw").write_bytes(b"slice-0")
    (tmp_path / "raw" / "slice1.raw").write_bytes(b"slice-1")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == ["raw/slice0.raw", "raw/slice1.raw", "submission.nhdr"]
        assert archive.read("raw/slice1.raw") == b"slice-1"


def test_validate_and_prepare_submission_rejects_detached_nrrd_with_missing_sidecar(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.nhdr` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.nhdr"
    submission_path.write_text("NRRD0005\nsizes: 4 5 6\ndata file: volume.raw\n", encoding="utf-8")

    with pytest.raises(SubmissionValidationError, match="missing referenced data file"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_detached_nrrd_with_unsafe_sidecar_path(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.nhdr` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.nhdr"
    submission_path.write_text("NRRD0005\nsizes: 4 5 6\ndata file: ../volume.raw\n", encoding="utf-8")

    with pytest.raises(SubmissionValidationError, match="unsafe data file"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_archives_analyze_pair_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.hdr` and matching `submission.img` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.hdr"
    submission_path.write_bytes(b"header")
    (tmp_path / "submission.img").write_bytes(b"volume")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == ["submission.hdr", "submission.img"]
        assert archive.read("submission.img") == b"volume"


def test_validate_and_prepare_submission_rejects_analyze_pair_with_missing_sidecar(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.img` and matching `submission.hdr` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.img"
    submission_path.write_bytes(b"volume")

    with pytest.raises(SubmissionValidationError, match="missing required pair sidecars"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_archives_obj_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.obj` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.obj"
    submission_path.write_text("mtllib materials/model.mtl\nv 0 0 0\n", encoding="utf-8")
    (tmp_path / "materials" / "textures").mkdir(parents=True)
    (tmp_path / "materials" / "model.mtl").write_text(
        "newmtl surface\nmap_Kd textures/diffuse.png\n",
        encoding="utf-8",
    )
    (tmp_path / "materials" / "textures" / "diffuse.png").write_bytes(b"texture")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == [
            "materials/model.mtl",
            "materials/textures/diffuse.png",
            "submission.obj",
        ]
        assert archive.read("materials/textures/diffuse.png") == b"texture"


def test_validate_and_prepare_submission_normalizes_obj_texture_parent_relative_path(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.obj` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.obj"
    submission_path.write_text("mtllib materials/model.mtl\nv 0 0 0\n", encoding="utf-8")
    (tmp_path / "materials").mkdir()
    (tmp_path / "textures").mkdir()
    (tmp_path / "materials" / "model.mtl").write_text(
        "newmtl surface\nmap_Kd ../textures/diffuse.png\n",
        encoding="utf-8",
    )
    (tmp_path / "textures" / "diffuse.png").write_bytes(b"texture")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == [
            "materials/model.mtl",
            "submission.obj",
            "textures/diffuse.png",
        ]
        assert archive.read("textures/diffuse.png") == b"texture"


def test_validate_and_prepare_submission_rejects_obj_with_missing_material(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.obj` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.obj"
    submission_path.write_text("mtllib materials/model.mtl\nv 0 0 0\n", encoding="utf-8")

    with pytest.raises(SubmissionValidationError, match="missing referenced material or texture"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_obj_with_unsafe_texture_path(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.obj` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.obj"
    submission_path.write_text("mtllib model.mtl\nv 0 0 0\n", encoding="utf-8")
    (tmp_path / "model.mtl").write_text("newmtl surface\nmap_Kd /tmp/diffuse.png\n", encoding="utf-8")

    with pytest.raises(SubmissionValidationError, match="unsafe material or texture"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_archives_ply_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.ply` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.ply"
    submission_path.write_text(
        "ply\nformat ascii 1.0\ncomment TextureFile textures/diffuse.png\nelement vertex 0\nend_header\n",
        encoding="ascii",
    )
    (tmp_path / "textures").mkdir()
    (tmp_path / "textures" / "diffuse.png").write_bytes(b"texture")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == ["submission.ply", "textures/diffuse.png"]
        assert archive.read("textures/diffuse.png") == b"texture"


def test_validate_and_prepare_submission_archives_las_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.las` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.las"
    submission_path.write_bytes(b"las")
    (tmp_path / "submission.prj").write_text("EPSG:4326\n", encoding="utf-8")
    (tmp_path / "submission.lax").write_bytes(b"index")
    (tmp_path / "submission.las.aux.xml").write_text("<PAMDataset />\n", encoding="utf-8")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == [
            "submission.las",
            "submission.las.aux.xml",
            "submission.lax",
            "submission.prj",
        ]
        assert archive.read("submission.lax") == b"index"


def test_validate_and_prepare_submission_keeps_plain_laz_without_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.laz` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.laz"
    submission_path.write_bytes(b"laz")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == submission_path


def test_validate_and_prepare_submission_decodes_ply_percent_encoded_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.ply` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.ply"
    submission_path.write_text(
        "ply\nformat ascii 1.0\nobj_info TextureFile textures/diffuse%20map.png\nelement vertex 0\nend_header\n",
        encoding="ascii",
    )
    (tmp_path / "textures").mkdir()
    (tmp_path / "textures" / "diffuse map.png").write_bytes(b"texture")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == ["submission.ply", "textures/diffuse map.png"]
        assert archive.read("textures/diffuse map.png") == b"texture"


def test_validate_and_prepare_submission_rejects_ply_with_missing_sidecar(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.ply` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.ply"
    submission_path.write_text(
        "ply\nformat ascii 1.0\ncomment TextureFile textures/diffuse.png\nelement vertex 0\nend_header\n",
        encoding="ascii",
    )

    with pytest.raises(SubmissionValidationError, match="missing referenced TextureFile"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_ply_with_external_url(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.ply` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.ply"
    submission_path.write_text(
        "ply\nformat ascii 1.0\ncomment TextureFile https://example.com/diffuse.png\nelement vertex 0\nend_header\n",
        encoding="ascii",
    )

    with pytest.raises(SubmissionValidationError, match="unsafe TextureFile"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_ply_with_encoded_path_traversal(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.ply` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.ply"
    submission_path.write_text(
        "ply\nformat ascii 1.0\ncomment TextureFile %2e%2e/secret.png\nelement vertex 0\nend_header\n",
        encoding="ascii",
    )

    with pytest.raises(SubmissionValidationError, match="unsafe TextureFile"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_archives_dae_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.dae` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.dae"
    submission_path.write_text(
        """
        <COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
          <library_images>
            <image id="diffuse"><init_from>textures/diffuse.png</init_from></image>
            <image id="inline"><init_from>data:image/png;base64,AAAA</init_from></image>
          </library_images>
        </COLLADA>
        """,
        encoding="utf-8",
    )
    (tmp_path / "textures").mkdir()
    (tmp_path / "textures" / "diffuse.png").write_bytes(b"texture")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == ["submission.dae", "textures/diffuse.png"]
        assert archive.read("textures/diffuse.png") == b"texture"


def test_validate_and_prepare_submission_decodes_dae_percent_encoded_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.dae` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.dae"
    submission_path.write_text(
        """
        <COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
          <library_images>
            <image id="diffuse"><init_from>textures/diffuse%20map.png</init_from></image>
          </library_images>
        </COLLADA>
        """,
        encoding="utf-8",
    )
    (tmp_path / "textures").mkdir()
    (tmp_path / "textures" / "diffuse map.png").write_bytes(b"texture")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == ["submission.dae", "textures/diffuse map.png"]
        assert archive.read("textures/diffuse map.png") == b"texture"


def test_validate_and_prepare_submission_preserves_nested_dae_layout(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.dae` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "meshes" / "submission.dae"
    submission_path.parent.mkdir()
    (tmp_path / "textures").mkdir()
    submission_path.write_text(
        """
        <COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
          <library_images>
            <image id="diffuse"><init_from>../textures/diffuse.png</init_from></image>
          </library_images>
        </COLLADA>
        """,
        encoding="utf-8",
    )
    (tmp_path / "textures" / "diffuse.png").write_bytes(b"texture")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "meshes" / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == ["meshes/submission.dae", "textures/diffuse.png"]
        assert archive.read("textures/diffuse.png") == b"texture"


def test_validate_and_prepare_submission_rejects_dae_with_missing_sidecar(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.dae` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.dae"
    submission_path.write_text(
        """
        <COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
          <library_images>
            <image id="diffuse"><init_from>textures/diffuse.png</init_from></image>
          </library_images>
        </COLLADA>
        """,
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="missing referenced external URI"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_dae_with_external_url(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.dae` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.dae"
    submission_path.write_text(
        """
        <COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
          <library_images>
            <image id="diffuse"><init_from>https://example.com/diffuse.png</init_from></image>
          </library_images>
        </COLLADA>
        """,
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="unsafe external URI"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_dae_with_encoded_path_traversal(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.dae` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.dae"
    submission_path.write_text(
        """
        <COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
          <library_images>
            <image id="diffuse"><init_from>%2e%2e/secret.png</init_from></image>
          </library_images>
        </COLLADA>
        """,
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="unsafe external URI"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_archives_x3d_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.x3d` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.x3d"
    submission_path.write_text(
        """
        <X3D>
          <Scene>
            <ImageTexture url='"textures/diffuse.png" "data:image/png;base64,AAAA"'/>
          </Scene>
        </X3D>
        """,
        encoding="utf-8",
    )
    (tmp_path / "textures").mkdir()
    (tmp_path / "textures" / "diffuse.png").write_bytes(b"texture")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == ["submission.x3d", "textures/diffuse.png"]
        assert archive.read("textures/diffuse.png") == b"texture"


def test_validate_and_prepare_submission_decodes_x3d_percent_encoded_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.x3d` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.x3d"
    submission_path.write_text(
        """
        <X3D>
          <Scene><ImageTexture url='"textures/diffuse%20map.png"'/></Scene>
        </X3D>
        """,
        encoding="utf-8",
    )
    (tmp_path / "textures").mkdir()
    (tmp_path / "textures" / "diffuse map.png").write_bytes(b"texture")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == ["submission.x3d", "textures/diffuse map.png"]
        assert archive.read("textures/diffuse map.png") == b"texture"


def test_validate_and_prepare_submission_preserves_nested_x3d_layout(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.x3d` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "scenes" / "submission.x3d"
    submission_path.parent.mkdir()
    (tmp_path / "textures").mkdir()
    submission_path.write_text(
        """
        <X3D>
          <Scene><ImageTexture url='"../textures/diffuse.png"'/></Scene>
        </X3D>
        """,
        encoding="utf-8",
    )
    (tmp_path / "textures" / "diffuse.png").write_bytes(b"texture")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "scenes" / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == ["scenes/submission.x3d", "textures/diffuse.png"]
        assert archive.read("textures/diffuse.png") == b"texture"


def test_validate_and_prepare_submission_rejects_x3d_with_missing_sidecar(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.x3d` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.x3d"
    submission_path.write_text(
        """
        <X3D>
          <Scene><ImageTexture url='"textures/diffuse.png"'/></Scene>
        </X3D>
        """,
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="missing referenced URL"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_x3d_with_external_url(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.x3d` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.x3d"
    submission_path.write_text(
        """
        <X3D>
          <Scene><ImageTexture url='"https://example.com/diffuse.png"'/></Scene>
        </X3D>
        """,
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="unsafe URL"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_x3d_with_encoded_path_traversal(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.x3d` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.x3d"
    submission_path.write_text(
        """
        <X3D>
          <Scene><ImageTexture url='"%2e%2e/secret.png"'/></Scene>
        </X3D>
        """,
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="unsafe URL"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_archives_gltf_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.gltf` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.gltf"
    submission_path.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "buffers": [{"uri": "buffers/scene.bin"}],
                "images": [
                    {"uri": "textures/diffuse.png"},
                    {"uri": "data:image/png;base64,AAAA"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "buffers").mkdir()
    (tmp_path / "textures").mkdir()
    (tmp_path / "buffers" / "scene.bin").write_bytes(b"buffer")
    (tmp_path / "textures" / "diffuse.png").write_bytes(b"texture")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == [
            "buffers/scene.bin",
            "submission.gltf",
            "textures/diffuse.png",
        ]
        assert archive.read("buffers/scene.bin") == b"buffer"
        assert archive.read("textures/diffuse.png") == b"texture"


def test_validate_and_prepare_submission_decodes_gltf_percent_encoded_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.gltf` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.gltf"
    submission_path.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "images": [{"uri": "textures/diffuse%20map.png"}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "textures").mkdir()
    (tmp_path / "textures" / "diffuse map.png").write_bytes(b"texture")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == ["submission.gltf", "textures/diffuse map.png"]
        assert archive.read("textures/diffuse map.png") == b"texture"


def test_validate_and_prepare_submission_preserves_nested_gltf_layout(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.gltf` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "scenes" / "submission.gltf"
    submission_path.parent.mkdir()
    (tmp_path / "textures").mkdir()
    submission_path.write_text(
        json.dumps({"asset": {"version": "2.0"}, "images": [{"uri": "../textures/diffuse.png"}]}),
        encoding="utf-8",
    )
    (tmp_path / "textures" / "diffuse.png").write_bytes(b"texture")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "scenes" / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == ["scenes/submission.gltf", "textures/diffuse.png"]
        assert archive.read("textures/diffuse.png") == b"texture"


def test_validate_and_prepare_submission_rejects_gltf_with_missing_sidecar(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.gltf` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.gltf"
    submission_path.write_text(
        json.dumps({"asset": {"version": "2.0"}, "buffers": [{"uri": "buffers/scene.bin"}]}),
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="missing referenced external URI"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_gltf_with_external_url(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.gltf` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.gltf"
    submission_path.write_text(
        json.dumps({"asset": {"version": "2.0"}, "images": [{"uri": "https://example.com/texture.png"}]}),
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="unsafe external URI"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_gltf_with_encoded_path_traversal(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.gltf` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.gltf"
    submission_path.write_text(
        json.dumps({"asset": {"version": "2.0"}, "buffers": [{"uri": "%2e%2e/secret.bin"}]}),
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="unsafe external URI"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_archives_usd_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.usda` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.usda"
    (tmp_path / "textures").mkdir()
    submission_path.write_text(
        "#usda 1.0\nasset inputs:file = @textures/diffuse.png@\n",
        encoding="utf-8",
    )
    (tmp_path / "textures" / "diffuse.png").write_bytes(b"texture")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == ["submission.usda", "textures/diffuse.png"]
        assert archive.read("textures/diffuse.png") == b"texture"


def test_validate_and_prepare_submission_preserves_nested_usd_layout(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.usda` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "scenes" / "submission.usda"
    submission_path.parent.mkdir()
    (tmp_path / "textures").mkdir()
    submission_path.write_text(
        "#usda 1.0\nasset inputs:file = @../textures/diffuse.png@\n",
        encoding="utf-8",
    )
    (tmp_path / "textures" / "diffuse.png").write_bytes(b"texture")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "scenes" / "submission.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == ["scenes/submission.usda", "textures/diffuse.png"]
        assert archive.read("textures/diffuse.png") == b"texture"


def test_validate_and_prepare_submission_rejects_usd_with_missing_sidecar(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.usda` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.usda"
    submission_path.write_text("#usda 1.0\nasset inputs:file = @textures/diffuse.png@\n", encoding="utf-8")

    with pytest.raises(SubmissionValidationError, match="missing referenced asset sidecars"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_usd_with_unsafe_sidecar(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `submission.usda` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.usda"
    submission_path.write_text("#usda 1.0\nasset inputs:file = @https://example.com/diffuse.png@\n", encoding="utf-8")

    with pytest.raises(SubmissionValidationError, match="unsafe asset sidecars"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_archives_model_index_shards(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `model.safetensors.index.json` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "model.safetensors.index.json"
    submission_path.write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.weight": "model-00001-of-00002.safetensors",
                    "layer.bias": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"shard-1")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"shard-2")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "model.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
        ]
        assert archive.read("model-00001-of-00002.safetensors") == b"shard-1"


def test_validate_and_prepare_submission_archives_tensorflow_checkpoint_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `model.ckpt.index` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "model.ckpt.index"
    submission_path.write_bytes(b"index")
    (tmp_path / "model.ckpt.data-00000-of-00002").write_bytes(b"shard-1")
    (tmp_path / "model.ckpt.data-00001-of-00002").write_bytes(b"shard-2")
    (tmp_path / "model.ckpt.meta").write_bytes(b"graph")
    (tmp_path / "checkpoint").write_text('model_checkpoint_path: "model.ckpt"\n', encoding="utf-8")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "model.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == [
            "checkpoint",
            "model.ckpt.data-00000-of-00002",
            "model.ckpt.data-00001-of-00002",
            "model.ckpt.index",
            "model.ckpt.meta",
        ]
        assert archive.read("model.ckpt.data-00000-of-00002") == b"shard-1"


def test_validate_and_prepare_submission_archives_model_artifact_sidecars(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `adapter_model.safetensors` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "adapter_model.safetensors"
    submission_path.write_bytes(b"weights")
    (tmp_path / "adapter_config.json").write_text('{"peft_type": "LORA"}\n', encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text('{"model_max_length": 512}\n', encoding="utf-8")

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == tmp_path / "adapter_model.zip"
    with zipfile.ZipFile(prepared) as archive:
        assert sorted(archive.namelist()) == [
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer_config.json",
        ]
        assert archive.read("adapter_model.safetensors") == b"weights"
        assert archive.read("adapter_config.json") == b'{"peft_type": "LORA"}\n'


def test_validate_and_prepare_submission_rejects_model_index_with_missing_shard(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `model.safetensors.index.json` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "model.safetensors.index.json"
    submission_path.write_text(
        json.dumps({"weight_map": {"layer.weight": "model-00001-of-00002.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="missing referenced shard"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_rejects_model_index_with_empty_shard(tmp_path: Path) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `model.safetensors.index.json` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "model.safetensors.index.json"
    shard_path = tmp_path / "model-00001-of-00001.safetensors"
    submission_path.write_text(
        json.dumps({"weight_map": {"layer.weight": shard_path.name}}),
        encoding="utf-8",
    )
    shard_path.write_bytes(b"")

    with pytest.raises(SubmissionValidationError, match="submission file is empty"):
        service.validate_and_prepare_submission(submission_path)


@pytest.mark.parametrize("shard_name", ["../model-00001-of-00001.safetensors", "/tmp/model.safetensors"])
def test_validate_and_prepare_submission_rejects_model_index_with_unsafe_shard_path(
    tmp_path: Path,
    shard_name: str,
) -> None:
    service, context_dir = _build_service(tmp_path)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nUpload `model.safetensors.index.json` for scoring.\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "model.safetensors.index.json"
    submission_path.write_text(
        json.dumps({"weight_map": {"layer.weight": shard_name}}),
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="unsafe shard paths"):
        service.validate_and_prepare_submission(submission_path)
