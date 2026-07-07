from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest
import zstandard as zstd

from kagglebot.exceptions import SubmissionValidationError
from kagglebot.submission.validate import validate_submission
from kagglebot.submission_service import SubmissionConfig, SubmissionService


def test_validate_and_prepare_accepts_normalized_blank_sample_header(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    data_dir = tmp_path / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    sample_path = context_dir / "sample_submission.csv"
    sample_path.write_text("id,,target\n1,-,0\n2,-,0\n", encoding="utf-8")
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,column_2,target\n1,-,0.1\n2,-,0.2\n", encoding="utf-8")

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == submission_path


def test_sniff_header_normalizes_blank_header_columns(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,,target\n1,x,0.1\n", encoding="utf-8")

    header, col_index = SubmissionService._sniff_header_and_column_index(
        submission_path=submission_path,
        delim=",",
        expected_columns=["id", "column_2", "target"],
    )

    assert header is True
    assert col_index == {"id": 0, "column_2": 1, "target": 2}


def test_validate_and_prepare_autofixes_tsv_headerless_and_aligns_to_sample_rows(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    data_dir = tmp_path / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    sample_path = context_dir / "sample_submission.csv"
    pd.DataFrame(
        {
            "col0": ["P1", "P1", "P2"],
            "col1": ["T1", "T2", "T1"],
            "col2": [0.1, 0.2, 0.3],
        }
    ).to_csv(sample_path, index=False)

    submission_path = tmp_path / "submission.csv"
    submission_path.write_text(
        "\n".join(
            [
                "P1\tT1\t0.9",
                "P2\tT1\t0.8",
                "P9\tT9\t0.7",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )

    prepared = service.validate_and_prepare_submission(submission_path)
    assert prepared.name.endswith(".autofixed.csv")

    validate_submission(str(prepared), str(sample_path))
    df = pd.read_csv(prepared)
    assert df.shape == (3, 3)
    assert df["col2"].tolist() == [0.9, 0.2, 0.8]


def test_validate_and_prepare_autofixes_compressed_csv_headerless_and_aligns_to_sample_rows(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    data_dir = tmp_path / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    sample_path = context_dir / "sample_submission.csv"
    pd.DataFrame(
        {
            "col0": ["P1", "P1", "P2"],
            "col1": ["T1", "T2", "T1"],
            "col2": [0.1, 0.2, 0.3],
        }
    ).to_csv(sample_path, index=False)

    submission_path = tmp_path / "submission.csv.gz"
    with gzip.open(submission_path, "wt", encoding="utf-8") as handle:
        handle.write("P1,T1,0.9\nP2,T1,0.8\nP9,T9,0.7\n")

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == "submission.autofixed.csv.gz"
    validate_submission(str(prepared), str(sample_path))
    df = pd.read_csv(prepared)
    assert df.shape == (3, 3)
    assert df["col2"].tolist() == [0.9, 0.2, 0.8]


def test_validate_and_prepare_does_not_aggregate_idless_single_prediction_parquet(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    data_dir = tmp_path / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    sample_path = context_dir / "sample_submission.csv"
    pd.DataFrame({"target": [0.0, 0.0, 0.0]}).to_csv(sample_path, index=False)

    submission_path = tmp_path / "submission.parquet"
    pd.DataFrame({"target": [0.7, 0.2, 0.9]}).to_parquet(submission_path, index=False)

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == submission_path
    validate_submission(str(prepared), str(sample_path), data_dir=data_dir)
    df = pd.read_parquet(prepared)
    assert df["target"].tolist() == [0.7, 0.2, 0.9]


def test_validate_and_prepare_does_not_aggregate_idless_multi_prediction_parquet(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    data_dir = tmp_path / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    sample_path = context_dir / "sample_submission.csv"
    pd.DataFrame({"target": [0.0, 0.0], "score": [0.0, 0.0]}).to_csv(sample_path, index=False)

    submission_path = tmp_path / "submission.parquet"
    pd.DataFrame({"target": [0.7, 0.2], "score": [0.3, 0.8]}).to_parquet(submission_path, index=False)

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == submission_path
    validate_submission(str(prepared), str(sample_path), data_dir=data_dir)
    df = pd.read_parquet(prepared)
    assert df.to_dict("list") == {"target": [0.7, 0.2], "score": [0.3, 0.8]}


def test_validate_and_prepare_autofixes_reordered_header_columns(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    data_dir = tmp_path / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    sample_path = context_dir / "sample_submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_csv(sample_path, index=False)

    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"target": [0.7, 0.2], "id": [1, 2]}).to_csv(submission_path, index=False)

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )

    prepared = service.validate_and_prepare_submission(submission_path)
    validate_submission(str(prepared), str(sample_path))
    df = pd.read_csv(prepared)
    assert list(df.columns) == ["id", "target"]
    assert df["target"].tolist() == [0.7, 0.2]


def test_validate_and_prepare_autofixes_semicolon_csv_and_preserves_delimiter(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    data_dir = tmp_path / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    sample_path = context_dir / "sample_submission.csv"
    sample_path.write_text("id;target\n1;0.0\n2;0.0\n", encoding="utf-8")
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("target;id\n0.7;1\n0.2;2\n", encoding="utf-8")

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )

    prepared = service.validate_and_prepare_submission(submission_path)

    validate_submission(str(prepared), str(sample_path))
    assert prepared.read_text(encoding="utf-8").splitlines()[0] == "id;target"
    df = pd.read_csv(prepared, sep=";")
    assert list(df.columns) == ["id", "target"]
    assert df["target"].tolist() == [0.7, 0.2]


def test_validate_and_prepare_autofixes_jsonl_columns_and_preserves_jsonl(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_path = tmp_path / "sample_submission.jsonl"
    sample_path.write_text('{"id": 1, "prediction": "A"}\n{"id": 2, "prediction": "B"}\n', encoding="utf-8")
    submission_path = tmp_path / "submission.jsonl"
    submission_path.write_text('{"ID": 1, "Category": "B"}\n{"ID": 2, "Category": "A"}\n', encoding="utf-8")

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == "submission.autofixed.jsonl"
    validate_submission(str(prepared), str(sample_path), data_dir=data_dir)
    df = pd.read_json(prepared, lines=True)
    assert list(df.columns) == ["id", "prediction"]
    assert df["prediction"].tolist() == ["B", "A"]


def test_validate_and_prepare_autofixes_wrapped_json_columns_and_preserves_json(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_path = tmp_path / "sample_submission.json"
    sample_path.write_text(
        '{"records":[{"id":1,"prediction":"A"},{"id":2,"prediction":"B"}]}',
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.json"
    submission_path.write_text(
        '{"rows":[{"ID":1,"Category":"B"},{"ID":2,"Category":"A"}]}',
        encoding="utf-8",
    )

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == "submission.autofixed.json"
    validate_submission(str(prepared), str(sample_path), data_dir=data_dir)
    df = pd.read_json(prepared)
    assert list(df.columns) == ["id", "prediction"]
    assert df["prediction"].tolist() == ["B", "A"]


def test_validate_and_prepare_autofixes_tsv_columns_and_preserves_tsv(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_path = tmp_path / "sample_submission.tsv"
    sample_path.write_text("id\tprediction\n1\tA\n2\tB\n", encoding="utf-8")
    submission_path = tmp_path / "submission.tsv"
    submission_path.write_text("ID\tCategory\n1\tB\n2\tA\n", encoding="utf-8")

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == "submission.autofixed.tsv"
    validate_submission(str(prepared), str(sample_path), data_dir=data_dir)
    df = pd.read_csv(prepared, sep="\t")
    assert list(df.columns) == ["id", "prediction"]
    assert df["prediction"].tolist() == ["B", "A"]


def test_validate_and_prepare_autofixes_psv_columns_and_preserves_psv(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_path = tmp_path / "sample_submission.psv"
    sample_path.write_text("id|prediction\n1|A\n2|B\n", encoding="utf-8")
    submission_path = tmp_path / "submission.psv"
    submission_path.write_text("ID|Category\n1|B\n2|A\n", encoding="utf-8")

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == "submission.autofixed.psv"
    validate_submission(str(prepared), str(sample_path), data_dir=data_dir)
    assert prepared.read_text(encoding="utf-8").splitlines()[0] == "id|prediction"
    df = pd.read_csv(prepared, sep="|")
    assert list(df.columns) == ["id", "prediction"]
    assert df["prediction"].tolist() == ["B", "A"]


def test_validate_and_prepare_autofixes_compressed_jsonl_columns_without_duplicate_suffix(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_path = tmp_path / "sample_submission.jsonl.zst"
    submission_path = tmp_path / "submission.jsonl.zst"
    compressor = zstd.ZstdCompressor()
    sample_path.write_bytes(compressor.compress(b'{"id":1,"prediction":"A"}\n{"id":2,"prediction":"B"}\n'))
    submission_path.write_bytes(compressor.compress(b'{"ID":1,"Category":"B"}\n{"ID":2,"Category":"A"}\n'))

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == "submission.autofixed.jsonl.zst"
    validate_submission(str(prepared), str(sample_path), data_dir=data_dir)
    df = pd.read_json(prepared, lines=True)
    assert list(df.columns) == ["id", "prediction"]
    assert df["prediction"].tolist() == ["B", "A"]


def test_validate_and_prepare_autofixes_parquet_duplicate_rows_and_preserves_parquet(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_path = tmp_path / "sample_submission.parquet"
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_parquet(sample_path, index=False)
    submission_path = tmp_path / "submission.parquet"
    pd.DataFrame({"id": [1, 1, 2], "target": [0.2, 0.8, 0.4]}).to_parquet(submission_path, index=False)

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == "submission.autofixed.parquet"
    validate_submission(str(prepared), str(sample_path), data_dir=data_dir)
    df = pd.read_parquet(prepared)
    assert list(df.columns) == ["id", "target"]
    assert df["id"].tolist() == [1, 2]
    assert df["target"].tolist() == [0.5, 0.4]


def test_validate_and_prepare_autofixes_excel_duplicate_rows_and_preserves_excel(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_path = tmp_path / "sample_submission.xlsx"
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_excel(sample_path, index=False)
    submission_path = tmp_path / "submission.xlsx"
    pd.DataFrame({"id": [1, 1, 2], "target": [0.2, 0.8, 0.4]}).to_excel(submission_path, index=False)

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == "submission.autofixed.xlsx"
    validate_submission(str(prepared), str(sample_path), data_dir=data_dir)
    df = pd.read_excel(prepared)
    assert list(df.columns) == ["id", "target"]
    assert df["id"].tolist() == [1, 2]
    assert df["target"].tolist() == [0.5, 0.4]


@pytest.mark.parametrize("suffix", [".orc", ".hdf", ".hdf5"])
def test_validate_and_prepare_autofixes_binary_duplicate_rows_and_preserves_suffix(
    tmp_path: Path,
    suffix: str,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_path = tmp_path / f"sample_submission{suffix}"
    submission_path = tmp_path / f"submission{suffix}"
    sample = pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]})
    submission = pd.DataFrame({"id": [1, 1, 2], "target": [0.2, 0.8, 0.4]})
    if suffix == ".orc":
        sample.to_orc(sample_path, index=False)
        submission.to_orc(submission_path, index=False)
    else:
        sample.to_hdf(sample_path, key="sample_submission", mode="w", format="table", index=False)
        submission.to_hdf(submission_path, key="submission", mode="w", format="table", index=False)

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == f"submission.autofixed{suffix}"
    validate_submission(str(prepared), str(sample_path), data_dir=data_dir)
    df = pd.read_orc(prepared) if suffix == ".orc" else pd.read_hdf(prepared)
    assert list(df.columns) == ["id", "target"]
    assert df["id"].tolist() == [1, 2]
    assert df["target"].tolist() == [0.5, 0.4]


def test_validate_and_prepare_autofixes_asset_id_and_prediction_string_aliases(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_path = tmp_path / "sample_submission.csv"
    pd.DataFrame(
        {
            "image_id": ["img_1.jpg", "img_2.jpg"],
            "prediction_string": ["", ""],
        }
    ).to_csv(sample_path, index=False)
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame(
        {
            "filename": ["img_1.jpg", "img_2.jpg"],
            "pred_string": ["0.9 1 2 3 4", "0.8 5 6 7 8"],
        }
    ).to_csv(submission_path, index=False)

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )

    prepared = service.validate_and_prepare_submission(submission_path)

    validate_submission(str(prepared), str(sample_path), data_dir=data_dir)
    df = pd.read_csv(prepared)
    assert list(df.columns) == ["image_id", "prediction_string"]
    assert df["prediction_string"].tolist() == ["0.9 1 2 3 4", "0.8 5 6 7 8"]


def test_validate_and_prepare_autofixes_medical_record_id_aliases(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_path = tmp_path / "sample_submission.csv"
    pd.DataFrame({"patient_id": ["p1", "p2"], "prediction": [0.0, 0.0]}).to_csv(sample_path, index=False)
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"record_id": ["p1", "p2"], "score": [0.7, 0.2]}).to_csv(submission_path, index=False)

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )

    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == "submission.autofixed.csv"
    validate_submission(str(prepared), str(sample_path), data_dir=data_dir)
    df = pd.read_csv(prepared)
    assert list(df.columns) == ["patient_id", "prediction"]
    assert df.to_dict("list") == {"patient_id": ["p1", "p2"], "prediction": [0.7, 0.2]}


def test_validate_and_prepare_compacts_large_csv_when_over_soft_limit(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    sample_path = tmp_path / "sample_submission.csv"
    rows = 250
    feature_cols = [f"c{idx}" for idx in range(40)]
    sample_df = pd.DataFrame({"id": [f"id_{i}" for i in range(rows)], **{c: 0.0 for c in feature_cols}})
    sample_df.to_csv(sample_path, index=False)

    submission_path = tmp_path / "submission.csv"
    submission_df = sample_df.copy()
    for c in feature_cols:
        submission_df[c] = (pd.Series(range(rows), dtype=float) / (rows + 1.0)) + (0.1234567890123)
    submission_df.to_csv(submission_path, index=False)

    monkeypatch.setattr("kagglebot.submission_service._KAGGLE_SUBMISSION_SOFT_MAX_BYTES", 2000)

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name.endswith(".compact.csv")
    assert prepared.stat().st_size < submission_path.stat().st_size
    validate_submission(str(prepared), str(sample_path))


def test_validate_and_prepare_compacts_large_tsv_when_over_soft_limit(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    sample_path = tmp_path / "sample_submission.tsv"
    rows = 220
    feature_cols = [f"c{idx}" for idx in range(32)]
    sample_df = pd.DataFrame({"id": [f"id_{i}" for i in range(rows)], **{c: 0.0 for c in feature_cols}})
    sample_df.to_csv(sample_path, sep="\t", index=False)

    submission_path = tmp_path / "submission.tsv"
    submission_df = sample_df.copy()
    for c in feature_cols:
        submission_df[c] = (pd.Series(range(rows), dtype=float) / (rows + 1.0)) + 0.1234567890123
    submission_df.to_csv(submission_path, sep="\t", index=False)

    monkeypatch.setattr("kagglebot.submission_service._KAGGLE_SUBMISSION_SOFT_MAX_BYTES", 2000)

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name.endswith(".compact.tsv")
    assert prepared.stat().st_size < submission_path.stat().st_size
    validate_submission(str(prepared), str(sample_path))


@pytest.mark.parametrize("suffix", [".csv.gz", ".csv.zst"])
def test_validate_and_prepare_compacts_large_compressed_csv_when_over_soft_limit(
    tmp_path: Path,
    monkeypatch,
    suffix: str,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    sample_path = tmp_path / "sample_submission.csv"
    rows = 160
    feature_cols = [f"c{idx}" for idx in range(25)]
    sample_df = pd.DataFrame({"id": [f"id_{i}" for i in range(rows)], **{c: 0.0 for c in feature_cols}})
    sample_df.to_csv(sample_path, index=False)

    submission_path = tmp_path / f"submission{suffix}"
    submission_df = sample_df.copy()
    for c in feature_cols:
        submission_df[c] = (pd.Series(range(rows), dtype=float) / (rows + 1.0)) + 0.123456789012345
    csv_payload = submission_df.to_csv(index=False).encode("utf-8")
    if suffix.endswith(".gz"):
        with gzip.open(submission_path, "wb", compresslevel=9) as handle:
            handle.write(csv_payload)
    else:
        submission_path.write_bytes(zstd.ZstdCompressor(level=9).compress(csv_payload))

    monkeypatch.setattr("kagglebot.submission_service._KAGGLE_SUBMISSION_SOFT_MAX_BYTES", 1)
    monkeypatch.setattr("kagglebot.submission_service._KAGGLE_SUBMISSION_COMPACT_MIN_BYTES_SAVED", 1)
    monkeypatch.setattr("kagglebot.submission_service._KAGGLE_SUBMISSION_COMPACT_MIN_RELATIVE_SAVED", 0.0)

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == f"submission.compact{suffix}"
    assert prepared.stat().st_size < submission_path.stat().st_size
    validate_submission(str(prepared), str(sample_path))


@pytest.mark.parametrize(
    ("suffix", "sample_suffix", "sep"),
    [
        (".tsv.gz", ".tsv", "\t"),
        (".psv.zst", ".psv", "|"),
    ],
)
def test_validate_and_prepare_compacts_large_compressed_delimited_text_when_over_soft_limit(
    tmp_path: Path,
    monkeypatch,
    suffix: str,
    sample_suffix: str,
    sep: str,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    sample_path = tmp_path / f"sample_submission{sample_suffix}"
    rows = 180
    feature_cols = [f"c{idx}" for idx in range(24)]
    sample_df = pd.DataFrame({"id": [f"id_{i}" for i in range(rows)], **{c: 0.0 for c in feature_cols}})
    sample_df.to_csv(sample_path, sep=sep, index=False)

    submission_path = tmp_path / f"submission{suffix}"
    submission_df = sample_df.copy()
    for c in feature_cols:
        submission_df[c] = (pd.Series(range(rows), dtype=float) / (rows + 1.0)) + 0.123456789012345
    payload = submission_df.to_csv(index=False, sep=sep).encode("utf-8")
    if suffix.endswith(".gz"):
        with gzip.open(submission_path, "wb", compresslevel=9) as handle:
            handle.write(payload)
    else:
        submission_path.write_bytes(zstd.ZstdCompressor(level=9).compress(payload))

    monkeypatch.setattr("kagglebot.submission_service._KAGGLE_SUBMISSION_SOFT_MAX_BYTES", 1)
    monkeypatch.setattr("kagglebot.submission_service._KAGGLE_SUBMISSION_COMPACT_MIN_BYTES_SAVED", 1)
    monkeypatch.setattr("kagglebot.submission_service._KAGGLE_SUBMISSION_COMPACT_MIN_RELATIVE_SAVED", 0.0)

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared.name == f"submission.compact{suffix}"
    assert prepared.stat().st_size < submission_path.stat().st_size
    validate_submission(str(prepared), str(sample_path))


def test_validate_and_prepare_skips_compact_when_savings_are_tiny(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    sample_path = tmp_path / "sample_submission.csv"
    sample_path.write_text("id,target\n1,0.0\n2,0.0\n3,0.0\n", encoding="utf-8")

    submission_path = tmp_path / "submission.csv"
    submission_path.write_text(
        "id,target\n1,325.0\n2,302.0\n3,307.0\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("kagglebot.submission_service._KAGGLE_SUBMISSION_SOFT_MAX_BYTES", 1)

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )
    prepared = service.validate_and_prepare_submission(submission_path)

    assert prepared == submission_path
    assert not submission_path.with_name("submission.compact.csv").exists()
    validate_submission(str(prepared), str(sample_path))


def test_validate_and_prepare_autofixes_header_only_sample_by_renaming_columns(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    sample_path.write_text("id,prediction\n", encoding="utf-8")

    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"ID": [1, 2], "Category": ["A", "B"]}).to_csv(submission_path, index=False)

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

    prepared = service.validate_and_prepare_submission(submission_path)
    validate_submission(str(prepared), str(sample_path))
    df = pd.read_csv(prepared)
    assert list(df.columns) == ["id", "prediction"]
    assert df["id"].tolist() == [1, 2]
    assert df["prediction"].tolist() == ["A", "B"]


def test_validate_and_prepare_rejects_header_only_submission(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    sample_path.write_text("id,prediction\n", encoding="utf-8")

    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,prediction\n", encoding="utf-8")

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

    with pytest.raises(SubmissionValidationError, match="submission has no data rows"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_autofixes_missing_required_id_suffix(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    data_dir = tmp_path / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "Kaggle_Prepared" / "val" / "MS").mkdir(parents=True, exist_ok=True)
    for stem in ("val_0001", "val_0002"):
        (data_dir / "Kaggle_Prepared" / "val" / "MS" / f"{stem}.tif").write_bytes(b"TIFF")

    sample_path = context_dir / "sample_submission.csv"
    sample_path.write_text("id,prediction\n", encoding="utf-8")
    (context_dir / "data.md").write_text(
        "## Submission Format\n\n"
        "A CSV file with the following columns:\n"
        "* `Id`: The filename (e.g., `val_a1b2c3d4.tif`)\n"
        "* `Category`: The predicted class\n",
        encoding="utf-8",
    )

    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"Id": ["val_0001", "val_0002"], "Category": ["Health", "Rust"]}).to_csv(submission_path, index=False)

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )

    prepared = service.validate_and_prepare_submission(submission_path)
    assert prepared.name.endswith(".autofixed.csv")
    validate_submission(str(prepared), str(sample_path), data_dir=data_dir)

    df = pd.read_csv(prepared)
    assert list(df.columns) == ["id", "prediction"]
    assert df["id"].tolist() == ["val_0001.tif", "val_0002.tif"]


def test_validate_and_prepare_autofixes_partial_compound_required_id_suffix(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    data_dir = tmp_path / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "scans" / "test").mkdir(parents=True, exist_ok=True)
    for stem in ("case_001", "case_002"):
        (data_dir / "scans" / "test" / f"{stem}.nii.gz").write_bytes(b"scan")

    sample_path = context_dir / "sample_submission.csv"
    sample_path.write_text("id,target\n", encoding="utf-8")

    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": ["case_001.nii", "case_002.nii"], "target": [0.1, 0.2]}).to_csv(
        submission_path,
        index=False,
    )

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )

    prepared = service.validate_and_prepare_submission(submission_path)
    assert prepared.name.endswith(".autofixed.csv")
    validate_submission(str(prepared), str(sample_path), data_dir=data_dir)

    df = pd.read_csv(prepared)
    assert df["id"].tolist() == ["case_001.nii.gz", "case_002.nii.gz"]


def test_validate_and_prepare_does_not_autofix_header_only_sample_when_mapping_is_ambiguous(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    sample_path.write_text("id,prediction\n", encoding="utf-8")

    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"foo": [1, 2], "bar": ["A", "B"]}).to_csv(submission_path, index=False)

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

    with pytest.raises(SubmissionValidationError):
        service.validate_and_prepare_submission(submission_path)
    assert not submission_path.with_name("submission.autofixed.csv").exists()


def test_validate_and_prepare_expands_placeholder_sample_to_test_ids(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    data_dir = tmp_path / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    sample_path = context_dir / "sample_submission.csv"
    pd.DataFrame(
        {
            "Stock code": ["X00001", "X00002", "X00003"],
            "IsDefault": [0, 0, 0],
        }
    ).to_csv(sample_path, index=False)

    train_path = data_dir / "train.csv"
    pd.DataFrame(
        {
            "Stock code": [f"T{i:05d}" for i in range(20)],
            "IsDefault": [0, 1] * 10,
        }
    ).to_csv(train_path, index=False)

    test_ids = ["X00001", "X00002", "X00003", *[f"X{i:05d}" for i in range(4, 21)]]
    test_path = data_dir / "test.csv"
    pd.DataFrame({"Stock code": test_ids, "f1": list(range(len(test_ids)))}).to_csv(test_path, index=False)

    submission_path = tmp_path / "submission.csv"
    pd.DataFrame(
        {
            "Stock code": ["X00001", "X00002", "X00003"],
            "IsDefault": [0.1, 0.2, 0.3],
        }
    ).to_csv(submission_path, index=False)

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )

    prepared = service.validate_and_prepare_submission(submission_path)
    assert prepared.name.endswith(".autofixed.csv")

    prepared_df = pd.read_csv(prepared)
    assert len(prepared_df) == len(test_ids)
    assert prepared_df["Stock code"].tolist() == test_ids
    validate_submission(str(prepared), str(sample_path), data_dir=data_dir)


def test_validate_and_prepare_expands_placeholder_sample_using_jsonl_train_and_test(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    data_dir = tmp_path / "data"
    context_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    sample_path = context_dir / "sample_submission.csv"
    pd.DataFrame(
        {
            "Stock code": ["X00001", "X00002", "X00003"],
            "IsDefault": [0, 0, 0],
        }
    ).to_csv(sample_path, index=False)

    (data_dir / "train.jsonl").write_text(
        "\n".join(f'{{"Stock code": "T{i:05d}", "IsDefault": {i % 2}}}' for i in range(20)) + "\n",
        encoding="utf-8",
    )
    test_ids = ["X00001", "X00002", "X00003", *[f"X{i:05d}" for i in range(4, 21)]]
    (data_dir / "test.jsonl").write_text(
        "\n".join(f'{{"Stock code": "{stock_code}", "f1": {idx}}}' for idx, stock_code in enumerate(test_ids)) + "\n",
        encoding="utf-8",
    )

    submission_path = tmp_path / "submission.csv"
    pd.DataFrame(
        {
            "Stock code": ["X00001", "X00002", "X00003"],
            "IsDefault": [0.1, 0.2, 0.3],
        }
    ).to_csv(submission_path, index=False)

    service = SubmissionService(
        SubmissionConfig(
            slug="demo",
            data_dir=data_dir,
            sample_submission_path=sample_path,
            submission_ledger_path=tmp_path / "ledger.jsonl",
            dry_run=True,
            force_submit=True,
        )
    )

    prepared = service.validate_and_prepare_submission(submission_path)

    prepared_df = pd.read_csv(prepared)
    assert len(prepared_df) == len(test_ids)
    assert prepared_df["Stock code"].tolist() == test_ids
    assert prepared_df["IsDefault"].iloc[:3].tolist() == [0.1, 0.2, 0.3]
    assert prepared_df["IsDefault"].iloc[3:].tolist() == [0.5] * (len(test_ids) - 3)
    validate_submission(str(prepared), str(sample_path), data_dir=data_dir)
