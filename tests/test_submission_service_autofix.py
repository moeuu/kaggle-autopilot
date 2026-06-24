from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kagglebot.exceptions import SubmissionValidationError
from kagglebot.submission.validate import validate_submission
from kagglebot.submission_service import SubmissionConfig, SubmissionService


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
