from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from kagglebot.exceptions import SubmissionValidationError
from kagglebot.submission_service import SubmissionConfig, SubmissionService


def _build_service(tmp_path: Path) -> SubmissionService:
    config = SubmissionConfig(
        slug="demo",
        data_dir=tmp_path / "data",
        sample_submission_path=tmp_path / "missing_sample_submission.csv",
        submission_ledger_path=tmp_path / "ledger.jsonl",
        dry_run=True,
        force_submit=False,
    )
    return SubmissionService(config)


def test_validate_and_prepare_submission_accepts_zip_without_sample_csv(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    submission_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(submission_path, "w") as archive:
        archive.writestr("123.tif", b"dummy")

    prepared = service.validate_and_prepare_submission(submission_path)
    assert prepared == submission_path


def test_validate_and_prepare_submission_rejects_invalid_zip(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    submission_path = tmp_path / "submission.zip"
    submission_path.write_bytes(b"not-a-zip")

    with pytest.raises(SubmissionValidationError, match="submission zip is invalid"):
        service.validate_and_prepare_submission(submission_path)


def test_validate_and_prepare_submission_converts_tabular_to_zip_when_required(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "submission_format.md").write_text(
        "## Submission Format\nYou must submit a ZIP file.\n",
        encoding="utf-8",
    )

    sample_path = context_dir / "sample_submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]}).to_csv(sample_path, index=False)

    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.2, 0.8]}).to_csv(submission_path, index=False)

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

    assert prepared.suffix == ".zip"
    with zipfile.ZipFile(prepared, "r") as archive:
        members = sorted(info.filename for info in archive.infolist() if not info.is_dir())
    assert members == ["submission.csv"]
