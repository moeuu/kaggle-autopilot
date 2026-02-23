from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kagglebot.exceptions import SubmissionValidationError
from kagglebot.submission.validate import validate_submission
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
