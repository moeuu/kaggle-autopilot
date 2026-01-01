"""Tests for run and submission ledgers."""

from __future__ import annotations

import json

import pytest

from kagglebot.exceptions import DuplicateSubmissionError
from kagglebot.history import RunLedger, SubmissionLedger
from kagglebot.validation import ensure_not_duplicate_submission


def test_run_ledger_creates_metadata(tmp_path):
    ledger = RunLedger.for_slug("demo", root=tmp_path)
    record = ledger.start_run(
        slug="demo",
        dry_run=True,
        force=False,
        submission_path=None,
        sample_path=None,
        message=None,
        argv=["kagglebot", "run", "demo"],
    )

    assert record.metadata_path.exists()
    payload = json.loads(record.metadata_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == record.run_id
    assert payload["slug"] == "demo"
    assert payload["dry_run"] is True


def test_submission_ledger_duplicate_detection(tmp_path):
    submission = tmp_path / "submission.csv"
    submission.write_text("id,target\n1,0.1\n", encoding="utf-8")

    ledger = SubmissionLedger.for_slug("demo", root=tmp_path)
    assert ledger.is_duplicate(str(submission)) is False

    ledger.record(str(submission), message="first", run_id="run-1", slug="demo")
    assert ledger.is_duplicate(str(submission)) is True

    with pytest.raises(DuplicateSubmissionError, match="Duplicate submission"):
        ensure_not_duplicate_submission(ledger, str(submission))
