"""Tests for submission ledgers."""

from __future__ import annotations

import pytest

from kagglebot.exceptions import DuplicateSubmissionError
from kagglebot.history import SubmissionLedger
from kagglebot.validation import ensure_not_duplicate_submission


def test_submission_ledger_duplicate_detection(tmp_path):
    submission = tmp_path / "submission.csv"
    submission.write_text("id,target\n1,0.1\n", encoding="utf-8")
    ledger = SubmissionLedger(tmp_path / "ledger.jsonl")
    assert ledger.is_duplicate(slug="demo", message="first", submission_path=submission) is False

    ledger.record(slug="demo", message="first", submission_path=submission, run_id="run-1")
    assert ledger.is_duplicate(slug="demo", message="first", submission_path=submission) is True
    assert ledger.is_duplicate(slug="demo", message="second", submission_path=submission) is False

    with pytest.raises(DuplicateSubmissionError, match="Duplicate submission"):
        ensure_not_duplicate_submission(ledger, slug="demo", message="first", submission_path=str(submission))
