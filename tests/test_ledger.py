"""Tests for submission ledgers."""

from __future__ import annotations

import json

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
    assert ledger.is_duplicate(slug="demo", message="second", submission_path=submission) is True

    with pytest.raises(DuplicateSubmissionError, match="Duplicate submission"):
        ensure_not_duplicate_submission(ledger, slug="demo", message="second", submission_path=str(submission))


def test_submission_ledger_records_offline_metadata_and_outcome(tmp_path):
    submission = tmp_path / "submission.csv"
    submission.write_text("id,target\n1,0.1\n", encoding="utf-8")
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"offline_value":0.918}\n', encoding="utf-8")
    ledger = SubmissionLedger(tmp_path / "ledger.jsonl")

    ledger.record(
        slug="demo",
        message="first",
        submission_path=submission,
        run_id="run-1",
        iteration=3,
        metrics_path=metrics,
        offline_score=0.918,
        score_source="cv",
        pipeline_name="xgb_cb_blend",
    )
    ledger.record_outcome(
        slug="demo",
        message="first",
        submission_path=submission,
        run_id="run-1",
        outcome={"score": 0.9162, "rank": 120, "total_teams": 1331},
    )

    rows = [json.loads(line) for line in ledger.ledger_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["event"] == "submit"
    assert rows[0]["iteration"] == 3
    assert rows[0]["metrics_path"] == str(metrics)
    assert rows[0]["offline_score"] == 0.918
    assert rows[0]["pipeline_name"] == "xgb_cb_blend"
    assert rows[1]["event"] == "outcome"
    assert rows[1]["outcome"]["score"] == 0.9162
