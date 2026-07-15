"""Tests for submission ledgers."""

from __future__ import annotations

import json
from datetime import UTC, datetime

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


def test_submission_ledger_uses_notebook_identity_instead_of_commit_marker_hash(tmp_path):
    submission = tmp_path / "submission.parquet"
    submission.write_bytes(b"same commit marker")
    ledger = SubmissionLedger(tmp_path / "ledger.jsonl")

    first_identity = "kernel:user/first@1"
    second_identity = "kernel:user/second@1"
    ledger.record(
        slug="demo-code-competition",
        message="first notebook",
        submission_path=submission,
        run_id="run-1",
        submission_kind="notebook",
        submission_identity=first_identity,
        submission_ref="12345",
    )

    assert ledger.is_duplicate(
        slug="demo-code-competition",
        message="same notebook again",
        submission_path=submission,
        submission_identity=first_identity,
    )
    assert not ledger.is_duplicate(
        slug="demo-code-competition",
        message="different notebook with the same marker",
        submission_path=submission,
        submission_identity=second_identity,
    )
    row = json.loads(ledger.ledger_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["submission_identity"] == first_identity
    assert row["submission_ref"] == "12345"


def test_submission_ledger_ignores_malformed_lines(tmp_path):
    submission = tmp_path / "submission.csv"
    submission.write_text("id,target\n1,0.1\n", encoding="utf-8")
    ledger = SubmissionLedger(tmp_path / "ledger.jsonl")
    ledger.ledger_path.write_text("{not json\n\n[]\n", encoding="utf-8")

    assert ledger.is_duplicate(slug="demo", message="first", submission_path=submission) is False
    assert ledger.last_submission_time() is None
    assert ledger.recent_submission_count(hours=24) == 0


def test_submission_ledger_parses_timestamps_with_utc_normalization(tmp_path):
    ledger = SubmissionLedger(tmp_path / "ledger.jsonl")
    ledger.ledger_path.write_text(
        "\n".join(
            [
                json.dumps({"event": "submit", "ts": "not-a-date"}),
                json.dumps({"event": "submit", "ts": "2026-06-25T21:00:00+09:00"}),
                json.dumps({"event": "outcome", "ts": "2026-06-26T00:00:00Z"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert ledger.last_submission_time() == datetime(2026, 6, 25, 12, tzinfo=UTC)


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
