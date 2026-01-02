"""Tests for submission validation helpers."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from kagglebot.exceptions import SubmissionRateLimitError
from kagglebot.history import SubmissionLedger
from kagglebot.validation import ensure_submission_rate_limit, validate_submission


def test_validate_submission_success():
    """Test validation passes for matching submissions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_path = Path(tmpdir) / "sample.csv"
        submission_path = Path(tmpdir) / "submission.csv"

        # Create matching sample and submission
        df = pd.DataFrame({"id": [1, 2, 3], "target": [0.5, 0.7, 0.3]})
        df.to_csv(sample_path, index=False)
        df.to_csv(submission_path, index=False)

        # Should not raise
        validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_column_mismatch():
    """Test validation fails when columns don't match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_path = Path(tmpdir) / "sample.csv"
        submission_path = Path(tmpdir) / "submission.csv"

        pd.DataFrame({"id": [1, 2, 3], "target": [0.5, 0.7, 0.3]}).to_csv(sample_path, index=False)
        pd.DataFrame({"id": [1, 2, 3], "score": [0.5, 0.7, 0.3]}).to_csv(submission_path, index=False)

        with pytest.raises(ValueError, match="columns do not match"):
            validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_row_count_mismatch():
    """Test validation fails when row counts don't match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_path = Path(tmpdir) / "sample.csv"
        submission_path = Path(tmpdir) / "submission.csv"

        pd.DataFrame({"id": [1, 2, 3], "target": [0.5, 0.7, 0.3]}).to_csv(sample_path, index=False)
        pd.DataFrame({"id": [1, 2], "target": [0.5, 0.7]}).to_csv(submission_path, index=False)

        with pytest.raises(ValueError, match="row count does not match"):
            validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_missing_id():
    """Test validation fails when id column has missing values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_path = Path(tmpdir) / "sample.csv"
        submission_path = Path(tmpdir) / "submission.csv"

        pd.DataFrame({"id": [1, 2, 3], "target": [0.5, 0.7, 0.3]}).to_csv(sample_path, index=False)
        pd.DataFrame({"id": [1, None, 3], "target": [0.5, 0.7, 0.3]}).to_csv(submission_path, index=False)

        with pytest.raises(ValueError, match="missing values in id column"):
            validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_all_nan_target():
    """Test validation fails when all target values are NaN."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_path = Path(tmpdir) / "sample.csv"
        submission_path = Path(tmpdir) / "submission.csv"

        pd.DataFrame({"id": [1, 2, 3], "target": [0.5, 0.7, 0.3]}).to_csv(sample_path, index=False)
        pd.DataFrame({"id": [1, 2, 3], "target": [None, None, None]}).to_csv(submission_path, index=False)

        with pytest.raises(ValueError, match="All values are NaN"):
            validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_id_mismatch():
    """Test validation fails when ids do not match sample submission."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_path = Path(tmpdir) / "sample.csv"
        submission_path = Path(tmpdir) / "submission.csv"

        pd.DataFrame({"id": [1, 2, 3], "target": [0.5, 0.7, 0.3]}).to_csv(sample_path, index=False)
        pd.DataFrame({"id": [1, 2, 4], "target": [0.5, 0.7, 0.3]}).to_csv(submission_path, index=False)

        with pytest.raises(ValueError, match="id values do not match"):
            validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_duplicate_id():
    """Test validation fails when ids are duplicated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_path = Path(tmpdir) / "sample.csv"
        submission_path = Path(tmpdir) / "submission.csv"

        pd.DataFrame({"id": [1, 2, 3], "target": [0.5, 0.7, 0.3]}).to_csv(sample_path, index=False)
        pd.DataFrame({"id": [1, 1, 3], "target": [0.5, 0.7, 0.3]}).to_csv(submission_path, index=False)

        with pytest.raises(ValueError, match="duplicate values"):
            validate_submission(str(sample_path), str(submission_path))


def test_submission_rate_limit(tmp_path):
    ledger = SubmissionLedger(tmp_path / "ledger.jsonl")
    now = datetime.now(UTC)
    entries = [
        {
            "ts": (now - timedelta(hours=1)).isoformat(),
            "sha256": "a",
            "fingerprint": "f1",
            "slug": "demo",
            "submission_path": "sub.csv",
            "message": "m1",
            "run_id": "r1",
        },
        {
            "ts": (now - timedelta(minutes=10)).isoformat(),
            "sha256": "b",
            "fingerprint": "f2",
            "slug": "demo",
            "submission_path": "sub2.csv",
            "message": "m2",
            "run_id": "r2",
        },
    ]
    ledger.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.ledger_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    with pytest.raises(SubmissionRateLimitError, match="cooldown"):
        ensure_submission_rate_limit(ledger, max_submissions_per_day=5, min_hours_between=1.0)
