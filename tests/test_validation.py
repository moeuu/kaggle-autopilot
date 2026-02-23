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


def test_validate_submission_long_format_allows_row_mismatch_and_duplicates(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.tsv"
    submission_path = tmp_path / "submission.tsv"

    sample_path.write_text(
        "id\tterm\tscore\nP1\tGO:0000001\t0.123\nP1\tGO:0000002\t0.456\n",
        encoding="utf-8",
    )
    submission_path.write_text(
        "P1\tGO:0000001\t0.999\nP1\tGO:0000003\t0.888\nP2\tGO:0000002\t0.777\n",
        encoding="utf-8",
    )

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_header_only_sample_allows_row_mismatch(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"

    sample_path.write_text("id,target\n", encoding="utf-8")
    pd.DataFrame({"id": [1, 2, 3], "target": [0.1, 0.2, 0.3]}).to_csv(submission_path, index=False)

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_handles_irregular_tsv(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.tsv"
    submission_path = tmp_path / "submission.csv"
    format_path = tmp_path / "submission_format.md"

    format_path.write_text("## Submission Format\n\nid,term,score\n", encoding="utf-8")
    sample_path.write_text(
        "A0A0C5B5G6\tGO:0000001\t0.123\nA0A0C5B5G6\tText\t0.456\tExtra text column\n",
        encoding="utf-8",
    )
    submission_path.write_text("id,term,score\nA0A0C5B5G6,GO:0000001,0.999\n", encoding="utf-8")

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_handles_tabbed_csv_with_commas(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"

    sample_path.write_text(
        "A0A0C5B5G6\tGO:0000001\t0.123\nA0A0C5B5G6\tText\t0.456\tInhibits, something\n",
        encoding="utf-8",
    )
    submission_path.write_text("A0A0C5B5G6\tGO:0000001\t0.999\n", encoding="utf-8")

    validate_submission(str(sample_path), str(submission_path))


def test_validate_submission_ignores_noisy_format_hint(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    format_path = tmp_path / "submission_format.md"

    format_path.write_text(
        "## Ahoy, welcome to Kaggle! You're in the right place. "
        "This is the legendary Titanic ML competition -- the best, first challenge. "
        "PassengerId,Survived\n",
        encoding="utf-8",
    )
    sample_path.write_text("PassengerId,Survived\n1,0\n", encoding="utf-8")
    submission_path.write_text("PassengerId,Survived\n1,0\n", encoding="utf-8")

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


def test_submission_rate_limit_default_cooldown_is_five_minutes(tmp_path):
    ledger = SubmissionLedger(tmp_path / "ledger.jsonl")
    now = datetime.now(UTC)
    entries = [
        {
            "ts": (now - timedelta(minutes=4)).isoformat(),
            "sha256": "a",
            "fingerprint": "f1",
            "slug": "demo",
            "submission_path": "sub.csv",
            "message": "m1",
            "run_id": "r1",
        }
    ]
    ledger.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.ledger_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    with pytest.raises(SubmissionRateLimitError, match="cooldown"):
        ensure_submission_rate_limit(ledger)
