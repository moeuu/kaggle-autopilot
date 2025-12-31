"""Tests for submission validation helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from kagglebot.validation import validate_submission


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
