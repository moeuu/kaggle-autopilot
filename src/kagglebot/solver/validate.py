from __future__ import annotations

from pathlib import Path

from kagglebot.validation import validate_submission


def validate_submission_file(sample_path: Path, submission_path: Path) -> None:
    validate_submission(str(sample_path), str(submission_path))
