from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kagglebot.exceptions import SubmissionValidationError
from kagglebot.submission_format import extract_submission_section, load_submission_format_hint, parse_submission_format


def validate_submission(sub_path: str, sample_path: str) -> None:
    """Strict local validation for Kaggle submissions."""
    sample_csv = Path(sample_path)
    submission_csv = Path(sub_path)

    problems: list[str] = []

    if not sample_csv.exists():
        problems.append(f"sample_submission.csv not found: {sample_csv}")
    if not submission_csv.exists():
        problems.append(f"submission.csv not found: {submission_csv}")
    if problems:
        raise SubmissionValidationError(_format_validation_message(problems))

    sample = pd.read_csv(sample_csv)
    submission = pd.read_csv(submission_csv)

    sample_has_data_rows = _has_data_rows(sample_csv)
    expected_columns = list(sample.columns)
    expected_source = "sample_submission.csv"
    hint_columns = _resolve_expected_columns_from_context(sample_csv)
    if hint_columns and (not expected_columns or sample.empty or not sample_has_data_rows):
        expected_columns = hint_columns
        expected_source = "submission_format/overview hint"

    actual_columns = list(submission.columns)
    if expected_columns != actual_columns:
        problems.append(
            "columns mismatch (order-sensitive):\n"
            f"  expected ({expected_source}): {expected_columns}\n"
            f"  actual:                     {actual_columns}"
        )

    if sample_has_data_rows and len(sample) != len(submission):
        problems.append(f"row count mismatch:\n  expected: {len(sample)}\n  actual:   {len(submission)}")

    if not expected_columns:
        problems.append("sample_submission has no columns")
    else:
        id_col = expected_columns[0]
        if id_col not in submission.columns:
            problems.append(f"id column missing in submission: '{id_col}'")
        else:
            id_values = submission[id_col]
            if id_values.isna().any():
                nan_count = int(id_values.isna().sum())
                problems.append(f"id column '{id_col}' contains NaN values: {nan_count}")
            enforce_unique_id = True
            if not sample_has_data_rows:
                enforce_unique_id = False
            elif id_col in sample.columns and sample[id_col].duplicated().any():
                # Long-format submissions can legitimately repeat ids (e.g., one row per label).
                enforce_unique_id = False
            if enforce_unique_id:
                dup_count = int(id_values.duplicated().sum())
                if dup_count > 0:
                    problems.append(f"id column '{id_col}' contains duplicate values: {dup_count}")

        prediction_columns = [col for col in expected_columns if col != id_col]
        for col in prediction_columns:
            if col not in submission.columns:
                continue

            sample_col = sample[col] if (col in sample.columns and sample_has_data_rows) else pd.Series(dtype=object)
            submission_col = submission[col]
            if _sample_column_is_numeric(sample_col):
                numeric = pd.to_numeric(submission_col, errors="coerce")
                nan_count = int(numeric.isna().sum())
                if nan_count > 0:
                    problems.append(f"prediction column '{col}' contains NaN/non-numeric values: {nan_count}")
                    continue
                values = numeric.to_numpy(dtype=float, copy=False)
                inf_count = int(np.isinf(values).sum())
                if inf_count > 0:
                    problems.append(f"prediction column '{col}' contains +/-inf values: {inf_count}")
                continue

            nan_count = int(submission_col.isna().sum())
            if nan_count > 0:
                problems.append(f"prediction column '{col}' contains NaN values: {nan_count}")
                continue

    if problems:
        raise SubmissionValidationError(_format_validation_message(problems))


def _format_validation_message(problems: list[str]) -> str:
    lines = ["Submission validation failed:", *[f"- {problem}" for problem in problems]]
    return "\n".join(lines)


def _sample_column_is_numeric(sample_col: pd.Series) -> bool:
    if sample_col.empty:
        return False
    if pd.api.types.is_numeric_dtype(sample_col):
        return True
    coerced = pd.to_numeric(sample_col, errors="coerce")
    return coerced.notna().all()


def _has_data_rows(path: Path) -> bool:
    non_empty = 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                non_empty += 1
                if non_empty >= 2:
                    return True
    except OSError:
        return True
    return False


def _resolve_expected_columns_from_context(sample_csv: Path) -> list[str] | None:
    for context_dir in _candidate_context_dirs(sample_csv):
        format_hint = load_submission_format_hint(context_dir / "submission_format.md")
        if format_hint is not None and format_hint.columns:
            return list(format_hint.columns)
        overview_path = context_dir / "overview.md"
        if not overview_path.exists():
            continue
        text = overview_path.read_text(encoding="utf-8", errors="ignore")
        section = extract_submission_section(text) or ""
        if not section.strip():
            continue
        overview_hint = parse_submission_format(section)
        if overview_hint.columns:
            return list(overview_hint.columns)
    return None


def _candidate_context_dirs(sample_csv: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(resolved)

    parent = sample_csv.parent
    add(parent)
    for ancestor in [parent, *parent.parents]:
        add(ancestor / "context")
    return candidates
