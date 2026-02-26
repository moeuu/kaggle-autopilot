from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kagglebot.exceptions import SubmissionValidationError
from kagglebot.submission_format import (
    columns_look_plausible,
    extract_submission_section,
    load_submission_format_hint,
    parse_submission_format,
)

_PLACEHOLDER_SAMPLE_MAX_ROWS = 10


def validate_submission(sub_path: str, sample_path: str, *, data_dir: str | Path | None = None) -> None:
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

    sample_delim = _sniff_delimiter(sample_csv, default=_default_delimiter_for_path(sample_csv))
    submission_delim = _sniff_delimiter(submission_csv, default=sample_delim)

    # Read sample header first so we can preserve id-column formatting (e.g., leading zeros).
    try:
        sample_header = pd.read_csv(sample_csv, sep=sample_delim, nrows=0)
        sample_columns = list(sample_header.columns)
    except Exception as exc:  # noqa: BLE001
        raise SubmissionValidationError(f"unable to read sample submission header: {sample_csv}") from exc

    sample_has_data_rows = _has_data_rows(sample_csv)
    expected_columns = sample_columns
    expected_source = "sample_submission.csv"
    hint_columns = _resolve_expected_columns_from_context(sample_csv)
    if hint_columns and _should_prefer_hint_columns(
        sample_has_data_rows=sample_has_data_rows,
        sample_columns=expected_columns,
        hint_columns=hint_columns,
    ):
        expected_columns = hint_columns
        expected_source = "submission_format/overview hint"

    if not expected_columns:
        raise SubmissionValidationError(_format_validation_message(["sample_submission has no columns"]))

    id_col = expected_columns[0]
    # Read frames with id_col forced to string to avoid losing leading zeros.
    try:
        sample = pd.read_csv(sample_csv, sep=sample_delim, dtype={id_col: str})
    except Exception:  # noqa: BLE001
        sample = pd.read_csv(sample_csv, sep=sample_delim)
    try:
        submission = pd.read_csv(submission_csv, sep=submission_delim, dtype={id_col: str})
    except Exception:  # noqa: BLE001
        submission = pd.read_csv(submission_csv, sep=submission_delim)

    actual_columns = list(submission.columns)
    if expected_columns != actual_columns:
        problems.append(
            "columns mismatch (order-sensitive):\n"
            f"  expected ({expected_source}): {expected_columns}\n"
            f"  actual:                     {actual_columns}"
        )
        if len(expected_columns) == len(actual_columns) and not (set(expected_columns) & set(actual_columns)):
            problems.append(
                "submission header does not resemble the expected columns "
                "(the file may be missing a header row or using an unexpected delimiter)"
            )

    expected_row_count = len(sample) if sample_has_data_rows else None
    expected_id_values: set[str] | None = None
    placeholder_sample = False

    if (
        data_dir is not None
        and sample_has_data_rows
        and len(sample) <= _PLACEHOLDER_SAMPLE_MAX_ROWS
        and (id_col in sample.columns)
        and (not sample[id_col].duplicated().any())
    ):
        test_ids = _maybe_load_test_ids(Path(data_dir), id_col=id_col)
        if test_ids is not None:
            sample_ids = sample[id_col].astype(str).tolist()
            if len(test_ids) >= max(len(sample_ids) * 3, len(sample_ids) + 10) and (
                _is_prefix(sample_ids, test_ids) or set(sample_ids).issubset(set(test_ids))
            ):
                placeholder_sample = True
                expected_row_count = len(test_ids)
                expected_id_values = set(test_ids)

    if expected_row_count is not None and len(submission) != expected_row_count:
        problems.append(f"row count mismatch:\n  expected: {expected_row_count}\n  actual:   {len(submission)}")

    if id_col not in submission.columns:
        if len(expected_columns) == len(actual_columns):
            # Most commonly: headerless submissions where the first data row becomes the header.
            problems.append(f"expected id column '{id_col}' is missing (submission appears to be missing a header row)")
        else:
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

        if placeholder_sample and expected_id_values is not None:
            sub_ids = [str(value) for value in id_values.tolist()]
            if set(sub_ids) != expected_id_values:
                missing = sorted(expected_id_values - set(sub_ids))[:5]
                extra = sorted(set(sub_ids) - expected_id_values)[:5]
                problems.append(
                    "id values mismatch (placeholder sample detected; validated against test ids):\n"
                    f"  missing (first 5): {missing}\n"
                    f"  extra (first 5):   {extra}"
                )

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


def _is_prefix(prefix: list[str], values: list[str]) -> bool:
    if not prefix:
        return False
    if len(prefix) > len(values):
        return False
    return all(a == b for a, b in zip(prefix, values, strict=False))


def _maybe_load_test_ids(data_dir: Path, *, id_col: str) -> list[str] | None:
    if not data_dir.exists():
        return None
    try:
        from kagglebot.solver.io import find_competition_files
    except Exception:
        return None
    try:
        _, test_path, _ = find_competition_files(data_dir)
    except Exception:  # noqa: BLE001
        return None
    if not test_path.exists():
        return None
    try:
        delim = _sniff_delimiter(test_path, default=_default_delimiter_for_path(test_path))
        test = pd.read_csv(test_path, sep=delim, usecols=[id_col], dtype={id_col: str})
    except Exception:  # noqa: BLE001
        return None
    if id_col not in test.columns:
        return None
    return [str(value) for value in test[id_col].tolist()]


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


def _should_prefer_hint_columns(
    *,
    sample_has_data_rows: bool,
    sample_columns: list[str],
    hint_columns: list[str],
) -> bool:
    if not hint_columns:
        return False
    if sample_has_data_rows:
        return False
    if not sample_columns:
        return True
    if list(sample_columns) == list(hint_columns):
        return False
    if not columns_look_plausible(sample_columns):
        return True
    return _sample_header_looks_placeholder(sample_columns)


def _sample_header_looks_placeholder(columns: list[str]) -> bool:
    normalized = [str(col).strip().lower() for col in columns if str(col).strip()]
    if len(normalized) != 2:
        return False
    id_like = normalized[0] in {"id", "row_id", "identifier"}
    target_like = normalized[1] in {"target", "label", "y", "value"}
    return bool(id_like and target_like)


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


def _default_delimiter_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return "\t"
    return ","


def _sniff_delimiter(path: Path, *, default: str = ",", max_lines: int = 100) -> str:
    candidates: list[str] = []
    for sep in (default, "\t", ","):
        if sep and sep not in candidates:
            candidates.append(sep)
    counts = {sep: 0 for sep in candidates}
    lines_seen = 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                lines_seen += 1
                for sep in candidates:
                    counts[sep] += line.count(sep)
                if lines_seen >= max_lines:
                    break
    except OSError:
        return default
    if lines_seen == 0:
        return default
    best = max(candidates, key=lambda sep: counts[sep])
    if counts[best] == 0:
        return default
    if counts.get(default, 0) >= counts[best]:
        return default
    return best


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
        if overview_hint.columns and columns_look_plausible(list(overview_hint.columns)):
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
