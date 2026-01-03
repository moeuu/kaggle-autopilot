from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from kagglebot.exceptions import DuplicateSubmissionError, SubmissionRateLimitError
from kagglebot.history import SubmissionLedger
from kagglebot.submission_format import load_submission_format_hint


def validate_submission(sample_path: str, submission_path: str) -> None:
    sample_path_obj = Path(sample_path)
    format_hint = load_submission_format_hint(sample_path_obj.with_name("submission_format.md"))
    sample = _read_submission_table(sample_path_obj, format_hint=format_hint)
    sub = _read_submission_table(Path(submission_path), format_hint=format_hint, expected_columns=list(sample.columns))

    # 1) Columns must match (including order).
    if list(sample.columns) != list(sub.columns):
        raise ValueError(
            "Submission columns do not match the sample submission file.\n"
            f"Expected: {list(sample.columns)}\n"
            f"Got:      {list(sub.columns)}"
        )

    # 2) Row count must match.
    if len(sample) != len(sub):
        raise ValueError(
            "Submission row count does not match the sample submission file.\n"
            f"Expected rows: {len(sample)}\n"
            f"Got rows:      {len(sub)}"
        )

    # 3) Basic id column checks.
    id_col = sample.columns[0]
    if sub[id_col].isna().any():
        raise ValueError(f"Submission contains missing values in id column '{id_col}'.")
    if sub[id_col].duplicated().any():
        raise ValueError(f"Submission contains duplicate values in id column '{id_col}'.")

    sample_ids = sample[id_col]
    sub_ids = sub[id_col]
    if sample_ids.isna().any():
        raise ValueError(f"Sample submission contains missing values in id column '{id_col}'.")
    if sample_ids.duplicated().any():
        raise ValueError(f"Sample submission contains duplicate values in id column '{id_col}'.")

    if set(sub_ids) != set(sample_ids):
        missing = sorted(set(sample_ids) - set(sub_ids))
        extra = sorted(set(sub_ids) - set(sample_ids))
        missing_preview = missing[:5]
        extra_preview = extra[:5]
        raise ValueError(
            "Submission id values do not match the sample submission file.\n"
            f"Missing ids (first 5): {missing_preview}\n"
            f"Extra ids (first 5):   {extra_preview}"
        )

    # 4) All-NaN target columns are not allowed (guard against bad output).
    for c in sample.columns[1:]:
        if sub[c].isna().all():
            raise ValueError(f"All values are NaN for target column '{c}'.")


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _read_submission_table(
    path: Path,
    *,
    format_hint=None,
    expected_columns: list[str] | None = None,
) -> pd.DataFrame:
    try:
        frame = _read_table(path)
    except Exception:
        frame = None
    if frame is not None:
        if expected_columns:
            if list(frame.columns) == expected_columns:
                return frame
            if _columns_look_like_header(frame.columns) or set(frame.columns) & set(expected_columns):
                return frame
        else:
            if not format_hint or not getattr(format_hint, "columns", None):
                return frame
            if list(frame.columns) == list(format_hint.columns or []):
                return frame
    return _read_table_relaxed(path, format_hint=format_hint, expected_columns=expected_columns)


def _read_table_relaxed(
    path: Path,
    *,
    format_hint=None,
    expected_columns: list[str] | None = None,
) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    sep = None
    if suffix == ".csv":
        sep = ","
    elif suffix == ".tsv":
        sep = "\t"
    elif suffix == ".txt":
        sep = getattr(format_hint, "delimiter", None) if format_hint is not None else None
        sep = sep or "\t"
    else:
        sep = getattr(format_hint, "delimiter", None) if format_hint is not None else None
        sep = sep or ","
    columns = expected_columns
    if columns is None and format_hint is not None:
        columns = format_hint.columns
    if columns:
        try:
            frame = pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip")
            if list(frame.columns) == list(columns):
                return frame
        except Exception:
            pass
    expected_cols = len(columns) if columns else _infer_column_count(path, sep)
    if expected_cols is None:
        return pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip")
    names = columns or [f"col{i}" for i in range(expected_cols)]
    filtered = _filter_delimited_text(path, sep=sep, expected_cols=expected_cols)
    if filtered is None:
        frame = pd.read_csv(
            path,
            sep=sep,
            header=None,
            names=names,
            usecols=list(range(expected_cols)),
            engine="python",
            on_bad_lines="skip",
        )
    else:
        frame = pd.read_csv(
            io.StringIO(filtered),
            sep=sep,
            header=None,
            names=names,
            engine="python",
        )
    if columns and not frame.empty and list(frame.iloc[0].astype(str)) == list(columns):
        frame = frame.iloc[1:].reset_index(drop=True)
    return frame


def _infer_column_count(path: Path, sep: str, max_lines: int = 200) -> int | None:
    from collections import Counter

    counts: Counter[int] = Counter()
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                counts[len(line.rstrip("\n").split(sep))] += 1
                if sum(counts.values()) >= max_lines:
                    break
    except OSError:
        return None
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _filter_delimited_text(path: Path, *, sep: str, expected_cols: int, max_lines: int | None = None) -> str | None:
    lines: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if len(line.rstrip("\n").split(sep)) != expected_cols:
                    continue
                lines.append(line)
                if max_lines is not None and len(lines) >= max_lines:
                    break
    except OSError:
        return None
    if not lines:
        return None
    return "".join(lines)


def _columns_look_like_header(columns: list[object]) -> bool:
    for col in columns:
        value = str(col)
        if any(ch.isdigit() for ch in value):
            return False
        if ":" in value or "/" in value:
            return False
    return True


def ensure_not_duplicate_submission(
    ledger: SubmissionLedger,
    *,
    slug: str,
    message: str,
    submission_path: str,
) -> None:
    if ledger.is_duplicate(slug=slug, message=message, submission_path=Path(submission_path)):
        raise DuplicateSubmissionError("Duplicate submission detected (hash already recorded).")


def ensure_submission_rate_limit(
    ledger: SubmissionLedger,
    *,
    max_submissions_per_day: int = 5,
    min_hours_between: float = 1.0,
) -> None:
    now = datetime.now(UTC)
    last_ts = ledger.last_submission_time()
    recent = ledger.recent_submission_count(hours=24)

    if recent >= max_submissions_per_day:
        raise SubmissionRateLimitError("Submission rate limit exceeded (max per day).")
    if last_ts is not None:
        elapsed = (now - last_ts).total_seconds() / 3600.0
        if elapsed < min_hours_between:
            raise SubmissionRateLimitError("Submission rate limit exceeded (cooldown).")
