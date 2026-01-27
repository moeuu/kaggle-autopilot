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
    expected_columns = list(sample.columns)
    if _looks_like_markdown_file(sample_path_obj) or _columns_look_like_markdown(expected_columns):
        sample = pd.DataFrame()
        expected_columns = []
    placeholder_sample = False
    sample_has_data_rows = _has_data_rows(sample_path_obj)
    if format_hint is not None and format_hint.columns:
        hint_columns = list(format_hint.columns)
        if not expected_columns or sample.empty or not sample_has_data_rows:
            placeholder_sample = True
            expected_columns = hint_columns
            sample = pd.DataFrame(columns=expected_columns)
    sub = _read_submission_table(Path(submission_path), format_hint=format_hint, expected_columns=expected_columns)

    # 1) Columns must match (including order).
    if expected_columns != list(sub.columns):
        raise ValueError(
            "Submission columns do not match the sample submission file.\n"
            f"Expected: {expected_columns}\n"
            f"Got:      {list(sub.columns)}"
        )

    # 2) Row count must match.
    # 3) Basic id column checks.
    id_col = expected_columns[0]
    if sub[id_col].isna().any():
        raise ValueError(f"Submission contains missing values in id column '{id_col}'.")
    sub_ids = sub[id_col]
    sample_ids = sample[id_col] if not sample.empty else pd.Series([], dtype=object)
    if not sample.empty and sample_ids.isna().any():
        raise ValueError(f"Sample submission contains missing values in id column '{id_col}'.")
    sample_has_duplicates = sample_ids.duplicated().any() if not sample.empty else False
    sub_has_duplicates = sub_ids.duplicated().any()
    if (
        not placeholder_sample
        and format_hint is not None
        and format_hint.columns
        and sample_has_data_rows
        and len(sample) <= 5
        and len(sub) > len(sample)
    ):
        placeholder_sample = True

    # If sample ids are unique, enforce strict row + id matching.
    # If sample ids repeat (long-format submissions), skip row-count/id-set checks.
    # If the sample submission has no data rows (header-only), skip row-count/id-set checks.
    if sample_has_data_rows and not sample_has_duplicates and not placeholder_sample:
        if len(sample) != len(sub):
            raise ValueError(
                "Submission row count does not match the sample submission file.\n"
                f"Expected rows: {len(sample)}\n"
                f"Got rows:      {len(sub)}"
            )
        if sub_has_duplicates:
            raise ValueError(f"Submission contains duplicate values in id column '{id_col}'.")

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


def _looks_like_markdown_file(path: Path, *, max_lines: int = 5) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            lines_checked = 0
            for line in handle:
                if not line.strip():
                    continue
                lines_checked += 1
                stripped = line.lstrip()
                if stripped.startswith("#") or stripped.startswith(">"):
                    return True
                if lines_checked >= max_lines:
                    break
    except OSError:
        return False
    return False


def _columns_look_like_markdown(columns: list[str]) -> bool:
    if not columns:
        return False
    for col in columns:
        text = str(col).strip()
        if not text:
            continue
        lowered = text.lower()
        if text.startswith("#") or "kaggle" in lowered:
            return True
        if "welcome to" in lowered or "competition" in lowered:
            return True
        if len(text) > 30 and " " in text:
            return True
    return False


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
    if frame is not None and _columns_contain_delimiters(frame.columns):
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
    sep = _sniff_delimiter(path, default=sep)
    columns = expected_columns
    if columns is None and format_hint is not None:
        columns = format_hint.columns
    inferred_cols = _infer_column_count(path, sep)
    if columns:
        if inferred_cols is not None and inferred_cols > 0 and inferred_cols < len(columns):
            columns = columns[:inferred_cols]
        try:
            frame = pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip")
            if list(frame.columns) == list(columns):
                return frame
        except Exception:
            pass
    expected_cols = len(columns) if columns else inferred_cols
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


def _sniff_delimiter(path: Path, *, default: str, max_lines: int = 100) -> str:
    candidates = []
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


def _columns_contain_delimiters(columns: list[object]) -> bool:
    if len(columns) != 1:
        return False
    value = str(columns[0])
    return "\t" in value or "," in value


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
