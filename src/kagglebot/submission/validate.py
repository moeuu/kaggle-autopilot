from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from kagglebot.asset_modality import DATA_ASSET_SUFFIXES, TABULAR_DATA_SUFFIXES, asset_suffix, is_data_asset_path
from kagglebot.baseline_tokens import (
    EMPTY_TEXT_PREDICTION_NAME_TOKENS,
    ID_LIKE_COLUMN_NAMES,
    TEXT_PREDICTION_NAME_TOKENS,
)
from kagglebot.exceptions import SubmissionValidationError
from kagglebot.solver.io import read_table
from kagglebot.submission_fidelity import build_identifier_cardinality_contract
from kagglebot.submission_format import (
    columns_look_plausible,
    extract_submission_section,
    load_submission_format_hint,
    parse_submission_format,
)
from kagglebot.submission_sample_discovery import (
    TABULAR_TEXT_SUFFIXES,
    default_delimited_text_separator,
    open_tabular_text,
    sniff_tabular_text_delimiter,
    tabular_file_has_data_rows,
    tabular_suffix,
)
from kagglebot.table_columns import normalize_table_column_names

_PLACEHOLDER_SAMPLE_MAX_ROWS = 100
_BACKTICK_TOKEN_RE = re.compile(r"`([^`\n]+)`")
_FILENAME_TOKEN_RE = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9._-]*\.(?P<ext>[A-Za-z0-9]{2,8})\b")
_COORD_COL_RE = re.compile(r"^(?:x|y|z)_\d+$", re.IGNORECASE)
_HIDDEN_FULL_TEST_CONTEXT_RE = re.compile(
    r"\bhidden(?:/|\s+or\s+|\s+and\s+|\s+)?(?:full\s+)?test\b"
    r"|\bfull\s+test\b"
    r"|\bpublic\s+test\s+set\s+is\s+dummy\b"
    r"|\bdummy\s+(?:public\s+)?test\b"
    r"|\bcode\s+competition\b"
    r"|\bnotebook(?:-only)?\s+submission",
    re.IGNORECASE,
)
_INCOMPLETE_DATA_RELEASE_CONTEXT_RE = re.compile(
    r"\bonly\s+(?:a\s+)?sample\s+of\s+the\s+(?:training\s+)?dataset\s+has\s+been\s+released\b"
    r"|\bfull\s+dataset\b.{0,120}\b(?:expected|will\s+be|to\s+be)\s+released\b",
    re.IGNORECASE | re.DOTALL,
)
_VARIABLE_INSTANCE_ROW_CONTEXT_RE = re.compile(
    r"\b(?:each|one)\s+row\s+(?:corresponds?\s+to|per)\s+"
    r"(?:one|a|each)?\s*(?:predicted|detected)?\s*(?:instance|object|item|event|mask|segment|filament)\b"
    r"|\b(?:tail\s+strings?|id\s+suffix(?:es)?)\b.{0,120}\b(?:rows?|ids?)\b.{0,60}\bunique\b"
    r"|\b(?:variable|arbitrary)\s+(?:number\s+of\s+)?rows?\s+per\s+(?:image|sample|entity|record)\b",
    re.IGNORECASE | re.DOTALL,
)
_COCO_RLE_CONTEXT_RE = re.compile(
    r"\bpycocotools\b|\b(?:compressed\s+)?coco[- ]?rle\b|\brle\s+counts?\b", re.IGNORECASE
)
_FIXED_MASK_SIZE_RE = re.compile(r"\b(?:size\D{0,30})?(\d{1,6})\s*[x×]\s*(\d{1,6})\s*(?:pixels?)?\b", re.IGNORECASE)
_FILE_ID_ASSET_SUFFIXES = DATA_ASSET_SUFFIXES - TABULAR_DATA_SUFFIXES


def validate_submission(sub_path: str, sample_path: str, *, data_dir: str | Path | None = None) -> None:
    """Strict local validation for Kaggle submissions."""
    sample_csv = Path(sample_path)
    submission_csv = Path(sub_path)

    problems: list[str] = []

    if not sample_csv.exists():
        problems.append(f"sample submission file not found: {sample_csv}")
    if not submission_csv.exists():
        problems.append(f"submission file not found: {submission_csv}")
    if problems:
        raise SubmissionValidationError(_format_validation_message(problems))

    sample_delim = _sniff_delimiter(sample_csv, default=_default_delimiter_for_path(sample_csv))
    submission_delim = _sniff_delimiter(submission_csv, default=sample_delim)

    # Read sample header first so we can preserve id-column formatting (e.g., leading zeros).
    try:
        sample_columns = _read_tabular_columns(sample_csv, sep=sample_delim)
    except Exception as exc:  # noqa: BLE001
        raise SubmissionValidationError(f"unable to read sample submission header: {sample_csv}") from exc
    if _columns_look_like_markdown(sample_columns):
        sample_columns = []

    sample_has_data_rows = _has_data_rows(sample_csv)
    expected_columns = sample_columns
    expected_source = sample_csv.name
    hint_columns = _resolve_expected_columns_from_context(sample_csv)
    sample_headerless = False
    if hint_columns and _should_prefer_hint_columns(
        sample_has_data_rows=sample_has_data_rows,
        sample_columns=expected_columns,
        hint_columns=hint_columns,
        sample_csv=sample_csv,
    ):
        sample_headerless = bool(sample_columns and _columns_look_headerless(sample_columns))
        expected_columns = hint_columns
        expected_source = "submission_format/overview hint"
    elif sample_columns and _columns_look_headerless(sample_columns):
        sample_headerless = True
        expected_columns = [f"col{idx}" for idx in range(len(sample_columns))]
        expected_source = f"{sample_csv.name} inferred headerless columns"

    if not expected_columns:
        raise SubmissionValidationError(_format_validation_message(["sample_submission has no columns"]))

    data_dir_path = Path(data_dir) if data_dir is not None else None
    id_col = _resolve_validation_id_column(expected_columns, data_dir_path=data_dir_path)
    # Read frames with id_col forced to string to avoid losing leading zeros.
    sample = _read_tabular_frame(
        sample_csv,
        sep=sample_delim,
        id_col=id_col,
        expected_columns=expected_columns,
        allow_headerless_fallback=sample_headerless,
    )
    sample_duplicate_ids = bool(sample_has_data_rows and id_col in sample.columns and sample[id_col].duplicated().any())
    submission_headerless_allowed = sample_headerless or (
        sample_duplicate_ids and not _columns_are_generic_placeholders(expected_columns)
    )
    submission = _read_tabular_frame(
        submission_csv,
        sep=submission_delim,
        id_col=id_col,
        expected_columns=expected_columns,
        allow_headerless_fallback=submission_headerless_allowed,
    )

    if len(submission) == 0:
        problems.append("submission has no data rows")

    actual_columns = list(submission.columns)
    if hint_columns and expected_columns == hint_columns:
        expected_with_anchor = _maybe_expected_columns_with_actual_anchor(
            actual_columns=actual_columns,
            hint_columns=hint_columns,
        )
        if expected_with_anchor is not None:
            expected_columns = expected_with_anchor
            expected_source = f"{expected_source} plus leading anchor column"
            id_col = _resolve_validation_id_column(expected_columns, data_dir_path=data_dir_path)

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
    if id_col is not None and sample_duplicate_ids:
        expected_row_count = None
    expected_id_values: set[str] | None = None
    expected_ids_source: str | None = None
    placeholder_sample = False

    if data_dir_path is not None and not sample_has_data_rows:
        eval_ids = _maybe_load_evaluation_ids(data_dir_path) if id_col is not None else None
        if eval_ids is not None:
            expected_row_count = len(eval_ids)
            expected_id_values = set(eval_ids)
            expected_ids_source = "evaluation directory ids"
        elif id_col is None:
            test_row_count = discover_test_row_count(data_dir_path)
            if test_row_count is not None:
                expected_row_count = test_row_count
        else:
            test_ids = discover_test_ids(data_dir_path, id_col=id_col)
            if test_ids is not None:
                expected_row_count = len(test_ids)
                expected_id_values = set(test_ids)
                expected_ids_source = "test data ids"

    if (
        id_col is not None
        and data_dir_path is not None
        and sample_has_data_rows
        and len(sample) <= _PLACEHOLDER_SAMPLE_MAX_ROWS
        and (id_col in sample.columns)
        and (not sample[id_col].duplicated().any())
    ):
        test_ids = discover_test_ids(data_dir_path, id_col=id_col)
        if test_ids is not None:
            sample_ids = sample[id_col].astype(str).tolist()
            if len(test_ids) >= max(len(sample_ids) * 3, len(sample_ids) + 10) and (
                _is_prefix(sample_ids, test_ids) or set(sample_ids).issubset(set(test_ids))
            ):
                placeholder_sample = True
                expected_row_count = len(test_ids)
                expected_id_values = set(test_ids)
    elif (
        id_col is None
        and data_dir_path is not None
        and sample_has_data_rows
        and len(sample) <= _PLACEHOLDER_SAMPLE_MAX_ROWS
    ):
        test_row_count = discover_test_row_count(data_dir_path)
        if test_row_count is not None and test_row_count > len(sample):
            placeholder_sample = True
            expected_row_count = test_row_count

    variable_instance_rows = _allows_variable_instance_rows(
        sample_csv=sample_csv,
        submission_csv=submission_csv,
    )
    if variable_instance_rows:
        expected_row_count = None
        expected_id_values = None
        placeholder_sample = False

    if (
        sample_has_data_rows
        and len(sample) <= _PLACEHOLDER_SAMPLE_MAX_ROWS
        and len(submission) <= len(sample)
        and not _looks_like_wide_single_row_submission(sample=sample, expected_columns=expected_columns)
        and (_has_hidden_full_test_context(sample_csv) or _has_incomplete_data_release_context(sample_csv))
    ):
        problems.append(
            "tiny static submission appears to use public placeholder rows for a hidden/full-test notebook "
            "competition; generate the submission artifact from runtime test ids or use notebook inference mode"
        )

    if expected_row_count is not None and len(submission) != expected_row_count:
        problems.append(f"row count mismatch:\n  expected: {expected_row_count}\n  actual:   {len(submission)}")

    anchor_columns = _resolve_anchor_columns(expected_columns)
    if sample_has_data_rows and anchor_columns:
        for column in anchor_columns:
            if column not in sample.columns or column not in submission.columns:
                continue
            sample_values = sample[column].where(sample[column].notna(), "").astype(str)
            submission_values = submission[column].where(submission[column].notna(), "").astype(str)
            if len(sample_values) != len(submission_values):
                continue
            if not sample_values.equals(submission_values):
                problems.append(
                    f"anchor column '{column}' must match {sample_csv.name} exactly for structured outputs."
                )

    if id_col is None:
        sub_ids: list[str] = []
    elif id_col not in submission.columns:
        if len(expected_columns) == len(actual_columns):
            # Most commonly: headerless submissions where the first data row becomes the header.
            problems.append(f"expected id column '{id_col}' is missing (submission appears to be missing a header row)")
        else:
            problems.append(f"id column missing in submission: '{id_col}'")
    else:
        id_values = submission[id_col]
        sub_ids = [str(value).strip() for value in id_values.tolist()]
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

        if variable_instance_rows:
            pass
        elif expected_id_values is not None:
            if set(sub_ids) != expected_id_values:
                missing = sorted(expected_id_values - set(sub_ids))[:5]
                extra = sorted(set(sub_ids) - expected_id_values)[:5]
                if placeholder_sample:
                    source_msg = "placeholder sample detected; validated against test ids"
                elif expected_ids_source:
                    source_msg = f"header-only sample detected; validated against {expected_ids_source}"
                else:
                    source_msg = "validated against expected test ids"
                problems.append(
                    f"id values mismatch ({source_msg}):\n  missing (first 5): {missing}\n  extra (first 5):   {extra}"
                )
        elif sample_has_data_rows and enforce_unique_id and id_col in sample.columns:
            sample_id_values = sample[id_col]
            if sample_id_values.isna().any():
                nan_count = int(sample_id_values.isna().sum())
                problems.append(f"sample id column '{id_col}' contains NaN values: {nan_count}")
            else:
                sample_ids = [str(value).strip() for value in sample_id_values.tolist()]
                if set(sub_ids) != set(sample_ids):
                    missing = sorted(set(sample_ids) - set(sub_ids))[:5]
                    extra = sorted(set(sub_ids) - set(sample_ids))[:5]
                    problems.append(
                        "id values mismatch (sample submission ids):\n"
                        f"  missing (first 5): {missing}\n"
                        f"  extra (first 5):   {extra}"
                    )
        required_id_suffix = infer_required_id_suffix(
            sample_csv=sample_csv,
            data_dir=data_dir_path,
            submission_ids=sub_ids,
        )
        if required_id_suffix:
            suffix_mismatches = [sid for sid in sub_ids if sid and asset_suffix(Path(sid)) != required_id_suffix]
            if suffix_mismatches:
                preview = suffix_mismatches[:5]
                problems.append(
                    f"id values appear to require '{required_id_suffix}' suffix based on context/data files "
                    f"(first 5 mismatches: {preview})"
                )

    prediction_columns = [col for col in expected_columns if id_col is None or col != id_col]
    for col in prediction_columns:
        if col not in submission.columns:
            continue

        sample_col = sample[col] if (col in sample.columns and sample_has_data_rows) else pd.Series(dtype=object)
        submission_col = submission[col]
        if _sample_column_is_numeric(sample_col) and not _prediction_column_allows_text_values(
            column=col,
            sample_csv=sample_csv,
        ):
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
            if _prediction_column_allows_empty_text_values(column=col, sample_csv=sample_csv):
                continue
            problems.append(f"prediction column '{col}' contains NaN values: {nan_count}")
            continue
        coco_shape = _declared_coco_rle_shape(column=col, sample_csv=sample_csv)
        if coco_shape is not None:
            invalid_count = 0
            first_error = ""
            for value in submission_col.astype(str):
                try:
                    counts = _decode_compressed_coco_rle_counts(value)
                    if sum(counts) != coco_shape[0] * coco_shape[1]:
                        raise ValueError(
                            f"run sum {sum(counts)} does not match declared mask size {coco_shape[0]}x{coco_shape[1]}"
                        )
                    if sum(counts[1::2]) <= 0:
                        raise ValueError("mask area is zero")
                except (IndexError, TypeError, ValueError) as exc:
                    invalid_count += 1
                    if not first_error:
                        first_error = str(exc)
            if invalid_count:
                problems.append(
                    f"prediction column '{col}' contains invalid compressed COCO RLE values: "
                    f"{invalid_count} (first error: {first_error})"
                )

    if problems:
        raise SubmissionValidationError(_format_validation_message(problems))


def _allows_variable_instance_rows(*, sample_csv: Path, submission_csv: Path) -> bool:
    if not _context_declares_variable_instance_rows(sample_csv):
        return False
    cardinality = build_identifier_cardinality_contract(
        sample_submission_path=sample_csv,
        submission_path=submission_csv,
        metrics=None,
        metrics_path=None,
    )
    return bool(cardinality.get("eligible"))


def _context_declares_variable_instance_rows(sample_csv: Path) -> bool:
    for context_dir in _candidate_context_dirs(sample_csv):
        for name in ("submission_format.md", "overview.md", "data.md", "rules.md"):
            path = context_dir / name
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if _VARIABLE_INSTANCE_ROW_CONTEXT_RE.search(text):
                return True
    return False


def _declared_coco_rle_shape(*, column: str, sample_csv: Path) -> tuple[int, int] | None:
    if "rle" not in column.strip().lower():
        return None
    for context_dir in _candidate_context_dirs(sample_csv):
        for name in ("submission_format.md", "overview.md", "data.md", "rules.md"):
            path = context_dir / name
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if not _COCO_RLE_CONTEXT_RE.search(text):
                continue
            match = _FIXED_MASK_SIZE_RE.search(text)
            if match is not None:
                return int(match.group(1)), int(match.group(2))
    return None


def _decode_compressed_coco_rle_counts(value: str) -> list[int]:
    if not value or value.strip() != value or value in {"0", "nan", "inf", "-inf"}:
        raise ValueError("RLE counts string is empty or a placeholder")
    counts: list[int] = []
    position = 0
    while position < len(value):
        decoded = 0
        shift = 0
        while True:
            if position >= len(value):
                raise ValueError("truncated compressed COCO RLE value")
            char = ord(value[position]) - 48
            position += 1
            if char < 0 or char > 0x3F:
                raise ValueError("compressed COCO RLE contains an out-of-range character")
            decoded |= (char & 0x1F) << (5 * shift)
            more = bool(char & 0x20)
            shift += 1
            if not more:
                if char & 0x10:
                    decoded |= -1 << (5 * shift)
                break
        if len(counts) > 2:
            decoded += counts[-2]
        if decoded < 0:
            raise ValueError("compressed COCO RLE contains a negative run")
        counts.append(decoded)
    return counts


def _prediction_column_allows_text_values(*, column: str, sample_csv: Path) -> bool:
    lowered_column = column.strip().lower()
    if any(token in lowered_column for token in TEXT_PREDICTION_NAME_TOKENS):
        return True
    for context_dir in _candidate_context_dirs(sample_csv):
        for name in ("submission_format.md", "data.md", "overview.md"):
            path = context_dir / name
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            text_markers = ("semicolon-separated", "citations", "empty string")
            if lowered_column in text and any(token in text for token in text_markers):
                return True
    return False


def _prediction_column_allows_empty_text_values(*, column: str, sample_csv: Path) -> bool:
    lowered_column = column.strip().lower()
    if not _prediction_column_allows_text_values(column=column, sample_csv=sample_csv):
        return False
    if any(token in lowered_column for token in EMPTY_TEXT_PREDICTION_NAME_TOKENS):
        return True
    empty_markers = (
        "empty string",
        "empty values",
        "blank",
        "missing prediction",
        "no prediction",
    )
    for context_dir in _candidate_context_dirs(sample_csv):
        for name in ("submission_format.md", "data.md", "overview.md"):
            path = context_dir / name
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            if lowered_column in text and any(marker in text for marker in empty_markers):
                return True
    return False


def _is_prefix(prefix: list[str], values: list[str]) -> bool:
    if not prefix:
        return False
    if len(prefix) > len(values):
        return False
    return all(a == b for a, b in zip(prefix, values, strict=False))


def _looks_like_wide_single_row_submission(*, sample: pd.DataFrame, expected_columns: list[str]) -> bool:
    if len(sample) != 1 or len(expected_columns) <= _PLACEHOLDER_SAMPLE_MAX_ROWS:
        return False
    return not _looks_like_validation_id_column(str(expected_columns[0]))


def _looks_like_validation_id_column(column: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
    if not normalized:
        return False
    compact = normalized.replace("_", "")
    if normalized in ID_LIKE_COLUMN_NAMES or compact in ID_LIKE_COLUMN_NAMES:
        return True
    if compact in {
        "id",
        "idx",
        "index",
        "rowid",
        "recordid",
        "sampleid",
        "imageid",
        "fileid",
        "filename",
        "file",
        "path",
        "name",
        "caseid",
        "patientid",
        "objectid",
        "seriesid",
    }:
        return True
    return compact.endswith("id") or compact.endswith("identifier")


def _resolve_validation_id_column(expected_columns: list[str], *, data_dir_path: Path | None) -> str | None:
    if not expected_columns:
        return None
    candidate = str(expected_columns[0])
    if _looks_like_validation_id_column(candidate):
        return candidate
    if len(expected_columns) == 1:
        return None
    test = _maybe_load_tabular_test_frame(data_dir_path) if data_dir_path is not None else None
    if test is not None and candidate in [str(col) for col in test.columns]:
        return candidate
    if data_dir_path is not None:
        return None
    return candidate


def discover_test_row_count(data_dir: Path) -> int | None:
    """Discover expected test row count from tabular test files or recognized test assets."""
    if not data_dir.exists():
        return None
    test = _maybe_load_tabular_test_frame(data_dir)
    if test is not None:
        return len(test)
    asset_ids = _maybe_load_test_asset_ids(data_dir)
    return len(asset_ids) if asset_ids is not None else None


def discover_test_ids(data_dir: Path, *, id_col: str) -> list[str] | None:
    """Discover expected test ids from tabular test files or recognized test assets."""
    if not data_dir.exists():
        return None
    test = _maybe_load_tabular_test_frame(data_dir)
    if test is not None and id_col in test.columns:
        return [str(value) for value in test[id_col].astype(str).tolist()]
    return _maybe_load_test_asset_ids(data_dir)


def _maybe_load_tabular_test_frame(data_dir: Path | None) -> pd.DataFrame | None:
    if data_dir is None:
        return None
    try:
        from kagglebot.solver.io import find_competition_files, read_table
    except Exception:
        return None
    try:
        _, test_path, _ = find_competition_files(data_dir)
    except Exception:  # noqa: BLE001
        return None
    if not test_path.exists():
        return None
    try:
        test = read_table(test_path)
    except Exception:  # noqa: BLE001
        return None
    test.columns = [str(col) for col in test.columns]
    return test


def _maybe_load_test_asset_ids(data_dir: Path) -> list[str] | None:
    ids: set[str] = set()
    for path in data_dir.rglob("*"):
        if not _is_file_id_asset_path(path):
            continue
        try:
            parts = path.relative_to(data_dir).parts
        except ValueError:
            parts = path.parts
        if not any("test" in part.lower() for part in parts):
            continue
        ids.add(path.name)
    if not ids:
        return None
    return sorted(ids)


def _is_file_id_asset_path(path: Path) -> bool:
    if not is_data_asset_path(path):
        return False
    return asset_suffix(path) in _FILE_ID_ASSET_SUFFIXES


def _maybe_load_evaluation_ids(data_dir: Path) -> list[str] | None:
    eval_root = data_dir / "ICPR02" / "kaggle" / "evaluation"
    if not eval_root.exists() or not eval_root.is_dir():
        return None

    sample_ids: list[str] = []
    for entry in sorted(eval_root.iterdir()):
        if not entry.is_dir():
            continue
        if any(entry.glob("B*.tif")):
            sample_ids.append(entry.name)
            continue
        nested = [d for d in entry.iterdir() if d.is_dir() and any(d.glob("B*.tif"))]
        if nested:
            sample_ids.append(entry.name)
    if not sample_ids:
        return None
    return sample_ids


def _resolve_anchor_columns(expected_columns: list[str]) -> list[str]:
    coord_positions = [index for index, column in enumerate(expected_columns) if _COORD_COL_RE.fullmatch(str(column))]
    if not coord_positions:
        return []
    first_coord = min(coord_positions)
    if first_coord <= 1:
        return []
    return expected_columns[1:first_coord]


def infer_required_id_suffix(*, sample_csv: Path, data_dir: Path | None, submission_ids: list[str]) -> str | None:
    """Infer a required id-file suffix (e.g., '.tif') when evidence is strong."""
    if data_dir is None or not data_dir.exists():
        return None
    id_stems = {
        stem
        for value in (str(raw).strip() for raw in submission_ids)
        if (stem := _submission_id_stem_for_suffix_inference(value))
    }
    if not id_stems:
        return None
    if _real_sample_has_suffixless_primary_ids(sample_csv):
        return None

    suffix_counts: dict[str, int] = {}
    for path in data_dir.rglob("*"):
        if not _is_file_id_asset_path(path):
            continue
        suffix = asset_suffix(path)
        stem = _remove_suffix(path.name, suffix)
        if stem not in id_stems:
            continue
        if not suffix:
            continue
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1

    if not suffix_counts:
        return None

    required_matches = len(id_stems)
    full_coverage = sorted(suffix for suffix, count in suffix_counts.items() if count >= required_matches)
    if not full_coverage:
        return None

    preferred = _preferred_id_suffix_from_context(sample_csv)
    if preferred and preferred in full_coverage:
        return preferred
    if len(full_coverage) == 1:
        return full_coverage[0]
    return None


def normalize_id_with_required_suffix(value: object, required_suffix: str) -> str:
    raw = str(value).strip()
    if not raw or not required_suffix:
        return raw
    current_suffix = asset_suffix(Path(raw))
    if current_suffix == required_suffix:
        return raw
    if current_suffix in _FILE_ID_ASSET_SUFFIXES:
        stem = _remove_suffix(raw, current_suffix)
        return f"{stem}{required_suffix}" if stem else raw
    if Path(raw).suffix:
        return raw
    return f"{raw}{required_suffix}"


def _submission_id_stem_for_suffix_inference(value: str) -> str | None:
    if not value:
        return None
    current_suffix = asset_suffix(Path(value))
    if current_suffix in _FILE_ID_ASSET_SUFFIXES:
        stem = _remove_suffix(value, current_suffix)
        return stem or None
    if Path(value).suffix:
        return None
    return value


def _remove_suffix(name: str, suffix: str) -> str:
    if suffix and name.lower().endswith(suffix):
        return name[: -len(suffix)]
    return Path(name).stem


def _real_sample_has_suffixless_primary_ids(sample_csv: Path) -> bool:
    if not _has_data_rows(sample_csv) or _is_synthesized_sample_submission(sample_csv):
        return False
    try:
        delim = _sniff_delimiter(sample_csv, default=_default_delimiter_for_path(sample_csv))
        sample_columns = _read_tabular_columns(sample_csv, sep=delim)
        if not sample_columns:
            return False
        id_col = sample_columns[0]
        sample = _read_tabular_frame(sample_csv, sep=delim, id_col=id_col)
        if id_col not in sample.columns:
            return False
        sample_ids = sample[id_col]
    except Exception:  # noqa: BLE001
        return False
    for raw in sample_ids.tolist():
        if pd.isna(raw):
            continue
        value = str(raw).strip()
        if value and not Path(value).suffix:
            return True
    return False


def _preferred_id_suffix_from_context(sample_csv: Path) -> str | None:
    counts: dict[str, int] = {}
    for context_dir in _candidate_context_dirs(sample_csv):
        for name in ("submission_format.md", "overview.md", "data.md", "rules.md", "discussion.md"):
            path = context_dir / name
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            section = extract_submission_section(text) or ""
            if not section.strip():
                continue
            for suffix in _extract_suffix_tokens(section):
                counts[suffix] = counts.get(suffix, 0) + 1
    if not counts:
        return None
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ranked[0][0]


def _extract_suffix_tokens(section: str) -> list[str]:
    suffixes: list[str] = []
    for token in _BACKTICK_TOKEN_RE.findall(section):
        suffix = _context_token_suffix(token)
        if _suffix_looks_plausible(suffix):
            suffixes.append(suffix)
    for match in _FILENAME_TOKEN_RE.finditer(section):
        suffix = _context_token_suffix(match.group(0))
        if _suffix_looks_plausible(suffix):
            suffixes.append(suffix)
    return suffixes


def _context_token_suffix(token: str) -> str:
    name = Path(str(token or "").strip())
    suffix = asset_suffix(name)
    if suffix in DATA_ASSET_SUFFIXES:
        return suffix
    return name.suffix.lower()


def _suffix_looks_plausible(suffix: str) -> bool:
    if not suffix.startswith("."):
        return False
    if suffix in DATA_ASSET_SUFFIXES:
        return True
    ext = suffix[1:]
    return 2 <= len(ext) <= 8 and ext.isalnum()


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
    sample_csv: Path,
) -> bool:
    if not hint_columns:
        return False
    if not sample_columns:
        return True
    if list(sample_columns) == list(hint_columns):
        return False
    if _columns_look_headerless(sample_columns):
        return True
    if sample_has_data_rows:
        return False
    if not _sample_header_looks_placeholder(sample_columns):
        return False
    if _hint_columns_strong_enough_to_override_placeholder(hint_columns):
        return True
    return _is_synthesized_sample_submission(sample_csv)


def _is_synthesized_sample_submission(path: Path) -> bool:
    name = path.name.lower()
    if "sample_submission_synth" in name:
        return True
    if ".kagglebot_cache" in {part.lower() for part in path.parts}:
        return True
    return False


def _sample_header_looks_placeholder(columns: list[str]) -> bool:
    normalized = [str(col).strip().lower() for col in columns if str(col).strip()]
    if len(normalized) != 2:
        return False
    id_like = normalized[0] in {"id", "row_id", "identifier"}
    target_like = normalized[1] in {
        "target",
        "label",
        "y",
        "value",
        "prediction",
        "pred",
        "category",
        "class",
        "classes",
    }
    return bool(id_like and target_like)


def _hint_columns_strong_enough_to_override_placeholder(columns: list[str]) -> bool:
    if not columns_look_plausible(columns):
        return False
    normalized = [str(col).strip().lower() for col in columns if str(col).strip()]
    if _sample_header_looks_placeholder(columns):
        return False
    documentation_table_headers = {
        ("column", "meaning"),
        ("column", "description"),
        ("column", "type"),
        ("field", "description"),
        ("name", "description"),
    }
    if tuple(normalized) in documentation_table_headers:
        return False
    generic_terms = {"column", "columns", "field", "name", "meaning", "description", "type", "format", "required"}
    return not all(col in generic_terms for col in normalized)


def _maybe_expected_columns_with_actual_anchor(
    *,
    actual_columns: list[str],
    hint_columns: list[str],
) -> list[str] | None:
    if len(actual_columns) != len(hint_columns) + 1:
        return None
    actual_targets = actual_columns[1:]
    if _normalized_columns(actual_targets) != _normalized_columns(hint_columns):
        return None
    return list(actual_columns)


def _normalized_columns(columns: list[str]) -> list[str]:
    return [str(col).strip().lower() for col in columns]


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
    return tabular_file_has_data_rows(path)


def _read_tabular_columns(path: Path, *, sep: str) -> list[str]:
    if _is_delimited_text_table(path):
        return list(_read_delimited_text_frame(path, sep=sep, nrows=0).columns)
    return list(_read_tabular_frame(path, sep=sep).columns)


def _read_tabular_frame(
    path: Path,
    *,
    sep: str,
    id_col: str | None = None,
    expected_columns: list[str] | None = None,
    allow_headerless_fallback: bool = False,
) -> pd.DataFrame:
    if _is_delimited_text_table(path):
        try:
            frame = _read_delimited_text_frame(path, sep=sep, dtype={id_col: str} if id_col else None)
        except Exception:  # noqa: BLE001
            frame = _read_delimited_text_frame(path, sep=sep)
        if (
            allow_headerless_fallback
            and expected_columns
            and list(frame.columns) != list(expected_columns)
            and _columns_look_headerless([str(column) for column in frame.columns])
        ):
            try:
                return _read_delimited_text_frame(
                    path,
                    sep=sep,
                    header=None,
                    names=expected_columns,
                    dtype={id_col: str} if id_col else None,
                    engine="python",
                    on_bad_lines="skip",
                )
            except Exception:  # noqa: BLE001
                return frame
        return frame
    frame = read_table(path)
    if id_col and id_col in frame.columns:
        frame[id_col] = frame[id_col].astype(str)
    return frame


def _read_delimited_text_frame(path: Path, **kwargs) -> pd.DataFrame:
    with open_tabular_text(path) as handle:
        frame = pd.read_csv(StringIO(handle.read()), **kwargs)
    frame.columns = normalize_table_column_names(frame.columns)
    return frame


def _columns_look_headerless(columns: list[str]) -> bool:
    if not columns:
        return False
    if len(columns) == 1 and _column_name_looks_like_prediction_header(columns[0]):
        return False
    if not columns_look_plausible(columns):
        return True
    data_value_count = sum(1 for column in columns if _column_name_looks_like_data_value(str(column)))
    return data_value_count >= max(1, len(columns) // 2)


def _column_name_looks_like_prediction_header(column: str) -> bool:
    compact = re.sub(r"[^a-z0-9]+", "", str(column).strip().lower())
    return compact in {
        "target",
        "prediction",
        "pred",
        "label",
        "score",
        "probability",
        "prob",
        "value",
        "y",
    }


def _column_name_looks_like_data_value(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    try:
        float(text)
    except ValueError:
        pass
    else:
        return True
    if re.fullmatch(r"[A-Z]{1,10}:\d{3,}", text):
        return True
    if re.fullmatch(r"(?:col|column)\d+", text, re.IGNORECASE):
        return False
    if re.fullmatch(r"[xyz]_\d+", text, re.IGNORECASE):
        return False
    if re.fullmatch(r"[A-Za-z]*\d+[A-Za-z0-9_.:-]*", text):
        return True
    return False


def _columns_are_generic_placeholders(columns: list[str]) -> bool:
    return all(re.fullmatch(r"col\d+", str(column).strip(), re.IGNORECASE) for column in columns)


def _default_delimiter_for_path(path: Path) -> str:
    return default_delimited_text_separator(tabular_suffix(path))


def _sniff_delimiter(path: Path, *, default: str = ",", max_lines: int = 100) -> str:
    if not _is_delimited_text_table(path):
        return default
    try:
        return sniff_tabular_text_delimiter(path)
    except Exception:  # noqa: BLE001
        pass
    candidates: list[str] = []
    for sep in (default, "\t", ",", ";", "|"):
        if sep and sep not in candidates:
            candidates.append(sep)
    counts = {sep: 0 for sep in candidates}
    first_counts: dict[str, int] | None = None
    lines_seen = 0
    try:
        with open_tabular_text(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                lines_seen += 1
                for sep in candidates:
                    counts[sep] += line.count(sep)
                if first_counts is None:
                    first_counts = {sep: line.count(sep) for sep in candidates}
                if lines_seen >= max_lines:
                    break
    except OSError:
        return default
    if lines_seen == 0:
        return default
    if first_counts is not None:
        if first_counts.get(default, 0) > 0:
            return default
        first_best = max(candidates, key=lambda sep: first_counts[sep])
        if first_counts[first_best] > 0:
            return first_best
    best = max(candidates, key=lambda sep: counts[sep])
    if counts[best] == 0:
        return default
    if counts.get(default, 0) >= counts[best]:
        return default
    return best


def _is_delimited_text_table(path: Path) -> bool:
    return tabular_suffix(path) in TABULAR_TEXT_SUFFIXES


def _resolve_expected_columns_from_context(sample_csv: Path) -> list[str] | None:
    for context_dir in _candidate_context_dirs(sample_csv):
        format_hint = load_submission_format_hint(context_dir / "submission_format.md")
        if format_hint is not None and format_hint.columns:
            return list(format_hint.columns)
        for name in ("overview.md", "data.md", "rules.md", "discussion.md"):
            path = context_dir / name
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            section = extract_submission_section(text) or ""
            if not section.strip():
                continue
            hint = parse_submission_format(section)
            if hint.columns and columns_look_plausible(list(hint.columns)):
                return list(hint.columns)
    return None


def _has_hidden_full_test_context(sample_csv: Path) -> bool:
    for context_dir in _candidate_context_dirs(sample_csv):
        for name in ("submission_format.md", "overview.md", "data.md", "rules.md", "discussion.md"):
            path = context_dir / name
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if _HIDDEN_FULL_TEST_CONTEXT_RE.search(text):
                return True
    return False


def _has_incomplete_data_release_context(sample_csv: Path) -> bool:
    for context_dir in _candidate_context_dirs(sample_csv):
        for name in ("submission_format.md", "overview.md", "data.md", "rules.md", "discussion.md"):
            path = context_dir / name
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if _INCOMPLETE_DATA_RELEASE_CONTEXT_RE.search(text):
                return True
    return False


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
