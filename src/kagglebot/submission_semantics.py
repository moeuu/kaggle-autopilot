from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd

from kagglebot.baseline_tokens import ID_LIKE_COLUMN_NAMES
from kagglebot.exceptions import SubmissionValidationError
from kagglebot.json_utils import load_json_object, write_json_object
from kagglebot.solver.io import read_table
from kagglebot.submission_sample_discovery import find_usable_sample_submissions

_PLACEHOLDER_TEXT_VALUES = frozenset(
    {
        "",
        '"',
        "''",
        "[]",
        "{}",
        "dummy",
        "missing",
        "none",
        "null",
        "placeholder",
        "test",
        "unknown",
    }
)
_FALLBACK_FLAG_KEYS = (
    "dummy_submission",
    "fallback_only",
    "fallback_submission",
    "placeholder_submission",
    "submission_fallback_used",
)
_SELECTED_PIPELINE_KEYS = ("selected_pipeline", "chosen_pipeline", "pipeline")
_EMITTED_PIPELINE_KEYS = (
    "emitted_pipeline",
    "inference_pipeline",
    "runtime_pipeline",
    "submission_pipeline",
)
_SUBMISSION_ROW_COUNT_KEYS = ("emitted_row_count", "submission_row_count", "submission_rows")
_SUBMISSION_FILENAME_KEYS = ("expected_output_file", "submission_filename", "submission_output_file")
_SUBMISSION_SHA_KEYS = ("submission_sha256", "submission_output_sha256")
_PREDICTION_COLUMN_PREFIXES = (
    "answer",
    "category",
    "class",
    "confidence",
    "label",
    "output",
    "pred",
    "prediction",
    "prob",
    "response",
    "risk",
    "score",
    "target",
    "value",
)


def analyze_submission_semantics(
    *,
    submission_path: Path,
    sample_submission_path: Path | None = None,
    data_dir: Path | None = None,
    metrics_path: Path | None = None,
    metrics_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a competition-independent semantic risk report for a tabular submission."""
    report: dict[str, object] = {
        "schema_version": 1,
        "submission_path": str(submission_path.resolve()),
        "submission_sha256": _sha256_file(submission_path),
        "applicable": False,
        "block_submit": False,
        "findings": [],
    }
    try:
        submission = read_table(submission_path)
    except Exception as exc:  # noqa: BLE001 - non-tabular submissions are validated elsewhere
        report["skip_reason"] = f"not_readable_as_tabular:{type(exc).__name__}"
        return report

    report["applicable"] = True
    submission.columns = [str(column) for column in submission.columns]
    prediction_columns = [column for column in submission.columns if not _looks_like_identifier_column(column)]
    report["row_count"] = int(len(submission))
    report["column_count"] = int(len(submission.columns))
    report["prediction_columns"] = prediction_columns
    if not prediction_columns or submission.empty:
        report["skip_reason"] = "no_prediction_cells"
        return report

    normalized = _normalized_values(submission[prediction_columns])
    unique_prediction_rows = int(normalized.drop_duplicates().shape[0])
    unique_by_column = {column: int(normalized[column].nunique(dropna=False)) for column in prediction_columns}
    report["unique_prediction_rows"] = unique_prediction_rows
    report["unique_prediction_values_by_column"] = unique_by_column

    findings: list[dict[str, object]] = []
    if len(submission) >= 3 and unique_prediction_rows == 1:
        findings.append(
            _finding(
                "row_constant_predictions",
                "all runtime submission rows have identical predictions; hidden-test inference is degenerate "
                f"across {len(submission)} rows and {len(prediction_columns)} prediction columns",
            )
        )

    if (
        len(submission) >= 2
        and len(prediction_columns) >= 3
        and unique_prediction_rows > 1
        and _all_prediction_columns_identical(normalized, prediction_columns)
    ):
        findings.append(
            _finding(
                "identical_prediction_heads",
                "all prediction columns are exact copies of one another; multi-output inference collapsed "
                f"across {len(prediction_columns)} prediction columns",
            )
        )

    if len(submission) >= 2 and _all_placeholder_text(submission[prediction_columns]):
        findings.append(
            _finding(
                "placeholder_text_predictions",
                "all prediction cells contain placeholder text rather than model/runtime outputs",
            )
        )

    sample, sample_source = _load_compatible_sample(
        submission=submission,
        sample_submission_path=sample_submission_path,
        data_dir=data_dir,
    )
    if sample is not None:
        sample_normalized = _normalized_values(sample[prediction_columns])
        if normalized.equals(sample_normalized):
            findings.append(
                _finding(
                    "sample_template_unchanged",
                    "submission prediction values are unchanged from the sample submission template; "
                    "no generated prediction evidence is present",
                )
            )
            report["sample_submission_path"] = str(sample_source.resolve()) if sample_source is not None else None

    metrics = (
        metrics_payload
        if isinstance(metrics_payload, dict)
        else load_json_object(metrics_path)
        if metrics_path
        else None
    )
    if metrics is not None:
        report["metrics_path"] = str(metrics_path.resolve()) if metrics_path is not None else None
        findings.extend(
            _metrics_findings(
                metrics=metrics,
                submission_path=submission_path,
                submission_sha256=str(report["submission_sha256"] or ""),
                row_count=len(submission),
            )
        )

    report["findings"] = findings
    report["block_submit"] = bool(findings)
    return report


def validate_autopilot_submission_semantics(
    *,
    submission_path: Path,
    sample_submission_path: Path | None,
    data_dir: Path | None,
    metrics_path: Path | None,
    report_path: Path | None,
) -> dict[str, object]:
    report = analyze_submission_semantics(
        submission_path=submission_path,
        sample_submission_path=sample_submission_path,
        data_dir=data_dir,
        metrics_path=metrics_path,
    )
    if report_path is not None:
        write_json_object(report_path, report, sort_keys=True)
    if report.get("block_submit") is not True:
        return report
    findings = report.get("findings")
    rows = findings if isinstance(findings, list) else []
    detail = "\n".join(f"  - [{row.get('code')}] {row.get('message')}" for row in rows if isinstance(row, dict))
    report_note = f"\n  report: {report_path}" if report_path is not None else ""
    raise SubmissionValidationError(
        "Autopilot semantic submission preflight blocked a likely implementation-error submission:\n"
        f"{detail}{report_note}\n"
        "Regenerate predictions from the selected model on the actual test/runtime inputs before submitting."
    )


def discover_submission_metrics_path(*, submission_path: Path, run_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for directory in (submission_path.parent, submission_path.parent.parent):
        candidates.extend((directory / "metrics.json", directory / "output" / "metrics.json"))
    for parent in submission_path.parents:
        if parent == run_dir.parent:
            break
        if parent.name.startswith("iter-"):
            candidates.extend((parent / "metrics.json", parent / "output" / "metrics.json"))
            break
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.is_file() and load_json_object(candidate) is not None:
            return candidate
    iteration_metrics = sorted(run_dir.glob("iter-*/metrics.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if iteration_metrics:
        return iteration_metrics[0]
    return None


def semantic_finding_messages(
    *,
    submission_path: Path,
    metrics_payload: dict[str, object] | None = None,
) -> list[str]:
    report = analyze_submission_semantics(
        submission_path=submission_path,
        metrics_payload=metrics_payload,
    )
    findings = report.get("findings")
    return (
        [str(row.get("message")) for row in findings if isinstance(row, dict) and str(row.get("message") or "").strip()]
        if isinstance(findings, list)
        else []
    )


def runtime_tabular_fidelity_findings(tabular: dict[str, object]) -> list[dict[str, str]]:
    """Return stable fidelity findings from the pure-stdlib runtime CSV audit."""
    findings: list[dict[str, str]] = []

    def add(code: str, message: str) -> None:
        findings.append({"code": code, "message": message})

    if tabular.get("parse_error"):
        add("tabular_output_parse_failed", "runtime could not parse the selected tabular output")
        return findings
    schema = tabular.get("schema")
    columns = schema.get("columns") if isinstance(schema, dict) else None
    if not isinstance(columns, list) or not columns:
        add("tabular_output_schema_missing", "runtime tabular output has no recorded schema")
    row_count = _runtime_nonnegative_int(tabular.get("row_count"))
    if row_count in {None, 0}:
        add("tabular_output_rows_missing", "runtime tabular output has no prediction rows")
    prediction_columns = tabular.get("prediction_columns")
    if not isinstance(prediction_columns, list) or not prediction_columns:
        add("tabular_prediction_columns_missing", "runtime tabular output has no prediction columns")
    if (_runtime_nonnegative_int(tabular.get("prediction_null_count")) or 0) > 0:
        add("tabular_prediction_nulls", "runtime tabular predictions contain null values")
    if (_runtime_nonnegative_int(tabular.get("prediction_nonfinite_count")) or 0) > 0:
        add("tabular_prediction_nonfinite", "runtime tabular predictions contain non-finite numeric values")

    identifier = tabular.get("identifier")
    if isinstance(identifier, dict):
        if identifier.get("unique") is not True:
            add("tabular_identifier_not_unique", "runtime tabular identifiers are not unique")
        if (_runtime_nonnegative_int(identifier.get("null_count")) or 0) > 0:
            add("tabular_identifier_nulls", "runtime tabular identifiers contain null values")
        order_digests = identifier.get("order_digests")
        if not isinstance(order_digests, dict) or not order_digests:
            add("tabular_identifier_order_digest_missing", "runtime identifier order digest is missing")

    dispersion = tabular.get("numeric_dispersion")
    rows = [item for item in dispersion if isinstance(item, dict)] if isinstance(dispersion, list) else []
    if isinstance(row_count, int) and row_count >= 3 and rows:
        known_unique_counts = [
            _runtime_nonnegative_int(item.get("unique_count"))
            for item in rows
            if item.get("unique_truncated") is not True
        ]
        if (
            known_unique_counts
            and len(known_unique_counts) == len(rows)
            and all(value == 1 for value in known_unique_counts)
        ):
            add("tabular_prediction_dispersion_collapsed", "all runtime prediction rows are constant")
    return findings


def _metrics_findings(
    *,
    metrics: dict[str, object],
    submission_path: Path,
    submission_sha256: str,
    row_count: int,
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for key in _FALLBACK_FLAG_KEYS:
        if _truthy(metrics.get(key)):
            findings.append(
                _finding(
                    "fallback_submission_output",
                    f"metrics report {key}=true; fallback/dummy output cannot be submitted automatically",
                )
            )
            break

    selected_pipeline = _first_text(metrics, _SELECTED_PIPELINE_KEYS)
    emitted_pipeline = _first_text(metrics, _EMITTED_PIPELINE_KEYS)
    if selected_pipeline and emitted_pipeline and selected_pipeline != emitted_pipeline:
        findings.append(
            _finding(
                "selected_emitted_pipeline_mismatch",
                f"selected pipeline {selected_pipeline!r} differs from emitted pipeline {emitted_pipeline!r}",
            )
        )

    expected_rows = _first_int(metrics, _SUBMISSION_ROW_COUNT_KEYS)
    if expected_rows is not None and expected_rows != row_count:
        findings.append(
            _finding(
                "metrics_submission_row_count_mismatch",
                f"metrics report {expected_rows} emitted rows but the submission contains {row_count}",
            )
        )

    expected_filename = _first_text(metrics, _SUBMISSION_FILENAME_KEYS)
    if expected_filename and Path(expected_filename).name != submission_path.name:
        findings.append(
            _finding(
                "metrics_submission_filename_mismatch",
                f"metrics expect output {Path(expected_filename).name!r} "
                f"but selected artifact is {submission_path.name!r}",
            )
        )

    expected_sha = _first_text(metrics, _SUBMISSION_SHA_KEYS).lower()
    if expected_sha and submission_sha256 and expected_sha != submission_sha256.lower():
        findings.append(
            _finding(
                "metrics_submission_hash_mismatch",
                "submission hash differs from the artifact hash recorded by the producing runtime",
            )
        )
    return findings


def _load_compatible_sample(
    *,
    submission: pd.DataFrame,
    sample_submission_path: Path | None,
    data_dir: Path | None,
) -> tuple[pd.DataFrame | None, Path | None]:
    candidates: list[Path] = []
    if sample_submission_path is not None:
        candidates.append(sample_submission_path)
    if data_dir is not None and data_dir.is_dir():
        candidates.extend(find_usable_sample_submissions(data_dir))
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen or not candidate.is_file():
            continue
        seen.add(resolved)
        try:
            sample = read_table(candidate)
        except Exception:  # noqa: BLE001
            continue
        sample.columns = [str(column) for column in sample.columns]
        if list(sample.columns) == list(submission.columns) and len(sample) == len(submission):
            prediction_columns = [column for column in submission.columns if not _looks_like_identifier_column(column)]
            if prediction_columns:
                return sample, candidate
    return None, None


def _normalized_values(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.where(frame.notna(), "<KAGGLEBOT_MISSING>").astype(str)


def _all_prediction_columns_identical(frame: pd.DataFrame, columns: list[str]) -> bool:
    first = frame[columns[0]]
    return all(first.equals(frame[column]) for column in columns[1:])


def _all_placeholder_text(frame: pd.DataFrame) -> bool:
    values: list[str] = []
    for value in frame.to_numpy().ravel().tolist():
        if isinstance(value, (int, float, complex)) and not isinstance(value, bool):
            return False
        values.append(str(value or "").strip().lower())
    return bool(values) and set(values).issubset(_PLACEHOLDER_TEXT_VALUES)


def _looks_like_identifier_column(value: object) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    compact = normalized.replace("_", "")
    if any(normalized == prefix or normalized.startswith(f"{prefix}_") for prefix in _PREDICTION_COLUMN_PREFIXES):
        return False
    normalized_ids = {name.replace("_", "") for name in ID_LIKE_COLUMN_NAMES}
    return normalized in ID_LIKE_COLUMN_NAMES or compact in normalized_ids or compact.endswith("id")


def _first_text(payload: dict[str, object], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return ""


def _first_int(payload: dict[str, object], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        try:
            value = int(payload.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if value >= 0:
            return value
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _runtime_nonnegative_int(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _finding(code: str, message: str) -> dict[str, object]:
    return {"code": code, "severity": "error", "message": message}


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()
