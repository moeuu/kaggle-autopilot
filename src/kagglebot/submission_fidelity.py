from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from kagglebot.baseline_tokens import ID_LIKE_COLUMN_NAMES
from kagglebot.exceptions import KernelFailedError, SubmissionCliError, SubmissionValidationError
from kagglebot.hashing import sha256_file_or_none, sha256_text
from kagglebot.json_utils import append_jsonl_record, load_json_object, load_jsonl_records, write_json_object
from kagglebot.kernel_runtime.submit_runtime_fidelity import (
    EXPECTED_FILE_NAME as EXPECTED_FILE_NAME,
)
from kagglebot.kernel_runtime.submit_runtime_fidelity import (
    RUNTIME_FILE_NAME as RUNTIME_FILE_NAME,
)
from kagglebot.kernel_runtime.submit_runtime_fidelity import (
    package_source_fingerprint,
)
from kagglebot.metric_matching import metrics_equivalent
from kagglebot.solver.io import read_table
from kagglebot.submission_semantics import runtime_tabular_fidelity_findings

REPORT_FILE_NAME = "submission_fidelity_report.json"
REPORT_SCHEMA_VERSION = 2
REPORT_TYPE = "SubmissionFidelityReport"
QUARANTINE_STATE_KEY = "submission_fidelity_quarantine"
QUARANTINE_LEDGER_EVENT = "submission_fidelity_quarantine"

_FIXED_ROWS_IDENTIFIER_MODE = "fixed_rows"
_VARIABLE_ROWS_IDENTIFIER_MODE = "variable_rows_by_base_identifier"
_IDENTIFIER_DIAGNOSTIC_LIMIT = 500

_CODE_OUTPUT_MODES = {"gateway", "inference"}
_TRUSTED_SCORE_SOURCES = frozenset({"cv", "cross_validation", "holdout", "offline", "oof"})
_SCORE_KEYS = ("selected_cv_mean", "offline_value", "primary_score", "score", "value")
_SUBMISSION_SHA_KEYS = ("submission_sha256", "submission_output_sha256", "reported_output_sha256")
_MISSING_ASSET_KEYS = ("missing_dependencies", "missing_kernel_sources", "missing_model_sources")
_FALLBACK_MARKERS = ("dummy", "emergency", "fallback", "placeholder", "sample", "template")
_FALLBACK_FLAG_KEYS = (
    "dummy_submission",
    "fallback_only",
    "fallback_submission",
    "placeholder_submission",
    "prediction_fallback_used",
    "submission_fallback_used",
)
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
_ERROR_NAME_RE = re.compile(
    r"(?:^kernel_error(?:[-_][^.]+)?\.txt$|^stderr\.txt$|^error(?:[-_][^.]+)?\.(?:log|txt)$|"
    r"^.*(?:exception|traceback).*\.(?:log|txt)$)",
    flags=re.IGNORECASE,
)


def build_submit_runtime_env(expected_metrics: Mapping[str, object] | None) -> dict[str, str]:
    """Build the optional runtime selection contract consumed by inference-aware kernels."""
    if not expected_metrics:
        return {}
    env: dict[str, str] = {}
    pipeline = _pipeline_name(expected_metrics)
    if pipeline:
        env["KAGGLEBOT_SELECTED_PIPELINE"] = pipeline
    score = _score(expected_metrics)
    if score is not None:
        env["KAGGLEBOT_SELECTED_OFFLINE_SCORE"] = f"{score:.17g}"
    if expected_metrics.get("reference_path_used") is True:
        env["KAGGLEBOT_REQUIRE_REFERENCE_PATH"] = "1"
    model_source = _text(expected_metrics.get("active_model_source"))
    if model_source:
        env["KAGGLEBOT_SELECTED_MODEL_SOURCE"] = model_source
    return env


def load_expected_submit_metrics_snapshot(paths: Iterable[Path]) -> dict[str, object] | None:
    """Load canonical metrics, enriching them only with a score-matching detailed payload."""
    payloads = [payload for path in paths if (payload := load_json_object(path)) is not None]
    if not payloads:
        return None

    canonical = payloads[0]
    canonical_score = _score(canonical)
    for candidate in payloads[1:]:
        if not _pipeline_name(candidate):
            continue
        candidate_score = _score(candidate)
        if canonical_score is not None and (
            candidate_score is None or not math.isclose(candidate_score, canonical_score, rel_tol=1e-6, abs_tol=1e-6)
        ):
            continue
        merged = dict(canonical)
        merged.update({key: value for key, value in candidate.items() if value is not None})
        return merged
    return dict(canonical)


def validate_file_submission_fidelity(
    *,
    slug: str,
    run_id: str,
    iteration: int,
    source_candidate_path: Path,
    prepared_submission_path: Path,
    code_fingerprint: str | None,
    metrics_path: Path | None,
    semantic_report_path: Path | None,
    sample_submission_path: Path | None,
    report_path: Path,
    expected_contract_path: Path,
    quarantine_state: Mapping[str, object] | None = None,
    score_value: float | None = None,
    score_direction: str | None = None,
) -> dict[str, object]:
    """Attest exact locally prepared bytes without claiming remote execution.

    Structural and format-specific validation remains owned by the existing
    submission validators. This report binds their prepared artifact to the
    selected candidate, evaluation provenance, bounded semantic evidence, and
    controller quarantine state.
    """
    metrics = load_json_object(metrics_path) if metrics_path is not None else None
    semantic = load_json_object(semantic_report_path) if semantic_report_path is not None else None
    expected = build_file_submission_fidelity_contract(
        slug=slug,
        run_id=run_id,
        iteration=iteration,
        source_candidate_path=source_candidate_path,
        prepared_submission_path=prepared_submission_path,
        code_fingerprint=code_fingerprint,
        metrics=metrics,
        metrics_path=metrics_path,
        sample_submission_path=sample_submission_path,
        score_value=score_value,
        score_direction=score_direction,
        strong_contract=_quarantine_active(quarantine_state),
    )
    expected_contract_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_object(expected_contract_path, expected, ensure_ascii=False, sort_keys=True)
    report = build_file_submission_fidelity_report(
        expected_contract=expected,
        expected_contract_path=expected_contract_path,
        source_candidate_path=source_candidate_path,
        prepared_submission_path=prepared_submission_path,
        metrics=metrics,
        metrics_path=metrics_path,
        semantic_report=semantic,
        semantic_report_path=semantic_report_path,
        sample_submission_path=sample_submission_path,
    )
    report = enforce_submission_fidelity_quarantine(report=report, quarantine_state=quarantine_state)
    report["report_path"] = str(report_path.resolve())
    report = finalize_submission_fidelity_report(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_object(report_path, report, ensure_ascii=False, sort_keys=True)
    if report.get("verdict") != "pass":
        _raise_normalized_fidelity_error(report, report_path=report_path)
    return report


def build_file_submission_fidelity_contract(
    *,
    slug: str,
    run_id: str,
    iteration: int,
    source_candidate_path: Path,
    prepared_submission_path: Path,
    code_fingerprint: str | None,
    metrics: Mapping[str, object] | None,
    metrics_path: Path | None,
    sample_submission_path: Path | None,
    score_value: float | None,
    score_direction: str | None,
    strong_contract: bool,
) -> dict[str, object]:
    metrics_payload = dict(metrics or {})
    provenance = _metric_provenance(
        metrics_payload,
        score_value=score_value,
        score_direction=score_direction,
    )
    local_evidence = _local_tabular_evidence(prepared_submission_path)
    sample_evidence = _local_tabular_evidence(sample_submission_path)
    identifier_cardinality = build_identifier_cardinality_contract(
        sample_submission_path=sample_submission_path,
        submission_path=prepared_submission_path,
        metrics=metrics_payload,
        metrics_path=metrics_path,
    )
    fallback_flags = {key: _truthy(metrics_payload.get(key)) for key in _FALLBACK_FLAG_KEYS if key in metrics_payload}
    source_counts = _bounded_source_counts(
        metrics_payload.get("test_prediction_distribution") or metrics_payload.get("prediction_source_distribution")
    )
    pipeline = _pipeline_name(metrics_payload)
    complete = bool(
        pipeline
        and provenance["metric"]
        and provenance["direction"] in {"maximize", "minimize"}
        and provenance["score_source"]
        and provenance["score"] is not None
        and sha256_file_or_none(prepared_submission_path)
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "contract": "kagglebot_submission_fidelity",
        "competition": slug,
        "run_id": run_id,
        "iteration": iteration,
        "artifact_mode": "file",
        "attestation_scope": "local_prepared_artifact",
        "strong_contract": bool(strong_contract or complete),
        "quarantine_required": bool(strong_contract),
        "code_fingerprint": _text(code_fingerprint) or None,
        "package_fingerprint": _text(code_fingerprint) or None,
        "selection": {
            "pipeline": pipeline or None,
            **provenance,
        },
        "selected_assets": _selected_assets(metrics_payload),
        "selected_local_candidate": {
            "path": str(source_candidate_path.resolve()),
            "name": source_candidate_path.name,
            "sha256": sha256_file_or_none(source_candidate_path),
            "size": _path_size(source_candidate_path),
        },
        "prepared_artifact": {
            "path": str(prepared_submission_path.resolve()),
            "filename": prepared_submission_path.name,
            "sha256": sha256_file_or_none(prepared_submission_path),
            "size": _path_size(prepared_submission_path),
            "metrics_reported_sha256": _first_text(metrics_payload, _SUBMISSION_SHA_KEYS) or None,
        },
        "output": {
            "filename": prepared_submission_path.name,
            "schema": local_evidence.get("schema") if local_evidence else None,
            "row_count": local_evidence.get("row_count") if local_evidence else None,
            "ordered_id_digest": _nested_value(local_evidence, "identifier", "composite_order_sha256"),
            "prediction_statistics": local_evidence.get("prediction_statistics") if local_evidence else None,
        },
        "sample_contract": sample_evidence,
        "identifier_cardinality": identifier_cardinality,
        "fallback": {
            "declared_flags": fallback_flags,
            "prediction_source_counts": source_counts,
        },
    }


def build_file_submission_fidelity_report(
    *,
    expected_contract: Mapping[str, object],
    expected_contract_path: Path,
    source_candidate_path: Path,
    prepared_submission_path: Path,
    metrics: Mapping[str, object] | None,
    metrics_path: Path | None,
    semantic_report: Mapping[str, object] | None,
    semantic_report_path: Path | None,
    sample_submission_path: Path | None,
) -> dict[str, object]:
    reasons: list[dict[str, str]] = []
    warnings: list[str] = []

    def add(code: str, message: str) -> None:
        if not any(item["code"] == code for item in reasons):
            reasons.append({"code": code, "message": message})

    expected_source = _nested_text(expected_contract, "selected_local_candidate", "sha256")
    expected_output = _nested_text(expected_contract, "prepared_artifact", "sha256")
    actual_source = sha256_file_or_none(source_candidate_path)
    actual_output = sha256_file_or_none(prepared_submission_path)
    metrics_output = _nested_text(expected_contract, "prepared_artifact", "metrics_reported_sha256")
    if not actual_source or actual_source != expected_source:
        add("selected_local_candidate_hash_mismatch", "selected source candidate changed after identity freeze")
    if not actual_output or actual_output != expected_output:
        add("prepared_artifact_hash_mismatch", "prepared submission bytes changed after identity freeze")
    evaluated_hashes = {value.lower() for value in (actual_source, actual_output) if value}
    if metrics_output and metrics_output.lower() not in evaluated_hashes:
        add("metrics_submission_hash_mismatch", "evaluation metrics bind to a different source/prepared artifact")
    elif expected_contract.get("quarantine_required") is True and not metrics_output:
        add(
            "metric_artifact_binding_missing",
            "anomaly quarantine requires metrics to report the evaluated submission SHA-256",
        )
    if prepared_submission_path.name != _nested_text(expected_contract, "output", "filename"):
        add("selected_output_filename_mismatch", "prepared submission filename changed after identity freeze")

    actual_tabular = _local_tabular_evidence(prepared_submission_path)
    sample_tabular = _local_tabular_evidence(sample_submission_path)
    actual_identifier_cardinality = build_identifier_cardinality_contract(
        sample_submission_path=sample_submission_path,
        submission_path=prepared_submission_path,
        metrics=metrics,
        metrics_path=metrics_path,
    )
    if actual_tabular is not None:
        for finding in runtime_tabular_fidelity_findings(actual_tabular):
            add(str(finding["code"]), str(finding["message"]))
    if actual_tabular is not None and sample_tabular is not None:
        _compare_local_tabular_contract(
            sample_tabular,
            actual_tabular,
            expected_identifier_cardinality=_nested_mapping(expected_contract, "identifier_cardinality"),
            actual_identifier_cardinality=actual_identifier_cardinality,
            add=add,
            warn=warnings.append,
        )

    semantic_findings = semantic_report.get("findings") if semantic_report else None
    if isinstance(semantic_findings, list):
        for item in semantic_findings:
            if not isinstance(item, Mapping):
                continue
            code = _text(item.get("code")) or "semantic_preflight_failed"
            add(code, _text(item.get("message")) or "semantic submission preflight failed")
    semantic_sha = _text(semantic_report.get("submission_sha256")) if semantic_report else ""
    if semantic_sha and actual_output and semantic_sha != actual_output:
        add("semantic_preflight_artifact_hash_mismatch", "semantic preflight inspected different prepared bytes")

    fallback = _nested_mapping(expected_contract, "fallback") or {}
    flags = fallback.get("declared_flags")
    if isinstance(flags, Mapping) and any(value is True for value in flags.values()):
        add("prediction_fallback_used", "selected metrics declare fallback/dummy prediction generation")
    source_counts = fallback.get("prediction_source_counts")
    if _fallback_sources_only(source_counts):
        add("prediction_sources_fallback_only", "all declared prediction sources are fallback/dummy paths")

    provenance = dict(_nested_mapping(expected_contract, "selection") or {})
    strong_contract = expected_contract.get("strong_contract") is True
    if provenance.get("trusted") is not True:
        if strong_contract:
            add("metric_provenance_untrusted", "strong fidelity contract requires trusted score provenance")
        else:
            warnings.append("legacy_unknown")
    missing_selection = [
        key for key in ("pipeline", "metric", "direction", "score_source", "score") if provenance.get(key) in (None, "")
    ]
    if missing_selection:
        if strong_contract:
            add(
                "selected_candidate_contract_incomplete",
                "selected candidate contract is missing: " + ", ".join(missing_selection),
            )
        elif "legacy_unknown" not in warnings:
            warnings.append("legacy_unknown")
    if actual_tabular is None and "legacy_unknown" not in warnings:
        warnings.append("legacy_unknown")

    reason_codes = sorted(item["code"] for item in reasons)
    contract_sha = sha256_file_or_none(expected_contract_path)
    package_fingerprint = _text(expected_contract.get("package_fingerprint"))
    attempt_fingerprint = sha256_text("\0".join((package_fingerprint, str(actual_output or ""))))
    supporting = [
        str(path.resolve())
        for path in (
            expected_contract_path,
            source_candidate_path,
            prepared_submission_path,
            metrics_path,
            semantic_report_path,
            sample_submission_path,
        )
        if path is not None and path.exists()
    ]
    ledger_path = _nested_text(actual_identifier_cardinality, "row_count_evidence", "prediction_ledger", "path")
    if ledger_path and Path(ledger_path).is_file():
        supporting.append(str(Path(ledger_path).resolve()))
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": REPORT_TYPE,
        "verdict": "fail" if reasons else "pass",
        "reason_codes": reason_codes,
        "warning_codes": sorted(set(warnings)),
        "reasons": sorted(reasons, key=lambda item: item["code"]),
        "competition": expected_contract.get("competition"),
        "run_id": expected_contract.get("run_id"),
        "iteration": expected_contract.get("iteration"),
        "artifact_mode": "file",
        "attestation_scope": "local_prepared_artifact",
        "remote_runtime_attested": False,
        "package_fingerprint": package_fingerprint or None,
        "code_fingerprint": _text(expected_contract.get("code_fingerprint")) or None,
        "expected_contract_sha256": contract_sha,
        "selected_candidate": dict(_nested_mapping(expected_contract, "selected_local_candidate") or {}),
        "selected_assets": dict(_nested_mapping(expected_contract, "selected_assets") or {}),
        "selected_output": {
            "path": str(prepared_submission_path.resolve()),
            "filename": prepared_submission_path.name,
            "size": _path_size(prepared_submission_path),
            "sha256": actual_output,
        },
        "metric_provenance": provenance,
        "fallback": dict(fallback),
        "prediction_evidence": actual_tabular,
        "identifier_cardinality": actual_identifier_cardinality,
        "attempt_fingerprint": attempt_fingerprint,
        "comparison": {
            "expected": {
                "source_candidate_sha256": expected_source or None,
                "prepared_artifact_sha256": expected_output or None,
                "filename": _nested_text(expected_contract, "output", "filename") or None,
                "sample_contract": sample_tabular,
            },
            "actual": {
                "source_candidate_sha256": actual_source,
                "prepared_artifact_sha256": actual_output,
                "filename": prepared_submission_path.name,
                "tabular_contract": actual_tabular,
                "metrics_sha256": sha256_file_or_none(metrics_path),
                "semantic_preflight_sha256": sha256_file_or_none(semantic_report_path),
            },
        },
        "supporting_artifact_paths": sorted(set(supporting)),
    }


def finalize_submission_fidelity_report(report: Mapping[str, object]) -> dict[str, object]:
    """Return a report with a stable fingerprint over its bounded evidence."""
    payload = dict(report)
    payload["schema_version"] = REPORT_SCHEMA_VERSION
    payload["report_type"] = REPORT_TYPE
    payload.pop("report_fingerprint", None)
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    payload["report_fingerprint"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return payload


def enforce_submission_fidelity_quarantine(
    *,
    report: Mapping[str, object],
    quarantine_state: Mapping[str, object] | None,
) -> dict[str, object]:
    """Fail closed unless an active quarantine receives one changed, trusted repair."""
    payload = dict(report)
    if not _quarantine_active(quarantine_state):
        return payload

    reasons = [dict(item) for item in payload.get("reasons", []) if isinstance(item, Mapping)]

    def add(code: str, message: str) -> None:
        if not any(_text(item.get("code")) == code for item in reasons):
            reasons.append({"code": code, "message": message})

    anomaly = _nested_mapping(quarantine_state, "anomaly") or {}
    current_output = _nested_text(payload, "selected_output", "sha256")
    current_package = _text(payload.get("package_fingerprint") or payload.get("code_fingerprint"))
    current_attempt = _text(payload.get("attempt_fingerprint"))
    anomalous_output = _text(anomaly.get("output_sha256"))
    anomalous_package = _text(anomaly.get("package_fingerprint") or anomaly.get("code_fingerprint"))
    failed_attempts = set(_string_values(quarantine_state.get("failed_attempt_fingerprints")))
    pending_attempt = _nested_mapping(quarantine_state, "pending_repair_attempt")

    if not anomalous_output or not anomalous_package:
        add(
            "quarantine_anomaly_identity_legacy_unknown",
            "active quarantine lacks the prior output/package hashes needed to prove a changed repair",
        )
    if not current_output or current_output == anomalous_output:
        add(
            "quarantine_output_fingerprint_unchanged",
            "anomaly quarantine requires prepared/downloaded output bytes to change",
        )
    if not current_package or current_package == anomalous_package:
        add(
            "quarantine_package_fingerprint_unchanged",
            "anomaly quarantine requires the code/package fingerprint to change",
        )
    if not current_attempt or current_attempt in failed_attempts:
        add(
            "quarantine_failed_attempt_unchanged",
            "this package/output attempt fingerprint already produced an anomalous outcome",
        )
    if pending_attempt is not None:
        add(
            "quarantine_repair_permit_already_used",
            "one repaired-candidate permit is already awaiting a leaderboard outcome",
        )
    provenance = _report_metric_provenance(payload)
    if provenance.get("trusted") is not True:
        add("metric_provenance_untrusted", "anomaly quarantine requires trusted score provenance")
    if payload.get("verdict") != "pass":
        add("quarantine_fidelity_report_failed", "anomaly quarantine requires a passing fidelity report")

    payload["metric_provenance"] = provenance
    payload["reasons"] = sorted(reasons, key=lambda item: _text(item.get("code")))
    payload["reason_codes"] = sorted({_text(item.get("code")) for item in reasons if _text(item.get("code"))})
    payload["verdict"] = "fail" if reasons else "pass"
    payload["quarantine"] = {
        "status": "active",
        "repair_permit": "granted" if not reasons else "blocked",
        "anomalous_output_sha256": anomalous_output or None,
        "anomalous_package_fingerprint": anomalous_package or None,
    }
    return payload


def reserve_quarantine_repair_attempt(
    *,
    report: Mapping[str, object],
    quarantine_state: Mapping[str, object] | None,
    save_run_state: Callable[[dict[str, object]], object],
    submission_ledger_path: Path | None = None,
    slug: str | None = None,
    run_id: str | None = None,
) -> bool:
    """Persist the single repaired-candidate permit immediately before submit."""
    if not _quarantine_active(quarantine_state):
        return False
    if report.get("verdict") != "pass":
        raise ValueError("cannot reserve a quarantine repair permit for a failed fidelity report")
    attempt_fingerprint = _text(report.get("attempt_fingerprint"))
    if not attempt_fingerprint:
        raise ValueError("cannot reserve a quarantine repair permit without an attempt fingerprint")
    if _nested_mapping(quarantine_state, "pending_repair_attempt") is not None:
        raise ValueError("a quarantine repair permit is already pending")
    state = dict(quarantine_state)
    state["pending_repair_attempt"] = {
        "reserved_at": datetime.now(UTC).isoformat(),
        "attempt_fingerprint": attempt_fingerprint,
        "report_fingerprint": finalize_submission_fidelity_report(report).get("report_fingerprint"),
        "output_sha256": _nested_value(report, "selected_output", "sha256"),
        "package_fingerprint": report.get("package_fingerprint") or report.get("code_fingerprint"),
    }
    state["repair_permits_granted"] = int(state.get("repair_permits_granted") or 0) + 1
    save_run_state({QUARANTINE_STATE_KEY: state})
    if submission_ledger_path is not None and slug and run_id:
        _append_quarantine_ledger_event(
            submission_ledger_path,
            slug=slug,
            run_id=run_id,
            action="repair_reserved",
            state=state,
            fidelity=fidelity_report_summary(report),
        )
    return True


def load_active_submission_fidelity_quarantine(
    *,
    run_state: Mapping[str, object],
    submission_ledger_path: Path,
    slug: str,
) -> dict[str, object] | None:
    """Resolve competition-wide ledger state, with run state as a legacy fallback."""
    current = _nested_mapping(run_state, QUARANTINE_STATE_KEY)
    records = [
        record
        for record in load_jsonl_records(submission_ledger_path)
        if record.get("event") == QUARANTINE_LEDGER_EVENT and _text(record.get("slug")) == slug
    ]
    if not records:
        return dict(current) if _quarantine_active(current) else None
    latest = records[-1]
    if latest.get("status") != "active" or latest.get("action") == "resolved":
        return None
    anomaly = {
        "output_sha256": latest.get("output_sha256"),
        "package_fingerprint": latest.get("package_fingerprint"),
        "report_fingerprint": latest.get("anomaly_report_fingerprint") or latest.get("report_fingerprint"),
        "attempt_fingerprint": latest.get("anomaly_attempt_fingerprint") or latest.get("attempt_fingerprint"),
        "reason_codes": list(latest.get("reason_codes") or ["legacy_unknown"]),
    }
    state: dict[str, object] = {
        "schema_version": 1,
        "status": "active",
        "activated_at": latest.get("activated_at") or latest.get("ts"),
        "updated_at": latest.get("ts"),
        "anomaly": anomaly,
        "failed_attempt_fingerprints": list(latest.get("failed_attempt_fingerprints") or []),
        "last_fidelity_report_path": latest.get("report_path"),
    }
    pending = latest.get("pending_repair_attempt")
    if isinstance(pending, Mapping):
        state["pending_repair_attempt"] = dict(pending)
    return state


def fidelity_report_summary(
    report: Mapping[str, object],
    *,
    report_path: Path | None = None,
) -> dict[str, object]:
    payload = finalize_submission_fidelity_report(report)
    expected = {
        "contract_sha256": payload.get("expected_contract_sha256"),
        "package_fingerprint": payload.get("package_fingerprint"),
    }
    actual = {
        "output_sha256": _nested_value(payload, "selected_output", "sha256"),
        "runtime_attestation_sha256": payload.get("runtime_attestation_sha256"),
        "metrics_sha256": payload.get("metrics_sha256")
        or _nested_value(payload, "comparison", "actual", "metrics_sha256"),
    }
    return {
        "schema_version": payload.get("schema_version"),
        "report_path": (
            str(report_path.resolve()) if report_path is not None else _text(payload.get("report_path")) or None
        ),
        "verdict": payload.get("verdict"),
        "reason_codes": list(payload.get("reason_codes") or []),
        "warning_codes": list(payload.get("warning_codes") or []),
        "report_fingerprint": payload.get("report_fingerprint"),
        "attempt_fingerprint": payload.get("attempt_fingerprint"),
        "artifact_mode": payload.get("artifact_mode"),
        "attestation_scope": payload.get("attestation_scope"),
        "remote_runtime_attested": payload.get("remote_runtime_attested"),
        "package_fingerprint": payload.get("package_fingerprint"),
        "code_fingerprint": payload.get("code_fingerprint"),
        "metric_provenance": _report_metric_provenance(payload),
        "expected_hashes": expected,
        "actual_hashes": actual,
        "supporting_artifact_paths": list(payload.get("supporting_artifact_paths") or [])[:32],
    }


def find_latest_submission_fidelity_report(
    run_dir: Path,
    *,
    output_sha256: str | None = None,
) -> tuple[Path, dict[str, object]] | None:
    candidates: list[Path] = []
    for pattern in ("submission_fidelity_report*.json", "**/submission_fidelity_report*.json"):
        candidates.extend(run_dir.glob(pattern))
    unique = {path.resolve(): path for path in candidates if path.is_file()}

    def modified(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    normalized_output = _text(output_sha256)
    for path in sorted(unique.values(), key=modified, reverse=True):
        report = load_json_object(path)
        if report is None:
            continue
        if normalized_output and _nested_text(report, "selected_output", "sha256") != normalized_output:
            continue
        return path, finalize_submission_fidelity_report(report)
    return None


def persist_leaderboard_outcome_quarantine(
    *,
    slug: str,
    run_id: str,
    run_state: Mapping[str, object],
    latest_submit_attempt: Mapping[str, object],
    anomaly: Mapping[str, object] | None,
    submission_ledger_path: Path,
    save_run_state: Callable[[dict[str, object]], object],
) -> str | None:
    """Persist anomaly activation/retention or resolve after a good leaderboard outcome."""
    prior = _nested_mapping(run_state, QUARANTINE_STATE_KEY)
    fidelity = latest_submit_attempt.get("submission_fidelity")
    fidelity = fidelity if isinstance(fidelity, Mapping) else {}
    attempt_fingerprint = _text(fidelity.get("attempt_fingerprint"))
    now = datetime.now(UTC).isoformat()

    if anomaly is None:
        if not _quarantine_active(prior):
            return None
        failed = set(_string_values(prior.get("failed_attempt_fingerprints"))) if prior else set()
        if (
            fidelity.get("verdict") != "pass"
            or not attempt_fingerprint
            or attempt_fingerprint in failed
            or _nested_value(fidelity, "metric_provenance", "trusted") is not True
        ):
            return None
        state = dict(prior or {})
        state.pop("pending_repair_attempt", None)
        state.update(
            {
                "status": "resolved",
                "resolved_at": now,
                "resolved_attempt_fingerprint": attempt_fingerprint,
                "resolved_report_fingerprint": fidelity.get("report_fingerprint"),
            }
        )
        save_run_state({QUARANTINE_STATE_KEY: state})
        _append_quarantine_ledger_event(
            submission_ledger_path,
            slug=slug,
            run_id=run_id,
            action="resolved",
            state=state,
            fidelity=fidelity,
        )
        return "resolved"

    failed = list(_string_values(prior.get("failed_attempt_fingerprints"))) if prior else []
    if attempt_fingerprint and attempt_fingerprint not in failed:
        failed.append(attempt_fingerprint)
    reason_codes = ["leaderboard_implementation_anomaly"]
    reason_codes.extend(_string_values(anomaly.get("signals")))
    fidelity_reason_codes = _string_values(fidelity.get("reason_codes"))
    reason_codes.extend(fidelity_reason_codes or (["legacy_unknown"] if not fidelity else []))
    anomaly_record = {
        "detected_at": now,
        "output_sha256": _nested_value(fidelity, "actual_hashes", "output_sha256")
        or latest_submit_attempt.get("sub_sha256"),
        "package_fingerprint": fidelity.get("package_fingerprint")
        or fidelity.get("code_fingerprint")
        or latest_submit_attempt.get("code_fingerprint")
        or run_state.get("last_submit_code_fingerprint"),
        "report_fingerprint": fidelity.get("report_fingerprint"),
        "attempt_fingerprint": attempt_fingerprint or None,
        "reason_codes": sorted(set(reason_codes)),
        "signals": list(anomaly.get("signals") or []),
    }
    state = dict(prior or {})
    state.pop("pending_repair_attempt", None)
    state.update(
        {
            "schema_version": 1,
            "status": "active",
            "activated_at": state.get("activated_at") or now,
            "updated_at": now,
            "anomaly": anomaly_record,
            "failed_attempt_fingerprints": failed[-64:],
            "last_fidelity_report_path": fidelity.get("report_path"),
        }
    )
    save_run_state({QUARANTINE_STATE_KEY: state})
    action = "retained" if _quarantine_active(prior) else "activated"
    _append_quarantine_ledger_event(
        submission_ledger_path,
        slug=slug,
        run_id=run_id,
        action=action,
        state=state,
        fidelity=fidelity,
    )
    return action


def _append_quarantine_ledger_event(
    path: Path,
    *,
    slug: str,
    run_id: str,
    action: str,
    state: Mapping[str, object],
    fidelity: Mapping[str, object],
) -> None:
    anomaly = _nested_mapping(state, "anomaly") or {}
    append_jsonl_record(
        path,
        {
            "ts": datetime.now(UTC).isoformat(),
            "event": QUARANTINE_LEDGER_EVENT,
            "action": action,
            "slug": slug,
            "run_id": run_id,
            "status": state.get("status"),
            "output_sha256": anomaly.get("output_sha256"),
            "package_fingerprint": anomaly.get("package_fingerprint"),
            "report_fingerprint": fidelity.get("report_fingerprint") or anomaly.get("report_fingerprint"),
            "attempt_fingerprint": fidelity.get("attempt_fingerprint") or anomaly.get("attempt_fingerprint"),
            "reason_codes": list(anomaly.get("reason_codes") or []),
            "activated_at": state.get("activated_at"),
            "anomaly_report_fingerprint": anomaly.get("report_fingerprint"),
            "anomaly_attempt_fingerprint": anomaly.get("attempt_fingerprint"),
            "failed_attempt_fingerprints": list(state.get("failed_attempt_fingerprints") or [])[-64:],
            "pending_repair_attempt": state.get("pending_repair_attempt"),
            "report_path": fidelity.get("report_path") or state.get("last_fidelity_report_path"),
            "supporting_artifact_paths": list(fidelity.get("supporting_artifact_paths") or [])[:32],
        },
    )


def _local_tabular_evidence(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.is_file():
        return None
    try:
        frame = read_table(path)
    except Exception:  # noqa: BLE001 - format validators own non-tabular artifacts
        return None
    frame.columns = [str(column)[:500] for column in frame.columns]
    columns = list(frame.columns)
    id_columns = [column for column in columns if _looks_like_id_column(column)]
    prediction_columns = [column for column in columns if column not in id_columns]
    identifier = None
    if id_columns:
        order_digests = {column: _series_order_digest(frame[column]) for column in id_columns[:32]}
        composite = hashlib.sha256()
        for values in frame[id_columns[:32]].itertuples(index=False, name=None):
            composite.update("\x1f".join(_normalized_cell(value) for value in values).encode("utf-8"))
            composite.update(b"\0")
        identifier = {
            "columns": id_columns[:32],
            "unique": not frame[id_columns].duplicated().any(),
            "null_count": int(frame[id_columns].isna().sum().sum()),
            "order_digests": order_digests,
            "composite_order_sha256": composite.hexdigest(),
        }
    dispersion: list[dict[str, object]] = []
    prediction_null_count = 0
    prediction_nonfinite_count = 0
    for column in prediction_columns[:100]:
        series = frame[column]
        prediction_null_count += int(series.isna().sum())
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_values = [float(value) for value in numeric.dropna().tolist()]
        finite_values = [value for value in numeric_values if math.isfinite(value)]
        prediction_nonfinite_count += len(numeric_values) - len(finite_values)
        dispersion.append(
            {
                "column": column,
                "null_count": int(series.isna().sum()),
                "nonfinite_count": len(numeric_values) - len(finite_values),
                "numeric_count": len(finite_values),
                "minimum": min(finite_values) if finite_values else None,
                "maximum": max(finite_values) if finite_values else None,
                "mean": sum(finite_values) / len(finite_values) if finite_values else None,
                "stddev": float(pd.Series(finite_values).std(ddof=0)) if finite_values else None,
                "unique_count": int(series.nunique(dropna=False)),
                "unique_truncated": False,
                "order_sha256": _series_order_digest(series),
            }
        )
    return {
        "applicable": True,
        "schema": {"columns": columns[:500], "column_count": len(columns)},
        "row_count": int(len(frame)),
        "null_count": int(frame.isna().sum().sum()),
        "nonfinite_count": prediction_nonfinite_count,
        "identifier": identifier,
        "prediction_columns": prediction_columns[:100],
        "prediction_null_count": prediction_null_count,
        "prediction_nonfinite_count": prediction_nonfinite_count,
        "numeric_dispersion": dispersion,
        "prediction_statistics": dispersion,
    }


def build_identifier_cardinality_contract(
    *,
    sample_submission_path: Path | None,
    submission_path: Path | None,
    metrics: Mapping[str, object] | None,
    metrics_path: Path | None,
) -> dict[str, object]:
    """Infer a conservative row-per-instance identifier contract.

    A variable-row submission is accepted only when its schema matches the
    sample, sample predictions are placeholders, it has one identifier column,
    every actual identifier maps uniquely to a sample base identifier, actual
    identifiers are unique, and at least one row expands a base as
    ``<base>_<nonempty suffix>``. The longest base match handles sample IDs that
    themselves contain underscores.
    """
    result: dict[str, object] = {
        "mode": _FIXED_ROWS_IDENTIFIER_MODE,
        "row_cardinality": "fixed",
        "identifier_relation": "exact",
        "candidate": False,
        "eligible": False,
    }
    if (
        sample_submission_path is None
        or submission_path is None
        or not sample_submission_path.is_file()
        or not submission_path.is_file()
    ):
        return result
    try:
        sample = read_table(sample_submission_path)
        actual = read_table(submission_path)
    except Exception:  # noqa: BLE001 - the format validators report unreadable tables
        return result

    sample.columns = [str(column)[:500] for column in sample.columns]
    actual.columns = [str(column)[:500] for column in actual.columns]
    sample_id_columns = [column for column in sample.columns if _looks_like_id_column(column)]
    actual_id_columns = [column for column in actual.columns if _looks_like_id_column(column)]
    schema_matches_sample = list(actual.columns) == list(sample.columns)
    prediction_columns = [column for column in sample.columns if column not in sample_id_columns]
    sample_predictions_are_placeholders = bool(prediction_columns) and all(
        _is_placeholder_prediction(value) for column in prediction_columns for value in sample[column].tolist()
    )
    actual_prediction_empty_count = sum(
        _is_empty_cell(value)
        for column in prediction_columns
        if column in actual.columns
        for value in actual[column].tolist()
    )
    result.update(
        {
            "sample_identifier_columns": sample_id_columns[:32],
            "actual_identifier_columns": actual_id_columns[:32],
            "actual_instance_row_count": int(len(actual)),
            "schema_matches_sample": schema_matches_sample,
            "sample_prediction_columns": prediction_columns[:100],
            "sample_predictions_are_placeholders": sample_predictions_are_placeholders,
            "actual_prediction_empty_count": actual_prediction_empty_count,
        }
    )
    if len(sample_id_columns) != 1 or actual_id_columns != sample_id_columns:
        return result

    identifier_column = sample_id_columns[0]
    expected_values = [_normalized_cell(value) for value in sample[identifier_column].tolist()]
    actual_values = [_normalized_cell(value) for value in actual[identifier_column].tolist()]
    base_occurrences: dict[str, int] = {}
    ordered_bases: list[str] = []
    for base in expected_values:
        if base not in base_occurrences:
            ordered_bases.append(base)
            base_occurrences[base] = 0
        base_occurrences[base] += 1
    rows_per_base = dict.fromkeys(ordered_bases, 0)
    unknown_identifiers: list[str] = []
    ambiguous_identifiers: list[str] = []
    malformed_identifiers: list[str] = []
    suffixed_row_count = 0
    exact_row_count = 0
    sorted_bases = sorted(base_occurrences, key=len, reverse=True)
    for identifier in actual_values:
        base = identifier if identifier in base_occurrences else ""
        if not base:
            base = next(
                (
                    candidate_base
                    for candidate_base in sorted_bases
                    if identifier.startswith(f"{candidate_base}_") and len(identifier) > len(candidate_base) + 1
                ),
                "",
            )
        if not base:
            if any(identifier == f"{candidate_base}_" for candidate_base in sorted_bases):
                malformed_identifiers.append(identifier)
                continue
            unknown_identifiers.append(identifier)
            continue
        if base_occurrences[base] != 1:
            ambiguous_identifiers.append(identifier)
            continue
        rows_per_base[base] += 1
        if identifier == base:
            exact_row_count += 1
        else:
            suffixed_row_count += 1

    actual_unique = len(set(actual_values)) == len(actual_values)
    expected_unique = all(count == 1 for count in base_occurrences.values())
    candidate = suffixed_row_count > 0 or bool(ambiguous_identifiers) or bool(malformed_identifiers)
    eligible = bool(
        candidate
        and schema_matches_sample
        and sample_predictions_are_placeholders
        and expected_unique
        and actual_unique
        and actual_prediction_empty_count == 0
        and not unknown_identifiers
        and not ambiguous_identifiers
        and not malformed_identifiers
    )
    missing_bases = [base for base in ordered_bases if rows_per_base[base] == 0]
    bounded_rows_per_base = ordered_bases[:_IDENTIFIER_DIAGNOSTIC_LIMIT]
    row_count_evidence = _variable_row_count_evidence(
        actual_row_count=len(actual),
        rows_per_base=rows_per_base,
        metrics=metrics,
        metrics_path=metrics_path,
        submission_path=submission_path,
    )
    result.update(
        {
            "mode": _VARIABLE_ROWS_IDENTIFIER_MODE if eligible else _FIXED_ROWS_IDENTIFIER_MODE,
            "row_cardinality": "variable_per_entity" if eligible else "fixed",
            "identifier_relation": "sample_id_plus_suffix" if eligible else "exact",
            "base_identifier_source": "sample_submission",
            "base_identifier_coverage": "subset" if eligible else "exact",
            "suffix_delimiter": "_",
            "candidate": candidate,
            "eligible": eligible,
            "identifier_column": identifier_column,
            "expected_image_base_count": len(ordered_bases),
            "expected_identifier_unique": expected_unique,
            "actual_identifier_unique": actual_unique,
            "exact_identifier_row_count": exact_row_count,
            "suffixed_identifier_row_count": suffixed_row_count,
            "unknown_base_count": len(unknown_identifiers),
            "unknown_base_examples": [value[:500] for value in unknown_identifiers[:20]],
            "ambiguous_base_count": len(ambiguous_identifiers),
            "ambiguous_base_examples": [value[:500] for value in ambiguous_identifiers[:20]],
            "malformed_suffix_count": len(malformed_identifiers),
            "malformed_suffix_examples": [value[:500] for value in malformed_identifiers[:20]],
            "missing_base_count": len(missing_bases),
            "missing_bases": [value[:500] for value in missing_bases[:_IDENTIFIER_DIAGNOSTIC_LIMIT]],
            "missing_bases_truncated": len(missing_bases) > _IDENTIFIER_DIAGNOSTIC_LIMIT,
            "rows_per_base": {base[:500]: rows_per_base[base] for base in bounded_rows_per_base},
            "rows_per_base_truncated": len(ordered_bases) > _IDENTIFIER_DIAGNOSTIC_LIMIT,
            "row_count_evidence": row_count_evidence,
        }
    )
    return result


def _is_placeholder_prediction(value: object) -> bool:
    if _is_empty_cell(value):
        return True
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric == 0.0


def _is_empty_cell(value: object) -> bool:
    if value is None:
        return True
    if not isinstance(value, (list, tuple, dict)) and bool(pd.isna(value)):
        return True
    return isinstance(value, str) and not value.strip()


def _variable_row_count_evidence(
    *,
    actual_row_count: int,
    rows_per_base: Mapping[str, int],
    metrics: Mapping[str, object] | None,
    metrics_path: Path | None,
    submission_path: Path,
) -> dict[str, object]:
    metrics_payload = metrics if isinstance(metrics, Mapping) else {}
    metrics_row_count_provided = (
        "test_submission_rows" in metrics_payload and metrics_payload.get("test_submission_rows") is not None
    )
    metrics_row_count = _strict_nonnegative_int(metrics_payload.get("test_submission_rows"))
    ledger_path = _find_prediction_ledger_path(
        metrics=metrics_payload,
        metrics_path=metrics_path,
        submission_path=submission_path,
    )
    return {
        "actual_row_count": int(actual_row_count),
        "metrics": {
            "test_submission_rows_provided": metrics_row_count_provided,
            "test_submission_rows": metrics_row_count,
            "valid": None if not metrics_row_count_provided else metrics_row_count is not None,
            "matches_actual": None if metrics_row_count is None else metrics_row_count == actual_row_count,
        },
        "prediction_ledger": _prediction_ledger_evidence(
            ledger_path=ledger_path,
            actual_row_count=actual_row_count,
            rows_per_base=rows_per_base,
        ),
    }


def _find_prediction_ledger_path(
    *,
    metrics: Mapping[str, object],
    metrics_path: Path | None,
    submission_path: Path,
) -> Path | None:
    candidates: list[Path] = []
    for key in ("test_prediction_ledger_path", "test_prediction_ledger"):
        raw = metrics.get(key)
        if not isinstance(raw, (str, Path)) or not str(raw).strip():
            continue
        candidate = Path(str(raw).strip())
        if not candidate.is_absolute() and metrics_path is not None:
            candidate = metrics_path.parent / candidate
        candidates.append(candidate)
    for directory in (
        metrics_path.parent if metrics_path is not None else None,
        submission_path.parent,
    ):
        if directory is None:
            continue
        candidates.extend(
            (directory / "test_prediction_ledger.csv", directory / "output" / "test_prediction_ledger.csv")
        )
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def _prediction_ledger_evidence(
    *,
    ledger_path: Path | None,
    actual_row_count: int,
    rows_per_base: Mapping[str, int],
) -> dict[str, object]:
    if ledger_path is None:
        return {"available": False}
    evidence: dict[str, object] = {
        "available": True,
        "path": str(ledger_path.resolve()),
        "sha256": sha256_file_or_none(ledger_path),
    }
    try:
        ledger = read_table(ledger_path)
    except Exception as exc:  # noqa: BLE001 - retain a bounded diagnostic for the fidelity report
        evidence.update({"valid": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
        return evidence
    ledger.columns = [str(column)[:500] for column in ledger.columns]
    count_column = next((column for column in ledger.columns if column.lower() == "prediction_count"), None)
    base_column = next(
        (
            column
            for preferred in ("stem", "base_identifier", "image_id", "id", "identifier")
            for column in ledger.columns
            if column.lower() == preferred
        ),
        None,
    )
    evidence.update(
        {
            "row_count": int(len(ledger)),
            "columns": list(ledger.columns)[:100],
            "count_column": count_column,
            "base_identifier_column": base_column,
        }
    )
    if count_column is None:
        evidence.update({"valid": False, "error": "prediction_count column is missing"})
        return evidence
    numeric_counts = pd.to_numeric(ledger[count_column], errors="coerce")
    parsed_counts = [_strict_nonnegative_int(value) for value in numeric_counts.tolist()]
    if any(value is None for value in parsed_counts):
        evidence.update({"valid": False, "error": "prediction_count contains invalid values"})
        return evidence
    ledger_count_values = [int(value) for value in parsed_counts if value is not None]
    ledger_total = sum(ledger_count_values)
    evidence.update(
        {
            "valid": True,
            "prediction_count_sum": ledger_total,
            "matches_actual": ledger_total == actual_row_count,
        }
    )
    if base_column is None:
        return evidence
    ledger_bases = [_normalized_cell(value) for value in ledger[base_column].tolist()]
    duplicate_bases = len(set(ledger_bases)) != len(ledger_bases)
    ledger_counts = dict(zip(ledger_bases, ledger_count_values, strict=False))
    unknown_bases = sorted(set(ledger_counts) - set(rows_per_base))
    missing_bases = sorted(set(rows_per_base) - set(ledger_counts))
    mismatched_bases = sorted(
        base for base in set(rows_per_base) & set(ledger_counts) if rows_per_base[base] != ledger_counts[base]
    )
    evidence.update(
        {
            "base_identifiers_unique": not duplicate_bases,
            "unknown_base_count": len(unknown_bases),
            "unknown_base_examples": unknown_bases[:20],
            "missing_base_count": len(missing_bases),
            "missing_base_examples": missing_bases[:20],
            "base_count_mismatch_count": len(mismatched_bases),
            "base_count_mismatch_examples": mismatched_bases[:20],
            "base_counts_match_actual": not duplicate_bases
            and not unknown_bases
            and not missing_bases
            and not mismatched_bases,
        }
    )
    return evidence


def _strict_nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        return None
    return int(numeric)


def _compare_local_tabular_contract(
    expected,  # noqa: ANN001
    actual,  # noqa: ANN001
    *,
    expected_identifier_cardinality,  # noqa: ANN001
    actual_identifier_cardinality,  # noqa: ANN001
    add,  # noqa: ANN001
    warn,  # noqa: ANN001
) -> None:
    expected_schema = _nested_list(expected, "schema", "columns")
    actual_schema = _nested_list(actual, "schema", "columns")
    if expected_schema and actual_schema != expected_schema:
        add("file_schema_mismatch", "prepared artifact schema differs from sample submission")

    actual_cardinality = actual_identifier_cardinality if isinstance(actual_identifier_cardinality, Mapping) else {}
    expected_cardinality = (
        expected_identifier_cardinality if isinstance(expected_identifier_cardinality, Mapping) else {}
    )
    variable_rows = (
        _text(expected_cardinality.get("mode")) == _VARIABLE_ROWS_IDENTIFIER_MODE
        and _text(actual_cardinality.get("mode")) == _VARIABLE_ROWS_IDENTIFIER_MODE
    )
    if actual_cardinality.get("candidate") is True:
        if (actual_cardinality.get("unknown_base_count") or 0) > 0:
            add("file_identifier_unknown_base", "prepared artifact contains identifiers with no sample image base")
        if (actual_cardinality.get("ambiguous_base_count") or 0) > 0:
            add(
                "file_identifier_base_mapping_ambiguous",
                "prepared artifact identifiers do not map unambiguously to sample image bases",
            )
        if (actual_cardinality.get("malformed_suffix_count") or 0) > 0:
            add(
                "file_identifier_suffix_malformed",
                "prepared artifact contains an instance identifier with an empty suffix",
            )
        if (actual_cardinality.get("actual_prediction_empty_count") or 0) > 0:
            add("file_prediction_value_empty", "prepared artifact contains empty prediction values")
    if _text(expected_cardinality.get("mode")) == _VARIABLE_ROWS_IDENTIFIER_MODE and not variable_rows:
        add(
            "file_identifier_cardinality_mode_mismatch",
            "prepared artifact no longer satisfies the frozen variable-row identifier contract",
        )
    if variable_rows:
        row_count_evidence = _nested_mapping(actual_cardinality, "row_count_evidence") or {}
        metrics_evidence = _nested_mapping(row_count_evidence, "metrics") or {}
        ledger_evidence = _nested_mapping(row_count_evidence, "prediction_ledger") or {}
        if metrics_evidence.get("test_submission_rows_provided") is True:
            if metrics_evidence.get("valid") is not True:
                add("file_metrics_row_count_invalid", "metrics.test_submission_rows is not a nonnegative integer")
            elif metrics_evidence.get("matches_actual") is not True:
                add(
                    "file_row_count_metrics_mismatch",
                    "prepared artifact row count differs from metrics.test_submission_rows",
                )
        if ledger_evidence.get("available") is True:
            if ledger_evidence.get("valid") is not True:
                add("file_prediction_ledger_invalid", "test prediction ledger is unreadable or invalid")
            else:
                if ledger_evidence.get("matches_actual") is not True:
                    add(
                        "file_row_count_ledger_mismatch",
                        "prepared artifact row count differs from the test prediction ledger sum",
                    )
                if (
                    ledger_evidence.get("base_identifier_column")
                    and ledger_evidence.get("base_counts_match_actual") is not True
                ):
                    add(
                        "file_identifier_ledger_count_mismatch",
                        "prepared artifact rows per base identifier differ from the test prediction ledger",
                    )
        if (actual_cardinality.get("missing_base_count") or 0) > 0:
            warn("file_identifier_bases_without_rows")
        return

    if expected.get("row_count") != actual.get("row_count"):
        add("file_row_count_mismatch", "prepared artifact row count differs from sample submission")
    expected_id = _nested_mapping(expected, "identifier")
    actual_id = _nested_mapping(actual, "identifier")
    if expected_id is not None and actual_id is not None:
        if _text(expected_id.get("composite_order_sha256")) != _text(actual_id.get("composite_order_sha256")):
            add("file_identifier_order_mismatch", "prepared artifact identifier order differs from sample submission")


def _metric_provenance(
    metrics: Mapping[str, object],
    *,
    score_value: float | None,
    score_direction: str | None,
) -> dict[str, object]:
    source = _text(metrics.get("score_source") or ("offline" if score_value is not None else "")).lower()
    metric = _metric_name(metrics)
    direction = _text(metrics.get("direction") or score_direction).lower()
    score = _score(metrics)
    if score is None:
        score = _finite_float(score_value)
    authoritative = metrics.get("authoritative")
    explicit_trusted = metrics.get("trusted") is True or metrics.get("score_trusted") is True
    trusted = bool(
        metric
        and direction in {"maximize", "minimize"}
        and score is not None
        and authoritative is not False
        and (explicit_trusted or source in _TRUSTED_SCORE_SOURCES)
    )
    return {
        "metric": metric or None,
        "direction": direction or None,
        "score_source": source or None,
        "score": score,
        "authoritative": authoritative,
        "trusted": trusted,
    }


def _first_text(payload: Mapping[str, object], keys: Iterable[str]) -> str:
    for key in keys:
        value = _text(payload.get(key))
        if value:
            return value
    return ""


def _report_metric_provenance(report: Mapping[str, object]) -> dict[str, object]:
    existing = report.get("metric_provenance")
    if isinstance(existing, Mapping):
        return dict(existing)
    selection = _nested_mapping(report, "comparison", "expected", "selection") or {}
    return _metric_provenance(selection, score_value=_finite_float(selection.get("score")), score_direction=None)


def _selected_assets(metrics: Mapping[str, object]) -> dict[str, object]:
    hashes: list[str] = []
    for key in ("artifact_hashes", "active_model_sha256", "model_artifact_sha256", "reference_artifact_sha256"):
        hashes.extend(_string_values(metrics.get(key)))
    return {
        "model_sources": _dedupe(
            [*_string_values(metrics.get("model_sources")), *_string_values(metrics.get("active_model_source"))]
        ),
        "reference_sources": _dedupe(
            [
                *_string_values(metrics.get("reference_sources")),
                *_string_values(metrics.get("active_reference_source") or metrics.get("reference_source")),
            ]
        ),
        "artifact_hashes": _dedupe(hashes),
    }


def _bounded_source_counts(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    raw = value.get("source_top10") or value.get("sources") or value
    if isinstance(raw, Mapping):
        return {str(key)[:200]: item for key, item in list(raw.items())[:20] if isinstance(item, (int, float))}
    return {}


def _fallback_sources_only(value: object) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    names = [_text(name).lower() for name in value]
    return bool(names) and all(any(marker in name for marker in _FALLBACK_MARKERS) for name in names)


def _looks_like_id_column(value: object) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", _text(value).lower()).strip("_")
    compact = normalized.replace("_", "")
    if any(normalized == prefix or normalized.startswith(f"{prefix}_") for prefix in _PREDICTION_COLUMN_PREFIXES):
        return False
    normalized_ids = {name.replace("_", "") for name in ID_LIKE_COLUMN_NAMES}
    return normalized in ID_LIKE_COLUMN_NAMES or compact in normalized_ids or compact.endswith("id")


def _series_order_digest(series: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in series.tolist():
        digest.update(_normalized_cell(value).encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()


def _normalized_cell(value: object) -> str:
    if value is None or (not isinstance(value, (list, tuple, dict)) and bool(pd.isna(value))):
        return "<KAGGLEBOT_MISSING>"
    return str(value)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "on"}


def _quarantine_active(value: Mapping[str, object] | None) -> bool:
    return isinstance(value, Mapping) and _text(value.get("status")).lower() == "active"


def build_submit_fidelity_expected_contract(
    *,
    package_dir: Path,
    slug: str,
    run_id: str,
    iteration: int,
    kernel_id: str,
    artifact_mode: str,
    expected_output_file: str,
    expected_metrics: Mapping[str, object] | None,
    selected_candidate_path: Path | None,
    requested_accelerator: str,
    executed_accelerator: str,
    machine_shape: str | None,
    capacity_fallback_used: bool,
) -> dict[str, object]:
    """Build the immutable selection/execution contract staged in a code kernel."""
    metrics = dict(expected_metrics or {})
    metadata = load_json_object(package_dir / "kernel-metadata.json") or {}
    selected_sha = sha256_file_or_none(selected_candidate_path)
    selected_size = _path_size(selected_candidate_path)
    artifact_hashes = _string_values(metrics.get("artifact_hashes"))
    for key in ("active_model_sha256", "model_artifact_sha256", "reference_artifact_sha256"):
        artifact_hashes.extend(_string_values(metrics.get(key)))
    active_model_source = _text(metrics.get("active_model_source"))
    active_reference_source = _text(metrics.get("active_reference_source") or metrics.get("reference_source"))
    model_sources = _dedupe(
        [
            *_string_values(metadata.get("model_sources")),
            *_string_values(metrics.get("model_sources")),
            *([active_model_source] if active_model_source else []),
        ]
    )
    reference_sources = _dedupe(
        [
            *_string_values(metrics.get("reference_sources")),
            *([active_reference_source] if active_reference_source else []),
        ]
    )
    score = _score(metrics)
    return {
        "schema_version": 1,
        "contract": "kagglebot_submit_runtime_fidelity",
        "competition": slug,
        "run_id": run_id,
        "iteration": iteration,
        "artifact_mode": _text(artifact_mode).lower(),
        "kernel": {"id": kernel_id},
        "selection": {
            "pipeline": _pipeline_name(metrics),
            "metric": _metric_name(metrics),
            "direction": _text(metrics.get("direction")).lower(),
            "score_source": _text(metrics.get("score_source")),
            "score": score,
            "authoritative": metrics.get("authoritative"),
            "reference_path_required": metrics.get("reference_path_used") is True,
        },
        "expected_assets": {
            "model_sources": model_sources,
            "reference_sources": reference_sources,
            "required_runtime_model_sources": [active_model_source] if active_model_source else [],
            "required_runtime_reference_sources": [active_reference_source] if active_reference_source else [],
            "artifact_hashes": _dedupe(artifact_hashes),
            "declared_dataset_sources": _string_values(metadata.get("dataset_sources")),
            "declared_kernel_sources": _string_values(metadata.get("kernel_sources")),
        },
        "output": {"filename": Path(expected_output_file).name},
        "accelerator": {
            "requested": _text(requested_accelerator).lower(),
            "executed": _text(executed_accelerator).lower(),
            "machine_shape": _text(machine_shape),
            "capacity_fallback_used": bool(capacity_fallback_used),
            "requires_gpu": _text(artifact_mode).lower() in {"gateway", "inference"}
            or _requires_gpu_execution(metrics),
        },
        "selected_local_candidate": {
            "name": selected_candidate_path.name if selected_candidate_path is not None else None,
            "kind": (
                "directory"
                if selected_candidate_path is not None and selected_candidate_path.is_dir()
                else "file"
                if selected_candidate_path is not None and selected_candidate_path.is_file()
                else None
            ),
            "size": selected_size,
            "sha256": selected_sha,
        },
        # The package digest excludes this expected file to avoid a circular
        # hash. The completed outer source fingerprint includes this file.
        "package_fingerprint": package_source_fingerprint(package_dir),
    }


def stage_submit_fidelity_expected_contract(
    *,
    package_dir: Path,
    slug: str,
    run_id: str,
    iteration: int,
    kernel_id: str,
    artifact_mode: str,
    expected_output_file: str,
    expected_metrics: Mapping[str, object] | None,
    selected_candidate_path: Path | None,
    requested_accelerator: str,
    executed_accelerator: str,
    machine_shape: str | None,
    capacity_fallback_used: bool,
) -> Path:
    """Write the canonical expected contract into a freshly built package."""
    path = package_dir / EXPECTED_FILE_NAME
    payload = build_submit_fidelity_expected_contract(
        package_dir=package_dir,
        slug=slug,
        run_id=run_id,
        iteration=iteration,
        kernel_id=kernel_id,
        artifact_mode=artifact_mode,
        expected_output_file=expected_output_file,
        expected_metrics=expected_metrics,
        selected_candidate_path=selected_candidate_path,
        requested_accelerator=requested_accelerator,
        executed_accelerator=executed_accelerator,
        machine_shape=machine_shape,
        capacity_fallback_used=capacity_fallback_used,
    )
    write_json_object(path, payload, ensure_ascii=False, sort_keys=True)
    return path


def validate_reference_submission_readiness(
    *,
    reproduction_report_path: Path,
    expected_metrics: Mapping[str, object] | None,
) -> None:
    """Protect scarce submit slots while the selected code reference is known to be unreproduced."""
    report = load_json_object(reproduction_report_path)
    if not report or report.get("blocks_novelty") is not True:
        return
    if _text(report.get("status")).lower() != "blocked":
        return
    baseline = _finite_float(report.get("campaign_baseline_score") or report.get("code_reference_score"))
    candidate = _score(expected_metrics or {})
    if candidate is None:
        candidate = _finite_float(report.get("candidate_score"))
    if baseline is None or candidate is None or candidate >= baseline:
        return
    gate_reason = _text(report.get("gate_reason")) or "reference_reproduction_blocked"
    raise KernelFailedError(
        "Refusing code-competition submission because the required public reference has not been reproduced: "
        f"candidate={candidate:.12g}, reference={baseline:.12g}, gate={gate_reason}. "
        "Restore its required model/kernel/dataset sources and validate that path before spending a submission slot."
    )


def validate_submit_kernel_runtime_fidelity(
    *,
    artifact_mode: str | None,
    expected_metrics: Mapping[str, object] | None,
    actual_metrics_path: Path | None,
    expected_contract_path: Path | None = None,
    runtime_fidelity_path: Path | None = None,
    submission_path: Path | None = None,
    package_dir: Path | None = None,
    report_path: Path | None = None,
    kernel_id: str | None = None,
    kernel_version: str | None = None,
    run_id: str | None = None,
    iteration: int | None = None,
    previous_report_paths: Iterable[Path] = (),
    quarantine_state: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    """Validate remote execution against selection and return one normalized report.

    Direct legacy callers that provide only metrics retain the original
    metrics-only behavior. Freshly staged code-submit packages pass the expected
    contract/runtime paths and therefore use the fail-closed attestation path.
    """
    mode = str(artifact_mode or "").strip().lower()
    if mode not in _CODE_OUTPUT_MODES:
        return None
    strong_contract = any(
        value is not None
        for value in (expected_contract_path, runtime_fidelity_path, submission_path, package_dir, report_path)
    )
    if not strong_contract:
        if not expected_metrics:
            return None
        if actual_metrics_path is None:
            _raise_legacy_fidelity_error(["remote inference did not produce metrics.json"])
        actual_metrics = load_json_object(actual_metrics_path)
        if not actual_metrics:
            _raise_legacy_fidelity_error([f"remote inference metrics are missing or invalid: {actual_metrics_path}"])
        problems = _fidelity_problems(expected_metrics, actual_metrics)
        if problems:
            _raise_legacy_fidelity_error(problems)
        return None

    resolved_report_path = report_path or (
        (actual_metrics_path or submission_path or Path.cwd()).parent / REPORT_FILE_NAME
    )
    previous_reports = [payload for path in previous_report_paths if (payload := load_json_object(path)) is not None]
    if resolved_report_path.exists():
        existing = load_json_object(resolved_report_path)
        if existing is not None:
            previous_reports.append(existing)
    report = build_submission_fidelity_report(
        artifact_mode=mode,
        expected_metrics=expected_metrics,
        actual_metrics_path=actual_metrics_path,
        expected_contract_path=expected_contract_path,
        runtime_fidelity_path=runtime_fidelity_path,
        submission_path=submission_path,
        package_dir=package_dir,
        kernel_id=kernel_id,
        kernel_version=kernel_version,
        run_id=run_id,
        iteration=iteration,
        previous_reports=previous_reports,
    )
    report = enforce_submission_fidelity_quarantine(report=report, quarantine_state=quarantine_state)
    report["report_path"] = str(resolved_report_path.resolve())
    report = finalize_submission_fidelity_report(report)
    write_json_object(resolved_report_path, report, ensure_ascii=False, sort_keys=True)
    if report.get("verdict") != "pass":
        _raise_normalized_fidelity_error(report, report_path=resolved_report_path)
    return report


def build_submission_fidelity_report(
    *,
    artifact_mode: str,
    expected_metrics: Mapping[str, object] | None,
    actual_metrics_path: Path | None,
    expected_contract_path: Path | None,
    runtime_fidelity_path: Path | None,
    submission_path: Path | None,
    package_dir: Path | None,
    kernel_id: str | None,
    kernel_version: str | None,
    run_id: str | None,
    iteration: int | None,
    previous_reports: Iterable[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Compare all expected/runtime/output evidence with stable reason codes."""
    reasons: list[dict[str, str]] = []

    def add(code: str, message: str) -> None:
        if any(item["code"] == code for item in reasons):
            return
        reasons.append({"code": code, "message": message})

    expected = load_json_object(expected_contract_path) if expected_contract_path is not None else None
    runtime = load_json_object(runtime_fidelity_path) if runtime_fidelity_path is not None else None
    actual_metrics = load_json_object(actual_metrics_path) if actual_metrics_path is not None else None
    contract_sha = sha256_file_or_none(expected_contract_path)
    runtime_sha = sha256_file_or_none(runtime_fidelity_path)
    metrics_sha = sha256_file_or_none(actual_metrics_path)
    selected_output_sha = sha256_file_or_none(submission_path)
    selected_output_size = _path_size(submission_path)
    output_evidence_root = (
        runtime_fidelity_path.parent
        if runtime_fidelity_path is not None
        else actual_metrics_path.parent
        if actual_metrics_path is not None
        else submission_path.parent
        if submission_path is not None
        else None
    )
    downloaded_error_paths = _downloaded_error_transcripts(output_evidence_root)

    if expected is None:
        add("expected_contract_missing_or_invalid", "submit_fidelity_expected.json is missing or invalid")
    if runtime is None:
        add("runtime_attestation_missing_or_invalid", "submit_fidelity_runtime.json is missing or invalid")

    expected_package = _text(expected.get("package_fingerprint")) if expected else ""
    runtime_package = _nested_text(runtime, "package", "source_sha256")
    runtime_contract_sha = _nested_text(runtime, "expected_contract", "sha256")
    if expected is not None:
        if expected.get("schema_version") != 1:
            add("expected_contract_schema_unsupported", "expected fidelity contract schema is not supported")
        selection = _nested_mapping(expected, "selection") or {}
        missing_selection_fields = [
            key for key in ("pipeline", "metric", "direction", "score_source") if not _text(selection.get(key))
        ]
        if missing_selection_fields:
            add(
                "expected_selection_contract_incomplete",
                "expected fidelity contract is missing: " + ", ".join(missing_selection_fields),
            )
        if _nested_text(expected, "output", "filename") == "":
            add("expected_output_contract_incomplete", "expected fidelity contract has no output filename")
        selected_local_candidate = _nested_mapping(expected, "selected_local_candidate") or {}
        if not _text(selected_local_candidate.get("sha256")):
            add(
                "selected_local_candidate_evidence_missing",
                "expected fidelity contract has no selected local candidate hash",
            )
        if _text(expected.get("artifact_mode")).lower() != artifact_mode:
            add("artifact_mode_mismatch", "expected fidelity contract artifact mode differs from submit mode")
        if run_id and _text(expected.get("run_id")) != run_id:
            add("run_identity_mismatch", "expected fidelity contract belongs to another run")
        if iteration is not None and expected.get("iteration") != iteration:
            add("iteration_identity_mismatch", "expected fidelity contract belongs to another iteration")
        expected_kernel_id = _nested_text(expected, "kernel", "id")
        if kernel_id and expected_kernel_id != kernel_id:
            add("kernel_identity_mismatch", "expected fidelity contract belongs to another kernel")
    if runtime is not None:
        if runtime.get("schema_version") != 1:
            add("runtime_attestation_schema_unsupported", "runtime fidelity attestation schema is not supported")
        if runtime.get("recorder_failure"):
            add("runtime_recorder_failed", "runtime fidelity recorder did not complete")
        if not contract_sha or runtime_contract_sha != contract_sha:
            add("expected_contract_digest_mismatch", "runtime did not execute the exact staged expected contract")
        if not expected_package or runtime_package != expected_package:
            add("package_fingerprint_mismatch", "runtime packaged-source digest differs from the selected package")
        if expected is not None:
            _compare_runtime_identity(expected, runtime, add=add)
            _compare_accelerator(expected, runtime, add=add)
            _compare_assets(expected, runtime, add=add)
        _check_input_inventory(runtime, add=add)
        _check_runtime_errors(runtime, add=add)
        _check_prediction_fallback(runtime, add=add)
    if downloaded_error_paths:
        add("runtime_error_transcript_present", "current kernel version emitted a non-empty error transcript")
        declared_error_paths = {
            _text(item.get("relative_path"))
            for item in _nested_list(runtime, "errors", "transcripts")
            if isinstance(item, Mapping)
        }
        undeclared = [
            path
            for path in downloaded_error_paths
            if output_evidence_root is not None
            and path.relative_to(output_evidence_root).as_posix() not in declared_error_paths
        ]
        if undeclared:
            add(
                "runtime_error_evidence_incomplete",
                "downloaded current-version error transcripts were absent from runtime attestation",
            )

    if package_dir is not None and package_dir.is_dir() and expected_package:
        try:
            local_package = package_source_fingerprint(package_dir)
        except OSError:
            local_package = ""
        if local_package != expected_package:
            add("local_package_mutated", "local package changed after the expected contract was staged")

    expected_output_file = _nested_text(expected, "output", "filename")
    runtime_selected = _nested_mapping(runtime, "outputs", "selected")
    runtime_candidates = _nested_list(runtime, "outputs", "candidates")
    candidate_count = _nested_int(runtime, "outputs", "candidate_count")
    selected_count = _nested_int(runtime, "outputs", "selected_candidate_count")
    if submission_path is None or not submission_path.is_file():
        add("selected_output_missing", "completed kernel output is missing the selected submission artifact")
    elif expected_output_file and submission_path.name != expected_output_file:
        add("selected_output_filename_mismatch", "selected output filename differs from the expected contract")
    if runtime is not None:
        if candidate_count is None or candidate_count != len(runtime_candidates):
            add("runtime_output_evidence_contradictory", "runtime output candidate count contradicts its inventory")
        if selected_count == 0 or runtime_selected is None:
            add("runtime_selected_output_missing", "runtime did not select the exact expected output")
        elif selected_count != 1:
            add("runtime_selected_output_ambiguous", "runtime found multiple candidates for the expected output")
        if runtime_selected is not None:
            runtime_relative_path = _text(runtime_selected.get("relative_path"))
            if (
                runtime_fidelity_path is not None
                and _safe_child(runtime_fidelity_path.parent, runtime_relative_path) is None
            ):
                add("runtime_selected_output_path_unsafe", "runtime selected output path is not safely relative")
            runtime_output_sha = _text(runtime_selected.get("sha256")).lower()
            runtime_output_name = _text(runtime_selected.get("filename"))
            runtime_output_size = _nonnegative_int(runtime_selected.get("size"))
            if expected_output_file and runtime_output_name != expected_output_file:
                add("runtime_output_filename_mismatch", "runtime selected an unexpected output filename")
            if not selected_output_sha or runtime_output_sha != selected_output_sha:
                add("selected_output_hash_mismatch", "downloaded output differs from the runtime-selected artifact")
            if selected_output_size is not None and runtime_output_size != selected_output_size:
                add("selected_output_size_mismatch", "downloaded output size differs from runtime evidence")
        if selected_count is not None and selected_count > 1:
            add("runtime_selected_output_ambiguous", "runtime found multiple candidates for the expected output")
        if expected_output_file:
            matching_candidates = [
                item
                for item in runtime_candidates
                if isinstance(item, Mapping) and _text(item.get("filename")) == expected_output_file
            ]
            if len(matching_candidates) != 1:
                add(
                    "runtime_output_candidate_set_invalid",
                    "runtime output candidates do not identify one exact output",
                )
            if selected_count is not None and selected_count != len(matching_candidates):
                add(
                    "runtime_output_evidence_contradictory",
                    "runtime selected-output count contradicts its candidate inventory",
                )
        runtime_metrics_sha = _nested_text(runtime, "outputs", "metrics_sha256")
        if not metrics_sha or runtime_metrics_sha != metrics_sha:
            add("metrics_hash_mismatch", "downloaded metrics.json differs from runtime evidence")

    if actual_metrics is None:
        add("runtime_metrics_missing_or_invalid", "completed inference metrics.json is missing or invalid")
    elif expected_metrics:
        for problem in _fidelity_problems(expected_metrics, actual_metrics):
            add("metrics_selection_contract_mismatch", problem)
    if expected is not None and actual_metrics is not None:
        _compare_metric_contract(expected, actual_metrics, add=add)
    if runtime is not None and actual_metrics is not None:
        _compare_metrics_output_claims(runtime, submission_path, selected_output_sha, add=add)

    tabular = _nested_value(runtime, "tabular_prediction") if runtime else None
    if isinstance(tabular, Mapping):
        # Hosted code competitions evaluate the served program during a hidden
        # rerun.  Their visible commit artifact is only a tiny gateway
        # placeholder, not the hidden prediction set, so ordinary prediction
        # dispersion/cardinality checks do not apply to this artifact.
        if artifact_mode not in _CODE_OUTPUT_MODES:
            for finding in runtime_tabular_fidelity_findings(dict(tabular)):
                add(str(finding["code"]), str(finding["message"]))
            _compare_tabular_input_contract(runtime, tabular, add=add)

    attempt_fingerprint = sha256_text("\0".join((expected_package, str(selected_output_sha or ""))))
    for previous in previous_reports:
        if previous.get("verdict") != "fail":
            continue
        if _text(previous.get("attempt_fingerprint")) == attempt_fingerprint:
            add(
                "unchanged_failed_fidelity_attempt",
                "a prior failed fidelity attempt has the same package, expected contract, and output hash",
            )
            break

    reason_codes = sorted(item["code"] for item in reasons)
    supporting_paths = [
        str(path.resolve())
        for path in (expected_contract_path, runtime_fidelity_path, actual_metrics_path, submission_path)
        if path is not None
    ]
    supporting_paths.extend(str(path.resolve()) for path in downloaded_error_paths)
    if runtime_fidelity_path is not None and runtime is not None:
        output_root = runtime_fidelity_path.parent
        for section, key in (("outputs", "candidates"), ("errors", "transcripts")):
            rows = _nested_list(runtime, section, key)
            for item in rows:
                if not isinstance(item, Mapping):
                    continue
                relative = _text(item.get("relative_path"))
                candidate = _safe_child(output_root, relative)
                if candidate is not None and candidate.is_file():
                    supporting_paths.append(str(candidate.resolve()))
    supporting_paths = sorted(set(supporting_paths))
    expected_accelerator = _nested_mapping(expected, "accelerator")
    runtime_accelerator = _nested_mapping(runtime, "accelerator")
    runtime_errors = _nested_mapping(runtime, "errors")
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": REPORT_TYPE,
        "verdict": "fail" if reasons else "pass",
        "reason_codes": reason_codes,
        "reasons": sorted(reasons, key=lambda item: item["code"]),
        "competition": _text(expected.get("competition")) if expected else None,
        "run_id": run_id or (_text(expected.get("run_id")) if expected else None),
        "iteration": iteration if iteration is not None else expected.get("iteration") if expected else None,
        "kernel_id": kernel_id or _nested_text(expected, "kernel", "id"),
        "kernel_version": kernel_version,
        "artifact_mode": artifact_mode,
        "attestation_scope": "remote_notebook_runtime",
        "remote_runtime_attested": True,
        "package_fingerprint": expected_package or None,
        "expected_contract_sha256": contract_sha,
        "runtime_attestation_sha256": runtime_sha,
        "selected_output": {
            "path": str(submission_path.resolve()) if submission_path is not None else None,
            "filename": submission_path.name if submission_path is not None else None,
            "size": selected_output_size,
            "sha256": selected_output_sha,
        },
        "metrics_sha256": metrics_sha,
        "metric_provenance": _metric_provenance(
            dict(_nested_mapping(expected, "selection") or {}),
            score_value=_finite_float(_nested_value(expected, "selection", "score")),
            score_direction=_nested_text(expected, "selection", "direction"),
        ),
        "attempt_fingerprint": attempt_fingerprint,
        "comparison": {
            "expected": {
                "package_fingerprint": expected_package or None,
                "contract_sha256": contract_sha,
                "output_filename": expected_output_file or None,
                "accelerator": dict(expected_accelerator) if expected_accelerator is not None else None,
                "selection": dict(_nested_mapping(expected, "selection") or {}),
                "assets": dict(_nested_mapping(expected, "expected_assets") or {}),
            },
            "actual": {
                "package_fingerprint": runtime_package or None,
                "contract_sha256": runtime_contract_sha or None,
                "output": dict(runtime_selected) if runtime_selected is not None else None,
                "accelerator": dict(runtime_accelerator) if runtime_accelerator is not None else None,
                "metrics_sha256": metrics_sha,
                "traceback_count": runtime_errors.get("traceback_count") if runtime_errors is not None else None,
            },
        },
        "supporting_artifact_paths": supporting_paths,
    }


def _compare_runtime_identity(expected, runtime, *, add) -> None:  # noqa: ANN001
    if _nested_text(runtime, "kernel", "contract_kernel_id") != _nested_text(expected, "kernel", "id"):
        add("runtime_kernel_identity_mismatch", "runtime contract kernel identity is contradictory")
    if _nested_text(runtime, "kernel", "run_id") != _text(expected.get("run_id")):
        add("runtime_run_identity_mismatch", "runtime attestation belongs to another run")
    if _nested_value(runtime, "kernel", "iteration") != expected.get("iteration"):
        add("runtime_iteration_identity_mismatch", "runtime attestation belongs to another iteration")
    environment_kernel = _nested_text(runtime, "kernel", "environment_kernel_id")
    if environment_kernel and environment_kernel != _nested_text(expected, "kernel", "id"):
        add("runtime_environment_kernel_mismatch", "runtime environment reports another kernel identity")


def _compare_accelerator(expected, runtime, *, add) -> None:  # noqa: ANN001
    expected_accelerator = _nested_mapping(expected, "accelerator") or {}
    runtime_accelerator = _nested_mapping(runtime, "accelerator") or {}
    for key in ("requested", "executed", "machine_shape"):
        expected_value = _text(expected_accelerator.get(key)).lower()
        actual_value = _text(runtime_accelerator.get(key)).lower()
        if expected_value != actual_value:
            add(f"accelerator_{key}_mismatch", f"runtime accelerator {key} differs from the package contract")
    expected_fallback = expected_accelerator.get("capacity_fallback_used") is True
    actual_fallback = runtime_accelerator.get("capacity_fallback_used") is True
    if expected_fallback != actual_fallback:
        add("accelerator_fallback_mismatch", "runtime capacity-fallback evidence is contradictory")
    if actual_fallback and expected_accelerator.get("requires_gpu") is True:
        add("gpu_required_capacity_fallback", "selected contract requires GPU-specific execution and ran on fallback")


def _compare_assets(expected, runtime, *, add) -> None:  # noqa: ANN001
    expected_assets = _nested_mapping(expected, "expected_assets") or {}
    runtime_metrics = _nested_mapping(runtime, "runtime_metrics") or {}
    runtime_models = set(_string_values(runtime_metrics.get("model_sources")))
    runtime_references = set(_string_values(runtime_metrics.get("reference_sources")))
    required_models = set(_string_values(expected_assets.get("required_runtime_model_sources")))
    required_references = set(_string_values(expected_assets.get("required_runtime_reference_sources")))
    if required_models - runtime_models:
        add("required_model_source_not_loaded", "runtime did not report the selected model source as loaded")
    if required_references - runtime_references:
        add("required_reference_source_not_loaded", "runtime did not report the selected reference source as loaded")
    expected_hashes = set(_string_values(expected_assets.get("artifact_hashes")))
    actual_hashes = set(_string_values(runtime_metrics.get("artifact_hashes")))
    if expected_hashes - actual_hashes:
        add("required_artifact_hash_not_loaded", "runtime artifact hashes do not include selected asset evidence")
    selection = _nested_mapping(expected, "selection") or {}
    if selection.get("reference_path_required") is True and runtime_metrics.get("reference_path_used") is not True:
        add("required_reference_path_not_used", "selected reference inference path was not used at runtime")


def _check_input_inventory(runtime, *, add) -> None:  # noqa: ANN001
    inventory = _nested_mapping(runtime, "input_inventory")
    if inventory is None or inventory.get("root_present") is not True:
        add("input_inventory_missing", "runtime did not inventory its current /kaggle/input tree")
        return
    if _nonnegative_int(inventory.get("file_count")) in {None, 0}:
        add("input_inventory_empty", "runtime input inventory contains no current-version input files")


def _check_runtime_errors(runtime, *, add) -> None:  # noqa: ANN001
    errors = _nested_mapping(runtime, "errors")
    if errors is None:
        add("runtime_error_evidence_missing", "runtime error evidence is missing")
        return
    if errors.get("unhandled_exception"):
        add("runtime_unhandled_exception", "runtime captured an unhandled exception")
    transcripts = errors.get("transcripts")
    rows = transcripts if isinstance(transcripts, list) else []
    nonempty = [item for item in rows if isinstance(item, Mapping) and item.get("nonempty") is True]
    transcript_tracebacks = sum(
        _nonnegative_int(item.get("traceback_count")) or 0 for item in rows if isinstance(item, Mapping)
    )
    reported_tracebacks = _nonnegative_int(errors.get("traceback_count"))
    if nonempty:
        add("runtime_error_transcript_present", "current kernel version emitted a non-empty error transcript")
    if reported_tracebacks is None:
        add("runtime_traceback_count_missing", "runtime traceback count is missing")
    elif reported_tracebacks < transcript_tracebacks:
        add("runtime_error_evidence_contradictory", "runtime traceback summary contradicts its error transcripts")


def _check_prediction_fallback(runtime, *, add) -> None:  # noqa: ANN001
    metrics = _nested_mapping(runtime, "runtime_metrics") or {}
    flags = metrics.get("fallback_flags")
    if isinstance(flags, Mapping) and any(value is True for value in flags.values()):
        add("prediction_fallback_used", "runtime metrics report fallback/dummy prediction generation")
    distribution = metrics.get("prediction_source_distribution")
    if not isinstance(distribution, Mapping):
        return
    raw_sources = distribution.get("source_top10") or distribution.get("sources")
    names: list[str] = []
    if isinstance(raw_sources, Mapping):
        names = [_text(name).lower() for name in raw_sources]
    elif isinstance(raw_sources, list):
        for item in raw_sources:
            if isinstance(item, (list, tuple)) and item:
                names.append(_text(item[0]).lower())
            elif isinstance(item, Mapping):
                names.append(_text(item.get("source") or item.get("name")).lower())
    if names and all(any(marker in name for marker in _FALLBACK_MARKERS) for name in names):
        add("prediction_sources_fallback_only", "all runtime prediction sources are fallback/dummy paths")


def _compare_metric_contract(expected, actual_metrics, *, add) -> None:  # noqa: ANN001
    selection = _nested_mapping(expected, "selection") or {}
    expected_pipeline = _text(selection.get("pipeline"))
    actual_pipeline = _pipeline_name(actual_metrics)
    if expected_pipeline and expected_pipeline != actual_pipeline:
        add("selected_pipeline_mismatch", "runtime pipeline differs from the canonical selected pipeline")
    expected_metric = _text(selection.get("metric"))
    actual_metric = _metric_name(actual_metrics)
    if expected_metric and (not actual_metric or not metrics_equivalent(expected_metric, actual_metric)):
        add("selected_metric_mismatch", "runtime metric differs from the canonical selected metric")
    expected_source = _text(selection.get("score_source"))
    actual_source = _text(actual_metrics.get("score_source"))
    if expected_source and expected_source != actual_source:
        add("metric_source_mismatch", "runtime metric source differs from the selected metric contract")
    expected_direction = _text(selection.get("direction")).lower()
    actual_direction = _text(actual_metrics.get("direction")).lower()
    if expected_direction and expected_direction != actual_direction:
        add("metric_direction_mismatch", "runtime metric direction differs from the selected metric contract")
    expected_score = _finite_float(selection.get("score"))
    actual_score = _score(actual_metrics)
    non_authoritative_reference = selection.get("authoritative") is False and expected_source.lower() in {
        "public_lb_reference",
        "public_leaderboard_reference",
    }
    if not non_authoritative_reference and expected_score is not None and actual_score is None:
        add("selected_score_missing", "runtime did not report the canonical selected score evidence")
    elif (
        not non_authoritative_reference
        and expected_score is not None
        and actual_score is not None
        and expected_direction in {"maximize", "minimize"}
    ):
        tolerance = max(abs(expected_score) * 0.25, 0.05)
        regressed = (
            expected_direction == "maximize"
            and actual_score < expected_score - tolerance
            or expected_direction == "minimize"
            and actual_score > expected_score + tolerance
        )
        if regressed:
            add("selected_score_regression", "runtime score regressed materially from the canonical selected score")


def _compare_metrics_output_claims(runtime, submission_path, selected_output_sha, *, add) -> None:  # noqa: ANN001
    metrics = _nested_mapping(runtime, "runtime_metrics") or {}
    reported_name = _text(metrics.get("reported_output_filename"))
    reported_sha = _text(metrics.get("reported_output_sha256")).lower()
    selected = _nested_mapping(runtime, "outputs", "selected") or {}
    if reported_name and submission_path is not None and Path(reported_name).name != submission_path.name:
        add("metrics_output_filename_mismatch", "metrics output filename contradicts the selected artifact")
    if reported_sha and selected_output_sha and reported_sha != selected_output_sha:
        add("metrics_output_hash_mismatch", "metrics output hash contradicts the selected artifact")
    reported_rows = _nonnegative_int(metrics.get("reported_row_count"))
    tabular = _nested_mapping(runtime, "tabular_prediction")
    tabular_rows = _nonnegative_int(tabular.get("row_count")) if tabular is not None else None
    if reported_rows is not None and tabular_rows is not None and reported_rows != tabular_rows:
        add("metrics_output_row_count_mismatch", "metrics row count contradicts the runtime output")
    selected_sha = _text(selected.get("sha256")).lower()
    if reported_sha and selected_sha and reported_sha != selected_sha:
        add("runtime_metrics_selected_hash_mismatch", "runtime metrics and selected-output hashes contradict")


def _compare_tabular_input_contract(runtime, tabular, *, add) -> None:  # noqa: ANN001
    contracts = _nested_list(runtime, "input_inventory", "tabular_identifier_contracts")
    rows = [item for item in contracts if isinstance(item, Mapping)]
    if not rows:
        return
    output_rows = _nonnegative_int(tabular.get("row_count"))
    output_schema = _nested_mapping(tabular, "schema") or {}
    output_columns = [str(item) for item in _nested_list(output_schema, "columns")]
    output_identifier = _nested_mapping(tabular, "identifier") or {}
    output_id_columns = set(map(str, _nested_list(output_identifier, "columns")))

    def rank(item: Mapping[str, object]) -> tuple[int, str]:
        input_rows = _nonnegative_int(item.get("row_count"))
        input_identifier = _nested_mapping(item, "identifier") or {}
        input_id_columns = set(map(str, _nested_list(input_identifier, "columns")))
        schema = _nested_mapping(item, "schema") or {}
        columns = [str(value) for value in _nested_list(schema, "columns")]
        same_rows = output_rows is not None and input_rows == output_rows
        same_ids = bool(output_id_columns & input_id_columns)
        exact_schema = bool(output_columns) and columns == output_columns
        priority = 0 if same_rows and exact_schema else 1 if same_rows and same_ids else 2
        return (priority, _text(item.get("relative_path")))

    selected = min(rows, key=rank)
    input_rows = _nonnegative_int(selected.get("row_count"))
    if output_rows is not None and input_rows is not None and output_rows != input_rows:
        add("tabular_input_output_row_count_mismatch", "runtime output row count differs from discovered test evidence")
    input_identifier = _nested_mapping(selected, "identifier") or {}
    input_order = _nested_mapping(input_identifier, "order_digests") or {}
    output_order = _nested_mapping(output_identifier, "order_digests") or {}
    common = sorted(set(map(str, input_order)) & set(map(str, output_order)))
    if common and any(_text(input_order[name]) != _text(output_order[name]) for name in common):
        add("tabular_identifier_order_mismatch", "runtime output identifier order differs from test/sample evidence")
    selected_name = Path(_text(selected.get("relative_path"))).name.lower()
    if "sample_submission" in selected_name or "answer_template" in selected_name:
        input_schema = _nested_mapping(selected, "schema") or {}
        input_columns = [str(value) for value in _nested_list(input_schema, "columns")]
        if input_columns and output_columns != input_columns:
            add("tabular_sample_schema_mismatch", "runtime output schema differs from the current sample contract")


def _raise_normalized_fidelity_error(report: Mapping[str, object], *, report_path: Path) -> None:
    codes = ",".join(map(str, report.get("reason_codes") or []))
    package = _text(report.get("package_fingerprint")) or "<missing>"
    selected = report.get("selected_output")
    selected_sha = _text(selected.get("sha256")) if isinstance(selected, Mapping) else ""
    diagnostic = (
        "Submission runtime fidelity validation failed: "
        f"reason_codes={codes or '<missing>'}; report={report_path}; "
        f"package_fingerprint={package}; selected_output_sha256={selected_sha or '<missing>'}"
    )
    error = SubmissionValidationError(diagnostic)
    error.reason_codes = list(report.get("reason_codes") or [])
    error.report_path = report_path
    error.fidelity_repair_required = True
    error.submission_fidelity = fidelity_report_summary(report, report_path=report_path)
    error.package_fingerprint = package
    error.selected_output_sha256 = selected_sha
    error.attempt_fingerprint = _text(report.get("attempt_fingerprint"))
    raise error


def _raise_legacy_fidelity_error(problems: Iterable[str]) -> None:
    diagnostic = "Invalid code submission runtime fidelity: " + "; ".join(problems)
    raise SubmissionCliError(
        "Notebook code-submission runtime result does not match the selected candidate.",
        command=[],
        exit_code=6,
        output=diagnostic,
        stdout="",
        stderr=diagnostic,
    )


def _fidelity_problems(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
) -> list[str]:
    problems: list[str] = []
    expected_pipeline = _pipeline_name(expected)
    actual_pipeline = _pipeline_name(actual)
    if expected_pipeline:
        if not actual_pipeline:
            problems.append(f"selected pipeline {expected_pipeline!r} was not reported")
        elif expected_pipeline != actual_pipeline:
            problems.append(f"pipeline changed from {expected_pipeline!r} to {actual_pipeline!r}")

    expected_metric = _metric_name(expected)
    actual_metric = _metric_name(actual)
    if expected_metric:
        if not actual_metric:
            problems.append(f"selected metric {expected_metric!r} was not reported")
        elif not metrics_equivalent(expected_metric, actual_metric):
            problems.append(f"metric changed from {expected_metric!r} to {actual_metric!r}")

    if expected.get("reference_path_used") is True and actual.get("reference_path_used") is not True:
        problems.append("selected reference execution path was not used")

    expected_model = _text(expected.get("active_model_source"))
    actual_model = _text(actual.get("active_model_source"))
    if expected_model:
        if not actual_model:
            problems.append(f"selected active model {expected_model!r} was not reported")
        elif expected_model != actual_model:
            problems.append(f"active model changed from {expected_model!r} to {actual_model!r}")

    for key in _MISSING_ASSET_KEYS:
        newly_missing = sorted(_string_set(actual.get(key)) - _string_set(expected.get(key)))
        if newly_missing:
            problems.append(f"{key} added {newly_missing!r}")

    non_authoritative_reference = expected.get("authoritative") is False and _text(
        expected.get("score_source")
    ).lower() in {"public_lb_reference", "public_leaderboard_reference"}
    expected_score = _score(expected)
    actual_score = _score(actual)
    direction = _text(expected.get("direction") or actual.get("direction")).lower()
    if not non_authoritative_reference and expected_score is not None and actual_score is None:
        problems.append(f"selected score {expected_score:.12g} was not reported")
    elif (
        not non_authoritative_reference
        and expected_score is not None
        and actual_score is not None
        and direction in {"maximize", "minimize"}
    ):
        tolerance = max(abs(expected_score) * 0.25, 0.05)
        regressed = (
            direction == "maximize"
            and actual_score < expected_score - tolerance
            or direction == "minimize"
            and actual_score > expected_score + tolerance
        )
        if regressed:
            problems.append(f"score regressed from {expected_score:.12g} to {actual_score:.12g} ({direction})")
    return problems


def _pipeline_name(payload: Mapping[str, object]) -> str:
    for key in ("chosen_pipeline", "selected_pipeline", "selected_profile", "pipeline"):
        value = _text(payload.get(key))
        if value:
            return value
    selected = payload.get("selected")
    if isinstance(selected, Mapping):
        return _text(selected.get("name") or selected.get("pipeline"))
    return ""


def _metric_name(payload: Mapping[str, object]) -> str:
    return _text(payload.get("metric") or payload.get("metric_name"))


def _score(payload: Mapping[str, object]) -> float | None:
    for key in _SCORE_KEYS:
        value = payload.get(key)
        try:
            parsed = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return parsed
    return None


def _finite_float(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _string_set(value: object) -> set[str]:
    return set(_string_values(value))


def _string_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Mapping):
        return [f"{key}:{item}" for key, item in sorted(value.items()) if _text(item)]
    if not isinstance(value, (list, tuple, set, frozenset)):
        return []
    return [text for item in value if (text := _text(item))]


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value[:2_000] for value in values if value))[:500]


def _requires_gpu_execution(metrics: Mapping[str, object]) -> bool:
    for key in ("requires_gpu", "gpu_required", "requires_gpu_execution"):
        value = metrics.get(key)
        if value is True or _text(value).lower() in {"1", "true", "yes", "on"}:
            return True
    return False


def _path_size(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    try:
        if path.is_file():
            return path.stat().st_size
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    except OSError:
        return None


def _nested_value(payload: Mapping[str, object] | None, *keys: str) -> object:
    value: object = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _nested_mapping(payload: Mapping[str, object] | None, *keys: str) -> Mapping[str, object] | None:
    value = _nested_value(payload, *keys)
    return value if isinstance(value, Mapping) else None


def _nested_list(payload: Mapping[str, object] | None, *keys: str) -> list[object]:
    value = _nested_value(payload, *keys)
    return value if isinstance(value, list) else []


def _nested_text(payload: Mapping[str, object] | None, *keys: str) -> str:
    return _text(_nested_value(payload, *keys))


def _nested_int(payload: Mapping[str, object] | None, *keys: str) -> int | None:
    return _nonnegative_int(_nested_value(payload, *keys))


def _nonnegative_int(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _safe_child(root: Path, relative: str) -> Path | None:
    if not relative or Path(relative).is_absolute():
        return None
    try:
        root_resolved = root.resolve()
        candidate = (root / relative).resolve()
    except OSError:
        return None
    return candidate if candidate.is_relative_to(root_resolved) else None


def _downloaded_error_transcripts(root: Path | None, *, max_files: int = 500) -> list[Path]:
    if root is None or not root.is_dir():
        return []
    paths: list[Path] = []
    try:
        candidates = root.rglob("*")
        for candidate in candidates:
            if len(paths) >= max_files:
                break
            if not candidate.is_file() or candidate.is_symlink() or not _ERROR_NAME_RE.search(candidate.name):
                continue
            try:
                raw = candidate.read_bytes()[: 2 * 1024 * 1024]
            except OSError:
                continue
            if raw.decode("utf-8", errors="replace").strip():
                paths.append(candidate)
    except OSError:
        return paths
    return paths


def _text(value: object) -> str:
    return str(value or "").strip()
