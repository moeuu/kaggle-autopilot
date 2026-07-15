from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from kagglebot.exceptions import KernelFailedError, SubmissionCliError, SubmissionValidationError
from kagglebot.hashing import sha256_file_or_none, sha256_text
from kagglebot.json_utils import load_json_object, write_json_object
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
from kagglebot.submission_semantics import runtime_tabular_fidelity_findings

REPORT_FILE_NAME = "submission_fidelity_report.json"

_CODE_OUTPUT_MODES = {"gateway", "inference"}
_SCORE_KEYS = ("selected_cv_mean", "offline_value", "primary_score", "score", "value")
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
            "requires_gpu": _requires_gpu_execution(metrics),
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
        for finding in runtime_tabular_fidelity_findings(dict(tabular)):
            add(str(finding["code"]), str(finding["message"]))
        _compare_tabular_input_contract(runtime, tabular, add=add)

    attempt_fingerprint = sha256_text(
        "\0".join((expected_package, str(contract_sha or ""), str(selected_output_sha or "")))
    )
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
        "schema_version": 1,
        "verdict": "fail" if reasons else "pass",
        "reason_codes": reason_codes,
        "reasons": sorted(reasons, key=lambda item: item["code"]),
        "competition": _text(expected.get("competition")) if expected else None,
        "run_id": run_id or (_text(expected.get("run_id")) if expected else None),
        "iteration": iteration if iteration is not None else expected.get("iteration") if expected else None,
        "kernel_id": kernel_id or _nested_text(expected, "kernel", "id"),
        "kernel_version": kernel_version,
        "artifact_mode": artifact_mode,
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
    for key in ("chosen_pipeline", "selected_pipeline", "pipeline"):
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
