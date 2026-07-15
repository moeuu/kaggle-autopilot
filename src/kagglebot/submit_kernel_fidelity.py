from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path

from kagglebot.exceptions import KernelFailedError, SubmissionCliError
from kagglebot.json_utils import load_json_object
from kagglebot.metric_matching import metrics_equivalent

_CODE_OUTPUT_MODES = {"gateway", "inference"}
_SCORE_KEYS = ("selected_cv_mean", "offline_value", "primary_score", "score", "value")
_MISSING_ASSET_KEYS = ("missing_dependencies", "missing_kernel_sources", "missing_model_sources")


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
) -> None:
    """Fail before Kaggle submission when the remote inference run degraded from selection."""
    if str(artifact_mode or "").strip().lower() not in _CODE_OUTPUT_MODES or not expected_metrics:
        return
    if actual_metrics_path is None:
        _raise_fidelity_error(["remote inference did not produce metrics.json"])
    actual_metrics = load_json_object(actual_metrics_path)
    if not actual_metrics:
        _raise_fidelity_error([f"remote inference metrics are missing or invalid: {actual_metrics_path}"])

    problems = _fidelity_problems(expected_metrics, actual_metrics)
    if problems:
        _raise_fidelity_error(problems)


def _raise_fidelity_error(problems: Iterable[str]) -> None:
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

    expected_score = _score(expected)
    actual_score = _score(actual)
    direction = _text(expected.get("direction") or actual.get("direction")).lower()
    if expected_score is not None and actual_score is None:
        problems.append(f"selected score {expected_score:.12g} was not reported")
    elif expected_score is not None and actual_score is not None and direction in {"maximize", "minimize"}:
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
    """Read the canonical metric across local and generated-kernel schemas."""
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
    if isinstance(value, str):
        return {value.strip()} if value.strip() else set()
    if not isinstance(value, (list, tuple, set, frozenset)):
        return set()
    return {text for item in value if (text := _text(item))}


def _text(value: object) -> str:
    return str(value or "").strip()
