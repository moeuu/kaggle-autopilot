from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedSubmitErrorClassification:
    kind: str
    reason: str
    retry_after_seconds: float


@dataclass(frozen=True)
class SubmitErrorClassificationResult:
    classification: dict[str, object]
    stderr: str
    normalized: NormalizedSubmitErrorClassification


def classify_submit_error_with_output_fallback(
    *,
    stdout: str,
    stderr: str,
    output: str,
    exit_code: int | None,
    classify_submit_error: Callable[[str, str, int | None], dict[str, object]],
    default_kind: str = "unknown",
    default_reason: str = "unclassified_submit_error",
    default_retry_after_seconds: float = 0.0,
) -> SubmitErrorClassificationResult:
    classification_stderr = stderr or ""
    classification = classify_submit_error(stdout, classification_stderr, exit_code)
    normalized = normalize_submit_error_classification(
        classification,
        default_kind=default_kind,
        default_reason=default_reason,
        default_retry_after_seconds=default_retry_after_seconds,
    )
    if normalized.reason == default_reason and output:
        classification_stderr = "\n".join(part for part in [classification_stderr, output] if part)
        classification = classify_submit_error(stdout, classification_stderr, exit_code)
        normalized = normalize_submit_error_classification(
            classification,
            default_kind=default_kind,
            default_reason=default_reason,
            default_retry_after_seconds=default_retry_after_seconds,
        )
    return SubmitErrorClassificationResult(
        classification=classification,
        stderr=classification_stderr,
        normalized=normalized,
    )


def normalize_submit_error_classification(
    classification: Mapping[str, object] | None,
    *,
    default_kind: str = "unknown",
    default_reason: str = "unclassified_submit_error",
    default_retry_after_seconds: float = 0.0,
) -> NormalizedSubmitErrorClassification:
    payload = classification or {}
    return NormalizedSubmitErrorClassification(
        kind=normalize_submit_error_text(payload.get("kind"), default=default_kind),
        reason=normalize_submit_error_text(payload.get("reason"), default=default_reason),
        retry_after_seconds=normalize_retry_after_seconds(
            payload.get("retry_after_seconds"),
            default=default_retry_after_seconds,
        ),
    )


def normalize_submit_error_text(value: object, *, default: str) -> str:
    text = str(value or "").strip()
    return text or default


def normalize_retry_after_seconds(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _finite_non_negative(default)
    seconds = float(value)
    if not math.isfinite(seconds):
        return _finite_non_negative(default)
    return max(seconds, 0.0)


def _finite_non_negative(value: float) -> float:
    seconds = float(value)
    if not math.isfinite(seconds):
        return 0.0
    return max(seconds, 0.0)
