from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedSubmitErrorClassification:
    kind: str
    reason: str
    retry_after_seconds: float


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
