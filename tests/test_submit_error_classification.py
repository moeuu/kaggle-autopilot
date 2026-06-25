from __future__ import annotations

from kagglebot.submit_error_classification import (
    classify_submit_error_with_output_fallback,
    normalize_retry_after_seconds,
    normalize_submit_error_classification,
)


def test_classify_submit_error_with_output_fallback_retries_blank_reason() -> None:
    calls: list[str] = []

    def classify(stdout: str, stderr: str, exit_code: int | None) -> dict[str, object]:  # noqa: ARG001
        calls.append(stderr)
        if "kernel must be specified" in stderr:
            return {
                "kind": "permanent",
                "reason": "notebook_submit_argument_missing",
                "retry_after_seconds": 7,
            }
        return {"kind": " ", "reason": " "}

    result = classify_submit_error_with_output_fallback(
        stdout="",
        stderr="",
        output="400 Client Error\nkernel must be specified",
        exit_code=1,
        classify_submit_error=classify,
    )

    assert calls == ["", "400 Client Error\nkernel must be specified"]
    assert result.stderr == "400 Client Error\nkernel must be specified"
    assert result.normalized.kind == "permanent"
    assert result.normalized.reason == "notebook_submit_argument_missing"
    assert result.normalized.retry_after_seconds == 7.0


def test_normalize_submit_error_classification_uses_defaults() -> None:
    normalized = normalize_submit_error_classification(
        {"kind": "", "reason": None, "retry_after_seconds": True},
        default_kind="unknown",
        default_reason="unclassified_submit_error",
        default_retry_after_seconds=3.0,
    )

    assert normalized.kind == "unknown"
    assert normalized.reason == "unclassified_submit_error"
    assert normalized.retry_after_seconds == 3.0


def test_normalize_retry_after_seconds_rejects_negative_and_non_finite() -> None:
    assert normalize_retry_after_seconds(-1) == 0.0
    assert normalize_retry_after_seconds(float("nan"), default=2.0) == 2.0
