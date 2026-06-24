from __future__ import annotations

from pathlib import Path

from kagglebot.submit_notebook import (
    build_kaggle_submit_kernel_kwargs,
    build_notebook_submit_reference,
    decide_ambiguous_notebook_submit_retry,
    decide_notebook_submit_artifact_mode,
    decide_submit_kernel_cpu_fallback,
    is_submit_kernel_push_error_text,
    normalize_notebook_submit_artifact_mode,
)


def test_normalize_notebook_submit_artifact_mode_defaults_to_wrapper() -> None:
    assert normalize_notebook_submit_artifact_mode(None) == "wrapper"
    assert normalize_notebook_submit_artifact_mode("") == "wrapper"
    assert normalize_notebook_submit_artifact_mode(" Inference ") == "inference"


def test_build_notebook_submit_reference_prefers_copied_artifact_path() -> None:
    reference = build_notebook_submit_reference(
        kernel_id="user/demo",
        submission_artifact_path=Path("/tmp/copied/submission-fixed.csv"),
        kernel_submission_path=Path("/kaggle/working/submission.csv"),
        version_label="7",
    )

    assert reference.kernel_ref == "user/demo"
    assert reference.submission_ref == "kernel:user/demo"
    assert reference.output_file == "submission-fixed.csv"
    assert reference.version == "7"


def test_build_notebook_submit_reference_uses_kernel_path_and_default_version() -> None:
    reference = build_notebook_submit_reference(
        kernel_id="user/demo",
        submission_artifact_path=None,
        kernel_submission_path=Path("/kaggle/working/submission.csv"),
        version_label=None,
    )

    assert reference.output_file == "submission.csv"
    assert reference.version == "1"


def test_build_kaggle_submit_kernel_kwargs_uses_reference_fields() -> None:
    reference = build_notebook_submit_reference(
        kernel_id="user/demo",
        submission_artifact_path=Path("/tmp/submission.csv"),
        kernel_submission_path=None,
        version_label="3",
    )

    assert build_kaggle_submit_kernel_kwargs(
        slug="demo-competition",
        reference=reference,
        message="submit message",
        dry_run=True,
    ) == {
        "slug": "demo-competition",
        "kernel": "user/demo",
        "message": "submit message",
        "output_file": "submission.csv",
        "version": "3",
        "dry_run": True,
    }


def test_decide_notebook_submit_artifact_mode_forces_inference_for_code_competition() -> None:
    decision = decide_notebook_submit_artifact_mode(
        requested_mode="wrapper",
        notebook_submit_required=True,
        code_competition=True,
        sample_data_rows=100,
        submission_data_rows=100,
    )

    assert decision.mode == "inference"
    assert decision.reason == "code_competition"


def test_decide_notebook_submit_artifact_mode_forces_inference_for_tiny_public_contract() -> None:
    decision = decide_notebook_submit_artifact_mode(
        requested_mode="wrapper",
        notebook_submit_required=True,
        code_competition=False,
        sample_data_rows=3,
        submission_data_rows=3,
    )

    assert decision.mode == "inference"
    assert decision.reason == "tiny_public_sample_notebook_contract"
    assert "hidden-test row mismatch" in decision.message


def test_decide_notebook_submit_artifact_mode_keeps_wrapper_for_regular_notebook_submit() -> None:
    decision = decide_notebook_submit_artifact_mode(
        requested_mode="wrapper",
        notebook_submit_required=True,
        code_competition=False,
        sample_data_rows=100,
        submission_data_rows=100,
    )

    assert decision.mode == "wrapper"
    assert decision.reason == ""


def test_decide_ambiguous_notebook_submit_retry_uses_output_fallback() -> None:
    calls: list[str] = []

    def classify(stdout: str, stderr: str, exit_code: int | None) -> dict[str, object]:  # noqa: ARG001
        calls.append(stderr)
        if "kernel must be specified" in stderr:
            return {"reason": "ambiguous_notebook_bad_request", "retry_after_seconds": 4}
        return {"reason": "unclassified_submit_error"}

    decision = decide_ambiguous_notebook_submit_retry(
        stdout="",
        stderr="",
        output="400 Client Error\nkernel must be specified as <owner>/<notebook>",
        exit_code=1,
        classify_submit_error=classify,
        should_retry_ambiguous=lambda *, reason, stdout, stderr: reason == "ambiguous_notebook_bad_request",
    )

    assert decision.retry is True
    assert decision.wait_seconds == 4.0
    assert "retrying same kernel submit in 4.0s" in decision.message
    assert calls == ["", "400 Client Error\nkernel must be specified as <owner>/<notebook>"]


def test_decide_ambiguous_notebook_submit_retry_rejects_generic_error() -> None:
    decision = decide_ambiguous_notebook_submit_retry(
        stdout="",
        stderr="generic bad request",
        output="",
        exit_code=1,
        classify_submit_error=lambda stdout, stderr, exit_code: {"reason": "bad_request"},
        should_retry_ambiguous=lambda *, reason, stdout, stderr: False,
    )

    assert decision.retry is False
    assert decision.wait_seconds == 0.0
    assert decision.message == ""
    assert decision.stderr == "generic bad request"


def test_decide_submit_kernel_cpu_fallback_allows_gpu_capacity_error() -> None:
    decision = decide_submit_kernel_cpu_fallback(
        accelerator="gpu",
        strict_accelerator=False,
        is_capacity_error=True,
        is_push_error=False,
    )

    assert decision.retry_on_cpu is True
    assert decision.reason == "Kaggle GPU capacity is unavailable"
    assert "retrying submit kernel on CPU" in decision.message


def test_decide_submit_kernel_cpu_fallback_allows_gpu_push_error() -> None:
    decision = decide_submit_kernel_cpu_fallback(
        accelerator="gpu",
        strict_accelerator=False,
        is_capacity_error=False,
        is_push_error=True,
    )

    assert decision.retry_on_cpu is True
    assert decision.reason == "Kaggle notebook push failed under GPU metadata"


def test_decide_submit_kernel_cpu_fallback_rejects_strict_or_non_gpu() -> None:
    strict = decide_submit_kernel_cpu_fallback(
        accelerator="gpu",
        strict_accelerator=True,
        is_capacity_error=True,
        is_push_error=True,
    )
    cpu = decide_submit_kernel_cpu_fallback(
        accelerator="cpu",
        strict_accelerator=False,
        is_capacity_error=True,
        is_push_error=True,
    )

    assert strict.retry_on_cpu is False
    assert strict.message == ""
    assert cpu.retry_on_cpu is False


def test_is_submit_kernel_push_error_text_detects_known_markers() -> None:
    assert is_submit_kernel_push_error_text(output="Kernel push error: Notebook not found") is True
    assert is_submit_kernel_push_error_text(stderr="kernel not found after push") is True
    assert is_submit_kernel_push_error_text(stdout="Kaggle kernel push failed") is True


def test_is_submit_kernel_push_error_text_rejects_generic_errors() -> None:
    assert is_submit_kernel_push_error_text(stderr="400 Client Error: Bad Request") is False
