from __future__ import annotations

from pathlib import Path

from kagglebot.submit_notebook import (
    build_kaggle_submit_kernel_kwargs,
    build_notebook_submit_output_reference,
    build_notebook_submit_reference,
    build_submit_kernel_run_kwargs,
    decide_ambiguous_notebook_submit_retry,
    decide_notebook_submit_artifact_mode,
    decide_notebook_submit_artifact_mode_for_paths,
    decide_submit_kernel_cpu_fallback,
    decide_submit_kernel_cpu_fallback_for_exception,
    infer_kernel_submit_version_label,
    is_submit_kernel_push_error,
    is_submit_kernel_push_error_text,
    normalize_notebook_submit_artifact_mode,
    resolve_notebook_submit_artifact_mode,
    run_kaggle_submit_kernel_with_retry,
    run_submit_kernel_with_cpu_fallback,
)


class SubmitKernelError(Exception):
    pass


class SubmitCliStubError(Exception):
    def __init__(
        self,
        *,
        stdout: str = "",
        stderr: str = "",
        output: str = "",
        exit_code: int | None = None,
    ) -> None:
        super().__init__(output or stderr or stdout)
        self.stdout = stdout
        self.stderr = stderr
        self.output = output
        self.exit_code = exit_code


def test_normalize_notebook_submit_artifact_mode_defaults_to_wrapper() -> None:
    assert normalize_notebook_submit_artifact_mode(None) == "wrapper"
    assert normalize_notebook_submit_artifact_mode("") == "wrapper"
    assert normalize_notebook_submit_artifact_mode(" Inference ") == "inference"


def test_resolve_notebook_submit_artifact_mode_uses_inference_for_code_competition_notebooks() -> None:
    assert (
        resolve_notebook_submit_artifact_mode(
            submit_mode="notebook",
            code_competition=True,
        )
        == "inference"
    )
    assert (
        resolve_notebook_submit_artifact_mode(
            submit_mode="kernel",
            code_competition=True,
        )
        == "inference"
    )
    assert (
        resolve_notebook_submit_artifact_mode(
            submit_mode="file",
            code_competition=True,
        )
        == "wrapper"
    )


def test_resolve_notebook_submit_artifact_mode_keeps_wrapper_for_regular_notebooks() -> None:
    assert (
        resolve_notebook_submit_artifact_mode(
            submit_mode="notebook",
            code_competition=False,
        )
        == "wrapper"
    )


def test_build_notebook_submit_reference_prefers_kernel_output_file_name() -> None:
    reference = build_notebook_submit_reference(
        kernel_id="user/demo",
        submission_artifact_path=Path("/tmp/copied/submission-fixed.csv"),
        kernel_submission_path=Path("/kaggle/working/submission.csv"),
        version_label="7",
    )

    assert reference.kernel_ref == "user/demo"
    assert reference.submission_ref == "kernel:user/demo"
    assert reference.output_file == "submission.csv"
    assert reference.version == "7"


def test_build_notebook_submit_reference_uses_copied_name_when_kernel_output_missing() -> None:
    reference = build_notebook_submit_reference(
        kernel_id="user/demo",
        submission_artifact_path=Path("/tmp/copied/submission-fixed.csv"),
        kernel_submission_path=None,
        version_label="7",
    )

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


def test_build_notebook_submit_output_reference_copies_kernel_submission(tmp_path: Path) -> None:
    copied_paths: list[Path] = []
    kernel_submission = Path("/kaggle/working/submission.csv")
    copied_submission = tmp_path / "submission.csv"

    output = build_notebook_submit_output_reference(
        kernel_id="user/demo",
        kernel_submission_path=kernel_submission,
        version_label="2",
        copy_submission_artifact=lambda source: copied_paths.append(source) or copied_submission,
    )

    assert copied_paths == [kernel_submission]
    assert output.submission_artifact_path == copied_submission
    assert output.reference.kernel_ref == "user/demo"
    assert output.reference.submission_ref == "kernel:user/demo"
    assert output.reference.output_file == "submission.csv"
    assert output.reference.version == "2"


def test_build_notebook_submit_output_reference_handles_missing_submission() -> None:
    output = build_notebook_submit_output_reference(
        kernel_id="user/demo",
        kernel_submission_path=None,
        version_label=None,
        copy_submission_artifact=lambda source: source,
    )

    assert output.submission_artifact_path is None
    assert output.reference.output_file == "submission.csv"
    assert output.reference.version == "1"


def test_infer_kernel_submit_version_label_from_push_logs(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "kernel_push-01.txt").write_text(
        "Kernel version 7 successfully pushed. check progress ...\n",
        encoding="utf-8",
    )
    assert infer_kernel_submit_version_label(logs_dir) == "7"


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


def test_build_submit_kernel_run_kwargs_normalizes_mode_and_preserves_fields(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.csv"
    kwargs = build_submit_kernel_run_kwargs(
        slug="demo",
        run_id="run-abcdef",
        iteration=3,
        base_dir=tmp_path,
        kaggle_username="user",
        kernel_name="kernel-name",
        accelerator="gpu",
        enable_internet=False,
        submission_path=submission_path,
        artifact_mode=" Inference ",
        dry_run=True,
        timeout_minutes=120,
    )

    assert kwargs == {
        "slug": "demo",
        "run_id": "run-abcdef",
        "iteration": 3,
        "base_dir": tmp_path,
        "kaggle_username": "user",
        "kernel_name": "kernel-name",
        "accelerator": "gpu",
        "enable_internet": False,
        "submission_path": submission_path,
        "mode": "inference",
        "dry_run": True,
        "timeout_minutes": 120,
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


def test_decide_notebook_submit_artifact_mode_for_paths_detects_tiny_contract(tmp_path: Path) -> None:
    sample_path = tmp_path / "context" / "sample_submission.csv"
    fallback_sample_path = tmp_path / "data" / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    fallback_sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text("id,target\n1,0\n2,0\n3,0\n", encoding="utf-8")
    fallback_sample_path.write_text("id,target\n1,0\n2,0\n3,0\n4,0\n", encoding="utf-8")
    submission_path.write_text("id,target\n1,0.1\n2,0.2\n3,0.3\n", encoding="utf-8")

    def count_rows(path: Path) -> int | None:
        return max(0, len(path.read_text(encoding="utf-8").splitlines()) - 1) if path.exists() else None

    decision = decide_notebook_submit_artifact_mode_for_paths(
        requested_mode="wrapper",
        notebook_submit_required=True,
        code_competition=False,
        sample_submission_path=sample_path,
        fallback_sample_submission_path=fallback_sample_path,
        submission_path=submission_path,
        count_csv_data_rows=count_rows,
    )

    assert decision.mode == "inference"
    assert decision.reason == "tiny_public_sample_notebook_contract"


def test_decide_notebook_submit_artifact_mode_forces_inference_for_empty_tiny_artifact() -> None:
    decision = decide_notebook_submit_artifact_mode(
        requested_mode="wrapper",
        notebook_submit_required=True,
        code_competition=False,
        sample_data_rows=3,
        submission_data_rows=0,
    )

    assert decision.mode == "inference"
    assert decision.reason == "tiny_public_sample_notebook_contract"


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


def test_decide_ambiguous_notebook_submit_retry_normalizes_retry_after() -> None:
    decision = decide_ambiguous_notebook_submit_retry(
        stdout="",
        stderr="kernel must be specified",
        output="",
        exit_code=1,
        classify_submit_error=lambda stdout, stderr, exit_code: {
            "reason": "ambiguous_notebook_bad_request",
            "retry_after_seconds": True,
        },
        should_retry_ambiguous=lambda *, reason, stdout, stderr: reason == "ambiguous_notebook_bad_request",
    )

    assert decision.retry is True
    assert decision.wait_seconds == 3.0


def test_decide_ambiguous_notebook_submit_retry_clamps_negative_retry_after() -> None:
    decision = decide_ambiguous_notebook_submit_retry(
        stdout="",
        stderr="kernel must be specified",
        output="",
        exit_code=1,
        classify_submit_error=lambda stdout, stderr, exit_code: {
            "reason": "ambiguous_notebook_bad_request",
            "retry_after_seconds": -4,
        },
        should_retry_ambiguous=lambda *, reason, stdout, stderr: reason == "ambiguous_notebook_bad_request",
    )

    assert decision.retry is True
    assert decision.wait_seconds == 0.0


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


def test_run_submit_kernel_with_cpu_fallback_retries_on_decision() -> None:
    calls: list[str] = []
    messages: list[str] = []

    def run_submit_kernel(**kwargs):  # noqa: ANN003
        calls.append(str(kwargs["accelerator"]))
        if kwargs["accelerator"] == "gpu":
            raise SubmitKernelError("gpu unavailable")
        return "kernel-result"

    result = run_submit_kernel_with_cpu_fallback(
        submit_kernel_kwargs={"accelerator": "gpu"},
        run_submit_kernel=run_submit_kernel,
        decide_cpu_fallback=lambda exc: decide_submit_kernel_cpu_fallback(
            accelerator="gpu",
            strict_accelerator=False,
            is_capacity_error=True,
            is_push_error=False,
        ),
        is_capacity_error=lambda exc: False,
        wrap_error=lambda exc: SubmitKernelError(f"wrapped: {exc}"),
        on_message=messages.append,
    )

    assert result == "kernel-result"
    assert calls == ["gpu", "cpu"]
    assert "retrying submit kernel on CPU" in messages[0]


def test_run_submit_kernel_with_cpu_fallback_wraps_retry_failure() -> None:
    def run_submit_kernel(**kwargs):  # noqa: ANN003, ARG001
        raise SubmitKernelError("still failed")

    try:
        run_submit_kernel_with_cpu_fallback(
            submit_kernel_kwargs={"accelerator": "gpu"},
            run_submit_kernel=run_submit_kernel,
            decide_cpu_fallback=lambda exc: decide_submit_kernel_cpu_fallback(
                accelerator="gpu",
                strict_accelerator=False,
                is_capacity_error=True,
                is_push_error=False,
            ),
            is_capacity_error=lambda exc: False,
            wrap_error=lambda exc: SubmitKernelError(f"wrapped: {exc}"),
            on_message=lambda message: None,
        )
    except SubmitKernelError as exc:
        assert str(exc) == "wrapped: still failed"
        assert isinstance(exc.__cause__, SubmitKernelError)
    else:  # pragma: no cover
        raise AssertionError("expected wrapped retry failure")


def test_run_kaggle_submit_kernel_with_retry_retries_ambiguous_error() -> None:
    calls = 0
    sleeps: list[float] = []
    messages: list[str] = []

    def run_submit(**kwargs):  # noqa: ANN003, ARG001
        nonlocal calls
        calls += 1
        if calls == 1:
            raise SubmitCliStubError(output="400 Client Error\nkernel must be specified", exit_code=1)
        return "submit-result"

    result = run_kaggle_submit_kernel_with_retry(
        submit_kwargs={"kernel": "user/demo"},
        run_kaggle_submit_kernel=run_submit,
        submit_error_types=SubmitCliStubError,
        classify_submit_error=lambda stdout, stderr, exit_code: (
            {"reason": "ambiguous_notebook_bad_request", "retry_after_seconds": 4}
            if "kernel must be specified" in stderr
            else {"reason": "unclassified_submit_error"}
        ),
        should_retry_ambiguous=lambda *, reason, stdout, stderr: reason == "ambiguous_notebook_bad_request",
        sleep=sleeps.append,
        on_message=messages.append,
    )

    assert result == "submit-result"
    assert calls == 2
    assert sleeps == [4.0]
    assert "retrying same kernel submit" in messages[0]


def test_run_kaggle_submit_kernel_with_retry_reraises_generic_error() -> None:
    error = SubmitCliStubError(stderr="generic bad request", exit_code=1)

    def run_submit(**kwargs):  # noqa: ANN003, ARG001
        raise error

    try:
        run_kaggle_submit_kernel_with_retry(
            submit_kwargs={"kernel": "user/demo"},
            run_kaggle_submit_kernel=run_submit,
            submit_error_types=SubmitCliStubError,
            classify_submit_error=lambda stdout, stderr, exit_code: {"reason": "bad_request"},
            should_retry_ambiguous=lambda *, reason, stdout, stderr: False,
            sleep=lambda seconds: None,
            on_message=lambda message: None,
        )
    except SubmitCliStubError as exc:
        assert exc is error
    else:  # pragma: no cover
        raise AssertionError("expected generic submit error")


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


def test_decide_submit_kernel_cpu_fallback_for_exception_uses_predicates() -> None:
    exc = SubmitKernelError("capacity")

    decision = decide_submit_kernel_cpu_fallback_for_exception(
        accelerator="gpu",
        strict_accelerator=False,
        exc=exc,
        is_capacity_error=lambda candidate: candidate is exc,
        is_push_error=lambda candidate: False,
    )

    assert decision.retry_on_cpu is True
    assert decision.reason == "Kaggle GPU capacity is unavailable"


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


def test_is_submit_kernel_push_error_reads_exception_fields() -> None:
    exc = SubmitCliStubError(output="Kernel push error: Notebook not found")

    assert is_submit_kernel_push_error(exc) is True


def test_is_submit_kernel_push_error_text_rejects_generic_errors() -> None:
    assert is_submit_kernel_push_error_text(stderr="400 Client Error: Bad Request") is False
