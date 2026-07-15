from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kagglebot.exceptions import (
    KaggleCliError,
    KernelCapacityError,
    SubmissionCliError,
    SubmissionValidationError,
)
from kagglebot.kernel_runtime.submit_runtime_fidelity import record_runtime_fidelity
from kagglebot.submit_kernel_fidelity import stage_submit_fidelity_expected_contract
from kagglebot.submit_notebook import (
    NotebookSubmitRunner,
    build_kaggle_submit_kernel_kwargs,
    build_notebook_submit_output_reference,
    build_notebook_submit_reference,
    build_notebook_submit_runner_for_run,
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
    notebook_kernel_submission_error,
    resolve_notebook_submit_artifact_mode,
    run_kaggle_submit_kernel_with_retry,
    run_notebook_kernel_submission,
    run_notebook_kernel_submission_for_run,
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


def test_build_notebook_submit_runner_for_run_wires_standard_error_detectors(tmp_path: Path) -> None:
    class Paths:
        base_dir = tmp_path / "artifacts" / "demo"

        @staticmethod
        def iter_dir(run_id: str, iteration: int) -> Path:
            return tmp_path / "artifacts" / "demo" / "runs" / run_id / f"iter-{iteration}"

    runner = build_notebook_submit_runner_for_run(
        slug="demo",
        run_id="run-1",
        paths=Paths(),
        kaggle_username="user",
        kernel_name="submit-kernel",
        accelerator="gpu",
        strict_accelerator=False,
        dry_run=True,
        timeout_minutes=30,
        infer_iteration_from_submission_path=lambda path: 1,
        resolve_kaggle_username=lambda value: str(value or ""),
        run_submit_kernel=lambda **kwargs: object(),
        run_kaggle_submit_kernel=lambda **kwargs: object(),
        copy_submission_artifact_to_iteration_dir=lambda **kwargs: tmp_path / "submission.csv",
        classify_submit_error=lambda stdout, stderr, exit_code: {},
        should_retry_ambiguous=lambda **kwargs: False,
        sleep=lambda seconds: None,
        on_message=lambda message: None,
    )

    assert isinstance(runner, NotebookSubmitRunner)
    assert runner.is_capacity_error(KernelCapacityError("capacity"))
    assert runner.is_push_error(KaggleCliError("Kernel push error: failed", command=[]))


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


def test_build_notebook_submit_output_reference_uses_expected_output_file_when_missing_submission() -> None:
    output = build_notebook_submit_output_reference(
        kernel_id="user/demo",
        kernel_submission_path=None,
        version_label=None,
        copy_submission_artifact=lambda source: source,
        expected_output_file="submission.csv.gz",
    )

    assert output.submission_artifact_path is None
    assert output.reference.output_file == "submission.csv.gz"
    assert output.reference.version == "1"


@pytest.mark.parametrize(
    "expected_output_file",
    [
        "predictions.vcf.gz",
        "volume.mrc",
        "elevation.hgt",
        "signals.vhdr",
        "graph.jsonld",
        "model.engine",
        "model.rknn",
        "model.dlc",
    ],
)
def test_build_notebook_submit_output_reference_uses_non_csv_expected_output_file(
    expected_output_file: str,
) -> None:
    output = build_notebook_submit_output_reference(
        kernel_id="user/demo",
        kernel_submission_path=None,
        version_label=None,
        copy_submission_artifact=lambda source: source,
        expected_output_file=expected_output_file,
    )

    assert output.submission_artifact_path is None
    assert output.reference.output_file == expected_output_file
    assert output.reference.version == "1"


def test_build_notebook_submit_output_reference_normalizes_template_expected_output_file() -> None:
    output = build_notebook_submit_output_reference(
        kernel_id="user/demo",
        kernel_submission_path=None,
        version_label=None,
        copy_submission_artifact=lambda source: source,
        expected_output_file="sample_submission.jsonlines.zst",
    )

    assert output.submission_artifact_path is None
    assert output.reference.output_file == "submission.jsonlines.zst"
    assert output.reference.version == "1"


def test_notebook_submit_runner_falls_back_to_source_submission_filename(monkeypatch, tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.onnx"
    submission_path.write_bytes(b"model")
    captured: dict[str, str | None] = {}

    def fake_run_notebook_kernel_submission_for_run(**kwargs):  # noqa: ANN001
        captured["expected_output_file"] = kwargs["expected_output_file"]
        return object(), "kernel:user/demo", None

    monkeypatch.setattr(
        "kagglebot.submit_notebook.run_notebook_kernel_submission_for_run",
        fake_run_notebook_kernel_submission_for_run,
    )
    runner = NotebookSubmitRunner(
        slug="demo",
        run_id="run-1",
        paths=object(),  # type: ignore[arg-type]
        kaggle_username=None,
        kernel_name=None,
        accelerator="cpu",
        strict_accelerator=False,
        dry_run=False,
        timeout_minutes=None,
        infer_iteration_from_submission_path=lambda _path: 1,
        resolve_kaggle_username=lambda _username: "user",
        run_submit_kernel=lambda **_kwargs: None,
        run_kaggle_submit_kernel=lambda **_kwargs: None,
        copy_submission_artifact_to_iteration_dir=lambda **_kwargs: submission_path,
        classify_submit_error=lambda *_args: {},
        should_retry_ambiguous=lambda **_kwargs: False,
        sleep=lambda _seconds: None,
        on_message=lambda _message: None,
        is_capacity_error=lambda _exc: False,
        is_push_error=lambda _exc: False,
    )

    runner.submit(submission_path=submission_path, message="submit", artifact_mode="wrapper")

    assert captured["expected_output_file"] == "submission.onnx"


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


def test_run_notebook_kernel_submission_for_run_resolves_iteration_and_paths(tmp_path: Path) -> None:
    calls: dict[str, object] = {}
    submission_path = tmp_path / "runs" / "run-1" / "iter-4" / "submission.csv"
    submission_path.parent.mkdir(parents=True)
    submission_path.write_text("id,target\n1,0.1\n", encoding="utf-8")
    kernel_submission_path = Path("/kaggle/working/submission.csv")
    copied_submission_path = tmp_path / "copied" / "submission.csv"
    submit_result = object()

    class Paths:
        base_dir = tmp_path / "artifacts" / "demo"

        @staticmethod
        def iter_dir(run_id: str, iteration: int) -> Path:
            return tmp_path / "artifacts" / "demo" / "runs" / run_id / f"iter-{iteration}"

    def fake_run_submit_kernel(**kwargs: object) -> SimpleNamespace:
        calls["kernel_kwargs"] = kwargs
        return SimpleNamespace(kernel_id="user/demo-kernel", submission_path=kernel_submission_path)

    def fake_submit_kernel(**kwargs: object) -> object:
        calls["submit_kwargs"] = kwargs
        return submit_result

    def fake_copy_submission_artifact(*, source: Path, iter_dir: Path) -> Path:
        calls["copy_source"] = source
        calls["copy_iter_dir"] = iter_dir
        return copied_submission_path

    result, reference, artifact_path = run_notebook_kernel_submission_for_run(
        slug="demo",
        run_id="run-1",
        paths=Paths(),
        kaggle_username="raw-user",
        kernel_name="submit-kernel",
        accelerator="gpu",
        strict_accelerator=False,
        submission_path=submission_path,
        message="submit message",
        artifact_mode="wrapper",
        dry_run=True,
        timeout_minutes=60,
        infer_iteration_from_submission_path=lambda path: 4 if path == submission_path else None,
        resolve_kaggle_username=lambda value: f"resolved-{value}",
        run_submit_kernel=fake_run_submit_kernel,
        run_kaggle_submit_kernel=fake_submit_kernel,
        copy_submission_artifact_to_iteration_dir=fake_copy_submission_artifact,
        classify_submit_error=lambda stdout, stderr, exit_code: {},
        should_retry_ambiguous=lambda **kwargs: False,
        sleep=lambda seconds: None,
        on_message=lambda message: calls.setdefault("message", message),
        is_capacity_error=lambda exc: False,
        is_push_error=lambda exc: False,
    )

    assert result is submit_result
    assert reference == "kernel:user/demo-kernel"
    assert artifact_path == copied_submission_path
    assert calls["kernel_kwargs"]["iteration"] == 4
    assert calls["kernel_kwargs"]["base_dir"] == tmp_path / "artifacts"
    assert calls["kernel_kwargs"]["kaggle_username"] == "resolved-raw-user"
    assert calls["kernel_kwargs"]["mode"] == "wrapper"
    assert calls["copy_source"] == kernel_submission_path
    assert calls["copy_iter_dir"] == tmp_path / "artifacts" / "demo" / "runs" / "run-1" / "iter-4"
    assert calls["submit_kwargs"]["kernel"] == "user/demo-kernel"
    assert calls["submit_kwargs"]["output_file"] == "submission.csv"


def test_notebook_submit_runner_binds_run_callbacks(tmp_path: Path) -> None:
    calls: dict[str, object] = {}
    submission_path = tmp_path / "runs" / "run-1" / "iter-3" / "submission.csv"
    submission_path.parent.mkdir(parents=True)
    submission_path.write_text("id,target\n1,0.1\n", encoding="utf-8")
    kernel_submission_path = Path("/kaggle/working/submission.csv")
    copied_submission_path = tmp_path / "copied" / "submission.csv"
    submit_result = object()

    class Paths:
        base_dir = tmp_path / "artifacts" / "demo"

        @staticmethod
        def iter_dir(run_id: str, iteration: int) -> Path:
            return tmp_path / "artifacts" / "demo" / "runs" / run_id / f"iter-{iteration}"

    def fake_run_submit_kernel(**kwargs: object) -> SimpleNamespace:
        calls["kernel_kwargs"] = kwargs
        return SimpleNamespace(kernel_id="user/demo-kernel", submission_path=kernel_submission_path)

    def fake_submit_kernel(**kwargs: object) -> object:
        calls["submit_kwargs"] = kwargs
        return submit_result

    def fake_copy_submission_artifact(*, source: Path, iter_dir: Path) -> Path:
        calls["copy_source"] = source
        calls["copy_iter_dir"] = iter_dir
        return copied_submission_path

    runner = NotebookSubmitRunner(
        slug="demo",
        run_id="run-1",
        paths=Paths(),
        kaggle_username="raw-user",
        kernel_name="submit-kernel",
        accelerator="gpu",
        strict_accelerator=False,
        dry_run=True,
        timeout_minutes=60,
        infer_iteration_from_submission_path=lambda path: 3 if path == submission_path else None,
        resolve_kaggle_username=lambda value: f"resolved-{value}",
        run_submit_kernel=fake_run_submit_kernel,
        run_kaggle_submit_kernel=fake_submit_kernel,
        copy_submission_artifact_to_iteration_dir=fake_copy_submission_artifact,
        classify_submit_error=lambda stdout, stderr, exit_code: {},
        should_retry_ambiguous=lambda **kwargs: False,
        sleep=lambda seconds: None,
        on_message=lambda message: calls.setdefault("message", message),
        is_capacity_error=lambda exc: False,
        is_push_error=lambda exc: False,
    )

    result, reference, artifact_path = runner.submit(
        submission_path=submission_path,
        message="submit message",
        artifact_mode="wrapper",
    )

    assert result is submit_result
    assert reference == "kernel:user/demo-kernel"
    assert artifact_path == copied_submission_path
    assert calls["kernel_kwargs"]["iteration"] == 3
    assert calls["kernel_kwargs"]["base_dir"] == tmp_path / "artifacts"
    assert calls["kernel_kwargs"]["kaggle_username"] == "resolved-raw-user"
    assert calls["copy_iter_dir"] == tmp_path / "artifacts" / "demo" / "runs" / "run-1" / "iter-3"
    assert calls["submit_kwargs"]["kernel"] == "user/demo-kernel"


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
        count_tabular_data_rows=count_rows,
    )

    assert decision.mode == "inference"
    assert decision.reason == "tiny_public_sample_notebook_contract"


def test_decide_notebook_submit_artifact_mode_for_paths_detects_jsonl_tiny_contract(tmp_path: Path) -> None:
    from kagglebot.context_artifacts import count_tabular_data_rows_capped

    sample_path = tmp_path / "context" / "sample_submission.jsonl"
    fallback_sample_path = tmp_path / "data" / "sample_submission.jsonl"
    submission_path = tmp_path / "submission.jsonl"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    fallback_sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(
        '{"id":1,"target":0}\n{"id":2,"target":0}\n{"id":3,"target":0}\n',
        encoding="utf-8",
    )
    fallback_sample_path.write_text(
        '{"id":1,"target":0}\n{"id":2,"target":0}\n{"id":3,"target":0}\n{"id":4,"target":0}\n',
        encoding="utf-8",
    )
    submission_path.write_text(
        '{"id":1,"target":0.1}\n{"id":2,"target":0.2}\n{"id":3,"target":0.3}\n',
        encoding="utf-8",
    )

    decision = decide_notebook_submit_artifact_mode_for_paths(
        requested_mode="wrapper",
        notebook_submit_required=True,
        code_competition=False,
        sample_submission_path=sample_path,
        fallback_sample_submission_path=fallback_sample_path,
        submission_path=submission_path,
        count_tabular_data_rows=count_tabular_data_rows_capped,
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


def test_run_notebook_kernel_submission_runs_kernel_and_submits_reference(tmp_path: Path) -> None:
    submission_path = tmp_path / "iter-2" / "submission.csv"
    submission_path.parent.mkdir(parents=True)
    submission_path.write_text("id,target\n1,0.1\n", encoding="utf-8")
    logs_dir = tmp_path / "iter-2" / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "kernel_push-001.txt").write_text("Kernel version 8 successfully pushed.\n", encoding="utf-8")

    captured_kernel_kwargs: dict[str, object] = {}
    captured_submit_kwargs: dict[str, object] = {}
    messages: list[str] = []
    copied_submission = tmp_path / "copied" / "submission.csv"

    def run_submit_kernel(**kwargs):  # noqa: ANN003
        captured_kernel_kwargs.update(kwargs)
        kernel_submission = tmp_path / "kernel-output" / "submission.csv"
        kernel_submission.parent.mkdir(parents=True)
        kernel_submission.write_text("id,target\n1,0.2\n", encoding="utf-8")
        return type(
            "KernelResult",
            (),
            {"kernel_id": "user/demo-submit", "submission_path": kernel_submission},
        )()

    def copy_submission(source: Path) -> Path:
        copied_submission.parent.mkdir(parents=True)
        copied_submission.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        return copied_submission

    def run_kaggle_submit_kernel(**kwargs):  # noqa: ANN003
        captured_submit_kwargs.update(kwargs)
        return type("SubmitResult", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    result, submission_ref, artifact_path = run_notebook_kernel_submission(
        slug="demo",
        run_id="run-1",
        iteration=2,
        iter_logs_dir=logs_dir,
        base_dir=tmp_path,
        kaggle_username="user",
        kernel_name="submit-kernel",
        accelerator="gpu",
        strict_accelerator=False,
        submission_path=submission_path,
        message="submit message",
        artifact_mode="inference",
        dry_run=False,
        timeout_minutes=60,
        run_submit_kernel=run_submit_kernel,
        run_kaggle_submit_kernel=run_kaggle_submit_kernel,
        copy_submission_artifact=copy_submission,
        classify_submit_error=lambda stdout, stderr, exit_code: {"reason": "unclassified_submit_error"},
        should_retry_ambiguous=lambda *, reason, stdout, stderr: False,
        sleep=lambda seconds: None,
        on_message=messages.append,
        is_capacity_error=lambda exc: False,
        is_push_error=lambda exc: False,
    )

    assert result.returncode == 0
    assert submission_ref == "kernel:user/demo-submit"
    assert artifact_path == copied_submission
    assert captured_kernel_kwargs["enable_internet"] is False
    assert captured_kernel_kwargs["mode"] == "inference"
    assert captured_submit_kwargs == {
        "slug": "demo",
        "kernel": "user/demo-submit",
        "message": "submit message",
        "output_file": "submission.csv",
        "version": "8",
        "dry_run": False,
    }
    assert messages == ["[cyan]submit notebook[/cyan]: user/demo-submit"]


def test_code_submission_runs_review_guard_executor_and_ledger_in_order(tmp_path: Path) -> None:
    submission_path = tmp_path / "iter-1" / "submission.json"
    submission_path.parent.mkdir(parents=True)
    submission_path.write_text("{}\n", encoding="utf-8")
    logs_dir = submission_path.parent / "logs"
    logs_dir.mkdir()
    (logs_dir / "kernel_push-001.txt").write_text("Kernel version 3 successfully pushed.\n", encoding="utf-8")
    output_dir = tmp_path / "kernel-output"
    output_dir.mkdir()
    remote_submission = output_dir / "submission.json"
    remote_submission.write_text('{"task": []}\n', encoding="utf-8")
    events: list[str] = []
    approval = object()
    permit = object()

    def review(**kwargs):  # noqa: ANN003
        events.append("review")
        assert kwargs["submission_path"] == remote_submission
        assert kwargs["kernel_version"] == "3"
        return approval

    def guard(**kwargs):  # noqa: ANN003
        events.append("guard")
        assert kwargs["approval"] is approval
        return permit

    def submit(**_kwargs):
        events.append("executor")
        return SimpleNamespace(returncode=0)

    def record(**kwargs):  # noqa: ANN003
        events.append("ledger")
        assert kwargs["permit"] is permit

    run_notebook_kernel_submission(
        slug="demo",
        run_id="run-1",
        iteration=1,
        iter_logs_dir=logs_dir,
        base_dir=tmp_path,
        kaggle_username="user",
        kernel_name=None,
        accelerator="gpu",
        strict_accelerator=False,
        submission_path=submission_path,
        message="submit",
        artifact_mode="gateway",
        dry_run=False,
        timeout_minutes=60,
        run_submit_kernel=lambda **_kwargs: SimpleNamespace(
            kernel_id="user/demo-submit",
            output_dir=output_dir,
            submission_path=remote_submission,
            metrics_path=None,
        ),
        run_kaggle_submit_kernel=submit,
        copy_submission_artifact=lambda source: source,
        classify_submit_error=lambda *_args: {},
        should_retry_ambiguous=lambda **_kwargs: False,
        sleep=lambda _seconds: None,
        on_message=lambda _message: None,
        is_capacity_error=lambda _exc: False,
        is_push_error=lambda _exc: False,
        expected_output_file="submission.json",
        review_code_submission=review,
        recheck_code_submission_guard=guard,
        record_code_submission_execution=record,
    )

    assert events == ["review", "guard", "executor", "ledger"]


def test_fresh_code_submission_validates_attestation_before_review_and_passes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slug = "demo"
    run_id = "run-1"
    iteration = 1
    submission_path = tmp_path / "iter-1" / "submission.csv"
    submission_path.parent.mkdir(parents=True)
    submission_path.write_text("id,target\n1,0.1\n2,0.9\n", encoding="utf-8")
    logs_dir = submission_path.parent / "logs"
    logs_dir.mkdir()
    (logs_dir / "kernel_push-001.txt").write_text("Kernel version 3 successfully pushed.\n", encoding="utf-8")
    package_dir = tmp_path / slug / "kernels" / run_id / "submit-iter-1"
    package_dir.mkdir(parents=True)
    (package_dir / "kernel.py").write_text("print('inference')\n", encoding="utf-8")
    (package_dir / "kernel-metadata.json").write_text(
        json.dumps({"model_sources": ["owner/model/1"]}),
        encoding="utf-8",
    )
    metrics = {
        "chosen_pipeline": "model",
        "metric": "accuracy",
        "direction": "maximize",
        "score_source": "cv",
        "score": 0.8,
        "active_model_source": "owner/model/1",
        "test_prediction_distribution": {"source_top10": [["model", 2]]},
    }
    expected_path = stage_submit_fidelity_expected_contract(
        package_dir=package_dir,
        slug=slug,
        run_id=run_id,
        iteration=iteration,
        kernel_id="user/demo-submit",
        artifact_mode="gateway",
        expected_output_file="submission.csv",
        expected_metrics=metrics,
        selected_candidate_path=submission_path,
        requested_accelerator="gpu",
        executed_accelerator="gpu",
        machine_shape="NvidiaTeslaT4",
        capacity_fallback_used=False,
    )
    output_dir = tmp_path / "remote-output"
    output_dir.mkdir()
    remote_submission = output_dir / "submission.csv"
    remote_submission.write_text("id,target\n2,0.8\n1,0.2\n", encoding="utf-8")
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "test.csv").write_text("id\n2\n1\n", encoding="utf-8")
    monkeypatch.setenv("KAGGLEBOT_FIDELITY_REQUESTED_ACCELERATOR", "gpu")
    monkeypatch.setenv("KAGGLEBOT_FIDELITY_EXECUTED_ACCELERATOR", "gpu")
    monkeypatch.setenv("KAGGLEBOT_FIDELITY_MACHINE_SHAPE", "NvidiaTeslaT4")
    record_runtime_fidelity(package_root=package_dir, output_root=output_dir, input_root=input_dir)
    runtime_path = output_dir / "submit_fidelity_runtime.json"
    events: list[str] = []
    approval = object()
    permit = object()

    def review(**kwargs):  # noqa: ANN003
        events.append("review")
        report_path = kwargs["fidelity_report_path"]
        assert json.loads(report_path.read_text(encoding="utf-8"))["verdict"] == "pass"
        return approval

    run_notebook_kernel_submission(
        slug=slug,
        run_id=run_id,
        iteration=iteration,
        iter_logs_dir=logs_dir,
        base_dir=tmp_path,
        kaggle_username="user",
        kernel_name=None,
        accelerator="gpu",
        strict_accelerator=False,
        submission_path=submission_path,
        message="submit",
        artifact_mode="gateway",
        dry_run=False,
        timeout_minutes=60,
        run_submit_kernel=lambda **_kwargs: SimpleNamespace(
            kernel_id="user/demo-submit",
            output_dir=output_dir,
            submission_path=remote_submission,
            metrics_path=metrics_path,
            fidelity_expected_path=expected_path,
            fidelity_runtime_path=runtime_path,
        ),
        run_kaggle_submit_kernel=lambda **_kwargs: events.append("executor"),
        copy_submission_artifact=lambda source: source,
        classify_submit_error=lambda *_args: {},
        should_retry_ambiguous=lambda **_kwargs: False,
        sleep=lambda _seconds: None,
        on_message=lambda _message: None,
        is_capacity_error=lambda _exc: False,
        is_push_error=lambda _exc: False,
        expected_output_file="submission.csv",
        expected_metrics_payload=metrics,
        review_code_submission=review,
        recheck_code_submission_guard=lambda **_kwargs: permit,
        record_code_submission_execution=lambda **_kwargs: events.append("ledger"),
    )

    assert events == ["review", "executor", "ledger"]


def test_code_submission_review_rejection_blocks_executor(tmp_path: Path) -> None:
    submission_path = tmp_path / "iter-1" / "submission.json"
    submission_path.parent.mkdir(parents=True)
    submission_path.write_text("{}\n", encoding="utf-8")
    logs_dir = submission_path.parent / "logs"
    logs_dir.mkdir()
    (logs_dir / "kernel_push-001.txt").write_text("Kernel version 3 successfully pushed.\n", encoding="utf-8")
    output_dir = tmp_path / "kernel-output"
    output_dir.mkdir()
    remote_submission = output_dir / "submission.json"
    remote_submission.write_text("{}\n", encoding="utf-8")
    cli_invoked = False

    def submit(**_kwargs):
        nonlocal cli_invoked
        cli_invoked = True

    with pytest.raises(SubmissionValidationError, match="rejected"):
        run_notebook_kernel_submission(
            slug="demo",
            run_id="run-1",
            iteration=1,
            iter_logs_dir=logs_dir,
            base_dir=tmp_path,
            kaggle_username="user",
            kernel_name=None,
            accelerator="gpu",
            strict_accelerator=False,
            submission_path=submission_path,
            message="submit",
            artifact_mode="gateway",
            dry_run=False,
            timeout_minutes=60,
            run_submit_kernel=lambda **_kwargs: SimpleNamespace(
                kernel_id="user/demo-submit",
                output_dir=output_dir,
                submission_path=remote_submission,
                metrics_path=None,
            ),
            run_kaggle_submit_kernel=submit,
            copy_submission_artifact=lambda source: source,
            classify_submit_error=lambda *_args: {},
            should_retry_ambiguous=lambda **_kwargs: False,
            sleep=lambda _seconds: None,
            on_message=lambda _message: None,
            is_capacity_error=lambda _exc: False,
            is_push_error=lambda _exc: False,
            expected_output_file="submission.json",
            review_code_submission=lambda **_kwargs: (_ for _ in ()).throw(SubmissionValidationError("rejected")),
        )

    assert cli_invoked is False


def test_run_notebook_kernel_submission_selects_exact_gateway_output_over_npy_diagnostics(
    tmp_path: Path,
) -> None:
    submission_path = tmp_path / "iter-2" / "test_array_mask.npy"
    submission_path.parent.mkdir(parents=True)
    submission_path.write_bytes(b"diagnostic input")
    logs_dir = tmp_path / "iter-2" / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "kernel_push-001.txt").write_text(
        "Kernel version 9 successfully pushed.\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "kernel-output"
    output_dir.mkdir()
    downloaded_mask = output_dir / "test_array_mask.npy"
    downloaded_mask.write_bytes(b"mask")
    downloaded_parquet = output_dir / "submission.parquet"
    downloaded_parquet.write_bytes(b"parquet")
    copied_sources: list[Path] = []
    captured_kernel_kwargs: dict[str, object] = {}
    captured_submit_kwargs: dict[str, object] = {}

    def run_submit_kernel(**kwargs):  # noqa: ANN003
        captured_kernel_kwargs.update(kwargs)
        return SimpleNamespace(
            kernel_id="user/arc-submit",
            output_dir=output_dir,
            submission_path=downloaded_mask,
        )

    def copy_submission(source: Path) -> Path:
        copied_sources.append(source)
        return tmp_path / "copied" / source.name

    def run_kaggle_submit_kernel(**kwargs):  # noqa: ANN003
        captured_submit_kwargs.update(kwargs)
        return SimpleNamespace(returncode=0)

    _result, submission_ref, artifact_path = run_notebook_kernel_submission(
        slug="arc-demo",
        run_id="run-1",
        iteration=2,
        iter_logs_dir=logs_dir,
        base_dir=tmp_path,
        kaggle_username="user",
        kernel_name=None,
        accelerator="gpu",
        strict_accelerator=False,
        submission_path=submission_path,
        message="submit",
        artifact_mode="gateway",
        dry_run=False,
        timeout_minutes=60,
        run_submit_kernel=run_submit_kernel,
        run_kaggle_submit_kernel=run_kaggle_submit_kernel,
        copy_submission_artifact=copy_submission,
        classify_submit_error=lambda *_args: {},
        should_retry_ambiguous=lambda **_kwargs: False,
        sleep=lambda _seconds: None,
        on_message=lambda _message: None,
        is_capacity_error=lambda _exc: False,
        is_push_error=lambda _exc: False,
        expected_output_file="submission.parquet",
    )

    assert submission_ref == "kernel:user/arc-submit"
    assert artifact_path == tmp_path / "copied" / "submission.parquet"
    assert copied_sources == [downloaded_parquet]
    assert captured_kernel_kwargs["mode"] == "gateway"
    assert captured_kernel_kwargs["expected_output_file"] == "submission.parquet"
    assert captured_submit_kwargs == {
        "slug": "arc-demo",
        "kernel": "user/arc-submit",
        "message": "submit",
        "output_file": "submission.parquet",
        "version": "9",
        "dry_run": False,
        "expected_output_file": "submission.parquet",
    }
    assert not any(".npy" in str(value) or "/data/" in str(value) for value in captured_submit_kwargs.values())


def test_run_notebook_kernel_submission_fails_before_cli_when_gateway_output_is_missing(
    tmp_path: Path,
) -> None:
    submission_path = tmp_path / "iter-2" / "test_array_mask.npy"
    submission_path.parent.mkdir(parents=True)
    submission_path.write_bytes(b"diagnostic")
    logs_dir = tmp_path / "iter-2" / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "kernel_push-001.txt").write_text(
        "Kernel version 4 successfully pushed.\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "kernel-output"
    output_dir.mkdir()
    mask = output_dir / "test_array_mask.npy"
    mask.write_bytes(b"mask")
    cli_invoked = False

    def run_kaggle_submit_kernel(**_kwargs):
        nonlocal cli_invoked
        cli_invoked = True

    with pytest.raises(SubmissionCliError, match="output contract is invalid") as exc:
        run_notebook_kernel_submission(
            slug="arc-demo",
            run_id="run-1",
            iteration=2,
            iter_logs_dir=logs_dir,
            base_dir=tmp_path,
            kaggle_username="user",
            kernel_name=None,
            accelerator="gpu",
            strict_accelerator=False,
            submission_path=submission_path,
            message="submit",
            artifact_mode="gateway",
            dry_run=False,
            timeout_minutes=60,
            run_submit_kernel=lambda **_kwargs: SimpleNamespace(
                kernel_id="user/arc-submit",
                output_dir=output_dir,
                submission_path=mask,
            ),
            run_kaggle_submit_kernel=run_kaggle_submit_kernel,
            copy_submission_artifact=lambda source: source,
            classify_submit_error=lambda *_args: {},
            should_retry_ambiguous=lambda **_kwargs: False,
            sleep=lambda _seconds: None,
            on_message=lambda _message: None,
            is_capacity_error=lambda _exc: False,
            is_push_error=lambda _exc: False,
            expected_output_file="submission.parquet",
        )

    assert cli_invoked is False
    assert "discovered_output=test_array_mask.npy" in exc.value.stderr
    assert exc.value.submission_ref == "kernel:user/arc-submit"
    assert exc.value.submission_artifact_path is None
    assert exc.value.code_output_file_name == "submission.parquet"


def test_run_notebook_kernel_submission_fails_before_cli_when_remote_pipeline_degrades(
    tmp_path: Path,
) -> None:
    submission_path = tmp_path / "iter-2" / "submission.json"
    submission_path.parent.mkdir(parents=True)
    submission_path.write_text("{}\n", encoding="utf-8")
    logs_dir = tmp_path / "iter-2" / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "kernel_push-001.txt").write_text(
        "Kernel version 5 successfully pushed.\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "kernel-output"
    output_dir.mkdir()
    remote_submission = output_dir / "submission.json"
    remote_submission.write_text("{}\n", encoding="utf-8")
    remote_metrics = output_dir / "metrics.json"
    remote_metrics.write_text(
        '{"metric":"accuracy","direction":"maximize","offline_value":0.46,"chosen_pipeline":"simple_baseline"}\n',
        encoding="utf-8",
    )
    cli_invoked = False

    def run_kaggle_submit_kernel(**_kwargs):
        nonlocal cli_invoked
        cli_invoked = True

    with pytest.raises(SubmissionCliError, match="does not match the selected candidate") as exc:
        run_notebook_kernel_submission(
            slug="arc-demo",
            run_id="run-1",
            iteration=2,
            iter_logs_dir=logs_dir,
            base_dir=tmp_path,
            kaggle_username="user",
            kernel_name=None,
            accelerator="gpu",
            strict_accelerator=False,
            submission_path=submission_path,
            message="submit",
            artifact_mode="inference",
            dry_run=False,
            timeout_minutes=60,
            run_submit_kernel=lambda **_kwargs: SimpleNamespace(
                kernel_id="user/arc-submit",
                output_dir=output_dir,
                submission_path=remote_submission,
                metrics_path=remote_metrics,
            ),
            run_kaggle_submit_kernel=run_kaggle_submit_kernel,
            copy_submission_artifact=lambda source: source,
            classify_submit_error=lambda *_args: {},
            should_retry_ambiguous=lambda **_kwargs: False,
            sleep=lambda _seconds: None,
            on_message=lambda _message: None,
            is_capacity_error=lambda _exc: False,
            is_push_error=lambda _exc: False,
            expected_output_file="submission.json",
            expected_metrics_payload={
                "metric": "accuracy",
                "direction": "maximize",
                "offline_value": 86.5,
                "chosen_pipeline": "qwen",
            },
        )

    assert cli_invoked is False
    assert "pipeline changed from 'qwen' to 'simple_baseline'" in exc.value.stderr
    assert exc.value.submission_ref == "kernel:user/arc-submit"
    assert exc.value.submission_artifact_path == remote_submission
    assert exc.value.kernel_version == "5"


def test_notebook_submit_error_preserves_kernel_and_gateway_artifact_context(tmp_path: Path) -> None:
    submission_path = tmp_path / "iter-2" / "test_array_mask.npy"
    submission_path.parent.mkdir(parents=True)
    submission_path.write_bytes(b"diagnostic")
    logs_dir = tmp_path / "iter-2" / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "kernel_push-001.txt").write_text(
        "Kernel version 5 successfully pushed.\n",
        encoding="utf-8",
    )
    gateway_output = tmp_path / "kernel-output" / "submission.parquet"
    gateway_output.parent.mkdir()
    gateway_output.write_bytes(b"parquet")
    copied_output = tmp_path / "copied" / "submission.parquet"
    submit_error = SubmissionCliError(
        "bad request",
        command=[],
        exit_code=1,
        stderr="400 Client Error: Bad Request",
    )

    with pytest.raises(SubmissionCliError) as exc:
        run_notebook_kernel_submission(
            slug="arc-demo",
            run_id="run-1",
            iteration=2,
            iter_logs_dir=logs_dir,
            base_dir=tmp_path,
            kaggle_username="user",
            kernel_name=None,
            accelerator="gpu",
            strict_accelerator=False,
            submission_path=submission_path,
            message="submit",
            artifact_mode="gateway",
            dry_run=False,
            timeout_minutes=60,
            run_submit_kernel=lambda **_kwargs: SimpleNamespace(
                kernel_id="user/arc-submit",
                output_dir=gateway_output.parent,
                submission_path=gateway_output,
            ),
            run_kaggle_submit_kernel=lambda **_kwargs: (_ for _ in ()).throw(submit_error),
            copy_submission_artifact=lambda _source: copied_output,
            classify_submit_error=lambda *_args: {"reason": "bad_request"},
            should_retry_ambiguous=lambda **_kwargs: False,
            sleep=lambda _seconds: None,
            on_message=lambda _message: None,
            is_capacity_error=lambda _exc: False,
            is_push_error=lambda _exc: False,
            expected_output_file="submission.parquet",
        )

    assert exc.value.submission_ref == "kernel:user/arc-submit"
    assert exc.value.submission_artifact_path == copied_output
    assert exc.value.code_output_file_name == "submission.parquet"
    assert exc.value.kernel_version == "5"


def test_run_notebook_kernel_submission_keeps_expected_output_file_when_kernel_path_missing(tmp_path: Path) -> None:
    submission_path = tmp_path / "iter-2" / "submission.csv.gz"
    submission_path.parent.mkdir(parents=True)
    submission_path.write_bytes(b"compressed")
    logs_dir = tmp_path / "iter-2" / "logs"
    logs_dir.mkdir(parents=True)
    captured_submit_kwargs: dict[str, object] = {}

    def run_submit_kernel(**kwargs):  # noqa: ANN003, ARG001
        return type("KernelResult", (), {"kernel_id": "user/demo-submit", "submission_path": None})()

    def run_kaggle_submit_kernel(**kwargs):  # noqa: ANN003
        captured_submit_kwargs.update(kwargs)
        return type("SubmitResult", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    run_notebook_kernel_submission(
        slug="demo",
        run_id="run-1",
        iteration=2,
        iter_logs_dir=logs_dir,
        base_dir=tmp_path,
        kaggle_username="user",
        kernel_name="submit-kernel",
        accelerator="gpu",
        strict_accelerator=False,
        submission_path=submission_path,
        message="submit message",
        artifact_mode="inference",
        dry_run=True,
        timeout_minutes=60,
        run_submit_kernel=run_submit_kernel,
        run_kaggle_submit_kernel=run_kaggle_submit_kernel,
        copy_submission_artifact=lambda source: source,
        classify_submit_error=lambda stdout, stderr, exit_code: {"reason": "unclassified_submit_error"},
        should_retry_ambiguous=lambda *, reason, stdout, stderr: False,
        sleep=lambda seconds: None,
        on_message=lambda message: None,
        is_capacity_error=lambda exc: False,
        is_push_error=lambda exc: False,
        expected_output_file="submission.csv.gz",
    )

    assert captured_submit_kwargs["output_file"] == "submission.csv.gz"


def test_run_notebook_kernel_submission_uses_wrapper_zip_name_for_directory_when_kernel_path_missing(
    tmp_path: Path,
) -> None:
    submission_path = tmp_path / "iter-2" / "model"
    submission_path.mkdir(parents=True)
    (submission_path / "config.json").write_text('{"architectures": ["Demo"]}\n', encoding="utf-8")
    (submission_path / "model.safetensors").write_bytes(b"weights")
    logs_dir = tmp_path / "iter-2" / "logs"
    logs_dir.mkdir(parents=True)
    captured_submit_kwargs: dict[str, object] = {}

    def run_submit_kernel(**kwargs):  # noqa: ANN003, ARG001
        return type("KernelResult", (), {"kernel_id": "user/demo-submit", "submission_path": None})()

    def run_kaggle_submit_kernel(**kwargs):  # noqa: ANN003
        captured_submit_kwargs.update(kwargs)
        return type("SubmitResult", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    run_notebook_kernel_submission(
        slug="demo",
        run_id="run-1",
        iteration=2,
        iter_logs_dir=logs_dir,
        base_dir=tmp_path,
        kaggle_username="user",
        kernel_name="submit-kernel",
        accelerator="gpu",
        strict_accelerator=False,
        submission_path=submission_path,
        message="submit message",
        artifact_mode="wrapper",
        dry_run=True,
        timeout_minutes=60,
        run_submit_kernel=run_submit_kernel,
        run_kaggle_submit_kernel=run_kaggle_submit_kernel,
        copy_submission_artifact=lambda source: source,
        classify_submit_error=lambda stdout, stderr, exit_code: {"reason": "unclassified_submit_error"},
        should_retry_ambiguous=lambda *, reason, stdout, stderr: False,
        sleep=lambda seconds: None,
        on_message=lambda message: None,
        is_capacity_error=lambda exc: False,
        is_push_error=lambda exc: False,
        expected_output_file="submission.hfmodel",
    )

    assert captured_submit_kwargs["output_file"] == "model.zip"


def test_notebook_kernel_submission_error_preserves_kaggle_cli_details() -> None:
    wrapped = notebook_kernel_submission_error(
        KaggleCliError(
            "push failed",
            command=["kaggle", "kernels", "push"],
            exit_code=123,
            output="full output",
            stdout="stdout text",
            stderr="stderr text",
        )
    )

    assert isinstance(wrapped, SubmissionCliError)
    assert wrapped.command == ["kaggle", "kernels", "push"]
    assert wrapped.exit_code == 123
    assert wrapped.output == "full output"
    assert wrapped.stdout == "stdout text"
    assert wrapped.stderr == "stderr text"


def test_notebook_kernel_submission_error_wraps_generic_exception() -> None:
    wrapped = notebook_kernel_submission_error(ValueError("bad kernel"))

    assert isinstance(wrapped, SubmissionCliError)
    assert wrapped.command == []
    assert wrapped.output == "bad kernel"
    assert wrapped.stderr == "bad kernel"


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
