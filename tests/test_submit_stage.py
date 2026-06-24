from __future__ import annotations

from pathlib import Path

from kagglebot.submit_stage import (
    build_submit_stage_success_record,
    classify_submit_stage_error,
    decide_initial_submit_stage_mode,
    decide_notebook_fallback_after_file_submit_error,
    decide_submit_stage_error_action,
    run_submit_stage_attempt,
)


class FileSubmitResult:
    def __init__(self, submission_path: Path) -> None:
        self.submission_path = submission_path


class SubmitResultStub:
    def __init__(
        self,
        *,
        stdout: object = "",
        stderr: object = "",
        exit_code: int | None = None,
        returncode: int | None = None,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        if exit_code is not None:
            self.exit_code = exit_code
        if returncode is not None:
            self.returncode = returncode


def test_decide_initial_submit_stage_mode_keeps_file_submit() -> None:
    decision = decide_initial_submit_stage_mode(
        requested_notebook_submit=False,
        notebook_submissions_only=False,
        notebook_submit_artifact_mode="wrapper",
        resolved_notebook_artifact_mode="inference",
    )

    assert decision.notebook_submit_required is False
    assert decision.notebook_fallback_activated is False
    assert decision.submission_artifact_mode == "wrapper"
    assert decision.messages == ()


def test_decide_initial_submit_stage_mode_uses_requested_notebook_mode() -> None:
    decision = decide_initial_submit_stage_mode(
        requested_notebook_submit=True,
        notebook_submissions_only=False,
        notebook_submit_artifact_mode="wrapper",
        resolved_notebook_artifact_mode="inference",
    )

    assert decision.notebook_submit_required is True
    assert decision.notebook_fallback_activated is True
    assert decision.submission_artifact_mode == "inference"
    assert decision.messages == ("[yellow]submit mode[/yellow]: using notebook submit",)


def test_decide_initial_submit_stage_mode_forces_notebook_only_competition() -> None:
    decision = decide_initial_submit_stage_mode(
        requested_notebook_submit=False,
        notebook_submissions_only=True,
        notebook_submit_artifact_mode="wrapper",
        resolved_notebook_artifact_mode="inference",
    )

    assert decision.notebook_submit_required is True
    assert decision.notebook_fallback_activated is True
    assert decision.submission_artifact_mode == "inference"
    assert decision.messages == (
        "[yellow]submit mode[/yellow]: notebook-only competition detected; forcing notebook submit",
        "[yellow]submit mode[/yellow]: using notebook submit",
    )


def test_run_submit_stage_attempt_uses_file_submit_result_path(tmp_path: Path) -> None:
    prepared_path = tmp_path / "prepared.csv"
    submitted_path = tmp_path / "submitted.csv"

    result = run_submit_stage_attempt(
        notebook_submit_required=False,
        file_submission_path=prepared_path,
        run_notebook_submit=lambda: (_ for _ in ()).throw(AssertionError("notebook should not run")),
        run_file_submit=lambda: FileSubmitResult(submitted_path),
    )

    assert isinstance(result.submission_result, FileSubmitResult)
    assert result.submission_reference == str(submitted_path)
    assert result.submission_artifact_path == submitted_path


def test_run_submit_stage_attempt_uses_notebook_submit_tuple(tmp_path: Path) -> None:
    notebook_artifact = tmp_path / "notebook-submission.csv"

    result = run_submit_stage_attempt(
        notebook_submit_required=True,
        file_submission_path=tmp_path / "prepared.csv",
        run_notebook_submit=lambda: ("notebook-result", "kernel:user/demo", notebook_artifact),
        run_file_submit=lambda: (_ for _ in ()).throw(AssertionError("file should not run")),
    )

    assert result.submission_result == "notebook-result"
    assert result.submission_reference == "kernel:user/demo"
    assert result.submission_artifact_path == notebook_artifact


def test_build_submit_stage_success_record_prefers_exit_code() -> None:
    record = build_submit_stage_success_record(
        submission_result=SubmitResultStub(stdout="ok", stderr="warn", exit_code=7, returncode=0),
        compute_error_fingerprint=lambda stdout, stderr: f"{stdout}:{stderr}",
    )

    assert record.exit_code == 7
    assert record.fingerprint == "ok:warn"
    assert record.stdout == "ok"
    assert record.stderr == "warn"


def test_build_submit_stage_success_record_uses_returncode_fallback() -> None:
    record = build_submit_stage_success_record(
        submission_result=SubmitResultStub(stdout=None, stderr=None, returncode=0),
        compute_error_fingerprint=lambda stdout, stderr: f"{stdout}:{stderr}",
    )

    assert record.exit_code == 0
    assert record.fingerprint == ":"
    assert record.stdout == ""
    assert record.stderr == ""


def test_classify_submit_stage_error_uses_output_fallback() -> None:
    calls: list[str] = []

    def classify(stdout: str, stderr: str, exit_code: int | None) -> dict[str, object]:  # noqa: ARG001
        calls.append(stderr)
        if "kernel must be specified" in stderr:
            return {
                "kind": "permanent",
                "reason": "ambiguous_notebook_bad_request",
                "retry_after_seconds": 4,
            }
        return {"reason": "unclassified_submit_error"}

    classification = classify_submit_stage_error(
        stdout="",
        stderr="",
        output="400 Client Error\nkernel must be specified",
        exit_code=1,
        classify_submit_error=classify,
    )

    assert classification.stderr == "400 Client Error\nkernel must be specified"
    assert classification.kind == "permanent"
    assert classification.reason == "ambiguous_notebook_bad_request"
    assert classification.retry_after_seconds == 4.0
    assert calls == ["", "400 Client Error\nkernel must be specified"]


def test_classify_submit_stage_error_defaults_unknown_kind_and_reason() -> None:
    classification = classify_submit_stage_error(
        stdout="",
        stderr="generic",
        output="",
        exit_code=1,
        classify_submit_error=lambda stdout, stderr, exit_code: {},
    )

    assert classification.stderr == "generic"
    assert classification.kind == "unknown"
    assert classification.reason == "unclassified_submit_error"
    assert classification.retry_after_seconds == 0.0


def test_decide_submit_stage_error_action_aborts_repeated_fingerprint() -> None:
    decision = decide_submit_stage_error_action(
        fingerprint_seen=True,
        same_fingerprint_retry_allowed=False,
        classification_kind="transient",
        classification_reason="network_or_timeout",
        attempt=1,
        max_attempts=3,
        retry_after_seconds=0.0,
        backoff_seconds=2.0,
    )

    assert decision.action == "abort"
    assert decision.error_kind == "transient"
    assert decision.reason == "same_error_fingerprint_recurred"
    assert "Same submit error fingerprint recurred" in decision.abort_message
    assert decision.messages == ()


def test_decide_submit_stage_error_action_retries_transient_with_allowance_message() -> None:
    decision = decide_submit_stage_error_action(
        fingerprint_seen=True,
        same_fingerprint_retry_allowed=True,
        classification_kind="transient",
        classification_reason="network_or_timeout",
        attempt=1,
        max_attempts=3,
        retry_after_seconds=5.0,
        backoff_seconds=2.0,
    )

    assert decision.action == "retry"
    assert decision.error_kind == "transient"
    assert decision.reason == "network_or_timeout"
    assert decision.wait_seconds == 5.0
    assert "same fingerprint matched previous failures" in decision.messages[0]
    assert "transient submit error" in decision.messages[1]


def test_decide_submit_stage_error_action_aborts_after_retry_budget() -> None:
    decision = decide_submit_stage_error_action(
        fingerprint_seen=False,
        same_fingerprint_retry_allowed=False,
        classification_kind="transient",
        classification_reason="network_or_timeout",
        attempt=3,
        max_attempts=3,
        retry_after_seconds=0.0,
        backoff_seconds=8.0,
    )

    assert decision.action == "abort"
    assert decision.reason == "network_or_timeout"
    assert decision.abort_message == "Transient submit error exceeded retry budget; aborting this run."
    assert "no further retries" in decision.messages[0]


def test_decide_notebook_fallback_after_file_submit_error_retries_as_notebook() -> None:
    decision = decide_notebook_fallback_after_file_submit_error(
        notebook_submit_required=False,
        notebook_fallback_activated=False,
        should_use_notebook_fallback=True,
        resolved_notebook_artifact_mode="inference",
        current_submission_artifact_mode="wrapper",
    )

    assert decision.retry_as_notebook is True
    assert decision.notebook_submit_required is True
    assert decision.notebook_fallback_activated is True
    assert decision.submission_artifact_mode == "inference"
    assert decision.messages == (
        "[yellow]submit mode[/yellow]: file submit indicates notebook submit is required; "
        "retrying via notebook submit automatically.",
    )


def test_decide_notebook_fallback_after_file_submit_error_rejects_already_activated() -> None:
    decision = decide_notebook_fallback_after_file_submit_error(
        notebook_submit_required=False,
        notebook_fallback_activated=True,
        should_use_notebook_fallback=True,
        resolved_notebook_artifact_mode="inference",
        current_submission_artifact_mode="wrapper",
    )

    assert decision.retry_as_notebook is False
    assert decision.notebook_submit_required is False
    assert decision.notebook_fallback_activated is True
    assert decision.submission_artifact_mode == "wrapper"
    assert decision.messages == ()


def test_decide_notebook_fallback_after_file_submit_error_rejects_non_notebook_error() -> None:
    decision = decide_notebook_fallback_after_file_submit_error(
        notebook_submit_required=False,
        notebook_fallback_activated=False,
        should_use_notebook_fallback=False,
        resolved_notebook_artifact_mode="inference",
        current_submission_artifact_mode="wrapper",
    )

    assert decision.retry_as_notebook is False
    assert decision.notebook_submit_required is False
    assert decision.notebook_fallback_activated is False
    assert decision.submission_artifact_mode == "wrapper"
