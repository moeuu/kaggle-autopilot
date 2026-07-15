from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kagglebot import submit_abort as _submit_abort
from kagglebot import submit_abort_specs as _submit_abort_specs
from kagglebot.submit_cli_error_resolution import SubmitStageRuntimeState, resolve_submit_cli_error_for_run


@dataclass(frozen=True)
class SubmitStageAttemptResult:
    submission_result: object
    submission_reference: str
    submission_artifact_path: Path | None


@dataclass(frozen=True)
class SubmitStageAttemptLoopResult:
    submission_result: object
    submission_reference: str
    submission_artifact_path: Path | None
    submit_stage_state: SubmitStageRuntimeState


def run_submit_stage_attempt(
    *,
    notebook_submit_required: bool,
    file_submission_path: Path,
    run_notebook_submit: Callable[[], tuple[object, str, Path | None]],
    run_file_submit: Callable[[], object],
) -> SubmitStageAttemptResult:
    if notebook_submit_required:
        notebook_result, notebook_ref, notebook_artifact_path = run_notebook_submit()
        return SubmitStageAttemptResult(
            submission_result=notebook_result,
            submission_reference=notebook_ref,
            submission_artifact_path=notebook_artifact_path,
        )

    file_result = run_file_submit()
    file_result_path = getattr(file_result, "submission_path", file_submission_path)
    return SubmitStageAttemptResult(
        submission_result=file_result,
        submission_reference=str(file_result_path),
        submission_artifact_path=file_result_path if isinstance(file_result_path, Path) else file_submission_path,
    )


def run_submit_stage_attempts_until_success_or_abort(
    *,
    run_dir: Path,
    run_id: str,
    state: SubmitStageRuntimeState,
    prepared_submission_path: Path,
    message: str,
    code_competition: bool,
    max_attempts: int,
    backoff_base_seconds: float,
    sample_submission_path: Path,
    fallback_sample_submission_path: Path,
    submit_code_fingerprint: str | None,
    run_state: dict[str, object],
    seen_fingerprints: set[str],
    run_notebook_submit: Callable[[SubmitStageRuntimeState], tuple[object, str, Path | None]],
    run_file_submit: Callable[[], object],
    submit_aborter: _submit_abort.SubmitRunAborter,
    submit_attempt_recorder: object,
    submit_retry_recorder: _submit_abort.SubmitRunRetryRecorder,
    submission_cli_error_types: tuple[type[BaseException], ...],
    local_guardrail_error_types: tuple[type[BaseException], ...],
    kaggle_cli_error_types: tuple[type[BaseException], ...],
    classify_submit_error: Callable[..., object],
    should_use_notebook_fallback: Callable[..., bool],
    resolve_notebook_submit_artifact_mode: Callable[..., str],
    decide_notebook_submit_artifact_mode_for_paths: Callable[..., object],
    count_tabular_data_rows: Callable[..., int | None],
    compute_error_fingerprint: Callable[..., str],
    decide_submit_fingerprint_reuse: Callable[..., object],
    compute_submit_backoff: Callable[..., float],
    save_run_state_for_run: Callable[[Path, dict[str, object]], object],
    is_missing_credentials_error: Callable[[object], bool],
    build_submit_aborted_error: Callable[[str], BaseException],
    sleep: Callable[[float], object],
    on_message: Callable[[str], object],
) -> SubmitStageAttemptLoopResult:
    del message
    submit_stage_state = state
    submission_result = None
    submission_reference = str(prepared_submission_path)
    submission_artifact_path: Path | None = prepared_submission_path
    resolved_max_attempts = max(1, max_attempts)

    for attempt in range(1, resolved_max_attempts + 1):
        try:
            submit_attempt_result = run_submit_stage_attempt(
                notebook_submit_required=submit_stage_state.notebook_submit_required,
                file_submission_path=prepared_submission_path,
                run_notebook_submit=lambda: run_notebook_submit(submit_stage_state),
                run_file_submit=run_file_submit,
            )
            submission_result = submit_attempt_result.submission_result
            submission_reference = submit_attempt_result.submission_reference
            submission_artifact_path = submit_attempt_result.submission_artifact_path
        except submission_cli_error_types as exc:
            exception_submission_ref = str(getattr(exc, "submission_ref", "") or "").strip()
            if exception_submission_ref:
                submission_reference = exception_submission_ref
            if hasattr(exc, "submission_artifact_path"):
                exception_artifact_path = getattr(exc, "submission_artifact_path")
                submission_artifact_path = (
                    exception_artifact_path if isinstance(exception_artifact_path, Path) else None
                )
            stdout = str(getattr(exc, "stdout", "") or "")
            stderr = str(getattr(exc, "stderr", "") or "")
            output = str(getattr(exc, "output", "") or "")
            exit_code = getattr(exc, "exit_code", None)
            submit_error_resolution = resolve_submit_cli_error_for_run(
                run_dir=run_dir,
                state=submit_stage_state,
                stdout=stdout,
                stderr=stderr,
                output=output,
                exit_code=exit_code if isinstance(exit_code, int) else None,
                attempt=attempt,
                max_attempts=resolved_max_attempts,
                backoff_base_seconds=backoff_base_seconds,
                classify_submit_error=classify_submit_error,
                should_use_notebook_fallback=should_use_notebook_fallback,
                code_competition=code_competition,
                sample_submission_path=sample_submission_path,
                fallback_sample_submission_path=fallback_sample_submission_path,
                submission_path=prepared_submission_path,
                resolve_notebook_submit_artifact_mode=resolve_notebook_submit_artifact_mode,
                decide_notebook_submit_artifact_mode_for_paths=decide_notebook_submit_artifact_mode_for_paths,
                count_tabular_data_rows=count_tabular_data_rows,
                compute_error_fingerprint=compute_error_fingerprint,
                decide_submit_fingerprint_reuse=decide_submit_fingerprint_reuse,
                compute_submit_backoff=compute_submit_backoff,
                seen_fingerprints=seen_fingerprints,
                run_state=run_state,
                code_fingerprint=submit_code_fingerprint,
                save_run_state_for_run=save_run_state_for_run,
                on_message=on_message,
            )
            submit_error_classification = submit_error_resolution.classification
            fallback_application = submit_error_resolution.fallback_application
            submit_stage_state = fallback_application.state
            if fallback_application.retry_as_notebook:
                continue
            fingerprint = submit_error_resolution.fingerprint
            error_action = submit_error_resolution.error_action
            if error_action is None:
                raise build_submit_aborted_error("Submit error resolution did not produce a retry or abort action.")
            if error_action.action == "abort":
                abort_spec = _submit_abort_specs.build_submit_stage_error_action_abort_spec(
                    action=error_action,
                    fingerprint=fingerprint,
                    stdout=stdout,
                    stderr=submit_error_classification.stderr,
                    exit_code=exit_code if isinstance(exit_code, int) else None,
                )
                return submit_aborter.abort(
                    submission_ref=submission_reference,
                    submission_artifact_path=submission_artifact_path,
                    artifact_mode=submit_stage_state.submission_artifact_mode,
                    code_fingerprint=submit_code_fingerprint,
                    **_submit_abort_specs.build_submit_abort_spec_kwargs(abort_spec),
                    submit_attempt_recorder=submit_attempt_recorder,
                )
            seen_fingerprints.add(fingerprint)
            if error_action.action == "retry":
                submit_retry_recorder.record(
                    submission_ref=submission_reference,
                    submission_artifact_path=submission_artifact_path,
                    fallback_submission_path=prepared_submission_path,
                    exit_code=exit_code if isinstance(exit_code, int) else None,
                    fingerprint=fingerprint,
                    action=error_action,
                    stdout=stdout,
                    stderr=submit_error_classification.stderr,
                    attempt=attempt,
                )
                sleep(error_action.wait_seconds)
                continue
        except local_guardrail_error_types as exc:
            abort_spec = _submit_abort_specs.resolve_local_submission_guardrail_abort_spec(
                error=exc,
                compute_error_fingerprint=compute_error_fingerprint,
            )
            return submit_aborter.abort(
                submission_ref=submission_reference,
                submission_artifact_path=submission_artifact_path,
                code_fingerprint=submit_code_fingerprint,
                **_submit_abort_specs.build_submit_abort_spec_kwargs(abort_spec),
                submit_attempt_recorder=submit_attempt_recorder,
            )
        except kaggle_cli_error_types as exc:
            abort_spec = _submit_abort_specs.resolve_kaggle_cli_submit_abort_spec(
                error=exc,
                is_missing_credentials_error=is_missing_credentials_error,
                compute_error_fingerprint=compute_error_fingerprint,
            )
            if abort_spec is None:
                raise
            return submit_aborter.abort(
                submission_ref=submission_reference,
                submission_artifact_path=submission_artifact_path,
                artifact_mode=submit_stage_state.submission_artifact_mode,
                code_fingerprint=submit_code_fingerprint,
                **_submit_abort_specs.build_submit_abort_spec_kwargs(abort_spec),
                submit_attempt_recorder=submit_attempt_recorder,
            )
        break

    if submission_result is None:
        raise build_submit_aborted_error("Submit failed before producing a submission result.")
    return SubmitStageAttemptLoopResult(
        submission_result=submission_result,
        submission_reference=submission_reference,
        submission_artifact_path=submission_artifact_path,
        submit_stage_state=submit_stage_state,
    )
