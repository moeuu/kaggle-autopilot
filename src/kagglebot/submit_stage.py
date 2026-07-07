from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from kagglebot import submit_abort_specs as _submit_abort_specs
from kagglebot import submit_attempts as _submit_attempts
from kagglebot import submit_outcome as _submit_outcome
from kagglebot import submit_preflight as _submit_preflight
from kagglebot import submit_stage_duplicate as _submit_stage_duplicate
from kagglebot import submit_stage_modes as _submit_stage_modes
from kagglebot.submit_cli_error_resolution import SubmitStageRuntimeState
from kagglebot.writeup import normalize_submit_mode


@dataclass(frozen=True)
class SubmitPreflightContext:
    duplicate_skip_result: dict[str, object] | None
    same_submission_path_skipped: bool
    submit_stage_state: SubmitStageRuntimeState | None
    code_competition: bool
    seen_fingerprints: set[str]


@dataclass(frozen=True)
class SubmitPreparedPreflightContext:
    prepared_context: _submit_preflight.SubmitPreparedRunContext
    preflight_context: SubmitPreflightContext


def wait_for_submission_outcome(**kwargs):
    return _submit_outcome.wait_for_submission_outcome(**kwargs)


def _resolve_pre_validation_submit_stage_state(
    *,
    submit_mode: object,
    notebook_submissions_only: bool,
    notebook_submit_artifact_mode: str | None,
    code_competition: bool,
    sample_submission_path: Path,
    fallback_sample_submission_path: Path,
    submission_path: Path,
    resolve_notebook_submit_artifact_mode: Callable[..., str],
    decide_notebook_submit_artifact_mode_for_paths: Callable[..., object],
    count_tabular_data_rows: Callable[[Path], int | None],
) -> SubmitStageRuntimeState | None:
    requested_notebook_submit = normalize_submit_mode(submit_mode, default="file") == "notebook"
    if not requested_notebook_submit and not notebook_submissions_only:
        return None
    return _submit_stage_modes.resolve_initial_submit_stage_runtime_state(
        submit_mode=submit_mode,
        notebook_submissions_only=notebook_submissions_only,
        notebook_submit_artifact_mode=notebook_submit_artifact_mode,
        code_competition=code_competition,
        sample_submission_path=sample_submission_path,
        fallback_sample_submission_path=fallback_sample_submission_path,
        submission_path=submission_path,
        resolve_notebook_submit_artifact_mode=resolve_notebook_submit_artifact_mode,
        decide_notebook_submit_artifact_mode_for_paths=decide_notebook_submit_artifact_mode_for_paths,
        count_tabular_data_rows=count_tabular_data_rows,
        on_message=lambda _message: None,
    )


def _uses_notebook_inference_submit_artifact(state: SubmitStageRuntimeState | None) -> bool:
    return bool(
        state is not None
        and state.notebook_submit_required
        and str(state.submission_artifact_mode or "").strip().lower() == "inference"
    )


def resolve_submit_preflight_for_run_or_abort(
    *,
    run_dir: Path,
    submission_ledger_path: Path,
    slug: str,
    run_id: str,
    message: str,
    submitted_at: datetime,
    source_submission_path: Path,
    prepared_submission_path: Path,
    prepared_submission_sha: str,
    code_fingerprint: str,
    allow_force: bool,
    run_state: dict[str, object],
    latest_submit_attempt: dict[str, object],
    submit_mode: object,
    notebook_submissions_only: bool,
    notebook_submit_artifact_mode: str | None,
    code_competition: bool,
    sample_submission_path: Path,
    fallback_sample_submission_path: Path,
    load_run_state: Callable[[Path], dict[str, object]],
    collect_duplicate_submission_sources: Callable[..., list[str]],
    decide_duplicate_submission_action: Callable[..., object],
    check_rules_accepted: Callable[[], bool],
    cli_error_types: tuple[type[BaseException], ...],
    is_missing_credentials_error: Callable[[BaseException], bool],
    rules_not_accepted_exit_code: int | None,
    resolve_notebook_submit_artifact_mode: Callable[..., str],
    decide_notebook_submit_artifact_mode_for_paths: Callable[..., object],
    count_tabular_data_rows: Callable[[Path], int | None],
    decide_same_submission_path_action: Callable[..., object],
    compute_error_fingerprint: Callable[[str, str], str],
    compute_submission_sha256: Callable[[Path | None], str | None],
    submit_aborter: object,
    submit_attempt_recorder: object,
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    on_message: Callable[[str], object],
    skip_static_duplicate_check: bool = False,
) -> SubmitPreflightContext:
    duplicate_skip_result = None
    if not skip_static_duplicate_check:
        duplicate_skip_result = _submit_stage_duplicate.resolve_duplicate_submission_for_run(
            run_dir=run_dir,
            submission_ledger_path=submission_ledger_path,
            slug=slug,
            run_id=run_id,
            message=message,
            submitted_at=submitted_at,
            submission_path=source_submission_path,
            prepared_submission_path=prepared_submission_path,
            prepared_submission_sha=prepared_submission_sha,
            code_fingerprint=code_fingerprint,
            allow_force=allow_force,
            load_run_state=load_run_state,
            collect_duplicate_submission_sources=collect_duplicate_submission_sources,
            decide_duplicate_submission_action=decide_duplicate_submission_action,
            compute_error_fingerprint=compute_error_fingerprint,
            record_submit_attempt_payloads=submit_attempt_recorder.record_payloads,
            stdout_tail_chars=stdout_tail_chars,
            stderr_tail_chars=stderr_tail_chars,
            on_message=on_message,
        )
    if duplicate_skip_result is not None:
        return SubmitPreflightContext(
            duplicate_skip_result=duplicate_skip_result,
            same_submission_path_skipped=False,
            submit_stage_state=None,
            code_competition=code_competition,
            seen_fingerprints=set(),
        )

    rules_resolution = _submit_preflight.resolve_rules_acceptance_for_submit(
        check_rules_accepted=check_rules_accepted,
        cli_error_types=cli_error_types,
        is_missing_credentials_error=is_missing_credentials_error,
        rules_not_accepted_exit_code=rules_not_accepted_exit_code,
        compute_error_fingerprint=compute_error_fingerprint,
    )
    if rules_resolution.abort_spec is not None:
        return submit_aborter.abort(
            submission_ref=prepared_submission_path,
            code_fingerprint=code_fingerprint,
            **_submit_abort_specs.build_submit_abort_spec_kwargs(rules_resolution.abort_spec),
            submit_attempt_recorder=submit_attempt_recorder,
        )

    submit_stage_state = _submit_stage_modes.resolve_initial_submit_stage_runtime_state(
        submit_mode=submit_mode,
        notebook_submissions_only=notebook_submissions_only,
        notebook_submit_artifact_mode=notebook_submit_artifact_mode,
        code_competition=code_competition,
        sample_submission_path=sample_submission_path,
        fallback_sample_submission_path=fallback_sample_submission_path,
        submission_path=prepared_submission_path,
        resolve_notebook_submit_artifact_mode=resolve_notebook_submit_artifact_mode,
        decide_notebook_submit_artifact_mode_for_paths=decide_notebook_submit_artifact_mode_for_paths,
        count_tabular_data_rows=count_tabular_data_rows,
        on_message=on_message,
    )

    same_submission_path_skipped = _submit_stage_duplicate.resolve_same_submission_path_for_run(
        run_id=run_id,
        run_state=run_state,
        latest_submit_attempt=latest_submit_attempt,
        prepared_submission_path=prepared_submission_path,
        current_submission_sha=prepared_submission_sha,
        submit_code_fingerprint=code_fingerprint,
        allow_force=allow_force,
        notebook_submit_required=submit_stage_state.notebook_submit_required,
        decide_same_submission_path_action=decide_same_submission_path_action,
        compute_submission_sha256=compute_submission_sha256,
        submit_attempt_recorder=submit_attempt_recorder,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
        on_message=on_message,
    )
    return SubmitPreflightContext(
        duplicate_skip_result=None,
        same_submission_path_skipped=same_submission_path_skipped,
        submit_stage_state=submit_stage_state,
        code_competition=code_competition,
        seen_fingerprints=_submit_attempts.build_seen_submit_fingerprint_set_for_run(
            run_dir=run_dir,
            run_state=run_state,
        ),
    )


def prepare_and_resolve_submit_preflight_for_run_or_abort(
    *,
    run_dir: Path,
    submission_ledger_path: Path,
    slug: str,
    run_id: str,
    message: str,
    submitted_at: datetime,
    source_submission_path: Path,
    input_submission_path: Path,
    validate_and_prepare: Callable[[Path], Path],
    validation_error_types: tuple[type[BaseException], ...],
    validation_exit_code: int | None,
    code_fingerprint: str,
    allow_force: bool,
    run_state: dict[str, object],
    latest_submit_attempt: dict[str, object],
    submit_mode: object,
    notebook_submissions_only: bool,
    notebook_submit_artifact_mode: str | None,
    code_competition: bool,
    sample_submission_path: Path,
    fallback_sample_submission_path: Path,
    load_run_state: Callable[[Path], dict[str, object]],
    collect_duplicate_submission_sources: Callable[..., list[str]],
    decide_duplicate_submission_action: Callable[..., object],
    check_rules_accepted: Callable[[], bool],
    cli_error_types: tuple[type[BaseException], ...],
    is_missing_credentials_error: Callable[[BaseException], bool],
    rules_not_accepted_exit_code: int | None,
    resolve_notebook_submit_artifact_mode: Callable[..., str],
    decide_notebook_submit_artifact_mode_for_paths: Callable[..., object],
    count_tabular_data_rows: Callable[[Path], int | None],
    decide_same_submission_path_action: Callable[..., object],
    compute_error_fingerprint: Callable[[str, str], str],
    compute_submission_sha256: Callable[[Path | None], str | None],
    submit_aborter: object,
    submit_attempt_recorder: object,
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    build_error: Callable[[str], BaseException],
    on_message: Callable[[str], object],
) -> SubmitPreparedPreflightContext:
    pre_validation_submit_stage_state = _resolve_pre_validation_submit_stage_state(
        submit_mode=submit_mode,
        notebook_submissions_only=notebook_submissions_only,
        notebook_submit_artifact_mode=notebook_submit_artifact_mode,
        code_competition=code_competition,
        sample_submission_path=sample_submission_path,
        fallback_sample_submission_path=fallback_sample_submission_path,
        submission_path=input_submission_path,
        resolve_notebook_submit_artifact_mode=resolve_notebook_submit_artifact_mode,
        decide_notebook_submit_artifact_mode_for_paths=decide_notebook_submit_artifact_mode_for_paths,
        count_tabular_data_rows=count_tabular_data_rows,
    )
    uses_notebook_inference_submit_artifact = _uses_notebook_inference_submit_artifact(
        pre_validation_submit_stage_state
    )
    if uses_notebook_inference_submit_artifact:
        prepared_context = _submit_preflight.SubmitPreparedRunContext(
            prepared_submission_path=input_submission_path,
            prepared_submission_sha=str(compute_submission_sha256(input_submission_path) or "").strip(),
        )
    else:
        prepared_context = _submit_preflight.prepare_submission_for_run_or_abort(
            input_submission_path=input_submission_path,
            validate_and_prepare=validate_and_prepare,
            validation_error_types=validation_error_types,
            validation_exit_code=validation_exit_code,
            compute_error_fingerprint=compute_error_fingerprint,
            submit_aborter=submit_aborter,
            submit_attempt_recorder=submit_attempt_recorder,
            code_fingerprint=code_fingerprint,
            compute_submission_sha256=compute_submission_sha256,
            build_error=build_error,
        )
    preflight_context = resolve_submit_preflight_for_run_or_abort(
        run_dir=run_dir,
        submission_ledger_path=submission_ledger_path,
        slug=slug,
        run_id=run_id,
        message=message,
        submitted_at=submitted_at,
        source_submission_path=source_submission_path,
        prepared_submission_path=prepared_context.prepared_submission_path,
        prepared_submission_sha=prepared_context.prepared_submission_sha,
        code_fingerprint=code_fingerprint,
        allow_force=allow_force,
        skip_static_duplicate_check=(
            uses_notebook_inference_submit_artifact
            and str(notebook_submit_artifact_mode or "").strip().lower() == "inference"
        ),
        run_state=run_state,
        latest_submit_attempt=latest_submit_attempt,
        submit_mode=submit_mode,
        notebook_submissions_only=notebook_submissions_only,
        notebook_submit_artifact_mode=notebook_submit_artifact_mode,
        code_competition=code_competition,
        sample_submission_path=sample_submission_path,
        fallback_sample_submission_path=fallback_sample_submission_path,
        load_run_state=load_run_state,
        collect_duplicate_submission_sources=collect_duplicate_submission_sources,
        decide_duplicate_submission_action=decide_duplicate_submission_action,
        check_rules_accepted=check_rules_accepted,
        cli_error_types=cli_error_types,
        is_missing_credentials_error=is_missing_credentials_error,
        rules_not_accepted_exit_code=rules_not_accepted_exit_code,
        resolve_notebook_submit_artifact_mode=resolve_notebook_submit_artifact_mode,
        decide_notebook_submit_artifact_mode_for_paths=decide_notebook_submit_artifact_mode_for_paths,
        count_tabular_data_rows=count_tabular_data_rows,
        decide_same_submission_path_action=decide_same_submission_path_action,
        compute_error_fingerprint=compute_error_fingerprint,
        compute_submission_sha256=compute_submission_sha256,
        submit_aborter=submit_aborter,
        submit_attempt_recorder=submit_attempt_recorder,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
        on_message=on_message,
    )
    return SubmitPreparedPreflightContext(
        prepared_context=prepared_context,
        preflight_context=preflight_context,
    )
