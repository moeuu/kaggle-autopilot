from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from kagglebot import submit_abort_specs as _submit_abort_specs
from kagglebot import submit_attempts as _submit_attempts
from kagglebot import submit_failure_context as _submit_failure_context
from kagglebot import submit_message as _submit_message
from kagglebot import submit_outcome_decisions as _submit_outcome_decisions
from kagglebot import submit_stage_duplicate as _submit_stage_duplicate
from kagglebot import submit_stage_modes as _submit_stage_modes
from kagglebot.history import SubmissionLedger
from kagglebot.submission.outcome_service import SubmissionOutcomePollingError, SubmissionOutcomeService
from kagglebot.submission_service import SubmissionConfig, SubmissionService
from kagglebot.submit_abort_specs import SubmitAbortSpec
from kagglebot.submit_cli_error_resolution import (
    SubmitStageErrorActionDecision,
    SubmitStageRuntimeState,
    resolve_submit_cli_error_for_run,
)


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


@dataclass(frozen=True)
class SubmitStageSuccessRecord:
    exit_code: int | None
    fingerprint: str
    stdout: str
    stderr: str


@dataclass(frozen=True)
class SubmitRunAborter:
    run_dir: Path
    run_id: str
    slug: str
    knowledge_paths: object
    problem_types: list[str]
    save_run_state_for_run: Callable[[Path, dict[str, object]], object]
    resolve_submit_abort_artifact_path: Callable[..., Path | None]
    persist_submit_abort_failure: Callable[..., object]
    load_run_state: Callable[[Path], dict[str, object]]
    load_latest_submit_attempt: Callable[[Path], dict[str, object]]
    has_successful_submit_attempt: Callable[[Path], bool]
    compute_submission_sha256: Callable[[Path | None], str | None]
    stdout_tail_chars: int
    stderr_tail_chars: int
    now_iso: Callable[[], str]
    normalize_detail: Callable[..., str]
    record_error_fix_insight: Callable[..., object]
    on_message: Callable[[str], object]
    build_error: Callable[[str], BaseException]

    def abort(
        self,
        *,
        submission_ref: str | Path,
        submission_artifact_path: Path | None = None,
        artifact_mode: str | None = None,
        code_fingerprint: str | None = None,
        fingerprint: str,
        error_kind: str,
        reason: str,
        message: str,
        stdout_tail: str,
        stderr_tail: str,
        exit_code: int | None,
        submit_attempt_recorder: object | None,
    ) -> None:
        abort_submit_for_run(
            run_dir=self.run_dir,
            run_id=self.run_id,
            slug=self.slug,
            knowledge_paths=self.knowledge_paths,
            problem_types=self.problem_types,
            submission_ref=submission_ref,
            submission_artifact_path=submission_artifact_path,
            artifact_mode=artifact_mode,
            code_fingerprint=code_fingerprint,
            fingerprint=fingerprint,
            error_kind=error_kind,
            reason=reason,
            message=message,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            exit_code=exit_code,
            submit_attempt_recorder=submit_attempt_recorder,
            save_run_state=lambda updates: self.save_run_state_for_run(self.run_dir, updates),
            resolve_submit_abort_artifact_path=self.resolve_submit_abort_artifact_path,
            persist_submit_abort_failure=self.persist_submit_abort_failure,
            load_run_state=self.load_run_state,
            load_latest_submit_attempt=self.load_latest_submit_attempt,
            has_successful_submit_attempt=self.has_successful_submit_attempt,
            compute_submission_sha256=self.compute_submission_sha256,
            stdout_tail_chars=self.stdout_tail_chars,
            stderr_tail_chars=self.stderr_tail_chars,
            now_iso=self.now_iso(),
            normalize_detail=self.normalize_detail,
            record_error_fix_insight=self.record_error_fix_insight,
            on_message=self.on_message,
            build_error=self.build_error,
        )


def build_submit_run_aborter_for_run(
    *,
    run_dir: Path,
    run_id: str,
    slug: str,
    knowledge_paths: object,
    problem_types: list[str],
    save_run_state_for_run: Callable[[Path, dict[str, object]], object],
    load_run_state: Callable[[Path], dict[str, object]],
    compute_submission_sha256: Callable[[Path | None], str | None],
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    now_iso: Callable[[], str],
    normalize_detail: Callable[..., str],
    record_error_fix_insight: Callable[..., object],
    on_message: Callable[[str], object],
    build_error: Callable[[str], BaseException],
) -> SubmitRunAborter:
    return SubmitRunAborter(
        run_dir=run_dir,
        run_id=run_id,
        slug=slug,
        knowledge_paths=knowledge_paths,
        problem_types=problem_types,
        save_run_state_for_run=save_run_state_for_run,
        resolve_submit_abort_artifact_path=_submit_failure_context.resolve_submit_abort_artifact_path,
        persist_submit_abort_failure=_submit_failure_context.persist_submit_abort_failure,
        load_run_state=load_run_state,
        load_latest_submit_attempt=_submit_attempts.load_latest_submit_attempt,
        has_successful_submit_attempt=_submit_attempts.has_successful_submit_attempt,
        compute_submission_sha256=compute_submission_sha256,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
        now_iso=now_iso,
        normalize_detail=normalize_detail,
        record_error_fix_insight=record_error_fix_insight,
        on_message=on_message,
        build_error=build_error,
    )


@dataclass(frozen=True)
class SubmitRunRetryRecorder:
    submit_attempt_recorder: object
    run_id: str
    slug: str
    problem_types: list[str]
    knowledge_paths: object
    compute_submission_sha256: Callable[[Path | None], str | None]
    stdout_tail_chars: int
    stderr_tail_chars: int
    normalize_detail: Callable[..., str]
    record_error_fix_insight: Callable[..., object]

    def record(
        self,
        *,
        submission_ref: str,
        submission_artifact_path: Path | None,
        fallback_submission_path: Path,
        exit_code: int | None,
        fingerprint: str,
        action: SubmitStageErrorActionDecision,
        stdout: str,
        stderr: str,
        attempt: int,
    ) -> bool:
        return record_submit_stage_retry_attempt(
            submit_attempt_recorder=self.submit_attempt_recorder,
            run_id=self.run_id,
            slug=self.slug,
            problem_types=self.problem_types,
            submission_ref=submission_ref,
            submission_artifact_path=submission_artifact_path,
            fallback_submission_path=fallback_submission_path,
            compute_submission_sha256=self.compute_submission_sha256,
            exit_code=exit_code,
            fingerprint=fingerprint,
            action=action,
            stdout=stdout,
            stderr=stderr,
            attempt=attempt,
            stdout_tail_chars=self.stdout_tail_chars,
            stderr_tail_chars=self.stderr_tail_chars,
            knowledge_paths=self.knowledge_paths,
            normalize_detail=self.normalize_detail,
            record_error_fix_insight=self.record_error_fix_insight,
        )


@dataclass(frozen=True)
class SubmitRunContext:
    submit_attempt_recorder: object
    autofix_attempt_context: _submit_failure_context.SubmitAutofixAttemptContext
    submit_code_fingerprint: str
    allow_force: bool
    input_submission_path: Path
    run_state: dict[str, object]
    latest_submit_attempt: dict[str, object]
    submit_aborter: SubmitRunAborter
    submit_retry_recorder: SubmitRunRetryRecorder


@dataclass(frozen=True)
class SubmitRuntimeContext:
    message: str
    submission_service: SubmissionService
    submitted_at: datetime


def build_submit_run_context(
    *,
    run_dir: Path,
    run_id: str,
    slug: str,
    submission_path: Path,
    src_root: Path,
    kernel_source_dir: Path,
    knowledge_paths: object,
    problem_types: list[str],
    force_submit: bool,
    force_resubmit: bool,
    save_run_state_for_run: Callable[[Path, dict[str, object]], object],
    load_run_state: Callable[[Path], dict[str, object]],
    compute_submit_code_fingerprint: Callable[..., str],
    compute_submission_sha256: Callable[[Path | None], str | None],
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    now_iso: Callable[[], str],
    normalize_detail: Callable[..., str],
    record_error_fix_insight: Callable[..., object],
    on_message: Callable[[str], object],
    build_error: Callable[[str], BaseException],
) -> SubmitRunContext:
    submit_attempt_recorder = _submit_attempts.build_submit_attempt_recorder_for_run(
        run_dir=run_dir,
        save_run_state_for_run=save_run_state_for_run,
    )
    autofix_attempt_context = _submit_failure_context.resolve_submit_autofix_context_for_run(
        run_dir=run_dir,
        submission_path=submission_path,
        load_run_state=load_run_state,
        save_run_state_for_run=save_run_state_for_run,
        now_iso=now_iso(),
    )
    if autofix_attempt_context.message:
        on_message(autofix_attempt_context.message)
    submit_code_fingerprint = compute_submit_code_fingerprint(
        src_root=src_root,
        kernel_source_dir=kernel_source_dir,
        sha256_or_none=compute_submission_sha256,
    )
    submit_aborter = build_submit_run_aborter_for_run(
        run_dir=run_dir,
        run_id=run_id,
        slug=slug,
        knowledge_paths=knowledge_paths,
        problem_types=problem_types,
        save_run_state_for_run=save_run_state_for_run,
        load_run_state=load_run_state,
        compute_submission_sha256=compute_submission_sha256,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
        now_iso=now_iso,
        normalize_detail=normalize_detail,
        record_error_fix_insight=record_error_fix_insight,
        on_message=on_message,
        build_error=build_error,
    )
    submit_retry_recorder = SubmitRunRetryRecorder(
        submit_attempt_recorder=submit_attempt_recorder,
        run_id=run_id,
        slug=slug,
        problem_types=problem_types,
        knowledge_paths=knowledge_paths,
        compute_submission_sha256=compute_submission_sha256,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
        normalize_detail=normalize_detail,
        record_error_fix_insight=record_error_fix_insight,
    )
    return SubmitRunContext(
        submit_attempt_recorder=submit_attempt_recorder,
        autofix_attempt_context=autofix_attempt_context,
        submit_code_fingerprint=submit_code_fingerprint,
        allow_force=force_submit or force_resubmit,
        input_submission_path=autofix_attempt_context.input_submission_path,
        run_state=autofix_attempt_context.run_state,
        latest_submit_attempt=autofix_attempt_context.latest_submit_attempt,
        submit_aborter=submit_aborter,
        submit_retry_recorder=submit_retry_recorder,
    )


def build_submit_runtime_context(
    *,
    slug: str,
    context_dir: Path,
    run_id: str,
    best_score: float | None,
    explicit_message: str | None,
    submission_path: Path,
    campaign_mode: str,
    target_direction: str,
    data_dir: Path,
    sample_submission_path: Path,
    submission_ledger_path: Path,
    dry_run: bool,
    force_submit: bool,
    now: Callable[[], datetime],
    on_message: Callable[[str], object],
) -> SubmitRuntimeContext:
    message = _submit_message.resolve_submission_message(
        context_dir=context_dir,
        run_id=run_id,
        best_score=best_score,
        explicit_message=explicit_message,
        submission_path=submission_path,
        campaign_mode=campaign_mode,
        target_direction=target_direction,
    )
    submission_service = SubmissionService(
        SubmissionConfig(
            slug=slug,
            data_dir=data_dir,
            sample_submission_path=sample_submission_path,
            submission_ledger_path=submission_ledger_path,
            dry_run=dry_run,
            force_submit=force_submit,
            bypass_rate_limit=False,
        )
    )
    on_message(f"[cyan]submit[/cyan]: {slug}")
    return SubmitRuntimeContext(
        message=message,
        submission_service=submission_service,
        submitted_at=now(),
    )


@dataclass(frozen=True)
class SubmitPreparedSubmissionResolution:
    prepared_submission_path: Path | None
    abort_spec: SubmitAbortSpec | None = None


@dataclass(frozen=True)
class SubmitPreparedRunContext:
    prepared_submission_path: Path
    prepared_submission_sha: str


@dataclass(frozen=True)
class SubmitPreflightContext:
    duplicate_skip_result: dict[str, object] | None
    same_submission_path_skipped: bool
    submit_stage_state: SubmitStageRuntimeState | None
    code_competition: bool
    seen_fingerprints: set[str]


@dataclass(frozen=True)
class SubmitPreparedPreflightContext:
    prepared_context: SubmitPreparedRunContext
    preflight_context: SubmitPreflightContext


@dataclass(frozen=True)
class SubmitRulesAcceptanceResolution:
    rules_accepted: bool
    abort_spec: SubmitAbortSpec | None = None


def resolve_prepared_submission_for_submit(
    *,
    input_submission_path: Path,
    validate_and_prepare: Callable[[Path], Path],
    validation_error_types: tuple[type[BaseException], ...],
    validation_exit_code: int | None,
    compute_error_fingerprint: Callable[[str, str], str],
) -> SubmitPreparedSubmissionResolution:
    try:
        return SubmitPreparedSubmissionResolution(prepared_submission_path=validate_and_prepare(input_submission_path))
    except validation_error_types as exc:
        return SubmitPreparedSubmissionResolution(
            prepared_submission_path=None,
            abort_spec=_submit_abort_specs.build_local_submission_validation_abort_spec(
                error=exc,
                exit_code=validation_exit_code,
                compute_error_fingerprint=compute_error_fingerprint,
            ),
        )


def require_prepared_submission_path(
    resolution: SubmitPreparedSubmissionResolution,
    *,
    build_error: Callable[[str], BaseException],
) -> Path:
    if resolution.prepared_submission_path is None:
        raise build_error("Submit validation did not produce a prepared submission path.")
    return resolution.prepared_submission_path


def prepare_submission_for_run_or_abort(
    *,
    input_submission_path: Path,
    validate_and_prepare: Callable[[Path], Path],
    validation_error_types: tuple[type[BaseException], ...],
    validation_exit_code: int | None,
    compute_error_fingerprint: Callable[[str, str], str],
    submit_aborter: object,
    submit_attempt_recorder: object,
    code_fingerprint: str,
    compute_submission_sha256: Callable[[Path | None], str | None],
    build_error: Callable[[str], BaseException],
) -> SubmitPreparedRunContext:
    prepared_resolution = resolve_prepared_submission_for_submit(
        input_submission_path=input_submission_path,
        validate_and_prepare=validate_and_prepare,
        validation_error_types=validation_error_types,
        validation_exit_code=validation_exit_code,
        compute_error_fingerprint=compute_error_fingerprint,
    )
    if prepared_resolution.abort_spec is not None:
        return submit_aborter.abort(
            submission_ref=input_submission_path,
            code_fingerprint=code_fingerprint,
            **_submit_abort_specs.build_submit_abort_spec_kwargs(prepared_resolution.abort_spec),
            submit_attempt_recorder=submit_attempt_recorder,
        )
    prepared_submission_path = require_prepared_submission_path(
        prepared_resolution,
        build_error=build_error,
    )
    return SubmitPreparedRunContext(
        prepared_submission_path=prepared_submission_path,
        prepared_submission_sha=str(compute_submission_sha256(prepared_submission_path) or "").strip(),
    )


def resolve_rules_acceptance_for_submit(
    *,
    check_rules_accepted: Callable[[], bool],
    cli_error_types: tuple[type[BaseException], ...],
    is_missing_credentials_error: Callable[[BaseException], bool],
    rules_not_accepted_exit_code: int | None,
    compute_error_fingerprint: Callable[[str, str], str],
) -> SubmitRulesAcceptanceResolution:
    try:
        rules_accepted = check_rules_accepted()
    except cli_error_types as exc:
        if not is_missing_credentials_error(exc):
            raise
        return SubmitRulesAcceptanceResolution(
            rules_accepted=False,
            abort_spec=_submit_abort_specs.build_kaggle_credentials_missing_abort_spec(
                stdout=str(getattr(exc, "stdout", "") or ""),
                stderr=str(getattr(exc, "stderr", "") or ""),
                output=str(getattr(exc, "output", "") or ""),
                exit_code=getattr(exc, "exit_code", None),
                compute_error_fingerprint=compute_error_fingerprint,
            ),
        )

    if not rules_accepted:
        return SubmitRulesAcceptanceResolution(
            rules_accepted=False,
            abort_spec=_submit_abort_specs.build_rules_not_accepted_abort_spec(
                exit_code=rules_not_accepted_exit_code,
                compute_error_fingerprint=compute_error_fingerprint,
            ),
        )
    return SubmitRulesAcceptanceResolution(rules_accepted=True)


@dataclass(frozen=True)
class SubmitOutcomeResolution:
    outcome: object
    abort_spec: SubmitAbortSpec | None = None


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
    count_csv_data_rows: Callable[[Path], int | None],
    decide_same_submission_path_action: Callable[..., object],
    compute_error_fingerprint: Callable[[str, str], str],
    compute_submission_sha256: Callable[[Path | None], str | None],
    submit_aborter: object,
    submit_attempt_recorder: object,
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    on_message: Callable[[str], object],
) -> SubmitPreflightContext:
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

    rules_resolution = resolve_rules_acceptance_for_submit(
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
        count_csv_data_rows=count_csv_data_rows,
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
    count_csv_data_rows: Callable[[Path], int | None],
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
    prepared_context = prepare_submission_for_run_or_abort(
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
        count_csv_data_rows=count_csv_data_rows,
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


def record_submit_stage_retry_attempt(
    *,
    submit_attempt_recorder: object,
    run_id: str,
    slug: str,
    problem_types: list[str],
    submission_ref: str,
    submission_artifact_path: Path | None,
    fallback_submission_path: Path,
    compute_submission_sha256: Callable[[Path | None], str | None],
    exit_code: int | None,
    fingerprint: str,
    action: SubmitStageErrorActionDecision,
    stdout: str,
    stderr: str,
    attempt: int,
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    knowledge_paths: object,
    normalize_detail: Callable[..., str],
    record_error_fix_insight: Callable[..., object],
) -> bool:
    return _submit_attempts.record_submit_retry_attempt_and_knowledge(
        submit_attempt_recorder=submit_attempt_recorder,
        run_id=run_id,
        slug=slug,
        problem_types=problem_types,
        submission_ref=submission_ref,
        submission_path=submission_artifact_path or fallback_submission_path,
        submission_sha256=compute_submission_sha256(submission_artifact_path),
        exit_code=exit_code,
        fingerprint=fingerprint,
        reason=action.reason,
        stdout=stdout,
        stderr=stderr,
        attempt=attempt,
        wait_seconds=action.wait_seconds,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
        knowledge_paths=knowledge_paths,
        infer_iteration=_submit_stage_duplicate.infer_iteration_from_submission_path,
        normalize_detail=normalize_detail,
        record_error_fix_insight=record_error_fix_insight,
    )


def record_submit_abort_for_run(
    *,
    run_dir: Path,
    run_id: str,
    slug: str,
    knowledge_paths: object,
    problem_types: list[str],
    submission_ref: str | Path,
    submission_artifact_path: Path | None,
    artifact_mode: str | None,
    code_fingerprint: str,
    fingerprint: str,
    error_kind: str,
    reason: str,
    message: str,
    stdout_tail: str,
    stderr_tail: str,
    exit_code: int | None,
    submit_attempt_recorder: object,
    resolve_submit_abort_artifact_path: Callable[..., Path | None],
    persist_submit_abort_failure: Callable[..., object],
    load_run_state: Callable[[Path], dict[str, object]],
    load_latest_submit_attempt: Callable[[Path], dict[str, object]],
    has_successful_submit_attempt: Callable[[Path], bool],
    compute_submission_sha256: Callable[[Path | None], str | None],
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    now_iso: str,
    normalize_detail: Callable[..., str],
    record_error_fix_insight: Callable[..., object],
    on_message: Callable[[str], object],
) -> None:
    submission_ref_text = str(submission_ref)
    artifact_path = resolve_submit_abort_artifact_path(
        submission_ref=submission_ref,
        submission_artifact_path=submission_artifact_path,
    )
    prior_state = load_run_state(run_dir)
    prior_submit_ok = bool(prior_state.get("submit_ok")) or has_successful_submit_attempt(run_dir)
    persist_submit_abort_failure(
        run_dir=run_dir,
        run_id=run_id,
        submission_ref=submission_ref_text,
        submission_sha256=compute_submission_sha256(artifact_path),
        artifact_path=artifact_path,
        artifact_mode=artifact_mode,
        code_fingerprint=code_fingerprint,
        fingerprint=fingerprint,
        error_kind=error_kind,
        reason=reason,
        message=message,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        exit_code=exit_code,
        prior_state=prior_state,
        prior_submit_ok=prior_submit_ok,
        submit_attempt_recorder=submit_attempt_recorder,
        load_latest_submit_attempt=load_latest_submit_attempt,
        load_run_state=load_run_state,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
        now_iso=now_iso,
    )
    _submit_attempts.record_submit_reason_knowledge(
        knowledge_paths=knowledge_paths,
        slug=slug,
        run_id=run_id,
        problem_types=problem_types,
        submission_path=artifact_path or Path(submission_ref_text),
        error_kind=error_kind,
        reason=reason,
        action_taken="abort",
        fingerprint=fingerprint,
        details=message,
        infer_iteration=_submit_stage_duplicate.infer_iteration_from_submission_path,
        normalize_detail=normalize_detail,
        record_error_fix_insight=record_error_fix_insight,
    )
    on_message(f"[red]submit aborted[/red]: {message}")


def abort_submit_for_run(
    *,
    run_dir: Path,
    run_id: str,
    slug: str,
    knowledge_paths: object,
    problem_types: list[str],
    submission_ref: str | Path,
    submission_artifact_path: Path | None,
    artifact_mode: str | None,
    code_fingerprint: str | None,
    fingerprint: str,
    error_kind: str,
    reason: str,
    message: str,
    stdout_tail: str,
    stderr_tail: str,
    exit_code: int | None,
    submit_attempt_recorder: object | None,
    save_run_state: Callable[[dict[str, object]], object],
    resolve_submit_abort_artifact_path: Callable[..., Path | None],
    persist_submit_abort_failure: Callable[..., object],
    load_run_state: Callable[[Path], dict[str, object]],
    load_latest_submit_attempt: Callable[[Path], dict[str, object]],
    has_successful_submit_attempt: Callable[[Path], bool],
    compute_submission_sha256: Callable[[Path | None], str | None],
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    now_iso: str,
    normalize_detail: Callable[..., str],
    record_error_fix_insight: Callable[..., object],
    on_message: Callable[[str], object],
    build_error: Callable[[str], BaseException],
) -> None:
    if submit_attempt_recorder is None:
        submit_attempt_recorder = _submit_attempts.SubmitAttemptRecorder(
            run_dir=run_dir,
            save_run_state=save_run_state,
        )
    record_submit_abort_for_run(
        run_dir=run_dir,
        run_id=run_id,
        slug=slug,
        knowledge_paths=knowledge_paths,
        problem_types=problem_types,
        submission_ref=submission_ref,
        submission_artifact_path=submission_artifact_path,
        artifact_mode=artifact_mode,
        code_fingerprint=code_fingerprint or "",
        fingerprint=fingerprint,
        error_kind=error_kind,
        reason=reason,
        message=message,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        exit_code=exit_code,
        submit_attempt_recorder=submit_attempt_recorder,
        resolve_submit_abort_artifact_path=resolve_submit_abort_artifact_path,
        persist_submit_abort_failure=persist_submit_abort_failure,
        load_run_state=load_run_state,
        load_latest_submit_attempt=load_latest_submit_attempt,
        has_successful_submit_attempt=has_successful_submit_attempt,
        compute_submission_sha256=compute_submission_sha256,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
        now_iso=now_iso,
        normalize_detail=normalize_detail,
        record_error_fix_insight=record_error_fix_insight,
        on_message=on_message,
    )
    raise build_error(message)


def wait_for_submission_outcome(
    *,
    slug: str,
    message: str,
    submitted_at: datetime,
    fetch_submission_rows: Callable[[str], list[dict[str, str]]],
    max_attempts: int | None,
    poll_interval_sec: float,
    max_fetch_errors: int,
) -> dict[str, object] | None:
    print(f"[cyan]submission polling[/cyan]: waiting for result (interval={poll_interval_sec:.0f}s)")
    service = SubmissionOutcomeService(
        fetch_rows=fetch_submission_rows,
        max_attempts=max_attempts,
        poll_interval_sec=poll_interval_sec,
        max_fetch_errors=max_fetch_errors,
    )
    return service.wait_for_outcome(
        slug=slug,
        message=message,
        submitted_at=submitted_at,
    )


def resolve_submission_outcome_after_submit(
    *,
    slug: str,
    message: str,
    submitted_at: datetime,
    deliverable_mode: str,
    fetch_submission_rows: Callable[[str], list[dict[str, str]]],
    max_attempts: int | None,
    poll_interval_sec: float,
    max_fetch_errors: int,
    normalize_detail: Callable[[str], str],
    compute_error_fingerprint: Callable[[str, str], str],
) -> SubmitOutcomeResolution:
    try:
        outcome = wait_for_submission_outcome(
            slug=slug,
            message=message,
            submitted_at=submitted_at,
            fetch_submission_rows=fetch_submission_rows,
            max_attempts=max_attempts,
            poll_interval_sec=poll_interval_sec,
            max_fetch_errors=max_fetch_errors,
        )
    except SubmissionOutcomePollingError as exc:
        return SubmitOutcomeResolution(
            outcome=None,
            abort_spec=_submit_abort_specs.build_submission_polling_error_abort_spec(
                error=exc,
                detail=exc.detail,
                normalize_detail=normalize_detail,
                compute_error_fingerprint=compute_error_fingerprint,
            ),
        )

    outcome_post_poll = _submit_outcome_decisions.evaluate_submission_outcome_after_poll(
        slug=slug,
        message=message,
        submitted_at=submitted_at,
        outcome=outcome,
        deliverable_mode=deliverable_mode,
        fetch_submission_rows=fetch_submission_rows,
        normalize_detail=normalize_detail,
    )
    if outcome_post_poll.abort_decision.should_abort:
        return SubmitOutcomeResolution(
            outcome=outcome_post_poll.outcome,
            abort_spec=_submit_abort_specs.build_submission_outcome_abort_spec(
                decision=outcome_post_poll.abort_decision,
                compute_error_fingerprint=compute_error_fingerprint,
            ),
        )
    return SubmitOutcomeResolution(outcome=outcome_post_poll.outcome)


def finalize_submit_outcome_for_run_or_abort(
    *,
    run_dir: Path,
    submission_ledger_path: Path,
    slug: str,
    run_id: str,
    message: str,
    submitted_at: datetime,
    submission_ref: str,
    submission_result: object,
    source_submission_path: Path,
    submission_artifact_path: Path | None,
    submit_stage_state: SubmitStageRuntimeState,
    code_fingerprint: str,
    deliverable_mode: str,
    fetch_submission_rows: Callable[[str], list[dict[str, str]]],
    max_attempts: int | None,
    poll_interval_sec: float,
    max_fetch_errors: int,
    normalize_detail: Callable[[str], str],
    compute_error_fingerprint: Callable[[str, str], str],
    compute_submission_sha256: Callable[[Path | None], str | None],
    load_run_state: Callable[[Path], dict[str, object]],
    record_submit_attempt_payloads: Callable[[object], object],
    submit_aborter: object,
    submit_attempt_recorder: object,
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    on_message: Callable[[str], object],
) -> dict[str, object]:
    outcome_resolution = resolve_submission_outcome_after_submit(
        slug=slug,
        message=message,
        submitted_at=submitted_at,
        deliverable_mode=deliverable_mode,
        fetch_submission_rows=fetch_submission_rows,
        max_attempts=max_attempts,
        poll_interval_sec=poll_interval_sec,
        max_fetch_errors=max_fetch_errors,
        normalize_detail=normalize_detail,
        compute_error_fingerprint=compute_error_fingerprint,
    )
    if outcome_resolution.abort_spec is not None:
        return submit_aborter.abort(
            submission_ref=submission_ref,
            submission_artifact_path=submission_artifact_path,
            artifact_mode=submit_stage_state.submission_artifact_mode,
            code_fingerprint=code_fingerprint,
            **_submit_abort_specs.build_submit_abort_spec_kwargs(outcome_resolution.abort_spec),
            submit_attempt_recorder=submit_attempt_recorder,
        )
    return record_successful_submit_for_run(
        run_dir=run_dir,
        submission_ledger_path=submission_ledger_path,
        slug=slug,
        run_id=run_id,
        message=message,
        submitted_at=submitted_at,
        submission_ref=submission_ref,
        submission_result=submission_result,
        submission_path=source_submission_path,
        submission_artifact_path=submission_artifact_path,
        outcome=outcome_resolution.outcome,
        code_fingerprint=code_fingerprint,
        load_run_state=load_run_state,
        compute_error_fingerprint=compute_error_fingerprint,
        compute_submission_sha256=compute_submission_sha256,
        record_submit_attempt_payloads=record_submit_attempt_payloads,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
        on_message=on_message,
    )


def build_submit_stage_success_record(
    *,
    submission_result: object,
    compute_error_fingerprint: Callable[[str, str], str],
) -> SubmitStageSuccessRecord:
    stdout = str(getattr(submission_result, "stdout", "") or "")
    stderr = str(getattr(submission_result, "stderr", "") or "")
    return SubmitStageSuccessRecord(
        exit_code=getattr(submission_result, "exit_code", getattr(submission_result, "returncode", None)),
        fingerprint=compute_error_fingerprint(stdout, stderr),
        stdout=stdout,
        stderr=stderr,
    )


def record_successful_submit_stage_result(
    *,
    run_id: str,
    message: str,
    submitted_at: datetime,
    submission_ref: str,
    submission_result: object,
    submission_path: Path,
    submission_artifact_path: Path | None,
    outcome: object,
    code_fingerprint: str,
    prior_state: dict[str, object],
    compute_error_fingerprint: Callable[[str, str], str],
    compute_submission_sha256: Callable[[Path | None], str | None],
    record_submit_attempt_payloads: Callable[[object], object],
    record_outcome: Callable[[Path, dict[str, object]], object],
    mark_failure_context_submitted: Callable[[str], object],
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    on_message: Callable[[str], object],
) -> dict[str, object]:
    submit_success_record = build_submit_stage_success_record(
        submission_result=submission_result,
        compute_error_fingerprint=compute_error_fingerprint,
    )
    submit_success_payloads = _submit_attempts.build_submit_success_record_payloads(
        run_id=run_id,
        submission_ref=submission_ref,
        submission_sha256=compute_submission_sha256(submission_artifact_path),
        exit_code=submit_success_record.exit_code,
        fingerprint=submit_success_record.fingerprint,
        code_fingerprint=code_fingerprint,
        stdout=submit_success_record.stdout,
        stderr=submit_success_record.stderr,
        prior_state=prior_state,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
    )
    record_submit_attempt_payloads(submit_success_payloads)
    on_message("[green]submission recorded[/green]")

    outcome_recording = _submit_attempts.decide_submit_outcome_recording(
        outcome=outcome,
        submission_artifact_exists=bool(submission_artifact_path is not None and submission_artifact_path.exists()),
    )
    on_message(outcome_recording.message)
    _submit_attempts.record_submit_outcome_if_available(
        decision=outcome_recording,
        submission_path=submission_artifact_path,
        record_outcome=record_outcome,
    )
    mark_failure_context_submitted(submission_ref)
    return _submit_attempts.build_successful_submit_result_payload(
        message=message,
        submission_ref=submission_ref,
        submitted_at=submitted_at,
        submission_path=submission_path,
        outcome=outcome,
        infer_iteration=_submit_stage_duplicate.infer_iteration_from_submission_path,
    )


def record_successful_submit_for_run(
    *,
    run_dir: Path,
    submission_ledger_path: Path,
    slug: str,
    run_id: str,
    message: str,
    submitted_at: datetime,
    submission_ref: str,
    submission_result: object,
    submission_path: Path,
    submission_artifact_path: Path | None,
    outcome: object,
    code_fingerprint: str,
    load_run_state: Callable[[Path], dict[str, object]],
    compute_error_fingerprint: Callable[[str, str], str],
    compute_submission_sha256: Callable[[Path | None], str | None],
    record_submit_attempt_payloads: Callable[[object], object],
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    on_message: Callable[[str], object],
) -> dict[str, object]:
    def record_outcome(path: Path, ledger_outcome: dict[str, object]) -> None:
        SubmissionLedger(submission_ledger_path).record_outcome(
            slug=slug,
            message=message,
            submission_path=path,
            run_id=run_id,
            outcome=ledger_outcome,
        )

    def mark_failure_context_submitted(submitted_ref: str) -> None:
        _submit_failure_context.mark_submit_failure_context_submitted(
            run_dir=run_dir,
            submission_ref=submitted_ref,
        )

    return record_successful_submit_stage_result(
        run_id=run_id,
        message=message,
        submitted_at=submitted_at,
        submission_ref=submission_ref,
        submission_result=submission_result,
        submission_path=submission_path,
        submission_artifact_path=submission_artifact_path,
        outcome=outcome,
        code_fingerprint=code_fingerprint,
        prior_state=load_run_state(run_dir),
        compute_error_fingerprint=compute_error_fingerprint,
        compute_submission_sha256=compute_submission_sha256,
        record_submit_attempt_payloads=record_submit_attempt_payloads,
        record_outcome=record_outcome,
        mark_failure_context_submitted=mark_failure_context_submitted,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
        on_message=on_message,
    )


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
    submit_aborter: SubmitRunAborter,
    submit_attempt_recorder: object,
    submit_retry_recorder: SubmitRunRetryRecorder,
    submission_cli_error_types: tuple[type[BaseException], ...],
    local_guardrail_error_types: tuple[type[BaseException], ...],
    kaggle_cli_error_types: tuple[type[BaseException], ...],
    classify_submit_error: Callable[..., object],
    should_use_notebook_fallback: Callable[..., bool],
    resolve_notebook_submit_artifact_mode: Callable[..., str],
    decide_notebook_submit_artifact_mode_for_paths: Callable[..., object],
    count_csv_data_rows: Callable[..., int | None],
    compute_error_fingerprint: Callable[..., str],
    decide_submit_fingerprint_reuse: Callable[..., object],
    compute_submit_backoff: Callable[..., float],
    save_run_state_for_run: Callable[[Path, dict[str, object]], object],
    is_missing_credentials_error: Callable[[object], bool],
    build_submit_aborted_error: Callable[[str], BaseException],
    sleep: Callable[[float], object],
    on_message: Callable[[str], object],
) -> SubmitStageAttemptLoopResult:
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
                count_csv_data_rows=count_csv_data_rows,
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
