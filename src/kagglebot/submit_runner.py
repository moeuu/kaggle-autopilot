from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Protocol

from kagglebot import submission_fidelity as _submission_fidelity
from kagglebot import submit_attempt_loop as _submit_attempt_loop
from kagglebot import submit_context as _submit_context
from kagglebot import submit_outcome as _submit_outcome
from kagglebot import submit_stage as _submit_stage
from kagglebot import submit_stage_duplicate as _submit_stage_duplicate
from kagglebot.exceptions import (
    DuplicateSubmissionError,
    KaggleCliError,
    RulesNotAcceptedError,
    SubmissionCliError,
    SubmissionRateLimitError,
    SubmissionValidationError,
)
from kagglebot.json_utils import load_json_object
from kagglebot.submission_format import load_submission_format_hint
from kagglebot.submission_output_naming import (
    all_submission_output_suffixes,
    output_filename_from_format_text,
    tabular_submission_output_suffixes,
)
from kagglebot.submission_sample_discovery import (
    TABULAR_INPUT_SUFFIXES,
    find_usable_sample_submissions,
    tabular_suffix,
)
from kagglebot.submission_semantics import (
    discover_submission_metrics_path,
    validate_autopilot_submission_semantics,
)

_EXPECTED_NOTEBOOK_OUTPUT_SUFFIXES = all_submission_output_suffixes()
_EXPECTED_NOTEBOOK_SAMPLE_OUTPUT_SUFFIXES = tabular_submission_output_suffixes()


class SubmitRunnerPaths(Protocol):
    kernel_source_dir: Path
    context_dir: Path
    data_dir: Path
    sample_submission_path: Path
    submission_ledger_path: Path

    def run_dir(self, run_id: str) -> Path: ...


class SubmitRunnerConfig(Protocol):
    submit: bool
    dry_run: bool
    slug: str
    paths: SubmitRunnerPaths
    knowledge_paths: object
    force_submit: bool
    message: str | None
    campaign_mode: str
    target_direction: str | None
    kaggle_username: str | None
    kernel_name: str | None
    accelerator: str
    strict_accelerator: bool
    time_budget_min: int | None


@dataclass(frozen=True)
class SubmitRunnerDependencies:
    load_competition_rule_constraints: Callable[[SubmitRunnerPaths], object]
    env_truthy: Callable[[str], bool]
    load_run_state: Callable[[Path], dict[str, object]]
    save_run_state: Callable[[Path, dict[str, object]], object]
    compute_submit_code_fingerprint: Callable[..., str | None]
    compute_submission_sha256: Callable[[Path | None], str | None]
    now_iso: Callable[[], str]
    now_datetime: Callable[[], datetime]
    normalize_error_text: Callable[..., str]
    record_error_fix_insight: Callable[..., object]
    build_error: Callable[[str], BaseException]
    check_rules_accepted: Callable[..., bool]
    infer_code_competition_from_paths: Callable[[SubmitRunnerPaths], bool]
    collect_duplicate_submission_sources: Callable[..., list[str]]
    decide_duplicate_submission_action: Callable[..., object]
    decide_same_submission_path_action: Callable[..., object]
    resolve_notebook_submit_artifact_mode: Callable[..., str]
    decide_notebook_submit_artifact_mode_for_paths: Callable[..., object]
    count_tabular_data_rows: Callable[[Path], int | None]
    resolve_kaggle_username: Callable[..., str]
    run_submit_kernel: Callable[..., object]
    run_kaggle_submit_kernel: Callable[..., object]
    review_code_submission: Callable[..., object]
    recheck_code_submission_guard: Callable[..., object]
    record_code_submission_execution: Callable[..., object]
    copy_submission_artifact_to_iteration_dir: Callable[..., object]
    classify_submit_error: Callable[..., dict[str, object]]
    should_retry_ambiguous_notebook_submit_error: Callable[..., bool]
    should_use_notebook_submit_fallback: Callable[..., bool]
    compute_error_fingerprint: Callable[..., str]
    decide_submit_fingerprint_reuse: Callable[..., object]
    compute_submit_backoff: Callable[..., float]
    is_missing_kaggle_credentials_error: Callable[[object], bool]
    deliverable_mode: Callable[[SubmitRunnerPaths], str]
    list_competition_submissions: Callable[..., list[dict[str, object]]]
    sleep: Callable[[float], object]
    on_message: Callable[[str], object]


@dataclass(frozen=True)
class SubmitRunnerLimits:
    stdout_tail_chars: int
    stderr_tail_chars: int
    max_transient_retries: int
    backoff_base_sec: float
    poll_max_attempts: int
    poll_interval_sec: float
    poll_max_fetch_errors: int


def attempt_submit_for_run(
    *,
    config: SubmitRunnerConfig,
    run_id: str,
    submission_path: Path,
    best_score: float | None,
    problem_types: list[str],
    submit_mode: str = "file",
    notebook_submit_artifact_mode: str = "wrapper",
    deps: SubmitRunnerDependencies,
    limits: SubmitRunnerLimits,
) -> dict[str, object] | None:
    if not config.submit or config.dry_run:
        return None
    run_dir = config.paths.run_dir(run_id)
    submit_run_context = _submit_context.build_submit_run_context(
        run_dir=run_dir,
        run_id=run_id,
        slug=config.slug,
        submission_path=submission_path,
        src_root=Path(__file__).resolve().parent,
        kernel_source_dir=config.paths.kernel_source_dir,
        knowledge_paths=config.knowledge_paths,
        problem_types=problem_types,
        force_submit=config.force_submit,
        force_resubmit=deps.env_truthy("KAGGLEBOT_FORCE_RESUBMIT"),
        load_run_state=deps.load_run_state,
        save_run_state_for_run=deps.save_run_state,
        compute_submit_code_fingerprint=deps.compute_submit_code_fingerprint,
        compute_submission_sha256=deps.compute_submission_sha256,
        stdout_tail_chars=limits.stdout_tail_chars,
        stderr_tail_chars=limits.stderr_tail_chars,
        now_iso=deps.now_iso,
        normalize_detail=deps.normalize_error_text,
        record_error_fix_insight=deps.record_error_fix_insight,
        on_message=deps.on_message,
        build_error=deps.build_error,
    )
    submit_attempt_recorder = submit_run_context.submit_attempt_recorder
    run_state = submit_run_context.run_state
    latest_submit_attempt = submit_run_context.latest_submit_attempt
    submit_code_fingerprint = submit_run_context.submit_code_fingerprint
    allow_force = submit_run_context.allow_force
    input_submission_path = submit_run_context.input_submission_path
    active_quarantine = _load_current_quarantine_state(
        run_state=run_state,
        submission_ledger_path=config.paths.submission_ledger_path,
        slug=config.slug,
    )
    if active_quarantine is not None and run_state.get(_submission_fidelity.QUARANTINE_STATE_KEY) != active_quarantine:
        deps.save_run_state(run_dir, {_submission_fidelity.QUARANTINE_STATE_KEY: active_quarantine})

    submit_runtime_context = _submit_context.build_submit_runtime_context(
        slug=config.slug,
        context_dir=config.paths.context_dir,
        run_id=run_id,
        best_score=best_score,
        explicit_message=config.message,
        submission_path=input_submission_path,
        campaign_mode=config.campaign_mode,
        target_direction=config.target_direction,
        data_dir=config.paths.data_dir,
        sample_submission_path=config.paths.sample_submission_path,
        submission_ledger_path=config.paths.submission_ledger_path,
        dry_run=config.dry_run,
        force_submit=config.force_submit,
        now=deps.now_datetime,
        on_message=deps.on_message,
    )
    message = submit_runtime_context.message
    submission_service = submit_runtime_context.submission_service
    submitted_at = submit_runtime_context.submitted_at

    submit_aborter = submit_run_context.submit_aborter
    submit_retry_recorder = submit_run_context.submit_retry_recorder

    constraints = deps.load_competition_rule_constraints(config.paths)
    fallback_sample_submission_path = _resolve_fallback_sample_submission_path(config.paths)
    prepared_preflight_context = _submit_stage.prepare_and_resolve_submit_preflight_for_run_or_abort(
        run_dir=run_dir,
        submission_ledger_path=config.paths.submission_ledger_path,
        slug=config.slug,
        run_id=run_id,
        message=message,
        submitted_at=submitted_at,
        source_submission_path=submission_path,
        input_submission_path=input_submission_path,
        validate_and_prepare=lambda path: _validate_and_prepare_autopilot_submission(
            submission_service=submission_service,
            submission_path=path,
            source_submission_path=input_submission_path,
            run_dir=run_dir,
            sample_submission_path=fallback_sample_submission_path,
            data_dir=config.paths.data_dir,
        ),
        validation_error_types=(SubmissionValidationError,),
        validation_exit_code=SubmissionValidationError.exit_code,
        code_fingerprint=submit_code_fingerprint,
        allow_force=allow_force,
        run_state=run_state,
        latest_submit_attempt=latest_submit_attempt,
        submit_mode=submit_mode,
        notebook_submissions_only=bool(getattr(constraints, "notebook_submissions_only", False)),
        notebook_submit_artifact_mode=notebook_submit_artifact_mode,
        code_competition=deps.infer_code_competition_from_paths(config.paths),
        sample_submission_path=config.paths.sample_submission_path,
        fallback_sample_submission_path=fallback_sample_submission_path,
        load_run_state=deps.load_run_state,
        collect_duplicate_submission_sources=deps.collect_duplicate_submission_sources,
        decide_duplicate_submission_action=deps.decide_duplicate_submission_action,
        check_rules_accepted=lambda: deps.check_rules_accepted(config.slug, dry_run=config.dry_run),
        cli_error_types=(KaggleCliError,),
        is_missing_credentials_error=deps.is_missing_kaggle_credentials_error,
        rules_not_accepted_exit_code=RulesNotAcceptedError.exit_code,
        resolve_notebook_submit_artifact_mode=deps.resolve_notebook_submit_artifact_mode,
        decide_notebook_submit_artifact_mode_for_paths=deps.decide_notebook_submit_artifact_mode_for_paths,
        count_tabular_data_rows=deps.count_tabular_data_rows,
        decide_same_submission_path_action=deps.decide_same_submission_path_action,
        compute_error_fingerprint=deps.compute_error_fingerprint,
        compute_submission_sha256=deps.compute_submission_sha256,
        submit_aborter=submit_aborter,
        submit_attempt_recorder=submit_attempt_recorder,
        stdout_tail_chars=limits.stdout_tail_chars,
        stderr_tail_chars=limits.stderr_tail_chars,
        build_error=deps.build_error,
        on_message=deps.on_message,
    )
    prepared_context = prepared_preflight_context.prepared_context
    prepared_submission_path = prepared_context.prepared_submission_path
    preflight_context = prepared_preflight_context.preflight_context
    if preflight_context.duplicate_skip_result is not None:
        return preflight_context.duplicate_skip_result
    if preflight_context.same_submission_path_skipped:
        return None
    submit_stage_state = preflight_context.submit_stage_state
    if submit_stage_state is None:
        raise deps.build_error("Submit preflight did not produce submit stage state.")
    code_competition = preflight_context.code_competition
    seen_fingerprints = preflight_context.seen_fingerprints
    if (
        code_competition
        and submit_stage_state.notebook_submit_required
        and str(submit_stage_state.submission_artifact_mode or "").strip().lower() == "inference"
    ):
        # "inference" is the legacy name for a code competition whose real
        # artifact is produced by the live gateway. Keep that distinction in
        # submit records so a local diagnostic input is never described as the
        # submitted artifact.
        submit_stage_state = replace(submit_stage_state, submission_artifact_mode="gateway")

    notebook_submitter = _build_notebook_submit_runner(
        config=config,
        run_id=run_id,
        deps=deps,
        submission_path=prepared_submission_path,
        quarantine_state=_load_current_quarantine_state(
            run_state=deps.load_run_state(run_dir),
            submission_ledger_path=config.paths.submission_ledger_path,
            slug=config.slug,
        ),
        save_run_state=lambda updates: deps.save_run_state(run_dir, updates),
    )

    def validate_local_submission_fidelity(current_state: object) -> dict[str, object] | None:
        notebook_required = bool(getattr(current_state, "notebook_submit_required", False))
        artifact_mode = str(getattr(current_state, "submission_artifact_mode", "") or "").strip().lower()
        if notebook_required and artifact_mode in {"gateway", "inference"}:
            return None
        iteration = _submit_stage_duplicate.infer_iteration_from_submission_path(input_submission_path) or 1
        logs_dir = run_dir / f"iter-{iteration}" / "logs"
        return _submission_fidelity.validate_file_submission_fidelity(
            slug=config.slug,
            run_id=run_id,
            iteration=iteration,
            source_candidate_path=input_submission_path,
            prepared_submission_path=prepared_submission_path,
            code_fingerprint=submit_code_fingerprint,
            metrics_path=discover_submission_metrics_path(
                submission_path=input_submission_path,
                run_dir=run_dir,
            ),
            semantic_report_path=run_dir / "submission_semantic_preflight.json",
            sample_submission_path=fallback_sample_submission_path,
            report_path=logs_dir / "submission_fidelity_report-file.json",
            expected_contract_path=logs_dir / "submission_fidelity_expected-file.json",
            quarantine_state=_load_current_quarantine_state(
                run_state=deps.load_run_state(run_dir),
                submission_ledger_path=config.paths.submission_ledger_path,
                slug=config.slug,
            ),
            score_value=best_score,
            score_direction=config.target_direction,
        )

    submit_attempt_loop_result = _submit_attempt_loop.run_submit_stage_attempts_until_success_or_abort(
        run_dir=run_dir,
        run_id=run_id,
        state=submit_stage_state,
        prepared_submission_path=prepared_submission_path,
        message=message,
        code_competition=code_competition,
        max_attempts=limits.max_transient_retries,
        backoff_base_seconds=limits.backoff_base_sec,
        sample_submission_path=config.paths.sample_submission_path,
        fallback_sample_submission_path=fallback_sample_submission_path,
        submit_code_fingerprint=submit_code_fingerprint,
        run_state=run_state,
        seen_fingerprints=seen_fingerprints,
        run_notebook_submit=lambda current_state: _run_notebook_submit_with_local_fidelity(
            current_state=current_state,
            validate_local_submission_fidelity=validate_local_submission_fidelity,
            quarantine_state=lambda: _load_current_quarantine_state(
                run_state=deps.load_run_state(run_dir),
                submission_ledger_path=config.paths.submission_ledger_path,
                slug=config.slug,
            ),
            save_run_state=lambda updates: deps.save_run_state(run_dir, updates),
            submission_ledger_path=config.paths.submission_ledger_path,
            slug=config.slug,
            run_id=run_id,
            submit=lambda: notebook_submitter.submit(
                submission_path=prepared_submission_path,
                message=message,
                artifact_mode=current_state.submission_artifact_mode,
            ),
        ),
        run_file_submit=lambda: _run_file_submit_with_fidelity(
            state=submit_stage_state,
            validate_local_submission_fidelity=validate_local_submission_fidelity,
            quarantine_state=lambda: _load_current_quarantine_state(
                run_state=deps.load_run_state(run_dir),
                submission_ledger_path=config.paths.submission_ledger_path,
                slug=config.slug,
            ),
            save_run_state=lambda updates: deps.save_run_state(run_dir, updates),
            submission_ledger_path=config.paths.submission_ledger_path,
            slug=config.slug,
            run_id=run_id,
            submit=lambda: submission_service.submit_prepared(
                prepared_path=prepared_submission_path,
                message=message,
                run_id=run_id,
                offline_score=best_score,
                score_source="offline",
            ),
        ),
        submit_aborter=submit_aborter,
        submit_attempt_recorder=submit_attempt_recorder,
        submit_retry_recorder=submit_retry_recorder,
        submission_cli_error_types=(SubmissionCliError,),
        local_guardrail_error_types=(
            DuplicateSubmissionError,
            SubmissionRateLimitError,
            SubmissionValidationError,
        ),
        kaggle_cli_error_types=(KaggleCliError,),
        classify_submit_error=deps.classify_submit_error,
        should_use_notebook_fallback=deps.should_use_notebook_submit_fallback,
        resolve_notebook_submit_artifact_mode=deps.resolve_notebook_submit_artifact_mode,
        decide_notebook_submit_artifact_mode_for_paths=deps.decide_notebook_submit_artifact_mode_for_paths,
        count_tabular_data_rows=deps.count_tabular_data_rows,
        compute_error_fingerprint=deps.compute_error_fingerprint,
        decide_submit_fingerprint_reuse=deps.decide_submit_fingerprint_reuse,
        compute_submit_backoff=deps.compute_submit_backoff,
        save_run_state_for_run=deps.save_run_state,
        is_missing_credentials_error=deps.is_missing_kaggle_credentials_error,
        build_submit_aborted_error=deps.build_error,
        sleep=deps.sleep,
        on_message=deps.on_message,
    )
    submit_stage_state = submit_attempt_loop_result.submit_stage_state
    outcome_message = _effective_submission_message(
        requested_message=message,
        submission_result=submit_attempt_loop_result.submission_result,
    )
    return _submit_outcome.finalize_submit_outcome_for_run_or_abort(
        run_dir=run_dir,
        submission_ledger_path=config.paths.submission_ledger_path,
        slug=config.slug,
        run_id=run_id,
        message=outcome_message,
        submitted_at=submitted_at,
        submission_ref=submit_attempt_loop_result.submission_reference,
        submission_result=submit_attempt_loop_result.submission_result,
        source_submission_path=submission_path,
        submission_artifact_path=submit_attempt_loop_result.submission_artifact_path,
        submit_stage_state=submit_stage_state,
        code_fingerprint=submit_code_fingerprint,
        deliverable_mode=deps.deliverable_mode(config.paths),
        fetch_submission_rows=lambda current_slug: deps.list_competition_submissions(current_slug, dry_run=False),
        max_attempts=limits.poll_max_attempts,
        poll_interval_sec=limits.poll_interval_sec,
        max_fetch_errors=limits.poll_max_fetch_errors,
        normalize_detail=lambda text: deps.normalize_error_text(text, max_chars=1200),
        submit_aborter=submit_aborter,
        submit_attempt_recorder=submit_attempt_recorder,
        load_run_state=deps.load_run_state,
        compute_error_fingerprint=deps.compute_error_fingerprint,
        compute_submission_sha256=deps.compute_submission_sha256,
        record_submit_attempt_payloads=submit_attempt_recorder.record_payloads,
        stdout_tail_chars=limits.stdout_tail_chars,
        stderr_tail_chars=limits.stderr_tail_chars,
        on_message=deps.on_message,
        wait_for_submission_outcome_func=_submit_stage.wait_for_submission_outcome,
    )


def _effective_submission_message(*, requested_message: str, submission_result: object) -> str:
    """Use the exact message sent by the submission service for outcome matching."""
    submitted_message = str(getattr(submission_result, "message", "") or "").strip()
    return submitted_message or requested_message


def _resolve_fallback_sample_submission_path(paths: SubmitRunnerPaths) -> Path:
    sample_path = paths.sample_submission_path
    if sample_path.is_file():
        return sample_path
    for root in (getattr(paths, "context_dir", None), paths.data_dir):
        if root is None:
            continue
        candidates = find_usable_sample_submissions(root)
        if candidates:
            return candidates[0]
    suffix = tabular_suffix(sample_path)
    if suffix in TABULAR_INPUT_SUFFIXES:
        return paths.data_dir / f"sample_submission{suffix}"
    return paths.data_dir / "sample_submission.csv"


def _validate_and_prepare_autopilot_submission(
    *,
    submission_service: object,
    submission_path: Path,
    source_submission_path: Path,
    run_dir: Path,
    sample_submission_path: Path,
    data_dir: Path,
) -> Path:
    """Apply structural and semantic validation before file/wrapper submission.

    Inference-mode Code submissions skip this local artifact hook and apply the
    same semantic checks to the completed remote runtime output during review.
    """
    prepared_path = submission_service.validate_and_prepare_submission(submission_path)
    metrics_path = discover_submission_metrics_path(
        submission_path=source_submission_path,
        run_dir=run_dir,
    )
    validate_autopilot_submission_semantics(
        submission_path=prepared_path,
        sample_submission_path=sample_submission_path,
        data_dir=data_dir,
        metrics_path=metrics_path,
        report_path=run_dir / "submission_semantic_preflight.json",
    )
    return prepared_path


def _build_notebook_submit_runner(
    *,
    config: SubmitRunnerConfig,
    run_id: str,
    deps: SubmitRunnerDependencies,
    submission_path: Path | None = None,
    quarantine_state: dict[str, object] | None = None,
    save_run_state: Callable[[dict[str, object]], object] | None = None,
):
    from kagglebot import submit_notebook as _submit_notebook

    constraints = deps.load_competition_rule_constraints(config.paths)
    return _submit_notebook.build_notebook_submit_runner_for_run(
        slug=config.slug,
        run_id=run_id,
        paths=config.paths,
        kaggle_username=config.kaggle_username,
        kernel_name=config.kernel_name,
        accelerator=config.accelerator,
        strict_accelerator=config.strict_accelerator,
        dry_run=config.dry_run,
        timeout_minutes=config.time_budget_min,
        infer_iteration_from_submission_path=_submit_stage_duplicate.infer_iteration_from_submission_path,
        resolve_kaggle_username=deps.resolve_kaggle_username,
        run_submit_kernel=deps.run_submit_kernel,
        run_kaggle_submit_kernel=deps.run_kaggle_submit_kernel,
        review_code_submission=deps.review_code_submission,
        recheck_code_submission_guard=lambda **kwargs: deps.recheck_code_submission_guard(
            **kwargs,
            submission_ledger_path=config.paths.submission_ledger_path,
            submission_limit_per_day=getattr(constraints, "submission_limit_per_day", None),
            fetch_submission_rows=lambda slug: deps.list_competition_submissions(slug, dry_run=False),
            force_submit=config.force_submit,
        ),
        record_code_submission_execution=lambda **kwargs: deps.record_code_submission_execution(
            **kwargs,
            submission_ledger_path=config.paths.submission_ledger_path,
            iteration_state_path=(
                config.paths.run_dir(run_id) / f"iter-{int(kwargs.get('iteration') or 0)}" / "iteration_state.json"
            ),
        ),
        copy_submission_artifact_to_iteration_dir=deps.copy_submission_artifact_to_iteration_dir,
        classify_submit_error=deps.classify_submit_error,
        should_retry_ambiguous=deps.should_retry_ambiguous_notebook_submit_error,
        sleep=deps.sleep,
        on_message=deps.on_message,
        expected_output_file=_expected_notebook_submit_output_file(
            config.paths,
            code_competition=deps.infer_code_competition_from_paths(config.paths),
            submission_path=submission_path,
        ),
        quarantine_state=quarantine_state,
        save_run_state=save_run_state,
        submission_ledger_path=config.paths.submission_ledger_path,
    )


def _load_current_quarantine_state(
    *,
    run_state: dict[str, object],
    submission_ledger_path: Path,
    slug: str,
) -> dict[str, object] | None:
    return _submission_fidelity.load_active_submission_fidelity_quarantine(
        run_state=run_state,
        submission_ledger_path=submission_ledger_path,
        slug=slug,
    )


def _run_notebook_submit_with_local_fidelity(
    *,
    current_state: object,
    validate_local_submission_fidelity: Callable[[object], dict[str, object] | None],
    quarantine_state: Callable[[], dict[str, object] | None],
    save_run_state: Callable[[dict[str, object]], object],
    submission_ledger_path: Path,
    slug: str,
    run_id: str,
    submit: Callable[[], tuple[object, str, Path | None]],
) -> tuple[object, str, Path | None]:
    report = validate_local_submission_fidelity(current_state)
    if report is not None:
        _submission_fidelity.reserve_quarantine_repair_attempt(
            report=report,
            quarantine_state=quarantine_state(),
            save_run_state=save_run_state,
            submission_ledger_path=submission_ledger_path,
            slug=slug,
            run_id=run_id,
        )
    return submit()


def _run_file_submit_with_fidelity(
    *,
    state: object,
    validate_local_submission_fidelity: Callable[[object], dict[str, object] | None],
    quarantine_state: Callable[[], dict[str, object] | None],
    save_run_state: Callable[[dict[str, object]], object],
    submission_ledger_path: Path,
    slug: str,
    run_id: str,
    submit: Callable[[], object],
) -> object:
    report = validate_local_submission_fidelity(state)
    if report is not None:
        _submission_fidelity.reserve_quarantine_repair_attempt(
            report=report,
            quarantine_state=quarantine_state(),
            save_run_state=save_run_state,
            submission_ledger_path=submission_ledger_path,
            slug=slug,
            run_id=run_id,
        )
    return submit()


def _expected_notebook_submit_output_file(
    paths: SubmitRunnerPaths,
    *,
    code_competition: bool = False,
    submission_path: Path | None = None,
) -> str | None:
    format_expected = _expected_notebook_submit_output_file_from_format(paths)
    if format_expected is not None:
        return format_expected
    if code_competition:
        history_expected = _expected_code_output_file_from_submission_history(paths)
        if history_expected is not None:
            return history_expected
        submission_name = Path(submission_path).name if submission_path is not None else ""
        lowered_submission_name = submission_name.lower()
        if submission_name and any(
            lowered_submission_name.endswith(suffix) for suffix in _EXPECTED_NOTEBOOK_OUTPUT_SUFFIXES
        ):
            return submission_name
    sample_path = _resolve_fallback_sample_submission_path(paths)
    suffix = tabular_suffix(sample_path)
    if suffix in _EXPECTED_NOTEBOOK_SAMPLE_OUTPUT_SUFFIXES:
        return f"submission{suffix}"
    return None


def _expected_code_output_file_from_submission_history(paths: SubmitRunnerPaths) -> str | None:
    context_dir = getattr(paths, "context_dir", None)
    if context_dir is None:
        return None
    payload = load_json_object(Path(context_dir) / "submission_history.json")
    if payload is None:
        return None
    records: list[object] = [payload.get("latest"), payload.get("best")]
    recent = payload.get("recent")
    if isinstance(recent, list):
        records.extend(recent)
    for record in records:
        if not isinstance(record, dict):
            continue
        label = str(record.get("label") or "").strip()
        lowered = label.lower()
        if not label or "/" in label or "\\" in label:
            continue
        if not any(lowered.endswith(suffix) for suffix in _EXPECTED_NOTEBOOK_OUTPUT_SUFFIXES):
            continue
        return label
    return None


def _expected_notebook_submit_output_file_from_format(paths: SubmitRunnerPaths) -> str | None:
    context_dir = getattr(paths, "context_dir", None)
    if context_dir is None:
        return None
    format_path = Path(context_dir) / "submission_format.md"
    hint = load_submission_format_hint(format_path)
    if hint is None or not hint.expected_suffixes:
        return None
    try:
        text = format_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        text = ""
    return output_filename_from_format_text(
        text,
        expected_suffixes=hint.expected_suffixes,
        allowed_suffixes=_EXPECTED_NOTEBOOK_OUTPUT_SUFFIXES,
    )
