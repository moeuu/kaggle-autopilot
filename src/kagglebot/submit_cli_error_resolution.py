from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kagglebot.submit_error_classification import classify_submit_error_with_output_fallback


@dataclass(frozen=True)
class SubmitStageRuntimeState:
    notebook_submit_required: bool
    notebook_fallback_activated: bool
    submission_artifact_mode: str


@dataclass(frozen=True)
class SubmitStageNotebookFallbackDecision:
    retry_as_notebook: bool
    notebook_submit_required: bool
    notebook_fallback_activated: bool
    submission_artifact_mode: str
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class SubmitStageNotebookFallbackRetryState:
    notebook_submit_required: bool
    notebook_fallback_activated: bool
    submission_artifact_mode: str
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class SubmitStageFallbackApplication:
    retry_as_notebook: bool
    state: SubmitStageRuntimeState


@dataclass(frozen=True)
class SubmitStageErrorClassification:
    classification: dict[str, object]
    stderr: str
    kind: str
    reason: str
    retry_after_seconds: float


@dataclass(frozen=True)
class SubmitStageErrorActionDecision:
    action: str
    error_kind: str
    reason: str
    wait_seconds: float = 0.0
    abort_message: str = ""
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class SubmitCliErrorResolution:
    classification: SubmitStageErrorClassification
    fallback_application: SubmitStageFallbackApplication
    fingerprint: str
    error_action: SubmitStageErrorActionDecision | None


def apply_notebook_fallback_retry_state(
    fallback_state: SubmitStageNotebookFallbackRetryState,
) -> SubmitStageRuntimeState:
    return SubmitStageRuntimeState(
        notebook_submit_required=fallback_state.notebook_submit_required,
        notebook_fallback_activated=fallback_state.notebook_fallback_activated,
        submission_artifact_mode=fallback_state.submission_artifact_mode,
    )


def decide_submit_stage_error_action(
    *,
    fingerprint_seen: bool,
    same_fingerprint_retry_allowed: bool,
    classification_kind: str,
    classification_reason: str,
    attempt: int,
    max_attempts: int,
    retry_after_seconds: float,
    backoff_seconds: float,
) -> SubmitStageErrorActionDecision:
    if fingerprint_seen and not same_fingerprint_retry_allowed:
        return SubmitStageErrorActionDecision(
            action="abort",
            error_kind=classification_kind,
            reason="same_error_fingerprint_recurred",
            abort_message="Same submit error fingerprint recurred; aborting this run to prevent infinite loop.",
        )

    messages: list[str] = []
    if fingerprint_seen and same_fingerprint_retry_allowed:
        messages.append(
            "[yellow]submit retry[/yellow]: same fingerprint matched previous failures, "
            "but code changed since last submit error; allowing one retry."
        )

    if classification_kind == "transient" and attempt < max_attempts:
        wait_seconds = max(backoff_seconds, retry_after_seconds)
        messages.append(
            "[yellow]submit retry[/yellow]: transient submit error "
            f"(reason={classification_reason}, attempt={attempt}/{max_attempts}, wait={wait_seconds:.1f}s)"
        )
        return SubmitStageErrorActionDecision(
            action="retry",
            error_kind="transient",
            reason=classification_reason,
            wait_seconds=wait_seconds,
            messages=tuple(messages),
        )

    abort_message = (
        "Submit failed and is not retryable in this run."
        if classification_kind != "transient"
        else "Transient submit error exceeded retry budget; aborting this run."
    )
    messages.append(
        "[red]submit aborted[/red]: "
        f"{classification_kind} submit error (reason={classification_reason}); no further retries in this run."
    )
    return SubmitStageErrorActionDecision(
        action="abort",
        error_kind=classification_kind,
        reason=classification_reason,
        abort_message=abort_message,
        messages=tuple(messages),
    )


def decide_submit_stage_error_action_from_classification(
    *,
    fingerprint_seen: bool,
    same_fingerprint_retry_allowed: bool,
    classification: SubmitStageErrorClassification,
    attempt: int,
    max_attempts: int,
    backoff_seconds: float,
) -> SubmitStageErrorActionDecision:
    return decide_submit_stage_error_action(
        fingerprint_seen=fingerprint_seen,
        same_fingerprint_retry_allowed=same_fingerprint_retry_allowed,
        classification_kind=classification.kind,
        classification_reason=classification.reason,
        attempt=attempt,
        max_attempts=max_attempts,
        retry_after_seconds=classification.retry_after_seconds,
        backoff_seconds=backoff_seconds,
    )


def classify_submit_stage_error(
    *,
    stdout: str,
    stderr: str,
    output: str,
    exit_code: int | None,
    classify_submit_error: Callable[[str, str, int | None], dict[str, object]],
) -> SubmitStageErrorClassification:
    result = classify_submit_error_with_output_fallback(
        stdout=stdout,
        stderr=stderr,
        output=output,
        exit_code=exit_code,
        classify_submit_error=classify_submit_error,
    )
    return SubmitStageErrorClassification(
        classification=result.classification,
        stderr=result.stderr,
        kind=result.normalized.kind,
        reason=result.normalized.reason,
        retry_after_seconds=result.normalized.retry_after_seconds,
    )


def decide_notebook_fallback_after_file_submit_error(
    *,
    notebook_submit_required: bool,
    notebook_fallback_activated: bool,
    should_use_notebook_fallback: bool,
    resolved_notebook_artifact_mode: str | None,
    current_submission_artifact_mode: str,
) -> SubmitStageNotebookFallbackDecision:
    if notebook_submit_required or notebook_fallback_activated or not should_use_notebook_fallback:
        return SubmitStageNotebookFallbackDecision(
            retry_as_notebook=False,
            notebook_submit_required=notebook_submit_required,
            notebook_fallback_activated=notebook_fallback_activated,
            submission_artifact_mode=current_submission_artifact_mode,
        )
    return SubmitStageNotebookFallbackDecision(
        retry_as_notebook=True,
        notebook_submit_required=True,
        notebook_fallback_activated=True,
        submission_artifact_mode=str(resolved_notebook_artifact_mode or "wrapper"),
        messages=(
            "[yellow]submit mode[/yellow]: file submit indicates notebook submit is required; "
            "retrying via notebook submit automatically.",
        ),
    )


def build_notebook_fallback_retry_state(
    *,
    fallback_decision: SubmitStageNotebookFallbackDecision,
    artifact_mode: str,
    artifact_message: str,
) -> SubmitStageNotebookFallbackRetryState:
    messages = list(fallback_decision.messages)
    if artifact_message:
        messages.append(artifact_message)
    return SubmitStageNotebookFallbackRetryState(
        notebook_submit_required=fallback_decision.notebook_submit_required,
        notebook_fallback_activated=fallback_decision.notebook_fallback_activated,
        submission_artifact_mode=artifact_mode,
        messages=tuple(messages),
    )


def apply_notebook_fallback_decision(
    *,
    state: SubmitStageRuntimeState,
    fallback_decision: SubmitStageNotebookFallbackDecision,
    resolve_artifact_mode: Callable[[str, bool], object],
    on_message: Callable[[str], object],
) -> SubmitStageFallbackApplication:
    if not fallback_decision.retry_as_notebook:
        return SubmitStageFallbackApplication(retry_as_notebook=False, state=state)

    artifact_mode_decision = resolve_artifact_mode(
        fallback_decision.submission_artifact_mode,
        fallback_decision.notebook_submit_required,
    )
    fallback_retry_state = build_notebook_fallback_retry_state(
        fallback_decision=fallback_decision,
        artifact_mode=str(getattr(artifact_mode_decision, "mode", "") or fallback_decision.submission_artifact_mode),
        artifact_message=str(getattr(artifact_mode_decision, "message", "") or ""),
    )
    for mode_message in fallback_retry_state.messages:
        on_message(mode_message)
    return SubmitStageFallbackApplication(
        retry_as_notebook=True,
        state=apply_notebook_fallback_retry_state(fallback_retry_state),
    )


def resolve_notebook_fallback_after_file_submit_error(
    *,
    state: SubmitStageRuntimeState,
    should_use_notebook_fallback: bool,
    code_competition: bool,
    sample_submission_path: Path,
    fallback_sample_submission_path: Path,
    submission_path: Path,
    resolve_notebook_submit_artifact_mode: Callable[..., str],
    decide_notebook_submit_artifact_mode_for_paths: Callable[..., object],
    count_tabular_data_rows: Callable[[Path], int | None],
    on_message: Callable[[str], object],
) -> SubmitStageFallbackApplication:
    resolved_notebook_artifact_mode = (
        resolve_notebook_submit_artifact_mode(
            submit_mode="notebook",
            code_competition=code_competition,
        )
        if should_use_notebook_fallback and not state.notebook_submit_required and not state.notebook_fallback_activated
        else None
    )
    fallback_decision = decide_notebook_fallback_after_file_submit_error(
        notebook_submit_required=state.notebook_submit_required,
        notebook_fallback_activated=state.notebook_fallback_activated,
        should_use_notebook_fallback=should_use_notebook_fallback,
        resolved_notebook_artifact_mode=resolved_notebook_artifact_mode,
        current_submission_artifact_mode=state.submission_artifact_mode,
    )
    return apply_notebook_fallback_decision(
        state=state,
        fallback_decision=fallback_decision,
        resolve_artifact_mode=lambda requested_mode, notebook_required: (
            decide_notebook_submit_artifact_mode_for_paths(
                requested_mode=requested_mode,
                notebook_submit_required=notebook_required,
                code_competition=code_competition,
                sample_submission_path=sample_submission_path,
                fallback_sample_submission_path=fallback_sample_submission_path,
                submission_path=submission_path,
                count_tabular_data_rows=count_tabular_data_rows,
            )
        ),
        on_message=on_message,
    )


def resolve_submit_cli_error(
    *,
    state: SubmitStageRuntimeState,
    stdout: str,
    stderr: str,
    output: str,
    exit_code: int | None,
    attempt: int,
    max_attempts: int,
    backoff_base_seconds: float,
    classify_submit_error: Callable[..., dict[str, object]],
    should_use_notebook_fallback: Callable[..., bool],
    code_competition: bool,
    sample_submission_path: Path,
    fallback_sample_submission_path: Path,
    submission_path: Path,
    resolve_notebook_submit_artifact_mode: Callable[..., str],
    decide_notebook_submit_artifact_mode_for_paths: Callable[..., object],
    count_tabular_data_rows: Callable[[Path], int | None],
    compute_error_fingerprint: Callable[[str, str], str],
    decide_submit_fingerprint_reuse: Callable[..., object],
    compute_submit_backoff: Callable[..., float],
    seen_fingerprints: set[str],
    run_state: dict[str, object],
    code_fingerprint: str,
    save_run_state: Callable[[dict[str, object]], object],
    on_message: Callable[[str], object],
) -> SubmitCliErrorResolution:
    classification = classify_submit_stage_error(
        stdout=stdout,
        stderr=stderr,
        output=output,
        exit_code=exit_code,
        classify_submit_error=classify_submit_error,
    )
    fallback_application = resolve_notebook_fallback_after_file_submit_error(
        state=state,
        should_use_notebook_fallback=should_use_notebook_fallback(
            reason=classification.reason,
            stdout=stdout,
            stderr=classification.stderr,
        ),
        code_competition=code_competition,
        sample_submission_path=sample_submission_path,
        fallback_sample_submission_path=fallback_sample_submission_path,
        submission_path=submission_path,
        resolve_notebook_submit_artifact_mode=resolve_notebook_submit_artifact_mode,
        decide_notebook_submit_artifact_mode_for_paths=decide_notebook_submit_artifact_mode_for_paths,
        count_tabular_data_rows=count_tabular_data_rows,
        on_message=on_message,
    )
    if fallback_application.retry_as_notebook:
        return SubmitCliErrorResolution(
            classification=classification,
            fallback_application=fallback_application,
            fingerprint="",
            error_action=None,
        )

    fingerprint = compute_error_fingerprint(stdout, stderr)
    fingerprint_reuse_decision = decide_submit_fingerprint_reuse(
        fingerprint=fingerprint,
        seen_fingerprints=seen_fingerprints,
        run_state=run_state,
        code_fingerprint=code_fingerprint,
        save_run_state=save_run_state,
    )
    error_action = decide_submit_stage_error_action_from_classification(
        fingerprint_seen=bool(getattr(fingerprint_reuse_decision, "fingerprint_seen", False)),
        same_fingerprint_retry_allowed=bool(
            getattr(fingerprint_reuse_decision, "same_fingerprint_retry_allowed", False)
        ),
        classification=classification,
        attempt=attempt,
        max_attempts=max_attempts,
        backoff_seconds=compute_submit_backoff(
            attempt=attempt,
            base_seconds=backoff_base_seconds,
        ),
    )
    for action_message in error_action.messages:
        on_message(action_message)
    return SubmitCliErrorResolution(
        classification=classification,
        fallback_application=fallback_application,
        fingerprint=fingerprint,
        error_action=error_action,
    )


def resolve_submit_cli_error_for_run(
    *,
    run_dir: Path,
    state: SubmitStageRuntimeState,
    stdout: str,
    stderr: str,
    output: str,
    exit_code: int | None,
    attempt: int,
    max_attempts: int,
    backoff_base_seconds: float,
    classify_submit_error: Callable[..., dict[str, object]],
    should_use_notebook_fallback: Callable[..., bool],
    code_competition: bool,
    sample_submission_path: Path,
    fallback_sample_submission_path: Path,
    submission_path: Path,
    resolve_notebook_submit_artifact_mode: Callable[..., str],
    decide_notebook_submit_artifact_mode_for_paths: Callable[..., object],
    count_tabular_data_rows: Callable[[Path], int | None],
    compute_error_fingerprint: Callable[[str, str], str],
    decide_submit_fingerprint_reuse: Callable[..., object],
    compute_submit_backoff: Callable[..., float],
    seen_fingerprints: set[str],
    run_state: dict[str, object],
    code_fingerprint: str,
    save_run_state_for_run: Callable[[Path, dict[str, object]], object],
    on_message: Callable[[str], object],
) -> SubmitCliErrorResolution:
    return resolve_submit_cli_error(
        state=state,
        stdout=stdout,
        stderr=stderr,
        output=output,
        exit_code=exit_code,
        attempt=attempt,
        max_attempts=max_attempts,
        backoff_base_seconds=backoff_base_seconds,
        classify_submit_error=classify_submit_error,
        should_use_notebook_fallback=should_use_notebook_fallback,
        code_competition=code_competition,
        sample_submission_path=sample_submission_path,
        fallback_sample_submission_path=fallback_sample_submission_path,
        submission_path=submission_path,
        resolve_notebook_submit_artifact_mode=resolve_notebook_submit_artifact_mode,
        decide_notebook_submit_artifact_mode_for_paths=decide_notebook_submit_artifact_mode_for_paths,
        count_tabular_data_rows=count_tabular_data_rows,
        compute_error_fingerprint=compute_error_fingerprint,
        decide_submit_fingerprint_reuse=decide_submit_fingerprint_reuse,
        compute_submit_backoff=compute_submit_backoff,
        seen_fingerprints=seen_fingerprints,
        run_state=run_state,
        code_fingerprint=code_fingerprint,
        save_run_state=lambda updates: save_run_state_for_run(run_dir, updates),
        on_message=on_message,
    )
