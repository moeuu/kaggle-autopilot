from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SubmitStageModeDecision:
    notebook_submit_required: bool
    notebook_fallback_activated: bool
    submission_artifact_mode: str
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class SubmitStageNotebookFallbackDecision:
    retry_as_notebook: bool
    notebook_submit_required: bool
    notebook_fallback_activated: bool
    submission_artifact_mode: str
    messages: tuple[str, ...] = ()


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
class SubmitStageAttemptResult:
    submission_result: object
    submission_reference: str
    submission_artifact_path: Path | None


def decide_initial_submit_stage_mode(
    *,
    requested_notebook_submit: bool,
    notebook_submissions_only: bool,
    notebook_submit_artifact_mode: str | None,
    resolved_notebook_artifact_mode: str | None,
) -> SubmitStageModeDecision:
    notebook_submit_required = bool(requested_notebook_submit)
    messages: list[str] = []
    if notebook_submissions_only and not notebook_submit_required:
        notebook_submit_required = True
        messages.append("[yellow]submit mode[/yellow]: notebook-only competition detected; forcing notebook submit")

    submission_artifact_mode = (
        str(resolved_notebook_artifact_mode or "wrapper")
        if notebook_submit_required
        else str(notebook_submit_artifact_mode or "wrapper")
    )
    if notebook_submit_required:
        messages.append("[yellow]submit mode[/yellow]: using notebook submit")

    return SubmitStageModeDecision(
        notebook_submit_required=notebook_submit_required,
        notebook_fallback_activated=notebook_submit_required,
        submission_artifact_mode=submission_artifact_mode,
        messages=tuple(messages),
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


def classify_submit_stage_error(
    *,
    stdout: str,
    stderr: str,
    output: str,
    exit_code: int | None,
    classify_submit_error: Callable[[str, str, int | None], dict[str, object]],
) -> SubmitStageErrorClassification:
    classification_stderr = stderr or ""
    classification = classify_submit_error(stdout, classification_stderr, exit_code)
    if str(classification.get("reason") or "unclassified_submit_error") == "unclassified_submit_error" and output:
        classification_stderr = "\n".join(part for part in [classification_stderr, output] if part)
        classification = classify_submit_error(stdout, classification_stderr, exit_code)
    retry_after = classification.get("retry_after_seconds")
    return SubmitStageErrorClassification(
        classification=classification,
        stderr=classification_stderr,
        kind=str(classification.get("kind") or "unknown"),
        reason=str(classification.get("reason") or "unclassified_submit_error"),
        retry_after_seconds=float(retry_after) if isinstance(retry_after, (int, float)) else 0.0,
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
