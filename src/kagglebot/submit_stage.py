from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


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
