from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kagglebot.submit_cli_error_resolution import SubmitStageRuntimeState
from kagglebot.writeup import normalize_submit_mode


@dataclass(frozen=True)
class SubmitStageModeDecision:
    notebook_submit_required: bool
    notebook_fallback_activated: bool
    submission_artifact_mode: str
    messages: tuple[str, ...] = ()


def resolve_iteration_submit_phase_state(
    *,
    submit_enabled: bool,
    daily_submission_limit_reached: bool,
    force_initial_submit: bool,
    quality_allows_submit: bool,
    force_submit: bool,
    submit_non_improving: bool,
    defer_submit_for_accuracy_frontier: bool,
    submit_limited_holdback: bool,
) -> str:
    state = "pending_submit" if submit_enabled else "disabled"
    if daily_submission_limit_reached:
        state = "daily_submission_limit_reached"
    elif force_initial_submit:
        state = "forced_initial_submit"
    elif submit_enabled and (not quality_allows_submit) and (not force_submit):
        state = "blocked_quality_guard"
    if submit_non_improving:
        state = "deferred_non_improving"
    if defer_submit_for_accuracy_frontier:
        state = "deferred_accuracy_frontier"
    if submit_limited_holdback:
        state = "deferred_for_final_slot"
    return state


def build_submit_stage_runtime_state(decision: SubmitStageModeDecision) -> SubmitStageRuntimeState:
    return SubmitStageRuntimeState(
        notebook_submit_required=decision.notebook_submit_required,
        notebook_fallback_activated=decision.notebook_fallback_activated,
        submission_artifact_mode=decision.submission_artifact_mode,
    )


def update_submit_stage_artifact_mode(
    state: SubmitStageRuntimeState,
    *,
    submission_artifact_mode: str,
) -> SubmitStageRuntimeState:
    return SubmitStageRuntimeState(
        notebook_submit_required=state.notebook_submit_required,
        notebook_fallback_activated=state.notebook_fallback_activated,
        submission_artifact_mode=submission_artifact_mode,
    )


def apply_initial_submit_stage_artifact_mode(
    *,
    mode_decision: SubmitStageModeDecision,
    resolve_artifact_mode: Callable[[str, bool], object],
    on_message: Callable[[str], object],
) -> SubmitStageRuntimeState:
    state = build_submit_stage_runtime_state(mode_decision)
    for mode_message in mode_decision.messages:
        on_message(mode_message)

    artifact_mode_decision = resolve_artifact_mode(
        state.submission_artifact_mode,
        state.notebook_submit_required,
    )
    state = update_submit_stage_artifact_mode(
        state,
        submission_artifact_mode=str(getattr(artifact_mode_decision, "mode", "") or state.submission_artifact_mode),
    )
    artifact_message = str(getattr(artifact_mode_decision, "message", "") or "").strip()
    if artifact_message:
        on_message(artifact_message)
    return state


def resolve_initial_submit_stage_runtime_state(
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
    on_message: Callable[[str], object],
) -> SubmitStageRuntimeState:
    requested_notebook_submit = normalize_submit_mode(submit_mode, default="file") == "notebook"
    resolved_notebook_artifact_mode = (
        resolve_notebook_submit_artifact_mode(
            submit_mode="notebook",
            code_competition=code_competition,
        )
        if requested_notebook_submit or notebook_submissions_only
        else None
    )
    mode_decision = decide_initial_submit_stage_mode(
        requested_notebook_submit=requested_notebook_submit,
        notebook_submissions_only=notebook_submissions_only,
        notebook_submit_artifact_mode=notebook_submit_artifact_mode,
        resolved_notebook_artifact_mode=resolved_notebook_artifact_mode,
    )
    return apply_initial_submit_stage_artifact_mode(
        mode_decision=mode_decision,
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

    requested_artifact_mode = str(notebook_submit_artifact_mode or "").strip().lower()
    if notebook_submit_required:
        if requested_artifact_mode == "inference":
            submission_artifact_mode = "inference"
        else:
            submission_artifact_mode = str(resolved_notebook_artifact_mode or requested_artifact_mode or "wrapper")
    else:
        submission_artifact_mode = str(notebook_submit_artifact_mode or "wrapper")
    if notebook_submit_required:
        messages.append("[yellow]submit mode[/yellow]: using notebook submit")

    return SubmitStageModeDecision(
        notebook_submit_required=notebook_submit_required,
        notebook_fallback_activated=notebook_submit_required,
        submission_artifact_mode=submission_artifact_mode,
        messages=tuple(messages),
    )
