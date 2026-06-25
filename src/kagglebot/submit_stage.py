from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from kagglebot import submit_attempts as _submit_attempts
from kagglebot.campaign import (
    CampaignCandidate,
    campaign_state_path,
    candidate_registry_path,
    format_campaign_submission_message,
    list_candidates,
    normalize_campaign_mode,
)
from kagglebot.json_utils import load_json_object
from kagglebot.scalar_utils import parse_finite_float, parse_int
from kagglebot.score_utils import should_update_best_score
from kagglebot.submission.outcome_service import SubmissionOutcomePollingError, SubmissionOutcomeService
from kagglebot.submit_error_classification import classify_submit_error_with_output_fallback
from kagglebot.writeup import normalize_submit_mode


@dataclass(frozen=True)
class SubmitStageModeDecision:
    notebook_submit_required: bool
    notebook_fallback_activated: bool
    submission_artifact_mode: str
    messages: tuple[str, ...] = ()


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
class SubmittedTrackingScoreDecision:
    online_score: float | None
    tracking_score: float | None
    tracking_source: str
    update_best_submitted_score: bool
    best_submitted_score: float | None


@dataclass(frozen=True)
class SubmissionRankState:
    rank_payload: dict[str, object]
    rank: int | None
    total_teams: int | None
    rank_percentile: float | None
    rank_source: str | None
    estimated_rank: int | None
    estimated_total_teams: int | None
    estimated_rank_percentile: float | None
    rank_estimate_source: str | None
    force_major_overhaul: bool
    force_reason: str | None
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class FallbackSubmitGateDecision:
    allow_submit: bool
    message: str


@dataclass(frozen=True)
class IterationSubmitImprovementGateDecision:
    submit_improvement_allowed: bool
    submit_non_improving: bool
    forced_submit_reason: str | None
    message: str


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


@dataclass(frozen=True)
class SubmitStageAttemptResult:
    submission_result: object
    submission_reference: str
    submission_artifact_path: Path | None


@dataclass(frozen=True)
class SubmitStageSuccessRecord:
    exit_code: int | None
    fingerprint: str
    stdout: str
    stderr: str


@dataclass(frozen=True)
class SubmissionKnowledgeContext:
    online_score: float
    outcome_bucket: str
    iteration: int | None


@dataclass(frozen=True)
class SubmitOutcomeAbortDecision:
    should_abort: bool
    error_kind: str = ""
    reason: str = ""
    message: str = ""
    detail: str = ""


@dataclass(frozen=True)
class SubmitAbortSpec:
    fingerprint: str
    error_kind: str
    reason: str
    message: str
    stdout_tail: str
    stderr_tail: str
    exit_code: int | None


@dataclass(frozen=True)
class SubmitPreparedSubmissionResolution:
    prepared_submission_path: Path | None
    abort_spec: SubmitAbortSpec | None = None


@dataclass(frozen=True)
class SubmitRulesAcceptanceResolution:
    rules_accepted: bool
    abort_spec: SubmitAbortSpec | None = None


def build_submit_abort_spec_kwargs(spec: SubmitAbortSpec) -> dict[str, object]:
    return {
        "fingerprint": spec.fingerprint,
        "error_kind": spec.error_kind,
        "reason": spec.reason,
        "message": spec.message,
        "stdout_tail": spec.stdout_tail,
        "stderr_tail": spec.stderr_tail,
        "exit_code": spec.exit_code,
    }


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
            abort_spec=build_local_submission_validation_abort_spec(
                error=exc,
                exit_code=validation_exit_code,
                compute_error_fingerprint=compute_error_fingerprint,
            ),
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
            abort_spec=build_kaggle_credentials_missing_abort_spec(
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
            abort_spec=build_rules_not_accepted_abort_spec(
                exit_code=rules_not_accepted_exit_code,
                compute_error_fingerprint=compute_error_fingerprint,
            ),
        )
    return SubmitRulesAcceptanceResolution(rules_accepted=True)


@dataclass(frozen=True)
class SubmissionOutcomePostPollDecision:
    outcome: object
    abort_decision: SubmitOutcomeAbortDecision


@dataclass(frozen=True)
class SubmitOutcomeResolution:
    outcome: object
    abort_spec: SubmitAbortSpec | None = None


FAILED_SUBMISSION_OUTCOME_STATUSES = {"error", "failed", "cancelled", "canceled"}
SCORELESS_COMPLETE_SUBMISSION_OUTCOME_STATUSES = {"complete", "completed"}


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


def normalize_submission_outcome_status(value: object) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "unknown"
    if "." in raw:
        prefix, _, suffix = raw.rpartition(".")
        if suffix and "status" in prefix:
            return suffix.strip()
    return raw


def infer_iteration_from_submission_path(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        name = path.parent.name
        if not name.startswith("iter-"):
            return None
        return int(name.split("-", 1)[1])
    except Exception:  # noqa: BLE001
        return None


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


def apply_notebook_fallback_retry_state(
    fallback_state: SubmitStageNotebookFallbackRetryState,
) -> SubmitStageRuntimeState:
    return SubmitStageRuntimeState(
        notebook_submit_required=fallback_state.notebook_submit_required,
        notebook_fallback_activated=fallback_state.notebook_fallback_activated,
        submission_artifact_mode=fallback_state.submission_artifact_mode,
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
    count_csv_data_rows: Callable[[Path], int | None],
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
                count_csv_data_rows=count_csv_data_rows,
            )
        ),
        on_message=on_message,
    )


def apply_same_submission_path_decision(
    *,
    decision: object,
    run_id: str,
    submission_path: Path,
    compute_submission_sha256: Callable[[Path | None], str | None],
    record_submit_attempt: Callable[[dict[str, object]], object],
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    on_message: Callable[[str], object],
) -> bool:
    action = str(getattr(decision, "action", "") or "").strip().lower()
    message = str(getattr(decision, "message", "") or "").strip()
    if action == "retry":
        if message:
            on_message(message)
        return False
    if action != "skip":
        return False

    if message:
        on_message(message)
    record_submit_attempt(
        _submit_attempts.build_same_submission_path_skip_attempt_payload(
            run_id=run_id,
            submission_ref=str(submission_path),
            submission_sha256=compute_submission_sha256(submission_path),
            fingerprint=str(getattr(decision, "fingerprint", "") or ""),
            reason=str(getattr(decision, "reason", "") or ""),
            stdout_tail_chars=stdout_tail_chars,
            stderr_tail_chars=stderr_tail_chars,
        )
    )
    return True


def apply_duplicate_submission_decision(
    *,
    decision: object,
    run_id: str,
    message: str,
    submitted_at: datetime,
    submission_path: Path,
    prepared_submission_path: Path,
    prepared_submission_sha: str,
    code_fingerprint: str,
    prior_state: dict[str, object],
    record_submit_attempt_payloads: Callable[[object], object],
    mark_duplicate_skipped: Callable[[str, str], object],
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    on_message: Callable[[str], object],
) -> dict[str, object] | None:
    action = str(getattr(decision, "action", "") or "").strip().lower()
    if action != "skip":
        return None

    decision_message = str(getattr(decision, "message", "") or "").strip()
    if decision_message:
        on_message(decision_message)
    reason = str(getattr(decision, "reason", "") or "")
    duplicate_sources = list(getattr(decision, "duplicate_sources", []) or [])
    submission_ref = str(prepared_submission_path)
    skip_payloads = _submit_attempts.build_duplicate_submit_skip_record_payloads(
        run_id=run_id,
        submission_ref=submission_ref,
        submission_sha256=prepared_submission_sha,
        fingerprint=str(getattr(decision, "fingerprint", "") or ""),
        code_fingerprint=code_fingerprint,
        reason=reason,
        prior_state=prior_state,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
        duplicate_sources=duplicate_sources,
    )
    record_submit_attempt_payloads(skip_payloads)
    mark_duplicate_skipped(submission_ref, reason)
    return _submit_attempts.build_duplicate_submit_skip_result_payload(
        message=message,
        submission_ref=submission_ref,
        submitted_at=submitted_at,
        submission_path=submission_path,
        reason=reason,
        duplicate_sources=duplicate_sources,
        infer_iteration=infer_iteration_from_submission_path,
    )


def submission_score_for_tracking(*, offline_score: float, online_score: float | None) -> tuple[float, str]:
    if isinstance(online_score, (int, float)):
        value = float(online_score)
        if math.isfinite(value):
            return value, "submission_public_score"
    return float(offline_score), "offline"


def decide_submitted_tracking_score_update(
    *,
    submission_result: dict[str, object],
    offline_score: float | None,
    previous_best_score: float | None,
    direction: str,
    min_improvement: float = 0.0,
) -> SubmittedTrackingScoreDecision:
    if offline_score is None:
        return SubmittedTrackingScoreDecision(
            online_score=None,
            tracking_score=None,
            tracking_source="unavailable",
            update_best_submitted_score=False,
            best_submitted_score=previous_best_score,
        )

    online_score: float | None = None
    outcome_payload = submission_result.get("outcome")
    if isinstance(outcome_payload, dict):
        online_score = parse_finite_float(outcome_payload.get("score"))
    tracking_score, tracking_source = submission_score_for_tracking(
        offline_score=offline_score,
        online_score=online_score,
    )
    should_update = should_update_best_score(previous_best_score, tracking_score, direction, min_improvement)
    return SubmittedTrackingScoreDecision(
        online_score=online_score,
        tracking_score=tracking_score,
        tracking_source=tracking_source,
        update_best_submitted_score=should_update,
        best_submitted_score=tracking_score if should_update else previous_best_score,
    )


def decide_fallback_submit_gate(
    *,
    submit_improved_only: bool,
    force_submit: bool,
    require_submit_improvement: bool,
    best_submittable_score: float | None,
    best_submitted_score: float | None,
    direction: str,
    min_improvement: float,
    final_iteration_reached: bool,
) -> FallbackSubmitGateDecision:
    if submit_improved_only and not force_submit and best_submitted_score is None:
        return FallbackSubmitGateDecision(allow_submit=False, message="")

    if (
        (require_submit_improvement or submit_improved_only)
        and not force_submit
        and best_submittable_score is not None
        and best_submitted_score is not None
    ):
        allow_submit = should_update_best_score(
            best_submitted_score, best_submittable_score, direction, min_improvement
        )
        if (not allow_submit) and final_iteration_reached and not submit_improved_only:
            return FallbackSubmitGateDecision(
                allow_submit=True,
                message=(
                    "[yellow]submit override[/yellow]: final iteration reached; "
                    "allowing fallback submit even though offline metric did not improve."
                ),
            )
        return FallbackSubmitGateDecision(allow_submit=allow_submit, message="")

    return FallbackSubmitGateDecision(allow_submit=True, message="")


def decide_iteration_submit_improvement_gate(
    *,
    submit_improved_only: bool,
    force_submit: bool,
    require_submit_improvement: bool,
    best_submitted_score: float | None,
    current_score: float,
    direction: str,
    min_improvement: float,
    final_iteration: bool,
    submit_enabled: bool,
    quality_allows_submit: bool,
    spare_daily_submission_slot: bool,
    submission_limit_per_day: int | None,
    forced_submit_reason: str | None,
    spare_submit_reason: str,
) -> IterationSubmitImprovementGateDecision:
    if submit_improved_only and not force_submit and best_submitted_score is None:
        if (
            submit_enabled
            and quality_allows_submit
            and (spare_daily_submission_slot or submission_limit_per_day is None)
        ):
            slot_reason = (
                "spare daily submission slots remain"
                if spare_daily_submission_slot
                else "no numeric daily submission limit is known"
            )
            return IterationSubmitImprovementGateDecision(
                submit_improvement_allowed=True,
                submit_non_improving=False,
                forced_submit_reason=forced_submit_reason or spare_submit_reason,
                message=(
                    f"[yellow]submit override[/yellow]: {slot_reason}; allowing submit without a prior "
                    "submitted checkpoint."
                ),
            )
        return IterationSubmitImprovementGateDecision(
            submit_improvement_allowed=False,
            submit_non_improving=True,
            forced_submit_reason=forced_submit_reason,
            message=("[yellow]submit deferred[/yellow]: submit_policy=improved requires a prior submitted checkpoint."),
        )

    if (require_submit_improvement or submit_improved_only) and not force_submit and best_submitted_score is not None:
        allowed = should_update_best_score(best_submitted_score, current_score, direction, min_improvement)
        if allowed:
            return IterationSubmitImprovementGateDecision(True, False, forced_submit_reason, "")
        if final_iteration and not submit_improved_only:
            return IterationSubmitImprovementGateDecision(
                submit_improvement_allowed=True,
                submit_non_improving=False,
                forced_submit_reason=forced_submit_reason,
                message=(
                    "[yellow]submit override[/yellow]: final iteration reached; "
                    "allowing submit even though score did not improve over submitted checkpoints."
                ),
            )
        if submit_enabled and spare_daily_submission_slot and quality_allows_submit:
            return IterationSubmitImprovementGateDecision(
                submit_improvement_allowed=True,
                submit_non_improving=False,
                forced_submit_reason=forced_submit_reason or spare_submit_reason,
                message=(
                    "[yellow]submit override[/yellow]: spare daily submission slots remain; "
                    "allowing non-improving checkpoint submit."
                ),
            )
        return IterationSubmitImprovementGateDecision(
            submit_improvement_allowed=False,
            submit_non_improving=True,
            forced_submit_reason=forced_submit_reason,
            message="[yellow]submit deferred[/yellow]: score did not improve over previous submitted checkpoint.",
        )

    return IterationSubmitImprovementGateDecision(True, False, forced_submit_reason, "")


def classify_submission_outcome(
    *,
    score: float,
    direction: str,
    target_score: float | None,
    top1_score: float | None,
) -> str:
    if target_score is not None and _meets_target(score, target_score, direction):
        return "good"
    if top1_score is not None:
        if direction == "minimize":
            gap = score - top1_score
        else:
            gap = top1_score - score
        scale = max(abs(top1_score), 1.0)
        if max(gap, 0.0) / scale <= 0.1:
            return "good"
    return "low"


def resolve_submission_knowledge_context(
    *,
    submission_result: dict[str, object] | None,
    metric_direction: str,
    target_score: float | None,
    top1_score: float | None,
) -> SubmissionKnowledgeContext | None:
    if not submission_result:
        return None
    outcome_payload = submission_result.get("outcome")
    if not isinstance(outcome_payload, dict):
        return None
    online_score = parse_finite_float(outcome_payload.get("score"))
    if online_score is None:
        return None
    submitted_iteration = submission_result.get("iteration")
    iteration_value = submitted_iteration if isinstance(submitted_iteration, int) else None
    return SubmissionKnowledgeContext(
        online_score=online_score,
        outcome_bucket=classify_submission_outcome(
            score=online_score,
            direction=metric_direction,
            target_score=target_score,
            top1_score=top1_score,
        ),
        iteration=iteration_value,
    )


def resolve_submission_knowledge_iteration(*, value: object, fallback_iteration: int | None) -> int:
    try:
        return int(value or (fallback_iteration or 1))
    except (TypeError, ValueError):
        return fallback_iteration or 1


def build_default_submission_problem_insight(
    *,
    iteration: int | None,
    diagnostics_text: str,
) -> dict[str, object]:
    resolved_iteration = iteration or 1
    return {
        "iteration": resolved_iteration,
        "why_poor": diagnostics_text,
        "how_improved": f"Submitted iteration {resolved_iteration} result after validation.",
        "delta_offline": None,
    }


def ensure_submission_problem_insights(
    *,
    pending_problem_insights: list[dict[str, object]],
    knowledge_context: SubmissionKnowledgeContext,
    load_diagnostics_text: Callable[[int], str],
) -> None:
    if pending_problem_insights:
        return
    diagnostics_text = ""
    if knowledge_context.iteration is not None:
        diagnostics_text = load_diagnostics_text(knowledge_context.iteration)
    pending_problem_insights.append(
        build_default_submission_problem_insight(
            iteration=knowledge_context.iteration,
            diagnostics_text=diagnostics_text,
        )
    )


def record_submission_knowledge_entries(
    *,
    knowledge_paths: object,
    slug: str,
    run_id: str,
    problem_types: list[str],
    pending_problem_insights: list[dict[str, object]],
    pending_error_fixes: list[dict[str, object]],
    knowledge_context: SubmissionKnowledgeContext,
    record_problem_type_insight: Callable[..., object],
    record_error_fix_insight: Callable[..., object],
) -> None:
    for item in pending_problem_insights:
        iteration = resolve_submission_knowledge_iteration(
            value=item.get("iteration"),
            fallback_iteration=knowledge_context.iteration,
        )
        record_problem_type_insight(
            knowledge_paths=knowledge_paths,
            slug=slug,
            run_id=run_id,
            iteration=iteration,
            problem_types=problem_types,
            why_poor=str(item.get("why_poor") or ""),
            how_improved=str(item.get("how_improved") or ""),
            delta_offline=item.get("delta_offline") if isinstance(item.get("delta_offline"), (int, float)) else None,
            outcome_bucket=knowledge_context.outcome_bucket,
            submission_score=knowledge_context.online_score,
        )
    for item in pending_error_fixes:
        iteration = resolve_submission_knowledge_iteration(
            value=item.get("iteration"),
            fallback_iteration=knowledge_context.iteration,
        )
        record_error_fix_insight(
            knowledge_paths=knowledge_paths,
            slug=slug,
            run_id=run_id,
            iteration=iteration,
            problem_types=problem_types,
            error_message=str(item.get("error_message") or ""),
            fix_summary=str(item.get("fix_summary") or ""),
            resolved=bool(item.get("resolved", True)),
            outcome_bucket=knowledge_context.outcome_bucket,
            submission_score=knowledge_context.online_score,
        )


def record_submission_knowledge(
    *,
    knowledge_paths: object,
    slug: str,
    run_id: str,
    problem_types: list[str],
    pending_problem_insights: list[dict[str, object]],
    pending_error_fixes: list[dict[str, object]],
    submission_result: dict[str, object] | None,
    metric_direction: str,
    target_score: float | None,
    top1_score: float | None,
    load_diagnostics_text: Callable[[int], str],
    record_problem_type_insight: Callable[..., object],
    record_error_fix_insight: Callable[..., object],
) -> bool:
    knowledge_context = resolve_submission_knowledge_context(
        submission_result=submission_result,
        metric_direction=metric_direction,
        target_score=target_score,
        top1_score=top1_score,
    )
    if knowledge_context is None:
        return False
    ensure_submission_problem_insights(
        pending_problem_insights=pending_problem_insights,
        knowledge_context=knowledge_context,
        load_diagnostics_text=load_diagnostics_text,
    )
    record_submission_knowledge_entries(
        knowledge_paths=knowledge_paths,
        slug=slug,
        run_id=run_id,
        problem_types=problem_types,
        pending_problem_insights=pending_problem_insights,
        pending_error_fixes=pending_error_fixes,
        knowledge_context=knowledge_context,
        record_problem_type_insight=record_problem_type_insight,
        record_error_fix_insight=record_error_fix_insight,
    )
    return True


def resolve_submission_rank_payload(
    *,
    slug: str,
    context_dir: Path,
    direction: str,
    outcome: dict[str, object],
    dry_run: bool,
    leaderboard_rank_for_score: Callable[..., dict[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    rank = _to_int(outcome.get("rank"))
    total_teams = _to_int(outcome.get("total_teams"))
    rank_percentile = _to_float(outcome.get("rank_percentile"))
    rank_source = outcome.get("rank_source")

    if rank is not None:
        payload["rank"] = rank
    if total_teams is not None:
        payload["total_teams"] = total_teams
    if rank_percentile is not None:
        payload["rank_percentile"] = rank_percentile
    if isinstance(rank_source, str) and rank_source.strip():
        payload["rank_source"] = rank_source.strip()

    if rank is None or total_teams is None:
        score = _to_float(outcome.get("score"))
        if score is not None:
            try:
                estimate = leaderboard_rank_for_score(
                    slug=slug,
                    output_dir=context_dir,
                    score=score,
                    direction=direction,
                    dry_run=dry_run,
                )
            except Exception:  # noqa: BLE001
                estimate = {}
            est_rank = _to_int(estimate.get("rank"))
            est_total = _to_int(estimate.get("total_teams"))
            est_percentile = _to_float(estimate.get("rank_percentile"))
            if est_rank is not None:
                payload["estimated_rank"] = est_rank
            if est_total is not None:
                payload["estimated_total_teams"] = est_total
            if est_percentile is not None:
                payload["estimated_rank_percentile"] = est_percentile
            if est_rank is not None and isinstance(estimate.get("source"), str):
                payload["rank_estimate_source"] = "leaderboard_score_estimate"

    resolved_rank = _to_int(payload.get("rank"))
    resolved_total = _to_int(payload.get("total_teams"))
    if resolved_rank is not None and resolved_total is not None and resolved_total > 0:
        payload.setdefault("rank_percentile", resolved_rank / resolved_total)
    return payload


def format_rank_force_reason(
    *,
    rank: int,
    total_teams: int,
    rank_percentile: float | None,
    max_percentile: float,
    min_teams: int,
    source: str | None,
) -> str:
    resolved_percentile = (rank / total_teams) if rank_percentile is None and total_teams > 0 else rank_percentile
    percentile_text = f"{(resolved_percentile or 0.0) * 100:.2f}%" if resolved_percentile is not None else "n/a"
    source_text = f" source={source}" if source else ""
    return (
        "Leaderboard rank indicates large headroom for improvement: "
        f"{rank}/{total_teams} (percentile={percentile_text}, threshold={max_percentile * 100:.2f}%, "
        f"min_teams={min_teams}).{source_text}"
    )


def format_submission_rank_message(
    *,
    rank: int,
    total_teams: int,
    rank_percentile: float | None,
    source: str | None,
    estimated: bool = False,
) -> str:
    resolved_percentile = (rank / total_teams) if rank_percentile is None and total_teams > 0 else rank_percentile
    percentile_text = f"{resolved_percentile * 100:.2f}%" if resolved_percentile is not None else "n/a"
    source_text = f" source={source}" if source else ""
    prefix = "[yellow]submission rank estimate[/yellow]" if estimated else "[cyan]submission rank[/cyan]"
    return f"{prefix}: {rank}/{total_teams} (percentile={percentile_text}){source_text}"


def resolve_submission_rank_state(
    *,
    rank_payload: dict[str, object],
    rank_force_major_max_percentile: float,
    rank_force_major_min_teams: int,
    should_force_major_overhaul_by_rank: Callable[..., bool],
) -> SubmissionRankState:
    submission_rank = _to_int(rank_payload.get("rank"))
    submission_total_teams = _to_int(rank_payload.get("total_teams"))
    submission_rank_percentile = _to_float(rank_payload.get("rank_percentile"))
    submission_rank_estimate = _to_int(rank_payload.get("estimated_rank"))
    submission_total_teams_estimate = _to_int(rank_payload.get("estimated_total_teams"))
    submission_rank_percentile_estimate = _to_float(rank_payload.get("estimated_rank_percentile"))

    estimate_source_raw = rank_payload.get("rank_estimate_source")
    submission_rank_estimate_source = (
        estimate_source_raw.strip() if isinstance(estimate_source_raw, str) and estimate_source_raw.strip() else None
    )
    source_raw = rank_payload.get("rank_source")
    submission_rank_source = str(source_raw) if source_raw is not None else None

    messages: list[str] = []
    rank_forced_major_overhaul = False
    rank_force_reason: str | None = None
    if submission_rank is not None and submission_total_teams is not None and submission_total_teams > 0:
        if submission_rank_percentile is None:
            submission_rank_percentile = submission_rank / submission_total_teams
        messages.append(
            format_submission_rank_message(
                rank=submission_rank,
                total_teams=submission_total_teams,
                rank_percentile=submission_rank_percentile,
                source=submission_rank_source,
            )
        )
        rank_forced_major_overhaul = should_force_major_overhaul_by_rank(
            rank=submission_rank,
            total_teams=submission_total_teams,
            max_percentile=rank_force_major_max_percentile,
            min_teams=rank_force_major_min_teams,
        )
        if rank_forced_major_overhaul:
            rank_force_reason = format_rank_force_reason(
                rank=submission_rank,
                total_teams=submission_total_teams,
                rank_percentile=submission_rank_percentile,
                max_percentile=rank_force_major_max_percentile,
                min_teams=rank_force_major_min_teams,
                source=submission_rank_source,
            )
            messages.append(f"[yellow]rank guard[/yellow]: {rank_force_reason}")
    elif (
        submission_rank_estimate is not None
        and submission_total_teams_estimate is not None
        and submission_total_teams_estimate > 0
    ):
        if submission_rank_percentile_estimate is None:
            submission_rank_percentile_estimate = submission_rank_estimate / submission_total_teams_estimate
        messages.append(
            format_submission_rank_message(
                rank=submission_rank_estimate,
                total_teams=submission_total_teams_estimate,
                rank_percentile=submission_rank_percentile_estimate,
                source=submission_rank_estimate_source,
                estimated=True,
            )
        )

    return SubmissionRankState(
        rank_payload=rank_payload,
        rank=submission_rank,
        total_teams=submission_total_teams,
        rank_percentile=submission_rank_percentile,
        rank_source=submission_rank_source,
        estimated_rank=submission_rank_estimate,
        estimated_total_teams=submission_total_teams_estimate,
        estimated_rank_percentile=submission_rank_percentile_estimate,
        rank_estimate_source=submission_rank_estimate_source,
        force_major_overhaul=rank_forced_major_overhaul,
        force_reason=rank_force_reason,
        messages=tuple(messages),
    )


def format_iteration_submit_status_message(
    *,
    iteration: int,
    max_iterations: int,
    submit_enabled: bool,
    submit_allowed_by_gate: bool,
    submit_phase_state: str,
    quality_reasons: list[str],
    competition_faithfulness: dict[str, object] | None = None,
) -> str | None:
    if not submit_enabled:
        return None
    if submit_allowed_by_gate:
        return f"[cyan]submit[/cyan]: iter {iteration}/{max_iterations} attempting submission now."
    detail = ""
    if quality_reasons and submit_phase_state == "blocked_quality_guard":
        detail = f" reasons={','.join(quality_reasons)}"
    if isinstance(competition_faithfulness, dict):
        detail = f"{detail}{_format_competition_faithfulness_detail(competition_faithfulness)}"
    return (
        "[cyan]submit[/cyan]: "
        f"iter {iteration}/{max_iterations} not attempted yet "
        f"(state={submit_phase_state}{detail})."
    )


def _format_competition_faithfulness_detail(competition_faithfulness: dict[str, object]) -> str:
    metric_detail = ""
    expected_metric = str(competition_faithfulness.get("expected_metric") or "").strip()
    actual_metric = str(competition_faithfulness.get("actual_metric") or "").strip()
    if expected_metric or actual_metric:
        metric_detail = f" metric={actual_metric or 'unknown'}/{expected_metric or 'unknown'}"

    split_detail = ""
    expected_split = str(competition_faithfulness.get("expected_split_strategy") or "").strip()
    actual_split = str(competition_faithfulness.get("actual_split_strategy") or "").strip()
    if expected_split or actual_split:
        split_detail = f" split={actual_split or 'unknown'}/{expected_split or 'unknown'}"

    dataset_mode = str(competition_faithfulness.get("dataset_mode") or "").strip()
    dataset_detail = f" dataset_mode={dataset_mode}" if dataset_mode else ""
    return f"{metric_detail}{split_detail}{dataset_detail}"


def _meets_target(value: float, target: float, direction: str) -> bool:
    return value <= target if str(direction).lower() == "minimize" else value >= target


def _to_float(value: object) -> float | None:
    return parse_finite_float(value)


def _to_int(value: object) -> int | None:
    return parse_int(value, allow_float=True)


def find_campaign_candidate_for_submission(
    *,
    candidates: list[CampaignCandidate],
    submission_path: Path | None,
    run_id: str,
    iteration: int | None,
) -> CampaignCandidate | None:
    if submission_path is not None:
        resolved_submission = str(submission_path)
        for candidate in candidates:
            if candidate.submission_path == resolved_submission:
                return candidate
    if iteration is not None:
        for candidate in candidates:
            if candidate.run_id == run_id and candidate.iteration == iteration:
                return candidate
    return None


def resolve_submission_message(
    *,
    context_dir: Path,
    run_id: str,
    best_score: float | None,
    explicit_message: str | None,
    submission_path: Path | None,
    campaign_mode: str | None,
    target_direction: str | None,
) -> str:
    iteration = infer_iteration_from_submission_path(submission_path)
    iteration_suffix = f" i={iteration}" if isinstance(iteration, int) else ""
    if explicit_message:
        base_message = explicit_message
    elif best_score is None:
        base_message = f"kb {run_id}{iteration_suffix}"
    else:
        base_message = f"kb {run_id}{iteration_suffix} offline={best_score:.4f}"
    if normalize_campaign_mode(campaign_mode, deliverable_mode="leaderboard") != "top1":
        return base_message

    campaign_candidate = find_campaign_candidate_for_submission(
        candidates=list_candidates(candidate_registry_path(context_dir)),
        submission_path=submission_path,
        run_id=run_id,
        iteration=iteration,
    )
    if campaign_candidate is None:
        return base_message

    campaign_state = load_json_object(campaign_state_path(context_dir))
    if not isinstance(campaign_state, dict):
        campaign_state = {}
    direction = str(campaign_state.get("direction") or target_direction or "minimize")
    return format_campaign_submission_message(
        base_message=base_message,
        campaign_state=campaign_state,
        candidate=campaign_candidate,
        offline_score=best_score,
        direction=direction,
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


def build_kaggle_credentials_missing_abort_spec(
    *,
    stdout: str,
    stderr: str,
    output: str,
    exit_code: int | None,
    compute_error_fingerprint: Callable[[str, str], str],
) -> SubmitAbortSpec:
    stderr_tail = stderr or output
    return SubmitAbortSpec(
        fingerprint=compute_error_fingerprint(stdout, stderr_tail),
        error_kind="permanent",
        reason="kaggle_credentials_missing",
        message="Kaggle credentials not configured. Set ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY.",
        stdout_tail=stdout,
        stderr_tail=stderr_tail,
        exit_code=exit_code,
    )


def build_rules_not_accepted_abort_spec(
    *,
    exit_code: int | None,
    compute_error_fingerprint: Callable[[str, str], str],
) -> SubmitAbortSpec:
    return SubmitAbortSpec(
        fingerprint=compute_error_fingerprint("", "rules_not_accepted"),
        error_kind="permanent",
        reason="rules_not_accepted",
        message="Competition rules are not accepted; aborting submit stage for this run.",
        stdout_tail="",
        stderr_tail="rules_not_accepted",
        exit_code=exit_code,
    )


def build_local_submission_guardrail_abort_spec(
    *,
    error: object,
    exit_code: int | None,
    compute_error_fingerprint: Callable[[str, str], str],
) -> SubmitAbortSpec:
    stderr_tail = str(error)
    return SubmitAbortSpec(
        fingerprint=compute_error_fingerprint("", stderr_tail),
        error_kind="permanent",
        reason="local_submission_guardrail",
        message=f"Local submission guardrail blocked submit: {stderr_tail}",
        stdout_tail="",
        stderr_tail=stderr_tail,
        exit_code=exit_code,
    )


def resolve_local_submission_guardrail_abort_spec(
    *,
    error: object,
    compute_error_fingerprint: Callable[[str, str], str],
    default_exit_code: int = 1,
) -> SubmitAbortSpec:
    return build_local_submission_guardrail_abort_spec(
        error=error,
        exit_code=getattr(error, "exit_code", default_exit_code),
        compute_error_fingerprint=compute_error_fingerprint,
    )


def resolve_kaggle_cli_submit_abort_spec(
    *,
    error: BaseException,
    is_missing_credentials_error: Callable[[BaseException], bool],
    compute_error_fingerprint: Callable[[str, str], str],
) -> SubmitAbortSpec | None:
    if not is_missing_credentials_error(error):
        return None
    return build_kaggle_credentials_missing_abort_spec(
        stdout=str(getattr(error, "stdout", "") or ""),
        stderr=str(getattr(error, "stderr", "") or ""),
        output=str(getattr(error, "output", "") or ""),
        exit_code=getattr(error, "exit_code", None),
        compute_error_fingerprint=compute_error_fingerprint,
    )


def build_local_submission_validation_abort_spec(
    *,
    error: object,
    exit_code: int | None,
    compute_error_fingerprint: Callable[[str, str], str],
) -> SubmitAbortSpec:
    stderr_tail = str(error)
    return SubmitAbortSpec(
        fingerprint=compute_error_fingerprint("", stderr_tail),
        error_kind="validation",
        reason="local_submission_validation_failed",
        message="Local submission validation failed; Kaggle CLI submit is skipped.",
        stdout_tail="",
        stderr_tail=stderr_tail,
        exit_code=exit_code,
    )


def build_submission_polling_error_abort_spec(
    *,
    error: object,
    detail: object,
    normalize_detail: Callable[[str], str],
    compute_error_fingerprint: Callable[[str, str], str],
) -> SubmitAbortSpec:
    normalized_detail = normalize_detail(str(detail or error))
    stderr_tail = normalized_detail or str(error)
    return SubmitAbortSpec(
        fingerprint=compute_error_fingerprint("", stderr_tail),
        error_kind="transient",
        reason="submission_polling_error",
        message="Submission outcome polling failed; aborting submit stage for this run.",
        stdout_tail="",
        stderr_tail=stderr_tail,
        exit_code=None,
    )


def build_submission_outcome_abort_spec(
    *,
    decision: SubmitOutcomeAbortDecision,
    compute_error_fingerprint: Callable[[str, str], str],
) -> SubmitAbortSpec:
    return SubmitAbortSpec(
        fingerprint=compute_error_fingerprint("", decision.detail),
        error_kind=decision.error_kind,
        reason=decision.reason,
        message=decision.message,
        stdout_tail="",
        stderr_tail=decision.detail,
        exit_code=None,
    )


def build_submit_stage_error_action_abort_spec(
    *,
    action: SubmitStageErrorActionDecision,
    fingerprint: str,
    stdout: str,
    stderr: str,
    exit_code: int | None,
) -> SubmitAbortSpec:
    return SubmitAbortSpec(
        fingerprint=fingerprint,
        error_kind=action.error_kind,
        reason=action.reason,
        message=action.abort_message,
        stdout_tail=stdout,
        stderr_tail=stderr,
        exit_code=exit_code,
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
        infer_iteration=infer_iteration_from_submission_path,
        normalize_detail=normalize_detail,
        record_error_fix_insight=record_error_fix_insight,
    )


def decide_submission_outcome_abort(
    *,
    outcome_status: str,
    outcome_score: object,
    deliverable_mode: str,
    raw_detail: str,
) -> SubmitOutcomeAbortDecision:
    if outcome_status in FAILED_SUBMISSION_OUTCOME_STATUSES:
        return SubmitOutcomeAbortDecision(
            should_abort=True,
            error_kind="validation",
            reason=f"submission_poll_status_{outcome_status}",
            message=(
                f"Submission finished with error status '{outcome_status}' during polling; "
                "aborting submit stage for this run."
            ),
            detail=raw_detail or outcome_status,
        )

    if (
        outcome_status in SCORELESS_COMPLETE_SUBMISSION_OUTCOME_STATUSES
        and outcome_score is None
        and str(deliverable_mode or "").strip().lower() == "leaderboard"
    ):
        detail = raw_detail
        if not detail:
            detail = (
                "Kaggle submission completed without a public/private score. "
                "For leaderboard submissions this usually indicates a scoring error, "
                "such as an invalid notebook-generated submission file."
            )
        elif "submission file" not in detail.lower() and "scoring error" not in detail.lower():
            detail = (
                detail + "\nKaggle scoring error inferred: leaderboard submission file completed "
                "without a public/private score."
            )
        return SubmitOutcomeAbortDecision(
            should_abort=True,
            error_kind="validation",
            reason=f"submission_poll_status_{outcome_status}_no_score",
            message=(
                f"Submission finished with status '{outcome_status}' but no score; "
                "treating as scoring failure for this leaderboard run."
            ),
            detail=detail,
        )

    return SubmitOutcomeAbortDecision(should_abort=False)


def evaluate_submission_outcome_after_poll(
    *,
    slug: str,
    message: str,
    submitted_at: datetime,
    outcome: object,
    deliverable_mode: str,
    fetch_submission_rows: Callable[[str], list[dict[str, str]]],
    normalize_detail: Callable[[str], str],
) -> SubmissionOutcomePostPollDecision:
    if not isinstance(outcome, dict):
        return SubmissionOutcomePostPollDecision(
            outcome=outcome,
            abort_decision=SubmitOutcomeAbortDecision(should_abort=False),
        )

    normalized_outcome = dict(outcome)
    outcome_status = normalize_submission_outcome_status(normalized_outcome.get("status"))
    normalized_outcome["status"] = outcome_status
    raw_detail = ""
    if outcome_status in FAILED_SUBMISSION_OUTCOME_STATUSES or (
        outcome_status in SCORELESS_COMPLETE_SUBMISSION_OUTCOME_STATUSES
        and normalized_outcome.get("score") is None
        and str(deliverable_mode or "").strip().lower() == "leaderboard"
    ):
        raw_detail = build_submission_outcome_error_detail(
            slug=slug,
            message=message,
            submitted_at=submitted_at,
            outcome=normalized_outcome,
            fetch_submission_rows=fetch_submission_rows,
            normalize_detail=normalize_detail,
        )
    return SubmissionOutcomePostPollDecision(
        outcome=normalized_outcome,
        abort_decision=decide_submission_outcome_abort(
            outcome_status=outcome_status,
            outcome_score=normalized_outcome.get("score"),
            deliverable_mode=deliverable_mode,
            raw_detail=raw_detail,
        ),
    )


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
            abort_spec=build_submission_polling_error_abort_spec(
                error=exc,
                detail=exc.detail,
                normalize_detail=normalize_detail,
                compute_error_fingerprint=compute_error_fingerprint,
            ),
        )

    outcome_post_poll = evaluate_submission_outcome_after_poll(
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
            abort_spec=build_submission_outcome_abort_spec(
                decision=outcome_post_poll.abort_decision,
                compute_error_fingerprint=compute_error_fingerprint,
            ),
        )
    return SubmitOutcomeResolution(outcome=outcome_post_poll.outcome)


def build_submission_outcome_error_detail(
    *,
    slug: str,
    message: str,
    submitted_at: datetime,
    outcome: dict[str, object],
    fetch_submission_rows: Callable[[str], list[dict[str, str]]],
    normalize_detail: Callable[[str], str],
) -> str:
    row: dict[str, object] | None = None
    raw_payload = outcome.get("raw")
    if isinstance(raw_payload, dict):
        row = dict(raw_payload)
    try:
        rows = fetch_submission_rows(slug)
        selector = SubmissionOutcomeService(fetch_rows=lambda current_slug: rows)
        matched = selector._select_submission_row(rows=rows, message=message, submitted_at=submitted_at)  # noqa: SLF001
        if isinstance(matched, dict):
            row = dict(matched)
    except Exception:  # noqa: BLE001
        pass

    details: list[str] = []
    if row:
        row_message = _extract_submission_row_message(row)
        if row_message:
            details.append(f"Kaggle reported: {row_message}")
        details.append(f"Kaggle submission row: {json.dumps(row, ensure_ascii=True)}")
    elif raw_payload is not None:
        details.append(f"Kaggle submission raw payload: {json.dumps(raw_payload, ensure_ascii=True)}")
    else:
        details.append(f"Kaggle submission status: {outcome.get('status') or 'unknown'}")
    return normalize_detail("\n".join(details))


def _extract_submission_row_message(row: dict[str, object]) -> str:
    for key in (
        "errorDescription",
        "error_description",
        "failureReason",
        "failure_reason",
        "error",
        "message",
        "comments",
        "comment",
        "description",
    ):
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


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
        infer_iteration=infer_iteration_from_submission_path,
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
    count_csv_data_rows: Callable[[Path], int | None],
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
                count_csv_data_rows=count_csv_data_rows,
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
    count_csv_data_rows: Callable[[Path], int | None],
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
        count_csv_data_rows=count_csv_data_rows,
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
