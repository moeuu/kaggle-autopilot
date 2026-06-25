from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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
from kagglebot.submission.outcome_service import SubmissionOutcomeService
from kagglebot.submit_error_classification import classify_submit_error_with_output_fallback


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
class SubmissionOutcomePostPollDecision:
    outcome: object
    abort_decision: SubmitOutcomeAbortDecision


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


def submission_score_for_tracking(*, offline_score: float, online_score: float | None) -> tuple[float, str]:
    if isinstance(online_score, (int, float)):
        value = float(online_score)
        if math.isfinite(value):
            return value, "submission_public_score"
    return float(offline_score), "offline"


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
