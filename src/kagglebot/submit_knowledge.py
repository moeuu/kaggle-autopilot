from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from kagglebot.scalar_utils import parse_finite_float
from kagglebot.score_utils import score_gap
from kagglebot.submission_policy import meets_target


@dataclass(frozen=True)
class SubmissionKnowledgeContext:
    online_score: float
    outcome_bucket: str
    iteration: int | None


def classify_submission_outcome(
    *,
    score: float,
    direction: str,
    target_score: float | None,
    top1_score: float | None,
) -> str:
    if target_score is not None and meets_target(score, target_score, direction):
        return "good"
    if top1_score is not None:
        gap = -(score_gap(current=score, reference=top1_score, direction=direction) or 0.0)
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
