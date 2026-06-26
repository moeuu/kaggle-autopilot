from __future__ import annotations

from pathlib import Path

from kagglebot import submit_stage_duplicate as _submit_stage_duplicate
from kagglebot.campaign import (
    CampaignCandidate,
    campaign_state_path,
    candidate_registry_path,
    format_campaign_submission_message,
    list_candidates,
    normalize_campaign_mode,
)
from kagglebot.json_utils import load_json_object


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
    iteration = _submit_stage_duplicate.infer_iteration_from_submission_path(submission_path)
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
