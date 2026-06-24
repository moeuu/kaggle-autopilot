from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.campaign import (
    CampaignCandidate,
    best_score,
    classify_against_campaign_baseline,
    list_candidates,
    recommend_blend_pairs,
    score_gap,
)
from kagglebot.json_utils import write_json_object
from kagglebot.scalar_utils import finite_float as _to_float

REFERENCE_REPRODUCTION_REPORT_FILENAME = "reference_reproduction_report.json"
PORTFOLIO_PLAN_FILENAME = "portfolio_plan.json"
BLEND_REPORT_FILENAME = "blend_report.json"

REQUIRED_PORTFOLIO_CATEGORIES = [
    "reference_reproduction",
    "strong_single",
    "feature_variant",
    "validation_variant",
    "blend",
]


def reference_reproduction_report_path(context_dir: Path) -> Path:
    return context_dir / REFERENCE_REPRODUCTION_REPORT_FILENAME


def build_reference_reproduction_report(
    *,
    context_dir: Path,
    campaign_state: dict[str, object],
    method_registry: dict[str, object] | None,
    direction: str,
    current_candidate: CampaignCandidate | None = None,
    code_reference_score: float | None = None,
    code_reference_source: str | None = None,
) -> dict[str, object]:
    reference_methods = _reference_methods(method_registry or {})
    historical_best = _to_float(campaign_state.get("historical_best_score"))
    champion = _to_float(campaign_state.get("champion_score"))
    baseline = best_score(direction=direction, scores=[historical_best, champion, code_reference_score])
    candidate_score = current_candidate.offline_score if current_candidate is not None else None
    baseline_status = classify_against_campaign_baseline(
        candidate_score=candidate_score,
        direction=direction,
        historical_best_score=baseline,
        champion_score=None,
    )
    current_is_reference = current_candidate is not None and current_candidate.category == "reference_reproduction"
    if current_candidate is None:
        status = "pending"
        blocks_novelty = True
        gate_reason = "reference_reproduction_not_attempted"
    elif not current_is_reference and reference_methods:
        status = "pending"
        blocks_novelty = True
        gate_reason = "reference_reproduction_required_before_novelty"
    elif baseline_status == "regression":
        status = "blocked"
        blocks_novelty = True
        gate_reason = "reference_reproduction_below_campaign_baseline"
    else:
        status = "passed"
        blocks_novelty = False
        gate_reason = "reference_reproduction_satisfied"

    payload = {
        "version": 1,
        "updated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "blocks_novelty": blocks_novelty,
        "gate_reason": gate_reason,
        "direction": direction,
        "campaign_id": campaign_state.get("campaign_id"),
        "campaign_baseline_score": baseline,
        "historical_best_score": historical_best,
        "champion_score": champion,
        "code_reference_score": code_reference_score,
        "code_reference_source": code_reference_source,
        "candidate_id": current_candidate.candidate_id if current_candidate is not None else None,
        "candidate_score": candidate_score,
        "baseline_delta": score_gap(current=candidate_score, reference=baseline, direction=direction),
        "reference_sources": reference_methods,
        "required_actions": _reference_required_actions(status=status, gate_reason=gate_reason),
    }
    path = reference_reproduction_report_path(context_dir)
    write_json_object(path, payload)
    return payload


def build_candidate_portfolio_plan(
    *,
    iter_dir: Path,
    registry_path: Path,
    method_registry: dict[str, object] | None,
    validation_registry: dict[str, object] | None,
    campaign_state: dict[str, object],
    run_id: str,
    iteration: int,
    direction: str,
) -> dict[str, object]:
    candidates = list_candidates(registry_path)
    active_validation_profile = _active_validation_profile(validation_registry)
    method_by_category = _method_by_category(method_registry or {})
    existing_by_category: dict[str, list[CampaignCandidate]] = {}
    for candidate in candidates:
        existing_by_category.setdefault(candidate.category, []).append(candidate)

    planned: list[dict[str, object]] = []
    for category in REQUIRED_PORTFOLIO_CATEGORIES:
        existing = _best_candidate(existing_by_category.get(category, []), direction=direction)
        method = method_by_category.get(category)
        candidate_id = existing.candidate_id if existing is not None else f"{run_id}-i{iteration:03d}-{category}"
        planned.append(
            {
                "candidate_id": candidate_id,
                "category": category,
                "status": "available" if existing is not None else "planned",
                "method_id": _method_id(method, fallback=f"{category}-default"),
                "validation_profile_id": active_validation_profile,
                "expected_oof_path": str(iter_dir / f"{candidate_id}.oof.npy"),
                "expected_prediction_path": str(iter_dir / f"{candidate_id}.test.npy"),
                "offline_score": existing.offline_score if existing is not None else None,
                "model_family": existing.model_family if existing is not None else None,
                "private_robustness_score": private_robustness_score(
                    existing,
                    campaign_state=campaign_state,
                )
                if existing is not None
                else None,
                "rank_score": _candidate_rank_score(existing, campaign_state=campaign_state)
                if existing is not None
                else None,
            }
        )

    blend_pairs = recommend_blend_pairs(candidates, direction=direction, max_pairs=5)
    payload = {
        "version": 1,
        "updated_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "iteration": iteration,
        "direction": direction,
        "campaign_id": campaign_state.get("campaign_id"),
        "active_validation_profile": active_validation_profile,
        "validation_priority": bool((validation_registry or {}).get("priority")),
        "required_categories": REQUIRED_PORTFOLIO_CATEGORIES,
        "missing_categories": [item["category"] for item in planned if item["status"] == "planned"],
        "candidates": planned,
        "blend_pairs": [{"left": left, "right": right} for left, right in blend_pairs],
        "next_action": _portfolio_next_action(planned, validation_registry or {}),
    }
    path = iter_dir / PORTFOLIO_PLAN_FILENAME
    write_json_object(path, payload)
    return payload


def build_blend_report(
    *,
    iter_dir: Path,
    registry_path: Path,
    campaign_state: dict[str, object],
    validation_registry: dict[str, object] | None,
    direction: str,
) -> dict[str, object]:
    candidates = list_candidates(registry_path)
    validation_trust = _validation_trust(campaign_state)
    validation_priority = bool((validation_registry or {}).get("priority"))
    pairs = recommend_blend_pairs(candidates, direction=direction, max_pairs=8)
    if validation_priority and validation_trust < 0.6:
        status = "deferred_for_validation_redesign"
        next_action = "Run split redesign before investing submissions in new blends."
    elif not pairs:
        status = "insufficient_diverse_candidates"
        next_action = "Create low-correlation OOF candidates before blending."
    else:
        status = "ready"
        next_action = "Generate rank, logit, weighted, and hill-climb blend candidates from OOF predictions."
    payload = {
        "version": 1,
        "updated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "direction": direction,
        "validation_trust": validation_trust,
        "validation_priority": validation_priority,
        "candidate_count": len(candidates),
        "blend_methods": ["rank_average", "logit_average", "weighted_average", "hill_climb_blend", "light_stack"],
        "recommended_pairs": [{"left": left, "right": right} for left, right in pairs],
        "next_action": next_action,
    }
    path = iter_dir / BLEND_REPORT_FILENAME
    write_json_object(path, payload)
    return payload


def select_method_id_for_category(method_registry: dict[str, object] | None, category: str) -> str | None:
    method = _method_by_category(method_registry or {}).get(category)
    return _method_id(method, fallback=None)


def private_robustness_score(candidate: CampaignCandidate, *, campaign_state: dict[str, object]) -> float:
    score = 0.7
    if candidate.fold_scores:
        mean = sum(candidate.fold_scores) / len(candidate.fold_scores)
        variance = sum((item - mean) ** 2 for item in candidate.fold_scores) / len(candidate.fold_scores)
        fold_std = math.sqrt(variance)
        score -= min(0.25, fold_std * 4.0)
    high_corr = [value for value in candidate.prediction_correlation.values() if value >= 0.98]
    score -= min(0.2, 0.05 * len(high_corr))
    if candidate.public_score is not None and candidate.offline_score is not None:
        delta = abs(float(candidate.public_score) - float(candidate.offline_score))
        score -= min(0.2, delta)
    if _to_float(campaign_state.get("offline_online_correlation")) is not None:
        corr = float(campaign_state["offline_online_correlation"])
        if corr < 0.25:
            score -= 0.2
    return round(max(0.0, min(1.0, score)), 6)


def _reference_methods(method_registry: dict[str, object]) -> list[dict[str, object]]:
    methods = method_registry.get("methods")
    if not isinstance(methods, list):
        return []
    refs: list[dict[str, object]] = []
    for item in methods:
        if not isinstance(item, dict) or item.get("status") != "active":
            continue
        category = str(item.get("candidate_category") or "")
        source_type = str(item.get("source_type") or "")
        if category == "reference_reproduction" or source_type in {"competition_specific", "discussion"}:
            refs.append(
                {
                    "method_id": item.get("method_id"),
                    "name": item.get("name"),
                    "source_ids": item.get("source_ids") if isinstance(item.get("source_ids"), list) else [],
                    "source_type": source_type,
                    "summary": item.get("summary"),
                }
            )
    return refs[:5]


def _reference_required_actions(*, status: str, gate_reason: str) -> list[str]:
    if status == "passed":
        return []
    if gate_reason == "reference_reproduction_below_campaign_baseline":
        return [
            "Diagnose implementation, feature, split, metric, and dependency differences against the reference source.",
            "Do not spend the next iteration on novelty until the reference gap is explained.",
        ]
    return [
        "Pick at least one competition-specific reference source and reproduce it as a candidate.",
        "Record attribution/source ids and the reference reproduction score.",
    ]


def _method_by_category(method_registry: dict[str, object]) -> dict[str, dict[str, object]]:
    methods = method_registry.get("methods")
    if not isinstance(methods, list):
        return {}
    selected: dict[str, dict[str, object]] = {}
    for item in methods:
        if not isinstance(item, dict) or item.get("status") != "active":
            continue
        category = str(item.get("candidate_category") or "strong_single")
        if category not in selected:
            selected[category] = item
    return selected


def _method_id(method: dict[str, object] | None, *, fallback: str | None) -> str | None:
    if not isinstance(method, dict):
        return fallback
    value = str(method.get("method_id") or "").strip()
    return value or fallback


def _active_validation_profile(validation_registry: dict[str, object] | None) -> str:
    if not isinstance(validation_registry, dict):
        return "default_cv"
    active = str(validation_registry.get("active_profile") or "").strip()
    return active or "default_cv"


def _best_candidate(candidates: list[CampaignCandidate], *, direction: str) -> CampaignCandidate | None:
    scored = [candidate for candidate in candidates if candidate.offline_score is not None]
    if not scored:
        return candidates[0] if candidates else None
    reverse = str(direction).lower() == "maximize"
    return sorted(scored, key=lambda item: float(item.offline_score), reverse=reverse)[0]


def _candidate_rank_score(candidate: CampaignCandidate, *, campaign_state: dict[str, object]) -> float:
    baseline = best_score(
        direction=candidate.direction,
        scores=[campaign_state.get("historical_best_score"), campaign_state.get("champion_score")],
    )
    gain = score_gap(current=candidate.offline_score, reference=baseline, direction=candidate.direction) or 0.0
    robust = private_robustness_score(candidate, campaign_state=campaign_state)
    return round(gain + (0.1 * robust), 6)


def _portfolio_next_action(planned: list[dict[str, object]], validation_registry: dict[str, object]) -> str:
    if bool(validation_registry.get("priority")):
        return "Run validation_variant candidates first and choose the split profile with better offline-online fit."
    missing = [str(item["category"]) for item in planned if item.get("status") == "planned"]
    if missing:
        return "Create missing portfolio candidates: " + ", ".join(missing)
    return "Promote low-correlation strong candidates into blend and allocator scoring."


def _validation_trust(campaign_state: dict[str, object]) -> float:
    corr = _to_float(campaign_state.get("offline_online_correlation"))
    if corr is None:
        return 0.5
    return max(0.0, min(1.0, (corr + 1.0) / 2.0))
