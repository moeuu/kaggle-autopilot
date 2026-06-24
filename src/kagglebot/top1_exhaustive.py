from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.campaign import (
    CampaignCandidate,
    allocate_submission,
    best_score,
    candidate_registry_path,
    classify_against_campaign_baseline,
    list_candidates,
)
from kagglebot.json_utils import write_json_object

WIN_CONTRACT_FILENAME = "win_contract.json"
PRIVATE_ROBUSTNESS_REPORT_FILENAME = "private_robustness_report.json"
PORTFOLIO_OPTIMIZER_REPORT_FILENAME = "portfolio_optimizer_report.json"
TOP1_EXHAUSTION_REPORT_FILENAME = "top1_exhaustion_report.json"

Top1SubmitPolicy = str

_REQUIRED_EXHAUSTIVE_CATEGORIES = [
    "reference_reproduction",
    "strong_single",
    "feature_variant",
    "validation_variant",
    "blend",
    "calibration",
]


def normalize_top1_submit_policy(value: str | None) -> Top1SubmitPolicy:
    normalized = str(value or "value_only").strip().lower()
    if normalized in {"value_only", "calibration", "final_lock"}:
        return normalized
    raise ValueError("top1_submit_policy must be one of: value_only, calibration, final_lock")


def win_contract_path(context_dir: Path) -> Path:
    return context_dir / WIN_CONTRACT_FILENAME


def private_robustness_report_path(context_dir: Path) -> Path:
    return context_dir / PRIVATE_ROBUSTNESS_REPORT_FILENAME


def portfolio_optimizer_report_path(iter_dir: Path) -> Path:
    return iter_dir / PORTFOLIO_OPTIMIZER_REPORT_FILENAME


def top1_exhaustion_report_path(context_dir: Path) -> Path:
    return context_dir / TOP1_EXHAUSTION_REPORT_FILENAME


def build_win_contract(
    *,
    context_dir: Path,
    slug: str,
    direction: str,
    campaign_state: dict[str, object],
    top1_info: dict[str, object] | None,
    submission_history: dict[str, object] | None,
    method_registry: dict[str, object] | None,
    source_registry: dict[str, object] | None,
    validation_registry: dict[str, object] | None,
    submission_limit_per_day: int | None = None,
) -> dict[str, object]:
    top1_score = _to_float((top1_info or {}).get("score"))
    historical_best = _to_float((submission_history or {}).get("best_score"))
    champion = _to_float(campaign_state.get("champion_score"))
    baseline = best_score(direction=direction, scores=[top1_score, historical_best, champion])
    active_methods = _string_list((method_registry or {}).get("active_method_ids"))
    active_sources = _string_list((source_registry or {}).get("active_source_ids"))
    payload = {
        "version": 1,
        "updated_at": datetime.now(UTC).isoformat(),
        "slug": slug,
        "direction": direction,
        "campaign_id": campaign_state.get("campaign_id"),
        "win_score_contract": {
            "top1_score": top1_score,
            "historical_best_score": historical_best,
            "champion_score": champion,
            "campaign_baseline_score": baseline,
            "top1_gap": campaign_state.get("top1_gap"),
        },
        "validation_contract": {
            "active_profile": (validation_registry or {}).get("active_profile"),
            "priority": bool((validation_registry or {}).get("priority")),
            "minimum_trust": 0.6,
        },
        "method_contract": {
            "required_categories": _REQUIRED_EXHAUSTIVE_CATEGORIES,
            "active_method_ids": active_methods,
            "active_source_ids": active_sources,
            "reference_required": True,
        },
        "submission_contract": {
            "submission_limit_per_day": submission_limit_per_day,
            "no_rules_auto_acceptance": True,
            "no_limit_bypass": True,
            "no_duplicate_submit": True,
        },
        "done_definition": [
            "reference reproduction attempted or explicitly diagnosed",
            "validation profile adopted with evidence or marked untrusted",
            "required candidate categories attempted or blocked with reasons",
            "portfolio optimizer selects no positive-value legal submission",
            "top1 exhaustion report records remaining gaps and blockers",
        ],
    }
    write_json_object(win_contract_path(context_dir), payload)
    return payload


def build_private_robustness_report(
    *,
    context_dir: Path,
    registry_path: Path,
    campaign_state: dict[str, object],
    validation_lab_report: dict[str, object] | None,
    direction: str,
) -> dict[str, object]:
    candidates = list_candidates(registry_path)
    active_profile = _active_validation_profile(validation_lab_report, campaign_state)
    rows = []
    for candidate in candidates:
        row = _candidate_robustness(candidate, campaign_state=campaign_state, active_profile=active_profile)
        rows.append(row)
    rows = sorted(rows, key=lambda item: float(item["private_robustness_score"]), reverse=True)
    payload = {
        "version": 1,
        "updated_at": datetime.now(UTC).isoformat(),
        "campaign_id": campaign_state.get("campaign_id"),
        "direction": direction,
        "active_validation_profile": active_profile,
        "candidate_count": len(candidates),
        "top_candidates": rows[:20],
        "risk_summary": {
            "public_overfit_risk_count": sum(1 for row in rows if "public_overfit_risk" in row["risk_flags"]),
            "high_correlation_count": sum(1 for row in rows if "high_prediction_correlation" in row["risk_flags"]),
            "baseline_regression_count": sum(1 for row in rows if "below_campaign_baseline" in row["risk_flags"]),
        },
    }
    write_json_object(private_robustness_report_path(context_dir), payload)
    return payload


def build_portfolio_optimizer_report(
    *,
    iter_dir: Path,
    registry_path: Path,
    campaign_state: dict[str, object],
    validation_registry: dict[str, object] | None,
    private_robustness_report: dict[str, object] | None,
    remaining_daily_slots: int | None,
    submit_policy: str | None,
    direction: str,
) -> dict[str, object]:
    policy = normalize_top1_submit_policy(submit_policy)
    candidates = list_candidates(registry_path)
    robustness_by_id = _robustness_by_candidate(private_robustness_report)
    validation_trust = _validation_trust(campaign_state, validation_registry)
    ranked = []
    for candidate in candidates:
        robustness = robustness_by_id.get(candidate.candidate_id, candidate.private_robustness_score)
        novelty = _novelty(candidate)
        regression_risk = _regression_risk(candidate, campaign_state=campaign_state, direction=direction)
        allocation = allocate_submission(
            candidate=candidate,
            campaign_state=campaign_state,
            remaining_daily_slots=remaining_daily_slots,
            novelty=novelty,
            validation_trust=validation_trust,
            regression_risk=regression_risk,
            information_value=_information_value(candidate, robustness=robustness, novelty=novelty),
            calibration_exception=policy == "calibration" and candidate.category == "calibration",
            force=False,
        )
        policy_allowed = _policy_allows_candidate(
            policy=policy,
            candidate=candidate,
            allocation_allowed=allocation.allow_submit,
        )
        ranked.append(
            {
                "candidate_id": candidate.candidate_id,
                "category": candidate.category,
                "offline_score": candidate.offline_score,
                "public_score": candidate.public_score,
                "method_id": candidate.method_id,
                "validation_profile_id": candidate.validation_profile_id,
                "private_robustness_score": robustness,
                "allocation": allocation.to_payload(),
                "policy_allowed": policy_allowed,
                "submit_value": round(
                    float(allocation.allocation_score) + (0.35 * float(robustness or 0.0)),
                    6,
                ),
            }
        )
    ranked = sorted(ranked, key=lambda item: float(item["submit_value"]), reverse=True)
    selected = next(
        (item for item in ranked if bool(item["policy_allowed"]) and bool(item["allocation"]["allow_submit"])),
        None,
    )
    payload = {
        "version": 1,
        "updated_at": datetime.now(UTC).isoformat(),
        "campaign_id": campaign_state.get("campaign_id"),
        "direction": direction,
        "submit_policy": policy,
        "remaining_daily_slots": remaining_daily_slots,
        "validation_trust": validation_trust,
        "selected_candidate_id": selected.get("candidate_id") if selected else None,
        "decision": "submit_selected_candidate" if selected else "no_positive_value_submission",
        "ranked_candidates": ranked[:50],
    }
    write_json_object(portfolio_optimizer_report_path(iter_dir), payload)
    return payload


def build_top1_exhaustion_report(
    *,
    context_dir: Path,
    run_id: str,
    iteration: int,
    campaign_state: dict[str, object],
    win_contract: dict[str, object] | None,
    method_registry: dict[str, object] | None,
    source_registry: dict[str, object] | None,
    validation_lab_report: dict[str, object] | None,
    private_robustness_report: dict[str, object] | None,
    portfolio_optimizer_report: dict[str, object] | None,
    experiment_graph: dict[str, object] | None,
) -> dict[str, object]:
    registry_path = candidate_registry_path(context_dir)
    candidates = list_candidates(registry_path)
    attempted_categories = sorted({candidate.category for candidate in candidates})
    missing_categories = [item for item in _REQUIRED_EXHAUSTIVE_CATEGORIES if item not in attempted_categories]
    blocked_nodes = _string_list((experiment_graph or {}).get("blocked_nodes"))
    active_methods = _string_list((method_registry or {}).get("active_method_ids"))
    active_sources = _string_list((source_registry or {}).get("active_source_ids"))
    selected_candidate = (portfolio_optimizer_report or {}).get("selected_candidate_id")
    payload = {
        "version": 1,
        "updated_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "iteration": iteration,
        "campaign_id": campaign_state.get("campaign_id"),
        "top1_gap": campaign_state.get("top1_gap"),
        "candidate_count": len(candidates),
        "attempted_categories": attempted_categories,
        "missing_categories": missing_categories,
        "active_method_count": len(active_methods),
        "active_source_count": len(active_sources),
        "validation_status": (validation_lab_report or {}).get("status"),
        "active_validation_profile": (validation_lab_report or {}).get("active_profile")
        or campaign_state.get("active_validation_profile"),
        "selected_candidate_id": selected_candidate,
        "portfolio_decision": (portfolio_optimizer_report or {}).get("decision"),
        "blocked_nodes": blocked_nodes,
        "exhaustion_status": _exhaustion_status(
            missing_categories=missing_categories,
            selected_candidate=selected_candidate,
            validation_lab_report=validation_lab_report,
            active_sources=active_sources,
        ),
        "remaining_work": _remaining_work(
            missing_categories=missing_categories,
            selected_candidate=selected_candidate,
            validation_lab_report=validation_lab_report,
            active_sources=active_sources,
            blocked_nodes=blocked_nodes,
        ),
        "win_contract_summary": (win_contract or {}).get("win_score_contract"),
        "private_robustness_summary": (private_robustness_report or {}).get("risk_summary"),
    }
    write_json_object(top1_exhaustion_report_path(context_dir), payload)
    _write_markdown(top1_exhaustion_report_path(context_dir).with_suffix(".md"), payload)
    return payload


def _candidate_robustness(
    candidate: CampaignCandidate,
    *,
    campaign_state: dict[str, object],
    active_profile: str | None,
) -> dict[str, object]:
    score = 0.72
    flags: list[str] = []
    if candidate.fold_scores:
        mean = sum(candidate.fold_scores) / len(candidate.fold_scores)
        fold_std = math.sqrt(sum((item - mean) ** 2 for item in candidate.fold_scores) / len(candidate.fold_scores))
        score -= min(0.25, fold_std * 4.0)
        if fold_std > 0.04:
            flags.append("fold_instability")
    high_corr = [value for value in candidate.prediction_correlation.values() if value >= 0.98]
    if high_corr:
        score -= min(0.2, 0.05 * len(high_corr))
        flags.append("high_prediction_correlation")
    if candidate.public_score is not None and candidate.offline_score is not None:
        delta = abs(float(candidate.public_score) - float(candidate.offline_score))
        score -= min(0.2, delta)
        if delta > 0.03:
            flags.append("public_overfit_risk")
    status = classify_against_campaign_baseline(
        candidate_score=candidate.offline_score,
        direction=candidate.direction,
        historical_best_score=campaign_state.get("historical_best_score"),  # type: ignore[arg-type]
        champion_score=campaign_state.get("champion_score"),  # type: ignore[arg-type]
    )
    if status == "regression":
        score -= 0.2
        flags.append("below_campaign_baseline")
    if active_profile and candidate.validation_profile_id and candidate.validation_profile_id != active_profile:
        score -= 0.08
        flags.append("inactive_validation_profile")
    return {
        "candidate_id": candidate.candidate_id,
        "category": candidate.category,
        "offline_score": candidate.offline_score,
        "public_score": candidate.public_score,
        "validation_profile_id": candidate.validation_profile_id,
        "private_robustness_score": round(max(0.0, min(1.0, score)), 6),
        "risk_flags": flags,
    }


def _active_validation_profile(
    validation_lab_report: dict[str, object] | None,
    campaign_state: dict[str, object],
) -> str | None:
    value = (validation_lab_report or {}).get("active_profile") or campaign_state.get("active_validation_profile")
    return str(value) if value else None


def _robustness_by_candidate(report: dict[str, object] | None) -> dict[str, float]:
    rows = (report or {}).get("top_candidates")
    if not isinstance(rows, list):
        return {}
    result: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("candidate_id") or "")
        score = _to_float(row.get("private_robustness_score"))
        if candidate_id and score is not None:
            result[candidate_id] = score
    return result


def _validation_trust(campaign_state: dict[str, object], validation_registry: dict[str, object] | None) -> float:
    corr = _to_float((validation_registry or {}).get("offline_online_correlation"))
    if corr is None:
        corr = _to_float(campaign_state.get("offline_online_correlation"))
    if corr is None:
        return 0.5
    return round(max(0.0, min(1.0, (corr + 1.0) / 2.0)), 6)


def _novelty(candidate: CampaignCandidate) -> float:
    if candidate.category in {"blend", "validation_variant", "calibration"}:
        return 0.72
    if candidate.category == "reference_reproduction":
        return 0.35
    return 0.55


def _regression_risk(
    candidate: CampaignCandidate,
    *,
    campaign_state: dict[str, object],
    direction: str,
) -> float:
    status = classify_against_campaign_baseline(
        candidate_score=candidate.offline_score,
        direction=direction,
        historical_best_score=campaign_state.get("historical_best_score"),  # type: ignore[arg-type]
        champion_score=campaign_state.get("champion_score"),  # type: ignore[arg-type]
    )
    if status == "regression":
        return 0.85
    if status == "unknown":
        return 0.45
    return 0.18


def _information_value(candidate: CampaignCandidate, *, robustness: float | None, novelty: float) -> float:
    value = 0.25 + (0.35 * novelty) + (0.25 * float(robustness or 0.0))
    if candidate.category in {"validation_variant", "calibration"}:
        value += 0.2
    if candidate.category == "blend":
        value += 0.1
    return round(max(0.0, min(1.0, value)), 6)


def _policy_allows_candidate(*, policy: str, candidate: CampaignCandidate, allocation_allowed: bool) -> bool:
    if policy == "calibration":
        return candidate.category == "calibration" or allocation_allowed
    if policy == "final_lock":
        return allocation_allowed and candidate.category != "calibration"
    return allocation_allowed


def _exhaustion_status(
    *,
    missing_categories: list[str],
    selected_candidate: object,
    validation_lab_report: dict[str, object] | None,
    active_sources: list[str],
) -> str:
    validation_status = str((validation_lab_report or {}).get("status") or "")
    if missing_categories:
        return "not_exhausted_missing_candidate_categories"
    if not active_sources:
        return "not_exhausted_no_active_sources"
    if validation_status not in {"active", "monitoring"}:
        return "not_exhausted_validation_untrusted"
    if selected_candidate:
        return "active_submission_opportunity"
    return "exhausted_no_positive_value_submission"


def _remaining_work(
    *,
    missing_categories: list[str],
    selected_candidate: object,
    validation_lab_report: dict[str, object] | None,
    active_sources: list[str],
    blocked_nodes: list[str],
) -> list[str]:
    work: list[str] = []
    if missing_categories:
        work.append("Run or explicitly block candidate categories: " + ", ".join(missing_categories))
    if not active_sources:
        work.append("Refresh research scout until at least one active attributed source exists.")
    if str((validation_lab_report or {}).get("status") or "") not in {"active", "monitoring"}:
        work.append("Run validation lab or record why validation evidence is unavailable.")
    if blocked_nodes:
        work.append("Resolve blocked experiment graph nodes: " + ", ".join(blocked_nodes[:8]))
    if selected_candidate:
        work.append(f"Submit or deliberately hold selected candidate: {selected_candidate}")
    if not work:
        work.append("No positive-value legal submission remains under current evidence and budget.")
    return work


def _write_markdown(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# Top1 Exhaustion Report",
        "",
        f"- status: {payload.get('exhaustion_status')}",
        f"- campaign_id: {payload.get('campaign_id')}",
        f"- top1_gap: {payload.get('top1_gap')}",
        f"- candidate_count: {payload.get('candidate_count')}",
        f"- selected_candidate_id: {payload.get('selected_candidate_id')}",
        "",
        "## Remaining Work",
        "",
    ]
    remaining_work = payload.get("remaining_work") if isinstance(payload.get("remaining_work"), list) else []
    for item in remaining_work:
        lines.append(f"- {item}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _to_float(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed
