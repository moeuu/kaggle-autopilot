from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.campaign import CampaignCandidate, list_candidates
from kagglebot.json_utils import write_json_object
from kagglebot.scalar_utils import non_nan_float as _to_float

VALIDATION_LAB_REPORT_FILENAME = "validation_lab_report.json"

ValidationLabMode = str


def normalize_validation_lab_mode(value: str | None) -> ValidationLabMode:
    normalized = str(value or "auto").strip().lower()
    if normalized in {"auto", "off", "force"}:
        return normalized
    raise ValueError("validation_lab must be one of: auto, off, force")


def validation_lab_report_path(context_dir: Path) -> Path:
    return context_dir / VALIDATION_LAB_REPORT_FILENAME


def run_validation_lab(
    *,
    context_dir: Path,
    validation_registry: dict[str, object] | None,
    candidate_registry_path: Path,
    campaign_state: dict[str, object] | None,
    mode: str | None = "auto",
) -> dict[str, object]:
    lab_mode = normalize_validation_lab_mode(mode)
    registry = dict(validation_registry or {})
    candidates = list_candidates(candidate_registry_path)
    if lab_mode == "off":
        payload = _report(
            mode=lab_mode,
            status="disabled",
            registry=registry,
            candidates=candidates,
            active_profile=_active_profile(registry),
            reason="validation_lab_disabled",
        )
        _write_report(context_dir, payload)
        return payload

    priority = bool(registry.get("priority"))
    if not priority and lab_mode != "force" and not _low_trust_campaign(campaign_state):
        payload = _report(
            mode=lab_mode,
            status="monitoring",
            registry=registry,
            candidates=candidates,
            active_profile=_active_profile(registry),
            reason="validation_trust_not_yet_problematic",
        )
        _write_report(context_dir, payload)
        _write_registry_if_changed(context_dir, registry)
        return payload

    profiles = _profile_evidence(registry=registry, candidates=candidates, campaign_state=campaign_state)
    active = _select_profile(profiles=profiles, fallback=_active_profile(registry))
    updated_registry = dict(registry)
    updated_registry["updated_at"] = datetime.now(UTC).isoformat()
    updated_registry["active_profile"] = active
    updated_registry["lab_status"] = "active"
    updated_registry["profiles"] = profiles
    updated_registry["next_action"] = (
        "validation_redesign" if bool(updated_registry.get("priority")) else "monitor_offline_online_fit"
    )

    payload = _report(
        mode=lab_mode,
        status="active",
        registry=updated_registry,
        candidates=candidates,
        active_profile=active,
        reason="validation_profiles_calibrated",
    )
    _write_report(context_dir, payload)
    _write_registry_if_changed(context_dir, updated_registry)
    return payload


def _profile_evidence(
    *,
    registry: dict[str, object],
    candidates: list[CampaignCandidate],
    campaign_state: dict[str, object] | None,
) -> list[dict[str, object]]:
    raw_profiles = registry.get("profiles")
    profiles = [dict(item) for item in raw_profiles if isinstance(item, dict)] if isinstance(raw_profiles, list) else []
    if not profiles:
        profiles = [
            {"profile_id": "default_cv", "split_family": "default", "priority": 0.3},
            {"profile_id": "group_or_proxy_cv", "split_family": "group_or_proxy", "priority": 0.55},
            {"profile_id": "time_aware_cv", "split_family": "time", "priority": 0.45},
            {"profile_id": "leak_safe_cv", "split_family": "leak_safe", "priority": 0.5},
            {"profile_id": "adversarial_proxy_cv", "split_family": "proxy", "priority": 0.52},
        ]

    by_profile: dict[str, list[CampaignCandidate]] = {}
    for candidate in candidates:
        profile_id = candidate.validation_profile_id or "default_cv"
        by_profile.setdefault(profile_id, []).append(candidate)

    campaign_corr = _to_float((campaign_state or {}).get("offline_online_correlation"))
    updated: list[dict[str, object]] = []
    for profile in profiles:
        profile_id = str(profile.get("profile_id") or "default_cv")
        profile_candidates = by_profile.get(profile_id, [])
        corr = _offline_online_correlation(profile_candidates)
        pair_count = sum(
            1
            for candidate in profile_candidates
            if candidate.offline_score is not None and candidate.public_score is not None
        )
        evidence_score = _evidence_score(
            profile=profile,
            pair_count=pair_count,
            profile_corr=corr,
            campaign_corr=campaign_corr,
            candidate_count=len(profile_candidates),
        )
        enriched = dict(profile)
        enriched["offline_online_correlation"] = corr if corr is not None else profile.get("offline_online_correlation")
        enriched["evidence_score"] = evidence_score
        enriched["candidate_count"] = len(profile_candidates)
        enriched["public_pair_count"] = pair_count
        enriched["run_status"] = "evaluated" if profile_candidates else str(profile.get("run_status") or "planned")
        updated.append(enriched)

    active = _select_profile(profiles=updated, fallback=_active_profile(registry))
    for profile in updated:
        profile["adoption_status"] = "adopted" if str(profile.get("profile_id")) == active else "candidate"
    return sorted(updated, key=lambda item: float(item.get("evidence_score") or 0.0), reverse=True)


def _select_profile(*, profiles: list[dict[str, object]], fallback: str) -> str:
    if not profiles:
        return fallback
    ranked = sorted(
        profiles,
        key=lambda item: (float(item.get("evidence_score") or 0.0), float(item.get("priority") or 0.0)),
        reverse=True,
    )
    return str(ranked[0].get("profile_id") or fallback)


def _evidence_score(
    *,
    profile: dict[str, object],
    pair_count: int,
    profile_corr: float | None,
    campaign_corr: float | None,
    candidate_count: int,
) -> float:
    priority = float(_to_float(profile.get("priority")) or 0.0)
    corr = profile_corr if profile_corr is not None else campaign_corr
    corr_score = 0.5 if corr is None else max(0.0, min(1.0, (float(corr) + 1.0) / 2.0))
    support = min(1.0, (0.35 * pair_count) + (0.12 * candidate_count))
    return round((0.42 * priority) + (0.38 * corr_score) + (0.20 * support), 6)


def _report(
    *,
    mode: str,
    status: str,
    registry: dict[str, object],
    candidates: list[CampaignCandidate],
    active_profile: str,
    reason: str,
) -> dict[str, object]:
    return {
        "version": 1,
        "updated_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "status": status,
        "reason": reason,
        "active_profile": active_profile,
        "profile_count": len(registry.get("profiles") or []) if isinstance(registry.get("profiles"), list) else 0,
        "candidate_count": len(candidates),
        "public_pair_count": sum(
            1 for item in candidates if item.offline_score is not None and item.public_score is not None
        ),
        "registry": registry,
    }


def _write_report(context_dir: Path, payload: dict[str, object]) -> None:
    write_json_object(validation_lab_report_path(context_dir), payload)


def _write_registry_if_changed(context_dir: Path, registry: dict[str, object]) -> None:
    write_json_object(context_dir / "validation_registry.json", registry)


def _active_profile(registry: dict[str, object]) -> str:
    return str(registry.get("active_profile") or "default_cv")


def _low_trust_campaign(campaign_state: dict[str, object] | None) -> bool:
    corr = _to_float((campaign_state or {}).get("offline_online_correlation"))
    if corr is not None and corr < 0.25:
        return True
    latest = _to_float((campaign_state or {}).get("latest_submission_score"))
    champion = _to_float(
        (campaign_state or {}).get("champion_score") or (campaign_state or {}).get("historical_best_score")
    )
    if latest is None or champion is None:
        return False
    direction = str((campaign_state or {}).get("direction") or "minimize").lower()
    return latest < champion if direction == "maximize" else latest > champion


def _offline_online_correlation(candidates: list[CampaignCandidate]) -> float | None:
    pairs = [
        (candidate.offline_score, candidate.public_score)
        for candidate in candidates
        if candidate.offline_score is not None and candidate.public_score is not None
    ]
    if len(pairs) < 2:
        return None
    offline = [float(left) for left, _right in pairs]
    online = [float(right) for _left, right in pairs]
    mean_offline = sum(offline) / len(offline)
    mean_online = sum(online) / len(online)
    numerator = sum((left - mean_offline) * (right - mean_online) for left, right in zip(offline, online, strict=True))
    denominator = math.sqrt(sum((left - mean_offline) ** 2 for left in offline)) * math.sqrt(
        sum((right - mean_online) ** 2 for right in online)
    )
    if denominator == 0.0:
        return None
    return round(numerator / denominator, 6)
