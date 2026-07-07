from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from kagglebot.hashing import sha256_file_or_none
from kagglebot.json_utils import load_json_object_or_empty, write_json_object
from kagglebot.scalar_utils import finite_float as _to_float
from kagglebot.scalar_utils import optional_int as _to_int
from kagglebot.scalar_utils import optional_str as _optional_str
from kagglebot.score_utils import best_score as _best_score
from kagglebot.score_utils import is_better_score as _is_better_score
from kagglebot.score_utils import score_gap as _score_gap
from kagglebot.solver.metrics import normalize_direction

CampaignMode = Literal["baseline", "top1"]

CAMPAIGN_STATE_FILENAME = "campaign_state.json"
CANDIDATE_REGISTRY_FILENAME = "candidate_registry.json"
TOP1_TARGET_RANK_PERCENTILE = 0.001
CandidateCategory = Literal[
    "reference_reproduction",
    "strong_single",
    "feature_variant",
    "validation_variant",
    "blend",
    "calibration",
]

_CATEGORIES: set[str] = {
    "reference_reproduction",
    "strong_single",
    "feature_variant",
    "validation_variant",
    "blend",
    "calibration",
}


@dataclass(frozen=True)
class CampaignCandidate:
    candidate_id: str
    category: CandidateCategory
    run_id: str
    iteration: int
    direction: str
    offline_score: float | None = None
    offline_std: float | None = None
    score_source: str | None = None
    public_score: float | None = None
    submission_path: str | None = None
    submission_sha256: str | None = None
    metrics_path: str | None = None
    oof_path: str | None = None
    prediction_path: str | None = None
    prediction_correlation: dict[str, float] = field(default_factory=dict)
    fold_scores: list[float] = field(default_factory=list)
    runtime_sec: float | None = None
    model_family: str | None = None
    feature_set: str | None = None
    method_id: str | None = None
    validation_profile_id: str | None = None
    private_robustness_score: float | None = None
    rank_score: float | None = None
    submitted: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, object] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> CampaignCandidate:
        category = str(payload.get("category") or "strong_single")
        if category not in _CATEGORIES:
            category = "strong_single"
        return cls(
            candidate_id=str(payload.get("candidate_id") or ""),
            category=category,  # type: ignore[arg-type]
            run_id=str(payload.get("run_id") or ""),
            iteration=int(payload.get("iteration") or 0),
            direction=_normalize_direction(payload.get("direction")),
            offline_score=_to_float(payload.get("offline_score")),
            offline_std=_to_float(payload.get("offline_std")),
            score_source=_optional_str(payload.get("score_source")),
            public_score=_to_float(payload.get("public_score")),
            submission_path=_optional_str(payload.get("submission_path")),
            submission_sha256=_optional_str(payload.get("submission_sha256")),
            metrics_path=_optional_str(payload.get("metrics_path")),
            oof_path=_optional_str(payload.get("oof_path")),
            prediction_path=_optional_str(payload.get("prediction_path")),
            prediction_correlation=_float_mapping(payload.get("prediction_correlation")),
            fold_scores=_float_list(payload.get("fold_scores")),
            runtime_sec=_to_float(payload.get("runtime_sec")),
            model_family=_optional_str(payload.get("model_family")),
            feature_set=_optional_str(payload.get("feature_set")),
            method_id=_optional_str(payload.get("method_id")),
            validation_profile_id=_optional_str(payload.get("validation_profile_id")),
            private_robustness_score=_to_float(payload.get("private_robustness_score")),
            rank_score=_to_float(payload.get("rank_score")),
            submitted=bool(payload.get("submitted", False)),
            created_at=str(payload.get("created_at") or datetime.now(UTC).isoformat()),
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        )


@dataclass(frozen=True)
class SubmissionAllocation:
    allow_submit: bool
    reason: str
    allocation_score: float
    expected_online_gain: float | None = None
    novelty: float | None = None
    validation_trust: float | None = None
    regression_risk: float | None = None
    remaining_daily_slots: int | None = None
    information_value: float | None = None

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


def normalize_campaign_mode(value: str | None, *, deliverable_mode: str | None = "leaderboard") -> CampaignMode:
    normalized = str(value or "").strip().lower()
    if normalized in {"baseline", "off", "none"}:
        return "baseline"
    if normalized in {"top1", "top-1", "leaderboard"}:
        return "top1"
    if normalized in {"", "auto"}:
        return "top1" if str(deliverable_mode or "leaderboard").strip().lower() == "leaderboard" else "baseline"
    raise ValueError("campaign_mode must be 'top1' or 'baseline'")


def campaign_state_path(context_dir: Path) -> Path:
    return context_dir / CAMPAIGN_STATE_FILENAME


def candidate_registry_path(context_dir: Path) -> Path:
    return context_dir / CANDIDATE_REGISTRY_FILENAME


def load_candidate_registry(path: Path) -> dict[str, object]:
    payload = load_json_object_or_empty(path)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        candidates = []
    return {
        "version": 1,
        "updated_at": str(payload.get("updated_at") or datetime.now(UTC).isoformat()),
        "candidates": [item for item in candidates if isinstance(item, dict)],
    }


def list_candidates(path: Path) -> list[CampaignCandidate]:
    registry = load_candidate_registry(path)
    return [
        CampaignCandidate.from_payload(item)
        for item in registry["candidates"]
        if isinstance(item, dict) and str(item.get("candidate_id") or "").strip()
    ]


def upsert_candidate(path: Path, candidate: CampaignCandidate) -> dict[str, object]:
    registry = load_candidate_registry(path)
    by_id: dict[str, dict[str, object]] = {}
    for item in registry["candidates"]:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id") or "").strip()
        if candidate_id:
            by_id[candidate_id] = dict(item)
    by_id[candidate.candidate_id] = candidate.to_payload()
    payload = {
        "version": 1,
        "updated_at": datetime.now(UTC).isoformat(),
        "candidates": sorted(by_id.values(), key=lambda item: str(item.get("candidate_id") or "")),
    }
    write_json_object(path, payload)
    return payload


def build_campaign_candidate(
    *,
    run_id: str,
    iteration: int,
    direction: str,
    category: str,
    offline_score: float | None,
    offline_std: float | None,
    score_source: str | None,
    submission_path: Path | None,
    metrics_path: Path | None,
    oof_path: Path | None = None,
    prediction_path: Path | None = None,
    model_family: str | None = None,
    feature_set: str | None = None,
    method_id: str | None = None,
    validation_profile_id: str | None = None,
    private_robustness_score: float | None = None,
    rank_score: float | None = None,
    fold_scores: list[float] | None = None,
    prediction_correlation: dict[str, float] | None = None,
    metadata: dict[str, object] | None = None,
) -> CampaignCandidate:
    normalized_category = category if category in _CATEGORIES else "strong_single"
    submission_sha = sha256_file_or_none(submission_path)
    return CampaignCandidate(
        candidate_id=f"{run_id}-i{iteration:03d}-{normalized_category}",
        category=normalized_category,  # type: ignore[arg-type]
        run_id=run_id,
        iteration=iteration,
        direction=_normalize_direction(direction),
        offline_score=offline_score,
        offline_std=offline_std,
        score_source=score_source,
        submission_path=str(submission_path) if submission_path is not None else None,
        submission_sha256=submission_sha,
        metrics_path=str(metrics_path) if metrics_path is not None else None,
        oof_path=str(oof_path) if oof_path is not None else None,
        prediction_path=str(prediction_path) if prediction_path is not None else None,
        model_family=model_family,
        feature_set=feature_set,
        method_id=method_id,
        validation_profile_id=validation_profile_id,
        private_robustness_score=private_robustness_score,
        rank_score=rank_score,
        fold_scores=fold_scores or [],
        prediction_correlation=prediction_correlation or {},
        metadata=metadata or {},
    )


def update_campaign_state(
    *,
    state_path: Path,
    registry_path: Path,
    slug: str,
    run_id: str,
    mode: CampaignMode,
    direction: str,
    top1_info: dict[str, object] | None,
    submission_history: dict[str, object] | None,
    latest_public_score: float | None = None,
    remaining_daily_slots: int | None = None,
    method_registry: dict[str, object] | None = None,
) -> dict[str, object]:
    previous = load_json_object_or_empty(state_path)
    candidates = list_candidates(registry_path)
    historical_best = _to_float((submission_history or {}).get("best_score"))
    top1_score = _to_float((top1_info or {}).get("score"))
    previous_champion = _to_float(previous.get("champion_score"))
    champion = best_score(
        direction=direction,
        scores=[
            previous_champion,
            historical_best,
            *[candidate.public_score for candidate in candidates],
            *[candidate.offline_score for candidate in candidates if candidate.submitted],
        ],
    )
    latest_score = _to_float(latest_public_score)
    if latest_score is None:
        latest_score = _to_float((submission_history or {}).get("latest_score"))
    payload = {
        "version": 1,
        "campaign_id": str(previous.get("campaign_id") or f"{slug}-{run_id}"),
        "mode": mode,
        "slug": slug,
        "run_id": run_id,
        "direction": _normalize_direction(direction),
        "target_rank_percentile": TOP1_TARGET_RANK_PERCENTILE if mode == "top1" else None,
        "champion_score": champion,
        "historical_best_score": historical_best,
        "latest_submission_score": latest_score,
        "top1_score": top1_score,
        "top1_gap": score_gap(current=champion, reference=top1_score, direction=direction),
        "offline_online_correlation": offline_online_correlation(candidates),
        "remaining_daily_slots": remaining_daily_slots,
        "active_method_ids": _string_list((method_registry or {}).get("active_method_ids")),
        "blocked_method_ids": _string_list((method_registry or {}).get("blocked_method_ids")),
        "active_validation_profile": (method_registry or {}).get("active_validation_profile"),
        "method_scout_updated_at": (method_registry or {}).get("updated_at"),
        "candidate_count": len(candidates),
        "candidates": [candidate.candidate_id for candidate in candidates],
        "submission_history": submission_history or {},
        "updated_at": datetime.now(UTC).isoformat(),
    }
    write_json_object(state_path, payload)
    return payload


def classify_against_campaign_baseline(
    *,
    candidate_score: float | None,
    direction: str,
    historical_best_score: float | None = None,
    champion_score: float | None = None,
) -> str:
    candidate = _to_float(candidate_score)
    baseline = best_score(direction=direction, scores=[historical_best_score, champion_score])
    if candidate is None or baseline is None:
        return "unknown"
    if is_better(candidate, baseline, direction=direction, min_delta=0.0):
        return "improvement"
    if math.isclose(candidate, baseline, rel_tol=0.0, abs_tol=1e-12):
        return "equal"
    return "regression"


def allocate_submission(
    *,
    candidate: CampaignCandidate,
    campaign_state: dict[str, object],
    is_duplicate: bool = False,
    remaining_daily_slots: int | None = None,
    expected_online_gain: float | None = None,
    novelty: float = 0.5,
    validation_trust: float | None = None,
    regression_risk: float | None = None,
    information_value: float | None = None,
    calibration_exception: bool = False,
    force: bool = False,
) -> SubmissionAllocation:
    slots = remaining_daily_slots
    if slots is None:
        slots = _to_int(campaign_state.get("remaining_daily_slots"))
    if slots is not None and slots <= 0 and not force:
        return SubmissionAllocation(False, "daily_submission_limit_reached", 0.0, remaining_daily_slots=slots)
    if is_duplicate and not force:
        return SubmissionAllocation(False, "duplicate_submission", 0.0, remaining_daily_slots=slots)

    direction = _normalize_direction(campaign_state.get("direction") or candidate.direction)
    historical_best = _to_float(campaign_state.get("historical_best_score"))
    champion = _to_float(campaign_state.get("champion_score"))
    baseline_status = classify_against_campaign_baseline(
        candidate_score=candidate.offline_score,
        direction=direction,
        historical_best_score=historical_best,
        champion_score=champion,
    )
    inferred_risk = regression_risk
    if inferred_risk is None:
        inferred_risk = 0.8 if baseline_status == "regression" else 0.2
    if baseline_status == "regression" and not calibration_exception and not force:
        return SubmissionAllocation(
            False,
            "below_campaign_baseline",
            0.0,
            expected_online_gain=expected_online_gain,
            novelty=novelty,
            validation_trust=validation_trust,
            regression_risk=inferred_risk,
            remaining_daily_slots=slots,
            information_value=information_value,
        )

    gain = (
        expected_online_gain
        if expected_online_gain is not None
        else _candidate_gain_vs_baseline(
            candidate=candidate,
            direction=direction,
            baseline=best_score(direction=direction, scores=[historical_best, champion]),
        )
    )
    trust = validation_trust
    if trust is None:
        correlation = _to_float(campaign_state.get("offline_online_correlation"))
        trust = 0.5 if correlation is None else max(0.0, min(1.0, (correlation + 1.0) / 2.0))
    info = information_value
    if info is None:
        info = _infer_information_value(candidate=candidate, novelty=novelty, validation_trust=trust)
    score = (
        float(gain or 0.0)
        + (0.2 * max(0.0, novelty))
        + (0.4 * max(0.0, trust))
        + (0.25 * max(0.0, info))
        - float(inferred_risk)
    )
    if calibration_exception:
        score += 0.25
    allow = force or calibration_exception or score > 0.0
    reason = "selected" if allow else "low_expected_value"
    if calibration_exception and allow:
        reason = "calibration_exception"
    return SubmissionAllocation(
        allow,
        reason,
        score,
        expected_online_gain=gain,
        novelty=novelty,
        validation_trust=trust,
        regression_risk=inferred_risk,
        remaining_daily_slots=slots,
        information_value=info,
    )


def format_campaign_submission_message(
    *,
    base_message: str,
    campaign_state: dict[str, object],
    candidate: CampaignCandidate,
    offline_score: float | None,
    direction: str,
) -> str:
    campaign_id = str(campaign_state.get("campaign_id") or "").strip()
    baseline = best_score(
        direction=direction,
        scores=[campaign_state.get("historical_best_score"), campaign_state.get("champion_score")],
    )
    delta = score_gap(current=offline_score, reference=baseline, direction=direction)
    parts = [
        base_message.strip(),
        f"campaign={campaign_id}" if campaign_id else "",
        f"candidate={candidate.candidate_id}",
    ]
    if offline_score is not None:
        parts.append(f"offline={offline_score:.6f}")
    if delta is not None:
        parts.append(f"baseline_delta={delta:+.6f}")
    return " ".join(part for part in parts if part)


def recommend_blend_pairs(
    candidates: list[CampaignCandidate],
    *,
    direction: str,
    max_pairs: int = 3,
    max_correlation: float = 0.95,
) -> list[tuple[str, str]]:
    scored = [candidate for candidate in candidates if candidate.offline_score is not None]
    scored = sorted(
        scored, key=lambda item: float(item.offline_score), reverse=_normalize_direction(direction) == "maximize"
    )
    pairs: list[tuple[str, str]] = []
    for left_index, left in enumerate(scored):
        for right in scored[left_index + 1 :]:
            corr = left.prediction_correlation.get(right.candidate_id)
            if corr is None:
                corr = right.prediction_correlation.get(left.candidate_id)
            if corr is not None and corr > max_correlation:
                continue
            pairs.append((left.candidate_id, right.candidate_id))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def offline_online_correlation(candidates: list[CampaignCandidate]) -> float | None:
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
    denom_left = math.sqrt(sum((left - mean_offline) ** 2 for left in offline))
    denom_right = math.sqrt(sum((right - mean_online) ** 2 for right in online))
    denominator = denom_left * denom_right
    if denominator == 0.0:
        return None
    return numerator / denominator


def is_better(candidate: float, baseline: float, *, direction: str, min_delta: float = 0.0) -> bool:
    return _is_better_score(candidate, baseline, direction=direction, min_delta=min_delta)


def best_score(*, direction: str, scores: list[object]) -> float | None:
    return _best_score(direction=direction, scores=scores)


def score_gap(*, current: object, reference: object, direction: str) -> float | None:
    return _score_gap(current=current, reference=reference, direction=direction)


def _candidate_gain_vs_baseline(
    *,
    candidate: CampaignCandidate,
    direction: str,
    baseline: float | None,
) -> float | None:
    if candidate.offline_score is None or baseline is None:
        return None
    return score_gap(current=candidate.offline_score, reference=baseline, direction=direction)


def _infer_information_value(*, candidate: CampaignCandidate, novelty: float, validation_trust: float) -> float:
    value = 0.25 + (0.35 * max(0.0, min(1.0, novelty)))
    if candidate.category in {"validation_variant", "calibration"}:
        value += 0.3
    if candidate.category == "blend" and candidate.prediction_correlation:
        value += 0.1
    if validation_trust < 0.4 and candidate.category not in {"validation_variant", "calibration"}:
        value -= 0.2
    return max(0.0, min(1.0, value))


def _normalize_direction(value: object) -> str:
    return normalize_direction(value, default="minimize") or "minimize"


def _float_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    parsed: dict[str, float] = {}
    for key, item in value.items():
        numeric = _to_float(item)
        if numeric is not None:
            parsed[str(key)] = numeric
    return parsed


def _float_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    parsed = [_to_float(item) for item in value]
    return [item for item in parsed if item is not None]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]
