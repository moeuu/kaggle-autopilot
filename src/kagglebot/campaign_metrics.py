from __future__ import annotations

from pathlib import Path

from kagglebot.autopilot_helpers import _to_float


def infer_campaign_candidate_category(
    *,
    iteration: int,
    kernel_metrics_payload: dict[str, object] | None,
    quality_reasons: list[str],
) -> str:
    payload = kernel_metrics_payload or {}
    for key in ("candidate_category", "campaign_category", "strategy_category"):
        value = str(payload.get(key) or "").strip().lower()
        if value:
            return value
    if "validation" in " ".join(quality_reasons).lower():
        return "validation_variant"
    if bool(payload.get("blend")) or bool(payload.get("ensemble")):
        return "blend"
    if iteration == 1:
        return "reference_reproduction"
    return "strong_single"


def infer_campaign_model_family(
    model_summary: dict[str, object],
    kernel_metrics_payload: dict[str, object] | None,
) -> str | None:
    for source in (model_summary, kernel_metrics_payload or {}):
        for key in ("model_family", "family", "model_type", "model_name"):
            value = source.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return None


def infer_campaign_feature_set(
    model_summary: dict[str, object],
    kernel_metrics_payload: dict[str, object] | None,
) -> str | None:
    for source in (model_summary, kernel_metrics_payload or {}):
        for key in ("feature_set", "features", "feature_version"):
            value = source.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return None


def extract_campaign_fold_scores(kernel_metrics_payload: dict[str, object] | None) -> list[float]:
    payload = kernel_metrics_payload or {}
    for key in ("fold_scores", "cv_scores", "scores_by_fold"):
        raw = payload.get(key)
        if not isinstance(raw, list):
            continue
        scores = [_to_float(item) for item in raw]
        return [float(score) for score in scores if score is not None]
    return []


def extract_campaign_prediction_correlation(kernel_metrics_payload: dict[str, object] | None) -> dict[str, float]:
    payload = kernel_metrics_payload or {}
    raw = payload.get("prediction_correlation")
    if not isinstance(raw, dict):
        raw = payload.get("prediction_correlations")
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, float] = {}
    for key, value in raw.items():
        numeric = _to_float(value)
        if numeric is not None:
            parsed[str(key)] = float(numeric)
    return parsed


def extract_campaign_artifact_path(kernel_metrics_payload: dict[str, object] | None, kind: str) -> Path | None:
    payload = kernel_metrics_payload or {}
    if kind == "oof":
        keys = ("oof_path", "oof_predictions_path", "oof_predictions")
    else:
        keys = ("prediction_path", "test_prediction_path", "test_predictions_path", "test_predictions")
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return Path(text)
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        nested = artifacts.get(kind) or artifacts.get(f"{kind}_path")
        if nested is not None and str(nested).strip():
            return Path(str(nested).strip())
    return None


def extract_campaign_method_id(kernel_metrics_payload: dict[str, object] | None) -> str | None:
    payload = kernel_metrics_payload or {}
    for key in ("method_id", "active_method_id", "campaign_method_id"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def extract_campaign_validation_profile_id(kernel_metrics_payload: dict[str, object] | None) -> str | None:
    payload = kernel_metrics_payload or {}
    for key in ("validation_profile_id", "active_validation_profile", "split_profile_id"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    readiness = payload.get("readiness")
    if isinstance(readiness, dict):
        value = readiness.get("validation_profile_id") or readiness.get("split_strategy")
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def campaign_prefers_validation_redesign(
    campaign_state: dict[str, object],
    method_registry: dict[str, object] | None,
) -> bool:
    if isinstance(method_registry, dict) and bool(method_registry.get("validation_priority")):
        return True
    corr = _to_float(campaign_state.get("offline_online_correlation"))
    if corr is not None and corr < 0.25:
        return True
    latest = _to_float(campaign_state.get("latest_submission_score"))
    champion = _to_float(campaign_state.get("champion_score") or campaign_state.get("historical_best_score"))
    if latest is None or champion is None:
        return False
    direction = str(campaign_state.get("direction") or "minimize").strip().lower()
    if direction == "maximize":
        return latest < champion
    return latest > champion
