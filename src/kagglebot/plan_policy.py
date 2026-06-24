from __future__ import annotations

import re

from kagglebot.json_utils import load_json_object
from kagglebot.paths import CompetitionPaths

SPLIT_STRATEGY_PRIORITY = {
    "kfold": 0,
    "stratified_kfold": 1,
    "group_kfold": 2,
    "timeseries_split": 3,
}

COMPETITION_EVAL_OVERRIDES: dict[str, dict[str, str]] = {
    "deep-past-initiative-machine-translation": {
        "metric_name": "Geometric Mean of the BLEU and the chrF++ scores",
        "direction": "maximize",
        "split_strategy": "group_kfold",
        "group_column_hint": "oare_id",
    }
}


def normalize_split_strategy_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    aliases = {
        "k": "kfold",
        "kfold": "kfold",
        "stratified": "stratified_kfold",
        "stratifiedkfold": "stratified_kfold",
        "stratified_kfold": "stratified_kfold",
        "group": "group_kfold",
        "groupkfold": "group_kfold",
        "group_kfold": "group_kfold",
        "group_kfold_oare_id": "group_kfold",
        "time": "timeseries_split",
        "timeseries": "timeseries_split",
        "timeseriessplit": "timeseries_split",
        "timeseries_split": "timeseries_split",
    }
    return aliases.get(normalized)


def competition_eval_override(slug: str) -> dict[str, str]:
    return dict(COMPETITION_EVAL_OVERRIDES.get(str(slug).strip().lower(), {}))


def apply_competition_eval_override(
    *,
    slug: str,
    payload: dict[str, object],
    include_spec_keys: bool = False,
) -> dict[str, object]:
    override = competition_eval_override(slug)
    if not override:
        return payload
    updated = dict(payload)
    updated["metric"] = override["metric_name"]
    updated["direction"] = override["direction"]
    updated["split_strategy_hint"] = override["split_strategy"]
    updated["group_column_hint"] = override["group_column_hint"]
    task = str(updated.get("task") or "").strip().lower()
    if task in {"classification", "binary", "multiclass", ""}:
        updated["task"] = "translation"
    updated["task_by_target"] = {"translation": "translation"}
    updated["prediction_kind_by_target"] = {"translation": "text"}
    updated["tags"] = ["text", "translation", "n_rows_small", "high_cardinality_cats"]
    if include_spec_keys:
        updated["metric_name"] = override["metric_name"]
        updated["split_strategy"] = override["split_strategy"]
    return updated


def infer_split_strategy_from_hint_text(text: str) -> str | None:
    lowered = text.strip().lower()
    if not lowered:
        return None
    direct = normalize_split_strategy_name(lowered)
    if direct is not None:
        return direct
    if re.search(r"\btime[-_\s]?series\b|\bchronolog\w*\b|\bforecast\w*\b", lowered):
        return "timeseries_split"
    if re.search(
        r"\bgroupkfold\b|\bgroup[_\s-]?kfold\b|\bgroup(?:ed)?[_\s-]?fold\b|\bgroup(?:ed)?[_\s-]?cv\b",
        lowered,
    ):
        return "group_kfold"
    if re.search(r"\bstratifiedkfold\b|\bstratified[_\s-]?kfold\b|\bstratified[_\s-]?cv\b", lowered):
        return "stratified_kfold"
    if re.search(r"\bk[-_\s]?fold\b", lowered):
        return "kfold"
    return None


def extract_plan_split_strategy_hints(plan_payload: dict[str, object]) -> list[str]:
    hints: list[str] = []

    evaluation_protocol = plan_payload.get("evaluation_protocol")
    if isinstance(evaluation_protocol, dict):
        for key in ("cv_type", "split_strategy"):
            raw = evaluation_protocol.get(key)
            if isinstance(raw, str) and raw.strip():
                hints.append(raw)

    toggles = plan_payload.get("toggles")
    if isinstance(toggles, dict):
        for key in ("CV_TYPE", "cv_type", "split_strategy", "SPLIT_STRATEGY"):
            raw = toggles.get(key)
            if isinstance(raw, str) and raw.strip():
                hints.append(raw)

    for key in ("cv_type", "split_strategy"):
        raw = plan_payload.get(key)
        if isinstance(raw, str) and raw.strip():
            hints.append(raw)

    return hints


def profile_has_temporal_signal(profile: dict[str, object]) -> bool:
    dtype_map_raw = profile.get("dtype_by_column")
    if not isinstance(dtype_map_raw, dict):
        return False
    temporal_name = re.compile(r"\b(date|datetime|timestamp|time)\b", flags=re.IGNORECASE)
    for name, dtype in dtype_map_raw.items():
        column_name = str(name)
        dtype_name = str(dtype).lower()
        if "datetime" in dtype_name or "timedelta" in dtype_name:
            return True
        if temporal_name.search(column_name):
            return True
    return False


def resolve_split_strategy_from_artifacts(
    *,
    paths: CompetitionPaths,
    split_strategy: object,
) -> tuple[str | None, str | None]:
    raw_current = str(split_strategy).strip() if isinstance(split_strategy, str) else ""
    normalized_current = normalize_split_strategy_name(raw_current)
    if raw_current and normalized_current is None:
        return raw_current, None
    if normalized_current not in {None, "kfold"}:
        return normalized_current, None

    plan_payload = load_json_object(paths.plan_path) or {}
    hints = extract_plan_split_strategy_hints(plan_payload)
    hinted_strategy: str | None = None
    for hint in hints:
        candidate = infer_split_strategy_from_hint_text(hint)
        if candidate is None:
            continue
        if hinted_strategy is None or (SPLIT_STRATEGY_PRIORITY[candidate] > SPLIT_STRATEGY_PRIORITY[hinted_strategy]):
            hinted_strategy = candidate

    if (
        hinted_strategy in {"timeseries_split", "group_kfold", "stratified_kfold"}
        and hinted_strategy != normalized_current
    ):
        return (
            hinted_strategy,
            f"split_strategy '{normalized_current or 'auto'}' -> '{hinted_strategy}' "
            "using plan evaluation hints for better local/public alignment.",
        )

    profile = load_json_object(paths.dataset_profile_path) or {}
    if (
        str(profile.get("modality", "")).strip().lower() == "timeseries"
        and profile_has_temporal_signal(profile)
        and normalized_current != "timeseries_split"
    ):
        return (
            "timeseries_split",
            f"split_strategy '{normalized_current or 'auto'}' -> 'timeseries_split' "
            "using dataset_profile temporal signal.",
        )

    task = str(profile.get("task", "")).strip().lower()
    if task in {"classification", "binary", "multiclass"} and normalized_current in {None, "kfold"}:
        return (
            "stratified_kfold",
            f"split_strategy '{normalized_current or 'auto'}' -> 'stratified_kfold' "
            "using dataset_profile classification task.",
        )

    return normalized_current, None
