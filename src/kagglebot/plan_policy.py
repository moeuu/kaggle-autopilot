from __future__ import annotations

import re
from dataclasses import dataclass

from kagglebot.json_utils import load_json_object
from kagglebot.metric_matching import canonical_metric_name_for_match, metrics_equivalent
from kagglebot.paths import CompetitionPaths
from kagglebot.score_sources import (
    DEFAULT_ACCEPTED_SCORE_SOURCES,
    normalize_score_source_list,
    normalize_score_source_name,
)
from kagglebot.solver.metrics import canonical_metric
from kagglebot.writeup import normalize_deliverable_mode, normalize_submit_mode

SPLIT_STRATEGY_PRIORITY = {
    "kfold": 0,
    "stratified_kfold": 1,
    "group_kfold": 2,
    "timeseries_split": 3,
}

DEFAULT_EVAL_SEEDS = (42, 2024, 777)
DEFAULT_EVAL_REPEATS = 2
EVAL_REPEAT_SEED_OFFSET = 1009
FULL_DATASET_REQUIRED_COMPETITIONS = frozenset({"urban-flood-modelling"})

COMPETITION_EVAL_OVERRIDES: dict[str, dict[str, str]] = {
    "deep-past-initiative-machine-translation": {
        "metric_name": "Geometric Mean of the BLEU and the chrF++ scores",
        "direction": "maximize",
        "split_strategy": "group_kfold",
        "group_column_hint": "oare_id",
    }
}


@dataclass(frozen=True)
class EvaluationSpecValues:
    metric_name: str | None = None
    direction: str | None = None
    split_strategy: str | None = None
    n_splits: int | None = None
    seed: int | None = None
    eval_seeds: tuple[int, ...] = ()
    repeats: int | None = None
    ci_method: str | None = None
    ci_alpha: int | float | None = None
    readiness_method: str | None = None
    readiness_k: int | float | None = None
    readiness_target_score: int | float | None = None
    submission_gate: str | None = None
    drift_enabled: bool | None = None
    drift_weight: int | float | None = None
    stop_min_delta: int | float | None = None
    stop_no_improve_patience: int | None = None
    stop_same_config_patience: int | None = None


@dataclass(frozen=True)
class TargetMetricDirectionDecision:
    target_metric: object
    target_direction: object
    override_split_strategy: str
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanScoreSourceDecision:
    score_source: str
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class SplitStrategyOverrideDecision:
    split_strategy: object
    messages: tuple[str, ...] = ()


def extract_evaluation_spec_values(eval_spec: dict[str, object]) -> EvaluationSpecValues:
    """Extract typed values from the persisted evaluation spec contract."""

    seeds: list[int] = []
    raw_seeds = eval_spec.get("seeds")
    if isinstance(raw_seeds, list):
        seeds = [item for item in raw_seeds if isinstance(item, int)]

    readiness_rule = eval_spec.get("readiness_rule") if isinstance(eval_spec.get("readiness_rule"), dict) else {}
    drift_cfg = eval_spec.get("drift_check") if isinstance(eval_spec.get("drift_check"), dict) else {}
    stop_policy = eval_spec.get("stop_policy") if isinstance(eval_spec.get("stop_policy"), dict) else {}

    return EvaluationSpecValues(
        metric_name=_str_or_none(eval_spec.get("metric_name")),
        direction=_str_or_none(eval_spec.get("direction")),
        split_strategy=_str_or_none(eval_spec.get("split_strategy")),
        n_splits=_int_or_none(eval_spec.get("n_splits")),
        seed=seeds[0] if seeds else None,
        eval_seeds=tuple(seeds),
        repeats=_int_or_none(eval_spec.get("repeats")),
        ci_method=_str_or_none(eval_spec.get("ci_method")),
        ci_alpha=_number_or_none(eval_spec.get("ci_alpha")),
        readiness_method=_str_or_none(readiness_rule.get("method")),
        readiness_k=_number_or_none(readiness_rule.get("k")),
        readiness_target_score=_number_or_none(readiness_rule.get("target_score")),
        submission_gate=_str_or_none(readiness_rule.get("submission_gate")),
        drift_enabled=drift_cfg.get("enabled") if isinstance(drift_cfg.get("enabled"), bool) else None,
        drift_weight=_number_or_none(drift_cfg.get("drift_weight")),
        stop_min_delta=_number_or_none(stop_policy.get("min_delta")),
        stop_no_improve_patience=_int_or_none(stop_policy.get("no_improve_patience")),
        stop_same_config_patience=_int_or_none(stop_policy.get("same_config_patience")),
    )


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _number_or_none(value: object) -> int | float | None:
    return value if isinstance(value, (int, float)) else None


def resolve_target_metric_direction(
    *,
    target_metric: object,
    target_direction: object,
    spec_metric: str | None,
    spec_direction: str | None,
    explicit_target_metric: bool,
    explicit_target_direction: bool,
    strict_competition_metric: bool,
    competition_override: dict[str, str],
) -> TargetMetricDirectionDecision:
    """Apply competition and evaluation-spec metric/direction policy."""

    resolved_metric = target_metric
    resolved_direction = target_direction
    messages: list[str] = []

    override_metric = str(competition_override.get("metric_name") or "").strip()
    override_direction = str(competition_override.get("direction") or "").strip().lower()
    override_split_strategy = str(competition_override.get("split_strategy") or "").strip()

    if override_metric:
        requested_metric = str(resolved_metric or "").strip()
        if requested_metric and not metrics_equivalent(requested_metric, override_metric):
            messages.append(
                "[yellow]note[/yellow]: competition override is active; "
                f"forcing target_metric '{requested_metric}' -> '{override_metric}'."
            )
        resolved_metric = override_metric

    if override_direction in {"minimize", "maximize"}:
        requested_direction = str(resolved_direction or "").strip().lower()
        if requested_direction and requested_direction != override_direction:
            messages.append(
                "[yellow]note[/yellow]: competition override is active; "
                f"forcing target_direction '{requested_direction}' -> '{override_direction}'."
            )
        resolved_direction = override_direction

    # Competition-specific overrides are authoritative and must not be undone by a stale
    # evaluation_spec.json contract from an earlier iteration.
    if strict_competition_metric and spec_metric and not competition_override:
        requested_metric = resolved_metric if isinstance(resolved_metric, str) else None
        requested_metric_norm = canonical_metric_name_for_match(requested_metric)
        spec_metric_norm = canonical_metric_name_for_match(spec_metric)
        if requested_metric_norm != spec_metric_norm:
            if explicit_target_metric and requested_metric:
                messages.append(
                    "[yellow]note[/yellow]: strict competition metric mode is enabled, "
                    "but keeping explicit target_metric "
                    f"'{requested_metric}' over evaluation_spec metric '{spec_metric}'."
                )
            elif requested_metric:
                messages.append(
                    "[yellow]note[/yellow]: strict competition metric mode is enabled; "
                    f"overriding target_metric '{requested_metric}' -> '{spec_metric}'."
                )
                resolved_metric = spec_metric
            else:
                resolved_metric = spec_metric

        if spec_direction in {"minimize", "maximize"}:
            requested_direction = str(resolved_direction or "").strip().lower()
            if requested_direction != spec_direction:
                if explicit_target_direction and requested_direction:
                    messages.append(
                        "[yellow]note[/yellow]: strict competition metric mode is enabled, "
                        "but keeping explicit target_direction "
                        f"'{requested_direction}' over evaluation_spec direction '{spec_direction}'."
                    )
                else:
                    messages.append(
                        "[yellow]note[/yellow]: strict competition metric mode is enabled; "
                        f"overriding target_direction '{requested_direction or 'auto'}' -> '{spec_direction}'."
                    )
                    resolved_direction = spec_direction

    return TargetMetricDirectionDecision(
        target_metric=resolved_metric,
        target_direction=resolved_direction,
        override_split_strategy=override_split_strategy,
        messages=tuple(messages),
    )


def resolve_plan_score_source(score_source: object) -> PlanScoreSourceDecision:
    """Restrict plan score source to generalizable local evaluation sources."""

    normalized = normalize_score_source_name(score_source)
    if normalized in {"cv", "holdout"}:
        return PlanScoreSourceDecision(score_source=normalized)
    return PlanScoreSourceDecision(
        score_source="cv",
        messages=("[yellow]note[/yellow]: non-generalizable score_source is not allowed; overriding to cv.",),
    )


def resolve_split_strategy_override(
    *,
    split_strategy: object,
    override_split_strategy: str | None,
) -> SplitStrategyOverrideDecision:
    """Apply competition split-strategy override after artifact/profile hints."""

    if not override_split_strategy:
        return SplitStrategyOverrideDecision(split_strategy=split_strategy)

    normalized_override = normalize_split_strategy_name(override_split_strategy)
    if normalized_override is None or split_strategy == normalized_override:
        return SplitStrategyOverrideDecision(split_strategy=split_strategy)

    return SplitStrategyOverrideDecision(
        split_strategy=normalized_override,
        messages=(
            "[yellow]note[/yellow]: competition override is active; "
            f"forcing split_strategy '{split_strategy or 'auto'}' -> '{normalized_override}'.",
        ),
    )


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


def build_evaluation_contract(
    *,
    slug: str,
    eval_spec: dict[str, object],
    target_metric: str | None,
    target_direction: str | None,
    split_strategy: str | None,
) -> dict[str, object]:
    faithfulness = eval_spec.get("faithfulness") if isinstance(eval_spec.get("faithfulness"), dict) else {}
    accepted_score_sources = normalize_score_source_list(
        faithfulness.get("accepted_score_sources") if isinstance(faithfulness, dict) else None
    )
    require_full_dataset_default = str(slug).strip().lower() in FULL_DATASET_REQUIRED_COMPETITIONS
    competition_override = competition_eval_override(slug)
    metric_source = (
        competition_override.get("metric_name") if competition_override.get("metric_name") else target_metric
    )
    direction_source = (
        competition_override.get("direction") if competition_override.get("direction") else target_direction
    )
    split_source = (
        competition_override.get("split_strategy") if competition_override.get("split_strategy") else split_strategy
    )
    expected_metric = canonical_metric(metric_source) if isinstance(metric_source, str) and metric_source else None
    return {
        "expected_metric": expected_metric,
        "expected_direction": (
            str(direction_source).strip().lower()
            if isinstance(direction_source, str) and str(direction_source).strip().lower() in {"minimize", "maximize"}
            else None
        ),
        "expected_split_strategy": normalize_split_strategy_name(split_source),
        "accepted_score_sources": accepted_score_sources or list(DEFAULT_ACCEPTED_SCORE_SOURCES),
        "require_metric_match": (
            bool(faithfulness.get("require_metric_match"))
            if isinstance(faithfulness, dict) and isinstance(faithfulness.get("require_metric_match"), bool)
            else True
        ),
        "require_split_match": (
            bool(faithfulness.get("require_split_match"))
            if isinstance(faithfulness, dict) and isinstance(faithfulness.get("require_split_match"), bool)
            else True
        ),
        "require_trusted_score_source": (
            bool(faithfulness.get("require_trusted_score_source"))
            if isinstance(faithfulness, dict) and isinstance(faithfulness.get("require_trusted_score_source"), bool)
            else True
        ),
        "require_competition_faithful": (
            bool(faithfulness.get("require_competition_faithful"))
            if isinstance(faithfulness, dict) and isinstance(faithfulness.get("require_competition_faithful"), bool)
            else True
        ),
        "require_full_dataset": (
            bool(faithfulness.get("require_full_dataset"))
            if isinstance(faithfulness, dict) and isinstance(faithfulness.get("require_full_dataset"), bool)
            else require_full_dataset_default
        ),
    }


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


def normalize_eval_seeds(
    value: object,
    *,
    fallback: list[int] | None = None,
    default_seeds: list[int] | tuple[int, ...],
) -> list[int]:
    candidates: list[int] = []
    source = value
    if source is None:
        source = fallback
    if isinstance(source, list):
        for item in source:
            if isinstance(item, int):
                candidates.append(int(item))
    seen: set[int] = set()
    normalized: list[int] = []
    for seed in candidates:
        if seed in seen:
            continue
        seen.add(seed)
        normalized.append(seed)
    if normalized:
        return normalized
    return list(default_seeds)


def normalize_default_eval_seeds(value: object, *, fallback: list[int] | None = None) -> list[int]:
    return normalize_eval_seeds(value, fallback=fallback, default_seeds=DEFAULT_EVAL_SEEDS)


def resolve_deliverable_mode(
    *,
    plan_value: object,
    spec_value: object,
    inferred_value: object,
    default: str = "leaderboard",
) -> str:
    plan_mode = normalize_deliverable_mode(plan_value, default="")
    spec_mode = normalize_deliverable_mode(spec_value, default="")
    inferred_mode = normalize_deliverable_mode(inferred_value, default="")
    return spec_mode or inferred_mode or plan_mode or default


def resolve_submit_mode(
    *,
    plan_value: object,
    spec_value: object,
    inferred_value: object,
    default: str = "file",
) -> str:
    plan_mode = normalize_submit_mode(plan_value, default="")
    spec_mode = normalize_submit_mode(spec_value, default="")
    inferred_mode = normalize_submit_mode(inferred_value, default="")
    return spec_mode or inferred_mode or plan_mode or default


def normalize_eval_repeats(value: object, *, fallback: int | None = None, default_repeats: int) -> int:
    resolved = value if isinstance(value, int) else fallback
    if isinstance(resolved, int):
        return max(1, min(resolved, 10))
    return int(default_repeats)


def normalize_default_eval_repeats(value: object, *, fallback: int | None = None) -> int:
    return normalize_eval_repeats(value, fallback=fallback, default_repeats=DEFAULT_EVAL_REPEATS)


def normalize_rank_force_percentile(value: object, *, fallback: float) -> float:
    if isinstance(value, (int, float)):
        parsed = float(value)
        if 0.0 < parsed <= 1.0:
            return parsed
    return float(fallback)


def normalize_rank_force_min_teams(value: object, *, fallback: int) -> int:
    if isinstance(value, int):
        return max(1, value)
    if isinstance(value, float):
        return max(1, int(value))
    return max(1, int(fallback))


def improvement_mode_rank(mode: str) -> int:
    return {"minor_tuning": 0, "moderate_update": 1, "major_overhaul": 2, "validation_redesign": 3}.get(mode, 0)


def upgrade_improvement_mode(current_mode: str, minimum_mode: str | None) -> str:
    if not minimum_mode:
        return current_mode
    if improvement_mode_rank(minimum_mode) > improvement_mode_rank(current_mode):
        return minimum_mode
    return current_mode


def expanded_eval_seeds(
    *,
    base_seeds: list[int],
    repeats: int,
    default_seeds: list[int] | tuple[int, ...],
    default_repeats: int,
    repeat_seed_offset: int,
) -> list[int]:
    seeds = normalize_eval_seeds(base_seeds, default_seeds=default_seeds)
    repeats_norm = normalize_eval_repeats(repeats, default_repeats=default_repeats)
    expanded: list[int] = []
    for repeat_idx in range(repeats_norm):
        offset = repeat_idx * repeat_seed_offset
        for seed in seeds:
            expanded.append(int(seed + offset))
    return expanded


def expanded_default_eval_seeds(*, base_seeds: list[int], repeats: int) -> list[int]:
    return expanded_eval_seeds(
        base_seeds=base_seeds,
        repeats=repeats,
        default_seeds=DEFAULT_EVAL_SEEDS,
        default_repeats=DEFAULT_EVAL_REPEATS,
        repeat_seed_offset=EVAL_REPEAT_SEED_OFFSET,
    )
