from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from kagglebot.json_utils import load_json_object, load_json_object_or_empty, write_json_object
from kagglebot.medals import DEFAULT_TARGET_MEDAL, normalize_target_medal, normalize_target_rank_percentile
from kagglebot.metric_matching import canonical_metric_name_for_match, metrics_equivalent
from kagglebot.paths import CompetitionPaths
from kagglebot.runtime_policy import is_heavy_deep_learning_modality
from kagglebot.scalar_utils import parse_int
from kagglebot.score_sources import (
    DEFAULT_ACCEPTED_SCORE_SOURCES,
    normalize_score_source_list,
    normalize_score_source_name,
)
from kagglebot.solver.metrics import canonical_metric
from kagglebot.types import PlanConfig
from kagglebot.writeup import infer_deliverable_mode_from_paths, normalize_deliverable_mode, normalize_submit_mode

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
FULL_DATASET_REQUIRED_LAYOUTS = frozenset({"flat_full"})
STOP_POLICY_ABORT_ALIASES = (
    "repeated_error_fingerprint_abort",
    "error_fingerprint_policy",
    "abort_on_repeated_error_fingerprint",
)
REQUIRED_SUITE_FIELDS = ("name", "train_mode", "feature_recipe", "lightweight", "promotion_stage")
HIGH_ACCURACY_TABULAR_REQUIRED_SUITES: tuple[dict[str, object], ...] = (
    {
        "name": "competition_only",
        "train_mode": "competition_only",
        "feature_recipe": "full",
        "lightweight": False,
        "promotion_stage": "full_eval",
    },
    {
        "name": "competition_plus_original",
        "train_mode": "competition_plus_original",
        "feature_recipe": "full",
        "lightweight": False,
        "promotion_stage": "ablation_fast",
    },
    {
        "name": "orig_signal_only",
        "train_mode": "competition_only",
        "feature_recipe": "orig_signal_only",
        "lightweight": True,
        "promotion_stage": "ablation_fast",
    },
)
VALID_SUITE_TRAIN_MODES = {"competition_only", "competition_plus_original", "original_only"}
VALID_SUITE_PROMOTION_STAGES = {"ablation_fast", "full_eval"}
ACCURACY_FIRST_DEFAULT_MAX_ITERATIONS = 5
LONG_ACCURACY_FIRST_MAX_ITERATIONS = 3
LONG_ACCURACY_FIRST_TIME_BUDGET_MIN = 12 * 60
ACCURACY_FIRST_MIN_PATIENCE = 4
ACCURACY_FIRST_MIN_CV_FOLDS = 5
ACCURACY_FIRST_EVAL_SEEDS = [42, 2024, 777]
HEAVY_MAX_FULL_TRAIN_FOLDS = 3

COMPETITION_EVAL_OVERRIDES: dict[str, dict[str, object]] = {
    "deep-past-initiative-machine-translation": {
        "metric_name": "Geometric Mean of the BLEU and the chrF++ scores",
        "direction": "maximize",
        "split_strategy": "group_kfold",
        "group_column_hint": "oare_id",
        "task": "translation",
        "task_by_target": {"translation": "translation"},
        "prediction_kind_by_target": {"translation": "text"},
        "tags": ["text", "translation", "n_rows_small", "high_cardinality_cats"],
    }
}
COMPETITION_PROFILE_OVERRIDE_KEYS = frozenset(
    {
        "task",
        "task_by_target",
        "prediction_kind_by_target",
        "tags",
    }
)


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
class TargetRequestDecision:
    target_metric: object
    target_score: object
    target_direction: object
    explicit_target_metric: bool
    explicit_target_direction: bool


@dataclass(frozen=True)
class PlanScoreSourceDecision:
    score_source: str
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class BaseEvaluationRequestDecision:
    score_source: str
    holdout_frac: object
    cv_folds: object
    split_strategy: object
    seed: object
    eval_seeds: list[int]
    eval_repeats: int
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class SplitStrategyOverrideDecision:
    split_strategy: object
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvalBudgetDecision:
    cv_folds: object
    eval_seeds: list[int]
    eval_repeats: int
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class MaxIterationsDecision:
    max_iterations: int
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class SubmitModeDecision:
    submit_mode: str
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class InternetDecision:
    internet: object
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class TimeBudgetDecision:
    time_budget_min: object
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeRequestDecision:
    time_budget_min: object
    kernel_name: object
    internet: object
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoopControlRequestDecision:
    max_total_min: object
    patience: object
    min_improvement: object
    requested_submit_policy: str
    requested_submission_gate: str | None


@dataclass(frozen=True)
class TargetObjectiveDecision:
    target_medal: str | None
    target_rank_percentile: float | None


@dataclass(frozen=True)
class ReadinessStopPolicyDecision:
    readiness_target_score: object
    readiness_method: object
    readiness_k: object
    ci_method: object
    ci_alpha: object
    drift_check: bool
    drift_weight: object
    stop_min_delta: object
    stop_no_improve_patience: object
    stop_same_config_patience: object


@dataclass(frozen=True)
class RankForcePolicyDecision:
    rank_force_major_max_percentile: float
    rank_force_major_min_teams: int


@dataclass(frozen=True)
class ResolvedPlan:
    deliverable_mode: str
    submit_mode: str
    code_competition: bool
    notebook_submit_artifact_mode: str
    target_medal: str | None
    target_rank_percentile: object
    target_metric: object
    target_score: object
    target_direction: object
    score_source: object
    holdout_frac: object
    cv_folds: object
    split_strategy: object
    seed: object
    eval_seeds: list[int]
    eval_repeats: int
    time_budget_min: object
    kernel_name: object
    internet: object
    max_iterations: int
    max_total_min: object
    patience: object
    min_improvement: object
    submit_policy: str
    submission_gate: str
    submission_limit_per_day: int | None
    readiness_target_score: object
    readiness_method: object
    readiness_k: object
    ci_method: object
    ci_alpha: object
    drift_check: object
    drift_weight: object
    stop_min_delta: object
    stop_no_improve_patience: object
    stop_same_config_patience: object
    rank_force_major_max_percentile: float
    rank_force_major_min_teams: int
    evaluation_contract: dict[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "deliverable_mode": self.deliverable_mode,
            "submit_mode": self.submit_mode,
            "code_competition": self.code_competition,
            "notebook_submit_artifact_mode": self.notebook_submit_artifact_mode,
            "target_medal": self.target_medal,
            "target_rank_percentile": self.target_rank_percentile,
            "target_metric": self.target_metric,
            "target_score": self.target_score,
            "target_direction": self.target_direction,
            "score_source": self.score_source,
            "holdout_frac": self.holdout_frac,
            "cv_folds": self.cv_folds,
            "split_strategy": self.split_strategy,
            "seed": self.seed,
            "eval_seeds": self.eval_seeds,
            "eval_repeats": self.eval_repeats,
            "time_budget_min": self.time_budget_min,
            "kernel_name": self.kernel_name,
            "internet": self.internet,
            "max_iterations": self.max_iterations,
            "max_total_min": self.max_total_min,
            "patience": self.patience,
            "min_improvement": self.min_improvement,
            "submit_policy": self.submit_policy,
            "submission_gate": self.submission_gate,
            "submission_limit_per_day": self.submission_limit_per_day,
            "readiness_target_score": self.readiness_target_score,
            "readiness_method": self.readiness_method,
            "readiness_k": self.readiness_k,
            "ci_method": self.ci_method,
            "ci_alpha": self.ci_alpha,
            "drift_check": self.drift_check,
            "drift_weight": self.drift_weight,
            "stop_min_delta": self.stop_min_delta,
            "stop_no_improve_patience": self.stop_no_improve_patience,
            "stop_same_config_patience": self.stop_same_config_patience,
            "rank_force_major_max_percentile": self.rank_force_major_max_percentile,
            "rank_force_major_min_teams": self.rank_force_major_min_teams,
            "evaluation_contract": self.evaluation_contract,
        }


def normalize_plan_payload(payload: dict[str, object]) -> dict[str, object]:
    """Normalize agent-produced plan payload shape without applying environment guardrails."""
    normalized = dict(payload)
    pipelines_raw = normalized.get("pipelines")
    if isinstance(pipelines_raw, list):
        pipelines: list[object] = []
        for index, item in enumerate(pipelines_raw):
            if not isinstance(item, dict):
                pipelines.append(item)
                continue
            pipeline = dict(item)
            name = pipeline.get("name")
            if not isinstance(name, str) or not name.strip():
                model_hint = ""
                models = pipeline.get("models")
                if isinstance(models, list) and models and isinstance(models[0], str) and models[0].strip():
                    model_hint = re.sub(r"[^a-zA-Z0-9]+", "_", models[0]).strip("_").lower()
                pipeline["name"] = model_hint or f"pipeline_{index + 1}"
            key_hyperparameters = pipeline.get("key_hyperparameters")
            if isinstance(key_hyperparameters, dict):
                pipeline["key_hyperparameters"] = _normalize_pipeline_hyperparameter_value(key_hyperparameters) or {}
            pipelines.append(pipeline)
        normalized["pipelines"] = pipelines

    suites_raw = normalized.get("suites")
    if isinstance(suites_raw, list):
        suites: list[object] = []
        for index, item in enumerate(suites_raw):
            if not isinstance(item, dict):
                suites.append(item)
                continue
            suite = dict(item)
            name = suite.get("name")
            if not isinstance(name, str) or not name.strip():
                train_mode = suite.get("train_mode")
                feature_recipe = suite.get("feature_recipe")
                hint = ""
                if isinstance(train_mode, str) and train_mode.strip():
                    hint = train_mode.strip().lower()
                elif isinstance(feature_recipe, str) and feature_recipe.strip():
                    hint = feature_recipe.strip().lower()
                hint = re.sub(r"[^a-zA-Z0-9]+", "_", hint).strip("_")
                suite["name"] = hint or f"suite_{index + 1}"
            suites.append(suite)
        normalized["suites"] = suites

    stop_policy_raw = normalized.get("stop_policy")
    if not isinstance(stop_policy_raw, dict):
        return normalized

    stop_policy = dict(stop_policy_raw)

    if "max_iterations" not in stop_policy:
        top_level_max_iterations = normalized.get("max_iterations")
        if isinstance(top_level_max_iterations, int):
            stop_policy["max_iterations"] = top_level_max_iterations

    if "error_fingerprint_abort" not in stop_policy:
        alias_value: object | None = None
        for alias in STOP_POLICY_ABORT_ALIASES:
            if alias in stop_policy:
                alias_value = stop_policy[alias]
                break
        stop_policy["error_fingerprint_abort"] = alias_value if alias_value is not None else True

    normalized["stop_policy"] = stop_policy
    return normalized


def _normalize_pipeline_hyperparameter_value(value: object) -> object | None:
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            normalized_item = _normalize_pipeline_hyperparameter_value(item)
            if normalized_item is None:
                continue
            normalized[str(key)] = normalized_item
        return normalized
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _normalize_pipeline_hyperparameter_value(value[0])
    return value


def repair_plan_payload_for_profile(
    payload: dict[str, object],
    profile: dict[str, object],
) -> dict[str, object]:
    normalized = normalize_plan_payload(payload)
    if not is_high_accuracy_tabular_blend_target(profile):
        return normalized

    repaired = dict(normalized)
    raw_candidates: list[tuple[object, bool]] = []
    suites_raw = repaired.get("suites")
    if isinstance(suites_raw, list):
        raw_candidates.extend((item, True) for item in suites_raw)

    alias_raw = repaired.pop("suite_aware_ablations", None)
    raw_candidates.extend((item, False) for item in _suite_alias_items(alias_raw))

    toggles_raw = repaired.get("toggles")
    if isinstance(toggles_raw, dict):
        toggles = dict(toggles_raw)
        toggle_alias_raw = toggles.pop("suite_ablations", None)
        raw_candidates.extend((item, False) for item in _suite_alias_items(toggle_alias_raw))
        repaired["toggles"] = toggles

    required_by_name = _required_high_accuracy_suites_by_name()
    required_suites: dict[str, dict[str, object]] = {}
    extra_suites: list[object] = []
    for item, preserve_invalid in raw_candidates:
        if isinstance(item, dict):
            suite = _complete_required_suite(item)
            required_name = _canonical_required_suite_name(suite)
            if required_name is not None:
                required_suites.setdefault(required_name, suite)
            elif preserve_invalid or _is_complete_suite(suite):
                extra_suites.append(suite)
            continue

        alias_name = _canonical_suite_name_from_alias_text(item)
        if alias_name is not None:
            required_suites.setdefault(alias_name, dict(required_by_name[alias_name]))
        elif preserve_invalid:
            extra_suites.append(item)

    for suite in HIGH_ACCURACY_TABULAR_REQUIRED_SUITES:
        required_name = str(suite["name"])
        required_suites.setdefault(required_name, dict(suite))

    repaired["suites"] = [
        required_suites[str(suite["name"])] for suite in HIGH_ACCURACY_TABULAR_REQUIRED_SUITES
    ] + extra_suites
    return normalize_plan_payload(repaired)


def validate_plan_payload(payload: dict[str, object], *, profile: dict[str, object] | None = None) -> list[str]:
    payload = normalize_plan_payload(payload)
    issues: list[str] = []
    required = [
        "target_metric",
        "target_direction",
        "target_score",
        "score_source",
        "holdout_frac",
        "cv_folds",
        "seed",
        "max_iterations",
        "patience",
        "min_improvement",
        "pipelines",
        "toggles",
        "evaluation_protocol",
        "stop_policy",
    ]
    for key in required:
        if key not in payload:
            issues.append(f"PLAN_JSON missing key: {key}.")
    if payload.get("target_direction") not in ("minimize", "maximize"):
        issues.append("PLAN_JSON target_direction must be 'minimize' or 'maximize'.")
    if payload.get("score_source") not in ("holdout", "cv"):
        issues.append("PLAN_JSON score_source must be one of: holdout, cv.")
    if not isinstance(payload.get("target_score"), (int, float)):
        issues.append("PLAN_JSON target_score must be a number.")
    suites = payload.get("suites")
    typed_suites: list[dict[str, object]] | None = None
    if suites is not None:
        if not isinstance(suites, list):
            issues.append("PLAN_JSON suites must be an array when provided.")
        else:
            typed_suites = []
            suite_names: set[str] = set()
            for index, item in enumerate(suites):
                if not isinstance(item, dict):
                    issues.append(f"PLAN_JSON suites[{index}] must be an object.")
                    continue
                typed_suites.append(item)
                for key in REQUIRED_SUITE_FIELDS:
                    if key not in item:
                        issues.append(f"PLAN_JSON suites[{index}] missing key: {key}.")
                name = item.get("name")
                if isinstance(name, str) and name.strip():
                    normalized_name = name.strip()
                    if normalized_name in suite_names:
                        issues.append(f"PLAN_JSON suites[{index}] duplicates suite name: {normalized_name}.")
                    suite_names.add(normalized_name)
                train_mode = item.get("train_mode")
                if train_mode is not None and train_mode not in VALID_SUITE_TRAIN_MODES:
                    issues.append(
                        "PLAN_JSON suites"
                        f"[{index}].train_mode must be one of: {', '.join(sorted(VALID_SUITE_TRAIN_MODES))}."
                    )
                feature_recipe = item.get("feature_recipe")
                if "feature_recipe" in item and (not isinstance(feature_recipe, str) or not feature_recipe.strip()):
                    issues.append(f"PLAN_JSON suites[{index}].feature_recipe must be a non-empty string.")
                lightweight = item.get("lightweight")
                if "lightweight" in item and not isinstance(lightweight, bool):
                    issues.append(f"PLAN_JSON suites[{index}].lightweight must be a boolean.")
                promotion_stage = item.get("promotion_stage")
                if promotion_stage is not None and promotion_stage not in VALID_SUITE_PROMOTION_STAGES:
                    issues.append(
                        "PLAN_JSON suites"
                        f"[{index}].promotion_stage must be one of: {', '.join(sorted(VALID_SUITE_PROMOTION_STAGES))}."
                    )
    pipelines = payload.get("pipelines")
    if not isinstance(pipelines, list):
        issues.append("PLAN_JSON pipelines must be an array.")
    else:
        if not 2 <= len(pipelines) <= 4:
            issues.append("PLAN_JSON pipelines must contain 2-4 entries.")
        for index, item in enumerate(pipelines):
            if not isinstance(item, dict):
                issues.append(f"PLAN_JSON pipelines[{index}] must be an object.")
                continue
            for key in (
                "name",
                "features",
                "models",
                "key_hyperparameters",
                "runtime_memory",
                "failure_modes",
                "fallbacks",
            ):
                if key not in item:
                    issues.append(f"PLAN_JSON pipelines[{index}] missing key: {key}.")
            key_hyperparameters = item.get("key_hyperparameters")
            if "key_hyperparameters" in item and not isinstance(key_hyperparameters, dict):
                issues.append(f"PLAN_JSON pipelines[{index}].key_hyperparameters must be an object.")
            elif isinstance(key_hyperparameters, dict):
                sequence_paths = _find_sequence_hyperparameter_paths(key_hyperparameters)
                if sequence_paths:
                    issues.append(
                        "PLAN_JSON pipelines"
                        f"[{index}].key_hyperparameters must contain scalar runtime values; found sequences at "
                        + ", ".join(sequence_paths)
                        + "."
                    )
        if not issues and profile and is_high_accuracy_tabular_blend_target(profile):
            typed_pipelines = [item for item in pipelines if isinstance(item, dict)]
            issues.extend(_validate_high_accuracy_tabular_plan(typed_pipelines, typed_suites))
    toggles = payload.get("toggles")
    if not isinstance(toggles, dict):
        issues.append("PLAN_JSON toggles must be an object.")
    elif not toggles:
        issues.append("PLAN_JSON toggles must not be empty.")
    evaluation_protocol = payload.get("evaluation_protocol")
    if not isinstance(evaluation_protocol, dict):
        issues.append("PLAN_JSON evaluation_protocol must be an object.")
    else:
        for key in ("cv_type", "n_folds", "seeds", "primary_metric"):
            if key not in evaluation_protocol:
                issues.append(f"PLAN_JSON evaluation_protocol missing key: {key}.")
        seeds = evaluation_protocol.get("seeds")
        if not isinstance(seeds, list) or len(seeds) < 1:
            issues.append("PLAN_JSON evaluation_protocol.seeds must be a non-empty list.")
        n_folds = evaluation_protocol.get("n_folds")
        if not isinstance(n_folds, int) or n_folds < 2:
            issues.append("PLAN_JSON evaluation_protocol.n_folds must be an integer >= 2.")
        cv_type = evaluation_protocol.get("cv_type")
        if not isinstance(cv_type, str) or not cv_type.strip():
            issues.append("PLAN_JSON evaluation_protocol.cv_type must be a non-empty string.")
        primary_metric = evaluation_protocol.get("primary_metric")
        if not isinstance(primary_metric, str) or not primary_metric.strip():
            issues.append("PLAN_JSON evaluation_protocol.primary_metric must be a non-empty string.")
    stop_policy = payload.get("stop_policy")
    if not isinstance(stop_policy, dict):
        issues.append("PLAN_JSON stop_policy must be an object.")
    else:
        if "max_iterations" not in stop_policy:
            issues.append("PLAN_JSON stop_policy missing key: max_iterations.")
        if "error_fingerprint_abort" not in stop_policy:
            issues.append("PLAN_JSON stop_policy missing key: error_fingerprint_abort.")
    return issues


def write_plan_payload(paths: CompetitionPaths, payload: dict[str, object]) -> None:
    payload = normalize_plan_payload(apply_plan_guardrails(paths, payload))
    existing = load_json_object_or_empty(paths.plan_path)
    merged = normalize_plan_payload({**existing, **payload})
    defaults = PlanConfig.from_dict(merged).to_dict()
    persisted = normalize_plan_payload({**merged, **defaults})
    write_json_object(paths.plan_path, persisted)


def load_plan_config(paths: CompetitionPaths) -> PlanConfig:
    payload = load_json_object(paths.plan_path)
    if payload is None:
        return PlanConfig()
    return PlanConfig.from_dict(payload)


def resolve_default_metric_from_artifacts(paths: CompetitionPaths, *, fallback: str = "rmse") -> str:
    plan = load_json_object(paths.plan_path)
    if isinstance(plan, dict):
        metric = _non_empty_string(plan.get("target_metric"))
        if metric is not None:
            return metric

    profile = load_json_object(paths.dataset_profile_path)
    if isinstance(profile, dict):
        for key in ("target_metric", "metric"):
            metric = _non_empty_string(profile.get(key))
            if metric is not None:
                return metric
    return fallback


def _non_empty_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def write_plan_config(paths: CompetitionPaths, plan: PlanConfig) -> None:
    existing = load_json_object_or_empty(paths.plan_path)
    payload = apply_plan_guardrails(paths, {**existing, **plan.to_dict()})
    write_json_object(paths.plan_path, payload)


def plan_config_from_resolved(
    resolved: dict[str, object],
    *,
    default_max_iterations: int,
    default_force_major_rank_max_percentile: float,
    default_force_major_rank_min_teams: int,
) -> PlanConfig:
    target_medal = normalize_target_medal(resolved.get("target_medal"), default=None)
    return PlanConfig(
        deliverable_mode=str(resolved.get("deliverable_mode") or "leaderboard"),
        submit_mode=str(resolved.get("submit_mode") or "file"),
        target_medal=target_medal,
        target_rank_percentile=normalize_target_rank_percentile(
            resolved.get("target_rank_percentile"),
            medal=target_medal,
            fallback=None,
        ),
        target_metric=resolved.get("target_metric"),  # type: ignore[arg-type]
        target_direction=str(resolved.get("target_direction") or "auto"),
        target_score=resolved.get("target_score"),  # type: ignore[arg-type]
        score_source=str(resolved.get("score_source") or "cv"),
        holdout_frac=resolved.get("holdout_frac"),  # type: ignore[arg-type]
        cv_folds=resolved.get("cv_folds"),  # type: ignore[arg-type]
        split_strategy=resolved.get("split_strategy"),  # type: ignore[arg-type]
        seed=resolved.get("seed"),  # type: ignore[arg-type]
        eval_seeds=resolved.get("eval_seeds"),  # type: ignore[arg-type]
        eval_repeats=resolved.get("eval_repeats"),  # type: ignore[arg-type]
        time_budget_min=resolved.get("time_budget_min"),  # type: ignore[arg-type]
        kernel_name=resolved.get("kernel_name"),  # type: ignore[arg-type]
        internet=str(resolved.get("internet") or "on"),
        max_iterations=int(resolved.get("max_iterations") or default_max_iterations),
        max_total_min=resolved.get("max_total_min"),  # type: ignore[arg-type]
        patience=int(resolved.get("patience") or 2),
        min_improvement=float(resolved.get("min_improvement") or 0.0),
        submit_policy=str(resolved.get("submit_policy") or "always"),
        submission_gate=str(resolved.get("submission_gate") or "always"),
        readiness_target_score=resolved.get("readiness_target_score"),  # type: ignore[arg-type]
        readiness_method=str(resolved.get("readiness_method") or "ci_bound"),
        readiness_k=float(resolved.get("readiness_k") or 1.0),
        ci_method=str(resolved.get("ci_method") or "normal"),
        ci_alpha=float(resolved.get("ci_alpha") or 0.05),
        drift_check=bool(resolved.get("drift_check", False)),
        drift_weight=float(resolved.get("drift_weight") or 1.0),
        stop_min_delta=float(resolved.get("stop_min_delta") or 0.0),
        stop_no_improve_patience=int(resolved.get("stop_no_improve_patience") or 0),
        stop_same_config_patience=int(resolved.get("stop_same_config_patience") or 0),
        rank_force_major_max_percentile=normalize_rank_force_percentile(
            resolved.get("rank_force_major_max_percentile"),
            fallback=default_force_major_rank_max_percentile,
        ),
        rank_force_major_min_teams=normalize_rank_force_min_teams(
            resolved.get("rank_force_major_min_teams"),
            fallback=default_force_major_rank_min_teams,
        ),
    )


def write_resolved_plan_config(
    paths: CompetitionPaths,
    resolved: dict[str, object],
    *,
    default_max_iterations: int,
    default_force_major_rank_max_percentile: float,
    default_force_major_rank_min_teams: int,
) -> None:
    write_plan_config(
        paths,
        plan_config_from_resolved(
            resolved,
            default_max_iterations=default_max_iterations,
            default_force_major_rank_max_percentile=default_force_major_rank_max_percentile,
            default_force_major_rank_min_teams=default_force_major_rank_min_teams,
        ),
    )


def apply_plan_guardrails(paths: CompetitionPaths, payload: dict[str, object]) -> dict[str, object]:
    guarded: dict[str, object] = dict(payload)

    raw_toggles = guarded.get("toggles")
    toggles: dict[str, object] | None = dict(raw_toggles) if isinstance(raw_toggles, dict) else {}
    if toggles is not None:
        guarded["toggles"] = toggles
        if isinstance(toggles.get("FAST_DEV"), bool) and bool(toggles.get("FAST_DEV")):
            toggles["FAST_DEV"] = False
            print("[yellow]plan guardrail[/yellow]: forcing FAST_DEV=False for production-quality evaluation.")
        _force_training_and_validation_toggles(toggles)

    raw_runtime_budget = guarded.get("runtime_budget")
    runtime_budget: dict[str, object] = dict(raw_runtime_budget) if isinstance(raw_runtime_budget, dict) else {}
    guarded["runtime_budget"] = runtime_budget
    _force_training_and_validation_runtime(runtime_budget)

    raw_eval_protocol = guarded.get("evaluation_protocol")
    evaluation_protocol: dict[str, object] | None = (
        dict(raw_eval_protocol) if isinstance(raw_eval_protocol, dict) else None
    )
    if evaluation_protocol is not None:
        guarded["evaluation_protocol"] = evaluation_protocol

    profile = _load_dataset_profile_payload(paths)
    task = str(profile.get("task") or "").strip().lower()
    modality = str(profile.get("modality") or "").strip().lower()
    top_class_ratio = _extract_top_class_ratio(profile)
    severe_imbalance = bool(task == "classification" and top_class_ratio is not None and top_class_ratio >= 0.98)
    leaderboard_mode = infer_deliverable_mode_from_paths(paths) != "writeup"

    if leaderboard_mode:
        target_medal = normalize_target_medal(guarded.get("target_medal"), default=DEFAULT_TARGET_MEDAL)
        target_rank_percentile = normalize_target_rank_percentile(
            guarded.get("target_rank_percentile"),
            medal=target_medal,
            fallback=None,
        )
        if target_medal is not None:
            guarded["target_medal"] = target_medal
        if target_rank_percentile is not None:
            guarded["target_rank_percentile"] = target_rank_percentile

    score_source = str(guarded.get("score_source") or "").strip().lower()
    if score_source not in {"cv", "holdout"}:
        guarded["score_source"] = "cv"
        print(
            "[yellow]plan guardrail[/yellow]: non-generalizable score_source is not allowed; forcing score_source=cv."
        )

    accuracy_first_cv = _should_force_accuracy_first_cv(modality=modality)
    heavy_deep_learning = is_heavy_deep_learning_modality(modality)
    if accuracy_first_cv:
        score_source = str(guarded.get("score_source") or "").strip().lower()
        if score_source != "cv":
            guarded["score_source"] = "cv"
            print("[yellow]plan guardrail[/yellow]: accuracy-first mode enabled; forcing score_source=cv.")

        cv_folds = _positive_int(guarded.get("cv_folds"))
        if cv_folds is None or cv_folds < ACCURACY_FIRST_MIN_CV_FOLDS:
            guarded["cv_folds"] = ACCURACY_FIRST_MIN_CV_FOLDS
            if evaluation_protocol is not None:
                evaluation_protocol["n_folds"] = ACCURACY_FIRST_MIN_CV_FOLDS
            print(
                "[yellow]plan guardrail[/yellow]: "
                f"accuracy-first mode enabled; forcing cv_folds>={ACCURACY_FIRST_MIN_CV_FOLDS}."
            )
        elif evaluation_protocol is not None:
            protocol_folds = _positive_int(evaluation_protocol.get("n_folds"))
            if protocol_folds is None or protocol_folds < cv_folds:
                evaluation_protocol["n_folds"] = cv_folds

    if heavy_deep_learning:
        cv_folds = _positive_int(guarded.get("cv_folds"))
        protocol_folds = _positive_int(evaluation_protocol.get("n_folds")) if evaluation_protocol is not None else None
        requested_folds = max([value for value in (cv_folds, protocol_folds) if value is not None], default=None)
        if requested_folds is not None and requested_folds > HEAVY_MAX_FULL_TRAIN_FOLDS:
            guarded["cv_folds"] = HEAVY_MAX_FULL_TRAIN_FOLDS
            if evaluation_protocol is not None:
                evaluation_protocol["n_folds"] = HEAVY_MAX_FULL_TRAIN_FOLDS
            print(
                "[yellow]plan guardrail[/yellow]: "
                f"heavy modality detected; capping full-training folds at {HEAVY_MAX_FULL_TRAIN_FOLDS}. "
                "Use cached embeddings, TTA, or lightweight heads for extra validation."
            )

    eval_seeds = _normalize_seed_list(guarded.get("eval_seeds"))
    protocol_seeds = _normalize_seed_list(evaluation_protocol.get("seeds")) if evaluation_protocol is not None else []
    if _should_force_multi_seed_evaluation(modality=modality, task=task, profile=profile) and (
        len(eval_seeds) < 3 or len(protocol_seeds) < 3
    ):
        guarded["eval_seeds"] = list(ACCURACY_FIRST_EVAL_SEEDS)
        if evaluation_protocol is not None:
            evaluation_protocol["seeds"] = list(ACCURACY_FIRST_EVAL_SEEDS)
        print(
            "[yellow]plan guardrail[/yellow]: "
            f"forcing evaluation seeds={ACCURACY_FIRST_EVAL_SEEDS} for lower-variance model ranking."
        )
    elif heavy_deep_learning:
        selected_seeds = eval_seeds or protocol_seeds or [ACCURACY_FIRST_EVAL_SEEDS[0]]
        if len(selected_seeds) > 1:
            print(
                "[yellow]plan guardrail[/yellow]: "
                f"heavy modality detected; using one full-training seed ({selected_seeds[0]}). "
                "Use extra seeds only for cheap heads/blends or final confirmation."
            )
            selected_seeds = [selected_seeds[0]]
        guarded["eval_seeds"] = list(selected_seeds)
        if evaluation_protocol is not None:
            evaluation_protocol["seeds"] = list(selected_seeds)

    time_budget_min = _positive_int(guarded.get("time_budget_min"))
    long_heavy_run = heavy_deep_learning and (
        time_budget_min is None or time_budget_min >= LONG_ACCURACY_FIRST_TIME_BUDGET_MIN
    )
    target_max_iterations = (
        LONG_ACCURACY_FIRST_MAX_ITERATIONS if long_heavy_run else ACCURACY_FIRST_DEFAULT_MAX_ITERATIONS
    )
    max_iterations = _positive_int(guarded.get("max_iterations"))
    if max_iterations != target_max_iterations:
        guarded["max_iterations"] = target_max_iterations
        max_iterations = target_max_iterations
        print(
            "[yellow]plan guardrail[/yellow]: "
            f"forcing max_iterations={target_max_iterations} "
            "for the selected runtime profile."
        )

    patience = _positive_int(guarded.get("patience"))
    if patience is None or patience < ACCURACY_FIRST_MIN_PATIENCE:
        guarded["patience"] = ACCURACY_FIRST_MIN_PATIENCE
        print(f"[yellow]plan guardrail[/yellow]: forcing patience>={ACCURACY_FIRST_MIN_PATIENCE} for stability.")

    raw_stop_policy = guarded.get("stop_policy")
    stop_policy = dict(raw_stop_policy) if isinstance(raw_stop_policy, dict) else {}
    guarded["stop_policy"] = stop_policy
    stop_max_iterations = _positive_int(stop_policy.get("max_iterations"))
    if stop_max_iterations != max_iterations:
        stop_policy["max_iterations"] = max_iterations
        print("[yellow]plan guardrail[/yellow]: aligning stop_policy.max_iterations with top-level max_iterations.")

    if severe_imbalance:
        score_source = str(guarded.get("score_source") or "").strip().lower()
        if score_source != "cv":
            guarded["score_source"] = "cv"
            print("[yellow]plan guardrail[/yellow]: detected severe class imbalance; forcing score_source=cv.")

        if evaluation_protocol is not None and _should_force_multi_seed_evaluation(
            modality=modality,
            task=task,
            profile=profile,
        ):
            seeds = evaluation_protocol.get("seeds")
            normalized_seeds = [seed for seed in seeds if isinstance(seed, int)] if isinstance(seeds, list) else []
            if len(normalized_seeds) < 3:
                evaluation_protocol["seeds"] = [42, 2024, 777]
                guarded["eval_seeds"] = [42, 2024, 777]
                print(
                    "[yellow]plan guardrail[/yellow]: "
                    "detected severe class imbalance; forcing evaluation seeds=[42, 2024, 777]."
                )

        if isinstance(guarded.get("cv_folds"), int):
            cv_folds = int(guarded["cv_folds"])
            adjusted_folds = _adjust_cv_folds_for_imbalance(profile=profile, cv_folds=cv_folds)
            if adjusted_folds < cv_folds:
                guarded["cv_folds"] = adjusted_folds
                if evaluation_protocol is not None:
                    evaluation_protocol["n_folds"] = adjusted_folds
                print(
                    "[yellow]plan guardrail[/yellow]: "
                    f"detected severe class imbalance; reducing cv_folds from {cv_folds} to {adjusted_folds}."
                )

        if toggles is not None:
            classifier_keys = [
                key for key in toggles if "CLASSIFIER" in key.upper() and key.upper() != "ALLOW_PRETRAINED_WEIGHTS"
            ]
            if classifier_keys and not any(bool(toggles.get(key)) for key in classifier_keys):
                toggles[classifier_keys[0]] = True
                print(
                    f"[yellow]plan guardrail[/yellow]: detected severe class imbalance; enabling {classifier_keys[0]}."
                )

    if toggles is not None and is_heavy_deep_learning_modality(modality):
        allow_pretrained = toggles.get("ALLOW_PRETRAINED_WEIGHTS")
        if isinstance(allow_pretrained, bool) and not allow_pretrained and not _rules_disallow_external_data(paths):
            toggles["ALLOW_PRETRAINED_WEIGHTS"] = True
            print(
                "[yellow]plan guardrail[/yellow]: "
                "modality suggests transfer learning and rules do not ban external data; "
                "enabling ALLOW_PRETRAINED_WEIGHTS."
            )

    return guarded


def is_high_accuracy_tabular_blend_target(profile: dict[str, object]) -> bool:
    modality = str(profile.get("modality") or "").strip().lower()
    task = str(profile.get("task") or "").strip().lower()
    tags_raw = profile.get("tags")
    tags = (
        [str(item).strip().lower() for item in tags_raw if isinstance(item, str)] if isinstance(tags_raw, list) else []
    )
    train_rows = _positive_int(profile.get("train_rows")) or 0
    categorical_count = (
        len(profile.get("categorical_columns", [])) if isinstance(profile.get("categorical_columns"), list) else 0
    )
    high_cardinality_count = (
        len(profile.get("high_cardinality_columns", []))
        if isinstance(profile.get("high_cardinality_columns"), list)
        else 0
    )
    binary_like = task == "binary" or "binary" in tags or (task == "classification" and "multiclass" not in tags)
    mixed_categorical = categorical_count >= 3 or high_cardinality_count >= 1
    return modality == "tabular" and binary_like and train_rows >= 5000 and mixed_categorical


def _find_sequence_hyperparameter_paths(value: object, *, prefix: str = "key_hyperparameters") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}"
            paths.extend(_find_sequence_hyperparameter_paths(item, prefix=child_prefix))
        return paths
    if isinstance(value, (list, tuple)):
        paths.append(prefix)
    return paths


def _suite_token(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _required_high_accuracy_suites_by_name() -> dict[str, dict[str, object]]:
    return {str(item["name"]): dict(item) for item in HIGH_ACCURACY_TABULAR_REQUIRED_SUITES}


def _canonical_required_suite_name(suite: dict[str, object]) -> str | None:
    name = _suite_token(suite.get("name"))
    train_mode = _suite_token(suite.get("train_mode"))
    feature_recipe = _suite_token(suite.get("feature_recipe"))
    if feature_recipe == "orig_signal_only" or name in {"orig_signal_only", "original_signal_only"}:
        return "orig_signal_only"
    if train_mode == "competition_plus_original" or name in {"competition_plus_original", "competition_plus_orig"}:
        return "competition_plus_original"
    if train_mode == "competition_only" or name in {"competition_only", "comp_only"}:
        return "competition_only"
    return None


def _canonical_suite_name_from_alias_text(value: object) -> str | None:
    token = _suite_token(value)
    if not token:
        return None
    if token in {"orig_signal_only", "original_signal_only"} or ("orig" in token and "signal" in token):
        return "orig_signal_only"
    if token in {"competition_plus_original", "competition_plus_orig"} or (
        "competition" in token and "original" in token
    ):
        return "competition_plus_original"
    if token in {"competition_only", "comp_only"} or ("competition" in token and "only" in token):
        return "competition_only"
    return None


def _complete_required_suite(suite: dict[str, object]) -> dict[str, object]:
    required_name = _canonical_required_suite_name(suite)
    if required_name is None:
        return dict(suite)
    defaults = _required_high_accuracy_suites_by_name()[required_name]
    extras = {key: value for key, value in suite.items() if key not in REQUIRED_SUITE_FIELDS}
    return {**defaults, **extras}


def _is_complete_suite(suite: dict[str, object]) -> bool:
    return all(key in suite for key in REQUIRED_SUITE_FIELDS)


def _suite_alias_items(value: object) -> list[object]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return list(value.values())
    return []


def _pipeline_texts(pipeline: dict[str, object]) -> list[str]:
    texts: list[str] = []
    for key in ("name",):
        value = pipeline.get(key)
        if isinstance(value, str) and value.strip():
            texts.append(value.strip().lower())
    for key in ("models", "features", "fallbacks", "failure_modes"):
        value = pipeline.get(key)
        if isinstance(value, list):
            texts.extend(str(item).strip().lower() for item in value if str(item).strip())
        elif isinstance(value, str) and value.strip():
            texts.append(value.strip().lower())
    return texts


def _validate_high_accuracy_tabular_plan(
    pipelines: list[dict[str, object]],
    suites: list[dict[str, object]] | None,
) -> list[str]:
    issues: list[str] = []
    family_counts = {"catboost": 0, "xgboost": 0, "lightgbm": 0}
    blend_present = False
    for pipeline in pipelines:
        texts = _pipeline_texts(pipeline)
        haystack = " ".join(texts)
        for family in family_counts:
            if family in haystack:
                family_counts[family] += 1
        if any(token in haystack for token in ("blend", "ensemble", "stack", "rank", "logit", "weighted")):
            blend_present = True
    if len(pipelines) < 3:
        issues.append("PLAN_JSON must include at least 3 pipelines for high-accuracy tabular binary search.")
    if family_counts["catboost"] < 1:
        issues.append("PLAN_JSON must include a CatBoost pipeline for raw categorical handling.")
    if family_counts["xgboost"] < 1:
        issues.append("PLAN_JSON must include an XGBoost pipeline with leak-safe encoded/stat features.")
    if family_counts["lightgbm"] < 1 and (family_counts["catboost"] + family_counts["xgboost"] < 3):
        issues.append("PLAN_JSON must include LightGBM or a second CatBoost/XGBoost variant.")
    if not blend_present:
        issues.append("PLAN_JSON must include at least one OOF blend/ensemble candidate.")
    if not suites:
        issues.append(
            "PLAN_JSON must include suite-aware ablations for high-accuracy tabular search "
            "(competition_only, competition_plus_original, orig_signal_only)."
        )
        return issues
    suite_names = {str(item.get("name") or "").strip() for item in suites}
    train_modes = {str(item.get("train_mode") or "").strip() for item in suites}
    feature_recipes = {str(item.get("feature_recipe") or "").strip() for item in suites}
    promotion_stages = {str(item.get("promotion_stage") or "").strip() for item in suites}
    if "competition_only" not in train_modes:
        issues.append("PLAN_JSON suites must include a competition_only suite.")
    if "competition_plus_original" not in train_modes:
        issues.append("PLAN_JSON suites must include a competition_plus_original suite.")
    if "orig_signal_only" not in feature_recipes and "orig_signal_only" not in suite_names:
        issues.append("PLAN_JSON suites must include an orig_signal_only ablation suite.")
    if "full_eval" not in promotion_stages:
        issues.append("PLAN_JSON suites must include at least one full_eval suite.")
    return issues


def _positive_int(value: object) -> int | None:
    parsed = parse_int(value, allow_commas=True, allow_float=True, require_integral_float=False)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def _force_training_and_validation_toggles(toggles: dict[str, object]) -> None:
    true_keys = {
        "ENABLE_TRAINING",
        "ENABLE_FULL_TRAINING",
        "ENABLE_REFERENCE_TRAINING",
        "RUN_VALIDATION_GENERATION",
        "ENABLE_VALIDATION_GENERATION",
        "RUN_VALIDATION",
        "ENABLE_VALIDATION",
        "SAVE_OOF",
        "SAVE_VALIDATION_PREDICTIONS",
    }
    false_keys = {
        "FAST_DEV",
        "SKIP_TRAINING",
        "DISABLE_TRAINING",
        "TRAINING_DISABLED",
        "SKIP_VALIDATION",
        "DISABLE_VALIDATION",
        "VALIDATION_DISABLED",
        "PACKAGING_ONLY",
        "ADAPTER_PACKAGING_ONLY",
        "ALLOW_IDENTITY_ADAPTER",
        "ALLOW_DEBUG_NOOP_ADAPTER",
        "ALLOW_NOOP_FALLBACK",
        "ALLOW_UNSCORED_SUBMISSION",
    }
    for key in true_keys:
        if toggles.get(key) is not True:
            toggles[key] = True
            print(f"[yellow]plan guardrail[/yellow]: forcing {key}=True.")
    for key in false_keys:
        if toggles.get(key) is not False:
            toggles[key] = False
            print(f"[yellow]plan guardrail[/yellow]: forcing {key}=False.")


def _force_training_and_validation_runtime(runtime_budget: dict[str, object]) -> None:
    minimum_ints = {
        "full_training_seeds": 1,
        "full_training_folds": 1,
        "validation_generation_max_samples": 64,
        "validation_generation_max_samples_rtx3060": 64,
        "validation_generation_max_samples_large_gpu": 256,
        "max_val_samples": 128,
        "max_validation_samples": 128,
        "num_steps_smoke": 10,
    }
    for key, minimum in minimum_ints.items():
        current = _positive_int(runtime_budget.get(key))
        if current is None or current < minimum:
            runtime_budget[key] = minimum
            print(f"[yellow]plan guardrail[/yellow]: forcing runtime_budget.{key}>={minimum}.")
    forced_true = {
        "enable_reference_training",
        "enable_training",
        "run_validation_generation",
        "enable_validation_generation",
        "run_validation",
        "enable_validation",
    }
    for key in forced_true:
        if runtime_budget.get(key) is not True:
            runtime_budget[key] = True
            print(f"[yellow]plan guardrail[/yellow]: forcing runtime_budget.{key}=true.")
    forced_false = {
        "packaging_only",
        "adapter_packaging_only",
        "allow_identity_adapter",
        "allow_debug_noop_adapter",
        "allow_noop_fallback",
        "allow_unscored_submission",
    }
    for key in forced_false:
        if runtime_budget.get(key) is not False:
            runtime_budget[key] = False
            print(f"[yellow]plan guardrail[/yellow]: forcing runtime_budget.{key}=false.")


def _normalize_seed_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    normalized: list[int] = []
    seen: set[int] = set()
    for item in value:
        if isinstance(item, int) and item not in seen:
            normalized.append(item)
            seen.add(item)
    return normalized


def _should_force_accuracy_first_cv(*, modality: str) -> bool:
    return not is_heavy_deep_learning_modality(modality)


def _should_force_multi_seed_evaluation(*, modality: str, task: str, profile: dict[str, object]) -> bool:
    if task == "classification":
        top_class_ratio = _extract_top_class_ratio(profile)
        if top_class_ratio is not None and top_class_ratio >= 0.98:
            return True
    return not is_heavy_deep_learning_modality(modality)


def _load_dataset_profile_payload(paths: CompetitionPaths) -> dict[str, object]:
    payload = load_json_object(paths.dataset_profile_path)
    return payload or {}


def _extract_top_class_ratio(profile: dict[str, object]) -> float | None:
    target_stats = profile.get("target_stats")
    if not isinstance(target_stats, dict):
        return None
    value = target_stats.get("top_class_ratio")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _adjust_cv_folds_for_imbalance(*, profile: dict[str, object], cv_folds: int) -> int:
    if cv_folds <= 2:
        return cv_folds
    train_rows = profile.get("train_rows")
    top_class_ratio = _extract_top_class_ratio(profile)
    if not isinstance(train_rows, int) or train_rows <= 0 or top_class_ratio is None:
        return cv_folds
    minority_count = int(round(float(train_rows) * max(0.0, 1.0 - top_class_ratio)))
    if minority_count <= 0:
        return cv_folds
    if minority_count < cv_folds * 2:
        if minority_count <= 3:
            return 2
        return max(2, min(cv_folds, minority_count // 2))
    return cv_folds


def _rules_disallow_external_data(paths: CompetitionPaths) -> bool:
    context = "\n".join(
        [
            _read_text(paths.rules_md_path),
            _read_text(paths.rules_html_path),
            _read_text(paths.overview_md_path),
        ]
    ).lower()
    if not context.strip():
        return False
    allow_patterns = (
        r"external data[^.\n]{0,80}allow",
        r"outside data[^.\n]{0,80}allow",
    )
    if any(re.search(pattern, context) for pattern in allow_patterns):
        return False
    deny_patterns = (
        r"external data[^.\n]{0,80}prohibit",
        r"outside data[^.\n]{0,80}prohibit",
        r"\bno external data\b",
        r"external data[^.\n]{0,80}forbidden",
    )
    return any(re.search(pattern, context) for pattern in deny_patterns)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


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


def needs_planning(
    *,
    agent: str | None,
    config_target_metric: object,
    config_target_score: object,
    config_target_direction: object,
    plan_target_metric: object,
    plan_target_score: object,
    plan_target_direction: object,
) -> bool:
    if agent in ("codex", "pipeline"):
        return True
    target_metric = config_target_metric or plan_target_metric
    target_score = config_target_score if config_target_score is not None else plan_target_score
    target_direction = config_target_direction or plan_target_direction
    if target_metric is None or target_score is None:
        return True
    return target_direction in (None, "auto")


def resolve_target_request(
    *,
    config_target_metric: object,
    config_target_score: object,
    config_target_direction: object,
    plan: PlanConfig,
    spec_values: EvaluationSpecValues,
) -> TargetRequestDecision:
    """Resolve raw target metric/score/direction before competition metric overrides."""

    return TargetRequestDecision(
        target_metric=_choose(config_target_metric, plan.target_metric, spec_values.metric_name),
        target_score=_choose(config_target_score, plan.target_score, spec_values.readiness_target_score),
        target_direction=_choose(config_target_direction, plan.target_direction, spec_values.direction or "auto"),
        explicit_target_metric=config_target_metric is not None or plan.target_metric is not None,
        explicit_target_direction=config_target_direction is not None or plan.target_direction is not None,
    )


def should_skip_planning_on_resume(
    *,
    resume_run: bool,
    plan_path: Path,
    kernel_path: Path,
    completion_path: Path,
    run_id: str,
    required_strategy_engine: str,
) -> bool:
    if not resume_run:
        return False
    if not plan_path.exists() or not kernel_path.exists():
        return False
    completion = load_json_object_or_empty(completion_path)
    return (
        completion.get("status") == "complete"
        and completion.get("run_id") == run_id
        and completion.get("strategy_engine") == required_strategy_engine
    )


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _number_or_none(value: object) -> int | float | None:
    return value if isinstance(value, (int, float)) else None


def _choose(value: object, fallback: object, default: object) -> object:
    if value is not None:
        return value
    if fallback is not None:
        return fallback
    return default


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


def resolve_base_evaluation_request(
    *,
    config_score_source: object,
    config_holdout_frac: object,
    config_cv_folds: object,
    config_seed: object,
    plan: PlanConfig,
    spec_values: EvaluationSpecValues,
) -> BaseEvaluationRequestDecision:
    """Resolve base local-evaluation request values before runtime budget caps."""

    score_source_decision = resolve_plan_score_source(_choose(config_score_source, plan.score_source, "cv"))
    return BaseEvaluationRequestDecision(
        score_source=score_source_decision.score_source,
        holdout_frac=_choose(config_holdout_frac, plan.holdout_frac, 0.2),
        cv_folds=_choose(
            config_cv_folds, plan.cv_folds, spec_values.n_splits if spec_values.n_splits is not None else 5
        ),
        split_strategy=_choose(None, plan.split_strategy, spec_values.split_strategy),
        seed=_choose(config_seed, plan.seed, spec_values.seed if spec_values.seed is not None else 42),
        eval_seeds=normalize_default_eval_seeds(plan.eval_seeds, fallback=list(spec_values.eval_seeds)),
        eval_repeats=normalize_default_eval_repeats(plan.eval_repeats, fallback=spec_values.repeats),
        messages=score_source_decision.messages,
    )


def resolve_target_objective(
    *,
    plan_target_medal: object,
    plan_target_rank_percentile: object,
    spec_target_medal: object,
    spec_target_rank_percentile: object,
    deliverable_mode: str,
    search_stop_rank_percentile: float | None,
    default_target_medal: str,
) -> TargetObjectiveDecision:
    """Resolve leaderboard medal/rank objectives from plan, evaluation spec, and competition policy."""

    plan_medal = normalize_target_medal(plan_target_medal, default=None)
    spec_medal = normalize_target_medal(spec_target_medal, default=None)
    default_medal = default_target_medal if deliverable_mode == "leaderboard" else None
    target_medal = spec_medal or plan_medal or default_medal

    target_rank_percentile = normalize_target_rank_percentile(
        plan_target_rank_percentile,
        medal=target_medal,
        fallback=None,
    )
    target_rank_percentile = normalize_target_rank_percentile(
        spec_target_rank_percentile,
        medal=spec_medal or target_medal,
        fallback=target_rank_percentile,
    )
    if search_stop_rank_percentile is not None:
        search_stop = float(search_stop_rank_percentile)
        target_rank_percentile = (
            search_stop if target_rank_percentile is None else min(float(target_rank_percentile), search_stop)
        )
    return TargetObjectiveDecision(
        target_medal=target_medal,
        target_rank_percentile=target_rank_percentile,
    )


def resolve_readiness_stop_policy(
    *,
    plan: PlanConfig,
    spec_values: EvaluationSpecValues,
    target_score: object,
    min_improvement: object,
    patience: object,
) -> ReadinessStopPolicyDecision:
    """Resolve readiness, drift, and no-improvement stop policy values."""

    return ReadinessStopPolicyDecision(
        readiness_target_score=_choose(
            None,
            plan.readiness_target_score,
            spec_values.readiness_target_score if spec_values.readiness_target_score is not None else target_score,
        ),
        readiness_method=_choose(None, plan.readiness_method, spec_values.readiness_method or "ci_bound"),
        readiness_k=_choose(
            None, plan.readiness_k, spec_values.readiness_k if spec_values.readiness_k is not None else 1.0
        ),
        ci_method=_choose(None, plan.ci_method, spec_values.ci_method or "normal"),
        ci_alpha=_choose(None, plan.ci_alpha, spec_values.ci_alpha if spec_values.ci_alpha is not None else 0.05),
        drift_check=bool(
            _choose(
                None,
                plan.drift_check,
                spec_values.drift_enabled if spec_values.drift_enabled is not None else False,
            )
        ),
        drift_weight=_choose(
            None,
            plan.drift_weight,
            spec_values.drift_weight if spec_values.drift_weight is not None else 1.0,
        ),
        stop_min_delta=_choose(
            None,
            plan.stop_min_delta,
            spec_values.stop_min_delta if spec_values.stop_min_delta is not None else min_improvement,
        ),
        stop_no_improve_patience=_choose(
            None,
            plan.stop_no_improve_patience,
            spec_values.stop_no_improve_patience if spec_values.stop_no_improve_patience is not None else patience,
        ),
        stop_same_config_patience=_choose(
            None,
            plan.stop_same_config_patience,
            spec_values.stop_same_config_patience if spec_values.stop_same_config_patience is not None else 0,
        ),
    )


def resolve_rank_force_policy(
    *,
    rank_force_major_max_percentile: object,
    rank_force_major_min_teams: object,
    target_rank_percentile: object,
    default_max_percentile: float,
    default_min_teams: int,
) -> RankForcePolicyDecision:
    """Resolve rank-based major-overhaul guard thresholds."""

    max_percentile = normalize_rank_force_percentile(
        rank_force_major_max_percentile,
        fallback=default_max_percentile,
    )
    if isinstance(target_rank_percentile, (int, float)):
        max_percentile = min(max_percentile, float(target_rank_percentile))
    min_teams = normalize_rank_force_min_teams(
        rank_force_major_min_teams,
        fallback=default_min_teams,
    )
    return RankForcePolicyDecision(
        rank_force_major_max_percentile=max_percentile,
        rank_force_major_min_teams=min_teams,
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


def resolve_eval_budget_policy(
    *,
    heavy_local_gpu: bool,
    cv_folds: object,
    seed: object,
    eval_seeds: list[int],
    eval_repeats: int,
    max_heavy_local_gpu_cv_folds: int,
    default_eval_seeds: list[int] | tuple[int, ...] = DEFAULT_EVAL_SEEDS,
    default_eval_repeats: int = DEFAULT_EVAL_REPEATS,
) -> EvalBudgetDecision:
    """Apply plan-level evaluation budget policy for heavy local GPU workloads."""

    messages: list[str] = []
    resolved_cv_folds = cv_folds
    cv_folds_int = parse_int(cv_folds, allow_commas=True, allow_float=True, require_integral_float=False)
    if cv_folds_int is not None:
        resolved_cv_folds = cv_folds_int
    if heavy_local_gpu and isinstance(resolved_cv_folds, int) and resolved_cv_folds > max_heavy_local_gpu_cv_folds:
        messages.append(
            "[yellow]note[/yellow]: heavy modality on local_gpu; "
            f"capping full-training cv_folds from {resolved_cv_folds} to {max_heavy_local_gpu_cv_folds}. "
            "Use cached embeddings/TTA/lightweight heads for extra validation instead of full folds."
        )
        resolved_cv_folds = max_heavy_local_gpu_cv_folds

    resolved_eval_seeds = list(eval_seeds)
    if heavy_local_gpu and len(resolved_eval_seeds) > 1:
        primary_seed = int(seed) if isinstance(seed, int) else resolved_eval_seeds[0]
        if primary_seed not in resolved_eval_seeds:
            primary_seed = resolved_eval_seeds[0]
        resolved_eval_seeds = [primary_seed]
        messages.append(
            "[yellow]note[/yellow]: heavy modality on local_gpu; "
            f"using one full-training seed ({primary_seed}) to stay inside the local runtime budget. "
            "Reserve extra seeds for cheap heads/blends or final confirmation."
        )
    elif len(resolved_eval_seeds) < 2:
        resolved_eval_seeds = list(default_eval_seeds)
        messages.append(
            "[yellow]note[/yellow]: evaluation seeds were single-seed; "
            f"upgrading to multi-seed defaults {list(default_eval_seeds)}."
        )

    resolved_eval_repeats = int(eval_repeats)
    if heavy_local_gpu and resolved_eval_repeats > 1:
        messages.append(
            "[yellow]note[/yellow]: heavy modality on local_gpu; "
            "using one full-training evaluation repeat to keep each iteration under the runtime budget."
        )
        resolved_eval_repeats = 1
    elif resolved_eval_repeats < 2:
        messages.append(
            "[yellow]note[/yellow]: evaluation repeats were < 2; "
            f"upgrading to default {default_eval_repeats} to reduce noise."
        )
        resolved_eval_repeats = int(default_eval_repeats)

    return EvalBudgetDecision(
        cv_folds=resolved_cv_folds,
        eval_seeds=resolved_eval_seeds,
        eval_repeats=resolved_eval_repeats,
        messages=tuple(messages),
    )


def resolve_plan_max_iterations(
    *,
    config_max_iterations: int | None,
    plan_max_iterations: object,
    default_max_iterations: int,
) -> MaxIterationsDecision:
    """Resolve CLI/plan max_iterations with legacy tolerant plan parsing."""

    if config_max_iterations is not None:
        return MaxIterationsDecision(max_iterations=max(1, int(config_max_iterations)))

    planned = parse_int(plan_max_iterations, allow_commas=True, allow_float=True, require_integral_float=False)
    if planned is not None and planned > 0:
        return MaxIterationsDecision(max_iterations=planned)

    messages: list[str] = []
    if planned is not None:
        messages.append(
            "[yellow]note[/yellow]: invalid plan max_iterations "
            f"({plan_max_iterations}); using default {default_max_iterations}."
        )
    return MaxIterationsDecision(max_iterations=int(default_max_iterations), messages=tuple(messages))


def resolve_heavy_local_gpu_max_iterations(
    *,
    heavy_local_gpu: bool,
    time_budget_min: object,
    max_iterations: int,
    long_iteration_budget_min: int,
    max_long_iterations: int,
) -> MaxIterationsDecision:
    """Cap long heavy local GPU loops so single iterations can run deeper."""

    if (
        heavy_local_gpu
        and (time_budget_min is None or time_budget_min >= long_iteration_budget_min)
        and max_iterations > max_long_iterations
    ):
        return MaxIterationsDecision(
            max_iterations=max_long_iterations,
            messages=(
                "[yellow]note[/yellow]: heavy long-running local_gpu plan detected; "
                f"capping max_iterations from {max_iterations} to {max_long_iterations} "
                "so accuracy-first iterations can run deeper.",
            ),
        )
    return MaxIterationsDecision(max_iterations=max_iterations)


def resolve_submit_mode_constraints(
    *,
    submit_mode: str,
    compute: object,
    code_competition: bool,
    notebook_submissions_only: bool,
) -> SubmitModeDecision:
    """Apply notebook/code competition submit-mode constraints."""

    resolved = str(submit_mode or "file")
    messages: list[str] = []
    if code_competition and resolved != "notebook":
        messages.append("[yellow]note[/yellow]: code competition detected; forcing submit_mode=notebook.")
        resolved = "notebook"
    if notebook_submissions_only and resolved != "notebook":
        messages.append(
            "[yellow]note[/yellow]: competition requires notebook-based submissions; forcing submit_mode=notebook."
        )
        resolved = "notebook"
    if resolved == "notebook" and not str(compute).startswith("kaggle_"):
        messages.append(
            "[yellow]note[/yellow]: notebook-based submission is selected; "
            "autopilot will auto-switch submit mode to notebook submit."
        )
    return SubmitModeDecision(submit_mode=resolved, messages=tuple(messages))


def resolve_internet_policy(*, internet: object, internet_must_be_off: bool) -> InternetDecision:
    """Normalize internet mode and apply competition internet constraints."""

    resolved = internet
    if resolved in (None, "auto"):
        resolved = "on"
    if internet_must_be_off and str(resolved).strip().lower() != "off":
        return InternetDecision(
            internet="off",
            messages=("[yellow]note[/yellow]: rules require internet disabled; forcing internet=off.",),
        )
    return InternetDecision(internet=resolved)


def resolve_runtime_request(
    *,
    config_time_budget_min: object,
    config_kernel_name: object,
    config_internet: object,
    plan: PlanConfig,
    internet_must_be_off: bool,
) -> RuntimeRequestDecision:
    """Resolve runtime request values before runtime-limit and local-budget caps."""

    internet_decision = resolve_internet_policy(
        internet=_choose(config_internet, plan.internet, "on"),
        internet_must_be_off=internet_must_be_off,
    )
    return RuntimeRequestDecision(
        time_budget_min=_choose(config_time_budget_min, plan.time_budget_min, None),
        kernel_name=_choose(config_kernel_name, plan.kernel_name, None),
        internet=internet_decision.internet,
        messages=internet_decision.messages,
    )


def resolve_loop_control_request(
    *,
    config_max_total_min: object,
    config_patience: object,
    config_min_improvement: object,
    config_submit_policy: object,
    plan: PlanConfig,
    spec_values: EvaluationSpecValues,
) -> LoopControlRequestDecision:
    """Resolve loop-control and submit-policy request values before submission-limit policy."""

    requested_gate_raw = _choose(None, plan.submission_gate, spec_values.submission_gate)
    requested_gate = str(requested_gate_raw or "").strip().lower() or None
    return LoopControlRequestDecision(
        max_total_min=_choose(config_max_total_min, plan.max_total_min, None),
        patience=_choose(config_patience, plan.patience, 2),
        min_improvement=_choose(config_min_improvement, plan.min_improvement, 0.0),
        requested_submit_policy=str(_choose(config_submit_policy, plan.submit_policy, "always") or "always"),
        requested_submission_gate=requested_gate,
    )


def resolve_time_budget_policy(
    *,
    time_budget_min: object,
    runtime_limit_min: int | None,
    local_budget_min: int | None,
    is_local_gpu: bool,
) -> TimeBudgetDecision:
    """Apply rules and local runtime caps to the requested time budget."""

    resolved = time_budget_min
    messages: list[str] = []
    if runtime_limit_min is not None:
        current_limit = int(resolved) if isinstance(resolved, (int, float)) else None
        if current_limit is None or current_limit > runtime_limit_min:
            messages.append(
                "[yellow]note[/yellow]: rules impose notebook runtime cap; "
                f"forcing time_budget_min={runtime_limit_min}."
            )
            resolved = runtime_limit_min
    if is_local_gpu:
        current_limit = int(resolved) if isinstance(resolved, (int, float)) else None
        if local_budget_min is not None and (current_limit is None or current_limit > local_budget_min):
            messages.append(
                "[yellow]note[/yellow]: local_gpu per-kernel budget limit active; "
                f"forcing time_budget_min={local_budget_min}. "
                "Unset KAGGLEBOT_LOCAL_GPU_TIME_BUDGET_MIN or set it to 0 for unlimited local runtime."
            )
            resolved = local_budget_min
    return TimeBudgetDecision(time_budget_min=resolved, messages=tuple(messages))


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


def normalize_competition_eval_override(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    metric = _first_non_empty_str(raw, "metric_name", "metric", "target_metric")
    direction = _first_non_empty_str(raw, "direction", "target_direction").lower()
    split_strategy = _first_non_empty_str(raw, "split_strategy", "split_strategy_hint")
    group_column_hint = _first_non_empty_str(raw, "group_column_hint", "group_column")
    override: dict[str, str] = {}
    if metric:
        override["metric_name"] = metric
    if direction in {"minimize", "maximize"}:
        override["direction"] = direction
    normalized_split = normalize_split_strategy_name(split_strategy) or infer_split_strategy_from_hint_text(
        split_strategy
    )
    if normalized_split is not None:
        override["split_strategy"] = normalized_split
    if group_column_hint:
        override["group_column_hint"] = group_column_hint
    return override


def _first_non_empty_str(raw: dict[object, object], *keys: str) -> str:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def competition_eval_override(slug: str, fallback_overrides: object | None = None) -> dict[str, str]:
    override = normalize_competition_eval_override(COMPETITION_EVAL_OVERRIDES.get(str(slug).strip().lower(), {}))
    policy_override = normalize_competition_eval_override(fallback_overrides)
    if policy_override:
        override.update(policy_override)
    return override


def competition_profile_override(slug: str) -> dict[str, object]:
    raw = COMPETITION_EVAL_OVERRIDES.get(str(slug).strip().lower(), {})
    if not isinstance(raw, dict):
        return {}
    return {key: value for key, value in raw.items() if key in COMPETITION_PROFILE_OVERRIDE_KEYS}


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
    if override.get("metric_name"):
        updated["metric"] = override["metric_name"]
    if override.get("direction"):
        updated["direction"] = override["direction"]
    if override.get("split_strategy"):
        updated["split_strategy_hint"] = override["split_strategy"]
    if override.get("group_column_hint"):
        updated["group_column_hint"] = override["group_column_hint"]
    updated.update(competition_profile_override(slug))
    if include_spec_keys:
        if override.get("metric_name"):
            updated["metric_name"] = override["metric_name"]
        if override.get("split_strategy"):
            updated["split_strategy"] = override["split_strategy"]
    return updated


def build_evaluation_contract(
    *,
    slug: str,
    eval_spec: dict[str, object],
    dataset_profile: dict[str, object] | None = None,
    competition_override: dict[str, str] | None = None,
    target_metric: str | None,
    target_direction: str | None,
    split_strategy: str | None,
) -> dict[str, object]:
    faithfulness = eval_spec.get("faithfulness") if isinstance(eval_spec.get("faithfulness"), dict) else {}
    accepted_score_sources = normalize_score_source_list(
        faithfulness.get("accepted_score_sources") if isinstance(faithfulness, dict) else None
    )
    require_full_dataset_default = resolve_require_full_dataset_default(
        slug=slug,
        dataset_profile=dataset_profile,
    )
    competition_override = competition_override if competition_override is not None else competition_eval_override(slug)
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


def resolve_require_full_dataset_default(
    *,
    slug: str,
    dataset_profile: dict[str, object] | None = None,
) -> bool:
    if isinstance(dataset_profile, dict):
        for key in ("require_full_dataset", "full_dataset_required"):
            value = dataset_profile.get(key)
            if isinstance(value, bool):
                return value
        data_root_layout = str(dataset_profile.get("data_root_layout") or "").strip().lower()
        if data_root_layout in FULL_DATASET_REQUIRED_LAYOUTS:
            return True
        data_resolution = dataset_profile.get("data_resolution")
        if isinstance(data_resolution, dict):
            for key in ("require_full_dataset", "full_dataset_required"):
                value = data_resolution.get(key)
                if isinstance(value, bool):
                    return value
            data_root_layout = str(data_resolution.get("data_root_layout") or "").strip().lower()
            if data_root_layout in FULL_DATASET_REQUIRED_LAYOUTS:
                return True
    return str(slug).strip().lower() in FULL_DATASET_REQUIRED_COMPETITIONS


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
    for name, dtype in dtype_map_raw.items():
        column_name = str(name)
        dtype_name = str(dtype).lower()
        if "datetime" in dtype_name or "timedelta" in dtype_name:
            return True
        if _column_name_has_temporal_token(column_name):
            return True
    return False


def _column_name_has_temporal_token(name: str) -> bool:
    tokens = [token for token in re.split(r"[^A-Za-z0-9]+", name.lower()) if token]
    if any(
        token in {"date", "datetime", "timestamp", "time", "day", "daynum", "week", "month", "year"} for token in tokens
    ):
        return True
    compact = "".join(tokens)
    return compact in {"dateblocknum", "daynum", "weekofyear"}


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

    plan_payload = load_json_object_or_empty(paths.plan_path)
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

    profile = load_json_object_or_empty(paths.dataset_profile_path)
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
