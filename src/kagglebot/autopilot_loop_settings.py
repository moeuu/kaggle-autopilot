from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from kagglebot import env_utils as _env_utils
from kagglebot import method_scout as _method_scout
from kagglebot import plan_policy as _plan_policy
from kagglebot.campaign import TOP1_TARGET_RANK_PERCENTILE, normalize_campaign_mode
from kagglebot.experiment_graph import normalize_portfolio_execution
from kagglebot.medals import normalize_target_medal, normalize_target_rank_percentile
from kagglebot.solver.metrics import infer_direction
from kagglebot.top1_exhaustive import normalize_top1_submit_policy
from kagglebot.validation_lab import normalize_validation_lab_mode
from kagglebot.writeup import normalize_deliverable_mode, normalize_submit_mode


class AutopilotLoopSettingsConfig(Protocol):
    campaign_mode: str
    portfolio_execution: str | None
    validation_lab: str | None
    research_scout: str | None
    top1_submit_policy: str | None
    top1_exhaustive: bool
    submit: bool


@dataclass(frozen=True)
class AutopilotLoopSettings:
    target_metric: str
    target_score: float
    metric_direction: str
    deliverable_mode: str
    campaign_mode: str
    portfolio_execution: str
    validation_lab_mode: str
    research_scout_mode: str
    top1_submit_policy: str
    submit_mode: str
    writeup_mode: bool
    submit_enabled: bool
    strict_competition_metric: bool
    require_submit_improvement: bool
    submit_improved_only: bool
    force_major_on_no_improve: bool
    max_iterations: int
    holdout_frac: float
    cv_folds: int
    split_strategy: str | None
    seed: int
    eval_seeds: list[int]
    eval_repeats: int
    score_source: str
    max_total_min: float | None
    time_budget_min: int | None
    kernel_name: object
    enable_internet: bool
    submission_gate: str
    submission_limit_per_day: int | None
    evaluation_contract: dict[str, object] | None
    readiness_target: float
    readiness_method: str
    readiness_k: float
    ci_method: str
    ci_alpha: float
    target_medal: str | None
    target_rank_percentile: float | None
    drift_check_enabled: bool
    drift_weight: float
    stop_min_delta: float
    stop_no_improve_patience: int
    stop_same_config_patience: int
    rank_force_major_max_percentile: float | None
    rank_force_major_min_teams: int | None


@dataclass(frozen=True)
class AutopilotLoopSettingsDefaults:
    strict_competition_metric: bool
    require_submit_improvement: bool
    force_major_on_no_improve: bool
    force_major_rank_max_percentile: float
    force_major_rank_min_teams: int


def resolve_autopilot_loop_settings(
    *,
    config: AutopilotLoopSettingsConfig,
    resolved: dict[str, object],
    target_metric: str,
    target_score: float,
    defaults: AutopilotLoopSettingsDefaults,
) -> AutopilotLoopSettings:
    """Normalize resolved-plan values used by the iteration loop.

    The loop still passes the mutable resolved-plan payload to legacy artifact
    writers, so this helper preserves the existing normalized write-backs while
    returning a typed view for orchestration code.
    """
    metric_direction = infer_direction(target_metric, resolved["target_direction"])
    resolved["target_direction"] = metric_direction
    deliverable_mode = normalize_deliverable_mode(resolved.get("deliverable_mode"), default="leaderboard")
    resolved["deliverable_mode"] = deliverable_mode
    campaign_mode = normalize_campaign_mode(config.campaign_mode, deliverable_mode=deliverable_mode)
    portfolio_execution = normalize_portfolio_execution(config.portfolio_execution)
    validation_lab_mode = normalize_validation_lab_mode(config.validation_lab)
    research_scout_mode = _method_scout.normalize_research_scout_mode(config.research_scout)
    top1_submit_policy = normalize_top1_submit_policy(config.top1_submit_policy)
    resolved["campaign_mode"] = campaign_mode
    resolved["portfolio_execution"] = portfolio_execution
    resolved["validation_lab"] = validation_lab_mode
    resolved["research_scout"] = research_scout_mode
    resolved["top1_exhaustive"] = bool(config.top1_exhaustive)
    resolved["top1_submit_policy"] = top1_submit_policy
    submit_mode = normalize_submit_mode(resolved.get("submit_mode"), default="file")
    resolved["submit_mode"] = submit_mode
    writeup_mode = deliverable_mode == "writeup"
    submit_enabled = bool(config.submit and not writeup_mode)
    strict_competition_metric = _env_utils.env_flag(
        "KAGGLEBOT_STRICT_COMPETITION_METRIC",
        default=defaults.strict_competition_metric,
    )
    require_submit_improvement = _env_utils.env_flag(
        "KAGGLEBOT_REQUIRE_SUBMIT_IMPROVEMENT",
        default=defaults.require_submit_improvement,
    )
    submit_improved_only = str(resolved.get("submit_policy") or "").strip().lower() == "improved"
    force_major_on_no_improve = _env_utils.env_flag(
        "KAGGLEBOT_FORCE_MAJOR_ON_NO_IMPROVE",
        default=defaults.force_major_on_no_improve,
    )

    max_iterations = max(1, int(resolved["max_iterations"]))
    holdout_frac = float(resolved["holdout_frac"])
    cv_folds = int(resolved["cv_folds"])
    split_strategy = str(resolved.get("split_strategy") or "").strip().lower() or None
    seed = int(resolved["seed"])
    eval_seeds = _plan_policy.normalize_default_eval_seeds(resolved.get("eval_seeds"), fallback=[seed])
    eval_repeats = _plan_policy.normalize_default_eval_repeats(
        resolved.get("eval_repeats"), fallback=_plan_policy.DEFAULT_EVAL_REPEATS
    )
    score_source = str(resolved["score_source"] or "cv")
    max_total_min_raw = resolved.get("max_total_min")
    max_total_min = float(max_total_min_raw) if isinstance(max_total_min_raw, (int, float)) else None
    time_budget_min_raw = resolved.get("time_budget_min")
    time_budget_min = int(time_budget_min_raw) if isinstance(time_budget_min_raw, (int, float)) else None
    kernel_name = resolved["kernel_name"]
    enable_internet = str(resolved["internet"]) == "on"
    submission_gate = str(resolved.get("submission_gate") or "always")
    submission_limit_per_day_raw = resolved.get("submission_limit_per_day")
    submission_limit_per_day = (
        int(submission_limit_per_day_raw)
        if isinstance(submission_limit_per_day_raw, (int, float)) and int(submission_limit_per_day_raw) > 0
        else None
    )
    evaluation_contract = (
        resolved.get("evaluation_contract") if isinstance(resolved.get("evaluation_contract"), dict) else None
    )
    readiness_target = float(resolved.get("readiness_target_score") or target_score)
    readiness_method = str(resolved.get("readiness_method") or "ci_bound")
    readiness_k = float(resolved.get("readiness_k") or 1.0)
    ci_method = str(resolved.get("ci_method") or "normal")
    ci_alpha = float(resolved.get("ci_alpha") or 0.05)
    target_medal = normalize_target_medal(resolved.get("target_medal"), default=None)
    target_rank_percentile = normalize_target_rank_percentile(
        resolved.get("target_rank_percentile"),
        medal=target_medal,
        fallback=None,
    )
    if campaign_mode == "top1" and target_rank_percentile is None:
        target_rank_percentile = TOP1_TARGET_RANK_PERCENTILE
        resolved["target_rank_percentile"] = target_rank_percentile
    drift_check_enabled = bool(resolved.get("drift_check", False))
    drift_weight = float(resolved.get("drift_weight") or 1.0)
    stop_min_delta = float(resolved.get("stop_min_delta") or 0.0)
    stop_no_improve_patience = int(resolved.get("stop_no_improve_patience") or 0)
    stop_same_config_patience = int(resolved.get("stop_same_config_patience") or 0)
    rank_force_policy = _plan_policy.resolve_rank_force_policy(
        rank_force_major_max_percentile=resolved.get("rank_force_major_max_percentile"),
        rank_force_major_min_teams=resolved.get("rank_force_major_min_teams"),
        target_rank_percentile=target_rank_percentile,
        default_max_percentile=defaults.force_major_rank_max_percentile,
        default_min_teams=defaults.force_major_rank_min_teams,
    )

    return AutopilotLoopSettings(
        target_metric=target_metric,
        target_score=target_score,
        metric_direction=metric_direction,
        deliverable_mode=deliverable_mode,
        campaign_mode=campaign_mode,
        portfolio_execution=portfolio_execution,
        validation_lab_mode=validation_lab_mode,
        research_scout_mode=research_scout_mode,
        top1_submit_policy=top1_submit_policy,
        submit_mode=submit_mode,
        writeup_mode=writeup_mode,
        submit_enabled=submit_enabled,
        strict_competition_metric=strict_competition_metric,
        require_submit_improvement=require_submit_improvement,
        submit_improved_only=submit_improved_only,
        force_major_on_no_improve=force_major_on_no_improve,
        max_iterations=max_iterations,
        holdout_frac=holdout_frac,
        cv_folds=cv_folds,
        split_strategy=split_strategy,
        seed=seed,
        eval_seeds=eval_seeds,
        eval_repeats=eval_repeats,
        score_source=score_source,
        max_total_min=max_total_min,
        time_budget_min=time_budget_min,
        kernel_name=kernel_name,
        enable_internet=enable_internet,
        submission_gate=submission_gate,
        submission_limit_per_day=submission_limit_per_day,
        evaluation_contract=evaluation_contract,
        readiness_target=readiness_target,
        readiness_method=readiness_method,
        readiness_k=readiness_k,
        ci_method=ci_method,
        ci_alpha=ci_alpha,
        target_medal=target_medal,
        target_rank_percentile=target_rank_percentile,
        drift_check_enabled=drift_check_enabled,
        drift_weight=drift_weight,
        stop_min_delta=stop_min_delta,
        stop_no_improve_patience=stop_no_improve_patience,
        stop_same_config_patience=stop_same_config_patience,
        rank_force_major_max_percentile=rank_force_policy.rank_force_major_max_percentile,
        rank_force_major_min_teams=rank_force_policy.rank_force_major_min_teams,
    )
