from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from kagglebot import (
    competition_rules,
    context_artifacts,
    env_utils,
    plan_policy,
    runtime_policy,
    submission_policy,
    submit_notebook,
)
from kagglebot.competition_policy import load_competition_policy
from kagglebot.paths import CompetitionPaths
from kagglebot.types import PlanConfig
from kagglebot.writeup import (
    infer_code_competition_from_paths,
    infer_deliverable_mode_from_paths,
    infer_submit_mode_from_paths,
)


class AutopilotPlanResolutionConfig(Protocol):
    paths: CompetitionPaths
    compute: str
    target_metric: str | None
    target_score: float | None
    target_direction: str | None
    score_source: str | None
    holdout_frac: float | None
    cv_folds: int | None
    seed: int
    time_budget_min: int | None
    kernel_name: str | None
    internet: str
    max_iterations: int | None
    max_total_min: int | None
    patience: int
    min_improvement: float
    submit_policy: str


@dataclass(frozen=True)
class AutopilotPlanResolutionDefaults:
    strict_competition_metric: bool
    target_medal: str
    limited_submission_gate: str
    max_iterations: int
    heavy_local_gpu_max_cv_folds: int
    long_local_gpu_iteration_budget_min: int
    long_local_gpu_max_iterations: int
    force_major_rank_max_percentile: float
    force_major_rank_min_teams: int


def resolve_plan_for_autopilot_config(
    *,
    plan: PlanConfig,
    config: AutopilotPlanResolutionConfig,
    defaults: AutopilotPlanResolutionDefaults,
    on_message: Callable[[str], None],
) -> dict[str, object]:
    return resolve_plan_for_autopilot(
        plan=plan,
        paths=config.paths,
        compute=config.compute,
        target_metric=config.target_metric,
        target_score=config.target_score,
        target_direction=config.target_direction,
        score_source=config.score_source,
        holdout_frac=config.holdout_frac,
        cv_folds=config.cv_folds,
        seed=config.seed,
        time_budget_min=config.time_budget_min,
        kernel_name=config.kernel_name,
        internet=config.internet,
        max_iterations=config.max_iterations,
        max_total_min=config.max_total_min,
        patience=config.patience,
        min_improvement=config.min_improvement,
        submit_policy=config.submit_policy,
        default_strict_competition_metric=defaults.strict_competition_metric,
        default_target_medal=defaults.target_medal,
        default_limited_submission_gate=defaults.limited_submission_gate,
        default_max_iterations=defaults.max_iterations,
        heavy_local_gpu_max_cv_folds=defaults.heavy_local_gpu_max_cv_folds,
        long_local_gpu_iteration_budget_min=defaults.long_local_gpu_iteration_budget_min,
        long_local_gpu_max_iterations=defaults.long_local_gpu_max_iterations,
        default_force_major_rank_max_percentile=defaults.force_major_rank_max_percentile,
        default_force_major_rank_min_teams=defaults.force_major_rank_min_teams,
        on_message=on_message,
    )


def resolve_plan_for_autopilot(
    *,
    plan: PlanConfig,
    paths: CompetitionPaths,
    compute: str,
    target_metric: str | None,
    target_score: float | None,
    target_direction: str | None,
    score_source: str | None,
    holdout_frac: float | None,
    cv_folds: int | None,
    seed: int,
    time_budget_min: int | None,
    kernel_name: str | None,
    internet: str,
    max_iterations: int | None,
    max_total_min: int | None,
    patience: int,
    min_improvement: float,
    submit_policy: str,
    default_strict_competition_metric: bool,
    default_target_medal: str,
    default_limited_submission_gate: str,
    default_max_iterations: int,
    heavy_local_gpu_max_cv_folds: int,
    long_local_gpu_iteration_budget_min: int,
    long_local_gpu_max_iterations: int,
    default_force_major_rank_max_percentile: float,
    default_force_major_rank_min_teams: int,
    on_message: Callable[[str], None],
) -> dict[str, object]:
    eval_spec = context_artifacts.load_evaluation_spec(
        slug=paths.slug,
        evaluation_spec_path=paths.context_dir / "evaluation_spec.json",
    )
    spec_values = plan_policy.extract_evaluation_spec_values(eval_spec)

    strict_competition_metric = env_utils.env_flag(
        "KAGGLEBOT_STRICT_COMPETITION_METRIC",
        default=default_strict_competition_metric,
    )
    deliverable_mode = plan_policy.resolve_deliverable_mode(
        plan_value=getattr(plan, "deliverable_mode", None),
        spec_value=eval_spec.get("deliverable_mode"),
        inferred_value=infer_deliverable_mode_from_paths(paths, default=""),
    )
    submit_mode = plan_policy.resolve_submit_mode(
        plan_value=getattr(plan, "submit_mode", None),
        spec_value=eval_spec.get("submit_mode"),
        inferred_value=infer_submit_mode_from_paths(paths, default=""),
    )
    competition_policy = load_competition_policy(paths)
    target_objective = plan_policy.resolve_target_objective(
        plan_target_medal=getattr(plan, "target_medal", None),
        plan_target_rank_percentile=getattr(plan, "target_rank_percentile", None),
        spec_target_medal=eval_spec.get("target_medal"),
        spec_target_rank_percentile=eval_spec.get("target_rank_percentile"),
        deliverable_mode=deliverable_mode,
        search_stop_rank_percentile=competition_policy.evaluation.search_stop_rank_percentile,
        default_target_medal=default_target_medal,
    )
    target_request = plan_policy.resolve_target_request(
        config_target_metric=target_metric,
        config_target_score=target_score,
        config_target_direction=target_direction,
        plan=plan,
        spec_values=spec_values,
    )
    resolved_target_metric = target_request.target_metric
    resolved_target_score = target_request.target_score
    resolved_target_direction = target_request.target_direction
    competition_override = plan_policy.competition_eval_override(
        paths.slug,
        fallback_overrides=competition_policy.evaluation.fallback_overrides,
    )
    metric_direction_decision = plan_policy.resolve_target_metric_direction(
        target_metric=resolved_target_metric,
        target_direction=resolved_target_direction,
        spec_metric=spec_values.metric_name,
        spec_direction=spec_values.direction,
        explicit_target_metric=target_request.explicit_target_metric,
        explicit_target_direction=target_request.explicit_target_direction,
        strict_competition_metric=strict_competition_metric,
        competition_override=competition_override,
    )
    resolved_target_metric = metric_direction_decision.target_metric
    resolved_target_direction = metric_direction_decision.target_direction
    override_split_strategy = metric_direction_decision.override_split_strategy
    _emit_messages(metric_direction_decision.messages, on_message=on_message)

    base_evaluation_request = plan_policy.resolve_base_evaluation_request(
        config_score_source=score_source,
        config_holdout_frac=holdout_frac,
        config_cv_folds=cv_folds,
        config_seed=seed,
        plan=plan,
        spec_values=spec_values,
    )
    resolved_score_source = base_evaluation_request.score_source
    _emit_messages(base_evaluation_request.messages, on_message=on_message)
    resolved_holdout_frac = base_evaluation_request.holdout_frac
    resolved_cv_folds = base_evaluation_request.cv_folds
    split_strategy = base_evaluation_request.split_strategy
    split_strategy, split_strategy_note = plan_policy.resolve_split_strategy_from_artifacts(
        paths=paths,
        split_strategy=split_strategy,
    )
    split_override_decision = plan_policy.resolve_split_strategy_override(
        split_strategy=split_strategy,
        override_split_strategy=override_split_strategy,
    )
    split_strategy = split_override_decision.split_strategy
    _emit_messages(split_override_decision.messages, on_message=on_message)
    if split_strategy_note:
        on_message(f"[yellow]note[/yellow]: {split_strategy_note}")

    dataset_profile = context_artifacts.load_dataset_profile(
        slug=paths.slug,
        dataset_profile_path=paths.dataset_profile_path,
    )
    profile_modality = str(dataset_profile.get("modality") or "").strip().lower()
    heavy_local_gpu = runtime_policy.is_local_gpu_compute(compute) and runtime_policy.is_heavy_deep_learning_modality(
        profile_modality
    )
    resolved_seed = base_evaluation_request.seed
    eval_seeds = base_evaluation_request.eval_seeds
    eval_repeats = base_evaluation_request.eval_repeats
    eval_budget_decision = plan_policy.resolve_eval_budget_policy(
        heavy_local_gpu=heavy_local_gpu,
        cv_folds=resolved_cv_folds,
        seed=resolved_seed,
        eval_seeds=eval_seeds,
        eval_repeats=eval_repeats,
        max_heavy_local_gpu_cv_folds=heavy_local_gpu_max_cv_folds,
    )
    resolved_cv_folds = eval_budget_decision.cv_folds
    eval_seeds = eval_budget_decision.eval_seeds
    eval_repeats = eval_budget_decision.eval_repeats
    _emit_messages(eval_budget_decision.messages, on_message=on_message)

    constraints = competition_rules.load_competition_rule_constraints(paths)
    code_competition = infer_code_competition_from_paths(paths)
    submit_mode_decision = plan_policy.resolve_submit_mode_constraints(
        submit_mode=submit_mode,
        compute=compute,
        code_competition=code_competition,
        notebook_submissions_only=constraints.notebook_submissions_only,
    )
    submit_mode = submit_mode_decision.submit_mode
    _emit_messages(submit_mode_decision.messages, on_message=on_message)
    notebook_submit_artifact_mode = submit_notebook.resolve_notebook_submit_artifact_mode(
        submit_mode=submit_mode,
        code_competition=code_competition,
    )

    runtime_request = plan_policy.resolve_runtime_request(
        config_time_budget_min=time_budget_min,
        config_kernel_name=kernel_name,
        config_internet=internet,
        plan=plan,
        internet_must_be_off=constraints.internet_must_be_off,
    )
    resolved_time_budget_min = runtime_request.time_budget_min
    resolved_kernel_name = runtime_request.kernel_name
    resolved_internet = runtime_request.internet
    _emit_messages(runtime_request.messages, on_message=on_message)
    runtime_limit_min = competition_rules.runtime_limit_for_compute(constraints=constraints, compute=compute)
    is_local_gpu_compute = runtime_policy.is_local_gpu_compute(compute)
    time_budget_decision = plan_policy.resolve_time_budget_policy(
        time_budget_min=resolved_time_budget_min,
        runtime_limit_min=runtime_limit_min,
        local_budget_min=runtime_policy.local_gpu_time_budget_limit_min() if is_local_gpu_compute else None,
        is_local_gpu=is_local_gpu_compute,
    )
    resolved_time_budget_min = time_budget_decision.time_budget_min
    _emit_messages(time_budget_decision.messages, on_message=on_message)

    max_iterations_decision = plan_policy.resolve_plan_max_iterations(
        config_max_iterations=max_iterations,
        plan_max_iterations=plan.max_iterations,
        default_max_iterations=default_max_iterations,
    )
    resolved_max_iterations = max_iterations_decision.max_iterations
    _emit_messages(max_iterations_decision.messages, on_message=on_message)
    max_iterations_decision = plan_policy.resolve_heavy_local_gpu_max_iterations(
        heavy_local_gpu=heavy_local_gpu,
        time_budget_min=resolved_time_budget_min,
        max_iterations=resolved_max_iterations,
        long_iteration_budget_min=long_local_gpu_iteration_budget_min,
        max_long_iterations=long_local_gpu_max_iterations,
    )
    resolved_max_iterations = max_iterations_decision.max_iterations
    _emit_messages(max_iterations_decision.messages, on_message=on_message)

    loop_control_request = plan_policy.resolve_loop_control_request(
        config_max_total_min=max_total_min,
        config_patience=patience,
        config_min_improvement=min_improvement,
        config_submit_policy=submit_policy,
        plan=plan,
        spec_values=spec_values,
    )
    submit_policy_decision = submission_policy.resolve_plan_submission_policy(
        config_submit_policy=submit_policy,
        requested_submit_policy=loop_control_request.requested_submit_policy,
        requested_submission_gate=loop_control_request.requested_submission_gate,
        submission_limit_detected=constraints.submission_limit_detected,
        default_limited_submission_gate=default_limited_submission_gate,
    )
    resolved_submit_policy = submit_policy_decision.submit_policy
    submission_gate = submit_policy_decision.submission_gate
    _emit_messages(submit_policy_decision.messages, on_message=on_message)

    readiness_stop_policy = plan_policy.resolve_readiness_stop_policy(
        plan=plan,
        spec_values=spec_values,
        target_score=resolved_target_score,
        min_improvement=loop_control_request.min_improvement,
        patience=loop_control_request.patience,
    )
    rank_force_policy = plan_policy.resolve_rank_force_policy(
        rank_force_major_max_percentile=plan.rank_force_major_max_percentile,
        rank_force_major_min_teams=plan.rank_force_major_min_teams,
        target_rank_percentile=target_objective.target_rank_percentile,
        default_max_percentile=default_force_major_rank_max_percentile,
        default_min_teams=default_force_major_rank_min_teams,
    )
    evaluation_contract = plan_policy.build_evaluation_contract(
        slug=paths.slug,
        eval_spec=eval_spec,
        dataset_profile=dataset_profile,
        competition_override=competition_override,
        target_metric=str(resolved_target_metric) if isinstance(resolved_target_metric, str) else None,
        target_direction=str(resolved_target_direction) if isinstance(resolved_target_direction, str) else None,
        split_strategy=str(split_strategy) if isinstance(split_strategy, str) else None,
    )

    return plan_policy.ResolvedPlan(
        deliverable_mode=deliverable_mode,
        submit_mode=submit_mode,
        code_competition=code_competition,
        notebook_submit_artifact_mode=notebook_submit_artifact_mode,
        target_medal=target_objective.target_medal,
        target_rank_percentile=target_objective.target_rank_percentile,
        target_metric=resolved_target_metric,
        target_score=resolved_target_score,
        target_direction=resolved_target_direction,
        score_source=resolved_score_source,
        holdout_frac=resolved_holdout_frac,
        cv_folds=resolved_cv_folds,
        split_strategy=split_strategy,
        seed=resolved_seed,
        eval_seeds=eval_seeds,
        eval_repeats=eval_repeats,
        time_budget_min=resolved_time_budget_min,
        kernel_name=resolved_kernel_name,
        internet=resolved_internet,
        max_iterations=resolved_max_iterations,
        max_total_min=loop_control_request.max_total_min,
        patience=loop_control_request.patience,
        min_improvement=loop_control_request.min_improvement,
        submit_policy=resolved_submit_policy,
        submission_gate=submission_gate,
        submission_limit_per_day=constraints.submission_limit_per_day,
        readiness_target_score=readiness_stop_policy.readiness_target_score,
        readiness_method=readiness_stop_policy.readiness_method,
        readiness_k=readiness_stop_policy.readiness_k,
        ci_method=readiness_stop_policy.ci_method,
        ci_alpha=readiness_stop_policy.ci_alpha,
        drift_check=readiness_stop_policy.drift_check,
        drift_weight=readiness_stop_policy.drift_weight,
        stop_min_delta=readiness_stop_policy.stop_min_delta,
        stop_no_improve_patience=readiness_stop_policy.stop_no_improve_patience,
        stop_same_config_patience=readiness_stop_policy.stop_same_config_patience,
        rank_force_major_max_percentile=rank_force_policy.rank_force_major_max_percentile,
        rank_force_major_min_teams=rank_force_policy.rank_force_major_min_teams,
        evaluation_contract=evaluation_contract,
    ).to_payload()


def _emit_messages(messages: tuple[str, ...], *, on_message: Callable[[str], None]) -> None:
    for message in messages:
        on_message(message)
