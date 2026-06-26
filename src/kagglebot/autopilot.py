from __future__ import annotations

import json
import os
import shlex
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rich import print

from kagglebot import agent_io as _agent_io
from kagglebot import agent_prompts as _agent_prompts
from kagglebot import agent_strategy as _agent_strategy
from kagglebot import autofix_restart as _autofix_restart
from kagglebot import autopilot_state as _autopilot_state
from kagglebot import campaign_metrics as _campaign_metrics
from kagglebot import code_reference as _code_reference
from kagglebot import competition_rules as _competition_rules
from kagglebot import context_artifacts as _context_artifacts
from kagglebot import diagnostics as _diagnostics
from kagglebot import env_utils as _env_utils
from kagglebot import iteration_metrics as _iteration_metrics
from kagglebot import iteration_signals as _iteration_signals
from kagglebot import json_utils as _json_utils
from kagglebot import kaggle_cli_errors as _kaggle_cli_errors
from kagglebot import kernel_errors as _kernel_errors
from kagglebot import kernel_metrics as _kernel_metrics
from kagglebot import kernel_quality as _kernel_quality
from kagglebot import kernel_snapshot as _kernel_snapshot
from kagglebot import knowledge_context as _knowledge_context
from kagglebot import leaderboard_policy as _leaderboard_policy
from kagglebot import loop_control as _loop_control
from kagglebot import method_scout as _method_scout
from kagglebot import metric_matching as _metric_matching
from kagglebot import plan_policy as _plan_policy
from kagglebot import plan_resolution as _plan_resolution
from kagglebot import runtime_fixes as _runtime_fixes
from kagglebot import score_progress as _score_progress
from kagglebot import score_utils as _score_utils
from kagglebot import submission_history as _submission_history
from kagglebot import submission_policy as _submission_policy
from kagglebot import submit_attempts as _submit_attempts
from kagglebot import submit_autofix as _submit_autofix
from kagglebot import submit_failure_context as _submit_failure_context
from kagglebot import submit_failure_policy as _submit_failure_policy
from kagglebot import submit_notebook as _submit_notebook
from kagglebot import submit_retry_policy as _submit_retry_policy
from kagglebot import submit_stage as _submit_stage
from kagglebot import verify_artifacts as _verify_artifacts
from kagglebot import watch_state as _watch_state
from kagglebot.agents.codex_runner import run_codex
from kagglebot.agents.identity import (
    IMPLEMENTATION_AGENT,
    STRATEGY_AGENT,
    planning_flow_summary,
    prompt_identity_format_args,
    render_prompt_identity,
)
from kagglebot.agents.strategy_runner import run_strategy
from kagglebot.campaign import (
    TOP1_TARGET_RANK_PERCENTILE,
    allocate_submission,
    build_campaign_candidate,
    campaign_state_path,
    candidate_registry_path,
    normalize_campaign_mode,
    update_campaign_state,
    upsert_candidate,
)
from kagglebot.competition_policy import load_competition_policy
from kagglebot.exceptions import (
    DuplicateSubmissionError,
    KaggleCliError,
    KaggleNetworkError,
    KernelCapacityError,
    KernelFailedError,
    KernelStillRunningError,
    RulesNotAcceptedError,
    SubmissionCliError,
    SubmissionRateLimitError,
    SubmissionValidationError,
    SubmitAbortedError,
)
from kagglebot.exec_utils import run_command
from kagglebot.experiment_executor import execute_experiment_graph
from kagglebot.experiment_graph import (
    append_campaign_outcome,
    build_experiment_graph,
    normalize_portfolio_execution,
    write_allocator_decision,
)
from kagglebot.hardware import render_hardware_constraints, resolve_hardware_profile
from kagglebot.hashing import sha256_file_or_none as _sha256_or_none
from kagglebot.history import new_run_id
from kagglebot.kaggle_api import (
    check_rules_accepted,
    leaderboard_rank_for_score,
    leaderboard_top1,
    list_competition_submissions,
)
from kagglebot.kernel_runner import resolve_kaggle_username, run_kernel, run_kernel_local, run_submit_kernel
from kagglebot.knowledge import (
    record_error_fix_insight,
    record_improvement,
    record_iteration,
    record_problem_type_insight,
    record_run,
)
from kagglebot.medals import (
    DEFAULT_TARGET_MEDAL,
    normalize_target_medal,
    normalize_target_rank_percentile,
)
from kagglebot.orchestrator.agent_pipeline import (
    AgentPipelineConfig,
    run_agent_pipeline,
)
from kagglebot.runners.base import RunContext
from kagglebot.runners.local_kernel import LocalKernelRunner
from kagglebot.scalar_utils import tolerant_finite_float, tolerant_int
from kagglebot.solver.metrics import infer_direction
from kagglebot.submission.guard import (
    classify_submit_error,
    compute_error_fingerprint,
    normalize_error_text,
    run_kaggle_submit_kernel,
)
from kagglebot.submission_service import SubmissionConfig, SubmissionService
from kagglebot.top1_campaign import (
    build_blend_report,
    build_candidate_portfolio_plan,
    build_reference_reproduction_report,
    private_robustness_score,
    select_method_id_for_category,
)
from kagglebot.top1_exhaustive import (
    build_portfolio_optimizer_report,
    build_private_robustness_report,
    build_top1_exhaustion_report,
    build_win_contract,
    format_top1_public_score_message,
    normalize_top1_submit_policy,
)
from kagglebot.types import PlanConfig
from kagglebot.validation_lab import normalize_validation_lab_mode, run_validation_lab
from kagglebot.validators import kernel_source_preflight_error
from kagglebot.write_guard import (
    _backup_guarded_files,
    _diff_snapshots,
    _enforce_allowlist_changes,
    _snapshot_tree,
    build_repair_write_policy,
)
from kagglebot.writeup import (
    build_writeup_bundle,
    infer_code_competition_from_paths,
    infer_deliverable_mode_from_paths,
    normalize_deliverable_mode,
    normalize_submit_mode,
)

if TYPE_CHECKING:
    from kagglebot.paths import CompetitionPaths, KnowledgePaths
    from kagglebot.solver.evaluate import EvaluationResult


@dataclass(frozen=True)
class AutopilotConfig:
    run_id: str | None
    slug: str
    competition_url: str | None
    paths: CompetitionPaths
    knowledge_paths: KnowledgePaths
    agent: str
    compute: str
    accelerator: str
    strict_accelerator: bool
    kaggle_username: str | None
    kernel_name: str | None
    internet: str | None
    time_budget_min: int | None
    seed: int | None
    score_source: str | None
    holdout_frac: float | None
    cv_folds: int | None
    target_metric: str | None
    target_score: float | None
    target_direction: str | None
    max_iterations: int | None
    max_total_min: int | None
    patience: int | None
    min_improvement: float | None
    submit: bool
    force_submit: bool
    message: str | None
    verify_cmd: str
    dry_run: bool
    submit_policy: str | None = None
    campaign_mode: str | None = "baseline"
    method_scout: str | None = "auto"
    research_scout: str | None = "auto"
    method_scout_max_sources: int = 12
    portfolio_execution: str | None = "serial"
    validation_lab: str | None = "auto"
    candidate_budget_min: int | None = None
    max_candidates_per_iteration: int | None = None
    top1_exhaustive: bool = False
    top1_submit_policy: str | None = "value_only"
    hardware_profile: str | None = "auto"


MAX_KERNEL_FIX_ATTEMPTS: int | None = 8
MAX_KERNEL_CAPACITY_RETRIES = 3
KERNEL_CAPACITY_RETRY_SLEEP = 30.0
MAX_KERNEL_CAPACITY_REPEAT = 6
MAX_KERNEL_REGISTRATION_RETRIES = 2
KERNEL_REGISTRATION_RETRY_SLEEP = 15.0
KERNEL_STILL_RUNNING_RETRY_SLEEP = 60.0
MAX_AUTOFIX_ATTEMPTS = 2
MAX_AUTOFIX_RESTARTS = 1
MAX_AUTOFIX_CODEX_PASSES = 3
MAX_KERNEL_FIX_CODEX_PASSES = 3
MAX_AGENT_CAPACITY_ATTEMPTS = 3
AGENT_CAPACITY_RETRY_SLEEP = 5.0
_ERROR_FIX_CODEX_MODEL = IMPLEMENTATION_AGENT.model
_ERROR_FIX_REASONING_EFFORT = IMPLEMENTATION_AGENT.reasoning_effort
_ERROR_STRATEGY_MODEL = STRATEGY_AGENT.model
_ERROR_STRATEGY_REASONING_EFFORT = STRATEGY_AGENT.reasoning_effort
_METRIC_FIX_CODEX_MODEL = IMPLEMENTATION_AGENT.model
_METRIC_FIX_REASONING_EFFORT = IMPLEMENTATION_AGENT.reasoning_effort
_MAX_METRIC_FIX_ATTEMPTS = 3
_MAX_METRIC_FIX_CODEX_PASSES = 4
_SUBMISSION_POLL_MAX_ATTEMPTS: int | None = None
_SUBMISSION_POLL_INTERVAL_SEC = 30.0
_SUBMISSION_POLL_MAX_FETCH_ERRORS = 3
_FORCED_INITIAL_SUBMIT_REASON = "initial_submit_contract_probe"
_SPARE_DAILY_SUBMIT_REASON = "spare_daily_submission_slot"
_SUBMIT_FAILED_DEFERRED_STATE = "submit_failed_deferred"
_SUBMIT_MAX_TRANSIENT_RETRIES = 3
_SUBMIT_BACKOFF_BASE_SEC = 2.0
_SUBMIT_STDERR_TAIL_CHARS = 1200
_SUBMIT_STDOUT_TAIL_CHARS = 1200
_DEFAULT_EVAL_SEEDS = list(_plan_policy.DEFAULT_EVAL_SEEDS)
_DEFAULT_MAX_ITERATIONS = 5
_LONG_LOCAL_GPU_ITERATION_BUDGET_MIN = 12 * 60
_LONG_LOCAL_GPU_MAX_ITERATIONS = 3
_HEAVY_LOCAL_GPU_MAX_CV_FOLDS = 3
_DEFAULT_FORCE_MAJOR_RANK_MAX_PERCENTILE = 0.35
_DEFAULT_FORCE_MAJOR_RANK_MIN_TEAMS = 200
_DEFAULT_TARGET_MEDAL = DEFAULT_TARGET_MEDAL
_DEFAULT_LIMITED_SUBMISSION_GATE = "readiness_or_final"
_DEFAULT_STRICT_COMPETITION_METRIC = True
_DEFAULT_REQUIRE_SUBMIT_IMPROVEMENT = True
_DEFAULT_FORCE_MAJOR_ON_NO_IMPROVE = True
_KERNEL_REGENERATE_MARKER_FILENAME = "kernel_regenerated_once.json"
_MAX_KERNEL_PREFLIGHT_FIX_ATTEMPTS = 2


def run_autopilot(config: AutopilotConfig) -> None:
    resume_id = os.environ.get("KAGGLEBOT_RESUME_RUN_ID")
    resume_slug = os.environ.get("KAGGLEBOT_RESUME_SLUG")
    resume_run = bool(config.run_id is None and resume_id and resume_slug == config.slug)
    if resume_run:
        run_id = resume_id
    else:
        run_id = config.run_id or new_run_id()
    if resume_id:
        os.environ.pop("KAGGLEBOT_RESUME_RUN_ID", None)
        os.environ.pop("KAGGLEBOT_RESUME_SLUG", None)
    resume_after_failure = resume_run
    attempt = 0
    submit_force_override = False
    try:
        while True:
            session = AutopilotSession(config=config, run_id=run_id, resume_run=resume_after_failure)
            try:
                return session.run()
            except RulesNotAcceptedError:
                raise
            except SubmitAbortedError as exc:
                if config.dry_run:
                    raise
                run_dir = config.paths.run_dir(run_id)
                submit_abort_autofix = _submit_failure_context.resolve_submit_abort_autofixability_for_run(
                    run_dir=run_dir,
                    load_run_state=_autopilot_state._load_run_state,
                )
                if submit_abort_autofix.message:
                    print(submit_abort_autofix.message)
                if not submit_abort_autofix.autofixable:
                    raise
                attempt += 1
                if attempt > MAX_AUTOFIX_ATTEMPTS:
                    raise
                print(
                    f"[yellow]autofix[/yellow]: submit stage failed; invoking "
                    f"{IMPLEMENTATION_AGENT.log_alias} to repair and retry submit"
                )
                if (not _submit_attempts.has_successful_submit_attempt(run_dir)) or (
                    _submit_failure_context.should_force_resubmit_after_submit_abort(
                        _autopilot_state._load_run_state(run_dir)
                    )
                ):
                    os.environ["KAGGLEBOT_FORCE_RESUBMIT"] = "1"
                    submit_force_override = True
                _run_autofix(config=config, run_id=run_id, attempt=attempt, error=exc)
                resume_after_failure = True
            except KernelCapacityError:
                raise
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                if config.dry_run:
                    raise
                if _runtime_fixes.is_non_autofixable_runtime_error(exc):
                    raise
                attempt += 1
                if attempt > MAX_AUTOFIX_ATTEMPTS:
                    raise
                print(f"[yellow]autofix[/yellow]: invoking {IMPLEMENTATION_AGENT.log_alias} to repair error")
                _run_autofix(config=config, run_id=run_id, attempt=attempt, error=exc)
                resume_after_failure = True
    finally:
        if submit_force_override:
            os.environ.pop("KAGGLEBOT_FORCE_RESUBMIT", None)


@dataclass(frozen=True)
class PlanningPhase:
    config: AutopilotConfig
    run_id: str
    resume_run: bool

    def execute(self, plan: PlanConfig) -> PlanConfig:
        if _plan_policy.should_skip_planning_on_resume(
            resume_run=self.resume_run,
            plan_path=self.config.paths.plan_path,
            kernel_path=self.config.paths.kernel_source_dir / "kernel.py",
        ):
            print("[yellow]resume[/yellow]: skipping planning after restart; reusing existing plan")
            return plan
        if _plan_policy.needs_planning(
            agent=self.config.agent,
            config_target_metric=self.config.target_metric,
            config_target_score=self.config.target_score,
            config_target_direction=self.config.target_direction,
            plan_target_metric=plan.target_metric,
            plan_target_score=plan.target_score,
            plan_target_direction=plan.target_direction,
        ):
            print("[cyan]plan[/cyan]: generating initial plan")
            _watch_state.update_watch_phase(
                self.config,
                self.run_id,
                "gpt_planning",
                detail="GPT is drafting the initial competition plan.",
            )
            _run_plan_and_initial(self.config, self.run_id)
            return _plan_policy.load_plan_config(self.config.paths)
        return plan


@dataclass(frozen=True)
class KnowledgePhase:
    config: AutopilotConfig

    def refresh(self) -> None:
        _knowledge_context.refresh_knowledge_hints(paths=self.config.paths, knowledge_paths=self.config.knowledge_paths)

    def load_dataset_profile(self) -> dict[str, object]:
        return _context_artifacts.load_dataset_profile(
            slug=self.config.paths.slug,
            dataset_profile_path=self.config.paths.dataset_profile_path,
        )

    def derive_problem_types(self) -> list[str]:
        return _knowledge_context.resolve_problem_types_from_profile(
            dataset_profile_path=self.config.paths.dataset_profile_path
        )


@dataclass(frozen=True)
class SubmissionPhase:
    config: AutopilotConfig
    run_id: str
    problem_types: list[str]
    submit_mode: str
    notebook_submit_artifact_mode: str = "wrapper"

    def attempt(self, *, submission_path: Path, best_score: float | None) -> dict[str, object] | None:
        return _attempt_submit(
            config=self.config,
            run_id=self.run_id,
            submission_path=submission_path,
            best_score=best_score,
            problem_types=self.problem_types,
            submit_mode=self.submit_mode,
            notebook_submit_artifact_mode=self.notebook_submit_artifact_mode,
        )


@dataclass(frozen=True)
class AutopilotSession:
    config: AutopilotConfig
    run_id: str
    resume_run: bool = False

    @property
    def planning(self) -> PlanningPhase:
        return PlanningPhase(config=self.config, run_id=self.run_id, resume_run=self.resume_run)

    @property
    def knowledge(self) -> KnowledgePhase:
        return KnowledgePhase(config=self.config)

    def run(self) -> None:
        _run_autopilot_core(self.config, self.run_id, resume_run=self.resume_run)


def _run_autopilot_core(config: AutopilotConfig, run_id: str, *, resume_run: bool = False) -> None:
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    _watch_state.update_watch_phase(config, run_id, "autopilot_starting")
    print(f"[green]run started[/green]: {run_id}")
    planning_phase = PlanningPhase(config=config, run_id=run_id, resume_run=resume_run)
    knowledge_phase = KnowledgePhase(config=config)
    plan = _plan_policy.load_plan_config(config.paths)
    if not config.paths.plan_path.exists():
        _plan_policy.write_plan_config(config.paths, plan)

    _watch_state.update_watch_phase(config, run_id, "leaderboard_fetching")
    print(f"[cyan]fetching leaderboard[/cyan]: {config.slug}")
    metric_hint = config.target_metric or plan.target_metric
    top1_info = leaderboard_top1(
        config.slug,
        config.paths.context_dir,
        dry_run=config.dry_run,
        metric_hint=metric_hint,
    )
    _json_utils.write_json_object(config.paths.top1_public_path, top1_info)
    print(format_top1_public_score_message(top1_info))
    _watch_state.update_watch_phase(config, run_id, "knowledge_refreshing")
    knowledge_phase.refresh()
    plan = planning_phase.execute(plan)

    _watch_state.update_watch_phase(config, run_id, "resolving_plan")
    resolved = _resolve_plan(plan, config)
    target_metric = resolved["target_metric"]
    target_score = resolved["target_score"]
    if target_metric is None or target_score is None:
        run_payload = _autopilot_state._build_run_payload(
            run_id=run_id,
            config=config,
            resolved=resolved,
            status="missing_target",
        )
        _json_utils.write_json_object(run_dir / "run.json", run_payload)
        return

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
        default=_DEFAULT_STRICT_COMPETITION_METRIC,
    )
    require_submit_improvement = _env_utils.env_flag(
        "KAGGLEBOT_REQUIRE_SUBMIT_IMPROVEMENT",
        default=_DEFAULT_REQUIRE_SUBMIT_IMPROVEMENT,
    )
    submit_improved_only = str(resolved.get("submit_policy") or "").strip().lower() == "improved"
    force_major_on_no_improve = _env_utils.env_flag(
        "KAGGLEBOT_FORCE_MAJOR_ON_NO_IMPROVE",
        default=_DEFAULT_FORCE_MAJOR_ON_NO_IMPROVE,
    )

    _plan_policy.write_resolved_plan_config(
        config.paths,
        resolved,
        default_max_iterations=_DEFAULT_MAX_ITERATIONS,
        default_force_major_rank_max_percentile=_DEFAULT_FORCE_MAJOR_RANK_MAX_PERCENTILE,
        default_force_major_rank_min_teams=_DEFAULT_FORCE_MAJOR_RANK_MIN_TEAMS,
    )
    _watch_state.update_watch_phase(config, run_id, "initializing_iterations")
    run_payload = _autopilot_state._build_run_payload(
        run_id=run_id,
        config=config,
        resolved=resolved,
        status="running",
    )
    _json_utils.write_json_object(run_dir / "run.json", run_payload)
    _kernel_snapshot.ensure_best_kernel_snapshot(paths=config.paths, run_dir=run_dir)

    record_run(
        knowledge_paths=config.knowledge_paths,
        run_id=run_id,
        slug=config.slug,
        compute=config.compute,
        goal_metric=target_metric,
        goal_score=target_score,
        direction=metric_direction,
    )
    dataset_profile = knowledge_phase.load_dataset_profile()
    problem_types = knowledge_phase.derive_problem_types()
    submission_phase = (
        SubmissionPhase(
            config=config,
            run_id=run_id,
            problem_types=problem_types,
            submit_mode=submit_mode,
            notebook_submit_artifact_mode=str(resolved.get("notebook_submit_artifact_mode") or "wrapper"),
        )
        if submit_enabled
        else None
    )
    best_score = None
    best_submission: Path | None = None
    best_submittable_score: float | None = None
    best_submittable_submission: Path | None = None
    best_high_potential_score: float | None = None
    best_high_potential_submission: Path | None = None
    best_high_potential_iteration: int | None = None
    best_high_potential_meta: dict[str, object] | None = None
    submitted = False
    pending_problem_insights: list[dict[str, object]] = []
    pending_error_fixes: list[dict[str, object]] = []
    last_submission_result: dict[str, object] | None = None
    fallback_submit_blocked_reason: str | None = None
    writeup_bundle_meta: dict[str, object] | None = None

    max_iterations = max(1, int(resolved["max_iterations"]))
    iteration_phase = _score_progress.IterationPhase(metric_direction=metric_direction)
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
        default_max_percentile=_DEFAULT_FORCE_MAJOR_RANK_MAX_PERCENTILE,
        default_min_teams=_DEFAULT_FORCE_MAJOR_RANK_MIN_TEAMS,
    )
    rank_force_major_max_percentile = rank_force_policy.rank_force_major_max_percentile
    rank_force_major_min_teams = rank_force_policy.rank_force_major_min_teams
    no_improve_streak = 0
    frontier_no_improve_streak = 0
    same_config_streak = 0
    last_config_hash: str | None = None
    eval_data_cache: dict[str, object] | None = None
    previous_readiness_score, noise_limited_streak = _iteration_metrics.resume_noise_guard_state(
        run_dir=config.paths.run_dir(run_id),
        max_iterations=max_iterations,
    )
    start_iteration, best_score, best_submission = _autopilot_state._resume_iteration_state(
        paths=config.paths,
        run_id=run_id,
        metric_direction=metric_direction,
        target_metric=target_metric,
        max_iterations=max_iterations,
        require_submit_phase=submit_enabled and not config.dry_run,
        load_kernel_metrics=_kernel_metrics.load_kernel_metrics,
        infer_iteration_from_submission_path=_submit_stage.infer_iteration_from_submission_path,
    )
    best_submitted_score = _autopilot_state._resume_best_submitted_offline_score(
        paths=config.paths,
        run_id=run_id,
        metric_direction=metric_direction,
        target_metric=target_metric,
        max_iterations=max_iterations,
        load_kernel_metrics=_kernel_metrics.load_kernel_metrics,
    )
    best_online_submission_score = _leaderboard_policy.resume_best_online_submission_score(
        paths=config.paths,
        run_id=run_id,
        direction=metric_direction,
        max_iterations=max_iterations,
    )
    previous_submission_history = _submission_history.load_previous_submission_history(
        slug=config.slug,
        history_path=config.paths.context_dir / "submission_history.json",
        direction=metric_direction,
        dry_run=config.dry_run,
        fetch_submission_rows=lambda current_slug: list_competition_submissions(current_slug, dry_run=False),
        on_message=print,
    )
    historical_best_submission_score = tolerant_finite_float(previous_submission_history.get("best_score"))
    if historical_best_submission_score is not None:
        if _score_utils.should_update_best_score(
            best_submitted_score,
            historical_best_submission_score,
            metric_direction,
            0.0,
        ):
            best_submitted_score = historical_best_submission_score
        best_online_submission_score = _leaderboard_policy.update_best_online_submission_score(
            current_best_score=best_online_submission_score,
            candidate_score=historical_best_submission_score,
            direction=metric_direction,
        )
        print(
            "[cyan]submission history[/cyan]: "
            f"best public score={historical_best_submission_score:.6f} "
            f"source={previous_submission_history.get('source') or 'unknown'}"
        )
    campaign_state: dict[str, object] = {}
    campaign_state_file = campaign_state_path(config.paths.context_dir)
    campaign_registry_file = candidate_registry_path(config.paths.context_dir)
    if campaign_mode == "top1":
        campaign_state = update_campaign_state(
            state_path=campaign_state_file,
            registry_path=campaign_registry_file,
            slug=config.slug,
            run_id=run_id,
            mode=campaign_mode,
            direction=metric_direction,
            top1_info=top1_info if isinstance(top1_info, dict) else {},
            submission_history=previous_submission_history,
        )
        print(f"[cyan]campaign[/cyan]: top1 mode active; state={campaign_state_file}")
    effective_method_scout = _method_scout.effective_method_scout_mode(
        requested_mode=config.method_scout,
        campaign_mode=campaign_mode,
    )
    method_registry: dict[str, object] = {}
    source_registry: dict[str, object] = {}
    validation_lab_report: dict[str, object] | None = None
    win_contract: dict[str, object] | None = None
    if effective_method_scout != "off":
        method_registry = _method_scout.run_method_scout(
            paths=config.paths,
            slug=config.slug,
            problem_types=problem_types,
            dataset_profile=dataset_profile,
            metric=target_metric,
            campaign_state=campaign_state,
            mode=effective_method_scout,
            research_mode=research_scout_mode,
            max_sources=int(config.method_scout_max_sources or 12),
        )
        source_registry = _json_utils.load_json_object_or_empty(config.paths.source_registry_path)
        if campaign_mode == "top1":
            campaign_state = update_campaign_state(
                state_path=campaign_state_file,
                registry_path=campaign_registry_file,
                slug=config.slug,
                run_id=run_id,
                mode=campaign_mode,
                direction=metric_direction,
                top1_info=top1_info if isinstance(top1_info, dict) else {},
                submission_history=previous_submission_history,
                method_registry=method_registry,
            )
            validation_registry = _json_utils.load_json_object_or_empty(config.paths.validation_registry_path)
            validation_lab_report = run_validation_lab(
                context_dir=config.paths.context_dir,
                validation_registry=validation_registry,
                candidate_registry_path=campaign_registry_file,
                campaign_state=campaign_state,
                mode=validation_lab_mode,
            )
            if isinstance(validation_lab_report.get("registry"), dict):
                method_registry["active_validation_profile"] = validation_lab_report["registry"].get("active_profile")
        print(f"[cyan]method scout[/cyan]: {config.paths.method_registry_path}")
    elif campaign_mode == "top1":
        method_registry = _json_utils.load_json_object_or_empty(config.paths.method_registry_path)
        source_registry = _json_utils.load_json_object_or_empty(config.paths.source_registry_path)
    if campaign_mode == "top1":
        validation_registry_for_contract = _json_utils.load_json_object_or_empty(config.paths.validation_registry_path)
        win_contract = build_win_contract(
            context_dir=config.paths.context_dir,
            slug=config.slug,
            direction=metric_direction,
            campaign_state=campaign_state,
            top1_info=top1_info if isinstance(top1_info, dict) else {},
            submission_history=previous_submission_history,
            method_registry=method_registry,
            source_registry=source_registry,
            validation_registry=validation_registry_for_contract,
            submission_limit_per_day=submission_limit_per_day,
        )
    resumed_best_readiness = _iteration_metrics.resume_best_readiness_score(
        run_dir=config.paths.run_dir(run_id),
        direction=metric_direction,
        max_iterations=max_iterations,
    )
    if resumed_best_readiness is not None and best_score is None:
        best_score = resumed_best_readiness
    best_submittable_score, best_submittable_submission = _autopilot_state._resume_best_submittable_iteration_state(
        paths=config.paths,
        run_id=run_id,
        metric_direction=metric_direction,
        target_metric=target_metric,
        max_iterations=max_iterations,
        load_kernel_metrics=_kernel_metrics.load_kernel_metrics,
        iteration_metrics_allow_submit=_iteration_metrics.iteration_metrics_allow_submit,
    )
    if start_iteration > 1:
        print(f"[yellow]resume[/yellow]: found completed iterations; resuming at {start_iteration}/{max_iterations}")
    loop_started_at = time.monotonic()
    last_completed_iteration = start_iteration - 1

    try:
        for iteration in range(start_iteration, max_iterations + 1):
            _watch_state.update_watch_phase(config, run_id, "iteration_starting", iteration=iteration)
            last_completed_iteration = iteration
            elapsed_total_min = (time.monotonic() - loop_started_at) / 60.0
            max_total_stop = _loop_control.decide_max_total_time_stop(
                elapsed_total_min=elapsed_total_min,
                max_total_min=max_total_min,
            )
            if max_total_stop.should_stop:
                _autopilot_state._apply_run_status(
                    run_payload,
                    status=max_total_stop.status,
                    stop_reason=max_total_stop.stop_reason,
                )
                print(max_total_stop.message)
                break
            iter_dir = config.paths.iter_dir(run_id, iteration)
            logs_dir = iter_dir / "logs"
            agent_dir = iter_dir / "agent"
            output_dir = iter_dir / "output"
            iter_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)
            agent_dir.mkdir(parents=True, exist_ok=True)

            print(f"[cyan]iteration[/cyan]: {iteration}/{max_iterations}")
            knowledge_phase.refresh()

            _watch_state.update_watch_phase(config, run_id, "verifying", iteration=iteration)
            _run_verify(config.verify_cmd, dry_run=config.dry_run, artifacts_dir=config.paths.artifacts_dir)

            submission_path = iter_dir / "submission.csv"
            metrics_path = iter_dir / "metrics.json"
            evaluation_report_path = iter_dir / "evaluation_report.json"
            evaluation = None
            kernel_metrics_payload: dict[str, object] | None = None
            kernel_metrics_artifact_path: Path | None = None
            evaluation_by_source: dict[str, EvaluationResult] = {}
            model_summary = {}
            accelerator_used = config.accelerator
            submit_retry_resume = _autopilot_state._load_submit_retry_artifacts(
                run_dir=run_dir,
                iter_dir=iter_dir,
                iteration=iteration,
                max_iterations=max_iterations,
                metric_direction=metric_direction,
                target_metric=target_metric,
                require_submit_phase=submit_enabled and not config.dry_run,
                load_kernel_metrics=_kernel_metrics.load_kernel_metrics,
            )
            if submit_retry_resume is not None:
                resume_submission_path, resume_metrics_path, resume_evaluation = submit_retry_resume
                if resume_submission_path != submission_path:
                    submission_path = _autopilot_state._copy_submission_artifact_to_iteration_dir(
                        source=resume_submission_path,
                        iter_dir=iter_dir,
                    )
                if resume_metrics_path != metrics_path:
                    metrics_path.write_bytes(resume_metrics_path.read_bytes())
                evaluation = resume_evaluation
                kernel_metrics_payload = _json_utils.load_json_object(resume_metrics_path)
                kernel_metrics_artifact_path = resume_metrics_path
                print(
                    "[yellow]resume[/yellow]: "
                    f"iter-{iteration} has completed training artifacts; retrying submit without retraining."
                )

            if evaluation is None:
                _watch_state.update_watch_phase(config, run_id, "kernel_preflight", iteration=iteration)
                _run_kernel_source_preflight_fixes(
                    config=config,
                    run_id=run_id,
                    iteration=iteration,
                    iter_dir=iter_dir,
                    pending_error_fixes=pending_error_fixes,
                )

            if evaluation is None and config.compute.startswith("kaggle_"):
                kaggle_user = resolve_kaggle_username(config.kaggle_username)
                _watch_state.update_watch_phase(config, run_id, "kaggle_kernel_running", iteration=iteration)
                print(f"[cyan]kernel run[/cyan]: {config.compute}")
                kernel_attempts = 0
                error_fingerprints: dict[str, int] = {}
                while True:
                    try:
                        kernel_result = run_kernel(
                            slug=config.slug,
                            run_id=run_id,
                            iteration=iteration,
                            base_dir=config.paths.base_dir.parent,
                            kaggle_username=kaggle_user,
                            kernel_name=kernel_name,
                            accelerator=config.accelerator,
                            enable_internet=enable_internet,
                            score_source=score_source,
                            metric=target_metric,
                            direction=metric_direction,
                            holdout_frac=holdout_frac,
                            cv_folds=cv_folds,
                            seed=seed,
                            dry_run=config.dry_run,
                            timeout_minutes=time_budget_min,
                            hardware_profile=config.hardware_profile,
                        )
                        if kernel_result.submission_path:
                            submission_path = _autopilot_state._copy_submission_artifact_to_iteration_dir(
                                source=kernel_result.submission_path,
                                iter_dir=iter_dir,
                            )
                        _autopilot_state._copy_kernel_support_artifacts_to_iteration_dir(
                            kernel_output_dir=kernel_result.output_dir,
                            iter_dir=iter_dir,
                        )
                        if kernel_result.metrics_path and kernel_result.metrics_path.exists():
                            kernel_metrics_artifact_path = kernel_result.metrics_path
                            kernel_metrics_payload = _json_utils.load_json_object(kernel_result.metrics_path)
                            evaluation = _kernel_metrics.load_kernel_metrics(
                                kernel_result.metrics_path,
                                metric_direction,
                                target_metric,
                            )
                        if evaluation is None:
                            raise KernelFailedError(
                                "Kernel metrics missing expected score; "
                                "ensure metrics.json includes a numeric metric value."
                            )
                        break
                    except RulesNotAcceptedError:
                        raise
                    except KaggleNetworkError as exc:
                        kernel_attempts += 1
                        error_text = _kernel_errors.format_kernel_error(exc)
                        _kernel_errors.record_kernel_error(
                            logs_dir=logs_dir,
                            attempt=kernel_attempts,
                            error_text=error_text,
                            error_fingerprints=error_fingerprints,
                            output_dir=output_dir,
                        )
                        raise
                    except KernelStillRunningError as exc:
                        error_text = _kernel_errors.format_kernel_error(exc)
                        logs_dir.mkdir(parents=True, exist_ok=True)
                        (logs_dir / "kernel_remote_still_running.txt").write_text(error_text + "\n", encoding="utf-8")
                        _watch_state.update_watch_phase(
                            config,
                            run_id,
                            "kaggle_kernel_still_running",
                            detail=(
                                "Kaggle notebook is still running remotely; "
                                "waiting instead of pushing a duplicate version."
                            ),
                            iteration=iteration,
                        )
                        print(
                            "[yellow]kernel still running[/yellow]: "
                            f"retrying status in {KERNEL_STILL_RUNNING_RETRY_SLEEP:.0f}s without pushing a new version"
                        )
                        time.sleep(KERNEL_STILL_RUNNING_RETRY_SLEEP)
                        continue
                    except KernelCapacityError as exc:
                        kernel_attempts += 1
                        error_text = _kernel_errors.format_kernel_error(exc)
                        _kernel_errors.record_kernel_error(
                            logs_dir=logs_dir,
                            attempt=kernel_attempts,
                            error_text=error_text,
                            error_fingerprints=error_fingerprints,
                            max_repeats=MAX_KERNEL_CAPACITY_REPEAT,
                            output_dir=output_dir,
                        )
                        capacity_retries = _env_utils.env_int(
                            "KAGGLEBOT_KERNEL_CAPACITY_RETRIES",
                            default=MAX_KERNEL_CAPACITY_RETRIES,
                        )
                        _watch_state.update_watch_phase(
                            config,
                            run_id,
                            "kaggle_gpu_no_capacity",
                            detail="Kaggle GPU session capacity is unavailable.",
                            iteration=iteration,
                        )
                        if kernel_attempts > capacity_retries:
                            raise
                        wait_seconds = KERNEL_CAPACITY_RETRY_SLEEP * kernel_attempts
                        print(
                            "[yellow]kaggle gpu limit reached[/yellow]: "
                            f"retrying in {wait_seconds:.0f}s (attempt {kernel_attempts})"
                        )
                        time.sleep(wait_seconds)
                        continue
                    except Exception as exc:  # noqa: BLE001
                        if _kernel_errors.is_kernel_registration_error(exc):
                            kernel_attempts += 1
                            error_text = _kernel_errors.format_kernel_error(exc)
                            _kernel_errors.record_kernel_error(
                                logs_dir=logs_dir,
                                attempt=kernel_attempts,
                                error_text=error_text,
                                error_fingerprints=error_fingerprints,
                                output_dir=output_dir,
                            )
                            if kernel_attempts > MAX_KERNEL_REGISTRATION_RETRIES:
                                raise
                            wait_seconds = KERNEL_REGISTRATION_RETRY_SLEEP * kernel_attempts
                            print(
                                "[yellow]kernel registration pending[/yellow]: "
                                f"retrying in {wait_seconds:.0f}s (attempt {kernel_attempts})"
                            )
                            time.sleep(wait_seconds)
                            continue
                        kernel_attempts += 1
                        error_text = _kernel_errors.format_kernel_error(exc)
                        try:
                            _kernel_errors.record_kernel_error(
                                logs_dir=logs_dir,
                                attempt=kernel_attempts,
                                error_text=error_text,
                                error_fingerprints=error_fingerprints,
                                output_dir=output_dir,
                            )
                        except KernelFailedError:
                            if _maybe_regenerate_kernel_sources_once(
                                config=config,
                                run_id=run_id,
                                iteration=iteration,
                                iter_dir=iter_dir,
                                attempt=kernel_attempts,
                                trigger_reason="repeated_error_fingerprint",
                            ):
                                error_fingerprints.clear()
                                continue
                            raise
                        if config.dry_run:
                            raise
                        if MAX_KERNEL_FIX_ATTEMPTS is not None and kernel_attempts > MAX_KERNEL_FIX_ATTEMPTS:
                            raise
                        print(
                            f"[yellow]kernel failed[/yellow]: invoking "
                            f"{IMPLEMENTATION_AGENT.log_alias} to fix (attempt {kernel_attempts})"
                        )
                        _run_kernel_fix(
                            config=config,
                            run_id=run_id,
                            iteration=iteration,
                            iter_dir=iter_dir,
                            error_message=error_text,
                            attempt=kernel_attempts,
                            pending_error_fixes=pending_error_fixes,
                        )
            elif evaluation is None:
                kernel_path = config.paths.kernel_source_dir / "kernel.py"
                if not kernel_path.exists():
                    raise RuntimeError(
                        "Local autopilot requires kernel.py, but "
                        f"{kernel_path} was not found. Run planning/implement to generate kernel.py first."
                    )
                _watch_state.update_watch_phase(config, run_id, "local_kernel_running", iteration=iteration)
                print(f"[cyan]kernel local run[/cyan]: {config.compute}")
                kernel_attempts = 0
                error_fingerprints = {}
                while True:
                    try:
                        kernel_result = run_kernel_local(
                            slug=config.slug,
                            run_id=run_id,
                            iteration=iteration,
                            base_dir=config.paths.base_dir.parent,
                            accelerator=config.accelerator,
                            score_source=score_source,
                            metric=target_metric,
                            direction=metric_direction,
                            holdout_frac=holdout_frac,
                            cv_folds=cv_folds,
                            seed=seed,
                            dry_run=config.dry_run,
                            timeout_minutes=time_budget_min,
                            strict_accelerator=config.strict_accelerator,
                            hardware_profile=config.hardware_profile,
                        )
                        if kernel_result.submission_path:
                            submission_path = _autopilot_state._copy_submission_artifact_to_iteration_dir(
                                source=kernel_result.submission_path,
                                iter_dir=iter_dir,
                            )
                        _autopilot_state._copy_kernel_support_artifacts_to_iteration_dir(
                            kernel_output_dir=kernel_result.output_dir,
                            iter_dir=iter_dir,
                        )
                        if kernel_result.metrics_path and kernel_result.metrics_path.exists():
                            kernel_metrics_artifact_path = kernel_result.metrics_path
                            kernel_metrics_payload = _json_utils.load_json_object(kernel_result.metrics_path)
                            evaluation = _kernel_metrics.load_kernel_metrics(
                                kernel_result.metrics_path,
                                metric_direction,
                                target_metric,
                            )
                        if evaluation is None:
                            raise KernelFailedError(
                                "Local kernel metrics missing expected score; "
                                "ensure metrics.json includes a numeric metric value."
                            )
                        break
                    except Exception as exc:  # noqa: BLE001
                        kernel_attempts += 1
                        error_text = _kernel_errors.format_kernel_error(exc)
                        try:
                            _kernel_errors.record_kernel_error(
                                logs_dir=logs_dir,
                                attempt=kernel_attempts,
                                error_text=error_text,
                                error_fingerprints=error_fingerprints,
                                output_dir=output_dir,
                            )
                        except KernelFailedError:
                            if _maybe_regenerate_kernel_sources_once(
                                config=config,
                                run_id=run_id,
                                iteration=iteration,
                                iter_dir=iter_dir,
                                attempt=kernel_attempts,
                                trigger_reason="repeated_error_fingerprint",
                            ):
                                error_fingerprints.clear()
                                continue
                            raise
                        if config.dry_run:
                            raise
                        if MAX_KERNEL_FIX_ATTEMPTS is not None and kernel_attempts > MAX_KERNEL_FIX_ATTEMPTS:
                            raise
                        print(
                            f"[yellow]local kernel failed[/yellow]: invoking "
                            f"{IMPLEMENTATION_AGENT.log_alias} to fix (attempt {kernel_attempts})"
                        )
                        _run_kernel_fix(
                            config=config,
                            run_id=run_id,
                            iteration=iteration,
                            iter_dir=iter_dir,
                            error_message=error_text,
                            attempt=kernel_attempts,
                            pending_error_fixes=pending_error_fixes,
                        )

            if evaluation is None:
                raise RuntimeError("No evaluation metrics produced.")
            metric_mismatch_detected = False
            metric_mismatch_reason: str | None = None
            metric_fix_attempts = 0
            metric_recheck_attempted = False
            while (
                evaluation.metric
                and target_metric
                and (not _metric_matching.metrics_equivalent(evaluation.metric, target_metric))
            ):
                corrected_direction, confident = _metric_matching.infer_metric_direction_for_mismatch(
                    evaluation.metric,
                    metric_direction,
                )
                confidence_text = "high" if confident else "fallback"
                official_metric_override = _score_progress.resolve_explicit_official_metric_override(
                    kernel_metrics_payload,
                    target_metric=target_metric,
                    evaluation_metric=evaluation.metric,
                )
                if official_metric_override:
                    metric_direction, _ = _metric_matching.infer_metric_direction_for_mismatch(
                        official_metric_override,
                        corrected_direction,
                    )
                    print(
                        "[yellow]metric mismatch[/yellow]: "
                        f"plan={target_metric}/{metric_direction}, "
                        f"kernel={evaluation.metric}/{corrected_direction}. "
                        "Kernel metrics.json declares an explicit official competition metric; "
                        "updating the run target to match it."
                    )
                    target_metric = official_metric_override
                    resolved["target_metric"] = target_metric
                    resolved["target_direction"] = metric_direction
                    _plan_policy.write_resolved_plan_config(
                        config.paths,
                        resolved,
                        default_max_iterations=_DEFAULT_MAX_ITERATIONS,
                        default_force_major_rank_max_percentile=_DEFAULT_FORCE_MAJOR_RANK_MAX_PERCENTILE,
                        default_force_major_rank_min_teams=_DEFAULT_FORCE_MAJOR_RANK_MIN_TEAMS,
                    )
                    from kagglebot.solver.evaluate import EvaluationResult

                    evaluation = EvaluationResult(
                        score_source=evaluation.score_source,
                        metric=target_metric,
                        direction=metric_direction,  # type: ignore[arg-type]
                        value=evaluation.value,
                        std=evaluation.std,
                        train_score=evaluation.train_score,
                        val_score=evaluation.val_score,
                        fold_scores=evaluation.fold_scores,
                    )
                    continue
                if strict_competition_metric:
                    metric_mismatch_reason = (
                        f"target={target_metric}/{metric_direction}, kernel={evaluation.metric}/{corrected_direction}"
                    )
                    if (
                        not metric_recheck_attempted
                        and kernel_metrics_artifact_path is not None
                        and kernel_metrics_artifact_path.exists()
                    ):
                        metric_recheck_attempted = True
                        print(
                            "[yellow]metric mismatch[/yellow]: "
                            f"{metric_mismatch_reason} "
                            f"(direction_confidence={confidence_text}). "
                            "Strict competition metric mode is enabled; attempting same-iteration metric recheck "
                            f"before invoking {IMPLEMENTATION_AGENT.display_name}."
                        )
                        evaluation, kernel_metrics_payload, submission_path = _rerun_kernel_for_metric_recheck(
                            config=config,
                            run_id=run_id,
                            iteration=iteration,
                            submission_path=submission_path,
                            iter_dir=iter_dir,
                            metrics_artifact_path=kernel_metrics_artifact_path,
                            kernel_name=kernel_name,
                            enable_internet=enable_internet,
                            score_source=score_source,
                            target_metric=target_metric,
                            metric_direction=metric_direction,
                            holdout_frac=holdout_frac,
                            cv_folds=cv_folds,
                            seed=seed,
                            time_budget_min=time_budget_min,
                        )
                        continue
                    metric_fix_attempts += 1
                    if metric_fix_attempts > _MAX_METRIC_FIX_ATTEMPTS:
                        metric_mismatch_detected = True
                        raise RuntimeError(
                            "Competition metric mismatch persisted after metric-only repairs "
                            f"(attempts={_MAX_METRIC_FIX_ATTEMPTS}, {metric_mismatch_reason})."
                        )
                    print(
                        "[yellow]metric mismatch[/yellow]: "
                        f"{metric_mismatch_reason} "
                        f"(direction_confidence={confidence_text}). "
                        "Strict competition metric mode is enabled; applying "
                        f"metric-only {IMPLEMENTATION_AGENT.display_name} fix "
                        f"(attempt {metric_fix_attempts}/{_MAX_METRIC_FIX_ATTEMPTS}) and re-running evaluation."
                    )
                    _run_metric_only_competition_metric_fix(
                        config=config,
                        run_id=run_id,
                        iteration=iteration,
                        iter_dir=iter_dir,
                        mismatch_reason=metric_mismatch_reason,
                        attempt=metric_fix_attempts,
                        pending_error_fixes=pending_error_fixes,
                    )
                    evaluation, kernel_metrics_payload, submission_path = _rerun_kernel_for_metric_recheck(
                        config=config,
                        run_id=run_id,
                        iteration=iteration,
                        submission_path=submission_path,
                        iter_dir=iter_dir,
                        metrics_artifact_path=kernel_metrics_artifact_path,
                        kernel_name=kernel_name,
                        enable_internet=enable_internet,
                        score_source=score_source,
                        target_metric=target_metric,
                        metric_direction=metric_direction,
                        holdout_frac=holdout_frac,
                        cv_folds=cv_folds,
                        seed=seed,
                        time_budget_min=time_budget_min,
                    )
                    metric_still_mismatched = bool(
                        evaluation.metric
                        and target_metric
                        and (not _metric_matching.metrics_equivalent(evaluation.metric, target_metric))
                    )
                    if metric_still_mismatched and (not config.dry_run) and (not config.compute.startswith("kaggle_")):
                        print(
                            "[yellow]metric mismatch[/yellow]: "
                            f"{metric_mismatch_reason}. "
                            "Metric-only fix was applied but metrics.json is still stale; "
                            "re-running local kernel once to materialize updated metric outputs."
                        )
                        kernel_result = run_kernel_local(
                            slug=config.slug,
                            run_id=run_id,
                            iteration=iteration,
                            base_dir=config.paths.base_dir.parent,
                            accelerator=config.accelerator,
                            score_source=score_source,
                            metric=target_metric,
                            direction=metric_direction,
                            holdout_frac=holdout_frac,
                            cv_folds=cv_folds,
                            seed=seed,
                            dry_run=config.dry_run,
                            timeout_minutes=time_budget_min,
                            strict_accelerator=config.strict_accelerator,
                            hardware_profile=config.hardware_profile,
                        )
                        if kernel_result.submission_path:
                            submission_path = _autopilot_state._copy_submission_artifact_to_iteration_dir(
                                source=kernel_result.submission_path,
                                iter_dir=iter_dir,
                            )
                        _autopilot_state._copy_kernel_support_artifacts_to_iteration_dir(
                            kernel_output_dir=kernel_result.output_dir,
                            iter_dir=iter_dir,
                        )
                        if kernel_result.metrics_path and kernel_result.metrics_path.exists():
                            kernel_metrics_artifact_path = kernel_result.metrics_path
                            kernel_metrics_payload = _json_utils.load_json_object(kernel_result.metrics_path)
                            evaluation = _kernel_metrics.load_kernel_metrics(
                                kernel_result.metrics_path,
                                metric_direction,
                                target_metric,
                            )
                        if evaluation is None:
                            raise KernelFailedError(
                                "Metric-only repair rerun failed: local kernel metrics missing expected score; "
                                "ensure metrics.json includes a numeric metric value."
                            )
                    continue
                if corrected_direction != metric_direction or evaluation.metric != target_metric:
                    print(
                        "[yellow]metric mismatch[/yellow]: "
                        f"plan={target_metric}/{metric_direction}, "
                        f"kernel={evaluation.metric}/{corrected_direction} "
                        f"(direction_confidence={confidence_text}). "
                        "Updating plan to match kernel metric."
                    )
                    metric_direction = corrected_direction
                    target_metric = evaluation.metric
                    resolved["target_metric"] = target_metric
                    resolved["target_direction"] = metric_direction
                    if isinstance(top1_info, dict) and isinstance(top1_info.get("score"), (int, float)):
                        target_score = float(top1_info["score"])
                        resolved["target_score"] = target_score
                    _plan_policy.write_resolved_plan_config(
                        config.paths,
                        resolved,
                        default_max_iterations=_DEFAULT_MAX_ITERATIONS,
                        default_force_major_rank_max_percentile=_DEFAULT_FORCE_MAJOR_RANK_MAX_PERCENTILE,
                        default_force_major_rank_min_teams=_DEFAULT_FORCE_MAJOR_RANK_MIN_TEAMS,
                    )
                    from kagglebot.solver.evaluate import EvaluationResult

                    evaluation = EvaluationResult(
                        score_source=evaluation.score_source,
                        metric=target_metric,
                        direction=metric_direction,  # type: ignore[arg-type]
                        value=evaluation.value,
                        std=evaluation.std,
                        train_score=evaluation.train_score,
                        val_score=evaluation.val_score,
                        fold_scores=evaluation.fold_scores,
                    )
            _watch_state.update_watch_phase(config, run_id, "evaluating_iteration", iteration=iteration)
            report, report_payload, eval_data_cache = _iteration_metrics.build_iteration_evaluation_report(
                config=config,
                run_id=run_id,
                iteration=iteration,
                evaluation=evaluation,
                evaluation_by_source=evaluation_by_source,
                metric_direction=metric_direction,
                cv_folds=cv_folds,
                split_strategy=split_strategy,
                seed=seed,
                eval_seeds=eval_seeds,
                eval_repeats=eval_repeats,
                score_source=score_source,
                ci_method=ci_method,
                ci_alpha=ci_alpha,
                readiness_method=readiness_method,
                readiness_k=readiness_k,
                drift_check_enabled=drift_check_enabled,
                drift_weight=drift_weight,
                eval_data_cache=eval_data_cache,
            )
            _iteration_metrics.append_run_evaluation_report(
                run_dir=run_dir, iteration=iteration, payload=report_payload
            )
            evaluation_report_path = iter_dir / "evaluation_report.json"
            _json_utils.write_json_object(evaluation_report_path, report_payload)

            readiness_score = report.readiness_score
            print(
                f"[green]iteration complete[/green]: {evaluation.metric}={evaluation.value:.6f} "
                f"(SRS={readiness_score:.6f})"
            )
            if evaluation_by_source:
                values_line = ", ".join(
                    f"{source}={result.value:.6f}" for source, result in evaluation_by_source.items()
                )
                print(f"[cyan]evaluation sources[/cyan]: {values_line}")
            top1_score = top1_info.get("score") if isinstance(top1_info, dict) else None
            decision_score = float(evaluation.value)
            decision_source = str(evaluation.score_source or "offline")
            top1_score_value = float(top1_score) if isinstance(top1_score, (int, float)) else None
            effective_best_score, best_score_guard = _score_progress.effective_best_score_for_progress(
                prev_best=best_score,
                current_score=decision_score,
                top1_score=top1_score_value,
                direction=metric_direction,
            )
            if best_score_guard is not None and effective_best_score is not None:
                print(
                    "[yellow]best-score guard[/yellow]: "
                    f"clipped previous best from {float(best_score_guard['prev_best']):.6f} "
                    f"to {float(best_score_guard['effective_best']):.6f} "
                    f"(top1={float(best_score_guard['top1_score']):.6f}, "
                    f"margin={float(best_score_guard['margin']):.6f})."
                )
                best_score = effective_best_score
            top1_tier_by_submission = False
            submission_rank: int | None = None
            submission_total_teams: int | None = None
            submission_rank_percentile: float | None = None
            submission_rank_source: str | None = None
            submission_rank_estimate: int | None = None
            submission_total_teams_estimate: int | None = None
            submission_rank_percentile_estimate: float | None = None
            submission_rank_estimate_source: str | None = None
            medal_target_met = False
            medal_minimum_improvement_mode: str | None = None
            medal_policy_reason: str | None = None
            rank_forced_major_overhaul = False
            rank_force_reason: str | None = None
            code_reference_score, code_reference_source = _code_reference.extract_code_reference_score(config.paths)
            code_reference_comparison_score = _score_progress.normalize_code_reference_score_for_comparison(
                current=decision_score,
                reference=code_reference_score,
                metric=evaluation.metric,
            )
            code_reference_delta_vs_current = (
                _score_progress.score_delta_vs_reference(
                    decision_score,
                    code_reference_comparison_score,
                    metric_direction,
                )
                if code_reference_comparison_score is not None
                else None
            )
            first_iteration_below_code_reference = bool(
                iteration == 1 and code_reference_delta_vs_current is not None and code_reference_delta_vs_current < 0.0
            )
            score_drop_vs_best = _score_progress.score_drop_vs_best(
                best_score=best_score,
                current_score=decision_score,
                direction=metric_direction,
            )
            severe_regression_detected = _score_progress.is_severe_regression_vs_best(
                metric=evaluation.metric,
                direction=metric_direction,
                best_score=best_score,
                current_score=decision_score,
            )
            conservative_feature_collapse = _score_progress.is_conservative_feature_collapse(kernel_metrics_payload)
            conservative_regression_detected = bool(severe_regression_detected and conservative_feature_collapse)

            quality_guard = _kernel_quality.build_kernel_quality_guard(
                evaluation=evaluation,
                kernel_metrics_payload=kernel_metrics_payload,
                evaluation_report=report,
                evaluation_contract=evaluation_contract,
                logs_dir=logs_dir,
                direction=metric_direction,
                iteration=iteration,
                max_iterations=max_iterations,
                force_submit=config.force_submit,
                code_reference_score=code_reference_score,
                code_reference_source=code_reference_source,
                metric_mismatch_detected=metric_mismatch_detected,
                metric_mismatch_reason=metric_mismatch_reason,
            )
            quality_allows_submit = bool(quality_guard.get("allow_submit", True))
            quality_reasons_raw = quality_guard.get("reasons")
            quality_reasons = (
                [str(item) for item in quality_reasons_raw if isinstance(item, str)]
                if isinstance(quality_reasons_raw, list)
                else []
            )
            accuracy_potential = _kernel_quality.build_accuracy_potential(
                score_source=evaluation.score_source,
                kernel_metrics_payload=kernel_metrics_payload,
                model_summary=model_summary,
                quality_guard=quality_guard,
                evaluation_contract=evaluation_contract,
            )
            non_generalizable_eval_detected = any(
                reason
                in {
                    "untrusted_score_source",
                    "oracle_override_detected",
                    "competition_metric_mismatch",
                    "competition_split_mismatch",
                    "competition_score_source_mismatch",
                    "competition_evaluation_unfaithful",
                    "missing_competitive_data",
                    "external_test_label_transfer_detected",
                }
                for reason in quality_reasons
            )
            quality_forced_major_overhaul = "below_code_reference_baseline" in quality_reasons
            quality_force_reason: str | None = None
            if quality_forced_major_overhaul:
                if code_reference_comparison_score is not None:
                    code_delta = _score_progress.score_delta_vs_reference(
                        decision_score,
                        code_reference_comparison_score,
                        metric_direction,
                    )
                    quality_force_reason = (
                        "Offline score is materially below code reference baseline: "
                        f"current={decision_score:.6f}, code_ref={code_reference_score:.6f}, "
                        f"comparison_ref={code_reference_comparison_score:.6f}, "
                        f"delta={code_delta:+.6f}, source={code_reference_source or 'unknown'}."
                    )
                else:
                    quality_force_reason = (
                        "Offline score is materially below code reference baseline detected by quality guard."
                    )
            force_initial_submit = _submission_policy.should_force_initial_submit(
                deliverable_mode=deliverable_mode,
                iteration=iteration,
                submit_enabled=submit_enabled,
                dry_run=config.dry_run,
                submit_policy=str(resolved.get("submit_policy") or ""),
                submission_limit_per_day=submission_limit_per_day,
            )
            is_final_iteration = iteration >= max_iterations
            successful_submit_count = _submission_policy.submission_count_for_daily_limit(
                slug=config.slug,
                fallback_count=_submit_attempts.count_successful_submit_attempts(run_dir),
                submission_limit_per_day=submission_limit_per_day,
                dry_run=config.dry_run,
                fetch_submission_rows=lambda current_slug, dry_run: list_competition_submissions(
                    current_slug,
                    dry_run=dry_run,
                ),
                on_warning=print,
            )
            spare_daily_submission_slot = _submission_policy.has_spare_daily_submission_slot(
                submission_limit_per_day=submission_limit_per_day,
                submissions_used_today=successful_submit_count,
                iteration=iteration,
                max_iterations=max_iterations,
            )
            extra_daily_submission_slot = False
            if isinstance(submission_limit_per_day, int) and submission_limit_per_day > 0:
                remaining_daily_slots = max(0, submission_limit_per_day - max(0, int(successful_submit_count)))
                remaining_iterations = max(1, int(max_iterations) - int(iteration) + 1)
                extra_daily_submission_slot = remaining_daily_slots > remaining_iterations
            else:
                remaining_daily_slots = None
            campaign_candidate = None
            campaign_allocation = None
            reference_reproduction_report: dict[str, object] | None = None
            portfolio_plan: dict[str, object] | None = None
            blend_report: dict[str, object] | None = None
            experiment_graph: dict[str, object] | None = None
            allocator_decision: dict[str, object] | None = None
            graph_execution_report: dict[str, object] | None = None
            validation_lab_report: dict[str, object] | None = None
            private_robustness_report: dict[str, object] | None = None
            portfolio_optimizer_report: dict[str, object] | None = None
            top1_exhaustion_report: dict[str, object] | None = None
            if campaign_mode == "top1":
                campaign_category = _campaign_metrics.infer_campaign_candidate_category(
                    iteration=iteration,
                    kernel_metrics_payload=kernel_metrics_payload,
                    quality_reasons=quality_reasons,
                )
                candidate_offline_std = tolerant_finite_float(evaluation.std)
                campaign_candidate = build_campaign_candidate(
                    run_id=run_id,
                    iteration=iteration,
                    direction=metric_direction,
                    category=campaign_category,
                    offline_score=decision_score,
                    offline_std=candidate_offline_std,
                    score_source=decision_source,
                    submission_path=submission_path,
                    metrics_path=metrics_path,
                    oof_path=_campaign_metrics.extract_campaign_artifact_path(kernel_metrics_payload, "oof"),
                    prediction_path=_campaign_metrics.extract_campaign_artifact_path(
                        kernel_metrics_payload, "prediction"
                    ),
                    model_family=_campaign_metrics.infer_campaign_model_family(model_summary, kernel_metrics_payload),
                    feature_set=_campaign_metrics.infer_campaign_feature_set(model_summary, kernel_metrics_payload),
                    method_id=_campaign_metrics.extract_campaign_method_id(kernel_metrics_payload)
                    or select_method_id_for_category(method_registry, campaign_category),
                    validation_profile_id=_campaign_metrics.extract_campaign_validation_profile_id(
                        kernel_metrics_payload
                    )
                    or str(method_registry.get("active_validation_profile") or "default_cv"),
                    fold_scores=_campaign_metrics.extract_campaign_fold_scores(kernel_metrics_payload),
                    prediction_correlation=_campaign_metrics.extract_campaign_prediction_correlation(
                        kernel_metrics_payload
                    ),
                    metadata={
                        "metric": evaluation.metric,
                        "readiness_score": readiness_score,
                        "quality_reasons": quality_reasons,
                    },
                )
                campaign_candidate = replace(
                    campaign_candidate,
                    private_robustness_score=private_robustness_score(
                        campaign_candidate,
                        campaign_state=campaign_state,
                    ),
                )
                upsert_candidate(campaign_registry_file, campaign_candidate)
                campaign_state = update_campaign_state(
                    state_path=campaign_state_file,
                    registry_path=campaign_registry_file,
                    slug=config.slug,
                    run_id=run_id,
                    mode=campaign_mode,
                    direction=metric_direction,
                    top1_info=top1_info if isinstance(top1_info, dict) else {},
                    submission_history=previous_submission_history,
                    remaining_daily_slots=remaining_daily_slots,
                    method_registry=method_registry,
                )
                reference_reproduction_report = build_reference_reproduction_report(
                    context_dir=config.paths.context_dir,
                    campaign_state=campaign_state,
                    method_registry=method_registry,
                    direction=metric_direction,
                    current_candidate=campaign_candidate,
                    code_reference_score=code_reference_comparison_score,
                    code_reference_source=code_reference_source,
                )
                validation_registry = _json_utils.load_json_object_or_empty(config.paths.validation_registry_path)
                validation_lab_report = run_validation_lab(
                    context_dir=config.paths.context_dir,
                    validation_registry=validation_registry,
                    candidate_registry_path=campaign_registry_file,
                    campaign_state=campaign_state,
                    mode=validation_lab_mode,
                )
                validation_registry = (
                    validation_lab_report.get("registry")
                    if isinstance(validation_lab_report.get("registry"), dict)
                    else validation_registry
                )
                if isinstance(validation_registry, dict):
                    campaign_state["active_validation_profile"] = validation_registry.get("active_profile")
                portfolio_plan = build_candidate_portfolio_plan(
                    iter_dir=iter_dir,
                    registry_path=campaign_registry_file,
                    method_registry=method_registry,
                    validation_registry=validation_registry,
                    campaign_state=campaign_state,
                    run_id=run_id,
                    iteration=iteration,
                    direction=metric_direction,
                )
                blend_report = build_blend_report(
                    iter_dir=iter_dir,
                    registry_path=campaign_registry_file,
                    campaign_state=campaign_state,
                    validation_registry=validation_registry,
                    direction=metric_direction,
                )
                experiment_graph = build_experiment_graph(
                    context_dir=config.paths.context_dir,
                    iter_dir=iter_dir,
                    run_id=run_id,
                    iteration=iteration,
                    portfolio_execution=portfolio_execution,
                    portfolio_plan=portfolio_plan,
                    reference_report=reference_reproduction_report,
                    blend_report=blend_report,
                    validation_registry=validation_registry,
                    method_registry=method_registry,
                    campaign_state=campaign_state,
                )
                if portfolio_execution != "off":
                    graph_execution = execute_experiment_graph(
                        graph=experiment_graph,
                        context=RunContext(
                            competition=config.competition_url or config.slug,
                            slug=config.slug,
                            run_id=run_id,
                            paths=config.paths,
                            workdir=config.paths.repo_root,
                            dry_run=config.dry_run,
                            force=False,
                            force_submit=config.force_submit,
                            message=config.message or f"kagglebot campaign {run_id}",
                            time_budget_minutes=int(config.time_budget_min or 60),
                            cv_folds=max(2, int(cv_folds)),
                            model_names=None,
                            use_stacking=False,
                            compute=config.compute,
                            accelerator=accelerator_used,
                            enable_internet=str(config.internet or "off").lower() == "on",
                            kaggle_username=config.kaggle_username,
                            strict_accelerator=config.strict_accelerator,
                            candidate_budget_minutes=config.candidate_budget_min,
                            max_candidates_per_iteration=config.max_candidates_per_iteration,
                        ),
                        runner=LocalKernelRunner(),
                        iter_dir=iter_dir,
                    )
                    graph_execution_report = graph_execution.to_payload()
                    experiment_graph = (
                        _json_utils.load_json_object_or_empty(iter_dir / "experiment_graph.json") or experiment_graph
                    )
                private_robustness_report = build_private_robustness_report(
                    context_dir=config.paths.context_dir,
                    registry_path=campaign_registry_file,
                    campaign_state=campaign_state,
                    validation_lab_report=validation_lab_report,
                    direction=metric_direction,
                )
                portfolio_optimizer_report = build_portfolio_optimizer_report(
                    iter_dir=iter_dir,
                    registry_path=campaign_registry_file,
                    campaign_state=campaign_state,
                    validation_registry=validation_registry,
                    private_robustness_report=private_robustness_report,
                    remaining_daily_slots=remaining_daily_slots,
                    submit_policy=top1_submit_policy,
                    direction=metric_direction,
                )
                source_registry = (
                    _json_utils.load_json_object_or_empty(config.paths.source_registry_path) or source_registry
                )
                top1_exhaustion_report = build_top1_exhaustion_report(
                    context_dir=config.paths.context_dir,
                    run_id=run_id,
                    iteration=iteration,
                    campaign_state=campaign_state,
                    win_contract=win_contract,
                    method_registry=method_registry,
                    source_registry=source_registry,
                    validation_lab_report=validation_lab_report,
                    private_robustness_report=private_robustness_report,
                    portfolio_optimizer_report=portfolio_optimizer_report,
                    experiment_graph=experiment_graph,
                )
            forced_submit_reason: str | None = None
            quality_submit_override = _submission_policy.decide_quality_submit_override(
                submit_enabled=submit_enabled,
                quality_allows_submit=quality_allows_submit,
                force_submit=config.force_submit,
                force_initial_submit=force_initial_submit,
                spare_daily_submission_slot=spare_daily_submission_slot,
                quality_reasons=quality_reasons,
                spare_reason=_SPARE_DAILY_SUBMIT_REASON,
            )
            quality_allows_submit = quality_submit_override.quality_allows_submit
            forced_submit_reason = quality_submit_override.forced_submit_reason
            if quality_submit_override.override_reason == _SPARE_DAILY_SUBMIT_REASON:
                print(
                    "[yellow]submit override[/yellow]: spare daily submission slots remain; "
                    "allowing submit through soft quality guard reasons."
                )
            if quality_submit_override.blocked_reason is not None:
                print(
                    "[yellow]submit blocked[/yellow]: kernel quality guard detected unstable evaluation "
                    f"({quality_submit_override.blocked_reason}); submission is deferred to a later iteration."
                )
            high_potential_improved = False
            if accuracy_potential.get("eligible"):
                if _score_progress.should_update_best_accuracy_candidate(
                    current_potential=accuracy_potential,
                    best_potential=best_high_potential_meta,
                    current_score=decision_score,
                    best_score=best_high_potential_score,
                    direction=metric_direction,
                ):
                    best_high_potential_score = decision_score
                    best_high_potential_submission = submission_path
                    best_high_potential_iteration = iteration
                    best_high_potential_meta = dict(accuracy_potential)
                    frontier_no_improve_streak = 0
                    high_potential_improved = True
                else:
                    frontier_no_improve_streak += 1
            if submit_enabled and (quality_allows_submit or config.force_submit):
                if _score_utils.should_update_best_score(best_submittable_score, decision_score, metric_direction, 0.0):
                    best_submittable_score = decision_score
                    best_submittable_submission = submission_path

            submit_improvement_allowed = True
            submit_non_improving = False
            defer_submit_for_accuracy_frontier = False
            submit_improvement_gate = _submit_stage.decide_iteration_submit_improvement_gate(
                submit_improved_only=submit_improved_only,
                force_submit=config.force_submit,
                require_submit_improvement=require_submit_improvement,
                best_submitted_score=best_submitted_score,
                current_score=decision_score,
                direction=metric_direction,
                min_improvement=stop_min_delta,
                final_iteration=is_final_iteration,
                submit_enabled=submit_enabled,
                quality_allows_submit=quality_allows_submit,
                spare_daily_submission_slot=spare_daily_submission_slot,
                submission_limit_per_day=submission_limit_per_day,
                forced_submit_reason=forced_submit_reason,
                spare_submit_reason=_SPARE_DAILY_SUBMIT_REASON,
            )
            submit_improvement_allowed = submit_improvement_gate.submit_improvement_allowed
            submit_non_improving = submit_improvement_gate.submit_non_improving
            forced_submit_reason = submit_improvement_gate.forced_submit_reason
            if submit_improvement_gate.message:
                print(submit_improvement_gate.message)
            if submit_enabled and isinstance(best_high_potential_meta, dict):
                current_priority = tolerant_int(accuracy_potential.get("frontier_priority")) or 0
                best_priority = tolerant_int(best_high_potential_meta.get("frontier_priority")) or 0
                if (
                    best_high_potential_submission is not None
                    and best_high_potential_submission != submission_path
                    and best_priority > current_priority
                    and (
                        not bool(best_high_potential_meta.get("faithful", False))
                        or not bool(best_high_potential_meta.get("trusted", False))
                    )
                ):
                    defer_submit_for_accuracy_frontier = True
                    print(
                        "[yellow]submit deferred[/yellow]: preserving a higher-potential unsubmitted candidate "
                        "instead of auto-submitting a weaker artifact."
                    )
            if (
                defer_submit_for_accuracy_frontier
                and extra_daily_submission_slot
                and quality_allows_submit
                and submit_improvement_allowed
            ):
                defer_submit_for_accuracy_frontier = False
                forced_submit_reason = forced_submit_reason or _SPARE_DAILY_SUBMIT_REASON
                print(
                    "[yellow]submit override[/yellow]: spare daily submission slots remain; "
                    "not preserving a higher-potential candidate for later."
                )
            allow_submit = _submission_policy.should_attempt_submit_for_readiness(
                gate=submission_gate,
                readiness_score=decision_score,
                readiness_target=target_score,
                direction=metric_direction,
                iteration=iteration,
                max_iterations=max_iterations,
                submission_limit_per_day=submission_limit_per_day,
                successful_submissions=successful_submit_count,
                top1_score=top1_score if isinstance(top1_score, (int, float)) else None,
            )
            if not submit_improvement_allowed:
                allow_submit = False
            if defer_submit_for_accuracy_frontier:
                allow_submit = False
            if (not quality_allows_submit) and (not config.force_submit):
                allow_submit = False
            submit_non_improving = submit_enabled and submit_non_improving
            daily_submission_limit_reached = (
                submit_enabled
                and isinstance(submission_limit_per_day, int)
                and submission_limit_per_day > 0
                and max(0, int(successful_submit_count)) >= submission_limit_per_day
            )
            limited_holdback_decision = _submission_policy.decide_limited_submission_holdback(
                submit_enabled=submit_enabled,
                submission_limit_per_day=submission_limit_per_day,
                quality_allows_submit=quality_allows_submit,
                submit_improvement_allowed=submit_improvement_allowed,
                successful_submit_count=successful_submit_count,
                max_iterations=max_iterations,
                allow_submit=allow_submit,
            )
            submit_limited_holdback = limited_holdback_decision.holdback
            if limited_holdback_decision.reason == "reserved_final_slot":
                print(
                    "[yellow]submit deferred[/yellow]: reserved final submission slot "
                    "until offline score reaches top1-tier, readiness target, or final iteration."
                )
            elif limited_holdback_decision.reason == "strict_limited_cadence":
                print(
                    "[yellow]submit deferred[/yellow]: strict limited-submission cadence "
                    "is active because daily limit is lower than max iterations."
                )
            initial_probe_decision = _submission_policy.decide_initial_submit_probe(
                force_initial_submit=force_initial_submit,
                quality_allows_submit=quality_allows_submit,
                force_submit=config.force_submit,
                quality_reasons=quality_reasons,
                allow_submit=allow_submit,
                forced_submit_reason=forced_submit_reason,
                probe_reason=_FORCED_INITIAL_SUBMIT_REASON,
            )
            force_initial_submit = initial_probe_decision.force_initial_submit
            quality_allows_submit = initial_probe_decision.quality_allows_submit
            allow_submit = initial_probe_decision.allow_submit
            forced_submit_reason = initial_probe_decision.forced_submit_reason
            if initial_probe_decision.soft_probe_override:
                print(
                    "[yellow]submit override[/yellow]: iter 1 only failed a soft baseline guard; "
                    "submitting the trained/validated artifact to probe the Kaggle contract."
                )
            if initial_probe_decision.skipped_reason == "quality_guard":
                print(
                    "[yellow]submit override skipped[/yellow]: "
                    "iter 1 artifact failed training/validation quality guard; "
                    "not probing with an untrusted output."
                )
            if initial_probe_decision.probe_forced:
                submit_non_improving = False
                defer_submit_for_accuracy_frontier = False
                submit_limited_holdback = False
                print("[yellow]submit override[/yellow]: forcing iter 1 submit to probe Kaggle submission contract.")
            if campaign_mode == "top1" and campaign_candidate is not None:
                campaign_allocation = allocate_submission(
                    candidate=campaign_candidate,
                    campaign_state=campaign_state,
                    remaining_daily_slots=remaining_daily_slots,
                    novelty=0.6 if campaign_candidate.category in {"blend", "validation_variant"} else 0.4,
                    calibration_exception=campaign_candidate.category == "calibration",
                    force=config.force_submit or force_initial_submit,
                )
                if not campaign_allocation.allow_submit and not config.force_submit:
                    allow_submit = False
                    submit_non_improving = False
                    defer_submit_for_accuracy_frontier = False
                    submit_limited_holdback = False
                    print(f"[yellow]campaign submit deferred[/yellow]: {campaign_allocation.reason}.")
                allocator_decision = write_allocator_decision(
                    iter_dir=iter_dir,
                    candidate=campaign_candidate,
                    allocation=campaign_allocation,
                    campaign_state=campaign_state,
                    experiment_graph=experiment_graph,
                )
                append_campaign_outcome(
                    context_dir=config.paths.context_dir,
                    run_id=run_id,
                    iteration=iteration,
                    phase="pre_submit",
                    candidate=campaign_candidate,
                    allocation=campaign_allocation,
                    campaign_state=campaign_state,
                    experiment_graph=experiment_graph,
                )
            if daily_submission_limit_reached:
                forced_submit_reason = None
                allow_submit = False
                submit_non_improving = False
                defer_submit_for_accuracy_frontier = False
                submit_limited_holdback = False
                print(
                    "[yellow]submit skipped[/yellow]: daily submission limit reached "
                    f"({successful_submit_count}/{submission_limit_per_day} used in the current 24h window)."
                )
            submit_phase_required = submit_enabled and not config.dry_run
            submit_allowed_by_gate = submit_enabled and allow_submit
            pre_submit_phase_state = _submit_stage.resolve_iteration_submit_phase_state(
                submit_enabled=submit_enabled,
                daily_submission_limit_reached=daily_submission_limit_reached,
                force_initial_submit=force_initial_submit,
                quality_allows_submit=quality_allows_submit,
                force_submit=config.force_submit,
                submit_non_improving=submit_non_improving,
                defer_submit_for_accuracy_frontier=defer_submit_for_accuracy_frontier,
                submit_limited_holdback=submit_limited_holdback,
            )
            pre_submit_phase_finished = (not submit_phase_required) or (not submit_allowed_by_gate)
            submit_status_message = _submit_stage.format_iteration_submit_status_message(
                iteration=iteration,
                max_iterations=max_iterations,
                submit_enabled=submit_enabled,
                submit_allowed_by_gate=submit_allowed_by_gate,
                submit_phase_state=pre_submit_phase_state,
                quality_reasons=quality_reasons,
                competition_faithfulness=quality_guard.get("competition_faithfulness")
                if isinstance(quality_guard.get("competition_faithfulness"), dict)
                else None,
            )
            if submit_status_message:
                print(submit_status_message)
            pre_submit_metrics_payload = _iteration_metrics.build_metrics_payload(
                run_id=run_id,
                iteration=iteration,
                evaluation=evaluation,
                target_score=target_score,
                met_target=_submission_policy.meets_target(decision_score, target_score, metric_direction),
                top1_info=top1_info if isinstance(top1_info, dict) else {},
                compute=config.compute,
                accelerator=accelerator_used,
                holdout_frac=holdout_frac,
                cv_folds=cv_folds,
                seed=seed,
                evaluation_by_source=evaluation_by_source,
                evaluation_report=report,
                readiness_target=readiness_target,
                evaluation_contract=evaluation_contract,
                competition_faithfulness=quality_guard.get("competition_faithfulness")
                if isinstance(quality_guard.get("competition_faithfulness"), dict)
                else None,
                accuracy_potential=accuracy_potential,
            )
            pre_submit_metrics_payload["checkpoint_phase"] = "pre_submit"
            pre_submit_metrics_payload["quality_guard"] = quality_guard
            pre_submit_metrics_payload["forced_submit_reason"] = forced_submit_reason or ""
            if campaign_mode == "top1":
                pre_submit_metrics_payload["campaign"] = {
                    "state_path": str(campaign_state_file),
                    "registry_path": str(campaign_registry_file),
                    "state": campaign_state,
                    "candidate": campaign_candidate.to_payload() if campaign_candidate is not None else None,
                    "allocation": campaign_allocation.to_payload() if campaign_allocation is not None else None,
                    "reference_reproduction_report": reference_reproduction_report,
                    "portfolio_plan": portfolio_plan,
                    "blend_report": blend_report,
                    "validation_lab_report": validation_lab_report,
                    "win_contract": win_contract,
                    "private_robustness_report": private_robustness_report,
                    "portfolio_optimizer_report": portfolio_optimizer_report,
                    "top1_exhaustion_report": top1_exhaustion_report,
                    "experiment_graph": experiment_graph,
                    "allocator_decision": allocator_decision,
                    "graph_execution_report": graph_execution_report,
                }
            _json_utils.write_json_object(metrics_path, pre_submit_metrics_payload)
            _autopilot_state._write_iteration_state_marker(
                iter_dir=iter_dir,
                run_id=run_id,
                iteration=iteration,
                submission_path=submission_path,
                metrics_path=metrics_path,
                evaluation_report_path=evaluation_report_path,
                submit_phase_required=submit_phase_required,
                submit_phase_finished=pre_submit_phase_finished,
                submit_allowed_by_gate=submit_allowed_by_gate,
                submit_phase_state=pre_submit_phase_state,
                forced_submit_reason=forced_submit_reason,
                submitted=False,
                readiness_score=readiness_score,
            )
            submission_result: dict[str, object] | None = None
            submission_skipped = False
            submit_failed_deferred = False
            online_score: float | None = None
            submit_phase_state = pre_submit_phase_state
            if submit_enabled and allow_submit and submission_phase is not None:
                try:
                    submission_result = submission_phase.attempt(
                        submission_path=submission_path,
                        best_score=decision_score,
                    )
                except SubmitAbortedError:
                    if _submit_failure_context.should_defer_submit_abort_to_next_iteration(
                        compute=config.compute,
                        failure_context=_submit_failure_context.load_submit_failure_context(run_dir),
                        iteration=iteration,
                        max_iterations=max_iterations,
                    ):
                        submit_failed_deferred = True
                        submit_phase_state = _SUBMIT_FAILED_DEFERRED_STATE
                        print(
                            "[yellow]submit deferred[/yellow]: non-final Kaggle GPU submit failed; "
                            "carrying the submit contract failure into the next iteration."
                        )
                    else:
                        run_payload["status"] = "submit_failed"
                        _json_utils.write_json_object(run_dir / "run.json", run_payload)
                        raise
                if submission_result:
                    if bool(submission_result.get("skipped")):
                        submission_skipped = True
                        submit_phase_state = str(submission_result.get("reason") or "skipped")
                    else:
                        submit_phase_state = "submitted"
                        submitted = True
                        last_submission_result = submission_result
                        outcome_payload = submission_result.get("outcome")
                        if isinstance(outcome_payload, dict):
                            online_score = tolerant_finite_float(outcome_payload.get("score"))
                            if online_score is not None:
                                print(f"[cyan]submission score[/cyan]: {online_score:.6f}")
                                if isinstance(top1_score, (int, float)):
                                    top1_tier_by_submission = _submission_policy.is_top1_tier(
                                        float(online_score),
                                        float(top1_score),
                                        metric_direction,
                                    )
                            rank_payload = _submit_stage.resolve_submission_rank_payload(
                                slug=config.slug,
                                context_dir=config.paths.context_dir,
                                direction=metric_direction,
                                outcome=outcome_payload,
                                dry_run=config.dry_run,
                                leaderboard_rank_for_score=leaderboard_rank_for_score,
                            )
                            if rank_payload:
                                rank_state = _submit_stage.resolve_submission_rank_state(
                                    rank_payload=rank_payload,
                                    rank_force_major_max_percentile=rank_force_major_max_percentile,
                                    rank_force_major_min_teams=rank_force_major_min_teams,
                                    should_force_major_overhaul_by_rank=(
                                        _leaderboard_policy.should_force_major_overhaul_by_rank
                                    ),
                                )
                                outcome_payload.update(rank_state.rank_payload)
                                submission_rank = rank_state.rank
                                submission_total_teams = rank_state.total_teams
                                submission_rank_percentile = rank_state.rank_percentile
                                submission_rank_source = rank_state.rank_source
                                submission_rank_estimate = rank_state.estimated_rank
                                submission_total_teams_estimate = rank_state.estimated_total_teams
                                submission_rank_percentile_estimate = rank_state.estimated_rank_percentile
                                submission_rank_estimate_source = rank_state.rank_estimate_source
                                rank_forced_major_overhaul = rank_state.force_major_overhaul
                                rank_force_reason = rank_state.force_reason
                                for message in rank_state.messages:
                                    print(message)
                        submitted_tracking_score, submitted_tracking_source = (
                            _submit_stage.submission_score_for_tracking(
                                offline_score=decision_score,
                                online_score=online_score,
                            )
                        )
                        if _score_utils.should_update_best_score(
                            best_submitted_score,
                            submitted_tracking_score,
                            metric_direction,
                            0.0,
                        ):
                            best_submitted_score = submitted_tracking_score
                            if submitted_tracking_source != "offline":
                                print(
                                    "[cyan]submit tracking[/cyan]: "
                                    f"updated best submitted score from {submitted_tracking_source}."
                                )
                else:
                    submit_phase_state = "dry_run" if config.dry_run else "attempted_no_result"
            if target_rank_percentile is not None and deliverable_mode == "leaderboard":
                medal_target_met = _leaderboard_policy.meets_rank_percentile_target(
                    rank_percentile=submission_rank_percentile,
                    estimated_rank_percentile=submission_rank_percentile_estimate,
                    target_rank_percentile=target_rank_percentile,
                )
                if not medal_target_met:
                    medal_minimum_improvement_mode = "moderate_update"
                    medal_policy_reason = _leaderboard_policy.build_medal_target_reason(
                        target_medal=target_medal,
                        target_rank_percentile=target_rank_percentile,
                        rank_percentile=submission_rank_percentile,
                        estimated_rank_percentile=submission_rank_percentile_estimate,
                    )
                    if medal_policy_reason:
                        print(f"[yellow]medal policy[/yellow]: {medal_policy_reason}")
            met_target = _submission_policy.meets_target(decision_score, target_score, metric_direction)
            top1_tier = _submission_policy.is_top1_tier(decision_score, top1_score, metric_direction)
            top1_tier_by_readiness = _submission_policy.is_top1_tier(readiness_score, top1_score, metric_direction)
            noise_guard_decision = _loop_control.update_readiness_noise_guard(
                previous_readiness_score=previous_readiness_score,
                readiness_score=readiness_score,
                report_std=report.std,
                noise_limited_streak=noise_limited_streak,
            )
            previous_readiness_score = noise_guard_decision.previous_readiness_score
            delta_srs_vs_prev = noise_guard_decision.delta_srs_vs_prev
            noise_threshold = noise_guard_decision.noise_threshold
            noise_limited_streak = noise_guard_decision.noise_limited_streak
            noise_forced_major_overhaul = noise_guard_decision.force_major_overhaul
            code_reference_forced_reproduction = bool(
                first_iteration_below_code_reference or conservative_regression_detected
            )
            code_reference_force_reason: str | None = None
            if first_iteration_below_code_reference and code_reference_score is not None:
                code_reference_force_reason = (
                    "First iteration is below /code reference baseline; "
                    f"current={decision_score:.6f}, code_ref={code_reference_score:.6f}, "
                    f"comparison_ref={code_reference_comparison_score:.6f}, "
                    f"delta={float(code_reference_delta_vs_current):+.6f}. "
                    "Next iteration must implement the required reference notebook path."
                )
            elif conservative_regression_detected:
                drop_text = (
                    f"{float(score_drop_vs_best):.6f}" if isinstance(score_drop_vs_best, (int, float)) else "unknown"
                )
                code_reference_force_reason = (
                    "Detected severe regression with conservative feature collapse "
                    f"(drop_vs_best={drop_text}, max_features={_score_progress.CONSERVATIVE_COLLAPSE_MAX_FEATURES}). "
                    "Next iteration must recover from code reference baseline instead of keeping the collapsed path."
                )
            major_overhaul_policy = _submission_policy.decide_major_overhaul_policy(
                noise_forced_major_overhaul=noise_forced_major_overhaul,
                rank_forced_major_overhaul=rank_forced_major_overhaul,
                quality_forced_major_overhaul=quality_forced_major_overhaul,
                code_reference_forced_reproduction=code_reference_forced_reproduction,
                noise_limited_streak=noise_limited_streak,
                rank_force_reason=rank_force_reason,
                quality_force_reason=quality_force_reason,
                code_reference_force_reason=code_reference_force_reason,
                quality_reasons=quality_reasons,
            )
            force_major_overhaul_next = major_overhaul_policy.force_major_overhaul
            forced_major_overhaul_reason = major_overhaul_policy.forced_major_overhaul_reason
            fallback_submit_blocked_reason = major_overhaul_policy.fallback_submit_blocked_reason

            metrics_payload = _iteration_metrics.build_metrics_payload(
                run_id=run_id,
                iteration=iteration,
                evaluation=evaluation,
                target_score=target_score,
                met_target=met_target,
                top1_info=top1_info,
                compute=config.compute,
                accelerator=accelerator_used,
                holdout_frac=holdout_frac,
                cv_folds=cv_folds,
                seed=seed,
                evaluation_by_source=evaluation_by_source,
                evaluation_report=report,
                readiness_target=readiness_target,
                evaluation_contract=evaluation_contract,
                competition_faithfulness=quality_guard.get("competition_faithfulness")
                if isinstance(quality_guard.get("competition_faithfulness"), dict)
                else None,
                accuracy_potential=accuracy_potential,
            )
            campaign_metrics_payload = None
            if campaign_mode == "top1":
                campaign_metrics_payload = {
                    "state_path": str(campaign_state_file),
                    "registry_path": str(campaign_registry_file),
                    "state": campaign_state,
                    "candidate": campaign_candidate.to_payload() if campaign_candidate is not None else None,
                    "allocation": campaign_allocation.to_payload() if campaign_allocation is not None else None,
                    "reference_reproduction_report": reference_reproduction_report,
                    "portfolio_plan": portfolio_plan,
                    "blend_report": blend_report,
                    "validation_lab_report": validation_lab_report,
                    "win_contract": win_contract,
                    "private_robustness_report": private_robustness_report,
                    "portfolio_optimizer_report": portfolio_optimizer_report,
                    "top1_exhaustion_report": top1_exhaustion_report,
                    "experiment_graph": experiment_graph,
                    "allocator_decision": allocator_decision,
                    "graph_execution_report": graph_execution_report,
                }
            metrics_payload = _iteration_metrics.build_final_metrics_payload(
                base_payload=metrics_payload,
                loop_decision_source=decision_source,
                loop_decision_value=decision_score,
                noise_guard=_iteration_metrics.build_noise_guard_payload(
                    delta_srs_vs_prev=delta_srs_vs_prev,
                    noise_threshold=noise_threshold,
                    noise_limited_streak=noise_limited_streak,
                    force_major_overhaul_next=force_major_overhaul_next,
                ),
                rank_guard=_iteration_metrics.build_rank_guard_payload(
                    target_medal=target_medal,
                    target_rank_percentile=target_rank_percentile,
                    target_rank_met=medal_target_met,
                    minimum_improvement_mode=medal_minimum_improvement_mode,
                    rank=submission_rank,
                    total_teams=submission_total_teams,
                    rank_percentile=submission_rank_percentile,
                    rank_source=submission_rank_source,
                    estimated_rank=submission_rank_estimate,
                    estimated_total_teams=submission_total_teams_estimate,
                    estimated_rank_percentile=submission_rank_percentile_estimate,
                    rank_estimate_source=submission_rank_estimate_source,
                    max_percentile=rank_force_major_max_percentile,
                    min_teams=rank_force_major_min_teams,
                    force_major_overhaul_next=rank_forced_major_overhaul,
                ),
                top1_tier_offline_decision=top1_tier,
                top1_tier_by_readiness=top1_tier_by_readiness,
                top1_tier_by_submission=top1_tier_by_submission,
                forced_submit_reason=forced_submit_reason,
                online_score=online_score,
                campaign_payload=campaign_metrics_payload,
                best_score_guard=best_score_guard,
                quality_guard=quality_guard,
                regression_guard=_iteration_metrics.build_regression_guard_payload(
                    best_score_before_iteration=best_score,
                    score_drop_vs_best=score_drop_vs_best,
                    severe_regression_detected=severe_regression_detected,
                    conservative_feature_collapse=conservative_feature_collapse,
                    conservative_regression_detected=conservative_regression_detected,
                    first_iteration_below_code_reference=first_iteration_below_code_reference,
                    code_reference_score=code_reference_score,
                    code_reference_comparison_score=code_reference_comparison_score,
                    code_reference_delta_vs_current=code_reference_delta_vs_current,
                    code_reference_forced_reproduction=code_reference_forced_reproduction,
                ),
            )
            _json_utils.write_json_object(metrics_path, metrics_payload)

            diff_summary = "Diff tracking disabled (git integration removed)."
            diagnostics = _diagnostics.build_diagnostics(
                evaluation=evaluation,
                model_summary=model_summary,
                best_score=best_score,
                target_score=target_score,
                dataset_profile=dataset_profile,
                top1_score=top1_score,
                top1_tier=top1_tier,
                diff_summary=diff_summary,
                evaluation_by_source=evaluation_by_source,
                loop_decision_score=decision_score,
                loop_decision_source=decision_source,
                quality_guard=quality_guard,
                accuracy_potential=accuracy_potential,
            )
            (iter_dir / "diagnostics.md").write_text(diagnostics, encoding="utf-8")

            competition_policy = load_competition_policy(config.paths)
            reference_inputs_manifest_payload = _json_utils.load_json_object(
                config.paths.reference_inputs_manifest_path
            )
            repair_signals = _iteration_signals.collect_iteration_repair_signals(
                kernel_metrics_payload=kernel_metrics_payload,
                diagnostics_text=diagnostics,
                reference_inputs_manifest_payload=reference_inputs_manifest_payload,
                enable_missing_ensemble_signal=competition_policy.repair.missing_ensemble_signal,
                enable_original_data_unused_signal=competition_policy.repair.original_data_unused_signal,
                enable_same_family_plateau_signal=competition_policy.repair.same_family_plateau_signal,
                direction=metric_direction,
                previous_best_offline=best_score,
                current_offline=decision_score,
                previous_best_online=best_online_submission_score,
                current_online=online_score,
                previous_submission_history=previous_submission_history,
                detect_subgroup_collapse_signal=_kernel_quality.detect_subgroup_collapse_signal,
                detect_online_history_regression_signal=(
                    _submission_history.detect_online_regression_vs_submission_history
                ),
            )
            best_online_submission_score = _leaderboard_policy.update_best_online_submission_score(
                current_best_score=best_online_submission_score,
                candidate_score=online_score,
                direction=metric_direction,
            )
            if campaign_mode == "top1" and campaign_candidate is not None:
                campaign_submission_succeeded = _campaign_metrics.campaign_submission_succeeded(
                    submission_result=submission_result,
                    submission_skipped=submission_skipped,
                )
                if campaign_submission_succeeded:
                    campaign_candidate = replace(
                        campaign_candidate,
                        submitted=True,
                        public_score=_campaign_metrics.campaign_public_score_from_online_score(online_score),
                    )
                    upsert_candidate(campaign_registry_file, campaign_candidate)
                campaign_state = update_campaign_state(
                    state_path=campaign_state_file,
                    registry_path=campaign_registry_file,
                    slug=config.slug,
                    run_id=run_id,
                    mode=campaign_mode,
                    direction=metric_direction,
                    top1_info=top1_info if isinstance(top1_info, dict) else {},
                    submission_history=previous_submission_history,
                    latest_public_score=online_score,
                    remaining_daily_slots=remaining_daily_slots,
                    method_registry=method_registry,
                )
                campaign_outcome_phase = _campaign_metrics.campaign_outcome_phase(
                    submission_result=submission_result,
                    submission_skipped=submission_skipped,
                )
                append_campaign_outcome(
                    context_dir=config.paths.context_dir,
                    run_id=run_id,
                    iteration=iteration,
                    phase=campaign_outcome_phase,
                    candidate=campaign_candidate,
                    allocation=campaign_allocation,
                    campaign_state=campaign_state,
                    experiment_graph=experiment_graph,
                )

            prefer_validation_redesign = (
                campaign_mode == "top1"
                and _campaign_metrics.campaign_prefers_validation_redesign(campaign_state, method_registry)
            )
            repair_signal_policy = _iteration_signals.apply_iteration_repair_signal_policy(
                iteration=iteration,
                orig_proba_signal=repair_signals.orig_proba_signal,
                original_data_unused_signal=repair_signals.original_data_unused_signal,
                pseudo_label_signal=repair_signals.pseudo_label_signal,
                missing_ensemble_signal=repair_signals.missing_ensemble_signal,
                same_family_plateau_signal=repair_signals.same_family_plateau_signal,
                subgroup_collapse_signal=repair_signals.subgroup_collapse_signal,
                online_mismatch_signal=repair_signals.online_mismatch_signal,
                online_history_regression_signal=repair_signals.online_history_regression_signal,
                minimum_improvement_mode=medal_minimum_improvement_mode,
                minimum_improvement_reason=medal_policy_reason,
                force_major_overhaul=force_major_overhaul_next,
                forced_major_overhaul_reason=forced_major_overhaul_reason,
                prefer_validation_redesign=prefer_validation_redesign,
                upgrade_improvement_mode=_plan_policy.upgrade_improvement_mode,
            )
            extra_policy_notes = repair_signal_policy.extra_policy_notes
            minimum_improvement_mode_next = repair_signal_policy.minimum_improvement_mode
            minimum_improvement_reason_next = repair_signal_policy.minimum_improvement_reason
            force_major_overhaul_next = repair_signal_policy.force_major_overhaul
            forced_major_overhaul_reason = repair_signal_policy.forced_major_overhaul_reason
            forced_validation_redesign_reason = repair_signal_policy.forced_validation_redesign_reason
            loop_signal_errors = repair_signal_policy.loop_signal_errors
            loop_signal_problems = repair_signal_policy.loop_signal_problems
            if repair_signal_policy.repair_signals is not None:
                metrics_payload["repair_signals"] = repair_signal_policy.repair_signals
            metrics_payload["previous_submission_history"] = previous_submission_history
            metrics_payload["next_iteration_policy"] = repair_signal_policy.next_iteration_policy
            _json_utils.write_json_object(metrics_path, metrics_payload)

            _iteration_signals.record_iteration_repair_signal_knowledge(
                knowledge_paths=config.knowledge_paths,
                slug=config.slug,
                run_id=run_id,
                iteration=iteration,
                problem_types=problem_types,
                loop_signal_errors=loop_signal_errors,
                loop_signal_problems=loop_signal_problems,
                submission_score=online_score,
                record_error_fix_insight=record_error_fix_insight,
                record_problem_type_insight=record_problem_type_insight,
            )

            if writeup_mode:
                writeup_bundle_meta = build_writeup_bundle(
                    paths=config.paths,
                    run_id=run_id,
                    iteration=iteration,
                    resolved=resolved,
                    evaluation=evaluation,
                    metrics_payload=metrics_payload,
                    top1_info=top1_info if isinstance(top1_info, dict) else None,
                )
                metrics_payload["deliverable_mode"] = "writeup"
                metrics_payload["writeup_bundle"] = writeup_bundle_meta
                _json_utils.write_json_object(metrics_path, metrics_payload)

            submit_phase_completion = _iteration_metrics.resolve_iteration_submit_phase_completion(
                submit_enabled=submit_enabled,
                allow_submit=allow_submit,
                submit_phase_required=submit_phase_required,
                submission_result=submission_result,
                submit_failed_deferred=submit_failed_deferred,
            )

            iteration_record_kwargs = _iteration_metrics.build_iteration_record_kwargs(
                knowledge_paths=config.knowledge_paths,
                run_id=run_id,
                iteration=iteration,
                evaluation=evaluation,
                top1_info=top1_info if isinstance(top1_info, dict) else {},
                met_target=met_target,
            )
            record_iteration(**iteration_record_kwargs)
            _autopilot_state._write_iteration_state_marker(
                iter_dir=iter_dir,
                run_id=run_id,
                iteration=iteration,
                submission_path=submission_path,
                metrics_path=metrics_path,
                evaluation_report_path=evaluation_report_path,
                submit_phase_required=submit_phase_required,
                submit_phase_finished=submit_phase_completion.submit_phase_finished,
                submit_allowed_by_gate=submit_phase_completion.submit_allowed_by_gate,
                submit_phase_state=submit_phase_state,
                forced_submit_reason=forced_submit_reason,
                submitted=submission_result is not None and not submission_skipped,
                readiness_score=readiness_score,
            )

            prev_best = best_score
            score_update_decision = _loop_control.decide_iteration_score_update(
                metric_mismatch_detected=metric_mismatch_detected,
                non_generalizable_eval_detected=non_generalizable_eval_detected,
                previous_best_score=prev_best,
                current_score=decision_score,
                submission_path=submission_path,
                no_improve_streak=no_improve_streak,
                stop_min_delta=stop_min_delta,
                conservative_regression_detected=conservative_regression_detected,
                delta_from_best=iteration_phase.delta_from_best,
                should_update_best=iteration_phase.should_update_best,
            )
            delta_offline = score_update_decision.delta_offline
            improved = score_update_decision.improved
            no_improve_streak = score_update_decision.no_improve_streak
            if score_update_decision.best_score is not None:
                best_score = score_update_decision.best_score
            if score_update_decision.best_submission is not None:
                best_submission = score_update_decision.best_submission
            if score_update_decision.capture_best_snapshot:
                _kernel_snapshot.capture_best_kernel_snapshot(paths=config.paths, run_dir=run_dir)
            if score_update_decision.restore_regression_snapshot:
                restored = _kernel_snapshot.restore_best_kernel_snapshot(paths=config.paths, run_dir=run_dir)
                if restored:
                    print(
                        "[yellow]kernel regression guard[/yellow]: "
                        "restored best-known kernel source after severe conservative regression."
                    )

            major_overhaul_decision = _loop_control.decide_no_improve_major_overhaul(
                force_enabled=force_major_on_no_improve,
                improved=improved,
                high_potential_improved=high_potential_improved,
                best_score_guarded=best_score_guard is not None,
                metric_name=evaluation.metric,
                current_score=decision_score,
                previous_best_score=float(prev_best) if prev_best is not None else None,
            )
            if major_overhaul_decision.skip_message:
                print(major_overhaul_decision.skip_message)
            if major_overhaul_decision.force_major_overhaul:
                force_major_overhaul_next = True
                forced_major_overhaul_reason = _loop_control.append_policy_reason(
                    forced_major_overhaul_reason,
                    major_overhaul_decision.reason,
                )

            current_config_hash = _diagnostics.pipeline_config_hash(
                model_summary=model_summary,
                metric=evaluation.metric,
                accelerator=accelerator_used,
            )
            config_streak = _loop_control.update_same_config_streak(
                current_config_hash=current_config_hash,
                last_config_hash=last_config_hash,
                same_config_streak=same_config_streak,
            )
            same_config_streak = config_streak.same_config_streak
            last_config_hash = config_streak.last_config_hash

            stagnation_track = _loop_control.select_stagnation_track(
                best_high_potential_score=best_high_potential_score,
                no_improve_streak=no_improve_streak,
                frontier_no_improve_streak=frontier_no_improve_streak,
            )
            stagnation_stop = _loop_control.decide_stagnation_stop(
                stop_allowed=not submit_enabled,
                no_improve_streak=stagnation_track.no_improve_streak,
                no_improve_patience=stop_no_improve_patience,
                stop_min_delta=stop_min_delta,
                track_label=stagnation_track.label,
                same_config_streak=same_config_streak,
                same_config_patience=stop_same_config_patience,
            )
            if stagnation_stop.should_stop:
                _autopilot_state._apply_run_status(run_payload, status="stopped", stop_reason=stagnation_stop.reason)
                print(f"[yellow]stop[/yellow]: {run_payload['stop_reason']}")
                break

            terminal_stop = _loop_control.decide_terminal_iteration_stop(
                confirmed_first_place=_score_progress.is_confirmed_first_place(
                    submission_rank,
                    submission_rank_source,
                ),
                iteration=iteration,
                max_iterations=max_iterations,
                submitted=submitted,
                allow_max_iteration_stop=False,
            )
            if terminal_stop.should_stop:
                _autopilot_state._apply_run_status(
                    run_payload,
                    status=terminal_stop.status,
                    stop_reason=terminal_stop.stop_reason,
                )
                if terminal_stop.message:
                    print(terminal_stop.message)
                break

            if top1_tier:
                print("[yellow]note[/yellow]: offline top1-tier reached; awaiting submission-score confirmation")

            terminal_stop = _loop_control.decide_terminal_iteration_stop(
                confirmed_first_place=False,
                iteration=iteration,
                max_iterations=max_iterations,
                submitted=submitted,
            )
            if terminal_stop.should_stop:
                _autopilot_state._apply_run_status(
                    run_payload,
                    status=terminal_stop.status,
                    stop_reason=terminal_stop.stop_reason,
                )
                break

            print("[cyan]improve[/cyan]: generating next iteration plan")
            _run_improvement(
                config=config,
                run_id=run_id,
                iteration=iteration,
                iter_dir=iter_dir,
                evaluation=evaluation,
                top1_info=top1_info,
                target_score=target_score,
                delta_offline=delta_offline,
                pending_problem_insights=pending_problem_insights,
                current_score=decision_score,
                current_score_source=decision_source,
                minimum_improvement_mode=minimum_improvement_mode_next,
                minimum_improvement_reason=minimum_improvement_reason_next,
                target_medal=target_medal,
                target_rank_percentile=target_rank_percentile,
                forced_improvement_mode=(
                    "validation_redesign"
                    if forced_validation_redesign_reason and not force_major_overhaul_next
                    else "major_overhaul"
                    if force_major_overhaul_next
                    else None
                ),
                forced_improvement_reason=forced_major_overhaul_reason or forced_validation_redesign_reason,
                extra_policy_notes=extra_policy_notes,
                enforce_code_reference_implementation=code_reference_forced_reproduction,
                code_reference_enforcement_reason=code_reference_force_reason,
                best_score_so_far=best_score,
                previous_submission_history=previous_submission_history,
            )
    except KeyboardInterrupt:
        run_payload["status"] = "interrupted"
        _json_utils.write_json_object(run_dir / "run.json", run_payload)
        print("[yellow]run interrupted[/yellow]")
        return

    fallback_submit_blocked_reason = _submission_policy.resolve_fallback_submit_blocked_reason(
        current_reason=fallback_submit_blocked_reason,
        best_high_potential_meta=best_high_potential_meta if isinstance(best_high_potential_meta, dict) else None,
        best_high_potential_submission=best_high_potential_submission,
        best_submittable_submission=best_submittable_submission,
    )

    if (
        submit_enabled
        and not submitted
        and best_submittable_submission is not None
        and fallback_submit_blocked_reason is None
    ):
        final_iteration_reached = last_completed_iteration >= max_iterations
        fallback_submit_gate = _submit_stage.decide_fallback_submit_gate(
            submit_improved_only=submit_improved_only,
            force_submit=config.force_submit,
            require_submit_improvement=require_submit_improvement,
            best_submittable_score=best_submittable_score,
            best_submitted_score=best_submitted_score,
            direction=metric_direction,
            min_improvement=stop_min_delta,
            final_iteration_reached=final_iteration_reached,
        )
        allow_fallback_submit = fallback_submit_gate.allow_submit
        if fallback_submit_gate.message:
            print(fallback_submit_gate.message)
        if allow_fallback_submit:
            fallback_iteration = _submit_stage.infer_iteration_from_submission_path(best_submittable_submission)
            score_text = (
                f" score={best_submittable_score:.6f}" if isinstance(best_submittable_score, (int, float)) else ""
            )
            if fallback_iteration is not None:
                fallback_label = (
                    f"best competition-faithful artifact from iter {fallback_iteration}/{max_iterations}{score_text}"
                )
                print(f"[cyan]submit[/cyan]: using {fallback_label}.")
            else:
                print(f"[cyan]submit[/cyan]: using best competition-faithful artifact{score_text}.")
            try:
                fallback_result = submission_phase.attempt(
                    submission_path=best_submittable_submission,
                    best_score=best_submittable_score,
                )
            except SubmitAbortedError:
                run_payload["status"] = "submit_failed"
                _json_utils.write_json_object(run_dir / "run.json", run_payload)
                raise
            if fallback_result:
                if bool(fallback_result.get("skipped")):
                    allow_fallback_submit = False
                    print(
                        "[yellow]submit skipped[/yellow]: "
                        f"{fallback_result.get('reason') or 'duplicate submission skipped'}."
                    )
                    fallback_result = None
                else:
                    submitted = True
                    last_submission_result = fallback_result
                    tracking_decision = _submit_stage.decide_submitted_tracking_score_update(
                        submission_result=fallback_result,
                        offline_score=best_submittable_score,
                        previous_best_score=best_submitted_score,
                        direction=metric_direction,
                    )
                    if tracking_decision.update_best_submitted_score:
                        best_submitted_score = tracking_decision.best_submitted_score
        else:
            print(
                "[yellow]submit skipped[/yellow]: fallback artifact is not better "
                "than previously submitted checkpoint score."
            )
    elif submit_enabled and not submitted and fallback_submit_blocked_reason is not None:
        print(
            "[yellow]submit skipped[/yellow]: latest iteration was not competition-faithful "
            f"({fallback_submit_blocked_reason}); refusing fallback submit from an older artifact."
        )
    elif submit_enabled and not submitted and best_submission is not None and best_submittable_submission is None:
        print(
            "[yellow]submit skipped[/yellow]: no competition-faithful fallback artifact "
            "(all candidates were blocked by quality guard)."
        )

    if submitted and last_submission_result:
        top1_score = top1_info.get("score") if isinstance(top1_info, dict) else None

        def load_submission_diagnostics(iteration: int) -> str:
            diagnostics_path = config.paths.iter_dir(run_id, iteration) / "diagnostics.md"
            if diagnostics_path.exists():
                return diagnostics_path.read_text(encoding="utf-8", errors="ignore")
            return ""

        _submit_stage.record_submission_knowledge(
            knowledge_paths=config.knowledge_paths,
            slug=config.slug,
            run_id=run_id,
            problem_types=problem_types,
            pending_problem_insights=pending_problem_insights,
            pending_error_fixes=pending_error_fixes,
            submission_result=last_submission_result,
            metric_direction=metric_direction,
            target_score=target_score,
            top1_score=top1_score if isinstance(top1_score, (int, float)) else None,
            load_diagnostics_text=load_submission_diagnostics,
            record_problem_type_insight=record_problem_type_insight,
            record_error_fix_insight=record_error_fix_insight,
        )
    _autopilot_state._apply_final_run_status(
        run_payload,
        submitted=submitted,
        has_submission_result=bool(last_submission_result),
        writeup_mode=writeup_mode,
        writeup_bundle_meta=writeup_bundle_meta,
    )

    run_payload["summary"] = _autopilot_state._build_run_summary_payload(
        best_score=best_score,
        best_submission=best_submission,
        best_submittable_score=best_submittable_score,
        best_submittable_submission=best_submittable_submission,
        best_high_potential_score=best_high_potential_score,
        best_high_potential_submission=best_high_potential_submission,
        best_high_potential_iteration=best_high_potential_iteration,
        best_high_potential_meta=best_high_potential_meta,
        fallback_submit_blocked_reason=fallback_submit_blocked_reason,
    )

    _json_utils.write_json_object(run_dir / "run.json", run_payload)


def _resolve_plan(plan: PlanConfig, config: AutopilotConfig) -> dict[str, object]:
    return _plan_resolution.resolve_plan_for_autopilot(
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
        default_strict_competition_metric=_DEFAULT_STRICT_COMPETITION_METRIC,
        default_target_medal=_DEFAULT_TARGET_MEDAL,
        default_limited_submission_gate=_DEFAULT_LIMITED_SUBMISSION_GATE,
        default_max_iterations=_DEFAULT_MAX_ITERATIONS,
        heavy_local_gpu_max_cv_folds=_HEAVY_LOCAL_GPU_MAX_CV_FOLDS,
        long_local_gpu_iteration_budget_min=_LONG_LOCAL_GPU_ITERATION_BUDGET_MIN,
        long_local_gpu_max_iterations=_LONG_LOCAL_GPU_MAX_ITERATIONS,
        default_force_major_rank_max_percentile=_DEFAULT_FORCE_MAJOR_RANK_MAX_PERCENTILE,
        default_force_major_rank_min_teams=_DEFAULT_FORCE_MAJOR_RANK_MIN_TEAMS,
        on_message=print,
    )


def _run_verify(verify_cmd: str, *, dry_run: bool, artifacts_dir: Path | None = None) -> None:
    _verify_artifacts.run_verify(
        verify_cmd,
        dry_run=dry_run,
        artifacts_dir=artifacts_dir,
        repo_root=Path.cwd(),
        run_command_fn=run_command,
    )


def _run_plan_and_initial(config: AutopilotConfig, run_id: str) -> None:
    print(f"[cyan]plan[/cyan]: {planning_flow_summary()}")
    _watch_state.update_watch_phase(
        config,
        run_id,
        "gpt_planning",
        detail=planning_flow_summary(),
    )
    planning_campaign_mode = normalize_campaign_mode(config.campaign_mode, deliverable_mode="leaderboard")
    pipeline_config = AgentPipelineConfig(
        slug=config.slug,
        competition_url=config.competition_url,
        compute=config.compute,
        accelerator=config.accelerator,
        internet=str(config.internet or "auto"),
        run_id=run_id,
        dry_run=config.dry_run,
        repo_root=config.paths.repo_root,
        method_scout=_method_scout.effective_method_scout_mode(
            requested_mode=config.method_scout,
            campaign_mode=planning_campaign_mode,
        ),
        method_scout_max_sources=int(config.method_scout_max_sources or 12),
        hardware_profile=config.hardware_profile,
        time_budget_min=config.time_budget_min,
    )
    run_agent_pipeline(paths=config.paths, config=pipeline_config)
    _watch_state.update_watch_phase(
        config,
        run_id,
        "verifying",
        detail="Verifying the generated plan and kernel scaffold.",
    )
    _run_verify(config.verify_cmd, dry_run=config.dry_run, artifacts_dir=config.paths.artifacts_dir)


def _run_kernel_source_preflight_fixes(
    *,
    config: AutopilotConfig,
    run_id: str,
    iteration: int,
    iter_dir: Path,
    pending_error_fixes: list[dict[str, object]] | None = None,
) -> None:
    """Fix deterministic kernel source issues before launching a kernel run."""
    attempt = 0
    while True:
        preflight_error = kernel_source_preflight_error(
            config.paths.kernel_source_dir,
            require_kaggle_input=False,
            format_error=_kernel_errors.format_kernel_error,
        )
        if preflight_error is None:
            return
        lowered = preflight_error.lower()
        if "requires kernel.py" in lowered:
            message = preflight_error
            if message.startswith("RuntimeError:"):
                message = message.split(":", 1)[1].strip()
            raise RuntimeError(message)
        attempt += 1
        if config.dry_run:
            raise KernelFailedError(preflight_error)
        if attempt > _MAX_KERNEL_PREFLIGHT_FIX_ATTEMPTS:
            raise KernelFailedError(f"Kernel source preflight failed after automatic fixes.\n{preflight_error}")
        print(
            "[yellow]kernel preflight[/yellow]: source contract check failed; "
            f"invoking {IMPLEMENTATION_AGENT.log_alias} fix (attempt {attempt}/{_MAX_KERNEL_PREFLIGHT_FIX_ATTEMPTS})"
        )
        _run_kernel_fix(
            config=config,
            run_id=run_id,
            iteration=iteration,
            iter_dir=iter_dir,
            error_message=preflight_error,
            attempt=attempt,
            pending_error_fixes=pending_error_fixes,
        )


def _run_improvement(
    config: AutopilotConfig,
    run_id: str,
    iteration: int,
    iter_dir: Path,
    evaluation: EvaluationResult,
    top1_info: dict[str, object],
    target_score: float,
    delta_offline: float | None,
    pending_problem_insights: list[dict[str, object]],
    current_score: float | None = None,
    current_score_source: str = "offline",
    minimum_improvement_mode: str | None = None,
    minimum_improvement_reason: str | None = None,
    target_medal: str | None = None,
    target_rank_percentile: float | None = None,
    forced_improvement_mode: str | None = None,
    forced_improvement_reason: str | None = None,
    extra_policy_notes: list[str] | None = None,
    enforce_code_reference_implementation: bool = False,
    code_reference_enforcement_reason: str | None = None,
    best_score_so_far: float | None = None,
    previous_submission_history: dict[str, object] | None = None,
) -> None:
    prompt_template = render_prompt_identity(config.paths.codex_improve_template.read_text(encoding="utf-8"))
    agent_dir = iter_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = agent_dir / "prompt.md"
    run_dir = config.paths.run_dir(run_id)
    (
        submit_failure_notes,
        submit_failure_force_reason,
    ) = _submit_failure_context.build_submit_failure_improvement_context(
        failure_context=_submit_failure_context.load_submit_failure_context(run_dir),
        latest_submit_attempt=_submit_attempts.load_latest_submit_attempt(run_dir),
    )
    top1_score = top1_info.get("score") if isinstance(top1_info, dict) else None
    effective_current_score = evaluation.value if current_score is None else current_score
    improvement_mode, top1_gap = _score_progress.classify_improvement_mode(
        effective_current_score,
        top1_score,
        evaluation.direction,
    )
    upgraded_mode = _plan_policy.upgrade_improvement_mode(improvement_mode, minimum_improvement_mode)
    if upgraded_mode != improvement_mode:
        print(
            "[yellow]improve mode floor[/yellow]: "
            f"{improvement_mode} -> {upgraded_mode} ({minimum_improvement_reason or 'policy'})"
        )
        improvement_mode = upgraded_mode
    if forced_improvement_mode:
        print(
            "[yellow]improve mode override[/yellow]: "
            f"{improvement_mode} -> {forced_improvement_mode} ({forced_improvement_reason or 'policy'})"
        )
        improvement_mode = forced_improvement_mode
    kernel_main_path = config.paths.kernel_source_dir / "kernel.py"
    code_reference_score, code_reference_source = _code_reference.extract_code_reference_score(config.paths)
    code_reference_comparison_score = _score_progress.normalize_code_reference_score_for_comparison(
        current=effective_current_score,
        reference=code_reference_score,
        metric=evaluation.metric,
    )
    code_reference_delta = (
        _score_progress.score_delta_vs_reference(
            effective_current_score,
            code_reference_comparison_score,
            evaluation.direction,
        )
        if code_reference_comparison_score is not None
        else None
    )
    code_reference_underperforming = bool(
        code_reference_score is not None and code_reference_delta is not None and code_reference_delta < 0
    )
    if code_reference_score is None:
        code_reference_status = "code_reference_unavailable"
    elif code_reference_underperforming:
        code_reference_status = "underperforming_code_reference"
    else:
        code_reference_status = "at_or_above_code_reference"
    required_reference_notebook = _code_reference.load_required_reference_notebook(config.paths)
    ensemble_reference_notebook = _code_reference.load_ensemble_reference_notebook(config.paths)
    competition_policy = load_competition_policy(config.paths)
    base_prompt_text = prompt_template.format(
        **prompt_identity_format_args(),
        slug=config.slug,
        iteration=iteration,
        plan_path=str(config.paths.plan_path),
        run_path=str(config.paths.run_dir(run_id) / "run.json"),
        metrics_path=str(iter_dir / "metrics.json"),
        diagnostics_path=str(iter_dir / "diagnostics.md"),
        logs_dir=str(iter_dir / "logs"),
        compute=config.compute,
        accelerator=config.accelerator,
        knowledge_hints=str(config.paths.knowledge_hints_path),
        metric=evaluation.metric,
        direction=evaluation.direction,
        current_score=f"{effective_current_score:.6f}",
        current_score_source=current_score_source,
        target_score=f"{target_score:.6f}",
        top1_score=str(top1_score or "unavailable"),
        top1_source=str(top1_info.get("source") or "unknown"),
        top1_gap="unavailable" if top1_gap is None else f"{top1_gap:.6f}",
        delta_offline="unavailable" if delta_offline is None else f"{delta_offline:.6f}",
        improvement_mode=improvement_mode,
        next_iteration=str(iteration + 1),
        rules_url=str(config.paths.rules_url_path),
        rules_md=str(config.paths.rules_md_path),
        rules_html=str(config.paths.rules_html_path),
        overview_md=str(config.paths.overview_md_path),
        data_md=str(config.paths.data_md_path),
        submission_format=str(config.paths.submission_format_md_path),
        dataset_profile=str(config.paths.dataset_profile_path),
        sample_submission=str(config.paths.sample_submission_path),
        code_md=str(config.paths.code_md_path),
        code_index=str(config.paths.code_notebooks_index_path),
        code_reference_score=("unavailable" if code_reference_score is None else f"{code_reference_score:.6f}"),
        code_reference_source=code_reference_source,
        code_reference_delta=("unavailable" if code_reference_delta is None else f"{code_reference_delta:+.6f}"),
        code_reference_status=code_reference_status,
        kernel_main=str(kernel_main_path),
    )
    if infer_deliverable_mode_from_paths(config.paths) == "writeup":
        base_prompt_text += (
            "\n\nWriteup mode is active for this competition.\n"
            "Do not optimize only for submission.csv production. Treat offline metrics and any CSV artifacts as "
            "proxy evidence supporting the final judged writeup package.\n"
        )
    if forced_improvement_reason:
        base_prompt_text += (
            "\n\nForced improvement mode policy is active.\n"
            f"Reason: {forced_improvement_reason}\n"
            "Do not propose minor_tuning; follow the forced improvement mode.\n"
        )
        if forced_improvement_mode == "validation_redesign":
            base_prompt_text += (
                "Mode is validation_redesign: first build and compare group/time/leak/proxy split candidates, "
                "calibrate against previous public outcomes, and only then rank new model-family changes.\n"
            )
    elif minimum_improvement_reason:
        base_prompt_text += (
            "\n\nMinimum improvement mode policy is active.\n"
            f"Reason: {minimum_improvement_reason}\n"
            "Do not propose minor_tuning while this policy remains active.\n"
        )
    if improvement_mode == "validation_redesign":
        base_prompt_text += (
            "\n\nValidation redesign campaign policy:\n"
            "- Treat online regression or low offline-online correlation as a split problem first.\n"
            "- Create validation_variant candidates for group, time, leak-safe, and proxy/adversarial splits.\n"
            "- Do not submit another model-only candidate until the active validation profile is justified.\n"
        )
    if target_rank_percentile is not None:
        medal_label = target_medal or "rank"
        base_prompt_text += (
            "\n\nMedal-aware search policy:\n"
            f"- target_medal: {medal_label}\n"
            f"- target_rank_percentile: {target_rank_percentile * 100:.2f}%\n"
            "- Until this leaderboard percentile is reached, keep search breadth high and "
            "avoid same-family-only tweaks.\n"
        )
    if _iteration_signals.requires_tabular_multi_family_policy(
        _context_artifacts.load_dataset_profile(
            slug=config.paths.slug,
            dataset_profile_path=config.paths.dataset_profile_path,
        )
    ):
        base_prompt_text += (
            "\n\nHigh-accuracy tabular policy is active.\n"
            "- This dataset is tabular binary with meaningful categorical structure.\n"
            "- The next iteration must keep multi-family exploration active.\n"
            "- Require CatBoost raw categorical, XGBoost with leak-safe target/stat encodings, "
            "and LightGBM or a second CatBoost/XGBoost variant.\n"
            "- If two or more model pipelines exist, require at least one OOF-based blend "
            "candidate (weighted/rank/logit blend).\n"
        )
    if competition_policy.active:
        policy_lines = ["\n\nCompetition policy override is active."]
        if competition_policy.required_capabilities:
            policy_lines.append(
                "- Required capabilities: "
                + ", ".join(capability for capability in competition_policy.required_capabilities if capability)
            )
        if competition_policy.has_capability("recoverable_original_dataset"):
            policy_lines.append(
                "- If staged reference/original datasets are available, wire them into training or feature "
                "generation instead of leaving them unused."
            )
        if competition_policy.has_capability("heterogeneous_tabular_ensemble"):
            policy_lines.append(
                "- Keep orthogonal model families active; do not spend the next iteration on same-family-only tuning."
            )
        if competition_policy.has_capability("requires_oof_blend"):
            policy_lines.append(
                "- Persist OOF predictions for each candidate and emit at least one weighted or rank blend artifact."
            )
        if competition_policy.has_capability("text_translation_seq2seq"):
            policy_lines.append(
                "- For translation/text seq2seq tasks, prefer reusable helpers from "
                "`src/kagglebot/kernel_runtime/text_translation.py` for normalization, metrics, MBR, retrieval, "
                "and consistency logic; keep competition-specific joins and dictionaries in `kernel.py`."
            )
        if competition_policy.has_capability("requires_grouped_text_cv"):
            policy_lines.append(
                "- Use grouped text CV keyed by the plan/runtime group columns; "
                "do not rank candidates with plain row-level splits."
            )
        if competition_policy.has_capability("requires_candidate_rerank"):
            policy_lines.append(
                "- Treat retrieval as a candidate source or fallback only; "
                "keep seq2seq + candidate rerank/MBR as the primary path."
            )
        if competition_policy.has_capability("supports_metadata_supervision"):
            policy_lines.append(
                "- If metadata supervision is useful, declare required aux inputs in "
                "plan.json `text_runtime.required_aux_inputs` "
                "and keep the matching/join heuristics inside `kernel.py`."
            )
        if competition_policy.has_capability("supports_soft_constraint_rewrite"):
            policy_lines.append(
                "- Prefer soft constraint rewrites and rerank bonuses for "
                "entity/quantity/unit handling instead of hard-coded decode constraints."
            )
        if competition_policy.prompt.ablation_groups:
            policy_lines.append(
                "- Required ablations: "
                + ", ".join(group for group in competition_policy.prompt.ablation_groups if group)
            )
        if competition_policy.prompt.min_model_families_before_stop is not None:
            policy_lines.append(
                f"- Minimum model families before stop: {competition_policy.prompt.min_model_families_before_stop}"
            )
        if competition_policy.prompt.require_oof_blend_before_stop:
            policy_lines.append("- Do not stop until at least one OOF blend candidate is implemented.")
        if competition_policy.evaluation.search_stop_rank_percentile is not None:
            policy_lines.append(
                "- Internal search target rank percentile: "
                f"{competition_policy.evaluation.search_stop_rank_percentile * 100:.2f}%"
            )
        if competition_policy.prompt.prefer_ensemble_reference and ensemble_reference_notebook is not None:
            policy_lines.append(f"- ensemble_kernel_id: {ensemble_reference_notebook.kernel_id}")
        if competition_policy.execution_hints:
            policy_lines.append(
                "- execution_hints: "
                + json.dumps(competition_policy.execution_hints, sort_keys=True, ensure_ascii=True)
            )
        for note in competition_policy.prompt.extra_notes:
            policy_lines.append(f"- {note}")
        base_prompt_text += "\n".join(policy_lines) + "\n"
    if best_score_so_far is not None:
        base_prompt_text += (
            "\n\nRegression Guard Policy:\n"
            f"- Best known offline score so far: {float(best_score_so_far):.6f}\n"
            "- Do NOT introduce conservative fallback paths that intentionally reduce model capacity "
            "or collapse features (e.g., tiny robust subsets) when they materially degrade offline quality.\n"
            "- If suspiciously high CV is detected, keep leak fixes but preserve competitive model strength "
            "instead of defaulting to a weak baseline.\n"
        )
    history_prompt = _submission_history.format_previous_submission_history_for_prompt(previous_submission_history)
    if history_prompt:
        base_prompt_text += "\n\nPrevious Kaggle Submission Results:\n" + history_prompt + "\n"
    method_registry_payload = _json_utils.load_json_object(config.paths.method_registry_path)
    method_prompt = (
        _method_scout.render_method_registry_for_prompt(method_registry_payload, max_methods=8)
        if isinstance(method_registry_payload, dict)
        else ""
    )
    if method_prompt:
        base_prompt_text += "\n\nCompetition-Specific Method Scout:\n" + method_prompt + "\n"
    if extra_policy_notes:
        note_lines = []
        for note in extra_policy_notes:
            clean = str(note).strip()
            if clean:
                note_lines.append(f"- {clean}")
        if note_lines:
            base_prompt_text += "\n\nAdditional repair targets:\n" + "\n".join(note_lines) + "\n"
    if submit_failure_notes:
        base_prompt_text += (
            "\n\nSubmit Contract Repair:\n" + "\n".join(f"- {note}" for note in submit_failure_notes) + "\n"
        )
        if submit_failure_force_reason:
            base_prompt_text += (
                "\nSubmit contract repair policy is active.\n"
                f"Reason: {submit_failure_force_reason}\n"
                "Repair the submission contract before spending iteration budget on further model tuning.\n"
            )
    code_reference_gate_lines = [
        "## Code Reference Gate",
        f"- Code snapshot: {config.paths.code_md_path}",
        f"- Code notebook index: {config.paths.code_notebooks_index_path}",
        (
            "- Code reference score: unavailable"
            if code_reference_score is None
            else (
                f"- Code reference score: {code_reference_score:.6f} "
                f"(comparison_score={code_reference_comparison_score:.6f}, "
                f"source: {code_reference_source}, delta_vs_current={code_reference_delta:+.6f})"
            )
        ),
        f"- Code reference status: {code_reference_status}",
    ]
    code_reference_mandatory = bool(code_reference_underperforming or enforce_code_reference_implementation)
    if code_reference_mandatory:
        code_reference_gate_lines.extend(
            [
                "",
                (
                    "Current score is below the code reference baseline."
                    if code_reference_underperforming
                    else "Code reference implementation is policy-mandatory for the next iteration."
                ),
                (
                    f"Enforcement reason: {code_reference_enforcement_reason}"
                    if code_reference_enforcement_reason
                    else "Enforcement reason: code-reference policy"
                ),
                "You MUST inspect code.md and code_notebooks_index.json and treat",
                "`Required Reference Notebook (Execution baseline)` as mandatory baseline context.",
            ]
        )
        if required_reference_notebook is not None:
            code_reference_gate_lines.extend(
                [
                    f"- required_kernel_id: {required_reference_notebook.kernel_id}",
                    f"- required_title: {required_reference_notebook.title}",
                    (
                        f"- required_source_file: {required_reference_notebook.source_file}"
                        if required_reference_notebook.source_file
                        else "- required_source_file: unavailable"
                    ),
                    (
                        f"- required_local_dir: {required_reference_notebook.local_dir}"
                        if required_reference_notebook.local_dir
                        else "- required_local_dir: unavailable"
                    ),
                    f"- required_marker: {_code_reference.code_reference_marker(required_reference_notebook)}",
                    (
                        "- required_model_family: tabicl"
                        if _code_reference.reference_requires_tabicl(required_reference_notebook)
                        else "- required_model_family: follow required notebook strategy"
                    ),
                ]
            )
        if ensemble_reference_notebook is not None and competition_policy.prompt.prefer_ensemble_reference:
            code_reference_gate_lines.extend(
                [
                    f"- ensemble_kernel_id: {ensemble_reference_notebook.kernel_id}",
                    f"- ensemble_title: {ensemble_reference_notebook.title}",
                    (
                        f"- ensemble_source_file: {ensemble_reference_notebook.source_file}"
                        if ensemble_reference_notebook.source_file
                        else "- ensemble_source_file: unavailable"
                    ),
                    "After reproducing the execution baseline, inspect the ensemble reference notebook "
                    "as the blend blueprint.",
                ]
            )
        code_reference_gate_lines.extend(
            [
                "Either reproduce that baseline path first or justify concrete blockers and implement",
                "the closest leak-free fallback in kernel.py.",
                "When implementing the required notebook path, add the exact marker comment shown above.",
            ]
        )
    base_prompt_text += "\n\n" + "\n".join(code_reference_gate_lines) + "\n"
    problem_type_knowledge = _knowledge_context.load_problem_type_knowledge_text(
        dataset_profile_path=config.paths.dataset_profile_path,
        knowledge_paths=config.knowledge_paths,
        include_research=False,
        unavailable_message="Problem-type knowledge unavailable: {error}",
    )
    hardware_profile = resolve_hardware_profile(config.hardware_profile, compute=config.compute)
    strategy_prompt = _agent_prompts.build_improvement_strategy_prompt(
        slug=config.slug,
        run_id=run_id,
        iteration=iteration,
        metric=evaluation.metric,
        direction=evaluation.direction,
        current_score=effective_current_score,
        current_score_source=current_score_source,
        target_score=target_score,
        top1_score=top1_score,
        top1_source=str(top1_info.get("source") or "unknown"),
        top1_gap=top1_gap,
        delta_offline=delta_offline,
        improvement_mode=improvement_mode,
        hardware_constraints=render_hardware_constraints(
            hardware_profile,
            compute=config.compute,
            time_budget_min=config.time_budget_min,
        ),
        codex_prompt=base_prompt_text,
        problem_type_knowledge=problem_type_knowledge,
    )
    strategy_dir = agent_dir / f"improve_strategy-{iteration:02d}"
    _watch_state.update_watch_phase(
        config,
        run_id,
        "gpt_improvement_thinking",
        detail="GPT is drafting the next improvement strategy.",
        iteration=iteration,
    )
    strategy_text = _run_improvement_strategy(
        prompt_text=strategy_prompt,
        output_dir=strategy_dir,
        dry_run=config.dry_run,
    )

    prompt_text = base_prompt_text
    if strategy_text:
        prompt_text = (
            f"# {IMPLEMENTATION_AGENT.display_name} Improvement Implementation\n\n"
            f"Implement the {STRATEGY_AGENT.display_name}-authored improvement prompt below as the primary plan.\n\n"
            f"## {STRATEGY_AGENT.display_name} Extra-High Improvement Prompt\n"
            f"{strategy_text}\n\n"
            "## Local Context (for file paths and constraints)\n"
            f"{base_prompt_text}\n"
        )

    prompt_path.write_text(prompt_text, encoding="utf-8")
    _agent_io.print_agent_prompt(
        log_alias=IMPLEMENTATION_AGENT.log_alias,
        prompt_path=prompt_path,
        prompt_text=prompt_text,
    )

    print(f"[cyan]improve[/cyan]: running {IMPLEMENTATION_AGENT.log_alias} implementer")
    _watch_state.update_watch_phase(
        config,
        run_id,
        "gpt_improvement_fixing",
        detail=f"{IMPLEMENTATION_AGENT.display_name} is applying the improvement strategy.",
        iteration=iteration,
    )
    # Codex runner always writes execution logs (codex_exec.jsonl / codex_last_message.txt)
    # under the provided output_dir (agent_dir). Include it in the allowlist so the guard
    # does not fail on its own transcripts.
    #
    # During improvement iterations we also update competition context (e.g. leaderboard snapshots,
    # eval advisor status) and run-scoped metadata (run.json, evaluation_report.json). Those are
    # side effects of Kagglebot itself rather than agent edits, so they must be allowlisted here
    # to avoid spurious write-guard failures.
    allowed_prefixes = build_repair_write_policy(
        repo_root=config.paths.repo_root,
        data_dir=config.paths.data_dir,
        kernels_dir=config.paths.kernels_dir,
        module_file=Path(__file__),
        extra_allowed_prefixes=[agent_dir],
    )

    def _run_improve_codex_pass(*, current_prompt_path: Path, stage_suffix: str) -> tuple[str, Path]:
        capacity_attempts = max(
            1,
            _env_utils.env_int("KAGGLEBOT_AGENT_CAPACITY_ATTEMPTS", default=MAX_AGENT_CAPACITY_ATTEMPTS),
        )
        for capacity_attempt in range(1, capacity_attempts + 1):
            pass_output_dir = (
                agent_dir
                if capacity_attempt == 1
                else agent_dir / f"improve_capacity_retry{stage_suffix}-{capacity_attempt:02d}"
            )
            guard_snapshot = _backup_guarded_files(config.paths.repo_root, allowed_prefixes)
            before = _snapshot_tree(config.paths.repo_root)
            result = run_codex(
                current_prompt_path,
                pass_output_dir,
                dry_run=config.dry_run,
                model=_ERROR_FIX_CODEX_MODEL,
                reasoning_effort=_ERROR_FIX_REASONING_EFFORT,
            )
            after = _snapshot_tree(config.paths.repo_root)
            _enforce_allowlist_changes(
                root=config.paths.repo_root,
                before=before,
                after=after,
                allowed_prefixes=allowed_prefixes,
                stage=f"improve_iteration_{iteration}{stage_suffix}",
                guard_snapshot=guard_snapshot,
                auto_repair=True,
            )
            response = _agent_io.read_agent_response(result.last_message_path)
            _agent_io.print_agent_response(
                log_alias=IMPLEMENTATION_AGENT.log_alias,
                response_path=result.last_message_path,
                response_text=response,
            )
            _agent_io.log_codex_sandbox_fallback(stage_label="improve", result=result)
            if result.returncode == 0:
                return response, result.last_message_path

            detail = _agent_io.agent_failure_detail(result, response)
            if _agent_io.is_agent_capacity_failure(result, response):
                if capacity_attempt < capacity_attempts:
                    wait_seconds = AGENT_CAPACITY_RETRY_SLEEP * capacity_attempt
                    print(
                        "[yellow]improve[/yellow]: "
                        f"{IMPLEMENTATION_AGENT.log_alias} capacity unavailable; "
                        f"retrying in {wait_seconds:.0f}s "
                        f"(attempt {capacity_attempt + 1}/{capacity_attempts})."
                    )
                    time.sleep(wait_seconds)
                    continue
                raise RuntimeError(
                    f"{IMPLEMENTATION_AGENT.display_name} improvement failed: transient_agent_capacity\n{detail}"
                )
            raise RuntimeError(f"{IMPLEMENTATION_AGENT.display_name} improvement failed.\n{detail}")
        raise RuntimeError(f"{IMPLEMENTATION_AGENT.display_name} improvement failed.")

    _watch_state.update_watch_phase(
        config,
        run_id,
        "gpt_improvement_fixing",
        detail=f"{IMPLEMENTATION_AGENT.display_name} is editing kernel code for the next iteration.",
        iteration=iteration,
    )
    response_text, _ = _run_improve_codex_pass(current_prompt_path=prompt_path, stage_suffix="")

    if code_reference_mandatory and required_reference_notebook is not None and not config.dry_run:
        kernel_path = config.paths.kernel_source_dir / "kernel.py"
        implementation_issues = _code_reference.validate_code_reference_implementation(
            kernel_path=kernel_path,
            reference=required_reference_notebook,
        )
        if implementation_issues:
            print(
                "[yellow]code reference guard[/yellow]: "
                "required reference implementation missing; rerunning "
                f"{IMPLEMENTATION_AGENT.log_alias} with strict repair prompt."
            )
            repair_prompt_path = agent_dir / f"code_reference_repair_prompt-{iteration:02d}.md"
            repair_prompt_text = _agent_prompts.build_code_reference_repair_prompt(
                base_prompt_text=base_prompt_text,
                reference=required_reference_notebook,
                issues=implementation_issues,
                kernel_path=kernel_path,
            )
            repair_prompt_path.write_text(repair_prompt_text, encoding="utf-8")
            _agent_io.print_agent_prompt(
                log_alias=IMPLEMENTATION_AGENT.log_alias,
                prompt_path=repair_prompt_path,
                prompt_text=repair_prompt_text,
            )
            repair_response, _ = _run_improve_codex_pass(
                current_prompt_path=repair_prompt_path,
                stage_suffix="_code_reference_repair",
            )
            implementation_issues = _code_reference.validate_code_reference_implementation(
                kernel_path=kernel_path,
                reference=required_reference_notebook,
            )
            if implementation_issues:
                issues_text = ", ".join(implementation_issues)
                raise RuntimeError(
                    f"Code reference implementation requirement not satisfied after repair pass (issues={issues_text})."
                )
            response_text = f"{response_text}\n\n{repair_response}".strip()

    _run_verify(config.verify_cmd, dry_run=config.dry_run, artifacts_dir=config.paths.artifacts_dir)
    summary = response_text
    diagnostics_text = ""
    diagnostics_path = iter_dir / "diagnostics.md"
    if diagnostics_path.exists():
        diagnostics_text = diagnostics_path.read_text(encoding="utf-8", errors="ignore")
    record_improvement(
        knowledge_paths=config.knowledge_paths,
        run_id=run_id,
        iteration=iteration,
        summary=summary.strip(),
        delta_offline=delta_offline,
    )
    pending_problem_insights.append(
        {
            "iteration": iteration,
            "why_poor": diagnostics_text,
            "how_improved": strategy_text or summary,
            "delta_offline": delta_offline,
        }
    )


def _run_improvement_strategy(*, prompt_text: str, output_dir: Path, dry_run: bool) -> str:
    return _agent_strategy.run_strategy_prompt(
        prompt_text=prompt_text,
        output_dir=output_dir,
        dry_run=dry_run,
        config=_agent_strategy.StrategyPromptRunConfig(
            prompt_filename="gpt_improvement_prompt.md",
            start_message="[cyan]improve[/cyan]: gpt drafting improvement prompt",
            failure_message=(
                f"[yellow]improve[/yellow]: gpt improvement strategy failed, "
                f"falling back to direct {IMPLEMENTATION_AGENT.log_alias} prompt"
            ),
            empty_message=(
                f"[yellow]improve[/yellow]: gpt improvement strategy empty, "
                f"falling back to direct {IMPLEMENTATION_AGENT.log_alias} prompt"
            ),
        ),
        run_strategy_func=run_strategy,
    )


def _run_kernel_fix(
    *,
    config: AutopilotConfig,
    run_id: str,
    iteration: int,
    iter_dir: Path,
    error_message: str,
    attempt: int,
    pending_error_fixes: list[dict[str, object]] | None = None,
    use_gpt_strategy: bool = True,
    codex_model: str | None = None,
    codex_reasoning_effort: str | None = None,
    prompt_prefix: str = "",
    max_codex_passes: int | None = None,
) -> None:
    agent_dir = iter_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    lightweight_note_path = agent_dir / f"kernel_fix_note-{attempt:02d}.txt"
    lightweight_fix = _maybe_apply_lightweight_runtime_fix(
        config=config,
        error_text=error_message,
        note_path=lightweight_note_path,
        stage_label="kernel fix",
    )
    if lightweight_fix:
        if pending_error_fixes is not None:
            pending_error_fixes.append(
                {
                    "iteration": iteration,
                    "error_message": error_message,
                    "fix_summary": f"Applied lightweight runtime autofix: {lightweight_fix}",
                    "resolved": True,
                }
            )
        return

    prompt_template = render_prompt_identity(config.paths.codex_kernel_fix_template.read_text(encoding="utf-8"))
    prompt_path = agent_dir / "kernel_fix_prompt.md"
    missing_module = _runtime_fixes.extract_missing_module(error_message)
    blocked_modules = _runtime_fixes.load_blocked_modules(config.paths.context_dir)
    if missing_module:
        # Keep dependency recovery paths open: do not auto-block newly missing modules.
        blocked_modules = [name for name in blocked_modules if name != missing_module]
        _runtime_fixes.save_blocked_modules(config.paths.context_dir, blocked_modules)
    blocked_text = "\n".join(f"- {name}" for name in blocked_modules) if blocked_modules else "None"
    prompt_text = prompt_template.format(
        **prompt_identity_format_args(),
        slug=config.slug,
        run_id=run_id,
        iteration=iteration,
        compute=config.compute,
        accelerator=config.accelerator,
        error_message=error_message,
        blocked_modules=blocked_text,
        logs_dir=str(iter_dir / "logs"),
        kernel_main=str(config.paths.kernel_source_dir / "kernel.py"),
        kernel_script=str(config.paths.kernel_run_dir(run_id) / "kernel.py"),
        rules_url=str(config.paths.rules_url_path),
        rules_md=str(config.paths.rules_md_path),
        overview_md=str(config.paths.overview_md_path),
        data_md=str(config.paths.data_md_path),
        submission_format=str(config.paths.submission_format_md_path),
        dataset_profile=str(config.paths.dataset_profile_path),
        sample_submission=str(config.paths.sample_submission_path),
    )
    subgroup_metrics_path = iter_dir / "output" / "metrics.json"
    subgroup_payload = _json_utils.load_json_object(subgroup_metrics_path) if subgroup_metrics_path.exists() else {}
    subgroup_collapse_signal = _kernel_quality.detect_subgroup_collapse_signal(
        kernel_metrics_payload=subgroup_payload if isinstance(subgroup_payload, dict) else None,
        direction="minimize",
    )
    if subgroup_collapse_signal is not None:
        prompt_text = (
            "Subgroup repair target:\n"
            f"- {subgroup_collapse_signal['note']}\n"
            "- Prefer subgroup-aware fixes over global retuning.\n"
            "- If selection or fallback logic is coarse, refine it to (model_id,node_type) granularity.\n\n"
            + prompt_text
        )
    if missing_module:
        prompt_text = (
            f"Missing dependency detected: {missing_module}\n"
            "Guard/disable only this missing package path. Keep actively using other available "
            "repo dependencies (torch/timm/torchvision/opencv/xgboost/lightgbm/catboost/"
            "transformers/tabicl/ultralytics/sklearn) and avoid silent capacity downgrades. "
            "If this package is required, add it via `uv add <package>` and update `pyproject.toml` "
            "+ `uv.lock`.\n\n" + prompt_text
        )
    if prompt_prefix.strip():
        prompt_text = f"{prompt_prefix.strip()}\n\n{prompt_text}"

    strategy_text = ""
    strategy_skip_reason: str | None = None
    if not use_gpt_strategy:
        strategy_skip_reason = "metric_fix_policy"
    else:
        strategy_skip_reason = _runtime_fixes.error_strategy_skip_reason(stage="kernel_fix", error_text=error_message)
    if strategy_skip_reason:
        print(
            "[yellow]kernel fix[/yellow]: "
            f"skipping gpt strategy ({strategy_skip_reason}); invoking {IMPLEMENTATION_AGENT.log_alias} fixer directly."
        )
    else:
        _watch_state.update_watch_phase(
            config,
            run_id,
            "gpt_kernel_fix_thinking",
            detail="GPT is analyzing the kernel failure and drafting a fix strategy.",
            iteration=iteration,
        )
        strategy_prompt = _agent_prompts.build_error_strategy_prompt(
            stage="kernel_fix",
            slug=config.slug,
            run_id=run_id,
            attempt=attempt,
            compute=config.compute,
            accelerator=config.accelerator,
            hardware_constraints=render_hardware_constraints(
                resolve_hardware_profile(config.hardware_profile, compute=config.compute),
                compute=config.compute,
                time_budget_min=config.time_budget_min,
            ),
            error_text=error_message,
            codex_prompt=prompt_text,
        )
        strategy_dir = agent_dir / f"kernel_fix_strategy-{attempt:02d}"
        strategy_text = _run_error_strategy(
            prompt_text=strategy_prompt,
            output_dir=strategy_dir,
            dry_run=config.dry_run,
            stage_label="kernel fix",
        )
    if strategy_text:
        prompt_text += (
            f"\n\n## {STRATEGY_AGENT.display_name} Extra-High Error-Fix Strategy\n"
            "Use the strategy below as guidance, then apply minimal targeted edits.\n\n"
            f"{strategy_text}\n"
        )

    base_prompt_text = f"Kernel fix attempt: {attempt}\n\n{prompt_text}"
    prompt_path.write_text(base_prompt_text, encoding="utf-8")
    attempt_path = agent_dir / f"kernel_fix_prompt-{attempt:02d}.md"
    attempt_path.write_text(base_prompt_text, encoding="utf-8")
    _agent_io.print_agent_prompt(
        log_alias=IMPLEMENTATION_AGENT.log_alias,
        prompt_path=prompt_path,
        prompt_text=base_prompt_text,
    )

    allowed_prefixes = build_repair_write_policy(
        repo_root=config.paths.repo_root,
        data_dir=config.paths.data_dir,
        kernels_dir=config.paths.kernels_dir,
        module_file=Path(__file__),
    )
    guard_snapshot = _backup_guarded_files(config.paths.repo_root, allowed_prefixes)
    codex_pass_limit = max(1, int(max_codex_passes or MAX_KERNEL_FIX_CODEX_PASSES))
    retry_feedback = ""
    last_response_text = ""
    for codex_pass in range(1, codex_pass_limit + 1):
        pass_prompt_text = (
            base_prompt_text
            if not retry_feedback
            else _agent_io.append_fix_retry_feedback(
                base_prompt=base_prompt_text,
                stage_label="kernel_fix",
                codex_pass=codex_pass - 1,
                failure_text=retry_feedback,
            )
        )
        pass_prompt_path = (
            prompt_path if codex_pass == 1 else agent_dir / f"kernel_fix_prompt-{attempt:02d}-pass-{codex_pass:02d}.md"
        )
        pass_prompt_path.write_text(pass_prompt_text, encoding="utf-8")
        if codex_pass > 1:
            print(
                "[yellow]kernel fix[/yellow]: "
                f"retrying {IMPLEMENTATION_AGENT.log_alias} pass "
                f"{codex_pass}/{codex_pass_limit} with previous failure context."
            )
        before = _snapshot_tree(config.paths.repo_root)
        pass_output_dir = (
            agent_dir if codex_pass == 1 else agent_dir / f"kernel_fix_pass-{attempt:02d}-{codex_pass:02d}"
        )
        print(f"[cyan]kernel fix[/cyan]: running {IMPLEMENTATION_AGENT.log_alias} fixer")
        _watch_state.update_watch_phase(
            config,
            run_id,
            "gpt_kernel_fix_fixing",
            detail=f"{IMPLEMENTATION_AGENT.display_name} is repairing the kernel failure.",
            iteration=iteration,
        )
        result = run_codex(
            pass_prompt_path,
            pass_output_dir,
            dry_run=config.dry_run,
            heartbeat_label="fixing error",
            model=codex_model or _ERROR_FIX_CODEX_MODEL,
            reasoning_effort=codex_reasoning_effort or _ERROR_FIX_REASONING_EFFORT,
        )
        after = _snapshot_tree(config.paths.repo_root)
        changed = _diff_snapshots(before, after)
        if changed:
            _enforce_allowlist_changes(
                root=config.paths.repo_root,
                before=before,
                after=after,
                allowed_prefixes=allowed_prefixes,
                stage=f"kernel_fix_attempt_{attempt}",
                guard_snapshot=guard_snapshot,
                auto_repair=True,
            )
            _maybe_restart_for_src_changes(
                config=config,
                run_id=run_id,
                changed=changed,
                stage=f"kernel_fix_attempt_{attempt}",
            )
        response_text = _agent_io.read_agent_response(result.last_message_path)
        _agent_io.print_agent_response(
            log_alias=IMPLEMENTATION_AGENT.log_alias,
            response_path=result.last_message_path,
            response_text=response_text,
        )
        _agent_io.log_codex_sandbox_fallback(stage_label="kernel fix", result=result)
        last_response_text = response_text
        if result.returncode != 0:
            retry_feedback = (
                f"{IMPLEMENTATION_AGENT.display_name} kernel-fix step failed with non-zero exit status.\n"
                f"returncode={result.returncode}\n"
                f"pass={codex_pass}/{codex_pass_limit}\n"
                f"response={response_text}"
            )
            if codex_pass < codex_pass_limit:
                continue
            raise RuntimeError(f"{IMPLEMENTATION_AGENT.display_name} kernel-fix step failed.")

        if not changed:
            regenerated = _maybe_regenerate_kernel_sources_once(
                config=config,
                run_id=run_id,
                iteration=iteration,
                iter_dir=iter_dir,
                attempt=attempt,
                trigger_reason="codex_no_changes",
            )
            if not regenerated:
                raise KernelFailedError(
                    "Kernel fix agent produced no file changes and regeneration fallback was already used."
                )
            try:
                _run_verify(config.verify_cmd, dry_run=config.dry_run, artifacts_dir=config.paths.artifacts_dir)
            except Exception as exc:  # noqa: BLE001
                retry_feedback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
                if codex_pass < codex_pass_limit:
                    continue
                raise
            if pending_error_fixes is not None:
                pending_error_fixes.append(
                    {
                        "iteration": iteration,
                        "error_message": error_message,
                        "fix_summary": (
                            f"{IMPLEMENTATION_AGENT.display_name} kernel-fix made no edits; "
                            "regenerated kernel sources once and verification passed."
                        ),
                        "resolved": True,
                    }
                )
            return

        try:
            _run_verify(config.verify_cmd, dry_run=config.dry_run, artifacts_dir=config.paths.artifacts_dir)
        except Exception as exc:  # noqa: BLE001
            retry_feedback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
            if codex_pass < codex_pass_limit:
                continue
            raise
        if pending_error_fixes is not None:
            pending_error_fixes.append(
                {
                    "iteration": iteration,
                    "error_message": error_message,
                    "fix_summary": strategy_text or last_response_text,
                    "resolved": True,
                }
            )
        return

    raise RuntimeError(
        f"Kernel fix exhausted {IMPLEMENTATION_AGENT.log_alias} retry passes without resolving the error."
    )


def _run_metric_only_competition_metric_fix(
    *,
    config: AutopilotConfig,
    run_id: str,
    iteration: int,
    iter_dir: Path,
    mismatch_reason: str,
    attempt: int,
    pending_error_fixes: list[dict[str, object]] | None = None,
) -> None:
    """Apply a metric-only kernel fix using the implementation agent without GPT strategy mediation."""
    policy_prefix = (
        "Metric-only repair policy:\n"
        "- Edit ONLY competition metric selection/reporting logic in kernel outputs.\n"
        "- Do NOT change model architecture, features, training schedule, folds, seeds, or ensembling.\n"
        "- Ensure metrics.json reports the official competition metric exactly.\n"
        "- Ensure submission.csv format stays unchanged.\n"
    )
    metric_fix_error = (
        "Competition metric mismatch detected in strict mode.\n"
        f"Details: {mismatch_reason}\n"
        "Apply a minimal metric-only fix and stop."
    )
    _run_kernel_fix(
        config=config,
        run_id=run_id,
        iteration=iteration,
        iter_dir=iter_dir,
        error_message=metric_fix_error,
        attempt=attempt,
        pending_error_fixes=pending_error_fixes,
        use_gpt_strategy=False,
        codex_model=_METRIC_FIX_CODEX_MODEL,
        codex_reasoning_effort=_METRIC_FIX_REASONING_EFFORT,
        prompt_prefix=policy_prefix,
        max_codex_passes=_MAX_METRIC_FIX_CODEX_PASSES,
    )


def _rerun_kernel_for_metric_recheck(
    *,
    config: AutopilotConfig,
    run_id: str,
    iteration: int,
    submission_path: Path,
    iter_dir: Path,
    metrics_artifact_path: Path | None,
    kernel_name: str | None,
    enable_internet: bool,
    score_source: str,
    target_metric: str | None,
    metric_direction: str,
    holdout_frac: float,
    cv_folds: int,
    seed: int,
    time_budget_min: int | None,
) -> tuple[EvaluationResult, dict[str, object] | None, Path]:
    """Recheck metric parsing from existing artifacts without retraining in the same iteration."""
    del (
        config,
        run_id,
        iteration,
        kernel_name,
        enable_internet,
        score_source,
        holdout_frac,
        cv_folds,
        seed,
        time_budget_min,
    )

    rechecked_submission_path = submission_path
    if not rechecked_submission_path.exists():
        resolved_submission = _autopilot_state._resolve_iteration_submission_artifact(iter_dir)
        if resolved_submission is None:
            raise RuntimeError(
                "Metric recheck failed: submission artifact is missing for same-iteration metric-only recheck."
            )
        rechecked_submission_path = _autopilot_state._copy_submission_artifact_to_iteration_dir(
            source=resolved_submission,
            iter_dir=iter_dir,
        )

    output_metrics_path = iter_dir / "output" / "metrics.json"
    metrics_candidates: list[Path] = []
    seen_metric_candidates: set[str] = set()

    def add_metrics_candidate(path: Path | None) -> None:
        if path is None:
            return
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen_metric_candidates:
            return
        seen_metric_candidates.add(key)
        metrics_candidates.append(path)

    add_metrics_candidate(metrics_artifact_path)
    add_metrics_candidate(iter_dir / "metrics.json")
    add_metrics_candidate(output_metrics_path)
    add_metrics_candidate(_autopilot_state._resolve_iteration_artifact(iter_dir, "metrics.json"))

    loaded_candidates: list[tuple[Path, dict[str, object], EvaluationResult | None]] = []
    for metrics_candidate in metrics_candidates:
        if not metrics_candidate.exists():
            continue
        candidate_payload = _json_utils.load_json_object(metrics_candidate)
        if candidate_payload is None:
            continue
        if str(candidate_payload.get("kind") or "").strip().lower() == "submit_only":
            continue
        candidate_evaluation = _kernel_metrics.load_kernel_metrics(
            metrics_candidate,
            metric_direction,
            target_metric,
        )
        loaded_candidates.append((metrics_candidate, candidate_payload, candidate_evaluation))

    valid_candidates = [candidate for candidate in loaded_candidates if candidate[2] is not None]
    selected_candidate: tuple[Path, dict[str, object], EvaluationResult | None] | None = None
    if valid_candidates:
        selected_candidate = valid_candidates[0]
        output_candidate = next(
            (candidate for candidate in valid_candidates if candidate[0] == output_metrics_path),
            None,
        )
        if output_candidate is not None:
            try:
                output_mtime = output_metrics_path.stat().st_mtime
                other_mtimes = [
                    candidate[0].stat().st_mtime
                    for candidate in valid_candidates
                    if candidate[0] != output_metrics_path
                ]
            except OSError:
                other_mtimes = []
                output_mtime = 0.0
            if not other_mtimes or output_mtime >= max(other_mtimes):
                selected_candidate = output_candidate
    elif loaded_candidates:
        selected_candidate = loaded_candidates[0]

    if selected_candidate is None:
        raise RuntimeError(
            "Metric recheck failed: metrics.json artifact is missing for same-iteration metric-only recheck."
        )

    resolved_metrics_path, payload, evaluation = selected_candidate
    metric_mismatch = bool(
        target_metric
        and evaluation
        and evaluation.metric
        and not _metric_matching.metrics_equivalent(evaluation.metric, target_metric)
    )
    needs_recompute = evaluation is None or metric_mismatch
    if needs_recompute:
        recomputed = _kernel_metrics.recompute_metric_from_oof_artifact(
            iter_dir=iter_dir,
            payload=payload,
            target_metric=target_metric,
            metric_direction=metric_direction,
            resolve_iteration_artifact=_autopilot_state._resolve_iteration_artifact,
        )
        if recomputed is not None:
            evaluation, payload = recomputed
            _kernel_metrics.persist_metric_recheck_payload(
                iter_dir=iter_dir,
                resolved_metrics_path=resolved_metrics_path,
                payload=payload,
            )
    if evaluation is None:
        raise RuntimeError("Metric recheck failed: kernel metrics missing expected score.")
    return evaluation, payload, rechecked_submission_path


def _run_autofix(*, config: AutopilotConfig, run_id: str, attempt: int, error: Exception) -> None:
    run_dir = config.paths.run_dir(run_id)
    autofix_dir = run_dir / "autofix" / f"attempt-{attempt}"
    autofix_dir.mkdir(parents=True, exist_ok=True)
    error_text = "".join(traceback.format_exception(type(error), error, error.__traceback__)).strip()
    submit_autofix = isinstance(error, SubmitAbortedError)
    submit_context = ""
    submit_file_fix_required = False
    submit_file_fix_baseline_path: Path | None = None
    submit_file_fix_baseline_sha256: str | None = None
    if isinstance(error, KaggleCliError):
        if error.command:
            error_text = f"{error_text}\n\nKaggle CLI command:\n{shlex.join(error.command)}"
        if error.output:
            error_text = f"{error_text}\n\nKaggle CLI output:\n{error.output}"
    if submit_autofix:
        failure_context = _submit_failure_context.load_submit_failure_context(run_dir)
        run_state = _autopilot_state._load_run_state(run_dir)
        latest_submit_attempt = _submit_attempts.load_latest_submit_attempt(run_dir)
        submit_context = _submit_failure_context.format_submit_autofix_context(
            failure_context=failure_context,
            run_state=run_state,
            latest_submit_attempt=latest_submit_attempt,
        )
        submit_file_fix_required = _submit_autofix.submit_file_fix_required_for_attempt(latest_submit_attempt)
        max_search_iteration = MAX_AUTOFIX_ATTEMPTS + MAX_KERNEL_FIX_ATTEMPTS + MAX_AUTOFIX_CODEX_PASSES

        def fallback_iteration_dirs():
            return (config.paths.iter_dir(run_id, iteration) for iteration in range(max_search_iteration, 0, -1))

        def save_repaired_submit_path(fixed: Path) -> None:
            _autopilot_state._save_run_state(run_dir, {"submit_autofix_submission_path": str(fixed)})

        if submit_file_fix_required:
            submit_file_fix_baseline_path = _submit_failure_context.resolve_submit_autofix_submission_artifact(
                run_state=run_state,
                latest_submit_attempt=latest_submit_attempt,
                failure_context=failure_context,
                fallback_iteration_dirs=fallback_iteration_dirs(),
                resolve_iteration_submission_artifact=_autopilot_state._resolve_iteration_submission_artifact,
            )
            submit_file_fix_baseline_sha256 = _sha256_or_none(submit_file_fix_baseline_path)
        repair_service = SubmissionService(
            SubmissionConfig(
                slug=config.slug,
                data_dir=config.paths.data_dir,
                sample_submission_path=config.paths.sample_submission_path,
                submission_ledger_path=config.paths.submission_ledger_path,
                dry_run=True,
                force_submit=True,
                bypass_rate_limit=True,
            )
        )
        preparation = _submit_autofix.prepare_submit_file_autofix_for_run(
            latest_submit_attempt=latest_submit_attempt,
            run_state=run_state,
            failure_context=failure_context,
            fallback_iteration_dirs=fallback_iteration_dirs,
            resolve_iteration_submission_artifact=_autopilot_state._resolve_iteration_submission_artifact,
            validate_and_prepare=repair_service.validate_and_prepare_submission,
            save_repaired_path=save_repaired_submit_path,
        )
        _prepared_submission_path, prepared_submission_summary = preparation.path, preparation.summary
        if prepared_submission_summary:
            submit_context = (
                f"{submit_context}\n\ndeterministic_submit_file_autofix:\n{prepared_submission_summary}".strip()
            )
            error_text = f"{error_text}\n\nDeterministic Submit File Autofix:\n{prepared_submission_summary}"
        if submit_context:
            error_text = f"{error_text}\n\nSubmit Failure Context:\n{submit_context}"
    attempt_tag = f"{attempt:02d}"
    header = f"autofix_attempt: {attempt}\n"
    error_path = autofix_dir / f"error-{attempt_tag}.txt"
    error_path.write_text(header + error_text + "\n", encoding="utf-8")
    (autofix_dir / "error.txt").write_text(header + error_text + "\n", encoding="utf-8")

    allowed_prefixes = build_repair_write_policy(
        repo_root=config.paths.repo_root,
        data_dir=config.paths.data_dir,
        kernels_dir=config.paths.kernels_dir,
        module_file=Path(__file__),
    )
    prompt_text = _agent_prompts.build_autofix_prompt(
        slug=config.slug,
        run_id=run_id,
        attempt=attempt,
        compute=config.compute,
        accelerator=config.accelerator,
        error_text=error_text,
        error_path=error_path,
        repo_root=config.paths.repo_root,
        run_dir=config.paths.run_dir(run_id),
        kernel_dir=config.paths.kernel_source_dir,
        context_dir=config.paths.context_dir,
        data_dir=config.paths.data_dir,
        prompts_dir=config.paths.prompts_dir,
        autopilot_path=Path(__file__).resolve(),
        allowed_prefixes=allowed_prefixes.allowed_prefixes,
        denied_prefixes=allowed_prefixes.denied_prefixes,
        submit_context=submit_context,
    )
    if submit_file_fix_required:
        prompt_text += _submit_failure_context.format_submit_file_repair_contract_prompt()
    strategy_stage = "submit_autofix" if submit_autofix else "autofix"
    strategy_label = "submit autofix" if submit_autofix else "autofix"
    print(
        f"[cyan]{strategy_label}[/cyan]: strategy={_ERROR_STRATEGY_MODEL}({_ERROR_STRATEGY_REASONING_EFFORT}) "
        f"-> fixer={_ERROR_FIX_CODEX_MODEL}({_ERROR_FIX_REASONING_EFFORT})"
    )
    thinking_phase = "gpt_submit_autofix_thinking" if submit_autofix else "gpt_autofix_thinking"
    fixing_phase = "gpt_submit_autofix_fixing" if submit_autofix else "gpt_autofix_fixing"
    _watch_state.update_watch_phase(
        config,
        run_id,
        thinking_phase,
        detail=f"GPT is analyzing the {strategy_label} failure and drafting a fix strategy.",
    )
    strategy_prompt = _agent_prompts.build_error_strategy_prompt(
        stage=strategy_stage,
        slug=config.slug,
        run_id=run_id,
        attempt=attempt,
        compute=config.compute,
        accelerator=config.accelerator,
        hardware_constraints=render_hardware_constraints(
            resolve_hardware_profile(config.hardware_profile, compute=config.compute),
            compute=config.compute,
            time_budget_min=config.time_budget_min,
        ),
        error_text=error_text,
        codex_prompt=prompt_text,
    )
    strategy_text = _run_error_strategy(
        prompt_text=strategy_prompt,
        output_dir=autofix_dir / "gpt_strategy",
        dry_run=config.dry_run,
        stage_label=strategy_label,
    )
    if not strategy_text.strip():
        print(
            f"[yellow]{strategy_label}[/yellow]: gpt strategy unavailable, "
            f"continuing with direct {IMPLEMENTATION_AGENT.log_alias} fix"
        )
    else:
        prompt_text += (
            f"\n\n## {STRATEGY_AGENT.display_name} Extra-High Error-Fix Strategy\n"
            "Use the strategy below as guidance, then apply minimal targeted edits.\n\n"
            f"{strategy_text}\n"
        )
    prompt_path = autofix_dir / "prompt.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    _agent_io.print_agent_prompt(
        log_alias=IMPLEMENTATION_AGENT.log_alias,
        prompt_path=prompt_path,
        prompt_text=prompt_text,
    )

    retry_feedback = ""
    for codex_pass in range(1, MAX_AUTOFIX_CODEX_PASSES + 1):
        pass_prompt_text = (
            prompt_text
            if not retry_feedback
            else _agent_io.append_fix_retry_feedback(
                base_prompt=prompt_text,
                stage_label="autofix",
                codex_pass=codex_pass - 1,
                failure_text=retry_feedback,
            )
        )
        pass_prompt_path = prompt_path if codex_pass == 1 else autofix_dir / f"prompt-pass-{codex_pass:02d}.md"
        pass_prompt_path.write_text(pass_prompt_text, encoding="utf-8")
        if codex_pass > 1:
            print(
                "[yellow]autofix[/yellow]: "
                f"retrying {IMPLEMENTATION_AGENT.log_alias} pass "
                f"{codex_pass}/{MAX_AUTOFIX_CODEX_PASSES} with previous failure context."
            )

        before = _snapshot_tree(config.paths.repo_root)
        pass_output_dir = autofix_dir if codex_pass == 1 else autofix_dir / f"pass-{codex_pass:02d}"
        _watch_state.update_watch_phase(
            config,
            run_id,
            fixing_phase,
            detail=f"{IMPLEMENTATION_AGENT.display_name} is applying the {strategy_label} fix.",
        )
        result = run_codex(
            pass_prompt_path,
            pass_output_dir,
            dry_run=config.dry_run,
            heartbeat_label="fixing error",
            model=_ERROR_FIX_CODEX_MODEL,
            reasoning_effort=_ERROR_FIX_REASONING_EFFORT,
        )
        after = _snapshot_tree(config.paths.repo_root)
        changed = _diff_snapshots(before, after)
        # Autofix often needs to regenerate staged outputs under artifacts/*/kernels and
        # run-level state files; do not apply write-guard restrictions in this stage.
        _maybe_restart_for_src_changes(
            config=config,
            run_id=run_id,
            changed=changed,
            stage=f"autofix_attempt_{attempt}",
        )
        response_text = _agent_io.read_agent_response(result.last_message_path)
        _agent_io.print_agent_response(
            log_alias=IMPLEMENTATION_AGENT.log_alias,
            response_path=result.last_message_path,
            response_text=response_text,
        )
        _agent_io.log_codex_sandbox_fallback(stage_label=strategy_label, result=result)
        if result.returncode != 0:
            retry_feedback = (
                f"{IMPLEMENTATION_AGENT.display_name} autofix step failed with non-zero exit status.\n"
                f"returncode={result.returncode}\n"
                f"pass={codex_pass}/{MAX_AUTOFIX_CODEX_PASSES}\n"
                f"response={response_text}"
            )
            if codex_pass < MAX_AUTOFIX_CODEX_PASSES:
                continue
            raise RuntimeError(f"{IMPLEMENTATION_AGENT.display_name} autofix step failed.")

        if submit_file_fix_required and not _submit_failure_context.submit_file_fix_contract_satisfied(
            run_state=_autopilot_state._load_run_state(run_dir),
            baseline_path=submit_file_fix_baseline_path,
            baseline_sha256=submit_file_fix_baseline_sha256,
            sha256_or_none=_sha256_or_none,
        ):
            retry_feedback = _submit_failure_context.format_submit_file_repair_contract_retry_feedback(
                baseline_path=submit_file_fix_baseline_path,
                baseline_sha256=submit_file_fix_baseline_sha256,
            )
            if codex_pass < MAX_AUTOFIX_CODEX_PASSES:
                continue
            raise RuntimeError("Submit autofix did not repair the submission artifact required by Kaggle.")

        try:
            _run_verify(config.verify_cmd, dry_run=config.dry_run, artifacts_dir=config.paths.artifacts_dir)
        except Exception as exc:  # noqa: BLE001
            retry_feedback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
            if codex_pass < MAX_AUTOFIX_CODEX_PASSES:
                continue
            raise
        return

    raise RuntimeError(f"Autofix exhausted {IMPLEMENTATION_AGENT.log_alias} retry passes without resolving the error.")


def _run_error_strategy(
    *,
    prompt_text: str,
    output_dir: Path,
    dry_run: bool,
    stage_label: str,
) -> str:
    return _agent_strategy.run_strategy_prompt(
        prompt_text=prompt_text,
        output_dir=output_dir,
        dry_run=dry_run,
        config=_agent_strategy.StrategyPromptRunConfig(
            prompt_filename="gpt_strategy_prompt.md",
            start_message=f"[cyan]{stage_label}[/cyan]: gpt analyzing error",
            detail_message=(
                f"[cyan]{stage_label}[/cyan]: strategy model={_ERROR_STRATEGY_MODEL} "
                f"reasoning={_ERROR_STRATEGY_REASONING_EFFORT}"
            ),
            failure_message=(
                f"[yellow]{stage_label}[/yellow]: gpt strategy failed, "
                f"continuing with direct {IMPLEMENTATION_AGENT.log_alias} fix"
            ),
            empty_message=(
                f"[yellow]{stage_label}[/yellow]: gpt strategy empty, "
                f"continuing with direct {IMPLEMENTATION_AGENT.log_alias} fix"
            ),
        ),
        run_strategy_func=run_strategy,
    )


def _maybe_regenerate_kernel_sources_once(
    *,
    config: AutopilotConfig,
    run_id: str,
    iteration: int,
    iter_dir: Path,
    attempt: int,
    trigger_reason: str,
) -> bool:
    """Regenerate authoritative kernel sources once when fix loops are stuck."""
    if config.dry_run:
        return False
    agent_dir = iter_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    marker_path = agent_dir / _KERNEL_REGENERATE_MARKER_FILENAME
    if marker_path.exists():
        return False

    marker_payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "trigger_reason": trigger_reason,
        "attempt": int(attempt),
        "iteration": int(iteration),
        "run_id": run_id,
    }
    _json_utils.write_json_object(marker_path, marker_payload)
    print(
        "[yellow]kernel fix[/yellow]: unresolved kernel error loop detected; "
        "regenerating kernel sources once before retry."
    )
    try:
        _run_plan_and_initial(config, run_id)
    except Exception as exc:  # noqa: BLE001
        note_path = agent_dir / f"kernel_regen_note-{attempt:02d}.txt"
        note_path.write_text(
            (f"kernel_regen_failed: regeneration fallback failed.\ntrigger_reason: {trigger_reason}\nerror: {exc}\n"),
            encoding="utf-8",
        )
        return False
    note_path = agent_dir / f"kernel_regen_note-{attempt:02d}.txt"
    note_path.write_text(
        (f"kernel_regen_applied: regeneration fallback succeeded.\ntrigger_reason: {trigger_reason}\n"),
        encoding="utf-8",
    )
    return True


def _maybe_apply_lightweight_runtime_fix(
    *,
    config: AutopilotConfig,
    error_text: str,
    note_path: Path,
    stage_label: str,
) -> str | None:
    actions: tuple[tuple[str, str, Callable[[AutopilotConfig, str], bool]], ...] = (
        (
            "column_fill.json",
            "missing column error",
            _runtime_fixes.maybe_write_column_fill,
        ),
        (
            "object_coerce.json",
            "numpy.object_ conversion error",
            _runtime_fixes.maybe_write_object_coerce,
        ),
        (
            "device_coerce.json",
            "torch device mismatch error",
            _runtime_fixes.maybe_write_device_coerce,
        ),
        (
            "column_map.json",
            "column alias mismatch",
            _runtime_fixes.maybe_write_column_map,
        ),
    )
    for artifact_name, reason, action in actions:
        try:
            changed = bool(action(config, error_text))
        except Exception:
            changed = False
        if not changed:
            continue
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note = (
            f"autofix_note: {artifact_name} created for {reason}.\n"
            "autofix will retry without modifying kernel sources.\n"
        )
        note_path.write_text(note, encoding="utf-8")
        print(
            f"[yellow]{stage_label}[/yellow]: wrote {artifact_name}; "
            f"retrying without {IMPLEMENTATION_AGENT.log_alias} edits"
        )
        return artifact_name
    return None


def _maybe_restart_for_src_changes(*, config: AutopilotConfig, run_id: str, changed: list[str], stage: str) -> None:
    _autofix_restart.maybe_restart_for_src_changes(
        dry_run=config.dry_run,
        run_dir=config.paths.run_dir(run_id),
        run_id=run_id,
        slug=config.slug,
        changed=changed,
        stage=stage,
        max_restarts=MAX_AUTOFIX_RESTARTS,
    )


def _attempt_submit(
    *,
    config: AutopilotConfig,
    run_id: str,
    submission_path: Path,
    best_score: float | None,
    problem_types: list[str],
    submit_mode: str = "file",
    notebook_submit_artifact_mode: str = "wrapper",
) -> dict[str, object] | None:
    if not config.submit or config.dry_run:
        return None
    run_dir = config.paths.run_dir(run_id)
    submit_attempt_recorder = _submit_attempts.SubmitAttemptRecorder(
        run_dir=run_dir,
        save_run_state=lambda updates: _autopilot_state._save_run_state(run_dir, updates),
    )
    autofix_attempt_context = _submit_failure_context.resolve_submit_autofix_context_for_run(
        run_dir=run_dir,
        submission_path=submission_path,
        load_run_state=_autopilot_state._load_run_state,
        save_run_state=lambda updates: _autopilot_state._save_run_state(run_dir, updates),
        now_iso=datetime.now(UTC).isoformat(),
    )
    run_state = autofix_attempt_context.run_state
    latest_submit_attempt = autofix_attempt_context.latest_submit_attempt
    submit_code_fingerprint = _submit_retry_policy.compute_submit_code_fingerprint(
        src_root=Path(__file__).resolve().parent,
        kernel_source_dir=config.paths.kernel_source_dir,
        sha256_or_none=_sha256_or_none,
    )
    allow_force = config.force_submit or _env_utils.env_truthy("KAGGLEBOT_FORCE_RESUBMIT")
    input_submission_path = autofix_attempt_context.input_submission_path
    if autofix_attempt_context.message:
        print(autofix_attempt_context.message)

    message = _submit_stage.resolve_submission_message(
        context_dir=config.paths.context_dir,
        run_id=run_id,
        best_score=best_score,
        explicit_message=config.message,
        submission_path=input_submission_path,
        campaign_mode=config.campaign_mode,
        target_direction=config.target_direction,
    )
    submission_service = SubmissionService(
        SubmissionConfig(
            slug=config.slug,
            data_dir=config.paths.data_dir,
            sample_submission_path=config.paths.sample_submission_path,
            submission_ledger_path=config.paths.submission_ledger_path,
            dry_run=config.dry_run,
            force_submit=config.force_submit,
            bypass_rate_limit=False,
        )
    )
    print(f"[cyan]submit[/cyan]: {config.slug}")
    submitted_at = datetime.now(UTC)

    submit_aborter = _submit_stage.SubmitRunAborter(
        run_dir=run_dir,
        run_id=run_id,
        slug=config.slug,
        knowledge_paths=config.knowledge_paths,
        problem_types=problem_types,
        save_run_state=lambda updates: _autopilot_state._save_run_state(run_dir, updates),
        resolve_submit_abort_artifact_path=_submit_failure_context.resolve_submit_abort_artifact_path,
        persist_submit_abort_failure=_submit_failure_context.persist_submit_abort_failure,
        load_run_state=_autopilot_state._load_run_state,
        load_latest_submit_attempt=_submit_attempts.load_latest_submit_attempt,
        has_successful_submit_attempt=_submit_attempts.has_successful_submit_attempt,
        compute_submission_sha256=_sha256_or_none,
        stdout_tail_chars=_SUBMIT_STDOUT_TAIL_CHARS,
        stderr_tail_chars=_SUBMIT_STDERR_TAIL_CHARS,
        now_iso=lambda: datetime.now(UTC).isoformat(),
        normalize_detail=normalize_error_text,
        record_error_fix_insight=record_error_fix_insight,
        on_message=print,
        build_error=SubmitAbortedError,
    )
    submit_retry_recorder = _submit_stage.SubmitRunRetryRecorder(
        submit_attempt_recorder=submit_attempt_recorder,
        run_id=run_id,
        slug=config.slug,
        problem_types=problem_types,
        knowledge_paths=config.knowledge_paths,
        compute_submission_sha256=_sha256_or_none,
        stdout_tail_chars=_SUBMIT_STDOUT_TAIL_CHARS,
        stderr_tail_chars=_SUBMIT_STDERR_TAIL_CHARS,
        normalize_detail=normalize_error_text,
        record_error_fix_insight=record_error_fix_insight,
    )

    prepared_resolution = _submit_stage.resolve_prepared_submission_for_submit(
        input_submission_path=input_submission_path,
        validate_and_prepare=submission_service.validate_and_prepare_submission,
        validation_error_types=(SubmissionValidationError,),
        validation_exit_code=SubmissionValidationError.exit_code,
        compute_error_fingerprint=compute_error_fingerprint,
    )
    if prepared_resolution.abort_spec is not None:
        return submit_aborter.abort(
            submission_ref=input_submission_path,
            code_fingerprint=submit_code_fingerprint,
            **_submit_stage.build_submit_abort_spec_kwargs(prepared_resolution.abort_spec),
            submit_attempt_recorder=submit_attempt_recorder,
        )
    prepared_submission_path = _submit_stage.require_prepared_submission_path(
        prepared_resolution,
        build_error=SubmitAbortedError,
    )

    prepared_submission_sha = str(_sha256_or_none(prepared_submission_path) or "").strip()
    duplicate_skip_result = _submit_stage.resolve_duplicate_submission_for_run(
        run_dir=run_dir,
        submission_ledger_path=config.paths.submission_ledger_path,
        slug=config.slug,
        run_id=run_id,
        message=message,
        submitted_at=submitted_at,
        submission_path=submission_path,
        prepared_submission_path=prepared_submission_path,
        prepared_submission_sha=prepared_submission_sha,
        code_fingerprint=submit_code_fingerprint,
        allow_force=allow_force,
        prior_state=_autopilot_state._load_run_state(run_dir),
        collect_duplicate_submission_sources=_submit_retry_policy.collect_duplicate_submission_sources,
        decide_duplicate_submission_action=_submit_retry_policy.decide_duplicate_submission_action,
        compute_error_fingerprint=compute_error_fingerprint,
        record_submit_attempt_payloads=submit_attempt_recorder.record_payloads,
        stdout_tail_chars=_SUBMIT_STDOUT_TAIL_CHARS,
        stderr_tail_chars=_SUBMIT_STDERR_TAIL_CHARS,
        on_message=print,
    )
    if duplicate_skip_result is not None:
        return duplicate_skip_result

    rules_resolution = _submit_stage.resolve_rules_acceptance_for_submit(
        check_rules_accepted=lambda: check_rules_accepted(config.slug, dry_run=config.dry_run),
        cli_error_types=(KaggleCliError,),
        is_missing_credentials_error=_kaggle_cli_errors.is_missing_kaggle_credentials_error,
        rules_not_accepted_exit_code=RulesNotAcceptedError.exit_code,
        compute_error_fingerprint=compute_error_fingerprint,
    )
    if rules_resolution.abort_spec is not None:
        return submit_aborter.abort(
            submission_ref=prepared_submission_path,
            code_fingerprint=submit_code_fingerprint,
            **_submit_stage.build_submit_abort_spec_kwargs(rules_resolution.abort_spec),
            submit_attempt_recorder=submit_attempt_recorder,
        )

    constraints = _competition_rules.load_competition_rule_constraints(config.paths)
    code_competition = infer_code_competition_from_paths(config.paths)
    submit_stage_state = _submit_stage.resolve_initial_submit_stage_runtime_state(
        submit_mode=submit_mode,
        notebook_submissions_only=constraints.notebook_submissions_only,
        notebook_submit_artifact_mode=notebook_submit_artifact_mode,
        code_competition=code_competition,
        sample_submission_path=config.paths.sample_submission_path,
        fallback_sample_submission_path=config.paths.data_dir / "sample_submission.csv",
        submission_path=prepared_submission_path,
        resolve_notebook_submit_artifact_mode=_submit_notebook.resolve_notebook_submit_artifact_mode,
        decide_notebook_submit_artifact_mode_for_paths=_submit_notebook.decide_notebook_submit_artifact_mode_for_paths,
        count_csv_data_rows=_context_artifacts.count_csv_data_rows_capped,
        on_message=print,
    )

    if _submit_stage.resolve_same_submission_path_for_run(
        run_id=run_id,
        run_state=run_state,
        latest_submit_attempt=latest_submit_attempt,
        prepared_submission_path=prepared_submission_path,
        current_submission_sha=str(_sha256_or_none(prepared_submission_path) or "").strip(),
        submit_code_fingerprint=submit_code_fingerprint,
        allow_force=allow_force,
        notebook_submit_required=submit_stage_state.notebook_submit_required,
        decide_same_submission_path_action=_submit_retry_policy.decide_same_submission_path_action,
        compute_submission_sha256=_sha256_or_none,
        submit_attempt_recorder=submit_attempt_recorder,
        stdout_tail_chars=_SUBMIT_STDOUT_TAIL_CHARS,
        stderr_tail_chars=_SUBMIT_STDERR_TAIL_CHARS,
        on_message=print,
    ):
        return None

    seen_fingerprints = _submit_attempts.build_seen_submit_fingerprint_set_for_run(
        run_dir=run_dir,
        run_state=run_state,
    )
    max_attempts = max(1, _SUBMIT_MAX_TRANSIENT_RETRIES)
    submission_result = None
    submission_reference = str(prepared_submission_path)
    submission_artifact_path: Path | None = prepared_submission_path

    notebook_submitter = _submit_notebook.NotebookSubmitRunner(
        slug=config.slug,
        run_id=run_id,
        paths=config.paths,
        kaggle_username=config.kaggle_username,
        kernel_name=config.kernel_name,
        accelerator=config.accelerator,
        strict_accelerator=config.strict_accelerator,
        dry_run=config.dry_run,
        timeout_minutes=config.time_budget_min,
        infer_iteration_from_submission_path=_submit_stage.infer_iteration_from_submission_path,
        resolve_kaggle_username=resolve_kaggle_username,
        run_submit_kernel=run_submit_kernel,
        run_kaggle_submit_kernel=run_kaggle_submit_kernel,
        copy_submission_artifact_to_iteration_dir=_autopilot_state._copy_submission_artifact_to_iteration_dir,
        classify_submit_error=classify_submit_error,
        should_retry_ambiguous=_submit_failure_policy.should_retry_ambiguous_notebook_submit_error,
        sleep=time.sleep,
        on_message=print,
        is_capacity_error=lambda exc: isinstance(exc, KernelCapacityError),
        is_push_error=lambda exc: isinstance(exc, KaggleCliError) and _submit_notebook.is_submit_kernel_push_error(exc),
    )

    for attempt in range(1, max_attempts + 1):
        try:
            submit_attempt_result = _submit_stage.run_submit_stage_attempt(
                notebook_submit_required=submit_stage_state.notebook_submit_required,
                file_submission_path=prepared_submission_path,
                run_notebook_submit=lambda: notebook_submitter.submit(
                    submission_path=prepared_submission_path,
                    message=message,
                    artifact_mode=submit_stage_state.submission_artifact_mode,
                ),
                run_file_submit=lambda: submission_service.submit_prepared(
                    prepared_path=prepared_submission_path,
                    message=message,
                    run_id=run_id,
                    offline_score=best_score,
                    score_source="offline",
                ),
            )
            submission_result = submit_attempt_result.submission_result
            submission_reference = submit_attempt_result.submission_reference
            submission_artifact_path = submit_attempt_result.submission_artifact_path
        except SubmissionCliError as exc:
            submit_error_resolution = _submit_stage.resolve_submit_cli_error(
                state=submit_stage_state,
                stdout=exc.stdout,
                stderr=exc.stderr or "",
                output=exc.output or "",
                exit_code=exc.exit_code,
                attempt=attempt,
                max_attempts=max_attempts,
                backoff_base_seconds=_SUBMIT_BACKOFF_BASE_SEC,
                classify_submit_error=classify_submit_error,
                should_use_notebook_fallback=_submit_failure_policy.should_use_notebook_submit_fallback,
                code_competition=code_competition,
                sample_submission_path=config.paths.sample_submission_path,
                fallback_sample_submission_path=config.paths.data_dir / "sample_submission.csv",
                submission_path=prepared_submission_path,
                resolve_notebook_submit_artifact_mode=_submit_notebook.resolve_notebook_submit_artifact_mode,
                decide_notebook_submit_artifact_mode_for_paths=_submit_notebook.decide_notebook_submit_artifact_mode_for_paths,
                count_csv_data_rows=_context_artifacts.count_csv_data_rows_capped,
                compute_error_fingerprint=compute_error_fingerprint,
                decide_submit_fingerprint_reuse=_submit_retry_policy.decide_submit_fingerprint_reuse,
                compute_submit_backoff=_submit_retry_policy.compute_submit_backoff,
                seen_fingerprints=seen_fingerprints,
                run_state=run_state,
                code_fingerprint=submit_code_fingerprint,
                save_run_state=lambda updates: _autopilot_state._save_run_state(run_dir, updates),
                on_message=print,
            )
            submit_error_classification = submit_error_resolution.classification
            fallback_application = submit_error_resolution.fallback_application
            submit_stage_state = fallback_application.state
            if fallback_application.retry_as_notebook:
                continue
            fingerprint = submit_error_resolution.fingerprint
            error_action = submit_error_resolution.error_action
            if error_action is None:
                raise SubmitAbortedError("Submit error resolution did not produce a retry or abort action.")
            if error_action.action == "abort":
                abort_spec = _submit_stage.build_submit_stage_error_action_abort_spec(
                    action=error_action,
                    fingerprint=fingerprint,
                    stdout=exc.stdout,
                    stderr=submit_error_classification.stderr,
                    exit_code=exc.exit_code,
                )
                return submit_aborter.abort(
                    submission_ref=submission_reference,
                    submission_artifact_path=submission_artifact_path,
                    artifact_mode=submit_stage_state.submission_artifact_mode,
                    code_fingerprint=submit_code_fingerprint,
                    **_submit_stage.build_submit_abort_spec_kwargs(abort_spec),
                    submit_attempt_recorder=submit_attempt_recorder,
                )
            seen_fingerprints.add(fingerprint)
            if error_action.action == "retry":
                submit_retry_recorder.record(
                    submission_ref=submission_reference,
                    submission_artifact_path=submission_artifact_path,
                    fallback_submission_path=prepared_submission_path,
                    exit_code=exc.exit_code,
                    fingerprint=fingerprint,
                    action=error_action,
                    stdout=exc.stdout,
                    stderr=submit_error_classification.stderr,
                    attempt=attempt,
                )
                time.sleep(error_action.wait_seconds)
                continue
        except (DuplicateSubmissionError, SubmissionRateLimitError) as exc:
            abort_spec = _submit_stage.resolve_local_submission_guardrail_abort_spec(
                error=exc,
                compute_error_fingerprint=compute_error_fingerprint,
            )
            return submit_aborter.abort(
                submission_ref=submission_reference,
                submission_artifact_path=submission_artifact_path,
                code_fingerprint=submit_code_fingerprint,
                **_submit_stage.build_submit_abort_spec_kwargs(abort_spec),
                submit_attempt_recorder=submit_attempt_recorder,
            )
        except KaggleCliError as exc:
            abort_spec = _submit_stage.resolve_kaggle_cli_submit_abort_spec(
                error=exc,
                is_missing_credentials_error=_kaggle_cli_errors.is_missing_kaggle_credentials_error,
                compute_error_fingerprint=compute_error_fingerprint,
            )
            if abort_spec is None:
                raise
            return submit_aborter.abort(
                submission_ref=submission_reference,
                submission_artifact_path=submission_artifact_path,
                artifact_mode=submit_stage_state.submission_artifact_mode,
                code_fingerprint=submit_code_fingerprint,
                **_submit_stage.build_submit_abort_spec_kwargs(abort_spec),
                submit_attempt_recorder=submit_attempt_recorder,
            )
        break

    if submission_result is None:
        raise SubmitAbortedError("Submit failed before producing a submission result.")
    submission_ref = submission_reference
    submission_for_submit_path = submission_artifact_path
    outcome_resolution = _submit_stage.resolve_submission_outcome_after_submit(
        slug=config.slug,
        message=message,
        submitted_at=submitted_at,
        deliverable_mode=infer_deliverable_mode_from_paths(config.paths, default="leaderboard"),
        fetch_submission_rows=lambda current_slug: list_competition_submissions(current_slug, dry_run=False),
        max_attempts=_SUBMISSION_POLL_MAX_ATTEMPTS,
        poll_interval_sec=_SUBMISSION_POLL_INTERVAL_SEC,
        max_fetch_errors=_SUBMISSION_POLL_MAX_FETCH_ERRORS,
        normalize_detail=lambda text: normalize_error_text(text, max_chars=1200),
        compute_error_fingerprint=compute_error_fingerprint,
    )
    if outcome_resolution.abort_spec is not None:
        return submit_aborter.abort(
            submission_ref=submission_ref,
            submission_artifact_path=submission_for_submit_path,
            artifact_mode=submit_stage_state.submission_artifact_mode,
            code_fingerprint=submit_code_fingerprint,
            **_submit_stage.build_submit_abort_spec_kwargs(outcome_resolution.abort_spec),
            submit_attempt_recorder=submit_attempt_recorder,
        )
    outcome = outcome_resolution.outcome

    return _submit_stage.record_successful_submit_for_run(
        run_dir=run_dir,
        submission_ledger_path=config.paths.submission_ledger_path,
        slug=config.slug,
        run_id=run_id,
        message=message,
        submitted_at=submitted_at,
        submission_ref=submission_ref,
        submission_result=submission_result,
        submission_path=submission_path,
        submission_artifact_path=submission_for_submit_path,
        outcome=outcome,
        code_fingerprint=submit_code_fingerprint,
        prior_state=_autopilot_state._load_run_state(run_dir),
        compute_error_fingerprint=compute_error_fingerprint,
        compute_submission_sha256=_sha256_or_none,
        record_submit_attempt_payloads=submit_attempt_recorder.record_payloads,
        stdout_tail_chars=_SUBMIT_STDOUT_TAIL_CHARS,
        stderr_tail_chars=_SUBMIT_STDERR_TAIL_CHARS,
        on_message=print,
    )
