from __future__ import annotations

import ast
import builtins
import csv
import hashlib
import json
import os
import random
import re
import shlex
import shutil
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from rich import print

from kagglebot.agents.codex_runner import run_codex
from kagglebot.agents.strategy_runner import run_strategy
from kagglebot.eval import (
    DriftChecker,
    EvaluationReport,
    SplitStrategyFactory,
    SubmissionReadinessScorer,
    UncertaintyEstimator,
    validate_evaluation_spec,
)
from kagglebot.exceptions import (
    DuplicateSubmissionError,
    KaggleCliError,
    KaggleNetworkError,
    KernelCapacityError,
    KernelFailedError,
    RulesNotAcceptedError,
    SubmissionCliError,
    SubmissionRateLimitError,
    SubmissionValidationError,
    SubmitAbortedError,
)
from kagglebot.exec_utils import run_command
from kagglebot.hashing import sha256_file
from kagglebot.history import new_run_id
from kagglebot.kaggle_api import (
    check_rules_accepted,
    leaderboard_rank_for_score,
    leaderboard_top1,
    list_competition_submissions,
)
from kagglebot.kernel_runner import (
    _collect_log_tail,
    resolve_kaggle_username,
    run_kernel,
    run_kernel_local,
)
from kagglebot.knowledge import (
    derive_problem_types,
    ensure_taxonomy,
    format_error_fix_insights,
    format_problem_type_insights,
    record_error_fix_insight,
    record_improvement,
    record_iteration,
    record_problem_type_insight,
    record_run,
    resolve_error_fix_insights,
    resolve_problem_type_insights,
    resolve_similar_improvements,
)
from kagglebot.orchestrator.agent_pipeline import (
    AgentPipelineConfig,
    _backup_guarded_files,
    _diff_snapshots,
    _enforce_allowlist_changes,
    _snapshot_tree,
    run_agent_pipeline,
)
from kagglebot.solver.io import load_competition_data
from kagglebot.solver.metrics import canonical_metric, compute_metric, infer_direction, metric_requires_proba
from kagglebot.submission.guard import (
    classify_submit_error,
    compute_error_fingerprint,
    normalize_error_text,
    run_kaggle_submit_kernel,
)
from kagglebot.submission.outcome_service import SubmissionOutcomePollingError, SubmissionOutcomeService
from kagglebot.submission_service import SubmissionConfig, SubmissionService
from kagglebot.types import PlanConfig
from kagglebot.validators import ensure_kernel_sources_valid


# Backward-compatible symbol for tests/extensions.
# Runtime no longer uses the legacy src local trainer path.
def train_evaluate_and_predict(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise RuntimeError(
        "Legacy src local trainer was removed. Use artifacts/<slug>/kernel/kernel.py via autopilot/train commands."
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


MAX_KERNEL_FIX_ATTEMPTS: int | None = 8
MAX_SAME_KERNEL_ERROR_REPEATS = 2
MAX_KERNEL_CAPACITY_RETRIES = 3
KERNEL_CAPACITY_RETRY_SLEEP = 30.0
MAX_KERNEL_CAPACITY_REPEAT = 6
MAX_KERNEL_REGISTRATION_RETRIES = 2
KERNEL_REGISTRATION_RETRY_SLEEP = 15.0
MAX_AUTOFIX_ATTEMPTS = 2
MAX_AUTOFIX_RESTARTS = 1
MAX_AUTOFIX_CODEX_PASSES = 3
MAX_KERNEL_FIX_CODEX_PASSES = 3
MAJOR_TOP1_GAP = 0.03
MODERATE_TOP1_GAP = 0.01
_ERROR_FIX_CODEX_MODEL = "gpt-5.3-codex"
_ERROR_FIX_REASONING_EFFORT = "extra_high"
_ERROR_STRATEGY_MODEL = "gpt-5.2"
_ERROR_STRATEGY_REASONING_EFFORT = "extra_high"
_METRIC_FIX_CODEX_MODEL = "gpt-5.3-codex"
_METRIC_FIX_REASONING_EFFORT = "extra_high"
_MAX_METRIC_FIX_ATTEMPTS = 3
_MAX_METRIC_FIX_CODEX_PASSES = 4
_SUBMISSION_POLL_MAX_ATTEMPTS: int | None = None
_SUBMISSION_POLL_INTERVAL_SEC = 30.0
_SUBMISSION_POLL_MAX_FETCH_ERRORS = 3
_FAILED_SUBMISSION_OUTCOME_STATUSES = {"error", "failed", "cancelled", "canceled"}
_SUBMIT_MAX_TRANSIENT_RETRIES = 3
_SUBMIT_BACKOFF_BASE_SEC = 2.0
_SUBMIT_STDERR_TAIL_CHARS = 1200
_SUBMIT_STDOUT_TAIL_CHARS = 1200
_KERNEL_PUSH_VERSION_RE = re.compile(r"Kernel version\s+(?P<version>\d+)\s+successfully pushed", re.IGNORECASE)
_CODE_SCORE_RE = re.compile(r"(?<!\d)(0\.\d{3,6})(?!\d)")
_NOTEBOOK_FALLBACK_HINTS = (
    "submit-notebook",
    "notebook",
    "/api/v1/competitions/submissions/submit/",
    "submission not allowed",
    "only accepts submissions from notebooks",
    "code competition submissions require both the output file name and the version label",
    "kernel must be specified as <owner>/<notebook>",
)
_ITERATION_STATE_FILENAME = "iteration_state.json"
_LEGACY_SUBMIT_PHASE_COMPLETE_ACTIONS = frozenset({"submit"})
_DEFAULT_EVAL_SEEDS = [42, 2024, 777]
_DEFAULT_EVAL_REPEATS = 2
_DEFAULT_MAX_ITERATIONS = 12
_EVAL_REPEAT_SEED_OFFSET = 1009
_DEFAULT_FORCE_MAJOR_RANK_MAX_PERCENTILE = 0.35
_DEFAULT_FORCE_MAJOR_RANK_MIN_TEAMS = 200
_DEFAULT_LIMITED_SUBMISSION_GATE = "readiness_or_final"
_DEFAULT_STRICT_COMPETITION_METRIC = True
_DEFAULT_REQUIRE_SUBMIT_IMPROVEMENT = True
_DEFAULT_FORCE_MAJOR_ON_NO_IMPROVE = True
_KERNEL_REGENERATE_MARKER_FILENAME = "kernel_regenerated_once.json"
_QUALITY_GUARD_BASELINE_REL_MARGIN = 0.01
_QUALITY_GUARD_BASELINE_ABS_MARGIN = 1e-6
_QUALITY_GUARD_MISMATCH_REL_MARGIN_MINIMIZE = 2.0
_QUALITY_GUARD_MISMATCH_REL_MARGIN_MAXIMIZE = 0.30
_QUALITY_GUARD_MISMATCH_ABS_MARGIN = 0.05
_QUALITY_GUARD_STEP_BUCKET_RATIO = 2.5
_QUALITY_GUARD_CODE_REF_REL_MARGIN = 0.0
_QUALITY_GUARD_CODE_REF_ABS_MARGIN = 0.02
_BEST_SCORE_OUTLIER_TOP1_ABS_MARGIN = 0.02
_BEST_SCORE_OUTLIER_TOP1_REL_MARGIN = 0.01
_REGRESSION_GUARD_ABS_DROP_PROB = 0.03
_REGRESSION_GUARD_ABS_DROP_DEFAULT = 0.10
_CONSERVATIVE_COLLAPSE_MAX_FEATURES = 5
_MAX_KERNEL_PREFLIGHT_FIX_ATTEMPTS = 2
_TRUSTED_SCORE_SOURCES = frozenset({"cv", "holdout", "consensus"})
_BEST_KERNEL_SNAPSHOT_FILENAME = "best_kernel.py"
_CODE_REFERENCE_IMPL_MARKER_PREFIX = "# KAGGLEBOT_CODE_REFERENCE_IMPLEMENTED:"
_NUMBER_WORD_TO_INT = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


@dataclass(frozen=True)
class _CompetitionRuleConstraints:
    notebook_submissions_only: bool = False
    internet_must_be_off: bool = False
    submission_limit_detected: bool = False
    submission_limit_per_day: int | None = None
    cpu_runtime_limit_min: int | None = None
    gpu_runtime_limit_min: int | None = None


@dataclass(frozen=True)
class _CodeReferenceNotebook:
    kernel_id: str
    title: str
    source_file: str | None = None
    local_dir: str | None = None
    summary: str = ""


class _TrainingLiveStdout:
    """Render a single carriage-return live line while preserving regular log lines."""

    def __init__(self, base_stream) -> None:
        self._base_stream = base_stream
        self._last_live_text = ""
        self._live_active = False

    def render_live(self, text: str) -> None:
        self._last_live_text = text
        self._base_stream.write(f"\r{text}")
        self._live_active = True

    def finish_live(self, text: str) -> None:
        self._last_live_text = text
        self._base_stream.write(f"\r{text}\n")
        self._live_active = False

    def write(self, s: str) -> int:
        if not s:
            return 0
        interrupted = False
        if self._live_active and any(ch not in "\r" for ch in s):
            self._base_stream.write("\n")
            self._live_active = False
            interrupted = True
        written = self._base_stream.write(s)
        if interrupted and s.endswith("\n") and self._last_live_text:
            self._base_stream.write(f"\r{self._last_live_text}")
            self._live_active = True
        return written

    def flush(self) -> None:
        self._base_stream.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._base_stream, "isatty", lambda: False)())

    @property
    def encoding(self) -> str | None:
        return getattr(self._base_stream, "encoding", None)

    @property
    def errors(self) -> str | None:
        return getattr(self._base_stream, "errors", None)

    def fileno(self) -> int:
        return self._base_stream.fileno()


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
                if not _is_submit_abort_autofixable(config=config, run_id=run_id):
                    raise
                attempt += 1
                if attempt > MAX_AUTOFIX_ATTEMPTS:
                    raise
                print("[yellow]autofix[/yellow]: submit stage failed; invoking codex to repair and retry submit")
                run_dir = config.paths.run_dir(run_id)
                if (not _has_successful_submit_attempt(run_dir)) or _should_force_resubmit_after_submit_abort(run_dir):
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
                if _is_non_autofixable_runtime_error(exc):
                    raise
                attempt += 1
                if attempt > MAX_AUTOFIX_ATTEMPTS:
                    raise
                print("[yellow]autofix[/yellow]: invoking codex to repair error")
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
        if _should_skip_planning(resume_run=self.resume_run, paths=self.config.paths):
            print("[yellow]resume[/yellow]: skipping planning after restart; reusing existing plan")
            return plan
        if _needs_planning(plan, self.config):
            print("[cyan]plan[/cyan]: generating initial plan")
            _run_plan_and_initial(self.config, self.run_id)
            return _load_plan(self.config.paths)
        return plan


@dataclass(frozen=True)
class KnowledgePhase:
    config: AutopilotConfig

    def refresh(self) -> None:
        _refresh_knowledge_hints(self.config)

    def load_dataset_profile(self) -> dict[str, object]:
        return _load_dataset_profile(self.config.paths)

    def derive_problem_types(self) -> list[str]:
        return derive_problem_types(self.load_dataset_profile())


@dataclass(frozen=True)
class IterationPhase:
    metric_direction: str

    def delta_from_best(self, best_score: float | None, current_score: float) -> float | None:
        if best_score is None:
            return None
        if self.metric_direction == "minimize":
            return best_score - current_score
        return current_score - best_score

    def should_update_best(self, best_score: float | None, current_score: float, min_improvement: float) -> bool:
        return _update_best_score(best_score, current_score, self.metric_direction, min_improvement)


@dataclass(frozen=True)
class SubmissionPhase:
    config: AutopilotConfig
    run_id: str
    problem_types: list[str]

    def attempt(self, *, submission_path: Path, best_score: float | None) -> dict[str, object] | None:
        return _attempt_submit(
            config=self.config,
            run_id=self.run_id,
            submission_path=submission_path,
            best_score=best_score,
            problem_types=self.problem_types,
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
    print(f"[green]run started[/green]: {run_id}")
    planning_phase = PlanningPhase(config=config, run_id=run_id, resume_run=resume_run)
    knowledge_phase = KnowledgePhase(config=config)
    plan = _load_plan(config.paths)
    if not config.paths.plan_path.exists():
        _write_plan(config.paths, plan)

    print(f"[cyan]fetching leaderboard[/cyan]: {config.slug}")
    metric_hint = config.target_metric or plan.target_metric
    top1_info = leaderboard_top1(
        config.slug,
        config.paths.context_dir,
        dry_run=config.dry_run,
        metric_hint=metric_hint,
    )
    config.paths.top1_public_path.write_text(json.dumps(top1_info, indent=2), encoding="utf-8")
    _print_top1_info(top1_info)
    knowledge_phase.refresh()
    plan = planning_phase.execute(plan)

    resolved = _resolve_plan(plan, config)
    target_metric = resolved["target_metric"]
    target_score = resolved["target_score"]
    if target_metric is None or target_score is None:
        run_payload = _build_run_payload(
            run_id=run_id,
            config=config,
            resolved=resolved,
            status="missing_target",
        )
        (run_dir / "run.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
        return

    metric_direction = infer_direction(target_metric, resolved["target_direction"])
    resolved["target_direction"] = metric_direction
    strict_competition_metric = _env_flag(
        "KAGGLEBOT_STRICT_COMPETITION_METRIC",
        default=_DEFAULT_STRICT_COMPETITION_METRIC,
    )
    require_submit_improvement = _env_flag(
        "KAGGLEBOT_REQUIRE_SUBMIT_IMPROVEMENT",
        default=_DEFAULT_REQUIRE_SUBMIT_IMPROVEMENT,
    )
    force_major_on_no_improve = _env_flag(
        "KAGGLEBOT_FORCE_MAJOR_ON_NO_IMPROVE",
        default=_DEFAULT_FORCE_MAJOR_ON_NO_IMPROVE,
    )

    _write_plan(config.paths, _resolved_plan(resolved))
    run_payload = _build_run_payload(
        run_id=run_id,
        config=config,
        resolved=resolved,
        status="running",
    )
    (run_dir / "run.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
    _ensure_best_kernel_snapshot(paths=config.paths, run_dir=run_dir)

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
    submission_phase = SubmissionPhase(config=config, run_id=run_id, problem_types=problem_types)
    best_score = None
    best_submission: Path | None = None
    best_submittable_score: float | None = None
    best_submittable_submission: Path | None = None
    submitted = False
    pending_problem_insights: list[dict[str, object]] = []
    pending_error_fixes: list[dict[str, object]] = []
    last_submission_result: dict[str, object] | None = None

    max_iterations = max(1, int(resolved["max_iterations"]))
    iteration_phase = IterationPhase(metric_direction=metric_direction)
    holdout_frac = float(resolved["holdout_frac"])
    cv_folds = int(resolved["cv_folds"])
    split_strategy = str(resolved.get("split_strategy") or "").strip().lower() or None
    seed = int(resolved["seed"])
    eval_seeds = _normalize_eval_seeds(resolved.get("eval_seeds"), fallback=[seed])
    eval_repeats = _normalize_eval_repeats(resolved.get("eval_repeats"), fallback=_DEFAULT_EVAL_REPEATS)
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
    readiness_target = float(resolved.get("readiness_target_score") or target_score)
    readiness_method = str(resolved.get("readiness_method") or "ci_bound")
    readiness_k = float(resolved.get("readiness_k") or 1.0)
    ci_method = str(resolved.get("ci_method") or "normal")
    ci_alpha = float(resolved.get("ci_alpha") or 0.05)
    drift_check_enabled = bool(resolved.get("drift_check", False))
    drift_weight = float(resolved.get("drift_weight") or 1.0)
    stop_min_delta = float(resolved.get("stop_min_delta") or 0.0)
    stop_no_improve_patience = int(resolved.get("stop_no_improve_patience") or 0)
    stop_same_config_patience = int(resolved.get("stop_same_config_patience") or 0)
    rank_force_major_max_percentile = _normalize_rank_force_percentile(
        resolved.get("rank_force_major_max_percentile"),
        fallback=_DEFAULT_FORCE_MAJOR_RANK_MAX_PERCENTILE,
    )
    rank_force_major_min_teams = _normalize_rank_force_min_teams(
        resolved.get("rank_force_major_min_teams"),
        fallback=_DEFAULT_FORCE_MAJOR_RANK_MIN_TEAMS,
    )
    no_improve_streak = 0
    same_config_streak = 0
    last_config_hash: str | None = None
    eval_data_cache: dict[str, object] | None = None
    previous_readiness_score, noise_limited_streak = _resume_noise_guard_state(
        run_dir=config.paths.run_dir(run_id),
        max_iterations=max_iterations,
    )
    start_iteration, best_score, best_submission = _resume_iteration_state(
        paths=config.paths,
        run_id=run_id,
        metric_direction=metric_direction,
        target_metric=target_metric,
        max_iterations=max_iterations,
        require_submit_phase=config.submit and not config.dry_run,
    )
    best_submitted_score = _resume_best_submitted_offline_score(
        paths=config.paths,
        run_id=run_id,
        metric_direction=metric_direction,
        target_metric=target_metric,
        max_iterations=max_iterations,
    )
    resumed_best_readiness = _resume_best_readiness_score(
        run_dir=config.paths.run_dir(run_id),
        direction=metric_direction,
        max_iterations=max_iterations,
    )
    if resumed_best_readiness is not None and best_score is None:
        best_score = resumed_best_readiness
    best_submittable_score = best_score
    best_submittable_submission = best_submission
    if start_iteration > 1:
        print(f"[yellow]resume[/yellow]: found completed iterations; resuming at {start_iteration}/{max_iterations}")
    loop_started_at = time.monotonic()
    last_completed_iteration = start_iteration - 1

    try:
        for iteration in range(start_iteration, max_iterations + 1):
            last_completed_iteration = iteration
            if max_total_min is not None and max_total_min > 0:
                elapsed_total_min = (time.monotonic() - loop_started_at) / 60.0
                if elapsed_total_min >= float(max_total_min):
                    run_payload["status"] = "stopped"
                    run_payload["stop_reason"] = (
                        f"max_total_min reached: elapsed={elapsed_total_min:.1f}m limit={float(max_total_min):.1f}m"
                    )
                    print(f"[yellow]stop[/yellow]: {run_payload['stop_reason']}")
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

            _run_verify(config.verify_cmd, dry_run=config.dry_run)

            submission_path = iter_dir / "submission.csv"
            metrics_path = iter_dir / "metrics.json"
            evaluation_report_path = iter_dir / "evaluation_report.json"
            evaluation = None
            kernel_metrics_payload: dict[str, object] | None = None
            kernel_metrics_artifact_path: Path | None = None
            evaluation_by_source: dict[str, EvaluationResult] = {}
            model_summary = {}
            accelerator_used = config.accelerator
            submit_retry_resume = _load_submit_retry_artifacts(
                run_dir=run_dir,
                iter_dir=iter_dir,
                iteration=iteration,
                max_iterations=max_iterations,
                metric_direction=metric_direction,
                target_metric=target_metric,
                require_submit_phase=config.submit and not config.dry_run,
            )
            if submit_retry_resume is not None:
                resume_submission_path, resume_metrics_path, resume_evaluation = submit_retry_resume
                if resume_submission_path != submission_path:
                    submission_path = _copy_submission_artifact_to_iteration_dir(
                        source=resume_submission_path,
                        iter_dir=iter_dir,
                    )
                if resume_metrics_path != metrics_path:
                    metrics_path.write_bytes(resume_metrics_path.read_bytes())
                evaluation = resume_evaluation
                kernel_metrics_payload = _load_json_object(resume_metrics_path)
                kernel_metrics_artifact_path = resume_metrics_path
                print(
                    "[yellow]resume[/yellow]: "
                    f"iter-{iteration} has completed training artifacts; retrying submit without retraining."
                )

            if evaluation is None:
                _run_kernel_source_preflight_fixes(
                    config=config,
                    run_id=run_id,
                    iteration=iteration,
                    iter_dir=iter_dir,
                    pending_error_fixes=pending_error_fixes,
                )

            if evaluation is None and config.compute.startswith("kaggle_"):
                kaggle_user = resolve_kaggle_username(config.kaggle_username)
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
                        )
                        if kernel_result.submission_path:
                            submission_path = _copy_submission_artifact_to_iteration_dir(
                                source=kernel_result.submission_path,
                                iter_dir=iter_dir,
                            )
                        _copy_kernel_support_artifacts_to_iteration_dir(
                            kernel_output_dir=kernel_result.output_dir,
                            iter_dir=iter_dir,
                        )
                        if kernel_result.metrics_path and kernel_result.metrics_path.exists():
                            kernel_metrics_artifact_path = kernel_result.metrics_path
                            kernel_metrics_payload = _load_json_object(kernel_result.metrics_path)
                            evaluation = _load_kernel_metrics(
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
                        error_text = _format_kernel_error(exc)
                        _record_kernel_error(
                            logs_dir=logs_dir,
                            attempt=kernel_attempts,
                            error_text=error_text,
                            error_fingerprints=error_fingerprints,
                            output_dir=output_dir,
                        )
                        raise
                    except KernelCapacityError as exc:
                        kernel_attempts += 1
                        error_text = _format_kernel_error(exc)
                        _record_kernel_error(
                            logs_dir=logs_dir,
                            attempt=kernel_attempts,
                            error_text=error_text,
                            error_fingerprints=error_fingerprints,
                            max_repeats=MAX_KERNEL_CAPACITY_REPEAT,
                            output_dir=output_dir,
                        )
                        if kernel_attempts > MAX_KERNEL_CAPACITY_RETRIES:
                            raise
                        wait_seconds = KERNEL_CAPACITY_RETRY_SLEEP * kernel_attempts
                        print(
                            "[yellow]kaggle gpu limit reached[/yellow]: "
                            f"retrying in {wait_seconds:.0f}s (attempt {kernel_attempts})"
                        )
                        time.sleep(wait_seconds)
                        continue
                    except Exception as exc:  # noqa: BLE001
                        if _is_kernel_registration_error(exc):
                            kernel_attempts += 1
                            error_text = _format_kernel_error(exc)
                            _record_kernel_error(
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
                        error_text = _format_kernel_error(exc)
                        try:
                            _record_kernel_error(
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
                        print(f"[yellow]kernel failed[/yellow]: invoking codex to fix (attempt {kernel_attempts})")
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
                        )
                        if kernel_result.submission_path:
                            submission_path = _copy_submission_artifact_to_iteration_dir(
                                source=kernel_result.submission_path,
                                iter_dir=iter_dir,
                            )
                        _copy_kernel_support_artifacts_to_iteration_dir(
                            kernel_output_dir=kernel_result.output_dir,
                            iter_dir=iter_dir,
                        )
                        if kernel_result.metrics_path and kernel_result.metrics_path.exists():
                            kernel_metrics_artifact_path = kernel_result.metrics_path
                            kernel_metrics_payload = _load_json_object(kernel_result.metrics_path)
                            evaluation = _load_kernel_metrics(
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
                        error_text = _format_kernel_error(exc)
                        try:
                            _record_kernel_error(
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
                            f"[yellow]local kernel failed[/yellow]: invoking codex to fix (attempt {kernel_attempts})"
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
            while evaluation.metric and target_metric and (not _metrics_equivalent(evaluation.metric, target_metric)):
                corrected_direction, confident = _infer_metric_direction_for_mismatch(
                    evaluation.metric,
                    metric_direction,
                )
                confidence_text = "high" if confident else "fallback"
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
                            "before invoking Codex."
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
                        "Strict competition metric mode is enabled; applying metric-only Codex fix "
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
                        and (not _metrics_equivalent(evaluation.metric, target_metric))
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
                        )
                        if kernel_result.submission_path:
                            submission_path = _copy_submission_artifact_to_iteration_dir(
                                source=kernel_result.submission_path,
                                iter_dir=iter_dir,
                            )
                        _copy_kernel_support_artifacts_to_iteration_dir(
                            kernel_output_dir=kernel_result.output_dir,
                            iter_dir=iter_dir,
                        )
                        if kernel_result.metrics_path and kernel_result.metrics_path.exists():
                            kernel_metrics_artifact_path = kernel_result.metrics_path
                            kernel_metrics_payload = _load_json_object(kernel_result.metrics_path)
                            evaluation = _load_kernel_metrics(
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
                    _write_plan(config.paths, _resolved_plan(resolved))
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
            report, report_payload, eval_data_cache = _build_iteration_evaluation_report(
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
            _append_run_evaluation_report(run_dir=run_dir, iteration=iteration, payload=report_payload)
            evaluation_report_path = iter_dir / "evaluation_report.json"
            evaluation_report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

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
            effective_best_score, best_score_guard = _effective_best_score_for_progress(
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
            rank_forced_major_overhaul = False
            rank_force_reason: str | None = None
            code_reference_score, code_reference_source = _extract_code_reference_score(config.paths)
            code_reference_delta_vs_current = (
                _score_delta_vs_reference(decision_score, code_reference_score, metric_direction)
                if code_reference_score is not None
                else None
            )
            first_iteration_below_code_reference = bool(
                iteration == 1 and code_reference_delta_vs_current is not None and code_reference_delta_vs_current < 0.0
            )
            score_drop_vs_best = _score_drop_vs_best(
                best_score=best_score,
                current_score=decision_score,
                direction=metric_direction,
            )
            severe_regression_detected = _is_severe_regression_vs_best(
                metric=evaluation.metric,
                direction=metric_direction,
                best_score=best_score,
                current_score=decision_score,
            )
            conservative_feature_collapse = _is_conservative_feature_collapse(kernel_metrics_payload)
            conservative_regression_detected = bool(severe_regression_detected and conservative_feature_collapse)

            quality_guard = _build_kernel_quality_guard(
                evaluation=evaluation,
                kernel_metrics_payload=kernel_metrics_payload,
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
            non_generalizable_eval_detected = any(
                reason in {"untrusted_score_source", "oracle_override_detected"} for reason in quality_reasons
            )
            quality_forced_major_overhaul = "below_code_reference_baseline" in quality_reasons
            quality_force_reason: str | None = None
            if quality_forced_major_overhaul:
                if code_reference_score is not None:
                    code_delta = _score_delta_vs_reference(decision_score, code_reference_score, metric_direction)
                    quality_force_reason = (
                        "Offline score is materially below code reference baseline: "
                        f"current={decision_score:.6f}, code_ref={code_reference_score:.6f}, "
                        f"delta={code_delta:+.6f}, source={code_reference_source or 'unknown'}."
                    )
                else:
                    quality_force_reason = (
                        "Offline score is materially below code reference baseline detected by quality guard."
                    )
            if config.submit and (not quality_allows_submit) and (not config.force_submit):
                reason_text = ", ".join(quality_reasons) if quality_reasons else "quality_guard_blocked_submit"
                print(
                    "[yellow]submit blocked[/yellow]: kernel quality guard detected unstable evaluation "
                    f"({reason_text}); submission is deferred to a later iteration."
                )
            if quality_allows_submit or config.force_submit:
                if _update_best_score(best_submittable_score, decision_score, metric_direction, 0.0):
                    best_submittable_score = decision_score
                    best_submittable_submission = submission_path

            is_final_iteration = iteration >= max_iterations
            successful_submit_count = _count_successful_submit_attempts(run_dir)
            submit_improvement_allowed = True
            submit_non_improving = False
            if require_submit_improvement and not config.force_submit and best_submitted_score is not None:
                submit_improvement_allowed = _update_best_score(
                    best_submitted_score,
                    decision_score,
                    metric_direction,
                    stop_min_delta,
                )
                if not submit_improvement_allowed:
                    if is_final_iteration:
                        print(
                            "[yellow]submit override[/yellow]: final iteration reached; "
                            "allowing submit even though offline metric did not improve."
                        )
                        submit_improvement_allowed = True
                    else:
                        submit_non_improving = True
                        print(
                            "[yellow]submit deferred[/yellow]: "
                            "offline metric did not improve over previous submitted checkpoint."
                        )
            allow_submit = _should_attempt_submit_for_readiness(
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
            if (not quality_allows_submit) and (not config.force_submit):
                allow_submit = False
            submit_non_improving = config.submit and submit_non_improving
            submit_limited_holdback = False
            if (
                config.submit
                and submission_limit_per_day is not None
                and quality_allows_submit
                and submit_improvement_allowed
            ):
                reserve_start = max(0, submission_limit_per_day - 1)
                strict_limit_mode = max_iterations > submission_limit_per_day
                if (successful_submit_count >= reserve_start and not allow_submit) or (
                    strict_limit_mode and (not allow_submit)
                ):
                    submit_limited_holdback = True
                    if successful_submit_count >= reserve_start:
                        print(
                            "[yellow]submit deferred[/yellow]: reserved final submission slot "
                            "until offline score reaches top1-tier, readiness target, or final iteration."
                        )
                    else:
                        print(
                            "[yellow]submit deferred[/yellow]: strict limited-submission cadence "
                            "is active because daily limit is lower than max iterations."
                        )
            submit_phase_required = config.submit and not config.dry_run
            submit_allowed_by_gate = config.submit and allow_submit
            pre_submit_phase_state = "disabled"
            if config.submit:
                pre_submit_phase_state = "pending_submit"
            if config.submit and (not quality_allows_submit) and (not config.force_submit):
                pre_submit_phase_state = "blocked_quality_guard"
            if submit_non_improving:
                pre_submit_phase_state = "deferred_non_improving"
            if submit_limited_holdback:
                pre_submit_phase_state = "deferred_for_final_slot"
            pre_submit_phase_finished = (not submit_phase_required) or (not submit_allowed_by_gate)
            pre_submit_metrics_payload = _build_metrics_payload(
                run_id=run_id,
                iteration=iteration,
                evaluation=evaluation,
                target_score=target_score,
                met_target=_meets_target(decision_score, target_score, metric_direction),
                top1_info=top1_info if isinstance(top1_info, dict) else {},
                compute=config.compute,
                accelerator=accelerator_used,
                holdout_frac=holdout_frac,
                cv_folds=cv_folds,
                seed=seed,
                evaluation_by_source=evaluation_by_source,
                evaluation_report=report,
                readiness_target=readiness_target,
            )
            pre_submit_metrics_payload["checkpoint_phase"] = "pre_submit"
            pre_submit_metrics_payload["quality_guard"] = quality_guard
            metrics_path.write_text(json.dumps(pre_submit_metrics_payload, indent=2), encoding="utf-8")
            _write_iteration_state_marker(
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
                submitted=False,
                readiness_score=readiness_score,
            )
            submission_result: dict[str, object] | None = None
            if submit_non_improving:
                submit_phase_state = "deferred_non_improving"
            elif submit_limited_holdback:
                submit_phase_state = "deferred_for_final_slot"
            else:
                submit_phase_state = "disabled"
            if config.submit and allow_submit:
                try:
                    submission_result = submission_phase.attempt(
                        submission_path=submission_path,
                        best_score=decision_score,
                    )
                except SubmitAbortedError:
                    run_payload["status"] = "submit_failed"
                    (run_dir / "run.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
                    raise
                if submission_result:
                    submit_phase_state = "submitted"
                    submitted = True
                    last_submission_result = submission_result
                    if _update_best_score(best_submitted_score, decision_score, metric_direction, 0.0):
                        best_submitted_score = decision_score
                    outcome_payload = submission_result.get("outcome")
                    if isinstance(outcome_payload, dict):
                        online_score = _to_float(outcome_payload.get("score"))
                        if online_score is not None:
                            print(f"[cyan]submission score[/cyan]: {online_score:.6f}")
                            if isinstance(top1_score, (int, float)):
                                top1_tier_by_submission = _is_top1_tier(
                                    float(online_score),
                                    float(top1_score),
                                    metric_direction,
                                )
                        rank_payload = _resolve_submission_rank_payload(
                            slug=config.slug,
                            context_dir=config.paths.context_dir,
                            direction=metric_direction,
                            outcome=outcome_payload,
                            dry_run=config.dry_run,
                        )
                        if rank_payload:
                            outcome_payload.update(rank_payload)
                            submission_rank = _to_int(rank_payload.get("rank"))
                            submission_total_teams = _to_int(rank_payload.get("total_teams"))
                            submission_rank_percentile = _to_float(rank_payload.get("rank_percentile"))
                            submission_rank_estimate = _to_int(rank_payload.get("estimated_rank"))
                            submission_total_teams_estimate = _to_int(rank_payload.get("estimated_total_teams"))
                            submission_rank_percentile_estimate = _to_float(
                                rank_payload.get("estimated_rank_percentile")
                            )
                            estimate_source_raw = rank_payload.get("rank_estimate_source")
                            if isinstance(estimate_source_raw, str) and estimate_source_raw.strip():
                                submission_rank_estimate_source = estimate_source_raw.strip()
                            source_raw = rank_payload.get("rank_source")
                            if source_raw is not None:
                                submission_rank_source = str(source_raw)
                            if (
                                submission_rank is not None
                                and submission_total_teams is not None
                                and submission_total_teams > 0
                            ):
                                if submission_rank_percentile is None:
                                    submission_rank_percentile = submission_rank / submission_total_teams
                                percentile_text = (
                                    f"{submission_rank_percentile * 100:.2f}%"
                                    if submission_rank_percentile is not None
                                    else "n/a"
                                )
                                source_text = f" source={submission_rank_source}" if submission_rank_source else ""
                                print(
                                    "[cyan]submission rank[/cyan]: "
                                    f"{submission_rank}/{submission_total_teams} "
                                    f"(percentile={percentile_text}){source_text}"
                                )
                                rank_forced_major_overhaul = _should_force_major_overhaul_by_rank(
                                    rank=submission_rank,
                                    total_teams=submission_total_teams,
                                    max_percentile=rank_force_major_max_percentile,
                                    min_teams=rank_force_major_min_teams,
                                )
                                if rank_forced_major_overhaul:
                                    rank_force_reason = _build_rank_force_reason(
                                        rank=submission_rank,
                                        total_teams=submission_total_teams,
                                        rank_percentile=submission_rank_percentile,
                                        max_percentile=rank_force_major_max_percentile,
                                        min_teams=rank_force_major_min_teams,
                                        source=submission_rank_source,
                                    )
                                    print(f"[yellow]rank guard[/yellow]: {rank_force_reason}")
                            elif (
                                submission_rank_estimate is not None
                                and submission_total_teams_estimate is not None
                                and submission_total_teams_estimate > 0
                            ):
                                if submission_rank_percentile_estimate is None:
                                    submission_rank_percentile_estimate = (
                                        submission_rank_estimate / submission_total_teams_estimate
                                    )
                                percentile_text = (
                                    f"{submission_rank_percentile_estimate * 100:.2f}%"
                                    if submission_rank_percentile_estimate is not None
                                    else "n/a"
                                )
                                source_text = (
                                    f" source={submission_rank_estimate_source}"
                                    if submission_rank_estimate_source
                                    else ""
                                )
                                print(
                                    "[yellow]submission rank estimate[/yellow]: "
                                    f"{submission_rank_estimate}/{submission_total_teams_estimate} "
                                    f"(percentile={percentile_text}){source_text}"
                                )
                else:
                    submit_phase_state = "dry_run" if config.dry_run else "attempted_no_result"
            met_target = _meets_target(decision_score, target_score, metric_direction)
            top1_tier = _is_top1_tier(decision_score, top1_score, metric_direction)
            top1_tier_by_readiness = _is_top1_tier(readiness_score, top1_score, metric_direction)
            delta_srs_vs_prev: float | None = None
            noise_threshold = 0.5 * max(float(report.std), 0.0)
            if previous_readiness_score is not None:
                delta_srs_vs_prev = abs(readiness_score - previous_readiness_score)
                if delta_srs_vs_prev < noise_threshold:
                    noise_limited_streak += 1
                else:
                    noise_limited_streak = 0
            previous_readiness_score = readiness_score
            noise_forced_major_overhaul = noise_limited_streak >= 2
            code_reference_forced_reproduction = bool(
                first_iteration_below_code_reference or conservative_regression_detected
            )
            code_reference_force_reason: str | None = None
            if first_iteration_below_code_reference and code_reference_score is not None:
                code_reference_force_reason = (
                    "First iteration is below /code reference baseline; "
                    f"current={decision_score:.6f}, code_ref={code_reference_score:.6f}, "
                    f"delta={float(code_reference_delta_vs_current):+.6f}. "
                    "Next iteration must implement the required reference notebook path."
                )
            elif conservative_regression_detected:
                drop_text = (
                    f"{float(score_drop_vs_best):.6f}" if isinstance(score_drop_vs_best, (int, float)) else "unknown"
                )
                code_reference_force_reason = (
                    "Detected severe regression with conservative feature collapse "
                    f"(drop_vs_best={drop_text}, max_features={_CONSERVATIVE_COLLAPSE_MAX_FEATURES}). "
                    "Next iteration must recover from code reference baseline instead of keeping the collapsed path."
                )
            force_major_overhaul_next = (
                noise_forced_major_overhaul
                or rank_forced_major_overhaul
                or quality_forced_major_overhaul
                or code_reference_forced_reproduction
            )
            forced_major_overhaul_reasons: list[str] = []
            if noise_forced_major_overhaul:
                forced_major_overhaul_reasons.append(
                    "Two consecutive iterations were noise-limited: "
                    f"|ΔSRS| < 0.5*CV std (streak={noise_limited_streak})."
                )
            if rank_forced_major_overhaul:
                forced_major_overhaul_reasons.append(
                    rank_force_reason or "Leaderboard rank indicates major improvement is still required."
                )
            if quality_forced_major_overhaul:
                forced_major_overhaul_reasons.append(
                    quality_force_reason
                    or "Quality guard requires major overhaul due to code-reference underperformance."
                )
            if code_reference_forced_reproduction:
                forced_major_overhaul_reasons.append(
                    code_reference_force_reason
                    or "Mandatory code-reference implementation is required in the next iteration."
                )
            forced_major_overhaul_reason = (
                " ".join(forced_major_overhaul_reasons) if forced_major_overhaul_reasons else None
            )

            metrics_payload = _build_metrics_payload(
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
            )
            metrics_payload["loop_decision"] = {
                "source": decision_source,
                "value": decision_score,
            }
            metrics_payload["noise_guard"] = {
                "delta_srs_vs_prev": delta_srs_vs_prev,
                "threshold": noise_threshold,
                "streak": noise_limited_streak,
                "force_major_overhaul_next": force_major_overhaul_next,
            }
            metrics_payload["rank_guard"] = {
                "rank": submission_rank,
                "total_teams": submission_total_teams,
                "rank_percentile": submission_rank_percentile,
                "rank_source": submission_rank_source,
                "estimated_rank": submission_rank_estimate,
                "estimated_total_teams": submission_total_teams_estimate,
                "estimated_rank_percentile": submission_rank_percentile_estimate,
                "rank_estimate_source": submission_rank_estimate_source,
                "max_percentile": rank_force_major_max_percentile,
                "min_teams": rank_force_major_min_teams,
                "force_major_overhaul_next": rank_forced_major_overhaul,
            }
            metrics_payload["top1_tier"] = {
                "offline_decision": top1_tier,
                "offline_readiness": top1_tier_by_readiness,
                "submission_score": top1_tier_by_submission,
            }
            if best_score_guard is not None:
                metrics_payload["best_score_guard"] = best_score_guard
            metrics_payload["quality_guard"] = quality_guard
            metrics_payload["regression_guard"] = {
                "best_score_before_iteration": best_score,
                "score_drop_vs_best": score_drop_vs_best,
                "severe_regression_detected": severe_regression_detected,
                "conservative_feature_collapse": conservative_feature_collapse,
                "conservative_regression_detected": conservative_regression_detected,
                "first_iteration_below_code_reference": first_iteration_below_code_reference,
                "code_reference_score": code_reference_score,
                "code_reference_delta_vs_current": code_reference_delta_vs_current,
                "code_reference_forced_reproduction": code_reference_forced_reproduction,
            }
            metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

            diff_summary = "Diff tracking disabled (git integration removed)."
            diagnostics = _build_diagnostics(
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
            )
            (iter_dir / "diagnostics.md").write_text(diagnostics, encoding="utf-8")

            submit_allowed_by_gate = config.submit and allow_submit
            submit_phase_finished = (
                (not submit_phase_required) or (not submit_allowed_by_gate) or (submission_result is not None)
            )

            iteration_record_kwargs = {
                "knowledge_paths": config.knowledge_paths,
                "run_id": run_id,
                "iteration": iteration,
                "score_source": evaluation.score_source,
                "offline_value": evaluation.value,
                "offline_std": evaluation.std,
                "top1_public_score": top1_info.get("score") if isinstance(top1_info, dict) else None,
                "met_target": met_target,
                "git_commit": None,
            }
            try:
                record_iteration(**iteration_record_kwargs)
            except TypeError as exc:
                if "submit_phase_finished" not in str(exc):
                    raise
                from kagglebot.knowledge import record_iteration as _record_iteration_canonical

                try:
                    _record_iteration_canonical(
                        **iteration_record_kwargs,
                        submit_phase_finished=submit_phase_finished,
                    )
                except TypeError as fallback_exc:
                    if "submit_phase_finished" not in str(fallback_exc):
                        raise
                    _record_iteration_canonical(**iteration_record_kwargs)
            _write_iteration_state_marker(
                iter_dir=iter_dir,
                run_id=run_id,
                iteration=iteration,
                submission_path=submission_path,
                metrics_path=metrics_path,
                evaluation_report_path=evaluation_report_path,
                submit_phase_required=submit_phase_required,
                submit_phase_finished=submit_phase_finished,
                submit_allowed_by_gate=submit_allowed_by_gate,
                submit_phase_state=submit_phase_state,
                submitted=submission_result is not None,
                readiness_score=readiness_score,
            )

            prev_best = best_score
            if metric_mismatch_detected or non_generalizable_eval_detected:
                delta_offline = None
                improved = False
            else:
                delta_offline = iteration_phase.delta_from_best(prev_best, decision_score)
                improved = iteration_phase.should_update_best(best_score, decision_score, stop_min_delta)
            if improved:
                best_score = decision_score
                best_submission = submission_path
                no_improve_streak = 0
                _capture_best_kernel_snapshot(paths=config.paths, run_dir=run_dir)
            else:
                no_improve_streak += 1
                if conservative_regression_detected:
                    restored = _restore_best_kernel_snapshot(paths=config.paths, run_dir=run_dir)
                    if restored:
                        print(
                            "[yellow]kernel regression guard[/yellow]: "
                            "restored best-known kernel source after severe conservative regression."
                        )

            if force_major_on_no_improve and (not improved):
                if best_score_guard is not None:
                    print(
                        "[yellow]improve guard[/yellow]: "
                        "skipping no-improve major-overhaul override because previous best "
                        "was clipped as an outlier."
                    )
                else:
                    force_major_overhaul_next = True
                    regression_reason = (
                        f"Offline {evaluation.metric} did not improve "
                        f"(current={decision_score:.6f}, best={float(prev_best):.6f})."
                        if prev_best is not None
                        else f"Offline {evaluation.metric} did not improve."
                    )
                    forced_major_overhaul_reason = (
                        f"{forced_major_overhaul_reason} {regression_reason}".strip()
                        if forced_major_overhaul_reason
                        else regression_reason
                    )

            current_config_hash = _pipeline_config_hash(
                model_summary=model_summary,
                metric=evaluation.metric,
                accelerator=accelerator_used,
            )
            if current_config_hash == last_config_hash:
                same_config_streak += 1
            else:
                same_config_streak = 0
            last_config_hash = current_config_hash

            if (not config.submit) and stop_no_improve_patience > 0 and no_improve_streak >= stop_no_improve_patience:
                run_payload["status"] = "stopped"
                run_payload["stop_reason"] = (
                    f"offline metric did not improve by >= {stop_min_delta:.6f} "
                    f"for {no_improve_streak} consecutive iterations"
                )
                print(f"[yellow]stop[/yellow]: {run_payload['stop_reason']}")
                break
            if (
                (not config.submit)
                and stop_same_config_patience > 0
                and same_config_streak >= stop_same_config_patience
            ):
                run_payload["status"] = "stopped"
                run_payload["stop_reason"] = (
                    f"model/pipeline config hash unchanged for {same_config_streak} consecutive iterations"
                )
                print(f"[yellow]stop[/yellow]: {run_payload['stop_reason']}")
                break

            if _is_confirmed_first_place(submission_rank, submission_rank_source):
                run_payload["status"] = "submitted" if submitted else "completed"
                run_payload["stop_reason"] = "submission_rank_1"
                print("[green]stop[/green]: submission rank reached #1")
                break

            if top1_tier:
                print("[yellow]note[/yellow]: offline top1-tier reached; awaiting submission-score confirmation")

            if iteration >= max_iterations:
                run_payload["status"] = "submitted" if submitted else "completed"
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
                forced_improvement_mode="major_overhaul" if force_major_overhaul_next else None,
                forced_improvement_reason=forced_major_overhaul_reason,
                enforce_code_reference_implementation=code_reference_forced_reproduction,
                code_reference_enforcement_reason=code_reference_force_reason,
                best_score_so_far=best_score,
            )
    except KeyboardInterrupt:
        run_payload["status"] = "interrupted"
        (run_dir / "run.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
        print("[yellow]run interrupted[/yellow]")
        return

    if config.submit and not submitted and best_submittable_submission is not None:
        final_iteration_reached = last_completed_iteration >= max_iterations
        allow_fallback_submit = True
        if (
            require_submit_improvement
            and not config.force_submit
            and best_submittable_score is not None
            and best_submitted_score is not None
        ):
            allow_fallback_submit = _update_best_score(
                best_submitted_score,
                best_submittable_score,
                metric_direction,
                stop_min_delta,
            )
            if (not allow_fallback_submit) and final_iteration_reached:
                print(
                    "[yellow]submit override[/yellow]: final iteration reached; "
                    "allowing fallback submit even though offline metric did not improve."
                )
                allow_fallback_submit = True
        if allow_fallback_submit:
            try:
                fallback_result = submission_phase.attempt(
                    submission_path=best_submittable_submission,
                    best_score=best_submittable_score,
                )
            except SubmitAbortedError:
                run_payload["status"] = "submit_failed"
                (run_dir / "run.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
                raise
            if fallback_result:
                submitted = True
                last_submission_result = fallback_result
                if best_submittable_score is not None and _update_best_score(
                    best_submitted_score,
                    best_submittable_score,
                    metric_direction,
                    0.0,
                ):
                    best_submitted_score = best_submittable_score
        else:
            print(
                "[yellow]submit skipped[/yellow]: fallback artifact is not better "
                "than previously submitted offline score."
            )
    elif config.submit and not submitted and best_submission is not None and best_submittable_submission is None:
        print(
            "[yellow]submit skipped[/yellow]: no quality-eligible fallback artifact "
            "(all candidates were blocked by quality guard)."
        )

    if submitted and last_submission_result:
        top1_score = top1_info.get("score") if isinstance(top1_info, dict) else None
        _record_submission_knowledge(
            config=config,
            run_id=run_id,
            problem_types=problem_types,
            pending_problem_insights=pending_problem_insights,
            pending_error_fixes=pending_error_fixes,
            submission_result=last_submission_result,
            metric_direction=metric_direction,
            target_score=target_score,
            top1_score=top1_score if isinstance(top1_score, (int, float)) else None,
        )
        run_payload["status"] = "submitted"
    elif run_payload.get("status") not in {"interrupted", "submit_failed"}:
        run_payload["status"] = "completed"

    (run_dir / "run.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")


def _evaluation_to_payload(evaluation: EvaluationResult) -> dict[str, object]:
    payload: dict[str, object] = {
        "score_source": evaluation.score_source,
        "metric": evaluation.metric,
        "direction": evaluation.direction,
        "value": evaluation.value,
        "std": evaluation.std,
    }
    if evaluation.fold_scores is not None:
        payload["fold_scores"] = list(evaluation.fold_scores)
    if evaluation.train_score is not None:
        payload["train_score"] = evaluation.train_score
    if evaluation.val_score is not None:
        payload["val_score"] = evaluation.val_score
    return payload


def _load_plan(paths: CompetitionPaths) -> PlanConfig:
    if not paths.plan_path.exists():
        return PlanConfig()
    try:
        payload = json.loads(paths.plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return PlanConfig()
    return PlanConfig.from_dict(payload)


def _write_plan(paths: CompetitionPaths, plan: PlanConfig) -> None:
    existing: dict[str, object] = {}
    if paths.plan_path.exists():
        try:
            loaded = json.loads(paths.plan_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except json.JSONDecodeError:
            existing = {}
    payload = {**existing, **plan.to_dict()}
    paths.plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _needs_planning(plan: PlanConfig, config: AutopilotConfig) -> bool:
    if config.agent in ("codex", "pipeline"):
        return True
    target_metric = config.target_metric or plan.target_metric
    target_score = config.target_score if config.target_score is not None else plan.target_score
    target_direction = config.target_direction or plan.target_direction
    if target_metric is None or target_score is None:
        return True
    return target_direction in (None, "auto")


def _should_skip_planning(*, resume_run: bool, paths: CompetitionPaths) -> bool:
    if not resume_run:
        return False
    if not paths.plan_path.exists():
        return False
    kernel_path = paths.kernel_source_dir / "kernel.py"
    return kernel_path.exists()


def _extract_submission_limit_per_day(lowered_rules_text: str) -> int | None:
    """Extract an explicit `N submissions per day` value from rules text."""
    candidates: list[int] = []

    for match in re.finditer(r"\((\d+)\)\s+submissions?\s+per\s+day", lowered_rules_text):
        candidates.append(int(match.group(1)))

    for match in re.finditer(r"\b(\d+)\s+submissions?\s+per\s+day\b", lowered_rules_text):
        candidates.append(int(match.group(1)))

    for match in re.finditer(r"\b([a-z]+)\s+submissions?\s+per\s+day\b", lowered_rules_text):
        number_word = match.group(1)
        if number_word in _NUMBER_WORD_TO_INT:
            candidates.append(_NUMBER_WORD_TO_INT[number_word])

    positive = [value for value in candidates if value > 0]
    if not positive:
        return None
    return min(positive)


def _load_competition_rule_constraints(paths: CompetitionPaths) -> _CompetitionRuleConstraints:
    text_parts: list[str] = []
    for path in (paths.rules_md_path, paths.rules_html_path):
        if not path.exists():
            continue
        try:
            text_parts.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    if not text_parts:
        return _CompetitionRuleConstraints()
    text = "\n".join(text_parts)
    lowered = text.lower()

    notebook_submissions_only = bool(
        re.search(r"submissions?\s+to\s+this\s+competition\s+must\s+be\s+made\s+through\s+notebooks?", lowered)
        or re.search(r"submissions?\s+must\s+be\s+made\s+through\s+notebooks?", lowered)
        or re.search(r"only\s+accepts?\s+submissions?\s+from\s+notebooks?", lowered)
    )
    internet_must_be_off = bool(
        re.search(r"internet\s+access\s+disabled", lowered)
        or re.search(r"enable[_\s-]?internet\s*[:=]\s*false", lowered)
    )
    submission_limit_per_day = _extract_submission_limit_per_day(lowered)
    submission_limit_detected = bool(
        re.search(r"maximum\s+number\s+of\s+submissions", lowered)
        or re.search(r"submission\s+limit", lowered)
        or re.search(r"\bmax(?:imum)?\s+submissions?\b", lowered)
        or re.search(r"\b\d+\s+submissions?\s+per\s+day\b", lowered)
        or re.search(r"\bdaily\s+submissions?\b", lowered)
        or submission_limit_per_day is not None
    )

    cpu_runtime_limit_min: int | None = None
    gpu_runtime_limit_min: int | None = None
    for match in re.finditer(
        r"\b(cpu|gpu)\s+notebook\s*<=?\s*(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b",
        lowered,
    ):
        device = match.group(1)
        hours = float(match.group(2))
        minutes = max(1, int(round(hours * 60)))
        if device == "cpu":
            cpu_runtime_limit_min = minutes if cpu_runtime_limit_min is None else min(cpu_runtime_limit_min, minutes)
        else:
            gpu_runtime_limit_min = minutes if gpu_runtime_limit_min is None else min(gpu_runtime_limit_min, minutes)

    return _CompetitionRuleConstraints(
        notebook_submissions_only=notebook_submissions_only,
        internet_must_be_off=internet_must_be_off,
        submission_limit_detected=submission_limit_detected,
        submission_limit_per_day=submission_limit_per_day,
        cpu_runtime_limit_min=cpu_runtime_limit_min,
        gpu_runtime_limit_min=gpu_runtime_limit_min,
    )


def _runtime_limit_for_compute(*, constraints: _CompetitionRuleConstraints, compute: str) -> int | None:
    normalized = str(compute or "").strip().lower()
    if normalized == "kaggle_gpu":
        return constraints.gpu_runtime_limit_min or constraints.cpu_runtime_limit_min
    if normalized == "kaggle_tpu":
        limits = [value for value in (constraints.gpu_runtime_limit_min, constraints.cpu_runtime_limit_min) if value]
        if limits:
            return min(limits)
        return None
    return None


def _resolve_plan(plan: PlanConfig, config: AutopilotConfig) -> dict[str, object]:
    def choose(value, fallback, default):
        if value is not None:
            return value
        if fallback is not None:
            return fallback
        return default

    eval_spec = _load_evaluation_spec(config.paths)
    spec_metric = eval_spec.get("metric_name") if isinstance(eval_spec.get("metric_name"), str) else None
    spec_direction = eval_spec.get("direction") if isinstance(eval_spec.get("direction"), str) else None
    spec_split = eval_spec.get("split_strategy") if isinstance(eval_spec.get("split_strategy"), str) else None
    spec_folds = eval_spec.get("n_splits") if isinstance(eval_spec.get("n_splits"), int) else None
    spec_seed: int | None = None
    spec_eval_seeds: list[int] = []
    spec_seeds = eval_spec.get("seeds")
    if isinstance(spec_seeds, list):
        for item in spec_seeds:
            if isinstance(item, int):
                spec_eval_seeds.append(item)
                if spec_seed is None:
                    spec_seed = item
    spec_repeats = eval_spec.get("repeats") if isinstance(eval_spec.get("repeats"), int) else None
    spec_ci_method = eval_spec.get("ci_method") if isinstance(eval_spec.get("ci_method"), str) else None
    spec_ci_alpha = eval_spec.get("ci_alpha") if isinstance(eval_spec.get("ci_alpha"), (int, float)) else None
    readiness_rule = eval_spec.get("readiness_rule") if isinstance(eval_spec.get("readiness_rule"), dict) else {}
    spec_readiness_method = readiness_rule.get("method") if isinstance(readiness_rule.get("method"), str) else None
    spec_readiness_k = readiness_rule.get("k") if isinstance(readiness_rule.get("k"), (int, float)) else None
    spec_readiness_target = (
        readiness_rule.get("target_score") if isinstance(readiness_rule.get("target_score"), (int, float)) else None
    )
    spec_submission_gate = (
        readiness_rule.get("submission_gate") if isinstance(readiness_rule.get("submission_gate"), str) else None
    )
    drift_cfg = eval_spec.get("drift_check") if isinstance(eval_spec.get("drift_check"), dict) else {}
    spec_drift_enabled = drift_cfg.get("enabled") if isinstance(drift_cfg.get("enabled"), bool) else None
    spec_drift_weight = (
        drift_cfg.get("drift_weight") if isinstance(drift_cfg.get("drift_weight"), (int, float)) else None
    )
    stop_policy = eval_spec.get("stop_policy") if isinstance(eval_spec.get("stop_policy"), dict) else {}
    spec_stop_min_delta = (
        stop_policy.get("min_delta") if isinstance(stop_policy.get("min_delta"), (int, float)) else None
    )
    spec_stop_no_improve = (
        stop_policy.get("no_improve_patience") if isinstance(stop_policy.get("no_improve_patience"), int) else None
    )
    spec_stop_same_config = (
        stop_policy.get("same_config_patience") if isinstance(stop_policy.get("same_config_patience"), int) else None
    )

    strict_competition_metric = _env_flag(
        "KAGGLEBOT_STRICT_COMPETITION_METRIC",
        default=_DEFAULT_STRICT_COMPETITION_METRIC,
    )
    target_metric = choose(config.target_metric, plan.target_metric, spec_metric)
    target_score = choose(config.target_score, plan.target_score, spec_readiness_target)
    target_direction = choose(config.target_direction, plan.target_direction, spec_direction or "auto")
    if strict_competition_metric and spec_metric:
        requested_metric = target_metric if isinstance(target_metric, str) else None
        requested_metric_norm = _canonical_metric_name_for_match(requested_metric)
        spec_metric_norm = _canonical_metric_name_for_match(spec_metric)
        if requested_metric_norm != spec_metric_norm:
            if requested_metric:
                print(
                    "[yellow]note[/yellow]: strict competition metric mode is enabled; "
                    f"overriding target_metric '{requested_metric}' -> '{spec_metric}'."
                )
            target_metric = spec_metric
        if spec_direction in {"minimize", "maximize"}:
            requested_direction = str(target_direction or "").strip().lower()
            if requested_direction != spec_direction:
                print(
                    "[yellow]note[/yellow]: strict competition metric mode is enabled; "
                    f"overriding target_direction '{requested_direction or 'auto'}' -> '{spec_direction}'."
                )
            target_direction = spec_direction
    score_source = str(choose(config.score_source, plan.score_source, "cv") or "cv")
    normalized_score_source = _normalize_score_source_name(score_source)
    if normalized_score_source not in {"cv", "holdout"}:
        print("[yellow]note[/yellow]: non-generalizable score_source is not allowed; overriding to cv.")
        score_source = "cv"
    holdout_frac = choose(config.holdout_frac, plan.holdout_frac, 0.2)
    cv_folds = choose(config.cv_folds, plan.cv_folds, spec_folds if spec_folds is not None else 5)
    split_strategy = choose(None, plan.split_strategy, spec_split)
    seed = choose(config.seed, plan.seed, spec_seed if spec_seed is not None else 42)
    eval_seeds = _normalize_eval_seeds(plan.eval_seeds, fallback=spec_eval_seeds)
    if len(eval_seeds) < 2:
        print(
            "[yellow]note[/yellow]: evaluation seeds were single-seed; "
            f"upgrading to multi-seed defaults {_DEFAULT_EVAL_SEEDS}."
        )
        eval_seeds = list(_DEFAULT_EVAL_SEEDS)
    eval_repeats = _normalize_eval_repeats(plan.eval_repeats, fallback=spec_repeats)
    if eval_repeats < 2:
        print(
            "[yellow]note[/yellow]: evaluation repeats were < 2; "
            f"upgrading to default {_DEFAULT_EVAL_REPEATS} to reduce noise."
        )
        eval_repeats = _DEFAULT_EVAL_REPEATS
    constraints = _load_competition_rule_constraints(config.paths)
    time_budget_min = choose(config.time_budget_min, plan.time_budget_min, None)
    kernel_name = choose(config.kernel_name, plan.kernel_name, None)
    internet = choose(config.internet, plan.internet, "on")
    if internet in (None, "auto"):
        internet = "on"
    if constraints.notebook_submissions_only and not str(config.compute).startswith("kaggle_"):
        print(
            "[yellow]note[/yellow]: competition requires notebook-based submissions; "
            "autopilot will auto-switch submit mode to notebook submit."
        )
    if constraints.internet_must_be_off and str(internet).strip().lower() != "off":
        print("[yellow]note[/yellow]: rules require internet disabled; forcing internet=off.")
        internet = "off"
    runtime_limit_min = _runtime_limit_for_compute(constraints=constraints, compute=config.compute)
    if runtime_limit_min is not None:
        current_limit = int(time_budget_min) if isinstance(time_budget_min, (int, float)) else None
        if current_limit is None or current_limit > runtime_limit_min:
            print(
                "[yellow]note[/yellow]: rules impose notebook runtime cap; "
                f"forcing time_budget_min={runtime_limit_min}."
            )
            time_budget_min = runtime_limit_min
    if config.max_iterations is None:
        planned_max_iterations = _to_int(plan.max_iterations)
        if planned_max_iterations is not None and planned_max_iterations > 0:
            max_iterations = planned_max_iterations
        else:
            max_iterations = _DEFAULT_MAX_ITERATIONS
            if planned_max_iterations is not None:
                print(
                    "[yellow]note[/yellow]: invalid plan max_iterations "
                    f"({plan.max_iterations}); using default {_DEFAULT_MAX_ITERATIONS}."
                )
    else:
        max_iterations = max(1, int(config.max_iterations))
    max_total_min = choose(config.max_total_min, plan.max_total_min, None)
    patience = choose(config.patience, plan.patience, 2)
    min_improvement = choose(config.min_improvement, plan.min_improvement, 0.0)
    requested_submit_policy = str(choose(None, plan.submit_policy, "always") or "always")
    requested_submission_gate_raw = choose(None, plan.submission_gate, spec_submission_gate)
    requested_submission_gate = str(requested_submission_gate_raw or "").strip().lower() or None
    if constraints.submission_limit_detected:
        submit_policy = _normalized_submit_policy(requested_submit_policy)
        default_gate = _submission_gate_for_policy(submit_policy)
        if requested_submission_gate is not None:
            submission_gate = _normalized_submission_gate(requested_submission_gate, default=default_gate)
        else:
            submission_gate = default_gate
        if submission_gate == "always" and requested_submission_gate is None and submit_policy == "always":
            submission_gate = _DEFAULT_LIMITED_SUBMISSION_GATE
            submit_policy = "readiness_or_final"
            print(
                "[yellow]note[/yellow]: submission limit detected in rules; "
                f"defaulting submission_gate={submission_gate}."
            )
    else:
        submit_policy = "always"
        submission_gate = "always"
        if _normalized_submit_policy(requested_submit_policy) != "always":
            print(
                "[yellow]note[/yellow]: no submission limit detected; "
                f"ignoring submit_policy='{requested_submit_policy}'."
            )
        normalized_requested_gate = (
            _normalized_submission_gate(requested_submission_gate, default="always")
            if requested_submission_gate
            else "always"
        )
        if requested_submission_gate and normalized_requested_gate != "always":
            print(
                "[yellow]note[/yellow]: no submission limit detected; "
                f"ignoring submission_gate='{requested_submission_gate}'."
            )
    readiness_target_score = choose(
        None,
        plan.readiness_target_score,
        spec_readiness_target if spec_readiness_target is not None else target_score,
    )
    readiness_method = choose(None, plan.readiness_method, spec_readiness_method or "ci_bound")
    readiness_k = choose(None, plan.readiness_k, spec_readiness_k if spec_readiness_k is not None else 1.0)
    ci_method = choose(None, plan.ci_method, spec_ci_method or "normal")
    ci_alpha = choose(None, plan.ci_alpha, spec_ci_alpha if spec_ci_alpha is not None else 0.05)
    drift_check = bool(
        choose(
            None,
            plan.drift_check,
            spec_drift_enabled if spec_drift_enabled is not None else False,
        )
    )
    drift_weight = choose(None, plan.drift_weight, spec_drift_weight if spec_drift_weight is not None else 1.0)
    stop_min_delta = choose(
        None,
        plan.stop_min_delta,
        spec_stop_min_delta if spec_stop_min_delta is not None else min_improvement,
    )
    stop_no_improve_patience = choose(
        None,
        plan.stop_no_improve_patience,
        spec_stop_no_improve if spec_stop_no_improve is not None else patience,
    )
    stop_same_config_patience = choose(
        None,
        plan.stop_same_config_patience,
        spec_stop_same_config if spec_stop_same_config is not None else 0,
    )
    rank_force_major_max_percentile = _normalize_rank_force_percentile(
        plan.rank_force_major_max_percentile,
        fallback=_DEFAULT_FORCE_MAJOR_RANK_MAX_PERCENTILE,
    )
    rank_force_major_min_teams = _normalize_rank_force_min_teams(
        plan.rank_force_major_min_teams,
        fallback=_DEFAULT_FORCE_MAJOR_RANK_MIN_TEAMS,
    )

    return {
        "target_metric": target_metric,
        "target_score": target_score,
        "target_direction": target_direction,
        "score_source": score_source,
        "holdout_frac": holdout_frac,
        "cv_folds": cv_folds,
        "split_strategy": split_strategy,
        "seed": seed,
        "eval_seeds": eval_seeds,
        "eval_repeats": eval_repeats,
        "time_budget_min": time_budget_min,
        "kernel_name": kernel_name,
        "internet": internet,
        "max_iterations": max_iterations,
        "max_total_min": max_total_min,
        "patience": patience,
        "min_improvement": min_improvement,
        "submit_policy": submit_policy,
        "submission_gate": submission_gate,
        "submission_limit_per_day": constraints.submission_limit_per_day,
        "readiness_target_score": readiness_target_score,
        "readiness_method": readiness_method,
        "readiness_k": readiness_k,
        "ci_method": ci_method,
        "ci_alpha": ci_alpha,
        "drift_check": drift_check,
        "drift_weight": drift_weight,
        "stop_min_delta": stop_min_delta,
        "stop_no_improve_patience": stop_no_improve_patience,
        "stop_same_config_patience": stop_same_config_patience,
        "rank_force_major_max_percentile": rank_force_major_max_percentile,
        "rank_force_major_min_teams": rank_force_major_min_teams,
    }


def _resolved_plan(resolved: dict[str, object]) -> PlanConfig:
    return PlanConfig(
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
        max_iterations=int(resolved.get("max_iterations") or 3),
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
        rank_force_major_max_percentile=_normalize_rank_force_percentile(
            resolved.get("rank_force_major_max_percentile"),
            fallback=_DEFAULT_FORCE_MAJOR_RANK_MAX_PERCENTILE,
        ),
        rank_force_major_min_teams=_normalize_rank_force_min_teams(
            resolved.get("rank_force_major_min_teams"),
            fallback=_DEFAULT_FORCE_MAJOR_RANK_MIN_TEAMS,
        ),
    )


def _build_run_payload(
    *,
    run_id: str,
    config: AutopilotConfig,
    resolved: dict[str, object],
    status: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "slug": config.slug,
        "started_at": datetime.now(UTC).isoformat(),
        "status": status,
        "config": {
            "agent": config.agent,
            "compute": config.compute,
            "accelerator": config.accelerator,
            "kaggle_username": config.kaggle_username,
            "kernel_name": resolved.get("kernel_name"),
            "internet": resolved.get("internet"),
            "score_source": resolved.get("score_source"),
            "holdout_frac": resolved.get("holdout_frac"),
            "cv_folds": resolved.get("cv_folds"),
            "split_strategy": resolved.get("split_strategy"),
            "target_metric": resolved.get("target_metric"),
            "target_score": resolved.get("target_score"),
            "target_direction": resolved.get("target_direction"),
            "max_iterations": resolved.get("max_iterations"),
            "max_total_min": resolved.get("max_total_min"),
            "patience": resolved.get("patience"),
            "min_improvement": resolved.get("min_improvement"),
            "time_budget_min": resolved.get("time_budget_min"),
            "seed": resolved.get("seed"),
            "eval_seeds": resolved.get("eval_seeds"),
            "eval_repeats": resolved.get("eval_repeats"),
            "submit_policy": resolved.get("submit_policy"),
            "submission_gate": resolved.get("submission_gate"),
            "submission_limit_per_day": resolved.get("submission_limit_per_day"),
            "readiness_target_score": resolved.get("readiness_target_score"),
            "readiness_method": resolved.get("readiness_method"),
            "readiness_k": resolved.get("readiness_k"),
            "ci_method": resolved.get("ci_method"),
            "ci_alpha": resolved.get("ci_alpha"),
            "drift_check": resolved.get("drift_check"),
            "drift_weight": resolved.get("drift_weight"),
            "stop_min_delta": resolved.get("stop_min_delta"),
            "stop_no_improve_patience": resolved.get("stop_no_improve_patience"),
            "stop_same_config_patience": resolved.get("stop_same_config_patience"),
            "rank_force_major_max_percentile": resolved.get("rank_force_major_max_percentile"),
            "rank_force_major_min_teams": resolved.get("rank_force_major_min_teams"),
            "submit": config.submit,
            "message": config.message,
        },
    }


def _load_dataset_profile(paths: CompetitionPaths) -> dict[str, object]:
    if not paths.dataset_profile_path.exists():
        return {}
    try:
        return json.loads(paths.dataset_profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_evaluation_spec(paths: CompetitionPaths) -> dict[str, object]:
    spec_path = paths.context_dir / "evaluation_spec.json"
    if not spec_path.exists():
        return {}
    try:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    spec, issues = validate_evaluation_spec(payload)
    if issues:
        issue_text = "; ".join(issues)
        print(f"[yellow]evaluation spec ignored[/yellow]: {issue_text}")
        return {}
    return spec or {}


def _normalize_eval_seeds(value: object, *, fallback: list[int] | None = None) -> list[int]:
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
    return list(_DEFAULT_EVAL_SEEDS)


def _normalize_eval_repeats(value: object, *, fallback: int | None = None) -> int:
    resolved = value if isinstance(value, int) else fallback
    if isinstance(resolved, int):
        return max(1, min(resolved, 10))
    return _DEFAULT_EVAL_REPEATS


def _normalize_rank_force_percentile(value: object, *, fallback: float) -> float:
    if isinstance(value, (int, float)):
        parsed = float(value)
        if 0.0 < parsed <= 1.0:
            return parsed
    return float(fallback)


def _normalize_rank_force_min_teams(value: object, *, fallback: int) -> int:
    if isinstance(value, int):
        return max(1, value)
    if isinstance(value, float):
        return max(1, int(value))
    return max(1, int(fallback))


def _expanded_eval_seeds(*, base_seeds: list[int], repeats: int) -> list[int]:
    seeds = _normalize_eval_seeds(base_seeds)
    repeats_norm = _normalize_eval_repeats(repeats)
    expanded: list[int] = []
    for repeat_idx in range(repeats_norm):
        offset = repeat_idx * _EVAL_REPEAT_SEED_OFFSET
        for seed in seeds:
            expanded.append(int(seed + offset))
    return expanded


def _refresh_knowledge_hints(config: AutopilotConfig) -> None:
    profile = _load_dataset_profile(config.paths)
    raw_tags = profile.get("tags", []) if isinstance(profile, dict) else []
    tags = [str(tag).strip() for tag in raw_tags if isinstance(tag, str) and str(tag).strip()]

    lines = ["# Knowledge Hints", ""]
    try:
        if not tags:
            lines.append("No dataset tags available yet; knowledge suggestions pending dataset profiling.")
        else:
            taxonomy = ensure_taxonomy(config.knowledge_paths)
            similar = resolve_similar_improvements(
                knowledge_paths=config.knowledge_paths,
                taxonomy=taxonomy,
                tags=tags,
            )
            if not similar:
                lines.append("No similar competitions found in knowledge base.")
            else:
                lines.append("Similar competitions and what improved score:")
                lines.append("")
                for item in similar:
                    slug = item.get("slug", "unknown")
                    overlap = item.get("overlap", 0)
                    summary = item.get("summary", "No summary recorded.")
                    lines.append(f"- {slug} ({overlap} tag overlap): {summary}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"Knowledge lookup failed: {exc}")

    config.paths.context_dir.mkdir(parents=True, exist_ok=True)
    config.paths.knowledge_hints_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_problem_type_knowledge_text(config: AutopilotConfig, *, limit: int = 5) -> str:
    profile = _load_dataset_profile(config.paths)
    problem_types = derive_problem_types(profile)
    try:
        insights = resolve_problem_type_insights(config.knowledge_paths, problem_types, limit=limit)
        error_insights = resolve_error_fix_insights(config.knowledge_paths, problem_types, limit=limit)
    except Exception as exc:  # noqa: BLE001
        return f"Problem-type knowledge unavailable: {exc}"
    sections = [
        format_problem_type_insights(insights, limit=limit),
        "",
        format_error_fix_insights(error_insights, limit=limit),
    ]
    return "\n".join(section for section in sections if section is not None)


def _run_verify(verify_cmd: str, *, dry_run: bool) -> None:
    if dry_run:
        return
    args = shlex.split(verify_cmd)

    def _is_pytest_invocation(cmd_args: list[str]) -> bool:
        for idx, item in enumerate(cmd_args):
            if item == "pytest" or item.endswith("/pytest"):
                return True
            if item == "-m" and idx + 1 < len(cmd_args) and cmd_args[idx + 1] == "pytest":
                return True
        return False

    env = None
    if _is_pytest_invocation(args):
        # Avoid crashes from unrelated third-party pytest plugins present in the environment
        # (e.g. system/site packages) by disabling auto-loading during verification.
        env = os.environ.copy()
        env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

    result = run_command(args, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Verification failed: {result.output}")


def _run_plan_and_initial(config: AutopilotConfig, run_id: str) -> None:
    print("[cyan]plan[/cyan]: codex(5.3) -> gpt(5.2) -> codex(5.3)")
    pipeline_config = AgentPipelineConfig(
        slug=config.slug,
        competition_url=config.competition_url,
        compute=config.compute,
        accelerator=config.accelerator,
        internet=str(config.internet or "auto"),
        run_id=run_id,
        dry_run=config.dry_run,
        repo_root=config.paths.repo_root,
    )
    run_agent_pipeline(paths=config.paths, config=pipeline_config)
    _run_verify(config.verify_cmd, dry_run=config.dry_run)


def _print_top1_info(top1_info: dict[str, object]) -> None:
    score = top1_info.get("score") if isinstance(top1_info, dict) else None
    source = top1_info.get("source") if isinstance(top1_info, dict) else None
    if score is None:
        print("[yellow]top1 public score[/yellow]: unavailable")
        return
    suffix = f" (source: {source})" if source else ""
    print(f"[cyan]top1 public score[/cyan]: {score}{suffix}")


def _print_agent_prompt(prompt_path: Path, prompt_text: str) -> None:
    print(f"[cyan]codex prompt[/cyan]: {prompt_path}")
    builtins.print(prompt_text.rstrip())
    builtins.print("")


def _read_agent_response(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").rstrip()


def _print_agent_response(response_path: Path, response_text: str) -> None:
    print(f"[cyan]codex response[/cyan]: {response_path}")
    builtins.print(response_text)
    builtins.print("")


def _tail_for_prompt(text: str, *, max_chars: int = 6000) -> str:
    normalized = (text or "").replace("\r", "\n").strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[-max_chars:]


def _append_fix_retry_feedback(
    *,
    base_prompt: str,
    stage_label: str,
    codex_pass: int,
    failure_text: str,
) -> str:
    clipped = _tail_for_prompt(failure_text, max_chars=6000)
    if not clipped:
        return base_prompt
    return (
        f"{base_prompt}\n\n"
        f"## Retry Feedback (pass {codex_pass})\n\n"
        f"The previous {stage_label} pass did not fully resolve the issue.\n"
        "Apply additional minimal edits focused on the remaining failure below.\n\n"
        "```\n"
        f"{clipped}\n"
        "```\n"
    )


def _load_json_object(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _evaluation_from_kernel_metrics_payload(
    payload: dict[str, object],
    *,
    direction: str,
    target_metric: str | None,
) -> EvaluationResult | None:
    """Build an evaluation result from kernel metrics payload with trust-aware source fallback."""
    from kagglebot.solver.evaluate import EvaluationResult

    metric_name, value = _extract_kernel_metric(payload, target_metric)
    if value is None:
        return None
    payload_direction_raw = payload.get("direction")
    if payload_direction_raw is None:
        payload_direction_raw = payload.get("target_direction")
    payload_direction = str(payload_direction_raw).strip().lower() if payload_direction_raw is not None else ""
    resolved_direction = direction
    if payload_direction in {"minimize", "maximize"}:
        resolved_direction = payload_direction

    std = payload.get("offline_std")
    if std is None:
        std = payload.get("std")
    if std is None:
        std = payload.get("selected_cv_std")
    std_value = float(std) if isinstance(std, (int, float)) else None

    fold_scores_raw = payload.get("fold_scores")
    fold_scores: list[float] | None = None
    if isinstance(fold_scores_raw, list):
        parsed_fold_scores = [float(item) for item in fold_scores_raw if isinstance(item, (int, float))]
        if parsed_fold_scores:
            fold_scores = parsed_fold_scores
            if std_value is None and len(parsed_fold_scores) > 1:
                std_value = float(np.std(parsed_fold_scores, ddof=1))

    score_source = _normalize_score_source_name(payload.get("score_source", "holdout"))
    if score_source == "holdout":
        for key in payload.keys():
            if isinstance(key, str) and key.lower().startswith("oof_"):
                score_source = "cv"
                break
    trusted_fallback_value = None
    if not _is_trusted_offline_score_source(score_source):
        trusted_fallback_value = _extract_trusted_cv_value_from_metrics_payload(payload)
        if trusted_fallback_value is not None:
            value = trusted_fallback_value
            score_source = "cv"

    return EvaluationResult(
        score_source=score_source,
        metric=metric_name or target_metric or "unknown",
        direction=resolved_direction,  # type: ignore[arg-type]
        value=float(value),
        std=std_value,
        train_score=None,
        val_score=None,
        fold_scores=fold_scores,
    )


def _load_kernel_metrics(metrics_path: Path, direction: str, target_metric: str | None) -> EvaluationResult | None:
    """Load kernel metrics from disk into a normalized evaluation result."""
    payload = _load_json_object(metrics_path)
    if payload is None:
        return None
    return _evaluation_from_kernel_metrics_payload(
        payload,
        direction=direction,
        target_metric=target_metric,
    )


def _normalize_score_source_name(value: object) -> str:
    """Normalize score_source labels for trust checks."""
    text = str(value or "").strip().lower()
    if not text:
        return "holdout"
    normalized = text.replace("-", "_").replace(" ", "_")
    alias_map = {
        "cross_validation": "cv",
        "crossval": "cv",
        "validation": "holdout",
        "lbproxy": "lb_proxy",
    }
    return alias_map.get(normalized, normalized)


def _is_trusted_offline_score_source(score_source: str) -> bool:
    """Return whether score source is trusted for offline model-selection decisions."""
    return _normalize_score_source_name(score_source) in _TRUSTED_SCORE_SOURCES


def _extract_trusted_cv_value_from_metrics_payload(payload: dict[str, object]) -> float | None:
    """Extract a CV-based fallback score from metrics payload when reported source is untrusted."""
    for key in (
        "cv_brier",
        "cv_score",
        "cv_mean",
        "selected_cv_mean",
        "best_cv",
        "oof_score",
        "oof_metric",
        "oof_brier",
    ):
        parsed = _to_float(payload.get(key))
        if parsed is not None:
            return float(parsed)

    fold_scores_raw = payload.get("fold_scores")
    if isinstance(fold_scores_raw, list):
        fold_scores = [float(item) for item in fold_scores_raw if isinstance(item, (int, float))]
        if fold_scores:
            return float(np.mean(fold_scores))
    return None


def _extract_kernel_metric(payload: dict[str, object], target_metric: str | None) -> tuple[str | None, float | None]:
    def as_number(value: object) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return _to_float(value)

    def normalize(text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalnum())

    def metric_hint() -> str | None:
        metric_raw = payload.get("metric")
        if isinstance(metric_raw, str) and metric_raw.strip():
            return metric_raw.strip()
        primary_raw = payload.get("primary_metric")
        if isinstance(primary_raw, str) and primary_raw.strip():
            return primary_raw.strip()
        return target_metric.strip() if isinstance(target_metric, str) and target_metric.strip() else None

    def pick_selected_metric(selected: dict[str, object]) -> tuple[str | None, float | None]:
        hint = metric_hint()
        hint_norm = normalize(hint) if hint else ""
        if "map" in hint_norm and "f1" in hint_norm:
            value = as_number(selected.get("combined_score"))
            if value is not None:
                return (hint, value)
        if "map" in hint_norm:
            value = as_number(selected.get("mean_map"))
            if value is not None:
                return (hint or "mean_map", value)
        if "f1" in hint_norm:
            value = as_number(selected.get("oof_f1"))
            if value is not None:
                return (hint or "f1", value)
        for key, name in (
            ("offline_value", hint),
            ("value", hint),
            ("score", hint),
            ("cv_mean", hint),
            ("combined_score", hint or "combined_score"),
            ("mean_map", hint or "mean_map"),
            ("oof_f1", hint or "f1"),
        ):
            value = as_number(selected.get(key))
            if value is not None:
                return (name, value)
        return (None, None)

    def strip_prefixes(text: str) -> str:
        lowered = text.lower()
        for prefix in (
            "val_",
            "train_",
            "test_",
            "oof_",
            "cv_",
            "holdout_",
            "offline_",
            "online_",
            "public_",
            "private_",
        ):
            if lowered.startswith(prefix):
                return text[len(prefix) :]
        return text

    def prefers_lower(metric: str) -> bool:
        return normalize(metric) in {"rmse", "rmsle", "mae", "mape", "logloss", "loss"}

    def pick_from_dict(metric_key: str, values: dict[str, object]) -> float | None:
        selection = payload.get("selection")
        if isinstance(selection, str) and selection in values:
            selected_value = as_number(values.get(selection))
            if selected_value is not None:
                return selected_value
        for key in ("selected", "average", "stacked", "best", "val", "oof", "score"):
            value = as_number(values.get(key))
            if value is not None:
                return value
        numeric = [parsed for raw in values.values() if (parsed := as_number(raw)) is not None]
        if not numeric:
            return None
        return min(numeric) if prefers_lower(metric_key) else max(numeric)

    selected_raw = payload.get("selected")
    if isinstance(selected_raw, dict):
        metric, value = pick_selected_metric(selected_raw)
        if value is not None:
            return (metric, value)

    selected_cv_mean = as_number(payload.get("selected_cv_mean"))
    if selected_cv_mean is not None:
        return (metric_hint(), selected_cv_mean)

    offline_value = as_number(payload.get("offline_value"))
    if offline_value is not None:
        return (str(payload.get("metric") or target_metric or "unknown"), offline_value)
    payload_value = as_number(payload.get("value"))
    if payload_value is not None:
        return (str(payload.get("metric") or target_metric or "unknown"), payload_value)
    score_value = as_number(payload.get("score"))
    if score_value is not None:
        return (str(payload.get("metric") or target_metric or "unknown"), score_value)

    hinted_metric = metric_hint()
    hinted_key = normalize(hinted_metric) if hinted_metric else ""
    if hinted_key:
        for key, val in payload.items():
            parsed = as_number(val)
            if parsed is None:
                continue
            normalized_key = normalize(str(key))
            normalized_base = normalize(strip_prefixes(str(key)))
            if normalized_key == hinted_key or normalized_base == hinted_key:
                return (hinted_metric, parsed)

    leaderboard_raw = payload.get("leaderboard")
    if isinstance(leaderboard_raw, list):
        selected_pipeline = payload.get("selected_pipeline")
        selected_name = selected_pipeline.strip() if isinstance(selected_pipeline, str) else ""
        best_value: float | None = None
        best_metric = metric_hint()
        for item in leaderboard_raw:
            if not isinstance(item, dict):
                continue
            if selected_name:
                pipeline = item.get("pipeline")
                if isinstance(pipeline, str) and pipeline.strip() != selected_name:
                    continue
            cv_mean = as_number(item.get("cv_mean"))
            if cv_mean is None:
                continue
            if best_value is None:
                best_value = cv_mean
            elif best_metric and prefers_lower(best_metric):
                best_value = min(best_value, cv_mean)
            else:
                best_value = max(best_value, cv_mean)
        if best_value is not None:
            return (best_metric, best_value)

    pipelines_raw = payload.get("pipelines")
    if isinstance(pipelines_raw, list):
        selected_name = None
        if isinstance(selected_raw, dict):
            maybe_name = selected_raw.get("name")
            if isinstance(maybe_name, str) and maybe_name.strip():
                selected_name = maybe_name.strip()

        best_value: float | None = None
        best_metric: str | None = None
        for item in pipelines_raw:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if selected_name and isinstance(name, str) and name.strip() != selected_name:
                continue
            metric, value = pick_selected_metric(item)
            if value is None:
                continue
            if best_value is None:
                best_value = value
                best_metric = metric
            elif best_metric and prefers_lower(best_metric):
                best_value = min(best_value, value)
            else:
                best_value = max(best_value, value)
        if best_value is not None:
            return (best_metric, best_value)

    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        metric_key = None
        if isinstance(key, str) and key.lower().startswith("oof_"):
            metric_key = key[4:]
        elif isinstance(key, str):
            metric_key = key
        if metric_key is None:
            continue
        if target_metric is not None and normalize(metric_key) != normalize(target_metric):
            continue
        picked = pick_from_dict(metric_key, value)
        if picked is not None:
            return (metric_key, picked)

    aliases: dict[str, tuple[str, ...]] = {
        "accuracy": ("accuracy", "acc"),
        "auc": ("auc", "rocauc", "roc_auc"),
        "brier_score": ("brier", "brier_score", "brierscore"),
        "fmax": ("fmax", "proxyfmax"),
        "f1": ("f1", "f1score"),
        "logloss": ("logloss", "log_loss"),
        "mae": ("mae",),
        "mape": ("mape",),
        "rmse": ("rmse",),
        "rmsle": ("rmsle",),
        "r2": ("r2", "r2score"),
    }

    target_key = normalize(target_metric) if target_metric else ""
    if target_key:
        wanted = set()
        for key, values in aliases.items():
            if target_key == normalize(key) or target_key in {normalize(v) for v in values}:
                wanted.update({normalize(v) for v in values})
                wanted.add(normalize(key))
                break
        if wanted:
            for key, val in payload.items():
                parsed = as_number(val)
                if parsed is None:
                    continue
                normalized_key = normalize(str(key))
                normalized_base = normalize(strip_prefixes(str(key)))
                if normalized_key in wanted or normalized_base in wanted:
                    return (str(target_metric), parsed)

    for key, val in payload.items():
        parsed = as_number(val)
        if parsed is None:
            continue
        normalized_key = normalize(str(key))
        normalized_base = normalize(strip_prefixes(str(key)))
        for metric_name, values in aliases.items():
            normalized_aliases = {normalize(v) for v in values}
            normalized_aliases.add(normalize(metric_name))
            if normalized_key in normalized_aliases or normalized_base in normalized_aliases:
                return (metric_name, parsed)

    return (str(target_metric) if target_metric else None, None)


def _metric_value_from_payload_item(item: dict[str, object]) -> float | None:
    for key in (
        "offline_value",
        "selected_cv_mean",
        "cv_mean",
        "score",
        "value",
        "combined_score",
        "mean_map",
        "oof_f1",
    ):
        value = _to_float(item.get(key))
        if value is not None:
            return value
    return None


def _extract_baseline_candidates_from_metrics_payload(payload: dict[str, object]) -> list[tuple[str, float]]:
    candidates: list[tuple[str, float]] = []

    pipelines_raw = payload.get("pipelines")
    if isinstance(pipelines_raw, list):
        for item in pipelines_raw:
            if not isinstance(item, dict):
                continue
            name_raw = item.get("name") or item.get("pipeline")
            name = str(name_raw).strip() if isinstance(name_raw, str) else ""
            lowered = name.lower()
            if "baseline" not in lowered and "persistence" not in lowered:
                continue
            score = _metric_value_from_payload_item(item)
            if score is not None:
                candidates.append((f"pipelines:{name or 'unnamed'}", float(score)))

    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        lowered = key.lower()
        if "baseline" not in lowered and "persistence" not in lowered:
            continue
        if isinstance(value, dict):
            score = _metric_value_from_payload_item(value)
            if score is not None:
                candidates.append((f"metrics:{key}", float(score)))
                continue
            for nested_key, nested_value in value.items():
                nested_score = _to_float(nested_value)
                if nested_score is None:
                    continue
                candidates.append((f"metrics:{key}.{nested_key}", float(nested_score)))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    score = _metric_value_from_payload_item(item)
                    if score is not None:
                        candidates.append((f"metrics:{key}[{index}]", float(score)))
                        continue
                parsed = _to_float(item)
                if parsed is not None:
                    candidates.append((f"metrics:{key}[{index}]", float(parsed)))
        else:
            parsed = _to_float(value)
            if parsed is not None:
                candidates.append((f"metrics:{key}", float(parsed)))

    return candidates


def _collect_kernel_log_text(logs_dir: Path | None) -> str:
    if logs_dir is None or not logs_dir.exists():
        return ""
    texts: list[str] = []
    for path in sorted(logs_dir.glob("*.log")):
        name = path.name.lower()
        if "stdout" not in name and "kernel" not in name:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not text:
            continue
        texts.append(text[-250_000:])
    if not texts:
        return ""
    return "\n".join(texts)


def _extract_validation_scores_from_log_text(log_text: str, metric_name: str | None) -> list[float]:
    if not log_text:
        return []
    pattern = re.compile(
        r"val_([^=\n]{1,80})\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        flags=re.IGNORECASE,
    )
    target_norm = _normalize_metric_name(metric_name)
    scores: list[float] = []
    for match in pattern.finditer(log_text):
        metric_label = match.group(1).strip()
        parsed = _to_float(match.group(2))
        if parsed is None:
            continue
        if target_norm:
            label_norm = _normalize_metric_name(metric_label)
            if label_norm and (target_norm not in label_norm and label_norm not in target_norm):
                continue
        scores.append(float(parsed))
    return scores


def _extract_baseline_scores_from_log_text(log_text: str) -> list[float]:
    if not log_text:
        return []
    pattern = re.compile(
        r"baseline[^=\n]{0,120}=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        flags=re.IGNORECASE,
    )
    scores: list[float] = []
    for match in pattern.finditer(log_text):
        parsed = _to_float(match.group(1))
        if parsed is not None:
            scores.append(float(parsed))
    return scores


def _is_significantly_worse(
    *,
    current: float,
    reference: float,
    direction: str,
    rel_margin: float,
    abs_margin: float,
) -> bool:
    margin = max(abs(reference) * max(rel_margin, 0.0), max(abs_margin, 0.0))
    if direction == "minimize":
        return (current - reference) > margin
    return (reference - current) > margin


def _build_kernel_quality_guard(
    *,
    evaluation: EvaluationResult,
    kernel_metrics_payload: dict[str, object] | None,
    logs_dir: Path | None,
    direction: str,
    iteration: int,
    max_iterations: int,
    force_submit: bool,
    code_reference_score: float | None = None,
    code_reference_source: str | None = None,
    metric_mismatch_detected: bool = False,
    metric_mismatch_reason: str | None = None,
) -> dict[str, object]:
    """Build submit guard signals that reject unstable or non-generalizable evaluations."""
    reasons: list[str] = []
    warnings: list[str] = []
    block_submit = False
    is_final_iteration = iteration >= max_iterations
    payload = kernel_metrics_payload or {}

    normalized_score_source = _normalize_score_source_name(evaluation.score_source)
    if not _is_trusted_offline_score_source(normalized_score_source):
        reasons.append("untrusted_score_source")
        warnings.append(f"score_source={normalized_score_source}")
        if not force_submit:
            block_submit = True

    oracle_payload = payload.get("oracle")
    if isinstance(oracle_payload, dict):
        oracle_mode = str(oracle_payload.get("mode_setting") or "").strip().lower()
        oracle_applied = bool(oracle_payload.get("applied"))
        if oracle_applied or (oracle_mode and oracle_mode != "off"):
            reasons.append("oracle_override_detected")
            warnings.append(f"oracle_mode={oracle_mode or 'unknown'}")
            if not force_submit:
                block_submit = True

    baseline_candidates = _extract_baseline_candidates_from_metrics_payload(payload)
    log_text = _collect_kernel_log_text(logs_dir)
    baseline_from_logs = _extract_baseline_scores_from_log_text(log_text)
    for index, score in enumerate(baseline_from_logs):
        baseline_candidates.append((f"logs:baseline[{index}]", float(score)))

    best_baseline_source: str | None = None
    best_baseline_score: float | None = None
    if baseline_candidates:
        if direction == "minimize":
            best_baseline_source, best_baseline_score = min(baseline_candidates, key=lambda item: item[1])
        else:
            best_baseline_source, best_baseline_score = max(baseline_candidates, key=lambda item: item[1])

    baseline_worse_than_reference = False
    if best_baseline_score is not None:
        baseline_worse_than_reference = _is_significantly_worse(
            current=float(evaluation.value),
            reference=float(best_baseline_score),
            direction=direction,
            rel_margin=_QUALITY_GUARD_BASELINE_REL_MARGIN,
            abs_margin=_QUALITY_GUARD_BASELINE_ABS_MARGIN,
        )
        if baseline_worse_than_reference:
            reasons.append("selected_worse_than_detected_baseline")
            if not is_final_iteration and not force_submit:
                block_submit = True

    validation_scores = _extract_validation_scores_from_log_text(log_text, evaluation.metric)
    best_validation: float | None = None
    severe_validation_mismatch = False
    if validation_scores:
        best_validation = min(validation_scores) if direction == "minimize" else max(validation_scores)
        mismatch_rel = (
            _QUALITY_GUARD_MISMATCH_REL_MARGIN_MINIMIZE
            if direction == "minimize"
            else _QUALITY_GUARD_MISMATCH_REL_MARGIN_MAXIMIZE
        )
        severe_validation_mismatch = _is_significantly_worse(
            current=float(evaluation.value),
            reference=float(best_validation),
            direction=direction,
            rel_margin=mismatch_rel,
            abs_margin=_QUALITY_GUARD_MISMATCH_ABS_MARGIN,
        )
        if severe_validation_mismatch:
            reasons.append("validation_metric_mismatch_vs_final_metric")
            if not is_final_iteration and not force_submit:
                block_submit = True

    step_bucket_payload = payload.get("cv_step_buckets")
    step_bucket_scores: list[float] = []
    if isinstance(step_bucket_payload, dict):
        for value in step_bucket_payload.values():
            parsed = _to_float(value)
            if parsed is not None:
                step_bucket_scores.append(float(parsed))
    step_bucket_collapse = False
    if len(step_bucket_scores) >= 4:
        median_bucket = float(np.median(step_bucket_scores))
        worst_bucket = float(max(step_bucket_scores))
        collapse_threshold = max(median_bucket * _QUALITY_GUARD_STEP_BUCKET_RATIO, median_bucket + 0.5)
        step_bucket_collapse = worst_bucket > collapse_threshold
        if step_bucket_collapse:
            warnings.append("cv_step_bucket_collapse_detected")
            if severe_validation_mismatch:
                reasons.append("severe_step_bucket_instability")
                if not is_final_iteration and not force_submit:
                    block_submit = True

    if metric_mismatch_detected:
        reasons.append("competition_metric_mismatch")
        if metric_mismatch_reason:
            warnings.append(f"metric_mismatch_detail={metric_mismatch_reason}")
        if not force_submit:
            block_submit = True

    below_code_reference = False
    code_delta: float | None = None
    if code_reference_score is not None:
        code_delta = _score_delta_vs_reference(float(evaluation.value), float(code_reference_score), direction)
        below_code_reference = _is_significantly_worse(
            current=float(evaluation.value),
            reference=float(code_reference_score),
            direction=direction,
            rel_margin=_QUALITY_GUARD_CODE_REF_REL_MARGIN,
            abs_margin=_QUALITY_GUARD_CODE_REF_ABS_MARGIN,
        )
        if below_code_reference:
            reasons.append("below_code_reference_baseline")
            warnings.append(
                "code_reference_score="
                f"{float(code_reference_score):.6f},current={float(evaluation.value):.6f},"
                f"delta={code_delta:+.6f},source={code_reference_source or 'unknown'}"
            )
            if not force_submit:
                block_submit = True

    allow_submit = not block_submit
    return {
        "allow_submit": allow_submit,
        "block_submit": block_submit,
        "is_final_iteration": is_final_iteration,
        "reasons": reasons,
        "warnings": warnings,
        "baseline": {
            "best_source": best_baseline_source,
            "best_score": best_baseline_score,
            "candidate_count": len(baseline_candidates),
            "selected_worse_than_baseline": baseline_worse_than_reference,
        },
        "metric_alignment": {
            "best_validation_score": best_validation,
            "validation_score_count": len(validation_scores),
            "severe_mismatch": severe_validation_mismatch,
        },
        "step_bucket": {
            "count": len(step_bucket_scores),
            "collapse_detected": step_bucket_collapse,
        },
        "code_reference": {
            "score": code_reference_score,
            "source": code_reference_source,
            "delta_vs_current": code_delta,
            "below_reference": below_code_reference,
            "abs_margin": _QUALITY_GUARD_CODE_REF_ABS_MARGIN,
            "rel_margin": _QUALITY_GUARD_CODE_REF_REL_MARGIN,
        },
    }


def _build_metrics_payload(
    *,
    run_id: str,
    iteration: int,
    evaluation: EvaluationResult,
    target_score: float,
    met_target: bool,
    top1_info: dict[str, object],
    compute: str,
    accelerator: str,
    holdout_frac: float,
    cv_folds: int,
    seed: int,
    evaluation_by_source: dict[str, EvaluationResult] | None = None,
    evaluation_report: EvaluationReport | None = None,
    readiness_target: float | None = None,
) -> dict[str, object]:
    payload = {
        "run_id": run_id,
        "iter": iteration,
        "metric": evaluation.metric,
        "direction": evaluation.direction,
        "score_source": evaluation.score_source,
        "offline_value": evaluation.value,
        "offline_std": evaluation.std,
        "target_score": target_score,
        "met_target": met_target,
        "top1_public_score": top1_info.get("score"),
        "top1_public_timestamp": top1_info.get("timestamp"),
        "compute": compute,
        "accelerator": accelerator,
        "timestamp": int(datetime.now(UTC).timestamp()),
        "folds": cv_folds if evaluation.score_source in {"cv", "consensus"} else None,
        "holdout_frac": holdout_frac if evaluation.score_source in {"holdout", "consensus"} else None,
        "seed": seed,
    }
    if evaluation_by_source:
        payload["offline_by_source"] = {
            source: _evaluation_to_payload(result) for source, result in evaluation_by_source.items()
        }
    if evaluation_report is not None:
        payload["readiness"] = {
            "score": evaluation_report.readiness_score,
            "mean": evaluation_report.mean,
            "std": evaluation_report.std,
            "ci_low": evaluation_report.ci_low,
            "ci_high": evaluation_report.ci_high,
            "target": readiness_target,
            "split_strategy": evaluation_report.split_strategy,
            "n_splits": evaluation_report.n_splits,
            "seeds": evaluation_report.seeds,
            "repeats": evaluation_report.repeats,
            "drift_auc": evaluation_report.drift_auc,
        }
    return payload


def _build_iteration_evaluation_report(
    *,
    config: AutopilotConfig,
    run_id: str,
    iteration: int,
    evaluation: EvaluationResult,
    evaluation_by_source: dict[str, EvaluationResult],
    metric_direction: str,
    cv_folds: int,
    split_strategy: str | None,
    seed: int,
    eval_seeds: list[int],
    eval_repeats: int,
    score_source: str,
    ci_method: str,
    ci_alpha: float,
    readiness_method: str,
    readiness_k: float,
    drift_check_enabled: bool,
    drift_weight: float,
    eval_data_cache: dict[str, object] | None,
) -> tuple[EvaluationReport, dict[str, object], dict[str, object] | None]:
    cache = _ensure_eval_data_cache(
        config=config,
        cv_folds=cv_folds,
        split_strategy=split_strategy,
        seed=seed,
        eval_seeds=eval_seeds,
        eval_repeats=eval_repeats,
        score_source=score_source,
        eval_data_cache=eval_data_cache,
    )
    fold_scores = _extract_fold_scores_for_report(evaluation=evaluation, evaluation_by_source=evaluation_by_source)

    ci_method_value = "bootstrap" if str(ci_method).lower() == "bootstrap" else "normal"
    uncertainty = UncertaintyEstimator.estimate(
        fold_scores,
        method=ci_method_value,  # type: ignore[arg-type]
        alpha=max(1e-6, min(float(ci_alpha), 0.5)),
        random_state=eval_seeds[0],
    )

    drift_auc = DriftChecker.adversarial_auc(
        cache.get("drift_train_x"),  # type: ignore[arg-type]
        cache.get("drift_test_x"),  # type: ignore[arg-type]
        enabled=bool(drift_check_enabled),
        random_state=eval_seeds[0],
        n_splits=max(2, min(cv_folds, 5)),
    )
    readiness_method_value = "mean_std" if str(readiness_method).lower() == "mean_std" else "ci_bound"
    readiness_score = SubmissionReadinessScorer.compute(
        direction=metric_direction,  # type: ignore[arg-type]
        mean_score=uncertainty.mean,
        std_score=uncertainty.std,
        ci_low=uncertainty.ci_low,
        ci_high=uncertainty.ci_high,
        method=readiness_method_value,  # type: ignore[arg-type]
        k=float(readiness_k),
        drift_auc=drift_auc,
        drift_enabled=bool(drift_check_enabled),
        drift_weight=float(drift_weight),
    )
    report = EvaluationReport(
        metric_name=evaluation.metric,
        direction=metric_direction,  # type: ignore[arg-type]
        split_strategy=str(cache.get("split_strategy") or "kfold"),  # type: ignore[arg-type]
        n_splits=int(cache.get("n_splits") or max(2, cv_folds)),
        seeds=list(eval_seeds),
        repeats=int(eval_repeats),
        per_fold_scores=[float(item) for item in fold_scores],
        mean=uncertainty.mean,
        std=uncertainty.std,
        ci_low=uncertainty.ci_low,
        ci_high=uncertainty.ci_high,
        drift_auc=drift_auc,
        readiness_score=readiness_score,
    )
    payload = report.to_dict()
    payload.update(
        {
            "run_id": run_id,
            "iteration": iteration,
            "metric_value": evaluation.value,
            "score_source": evaluation.score_source,
            "split_index_fingerprints": cache.get("split_index_fingerprints", []),
        }
    )
    return report, payload, cache


def _extract_fold_scores_for_report(
    *,
    evaluation: EvaluationResult,
    evaluation_by_source: dict[str, EvaluationResult],
) -> list[float]:
    cv_eval = evaluation_by_source.get("cv")
    if cv_eval is not None and cv_eval.fold_scores:
        return [float(value) for value in cv_eval.fold_scores]
    if evaluation.fold_scores:
        return [float(value) for value in evaluation.fold_scores]
    if evaluation_by_source:
        return [float(item.value) for item in evaluation_by_source.values()]
    return [float(evaluation.value)]


def _ensure_eval_data_cache(
    *,
    config: AutopilotConfig,
    cv_folds: int,
    split_strategy: str | None,
    seed: int,
    eval_seeds: list[int],
    eval_repeats: int,
    score_source: str,
    eval_data_cache: dict[str, object] | None,
) -> dict[str, object]:
    if eval_data_cache is not None:
        return eval_data_cache
    fallback = {
        "split_strategy": "kfold",
        "n_splits": max(2, int(cv_folds)),
        "split_index_fingerprints": [],
        "drift_train_x": None,
        "drift_test_x": None,
    }
    if score_source == "holdout":
        return fallback
    try:
        data = load_competition_data(config.paths.data_dir)
    except Exception:  # noqa: BLE001
        return fallback
    try:
        y = np.asarray(data.train[data.target_column])
        expanded_seeds = _expanded_eval_seeds(base_seeds=eval_seeds, repeats=eval_repeats)
        split = SplitStrategyFactory.create(y=y, strategy=split_strategy, n_splits=cv_folds, seed=seed)
        fingerprints: list[dict[str, object]] = []
        for expanded_seed in expanded_seeds:
            split_for_seed = SplitStrategyFactory.create(
                y=y,
                strategy=split_strategy,
                n_splits=cv_folds,
                seed=expanded_seed,
            )
            fingerprints.extend(_build_split_index_fingerprints(split=split_for_seed, y=y, seed=expanded_seed))
    except Exception:  # noqa: BLE001
        split = SplitStrategyFactory.create(y=[0, 1, 0, 1], strategy="kfold", n_splits=2, seed=seed)
        fingerprints = []
    drift_cols_raw = list(data.feature_columns or [])
    drift_cols: list[str] = []
    seen_drift_cols: set[str] = set()
    for col in drift_cols_raw:
        if col in seen_drift_cols:
            continue
        seen_drift_cols.add(col)
        drift_cols.append(col)
    try:
        common_cols = set(data.train.columns).intersection(set(data.test.columns))
        drift_cols = [col for col in drift_cols if col in common_cols]
        drift_train_x = data.train.reindex(columns=drift_cols).copy() if drift_cols else None
        drift_test_x = data.test.reindex(columns=drift_cols).copy() if drift_cols else None
    except Exception:  # noqa: BLE001
        drift_train_x = None
        drift_test_x = None
    return {
        "split_strategy": split.name,
        "n_splits": split.n_splits,
        "split_index_fingerprints": fingerprints,
        "drift_train_x": drift_train_x,
        "drift_test_x": drift_test_x,
    }


def _build_split_index_fingerprints(*, split: object, y: np.ndarray, seed: int) -> list[dict[str, object]]:
    split_strategy = split  # local alias for readability
    name = getattr(split_strategy, "name", "kfold")
    splitter = getattr(split_strategy, "splitter", None)
    if splitter is None:
        return []
    groups = None
    x_dummy = np.zeros((len(y), 1), dtype=np.float64)
    records: list[dict[str, object]] = []
    split_iter = _iter_split_indices(name=name, splitter=splitter, x=x_dummy, y=y, groups=groups)
    for fold_idx, (train_idx, valid_idx) in enumerate(split_iter):
        train_arr = np.asarray(train_idx, dtype=np.int64)
        valid_arr = np.asarray(valid_idx, dtype=np.int64)
        records.append(
            {
                "seed": seed,
                "fold": fold_idx,
                "train_size": int(train_arr.size),
                "valid_size": int(valid_arr.size),
                "train_hash": hashlib.sha256(train_arr.tobytes()).hexdigest()[:16],
                "valid_hash": hashlib.sha256(valid_arr.tobytes()).hexdigest()[:16],
            }
        )
    return records


def _iter_split_indices(*, name: str, splitter: object, x: np.ndarray, y: np.ndarray, groups: object):
    if name == "timeseries_split":
        yield from splitter.split(x)  # type: ignore[union-attr]
        return
    if name == "group_kfold" and groups is not None:
        yield from splitter.split(x, y, groups=groups)  # type: ignore[union-attr]
        return
    if name == "stratified_kfold":
        yield from splitter.split(x, y)  # type: ignore[union-attr]
        return
    yield from splitter.split(x, y)  # type: ignore[union-attr]


def _append_run_evaluation_report(*, run_dir: Path, iteration: int, payload: dict[str, object]) -> None:
    path = run_dir / "evaluation_report.json"
    state: dict[str, object] = {"latest_iteration": iteration, "latest": payload, "history": [payload]}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                history_raw = existing.get("history", [])
                if isinstance(history_raw, list):
                    history = [item for item in history_raw if isinstance(item, dict)]
                else:
                    history = []
                history = [item for item in history if item.get("iteration") != iteration]
                history.append(payload)
                history.sort(key=lambda item: int(item.get("iteration", 0)))
                state["history"] = history
        except json.JSONDecodeError:
            pass
    state["latest_iteration"] = iteration
    state["latest"] = payload
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _resume_best_readiness_score(*, run_dir: Path, direction: str, max_iterations: int) -> float | None:
    path = run_dir / "evaluation_report.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    history = payload.get("history")
    if not isinstance(history, list):
        latest = payload.get("latest")
        history = [latest] if isinstance(latest, dict) else []
    best: float | None = None
    for item in history:
        if not isinstance(item, dict):
            continue
        iteration = _to_int(item.get("iteration"))
        if iteration is not None and iteration > max_iterations:
            continue
        score = _to_float(item.get("readiness_score"))
        if score is None:
            continue
        if _update_best_score(best, score, direction, 0.0):
            best = score
    return best


def _resume_noise_guard_state(*, run_dir: Path, max_iterations: int) -> tuple[float | None, int]:
    path = run_dir / "evaluation_report.json"
    if not path.exists():
        return None, 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, 0
    if not isinstance(payload, dict):
        return None, 0
    history = payload.get("history")
    if not isinstance(history, list):
        latest = payload.get("latest")
        history = [latest] if isinstance(latest, dict) else []
    rows: list[dict[str, object]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        iteration = _to_int(item.get("iteration"))
        if iteration is None or iteration > max_iterations:
            continue
        rows.append(item)
    if not rows:
        return None, 0
    rows.sort(key=lambda item: int(_to_int(item.get("iteration")) or 0))
    streak = 0
    prev_score: float | None = None
    for item in rows:
        score = _to_float(item.get("readiness_score"))
        std = _to_float(item.get("std"))
        if score is None:
            continue
        if prev_score is not None and std is not None:
            threshold = 0.5 * max(std, 0.0)
            if abs(score - prev_score) < threshold:
                streak += 1
            else:
                streak = 0
        prev_score = score
    return prev_score, streak


def _non_final_submission_checkpoints(*, max_iterations: int, non_final_slots: int) -> set[int]:
    """Spread non-final submit slots across the loop to avoid early budget burn."""
    if max_iterations <= 1 or non_final_slots <= 0:
        return set()
    last_non_final = max_iterations - 1
    if non_final_slots >= last_non_final:
        return set(range(1, max_iterations))

    checkpoints: set[int] = set()
    for idx in range(1, non_final_slots + 1):
        # Integer spacing over [1, max_iterations-1], leaving room for final slot.
        candidate = (idx * max_iterations) // (non_final_slots + 1)
        candidate = max(1, min(last_non_final, candidate))
        checkpoints.add(candidate)

    if len(checkpoints) < non_final_slots:
        for candidate in range(last_non_final, 0, -1):
            checkpoints.add(candidate)
            if len(checkpoints) >= non_final_slots:
                break
    return checkpoints


def _should_attempt_submit_for_readiness(
    *,
    gate: str,
    readiness_score: float | None,
    readiness_target: float,
    direction: str,
    iteration: int,
    max_iterations: int,
    submission_limit_per_day: int | None = None,
    successful_submissions: int = 0,
    top1_score: float | None = None,
) -> bool:
    normalized = _normalized_submission_gate(gate, default="always")
    is_final_iteration = iteration >= max_iterations
    met_target = readiness_score is not None and _meets_target(readiness_score, readiness_target, direction)
    top1_tier = readiness_score is not None and _is_top1_tier(readiness_score, top1_score, direction)

    if normalized in {"final_only", "at_final"}:
        return is_final_iteration
    if normalized in {"readiness_only", "readiness_target", "on_target_only"}:
        return met_target

    if isinstance(submission_limit_per_day, int) and submission_limit_per_day > 0:
        if is_final_iteration:
            return True

        non_final_slots = max(0, submission_limit_per_day - 1)
        if non_final_slots <= 0:
            return False

        if successful_submissions >= non_final_slots:
            return top1_tier or met_target

        if max_iterations > submission_limit_per_day:
            checkpoints = _non_final_submission_checkpoints(
                max_iterations=max_iterations,
                non_final_slots=non_final_slots,
            )
            return (iteration in checkpoints) or top1_tier or met_target

        return True

    if normalized in {"always", "each_iteration"}:
        return True
    if normalized in {"readiness_or_final", "target_or_final"}:
        return met_target or is_final_iteration
    if readiness_score is None:
        return is_final_iteration
    return met_target or is_final_iteration


def _submission_gate_for_policy(policy: str | None) -> str:
    normalized = _normalized_submit_policy(policy)
    if normalized in {"always", "each_iteration"}:
        return "always"
    if normalized in {"final_only", "at_final"}:
        return "final_only"
    if normalized in {"readiness_only", "readiness_target", "on_target_only"}:
        return "readiness_only"
    if normalized in {"readiness_or_final", "target_or_final"}:
        return "readiness_or_final"
    return "always"


def _normalized_submit_policy(policy: str | None) -> str:
    normalized = str(policy or "").strip().lower()
    if normalized in {"always", "each_iteration"}:
        return "always"
    if normalized in {"final_only", "at_final"}:
        return "final_only"
    if normalized in {"readiness_only", "readiness_target", "on_target_only"}:
        return "readiness_only"
    if normalized in {"readiness_or_final", "target_or_final"}:
        return "readiness_or_final"
    return "always"


def _normalized_submission_gate(gate: str | None, *, default: str) -> str:
    normalized = str(gate or "").strip().lower()
    if normalized in {"always", "each_iteration"}:
        return "always"
    if normalized in {"final_only", "at_final"}:
        return "final_only"
    if normalized in {"readiness_only", "readiness_target", "on_target_only"}:
        return "readiness_only"
    if normalized in {"readiness_or_final", "target_or_final"}:
        return "readiness_or_final"
    return default


def _pipeline_config_hash(*, model_summary: dict[str, object], metric: str, accelerator: str) -> str:
    stable_payload: dict[str, object] = {
        "metric": metric,
        "accelerator": accelerator,
    }
    for key, value in model_summary.items():
        if key in {"evaluation_by_source", "timing", "elapsed", "duration"}:
            continue
        stable_payload[key] = value
    encoded = json.dumps(stable_payload, sort_keys=True, default=_diagnostics_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _diagnostics_json_default(obj: object) -> object:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted([str(item) for item in obj])
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    for attr in ("item", "tolist"):
        func = getattr(obj, attr, None)
        if callable(func):
            try:
                return func()
            except Exception:  # noqa: BLE001
                break
    return {
        "__type__": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
        "__repr__": repr(obj),
    }


def _build_diagnostics(
    *,
    evaluation: EvaluationResult,
    model_summary: dict[str, object],
    best_score: float | None,
    target_score: float,
    dataset_profile: dict[str, object],
    top1_score: float | None,
    top1_tier: bool,
    diff_summary: str,
    evaluation_by_source: dict[str, EvaluationResult] | None = None,
    loop_decision_score: float | None = None,
    loop_decision_source: str = "offline",
    quality_guard: dict[str, object] | None = None,
) -> str:
    direction = evaluation.direction
    decision_score = evaluation.value if loop_decision_score is None else loop_decision_score
    delta_to_target = target_score - decision_score if direction == "minimize" else decision_score - target_score
    best_line = best_score if best_score is not None else decision_score
    trend = "improving" if best_score is None or _meets_target(decision_score, best_line, direction) else "stalled"
    top1_delta = None
    if top1_score is not None:
        top1_delta = top1_score - decision_score if direction == "minimize" else decision_score - top1_score
    gap = None
    if evaluation.train_score is not None and evaluation.val_score is not None:
        gap = evaluation.train_score - evaluation.val_score

    dataset_lines = []
    if dataset_profile:
        dataset_lines = [
            f"- Train rows/cols: {dataset_profile.get('train_rows')} / {dataset_profile.get('train_cols')}",
            f"- Test rows/cols: {dataset_profile.get('test_rows')} / {dataset_profile.get('test_cols')}",
            f"- Missingness: {dataset_profile.get('missingness')}",
            f"- Categorical cols: {len(dataset_profile.get('categorical_columns', []))}",
            f"- High-cardinality cols: {len(dataset_profile.get('high_cardinality_columns', []))}",
        ]
    else:
        dataset_lines = ["- Dataset profile unavailable."]

    lines = [
        "# Diagnostics",
        "",
        f"Loop decision: source={loop_decision_source} score={decision_score:.6f}",
        f"Score vs target: {decision_score:.6f} vs {target_score:.6f} (delta {delta_to_target:.6f})",
        f"Best so far: {best_line:.6f} ({trend})",
        f"Evaluation: {evaluation.score_source}",
    ]
    if evaluation_by_source:
        lines.append(
            "Offline by source: "
            + ", ".join(f"{source}={result.value:.6f}" for source, result in evaluation_by_source.items())
        )
    else:
        lines.append(f"Offline ({evaluation.score_source}): {evaluation.value:.6f}")
    if top1_score is None:
        lines.append("Top1 public score: unavailable")
    else:
        lines.append(f"Top1 public score: {top1_score:.6f} (delta {top1_delta:.6f}, top1-tier={top1_tier})")
    if gap is not None:
        lines.append(f"Train/val gap: {gap:.6f}")
    if evaluation.std is not None:
        lines.append(f"CV std: {evaluation.std:.6f}")
    if quality_guard:
        reasons = quality_guard.get("reasons")
        warning_values = quality_guard.get("warnings")
        reason_text = (
            ", ".join(str(item) for item in reasons if isinstance(item, str))
            if isinstance(reasons, list) and reasons
            else "none"
        )
        warning_text = (
            ", ".join(str(item) for item in warning_values if isinstance(item, str))
            if isinstance(warning_values, list) and warning_values
            else "none"
        )
        lines.append(
            "Kernel quality guard: "
            f"allow_submit={bool(quality_guard.get('allow_submit', True))}, "
            f"reasons={reason_text}, warnings={warning_text}"
        )
    lines += [
        "",
        "Dataset summary:",
        *dataset_lines,
        "",
        "Pipeline summary:",
        json.dumps(model_summary, indent=2, default=_diagnostics_json_default),
        "",
        "Suspected causes:",
        "- Underfit if train/val both low; overfit if gap large.",
        "- Check categorical encoding, leakage, and missing value handling.",
        "",
        "Next improvements (ranked):",
        "1) Try a stronger model or tuning.",
        "2) Add features or target transformations.",
        "3) Adjust validation strategy.",
        "",
        "Diff summary:",
        diff_summary or "No code changes.",
    ]
    return "\n".join(lines) + "\n"


def _format_kernel_error(exc: Exception) -> str:
    trace = traceback.format_exc()
    header = f"{exc.__class__.__name__}: {exc}".strip()
    if isinstance(exc, KaggleCliError) and getattr(exc, "output", ""):
        header = f"{header}\nKaggle CLI output:\n{exc.output}".strip()
    if trace and trace != "NoneType: None\n":
        return f"{header}\n{trace}".strip()
    return header


def _kernel_source_preflight_error(*, config: AutopilotConfig) -> str | None:
    """Return source contract validation error text, or None when ready."""
    kernel_dir = config.paths.kernel_source_dir
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return (
            "RuntimeError: Local autopilot requires kernel.py, but "
            f"{kernel_path} was not found. "
            "Run planning/implement to generate kernel.py first."
        )
    try:
        ensure_kernel_sources_valid(kernel_dir, require_kaggle_input=False)
    except Exception as exc:  # noqa: BLE001
        return _format_kernel_error(exc)
    return None


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
        preflight_error = _kernel_source_preflight_error(config=config)
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
            f"invoking codex fix (attempt {attempt}/{_MAX_KERNEL_PREFLIGHT_FIX_ATTEMPTS})"
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


def _fingerprint_error(message: str) -> str:
    normalized = " ".join(message.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def _record_kernel_error(
    *,
    logs_dir: Path,
    attempt: int,
    error_text: str,
    error_fingerprints: dict[str, int],
    max_repeats: int | None = None,
    output_dir: Path | None = None,
) -> None:
    enriched_error = error_text
    if output_dir is not None and output_dir.exists():
        log_tail = _collect_log_tail(output_dir, max_lines=200)
        if log_tail and log_tail not in enriched_error:
            enriched_error = f"{enriched_error}\n\n--- kernel log tail ---\n{log_tail}"
    fingerprint = _fingerprint_error(enriched_error)
    error_fingerprints[fingerprint] = error_fingerprints.get(fingerprint, 0) + 1
    repeat_limit = MAX_SAME_KERNEL_ERROR_REPEATS if max_repeats is None else max_repeats
    if repeat_limit is not None and error_fingerprints[fingerprint] > repeat_limit:
        raise KernelFailedError(
            "Kernel failure repeated with the same error; aborting auto-fix loop to avoid an infinite retry."
        )
    attempt_tag = f"{attempt:02d}"
    header = (
        f"kernel_attempt: {attempt}\n"
        f"error_fingerprint: {fingerprint}\n"
        f"error_repeat: {error_fingerprints[fingerprint]}\n"
    )
    numbered_path = logs_dir / f"kernel_error-{attempt_tag}.txt"
    numbered_path.write_text(header + enriched_error + "\n", encoding="utf-8")
    (logs_dir / "kernel_error.txt").write_text(header + enriched_error + "\n", encoding="utf-8")


def _is_kernel_registration_error(exc: Exception) -> bool:
    if isinstance(exc, KernelFailedError) and "kernel not found after push" in str(exc).lower():
        return True
    if isinstance(exc, KaggleCliError) and "kernels/status" in str(getattr(exc, "output", "")).lower():
        return True
    return False


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
    forced_improvement_mode: str | None = None,
    forced_improvement_reason: str | None = None,
    enforce_code_reference_implementation: bool = False,
    code_reference_enforcement_reason: str | None = None,
    best_score_so_far: float | None = None,
) -> None:
    prompt_template = config.paths.codex_improve_template.read_text(encoding="utf-8")
    agent_dir = iter_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = agent_dir / "prompt.md"
    top1_score = top1_info.get("score") if isinstance(top1_info, dict) else None
    effective_current_score = evaluation.value if current_score is None else current_score
    improvement_mode, top1_gap = _classify_improvement_mode(
        effective_current_score,
        top1_score,
        evaluation.direction,
    )
    if forced_improvement_mode:
        print(
            "[yellow]improve mode override[/yellow]: "
            f"{improvement_mode} -> {forced_improvement_mode} ({forced_improvement_reason or 'policy'})"
        )
        improvement_mode = forced_improvement_mode
    kernel_main_path = config.paths.kernel_source_dir / "kernel.py"
    code_reference_score, code_reference_source = _extract_code_reference_score(config.paths)
    code_reference_delta = (
        _score_delta_vs_reference(effective_current_score, code_reference_score, evaluation.direction)
        if code_reference_score is not None
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
    required_reference_notebook = _load_required_reference_notebook(config.paths)
    base_prompt_text = prompt_template.format(
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
    if forced_improvement_reason:
        base_prompt_text += (
            "\n\nForced improvement mode policy is active.\n"
            f"Reason: {forced_improvement_reason}\n"
            "Do not propose minor_tuning; make a major_overhaul plan.\n"
        )
    if best_score_so_far is not None:
        base_prompt_text += (
            "\n\nRegression Guard Policy:\n"
            f"- Best known offline score so far: {float(best_score_so_far):.6f}\n"
            "- Do NOT introduce conservative fallback paths that intentionally reduce model capacity "
            "or collapse features (e.g., tiny robust subsets) when they materially degrade offline quality.\n"
            "- If suspiciously high CV is detected, keep leak fixes but preserve competitive model strength "
            "instead of defaulting to a weak baseline.\n"
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
                f"(source: {code_reference_source}, delta_vs_current={code_reference_delta:+.6f})"
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
                    f"- required_marker: {_code_reference_marker(required_reference_notebook)}",
                    (
                        "- required_model_family: tabicl"
                        if _reference_requires_tabicl(required_reference_notebook)
                        else "- required_model_family: follow required notebook strategy"
                    ),
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
    problem_type_knowledge = _load_problem_type_knowledge_text(config)
    strategy_prompt = _build_improvement_strategy_prompt(
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
        codex_prompt=base_prompt_text,
        problem_type_knowledge=problem_type_knowledge,
    )
    strategy_dir = agent_dir / f"improve_strategy-{iteration:02d}"
    strategy_text = _run_improvement_strategy(
        prompt_text=strategy_prompt,
        output_dir=strategy_dir,
        dry_run=config.dry_run,
    )

    prompt_text = base_prompt_text
    if strategy_text:
        prompt_text = (
            "# Codex Improvement Implementation\n\n"
            "Implement the GPT-authored improvement prompt below as the primary plan.\n\n"
            "## GPT 5.2 Extra-High Improvement Prompt\n"
            f"{strategy_text}\n\n"
            "## Local Context (for file paths and constraints)\n"
            f"{base_prompt_text}\n"
        )

    prompt_path.write_text(prompt_text, encoding="utf-8")
    _print_agent_prompt(prompt_path, prompt_text)

    print("[cyan]improve[/cyan]: running codex implementer")
    # Codex runner always writes execution logs (codex_exec.jsonl / codex_last_message.txt)
    # under the provided output_dir (agent_dir). Include it in the allowlist so the guard
    # does not fail on its own transcripts.
    #
    # During improvement iterations we also update competition context (e.g. leaderboard snapshots,
    # eval advisor status) and run-scoped metadata (run.json, evaluation_report.json). Those are
    # side effects of Kagglebot itself rather than agent edits, so they must be allowlisted here
    # to avoid spurious write-guard failures.
    allowed_prefixes = [
        config.paths.kernel_source_dir,
        config.paths.context_dir,
        config.paths.run_dir(run_id),
        agent_dir,
        config.paths.repo_root / "pyproject.toml",
        config.paths.repo_root / "uv.lock",
    ]

    def _run_improve_codex_pass(*, current_prompt_path: Path, stage_suffix: str) -> tuple[str, Path]:
        guard_snapshot = _backup_guarded_files(config.paths.repo_root, allowed_prefixes)
        before = _snapshot_tree(config.paths.repo_root)
        result = run_codex(
            current_prompt_path,
            agent_dir,
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
        response = _read_agent_response(result.last_message_path)
        _print_agent_response(result.last_message_path, response)
        if result.returncode != 0:
            raise RuntimeError("Codex improvement failed.")
        return response, result.last_message_path

    response_text, _ = _run_improve_codex_pass(current_prompt_path=prompt_path, stage_suffix="")

    if code_reference_mandatory and required_reference_notebook is not None and not config.dry_run:
        kernel_path = config.paths.kernel_source_dir / "kernel.py"
        implementation_issues = _validate_code_reference_implementation(
            kernel_path=kernel_path,
            reference=required_reference_notebook,
        )
        if implementation_issues:
            print(
                "[yellow]code reference guard[/yellow]: "
                "required reference implementation missing; rerunning codex with strict repair prompt."
            )
            repair_prompt_path = agent_dir / f"code_reference_repair_prompt-{iteration:02d}.md"
            repair_prompt_text = _build_code_reference_repair_prompt(
                base_prompt_text=base_prompt_text,
                reference=required_reference_notebook,
                issues=implementation_issues,
                kernel_path=kernel_path,
            )
            repair_prompt_path.write_text(repair_prompt_text, encoding="utf-8")
            _print_agent_prompt(repair_prompt_path, repair_prompt_text)
            repair_response, _ = _run_improve_codex_pass(
                current_prompt_path=repair_prompt_path,
                stage_suffix="_code_reference_repair",
            )
            implementation_issues = _validate_code_reference_implementation(
                kernel_path=kernel_path,
                reference=required_reference_notebook,
            )
            if implementation_issues:
                issues_text = ", ".join(implementation_issues)
                raise RuntimeError(
                    f"Code reference implementation requirement not satisfied after repair pass (issues={issues_text})."
                )
            response_text = f"{response_text}\n\n{repair_response}".strip()

    _run_verify(config.verify_cmd, dry_run=config.dry_run)
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


def _build_improvement_strategy_prompt(
    *,
    slug: str,
    run_id: str,
    iteration: int,
    metric: str,
    direction: str,
    current_score: float,
    current_score_source: str,
    target_score: float,
    top1_score: float | None,
    top1_source: str,
    top1_gap: float | None,
    delta_offline: float | None,
    improvement_mode: str,
    codex_prompt: str,
    problem_type_knowledge: str,
) -> str:
    return f"""\
# Kagglebot Improvement Strategy

You are GPT 5.2 in extra-high reasoning mode.
Design a concrete improvement prompt for Codex 5.3 (extra-high), which will implement changes.

Competition: {slug}
Run ID: {run_id}
Iteration: {iteration}
Metric: {metric} ({direction})
Current score: {current_score:.6f} (source: {current_score_source})
Target score: {target_score:.6f}
Top1 score: {"unavailable" if top1_score is None else f"{top1_score:.6f}"}
Top1 source: {top1_source}
Top1 gap: {"unavailable" if top1_gap is None else f"{top1_gap:.6f}"}
Delta vs previous best: {"unavailable" if delta_offline is None else f"{delta_offline:.6f}"}
Improvement mode: {improvement_mode}

## Existing Codex Improvement Prompt Draft

```
{codex_prompt}
```

## Problem-Type Knowledge (Past Causes and Fixes)

{problem_type_knowledge}

## Required Output

Return concise, actionable implementation instructions for Codex:
1) What to change and why (root-cause hypothesis of current gap).
2) Exact file-level edits and model/training changes.
3) Validation checks after edits (what metrics/logs to confirm).
4) Fallback if the first plan fails.

Do not include chain-of-thought.
"""


def _run_improvement_strategy(*, prompt_text: str, output_dir: Path, dry_run: bool) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "gpt_improvement_prompt.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    print("[cyan]improve[/cyan]: gpt drafting improvement prompt")
    result = run_strategy(prompt_path, output_dir, dry_run=dry_run)
    strategy_text = _read_agent_response(result.last_message_path).strip()
    if result.returncode != 0:
        print("[yellow]improve[/yellow]: gpt improvement strategy failed, falling back to direct codex prompt")
        return ""
    if not strategy_text:
        print("[yellow]improve[/yellow]: gpt improvement strategy empty, falling back to direct codex prompt")
        return ""
    return strategy_text


def _error_strategy_skip_reason(*, stage: str, error_text: str) -> str | None:
    """Return a deterministic reason to skip GPT strategy analysis, if any."""
    normalized_stage = str(stage or "").strip().lower()
    lowered = normalize_error_text(error_text or "", max_chars=8000).lower()
    if not lowered:
        return None

    cross_stage_patterns = (
        (
            "competition metric mismatch",
            "strict competition metric mismatch escalation is deterministic",
        ),
    )
    for needle, reason in cross_stage_patterns:
        if needle in lowered:
            return reason

    common_patterns = (
        (
            "kernel source validation failed",
            "deterministic kernel source validation failure",
        ),
        (
            "do not reference metrics.json output",
            "missing metrics.json output contract is deterministic",
        ),
        (
            "do not reference submission.csv output",
            "missing submission.csv output contract is deterministic",
        ),
        (
            "unexpected keyword argument 'evaluation_strategy'",
            "known transformers eval_strategy API mismatch",
        ),
        (
            "modulenotfounderror: no module named",
            "deterministic missing module error",
        ),
        (
            "keyerror:",
            "deterministic dataframe key/column error",
        ),
        (
            "not in index",
            "deterministic dataframe column mismatch",
        ),
        (
            "missing columns",
            "deterministic missing-column error",
        ),
        (
            "data directory not found:",
            "deterministic local data path resolution failure",
        ),
        (
            "unable to resolve competition data root",
            "deterministic competition data path resolution failure",
        ),
    )
    if normalized_stage != "submit_autofix":
        for needle, reason in common_patterns:
            if needle in lowered:
                return reason

    if normalized_stage == "submit_autofix":
        submit_patterns = (
            (
                "cannot use internet access in this competition",
                "competition internet policy violation is deterministic",
            ),
            (
                "disable internet in the notebook editor",
                "competition internet policy violation is deterministic",
            ),
            (
                "submission file must be named submission.csv",
                "submission filename contract violation is deterministic",
            ),
        )
        for needle, reason in submit_patterns:
            if needle in lowered:
                return reason
    return None


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

    prompt_template = config.paths.codex_kernel_fix_template.read_text(encoding="utf-8")
    prompt_path = agent_dir / "kernel_fix_prompt.md"
    missing_module = _extract_missing_module(error_message)
    blocked_modules = _load_blocked_modules(config.paths.context_dir)
    if missing_module:
        # Keep dependency recovery paths open: do not auto-block newly missing modules.
        blocked_modules = [name for name in blocked_modules if name != missing_module]
        _save_blocked_modules(config.paths.context_dir, blocked_modules)
    blocked_text = "\n".join(f"- {name}" for name in blocked_modules) if blocked_modules else "None"
    prompt_text = prompt_template.format(
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
        strategy_skip_reason = _error_strategy_skip_reason(stage="kernel_fix", error_text=error_message)
    if strategy_skip_reason:
        print(
            "[yellow]kernel fix[/yellow]: "
            f"skipping gpt strategy ({strategy_skip_reason}); invoking codex fixer directly."
        )
    else:
        strategy_prompt = _build_error_strategy_prompt(
            stage="kernel_fix",
            slug=config.slug,
            run_id=run_id,
            attempt=attempt,
            compute=config.compute,
            accelerator=config.accelerator,
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
            "\n\n## GPT 5.2 Extra-High Error-Fix Strategy\n"
            "Use the strategy below as guidance, then apply minimal targeted edits.\n\n"
            f"{strategy_text}\n"
        )

    base_prompt_text = f"Kernel fix attempt: {attempt}\n\n{prompt_text}"
    prompt_path.write_text(base_prompt_text, encoding="utf-8")
    attempt_path = agent_dir / f"kernel_fix_prompt-{attempt:02d}.md"
    attempt_path.write_text(base_prompt_text, encoding="utf-8")
    _print_agent_prompt(prompt_path, base_prompt_text)

    allowed_prefixes = [
        config.paths.repo_root / "src",
        config.paths.repo_root / "docs",
        config.paths.repo_root / "tests",
        config.paths.repo_root / "pyproject.toml",
        config.paths.repo_root / "uv.lock",
        config.paths.kernel_source_dir,
        config.paths.context_dir,
        config.paths.runs_dir,
        config.paths.prompts_dir,
        config.paths.submissions_dir,
        config.paths.kernels_dir,
        config.paths.base_dir / "outputs",
    ]
    guard_snapshot = _backup_guarded_files(config.paths.repo_root, allowed_prefixes)
    codex_pass_limit = max(1, int(max_codex_passes or MAX_KERNEL_FIX_CODEX_PASSES))
    retry_feedback = ""
    last_response_text = ""
    for codex_pass in range(1, codex_pass_limit + 1):
        pass_prompt_text = (
            base_prompt_text
            if not retry_feedback
            else _append_fix_retry_feedback(
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
                f"retrying codex pass {codex_pass}/{codex_pass_limit} with previous failure context."
            )
        before = _snapshot_tree(config.paths.repo_root)
        pass_output_dir = (
            agent_dir if codex_pass == 1 else agent_dir / f"kernel_fix_pass-{attempt:02d}-{codex_pass:02d}"
        )
        print("[cyan]kernel fix[/cyan]: running codex fixer")
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
        response_text = _read_agent_response(result.last_message_path)
        _print_agent_response(result.last_message_path, response_text)
        last_response_text = response_text
        if result.returncode != 0:
            retry_feedback = (
                "Codex kernel-fix step failed with non-zero exit status.\n"
                f"returncode={result.returncode}\n"
                f"pass={codex_pass}/{codex_pass_limit}\n"
                f"response={response_text}"
            )
            if codex_pass < codex_pass_limit:
                continue
            raise RuntimeError("Codex kernel-fix step failed.")

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
                _run_verify(config.verify_cmd, dry_run=config.dry_run)
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
                            "Codex kernel-fix made no edits; regenerated kernel sources once and verification passed."
                        ),
                        "resolved": True,
                    }
                )
            return

        try:
            _run_verify(config.verify_cmd, dry_run=config.dry_run)
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

    raise RuntimeError("Kernel fix exhausted codex retry passes without resolving the error.")


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
    """Apply a metric-only kernel fix using Codex without GPT strategy mediation."""
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
        resolved_submission = _resolve_iteration_submission_artifact(iter_dir)
        if resolved_submission is None:
            raise RuntimeError(
                "Metric recheck failed: submission.csv artifact is missing for same-iteration metric-only recheck."
            )
        rechecked_submission_path = _copy_submission_artifact_to_iteration_dir(
            source=resolved_submission,
            iter_dir=iter_dir,
        )

    output_metrics_path = iter_dir / "output" / "metrics.json"
    resolved_metrics_path = output_metrics_path if output_metrics_path.exists() else metrics_artifact_path
    if resolved_metrics_path is None or (not resolved_metrics_path.exists()):
        resolved_metrics_path = _resolve_iteration_artifact(iter_dir, "metrics.json")
    if resolved_metrics_path is None:
        raise RuntimeError(
            "Metric recheck failed: metrics.json artifact is missing for same-iteration metric-only recheck."
        )

    payload = _load_json_object(resolved_metrics_path)
    evaluation = _load_kernel_metrics(
        resolved_metrics_path,
        metric_direction,
        target_metric,
    )
    metric_mismatch = bool(
        target_metric and evaluation and evaluation.metric and not _metrics_equivalent(evaluation.metric, target_metric)
    )
    needs_recompute = evaluation is None or metric_mismatch
    if needs_recompute:
        recomputed = _recompute_metric_from_oof_artifact(
            iter_dir=iter_dir,
            payload=payload,
            target_metric=target_metric,
            metric_direction=metric_direction,
        )
        if recomputed is not None:
            evaluation, payload = recomputed
            _persist_metric_recheck_payload(
                iter_dir=iter_dir,
                resolved_metrics_path=resolved_metrics_path,
                payload=payload,
            )
    if evaluation is None:
        raise RuntimeError("Metric recheck failed: kernel metrics missing expected score.")
    return evaluation, payload, rechecked_submission_path


def _recompute_metric_from_oof_artifact(
    *,
    iter_dir: Path,
    payload: dict[str, object] | None,
    target_metric: str | None,
    metric_direction: str,
) -> tuple[EvaluationResult, dict[str, object]] | None:
    """Recompute target metric from cached OOF predictions without rerunning training."""
    if not target_metric:
        return None
    oof_path = _resolve_iteration_artifact(iter_dir, "oof_predictions.csv")
    if oof_path is None or not oof_path.exists():
        return None
    try:
        import pandas as pd
    except Exception:
        return None
    try:
        oof = pd.read_csv(oof_path)
    except Exception:
        return None
    if oof.empty:
        return None

    y_col = _pick_oof_target_column(oof)
    pred_col = _pick_oof_prediction_column(oof, metric=target_metric)
    if y_col is None or pred_col is None:
        return None

    y_series = pd.to_numeric(oof[y_col], errors="coerce")
    pred_series = pd.to_numeric(oof[pred_col], errors="coerce")
    valid_mask = y_series.notna() & pred_series.notna()
    if int(valid_mask.sum()) < 2:
        return None
    y_values = y_series[valid_mask].to_numpy()
    pred_values = pred_series[valid_mask].to_numpy()

    try:
        metric_value = float(compute_metric(target_metric, y_values, pred_values))
    except Exception:
        return None

    metric_name = canonical_metric(target_metric)
    direction = infer_direction(metric_name, metric_direction)
    score_source_raw = payload.get("score_source") if isinstance(payload, dict) else None
    score_source = (
        str(score_source_raw).strip() if isinstance(score_source_raw, str) and str(score_source_raw).strip() else "cv"
    )
    std_value = _to_float(payload.get("offline_std")) if isinstance(payload, dict) else None
    train_score = _to_float(payload.get("train_score")) if isinstance(payload, dict) else None
    val_score = _to_float(payload.get("val_score")) if isinstance(payload, dict) else None
    fold_scores = _extract_numeric_list(payload.get("fold_scores")) if isinstance(payload, dict) else None

    from kagglebot.solver.evaluate import EvaluationResult

    evaluation = EvaluationResult(
        score_source=score_source,
        metric=metric_name,
        direction=direction,  # type: ignore[arg-type]
        value=metric_value,
        std=std_value,
        train_score=train_score,
        val_score=val_score,
        fold_scores=fold_scores,
    )
    updated_payload = dict(payload) if isinstance(payload, dict) else {}
    updated_payload["metric"] = metric_name
    updated_payload["direction"] = direction
    updated_payload["score_source"] = score_source
    updated_payload["offline_value"] = metric_value
    updated_payload["value"] = metric_value
    updated_payload["metric_recheck_source"] = f"oof_predictions:{oof_path.name}"
    updated_payload["metric_recheck_without_retrain"] = True
    loop_decision = updated_payload.get("loop_decision")
    if isinstance(loop_decision, dict):
        loop_decision["source"] = score_source
        loop_decision["value"] = metric_value
    else:
        updated_payload["loop_decision"] = {"source": score_source, "value": metric_value}
    return evaluation, updated_payload


def _pick_oof_target_column(frame) -> str | None:  # type: ignore[no-untyped-def]
    """Return the target column name from an OOF prediction table."""
    columns = [str(col) for col in frame.columns]
    normalized = {col.lower().strip(): col for col in columns}
    for key in ("y", "target", "label", "y_true", "isdefault", "is_default"):
        if key in normalized:
            return normalized[key]
    return None


def _pick_oof_prediction_column(frame, *, metric: str) -> str | None:  # type: ignore[no-untyped-def]
    """Return the most suitable prediction column for the requested metric."""
    columns = [str(col) for col in frame.columns]
    normalized = {col.lower().strip(): col for col in columns}
    is_prob_metric = bool(metric_requires_proba(metric))

    if is_prob_metric:
        for key in ("oof_proba", "pred_proba", "prediction_proba", "probability", "proba", "score"):
            if key in normalized:
                return normalized[key]
        for col in columns:
            lowered = col.lower()
            if "proba" in lowered or "prob" in lowered or "score" in lowered:
                return col
    for key in ("oof_pred", "prediction", "pred", "y_pred"):
        if key in normalized:
            return normalized[key]
    if is_prob_metric:
        return None
    for col in columns:
        lowered = col.lower()
        if any(token in lowered for token in ("pred", "score", "proba", "prob")):
            return col
    return None


def _extract_numeric_list(value: object) -> list[float] | None:
    """Return parsed numeric list or None when payload value is not a numeric list."""
    if not isinstance(value, list):
        return None
    parsed = [float(item) for item in value if isinstance(item, (int, float))]
    return parsed or None


def _persist_metric_recheck_payload(*, iter_dir: Path, resolved_metrics_path: Path, payload: dict[str, object]) -> None:
    """Persist recomputed metric payload to canonical iteration metrics artifacts."""
    serialized = json.dumps(payload, indent=2)
    candidates = [resolved_metrics_path, iter_dir / "metrics.json", iter_dir / "output" / "metrics.json"]
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized, encoding="utf-8")


def _run_autofix(*, config: AutopilotConfig, run_id: str, attempt: int, error: Exception) -> None:
    run_dir = config.paths.run_dir(run_id)
    autofix_dir = run_dir / "autofix" / f"attempt-{attempt}"
    autofix_dir.mkdir(parents=True, exist_ok=True)
    error_text = "".join(traceback.format_exception(type(error), error, error.__traceback__)).strip()
    submit_autofix = isinstance(error, SubmitAbortedError)
    submit_context = ""
    if isinstance(error, KaggleCliError):
        if error.command:
            error_text = f"{error_text}\n\nKaggle CLI command:\n{shlex.join(error.command)}"
        if error.output:
            error_text = f"{error_text}\n\nKaggle CLI output:\n{error.output}"
    if submit_autofix:
        submit_context = _build_submit_autofix_context(run_dir)
        if submit_context:
            error_text = f"{error_text}\n\nSubmit Failure Context:\n{submit_context}"
    attempt_tag = f"{attempt:02d}"
    header = f"autofix_attempt: {attempt}\n"
    error_path = autofix_dir / f"error-{attempt_tag}.txt"
    error_path.write_text(header + error_text + "\n", encoding="utf-8")
    (autofix_dir / "error.txt").write_text(header + error_text + "\n", encoding="utf-8")

    allowed_prefixes = _autofix_allowed_prefixes(config)
    prompt_text = _build_autofix_prompt(
        config=config,
        run_id=run_id,
        attempt=attempt,
        error_text=error_text,
        error_path=error_path,
        allowed_prefixes=allowed_prefixes,
        submit_context=submit_context,
    )
    strategy_stage = "submit_autofix" if submit_autofix else "autofix"
    strategy_label = "submit autofix" if submit_autofix else "autofix"
    print(
        f"[cyan]{strategy_label}[/cyan]: strategy={_ERROR_STRATEGY_MODEL}({_ERROR_STRATEGY_REASONING_EFFORT}) "
        f"-> fixer={_ERROR_FIX_CODEX_MODEL}({_ERROR_FIX_REASONING_EFFORT})"
    )
    strategy_prompt = _build_error_strategy_prompt(
        stage=strategy_stage,
        slug=config.slug,
        run_id=run_id,
        attempt=attempt,
        compute=config.compute,
        accelerator=config.accelerator,
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
        raise RuntimeError(
            "Autofix strategy generation failed: strict autofix path requires GPT strategy output before Codex edits."
        )
    prompt_text += (
        "\n\n## GPT 5.2 Extra-High Error-Fix Strategy\n"
        "Use the strategy below as guidance, then apply minimal targeted edits.\n\n"
        f"{strategy_text}\n"
    )
    prompt_path = autofix_dir / "prompt.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    _print_agent_prompt(prompt_path, prompt_text)

    retry_feedback = ""
    for codex_pass in range(1, MAX_AUTOFIX_CODEX_PASSES + 1):
        pass_prompt_text = (
            prompt_text
            if not retry_feedback
            else _append_fix_retry_feedback(
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
                f"retrying codex pass {codex_pass}/{MAX_AUTOFIX_CODEX_PASSES} with previous failure context."
            )

        before = _snapshot_tree(config.paths.repo_root)
        pass_output_dir = autofix_dir if codex_pass == 1 else autofix_dir / f"pass-{codex_pass:02d}"
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
        response_text = _read_agent_response(result.last_message_path)
        _print_agent_response(result.last_message_path, response_text)
        if result.returncode != 0:
            retry_feedback = (
                "Codex autofix step failed with non-zero exit status.\n"
                f"returncode={result.returncode}\n"
                f"pass={codex_pass}/{MAX_AUTOFIX_CODEX_PASSES}\n"
                f"response={response_text}"
            )
            if codex_pass < MAX_AUTOFIX_CODEX_PASSES:
                continue
            raise RuntimeError("Codex autofix step failed.")

        try:
            _run_verify(config.verify_cmd, dry_run=config.dry_run)
        except Exception as exc:  # noqa: BLE001
            retry_feedback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
            if codex_pass < MAX_AUTOFIX_CODEX_PASSES:
                continue
            raise
        return

    raise RuntimeError("Autofix exhausted codex retry passes without resolving the error.")


def _autofix_allowed_prefixes(config: AutopilotConfig) -> list[Path]:
    # Keep src writable during autofix so runtime/framework issues in core code can be repaired.
    # Do not grant broad competition-root/kernels write access; fixes must target authoritative
    # sources (src/kernel/context/prompts) rather than generated staged artifacts.
    candidates = [
        config.paths.repo_root / "src",
        config.paths.repo_root / "docs",
        config.paths.repo_root / "tests",
        config.paths.repo_root / "pyproject.toml",
        config.paths.repo_root / "uv.lock",
        # Allow competition-scoped prompt/context/submission artifacts and authoritative kernel.
        config.paths.kernel_source_dir,
        config.paths.context_dir,
        config.paths.runs_dir,
        config.paths.prompts_dir,
        config.paths.submissions_dir,
    ]
    module_src_root = Path(__file__).resolve().parents[1]
    # In some test/runtime setups, config.repo_root can differ from the imported module tree.
    # Include the active src root so autofix can patch the file that raised the exception.
    if module_src_root.name == "src":
        candidates.append(module_src_root)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _build_autofix_prompt(
    *,
    config: AutopilotConfig,
    run_id: str,
    attempt: int,
    error_text: str,
    error_path: Path,
    allowed_prefixes: list[Path],
    submit_context: str = "",
) -> str:
    allowed_list = "\n".join(f"- {path}" for path in allowed_prefixes)
    submit_context_block = ""
    if submit_context:
        submit_context_block = (
            "\n## Submit Context\n\n"
            "This is a submit-stage failure. Prioritize fixing the submission path and output format first.\n\n"
            "```\n"
            f"{submit_context}\n"
            "```\n"
        )
    return f"""\
# Kagglebot Codex: Auto-Fix

## Context

Competition: {config.slug}
Run ID: {run_id}
Attempt: {attempt}
Compute: {config.compute} ({config.accelerator})

## Error

```
{error_text}
```

Error log file: {error_path}
{submit_context_block}

## Relevant Paths

- repo_root: {config.paths.repo_root}
- run_dir: {config.paths.run_dir(run_id)}
- kernel_dir: {config.paths.kernel_source_dir}
- context_dir: {config.paths.context_dir}
- data_dir: {config.paths.data_dir}
- prompts_dir: {config.paths.prompts_dir}
- autopilot: {Path(__file__).resolve()}

## Allowed Edit Scope

{allowed_list}

## Task

1) Identify root cause of the failure.
2) Apply minimal, targeted fixes so autopilot can continue.
3) Do not touch datasets or credentials.
   Prefer already-installed dependencies; add new dependencies only with clear justification.
   If a dependency must be added, use `uv add <package>` and keep `pyproject.toml` + `uv.lock` consistent.
4) Explain what you changed in your response.
"""


def _build_error_strategy_prompt(
    *,
    stage: str,
    slug: str,
    run_id: str,
    attempt: int,
    compute: str,
    accelerator: str,
    error_text: str,
    codex_prompt: str,
) -> str:
    return f"""\
# Kagglebot GPT Error Strategy

You are GPT 5.2 in extra-high reasoning mode.
Think through the failure and produce a concrete fix strategy for Codex 5.3 (extra high), which will apply edits.

Stage: {stage}
Competition: {slug}
Run ID: {run_id}
Attempt: {attempt}
Compute: {compute} ({accelerator})

## Error

```
{error_text}
```

## Codex Fix Prompt (current)

```
{codex_prompt}
```

## Required Output

Return concise, actionable instructions for Codex:
1) Root cause hypothesis.
2) Minimal file edits (paths + what to change).
3) Safety checks to run after edits.
4) Fallback if the first fix does not work.
"""


def _run_error_strategy(
    *,
    prompt_text: str,
    output_dir: Path,
    dry_run: bool,
    stage_label: str,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "gpt_strategy_prompt.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    print(f"[cyan]{stage_label}[/cyan]: gpt analyzing error")
    print(
        f"[cyan]{stage_label}[/cyan]: strategy model={_ERROR_STRATEGY_MODEL} "
        f"reasoning={_ERROR_STRATEGY_REASONING_EFFORT}"
    )
    result = run_strategy(prompt_path, output_dir, dry_run=dry_run)
    strategy_text = _read_agent_response(result.last_message_path).strip()
    if result.returncode != 0:
        print(f"[yellow]{stage_label}[/yellow]: gpt strategy failed, continuing with direct codex fix")
        return ""
    if not strategy_text:
        print(f"[yellow]{stage_label}[/yellow]: gpt strategy empty, continuing with direct codex fix")
        return ""
    return strategy_text


_COLUMN_MAP_FILENAME = "column_map.json"
_COLUMN_FILL_FILENAME = "column_fill.json"
_OBJECT_COERCE_FILENAME = "object_coerce.json"
_DEVICE_COERCE_FILENAME = "device_coerce.json"
_BLOCKED_MODULES_FILENAME = "blocked_modules.json"
_MISSING_MODULE_RE = re.compile(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]")
_MISSING_COLUMNS_RE = re.compile(r"missing columns?:\s*\[([^\]]+)\]", re.IGNORECASE)
_MISSING_COLUMNS_KEYERROR_RE = re.compile(
    r"KeyError:\s*[\"']?\[([^\]]+)\]\s*not in index[\"']?",
    re.IGNORECASE,
)
_MISSING_COLUMNS_FILE_RE = re.compile(
    r"([A-Za-z0-9_.-]+\.(?:csv|tsv|txt|parquet|json|jsonl))\s+missing columns",
    re.IGNORECASE,
)
_OBJECT_DTYPE_RE = re.compile(r"numpy\.object_", re.IGNORECASE)
_DEVICE_MISMATCH_RE = re.compile(
    r"Expected all tensors to be on the same device|found at least two devices",
    re.IGNORECASE,
)
_COLUMN_ERROR_PATTERNS = (
    "could not resolve column",
    "unable to locate session",
    "missing columns",
    "not in index",
    "are in the [columns]",
)


def _maybe_write_column_fill(config: AutopilotConfig, error_text: str) -> bool:
    raw_error = error_text or ""
    file_name: str | None = None
    match = _MISSING_COLUMNS_RE.search(raw_error)
    if match:
        missing_columns = _parse_missing_columns(match.group(1))
        file_match = _MISSING_COLUMNS_FILE_RE.search(raw_error)
        file_name = file_match.group(1) if file_match else None
    else:
        keyerror_match = _MISSING_COLUMNS_KEYERROR_RE.search(raw_error)
        if not keyerror_match:
            return False
        missing_columns = _parse_missing_columns(keyerror_match.group(1))
    if not missing_columns:
        return False

    deduped_missing: list[str] = []
    seen_missing: set[str] = set()
    for column in missing_columns:
        normalized = str(column).strip()
        if not normalized or normalized in seen_missing:
            continue
        seen_missing.add(normalized)
        deduped_missing.append(normalized)
    if not deduped_missing:
        return False

    context_dir = config.paths.context_dir
    fill_path = context_dir / _COLUMN_FILL_FILENAME
    payload: dict[str, object] = {}
    if fill_path.exists():
        try:
            loaded = json.loads(fill_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, json.JSONDecodeError):
            payload = {}

    changed = not fill_path.exists()
    files_payload = payload.get("files")
    files: dict[str, list[str]]
    if isinstance(files_payload, dict):
        files = {}
        for key, value in files_payload.items():
            if not isinstance(key, str) or not isinstance(value, list):
                continue
            cleaned: list[str] = []
            seen_cols: set[str] = set()
            for col in value:
                col_name = str(col).strip()
                if not col_name or col_name in seen_cols:
                    continue
                seen_cols.add(col_name)
                cleaned.append(col_name)
            files[key] = cleaned
    else:
        files = {}
        if files_payload is not None:
            changed = True

    if file_name:
        existing = files.get(file_name, [])
        merged = list(existing)
        for col in deduped_missing:
            if col not in merged:
                merged.append(col)
        if merged != existing:
            files[file_name] = merged
            changed = True
    else:
        missing_payload = payload.get("missing_columns")
        if isinstance(missing_payload, list):
            existing_missing: list[str] = []
            seen_cols: set[str] = set()
            for col in missing_payload:
                col_name = str(col).strip()
                if not col_name or col_name in seen_cols:
                    continue
                seen_cols.add(col_name)
                existing_missing.append(col_name)
        else:
            existing_missing = []
            if missing_payload is not None:
                changed = True
        merged_missing = list(existing_missing)
        for col in deduped_missing:
            if col not in merged_missing:
                merged_missing.append(col)
        if merged_missing != existing_missing:
            payload["missing_columns"] = merged_missing
            changed = True

    if not changed:
        return False

    payload["files"] = files
    payload.setdefault("source", "autofix")
    payload.setdefault("created_at", datetime.now(UTC).isoformat())
    payload["updated_at"] = datetime.now(UTC).isoformat()
    if "missing_columns" not in payload:
        payload["missing_columns"] = []
    context_dir.mkdir(parents=True, exist_ok=True)
    fill_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return True


def _maybe_write_object_coerce(config: AutopilotConfig, error_text: str) -> bool:
    if not _OBJECT_DTYPE_RE.search(error_text or ""):
        return False
    context_dir = config.paths.context_dir
    coerce_path = context_dir / _OBJECT_COERCE_FILENAME
    if coerce_path.exists():
        return False
    payload = {
        "source": "autofix",
        "created_at": datetime.now(UTC).isoformat(),
        "enabled": True,
        "reason": "numpy.object_ conversion error",
    }
    context_dir.mkdir(parents=True, exist_ok=True)
    coerce_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return True


def _maybe_write_device_coerce(config: AutopilotConfig, error_text: str) -> bool:
    if not _DEVICE_MISMATCH_RE.search(error_text or ""):
        return False
    context_dir = config.paths.context_dir
    coerce_path = context_dir / _DEVICE_COERCE_FILENAME
    if coerce_path.exists():
        return False
    payload = {
        "source": "autofix",
        "created_at": datetime.now(UTC).isoformat(),
        "enabled": True,
        "prefer": "cuda",
        "reason": "torch device mismatch error",
    }
    context_dir.mkdir(parents=True, exist_ok=True)
    coerce_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return True


def _parse_missing_columns(raw: str) -> list[str]:
    text = raw.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return []
    try:
        parsed = ast.literal_eval(f"[{text}]")
    except Exception:
        parsed = [item.strip().strip("'\"") for item in text.split(",") if item.strip()]
    if isinstance(parsed, (list, tuple)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return []


def _maybe_write_column_map(config: AutopilotConfig, error_text: str) -> bool:
    lowered = error_text.lower()
    if not any(pattern in lowered for pattern in _COLUMN_ERROR_PATTERNS):
        return False
    context_dir = config.paths.context_dir
    map_path = context_dir / _COLUMN_MAP_FILENAME
    if map_path.exists():
        return False
    columns_by_file = _scan_tabular_headers(config.paths.data_dir)
    if not columns_by_file:
        return False
    candidate_groups = _extract_candidate_groups(error_text)
    if not candidate_groups:
        return False
    mapping = _infer_column_mapping(columns_by_file, candidate_groups)
    if not mapping:
        return False
    payload = {
        "mapping": mapping,
        "source": "autofix",
        "created_at": datetime.now(UTC).isoformat(),
        "candidates": candidate_groups,
        "files": columns_by_file,
    }
    context_dir.mkdir(parents=True, exist_ok=True)
    map_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return True


def _scan_tabular_headers(data_dir: Path) -> dict[str, list[str]]:
    columns: dict[str, list[str]] = {}
    if not data_dir.exists():
        return columns
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".csv", ".tsv"}:
            continue
        header = _read_header(path)
        if not header:
            continue
        try:
            rel = str(path.relative_to(data_dir))
        except ValueError:
            rel = str(path)
        columns[rel] = header
    return columns


def _read_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            line = handle.readline()
    except OSError:
        return []
    if not line:
        return []
    sep = "\t" if "\t" in line and path.suffix.lower() == ".tsv" else ","
    try:
        row = next(csv.reader([line], delimiter=sep))
    except Exception:
        return []
    return [col.strip().strip('"').strip("'") for col in row if col.strip()]


def _extract_candidate_groups(error_text: str) -> list[list[str]]:
    groups: list[list[str]] = []
    list_match = re.findall(r"candidates:\s*\[([^\]]+)\]", error_text, flags=re.IGNORECASE)
    for match in list_match:
        try:
            items = [item.strip().strip("'\"") for item in match.split(",") if item.strip()]
        except Exception:
            items = []
        if items:
            groups.append(items)
    slash_match = re.findall(r"([A-Za-z0-9_]+)\s*/\s*([A-Za-z0-9_]+)", error_text)
    for left, right in slash_match:
        groups.append([left, right])
    lowered = error_text.lower()
    if "session" in lowered or "visit" in lowered:
        groups.append(["session_id", "visit_id"])
    if "product" in lowered or "item" in lowered:
        groups.append(["product_id", "item_id", "sku"])
    deduped: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for group in groups:
        norm = tuple(group)
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(group)
    return deduped


def _infer_column_mapping(columns_by_file: dict[str, list[str]], groups: list[list[str]]) -> dict[str, str]:
    all_columns = []
    for cols in columns_by_file.values():
        all_columns.extend(cols)
    mapping: dict[str, str] = {}
    normalized = {_normalize_column(col): col for col in all_columns}
    for group in groups:
        normalized_group = _normalize_group_tokens(group)
        if not normalized_group:
            continue
        canonical = normalized_group[0]
        match = _match_column(normalized_group, normalized, all_columns)
        if match and match not in mapping:
            mapping[match] = canonical
    return mapping


def _match_column(group: list[str], normalized: dict[str, str], all_columns: list[str]) -> str | None:
    for cand in group:
        norm = _normalize_column(cand)
        if norm in normalized:
            return normalized[norm]
    keywords = _keywords_from_group(group)
    if not keywords:
        return None
    best: tuple[int, str] | None = None
    for col in all_columns:
        score = _keyword_score(col, keywords)
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, col)
    return best[1] if best else None


def _normalize_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _normalize_group_tokens(group: list[object]) -> list[str]:
    normalized: list[str] = []
    for cand in group:
        if cand is None:
            continue
        text = cand if isinstance(cand, str) else str(cand)
        stripped = text.strip()
        if stripped:
            normalized.append(stripped)
    return normalized


def _keywords_from_group(group: list[str]) -> set[str]:
    keywords: set[str] = set()
    for cand in group:
        for token in re.split(r"[^a-zA-Z0-9]+", cand):
            if token:
                keywords.add(token.lower())
    return keywords


def _keyword_score(column: str, keywords: set[str]) -> int:
    lowered = column.lower()
    return sum(1 for key in keywords if key in lowered)


def _extract_missing_module(error_text: str) -> str | None:
    match = _MISSING_MODULE_RE.search(error_text or "")
    if not match:
        return None
    return match.group(1)


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
    marker_path.write_text(json.dumps(marker_payload, indent=2), encoding="utf-8")
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
            _maybe_write_column_fill,
        ),
        (
            "object_coerce.json",
            "numpy.object_ conversion error",
            _maybe_write_object_coerce,
        ),
        (
            "device_coerce.json",
            "torch device mismatch error",
            _maybe_write_device_coerce,
        ),
        (
            "column_map.json",
            "column alias mismatch",
            _maybe_write_column_map,
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
        print(f"[yellow]{stage_label}[/yellow]: wrote {artifact_name}; retrying without codex edits")
        return artifact_name
    return None


def _load_blocked_modules(context_dir: Path) -> list[str]:
    path = context_dir / _BLOCKED_MODULES_FILENAME
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, list):
        return [str(item) for item in payload if item]
    return []


def _save_blocked_modules(context_dir: Path, modules: list[str]) -> None:
    context_dir.mkdir(parents=True, exist_ok=True)
    path = context_dir / _BLOCKED_MODULES_FILENAME
    if modules:
        path.write_text(json.dumps(modules, indent=2), encoding="utf-8")
        return
    if path.exists():
        path.unlink()


def _record_blocked_module(context_dir: Path, module: str) -> list[str]:
    context_dir.mkdir(parents=True, exist_ok=True)
    existing = _load_blocked_modules(context_dir)
    if module not in existing:
        existing.append(module)
        _save_blocked_modules(context_dir, existing)
    return existing


def _write_iteration_state_marker(
    *,
    iter_dir: Path,
    run_id: str,
    iteration: int,
    submission_path: Path,
    metrics_path: Path,
    evaluation_report_path: Path,
    submit_phase_required: bool,
    submit_phase_finished: bool | None = None,
    submit_allowed_by_gate: bool,
    submit_phase_state: str,
    submitted: bool,
    readiness_score: float,
) -> None:
    if submit_phase_finished is None:
        submit_phase_finished = (not submit_phase_required) or (not submit_allowed_by_gate) or submitted

    payload = {
        "run_id": run_id,
        "iteration": iteration,
        "iteration_complete": True,
        "trained": True,
        "submission_exists": submission_path.exists(),
        "submission_path": str(submission_path),
        "metrics_exists": metrics_path.exists(),
        "metrics_path": str(metrics_path),
        "evaluation_report_exists": evaluation_report_path.exists(),
        "evaluation_report_path": str(evaluation_report_path),
        "submit_phase_required": submit_phase_required,
        "submit_phase_finished": submit_phase_finished,
        "submit_allowed_by_gate": submit_allowed_by_gate,
        "submit_phase_state": submit_phase_state,
        "submitted": submitted,
        "readiness_score": readiness_score,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    path = iter_dir / _ITERATION_STATE_FILENAME
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_iteration_state_marker(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _is_iteration_marker_complete(payload: dict[str, object], *, require_submit_phase: bool) -> bool:
    if not payload:
        return False
    if not bool(payload.get("iteration_complete")):
        return False
    if require_submit_phase and not bool(payload.get("submit_phase_finished")):
        return False
    if require_submit_phase and bool(payload.get("submit_allowed_by_gate")) and not bool(payload.get("submitted")):
        return False
    return True


def _load_submit_phase_completed_iterations(run_dir: Path) -> set[int]:
    attempts_path = run_dir / "submit_attempts.jsonl"
    if not attempts_path.exists():
        return set()
    completed: set[int] = set()
    try:
        lines = attempts_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return completed
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        action = str(payload.get("action_taken") or "").strip().lower()
        if action not in _LEGACY_SUBMIT_PHASE_COMPLETE_ACTIONS:
            continue
        iteration = _to_int(payload.get("iteration"))
        if iteration is None:
            sub_path = str(payload.get("sub_path") or "").strip()
            if sub_path:
                iteration = _infer_iteration_from_submission_path(Path(sub_path))
        if iteration is not None and iteration > 0:
            completed.add(iteration)
    return completed


def _resume_best_submitted_offline_score(
    *,
    paths: CompetitionPaths,
    run_id: str,
    metric_direction: str,
    target_metric: str,
    max_iterations: int,
) -> float | None:
    """Resume the best offline score among iterations that actually submitted."""
    run_dir = paths.run_dir(run_id)
    if not run_dir.exists():
        return None
    best_score: float | None = None
    for iter_dir in sorted(run_dir.glob("iter-*")):
        if not iter_dir.is_dir():
            continue
        try:
            iteration = int(iter_dir.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        if iteration > max_iterations:
            continue
        marker_path = iter_dir / _ITERATION_STATE_FILENAME
        marker_payload = _load_iteration_state_marker(marker_path)
        if not bool(marker_payload.get("submitted")):
            continue
        metrics_path = iter_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        evaluation = _load_kernel_metrics(metrics_path, metric_direction, target_metric)
        if evaluation is None:
            continue
        if _update_best_score(best_score, evaluation.value, metric_direction, 0.0):
            best_score = evaluation.value
    return best_score


def _resume_iteration_state(
    *,
    paths: CompetitionPaths,
    run_id: str,
    metric_direction: str,
    target_metric: str,
    max_iterations: int,
    require_submit_phase: bool = False,
) -> tuple[int, float | None, Path | None]:
    run_dir = paths.run_dir(run_id)
    if not run_dir.exists():
        return 1, None, None
    best_score: float | None = None
    best_submission: Path | None = None
    completed_iters: list[int] = []
    legacy_submit_phase_iters = _load_submit_phase_completed_iterations(run_dir) if require_submit_phase else set()
    for iter_dir in sorted(run_dir.glob("iter-*")):
        if not iter_dir.is_dir():
            continue
        try:
            iteration = int(iter_dir.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        if iteration > max_iterations:
            continue
        submission_path = _resolve_iteration_submission_artifact(iter_dir)
        metrics_path = iter_dir / "metrics.json"
        if submission_path is None and not metrics_path.exists():
            continue
        if submission_path is not None and not metrics_path.exists():
            print(
                "[yellow]resume[/yellow]: "
                f"iter-{iteration} has submission artifact but no metrics.json; treating as incomplete."
            )
            continue
        if metrics_path.exists() and submission_path is None:
            print(
                "[yellow]resume[/yellow]: "
                f"iter-{iteration} has metrics.json but no submission artifact; treating as incomplete."
            )
            continue

        marker_path = iter_dir / _ITERATION_STATE_FILENAME
        marker_payload = _load_iteration_state_marker(marker_path)
        marker_complete = _is_iteration_marker_complete(marker_payload, require_submit_phase=require_submit_phase)
        legacy_complete = False
        if not marker_complete:
            if require_submit_phase:
                legacy_complete = iteration in legacy_submit_phase_iters
            else:
                legacy_complete = True
            if not legacy_complete:
                phase_note = "submit phase completion" if require_submit_phase else "completion marker"
                print(
                    "[yellow]resume[/yellow]: "
                    f"iter-{iteration} missing {phase_note} ({marker_path.name}); treating as incomplete."
                )
                continue
            print(
                "[yellow]resume[/yellow]: "
                f"iter-{iteration} has no {marker_path.name}; inferred completion from legacy artifacts."
            )

        evaluation = None
        try:
            evaluation = _load_kernel_metrics(metrics_path, metric_direction, target_metric)
        except Exception:  # noqa: BLE001
            evaluation = None
        if evaluation is None:
            print(
                "[yellow]resume[/yellow]: "
                f"{metrics_path} is missing a valid offline metric; treating iter-{iteration} as incomplete."
            )
            continue

        completed_iters.append(iteration)
        if submission_path is None:
            continue
        if best_submission is None:
            best_submission = submission_path
        if best_score is None:
            best_score = evaluation.value
        else:
            if _meets_target(evaluation.value, best_score, metric_direction):
                best_score = evaluation.value
                best_submission = submission_path
    if not completed_iters:
        return 1, best_score, best_submission
    next_iter = max(completed_iters) + 1
    return next_iter, best_score, best_submission


def _newest_existing_path(candidates: list[Path]) -> Path | None:
    existing: list[tuple[float, int, Path]] = []
    for candidate in candidates:
        try:
            if not candidate.exists():
                continue
            stat = candidate.stat()
            existing.append((float(stat.st_mtime), int(stat.st_size), candidate))
        except OSError:
            continue
    if not existing:
        return None
    existing.sort(reverse=True)
    return existing[0][2]


def _resolve_iteration_artifact(iter_dir: Path, filename: str) -> Path | None:
    primary = _newest_existing_path(
        [
            iter_dir / filename,
            iter_dir / "output" / filename,
        ]
    )
    if primary is not None:
        return primary

    # Fallback for resumed/older runs where support artifacts were left only in
    # staged kernel directories instead of iter-*/output.
    run_dir = iter_dir.parent
    runs_dir = run_dir.parent
    competition_dir = runs_dir.parent
    kernel_run_dir = competition_dir / "kernels" / run_dir.name
    fallback_candidates: list[Path] = [
        kernel_run_dir / "outputs" / filename,
        competition_dir / "kernel" / "outputs" / filename,
    ]
    try:
        iteration = int(iter_dir.name.split("-", 1)[1])
    except (IndexError, ValueError):
        iteration = None
    if iteration is not None:
        fallback_candidates.extend(
            [
                kernel_run_dir / f"local-iter-{iteration}" / "outputs" / filename,
                kernel_run_dir / f"submit-iter-{iteration}" / "outputs" / filename,
            ]
        )
    for root in (kernel_run_dir, competition_dir / "kernel" / "outputs"):
        if not root.exists():
            continue
        try:
            fallback_candidates.extend(path for path in root.rglob(filename) if path.is_file())
        except OSError:
            continue
    return _newest_existing_path(fallback_candidates)


def _resolve_iteration_submission_artifact(iter_dir: Path) -> Path | None:
    candidates: list[Path] = [iter_dir / "submission.csv", iter_dir / "output" / "submission.csv"]
    for root in (iter_dir, iter_dir / "output"):
        if not root.exists():
            continue
        try:
            for path in root.rglob("submission.*"):
                if path.is_file():
                    candidates.append(path)
        except OSError:
            continue
    return _newest_existing_path(candidates)


def _copy_submission_artifact_to_iteration_dir(*, source: Path, iter_dir: Path) -> Path:
    destination = iter_dir / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if source.resolve() == destination.resolve():
            return destination
    except OSError:
        pass
    shutil.copy2(source, destination)
    return destination


def _copy_kernel_support_artifacts_to_iteration_dir(*, kernel_output_dir: Path, iter_dir: Path) -> None:
    """Copy optional kernel support artifacts into the canonical iteration output directory."""
    if not kernel_output_dir.exists():
        return
    output_dir = iter_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("oof_predictions.csv", "split_diagnostics.json", "feature_suspects.csv"):
        source = kernel_output_dir / filename
        if not source.exists() or not source.is_file():
            continue
        destination = output_dir / filename
        try:
            if source.resolve() == destination.resolve():
                continue
        except OSError:
            pass
        shutil.copy2(source, destination)


def _latest_iteration_with_training_artifacts(*, run_dir: Path, max_iterations: int) -> int | None:
    latest: int | None = None
    for iter_dir in sorted(run_dir.glob("iter-*")):
        if not iter_dir.is_dir():
            continue
        try:
            iteration = int(iter_dir.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if iteration > max_iterations:
            continue
        submission_path = _resolve_iteration_submission_artifact(iter_dir)
        metrics_path = _resolve_iteration_artifact(iter_dir, "metrics.json")
        if submission_path is None or metrics_path is None:
            continue
        if latest is None or iteration > latest:
            latest = iteration
    return latest


def _load_submit_retry_artifacts(
    *,
    run_dir: Path,
    iter_dir: Path,
    iteration: int,
    max_iterations: int,
    metric_direction: str,
    target_metric: str,
    require_submit_phase: bool,
) -> tuple[Path, Path, EvaluationResult] | None:
    if not require_submit_phase:
        return None

    marker_payload = _load_iteration_state_marker(iter_dir / _ITERATION_STATE_FILENAME)
    marker_pending = (
        bool(marker_payload.get("trained"))
        and bool(marker_payload.get("submit_allowed_by_gate"))
        and (not bool(marker_payload.get("submit_phase_finished")))
    )

    legacy_pending = False
    if not marker_pending:
        latest_iter = _latest_iteration_with_training_artifacts(run_dir=run_dir, max_iterations=max_iterations)
        run_state = _load_run_state(run_dir)
        if bool(run_state.get("submit_attempted")) and not bool(run_state.get("submit_ok")):
            legacy_pending = latest_iter == iteration
        elif not marker_payload and latest_iter == iteration:
            # Legacy runs may have submit failures without iteration_state/run_state.
            # If the latest iteration already has both artifacts, prefer submit-only resume.
            legacy_pending = True
    if not (marker_pending or legacy_pending):
        return None

    submission_path = _resolve_iteration_submission_artifact(iter_dir)
    metrics_path = _resolve_iteration_artifact(iter_dir, "metrics.json")
    if submission_path is None or metrics_path is None:
        return None
    evaluation = _load_kernel_metrics(metrics_path, metric_direction, target_metric)
    if evaluation is None:
        return None
    return submission_path, metrics_path, evaluation


def _maybe_restart_for_src_changes(*, config: AutopilotConfig, run_id: str, changed: list[str], stage: str) -> None:
    if config.dry_run:
        return
    if os.environ.get("KAGGLEBOT_NO_RESTART") == "1":
        return
    if not any(path.startswith("src/") for path in changed):
        return
    run_dir = config.paths.run_dir(run_id)
    state_path = run_dir / "autofix_restart.json"
    state = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
    stage_family = _restart_stage_family(stage)
    counts_by_stage: dict[str, int] = {}
    raw_counts = state.get("counts_by_stage")
    if isinstance(raw_counts, dict):
        for key, value in raw_counts.items():
            if not isinstance(key, str):
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                counts_by_stage[key] = parsed
    elif int(state.get("count", 0)) > 0:
        # Backward compatibility with legacy single-counter state files.
        legacy_stage = str(state.get("last_stage") or "").strip()
        legacy_family = _restart_stage_family(legacy_stage) if legacy_stage else "legacy"
        counts_by_stage[legacy_family] = int(state.get("count", 0))
    stage_count = int(counts_by_stage.get(stage_family, 0))
    if stage_count >= MAX_AUTOFIX_RESTARTS:
        print(f"[yellow]autofix[/yellow]: src changes detected in {stage}, restart limit reached")
        return
    counts_by_stage[stage_family] = stage_count + 1
    state["counts_by_stage"] = counts_by_stage
    state["count"] = sum(counts_by_stage.values())
    state["last_stage"] = stage
    state["last_stage_family"] = stage_family
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"[yellow]autofix[/yellow]: src changes detected in {stage}; restarting to reload code")
    os.environ["KAGGLEBOT_RESUME_RUN_ID"] = run_id
    os.environ["KAGGLEBOT_RESUME_SLUG"] = config.slug
    os.execv(sys.executable, [sys.executable, *sys.argv])


def _restart_stage_family(stage: str) -> str:
    normalized = str(stage or "").strip().lower()
    if not normalized:
        return "unknown"
    return normalized.split("_attempt_", 1)[0]


def _attempt_submit(
    *,
    config: AutopilotConfig,
    run_id: str,
    submission_path: Path,
    best_score: float | None,
    problem_types: list[str],
) -> dict[str, object] | None:
    if not config.submit or config.dry_run:
        return None
    run_dir = config.paths.run_dir(run_id)
    run_state = _load_run_state(run_dir)
    submit_code_fingerprint = _compute_submit_code_fingerprint(config)
    allow_force = config.force_submit or _env_truthy("KAGGLEBOT_FORCE_RESUBMIT")

    message = _submission_message(config, run_id, best_score, submission_path=submission_path)
    submission_service = SubmissionService(
        SubmissionConfig(
            slug=config.slug,
            data_dir=config.paths.data_dir,
            sample_submission_path=config.paths.sample_submission_path,
            submission_ledger_path=config.paths.submission_ledger_path,
            dry_run=config.dry_run,
            force_submit=config.force_submit,
            bypass_rate_limit=True,
        )
    )
    print(f"[cyan]submit[/cyan]: {config.slug}")
    submitted_at = datetime.now(UTC)

    try:
        prepared_submission_path = submission_service.validate_and_prepare_submission(submission_path)
    except SubmissionValidationError as exc:
        fingerprint = compute_error_fingerprint("", str(exc))
        return _abort_submit_for_run(
            config=config,
            run_id=run_id,
            problem_types=problem_types,
            submission_ref=submission_path,
            code_fingerprint=submit_code_fingerprint,
            fingerprint=fingerprint,
            error_kind="validation",
            reason="local_submission_validation_failed",
            message="Local submission validation failed; Kaggle CLI submit is skipped.",
            stdout_tail="",
            stderr_tail=str(exc),
            exit_code=SubmissionValidationError.exit_code,
        )

    try:
        rules_accepted = check_rules_accepted(config.slug, dry_run=config.dry_run)
    except KaggleCliError as exc:
        if _is_missing_kaggle_credentials_error(exc):
            return _abort_submit_for_run(
                config=config,
                run_id=run_id,
                problem_types=problem_types,
                submission_ref=prepared_submission_path,
                code_fingerprint=submit_code_fingerprint,
                fingerprint=compute_error_fingerprint(exc.stdout, exc.stderr or exc.output),
                error_kind="permanent",
                reason="kaggle_credentials_missing",
                message=("Kaggle credentials not configured. Set ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY."),
                stdout_tail=exc.stdout,
                stderr_tail=exc.stderr or exc.output,
                exit_code=exc.exit_code,
            )
        raise
    if not rules_accepted:
        return _abort_submit_for_run(
            config=config,
            run_id=run_id,
            problem_types=problem_types,
            submission_ref=prepared_submission_path,
            code_fingerprint=submit_code_fingerprint,
            fingerprint=compute_error_fingerprint("", "rules_not_accepted"),
            error_kind="permanent",
            reason="rules_not_accepted",
            message="Competition rules are not accepted; aborting submit stage for this run.",
            stdout_tail="",
            stderr_tail="rules_not_accepted",
            exit_code=RulesNotAcceptedError.exit_code,
        )

    constraints = _load_competition_rule_constraints(config.paths)
    notebook_submit_required = bool(constraints.notebook_submissions_only)
    notebook_fallback_activated = notebook_submit_required
    if notebook_submit_required:
        print("[yellow]submit mode[/yellow]: notebook-only competition detected; using notebook submit")

    if not notebook_submit_required:
        last_submission_path = str(run_state.get("last_submission_path") or "").strip()
        last_reason = str(run_state.get("last_reason") or "").strip().lower()
        last_code_fingerprint = str(run_state.get("last_submit_code_fingerprint") or "").strip()
        allow_same_path_retry_reasons = {
            "bad_request",
            "notebook_only_submission_required",
            "unclassified_submit_error",
        }
        if last_submission_path and Path(last_submission_path) == prepared_submission_path and not allow_force:
            if last_reason in allow_same_path_retry_reasons:
                print(
                    "[yellow]submit retry[/yellow]: previous submit failed with "
                    f"reason={last_reason}; retrying same artifact to allow notebook fallback."
                )
            elif last_code_fingerprint and last_code_fingerprint != submit_code_fingerprint:
                print(
                    "[yellow]submit retry[/yellow]: same artifact path but submit code changed; retrying in this run."
                )
            else:
                print("[yellow]submit skipped[/yellow]: same submission file already attempted in this run")
                known_fingerprint = str(
                    run_state.get("last_submit_fingerprint") or run_state.get("last_fingerprint") or ""
                )
                _append_submit_attempt(
                    run_dir=run_dir,
                    payload={
                        "run_id": run_id,
                        "sub_path": str(prepared_submission_path),
                        "sub_sha256": _sha256_or_none(prepared_submission_path),
                        "exit_code": None,
                        "ok": False,
                        "fingerprint": known_fingerprint,
                        "error_kind": "unknown",
                        "action_taken": "skip",
                        "reason": "same_submission_path_reused_in_run",
                        "stdout_tail": "",
                        "stderr_tail": "",
                    },
                )
                return None

    seen_fingerprints = set(_load_submit_fingerprints(run_dir))
    state_fingerprint = str(run_state.get("last_submit_fingerprint") or run_state.get("last_fingerprint") or "").strip()
    if state_fingerprint:
        seen_fingerprints.add(state_fingerprint)
    max_attempts = max(1, _SUBMIT_MAX_TRANSIENT_RETRIES)
    submission_result = None
    submission_reference = str(prepared_submission_path)
    submission_artifact_path: Path | None = prepared_submission_path
    for attempt in range(1, max_attempts + 1):
        try:
            if notebook_submit_required:
                notebook_result, notebook_ref, notebook_artifact_path = _submit_with_notebook_kernel(
                    config=config,
                    run_id=run_id,
                    submission_path=prepared_submission_path,
                    message=message,
                )
                submission_result = notebook_result
                submission_reference = notebook_ref
                submission_artifact_path = notebook_artifact_path
            else:
                submission_result = submission_service.submit_prepared(
                    prepared_path=prepared_submission_path,
                    message=message,
                    run_id=run_id,
                )
                submission_reference = str(submission_result.submission_path)
                submission_artifact_path = submission_result.submission_path
        except SubmissionCliError as exc:
            classification_stderr = exc.stderr or ""
            classification = classify_submit_error(exc.stdout, classification_stderr, exc.exit_code)
            if str(classification.get("reason") or "unclassified_submit_error") == "unclassified_submit_error" and (
                exc.output
            ):
                fallback_stderr = "\n".join(part for part in [classification_stderr, exc.output] if part)
                classification = classify_submit_error(exc.stdout, fallback_stderr, exc.exit_code)
                classification_stderr = fallback_stderr
            classification_kind = str(classification.get("kind") or "unknown")
            classification_reason = str(classification.get("reason") or "unclassified_submit_error")
            if (
                (not notebook_submit_required)
                and (not notebook_fallback_activated)
                and _should_use_notebook_submit_fallback(
                    reason=classification_reason,
                    stdout=exc.stdout,
                    stderr=exc.stderr,
                )
            ):
                notebook_submit_required = True
                notebook_fallback_activated = True
                print(
                    "[yellow]submit mode[/yellow]: file submit indicates notebook submit is required; "
                    "retrying via notebook submit automatically."
                )
                continue
            fingerprint = compute_error_fingerprint(exc.stdout, exc.stderr)
            if fingerprint in seen_fingerprints:
                if _consume_same_submit_fingerprint_retry_allowance(
                    run_dir=run_dir,
                    run_state=run_state,
                    fingerprint=fingerprint,
                    code_fingerprint=submit_code_fingerprint,
                ):
                    print(
                        "[yellow]submit retry[/yellow]: same fingerprint matched previous failures, "
                        "but code changed since last submit error; allowing one retry."
                    )
                else:
                    return _abort_submit_for_run(
                        config=config,
                        run_id=run_id,
                        problem_types=problem_types,
                        submission_ref=submission_reference,
                        submission_artifact_path=submission_artifact_path,
                        code_fingerprint=submit_code_fingerprint,
                        fingerprint=fingerprint,
                        error_kind=classification_kind,
                        reason="same_error_fingerprint_recurred",
                        message="Same submit error fingerprint recurred; aborting this run to prevent infinite loop.",
                        stdout_tail=exc.stdout,
                        stderr_tail=classification_stderr,
                        exit_code=exc.exit_code,
                    )
            seen_fingerprints.add(fingerprint)
            retry_after = classification.get("retry_after_seconds")
            retry_after_value = float(retry_after) if isinstance(retry_after, (int, float)) else 0.0
            if classification_kind == "transient" and attempt < max_attempts:
                wait_seconds = max(_compute_submit_backoff(attempt), retry_after_value)
                print(
                    "[yellow]submit retry[/yellow]: transient submit error "
                    f"(reason={classification_reason}, attempt={attempt}/{max_attempts}, wait={wait_seconds:.1f}s)"
                )
                _append_submit_attempt(
                    run_dir=run_dir,
                    payload={
                        "run_id": run_id,
                        "sub_path": submission_reference,
                        "sub_sha256": _sha256_or_none(submission_artifact_path),
                        "exit_code": exc.exit_code,
                        "ok": False,
                        "fingerprint": fingerprint,
                        "error_kind": "transient",
                        "action_taken": "retry",
                        "reason": classification_reason,
                        "stdout_tail": exc.stdout[-_SUBMIT_STDOUT_TAIL_CHARS:],
                        "stderr_tail": classification_stderr[-_SUBMIT_STDERR_TAIL_CHARS:],
                    },
                )
                _record_submit_reason_knowledge(
                    config=config,
                    run_id=run_id,
                    problem_types=problem_types,
                    submission_path=submission_artifact_path or prepared_submission_path,
                    error_kind="transient",
                    reason=classification_reason,
                    action_taken="retry",
                    fingerprint=fingerprint,
                    details=f"attempt={attempt}; wait={wait_seconds:.1f}s",
                )
                time.sleep(wait_seconds)
                continue
            print(
                "[red]submit aborted[/red]: "
                f"{classification_kind} submit error (reason={classification_reason}); no further retries in this run."
            )
            return _abort_submit_for_run(
                config=config,
                run_id=run_id,
                problem_types=problem_types,
                submission_ref=submission_reference,
                submission_artifact_path=submission_artifact_path,
                code_fingerprint=submit_code_fingerprint,
                fingerprint=fingerprint,
                error_kind=classification_kind,
                reason=classification_reason,
                message=(
                    "Submit failed and is not retryable in this run."
                    if classification_kind != "transient"
                    else "Transient submit error exceeded retry budget; aborting this run."
                ),
                stdout_tail=exc.stdout,
                stderr_tail=classification_stderr,
                exit_code=exc.exit_code,
            )
        except (DuplicateSubmissionError, SubmissionRateLimitError) as exc:
            fingerprint = compute_error_fingerprint("", str(exc))
            return _abort_submit_for_run(
                config=config,
                run_id=run_id,
                problem_types=problem_types,
                submission_ref=submission_reference,
                submission_artifact_path=submission_artifact_path,
                code_fingerprint=submit_code_fingerprint,
                fingerprint=fingerprint,
                error_kind="permanent",
                reason="local_submission_guardrail",
                message=f"Local submission guardrail blocked submit: {exc}",
                stdout_tail="",
                stderr_tail=str(exc),
                exit_code=getattr(exc, "exit_code", 1),
            )
        except KaggleCliError as exc:
            if _is_missing_kaggle_credentials_error(exc):
                return _abort_submit_for_run(
                    config=config,
                    run_id=run_id,
                    problem_types=problem_types,
                    submission_ref=submission_reference,
                    submission_artifact_path=submission_artifact_path,
                    code_fingerprint=submit_code_fingerprint,
                    fingerprint=compute_error_fingerprint(exc.stdout, exc.stderr or exc.output),
                    error_kind="permanent",
                    reason="kaggle_credentials_missing",
                    message=(
                        "Kaggle credentials not configured. Set ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY."
                    ),
                    stdout_tail=exc.stdout,
                    stderr_tail=exc.stderr or exc.output,
                    exit_code=exc.exit_code,
                )
            raise
        break

    if submission_result is None:
        raise SubmitAbortedError("Submit failed before producing a submission result.")
    submission_ref = submission_reference
    submission_for_submit_path = submission_artifact_path
    submit_exit_code = getattr(submission_result, "exit_code", getattr(submission_result, "returncode", None))
    stdout_tail = submission_result.stdout[-_SUBMIT_STDOUT_TAIL_CHARS:]
    stderr_tail = submission_result.stderr[-_SUBMIT_STDERR_TAIL_CHARS:]
    fingerprint = compute_error_fingerprint(submission_result.stdout, submission_result.stderr)
    _append_submit_attempt(
        run_dir=run_dir,
        payload={
            "run_id": run_id,
            "sub_path": submission_ref,
            "sub_sha256": _sha256_or_none(submission_for_submit_path),
            "exit_code": submit_exit_code,
            "ok": True,
            "fingerprint": fingerprint,
            "error_kind": "none",
            "action_taken": "submit",
            "reason": "submitted",
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        },
    )
    _save_run_state(
        run_dir,
        {
            "submit_attempted": True,
            "submit_ok": True,
            "last_submit_fingerprint": fingerprint,
            "last_fingerprint": fingerprint,
            "last_submit_code_fingerprint": submit_code_fingerprint,
            "last_error_kind": "none",
            "last_action": "submit",
            "last_reason": "submitted",
            "last_submission_path": submission_ref,
            "submit_attempts_count": int(_load_run_state(run_dir).get("submit_attempts_count", 0)) + 1,
        },
    )
    print("[green]submission recorded[/green]")
    try:
        outcome = _wait_for_submission_outcome(
            slug=config.slug,
            message=message,
            submitted_at=submitted_at,
        )
    except SubmissionOutcomePollingError as exc:
        detail = normalize_error_text(exc.detail or str(exc), max_chars=1200)
        return _abort_submit_for_run(
            config=config,
            run_id=run_id,
            problem_types=problem_types,
            submission_ref=submission_ref,
            submission_artifact_path=submission_for_submit_path,
            code_fingerprint=submit_code_fingerprint,
            fingerprint=compute_error_fingerprint("", detail or str(exc)),
            error_kind="transient",
            reason="submission_polling_error",
            message="Submission outcome polling failed; aborting submit stage for this run.",
            stdout_tail="",
            stderr_tail=detail or str(exc),
            exit_code=None,
        )

    if isinstance(outcome, dict):
        outcome_status = _normalize_submission_outcome_status(outcome.get("status"))
        outcome["status"] = outcome_status
        if outcome_status in _FAILED_SUBMISSION_OUTCOME_STATUSES:
            raw_payload = outcome.get("raw")
            if isinstance(raw_payload, dict):
                raw_detail = normalize_error_text(json.dumps(raw_payload, ensure_ascii=True), max_chars=1200)
            else:
                raw_detail = normalize_error_text(str(raw_payload or outcome_status), max_chars=1200)
            return _abort_submit_for_run(
                config=config,
                run_id=run_id,
                problem_types=problem_types,
                submission_ref=submission_ref,
                submission_artifact_path=submission_for_submit_path,
                code_fingerprint=submit_code_fingerprint,
                fingerprint=compute_error_fingerprint("", raw_detail or outcome_status),
                error_kind="validation",
                reason=f"submission_poll_status_{outcome_status}",
                message=(
                    f"Submission finished with error status '{outcome_status}' during polling; "
                    "aborting submit stage for this run."
                ),
                stdout_tail="",
                stderr_tail=raw_detail or outcome_status,
                exit_code=None,
            )

    if isinstance(outcome, dict) and outcome.get("score") is not None:
        print(
            "[cyan]submission result[/cyan]: "
            f"status={outcome.get('status') or 'unknown'} score={float(outcome['score']):.6f}"
        )
    else:
        print("[yellow]submission result[/yellow]: score not available yet; knowledge update skipped")
    return {
        "message": message,
        "submission_path": submission_ref,
        "submitted_at": submitted_at.isoformat(),
        "iteration": _infer_iteration_from_submission_path(submission_path),
        "outcome": outcome,
    }


def _submit_with_notebook_kernel(
    *,
    config: AutopilotConfig,
    run_id: str,
    submission_path: Path,
    message: str,
):
    """Execute a Kaggle notebook kernel and submit via kernel reference."""
    iteration = _infer_iteration_from_submission_path(submission_path) or 1
    iter_dir = config.paths.iter_dir(run_id, iteration)
    kaggle_user = resolve_kaggle_username(config.kaggle_username)
    enable_internet = str(config.internet or "on").strip().lower() == "on"
    try:
        kernel_result = run_kernel(
            slug=config.slug,
            run_id=run_id,
            iteration=iteration,
            base_dir=config.paths.base_dir.parent,
            kaggle_username=kaggle_user,
            kernel_name=config.kernel_name,
            accelerator=config.accelerator,
            enable_internet=enable_internet,
            score_source="cv",
            metric="unknown",
            direction="maximize",
            holdout_frac=0.2,
            cv_folds=5,
            seed=42,
            dry_run=config.dry_run,
            timeout_minutes=config.time_budget_min,
        )
    except Exception as exc:  # noqa: BLE001
        raise SubmissionCliError(
            "Notebook submission fallback failed while running Kaggle kernel.",
            command=[],
            output=str(exc),
            stdout="",
            stderr=str(exc),
        ) from exc

    submission_artifact_path: Path | None = None
    if kernel_result.submission_path:
        submission_artifact_path = _copy_submission_artifact_to_iteration_dir(
            source=kernel_result.submission_path,
            iter_dir=iter_dir,
        )
    kernel_ref = kernel_result.kernel_id
    output_file = (
        (submission_artifact_path or kernel_result.submission_path).name
        if (submission_artifact_path or kernel_result.submission_path)
        else "submission.csv"
    )
    version = _infer_kernel_submit_version_label(iter_dir / "logs") or "1"
    print(f"[cyan]submit notebook[/cyan]: {kernel_ref}")
    submit_result = run_kaggle_submit_kernel(
        slug=config.slug,
        kernel=kernel_ref,
        message=message,
        output_file=output_file,
        version=version,
        dry_run=config.dry_run,
    )
    return submit_result, f"kernel:{kernel_ref}", submission_artifact_path


def _should_use_notebook_submit_fallback(*, reason: str, stdout: str, stderr: str) -> bool:
    """Return True only when submit errors clearly indicate notebook-only submission."""
    normalized_reason = str(reason or "").strip().lower()
    if normalized_reason == "notebook_only_submission_required":
        return True
    if normalized_reason not in {"bad_request", "unclassified_submit_error", "unknown"}:
        return False
    detail = f"{stdout}\n{stderr}".lower()
    return any(hint in detail for hint in _NOTEBOOK_FALLBACK_HINTS)


def _infer_kernel_submit_version_label(logs_dir: Path | None) -> str | None:
    """Read pushed kernel version from kernel push logs for notebook submit."""
    if logs_dir is None or not logs_dir.exists():
        return None
    candidates = sorted(logs_dir.glob("kernel_push-*.txt"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        match = _KERNEL_PUSH_VERSION_RE.search(text)
        if match:
            version = str(match.group("version") or "").strip()
            if version:
                return version
    return None


def _abort_submit_for_run(
    *,
    config: AutopilotConfig,
    run_id: str,
    problem_types: list[str],
    submission_ref: str | Path,
    submission_artifact_path: Path | None = None,
    code_fingerprint: str | None = None,
    fingerprint: str,
    error_kind: str,
    reason: str,
    message: str,
    stdout_tail: str,
    stderr_tail: str,
    exit_code: int | None,
) -> None:
    run_dir = config.paths.run_dir(run_id)
    submission_ref_text = str(submission_ref)
    artifact_path: Path | None
    if submission_artifact_path is not None:
        artifact_path = submission_artifact_path
    elif isinstance(submission_ref, Path):
        artifact_path = submission_ref
    else:
        artifact_path = None
    _append_submit_attempt(
        run_dir=run_dir,
        payload={
            "run_id": run_id,
            "sub_path": submission_ref_text,
            "sub_sha256": _sha256_or_none(artifact_path),
            "exit_code": exit_code,
            "ok": False,
            "fingerprint": fingerprint,
            "code_fingerprint": code_fingerprint or "",
            "error_kind": error_kind,
            "action_taken": "abort",
            "reason": reason,
            "stdout_tail": stdout_tail[-_SUBMIT_STDOUT_TAIL_CHARS:],
            "stderr_tail": stderr_tail[-_SUBMIT_STDERR_TAIL_CHARS:],
        },
    )
    prior = _load_run_state(run_dir)
    prior_ok = bool(prior.get("submit_ok")) or _has_successful_submit_attempt(run_dir)
    _save_run_state(
        run_dir,
        {
            "submit_attempted": True,
            "submit_ok": prior_ok,
            "last_submit_fingerprint": fingerprint,
            "last_fingerprint": fingerprint,
            "last_submit_code_fingerprint": code_fingerprint or "",
            "last_error_kind": error_kind,
            "last_action": "abort",
            "last_reason": reason,
            "last_submission_path": submission_ref_text,
            "submit_attempts_count": int(prior.get("submit_attempts_count", 0)) + 1,
        },
    )
    knowledge_submission_path = artifact_path or Path(submission_ref_text)
    _record_submit_reason_knowledge(
        config=config,
        run_id=run_id,
        problem_types=problem_types,
        submission_path=knowledge_submission_path,
        error_kind=error_kind,
        reason=reason,
        action_taken="abort",
        fingerprint=fingerprint,
        details=message,
    )
    print(f"[red]submit aborted[/red]: {message}")
    raise SubmitAbortedError(message)


def _append_submit_attempt(*, run_dir: Path, payload: dict[str, object]) -> None:
    record = {
        "ts": datetime.now(UTC).isoformat(),
        **payload,
    }
    attempts_path = run_dir / "submit_attempts.jsonl"
    attempts_path.parent.mkdir(parents=True, exist_ok=True)
    with attempts_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _load_run_state(run_dir: Path) -> dict[str, object]:
    state_path = run_dir / "run_state.json"
    if not state_path.exists():
        attempted = _has_submit_attempt_records(run_dir)
        return {"submit_attempted": attempted, "submit_ok": False}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    if not payload.get("submit_attempted"):
        payload["submit_attempted"] = _has_submit_attempt_records(run_dir)
    if "last_submit_fingerprint" not in payload and payload.get("last_fingerprint"):
        payload["last_submit_fingerprint"] = payload.get("last_fingerprint")
    if "last_fingerprint" not in payload and payload.get("last_submit_fingerprint"):
        payload["last_fingerprint"] = payload.get("last_submit_fingerprint")
    if bool(payload.get("submit_attempted")) and not bool(payload.get("submit_ok")):
        if _has_successful_submit_attempt(run_dir):
            payload["submit_ok"] = True
    return payload


def _save_run_state(run_dir: Path, updates: dict[str, object]) -> None:
    state = _load_run_state(run_dir)
    state.update(updates)
    state["submit_attempted"] = bool(state.get("submit_attempted")) or _has_submit_attempt_records(run_dir)
    state["submit_ok"] = bool(state.get("submit_ok")) or _has_successful_submit_attempt(run_dir)
    state["updated_at"] = datetime.now(UTC).isoformat()
    state_path = run_dir / "run_state.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _has_submit_attempt_records(run_dir: Path) -> bool:
    attempts_path = run_dir / "submit_attempts.jsonl"
    if not attempts_path.exists():
        return False
    try:
        for line in attempts_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                return True
    except OSError:
        return False
    return False


def _has_successful_submit_attempt(run_dir: Path) -> bool:
    attempts_path = run_dir / "submit_attempts.jsonl"
    if not attempts_path.exists():
        return False
    try:
        lines = attempts_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and bool(payload.get("ok")):
            return True
    return False


def _count_successful_submit_attempts(run_dir: Path) -> int:
    """Count successful Kaggle submit actions recorded for this run."""
    attempts_path = run_dir / "submit_attempts.jsonl"
    if not attempts_path.exists():
        return 0
    try:
        lines = attempts_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    count = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if not bool(payload.get("ok")):
            continue
        action_taken = str(payload.get("action_taken") or "").strip().lower()
        if action_taken and action_taken != "submit":
            continue
        count += 1
    return count


def _load_submit_fingerprints(run_dir: Path) -> list[str]:
    fingerprints: list[str] = []
    attempts_path = run_dir / "submit_attempts.jsonl"
    if not attempts_path.exists():
        return fingerprints
    try:
        lines = attempts_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return fingerprints
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        fingerprint = str(row.get("fingerprint") or "").strip()
        if not fingerprint:
            continue
        fingerprints.append(fingerprint)
    return fingerprints


def _load_latest_submit_attempt(run_dir: Path) -> dict[str, object]:
    attempts_path = run_dir / "submit_attempts.jsonl"
    if not attempts_path.exists():
        return {}
    try:
        lines = attempts_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for raw in reversed(lines):
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _build_submit_autofix_context(run_dir: Path) -> str:
    state = _load_run_state(run_dir)
    latest = _load_latest_submit_attempt(run_dir)
    lines: list[str] = []
    state_keys = (
        "submit_attempted",
        "submit_ok",
        "last_error_kind",
        "last_reason",
        "last_action",
        "last_submit_fingerprint",
        "last_submission_path",
    )
    lines.append("run_state:")
    for key in state_keys:
        value = state.get(key)
        if value in (None, ""):
            continue
        lines.append(f"- {key}: {value}")

    if latest:
        lines.append("latest_submit_attempt:")
        for key in (
            "ts",
            "ok",
            "exit_code",
            "error_kind",
            "reason",
            "action_taken",
            "fingerprint",
            "sub_path",
        ):
            value = latest.get(key)
            if value in (None, ""):
                continue
            lines.append(f"- {key}: {value}")
        stdout_tail = normalize_error_text(str(latest.get("stdout_tail") or ""), max_chars=1200)
        stderr_tail = normalize_error_text(str(latest.get("stderr_tail") or ""), max_chars=1200)
        if stdout_tail:
            lines.append(f"- stdout_tail: {stdout_tail}")
        if stderr_tail:
            lines.append(f"- stderr_tail: {stderr_tail}")
    return "\n".join(lines).strip()


def _sha256_or_none(path: Path | None) -> str | None:
    """Return SHA256 for an existing file path, otherwise None."""
    if path is None:
        return None
    if not path.exists():
        return None
    try:
        return sha256_file(str(path))
    except OSError:
        return None


def _compute_submit_code_fingerprint(config: AutopilotConfig) -> str:
    """Compute a stable fingerprint of submit-relevant local code."""
    hasher = hashlib.sha256()
    root_specs = (
        ("src", Path(__file__).resolve().parent),
        ("kernel", config.paths.kernel_source_dir),
    )
    for label, root in root_specs:
        if not root.exists() or not root.is_dir():
            hasher.update(f"{label}:<missing>\n".encode())
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts:
                continue
            if path.suffix.lower() in {".pyc", ".pyo"}:
                continue
            try:
                rel = path.relative_to(root).as_posix()
            except ValueError:
                rel = path.name
            hasher.update(f"{label}:{rel}\n".encode())
            hasher.update((_sha256_or_none(path) or "missing").encode())
            hasher.update(b"\n")
    return hasher.hexdigest()


def _consume_same_submit_fingerprint_retry_allowance(
    *,
    run_dir: Path,
    run_state: dict[str, object],
    fingerprint: str,
    code_fingerprint: str,
) -> bool:
    """Allow one repeated-fingerprint retry after code changes since last submit error."""
    last_code_fingerprint = str(run_state.get("last_submit_code_fingerprint") or "").strip()
    prior_error_fingerprint = str(
        run_state.get("last_submit_fingerprint") or run_state.get("last_fingerprint") or ""
    ).strip()
    if not code_fingerprint:
        return False

    consumed_code_fingerprint = str(run_state.get("same_fp_allowance_code_fingerprint") or "").strip()
    consumed_error_fingerprint = str(run_state.get("same_fp_allowance_error_fingerprint") or "").strip()
    if consumed_code_fingerprint == code_fingerprint and consumed_error_fingerprint == fingerprint:
        return False

    # Backward compatibility for runs recorded before code_fingerprint tracking existed.
    # In that case we cannot compare "before vs after" code reliably, so allow exactly once.
    if not last_code_fingerprint:
        if not prior_error_fingerprint or prior_error_fingerprint != fingerprint:
            return False
        run_state["same_fp_allowance_code_fingerprint"] = code_fingerprint
        run_state["same_fp_allowance_error_fingerprint"] = fingerprint
        _save_run_state(
            run_dir,
            {
                "same_fp_allowance_code_fingerprint": code_fingerprint,
                "same_fp_allowance_error_fingerprint": fingerprint,
            },
        )
        return True

    if code_fingerprint == last_code_fingerprint:
        return False

    run_state["same_fp_allowance_code_fingerprint"] = code_fingerprint
    run_state["same_fp_allowance_error_fingerprint"] = fingerprint
    _save_run_state(
        run_dir,
        {
            "same_fp_allowance_code_fingerprint": code_fingerprint,
            "same_fp_allowance_error_fingerprint": fingerprint,
        },
    )
    return True


def _compute_submit_backoff(attempt: int) -> float:
    base = _SUBMIT_BACKOFF_BASE_SEC * (2 ** max(0, attempt - 1))
    jitter = random.uniform(0.0, 1.0)
    return base + jitter


def _env_flag(name: str, *, default: bool) -> bool:
    """Parse a boolean environment flag with an explicit default fallback."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _should_force_resubmit_after_submit_abort(run_dir: Path) -> bool:
    state = _load_run_state(run_dir)
    reason = str(state.get("last_reason") or "").strip().lower()
    if not reason:
        return False
    if reason in {"submission_polling_error", "submission_polling_timeout", "submission_polling_invalid_payload"}:
        return True
    return reason.startswith("submission_poll_status_")


def _is_submit_abort_autofixable(*, config: AutopilotConfig, run_id: str) -> bool:
    run_dir = config.paths.run_dir(run_id)
    state = _load_run_state(run_dir)
    kind = str(state.get("last_error_kind") or "").strip().lower()
    reason = str(state.get("last_reason") or "").strip().lower()
    if kind in {"validation", "transient", "unknown"}:
        return True
    if reason == "same_error_fingerprint_recurred":
        return True
    print(
        "[yellow]autofix skipped[/yellow]: submit abort is not safely auto-fixable "
        f"(kind={kind or 'unknown'}, reason={reason or 'unknown'})"
    )
    return False


def _is_non_autofixable_runtime_error(error: Exception) -> bool:
    text = str(error).strip().lower()
    if not text:
        return False
    return "requires kernel.py" in text or "kernel-first training" in text


def _record_submit_reason_knowledge(
    *,
    config: AutopilotConfig,
    run_id: str,
    problem_types: list[str],
    submission_path: Path,
    error_kind: str,
    reason: str,
    action_taken: str,
    fingerprint: str,
    details: str,
) -> None:
    iteration = _infer_iteration_from_submission_path(submission_path) or 1
    summary = normalize_error_text(details, max_chars=1200)
    message = f"submit_error kind={error_kind} reason={reason} fingerprint={fingerprint}"
    fix = f"submit_action={action_taken}; detail={summary}"
    try:
        record_error_fix_insight(
            knowledge_paths=config.knowledge_paths,
            slug=config.slug,
            run_id=run_id,
            iteration=iteration,
            problem_types=problem_types,
            error_message=message,
            fix_summary=fix,
            resolved=False,
            outcome_bucket="unknown",
            submission_score=None,
        )
    except Exception:  # noqa: BLE001
        # Knowledge recording must not block submit abort/retry control.
        return


def _is_missing_kaggle_credentials_error(exc: KaggleCliError) -> bool:
    text = (exc.message or "") + "\n" + (exc.output or "")
    lowered = text.lower()
    if "kaggle.json" in lowered and "could not find" in lowered:
        return True
    if "kaggle.json" in lowered and "environment method" in lowered:
        return True
    if "api.authenticate" in lowered and "kaggle.json" in lowered:
        return True
    return False


def _wait_for_submission_outcome(
    *,
    slug: str,
    message: str,
    submitted_at: datetime,
    max_attempts: int | None = _SUBMISSION_POLL_MAX_ATTEMPTS,
    poll_interval_sec: float = _SUBMISSION_POLL_INTERVAL_SEC,
) -> dict[str, object] | None:
    print(f"[cyan]submission polling[/cyan]: waiting for result (interval={poll_interval_sec:.0f}s)")
    service = SubmissionOutcomeService(
        fetch_rows=lambda current_slug: list_competition_submissions(current_slug, dry_run=False),
        max_attempts=max_attempts,
        poll_interval_sec=poll_interval_sec,
        max_fetch_errors=_SUBMISSION_POLL_MAX_FETCH_ERRORS,
    )
    return service.wait_for_outcome(
        slug=slug,
        message=message,
        submitted_at=submitted_at,
    )


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_submission_outcome_status(value: object) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "unknown"
    if "." in raw:
        prefix, _, suffix = raw.rpartition(".")
        if suffix and "status" in prefix:
            return suffix.strip()
    return raw


def _to_int(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return None


def _resolve_submission_rank_payload(
    *,
    slug: str,
    context_dir: Path,
    direction: str,
    outcome: dict[str, object],
    dry_run: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    rank = _to_int(outcome.get("rank"))
    total_teams = _to_int(outcome.get("total_teams"))
    rank_percentile = _to_float(outcome.get("rank_percentile"))
    rank_source = outcome.get("rank_source")

    if rank is not None:
        payload["rank"] = rank
    if total_teams is not None:
        payload["total_teams"] = total_teams
    if rank_percentile is not None:
        payload["rank_percentile"] = rank_percentile
    if isinstance(rank_source, str) and rank_source.strip():
        payload["rank_source"] = rank_source.strip()

    if rank is None or total_teams is None:
        score = _to_float(outcome.get("score"))
        if score is not None:
            try:
                estimate = leaderboard_rank_for_score(
                    slug=slug,
                    output_dir=context_dir,
                    score=score,
                    direction=direction,
                    dry_run=dry_run,
                )
            except Exception:  # noqa: BLE001
                estimate = {}
            est_rank = _to_int(estimate.get("rank"))
            est_total = _to_int(estimate.get("total_teams"))
            est_percentile = _to_float(estimate.get("rank_percentile"))
            if est_rank is not None:
                payload["estimated_rank"] = est_rank
            if est_total is not None:
                payload["estimated_total_teams"] = est_total
            if est_percentile is not None:
                payload["estimated_rank_percentile"] = est_percentile
            if est_rank is not None and isinstance(estimate.get("source"), str):
                payload["rank_estimate_source"] = "leaderboard_score_estimate"

    resolved_rank = _to_int(payload.get("rank"))
    resolved_total = _to_int(payload.get("total_teams"))
    if resolved_rank is not None and resolved_total is not None and resolved_total > 0:
        payload.setdefault("rank_percentile", resolved_rank / resolved_total)
    return payload


def _should_force_major_overhaul_by_rank(
    *,
    rank: int | None,
    total_teams: int | None,
    max_percentile: float,
    min_teams: int,
) -> bool:
    if rank is None or total_teams is None:
        return False
    if total_teams < max(1, min_teams):
        return False
    if rank <= 0:
        return False
    rank_percentile = rank / total_teams
    return rank_percentile > max_percentile


def _build_rank_force_reason(
    *,
    rank: int,
    total_teams: int,
    rank_percentile: float | None,
    max_percentile: float,
    min_teams: int,
    source: str | None,
) -> str:
    resolved_percentile = (rank / total_teams) if rank_percentile is None and total_teams > 0 else rank_percentile
    percentile_text = f"{(resolved_percentile or 0.0) * 100:.2f}%" if resolved_percentile is not None else "n/a"
    source_text = f" source={source}" if source else ""
    return (
        "Leaderboard rank indicates large headroom for improvement: "
        f"{rank}/{total_teams} (percentile={percentile_text}, threshold={max_percentile * 100:.2f}%, "
        f"min_teams={min_teams}).{source_text}"
    )


def _infer_iteration_from_submission_path(path: Path) -> int | None:
    try:
        name = path.parent.name
        if not name.startswith("iter-"):
            return None
        return int(name.split("-", 1)[1])
    except Exception:  # noqa: BLE001
        return None


def _record_submission_knowledge(
    *,
    config: AutopilotConfig,
    run_id: str,
    problem_types: list[str],
    pending_problem_insights: list[dict[str, object]],
    pending_error_fixes: list[dict[str, object]],
    submission_result: dict[str, object] | None,
    metric_direction: str,
    target_score: float | None,
    top1_score: float | None,
) -> None:
    if not submission_result:
        return
    outcome_payload = submission_result.get("outcome")
    if not isinstance(outcome_payload, dict):
        return
    online_score = _to_float(outcome_payload.get("score"))
    if online_score is None:
        return
    outcome_bucket = _classify_submission_outcome(
        score=online_score,
        direction=metric_direction,
        target_score=target_score,
        top1_score=top1_score,
    )
    submitted_iteration = submission_result.get("iteration")
    iteration_value = submitted_iteration if isinstance(submitted_iteration, int) else None
    if not pending_problem_insights:
        diagnostics_text = ""
        if iteration_value is not None:
            diagnostics_path = config.paths.iter_dir(run_id, iteration_value) / "diagnostics.md"
            if diagnostics_path.exists():
                diagnostics_text = diagnostics_path.read_text(encoding="utf-8", errors="ignore")
        pending_problem_insights.append(
            {
                "iteration": iteration_value or 1,
                "why_poor": diagnostics_text,
                "how_improved": f"Submitted iteration {iteration_value or 1} result after validation.",
                "delta_offline": None,
            }
        )
    for item in pending_problem_insights:
        try:
            iteration = int(item.get("iteration") or (iteration_value or 1))
        except (TypeError, ValueError):
            iteration = iteration_value or 1
        record_problem_type_insight(
            knowledge_paths=config.knowledge_paths,
            slug=config.slug,
            run_id=run_id,
            iteration=iteration,
            problem_types=problem_types,
            why_poor=str(item.get("why_poor") or ""),
            how_improved=str(item.get("how_improved") or ""),
            delta_offline=item.get("delta_offline") if isinstance(item.get("delta_offline"), (int, float)) else None,
            outcome_bucket=outcome_bucket,
            submission_score=online_score,
        )
    for item in pending_error_fixes:
        try:
            iteration = int(item.get("iteration") or (iteration_value or 1))
        except (TypeError, ValueError):
            iteration = iteration_value or 1
        record_error_fix_insight(
            knowledge_paths=config.knowledge_paths,
            slug=config.slug,
            run_id=run_id,
            iteration=iteration,
            problem_types=problem_types,
            error_message=str(item.get("error_message") or ""),
            fix_summary=str(item.get("fix_summary") or ""),
            resolved=bool(item.get("resolved", True)),
            outcome_bucket=outcome_bucket,
            submission_score=online_score,
        )


def _classify_submission_outcome(
    *,
    score: float,
    direction: str,
    target_score: float | None,
    top1_score: float | None,
) -> str:
    if target_score is not None and _meets_target(score, target_score, direction):
        return "good"
    if top1_score is not None:
        if direction == "minimize":
            gap = score - top1_score
        else:
            gap = top1_score - score
        scale = max(abs(top1_score), 1.0)
        if max(gap, 0.0) / scale <= 0.1:
            return "good"
    return "low"


def _submission_message(
    config: AutopilotConfig,
    run_id: str,
    best_score: float | None,
    *,
    submission_path: Path | None = None,
) -> str:
    if config.message:
        return config.message
    iteration = _infer_iteration_from_submission_path(submission_path) if submission_path is not None else None
    iteration_suffix = f" i={iteration}" if isinstance(iteration, int) else ""
    if best_score is None:
        return f"kb {run_id}{iteration_suffix}"
    return f"kb {run_id}{iteration_suffix} offline={best_score:.4f}"


def _normalize_metric_name(name: str | None) -> str:
    """Normalize a metric label for loose string comparison."""
    if not name:
        return ""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _canonical_metric_name_for_match(name: str | None) -> str:
    """Return a canonical normalized metric token for mismatch checks."""
    normalized = _normalize_metric_name(name)
    if not normalized:
        return ""
    canonical = _normalize_metric_name(canonical_metric(str(name)))
    metric = canonical or normalized
    alias_map = {
        "brier": "brierscore",
        "brierloss": "brierscore",
        "brierscoreloss": "brierscore",
    }
    return alias_map.get(metric, metric)


def _metrics_equivalent(left: str | None, right: str | None) -> bool:
    """Return True when two metric labels represent the same metric."""
    left_metric = _canonical_metric_name_for_match(left)
    right_metric = _canonical_metric_name_for_match(right)
    return bool(left_metric) and left_metric == right_metric


def _infer_metric_direction_for_mismatch(metric: str, fallback_direction: str) -> tuple[str, bool]:
    metric_name = canonical_metric(metric)
    if metric_name in {"rmse", "rmsle", "mae", "mape", "mse", "logloss"}:
        return "minimize", True
    if metric_name in {"auc", "accuracy", "f1", "precision", "recall", "average_precision", "r2", "r_squared"}:
        return "maximize", True
    metric_lower = metric.lower()
    if any(key in metric_lower for key in ["loss", "error"]):
        return "minimize", True
    if any(key in metric_lower for key in ["auc", "accuracy", "f1", "precision", "recall", "ap", "r2", "map"]):
        return "maximize", True
    return fallback_direction, False


def _meets_target(value: float, target: float, direction: str) -> bool:
    if direction == "minimize":
        return value <= target
    return value >= target


def _is_top1_tier(value: float, top1_score: float | None, direction: str) -> bool:
    if top1_score is None:
        return False
    if direction == "minimize":
        return value <= top1_score
    return value >= top1_score


def _is_confirmed_first_place(rank: int | None, source: str | None) -> bool:
    if rank != 1:
        return False
    if source is None:
        return True
    normalized = source.strip().lower()
    return normalized not in {"leaderboard_score_estimate", "score_estimate"}


def _classify_improvement_mode(value: float, top1_score: float | None, direction: str) -> tuple[str, float | None]:
    if top1_score is None:
        return "major_overhaul", None
    gap = top1_score - value if direction == "maximize" else value - top1_score
    if gap >= MAJOR_TOP1_GAP:
        return "major_overhaul", gap
    if gap >= MODERATE_TOP1_GAP:
        return "moderate_update", gap
    return "minor_tuning", gap


def _score_delta_vs_reference(current: float, reference: float, direction: str) -> float:
    """Return signed delta where positive means current is better than reference."""
    if direction == "minimize":
        return reference - current
    return current - reference


def _score_drop_vs_best(*, best_score: float | None, current_score: float, direction: str) -> float | None:
    if best_score is None:
        return None
    if direction == "maximize":
        return float(best_score) - float(current_score)
    return float(current_score) - float(best_score)


def _regression_drop_threshold(*, metric: str, direction: str) -> float:
    if direction == "maximize":
        metric_name = canonical_metric(metric)
        if metric_name in {"auc", "accuracy", "f1", "precision", "recall", "average_precision", "r2", "r_squared"}:
            return _REGRESSION_GUARD_ABS_DROP_PROB
    return _REGRESSION_GUARD_ABS_DROP_DEFAULT


def _is_severe_regression_vs_best(
    *, metric: str, direction: str, best_score: float | None, current_score: float
) -> bool:
    drop = _score_drop_vs_best(best_score=best_score, current_score=current_score, direction=direction)
    if drop is None:
        return False
    threshold = _regression_drop_threshold(metric=metric, direction=direction)
    return drop > threshold


def _is_conservative_feature_collapse(payload: dict[str, object] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    robust_subset_payload = payload.get("robust_subset_report")
    if not isinstance(robust_subset_payload, dict):
        return False
    selected_feature_count_raw = payload.get("selected_feature_count")
    selected_feature_count = (
        int(selected_feature_count_raw) if isinstance(selected_feature_count_raw, (int, float)) else None
    )
    selected_features = robust_subset_payload.get("selected_features")
    selected_subset_size = len(selected_features) if isinstance(selected_features, list) else None
    if selected_feature_count is not None and selected_feature_count <= _CONSERVATIVE_COLLAPSE_MAX_FEATURES:
        return True
    if selected_subset_size is not None and selected_subset_size <= _CONSERVATIVE_COLLAPSE_MAX_FEATURES:
        return True
    return False


def _best_kernel_snapshot_path(run_dir: Path) -> Path:
    return run_dir / _BEST_KERNEL_SNAPSHOT_FILENAME


def _capture_best_kernel_snapshot(*, paths: CompetitionPaths, run_dir: Path) -> bool:
    kernel_path = paths.kernel_source_dir / "kernel.py"
    if not kernel_path.exists():
        return False
    snapshot_path = _best_kernel_snapshot_path(run_dir)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(kernel_path, snapshot_path)
    except OSError:
        return False
    return True


def _ensure_best_kernel_snapshot(*, paths: CompetitionPaths, run_dir: Path) -> None:
    snapshot_path = _best_kernel_snapshot_path(run_dir)
    if snapshot_path.exists():
        return
    _capture_best_kernel_snapshot(paths=paths, run_dir=run_dir)


def _restore_best_kernel_snapshot(*, paths: CompetitionPaths, run_dir: Path) -> bool:
    snapshot_path = _best_kernel_snapshot_path(run_dir)
    kernel_path = paths.kernel_source_dir / "kernel.py"
    if not snapshot_path.exists():
        return False
    try:
        shutil.copy2(snapshot_path, kernel_path)
    except OSError:
        return False
    return True


def _effective_best_score_for_progress(
    *,
    prev_best: float | None,
    current_score: float,
    top1_score: float | None,
    direction: str,
) -> tuple[float | None, dict[str, object] | None]:
    """
    Clamp an implausible previous best into a top1-proximate band before no-improve checks.

    This avoids driving improvement-mode escalation from a stale outlier best score.
    """
    if prev_best is None or top1_score is None:
        return prev_best, None

    margin = _BEST_SCORE_OUTLIER_TOP1_ABS_MARGIN + (
        _BEST_SCORE_OUTLIER_TOP1_REL_MARGIN * max(abs(float(top1_score)), 1.0)
    )
    if direction == "maximize":
        cap = float(top1_score) + margin
        if float(prev_best) > cap and float(current_score) <= cap:
            return cap, {
                "applied": True,
                "reason": "clip_prev_best_above_top1_band",
                "prev_best": float(prev_best),
                "effective_best": cap,
                "top1_score": float(top1_score),
                "margin": float(margin),
            }
        return prev_best, None

    floor = float(top1_score) - margin
    if float(prev_best) < floor and float(current_score) >= floor:
        return floor, {
            "applied": True,
            "reason": "clip_prev_best_below_top1_band",
            "prev_best": float(prev_best),
            "effective_best": floor,
            "top1_score": float(top1_score),
            "margin": float(margin),
        }
    return prev_best, None


def _extract_score_from_text(text: str) -> float | None:
    for match in _CODE_SCORE_RE.finditer(text):
        value = _to_float(match.group(1))
        if value is None:
            continue
        if 0.0 <= value <= 1.0:
            return value
    return None


def _extract_code_reference_score_from_index(path: Path) -> tuple[float | None, str]:
    if not path.exists():
        return None, "missing_code_index"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, "invalid_code_index"
    if not isinstance(payload, dict):
        return None, "invalid_code_index"
    notebooks = payload.get("notebooks")
    if not isinstance(notebooks, list) or not notebooks:
        return None, "empty_code_index"
    required_id_raw = payload.get("required_reference_kernel_id")
    required_id = str(required_id_raw).strip() if isinstance(required_id_raw, str) else ""
    selected: dict[str, object] | None = None
    if required_id:
        for row in notebooks:
            if not isinstance(row, dict):
                continue
            if str(row.get("kernel_id") or "").strip() == required_id:
                selected = row
                break
    if selected is None:
        for row in notebooks:
            if isinstance(row, dict):
                selected = row
                break
    if selected is None:
        return None, "empty_code_index"

    kernel_id = str(selected.get("kernel_id") or "top_entry").strip() or "top_entry"
    score = _to_float(selected.get("score"))
    if score is not None:
        return score, f"code_index:{kernel_id}"

    title_score = _extract_score_from_text(str(selected.get("title") or ""))
    if title_score is not None:
        return title_score, f"code_title:{kernel_id}"
    return None, "code_index_without_numeric_score"


def _extract_code_reference_score_from_markdown(path: Path) -> tuple[float | None, str]:
    if not path.exists():
        return None, "missing_code_md"
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None, "missing_code_md"
    if not text.strip():
        return None, "empty_code_md"

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "notebook_score:" not in line.lower():
            continue
        score = _extract_score_from_text(line)
        if score is not None:
            return score, "code_md:notebook_score"

    lowered = text.lower()
    required_start = lowered.find("required reference notebook")
    if required_start >= 0:
        required_section = text[required_start : required_start + 2200]
        score = _extract_score_from_text(required_section)
        if score is not None:
            return score, "code_md:required_reference_section"

    top_snapshot = text[:3000]
    score = _extract_score_from_text(top_snapshot)
    if score is not None:
        return score, "code_md:top_snapshot"
    return None, "code_md_without_numeric_score"


def _extract_code_reference_score(paths: CompetitionPaths) -> tuple[float | None, str]:
    score, source = _extract_code_reference_score_from_index(paths.code_notebooks_index_path)
    if score is not None:
        return score, source
    score, source = _extract_code_reference_score_from_markdown(paths.code_md_path)
    if score is not None:
        return score, source
    if source and source != "code_md_without_numeric_score":
        return None, source
    return None, "unavailable"


def _load_required_reference_notebook(paths: CompetitionPaths) -> _CodeReferenceNotebook | None:
    index_path = paths.code_notebooks_index_path
    if not index_path.exists():
        return None
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    notebooks = payload.get("notebooks")
    if not isinstance(notebooks, list) or not notebooks:
        return None
    required_id_raw = payload.get("required_reference_kernel_id")
    required_id = str(required_id_raw).strip() if isinstance(required_id_raw, str) else ""
    selected: dict[str, object] | None = None
    if required_id:
        for row in notebooks:
            if not isinstance(row, dict):
                continue
            if str(row.get("kernel_id") or "").strip() == required_id:
                selected = row
                break
    if selected is None:
        for row in notebooks:
            if isinstance(row, dict):
                selected = row
                break
    if selected is None:
        return None
    kernel_id = str(selected.get("kernel_id") or "").strip()
    if not kernel_id:
        return None
    title = str(selected.get("title") or kernel_id).strip() or kernel_id
    source_file_raw = selected.get("source_file")
    source_file = str(source_file_raw).strip() if isinstance(source_file_raw, str) and source_file_raw else None
    local_dir_raw = selected.get("local_dir")
    local_dir = str(local_dir_raw).strip() if isinstance(local_dir_raw, str) and local_dir_raw else None
    summary_raw = selected.get("summary")
    summary = str(summary_raw).strip() if isinstance(summary_raw, str) and summary_raw else ""
    return _CodeReferenceNotebook(
        kernel_id=kernel_id,
        title=title,
        source_file=source_file,
        local_dir=local_dir,
        summary=summary,
    )


def _reference_requires_tabicl(reference: _CodeReferenceNotebook) -> bool:
    text = " ".join([reference.kernel_id, reference.title, reference.summary]).lower()
    return "tabicl" in text


def _code_reference_marker(reference: _CodeReferenceNotebook) -> str:
    return f"{_CODE_REFERENCE_IMPL_MARKER_PREFIX} {reference.kernel_id}"


def _validate_code_reference_implementation(*, kernel_path: Path, reference: _CodeReferenceNotebook) -> list[str]:
    if not kernel_path.exists():
        return ["kernel_source_missing"]
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    lowered = text.lower()
    marker = _code_reference_marker(reference).lower()
    issues: list[str] = []
    if marker not in lowered:
        issues.append("missing_code_reference_marker")
    if _reference_requires_tabicl(reference) and "tabicl" not in lowered:
        issues.append("missing_tabicl_implementation_path")
    return issues


def _build_code_reference_repair_prompt(
    *,
    base_prompt_text: str,
    reference: _CodeReferenceNotebook,
    issues: list[str],
    kernel_path: Path,
) -> str:
    issues_text = ", ".join(issues) if issues else "unknown"
    tabicl_required = _reference_requires_tabicl(reference)
    tabicl_line = (
        "- This reference appears to be TabICL-based. You MUST include a real TabICL path in kernel.py."
        if tabicl_required
        else "- TabICL path is optional for this reference notebook."
    )
    return (
        "# Codex Improvement Repair: Mandatory Code Reference Implementation\n\n"
        "The previous change did not satisfy mandatory code-reference implementation requirements.\n\n"
        f"- Failed checks: {issues_text}\n"
        f"- Required notebook: {reference.kernel_id} ({reference.title})\n"
        f"- Kernel path: {kernel_path}\n"
        f"- Required marker: `{_code_reference_marker(reference)}`\n"
        f"{tabicl_line}\n\n"
        "Make minimal edits to kernel.py so all checks pass.\n"
        "Do not weaken the model by collapsing to tiny conservative feature subsets that reduce offline score.\n\n"
        "## Original Improvement Context\n\n"
        f"{base_prompt_text}\n"
    )


def _update_best_score(best: float | None, current: float, direction: str, min_improvement: float) -> bool:
    """Check if current score represents an improvement over best score.

    Uses a small epsilon (1e-9) to handle floating point precision issues.
    """
    if best is None:
        return True

    eps = 1e-9  # Small tolerance for floating point comparison
    if direction == "minimize":
        improvement = best - current
        return improvement >= (min_improvement - eps)
    else:  # maximize
        improvement = current - best
        return improvement >= (min_improvement - eps)
