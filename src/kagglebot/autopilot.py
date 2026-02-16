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
import sys
import time
import traceback
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
from kagglebot.kernel_runner import _collect_log_tail, resolve_kaggle_username, run_kernel, run_kernel_local
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
from kagglebot.solver.metrics import canonical_metric, infer_direction
from kagglebot.submission.guard import (
    classify_submit_error,
    compute_error_fingerprint,
    normalize_error_text,
)
from kagglebot.submission.outcome_service import SubmissionOutcomeService
from kagglebot.submission_service import SubmissionConfig, SubmissionService
from kagglebot.types import PlanConfig


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


MAX_KERNEL_FIX_ATTEMPTS: int | None = None
MAX_SAME_KERNEL_ERROR_REPEATS = 2
MAX_KERNEL_CAPACITY_RETRIES = 3
KERNEL_CAPACITY_RETRY_SLEEP = 30.0
MAX_KERNEL_CAPACITY_REPEAT = 6
MAX_KERNEL_REGISTRATION_RETRIES = 2
KERNEL_REGISTRATION_RETRY_SLEEP = 15.0
MAX_AUTOFIX_ATTEMPTS = 2
MAX_AUTOFIX_RESTARTS = 1
MAJOR_TOP1_GAP = 0.03
MODERATE_TOP1_GAP = 0.01
_ERROR_FIX_CODEX_MODEL = "gpt-5.3-codex"
_ERROR_FIX_REASONING_EFFORT = "extra_high"
_ERROR_STRATEGY_MODEL = "gpt-5.2"
_ERROR_STRATEGY_REASONING_EFFORT = "extra_high"
_SUBMISSION_POLL_MAX_ATTEMPTS: int | None = None
_SUBMISSION_POLL_INTERVAL_SEC = 30.0
_SUBMIT_MAX_TRANSIENT_RETRIES = 3
_SUBMIT_BACKOFF_BASE_SEC = 2.0
_SUBMIT_STDERR_TAIL_CHARS = 1200
_SUBMIT_STDOUT_TAIL_CHARS = 1200
_ITERATION_STATE_FILENAME = "iteration_state.json"
_LEGACY_SUBMIT_PHASE_COMPLETE_ACTIONS = frozenset({"submit"})
_DEFAULT_EVAL_SEEDS = [42, 2024, 777]
_DEFAULT_EVAL_REPEATS = 2
_EVAL_REPEAT_SEED_OFFSET = 1009
_DEFAULT_FORCE_MAJOR_RANK_MAX_PERCENTILE = 0.35
_DEFAULT_FORCE_MAJOR_RANK_MIN_TEAMS = 200


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
    session = AutopilotSession(config=config, run_id=run_id, resume_run=resume_run)
    attempt = 0
    submit_force_override = False
    try:
        while True:
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
                os.environ["KAGGLEBOT_FORCE_RESUBMIT"] = "1"
                submit_force_override = True
                _run_autofix(config=config, run_id=run_id, attempt=attempt, error=exc)
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
    top1_info = leaderboard_top1(config.slug, config.paths.context_dir, dry_run=config.dry_run)
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

    _write_plan(config.paths, _resolved_plan(resolved))
    run_payload = _build_run_payload(
        run_id=run_id,
        config=config,
        resolved=resolved,
        status="running",
    )
    (run_dir / "run.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")

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
    score_source = str(resolved["score_source"] or "auto")
    max_total_min_raw = resolved.get("max_total_min")
    max_total_min = float(max_total_min_raw) if isinstance(max_total_min_raw, (int, float)) else None
    kernel_name = resolved["kernel_name"]
    enable_internet = str(resolved["internet"]) == "on"
    submission_gate = str(resolved.get("submission_gate") or "always")
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
    resumed_best_readiness = _resume_best_readiness_score(
        run_dir=config.paths.run_dir(run_id),
        direction=metric_direction,
        max_iterations=max_iterations,
    )
    if resumed_best_readiness is not None:
        best_score = resumed_best_readiness
    if start_iteration > 1:
        print(f"[yellow]resume[/yellow]: found completed iterations; resuming at {start_iteration}/{max_iterations}")
    loop_started_at = time.monotonic()

    try:
        for iteration in range(start_iteration, max_iterations + 1):
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
            evaluation = None
            evaluation_by_source: dict[str, EvaluationResult] = {}
            model_summary = {}
            accelerator_used = config.accelerator

            if config.compute.startswith("kaggle_"):
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
                            timeout_minutes=None,
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
                        _record_kernel_error(
                            logs_dir=logs_dir,
                            attempt=kernel_attempts,
                            error_text=error_text,
                            error_fingerprints=error_fingerprints,
                            output_dir=output_dir,
                        )
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
                if kernel_result.submission_path:
                    submission_path.write_bytes(kernel_result.submission_path.read_bytes())
                if kernel_result.metrics_path and kernel_result.metrics_path.exists():
                    evaluation = _load_kernel_metrics(kernel_result.metrics_path, metric_direction, target_metric)
                if evaluation is None:
                    raise KernelFailedError(
                        "Kernel metrics missing expected score; ensure metrics.json includes a numeric metric value."
                    )
            else:
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
                            timeout_minutes=config.time_budget_min,
                            strict_accelerator=config.strict_accelerator,
                        )
                        break
                    except Exception as exc:  # noqa: BLE001
                        kernel_attempts += 1
                        error_text = _format_kernel_error(exc)
                        _record_kernel_error(
                            logs_dir=logs_dir,
                            attempt=kernel_attempts,
                            error_text=error_text,
                            error_fingerprints=error_fingerprints,
                            output_dir=output_dir,
                        )
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

                if kernel_result.submission_path:
                    submission_path.write_bytes(kernel_result.submission_path.read_bytes())
                if kernel_result.metrics_path and kernel_result.metrics_path.exists():
                    evaluation = _load_kernel_metrics(kernel_result.metrics_path, metric_direction, target_metric)
                if evaluation is None:
                    raise KernelFailedError(
                        "Local kernel metrics missing expected score; "
                        "ensure metrics.json includes a numeric metric value."
                    )

            if evaluation is None:
                raise RuntimeError("No evaluation metrics produced.")
            if evaluation.metric and target_metric:
                normalized_eval = _normalize_metric_name(evaluation.metric)
                normalized_target = _normalize_metric_name(target_metric)
                if normalized_eval and normalized_eval != normalized_target:
                    corrected_direction, confident = _infer_metric_direction_for_mismatch(
                        evaluation.metric,
                        metric_direction,
                    )
                    if corrected_direction != metric_direction or evaluation.metric != target_metric:
                        confidence_text = "high" if confident else "fallback"
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
            decision_score = readiness_score
            decision_source = "readiness"
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

            allow_submit = _should_attempt_submit_for_readiness(
                gate=submission_gate,
                readiness_score=readiness_score,
                readiness_target=readiness_target,
                direction=metric_direction,
                iteration=iteration,
                max_iterations=max_iterations,
            )
            submission_result: dict[str, object] | None = None
            submit_phase_state = "disabled"
            if config.submit and allow_submit:
                try:
                    submission_result = submission_phase.attempt(
                        submission_path=submission_path,
                        best_score=best_score if best_score is not None else readiness_score,
                    )
                except SubmitAbortedError:
                    run_payload["status"] = "submit_failed"
                    (run_dir / "run.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
                    raise
                if submission_result:
                    submit_phase_state = "submitted"
                    submitted = True
                    last_submission_result = submission_result
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
            elif config.submit:
                submit_phase_state = "skipped_gate"
                print(
                    "[yellow]submit gate[/yellow]: "
                    f"gate={submission_gate} readiness={readiness_score:.6f} target={readiness_target:.6f} -> skipped"
                )

            met_target = _meets_target(readiness_score, readiness_target, metric_direction)
            top1_tier = _is_top1_tier(readiness_score, top1_score, metric_direction)
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
            force_major_overhaul_next = noise_forced_major_overhaul or rank_forced_major_overhaul
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
                "offline_readiness": top1_tier,
                "submission_score": top1_tier_by_submission,
            }
            metrics_path = iter_dir / "metrics.json"
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
            )
            (iter_dir / "diagnostics.md").write_text(diagnostics, encoding="utf-8")

            submit_phase_required = config.submit and not config.dry_run
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
            delta_offline = iteration_phase.delta_from_best(prev_best, readiness_score)
            improved = iteration_phase.should_update_best(best_score, readiness_score, stop_min_delta)
            if improved:
                best_score = readiness_score
                best_submission = submission_path
                no_improve_streak = 0
            else:
                no_improve_streak += 1

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
                    f"readiness_score did not improve by >= {stop_min_delta:.6f} "
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
            )
    except KeyboardInterrupt:
        run_payload["status"] = "interrupted"
        (run_dir / "run.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
        print("[yellow]run interrupted[/yellow]")
        return

    allow_final_submit = _should_attempt_submit_for_readiness(
        gate=submission_gate,
        readiness_score=best_score,
        readiness_target=readiness_target,
        direction=metric_direction,
        iteration=max_iterations,
        max_iterations=max_iterations,
    )
    if config.submit and not submitted and best_submission is not None and allow_final_submit:
        try:
            fallback_result = submission_phase.attempt(
                submission_path=best_submission,
                best_score=best_score,
            )
        except SubmitAbortedError:
            run_payload["status"] = "submit_failed"
            (run_dir / "run.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
            raise
        if fallback_result:
            submitted = True
            last_submission_result = fallback_result
    elif config.submit and not submitted and best_submission is not None and not allow_final_submit:
        print(
            "[yellow]submit gate[/yellow]: "
            f"final submit skipped by gate={submission_gate} "
            f"(best_readiness={best_score if best_score is not None else 'n/a'}, target={readiness_target:.6f})"
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

    target_metric = choose(config.target_metric, plan.target_metric, spec_metric)
    target_score = choose(config.target_score, plan.target_score, spec_readiness_target)
    target_direction = choose(config.target_direction, plan.target_direction, spec_direction or "auto")
    score_source = str(choose(config.score_source, plan.score_source, "auto") or "auto")
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
    time_budget_min = choose(config.time_budget_min, plan.time_budget_min, None)
    kernel_name = choose(config.kernel_name, plan.kernel_name, None)
    internet = choose(config.internet, plan.internet, "on")
    if internet in (None, "auto"):
        internet = "on"
    default_max_iterations = 3
    if config.max_iterations is None:
        max_iterations = default_max_iterations
        if plan.max_iterations not in (None, default_max_iterations):
            print(
                f"[yellow]note[/yellow]: plan max_iterations={plan.max_iterations} ignored; "
                f"using default {default_max_iterations}. "
                "Use --max-iterations to override."
            )
    else:
        max_iterations = config.max_iterations
    max_total_min = choose(config.max_total_min, plan.max_total_min, None)
    patience = choose(config.patience, plan.patience, 2)
    min_improvement = choose(config.min_improvement, plan.min_improvement, 0.0)
    submit_policy = str(choose(None, plan.submit_policy, "always") or "always")
    policy_submission_gate = _submission_gate_for_policy(submit_policy)
    submission_gate = choose(None, plan.submission_gate, spec_submission_gate or policy_submission_gate)
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
        score_source=str(resolved.get("score_source") or "auto"),
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
    result = run_command(args)
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


def _load_kernel_metrics(metrics_path: Path, direction: str, target_metric: str | None) -> EvaluationResult | None:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    from kagglebot.solver.evaluate import EvaluationResult

    metric_name, value = _extract_kernel_metric(payload, target_metric)
    if value is None:
        return None
    payload_direction_raw = payload.get("direction")
    payload_direction = str(payload_direction_raw).strip().lower() if payload_direction_raw is not None else ""
    resolved_direction = direction
    if payload_direction in {"minimize", "maximize"}:
        resolved_direction = payload_direction

    std = payload.get("offline_std")
    if std is None:
        std = payload.get("std")
    std_value = float(std) if isinstance(std, (int, float)) else None

    fold_scores_raw = payload.get("fold_scores")
    fold_scores: list[float] | None = None
    if isinstance(fold_scores_raw, list):
        parsed_fold_scores = [float(item) for item in fold_scores_raw if isinstance(item, (int, float))]
        if parsed_fold_scores:
            fold_scores = parsed_fold_scores
            if std_value is None and len(parsed_fold_scores) > 1:
                std_value = float(np.std(parsed_fold_scores, ddof=1))

    score_source = payload.get("score_source", "holdout")
    if score_source == "holdout":
        for key in payload.keys():
            if isinstance(key, str) and key.lower().startswith("oof_"):
                score_source = "cv"
                break

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


def _extract_kernel_metric(payload: dict[str, object], target_metric: str | None) -> tuple[str | None, float | None]:
    def is_number(value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def normalize(text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalnum())

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
        if isinstance(selection, str) and selection in values and is_number(values[selection]):
            return float(values[selection])
        for key in ("selected", "average", "stacked", "best", "val", "oof", "score"):
            if key in values and is_number(values[key]):
                return float(values[key])
        numeric = [float(v) for v in values.values() if is_number(v)]
        if not numeric:
            return None
        return min(numeric) if prefers_lower(metric_key) else max(numeric)

    if is_number(payload.get("offline_value")):
        return (str(payload.get("metric") or target_metric or "unknown"), float(payload["offline_value"]))
    if is_number(payload.get("value")):
        return (str(payload.get("metric") or target_metric or "unknown"), float(payload["value"]))
    if is_number(payload.get("score")):
        return (str(payload.get("metric") or target_metric or "unknown"), float(payload["score"]))

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
                if not is_number(val):
                    continue
                normalized_key = normalize(str(key))
                normalized_base = normalize(strip_prefixes(str(key)))
                if normalized_key in wanted or normalized_base in wanted:
                    return (str(target_metric), float(val))

    for key, val in payload.items():
        if not is_number(val):
            continue
        normalized_key = normalize(str(key))
        normalized_base = normalize(strip_prefixes(str(key)))
        for metric_name, values in aliases.items():
            normalized_aliases = {normalize(v) for v in values}
            normalized_aliases.add(normalize(metric_name))
            if normalized_key in normalized_aliases or normalized_base in normalized_aliases:
                return (metric_name, float(val))

    return (str(target_metric) if target_metric else None, None)


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
    drift_train_x = data.train[data.feature_columns].copy() if data.feature_columns else None
    drift_test_x = data.test[data.feature_columns].copy() if data.feature_columns else None
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


def _should_attempt_submit_for_readiness(
    *,
    gate: str,
    readiness_score: float | None,
    readiness_target: float,
    direction: str,
    iteration: int,
    max_iterations: int,
) -> bool:
    normalized = str(gate or "always").strip().lower()
    if normalized in {"always", "each_iteration"}:
        return True
    if normalized in {"final_only", "at_final"}:
        return iteration >= max_iterations
    if readiness_score is None:
        return iteration >= max_iterations
    met_target = _meets_target(readiness_score, readiness_target, direction)
    if normalized in {"readiness_only", "readiness_target", "on_target_only"}:
        return met_target
    if normalized in {"readiness_or_final", "target_or_final"}:
        return met_target or iteration >= max_iterations
    return met_target or iteration >= max_iterations


def _submission_gate_for_policy(policy: str | None) -> str:
    normalized = str(policy or "").strip().lower()
    if normalized in {"always", "each_iteration"}:
        return "always"
    if normalized in {"final_only", "at_final"}:
        return "final_only"
    if normalized in {"readiness_only", "readiness_target"}:
        return "readiness_only"
    if normalized in {"on_target_only", "target_or_final", "readiness_or_final"}:
        return "readiness_or_final"
    return "always"


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
        kernel_main=str(kernel_main_path),
    )
    if forced_improvement_reason:
        base_prompt_text += (
            "\n\nForced improvement mode policy is active.\n"
            f"Reason: {forced_improvement_reason}\n"
            "Do not propose minor_tuning; make a major_overhaul plan.\n"
        )
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
    allowed_prefixes = [config.paths.kernel_source_dir]
    guard_snapshot = _backup_guarded_files(config.paths.repo_root, allowed_prefixes)
    before = _snapshot_tree(config.paths.repo_root)
    result = run_codex(
        prompt_path,
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
        stage=f"improve_iteration_{iteration}",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )
    response_text = _read_agent_response(result.last_message_path)
    _print_agent_response(result.last_message_path, response_text)
    if result.returncode != 0:
        raise RuntimeError("Codex improvement failed.")

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


def _run_kernel_fix(
    *,
    config: AutopilotConfig,
    run_id: str,
    iteration: int,
    iter_dir: Path,
    error_message: str,
    attempt: int,
    pending_error_fixes: list[dict[str, object]] | None = None,
) -> None:
    prompt_template = config.paths.codex_kernel_fix_template.read_text(encoding="utf-8")
    agent_dir = iter_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = agent_dir / "kernel_fix_prompt.md"
    missing_module = _extract_missing_module(error_message)
    blocked_modules = _load_blocked_modules(config.paths.context_dir)
    if missing_module:
        blocked_modules = _record_blocked_module(config.paths.context_dir, missing_module)
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
            "Avoid this package or guard it with a fallback; prefer Kaggle-default libraries.\n\n" + prompt_text
        )

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

    prompt_text = f"Kernel fix attempt: {attempt}\n\n{prompt_text}"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    attempt_path = agent_dir / f"kernel_fix_prompt-{attempt:02d}.md"
    attempt_path.write_text(prompt_text, encoding="utf-8")
    _print_agent_prompt(prompt_path, prompt_text)

    allowed_prefixes = [
        config.paths.repo_root / "src",
        config.paths.repo_root / "docs",
        config.paths.repo_root / "tests",
        config.paths.kernel_source_dir,
        config.paths.context_dir,
        config.paths.runs_dir,
        config.paths.prompts_dir,
    ]
    guard_snapshot = _backup_guarded_files(config.paths.repo_root, allowed_prefixes)
    before = _snapshot_tree(config.paths.repo_root)

    print("[cyan]kernel fix[/cyan]: running codex fixer")
    result = run_codex(
        prompt_path,
        agent_dir,
        dry_run=config.dry_run,
        heartbeat_label="fixing error",
        model=_ERROR_FIX_CODEX_MODEL,
        reasoning_effort=_ERROR_FIX_REASONING_EFFORT,
    )
    after = _snapshot_tree(config.paths.repo_root)
    changed = _diff_snapshots(before, after)
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
    if result.returncode != 0:
        raise RuntimeError("Codex kernel-fix step failed.")
    _run_verify(config.verify_cmd, dry_run=config.dry_run)
    if pending_error_fixes is not None:
        pending_error_fixes.append(
            {
                "iteration": iteration,
                "error_message": error_message,
                "fix_summary": strategy_text or response_text,
                "resolved": True,
            }
        )


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

    if not submit_autofix and _maybe_write_column_fill(config, error_text):
        note_path = autofix_dir / "note.txt"
        note = (
            "autofix_note: column_fill.json created for missing column error.\n"
            "autofix will retry without modifying kernel sources.\n"
        )
        note_path.write_text(note, encoding="utf-8")
        print("[yellow]autofix[/yellow]: wrote column_fill.json; retrying without kernel edits")
        return

    if not submit_autofix and _maybe_write_object_coerce(config, error_text):
        note_path = autofix_dir / "note.txt"
        note = (
            "autofix_note: object_coerce.json created for numpy.object_ conversion error.\n"
            "autofix will retry without modifying kernel sources.\n"
        )
        note_path.write_text(note, encoding="utf-8")
        print("[yellow]autofix[/yellow]: wrote object_coerce.json; retrying without kernel edits")
        return

    if not submit_autofix and _maybe_write_device_coerce(config, error_text):
        note_path = autofix_dir / "note.txt"
        note = (
            "autofix_note: device_coerce.json created for torch device mismatch error.\n"
            "autofix will retry without modifying kernel sources.\n"
        )
        note_path.write_text(note, encoding="utf-8")
        print("[yellow]autofix[/yellow]: wrote device_coerce.json; retrying without kernel edits")
        return

    if not submit_autofix and _maybe_write_column_map(config, error_text):
        note_path = autofix_dir / "note.txt"
        note = (
            "autofix_note: column_map.json created for missing column error.\n"
            "autofix will retry without modifying kernel sources.\n"
        )
        note_path.write_text(note, encoding="utf-8")
        print("[yellow]autofix[/yellow]: wrote column_map.json; retrying without kernel edits")
        return

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
    if submit_autofix:
        print(
            f"[cyan]submit autofix[/cyan]: strategy={_ERROR_STRATEGY_MODEL}({_ERROR_STRATEGY_REASONING_EFFORT}) "
            f"-> fixer={_ERROR_FIX_CODEX_MODEL}({_ERROR_FIX_REASONING_EFFORT})"
        )
    strategy_prompt = _build_error_strategy_prompt(
        stage="submit_autofix" if submit_autofix else "autofix",
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
        stage_label="submit autofix" if submit_autofix else "autofix",
    )
    if strategy_text:
        prompt_text += (
            "\n\n## GPT 5.2 Extra-High Error-Fix Strategy\n"
            "Use the strategy below as guidance, then apply minimal targeted edits.\n\n"
            f"{strategy_text}\n"
        )
    prompt_path = autofix_dir / "prompt.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    _print_agent_prompt(prompt_path, prompt_text)

    guard_snapshot = _backup_guarded_files(config.paths.repo_root, allowed_prefixes)
    before = _snapshot_tree(config.paths.repo_root)
    result = run_codex(
        prompt_path,
        autofix_dir,
        dry_run=config.dry_run,
        heartbeat_label="fixing error",
        model=_ERROR_FIX_CODEX_MODEL,
        reasoning_effort=_ERROR_FIX_REASONING_EFFORT,
    )
    after = _snapshot_tree(config.paths.repo_root)
    changed = _diff_snapshots(before, after)
    _enforce_allowlist_changes(
        root=config.paths.repo_root,
        before=before,
        after=after,
        allowed_prefixes=allowed_prefixes,
        stage=f"autofix_attempt_{attempt}",
        guard_snapshot=guard_snapshot,
        auto_repair=True,
    )
    _maybe_restart_for_src_changes(
        config=config,
        run_id=run_id,
        changed=changed,
        stage=f"autofix_attempt_{attempt}",
    )
    response_text = _read_agent_response(result.last_message_path)
    _print_agent_response(result.last_message_path, response_text)
    if result.returncode != 0:
        raise RuntimeError("Codex autofix step failed.")
    _run_verify(config.verify_cmd, dry_run=config.dry_run)


def _autofix_allowed_prefixes(config: AutopilotConfig) -> list[Path]:
    # Keep src writable during autofix so runtime/framework issues in core code can be repaired.
    # Do not grant broad competition-root/kernels write access; fixes must target authoritative
    # sources (src/kernel/context/prompts) rather than generated staged artifacts.
    candidates = [
        config.paths.repo_root / "src",
        config.paths.repo_root / "docs",
        config.paths.repo_root / "tests",
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
3) Do not touch datasets or credentials. Do not add new dependencies without justification.
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
)


def _maybe_write_column_fill(config: AutopilotConfig, error_text: str) -> bool:
    match = _MISSING_COLUMNS_RE.search(error_text or "")
    if not match:
        return False
    missing_columns = _parse_missing_columns(match.group(1))
    if not missing_columns:
        return False
    context_dir = config.paths.context_dir
    fill_path = context_dir / _COLUMN_FILL_FILENAME
    if fill_path.exists():
        return False
    file_match = _MISSING_COLUMNS_FILE_RE.search(error_text or "")
    file_name = file_match.group(1) if file_match else None
    payload = {
        "source": "autofix",
        "created_at": datetime.now(UTC).isoformat(),
        "files": {file_name: missing_columns} if file_name else {},
        "missing_columns": [] if file_name else missing_columns,
    }
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


def _record_blocked_module(context_dir: Path, module: str) -> list[str]:
    context_dir.mkdir(parents=True, exist_ok=True)
    existing = _load_blocked_modules(context_dir)
    if module not in existing:
        existing.append(module)
        path = context_dir / _BLOCKED_MODULES_FILENAME
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
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
        submission_path = iter_dir / "submission.csv"
        metrics_path = iter_dir / "metrics.json"
        if not submission_path.exists() and not metrics_path.exists():
            continue
        if submission_path.exists() and not metrics_path.exists():
            print(
                "[yellow]resume[/yellow]: "
                f"iter-{iteration} has submission.csv but no metrics.json; treating as incomplete."
            )
            continue
        if metrics_path.exists() and not submission_path.exists():
            print(
                "[yellow]resume[/yellow]: "
                f"iter-{iteration} has metrics.json but no submission.csv; treating as incomplete."
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
    count = int(state.get("count", 0))
    if count >= MAX_AUTOFIX_RESTARTS:
        print(f"[yellow]autofix[/yellow]: src changes detected in {stage}, restart limit reached")
        return
    state["count"] = count + 1
    state["last_stage"] = stage
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"[yellow]autofix[/yellow]: src changes detected in {stage}; restarting to reload code")
    os.environ["KAGGLEBOT_RESUME_RUN_ID"] = run_id
    os.environ["KAGGLEBOT_RESUME_SLUG"] = config.slug
    os.execv(sys.executable, [sys.executable, *sys.argv])


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
    allow_force = config.force_submit or _env_truthy("KAGGLEBOT_FORCE_RESUBMIT")
    last_submission_path = str(run_state.get("last_submission_path") or "").strip()
    if last_submission_path and Path(last_submission_path) == submission_path and not allow_force:
        print("[yellow]submit skipped[/yellow]: same submission file already attempted in this run")
        known_fingerprint = str(run_state.get("last_submit_fingerprint") or run_state.get("last_fingerprint") or "")
        _append_submit_attempt(
            run_dir=run_dir,
            payload={
                "run_id": run_id,
                "sub_path": str(submission_path),
                "sub_sha256": _sha256_or_none(submission_path),
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
            submission_path=submission_path,
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
                submission_path=prepared_submission_path,
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
            submission_path=prepared_submission_path,
            fingerprint=compute_error_fingerprint("", "rules_not_accepted"),
            error_kind="permanent",
            reason="rules_not_accepted",
            message="Competition rules are not accepted; aborting submit stage for this run.",
            stdout_tail="",
            stderr_tail="rules_not_accepted",
            exit_code=RulesNotAcceptedError.exit_code,
        )

    seen_fingerprints = set(_load_submit_fingerprints(run_dir))
    state_fingerprint = str(run_state.get("last_submit_fingerprint") or run_state.get("last_fingerprint") or "").strip()
    if state_fingerprint:
        seen_fingerprints.add(state_fingerprint)
    max_attempts = max(1, _SUBMIT_MAX_TRANSIENT_RETRIES)
    for attempt in range(1, max_attempts + 1):
        try:
            submission_result = submission_service.submit_prepared(
                prepared_path=prepared_submission_path,
                message=message,
                run_id=run_id,
            )
        except SubmissionCliError as exc:
            classification = classify_submit_error(exc.stdout, exc.stderr, exc.exit_code)
            classification_kind = str(classification.get("kind") or "unknown")
            classification_reason = str(classification.get("reason") or "unclassified_submit_error")
            fingerprint = compute_error_fingerprint(exc.stdout, exc.stderr)
            if fingerprint in seen_fingerprints:
                return _abort_submit_for_run(
                    config=config,
                    run_id=run_id,
                    problem_types=problem_types,
                    submission_path=prepared_submission_path,
                    fingerprint=fingerprint,
                    error_kind=classification_kind,
                    reason="same_error_fingerprint_recurred",
                    message="Same submit error fingerprint recurred; aborting this run to prevent infinite loop.",
                    stdout_tail=exc.stdout,
                    stderr_tail=exc.stderr,
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
                        "sub_path": str(prepared_submission_path),
                        "sub_sha256": _sha256_or_none(prepared_submission_path),
                        "exit_code": exc.exit_code,
                        "ok": False,
                        "fingerprint": fingerprint,
                        "error_kind": "transient",
                        "action_taken": "retry",
                        "reason": classification_reason,
                        "stdout_tail": exc.stdout[-_SUBMIT_STDOUT_TAIL_CHARS:],
                        "stderr_tail": exc.stderr[-_SUBMIT_STDERR_TAIL_CHARS:],
                    },
                )
                _record_submit_reason_knowledge(
                    config=config,
                    run_id=run_id,
                    problem_types=problem_types,
                    submission_path=prepared_submission_path,
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
                submission_path=prepared_submission_path,
                fingerprint=fingerprint,
                error_kind=classification_kind,
                reason=classification_reason,
                message=(
                    "Submit failed and is not retryable in this run."
                    if classification_kind != "transient"
                    else "Transient submit error exceeded retry budget; aborting this run."
                ),
                stdout_tail=exc.stdout,
                stderr_tail=exc.stderr,
                exit_code=exc.exit_code,
            )
        except (DuplicateSubmissionError, SubmissionRateLimitError) as exc:
            fingerprint = compute_error_fingerprint("", str(exc))
            return _abort_submit_for_run(
                config=config,
                run_id=run_id,
                problem_types=problem_types,
                submission_path=prepared_submission_path,
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
                    submission_path=prepared_submission_path,
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

    submission_for_submit = submission_result.submission_path
    stdout_tail = submission_result.stdout[-_SUBMIT_STDOUT_TAIL_CHARS:]
    stderr_tail = submission_result.stderr[-_SUBMIT_STDERR_TAIL_CHARS:]
    fingerprint = compute_error_fingerprint(submission_result.stdout, submission_result.stderr)
    _append_submit_attempt(
        run_dir=run_dir,
        payload={
            "run_id": run_id,
            "sub_path": str(submission_for_submit),
            "sub_sha256": _sha256_or_none(submission_for_submit),
            "exit_code": submission_result.exit_code,
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
            "last_error_kind": "none",
            "last_action": "submit",
            "last_reason": "submitted",
            "last_submission_path": str(submission_for_submit),
            "submit_attempts_count": int(_load_run_state(run_dir).get("submit_attempts_count", 0)) + 1,
        },
    )
    print("[green]submission recorded[/green]")
    outcome = _wait_for_submission_outcome(
        slug=config.slug,
        message=message,
        submitted_at=submitted_at,
    )
    if outcome and outcome.get("score") is not None:
        print(
            "[cyan]submission result[/cyan]: "
            f"status={outcome.get('status') or 'unknown'} score={float(outcome['score']):.6f}"
        )
    else:
        print("[yellow]submission result[/yellow]: score not available yet; knowledge update skipped")
    return {
        "message": message,
        "submission_path": str(submission_for_submit),
        "submitted_at": submitted_at.isoformat(),
        "iteration": _infer_iteration_from_submission_path(submission_path),
        "outcome": outcome,
    }


def _abort_submit_for_run(
    *,
    config: AutopilotConfig,
    run_id: str,
    problem_types: list[str],
    submission_path: Path,
    fingerprint: str,
    error_kind: str,
    reason: str,
    message: str,
    stdout_tail: str,
    stderr_tail: str,
    exit_code: int | None,
) -> None:
    run_dir = config.paths.run_dir(run_id)
    _append_submit_attempt(
        run_dir=run_dir,
        payload={
            "run_id": run_id,
            "sub_path": str(submission_path),
            "sub_sha256": _sha256_or_none(submission_path),
            "exit_code": exit_code,
            "ok": False,
            "fingerprint": fingerprint,
            "error_kind": error_kind,
            "action_taken": "abort",
            "reason": reason,
            "stdout_tail": stdout_tail[-_SUBMIT_STDOUT_TAIL_CHARS:],
            "stderr_tail": stderr_tail[-_SUBMIT_STDERR_TAIL_CHARS:],
        },
    )
    prior = _load_run_state(run_dir)
    _save_run_state(
        run_dir,
        {
            "submit_attempted": True,
            "submit_ok": False,
            "last_submit_fingerprint": fingerprint,
            "last_fingerprint": fingerprint,
            "last_error_kind": error_kind,
            "last_action": "abort",
            "last_reason": reason,
            "last_submission_path": str(submission_path),
            "submit_attempts_count": int(prior.get("submit_attempts_count", 0)) + 1,
        },
    )
    _record_submit_reason_knowledge(
        config=config,
        run_id=run_id,
        problem_types=problem_types,
        submission_path=submission_path,
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
    return payload


def _save_run_state(run_dir: Path, updates: dict[str, object]) -> None:
    state = _load_run_state(run_dir)
    state.update(updates)
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


def _sha256_or_none(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return sha256_file(str(path))
    except OSError:
        return None


def _compute_submit_backoff(attempt: int) -> float:
    base = _SUBMIT_BACKOFF_BASE_SEC * (2 ** max(0, attempt - 1))
    jitter = random.uniform(0.0, 1.0)
    return base + jitter


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
    iteration_suffix = f" iter={iteration}" if isinstance(iteration, int) else ""
    if best_score is None:
        return f"kagglebot {config.slug} {run_id}{iteration_suffix}"
    return f"kagglebot {config.slug} {run_id}{iteration_suffix} best_offline={best_score:.6f}"


def _normalize_metric_name(name: str | None) -> str:
    if not name:
        return ""
    return "".join(ch for ch in name.lower() if ch.isalnum())


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
