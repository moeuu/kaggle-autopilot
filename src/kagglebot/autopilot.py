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

from rich import print

from kagglebot.agents.codex_runner import run_codex
from kagglebot.agents.strategy_runner import run_strategy
from kagglebot.compute import Compute
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
    leaderboard_top1,
    list_competition_submissions,
)
from kagglebot.kernel_runner import _collect_log_tail, resolve_kaggle_username, run_kernel
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
from kagglebot.solver.initial_model import train_evaluate_and_predict
from kagglebot.solver.metrics import infer_direction
from kagglebot.submission.guard import (
    classify_submit_error,
    compute_error_fingerprint,
    normalize_error_text,
)
from kagglebot.submission_service import SubmissionConfig, SubmissionService
from kagglebot.types import PlanConfig

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
_SUBMISSION_POLL_MAX_ATTEMPTS = 20
_SUBMISSION_POLL_INTERVAL_SEC = 30.0
_SUBMIT_MAX_TRANSIENT_RETRIES = 3
_SUBMIT_BACKOFF_BASE_SEC = 2.0
_SUBMIT_STDERR_TAIL_CHARS = 1200
_SUBMIT_STDOUT_TAIL_CHARS = 1200


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
    attempt = 0
    while True:
        try:
            return _run_autopilot_core(config, run_id, resume_run=resume_run)
        except RulesNotAcceptedError:
            raise
        except SubmitAbortedError:
            raise
        except KernelCapacityError:
            raise
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            if config.dry_run:
                raise
            attempt += 1
            if attempt > MAX_AUTOFIX_ATTEMPTS:
                raise
            print("[yellow]autofix[/yellow]: invoking codex to repair error")
            _run_autofix(config=config, run_id=run_id, attempt=attempt, error=exc)


def _run_autopilot_core(config: AutopilotConfig, run_id: str, *, resume_run: bool = False) -> None:
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[green]run started[/green]: {run_id}")
    plan = _load_plan(config.paths)
    if not config.paths.plan_path.exists():
        _write_plan(config.paths, plan)

    print(f"[cyan]fetching leaderboard[/cyan]: {config.slug}")
    top1_info = leaderboard_top1(config.slug, config.paths.context_dir, dry_run=config.dry_run)
    config.paths.top1_public_path.write_text(json.dumps(top1_info, indent=2), encoding="utf-8")
    _print_top1_info(top1_info)
    _refresh_knowledge_hints(config)

    if _should_skip_planning(resume_run=resume_run, paths=config.paths):
        print("[yellow]resume[/yellow]: skipping planning after restart; reusing existing plan")
    elif _needs_planning(plan, config):
        print("[cyan]plan[/cyan]: generating initial plan")
        _run_plan_and_initial(config, run_id)
        plan = _load_plan(config.paths)

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
    if run_payload.get("status") == "running":
        run_payload["status"] = "completed"
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
    dataset_profile = _load_dataset_profile(config.paths)
    problem_types = derive_problem_types(dataset_profile)
    best_score = None
    best_submission: Path | None = None
    submitted = False
    pending_problem_insights: list[dict[str, object]] = []
    pending_error_fixes: list[dict[str, object]] = []
    submit_candidate: Path | None = None
    submit_candidate_score: float | None = None
    submit_top1_score: float | None = None

    max_iterations = max(1, int(resolved["max_iterations"]))
    holdout_frac = float(resolved["holdout_frac"])
    cv_folds = int(resolved["cv_folds"])
    seed = int(resolved["seed"])
    score_source = str(resolved["score_source"] or "auto")
    kernel_name = resolved["kernel_name"]
    enable_internet = str(resolved["internet"]) == "on"
    start_iteration, best_score, best_submission = _resume_iteration_state(
        paths=config.paths,
        run_id=run_id,
        metric_direction=metric_direction,
        target_metric=target_metric,
        max_iterations=max_iterations,
    )
    if start_iteration > 1:
        print(f"[yellow]resume[/yellow]: found completed iterations; resuming at {start_iteration}/{max_iterations}")

    try:
        for iteration in range(start_iteration, max_iterations + 1):
            iter_dir = config.paths.iter_dir(run_id, iteration)
            logs_dir = iter_dir / "logs"
            agent_dir = iter_dir / "agent"
            output_dir = iter_dir / "output"
            iter_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)
            agent_dir.mkdir(parents=True, exist_ok=True)

            print(f"[cyan]iteration[/cyan]: {iteration}/{max_iterations}")
            _refresh_knowledge_hints(config)

            _run_verify(config.verify_cmd, dry_run=config.dry_run)

            submission_path = iter_dir / "submission.csv"
            evaluation = None
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
                compute_enum = Compute(config.compute)
                print(f"[cyan]training[/cyan]: {config.compute}")
                outcome = train_evaluate_and_predict(
                    data_dir=config.paths.data_dir,
                    output_path=submission_path,
                    compute=compute_enum,
                    strict_accelerator=config.strict_accelerator,
                    seed=seed,
                    score_source=score_source,
                    metric=target_metric,
                    direction=metric_direction,
                    holdout_frac=holdout_frac,
                    cv_folds=cv_folds,
                    plan_score_source=score_source,
                    target_override=None,
                )
                evaluation = outcome.evaluation
                model_summary = outcome.model_summary
                accelerator_used = outcome.accelerator

            if evaluation is None:
                raise RuntimeError("No evaluation metrics produced.")
            if evaluation.metric and target_metric:
                normalized_eval = _normalize_metric_name(evaluation.metric)
                normalized_target = _normalize_metric_name(target_metric)
                if normalized_eval and normalized_eval != normalized_target:
                    corrected_direction = infer_direction(evaluation.metric, "auto")
                    if corrected_direction != metric_direction or evaluation.metric != target_metric:
                        print(
                            "[yellow]metric mismatch[/yellow]: "
                            f"plan={target_metric}/{metric_direction}, "
                            f"kernel={evaluation.metric}/{corrected_direction}. "
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
            print(f"[green]iteration complete[/green]: {evaluation.metric}={evaluation.value:.6f}")

            met_target = _meets_target(evaluation.value, target_score, metric_direction)
            top1_score = top1_info.get("score") if isinstance(top1_info, dict) else None
            top1_tier = _is_top1_tier(evaluation.value, top1_score, metric_direction)

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
            )
            (iter_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

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
            )
            (iter_dir / "diagnostics.md").write_text(diagnostics, encoding="utf-8")

            record_iteration(
                knowledge_paths=config.knowledge_paths,
                run_id=run_id,
                iteration=iteration,
                score_source=evaluation.score_source,
                offline_value=evaluation.value,
                offline_std=evaluation.std,
                top1_public_score=top1_info.get("score") if isinstance(top1_info, dict) else None,
                met_target=met_target,
                git_commit=None,
            )

            prev_best = best_score
            delta_offline = None
            if prev_best is not None:
                delta_offline = (
                    prev_best - evaluation.value if metric_direction == "minimize" else evaluation.value - prev_best
                )
            improved = _update_best_score(best_score, evaluation.value, metric_direction, 0.0)
            if improved:
                best_score = evaluation.value
                best_submission = submission_path

            if top1_tier:
                if config.submit:
                    submit_candidate = submission_path
                    submit_candidate_score = best_score or evaluation.value
                    submit_top1_score = top1_score if isinstance(top1_score, (int, float)) else None
                else:
                    run_payload["status"] = "completed"
                break

            if iteration >= max_iterations:
                if config.submit and best_submission and not submitted:
                    submit_candidate = best_submission
                    submit_candidate_score = best_score
                    submit_top1_score = top1_score if isinstance(top1_score, (int, float)) else None
                elif not submitted:
                    run_payload["status"] = "completed"
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
            )
    except KeyboardInterrupt:
        run_payload["status"] = "interrupted"
        (run_dir / "run.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
        print("[yellow]run interrupted[/yellow]")
        return

    if config.submit and submit_candidate and not submitted:
        try:
            submission_result = _attempt_submit(
                config=config,
                run_id=run_id,
                submission_path=submit_candidate,
                best_score=submit_candidate_score,
                problem_types=problem_types,
            )
        except SubmitAbortedError:
            run_payload["status"] = "submit_failed"
            (run_dir / "run.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
            raise
        if submission_result:
            _record_submission_knowledge(
                config=config,
                run_id=run_id,
                problem_types=problem_types,
                pending_problem_insights=pending_problem_insights,
                pending_error_fixes=pending_error_fixes,
                submission_result=submission_result,
                metric_direction=metric_direction,
                target_score=target_score,
                top1_score=submit_top1_score,
            )
            submitted = True
            run_payload["status"] = "submitted"
        else:
            run_payload["status"] = "completed"

    (run_dir / "run.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")


def _load_plan(paths: CompetitionPaths) -> PlanConfig:
    if not paths.plan_path.exists():
        return PlanConfig()
    try:
        payload = json.loads(paths.plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return PlanConfig()
    return PlanConfig.from_dict(payload)


def _write_plan(paths: CompetitionPaths, plan: PlanConfig) -> None:
    paths.plan_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")


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
    if paths.plan_path.exists():
        agent_dir = paths.context_dir / "agent"
        kernel_path = paths.kernel_source_dir / "kernel.py"
        if agent_dir.exists() or kernel_path.exists():
            return True
    return False


def _resolve_plan(plan: PlanConfig, config: AutopilotConfig) -> dict[str, object]:
    def choose(value, fallback, default):
        if value is not None:
            return value
        if fallback is not None:
            return fallback
        return default

    target_metric = choose(config.target_metric, plan.target_metric, None)
    target_score = choose(config.target_score, plan.target_score, None)
    target_direction = choose(config.target_direction, plan.target_direction, "auto")
    score_source = choose(config.score_source, plan.score_source, "auto")
    holdout_frac = choose(config.holdout_frac, plan.holdout_frac, 0.2)
    cv_folds = choose(config.cv_folds, plan.cv_folds, 5)
    seed = choose(config.seed, plan.seed, 42)
    time_budget_min = choose(config.time_budget_min, plan.time_budget_min, None)
    kernel_name = choose(config.kernel_name, plan.kernel_name, None)
    internet = choose(config.internet, plan.internet, "on")
    if internet in (None, "auto"):
        internet = "on"
    default_max_iterations = 1
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
    submit_policy = choose(None, plan.submit_policy, "on_target_only")

    return {
        "target_metric": target_metric,
        "target_score": target_score,
        "target_direction": target_direction,
        "score_source": score_source,
        "holdout_frac": holdout_frac,
        "cv_folds": cv_folds,
        "seed": seed,
        "time_budget_min": time_budget_min,
        "kernel_name": kernel_name,
        "internet": internet,
        "max_iterations": max_iterations,
        "max_total_min": max_total_min,
        "patience": patience,
        "min_improvement": min_improvement,
        "submit_policy": submit_policy,
    }


def _resolved_plan(resolved: dict[str, object]) -> PlanConfig:
    return PlanConfig(
        target_metric=resolved.get("target_metric"),  # type: ignore[arg-type]
        target_direction=str(resolved.get("target_direction") or "auto"),
        target_score=resolved.get("target_score"),  # type: ignore[arg-type]
        score_source=str(resolved.get("score_source") or "auto"),
        holdout_frac=resolved.get("holdout_frac"),  # type: ignore[arg-type]
        cv_folds=resolved.get("cv_folds"),  # type: ignore[arg-type]
        seed=resolved.get("seed"),  # type: ignore[arg-type]
        time_budget_min=resolved.get("time_budget_min"),  # type: ignore[arg-type]
        kernel_name=resolved.get("kernel_name"),  # type: ignore[arg-type]
        internet=str(resolved.get("internet") or "on"),
        max_iterations=int(resolved.get("max_iterations") or 1),
        max_total_min=resolved.get("max_total_min"),  # type: ignore[arg-type]
        patience=int(resolved.get("patience") or 2),
        min_improvement=float(resolved.get("min_improvement") or 0.0),
        submit_policy=str(resolved.get("submit_policy") or "on_target_only"),
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
            "target_metric": resolved.get("target_metric"),
            "target_score": resolved.get("target_score"),
            "target_direction": resolved.get("target_direction"),
            "max_iterations": resolved.get("max_iterations"),
            "max_total_min": resolved.get("max_total_min"),
            "patience": resolved.get("patience"),
            "min_improvement": resolved.get("min_improvement"),
            "time_budget_min": resolved.get("time_budget_min"),
            "seed": resolved.get("seed"),
            "submit_policy": resolved.get("submit_policy"),
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

    std = payload.get("offline_std")
    if std is None:
        std = payload.get("std")

    score_source = payload.get("score_source", "holdout")
    if score_source == "holdout":
        for key in payload.keys():
            if isinstance(key, str) and key.lower().startswith("oof_"):
                score_source = "cv"
                break

    return EvaluationResult(
        score_source=score_source,
        metric=metric_name or target_metric or "unknown",
        direction=direction,  # type: ignore[arg-type]
        value=float(value),
        std=std,
        train_score=None,
        val_score=None,
        fold_scores=None,
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
) -> dict[str, object]:
    return {
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
        "folds": cv_folds if evaluation.score_source == "cv" else None,
        "holdout_frac": holdout_frac if evaluation.score_source == "holdout" else None,
        "seed": seed,
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
) -> str:
    direction = evaluation.direction
    delta_to_target = target_score - evaluation.value if direction == "minimize" else evaluation.value - target_score
    best_line = best_score if best_score is not None else evaluation.value
    trend = "improving" if best_score is None or _meets_target(evaluation.value, best_line, direction) else "stalled"
    top1_delta = None
    if top1_score is not None:
        top1_delta = top1_score - evaluation.value if direction == "minimize" else evaluation.value - top1_score
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
        f"Score: {evaluation.value:.6f} vs target {target_score:.6f} (delta {delta_to_target:.6f})",
        f"Best so far: {best_line:.6f} ({trend})",
        f"Evaluation: {evaluation.score_source}",
    ]
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
        json.dumps(model_summary, indent=2),
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
) -> None:
    prompt_template = config.paths.codex_improve_template.read_text(encoding="utf-8")
    agent_dir = iter_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = agent_dir / "prompt.md"
    top1_score = top1_info.get("score") if isinstance(top1_info, dict) else None
    improvement_mode, top1_gap = _classify_improvement_mode(evaluation.value, top1_score, evaluation.direction)
    kernel_main_path = config.paths.kernel_source_dir / "kernel.py"
    model_family = _infer_kernel_model_family(kernel_main_path)
    nn_upgrade_required = (
        iteration == 1
        and improvement_mode == "major_overhaul"
        and model_family == "tree_only"
        and top1_gap is not None
        and top1_gap > 0
    )
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
        current_score=f"{evaluation.value:.6f}",
        target_score=f"{target_score:.6f}",
        top1_score=str(top1_score or "unavailable"),
        top1_source=str(top1_info.get("source") or "unknown"),
        top1_gap="unavailable" if top1_gap is None else f"{top1_gap:.6f}",
        delta_offline="unavailable" if delta_offline is None else f"{delta_offline:.6f}",
        improvement_mode=improvement_mode,
        model_family_hint=model_family,
        nn_upgrade_required="yes" if nn_upgrade_required else "no",
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
    problem_type_knowledge = _load_problem_type_knowledge_text(config)
    strategy_prompt = _build_improvement_strategy_prompt(
        slug=config.slug,
        run_id=run_id,
        iteration=iteration,
        metric=evaluation.metric,
        direction=evaluation.direction,
        current_score=evaluation.value,
        target_score=target_score,
        top1_score=top1_score,
        top1_source=str(top1_info.get("source") or "unknown"),
        top1_gap=top1_gap,
        delta_offline=delta_offline,
        improvement_mode=improvement_mode,
        model_family=model_family,
        nn_upgrade_required=nn_upgrade_required,
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
    result = run_codex(
        prompt_path,
        agent_dir,
        dry_run=config.dry_run,
        model=_ERROR_FIX_CODEX_MODEL,
        reasoning_effort=_ERROR_FIX_REASONING_EFFORT,
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
    target_score: float,
    top1_score: float | None,
    top1_source: str,
    top1_gap: float | None,
    delta_offline: float | None,
    improvement_mode: str,
    model_family: str,
    nn_upgrade_required: bool,
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
Current score: {current_score:.6f}
Target score: {target_score:.6f}
Top1 score: {"unavailable" if top1_score is None else f"{top1_score:.6f}"}
Top1 source: {top1_source}
Top1 gap: {"unavailable" if top1_gap is None else f"{top1_gap:.6f}"}
Delta offline: {"unavailable" if delta_offline is None else f"{delta_offline:.6f}"}
Improvement mode: {improvement_mode}
Model family hint: {model_family}
NN upgrade required: {"yes" if nn_upgrade_required else "no"}

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
    if isinstance(error, KaggleCliError):
        if error.command:
            error_text = f"{error_text}\n\nKaggle CLI command:\n{shlex.join(error.command)}"
        if error.output:
            error_text = f"{error_text}\n\nKaggle CLI output:\n{error.output}"
    attempt_tag = f"{attempt:02d}"
    header = f"autofix_attempt: {attempt}\n"
    error_path = autofix_dir / f"error-{attempt_tag}.txt"
    error_path.write_text(header + error_text + "\n", encoding="utf-8")
    (autofix_dir / "error.txt").write_text(header + error_text + "\n", encoding="utf-8")

    if _maybe_write_column_fill(config, error_text):
        note_path = autofix_dir / "note.txt"
        note = (
            "autofix_note: column_fill.json created for missing column error.\n"
            "autofix will retry without modifying kernel sources.\n"
        )
        note_path.write_text(note, encoding="utf-8")
        print("[yellow]autofix[/yellow]: wrote column_fill.json; retrying without kernel edits")
        return

    if _maybe_write_object_coerce(config, error_text):
        note_path = autofix_dir / "note.txt"
        note = (
            "autofix_note: object_coerce.json created for numpy.object_ conversion error.\n"
            "autofix will retry without modifying kernel sources.\n"
        )
        note_path.write_text(note, encoding="utf-8")
        print("[yellow]autofix[/yellow]: wrote object_coerce.json; retrying without kernel edits")
        return

    if _maybe_write_device_coerce(config, error_text):
        note_path = autofix_dir / "note.txt"
        note = (
            "autofix_note: device_coerce.json created for torch device mismatch error.\n"
            "autofix will retry without modifying kernel sources.\n"
        )
        note_path.write_text(note, encoding="utf-8")
        print("[yellow]autofix[/yellow]: wrote device_coerce.json; retrying without kernel edits")
        return

    if _maybe_write_column_map(config, error_text):
        note_path = autofix_dir / "note.txt"
        note = (
            "autofix_note: column_map.json created for missing column error.\n"
            "autofix will retry without modifying kernel sources.\n"
        )
        note_path.write_text(note, encoding="utf-8")
        print("[yellow]autofix[/yellow]: wrote column_map.json; retrying without kernel edits")
        return

    allowed_prefixes = [
        config.paths.repo_root / "src",
        config.paths.repo_root / "docs",
        config.paths.repo_root / "tests",
        config.paths.kernel_source_dir,
        config.paths.context_dir,
        config.paths.runs_dir,
        config.paths.prompts_dir,
    ]
    prompt_text = _build_autofix_prompt(
        config=config,
        run_id=run_id,
        attempt=attempt,
        error_text=error_text,
        error_path=error_path,
        allowed_prefixes=allowed_prefixes,
    )
    strategy_prompt = _build_error_strategy_prompt(
        stage="autofix",
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
        stage_label="autofix",
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


def _build_autofix_prompt(
    *,
    config: AutopilotConfig,
    run_id: str,
    attempt: int,
    error_text: str,
    error_path: Path,
    allowed_prefixes: list[Path],
) -> str:
    allowed_list = "\n".join(f"- {path}" for path in allowed_prefixes)
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

## Relevant Paths

- repo_root: {config.paths.repo_root}
- run_dir: {config.paths.run_dir(run_id)}
- kernel_dir: {config.paths.kernel_source_dir}
- context_dir: {config.paths.context_dir}
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
        canonical = group[0]
        match = _match_column(group, normalized, all_columns)
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


def _resume_iteration_state(
    *,
    paths: CompetitionPaths,
    run_id: str,
    metric_direction: str,
    target_metric: str,
    max_iterations: int,
) -> tuple[int, float | None, Path | None]:
    run_dir = paths.run_dir(run_id)
    if not run_dir.exists():
        return 1, None, None
    best_score: float | None = None
    best_submission: Path | None = None
    completed_iters: list[int] = []
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
        evaluation = None
        if metrics_path.exists():
            try:
                evaluation = _load_kernel_metrics(metrics_path, metric_direction, target_metric)
            except Exception:  # noqa: BLE001
                evaluation = None
        if evaluation is None and submission_path.exists():
            completed_iters.append(iteration)
            if best_submission is None:
                best_submission = submission_path
            print(
                "[yellow]resume[/yellow]: "
                f"{metrics_path} missing; treating iter-{iteration} as complete based on submission.csv."
            )
            continue
        if evaluation is None:
            continue
        completed_iters.append(iteration)
        if best_submission is None and submission_path.exists():
            best_submission = submission_path
        if best_score is None:
            best_score = evaluation.value
        else:
            if _meets_target(evaluation.value, best_score, metric_direction):
                best_score = evaluation.value
                if submission_path.exists():
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
    if run_state.get("submit_attempted") and not allow_force:
        print("[yellow]submit skipped[/yellow]: this run already attempted submission; use --force-submit to override")
        _append_submit_attempt(
            run_dir=run_dir,
            payload={
                "run_id": run_id,
                "sub_path": str(submission_path),
                "sub_sha256": _sha256_or_none(submission_path),
                "exit_code": None,
                "ok": False,
                "fingerprint": str(run_state.get("last_fingerprint") or ""),
                "error_kind": "unknown",
                "action_taken": "skip",
                "reason": "submit_already_attempted_in_run",
                "stdout_tail": "",
                "stderr_tail": "",
            },
        )
        return None

    try:
        rules_accepted = check_rules_accepted(config.slug, dry_run=config.dry_run)
    except KaggleCliError as exc:
        if _is_missing_kaggle_credentials_error(exc):
            return _abort_submit_for_run(
                config=config,
                run_id=run_id,
                problem_types=problem_types,
                submission_path=submission_path,
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
            submission_path=submission_path,
            fingerprint=compute_error_fingerprint("", "rules_not_accepted"),
            error_kind="permanent",
            reason="rules_not_accepted",
            message="Competition rules are not accepted; aborting submit stage for this run.",
            stdout_tail="",
            stderr_tail="rules_not_accepted",
            exit_code=RulesNotAcceptedError.exit_code,
        )

    message = _submission_message(config, run_id, best_score)
    submission_service = SubmissionService(
        SubmissionConfig(
            slug=config.slug,
            data_dir=config.paths.data_dir,
            sample_submission_path=config.paths.sample_submission_path,
            submission_ledger_path=config.paths.submission_ledger_path,
            dry_run=config.dry_run,
            force_submit=config.force_submit,
        )
    )
    print(f"[cyan]submit[/cyan]: {config.slug}")
    submitted_at = datetime.now(UTC)
    seen_fingerprints = set(_load_submit_fingerprints(run_dir))
    max_attempts = max(1, _SUBMIT_MAX_TRANSIENT_RETRIES)
    for attempt in range(1, max_attempts + 1):
        try:
            submission_result = submission_service.submit(
                submission_path=submission_path,
                message=message,
                run_id=run_id,
            )
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
        except SubmissionCliError as exc:
            classification = classify_submit_error(exc.stdout, exc.stderr, exc.exit_code)
            fingerprint = compute_error_fingerprint(exc.stdout, exc.stderr)
            if fingerprint in seen_fingerprints:
                return _abort_submit_for_run(
                    config=config,
                    run_id=run_id,
                    problem_types=problem_types,
                    submission_path=submission_path,
                    fingerprint=fingerprint,
                    error_kind=classification.kind,
                    reason="same_error_fingerprint_recurred",
                    message="Same submit error fingerprint recurred; aborting this run to prevent infinite loop.",
                    stdout_tail=exc.stdout,
                    stderr_tail=exc.stderr,
                    exit_code=exc.exit_code,
                )
            seen_fingerprints.add(fingerprint)
            if classification.kind == "transient" and attempt < max_attempts:
                wait_seconds = _compute_submit_backoff(attempt)
                print(
                    "[yellow]submit retry[/yellow]: transient submit error "
                    f"(reason={classification.reason}, attempt={attempt}/{max_attempts}, wait={wait_seconds:.1f}s)"
                )
                _append_submit_attempt(
                    run_dir=run_dir,
                    payload={
                        "run_id": run_id,
                        "sub_path": str(submission_path),
                        "sub_sha256": _sha256_or_none(submission_path),
                        "exit_code": exc.exit_code,
                        "ok": False,
                        "fingerprint": fingerprint,
                        "error_kind": "transient",
                        "action_taken": "retry",
                        "reason": classification.reason,
                        "stdout_tail": exc.stdout[-_SUBMIT_STDOUT_TAIL_CHARS:],
                        "stderr_tail": exc.stderr[-_SUBMIT_STDERR_TAIL_CHARS:],
                    },
                )
                _record_submit_reason_knowledge(
                    config=config,
                    run_id=run_id,
                    problem_types=problem_types,
                    submission_path=submission_path,
                    error_kind="transient",
                    reason=classification.reason,
                    action_taken="retry",
                    fingerprint=fingerprint,
                    details=f"attempt={attempt}; wait={wait_seconds:.1f}s",
                )
                time.sleep(wait_seconds)
                continue
            print(
                "[red]submit aborted[/red]: "
                f"{classification.kind} submit error (reason={classification.reason}); no further retries in this run."
            )
            return _abort_submit_for_run(
                config=config,
                run_id=run_id,
                problem_types=problem_types,
                submission_path=submission_path,
                fingerprint=fingerprint,
                error_kind=classification.kind,
                reason=classification.reason,
                message=(
                    "Submit failed and is not retryable in this run."
                    if classification.kind != "transient"
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
                submission_path=submission_path,
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
                    submission_path=submission_path,
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
        return {"submit_attempted": attempted}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    if not payload.get("submit_attempted"):
        payload["submit_attempted"] = _has_submit_attempt_records(run_dir)
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


def _sha256_or_none(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return sha256_file(str(path))
    except OSError:
        return None


def _compute_submit_backoff(attempt: int) -> float:
    base = _SUBMIT_BACKOFF_BASE_SEC * (2 ** max(0, attempt - 1))
    jitter = random.uniform(0.0, 0.75)
    return base + jitter


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
    max_attempts: int = _SUBMISSION_POLL_MAX_ATTEMPTS,
    poll_interval_sec: float = _SUBMISSION_POLL_INTERVAL_SEC,
) -> dict[str, object] | None:
    for attempt in range(1, max_attempts + 1):
        try:
            rows = list_competition_submissions(slug, dry_run=False)
        except Exception:  # noqa: BLE001
            return None
        match = _select_submission_row(rows=rows, message=message, submitted_at=submitted_at)
        if match is not None:
            status = _extract_submission_status(match)
            score = _extract_submission_score(match)
            if score is not None:
                return {
                    "status": status,
                    "score": score,
                    "raw": match,
                    "checked_at": datetime.now(UTC).isoformat(),
                }
            if status in {"complete", "completed", "error", "failed", "cancelled"}:
                return {
                    "status": status,
                    "score": None,
                    "raw": match,
                    "checked_at": datetime.now(UTC).isoformat(),
                }
        if attempt < max_attempts:
            time.sleep(poll_interval_sec)
    return None


def _select_submission_row(
    *,
    rows: list[dict[str, str]],
    message: str,
    submitted_at: datetime,
) -> dict[str, str] | None:
    if not rows:
        return None
    target = message.strip()
    with_message = [row for row in rows if _row_matches_submission_message(row, target)]
    candidates = with_message if with_message else rows
    rows_with_ts: list[tuple[datetime, dict[str, str]]] = []
    for row in candidates:
        ts = _parse_submission_row_time(row)
        if ts is None:
            continue
        rows_with_ts.append((ts, row))
    if rows_with_ts:
        window_start = submitted_at.timestamp() - 3600
        recent = [item for item in rows_with_ts if item[0].timestamp() >= window_start]
        source = recent or rows_with_ts
        source.sort(key=lambda item: item[0], reverse=True)
        return source[0][1]
    return candidates[0]


def _row_matches_submission_message(row: dict[str, str], message: str) -> bool:
    if not message:
        return False
    for key in ("description", "message", "comments", "comment"):
        value = _get_row_value_ci(row, key)
        if value and value.strip() == message:
            return True
    return False


def _parse_submission_row_time(row: dict[str, str]) -> datetime | None:
    for key in ("date", "submittedDate", "submitted_date", "createdAt", "created_at", "timestamp"):
        value = _get_row_value_ci(row, key)
        if not value:
            continue
        parsed = _parse_datetime(value)
        if parsed is not None:
            return parsed
    return None


def _extract_submission_status(row: dict[str, str]) -> str:
    for key in ("status", "state"):
        value = _get_row_value_ci(row, key)
        if value:
            return value.strip().lower()
    return "unknown"


def _extract_submission_score(row: dict[str, str]) -> float | None:
    for key in ("publicScore", "public_score", "score", "privateScore", "private_score"):
        value = _get_row_value_ci(row, key)
        parsed = _to_float(value)
        if parsed is not None:
            return parsed
    return None


def _get_row_value_ci(row: dict[str, str], key: str) -> str | None:
    target = key.strip().lower()
    for current_key, value in row.items():
        if current_key.strip().lower() == target:
            return value
    return None


def _parse_datetime(value: str) -> datetime | None:
    raw = str(value).strip()
    if not raw:
        return None
    normalized = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y/%m/%d %H:%M:%S",
        ):
            try:
                dt = datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue
        else:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


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


def _submission_message(config: AutopilotConfig, run_id: str, best_score: float | None) -> str:
    if config.message:
        return config.message
    if best_score is None:
        return f"kagglebot {config.slug} {run_id}"
    return f"kagglebot {config.slug} {run_id} best_offline={best_score:.6f}"


def _normalize_metric_name(name: str | None) -> str:
    if not name:
        return ""
    return "".join(ch for ch in name.lower() if ch.isalnum())


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


def _classify_improvement_mode(value: float, top1_score: float | None, direction: str) -> tuple[str, float | None]:
    if top1_score is None:
        return "major_overhaul", None
    gap = top1_score - value if direction == "maximize" else value - top1_score
    if gap >= MAJOR_TOP1_GAP:
        return "major_overhaul", gap
    if gap >= MODERATE_TOP1_GAP:
        return "moderate_update", gap
    return "minor_tuning", gap


def _infer_kernel_model_family(kernel_main: Path) -> str:
    if not kernel_main.exists():
        return "unknown"
    text = kernel_main.read_text(encoding="utf-8", errors="ignore").lower()
    tree_markers = (
        "lightgbm",
        "lgbm",
        "xgboost",
        "catboost",
        "randomforest",
        "gradientboosting",
        "histgradientboosting",
        "xgb.",
        "lgb.",
    )
    nn_markers = (
        "torch",
        "tensorflow",
        "keras",
        "pytorch",
        "tabnet",
        "ft-transformer",
        "ft_transformer",
        "transformer",
        "mlp",
    )
    has_tree = any(marker in text for marker in tree_markers)
    has_nn = any(marker in text for marker in nn_markers)
    if has_tree and has_nn:
        return "hybrid(tree+nn)"
    if has_tree:
        return "tree_only"
    if has_nn:
        return "nn_only"
    return "unknown"


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
