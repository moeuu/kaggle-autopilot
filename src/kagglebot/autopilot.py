from __future__ import annotations

import json
import shlex
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from kagglebot.agents.codex_runner import run_codex
from kagglebot.compute import Compute
from kagglebot.exceptions import RulesNotAcceptedError
from kagglebot.exec_utils import run_command
from kagglebot.history import SubmissionLedger, new_run_id
from kagglebot.kaggle_api import check_rules_accepted, leaderboard_top1, submit_competition
from kagglebot.kernel_runner import resolve_kaggle_username, run_kernel
from kagglebot.knowledge import record_improvement, record_iteration, record_run
from kagglebot.solver.baseline import train_evaluate_and_predict
from kagglebot.solver.metrics import infer_direction
from kagglebot.types import PlanConfig
from kagglebot.validation import ensure_not_duplicate_submission, ensure_submission_rate_limit, validate_submission

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


def run_autopilot(config: AutopilotConfig) -> None:
    start_time = time.monotonic()
    run_id = config.run_id or new_run_id()
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    plan = _load_plan(config.paths)
    if not config.paths.plan_path.exists():
        _write_plan(config.paths, plan)

    top1_info = leaderboard_top1(config.slug, config.paths.context_dir, dry_run=config.dry_run)
    config.paths.top1_public_path.write_text(json.dumps(top1_info, indent=2), encoding="utf-8")

    if _needs_planning(plan, config):
        _run_plan_and_baseline(config, run_id)
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
    best_score = None
    no_improve = 0

    last_submission: Path | None = None
    max_iterations = int(resolved["max_iterations"])
    max_total_min = int(resolved["max_total_min"])
    patience = int(resolved["patience"])
    min_improvement = float(resolved["min_improvement"])
    holdout_frac = float(resolved["holdout_frac"])
    cv_folds = int(resolved["cv_folds"])
    seed = int(resolved["seed"])
    score_source = str(resolved["score_source"] or "auto")
    kernel_name = resolved["kernel_name"]
    enable_internet = str(resolved["internet"]) == "on"

    for iteration in range(1, max_iterations + 1):
        iter_dir = config.paths.iter_dir(run_id, iteration)
        logs_dir = iter_dir / "logs"
        agent_dir = iter_dir / "agent"
        iter_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        agent_dir.mkdir(parents=True, exist_ok=True)

        if _time_exceeded(start_time, max_total_min):
            break

        _run_verify(config.verify_cmd, dry_run=config.dry_run)

        submission_path = iter_dir / "submission.csv"
        evaluation = None
        model_summary = {}
        accelerator_used = config.accelerator

        if config.compute.startswith("kaggle_"):
            kaggle_user = resolve_kaggle_username(config.kaggle_username)
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
                timeout_minutes=resolved["time_budget_min"] or max_total_min,
            )
            if kernel_result.submission_path:
                submission_path.write_bytes(kernel_result.submission_path.read_bytes())
            if kernel_result.metrics_path and kernel_result.metrics_path.exists():
                evaluation = _load_kernel_metrics(kernel_result.metrics_path, metric_direction)
        else:
            compute_enum = Compute(config.compute)
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
        last_submission = submission_path

        met_target = _meets_target(
            evaluation.value,
            target_score,
            metric_direction,
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
        )
        (iter_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

        diagnostics = _build_diagnostics(
            evaluation=evaluation,
            model_summary=model_summary,
            best_score=best_score,
            target_score=target_score,
            dataset_profile=dataset_profile,
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
        improved = _update_best_score(best_score, evaluation.value, metric_direction, min_improvement)
        if improved:
            best_score = evaluation.value
            no_improve = 0
        else:
            no_improve += 1
        delta_offline = None
        if prev_best is not None:
            delta_offline = (
                prev_best - evaluation.value if metric_direction == "minimize" else evaluation.value - prev_best
            )

        if met_target:
            _attempt_submit(
                config=config,
                run_id=run_id,
                submission_path=submission_path,
                best_score=best_score or evaluation.value,
            )
            break

        if no_improve >= patience:
            break

        if iteration >= max_iterations:
            break

        if _time_exceeded(start_time, max_total_min):
            break

        _run_improvement(
            config=config,
            run_id=run_id,
            iteration=iteration,
            iter_dir=iter_dir,
            delta_offline=delta_offline,
        )

    if _submit_at_final(resolved["submit_policy"]) and config.submit and not config.dry_run and last_submission:
        if last_submission.exists():
            _attempt_submit(config=config, run_id=run_id, submission_path=last_submission, best_score=best_score)


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
    target_metric = config.target_metric or plan.target_metric
    target_score = config.target_score if config.target_score is not None else plan.target_score
    target_direction = config.target_direction or plan.target_direction
    if target_metric is None or target_score is None:
        return True
    return target_direction in (None, "auto")


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
    time_budget_min = choose(config.time_budget_min, plan.time_budget_min, 60)
    kernel_name = choose(config.kernel_name, plan.kernel_name, None)
    internet = choose(config.internet, plan.internet, "auto")
    if internet in (None, "auto"):
        internet = "off"
    max_iterations = choose(config.max_iterations, plan.max_iterations, 5)
    max_total_min = choose(config.max_total_min, plan.max_total_min, 240)
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
        internet=str(resolved.get("internet") or "auto"),
        max_iterations=int(resolved.get("max_iterations") or 5),
        max_total_min=int(resolved.get("max_total_min") or 240),
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


def _submit_at_final(policy: str | None) -> bool:
    value = (policy or "").lower()
    return value in {"submit_at_final", "force_at_final", "allow_final"}


def _load_dataset_profile(paths: CompetitionPaths) -> dict[str, object]:
    if not paths.dataset_profile_path.exists():
        return {}
    try:
        return json.loads(paths.dataset_profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _run_verify(verify_cmd: str, *, dry_run: bool) -> None:
    if dry_run:
        return
    args = shlex.split(verify_cmd)
    result = run_command(args)
    if result.returncode != 0:
        raise RuntimeError(f"Verification failed: {result.output}")


def _run_plan_and_baseline(config: AutopilotConfig, run_id: str) -> None:
    iter_dir = config.paths.iter_dir(run_id, 0)
    agent_dir = iter_dir / "agent"
    iter_dir.mkdir(parents=True, exist_ok=True)
    agent_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = config.paths.codex_plan_and_baseline_prompt.read_text(encoding="utf-8")
    prompt_path = agent_dir / "prompt.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    if config.dry_run:
        return
    result = run_codex(prompt_path, agent_dir, dry_run=False)
    if result.returncode != 0:
        raise RuntimeError("Codex planning/baseline step failed.")
    _run_verify(config.verify_cmd, dry_run=config.dry_run)


def _load_kernel_metrics(metrics_path: Path, direction: str):
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    from kagglebot.solver.evaluate import EvaluationResult

    return EvaluationResult(
        score_source=payload.get("score_source", "holdout"),
        metric=payload.get("metric", "rmse"),
        direction=direction,  # type: ignore[arg-type]
        value=float(payload.get("offline_value", 0.0)),
        std=payload.get("offline_std"),
        train_score=None,
        val_score=None,
        fold_scores=None,
    )


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
) -> str:
    direction = evaluation.direction
    delta_to_target = (
        target_score - evaluation.value if direction == "minimize" else evaluation.value - target_score
    )
    best_line = best_score if best_score is not None else evaluation.value
    trend = (
        "improving"
        if best_score is None or _meets_target(evaluation.value, best_line, direction)
        else "stalled"
    )
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
    ]
    return "\n".join(lines) + "\n"


def _run_improvement(
    config: AutopilotConfig,
    run_id: str,
    iteration: int,
    iter_dir: Path,
    delta_offline: float | None,
) -> None:
    prompt_template = config.paths.codex_improve_template.read_text(encoding="utf-8")
    agent_dir = iter_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = agent_dir / "prompt.md"
    prompt_path.write_text(
        prompt_template.format(
            slug=config.slug,
            iteration=iteration,
            plan_path=str(config.paths.plan_path),
            run_path=str(config.paths.run_dir(run_id) / "run.json"),
            metrics_path=str(iter_dir / "metrics.json"),
            diagnostics_path=str(iter_dir / "diagnostics.md"),
            logs_dir=str(iter_dir / "logs"),
        ),
        encoding="utf-8",
    )

    result = run_codex(prompt_path, agent_dir, dry_run=config.dry_run)
    if result.returncode != 0:
        raise RuntimeError("Codex improvement failed.")

    _run_verify(config.verify_cmd, dry_run=config.dry_run)
    summary = result.last_message_path.read_text(encoding="utf-8")
    record_improvement(
        knowledge_paths=config.knowledge_paths,
        run_id=run_id,
        iteration=iteration,
        summary=summary.strip(),
        delta_offline=delta_offline,
    )


def _attempt_submit(*, config: AutopilotConfig, run_id: str, submission_path: Path, best_score: float | None) -> None:
    if not config.submit or config.dry_run:
        return
    if not check_rules_accepted(config.slug, dry_run=config.dry_run):
        raise RulesNotAcceptedError("Competition rules not accepted.")
    sample_path = config.paths.sample_submission_path
    if not sample_path.exists():
        from kagglebot.solver.io import find_competition_files

        _, _, sample_path = find_competition_files(config.paths.data_dir)
    validate_submission(str(sample_path), str(submission_path))
    ledger = SubmissionLedger(config.paths.submission_ledger_path)
    ensure_submission_rate_limit(ledger)
    if not config.force_submit:
        ensure_not_duplicate_submission(
            ledger,
            slug=config.slug,
            message=_submission_message(config, run_id, best_score),
            submission_path=str(submission_path),
        )
    submit_competition(
        config.slug, submission_path, _submission_message(config, run_id, best_score), dry_run=config.dry_run
    )
    ledger.record(
        slug=config.slug,
        message=_submission_message(config, run_id, best_score),
        submission_path=submission_path,
        run_id=run_id,
    )


def _submission_message(config: AutopilotConfig, run_id: str, best_score: float | None) -> str:
    if config.message:
        return config.message
    if best_score is None:
        return f"kagglebot {config.slug} {run_id}"
    return f"kagglebot {config.slug} {run_id} best_offline={best_score:.6f}"


def _meets_target(value: float, target: float, direction: str) -> bool:
    if direction == "minimize":
        return value <= target
    return value >= target


def _update_best_score(best: float | None, current: float, direction: str, min_improvement: float) -> bool:
    if best is None:
        return True
    if direction == "minimize":
        return (best - current) >= min_improvement
    return (current - best) >= min_improvement


def _time_exceeded(start_time: float, max_total_min: int) -> bool:
    elapsed = (time.monotonic() - start_time) / 60
    return elapsed >= max_total_min
