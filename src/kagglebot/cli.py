from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import typer
from rich import print

from kagglebot.agents.codex_runner import run_codex
from kagglebot.agents.identity import ORACLE_IMPLEMENTATION_AGENT
from kagglebot.agents.strategy_runner import run_strategy
from kagglebot.autopilot import AutopilotConfig, run_autopilot
from kagglebot.autopilot_state import ResumeRunResolutionError, resolve_resume_run_id
from kagglebot.bootstrap import bootstrap_competition
from kagglebot.campaign import normalize_campaign_mode
from kagglebot.competition import parse_competition_slug
from kagglebot.competition_submission_formats import crawl_submission_formats
from kagglebot.compute import Compute, resolve_accelerator
from kagglebot.env_utils import env_flag
from kagglebot.eval import EvaluationAdvisor
from kagglebot.event_notifications import EventDispatcher, dispatch_events_once
from kagglebot.exceptions import KaggleBotError, RulesNotAcceptedError, SubmitAbortedError
from kagglebot.experiment_graph import normalize_portfolio_execution
from kagglebot.hardware import resolve_hardware_profile
from kagglebot.history import new_run_id
from kagglebot.kaggle_api import check_rules_accepted
from kagglebot.kernel_runner import resolve_kaggle_username, run_kernel, run_kernel_local
from kagglebot.knowledge import knowledge_search, knowledge_show
from kagglebot.method_scout import normalize_method_scout_mode, normalize_research_scout_mode
from kagglebot.paths import CompetitionPaths, KnowledgePaths, resolve_artifacts_dir
from kagglebot.plan_policy import resolve_default_metric_from_artifacts
from kagglebot.score_sources import normalize_generalizable_score_source
from kagglebot.self_improvement import SelfImprovementConfig, run_self_improvement_cycle
from kagglebot.solver.metrics import infer_direction
from kagglebot.submission_artifacts import store_submission_artifact
from kagglebot.submission_policy import normalize_watch_submit_policy
from kagglebot.submission_service import SubmissionConfig, SubmissionService
from kagglebot.supervisor import WatchConfig, run_watch_forever, run_watch_once, run_watch_self_improvement
from kagglebot.top1_exhaustive import normalize_top1_submit_policy
from kagglebot.validation_lab import normalize_validation_lab_mode
from kagglebot.verify_artifacts import run_repo_verify

app = typer.Typer(add_completion=False, help="Kaggle competition automation CLI (safe by default).")
knowledge_app = typer.Typer(add_completion=False, help="Knowledge base commands.")
app.add_typer(knowledge_app, name="knowledge")

ARTIFACTS_DIR_ENV = "KAGGLEBOT_ARTIFACTS_DIR"
DEFAULT_ARTIFACTS_DIR = Path("artifacts")
DEFAULT_KAGGLE_GPU_MIN_QUOTA_HOURS = 15.0


@dataclass(frozen=True)
class AppContext:
    workdir: Path
    artifacts_dir: Path
    dry_run: bool
    interactive: bool
    log_level: str
    force: bool


def _sidecar_min_gpu_quota_minutes(
    *,
    min_gpu_quota_hours_for_new_comp: float | None,
    max_total_min: int | None,
    time_budget_min: int | None,
    max_iterations: int,
) -> int | None:
    if min_gpu_quota_hours_for_new_comp is not None:
        if min_gpu_quota_hours_for_new_comp <= 0:
            return None
        return int(min_gpu_quota_hours_for_new_comp * 60)
    return int(DEFAULT_KAGGLE_GPU_MIN_QUOTA_HOURS * 60)


def _normalize_self_improvement_interval(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    return float(value)


def _preferred_artifacts_dir() -> Path:
    configured = os.environ.get(ARTIFACTS_DIR_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_ARTIFACTS_DIR


@app.callback()
def main(
    ctx: typer.Context,
    artifacts_dir: Path | None = typer.Option(
        None,
        "--artifacts-dir",
        help="Artifacts directory. Defaults to ./artifacts; override with KAGGLEBOT_ARTIFACTS_DIR.",
    ),
    workdir: Path = typer.Option(Path("."), "--workdir", help="Working directory."),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run", help="Skip external side effects."),
    interactive: bool = typer.Option(
        False,
        "--interactive/--non-interactive",
        help="Enable interactive prompts (default: non-interactive).",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level."),
    force: bool = typer.Option(False, "--force", help="Allow external side effects (downloads/submissions)."),
) -> None:
    resolved_artifacts_dir = resolve_artifacts_dir(
        workdir.resolve(),
        artifacts_dir if artifacts_dir is not None else _preferred_artifacts_dir(),
    )
    ctx.obj = AppContext(
        workdir=workdir.resolve(),
        artifacts_dir=resolved_artifacts_dir,
        dry_run=dry_run,
        interactive=interactive,
        log_level=log_level,
        force=force,
    )


@app.command()
def bootstrap(
    ctx: typer.Context,
    competition: str = typer.Argument(..., help="Competition URL or slug."),
    download: bool = typer.Option(False, "--download/--no-download", help="Download competition data."),
    rules_source: str = typer.Option("url", "--rules-source", help="Rules capture source: none, url, file."),
    rules_file: Path | None = typer.Option(None, "--rules-file", help="Rules file path when rules-source=file."),
    quiet: bool = typer.Option(True, "--quiet/--no-quiet", help="Use --quiet for Kaggle CLI download."),
) -> None:
    cfg = ctx.obj
    slug = parse_competition_slug(competition)
    paths = CompetitionPaths(slug=slug, artifacts_dir=cfg.artifacts_dir, repo_root=cfg.workdir)
    knowledge_paths = KnowledgePaths(workdir=cfg.workdir)

    meta_path = bootstrap_competition(
        slug=slug,
        competition_url=competition if "kaggle.com" in competition else None,
        paths=paths,
        knowledge_paths=knowledge_paths,
        rules_source=rules_source,
        rules_file=rules_file,
        download=download,
        quiet=quiet,
        force=cfg.force,
        dry_run=cfg.dry_run,
        download_progress_callback=_print_download_progress,
    )
    print(f"[green]bootstrap complete[/green]: {meta_path}")


@app.command()
def implement(
    ctx: typer.Context,
    competition: str = typer.Argument(..., help="Competition URL or slug."),
    verify_cmd: str = typer.Option("uv run pytest -q", "--verify-cmd", help="Verification command."),
) -> None:
    cfg = ctx.obj
    slug = parse_competition_slug(competition)
    paths = CompetitionPaths(slug=slug, artifacts_dir=cfg.artifacts_dir, repo_root=cfg.workdir)
    knowledge_paths = KnowledgePaths(workdir=cfg.workdir)
    bootstrap_competition(
        slug=slug,
        competition_url=competition if "kaggle.com" in competition else None,
        paths=paths,
        knowledge_paths=knowledge_paths,
        download=False,
        force=cfg.force,
        dry_run=cfg.dry_run,
    )

    run_id = new_run_id()
    run_dir = paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    agent_dir = run_dir / "codex"
    prompt_path = paths.codex_plan_and_implement_prompt
    strategy_dir = run_dir / "oracle"
    strategy_prompt_path = run_dir / "oracle_implementation_prompt.md"
    implementation_prompt = prompt_path.read_text(encoding="utf-8")
    strategy_prompt_path.write_text(
        "Review the implementation request below without editing files. Return a concrete implementation brief "
        "for Codex, including risks, exact files, and tests.\n\n" + implementation_prompt,
        encoding="utf-8",
    )
    strategy_result = run_strategy(strategy_prompt_path, strategy_dir, dry_run=cfg.dry_run, engine="oracle")
    strategy_text = strategy_result.stdout.strip()
    if not cfg.dry_run and (strategy_result.returncode != 0 or not strategy_text):
        raise KaggleBotError("Oracle implementation strategy is required before Codex implementation.")
    mediated_prompt_path = run_dir / "codex_implementation_prompt.md"
    mediated_prompt_path.write_text(
        implementation_prompt
        + "\n\n## Oracle implementation brief\n"
        + (strategy_text or "DRY RUN: Oracle strategy not executed.")
        + "\n",
        encoding="utf-8",
    )

    run_codex(
        mediated_prompt_path,
        agent_dir,
        dry_run=cfg.dry_run,
        model=ORACLE_IMPLEMENTATION_AGENT.model,
        reasoning_effort=ORACLE_IMPLEMENTATION_AGENT.reasoning_effort,
        reasoning_profile=ORACLE_IMPLEMENTATION_AGENT.reasoning_profile,
        cli_profile=ORACLE_IMPLEMENTATION_AGENT.cli_profile,
        cwd=cfg.workdir,
    )

    run_repo_verify(verify_cmd, dry_run=cfg.dry_run, artifacts_dir=cfg.artifacts_dir)
    print(f"[green]agent logs[/green]: {agent_dir}")


@app.command()
def train(
    ctx: typer.Context,
    competition: str = typer.Argument(..., help="Competition URL or slug."),
    compute: Compute = typer.Option(Compute.local_gpu, "--compute", help="Compute target for training."),
    accelerator: str = typer.Option("auto", "--accelerator", help="Accelerator: auto, gpu, tpu."),
    kaggle_username: str | None = typer.Option(None, "--kaggle-username", help="Kaggle username for kernels."),
    kernel_name: str | None = typer.Option(None, "--kernel-name", help="Kernel slug override."),
    internet: str = typer.Option("off", "--internet", help="Kernel internet: auto|off|on."),
    time_budget_min: int | None = typer.Option(1440, "--time-budget-min", help="Time budget in minutes."),
    seed: int | None = typer.Option(None, "--seed", help="Random seed."),
    hardware_profile: str | None = typer.Option(
        "auto",
        "--hardware-profile",
        help="Runtime profile for kernel env: auto, rtx3060, rtx5090, kaggle_p100, kaggle_rtx_pro_6000, etc.",
    ),
    strict_accelerator: bool = typer.Option(False, "--strict-accelerator", help="Fail if GPU unavailable."),
) -> None:
    cfg = ctx.obj
    slug = parse_competition_slug(competition)
    paths = CompetitionPaths(slug=slug, artifacts_dir=cfg.artifacts_dir, repo_root=cfg.workdir)
    knowledge_paths = KnowledgePaths(workdir=cfg.workdir)
    bootstrap_competition(
        slug=slug,
        competition_url=competition if "kaggle.com" in competition else None,
        paths=paths,
        knowledge_paths=knowledge_paths,
        download=False,
        force=cfg.force,
        dry_run=cfg.dry_run,
    )

    resolved_accelerator = _resolve_accelerator(compute.value, accelerator)
    try:
        resolve_hardware_profile(hardware_profile, compute=compute.value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--hardware-profile") from exc
    resolved_time_budget = time_budget_min if time_budget_min is not None else 60
    resolved_seed = seed if seed is not None else 42
    metric = resolve_default_metric_from_artifacts(paths)
    direction = infer_direction(metric, "auto")
    run_id = new_run_id()
    run_dir = paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    submission_path: Path | None = None

    if compute.value.startswith("kaggle_"):
        if not cfg.force and not cfg.dry_run:
            raise typer.BadParameter("Refusing to run Kaggle kernel without --force.")
        kaggle_user = resolve_kaggle_username(kaggle_username)
        try:
            kernel_result = run_kernel(
                slug=slug,
                run_id=run_id,
                iteration=0,
                base_dir=paths.base_dir.parent,
                kaggle_username=kaggle_user,
                kernel_name=kernel_name,
                accelerator=resolved_accelerator,
                enable_internet=internet == "on",
                score_source="holdout",
                metric=metric,
                direction=direction,
                holdout_frac=0.2,
                cv_folds=5,
                seed=resolved_seed,
                dry_run=cfg.dry_run,
                timeout_minutes=resolved_time_budget,
                hardware_profile=hardware_profile,
            )
        except RulesNotAcceptedError:
            _print_rules(slug)
            raise typer.Exit(code=2)
        if kernel_result.submission_path:
            submission_path = store_submission_artifact(
                source=kernel_result.submission_path,
                destination_dir=paths.submissions_dir,
                run_id=run_id,
            )
    else:
        kernel_path = paths.kernel_source_dir / "kernel.py"
        if not kernel_path.exists():
            raise typer.BadParameter(f"Local training now requires kernel.py, but it was not found: {kernel_path}")
        kernel_result = run_kernel_local(
            slug=slug,
            run_id=run_id,
            iteration=0,
            base_dir=paths.base_dir.parent,
            accelerator=resolved_accelerator,
            score_source="holdout",
            metric=metric,
            direction=direction,
            holdout_frac=0.2,
            cv_folds=5,
            seed=resolved_seed,
            dry_run=cfg.dry_run,
            timeout_minutes=resolved_time_budget,
            strict_accelerator=strict_accelerator,
            hardware_profile=hardware_profile,
            plan_path=paths.plan_path,
        )
        if kernel_result.submission_path:
            submission_path = store_submission_artifact(
                source=kernel_result.submission_path,
                destination_dir=paths.submissions_dir,
                run_id=run_id,
            )
        if kernel_result.metrics_path and kernel_result.metrics_path.exists():
            metrics_path = run_dir / "metrics.json"
            metrics_path.write_bytes(kernel_result.metrics_path.read_bytes())

    if submission_path is None:
        raise typer.BadParameter("Training completed but no submission artifact was produced.")
    validation_service = SubmissionService(
        SubmissionConfig(
            slug=slug,
            data_dir=paths.data_dir,
            sample_submission_path=paths.sample_submission_path,
            submission_ledger_path=paths.submission_ledger_path,
            dry_run=cfg.dry_run,
            force_submit=True,
            bypass_rate_limit=True,
        )
    )
    submission_path = validation_service.validate_and_prepare_submission(submission_path)
    print(f"[green]submission written[/green]: {submission_path}")


@app.command()
def submit(
    ctx: typer.Context,
    competition: str = typer.Argument(..., help="Competition URL or slug."),
    file: Path = typer.Option(..., "-f", "--file", help="Submission file path."),
    message: str = typer.Option(..., "-m", "--message", help="Submission message."),
    force_submit: bool = typer.Option(False, "--force-submit", "--force", help="Allow duplicate submissions."),
    out_of_band: bool = typer.Option(
        False,
        "--out-of-band",
        help="Record this submit outside the normal run/iteration ledger linkage.",
    ),
    submission_kind: str | None = typer.Option(
        None,
        "--submission-kind",
        help="Optional ledger label, e.g. external_test_label_transfer.",
    ),
) -> None:
    cfg = ctx.obj
    slug = parse_competition_slug(competition)
    paths = CompetitionPaths(slug=slug, artifacts_dir=cfg.artifacts_dir, repo_root=cfg.workdir)
    if not file.exists():
        raise typer.BadParameter(f"Submission file not found: {file}")

    if not (cfg.force or force_submit) and not cfg.dry_run:
        raise typer.BadParameter("Refusing to submit without --force or --force-submit.")

    if not check_rules_accepted(slug, dry_run=cfg.dry_run):
        _print_rules(slug)
        raise typer.Exit(code=2)

    submission_service = SubmissionService(
        SubmissionConfig(
            slug=slug,
            data_dir=paths.data_dir,
            sample_submission_path=paths.sample_submission_path,
            submission_ledger_path=paths.submission_ledger_path,
            dry_run=cfg.dry_run,
            force_submit=force_submit,
        )
    )
    recorded_kind = submission_kind
    if out_of_band and not recorded_kind:
        recorded_kind = "out_of_band_manual"
    submission_service.submit(
        submission_path=file,
        message=message,
        run_id=None,
        submission_kind=recorded_kind,
        out_of_band=out_of_band,
    )
    print("[green]submission complete[/green]")


@app.command()
def autopilot(
    ctx: typer.Context,
    competition: str = typer.Argument(..., help="Competition URL or slug."),
    compute: Compute = typer.Option(..., "--compute", help="Compute target."),
    rules_file: Path | None = typer.Option(None, "--rules-file", help="Path to rules file (md/txt/html)."),
    target_metric: str | None = typer.Option(None, "--target-metric", help="Target metric override."),
    target_score: float | None = typer.Option(None, "--target-score", help="Target score override."),
    target_direction: str | None = typer.Option(None, "--target-direction", help="minimize|maximize|auto"),
    score_source: str | None = typer.Option(None, "--score-source", help="holdout|cv"),
    holdout_frac: float | None = typer.Option(None, "--holdout-frac", help="Holdout fraction."),
    cv_folds: int | None = typer.Option(None, "--cv-folds", help="CV folds."),
    max_iterations: int | None = typer.Option(
        None,
        "--max-iterations",
        "--max-iteration",
        "--max-iter",
        "--iter",
        help="Max iterations.",
    ),
    max_total_min: int | None = typer.Option(None, "--max-total-min", help="Max total minutes."),
    patience: int | None = typer.Option(None, "--patience", help="Patience iterations."),
    min_improvement: float | None = typer.Option(None, "--min-improvement", help="Minimum improvement."),
    force_submit: bool = typer.Option(False, "--force-submit", help="Allow duplicate submissions."),
    accelerator: str = typer.Option("auto", "--accelerator", help="auto|gpu|tpu"),
    kaggle_username: str | None = typer.Option(None, "--kaggle-username", help="Kaggle username."),
    kernel_name: str | None = typer.Option(None, "--kernel-name", help="Kernel name override."),
    internet: str | None = typer.Option("on", "--internet", help="auto|off|on"),
    time_budget_min: int | None = typer.Option(1440, "--time-budget-min", help="Time budget in minutes."),
    seed: int | None = typer.Option(None, "--seed", help="Random seed."),
    verify_cmd: str = typer.Option("uv run pytest -q", "--verify-cmd", help="Verification command."),
    message: str | None = typer.Option(None, "-m", "--message", help="Submission message override."),
    campaign_mode: str = typer.Option(
        "top1",
        "--campaign-mode",
        help="Campaign behavior for leaderboard competitions: top1 or baseline.",
    ),
    method_scout: str = typer.Option(
        "auto",
        "--method-scout",
        help="Competition-specific method discovery: auto, off, refresh.",
    ),
    research_scout: str = typer.Option(
        "auto",
        "--research-scout",
        help="Research/source registry ingestion: auto, off, refresh.",
    ),
    method_scout_max_sources: int = typer.Option(
        12,
        "--method-scout-max-sources",
        help="Maximum research sources/query seeds to use when building method registry.",
    ),
    portfolio_execution: str = typer.Option(
        "serial",
        "--portfolio-execution",
        help="Top1 portfolio execution mode: off, serial, parallel, or budgeted.",
    ),
    validation_lab: str = typer.Option(
        "auto",
        "--validation-lab",
        help="Validation redesign lab: auto, off, or force.",
    ),
    candidate_budget_min: int | None = typer.Option(
        None,
        "--candidate-budget",
        "--candidate-budget-min",
        help="Per-candidate budget hint in minutes for budgeted portfolio execution.",
    ),
    max_candidates_per_iteration: int | None = typer.Option(
        None,
        "--max-candidates-per-iteration",
        help="Maximum portfolio candidates to run per iteration when supported.",
    ),
    top1_exhaustive: bool = typer.Option(
        False,
        "--top1-exhaustive/--no-top1-exhaustive",
        help="Enable exhaustive top1 campaign artifacts and safe search defaults.",
    ),
    top1_submit_policy: str = typer.Option(
        "value_only",
        "--top1-submit-policy",
        help="Top1 portfolio submit ranking policy: value_only, calibration, or final_lock.",
    ),
    hardware_profile: str | None = typer.Option(
        "auto",
        "--hardware-profile",
        help="Runtime planning profile: auto, rtx3060, rtx5090, kaggle_p100, kaggle_rtx_pro_6000, etc.",
    ),
    strict_accelerator: bool = typer.Option(False, "--strict-accelerator", help="Fail if GPU unavailable."),
    auto_eval_spec: bool = typer.Option(
        True,
        "--auto-eval-spec/--no-auto-eval-spec",
        help="Run GPT-5.5 advisor once to generate/freeze context/evaluation_spec.json (default: on).",
    ),
    resume_run_id: str | None = typer.Option(None, "--resume-run-id", help="Resume an existing run by run ID."),
    resume_latest: bool = typer.Option(False, "--resume-latest", help="Resume the most recent run."),
) -> None:
    cfg = ctx.obj
    if not cfg.force and not cfg.dry_run:
        raise typer.BadParameter(
            "Refusing to run autopilot downloads, remote kernels, or submissions without --force. "
            "Use --dry-run for a side-effect-free preview.",
            param_hint="--force",
        )
    slug = parse_competition_slug(competition)
    paths = CompetitionPaths(slug=slug, artifacts_dir=cfg.artifacts_dir, repo_root=cfg.workdir)
    knowledge_paths = KnowledgePaths(workdir=cfg.workdir)

    resolved_accelerator = _resolve_accelerator(compute.value, accelerator)
    try:
        requested_resume_id = resolve_resume_run_id(
            paths=paths,
            resume_run_id=resume_run_id,
            resume_latest=resume_latest,
        )
    except ResumeRunResolutionError as exc:
        raise typer.BadParameter(str(exc), param_hint=exc.param_hint) from exc
    resume_id = os.environ.get("KAGGLEBOT_RESUME_RUN_ID")
    resume_slug = os.environ.get("KAGGLEBOT_RESUME_SLUG")
    resume_run = bool(resume_id and resume_slug == slug)
    if requested_resume_id is not None:
        if resume_run and resume_id != requested_resume_id:
            raise typer.BadParameter(
                "Autofix resume context conflicts with requested resume run ID. Retry without --resume-run-id.",
                param_hint="--resume-run-id",
            )
        if not resume_run:
            os.environ["KAGGLEBOT_RESUME_RUN_ID"] = requested_resume_id
            os.environ["KAGGLEBOT_RESUME_SLUG"] = slug
            resume_id = requested_resume_id
            resume_run = True
        print(f"[yellow]resume[/yellow]: requested run {requested_resume_id}")

    if resume_run and paths.context_dir.exists():
        print("[yellow]resume[/yellow]: skipping bootstrap; reusing existing context")
    else:
        if cfg.dry_run:
            print(f"[yellow]DRY RUN[/yellow]: would download data to {paths.data_dir}")
        else:
            print(f"[cyan]downloading data[/cyan]: {paths.data_dir}")

        rules_source = "file" if rules_file else "url"
        bootstrap_competition(
            slug=slug,
            competition_url=competition if "kaggle.com" in competition else None,
            paths=paths,
            knowledge_paths=knowledge_paths,
            rules_source=rules_source,
            rules_file=rules_file,
            download=not cfg.dry_run,
            force=False,
            dry_run=cfg.dry_run,
            download_progress_callback=_print_download_progress,
        )
        if not cfg.dry_run:
            print(f"[green]download complete[/green]: {paths.data_dir}")

    if auto_eval_spec:
        advisor = EvaluationAdvisor(
            paths=paths,
            slug=slug,
            dry_run=cfg.dry_run,
            force=env_flag("KAGGLEBOT_REFRESH_EVALUATION_SPEC", default=False),
        )
        spec, source = advisor.ensure_spec()
        metric_name = spec.get("metric_name")
        split_strategy = spec.get("split_strategy")
        print(
            "[cyan]evaluation advisor[/cyan]: "
            f"{source} -> {advisor.spec_path} "
            f"(metric={metric_name}, split={split_strategy})"
        )

    if score_source is not None:
        try:
            score_source = normalize_generalizable_score_source(score_source)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--score-source") from exc
    try:
        normalized_campaign_mode = normalize_campaign_mode(campaign_mode, deliverable_mode="leaderboard")
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--campaign-mode") from exc
    try:
        normalized_method_scout = normalize_method_scout_mode(method_scout)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--method-scout") from exc
    try:
        normalized_research_scout = normalize_research_scout_mode(research_scout)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--research-scout") from exc
    if method_scout_max_sources <= 0:
        raise typer.BadParameter(
            "--method-scout-max-sources must be positive.",
            param_hint="--method-scout-max-sources",
        )
    try:
        normalized_portfolio_execution = normalize_portfolio_execution(portfolio_execution)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--portfolio-execution") from exc
    try:
        normalized_validation_lab = normalize_validation_lab_mode(validation_lab)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--validation-lab") from exc
    if candidate_budget_min is not None and candidate_budget_min <= 0:
        raise typer.BadParameter("--candidate-budget must be positive.", param_hint="--candidate-budget")
    if max_candidates_per_iteration is not None and max_candidates_per_iteration <= 0:
        raise typer.BadParameter(
            "--max-candidates-per-iteration must be positive.",
            param_hint="--max-candidates-per-iteration",
        )
    try:
        normalized_top1_submit_policy = normalize_top1_submit_policy(top1_submit_policy)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--top1-submit-policy") from exc
    if top1_exhaustive:
        normalized_campaign_mode = "top1"
        if normalized_method_scout == "auto":
            normalized_method_scout = "refresh"
        if normalized_research_scout == "auto":
            normalized_research_scout = "refresh"
        if normalized_portfolio_execution == "serial":
            normalized_portfolio_execution = "budgeted"
        if normalized_validation_lab == "auto":
            normalized_validation_lab = "force"
        if candidate_budget_min is None:
            candidate_budget_min = 60
        if max_candidates_per_iteration is None:
            max_candidates_per_iteration = 3
    try:
        resolve_hardware_profile(hardware_profile, compute=compute.value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--hardware-profile") from exc

    run_id = None if resume_run else new_run_id()
    config = AutopilotConfig(
        run_id=run_id,
        slug=slug,
        competition_url=competition if "kaggle.com" in competition else None,
        paths=paths,
        knowledge_paths=knowledge_paths,
        agent="codex",
        compute=compute.value,
        accelerator=resolved_accelerator,
        strict_accelerator=strict_accelerator,
        kaggle_username=kaggle_username,
        kernel_name=kernel_name,
        internet=internet,
        time_budget_min=time_budget_min,
        seed=seed,
        score_source=score_source,
        holdout_frac=holdout_frac,
        cv_folds=cv_folds,
        target_metric=target_metric,
        target_score=target_score,
        target_direction=target_direction,
        max_iterations=max_iterations,
        max_total_min=max_total_min,
        patience=patience,
        min_improvement=min_improvement,
        submit=True,
        force_submit=force_submit,
        message=message,
        verify_cmd=verify_cmd,
        dry_run=cfg.dry_run,
        campaign_mode=normalized_campaign_mode,
        method_scout=normalized_method_scout,
        research_scout=normalized_research_scout,
        method_scout_max_sources=method_scout_max_sources,
        portfolio_execution=normalized_portfolio_execution,
        validation_lab=normalized_validation_lab,
        candidate_budget_min=candidate_budget_min,
        max_candidates_per_iteration=max_candidates_per_iteration,
        top1_exhaustive=top1_exhaustive,
        top1_submit_policy=normalized_top1_submit_policy,
        hardware_profile=hardware_profile,
    )
    try:
        run_autopilot(config)
    except RulesNotAcceptedError:
        _print_rules(slug)
        raise typer.Exit(code=2)
    except SubmitAbortedError as exc:
        print(f"[red]submit aborted[/red]: {exc}")
        raise typer.Exit(code=exc.exit_code)


@app.command()
def watch(
    ctx: typer.Context,
    once: bool = typer.Option(False, "--once", help="Run one select/autopilot cycle and exit."),
    compute: Compute = typer.Option(Compute.local_gpu, "--compute", help="Compute target."),
    accelerator: str = typer.Option("auto", "--accelerator", help="auto|gpu|tpu"),
    kaggle_username: str | None = typer.Option(None, "--kaggle-username", help="Kaggle username."),
    kernel_name: str | None = typer.Option(None, "--kernel-name", help="Kernel name override."),
    internet: str | None = typer.Option("on", "--internet", help="auto|off|on"),
    time_budget_min: int | None = typer.Option(
        1440,
        "--time-budget-min",
        help="Per-kernel time budget in minutes (default: 1440 for local GPU).",
    ),
    seed: int | None = typer.Option(None, "--seed", help="Random seed."),
    score_source: str | None = typer.Option(None, "--score-source", help="holdout|cv"),
    holdout_frac: float | None = typer.Option(None, "--holdout-frac", help="Holdout fraction."),
    cv_folds: int | None = typer.Option(None, "--cv-folds", help="CV folds."),
    max_iterations: int = typer.Option(5, "--max-iterations", min=1, help="Max autopilot iterations per competition."),
    max_total_min: int | None = typer.Option(
        None,
        "--max-total-min",
        min=1,
        help="Max minutes per competition. Omit for no wall-clock limit.",
    ),
    patience: int | None = typer.Option(None, "--patience", help="Patience iterations."),
    min_improvement: float | None = typer.Option(None, "--min-improvement", help="Minimum improvement."),
    submit_policy: str = typer.Option("improved", "--submit-policy", help="improved|none"),
    verify_cmd: str = typer.Option("uv run pytest -q", "--verify-cmd", help="Verification command."),
    auto_eval_spec: bool = typer.Option(
        True,
        "--auto-eval-spec/--no-auto-eval-spec",
        help="Generate/freeze context/evaluation_spec.json before autopilot.",
    ),
    page_limit: int = typer.Option(5, "--page-limit", min=1, help="Entered competition list page limit."),
    sleep_empty_sec: int = typer.Option(1800, "--sleep-empty-sec", min=1, help="Sleep when no candidates exist."),
    sleep_error_sec: int = typer.Option(300, "--sleep-error-sec", min=1, help="Sleep after skipped/failed cycles."),
    cooldown_hours: float = typer.Option(24.0, "--cooldown-hours", min=0.0, help="Cooldown after finish/failure."),
    allow_slug: list[str] | None = typer.Option(None, "--allow-slug", help="Only consider this slug; repeatable."),
    block_slug: list[str] | None = typer.Option(None, "--block-slug", help="Never consider this slug; repeatable."),
    strict_accelerator: bool = typer.Option(False, "--strict-accelerator", help="Fail if GPU unavailable."),
    self_improvement_interval_hours: float = typer.Option(
        6.0,
        "--self-improvement-interval-hours",
        min=0.0,
        help="Run Oracle/Codex self-improvement at most this often during watch. Use 0 to disable.",
    ),
    self_improvement_codex: bool = typer.Option(
        True,
        "--self-improvement-codex/--no-self-improvement-codex",
        help="Let periodic self-improvement ask Oracle/GPT for a brief and call Codex when the git worktree is clean.",
    ),
    self_improvement_publish: bool = typer.Option(
        True,
        "--self-improvement-publish/--no-self-improvement-publish",
        help="After Oracle/Codex self-improvement succeeds, verify, commit, and push repo code changes.",
    ),
    top1_exhaustive: bool = typer.Option(
        True,
        "--top1-exhaustive/--no-top1-exhaustive",
        help="Run watched competitions with exhaustive top1 campaign artifacts.",
    ),
    top1_submit_policy: str = typer.Option(
        "value_only",
        "--top1-submit-policy",
        help="Top1 portfolio submit ranking policy: value_only, calibration, or final_lock.",
    ),
    hardware_profile: str | None = typer.Option(
        "auto",
        "--hardware-profile",
        help="Runtime planning profile: auto, rtx3060, rtx5090, kaggle_p100, kaggle_rtx_pro_6000, etc.",
    ),
) -> None:
    cfg = ctx.obj
    try:
        normalized_submit_policy = normalize_watch_submit_policy(submit_policy)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--submit-policy") from exc
    if normalized_submit_policy != "none" and not cfg.force and not cfg.dry_run:
        raise typer.BadParameter("Refusing to run watch with submissions enabled without --force.")
    if self_improvement_codex and self_improvement_interval_hours > 0 and not cfg.force and not cfg.dry_run:
        raise typer.BadParameter("Refusing to run Oracle/Codex self-improvement without --force.")
    if self_improvement_publish and not self_improvement_codex:
        raise typer.BadParameter("--self-improvement-publish requires --self-improvement-codex.")
    try:
        normalized_top1_submit_policy = normalize_top1_submit_policy(top1_submit_policy)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--top1-submit-policy") from exc
    try:
        resolve_hardware_profile(hardware_profile, compute=compute.value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--hardware-profile") from exc

    watch_config = WatchConfig(
        workdir=cfg.workdir,
        artifacts_dir=cfg.artifacts_dir,
        compute=compute.value,
        accelerator=_resolve_accelerator(compute.value, accelerator),
        strict_accelerator=strict_accelerator,
        kaggle_username=kaggle_username,
        kernel_name=kernel_name,
        internet=internet,
        time_budget_min=time_budget_min,
        seed=seed,
        score_source=score_source,
        holdout_frac=holdout_frac,
        cv_folds=cv_folds,
        max_iterations=max_iterations,
        max_total_min=max_total_min,
        patience=patience,
        min_improvement=min_improvement,
        submit_policy=normalized_submit_policy,
        verify_cmd=verify_cmd,
        auto_eval_spec=auto_eval_spec,
        page_limit=page_limit,
        allow_slugs=tuple(allow_slug or ()),
        block_slugs=tuple(block_slug or ()),
        cooldown_hours=cooldown_hours,
        dry_run=cfg.dry_run,
        force=cfg.force,
        self_improvement_interval_hours=_normalize_self_improvement_interval(self_improvement_interval_hours),
        self_improvement_codex=self_improvement_codex,
        self_improvement_publish=self_improvement_publish,
        top1_exhaustive=top1_exhaustive,
        top1_submit_policy=normalized_top1_submit_policy,
        hardware_profile=hardware_profile,
    )
    if once:
        result = run_watch_once(watch_config)
        run_watch_self_improvement(watch_config)
        EventDispatcher(artifacts_dir=cfg.artifacts_dir).flush()
        print(f"[green]watch cycle[/green]: {result.status} slug={result.slug} run_id={result.run_id}")
        return
    run_watch_forever(
        watch_config,
        sleep_empty_sec=sleep_empty_sec,
        sleep_error_sec=sleep_error_sec,
    )


@app.command("watch-kaggle-gpu-sidecar")
def watch_kaggle_gpu_sidecar(
    ctx: typer.Context,
    once: bool = typer.Option(False, "--once", help="Run one Kaggle GPU sidecar cycle and exit."),
    interval_sec: int = typer.Option(1800, "--interval-sec", min=1, help="Sleep after no capacity/no candidates."),
    sleep_error_sec: int = typer.Option(300, "--sleep-error-sec", min=1, help="Sleep after skipped/failed cycles."),
    kaggle_username: str | None = typer.Option(None, "--kaggle-username", help="Kaggle username."),
    kernel_name: str | None = typer.Option(None, "--kernel-name", help="Kernel name override."),
    internet: str | None = typer.Option("on", "--internet", help="auto|off|on"),
    time_budget_min: int | None = typer.Option(600, "--time-budget-min", help="Per-kernel time budget in minutes."),
    seed: int | None = typer.Option(None, "--seed", help="Random seed."),
    score_source: str | None = typer.Option(None, "--score-source", help="holdout|cv"),
    holdout_frac: float | None = typer.Option(None, "--holdout-frac", help="Holdout fraction."),
    cv_folds: int | None = typer.Option(None, "--cv-folds", help="CV folds."),
    max_iterations: int = typer.Option(3, "--max-iterations", min=1, help="Max sidecar iterations per competition."),
    max_total_min: int | None = typer.Option(
        1800, "--max-total-min", min=1, help="Max minutes per sidecar competition."
    ),
    patience: int | None = typer.Option(2, "--patience", help="Patience iterations."),
    min_improvement: float | None = typer.Option(None, "--min-improvement", help="Minimum improvement."),
    submit_policy: str = typer.Option("improved", "--submit-policy", help="improved|none"),
    verify_cmd: str = typer.Option("uv run pytest -q", "--verify-cmd", help="Verification command."),
    auto_eval_spec: bool = typer.Option(
        True,
        "--auto-eval-spec/--no-auto-eval-spec",
        help="Generate/freeze context/evaluation_spec.json before autopilot.",
    ),
    page_limit: int = typer.Option(5, "--page-limit", min=1, help="Entered competition list page limit."),
    cooldown_hours: float = typer.Option(24.0, "--cooldown-hours", min=0.0, help="Cooldown after finish/failure."),
    max_data_gb: float = typer.Option(
        2.0,
        "--max-data-gb",
        min=0.01,
        help="Deprecated; lightweight selection is based on estimated training time.",
    ),
    max_training_min: int = typer.Option(
        600,
        "--max-training-min",
        min=1,
        help="Only run candidates with estimated training time at or below this many minutes.",
    ),
    min_gpu_quota_hours_for_new_comp: float | None = typer.Option(
        None,
        "--min-gpu-quota-hours-for-new-comp",
        min=0.0,
        help=(
            "Do not start a new Kaggle GPU competition unless at least this many GPU hours remain. "
            "Defaults to the existing Kaggle GPU floor (15h). Use 0 to disable."
        ),
    ),
    allow_slug: list[str] | None = typer.Option(None, "--allow-slug", help="Only consider this slug; repeatable."),
    block_slug: list[str] | None = typer.Option(None, "--block-slug", help="Never consider this slug; repeatable."),
    self_improvement_interval_hours: float = typer.Option(
        6.0,
        "--self-improvement-interval-hours",
        min=0.0,
        help="Run Oracle/Codex self-improvement at most this often during sidecar watch. Use 0 to disable.",
    ),
    self_improvement_codex: bool = typer.Option(
        True,
        "--self-improvement-codex/--no-self-improvement-codex",
        help="Let periodic self-improvement ask Oracle/GPT for a brief and call Codex when the git worktree is clean.",
    ),
    self_improvement_publish: bool = typer.Option(
        True,
        "--self-improvement-publish/--no-self-improvement-publish",
        help="After Oracle/Codex self-improvement succeeds, verify, commit, and push repo code changes.",
    ),
    hardware_profile: str | None = typer.Option(
        "kaggle_p100",
        "--hardware-profile",
        help="Runtime planning profile: kaggle_p100, kaggle_t4, kaggle_rtx_pro_6000, rtx3060, rtx5090, etc.",
    ),
) -> None:
    cfg = ctx.obj
    try:
        normalized_submit_policy = normalize_watch_submit_policy(submit_policy)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--submit-policy") from exc
    if normalized_submit_policy != "none" and not cfg.force and not cfg.dry_run:
        raise typer.BadParameter("Refusing to run Kaggle GPU sidecar with submissions enabled without --force.")
    if self_improvement_codex and self_improvement_interval_hours > 0 and not cfg.force and not cfg.dry_run:
        raise typer.BadParameter("Refusing to run Oracle/Codex self-improvement without --force.")
    if self_improvement_publish and not self_improvement_codex:
        raise typer.BadParameter("--self-improvement-publish requires --self-improvement-codex.")
    try:
        resolve_hardware_profile(hardware_profile, compute=Compute.kaggle_gpu.value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--hardware-profile") from exc

    watch_config = WatchConfig(
        workdir=cfg.workdir,
        artifacts_dir=cfg.artifacts_dir,
        compute=Compute.kaggle_gpu.value,
        accelerator="gpu",
        strict_accelerator=False,
        kaggle_username=kaggle_username,
        kernel_name=kernel_name,
        internet=internet,
        time_budget_min=time_budget_min,
        seed=seed,
        score_source=score_source,
        holdout_frac=holdout_frac,
        cv_folds=cv_folds,
        max_iterations=max_iterations,
        max_total_min=max_total_min,
        patience=patience,
        min_improvement=min_improvement,
        submit_policy=normalized_submit_policy,
        verify_cmd=verify_cmd,
        auto_eval_spec=auto_eval_spec,
        page_limit=page_limit,
        allow_slugs=tuple(allow_slug or ()),
        block_slugs=tuple(block_slug or ()),
        cooldown_hours=cooldown_hours,
        dry_run=cfg.dry_run,
        force=cfg.force,
        state_scope="kaggle_gpu",
        lightweight_only=True,
        lightweight_max_data_bytes=None,
        lightweight_max_training_min=max_training_min,
        kaggle_gpu_min_available_minutes_for_new_competition=_sidecar_min_gpu_quota_minutes(
            min_gpu_quota_hours_for_new_comp=min_gpu_quota_hours_for_new_comp,
            max_total_min=max_total_min,
            time_budget_min=time_budget_min,
            max_iterations=max_iterations,
        ),
        kaggle_gpu_quota_web_lookup=True,
        self_improvement_interval_hours=_normalize_self_improvement_interval(self_improvement_interval_hours),
        self_improvement_codex=self_improvement_codex,
        self_improvement_publish=self_improvement_publish,
        hardware_profile=hardware_profile,
    )
    if once:
        result = run_watch_once(watch_config)
        run_watch_self_improvement(watch_config)
        EventDispatcher(artifacts_dir=cfg.artifacts_dir).flush()
        print(f"[green]kaggle gpu sidecar[/green]: {result.status} slug={result.slug} run_id={result.run_id}")
        return
    run_watch_forever(
        watch_config,
        sleep_empty_sec=interval_sec,
        sleep_error_sec=sleep_error_sec,
    )


@app.command("self-improve")
def self_improve(
    ctx: typer.Context,
    max_runs: int = typer.Option(500, "--max-runs", min=1, help="Historical run count to analyze."),
    interval_hours: float = typer.Option(
        0.0,
        "--interval-hours",
        min=0.0,
        help="Skip if a newer report exists. Default 0 forces a manual run.",
    ),
    codex: bool = typer.Option(
        True,
        "--codex/--no-codex",
        help="Ask Oracle/GPT for an improvement brief, then call Codex to implement one structural improvement.",
    ),
    publish: bool = typer.Option(
        True,
        "--publish/--no-publish",
        help="After Oracle/Codex self-improvement succeeds, verify, commit, and push repo code changes.",
    ),
) -> None:
    cfg = ctx.obj
    if codex and not cfg.force and not cfg.dry_run:
        raise typer.BadParameter("Refusing to run Oracle/Codex self-improvement without --force.")
    if publish and not codex:
        raise typer.BadParameter("--publish requires --codex.")
    result = run_self_improvement_cycle(
        SelfImprovementConfig(
            artifacts_dir=cfg.artifacts_dir,
            knowledge_paths=KnowledgePaths(workdir=cfg.workdir),
            max_runs=max_runs,
            min_interval_hours=_normalize_self_improvement_interval(interval_hours),
            invoke_codex=codex,
            publish_codex_changes=publish,
            force=interval_hours <= 0,
            dry_run=cfg.dry_run,
        )
    )
    print(f"[green]self-improvement[/green]: {json.dumps(result, sort_keys=True)}")


@app.command("event-dispatch")
def event_dispatch(
    ctx: typer.Context,
    heartbeat_sec: int = typer.Option(1800, "--heartbeat-sec", min=1, help="Send unchanged status at this interval."),
    force: bool = typer.Option(False, "--force", help="Send even if the status snapshot has not changed."),
) -> None:
    """Flush the durable event outbox once; watch dispatches automatically."""
    cfg = ctx.obj
    dispatch_events_once(
        artifacts_dir=cfg.artifacts_dir,
        heartbeat_sec=heartbeat_sec,
        force=force,
    )


@app.command("crawl-submission-formats")
def crawl_submission_formats_cmd(
    ctx: typer.Context,
    output_dir: Path = typer.Option(
        Path("artifacts/competition-submission-formats"),
        "--output-dir",
        help="Directory for raw JSONL, normalized CSV, and summary outputs.",
    ),
    max_prefix_depth: int = typer.Option(
        1,
        "--max-prefix-depth",
        min=1,
        help="Adaptive Kaggle API search-prefix depth for discovering historical competitions.",
    ),
    max_pages_per_search: int = typer.Option(
        2,
        "--max-pages-per-search",
        min=1,
        help="Maximum Kaggle API pages to fetch per category/group/search prefix.",
    ),
    max_competitions: int | None = typer.Option(
        None,
        "--max-competitions",
        min=1,
        help="Optional cap on the number of competition pages to scrape.",
    ),
    fetch_rules_pages: bool = typer.Option(
        True,
        "--fetch-rules-pages/--no-fetch-rules-pages",
        help="Fetch each competition's rules page in addition to the overview page when needed.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Reuse existing raw JSONL output and skip already crawled competition slugs.",
    ),
    entered_only: bool = typer.Option(
        False,
        "--entered-only/--all-competitions",
        help="Audit only competitions joined by the authenticated Kaggle account.",
    ),
) -> None:
    cfg = ctx.obj
    summary = crawl_submission_formats(
        output_dir=resolve_artifacts_dir(cfg.workdir, output_dir),
        max_prefix_depth=max_prefix_depth,
        max_pages_per_search=max_pages_per_search,
        max_competitions=max_competitions,
        fetch_rules_pages=fetch_rules_pages,
        resume=resume,
        entered_only=entered_only,
    )
    print(
        "[green]crawl complete[/green]: "
        f"{summary['competition_count']} competitions, "
        f"{summary['review_required_count']} require review"
    )


@knowledge_app.command("show")
def knowledge_show_cmd(
    ctx: typer.Context,
    slug: str = typer.Argument(..., help="Competition slug."),
) -> None:
    cfg = ctx.obj
    knowledge_paths = KnowledgePaths(workdir=cfg.workdir)
    payload = knowledge_show(knowledge_paths, slug)
    print(json.dumps(payload, indent=2))


@knowledge_app.command("search")
def knowledge_search_cmd(
    ctx: typer.Context,
    tag: list[str] = typer.Option(..., "--tag", help="Tag filters."),
    limit: int = typer.Option(5, "--limit", help="Result limit."),
) -> None:
    cfg = ctx.obj
    knowledge_paths = KnowledgePaths(workdir=cfg.workdir)
    results = knowledge_search(knowledge_paths, tag, limit)
    print(json.dumps(results, indent=2))


def _print_download_progress(done_files: int, total_files: int, file_name: str | None) -> None:
    if total_files <= 0:
        return
    percent = (done_files / total_files) * 100.0
    detail = f" - {Path(file_name).name}" if file_name else ""
    print(f"[cyan]download progress[/cyan]: {done_files}/{total_files} ({percent:.1f}%){detail}")


def _resolve_accelerator(compute: str, accelerator: str) -> str:
    try:
        return resolve_accelerator(compute, accelerator)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--accelerator") from exc


def _print_rules(slug: str) -> None:
    print(f"[red]Rules not accepted[/red]. Visit: https://www.kaggle.com/competitions/{slug}/rules")
