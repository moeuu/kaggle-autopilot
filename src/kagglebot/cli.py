from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import typer
from rich import print

from kagglebot.agents.codex_runner import run_codex
from kagglebot.autopilot import AutopilotConfig, run_autopilot
from kagglebot.bootstrap import bootstrap_competition
from kagglebot.competition import parse_competition_slug
from kagglebot.compute import Compute
from kagglebot.exceptions import KaggleBotError, RulesNotAcceptedError
from kagglebot.exec_utils import run_command
from kagglebot.history import SubmissionLedger, new_run_id
from kagglebot.kaggle_api import check_rules_accepted, submit_competition
from kagglebot.kernel_runner import resolve_kaggle_username, run_kernel
from kagglebot.knowledge import knowledge_search, knowledge_show
from kagglebot.paths import CompetitionPaths, KnowledgePaths, resolve_artifacts_dir
from kagglebot.solver.baseline import train_evaluate_and_predict
from kagglebot.solver.metrics import infer_direction
from kagglebot.validation import ensure_not_duplicate_submission, ensure_submission_rate_limit, validate_submission

app = typer.Typer(add_completion=False, help="Kaggle competition automation CLI (safe by default).")
knowledge_app = typer.Typer(add_completion=False, help="Knowledge base commands.")
app.add_typer(knowledge_app, name="knowledge")


@dataclass(frozen=True)
class AppContext:
    workdir: Path
    artifacts_dir: Path
    dry_run: bool
    interactive: bool
    log_level: str
    force: bool


@app.callback()
def main(
    ctx: typer.Context,
    artifacts_dir: Path = typer.Option(Path("artifacts"), "--artifacts-dir", help="Artifacts directory."),
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
    ctx.obj = AppContext(
        workdir=workdir.resolve(),
        artifacts_dir=resolve_artifacts_dir(workdir.resolve(), artifacts_dir),
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
    rules_source: str = typer.Option("fetch", "--rules-source", help="Rules capture source: none, url, fetch, file."),
    rules_file: Path | None = typer.Option(None, "--rules-file", help="Rules file path when rules-source=file."),
    quiet: bool = typer.Option(True, "--quiet/--no-quiet", help="Use --quiet for Kaggle CLI download."),
) -> None:
    cfg = ctx.obj
    slug = parse_competition_slug(competition)
    paths = CompetitionPaths(slug=slug, artifacts_dir=cfg.artifacts_dir)
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
    )
    print(f"[green]bootstrap complete[/green]: {meta_path}")


@app.command()
def implement(
    ctx: typer.Context,
    competition: str = typer.Argument(..., help="Competition URL or slug."),
    agent: str = typer.Option(..., "--agent", help="Agent to run (codex or claude)."),
    verify_cmd: str = typer.Option("uv run pytest -q", "--verify-cmd", help="Verification command."),
) -> None:
    cfg = ctx.obj
    slug = parse_competition_slug(competition)
    paths = CompetitionPaths(slug=slug, artifacts_dir=cfg.artifacts_dir)
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
    agent_dir = run_dir / agent
    prompt_path = paths.codex_plan_and_baseline_prompt

    if agent != "codex":
        print("[yellow]agent override[/yellow]: using codex exec per safety policy.")
    run_codex(prompt_path, agent_dir, dry_run=cfg.dry_run)

    _run_verify(verify_cmd, cfg.dry_run)
    print(f"[green]agent logs[/green]: {agent_dir}")


@app.command()
def train(
    ctx: typer.Context,
    competition: str = typer.Argument(..., help="Competition URL or slug."),
    compute: Compute = typer.Option(Compute.local_cpu, "--compute", help="Compute target for training."),
    accelerator: str = typer.Option("auto", "--accelerator", help="Accelerator: auto, cpu, gpu, tpu."),
    kaggle_username: str | None = typer.Option(None, "--kaggle-username", help="Kaggle username for kernels."),
    kernel_name: str | None = typer.Option(None, "--kernel-name", help="Kernel slug override."),
    internet: str = typer.Option("off", "--internet", help="Kernel internet: auto|off|on."),
    time_budget_min: int | None = typer.Option(None, "--time-budget-min", help="Time budget in minutes."),
    seed: int | None = typer.Option(None, "--seed", help="Random seed."),
    strict_accelerator: bool = typer.Option(False, "--strict-accelerator", help="Fail if GPU unavailable."),
) -> None:
    cfg = ctx.obj
    slug = parse_competition_slug(competition)
    paths = CompetitionPaths(slug=slug, artifacts_dir=cfg.artifacts_dir)
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
    resolved_time_budget = time_budget_min if time_budget_min is not None else 60
    resolved_seed = seed if seed is not None else 42
    metric = _default_metric(paths)
    direction = infer_direction(metric, "auto")
    run_id = new_run_id()
    run_dir = paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    submission_path = paths.submissions_dir / f"{run_id}_submission.csv"

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
            )
        except RulesNotAcceptedError:
            _print_rules(slug)
            raise typer.Exit(code=2)
        if kernel_result.submission_path:
            submission_path.write_bytes(kernel_result.submission_path.read_bytes())
    else:
        outcome = train_evaluate_and_predict(
            data_dir=paths.data_dir,
            output_path=submission_path,
            compute=compute,
            strict_accelerator=strict_accelerator,
            seed=resolved_seed,
            score_source="holdout",
            metric=metric,
            direction="auto",
            holdout_frac=0.2,
            cv_folds=5,
            plan_score_source=None,
            target_override=None,
        )
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "metric": outcome.evaluation.metric,
                    "offline_value": outcome.evaluation.value,
                    "score_source": outcome.evaluation.score_source,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"[green]submission written[/green]: {submission_path}")


@app.command()
def submit(
    ctx: typer.Context,
    competition: str = typer.Argument(..., help="Competition URL or slug."),
    file: Path = typer.Option(..., "-f", "--file", help="Submission CSV path."),
    message: str = typer.Option(..., "-m", "--message", help="Submission message."),
    force_submit: bool = typer.Option(False, "--force-submit", help="Allow duplicate submissions."),
) -> None:
    cfg = ctx.obj
    slug = parse_competition_slug(competition)
    paths = CompetitionPaths(slug=slug, artifacts_dir=cfg.artifacts_dir)
    if not file.exists():
        raise typer.BadParameter(f"Submission file not found: {file}")

    if not cfg.force and not cfg.dry_run:
        raise typer.BadParameter("Refusing to submit without --force.")

    if not check_rules_accepted(slug, dry_run=cfg.dry_run):
        _print_rules(slug)
        raise typer.Exit(code=2)

    sample = paths.sample_submission_path
    if not sample.exists():
        from kagglebot.solver.io import find_competition_files

        _, _, sample = find_competition_files(paths.data_dir)

    validate_submission(str(sample), str(file))

    ledger = SubmissionLedger(paths.submission_ledger_path)
    ensure_submission_rate_limit(ledger)
    if not force_submit:
        ensure_not_duplicate_submission(ledger, slug=slug, message=message, submission_path=str(file))

    submit_competition(slug, file, message, dry_run=cfg.dry_run)
    ledger.record(slug=slug, message=message, submission_path=file, run_id=None)
    print("[green]submission complete[/green]")


@app.command()
def autopilot(
    ctx: typer.Context,
    competition: str = typer.Argument(..., help="Competition URL or slug."),
    agent: str = typer.Option(..., "--agent", help="Agent to run (codex or claude)."),
    compute: Compute = typer.Option(..., "--compute", help="Compute target."),
    submit: bool = typer.Option(False, "--submit/--no-submit", help="Submit when target met."),
    target_metric: str | None = typer.Option(None, "--target-metric", help="Target metric override."),
    target_score: float | None = typer.Option(None, "--target-score", help="Target score override."),
    target_direction: str | None = typer.Option(None, "--target-direction", help="minimize|maximize|auto"),
    score_source: str | None = typer.Option(None, "--score-source", help="auto|holdout|cv|test"),
    holdout_frac: float | None = typer.Option(None, "--holdout-frac", help="Holdout fraction."),
    cv_folds: int | None = typer.Option(None, "--cv-folds", help="CV folds."),
    max_iterations: int | None = typer.Option(None, "--max-iterations", help="Max iterations."),
    max_total_min: int | None = typer.Option(None, "--max-total-min", help="Max total minutes."),
    patience: int | None = typer.Option(None, "--patience", help="Patience iterations."),
    min_improvement: float | None = typer.Option(None, "--min-improvement", help="Minimum improvement."),
    force_submit: bool = typer.Option(False, "--force-submit", help="Allow duplicate submissions."),
    accelerator: str = typer.Option("auto", "--accelerator", help="auto|cpu|gpu|tpu"),
    kaggle_username: str | None = typer.Option(None, "--kaggle-username", help="Kaggle username."),
    kernel_name: str | None = typer.Option(None, "--kernel-name", help="Kernel name override."),
    internet: str | None = typer.Option(None, "--internet", help="auto|off|on"),
    time_budget_min: int | None = typer.Option(None, "--time-budget-min", help="Time budget in minutes."),
    seed: int | None = typer.Option(None, "--seed", help="Random seed."),
    verify_cmd: str = typer.Option("uv run pytest -q", "--verify-cmd", help="Verification command."),
    message: str | None = typer.Option(None, "-m", "--message", help="Submission message override."),
    strict_accelerator: bool = typer.Option(False, "--strict-accelerator", help="Fail if GPU unavailable."),
) -> None:
    cfg = ctx.obj
    slug = parse_competition_slug(competition)
    paths = CompetitionPaths(slug=slug, artifacts_dir=cfg.artifacts_dir)
    knowledge_paths = KnowledgePaths(workdir=cfg.workdir)

    resolved_accelerator = _resolve_accelerator(compute.value, accelerator)
    if cfg.dry_run:
        print(f"[yellow]DRY RUN[/yellow]: would download data to {paths.data_dir}")
    else:
        print(f"[cyan]downloading data[/cyan]: {paths.data_dir}")

    bootstrap_competition(
        slug=slug,
        competition_url=competition if "kaggle.com" in competition else None,
        paths=paths,
        knowledge_paths=knowledge_paths,
        rules_source="fetch",
        download=not cfg.dry_run,
        force=False,
        dry_run=cfg.dry_run,
    )
    if not cfg.dry_run:
        print(f"[green]download complete[/green]: {paths.data_dir}")

    run_id = new_run_id()
    config = AutopilotConfig(
        run_id=run_id,
        slug=slug,
        competition_url=competition if "kaggle.com" in competition else None,
        paths=paths,
        knowledge_paths=knowledge_paths,
        agent=agent,
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
        submit=submit,
        force_submit=force_submit,
        message=message,
        verify_cmd=verify_cmd,
        dry_run=cfg.dry_run,
    )
    try:
        run_autopilot(config)
    except RulesNotAcceptedError:
        _print_rules(slug)
        raise typer.Exit(code=2)


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


def _run_verify(cmd: str, dry_run: bool) -> None:
    if dry_run:
        return
    import shlex

    result = run_command(shlex.split(cmd))
    if result.returncode != 0:
        raise KaggleBotError(f"Verification failed: {result.output}")


def _resolve_accelerator(compute: str, accelerator: str) -> str:
    if accelerator == "auto":
        if compute == "local_cpu":
            return "cpu"
        if compute == "local_gpu":
            return "gpu"
        if compute == "kaggle_gpu":
            return "gpu"
        if compute == "kaggle_tpu":
            return "tpu"
    if compute == "local_cpu" and accelerator not in {"cpu"}:
        raise typer.BadParameter("--accelerator must be cpu for local_cpu.")
    if compute == "local_gpu" and accelerator not in {"gpu"}:
        raise typer.BadParameter("--accelerator must be gpu for local_gpu.")
    if compute == "kaggle_gpu" and accelerator not in {"gpu"}:
        raise typer.BadParameter("--accelerator must be gpu for kaggle_gpu.")
    if compute == "kaggle_tpu" and accelerator not in {"tpu"}:
        raise typer.BadParameter("--accelerator must be tpu for kaggle_tpu.")
    return accelerator


def _print_rules(slug: str) -> None:
    print(f"[red]Rules not accepted[/red]. Visit: https://www.kaggle.com/competitions/{slug}/rules")


def _default_metric(paths: CompetitionPaths) -> str:
    profile_path = paths.dataset_profile_path
    if profile_path.exists():
        try:
            data = json.loads(profile_path.read_text(encoding="utf-8"))
            metric = data.get("metric")
            if isinstance(metric, str):
                return metric
        except json.JSONDecodeError:
            pass
    return "rmse"
