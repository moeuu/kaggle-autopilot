from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import typer
from rich import print

from kagglebot.agents.codex_runner import run_codex
from kagglebot.autopilot import AutopilotConfig, run_autopilot
from kagglebot.bootstrap import bootstrap_competition
from kagglebot.competition import parse_competition_slug
from kagglebot.compute import Compute
from kagglebot.eval import EvaluationAdvisor
from kagglebot.exceptions import KaggleBotError, RulesNotAcceptedError, SubmitAbortedError
from kagglebot.exec_utils import run_command
from kagglebot.history import new_run_id
from kagglebot.kaggle_api import check_rules_accepted
from kagglebot.kernel_runner import resolve_kaggle_username, run_kernel, run_kernel_local
from kagglebot.knowledge import knowledge_search, knowledge_show
from kagglebot.paths import CompetitionPaths, KnowledgePaths, resolve_artifacts_dir
from kagglebot.solver.io import find_competition_files
from kagglebot.solver.metrics import infer_direction
from kagglebot.submission.validate import validate_submission
from kagglebot.submission_service import SubmissionConfig, SubmissionService

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
    rules_source: str = typer.Option("url", "--rules-source", help="Rules capture source: none, url, file."),
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
    agent_dir = run_dir / "codex"
    prompt_path = paths.codex_plan_and_implement_prompt

    run_codex(prompt_path, agent_dir, dry_run=cfg.dry_run)

    _run_verify(verify_cmd, cfg.dry_run)
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
        )
        if kernel_result.submission_path:
            submission_path.write_bytes(kernel_result.submission_path.read_bytes())
        if kernel_result.metrics_path and kernel_result.metrics_path.exists():
            metrics_path = run_dir / "metrics.json"
            metrics_path.write_bytes(kernel_result.metrics_path.read_bytes())

    _, _, sample_path = find_competition_files(paths.data_dir)
    validate_submission(str(submission_path), str(sample_path))
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
    submission_service.submit(submission_path=file, message=message, run_id=None)
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
    score_source: str | None = typer.Option(None, "--score-source", help="auto|holdout|cv|test"),
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
    time_budget_min: int | None = typer.Option(None, "--time-budget-min", help="Time budget in minutes."),
    seed: int | None = typer.Option(None, "--seed", help="Random seed."),
    verify_cmd: str = typer.Option("uv run pytest -q", "--verify-cmd", help="Verification command."),
    message: str | None = typer.Option(None, "-m", "--message", help="Submission message override."),
    strict_accelerator: bool = typer.Option(False, "--strict-accelerator", help="Fail if GPU unavailable."),
    auto_eval_spec: bool = typer.Option(
        True,
        "--auto-eval-spec/--no-auto-eval-spec",
        help="Run GPT-5.2 advisor once to generate/freeze context/evaluation_spec.json (default: on).",
    ),
    resume_run_id: str | None = typer.Option(None, "--resume-run-id", help="Resume an existing run by run ID."),
    resume_latest: bool = typer.Option(False, "--resume-latest", help="Resume the most recent run."),
) -> None:
    cfg = ctx.obj
    slug = parse_competition_slug(competition)
    paths = CompetitionPaths(slug=slug, artifacts_dir=cfg.artifacts_dir)
    knowledge_paths = KnowledgePaths(workdir=cfg.workdir)

    resolved_accelerator = _resolve_accelerator(compute.value, accelerator)
    requested_resume_id = _resolve_resume_run_id(
        paths=paths,
        resume_run_id=resume_run_id,
        resume_latest=resume_latest,
    )
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
        )
        if not cfg.dry_run:
            print(f"[green]download complete[/green]: {paths.data_dir}")

    if auto_eval_spec:
        advisor = EvaluationAdvisor(
            paths=paths,
            slug=slug,
            dry_run=cfg.dry_run,
            force=cfg.force,
        )
        spec, source = advisor.ensure_spec()
        metric_name = spec.get("metric_name")
        split_strategy = spec.get("split_strategy")
        print(
            "[cyan]evaluation advisor[/cyan]: "
            f"{source} -> {advisor.spec_path} "
            f"(metric={metric_name}, split={split_strategy})"
        )

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
    )
    try:
        run_autopilot(config)
    except RulesNotAcceptedError:
        _print_rules(slug)
        raise typer.Exit(code=2)
    except SubmitAbortedError as exc:
        print(f"[red]submit aborted[/red]: {exc}")
        raise typer.Exit(code=exc.exit_code)


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
        if compute == "local_gpu":
            return "gpu"
        if compute == "kaggle_gpu":
            return "gpu"
        if compute == "kaggle_tpu":
            return "tpu"
    if compute == "local_gpu" and accelerator not in {"gpu"}:
        raise typer.BadParameter("--accelerator must be gpu for local_gpu.")
    if compute == "kaggle_gpu" and accelerator not in {"gpu"}:
        raise typer.BadParameter("--accelerator must be gpu for kaggle_gpu.")
    if compute == "kaggle_tpu" and accelerator not in {"tpu"}:
        raise typer.BadParameter("--accelerator must be tpu for kaggle_tpu.")
    return accelerator


def _resolve_resume_run_id(
    *,
    paths: CompetitionPaths,
    resume_run_id: str | None,
    resume_latest: bool,
) -> str | None:
    if resume_run_id and resume_latest:
        raise typer.BadParameter(
            "Use either --resume-run-id or --resume-latest, not both.",
            param_hint="--resume-run-id",
        )
    if resume_run_id:
        candidate = resume_run_id.strip()
        if not candidate:
            raise typer.BadParameter("--resume-run-id cannot be empty.", param_hint="--resume-run-id")
        if not paths.run_dir(candidate).exists():
            raise typer.BadParameter(f"Run ID not found: {candidate}", param_hint="--resume-run-id")
        return candidate
    if not resume_latest:
        return None
    latest = _find_latest_run_id(paths)
    if latest is None:
        raise typer.BadParameter(f"No prior runs found under {paths.runs_dir}", param_hint="--resume-latest")
    return latest


def _find_latest_run_id(paths: CompetitionPaths) -> str | None:
    runs_dir = paths.runs_dir
    if not runs_dir.exists():
        return None
    latest_name: str | None = None
    latest_mtime: float | None = None
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        try:
            mtime = run_dir.stat().st_mtime
        except OSError:
            continue
        if latest_mtime is None or mtime > latest_mtime:
            latest_name = run_dir.name
            latest_mtime = mtime
    return latest_name


def _print_rules(slug: str) -> None:
    print(f"[red]Rules not accepted[/red]. Visit: https://www.kaggle.com/competitions/{slug}/rules")


def _default_metric(paths: CompetitionPaths) -> str:
    plan_path = paths.plan_path
    if plan_path.exists():
        try:
            data = json.loads(plan_path.read_text(encoding="utf-8"))
            metric = data.get("target_metric")
            if isinstance(metric, str) and metric.strip():
                return metric
        except json.JSONDecodeError:
            pass

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
