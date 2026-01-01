from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich import print

from kagglebot.analyzer import UnsupportedCompetitionError, analyze_competition
from kagglebot.bootstrap import bootstrap_competition
from kagglebot.competition import parse_competition_slug, rules_url_for_slug
from kagglebot.compute import Compute, compute_to_runner_and_accelerator
from kagglebot.history import RunLedger, SubmissionLedger
from kagglebot.kaggle_cli import KaggleCliError, RulesNotAcceptedError, download_competition, submit_competition
from kagglebot.paths import CompetitionPaths, repo_root
from kagglebot.runners import KaggleNotebookRunner, LocalRunner, RunContext
from kagglebot.training import predict_tabular, train_tabular
from kagglebot.validation import ensure_not_duplicate_submission, ensure_submission_rate_limit, validate_submission

EXIT_RULES_NOT_ACCEPTED = 2
EXIT_INVALID_COMPETITION = 3
EXIT_DOWNLOAD_FAILED = 4
EXIT_TRAINING_FAILED = 5
EXIT_VALIDATION_FAILED = 6
EXIT_SUBMISSION_FAILED = 7
EXIT_DUPLICATE_SUBMISSION = 8

app = typer.Typer(add_completion=False, help="Kaggle competition automation CLI (safe by default).")


@app.command()
def bootstrap(
    competition: str = typer.Argument(..., help="Competition URL or slug."),
    force: bool = typer.Option(False, "--force", help="Overwrite the local config if it exists."),
) -> None:
    """
    Prepare workspace directories and write a config file.
    Does not join competitions or perform network actions.
    """
    slug = _resolve_slug(competition)
    config_path = bootstrap_competition(slug=slug, force=force)
    print(f"[green]bootstrap complete[/green]: {config_path}")


@app.command()
def download(
    competition: str = typer.Argument(..., help="Competition URL or slug."),
    force: bool = typer.Option(False, "--force", help="Allow Kaggle CLI download."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Force re-download even if files exist."),
) -> None:
    """Download competition data via Kaggle CLI into data/<slug>/raw."""
    slug = _resolve_slug(competition)
    if not force:
        _refuse_side_effect("download competition data")
    paths = CompetitionPaths(slug=slug, repo_root=repo_root())
    bootstrap_competition(slug=slug, force=False)
    try:
        download_competition(slug, paths.data_raw, overwrite=overwrite)
    except RulesNotAcceptedError:
        _print_rules_and_exit(slug)
    except KaggleCliError as exc:
        _print_kaggle_error(exc, action="download")
        raise typer.Exit(code=EXIT_DOWNLOAD_FAILED)
    print(f"[green]download complete[/green]: {paths.data_raw}")


@app.command()
def train(
    competition: str = typer.Argument(..., help="Competition URL or slug."),
    time_budget_minutes: int = typer.Option(
        60,
        "--time-budget",
        "--time-budget-min",
        help="Max training time in minutes.",
    ),
    models: str | None = typer.Option(None, "--models", help="Comma-separated model list."),
    cv_folds: int = typer.Option(5, "--cv-folds", help="Number of cross-validation folds."),
    no_stacking: bool = typer.Option(False, "--no-stacking", help="Disable stacking."),
) -> None:
    """Train a baseline model and write artifacts into artifacts/<slug>/."""
    slug = _resolve_slug(competition)
    model_list = _parse_models(models)
    paths = CompetitionPaths(slug=slug, repo_root=repo_root())
    bootstrap_competition(slug=slug, force=False)
    try:
        analysis = analyze_competition(
            slug=slug,
            paths=paths,
            time_budget_minutes=time_budget_minutes,
            cv_folds=cv_folds,
            models=model_list,
            use_stacking=not no_stacking,
        )
    except UnsupportedCompetitionError as exc:
        print(f"[red]unsupported competition[/red]: {exc}")
        raise typer.Exit(code=EXIT_INVALID_COMPETITION) from exc
    except Exception as exc:  # noqa: BLE001
        print(f"[red]analysis failed[/red]: {exc}")
        raise typer.Exit(code=EXIT_TRAINING_FAILED) from exc
    try:
        result = train_tabular(
            analysis.metadata,
            paths=paths,
            time_budget_minutes=time_budget_minutes,
            model_names=model_list,
            cv_folds=cv_folds,
            accelerator="none",
            strict_accelerator=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[red]training failed[/red]: {exc}")
        raise typer.Exit(code=EXIT_TRAINING_FAILED) from exc
    print(f"[green]analysis saved[/green]: {analysis.analysis_path}")
    print(f"[green]model saved[/green]: {result.model_path}")
    print(f"[green]report saved[/green]: {result.report_path}")


@app.command()
def predict(
    competition: str = typer.Argument(..., help="Competition URL or slug."),
) -> None:
    """Generate a submission from the latest trained baseline model."""
    slug = _resolve_slug(competition)
    paths = CompetitionPaths(slug=slug, repo_root=repo_root())
    bootstrap_competition(slug=slug, force=False)
    try:
        analysis = analyze_competition(
            slug=slug,
            paths=paths,
            time_budget_minutes=60,
            cv_folds=5,
            models=None,
            use_stacking=False,
        )
        submission_path = predict_tabular(analysis.metadata, paths=paths)
    except UnsupportedCompetitionError as exc:
        print(f"[red]unsupported competition[/red]: {exc}")
        raise typer.Exit(code=EXIT_INVALID_COMPETITION) from exc
    except Exception as exc:  # noqa: BLE001
        print(f"[red]prediction failed[/red]: {exc}")
        raise typer.Exit(code=EXIT_TRAINING_FAILED) from exc
    print(f"[green]submission written[/green]: {submission_path}")


@app.command()
def submit(
    competition: str = typer.Argument(..., help="Competition URL or slug."),
    message: str = typer.Option(..., "--message", help="Submission message."),
    force: bool = typer.Option(False, "--force", help="Allow Kaggle CLI submission."),
    force_submit: bool = typer.Option(False, "--force-submit", help="Allow duplicate submissions."),
) -> None:
    """Validate and submit artifacts/<slug>/submissions/submission.csv to Kaggle."""
    slug = _resolve_slug(competition)
    if not force:
        _refuse_side_effect("submit to Kaggle")
    paths = CompetitionPaths(slug=slug, repo_root=repo_root())
    submission_path = paths.submission_csv
    sample_path = paths.data_raw / "sample_submission.csv"

    try:
        validate_submission(str(sample_path), str(submission_path))
    except Exception as exc:  # noqa: BLE001
        print(f"[red]submission validation failed[/red]: {exc}")
        raise typer.Exit(code=EXIT_VALIDATION_FAILED) from exc

    ledger = SubmissionLedger.for_slug(slug)
    ensure_submission_rate_limit(ledger)
    if not force_submit:
        try:
            ensure_not_duplicate_submission(ledger, str(submission_path), slug=slug, message=message)
        except ValueError as exc:
            print(f"[red]{exc}[/red]")
            raise typer.Exit(code=EXIT_DUPLICATE_SUBMISSION) from exc

    try:
        submit_competition(slug, submission_path, message)
    except RulesNotAcceptedError:
        _print_rules_and_exit(slug)
    except KaggleCliError as exc:
        _print_kaggle_error(exc, action="submit")
        raise typer.Exit(code=EXIT_SUBMISSION_FAILED) from exc

    ledger.record(str(submission_path), message=message, run_id=None, slug=slug)
    print("[green]submission recorded[/green]")


@app.command()
def run(
    competition: str = typer.Argument(..., help="Competition URL or slug."),
    submit: bool = typer.Option(False, "--submit/--no-submit", help="Submit after validation."),
    compute: Compute = typer.Option(Compute.local_cpu, "--compute", help="Compute target for training."),
    enable_internet: bool = typer.Option(False, "--enable-internet", help="Enable internet in Kaggle notebook."),
    kaggle_username: str | None = typer.Option(
        None,
        "--kaggle-username",
        help="Kaggle username for Kaggle notebook kernels (defaults to config/env).",
    ),
    workdir: Path | None = typer.Option(
        None,
        "--workdir",
        help="Base working directory for artifacts/data (default: current working directory).",
    ),
    time_budget_minutes: int = typer.Option(
        60,
        "--time-budget",
        "--time-budget-min",
        help="Max training time in minutes.",
    ),
    config: Path | None = typer.Option(None, "--config", help="Optional config path."),
    resume: str | None = typer.Option(None, "--resume", help="Resume from a previous run id."),
    models: str | None = typer.Option(None, "--models", help="Comma-separated model list."),
    cv_folds: int = typer.Option(5, "--cv-folds", help="Number of cross-validation folds."),
    no_stacking: bool = typer.Option(False, "--no-stacking", help="Disable stacking."),
    message: str | None = typer.Option(None, "--message", help="Submission message."),
    force: bool = typer.Option(False, "--force", help="Allow side effects beyond validation."),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Skip Kaggle CLI operations."),
    force_submit: bool = typer.Option(False, "--force-submit", help="Allow duplicate submissions."),
    strict_accelerator: bool = typer.Option(
        False,
        "--strict-accelerator",
        help="Fail if requested accelerator is unavailable (local_gpu).",
    ),
) -> None:
    """
    Run the end-to-end pipeline: download → train → predict → validate → submit.
    """
    _ = (config, resume)
    slug = _resolve_slug(competition)
    model_list = _parse_models(models)

    if submit and not message:
        print("[red]Submission requires --message.[/red]")
        raise typer.Exit(code=EXIT_SUBMISSION_FAILED)
    submit_message = message or ""

    base_root = workdir if workdir is not None else repo_root()
    paths = CompetitionPaths(slug=slug, repo_root=base_root)
    bootstrap_competition(slug=slug, force=False, root=base_root)

    run_ledger = RunLedger.for_slug(slug, root=base_root)
    run_record = run_ledger.start_run(
        slug=slug,
        dry_run=dry_run,
        force=force,
        submission_path=str(paths.submission_csv),
        sample_path=str(paths.data_raw / "sample_submission.csv"),
        message=message,
        argv=list(sys.argv),
    )
    print(f"[green]run started[/green]: {run_record.run_id}")

    selection = compute_to_runner_and_accelerator(compute)
    if selection.runner == "kaggle_notebook" and not dry_run and not force:
        _refuse_side_effect("execute Kaggle notebook runner")

    context = RunContext(
        competition=competition,
        slug=slug,
        run_id=run_record.run_id,
        paths=paths,
        workdir=base_root,
        dry_run=dry_run,
        submit=submit,
        force=force,
        force_submit=force_submit,
        message=submit_message,
        time_budget_minutes=time_budget_minutes,
        cv_folds=cv_folds,
        model_names=model_list,
        use_stacking=not no_stacking,
        compute=compute.value,
        accelerator=selection.accelerator,
        enable_internet=enable_internet,
        kaggle_username=kaggle_username,
        strict_accelerator=strict_accelerator,
    )

    runner_impl = LocalRunner() if selection.runner == "local" else KaggleNotebookRunner()
    try:
        result = runner_impl.run(context)
    except UnsupportedCompetitionError as exc:
        print(f"[red]unsupported competition[/red]: {exc}")
        raise typer.Exit(code=EXIT_INVALID_COMPETITION) from exc
    except RulesNotAcceptedError:
        _print_rules_and_exit(slug)
    except KaggleCliError as exc:
        _print_kaggle_error(exc, action="run")
        raise typer.Exit(code=EXIT_DOWNLOAD_FAILED) from exc
    except Exception as exc:  # noqa: BLE001
        print(f"[red]run failed[/red]: {exc}")
        raise typer.Exit(code=EXIT_TRAINING_FAILED) from exc

    submission_path = result.submission_path
    if submission_path is None:
        print("[yellow]no submission produced[/yellow]: dry-run or kernel output missing.")
        return

    sample_path = paths.data_raw / "sample_submission.csv"
    if not sample_path.exists():
        if dry_run:
            print("[yellow]sample_submission.csv missing[/yellow]: skipping validation in dry-run.")
            return
        if not force:
            _refuse_side_effect("download competition data for validation")
        try:
            download_competition(slug, paths.data_raw, overwrite=False)
        except RulesNotAcceptedError:
            _print_rules_and_exit(slug)
        except KaggleCliError as exc:
            _print_kaggle_error(exc, action="download")
            raise typer.Exit(code=EXIT_DOWNLOAD_FAILED) from exc

    try:
        validate_submission(str(sample_path), str(submission_path))
    except Exception as exc:  # noqa: BLE001
        print(f"[red]submission validation failed[/red]: {exc}")
        raise typer.Exit(code=EXIT_VALIDATION_FAILED) from exc
    print(f"[green]submission validated[/green]: {submission_path}")

    if not submit:
        print("[yellow]Submission skipped[/yellow]: pass --submit to upload.")
        return

    if dry_run:
        print("[yellow]DRY RUN[/yellow]: submission skipped.")
        return

    if not force:
        _refuse_side_effect("submit to Kaggle")

    ledger = SubmissionLedger.for_slug(slug, root=base_root)
    ensure_submission_rate_limit(ledger)
    if not force_submit:
        try:
            ensure_not_duplicate_submission(ledger, str(submission_path), slug=slug, message=submit_message)
        except ValueError as exc:
            print(f"[red]{exc}[/red]")
            raise typer.Exit(code=EXIT_DUPLICATE_SUBMISSION) from exc

    try:
        submit_competition(slug, Path(submission_path), submit_message)
    except RulesNotAcceptedError:
        _print_rules_and_exit(slug)
    except KaggleCliError as exc:
        _print_kaggle_error(exc, action="submit")
        raise typer.Exit(code=EXIT_SUBMISSION_FAILED) from exc

    ledger.record(str(submission_path), message=submit_message, run_id=run_record.run_id, slug=slug)
    print("[green]submission recorded[/green]")


def _resolve_slug(competition: str) -> str:
    try:
        return parse_competition_slug(competition)
    except ValueError as exc:
        print(f"[red]Invalid competition[/red]: {exc}")
        raise typer.Exit(code=EXIT_INVALID_COMPETITION) from exc


def _refuse_side_effect(action: str) -> None:
    print(f"[red]Refusing to {action} without --force.[/red]")
    raise typer.Exit(code=1)


def _print_rules_and_exit(slug: str) -> None:
    print("[red]Competition rules not accepted.[/red]")
    print(f"Visit: {rules_url_for_slug(slug)}")
    raise typer.Exit(code=EXIT_RULES_NOT_ACCEPTED)


def _print_kaggle_error(exc: KaggleCliError, action: str) -> None:
    print(f"[red]Kaggle CLI {action} failed[/red]: {exc.message}")
    if exc.output:
        print(exc.output)


def _parse_models(models: str | None) -> list[str] | None:
    if not models:
        return None
    cleaned = [m.strip().lower() for m in models.split(",") if m.strip()]
    return cleaned or None


if __name__ == "__main__":
    app()
