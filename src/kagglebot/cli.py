from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich import print

from kagglebot.bootstrap import bootstrap_competition
from kagglebot.competition import parse_competition_slug, rules_url_for_slug
from kagglebot.history import RunLedger, SubmissionLedger
from kagglebot.kaggle_cli import KaggleCliError, RulesNotAcceptedError, download_competition, submit_competition
from kagglebot.paths import CompetitionPaths, repo_root
from kagglebot.tabular_baseline import train_and_make_submission
from kagglebot.validation import ensure_not_duplicate_submission, validate_submission

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
    time_budget_minutes: int = typer.Option(60, "--time-budget", help="Max training time in minutes."),
    models: str | None = typer.Option(None, "--models", help="Comma-separated model list."),
    cv_folds: int = typer.Option(5, "--cv-folds", help="Number of cross-validation folds."),
    no_stacking: bool = typer.Option(False, "--no-stacking", help="Disable stacking."),
) -> None:
    """Train a baseline model and write artifacts into artifacts/<slug>/."""
    _ = (time_budget_minutes, models, cv_folds, no_stacking)
    slug = _resolve_slug(competition)
    try:
        outputs = train_and_make_submission(slug)
    except Exception as exc:  # noqa: BLE001
        print(f"[red]training failed[/red]: {exc}")
        raise typer.Exit(code=EXIT_TRAINING_FAILED) from exc
    print(f"[green]model saved[/green]: {outputs.model_path}")
    print(f"[green]submission written[/green]: {outputs.submission}")


@app.command()
def predict(
    competition: str = typer.Argument(..., help="Competition URL or slug."),
) -> None:
    """Generate a submission from the latest trained baseline model."""
    slug = _resolve_slug(competition)
    try:
        outputs = train_and_make_submission(slug)
    except Exception as exc:  # noqa: BLE001
        print(f"[red]prediction failed[/red]: {exc}")
        raise typer.Exit(code=EXIT_TRAINING_FAILED) from exc
    print(f"[green]submission written[/green]: {outputs.submission}")


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
    if not force_submit:
        try:
            ensure_not_duplicate_submission(ledger, str(submission_path))
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

    ledger.record(str(submission_path), message=message, run_id=None)
    print("[green]submission recorded[/green]")


@app.command()
def run(
    competition: str = typer.Argument(..., help="Competition URL or slug."),
    submit: bool = typer.Option(False, "--submit/--no-submit", help="Submit after validation."),
    time_budget_minutes: int = typer.Option(60, "--time-budget", help="Max training time in minutes."),
    config: Path | None = typer.Option(None, "--config", help="Optional config path."),
    resume: str | None = typer.Option(None, "--resume", help="Resume from a previous run id."),
    models: str | None = typer.Option(None, "--models", help="Comma-separated model list."),
    cv_folds: int = typer.Option(5, "--cv-folds", help="Number of cross-validation folds."),
    no_stacking: bool = typer.Option(False, "--no-stacking", help="Disable stacking."),
    message: str | None = typer.Option(None, "--message", help="Submission message."),
    force: bool = typer.Option(False, "--force", help="Allow side effects beyond validation."),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Skip Kaggle download and submission."),
    force_submit: bool = typer.Option(False, "--force-submit", help="Allow duplicate submissions."),
) -> None:
    """
    Run the end-to-end pipeline: download → train → predict → validate → submit.
    """
    _ = (time_budget_minutes, config, resume, models, cv_folds, no_stacking)
    slug = _resolve_slug(competition)

    if submit and not message:
        print("[red]Submission requires --message.[/red]")
        raise typer.Exit(code=EXIT_SUBMISSION_FAILED)
    submit_message = message or ""

    paths = CompetitionPaths(slug=slug, repo_root=repo_root())
    bootstrap_competition(slug=slug, force=False)

    run_ledger = RunLedger.for_slug(slug)
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

    if not dry_run:
        if not force:
            _refuse_side_effect("download competition data")
        try:
            download_competition(slug, paths.data_raw, overwrite=False)
        except RulesNotAcceptedError:
            _print_rules_and_exit(slug)
        except KaggleCliError as exc:
            _print_kaggle_error(exc, action="download")
            raise typer.Exit(code=EXIT_DOWNLOAD_FAILED) from exc

    try:
        outputs = train_and_make_submission(slug)
    except Exception as exc:  # noqa: BLE001
        print(f"[red]training failed[/red]: {exc}")
        raise typer.Exit(code=EXIT_TRAINING_FAILED) from exc

    try:
        validate_submission(outputs.sample_submission, outputs.submission)
    except Exception as exc:  # noqa: BLE001
        print(f"[red]submission validation failed[/red]: {exc}")
        raise typer.Exit(code=EXIT_VALIDATION_FAILED) from exc
    print(f"[green]submission validated[/green]: {outputs.submission}")

    if not submit:
        print("[yellow]Submission skipped[/yellow]: pass --submit to upload.")
        return

    if dry_run:
        print("[yellow]DRY RUN[/yellow]: submission skipped.")
        return

    if not force:
        _refuse_side_effect("submit to Kaggle")

    ledger = SubmissionLedger.for_slug(slug)
    if not force_submit:
        try:
            ensure_not_duplicate_submission(ledger, outputs.submission)
        except ValueError as exc:
            print(f"[red]{exc}[/red]")
            raise typer.Exit(code=EXIT_DUPLICATE_SUBMISSION) from exc

    try:
        submit_competition(slug, Path(outputs.submission), submit_message)
    except RulesNotAcceptedError:
        _print_rules_and_exit(slug)
    except KaggleCliError as exc:
        _print_kaggle_error(exc, action="submit")
        raise typer.Exit(code=EXIT_SUBMISSION_FAILED) from exc

    ledger.record(outputs.submission, message=submit_message, run_id=run_record.run_id)
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


if __name__ == "__main__":
    app()
