from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich import print

from kagglebot.bootstrap import bootstrap_competition
from kagglebot.history import RunLedger, SubmissionLedger
from kagglebot.paths import CompetitionPaths, repo_root
from kagglebot.validation import ensure_not_duplicate_submission, validate_submission

app = typer.Typer(add_completion=False, help="Kaggle competition automation CLI (safe by default).")


@app.command()
def bootstrap(
    slug: str = typer.Argument(..., help="Competition slug (URL suffix)."),
    force: bool = typer.Option(False, "--force", help="Overwrite the local config if it exists."),
):
    """
    Prepare workspace directories and write a config file.
    Does not join competitions or perform network actions.
    """
    config_path = bootstrap_competition(slug=slug, force=force)
    print(f"[green]bootstrap complete[/green]: {config_path}")


@app.command()
def run(
    slug: str = typer.Argument(..., help="Competition slug (URL suffix)."),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Default to dry-run (no side effects)."),
    force: bool = typer.Option(
        False,
        "--force",
        help="Allow side effects (submission recording or future network actions).",
    ),
    submission: Path | None = typer.Option(None, "--submission", help="Path to submission CSV to validate and record."),
    sample: Path | None = typer.Option(
        None, "--sample", help="Path to sample_submission.csv (defaults to data/<slug>/raw)."
    ),
    message: str = typer.Option("manual run", "--message", help="Message for ledger entries."),
):
    """
    Safety-first run skeleton:
    - bootstrap (workspace dirs + config)
    - record run metadata
    - optionally validate submission vs sample_submission.csv
    - record submission ledger only when --no-dry-run and --force
    """
    config_path = bootstrap_competition(slug=slug, force=force)
    paths = CompetitionPaths(slug=slug, repo_root=repo_root())

    sample_path = None
    if submission is not None:
        sample_path = sample if sample is not None else paths.data_raw / "sample_submission.csv"

    run_ledger = RunLedger.for_slug(slug)
    run_record = run_ledger.start_run(
        slug=slug,
        dry_run=dry_run,
        force=force,
        submission_path=str(submission) if submission else None,
        sample_path=str(sample_path) if sample_path else None,
        message=message,
        argv=list(sys.argv),
    )
    print(f"[green]run started[/green]: {run_record.run_id}")
    print(f"[green]workspace[/green]: {config_path}")

    ledger = SubmissionLedger.for_slug(slug)
    if submission is not None:
        if sample_path is None:
            raise RuntimeError("Sample path resolution failed.")
        if not sample_path.exists():
            raise FileNotFoundError(f"Missing sample submission at {sample_path}")
        validate_submission(str(sample_path), str(submission))
        ensure_not_duplicate_submission(ledger, str(submission))
        print(f"[green]submission validated[/green]: {submission}")

    if dry_run:
        print("[yellow]DRY RUN[/yellow]: no side effects. Use --no-dry-run --force to proceed.")
        return

    if not force:
        print("[red]Refusing to perform side effects without --force.[/red]")
        raise typer.Exit(code=2)

    if submission is None:
        print("[yellow]No submission provided; nothing to record.[/yellow]")
        return

    ledger.record(str(submission), message=message, run_id=run_record.run_id)
    print("[green]submission recorded[/green]")


if __name__ == "__main__":
    app()
