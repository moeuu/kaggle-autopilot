from __future__ import annotations

import typer
from rich import print

from kagglebot.bootstrap import bootstrap_competition
from kagglebot.history import SubmissionHistory
from kagglebot.kaggle_cli import kaggle_submit
from kagglebot.tabular_baseline import train_and_make_submission
from kagglebot.validation import validate_submission

app = typer.Typer(add_completion=False, help="Kaggle competition automation CLI (safe by default).")


@app.command()
def bootstrap(
    slug: str = typer.Argument(..., help="Competition slug (URL suffix)."),
    force: bool = typer.Option(False, "--force", help="Re-download and overwrite existing data."),
):
    """
    Download & unzip competition files into data/<slug>/raw.
    Does NOT auto-accept rules; if not accepted, prints URL and exits.
    """
    bootstrap_competition(slug=slug, force=force)
    print("[green]bootstrap complete[/green]")


@app.command()
def run(
    slug: str = typer.Argument(..., help="Competition slug (URL suffix)."),
    submit: bool = typer.Option(False, "--submit", help="Actually submit to Kaggle (default: dry-run)."),
    message: str = typer.Option("auto baseline", "--message", help="Submission message."),
    force_submit: bool = typer.Option(False, "--force-submit", help="Allow duplicate submission hash."),
    force_download: bool = typer.Option(False, "--force-download", help="Force re-download data in bootstrap."),
):
    """
    End-to-end (safe by default):
    - bootstrap (download/unzip)
    - train baseline (tabular MVP)
    - generate submission.csv
    - validate vs sample_submission.csv
    - optional submit with guardrails
    """
    # 1) Bootstrap (downloads/unzips)
    bootstrap_competition(slug=slug, force=force_download)

    # 2) Train & generate submission.csv
    paths = train_and_make_submission(slug=slug)

    # 3) Validate submission strictly
    validate_submission(sample_path=paths.sample_submission, submission_path=paths.submission)

    print(f"[green]submission ready[/green]: {paths.submission}")

    # 4) Dry-run default
    if not submit:
        print("[yellow]DRY RUN[/yellow] (no submission). Use --submit to submit.")
        return

    # 5) Duplicate submission prevention
    history = SubmissionHistory.for_slug(slug)
    if history.is_duplicate(paths.submission) and not force_submit:
        print(
            "[red]Refusing to submit[/red]: identical submission already recorded.\n"
            "Use --force-submit if you really want to submit the same file again."
        )
        raise typer.Exit(code=2)

    # 6) Submit
    kaggle_submit(slug=slug, submission_file=paths.submission, message=message)

    # 7) Record history
    history.record(paths.submission, message=message)
    print("[green]submitted[/green]")


if __name__ == "__main__":
    app()
