from __future__ import annotations

import shlex
import subprocess
import sys
from enum import Enum
from pathlib import Path

import typer
from rich import print

from kagglebot.agents.claude_runner import run_claude
from kagglebot.agents.codex_runner import run_codex
from kagglebot.bootstrap import bootstrap_competition
from kagglebot.competition import parse_competition_slug, rules_url_for_slug
from kagglebot.compute import Compute, compute_to_runner_and_accelerator
from kagglebot.exceptions import (
    DuplicateSubmissionError,
    GPUNotAvailableError,
    KaggleCliError,
    KernelFailedError,
    KernelTimeoutError,
    RulesNotAcceptedError,
    SubmissionRateLimitError,
    ValidationError,
)
from kagglebot.git_ops import branch_exists, commit_all, create_branch, ensure_clean_worktree, ensure_git_repo
from kagglebot.history import RunLedger, SubmissionLedger
from kagglebot.kaggle_api import check_rules_accepted, submit_competition
from kagglebot.kernel_runner import resolve_kaggle_username, run_kernel
from kagglebot.paths import CompetitionPaths, repo_root
from kagglebot.solver.baseline import train_and_predict
from kagglebot.solver.validate import validate_submission_file
from kagglebot.validation import ensure_not_duplicate_submission, ensure_submission_rate_limit

app = typer.Typer(add_completion=False, help="Kaggle competition automation CLI (safe by default).")


class Agent(str, Enum):
    codex = "codex"
    claude = "claude"


@app.command()
def bootstrap(
    competition: str = typer.Argument(..., help="Competition URL or slug."),
    download: bool = typer.Option(False, "--download/--no-download", help="Download competition data."),
    rules_source: str = typer.Option(
        "url",
        "--rules-source",
        help="Rules capture source: none, url, fetch, file.",
    ),
    rules_file: Path | None = typer.Option(
        None,
        "--rules-file",
        help="Rules file to copy when --rules-source file.",
    ),
    force: bool = typer.Option(False, "--force", help="Allow Kaggle CLI download."),
    quiet: bool = typer.Option(True, "--quiet/--no-quiet", help="Use --quiet for Kaggle CLI download."),
    workdir: Path | None = typer.Option(None, "--workdir", help="Base working directory."),
) -> None:
    slug = _resolve_slug(competition)
    base_root = workdir if workdir is not None else repo_root()

    if download and not force:
        _refuse_side_effect("download competition data")

    try:
        meta_path = bootstrap_competition(
            slug=slug,
            root=base_root,
            force=force,
            rules_source=rules_source,
            rules_file=rules_file,
            download=download,
            quiet=quiet,
        )
    except RulesNotAcceptedError:
        _print_rules_and_exit(slug)
    except KaggleCliError as exc:
        _print_kaggle_error(exc, action="download")
        raise typer.Exit(code=exc.exit_code) from exc

    print(f"[green]bootstrap complete[/green]: {meta_path}")


@app.command()
def implement(
    competition: str = typer.Argument(..., help="Competition URL or slug."),
    agent: Agent = typer.Option(..., "--agent", help="Agent to run (codex or claude)."),
    commit: bool = typer.Option(False, "--commit/--no-commit", help="Commit changes after verification."),
    verify_cmd: str = typer.Option("uv run pytest -q", "--verify-cmd", help="Verification command."),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run", help="Skip running the agent."),
    workdir: Path | None = typer.Option(None, "--workdir", help="Base working directory."),
) -> None:
    slug = _resolve_slug(competition)
    base_root = workdir if workdir is not None else repo_root()

    paths = CompetitionPaths(slug=slug, repo_root=base_root)
    bootstrap_competition(slug=slug, root=base_root)

    run_ledger = RunLedger.for_slug(slug, root=base_root)
    run_record = run_ledger.start_run(
        slug=slug,
        dry_run=dry_run,
        force=False,
        submission_path=None,
        sample_path=None,
        message=None,
        argv=list(sys.argv),
        extra={"agent": agent.value, "command": "implement"},
    )
    print(f"[green]run started[/green]: {run_record.run_id}")

    agent_dir = paths.runs_dir / run_record.run_id / agent.value
    prompt_path = paths.prompts_dir / f"{agent.value}.md"
    if not prompt_path.exists():
        raise typer.Exit(code=1)

    _run_agent(
        agent=agent,
        prompt_path=prompt_path,
        agent_dir=agent_dir,
        branch_name=f"bot/{slug}/{run_record.run_id}",
        verify_cmd=verify_cmd,
        commit=commit,
        dry_run=dry_run,
        base_root=base_root,
        slug=slug,
        run_id=run_record.run_id,
    )
    print(f"[green]agent logs[/green]: {agent_dir}")


@app.command()
def train(
    competition: str = typer.Argument(..., help="Competition URL or slug."),
    compute: Compute = typer.Option(Compute.local_cpu, "--compute", help="Compute target for training."),
    kaggle_username: str | None = typer.Option(
        None,
        "--kaggle-username",
        help="Kaggle username for Kaggle kernel runs.",
    ),
    enable_internet: bool = typer.Option(False, "--enable-internet", help="Enable internet in Kaggle kernels."),
    strict_accelerator: bool = typer.Option(
        False,
        "--strict-accelerator",
        help="Fail if requested accelerator is unavailable (local_gpu).",
    ),
    kernel_name: str | None = typer.Option(None, "--kernel-name", help="Custom kernel slug."),
    seed: int = typer.Option(42, "--seed", help="Random seed for local training."),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run", help="Skip Kaggle kernel execution."),
    force: bool = typer.Option(False, "--force", help="Allow Kaggle CLI operations."),
    workdir: Path | None = typer.Option(None, "--workdir", help="Base working directory."),
) -> None:
    slug = _resolve_slug(competition)
    base_root = workdir if workdir is not None else repo_root()
    paths = CompetitionPaths(slug=slug, repo_root=base_root)
    bootstrap_competition(slug=slug, root=base_root)

    run_ledger = RunLedger.for_slug(slug, root=base_root)
    run_record = run_ledger.start_run(
        slug=slug,
        dry_run=dry_run,
        force=force,
        submission_path=None,
        sample_path=None,
        message=None,
        argv=list(sys.argv),
        extra={"compute": compute.value, "command": "train"},
    )

    selection = compute_to_runner_and_accelerator(compute)
    if selection.runner == "kaggle_notebook" and not dry_run and not force:
        _refuse_side_effect("execute Kaggle notebook runner")

    data_dir = paths.data_dir
    if selection.runner == "local" and not _has_csvs(data_dir):
        print("[red]No data found[/red]: run bootstrap with --download first.")
        raise typer.Exit(code=1)

    submission_path = paths.submissions_dir / f"{run_record.run_id}_submission.csv"

    if selection.runner == "local":
        try:
            result = train_and_predict(
                data_dir=data_dir,
                output_path=submission_path,
                compute=compute,
                strict_accelerator=strict_accelerator,
                seed=seed,
                metrics_path=paths.runs_dir / run_record.run_id / "metrics.json",
            )
        except GPUNotAvailableError as exc:
            print(f"[red]{exc}[/red]")
            raise typer.Exit(code=exc.exit_code) from exc
        except Exception as exc:  # noqa: BLE001
            print(f"[red]training failed[/red]: {exc}")
            raise typer.Exit(code=5) from exc
        print(f"[green]submission written[/green]: {result.submission_path}")
        return

    try:
        kaggle_user = resolve_kaggle_username(kaggle_username)
    except ValueError as exc:
        print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    try:
        kernel_result = run_kernel(
            slug=slug,
            run_id=run_record.run_id,
            base_dir=paths.artifacts,
            kaggle_username=kaggle_user,
            kernel_name=kernel_name,
            accelerator=selection.accelerator,
            enable_internet=enable_internet,
            dry_run=dry_run,
        )
    except RulesNotAcceptedError:
        _print_rules_and_exit(slug)
    except KaggleCliError as exc:
        _print_kaggle_error(exc, action="run kernel")
        raise typer.Exit(code=exc.exit_code) from exc
    except (KernelTimeoutError, KernelFailedError) as exc:
        print(f"[red]{exc}[/red]")
        raise typer.Exit(code=exc.exit_code) from exc

    if kernel_result.submission_path is None:
        print("[yellow]no submission produced[/yellow]: dry-run or kernel output missing.")
        return

    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_bytes(kernel_result.submission_path.read_bytes())
    print(f"[green]submission written[/green]: {submission_path}")


@app.command()
def submit(
    competition: str = typer.Argument(..., help="Competition URL or slug."),
    file: Path = typer.Option(..., "-f", "--file", help="Submission CSV file."),
    message: str = typer.Option(..., "-m", "--message", help="Submission message."),
    force: bool = typer.Option(False, "--force", help="Allow Kaggle CLI submission."),
    force_duplicate: bool = typer.Option(False, "--force-duplicate", help="Allow duplicate submissions."),
    workdir: Path | None = typer.Option(None, "--workdir", help="Base working directory."),
) -> None:
    slug = _resolve_slug(competition)
    if not file.exists():
        raise typer.Exit(code=1)
    if not force:
        _refuse_side_effect("submit to Kaggle")

    base_root = workdir if workdir is not None else repo_root()
    paths = CompetitionPaths(slug=slug, repo_root=base_root)
    sample_path = _find_sample_submission(paths.data_dir)
    if sample_path:
        try:
            validate_submission_file(sample_path, file)
        except Exception as exc:  # noqa: BLE001
            print(f"[red]submission validation failed[/red]: {exc}")
            raise typer.Exit(code=ValidationError.exit_code) from exc

    ledger = SubmissionLedger.for_slug(slug, root=base_root)
    try:
        ensure_submission_rate_limit(ledger)
    except SubmissionRateLimitError as exc:
        print(f"[red]{exc}[/red]")
        raise typer.Exit(code=exc.exit_code) from exc
    if not force_duplicate:
        try:
            ensure_not_duplicate_submission(ledger, str(file))
        except DuplicateSubmissionError as exc:
            print(f"[red]{exc}[/red]")
            raise typer.Exit(code=exc.exit_code) from exc

    try:
        if not check_rules_accepted(slug):
            _print_rules_and_exit(slug)
    except KaggleCliError as exc:
        _print_kaggle_error(exc, action="rules check")
        raise typer.Exit(code=exc.exit_code) from exc

    try:
        submit_competition(slug, file, message)
    except RulesNotAcceptedError:
        _print_rules_and_exit(slug)
    except KaggleCliError as exc:
        _print_kaggle_error(exc, action="submit")
        raise typer.Exit(code=exc.exit_code) from exc

    ledger.record(str(file), message=message, run_id=None, slug=slug)
    print("[green]submission recorded[/green]")


@app.command()
def run(
    competition: str = typer.Argument(..., help="Competition URL or slug."),
    agent: Agent = typer.Option(..., "--agent", help="Agent to run (codex or claude)."),
    compute: Compute = typer.Option(Compute.local_cpu, "--compute", help="Compute target for training."),
    submit: bool = typer.Option(False, "--submit/--no-submit", help="Submit after validation."),
    message: str | None = typer.Option(None, "-m", "--message", help="Submission message."),
    kaggle_username: str | None = typer.Option(
        None,
        "--kaggle-username",
        help="Kaggle username for Kaggle kernel runs.",
    ),
    enable_internet: bool = typer.Option(False, "--enable-internet", help="Enable internet in Kaggle kernels."),
    strict_accelerator: bool = typer.Option(
        False,
        "--strict-accelerator",
        help="Fail if requested accelerator is unavailable (local_gpu).",
    ),
    kernel_name: str | None = typer.Option(None, "--kernel-name", help="Custom kernel slug."),
    seed: int = typer.Option(42, "--seed", help="Random seed for local training."),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run", help="Skip external commands."),
    force: bool = typer.Option(False, "--force", help="Allow Kaggle CLI operations."),
    force_duplicate: bool = typer.Option(False, "--force-duplicate", help="Allow duplicate submissions."),
    commit: bool = typer.Option(False, "--commit/--no-commit", help="Commit after agent verification."),
    verify_cmd: str = typer.Option("uv run pytest -q", "--verify-cmd", help="Verification command."),
    workdir: Path | None = typer.Option(None, "--workdir", help="Base working directory."),
) -> None:
    slug = _resolve_slug(competition)
    base_root = workdir if workdir is not None else repo_root()
    paths = CompetitionPaths(slug=slug, repo_root=base_root)

    selection = compute_to_runner_and_accelerator(compute)
    if selection.runner == "kaggle_notebook" and not dry_run and not force:
        _refuse_side_effect("execute Kaggle notebook runner")

    if submit and not message:
        print("[red]Submission requires --message.[/red]")
        raise typer.Exit(code=1)

    download_needed = not _has_csvs(paths.data_dir)
    if download_needed and not dry_run and not force:
        _refuse_side_effect("download competition data")

    try:
        bootstrap_competition(
            slug=slug,
            root=base_root,
            force=force,
            download=download_needed and not dry_run,
        )
    except RulesNotAcceptedError:
        _print_rules_and_exit(slug)
    except KaggleCliError as exc:
        _print_kaggle_error(exc, action="download")
        raise typer.Exit(code=exc.exit_code) from exc

    run_ledger = RunLedger.for_slug(slug, root=base_root)
    run_record = run_ledger.start_run(
        slug=slug,
        dry_run=dry_run,
        force=force,
        submission_path=None,
        sample_path=None,
        message=message,
        argv=list(sys.argv),
        extra={"compute": compute.value, "agent": agent.value, "command": "run"},
    )

    agent_dir = paths.runs_dir / run_record.run_id / agent.value
    prompt_path = paths.prompts_dir / f"{agent.value}.md"
    _run_agent(
        agent=agent,
        prompt_path=prompt_path,
        agent_dir=agent_dir,
        branch_name=f"bot/{slug}/{run_record.run_id}",
        verify_cmd=verify_cmd,
        commit=commit,
        dry_run=dry_run,
        base_root=base_root,
        slug=slug,
        run_id=run_record.run_id,
    )

    submission_path = paths.submissions_dir / f"{run_record.run_id}_submission.csv"
    if selection.runner == "local" and not _has_csvs(paths.data_dir):
        if dry_run:
            print("[yellow]DRY RUN[/yellow]: no local data available for training.")
            return
        print("[red]No data found[/red]: run bootstrap with --download first.")
        raise typer.Exit(code=1)
    try:
        if selection.runner == "local":
            result = train_and_predict(
                data_dir=paths.data_dir,
                output_path=submission_path,
                compute=compute,
                strict_accelerator=strict_accelerator,
                seed=seed,
                metrics_path=paths.runs_dir / run_record.run_id / "metrics.json",
            )
            submission_path = result.submission_path
        else:
            try:
                kaggle_user = resolve_kaggle_username(kaggle_username)
            except ValueError as exc:
                print(f"[red]{exc}[/red]")
                raise typer.Exit(code=1) from exc
            kernel_result = run_kernel(
                slug=slug,
                run_id=run_record.run_id,
                base_dir=paths.artifacts,
                kaggle_username=kaggle_user,
                kernel_name=kernel_name,
                accelerator=selection.accelerator,
                enable_internet=enable_internet,
                dry_run=dry_run,
            )
            if kernel_result.submission_path is None:
                print("[yellow]no submission produced[/yellow]: dry-run or kernel output missing.")
                return
            submission_path.parent.mkdir(parents=True, exist_ok=True)
            submission_path.write_bytes(kernel_result.submission_path.read_bytes())
    except GPUNotAvailableError as exc:
        print(f"[red]{exc}[/red]")
        raise typer.Exit(code=exc.exit_code) from exc
    except RulesNotAcceptedError:
        _print_rules_and_exit(slug)
    except KaggleCliError as exc:
        _print_kaggle_error(exc, action="run kernel")
        raise typer.Exit(code=exc.exit_code) from exc
    except (KernelTimeoutError, KernelFailedError) as exc:
        print(f"[red]{exc}[/red]")
        raise typer.Exit(code=exc.exit_code) from exc
    except Exception as exc:  # noqa: BLE001
        print(f"[red]training failed[/red]: {exc}")
        raise typer.Exit(code=5) from exc

    sample_path = _find_sample_submission(paths.data_dir)
    if sample_path:
        try:
            validate_submission_file(sample_path, submission_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[red]submission validation failed[/red]: {exc}")
            raise typer.Exit(code=ValidationError.exit_code) from exc
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
    try:
        ensure_submission_rate_limit(ledger)
    except SubmissionRateLimitError as exc:
        print(f"[red]{exc}[/red]")
        raise typer.Exit(code=exc.exit_code) from exc
    if not force_duplicate:
        try:
            ensure_not_duplicate_submission(ledger, str(submission_path))
        except DuplicateSubmissionError as exc:
            print(f"[red]{exc}[/red]")
            raise typer.Exit(code=exc.exit_code) from exc

    try:
        if not check_rules_accepted(slug):
            _print_rules_and_exit(slug)
    except KaggleCliError as exc:
        _print_kaggle_error(exc, action="rules check")
        raise typer.Exit(code=exc.exit_code) from exc

    try:
        submit_competition(slug, submission_path, message or "")
    except RulesNotAcceptedError:
        _print_rules_and_exit(slug)
    except KaggleCliError as exc:
        _print_kaggle_error(exc, action="submit")
        raise typer.Exit(code=exc.exit_code) from exc
    ledger.record(str(submission_path), message=message or "", run_id=run_record.run_id, slug=slug)
    print("[green]submission recorded[/green]")


def _resolve_slug(competition: str) -> str:
    try:
        return parse_competition_slug(competition)
    except ValueError as exc:
        print(f"[red]Invalid competition[/red]: {exc}")
        raise typer.Exit(code=3) from exc


def _refuse_side_effect(action: str) -> None:
    print(f"[red]Refusing to {action} without --force.[/red]")
    raise typer.Exit(code=1)


def _print_rules_and_exit(slug: str) -> None:
    print("[red]Competition rules not accepted.[/red]")
    print(f"Visit: {rules_url_for_slug(slug)}")
    raise typer.Exit(code=2)


def _print_kaggle_error(exc: KaggleCliError, action: str) -> None:
    print(f"[red]Kaggle CLI {action} failed[/red]: {exc.message}")
    if exc.output:
        print(exc.output)


def _run_agent(
    *,
    agent: Agent,
    prompt_path: Path,
    agent_dir: Path,
    branch_name: str,
    verify_cmd: str,
    commit: bool,
    dry_run: bool,
    base_root: Path,
    slug: str,
    run_id: str,
) -> None:
    if not prompt_path.exists():
        print(f"[red]Prompt not found[/red]: {prompt_path}")
        raise typer.Exit(code=1)

    if dry_run:
        if agent == Agent.codex:
            run_codex(prompt_path, agent_dir, dry_run=True)
        else:
            run_claude(prompt_path, agent_dir, dry_run=True)
        print("[yellow]DRY RUN[/yellow]: agent execution skipped.")
        return

    ensure_git_repo(cwd=base_root)
    ensure_clean_worktree(cwd=base_root)
    if branch_exists(branch_name, cwd=base_root):
        print(f"[red]Branch already exists[/red]: {branch_name}")
        raise typer.Exit(code=1)
    create_branch(branch_name, cwd=base_root)

    if agent == Agent.codex:
        run_codex(prompt_path, agent_dir)
    else:
        run_claude(prompt_path, agent_dir)

    verify_log = agent_dir / "verify.log"
    verify_result = _run_verify(verify_cmd, cwd=base_root, log_path=verify_log)
    if verify_result != 0:
        print(f"[red]verification failed[/red]: {verify_log}")
        raise typer.Exit(code=1)

    if commit:
        commit_all(f"kagglebot: implement {slug} run {run_id}", cwd=base_root)


def _run_verify(command: str, *, cwd: Path, log_path: Path) -> int:
    args = shlex.split(command)
    completed = subprocess.run(args, capture_output=True, text=True, check=False, cwd=str(cwd))
    log_path.write_text(
        "".join([completed.stdout or "", completed.stderr or ""]),
        encoding="utf-8",
    )
    return completed.returncode


def _has_csvs(data_dir: Path) -> bool:
    return any(data_dir.rglob("*.csv"))


def _find_sample_submission(data_dir: Path) -> Path | None:
    for path in data_dir.rglob("sample_submission.csv"):
        if path.is_file():
            return path
    return None


if __name__ == "__main__":
    app()
