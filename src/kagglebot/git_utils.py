from __future__ import annotations

from pathlib import Path

from rich import print

from kagglebot.exceptions import KaggleBotError
from kagglebot.exec_utils import run_command


def ensure_main_branch(repo_root: Path, *, slug: str, run_id: str) -> None:
    _assert_git_repo(repo_root)
    if _is_dirty(repo_root):
        message = f"kagglebot:{slug}:{run_id}"
        print("[yellow]git:[/yellow] Stashing dirty state (including untracked files)")
        _run_git(["stash", "push", "-u", "-m", message], repo_root)
    branch = _current_branch(repo_root)
    if branch != "main":
        print(f"[yellow]git:[/yellow] Switching branch {branch} -> main")
        _run_git(["checkout", "main"], repo_root)
        if _current_branch(repo_root) != "main":
            raise KaggleBotError("Unable to switch to main branch.")


def write_working_diff(repo_root: Path, output_path: Path) -> str:
    diff = _run_git(["diff"], repo_root)
    output_path.write_text(diff, encoding="utf-8")
    summary = _run_git(["diff", "--stat"], repo_root)
    return summary.strip()


def _assert_git_repo(repo_root: Path) -> None:
    result = run_command(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo_root)
    if result.returncode != 0 or result.output.strip() != "true":
        raise KaggleBotError("Not a git repository; unable to enforce main-only policy.")


def _current_branch(repo_root: Path) -> str:
    return _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)


def _is_dirty(repo_root: Path) -> bool:
    result = run_command(["git", "status", "--porcelain"], cwd=repo_root)
    if result.returncode != 0:
        raise KaggleBotError(f"git status failed: {result.output}")
    return bool(result.stdout.strip())


def _run_git(args: list[str], repo_root: Path) -> str:
    result = run_command(["git", *args], cwd=repo_root)
    if result.returncode != 0:
        raise KaggleBotError(f"git {' '.join(args)} failed: {result.output}")
    return result.stdout.strip()
