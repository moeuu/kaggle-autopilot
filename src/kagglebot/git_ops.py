from __future__ import annotations

import subprocess
from pathlib import Path

from kagglebot.exceptions import GitError


def _run_git(args: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(cwd) if cwd else None,
    )
    output = "".join([completed.stdout or "", completed.stderr or ""]).strip()
    if completed.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {output}")
    return output


def ensure_git_repo(*, cwd: Path | None = None) -> None:
    _run_git(["rev-parse", "--is-inside-work-tree"], cwd=cwd)


def ensure_clean_worktree(*, cwd: Path | None = None) -> None:
    status = _run_git(["status", "--porcelain"], cwd=cwd)
    if status.strip():
        raise GitError("Git worktree is dirty. Commit or stash changes before running implement.")


def current_branch(*, cwd: Path | None = None) -> str:
    return _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd).strip()


def branch_exists(name: str, *, cwd: Path | None = None) -> bool:
    output = _run_git(["branch", "--list", name], cwd=cwd)
    return bool(output.strip())


def create_branch(name: str, *, cwd: Path | None = None) -> None:
    _run_git(["checkout", "-b", name], cwd=cwd)


def commit_all(message: str, *, cwd: Path | None = None) -> None:
    _run_git(["add", "-A"], cwd=cwd)
    _run_git(["commit", "-m", message], cwd=cwd)
