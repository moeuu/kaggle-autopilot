from __future__ import annotations

import fcntl
import json
import os
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from kagglebot.exec_utils import run_command
from kagglebot.json_utils import append_jsonl_record


class RepositoryTransactionError(RuntimeError):
    pass


class RepositoryTransactionLockedError(RepositoryTransactionError):
    pass


@dataclass(frozen=True)
class RepositoryBaseline:
    workdir: Path
    branch: str
    upstream: str
    remote: str
    remote_branch: str
    repository_url: str
    head_sha: str
    remote_sha: str

    @property
    def commit_url(self) -> str:
        return f"{self.repository_url}/commit/{self.head_sha}"

    def prompt_header(self) -> str:
        return "\n".join(
            (
                f"Repository URL: {self.repository_url}",
                f"Baseline commit SHA: {self.head_sha}",
                f"Baseline commit URL: {self.commit_url}",
                "Baseline status: clean and verified on the configured upstream",
            )
        )


@contextmanager
def repository_transaction_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RepositoryTransactionLockedError(
                f"repository self-improvement lock is already held: {lock_path}"
            ) from exc
        _write_lock_owner(handle)
        yield handle
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def verify_clean_pushed_repository(workdir: Path) -> RepositoryBaseline:
    root = _git_stdout(workdir, "rev-parse", "--show-toplevel")
    repository_root = Path(root).resolve()
    status = _git_stdout(repository_root, "status", "--porcelain")
    if status:
        raise RepositoryTransactionError("repository worktree must be clean before Oracle consultation")
    branch = _git_stdout(repository_root, "branch", "--show-current")
    if not branch:
        raise RepositoryTransactionError("repository self-improvement does not support detached HEAD")
    upstream = _git_stdout(repository_root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    remote, separator, remote_branch = upstream.partition("/")
    if not separator or not remote or not remote_branch:
        raise RepositoryTransactionError(f"unable to resolve configured upstream from {upstream!r}")
    remote_url = _git_stdout(repository_root, "remote", "get-url", remote)
    repository_url = canonical_repository_url(remote_url)
    head_sha = _git_stdout(repository_root, "rev-parse", "HEAD")
    remote_sha = _remote_branch_sha(repository_root, remote, remote_branch)
    if head_sha != remote_sha:
        raise RepositoryTransactionError(
            f"local HEAD is not the pushed upstream commit: local={head_sha} upstream={remote_sha}"
        )
    return RepositoryBaseline(
        workdir=repository_root,
        branch=branch,
        upstream=upstream,
        remote=remote,
        remote_branch=remote_branch,
        repository_url=repository_url,
        head_sha=head_sha,
        remote_sha=remote_sha,
    )


def revalidate_repository_baseline(baseline: RepositoryBaseline) -> None:
    current = verify_clean_pushed_repository(baseline.workdir)
    if current.head_sha != baseline.head_sha or current.repository_url != baseline.repository_url:
        raise RepositoryTransactionError(
            "repository baseline changed after Oracle consultation; a fresh Oracle response is required"
        )


def canonical_repository_url(value: str) -> str:
    raw = value.strip()
    match = re.fullmatch(r"git@([^:]+):(.+?)(?:\.git)?", raw)
    if match:
        return f"https://{match.group(1)}/{match.group(2).removesuffix('.git')}"
    match = re.fullmatch(r"ssh://git@([^/]+)/(.+?)(?:\.git)?", raw)
    if match:
        return f"https://{match.group(1)}/{match.group(2).removesuffix('.git')}"
    if raw.startswith(("https://", "http://")):
        return raw.removesuffix("/").removesuffix(".git")
    raise RepositoryTransactionError(f"unsupported repository remote URL: {value}")


def validate_repository_oracle_response(text: str, baseline: RepositoryBaseline) -> dict[str, object]:
    stripped = text.strip()
    required = (
        "===DECISION===",
        "===EVIDENCE===",
        "===REPO_IMPROVEMENT_PLAN_JSON===",
        "===GUARDRAILS===",
    )
    missing = [marker for marker in required if marker not in stripped]
    if missing:
        raise RepositoryTransactionError(f"Oracle response is missing required sections: {', '.join(missing)}")
    if baseline.repository_url not in stripped or baseline.head_sha not in stripped:
        raise RepositoryTransactionError("Oracle response did not echo the exact repository URL and baseline SHA")
    plan_text = stripped.split("===REPO_IMPROVEMENT_PLAN_JSON===", 1)[1].split("===GUARDRAILS===", 1)[0].strip()
    if plan_text.startswith("```"):
        plan_text = re.sub(r"^```(?:json)?\s*", "", plan_text)
        plan_text = re.sub(r"\s*```$", "", plan_text)
    try:
        plan = json.loads(plan_text)
    except json.JSONDecodeError as exc:
        raise RepositoryTransactionError(f"Oracle repository plan JSON is invalid: {exc}") from exc
    if not isinstance(plan, dict):
        raise RepositoryTransactionError("Oracle repository plan must be a JSON object")
    if plan.get("repository_url") != baseline.repository_url or plan.get("baseline_sha") != baseline.head_sha:
        raise RepositoryTransactionError("Oracle repository plan targets a different repository baseline")
    for key in ("proposed_files", "acceptance_tests", "rollback_strategy"):
        value = plan.get(key)
        if value in (None, "", []):
            raise RepositoryTransactionError(f"Oracle repository plan is missing {key}")
    return plan


def write_transaction_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_transaction_event(path: Path, *, state: str, **payload: object) -> None:
    append_jsonl_record(
        path,
        {"ts": datetime.now(UTC).isoformat(), "state": state, **payload},
        sort_keys=True,
    )


def baseline_payload(baseline: RepositoryBaseline) -> dict[str, object]:
    payload = asdict(baseline)
    payload["workdir"] = str(baseline.workdir)
    payload["commit_url"] = baseline.commit_url
    return payload


def _git_stdout(workdir: Path, *args: str) -> str:
    result = run_command(["git", *args], cwd=workdir)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RepositoryTransactionError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def _remote_branch_sha(workdir: Path, remote: str, branch: str) -> str:
    result = run_command(["git", "ls-remote", "--heads", remote, f"refs/heads/{branch}"], cwd=workdir)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RepositoryTransactionError(f"unable to read upstream branch SHA: {detail}")
    line = next((item for item in result.stdout.splitlines() if item.strip()), "")
    sha = line.split(maxsplit=1)[0] if line else ""
    if not re.fullmatch(r"[0-9a-f]{40,64}", sha):
        raise RepositoryTransactionError(f"upstream branch {remote}/{branch} was not found")
    return sha


def _write_lock_owner(handle: TextIO) -> None:
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "host": os.uname().nodename,
                "acquired_at": datetime.now(UTC).isoformat(),
            },
            sort_keys=True,
        )
        + "\n"
    )
    handle.flush()
