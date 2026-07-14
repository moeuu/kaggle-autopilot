from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagglebot.exec_utils import CommandResult
from kagglebot.repository_transaction import (
    RepositoryBaseline,
    RepositoryTransactionError,
    canonical_repository_url,
    validate_repository_oracle_response,
    verify_clean_pushed_repository,
)


def test_verify_clean_pushed_repository_uses_configured_upstream(monkeypatch, tmp_path: Path) -> None:
    sha = "a" * 40
    outputs = {
        ("rev-parse", "--show-toplevel"): str(tmp_path),
        ("status", "--porcelain"): "",
        ("branch", "--show-current"): "main",
        ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"): "upstream/main",
        ("remote", "get-url", "upstream"): "git@github.com:example/kaggle-autopilot.git",
        ("rev-parse", "HEAD"): sha,
        ("ls-remote", "--heads", "upstream", "refs/heads/main"): f"{sha}\trefs/heads/main\n",
    }

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:  # noqa: ARG001
        key = tuple(args[1:])
        return CommandResult(args=args, returncode=0, stdout=outputs[key], stderr="", duration_sec=0.01)

    monkeypatch.setattr("kagglebot.repository_transaction.run_command", fake_run_command)

    baseline = verify_clean_pushed_repository(tmp_path)

    assert baseline.repository_url == "https://github.com/example/kaggle-autopilot"
    assert baseline.upstream == "upstream/main"
    assert baseline.head_sha == sha


def test_verify_clean_pushed_repository_rejects_unpushed_head(monkeypatch, tmp_path: Path) -> None:
    local_sha = "a" * 40
    remote_sha = "b" * 40
    outputs = iter(
        [
            str(tmp_path),
            "",
            "main",
            "origin/main",
            "https://github.com/example/repo.git",
            local_sha,
            f"{remote_sha}\trefs/heads/main\n",
        ]
    )

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:  # noqa: ARG001
        return CommandResult(args=args, returncode=0, stdout=next(outputs), stderr="", duration_sec=0.01)

    monkeypatch.setattr("kagglebot.repository_transaction.run_command", fake_run_command)

    with pytest.raises(RepositoryTransactionError, match="not the pushed upstream"):
        verify_clean_pushed_repository(tmp_path)


def test_validate_repository_oracle_response_binds_exact_baseline(tmp_path: Path) -> None:
    baseline = RepositoryBaseline(
        workdir=tmp_path,
        branch="main",
        upstream="origin/main",
        remote="origin",
        remote_branch="main",
        repository_url="https://github.com/example/repo",
        head_sha="a" * 40,
        remote_sha="a" * 40,
    )
    plan = {
        "schema_version": 1,
        "repository_url": baseline.repository_url,
        "baseline_sha": baseline.head_sha,
        "proposed_files": ["src/example.py"],
        "acceptance_tests": ["uv run pytest -q"],
        "rollback_strategy": "revert the implementation commit",
    }
    response = (
        "===DECISION===\nImplement one fix.\n"
        "===EVIDENCE===\nRepeated failures.\n"
        f"===REPO_IMPROVEMENT_PLAN_JSON===\n```json\n{json.dumps(plan)}\n```\n"
        "===GUARDRAILS===\nPreserve submission safety.\n"
    )

    assert validate_repository_oracle_response(response, baseline) == plan

    with pytest.raises(RepositoryTransactionError, match="baseline SHA|different repository baseline"):
        validate_repository_oracle_response(response.replace(baseline.head_sha, "b" * 40), baseline)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("git@github.com:owner/repo.git", "https://github.com/owner/repo"),
        ("https://github.com/owner/repo.git", "https://github.com/owner/repo"),
    ],
)
def test_canonical_repository_url(raw: str, expected: str) -> None:
    assert canonical_repository_url(raw) == expected
