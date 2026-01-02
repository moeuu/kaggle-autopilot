"""Tests for git utility helpers."""

from __future__ import annotations

from types import SimpleNamespace

from kagglebot import git_utils


def test_ensure_main_branch_stash_and_checkout(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []
    branch_calls = {"count": 0}

    def fake_run_command(args, cwd=None, **kwargs):  # noqa: ARG001
        calls.append(args)
        if args[:3] == ["git", "rev-parse", "--is-inside-work-tree"]:
            return SimpleNamespace(returncode=0, stdout="true\n", stderr="", output="true")
        if args[:3] == ["git", "status", "--porcelain"]:
            return SimpleNamespace(returncode=0, stdout=" M file.txt\n", stderr="", output=" M file.txt")
        if args[:4] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            branch_calls["count"] += 1
            branch = "feature" if branch_calls["count"] == 1 else "main"
            return SimpleNamespace(returncode=0, stdout=f"{branch}\n", stderr="", output=branch)
        if args[:3] == ["git", "stash", "push"]:
            return SimpleNamespace(returncode=0, stdout="Saved\n", stderr="", output="Saved")
        if args[:2] == ["git", "checkout"]:
            return SimpleNamespace(returncode=0, stdout="Switched\n", stderr="", output="Switched")
        raise AssertionError(f"Unexpected git command: {args}")

    monkeypatch.setattr(git_utils, "run_command", fake_run_command)
    git_utils.ensure_main_branch(tmp_path, slug="demo", run_id="run-1")

    assert ["git", "stash", "push", "-u", "-m", "kagglebot:demo:run-1"] in calls
    assert ["git", "checkout", "main"] in calls


def test_write_working_diff(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    def fake_run_command(args, cwd=None, **kwargs):  # noqa: ARG001
        calls.append(args)
        if args[:3] == ["git", "diff", "--stat"]:
            return SimpleNamespace(returncode=0, stdout="file.txt | 1 +\n", stderr="", output="file.txt | 1 +")
        if args[:2] == ["git", "diff"]:
            return SimpleNamespace(returncode=0, stdout="diff --git a/file.txt b/file.txt\n", stderr="", output="")
        if args[:3] == ["git", "rev-parse", "--is-inside-work-tree"]:
            return SimpleNamespace(returncode=0, stdout="true\n", stderr="", output="true")
        return SimpleNamespace(returncode=0, stdout="", stderr="", output="")

    monkeypatch.setattr(git_utils, "run_command", fake_run_command)
    output_path = tmp_path / "code.diff"
    summary = git_utils.write_working_diff(tmp_path, output_path)

    assert output_path.exists()
    assert "file.txt" in output_path.read_text(encoding="utf-8")
    assert "file.txt" in summary
