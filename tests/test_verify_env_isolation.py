from __future__ import annotations

from pathlib import Path

from kagglebot.exec_utils import CommandResult
from kagglebot.verify_artifacts import run_repo_verify


def test_run_verify_disables_pytest_plugin_autoload(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_command(args, **kwargs):  # noqa: ANN003
        captured["args"] = args
        captured["env"] = kwargs.get("env")
        return CommandResult(args=args, returncode=0, stdout="", stderr="", duration_sec=0.0)

    monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)

    run_repo_verify("uv run pytest -q", dry_run=False, run_command_fn=fake_run_command)

    env = captured.get("env")
    assert isinstance(env, dict)
    assert env.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1"
    assert captured["args"] == ["uv", "run", "pytest", "-p", "xdist.plugin", "-q"]


def test_run_verify_respects_existing_pytest_disable_env(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_command(args, **kwargs):  # noqa: ANN003
        captured["env"] = kwargs.get("env")
        return CommandResult(args=args, returncode=0, stdout="", stderr="", duration_sec=0.0)

    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "0")

    run_repo_verify("pytest -q", dry_run=False, run_command_fn=fake_run_command)

    env = captured.get("env")
    assert isinstance(env, dict)
    assert env.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "0"


def test_run_verify_does_not_modify_env_for_non_pytest(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_command(args, **kwargs):  # noqa: ANN003
        captured["env"] = kwargs.get("env")
        return CommandResult(args=args, returncode=0, stdout="", stderr="", duration_sec=0.0)

    run_repo_verify("uv run ruff check .", dry_run=False, run_command_fn=fake_run_command)

    assert captured.get("env") is None


def test_run_verify_mirrors_external_artifacts_before_pytest(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "tests").mkdir()
    external_artifacts = tmp_path / "external-artifacts"
    source_kernel = external_artifacts / "demo" / "kernel"
    source_kernel.mkdir(parents=True)
    expected_kernel = repo_root / "artifacts" / "demo" / "kernel" / "kernel.py"
    (source_kernel / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")

    def fake_run_command(args, **kwargs):  # noqa: ANN003
        assert expected_kernel.exists()
        assert expected_kernel.read_text(encoding="utf-8") == "VALUE = 1\n"
        return CommandResult(args=args, returncode=0, stdout="", stderr="", duration_sec=0.0)

    monkeypatch.chdir(repo_root)

    run_repo_verify("pytest -q", dry_run=False, artifacts_dir=external_artifacts, run_command_fn=fake_run_command)
