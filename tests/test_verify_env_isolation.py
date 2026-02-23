from __future__ import annotations

from kagglebot import autopilot as autopilot_module
from kagglebot.exec_utils import CommandResult


def test_run_verify_disables_pytest_plugin_autoload(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_command(args, **kwargs):  # noqa: ANN003
        captured["args"] = args
        captured["env"] = kwargs.get("env")
        return CommandResult(args=args, returncode=0, stdout="", stderr="", duration_sec=0.0)

    monkeypatch.setattr(autopilot_module, "run_command", fake_run_command)
    monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)

    autopilot_module._run_verify("uv run pytest -q", dry_run=False)

    env = captured.get("env")
    assert isinstance(env, dict)
    assert env.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1"


def test_run_verify_respects_existing_pytest_disable_env(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_command(args, **kwargs):  # noqa: ANN003
        captured["env"] = kwargs.get("env")
        return CommandResult(args=args, returncode=0, stdout="", stderr="", duration_sec=0.0)

    monkeypatch.setattr(autopilot_module, "run_command", fake_run_command)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "0")

    autopilot_module._run_verify("pytest -q", dry_run=False)

    env = captured.get("env")
    assert isinstance(env, dict)
    assert env.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "0"


def test_run_verify_does_not_modify_env_for_non_pytest(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_command(args, **kwargs):  # noqa: ANN003
        captured["env"] = kwargs.get("env")
        return CommandResult(args=args, returncode=0, stdout="", stderr="", duration_sec=0.0)

    monkeypatch.setattr(autopilot_module, "run_command", fake_run_command)

    autopilot_module._run_verify("uv run ruff check .", dry_run=False)

    assert captured.get("env") is None
