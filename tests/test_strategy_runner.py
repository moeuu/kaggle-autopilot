from __future__ import annotations

import json
import subprocess
from pathlib import Path

from kagglebot.agents import strategy_runner
from kagglebot.agents.identity import resolve_oracle_model
from kagglebot.exec_utils import CommandResult


def test_run_strategy_uses_full_auto_and_sandbox_flags_when_supported(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLEBOT_AGENT_SANDBOX_MODE", "workspace-write")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    captured_args: list[str] = []

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:
        nonlocal captured_args
        captured_args = args
        last_message_path = Path(args[args.index("--output-last-message") + 1])
        last_message_path.write_text("final strategy\n", encoding="utf-8")
        return CommandResult(args=args, returncode=0, stdout="full transcript\n", stderr="", duration_sec=0.01)

    monkeypatch.setattr(strategy_runner, "_supported_flags", lambda: {"--full-auto", "--sandbox"})
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False, engine="codex")

    assert "--full-auto" in captured_args
    assert "--sandbox" in captured_args
    assert "-a" not in captured_args
    assert result.stdout == "final strategy"
    assert (tmp_path / "strategy_exec.txt").read_text(encoding="utf-8") == "full transcript\n"


def test_run_strategy_uses_full_auto_when_short_approval_flag_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLEBOT_AGENT_SANDBOX_MODE", "workspace-write")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    captured_args: list[str] = []

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:
        nonlocal captured_args
        captured_args = args
        return CommandResult(args=args, returncode=0, stdout="fallback output\n", stderr="", duration_sec=0.01)

    monkeypatch.setattr(strategy_runner, "_supported_flags", lambda: {"--full-auto"})
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False, engine="codex")

    assert "--full-auto" in captured_args
    assert "-a" not in captured_args
    assert "--sandbox" not in captured_args
    assert result.stdout == "fallback output"


def test_run_strategy_retries_without_sandbox_on_bootstrap_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLEBOT_AGENT_SANDBOX_MODE", "fallback")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    captured_args: list[list[str]] = []
    calls = {"count": 0}

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:  # noqa: ARG001
        calls["count"] += 1
        captured_args.append(args)
        last_message_path = Path(args[args.index("--output-last-message") + 1])
        last_message_path.parent.mkdir(parents=True, exist_ok=True)
        if calls["count"] == 1:
            last_message_path.write_text("sandbox failed\n", encoding="utf-8")
            return CommandResult(
                args=args,
                returncode=1,
                stdout="",
                stderr="bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted\n",
                duration_sec=0.01,
            )
        last_message_path.write_text("final strategy\n", encoding="utf-8")
        return CommandResult(args=args, returncode=0, stdout="fallback output\n", stderr="", duration_sec=0.01)

    monkeypatch.setattr(
        strategy_runner,
        "_supported_flags",
        lambda: {"--full-auto", "--sandbox", "--dangerously-bypass-approvals-and-sandbox"},
    )
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False, engine="codex")

    assert result.returncode == 0
    assert result.used_sandbox_fallback is True
    assert result.sandbox_failure_excerpt is not None
    assert "bwrap:" in result.sandbox_failure_excerpt
    assert calls["count"] == 2
    assert "--sandbox" in captured_args[0]
    assert "--dangerously-bypass-approvals-and-sandbox" in captured_args[1]
    assert "--full-auto" not in captured_args[1]


def test_run_strategy_falls_back_to_danger_full_access_when_dangerous_flag_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLEBOT_AGENT_SANDBOX_MODE", "fallback")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    captured_args: list[list[str]] = []
    calls = {"count": 0}

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:  # noqa: ARG001
        calls["count"] += 1
        captured_args.append(args)
        last_message_path = Path(args[args.index("--output-last-message") + 1])
        last_message_path.parent.mkdir(parents=True, exist_ok=True)
        if calls["count"] == 1:
            last_message_path.write_text("sandbox failed\n", encoding="utf-8")
            return CommandResult(
                args=args,
                returncode=1,
                stdout="",
                stderr="bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted\n",
                duration_sec=0.01,
            )
        last_message_path.write_text("final strategy\n", encoding="utf-8")
        return CommandResult(args=args, returncode=0, stdout="fallback output\n", stderr="", duration_sec=0.01)

    monkeypatch.setattr(strategy_runner, "_supported_flags", lambda: {"--full-auto", "--sandbox"})
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False, engine="codex")

    assert result.returncode == 0
    assert result.used_sandbox_fallback is True
    assert calls["count"] == 2
    assert "--dangerously-bypass-approvals-and-sandbox" not in captured_args[1]
    assert "--sandbox" in captured_args[1]
    assert "danger-full-access" in captured_args[1]


def test_run_strategy_does_not_retry_non_sandbox_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLEBOT_AGENT_SANDBOX_MODE", "fallback")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    calls = {"count": 0}

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:  # noqa: ARG001
        calls["count"] += 1
        last_message_path = Path(args[args.index("--output-last-message") + 1])
        last_message_path.parent.mkdir(parents=True, exist_ok=True)
        last_message_path.write_text("ordinary failure\n", encoding="utf-8")
        return CommandResult(
            args=args,
            returncode=1,
            stdout="",
            stderr="RuntimeError: ordinary failure\n",
            duration_sec=0.01,
        )

    monkeypatch.setattr(
        strategy_runner,
        "_supported_flags",
        lambda: {"--full-auto", "--sandbox", "--dangerously-bypass-approvals-and-sandbox"},
    )
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False, engine="codex")

    assert result.returncode == 1
    assert result.used_sandbox_fallback is False
    assert result.sandbox_failure_excerpt is None
    assert calls["count"] == 1


def test_run_strategy_uses_permissive_mode_by_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KAGGLEBOT_AGENT_SANDBOX_MODE", raising=False)
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    captured_args: list[str] = []

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:
        nonlocal captured_args
        captured_args = args
        last_message_path = Path(args[args.index("--output-last-message") + 1])
        last_message_path.write_text("final strategy\n", encoding="utf-8")
        return CommandResult(args=args, returncode=0, stdout="permissive output\n", stderr="", duration_sec=0.01)

    monkeypatch.setattr(
        strategy_runner,
        "_supported_flags",
        lambda: {"--full-auto", "--sandbox", "--dangerously-bypass-approvals-and-sandbox"},
    )
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False, engine="codex")

    assert result.returncode == 0
    assert result.sandbox_policy_mode == "permissive"
    assert result.used_sandbox_fallback is False
    assert "--dangerously-bypass-approvals-and-sandbox" in captured_args
    assert "--sandbox" not in captured_args


def test_run_strategy_workspace_write_mode_skips_permissive_retry(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLEBOT_AGENT_SANDBOX_MODE", "workspace-write")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    calls = {"count": 0}

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:  # noqa: ARG001
        calls["count"] += 1
        last_message_path = Path(args[args.index("--output-last-message") + 1])
        last_message_path.parent.mkdir(parents=True, exist_ok=True)
        last_message_path.write_text("sandbox failed\n", encoding="utf-8")
        return CommandResult(
            args=args,
            returncode=1,
            stdout="",
            stderr="bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted\n",
            duration_sec=0.01,
        )

    monkeypatch.setattr(
        strategy_runner,
        "_supported_flags",
        lambda: {"--full-auto", "--sandbox", "--dangerously-bypass-approvals-and-sandbox"},
    )
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False, engine="codex")

    assert result.returncode == 1
    assert result.sandbox_policy_mode == "workspace-write"
    assert result.used_sandbox_fallback is False
    assert calls["count"] == 1


def test_run_strategy_defaults_to_auto_and_uses_oracle_when_available(monkeypatch, tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    captured_args: list[str] = []

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:  # noqa: ARG001
        nonlocal captured_args
        captured_args = args
        return CommandResult(
            args=args,
            returncode=0,
            stdout="oracle strategy output\n",
            stderr="",
            duration_sec=0.01,
        )

    monkeypatch.setenv("KAGGLEBOT_ORACLE_COMMAND", "oracle")
    monkeypatch.setattr(strategy_runner, "_oracle_available", lambda: True)
    monkeypatch.setattr(
        strategy_runner,
        "_maybe_start_oracle_browser",
        lambda extra_args: strategy_runner.OracleBrowserBootstrap(args=[]),
    )
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False)

    assert result.engine == "oracle"
    assert result.stdout == "oracle strategy output"
    assert "--engine" in captured_args
    assert "browser" in captured_args
    assert "--wait" in captured_args
    assert "--model" in captured_args
    assert resolve_oracle_model() in captured_args
    assert "--force" in captured_args
    assert "-p" in captured_args
    assert (
        "Read the attached Kagglebot strategy prompt file and any attached Kagglebot context bundle files."
        in (captured_args[captured_args.index("-p") + 1])
    )
    assert "Authorized benign use" in captured_args[captured_args.index("-p") + 1]
    assert "\n\nstrategy prompt" not in captured_args[captured_args.index("-p") + 1]
    assert "--file" in captured_args
    assert str(tmp_path / "oracle_strategy_prompt.md") in captured_args
    assert (tmp_path / "oracle_strategy_prompt.md").exists()
    assert (tmp_path / "strategy_last_message.txt").read_text(encoding="utf-8") == "oracle strategy output\n"


def test_run_strategy_oracle_does_not_treat_browser_error_transcript_as_response(monkeypatch, tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:  # noqa: ARG001
        output_path = Path(args[args.index("--write-output") + 1])
        output_path.write_text(
            "\U0001f9ff oracle 0.16.0 — test\n"
            "ERROR: Prompt did not appear in conversation before timeout (likely too large).\n"
            "User error (browser-automation): Prompt did not appear in conversation before timeout.\n",
            encoding="utf-8",
        )
        return CommandResult(args=args, returncode=1, stdout="", stderr="", duration_sec=0.01)

    monkeypatch.setattr(
        strategy_runner,
        "_maybe_start_oracle_browser",
        lambda extra_args: strategy_runner.OracleBrowserBootstrap(args=[]),
    )
    monkeypatch.setenv("KAGGLEBOT_ORACLE_FALLBACK_CODEX", "0")
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False, engine="oracle")

    assert result.returncode == 1
    assert "Prompt did not appear in conversation before timeout" in result.stderr


def test_run_strategy_auto_falls_back_to_codex_when_oracle_unavailable(monkeypatch, tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    monkeypatch.setattr(strategy_runner, "_oracle_command", lambda: [])
    fallback_calls: list[tuple[str | None, str | None]] = []

    def fake_codex_strategy(
        prompt_path: Path,
        output_dir: Path,
        *,
        dry_run: bool,
        model: str | None,
        reasoning_effort: str | None,
    ) -> strategy_runner.StrategyResult:
        fallback_calls.append((model, reasoning_effort))
        transcript_path = output_dir / "strategy_exec.txt"
        last_message_path = output_dir / "strategy_last_message.txt"
        transcript_path.write_text("Codex fallback response\n", encoding="utf-8")
        last_message_path.write_text("Codex fallback response\n", encoding="utf-8")
        return strategy_runner.StrategyResult(
            transcript_path=transcript_path,
            last_message_path=last_message_path,
            returncode=0,
            stdout="Codex fallback response",
            stderr="",
            engine="codex",
        )

    monkeypatch.setattr(strategy_runner, "_run_codex_strategy", fake_codex_strategy)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False, engine="auto")

    assert result.engine == "codex"
    assert result.returncode == 0
    assert result.stdout == "Codex fallback response"
    assert fallback_calls == [("gpt-5.6-sol", "xhigh")]
    assert "Oracle-to-Codex fallback" in result.stderr
    assert (tmp_path / "oracle_strategy_exec.txt").exists()
    assert (tmp_path / "oracle_fallback_reason.txt").exists()


def test_run_strategy_oracle_uses_configured_command_and_args(monkeypatch, tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    captured_args: list[str] = []

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:  # noqa: ARG001
        nonlocal captured_args
        captured_args = args
        return CommandResult(args=args, returncode=0, stdout="oracle output\n", stderr="", duration_sec=0.01)

    monkeypatch.setenv("KAGGLEBOT_ORACLE_COMMAND", "npx -y @steipete/oracle")
    monkeypatch.setenv("KAGGLEBOT_ORACLE_ARGS", "--engine browser --browser-manual-login")
    monkeypatch.setenv("KAGGLEBOT_ORACLE_MODEL", "pinned-pro")
    monkeypatch.setattr(
        strategy_runner,
        "_maybe_start_oracle_browser",
        lambda extra_args: strategy_runner.OracleBrowserBootstrap(args=[]),
    )
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False, engine="oracle")

    assert result.engine == "oracle"
    assert captured_args[:3] == ["npx", "-y", "@steipete/oracle"]
    assert "--engine" in captured_args
    assert "browser" in captured_args
    assert "--browser-manual-login" in captured_args
    assert "--wait" in captured_args
    assert captured_args[captured_args.index("--model") + 1] == "pinned-pro"


def test_run_strategy_oracle_uses_two_hour_default_timeout_outside_pytest(monkeypatch, tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    captured_timeout: float | None = None

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:
        nonlocal captured_timeout
        captured_timeout = kwargs.get("timeout")
        return CommandResult(args=args, returncode=0, stdout="oracle output\n", stderr="", duration_sec=0.01)

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("KAGGLEBOT_STRATEGY_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("KAGGLEBOT_ORACLE_STRATEGY_TIMEOUT_SEC", raising=False)
    monkeypatch.setattr(
        strategy_runner,
        "_maybe_start_oracle_browser",
        lambda extra_args: strategy_runner.OracleBrowserBootstrap(args=[]),
    )
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False, engine="oracle")

    assert result.engine == "oracle"
    assert captured_timeout == 7200.0


def test_run_strategy_oracle_timeout_falls_back_to_sol_xhigh(monkeypatch, tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    calls: list[tuple[list[str], float | None]] = []

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:
        calls.append((args, kwargs.get("timeout")))
        if "--write-output" in args:
            raise subprocess.TimeoutExpired(args, kwargs.get("timeout"))
        last_message_path = Path(args[args.index("--output-last-message") + 1])
        last_message_path.write_text("Codex recovered the strategy\n", encoding="utf-8")
        return CommandResult(
            args=args,
            returncode=0,
            stdout="Codex transcript\n",
            stderr="",
            duration_sec=0.01,
        )

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("KAGGLEBOT_STRATEGY_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("KAGGLEBOT_ORACLE_STRATEGY_TIMEOUT_SEC", raising=False)
    monkeypatch.setattr(
        strategy_runner,
        "_maybe_start_oracle_browser",
        lambda extra_args: strategy_runner.OracleBrowserBootstrap(args=[]),
    )
    monkeypatch.setattr(strategy_runner, "_supported_flags", lambda: set())
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False, engine="oracle")

    oracle_args, oracle_timeout = calls[0]
    codex_args, _ = calls[1]
    assert "--write-output" in oracle_args
    assert oracle_timeout == 7200.0
    assert codex_args[codex_args.index("-m") + 1] == "gpt-5.6-sol"
    assert 'model_reasoning_effort="xhigh"' in codex_args
    assert result.engine == "codex"
    assert result.stdout == "Codex recovered the strategy"
    assert "timed out after 7200s" in result.stderr
    assert "timed out after 7200s" in (tmp_path / "oracle_strategy_exec.txt").read_text(encoding="utf-8")


def test_run_strategy_oracle_timeout_override_wins_over_global_timeout(monkeypatch, tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    captured_timeout: float | None = None

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:
        nonlocal captured_timeout
        captured_timeout = kwargs.get("timeout")
        return CommandResult(args=args, returncode=0, stdout="oracle output\n", stderr="", duration_sec=0.01)

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("KAGGLEBOT_STRATEGY_TIMEOUT_SEC", "10")
    monkeypatch.setenv("KAGGLEBOT_ORACLE_STRATEGY_TIMEOUT_SEC", "5400")
    monkeypatch.setattr(
        strategy_runner,
        "_maybe_start_oracle_browser",
        lambda extra_args: strategy_runner.OracleBrowserBootstrap(args=[]),
    )
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False, engine="oracle")

    assert result.engine == "oracle"
    assert captured_timeout == 5400.0


def test_run_strategy_oracle_auto_bootstraps_remote_chrome(monkeypatch, tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    captured_args: list[str] = []
    closed = {"value": False}

    class FakeBootstrap:
        args = ["--remote-chrome", "127.0.0.1:9222"]

        def close(self) -> None:
            closed["value"] = True

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:  # noqa: ARG001
        nonlocal captured_args
        captured_args = args
        return CommandResult(args=args, returncode=0, stdout="oracle output\n", stderr="", duration_sec=0.01)

    monkeypatch.delenv("KAGGLEBOT_ORACLE_ARGS", raising=False)
    monkeypatch.setattr(strategy_runner, "_maybe_start_oracle_browser", lambda extra_args: FakeBootstrap())
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False, engine="oracle")

    assert result.engine == "oracle"
    assert captured_args[captured_args.index("--remote-chrome") + 1] == "127.0.0.1:9222"
    assert closed["value"] is True


def test_oracle_browser_bootstrap_skips_when_route_is_explicit(monkeypatch) -> None:
    called = {"value": False}

    def fake_ready(port: int) -> bool:  # noqa: ARG001
        called["value"] = True
        return True

    monkeypatch.setattr(strategy_runner, "_oracle_remote_chrome_ready", fake_ready)

    result = strategy_runner._maybe_start_oracle_browser(["--remote-chrome", "example.com:9222"])

    assert result.args == []
    assert called["value"] is False


def test_oracle_browser_bootstrap_reuses_ready_remote_chrome(monkeypatch) -> None:
    monkeypatch.delenv("KAGGLEBOT_ORACLE_ENGINE", raising=False)
    monkeypatch.setenv("KAGGLEBOT_ORACLE_BROWSER_MODEL_STRATEGY", "ignore")
    monkeypatch.setenv("KAGGLEBOT_ORACLE_BROWSER_THINKING_TIME", "light")
    monkeypatch.setenv("KAGGLEBOT_ORACLE_BROWSER_PORT", "9333")
    monkeypatch.setattr(strategy_runner, "_oracle_remote_chrome_ready", lambda port: port == 9333)

    result = strategy_runner._maybe_start_oracle_browser([])

    assert result.args == [
        "--remote-chrome",
        "127.0.0.1:9333",
        "--browser-model-strategy",
        "select",
        "--browser-thinking-time",
        "extended",
        "--browser-attachments",
        "auto",
        "--browser-input-timeout",
        "3600s",
        "--browser-timeout",
        "24h",
        "--browser-archive",
        "always",
    ]
    assert result.process is None
    assert result.temp_profile_dir is None


def test_oracle_browser_bootstrap_keeps_explicit_model_strategy(monkeypatch) -> None:
    monkeypatch.setattr(strategy_runner, "_oracle_remote_chrome_ready", lambda port: True)  # noqa: ARG005

    result = strategy_runner._maybe_start_oracle_browser(["--browser-model-strategy", "select"])

    assert result.args == [
        "--remote-chrome",
        "127.0.0.1:9222",
        "--browser-thinking-time",
        "extended",
        "--browser-attachments",
        "auto",
        "--browser-input-timeout",
        "3600s",
        "--browser-timeout",
        "24h",
        "--browser-archive",
        "always",
    ]


def test_oracle_browser_bootstrap_keeps_explicit_archive_mode(monkeypatch) -> None:
    monkeypatch.setattr(strategy_runner, "_oracle_remote_chrome_ready", lambda port: True)  # noqa: ARG005

    result = strategy_runner._maybe_start_oracle_browser(["--browser-archive", "never"])

    assert "--browser-archive" not in result.args


def test_oracle_browser_bootstrap_does_not_duplicate_explicit_thinking_time(monkeypatch) -> None:
    monkeypatch.setattr(strategy_runner, "_oracle_remote_chrome_ready", lambda port: True)  # noqa: ARG005

    result = strategy_runner._maybe_start_oracle_browser(["--browser-thinking-time", "light"])

    assert "--browser-thinking-time" not in result.args


def test_prepare_oracle_chrome_profile_excludes_live_browser_locks(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "Local State").write_text("state", encoding="utf-8")
    for name in strategy_runner._ORACLE_CHROME_PROFILE_ROOT_EXCLUDES:
        (source / name).write_text("live", encoding="utf-8")
    destination = tmp_path / "copy"
    destination.mkdir()

    monkeypatch.delenv("KAGGLEBOT_ORACLE_CHROME_USER_DATA_DIR", raising=False)
    monkeypatch.setenv("KAGGLEBOT_ORACLE_CHROME_COPY_PROFILE", str(source))
    monkeypatch.setattr(strategy_runner, "mkdtemp", lambda prefix: str(destination))
    monkeypatch.setattr(strategy_runner.shutil, "which", lambda command: None)

    profile_dir, temp_profile_dir = strategy_runner._prepare_oracle_chrome_profile()

    assert profile_dir == destination
    assert temp_profile_dir == destination
    assert (destination / "Local State").read_text(encoding="utf-8") == "state"
    for name in strategy_runner._ORACLE_CHROME_PROFILE_ROOT_EXCLUDES:
        assert not (destination / name).exists()


def test_prepare_oracle_chrome_profile_passes_root_excludes_to_rsync(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "copy"
    destination.mkdir()
    captured_args: list[str] = []

    def fake_run(args: list[str], **kwargs) -> None:  # noqa: ARG001
        captured_args.extend(args)

    monkeypatch.delenv("KAGGLEBOT_ORACLE_CHROME_USER_DATA_DIR", raising=False)
    monkeypatch.setenv("KAGGLEBOT_ORACLE_CHROME_COPY_PROFILE", str(source))
    monkeypatch.setattr(strategy_runner, "mkdtemp", lambda prefix: str(destination))
    monkeypatch.setattr(strategy_runner.shutil, "which", lambda command: "/usr/bin/rsync")
    monkeypatch.setattr(strategy_runner.subprocess, "run", fake_run)

    strategy_runner._prepare_oracle_chrome_profile()

    for name in strategy_runner._ORACLE_CHROME_PROFILE_ROOT_EXCLUDES:
        assert f"--exclude=/{name}" in captured_args


def test_oracle_browser_bootstrap_skips_api_engine(monkeypatch) -> None:
    monkeypatch.setenv("KAGGLEBOT_ORACLE_ENGINE", "api")

    result = strategy_runner._maybe_start_oracle_browser([])

    assert result.args == []


def test_run_strategy_oracle_allows_api_engine_override(monkeypatch, tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    captured_args: list[str] = []

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:  # noqa: ARG001
        nonlocal captured_args
        captured_args = args
        return CommandResult(args=args, returncode=0, stdout="oracle output\n", stderr="", duration_sec=0.01)

    monkeypatch.setenv("KAGGLEBOT_ORACLE_ENGINE", "api")
    monkeypatch.setattr(
        strategy_runner,
        "_maybe_start_oracle_browser",
        lambda extra_args: strategy_runner.OracleBrowserBootstrap(args=[]),
    )
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False, engine="oracle")

    assert result.engine == "oracle"
    assert captured_args[captured_args.index("--engine") + 1] == "api"
    assert "--file" not in captured_args
    assert "strategy prompt" in captured_args[captured_args.index("-p") + 1]


def test_run_strategy_oracle_can_disable_inline_prompt(monkeypatch, tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    captured_args: list[str] = []

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:  # noqa: ARG001
        nonlocal captured_args
        captured_args = args
        return CommandResult(args=args, returncode=0, stdout="oracle output\n", stderr="", duration_sec=0.01)

    monkeypatch.setenv("KAGGLEBOT_ORACLE_INLINE_PROMPT", "0")
    monkeypatch.setattr(
        strategy_runner,
        "_maybe_start_oracle_browser",
        lambda extra_args: strategy_runner.OracleBrowserBootstrap(args=[]),
    )
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False, engine="oracle")

    assert result.engine == "oracle"
    assert "--file" in captured_args
    assert str(tmp_path / "oracle_strategy_prompt.md") in captured_args


def test_run_strategy_oracle_attaches_context_bundle_when_inline_disabled(monkeypatch, tmp_path: Path) -> None:
    strategy_dir = tmp_path / "agent" / "strategy"
    strategy_dir.mkdir(parents=True)
    prompt_path = strategy_dir / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    bundle_path = tmp_path / "agent" / "strategy_context_bundle.md"
    bundle_path.write_text("competition context bundle", encoding="utf-8")
    captured_args: list[str] = []

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:  # noqa: ARG001
        nonlocal captured_args
        captured_args = args
        return CommandResult(args=args, returncode=0, stdout="oracle output\n", stderr="", duration_sec=0.01)

    monkeypatch.setenv("KAGGLEBOT_ORACLE_INLINE_PROMPT", "0")
    monkeypatch.setattr(
        strategy_runner,
        "_maybe_start_oracle_browser",
        lambda extra_args: strategy_runner.OracleBrowserBootstrap(args=[]),
    )
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, strategy_dir, dry_run=False, engine="oracle")

    assert result.engine == "oracle"
    file_index = captured_args.index("--file")
    attached = captured_args[file_index + 1 :]
    assert str(strategy_dir / "oracle_strategy_prompt.md") in attached
    assert str(bundle_path) in attached


def test_oracle_context_attaches_complete_context_and_permitted_package(monkeypatch, tmp_path: Path) -> None:
    context_dir = tmp_path / "artifacts" / "demo" / "context"
    strategy_dir = context_dir / "agent" / "strategy"
    data_dir = context_dir.parent / "data"
    strategy_dir.mkdir(parents=True)
    data_dir.mkdir()
    prompt_path = strategy_dir / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    (context_dir / "rules.md").write_text(
        "Competition data may be processed by external tools for this competition.\n",
        encoding="utf-8",
    )
    (context_dir / "overview.md").write_text("complete overview\n", encoding="utf-8")
    (context_dir / "dataset_profile.json").write_text("{}\n", encoding="utf-8")
    (context_dir / "top1_public.json").write_text('{"score": 0.9}\n', encoding="utf-8")
    (context_dir / "kaggle_discovery.md").write_text(
        "# Kaggle Discovery Snapshot\n\n- ARC model candidate\n",
        encoding="utf-8",
    )
    brief_path = context_dir / "agent" / "brief_for_strategy.md"
    brief_path.parent.mkdir(exist_ok=True)
    brief_path.write_text("complete Codex brief\n", encoding="utf-8")
    package_path = data_dir / "demo.zip"
    package_path.write_bytes(b"x" * (1024 * 1024 + 1))
    captured_args: list[str] = []

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:  # noqa: ARG001
        nonlocal captured_args
        captured_args = args
        return CommandResult(args=args, returncode=0, stdout="oracle output\n", stderr="", duration_sec=0.01)

    monkeypatch.setattr(
        strategy_runner,
        "_maybe_start_oracle_browser",
        lambda extra_args: strategy_runner.OracleBrowserBootstrap(args=[]),
    )
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    strategy_runner.run_strategy(prompt_path, strategy_dir, dry_run=False, engine="oracle")

    attached = captured_args[captured_args.index("--file") + 1 :]
    context_bundle_path = strategy_dir / "oracle_canonical_context.md"
    assert str(context_bundle_path) in attached
    context_bundle = context_bundle_path.read_text(encoding="utf-8")
    assert str(context_dir / "rules.md") in context_bundle
    assert "Competition data may be processed" in context_bundle
    assert "complete overview" in context_bundle
    assert '"score": 0.9' in context_bundle
    assert "ARC model candidate" in context_bundle
    assert "complete Codex brief" in context_bundle
    assert str(package_path) in attached
    assert captured_args[captured_args.index("--max-file-size-bytes") + 1] == str(package_path.stat().st_size)
    manifest_path = strategy_dir / "oracle_context_manifest.md"
    assert str(manifest_path) in attached
    assert "attached: 1 canonical package file(s)" in manifest_path.read_text(encoding="utf-8")


def test_oracle_improvement_context_resolves_competition_root_and_attaches_runtime_files(tmp_path: Path) -> None:
    competition_dir = tmp_path / "artifacts" / "demo"
    context_dir = competition_dir / "context"
    iter_dir = competition_dir / "runs" / "run-1" / "iter-2"
    strategy_dir = iter_dir / "agent" / "improve_strategy-01"
    logs_dir = iter_dir / "logs"
    strategy_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    (context_dir).mkdir(parents=True)
    (context_dir / "overview.md").write_text("competition overview\n", encoding="utf-8")
    (competition_dir / "plan.json").write_text('{"target_metric":"rmse"}\n', encoding="utf-8")
    kernel_path = competition_dir / "kernel" / "kernel.py"
    kernel_path.parent.mkdir()
    kernel_path.write_text("print('current kernel')\n", encoding="utf-8")
    run_path = iter_dir.parent / "run.json"
    run_path.write_text('{"status":"running"}\n', encoding="utf-8")
    (iter_dir / "metrics.json").write_text('{"offline_value":0.42}\n', encoding="utf-8")
    (iter_dir / "diagnostics.md").write_text("validation drift detected\n", encoding="utf-8")
    (iter_dir / "iteration_evidence.json").write_text(
        '{"transitions_observed":[{"hypothesis_outcome":"not_supported"}]}\n',
        encoding="utf-8",
    )
    (iter_dir / "iteration_evidence.md").write_text("# Iteration Evidence\n", encoding="utf-8")
    (iter_dir / "kernel_before_improvement.py").write_text("print('frozen kernel')\n", encoding="utf-8")
    (iter_dir / "plan_before_improvement.json").write_text('{"model":"baseline"}\n', encoding="utf-8")
    (logs_dir / "kernel_stderr.txt").write_text("CUDA out of memory\n", encoding="utf-8")
    prompt_path = strategy_dir / "gpt_improvement_prompt.md"
    prompt_path.write_text("improve this run", encoding="utf-8")

    plan = strategy_runner._build_oracle_attachment_plan(
        prompt_path=prompt_path,
        oracle_prompt_path=strategy_dir / "oracle_strategy_prompt.md",
        output_dir=strategy_dir,
        inline_prompt=True,
    )

    assert strategy_runner._oracle_context_dir(prompt_path) == context_dir
    assert strategy_dir / "oracle_context_manifest.md" in plan.paths
    bundle = (strategy_dir / "oracle_canonical_context.md").read_text(encoding="utf-8")
    for expected in (
        kernel_path,
        run_path,
        iter_dir / "metrics.json",
        iter_dir / "diagnostics.md",
        iter_dir / "iteration_evidence.json",
        iter_dir / "iteration_evidence.md",
        iter_dir / "kernel_before_improvement.py",
        iter_dir / "plan_before_improvement.json",
    ):
        assert str(expected) in bundle
    assert "CUDA out of memory" in bundle
    assert "not_supported" in bundle


def test_oracle_autofix_context_attaches_saved_error_and_current_kernel(tmp_path: Path) -> None:
    competition_dir = tmp_path / "artifacts" / "demo"
    context_dir = competition_dir / "context"
    attempt_dir = competition_dir / "runs" / "run-1" / "autofix" / "attempt-1"
    strategy_dir = attempt_dir / "gpt_strategy"
    strategy_dir.mkdir(parents=True)
    context_dir.mkdir(parents=True)
    (context_dir / "dataset_profile.json").write_text("{}\n", encoding="utf-8")
    kernel_path = competition_dir / "kernel" / "kernel.py"
    kernel_path.parent.mkdir()
    kernel_path.write_text("raise RuntimeError('bad shape')\n", encoding="utf-8")
    error_path = attempt_dir / "error-1.txt"
    error_path.write_text("ValueError: submission row mismatch\n", encoding="utf-8")
    prompt_path = strategy_dir / "gpt_strategy_prompt.md"
    prompt_path.write_text("repair the failure", encoding="utf-8")

    strategy_runner._build_oracle_attachment_plan(
        prompt_path=prompt_path,
        oracle_prompt_path=strategy_dir / "oracle_strategy_prompt.md",
        output_dir=strategy_dir,
        inline_prompt=True,
    )

    bundle = (strategy_dir / "oracle_canonical_context.md").read_text(encoding="utf-8")
    assert str(error_path) in bundle
    assert "ValueError: submission row mismatch" in bundle
    assert str(kernel_path) in bundle


def test_oracle_repository_improvement_attaches_report_backlog_and_knowledge_context(tmp_path: Path) -> None:
    improvement_dir = tmp_path / "artifacts" / "_self_improvement"
    transaction_dir = improvement_dir / "transactions" / "20260714T000000Z"
    strategy_dir = transaction_dir / "strategy"
    strategy_dir.mkdir(parents=True)
    expected_files = {
        "latest.json": '{"recommended_actions":["repair validation"]}\n',
        "latest.md": "# Improvement report\n",
        "strategy_context.md": "Prioritize failed submissions\n",
        "experiment_backlog.json": '[{"title":"validation redesign"}]\n',
        "skill_candidates.json": '[{"name":"submission repair"}]\n',
        "outcomes.jsonl": '{"status":"failed"}\n',
    }
    for name, content in expected_files.items():
        (improvement_dir / name).write_text(content, encoding="utf-8")
    prompt_path = transaction_dir / "strategy_prompt.md"
    prompt_path.write_text("review repository", encoding="utf-8")

    plan = strategy_runner._build_oracle_attachment_plan(
        prompt_path=prompt_path,
        oracle_prompt_path=strategy_dir / "oracle_strategy_prompt.md",
        output_dir=strategy_dir,
        inline_prompt=True,
    )

    assert strategy_dir / "oracle_context_manifest.md" in plan.paths
    bundle = (strategy_dir / "oracle_canonical_context.md").read_text(encoding="utf-8")
    for name, content in expected_files.items():
        assert str(improvement_dir / name) in bundle
        assert content.strip() in bundle


def test_oracle_context_never_attaches_data_for_rules_with_third_party_restriction(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KAGGLEBOT_ORACLE_COMPETITION_DATA", raising=False)
    context_dir = tmp_path / "artifacts" / "demo" / "context"
    strategy_dir = context_dir / "agent" / "strategy"
    data_dir = context_dir.parent / "data"
    strategy_dir.mkdir(parents=True)
    data_dir.mkdir()
    prompt_path = strategy_dir / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    (context_dir / "rules.md").write_text(
        "You agree not to transmit Competition Data to any party not participating in the Competition.\n",
        encoding="utf-8",
    )
    package_path = data_dir / "demo.zip"
    package_path.write_bytes(b"competition data")

    plan = strategy_runner._build_oracle_attachment_plan(
        prompt_path=prompt_path,
        oracle_prompt_path=strategy_dir / "oracle_strategy_prompt.md",
        output_dir=strategy_dir,
        inline_prompt=True,
    )

    assert package_path not in plan.paths
    assert plan.data_paths == ()
    assert "restrict third-party data transmission" in plan.data_decision
    manifest = (strategy_dir / "oracle_context_manifest.md").read_text(encoding="utf-8")
    assert 'matched "not to transmit"' in manifest


def test_oracle_context_owner_authorized_mode_attaches_package_despite_transmission_wording(
    monkeypatch, tmp_path: Path
) -> None:
    context_dir = tmp_path / "artifacts" / "demo" / "context"
    data_dir = context_dir.parent / "data"
    context_dir.mkdir(parents=True)
    data_dir.mkdir()
    (context_dir / "rules.md").write_text(
        "You agree not to transmit Competition Data to any party not participating in the Competition.\n",
        encoding="utf-8",
    )
    package_path = data_dir / "demo.zip"
    package_path.write_bytes(b"competition data")
    monkeypatch.setenv("KAGGLEBOT_ORACLE_COMPETITION_DATA", "owner-authorized")

    paths, decision = strategy_runner._oracle_competition_data_attachments(context_dir)

    assert paths == [package_path]
    assert "owner-authorized processing" in decision


def test_run_strategy_oracle_keeps_explicit_extra_engine_and_wait(monkeypatch, tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    captured_args: list[str] = []

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:  # noqa: ARG001
        nonlocal captured_args
        captured_args = args
        return CommandResult(args=args, returncode=0, stdout="oracle output\n", stderr="", duration_sec=0.01)

    monkeypatch.setenv("KAGGLEBOT_ORACLE_ARGS", "--engine api --wait")
    monkeypatch.setattr(
        strategy_runner,
        "_maybe_start_oracle_browser",
        lambda extra_args: strategy_runner.OracleBrowserBootstrap(args=[]),
    )
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False, engine="oracle")

    assert result.engine == "oracle"
    assert captured_args.count("--engine") == 1
    assert captured_args[captured_args.index("--engine") + 1] == "api"
    assert captured_args.count("--wait") == 1


def test_find_oracle_session_archive_status_matches_write_output_path(monkeypatch, tmp_path: Path) -> None:
    oracle_home = tmp_path / "oracle"
    session_dir = oracle_home / "sessions" / "run-1"
    session_dir.mkdir(parents=True)
    transcript_path = tmp_path / "output" / "strategy_exec.txt"
    transcript_path.parent.mkdir()
    (session_dir / "meta.json").write_text(
        json.dumps(
            {
                "options": {"writeOutputPath": str(transcript_path.resolve())},
                "browser": {
                    "archive": {
                        "archived": False,
                        "reason": "conversation-menu-not-found",
                        "conversationUrl": "https://chatgpt.com/c/example",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ORACLE_HOME", str(oracle_home))

    report = strategy_runner._find_oracle_session_archive_status(transcript_path)

    assert report["conversationUrl"] == "https://chatgpt.com/c/example"
    assert report["oracleSession"] == str(session_dir)


def test_archive_oracle_conversation_via_cdp_parses_success(monkeypatch, tmp_path: Path) -> None:
    module_path = tmp_path / "chrome-remote-interface"
    module_path.mkdir()
    monkeypatch.setattr(strategy_runner, "_oracle_cdp_module_path", lambda: module_path)
    monkeypatch.setattr(strategy_runner, "_oracle_node_command", lambda: "/usr/bin/node")
    monkeypatch.setattr(
        strategy_runner,
        "run_command",
        lambda args, **kwargs: CommandResult(
            args=args,
            returncode=0,
            stdout='{"archived":true,"status":200,"response":"{\\"success\\":true}"}\n',
            stderr="",
            duration_sec=0.01,
        ),
    )

    report = strategy_runner._archive_oracle_conversation_via_cdp(
        conversation_url="https://chatgpt.com/c/example",
        host="127.0.0.1",
        port=9222,
    )

    assert report["archived"] is True
    assert report["fallbackAttempted"] is True


def test_start_oracle_browser_attachment_compatibility_uses_exact_filenames(monkeypatch, tmp_path: Path) -> None:
    module_path = tmp_path / "chrome-remote-interface"
    module_path.mkdir()
    attachment_a = tmp_path / "context.md"
    attachment_b = tmp_path / "data-part.zip"
    attachment_a.touch()
    attachment_b.touch()
    captured: dict[str, object] = {}

    class FakeProcess:
        def poll(self):
            return None

        def terminate(self):
            captured["terminated"] = True

        def wait(self, *, timeout):
            captured["wait_timeout"] = timeout

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(strategy_runner, "_oracle_cdp_module_path", lambda: module_path)
    monkeypatch.setattr(strategy_runner, "_oracle_node_command", lambda: "/opt/node")
    monkeypatch.setattr(strategy_runner.subprocess, "Popen", fake_popen)

    compatibility = strategy_runner._start_oracle_browser_attachment_compatibility(
        args=["--remote-chrome", "127.0.0.1:9333"],
        attachment_paths=[attachment_a, attachment_b, attachment_a],
    )
    compatibility.close()

    args = captured["args"]
    assert args[:3] == ["/opt/node", "-e", strategy_runner._ORACLE_ATTACHMENT_COMPATIBILITY_CDP_SCRIPT]
    assert args[3:6] == [str(module_path), "127.0.0.1", "9333"]
    assert json.loads(args[6]) == ["context.md", "data-part.zip"]
    assert captured["kwargs"] == {
        "stdout": strategy_runner.subprocess.DEVNULL,
        "stderr": strategy_runner.subprocess.DEVNULL,
    }
    assert captured["terminated"] is True
    assert captured["wait_timeout"] == 5


def test_oracle_attachment_compatibility_script_is_locale_independent() -> None:
    script = strategy_runner._ORACLE_ATTACHMENT_COMPATIBILITY_CDP_SCRIPT

    assert "label.includes(candidate)" in script
    assert "label.includes(stem)" in script
    assert "'Remove ' + name" in script
    assert "削除" not in script


def test_run_strategy_preserves_successful_oracle_output_when_archive_is_unverified(
    monkeypatch,
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    monkeypatch.setattr(
        strategy_runner,
        "_maybe_start_oracle_browser",
        lambda extra_args: strategy_runner.OracleBrowserBootstrap(args=["--remote-chrome", "127.0.0.1:9222"]),
    )
    monkeypatch.setattr(
        strategy_runner,
        "run_command",
        lambda args, **kwargs: CommandResult(
            args=args,
            returncode=0,
            stdout="Oracle response",
            stderr="",
            duration_sec=0.01,
        ),
    )
    monkeypatch.setattr(
        strategy_runner,
        "_ensure_oracle_conversation_archived",
        lambda **kwargs: {"archived": False, "fallbackReason": "verification-failed"},
    )

    result = strategy_runner.run_strategy(prompt_path, tmp_path, engine="oracle")

    assert result.stdout == "Oracle response"
    assert result.returncode == 0
    assert "archive verification warning" in result.stderr.lower()


def test_oracle_context_splits_large_browser_package_without_changing_bytes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    context_dir = tmp_path / "artifacts" / "demo" / "context"
    strategy_dir = context_dir / "agent" / "strategy"
    data_dir = context_dir.parent / "data"
    strategy_dir.mkdir(parents=True)
    data_dir.mkdir()
    prompt_path = strategy_dir / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    (context_dir / "rules.md").write_text("Data processing is permitted.\n", encoding="utf-8")
    package_path = data_dir / "demo.zip"
    package_bytes = bytes(range(251)) * 41
    package_path.write_bytes(package_bytes)
    monkeypatch.setattr(strategy_runner, "_ORACLE_REMOTE_DATA_PART_BYTES", 1024)

    plan = strategy_runner._build_oracle_attachment_plan(
        prompt_path=prompt_path,
        oracle_prompt_path=strategy_dir / "oracle_strategy_prompt.md",
        output_dir=strategy_dir,
        inline_prompt=True,
        split_large_data=True,
    )

    part_paths = [path for path in plan.paths if path.parent.name == "oracle_data_parts"]
    assert len(part_paths) == 11
    assert all(path.stat().st_size <= 1024 for path in part_paths)
    assert b"".join(path.read_bytes() for path in part_paths) == package_bytes
    manifest = (strategy_dir / "oracle_context_manifest.md").read_text(encoding="utf-8")
    assert "concatenate the following parts in listed order" in manifest
    assert strategy_runner._sha256_file(package_path) in manifest


def test_run_strategy_does_not_reuse_stale_oracle_transcript(monkeypatch, tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    (tmp_path / "strategy_exec.txt").write_text("stale Oracle response\n", encoding="utf-8")
    monkeypatch.setattr(
        strategy_runner,
        "_maybe_start_oracle_browser",
        lambda extra_args: strategy_runner.OracleBrowserBootstrap(args=[]),
    )
    monkeypatch.setenv("KAGGLEBOT_ORACLE_FALLBACK_CODEX", "0")
    monkeypatch.setattr(
        strategy_runner,
        "run_command",
        lambda args, **kwargs: CommandResult(
            args=args,
            returncode=1,
            stdout="current Oracle transfer error",
            stderr="attachment failed",
            duration_sec=0.01,
        ),
    )

    result = strategy_runner.run_strategy(prompt_path, tmp_path, engine="oracle")

    assert result.returncode == 1
    assert result.stdout == "current Oracle transfer error"
    assert "stale Oracle response" not in result.stdout


def test_run_strategy_accepts_current_oracle_response_despite_cleanup_exit(monkeypatch, tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    monkeypatch.setattr(
        strategy_runner,
        "_maybe_start_oracle_browser",
        lambda extra_args: strategy_runner.OracleBrowserBootstrap(args=[]),
    )

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:  # noqa: ARG001
        output_path = Path(args[args.index("--write-output") + 1])
        output_path.write_text("current complete Oracle response\n", encoding="utf-8")
        return CommandResult(args=args, returncode=1, stdout="", stderr="cleanup failed", duration_sec=0.01)

    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, engine="oracle")

    assert result.returncode == 0
    assert result.stdout == "current complete Oracle response"
    assert "validating the response content" in result.stderr
