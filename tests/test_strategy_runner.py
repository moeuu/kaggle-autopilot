from __future__ import annotations

from pathlib import Path

from kagglebot.agents import strategy_runner
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
    assert "gpt-5.5-pro" in captured_args
    assert "--force" in captured_args
    assert "-p" in captured_args
    assert (
        "Use the Kagglebot strategy prompt below as the complete context."
        in captured_args[captured_args.index("-p") + 1]
    )
    assert "strategy prompt" in captured_args[captured_args.index("-p") + 1]
    assert "--file" not in captured_args
    assert (tmp_path / "oracle_strategy_prompt.md").exists()
    assert (tmp_path / "strategy_last_message.txt").read_text(encoding="utf-8") == "oracle strategy output\n"


def test_run_strategy_auto_falls_back_to_codex_when_oracle_unavailable(monkeypatch, tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("strategy prompt", encoding="utf-8")
    captured_args: list[str] = []

    def fake_run_command(args: list[str], **kwargs) -> CommandResult:  # noqa: ARG001
        nonlocal captured_args
        captured_args = args
        last_message_path = Path(args[args.index("--output-last-message") + 1])
        last_message_path.write_text("codex strategy output\n", encoding="utf-8")
        return CommandResult(args=args, returncode=0, stdout="full transcript\n", stderr="", duration_sec=0.01)

    monkeypatch.setattr(strategy_runner, "_oracle_available", lambda: False)
    monkeypatch.setattr(strategy_runner, "_supported_flags", lambda: {"--full-auto"})
    monkeypatch.setattr(strategy_runner, "run_command", fake_run_command)

    result = strategy_runner.run_strategy(prompt_path, tmp_path, dry_run=False, engine="auto")

    assert result.engine == "codex"
    assert captured_args[0] == "codex"
    assert result.stdout == "codex strategy output"


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
    monkeypatch.setenv("KAGGLEBOT_ORACLE_MODEL", "gpt-5.5-pro")
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


def test_run_strategy_oracle_uses_long_default_timeout_outside_pytest(monkeypatch, tmp_path: Path) -> None:
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
    assert captured_timeout == 3900.0


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
    monkeypatch.setenv("KAGGLEBOT_ORACLE_BROWSER_PORT", "9333")
    monkeypatch.setattr(strategy_runner, "_oracle_remote_chrome_ready", lambda port: port == 9333)

    result = strategy_runner._maybe_start_oracle_browser([])

    assert result.args == [
        "--remote-chrome",
        "127.0.0.1:9333",
        "--browser-model-strategy",
        "ignore",
        "--browser-attachments",
        "auto",
        "--browser-input-timeout",
        "600s",
        "--browser-timeout",
        "60m",
    ]
    assert result.process is None
    assert result.temp_profile_dir is None


def test_oracle_browser_bootstrap_keeps_explicit_model_strategy(monkeypatch) -> None:
    monkeypatch.setattr(strategy_runner, "_oracle_remote_chrome_ready", lambda port: True)  # noqa: ARG005

    result = strategy_runner._maybe_start_oracle_browser(["--browser-model-strategy", "select"])

    assert result.args == [
        "--remote-chrome",
        "127.0.0.1:9222",
        "--browser-attachments",
        "auto",
        "--browser-input-timeout",
        "600s",
        "--browser-timeout",
        "60m",
    ]


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
