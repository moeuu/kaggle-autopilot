from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from kagglebot.agents.identity import STRATEGY_AGENT, render_prompt_identity
from kagglebot.agents.sandbox_fallback import (
    append_sandbox_args,
    detect_sandbox_startup_failure,
    resolve_agent_sandbox_mode,
)
from kagglebot.exec_utils import CommandResult, run_command

_DEFAULT_MODEL = STRATEGY_AGENT.model
_DEFAULT_REASONING_EFFORT = STRATEGY_AGENT.reasoning_effort
_DEFAULT_TIMEOUT_SEC = 600.0
_PYTEST_TIMEOUT_SEC = 2.0
_RUNNER_LABEL = STRATEGY_AGENT.log_alias


@dataclass(frozen=True)
class StrategyResult:
    transcript_path: Path
    last_message_path: Path
    returncode: int
    stdout: str
    stderr: str
    sandbox_policy_mode: str = "permissive"
    used_sandbox_fallback: bool = False
    sandbox_failure_excerpt: str | None = None


def run_strategy(prompt_path: Path, output_dir: Path, *, dry_run: bool = False) -> StrategyResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = render_prompt_identity(prompt_path.read_text(encoding="utf-8"))
    transcript_path = output_dir / "strategy_exec.txt"
    last_message_path = output_dir / "strategy_last_message.txt"

    if dry_run:
        transcript_path.write_text("", encoding="utf-8")
        last_message_path.write_text("DRY RUN: strategy not executed.\n", encoding="utf-8")
        return StrategyResult(
            transcript_path=transcript_path,
            last_message_path=last_message_path,
            returncode=0,
            stdout="",
            stderr="",
            sandbox_policy_mode=resolve_agent_sandbox_mode(),
        )

    normalized_effort = _normalize_reasoning_effort(_DEFAULT_REASONING_EFFORT)
    timeout = float(os.environ.get("KAGGLEBOT_STRATEGY_TIMEOUT_SEC", str(_DEFAULT_TIMEOUT_SEC)))
    if os.environ.get("PYTEST_CURRENT_TEST"):
        timeout = float(os.environ.get("KAGGLEBOT_PYTEST_STRATEGY_TIMEOUT_SEC", str(_PYTEST_TIMEOUT_SEC)))
    args = [
        STRATEGY_AGENT.cli_command,
        "exec",
        "-m",
        _DEFAULT_MODEL,
        "-c",
        f'model_reasoning_effort="{normalized_effort}"',
    ]
    supported = _supported_flags()
    sandbox_policy_mode = resolve_agent_sandbox_mode()
    sandbox_mode = "workspace-write" if sandbox_policy_mode in {"fallback", "workspace-write"} else "danger-full-access"
    dangerously_bypass = sandbox_policy_mode == "permissive"
    append_sandbox_args(
        args,
        supported,
        sandbox_mode=sandbox_mode,
        dangerously_bypass=dangerously_bypass,
        include_full_auto=not dangerously_bypass,
    )
    if "--search" in supported:
        args.append("--search")
    args += [
        "--output-last-message",
        str(last_message_path),
        "-",
    ]
    stop_event = threading.Event()
    start_time = time.monotonic()
    print(f"{_RUNNER_LABEL}: sandbox mode {sandbox_policy_mode}", flush=True)
    print(f"{_RUNNER_LABEL} running... (0s total)", flush=True)
    heartbeat = threading.Thread(target=_heartbeat, args=(stop_event, start_time), daemon=True)
    heartbeat.start()
    try:
        result = run_command(args, input_text=prompt_text, timeout=timeout)
        sandbox_failure_excerpt = detect_sandbox_startup_failure(
            result.stdout,
            result.stderr,
            last_message_path.read_text(encoding="utf-8", errors="ignore") if last_message_path.exists() else "",
        )
        used_sandbox_fallback = False
        if sandbox_policy_mode == "fallback" and result.returncode != 0 and sandbox_failure_excerpt is not None:
            used_sandbox_fallback = True
            print(f"{_RUNNER_LABEL}: sandbox startup failed; retrying without sandbox", flush=True)
            retry_args = [
                STRATEGY_AGENT.cli_command,
                "exec",
                "-m",
                _DEFAULT_MODEL,
                "-c",
                f'model_reasoning_effort="{normalized_effort}"',
            ]
            append_sandbox_args(
                retry_args,
                supported,
                sandbox_mode="danger-full-access",
                dangerously_bypass=True,
                include_full_auto=False,
            )
            if "--search" in supported:
                retry_args.append("--search")
            retry_args += [
                "--output-last-message",
                str(last_message_path),
                "-",
            ]
            retry_result = run_command(retry_args, input_text=prompt_text, timeout=timeout)
            stdout_text = result.stdout + retry_result.stdout
            stderr_chunks = [chunk for chunk in (result.stderr, retry_result.stderr) if chunk.strip()]
            stderr_text = "\n\n".join(stderr_chunks)
            if sandbox_failure_excerpt:
                if stderr_text:
                    stderr_text = f"{sandbox_failure_excerpt}\n\n{stderr_text}"
                else:
                    stderr_text = sandbox_failure_excerpt
            result = CommandResult(
                args=retry_result.args,
                returncode=retry_result.returncode,
                stdout=stdout_text,
                stderr=stderr_text,
                duration_sec=retry_result.duration_sec,
            )
        else:
            used_sandbox_fallback = False
    except subprocess.TimeoutExpired:
        total_elapsed = int(time.monotonic() - start_time)
        stop_event.set()
        heartbeat.join(timeout=1.0)
        message = f"Strategy runner timed out after {int(timeout)}s (elapsed={total_elapsed}s)."
        transcript_path.write_text(message + "\n", encoding="utf-8")
        last_message_path.write_text(message + "\n", encoding="utf-8")
        return StrategyResult(
            transcript_path=transcript_path,
            last_message_path=last_message_path,
            returncode=124,
            stdout=message,
            stderr=message,
            sandbox_policy_mode=sandbox_policy_mode,
        )
    finally:
        stop_event.set()
        heartbeat.join(timeout=1.0)
    total_elapsed = int(time.monotonic() - start_time)
    print(f"{_RUNNER_LABEL} done... ({total_elapsed}s total, exit={result.returncode})", flush=True)
    transcript_path.write_text(result.stdout, encoding="utf-8")
    last_message = last_message_path.read_text(encoding="utf-8").strip() if last_message_path.exists() else ""
    if not last_message:
        last_message = result.stdout.strip()
        last_message_path.write_text((last_message + "\n") if last_message else "", encoding="utf-8")
    return StrategyResult(
        transcript_path=transcript_path,
        last_message_path=last_message_path,
        returncode=result.returncode,
        stdout=last_message,
        stderr=result.stderr,
        sandbox_policy_mode=sandbox_policy_mode,
        used_sandbox_fallback=used_sandbox_fallback,
        sandbox_failure_excerpt=sandbox_failure_excerpt if used_sandbox_fallback else None,
    )


def _normalize_reasoning_effort(effort: str) -> str:
    normalized = effort.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"extra_high", "xhgih"}:
        return "xhigh"
    return normalized


def _heartbeat(stop_event: threading.Event, start_time: float, interval: float = 30.0) -> None:
    while not stop_event.wait(interval):
        elapsed = int(time.monotonic() - start_time)
        print(f"{_RUNNER_LABEL} running... ({elapsed}s total)", flush=True)


@lru_cache(maxsize=1)
def _codex_help() -> str:
    try:
        result = run_command([STRATEGY_AGENT.cli_command, "exec", "--help"])
    except FileNotFoundError:
        return ""
    if result.returncode != 0:
        return ""
    return result.output


def _supported_flags() -> set[str]:
    text = _codex_help()
    flags: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        for token in line.split():
            if not token.startswith("-"):
                break
            flags.add(token.rstrip(","))
    return flags
