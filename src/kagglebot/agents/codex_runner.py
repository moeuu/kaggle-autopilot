from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from kagglebot.agents.identity import IMPLEMENTATION_AGENT, render_prompt_identity
from kagglebot.agents.sandbox_fallback import (
    append_sandbox_args,
    detect_sandbox_startup_failure,
    resolve_agent_sandbox_mode,
)
from kagglebot.exec_utils import run_command

_COMMAND_LOG_FIRST_WIDTH = 100
_COMMAND_LOG_SECOND_WIDTH = 100
_RUNNER_LABEL = IMPLEMENTATION_AGENT.log_alias


@dataclass(frozen=True)
class CodexResult:
    transcript_path: Path
    last_message_path: Path
    returncode: int
    stdout: str
    stderr: str
    sandbox_policy_mode: str = "permissive"
    used_sandbox_fallback: bool = False
    sandbox_failure_excerpt: str | None = None


def run_codex(
    prompt_path: Path,
    output_dir: Path,
    *,
    dry_run: bool = False,
    heartbeat_label: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> CodexResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = render_prompt_identity(prompt_path.read_text(encoding="utf-8"))
    transcript_path = output_dir / "codex_exec.jsonl"
    last_message_path = output_dir / "codex_last_message.txt"

    if dry_run:
        transcript_path.write_text("", encoding="utf-8")
        last_message_path.write_text(f"DRY RUN: {_RUNNER_LABEL} not executed.\n", encoding="utf-8")
        return CodexResult(
            transcript_path=transcript_path,
            last_message_path=last_message_path,
            returncode=0,
            stdout="",
            stderr="",
            sandbox_policy_mode=resolve_agent_sandbox_mode(),
        )

    args = [IMPLEMENTATION_AGENT.cli_command, "exec"]
    if model:
        args += ["-m", model]
    normalized_effort = _normalize_reasoning_effort(reasoning_effort)
    if normalized_effort:
        args += ["-c", f'model_reasoning_effort="{normalized_effort}"']
    supported = _supported_flags()
    sandbox_policy_mode = resolve_agent_sandbox_mode()
    sandbox_mode = "workspace-write" if sandbox_policy_mode in {"fallback", "workspace-write"} else "danger-full-access"
    dangerously_bypass = sandbox_policy_mode == "permissive"
    args += _build_shared_args(
        supported=supported,
        last_message_path=last_message_path,
        sandbox_mode=sandbox_mode,
        dangerously_bypass=dangerously_bypass,
    )
    print(f"{_RUNNER_LABEL}: sandbox mode {sandbox_policy_mode}", flush=True)
    stop_event = threading.Event()
    start_time = time.monotonic()
    label = heartbeat_label.strip() if heartbeat_label else None
    label_suffix = f" [{label}]" if label else ""
    print(f"{_RUNNER_LABEL} running... (0s total){label_suffix}", flush=True)
    heartbeat = threading.Thread(
        target=_heartbeat,
        args=(stop_event, start_time, label),
        daemon=True,
    )
    heartbeat.start()
    stdout_attempts: list[str] = []
    stderr_attempts: list[str] = []
    returncode = 0
    used_sandbox_fallback = False
    sandbox_failure_excerpt: str | None = None
    try:
        stdout_text, stderr_text, returncode = _run_codex_process(args=args, prompt_text=prompt_text)
        stdout_attempts.append(stdout_text)
        if stderr_text:
            stderr_attempts.append(stderr_text)
        sandbox_failure_excerpt = detect_sandbox_startup_failure(
            stdout_text,
            stderr_text,
            _read_last_message(last_message_path),
        )
        if sandbox_policy_mode == "fallback" and returncode != 0 and sandbox_failure_excerpt is not None:
            used_sandbox_fallback = True
            print(f"{_RUNNER_LABEL}: sandbox startup failed; retrying without sandbox", flush=True)
            retry_args = [IMPLEMENTATION_AGENT.cli_command, "exec"]
            if model:
                retry_args += ["-m", model]
            if normalized_effort:
                retry_args += ["-c", f'model_reasoning_effort="{normalized_effort}"']
            retry_args += _build_shared_args(
                supported=supported,
                last_message_path=last_message_path,
                sandbox_mode="danger-full-access",
                dangerously_bypass=True,
            )
            stdout_text, stderr_text, returncode = _run_codex_process(args=retry_args, prompt_text=prompt_text)
            stdout_attempts.append(stdout_text)
            if stderr_text:
                stderr_attempts.append(stderr_text)
    finally:
        stop_event.set()
        heartbeat.join(timeout=1.0)
    total_elapsed = int(time.monotonic() - start_time)
    print(f"{_RUNNER_LABEL} done... ({total_elapsed}s total, exit={returncode}){label_suffix}", flush=True)
    stdout_text = "".join(stdout_attempts)
    stderr_text = "\n\n".join(chunk for chunk in stderr_attempts if chunk.strip())
    if used_sandbox_fallback and sandbox_failure_excerpt:
        if stderr_text:
            stderr_text = f"{sandbox_failure_excerpt}\n\n{stderr_text}"
        else:
            stderr_text = sandbox_failure_excerpt
    transcript_path.write_text(stdout_text, encoding="utf-8")
    if not last_message_path.exists():
        last_message_path.write_text("", encoding="utf-8")
    return CodexResult(
        transcript_path=transcript_path,
        last_message_path=last_message_path,
        returncode=returncode,
        stdout=stdout_text,
        stderr=stderr_text,
        sandbox_policy_mode=sandbox_policy_mode,
        used_sandbox_fallback=used_sandbox_fallback,
        sandbox_failure_excerpt=sandbox_failure_excerpt,
    )


def _build_shared_args(
    *,
    supported: set[str],
    last_message_path: Path,
    sandbox_mode: str,
    dangerously_bypass: bool = False,
) -> list[str]:
    args: list[str] = []
    append_sandbox_args(
        args,
        supported,
        sandbox_mode=sandbox_mode,
        dangerously_bypass=dangerously_bypass,
        include_full_auto=not dangerously_bypass and sandbox_mode == "workspace-write",
    )
    args += [
        "--json",
        "--output-last-message",
        str(last_message_path),
        "-",
    ]
    return args


def _run_codex_process(*, args: list[str], prompt_text: str) -> tuple[str, str, int]:
    stdout_chunks: list[str] = []
    stderr_text = ""
    proc = subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.stdin is not None:
        proc.stdin.write(prompt_text)
        proc.stdin.close()
    if proc.stdout is not None:
        for line in proc.stdout:
            stdout_chunks.append(line)
            _emit_codex_event(line)
    if proc.stderr is not None:
        stderr_text = proc.stderr.read()
    return "".join(stdout_chunks), stderr_text, proc.wait()


def _read_last_message(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _normalize_reasoning_effort(effort: str | None) -> str | None:
    if effort is None:
        return None
    normalized = effort.strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return None
    if normalized in {"extra_high", "xhgih"}:
        return "xhigh"
    return normalized


def _heartbeat(
    stop_event: threading.Event,
    start_time: float,
    label: str | None,
    interval: float = 30.0,
) -> None:
    while not stop_event.wait(interval):
        elapsed = int(time.monotonic() - start_time)
        label_suffix = f" [{label}]" if label else ""
        print(f"{_RUNNER_LABEL} running... ({elapsed}s total){label_suffix}", flush=True)


def _emit_codex_event(line: str) -> None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return
    item = payload.get("item") if isinstance(payload, dict) else None
    if not isinstance(item, dict):
        return
    item_type = item.get("type")
    if item_type == "reasoning":
        return
    if item_type == "command_execution":
        command = item.get("command", "").strip()
        status = item.get("status", "")
        exit_code = item.get("exit_code")
        first_line, second_line = _format_command_for_log(command)
        if status == "completed":
            suffix = f" (exit {exit_code})" if exit_code is not None else ""
            if second_line:
                print(f"{_RUNNER_LABEL}: command completed: {first_line}")
                print(f"  {second_line}{suffix}")
            else:
                print(f"{_RUNNER_LABEL}: command completed: {first_line}{suffix}")
        elif status:
            if second_line:
                print(f"{_RUNNER_LABEL}: command {status}: {first_line}")
                print(f"  {second_line}")
            else:
                print(f"{_RUNNER_LABEL}: command {status}: {first_line}")
        return
    if item_type == "file_change":
        changes = item.get("changes", [])
        if isinstance(changes, list):
            for change in changes:
                if not isinstance(change, dict):
                    continue
                path = change.get("path")
                kind = change.get("kind")
                if path and kind:
                    print(f"{_RUNNER_LABEL}: {kind} {path}")
        return


@lru_cache(maxsize=1)
def _codex_help() -> str:
    try:
        result = run_command([IMPLEMENTATION_AGENT.cli_command, "exec", "--help"])
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


def _format_command_for_log(command: str) -> tuple[str, str]:
    normalized = " ".join(command.split())
    if not normalized:
        return "", ""
    first_line, remainder = _split_by_width(normalized, _COMMAND_LOG_FIRST_WIDTH)
    if not remainder:
        return first_line, ""
    second_line, tail = _split_by_width(remainder, _COMMAND_LOG_SECOND_WIDTH)
    if tail:
        second_line = second_line.rstrip()
        if len(second_line) >= _COMMAND_LOG_SECOND_WIDTH:
            second_line = second_line[: _COMMAND_LOG_SECOND_WIDTH - 3].rstrip()
        second_line = f"{second_line}..."
    return first_line, second_line


def _split_by_width(text: str, width: int) -> tuple[str, str]:
    if len(text) <= width:
        return text, ""
    split_at = text.rfind(" ", 0, width + 1)
    if split_at <= 0:
        split_at = width
    head = text[:split_at].strip()
    tail = text[split_at:].strip()
    return head, tail
