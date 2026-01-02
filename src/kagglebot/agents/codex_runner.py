from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from kagglebot.exec_utils import run_command


@dataclass(frozen=True)
class CodexResult:
    transcript_path: Path
    last_message_path: Path
    returncode: int
    stdout: str
    stderr: str


def run_codex(prompt_path: Path, output_dir: Path, *, dry_run: bool = False) -> CodexResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = prompt_path.read_text(encoding="utf-8")
    transcript_path = output_dir / "codex_exec.jsonl"
    last_message_path = output_dir / "codex_last_message.txt"

    if dry_run:
        transcript_path.write_text("", encoding="utf-8")
        last_message_path.write_text("DRY RUN: codex not executed.\n", encoding="utf-8")
        return CodexResult(
            transcript_path=transcript_path,
            last_message_path=last_message_path,
            returncode=0,
            stdout="",
            stderr="",
        )

    args = ["codex", "exec"]
    supported = _supported_flags()
    if "--full-auto" in supported:
        args.append("--full-auto")
    elif "-a" in supported:
        args += ["-a", "never"]
    if "--sandbox" in supported:
        args += ["--sandbox", "workspace-write"]
    if "--search" in supported:
        args.append("--search")
    args += [
        "--json",
        "--output-last-message",
        str(last_message_path),
        "-",
    ]
    if "--search" in args:
        print("codex: search enabled")
    stop_event = threading.Event()
    heartbeat = threading.Thread(target=_heartbeat, args=(stop_event,), daemon=True)
    heartbeat.start()
    stdout_chunks: list[str] = []
    stderr_text = ""
    returncode = 0
    try:
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
        returncode = proc.wait()
    finally:
        stop_event.set()
        heartbeat.join(timeout=1.0)
    stdout_text = "".join(stdout_chunks)
    transcript_path.write_text(stdout_text, encoding="utf-8")
    if not last_message_path.exists():
        last_message_path.write_text("", encoding="utf-8")
    return CodexResult(
        transcript_path=transcript_path,
        last_message_path=last_message_path,
        returncode=returncode,
        stdout=stdout_text,
        stderr=stderr_text,
    )


def _heartbeat(stop_event: threading.Event, interval: float = 30.0) -> None:
    while not stop_event.wait(interval):
        print("codex: still running...")


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
        output = item.get("aggregated_output") or item.get("stdout") or ""
        if status == "completed":
            suffix = f" (exit {exit_code})" if exit_code is not None else ""
            print(f"codex: command completed: {command}{suffix}")
            if output:
                print(output.rstrip())
        elif status:
            print(f"codex: command {status}: {command}")
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
                    print(f"codex: {kind} {path}")
        return


@lru_cache(maxsize=1)
def _codex_help() -> str:
    try:
        result = run_command(["codex", "exec", "--help"])
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
