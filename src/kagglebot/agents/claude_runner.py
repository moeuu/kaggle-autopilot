from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

from kagglebot.exec_utils import run_command

_DEFAULT_MODEL = "opus"


@dataclass(frozen=True)
class ClaudeResult:
    transcript_path: Path
    last_message_path: Path
    returncode: int
    stdout: str
    stderr: str


def run_claude(prompt_path: Path, output_dir: Path, *, dry_run: bool = False) -> ClaudeResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = prompt_path.read_text(encoding="utf-8")
    transcript_path = output_dir / "claude_exec.txt"
    last_message_path = output_dir / "claude_last_message.txt"

    if dry_run:
        transcript_path.write_text("", encoding="utf-8")
        last_message_path.write_text("DRY RUN: claude not executed.\n", encoding="utf-8")
        return ClaudeResult(
            transcript_path=transcript_path,
            last_message_path=last_message_path,
            returncode=0,
            stdout="",
            stderr="",
        )

    args = ["claude", "-p", "--model", _DEFAULT_MODEL, prompt_text]
    stop_event = threading.Event()
    start_time = time.monotonic()
    heartbeat = threading.Thread(target=_heartbeat, args=(stop_event, start_time), daemon=True)
    heartbeat.start()
    try:
        result = run_command(args)
    finally:
        stop_event.set()
        heartbeat.join(timeout=1.0)
    transcript_path.write_text(result.stdout, encoding="utf-8")
    last_line = ""
    for line in result.stdout.splitlines():
        if line.strip():
            last_line = line
    last_message_path.write_text(last_line + "\n", encoding="utf-8")
    return ClaudeResult(
        transcript_path=transcript_path,
        last_message_path=last_message_path,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def _heartbeat(stop_event: threading.Event, start_time: float, interval: float = 30.0) -> None:
    while not stop_event.wait(interval):
        elapsed = int(time.monotonic() - start_time)
        print(f"claude: still running... ({elapsed}s)")
