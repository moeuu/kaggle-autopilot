from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

from kagglebot.exec_utils import run_command

_DEFAULT_MODEL = "gpt-5.2"
_DEFAULT_REASONING_EFFORT = "extra_high"


@dataclass(frozen=True)
class StrategyResult:
    transcript_path: Path
    last_message_path: Path
    returncode: int
    stdout: str
    stderr: str


def run_strategy(prompt_path: Path, output_dir: Path, *, dry_run: bool = False) -> StrategyResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = prompt_path.read_text(encoding="utf-8")
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
        )

    normalized_effort = _normalize_reasoning_effort(_DEFAULT_REASONING_EFFORT)
    args = [
        "codex",
        "exec",
        "-m",
        _DEFAULT_MODEL,
        "-c",
        f'model_reasoning_effort="{normalized_effort}"',
        "-a",
        "never",
        "--sandbox",
        "workspace-write",
        "--output-last-message",
        str(last_message_path),
        "-",
    ]
    stop_event = threading.Event()
    start_time = time.monotonic()
    print("gpt running... (0s total)", flush=True)
    heartbeat = threading.Thread(target=_heartbeat, args=(stop_event, start_time), daemon=True)
    heartbeat.start()
    try:
        result = run_command(args, input_text=prompt_text)
    finally:
        stop_event.set()
        heartbeat.join(timeout=1.0)
    total_elapsed = int(time.monotonic() - start_time)
    print(f"gpt done... ({total_elapsed}s total, exit={result.returncode})", flush=True)
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
    )


def _normalize_reasoning_effort(effort: str) -> str:
    normalized = effort.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized == "extra_high":
        return "high"
    return normalized


def _heartbeat(stop_event: threading.Event, start_time: float, interval: float = 30.0) -> None:
    while not stop_event.wait(interval):
        elapsed = int(time.monotonic() - start_time)
        print(f"gpt running... ({elapsed}s total)", flush=True)
