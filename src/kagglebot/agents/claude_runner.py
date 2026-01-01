from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.types import AgentResult


def run_claude(prompt_path: Path, output_dir: Path, *, dry_run: bool = False) -> AgentResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = prompt_path.read_text(encoding="utf-8")

    if dry_run:
        transcript = {
            "command": ["claude", "-p", "<prompt>"],
            "prompt_path": str(prompt_path),
            "dry_run": True,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "timestamp": datetime.now(UTC).isoformat(),
        }
        transcript_path = output_dir / "transcript.json"
        transcript_path.write_text(json.dumps(transcript, indent=2), encoding="utf-8")
        last_message_path = output_dir / "last_message.txt"
        last_message_path.write_text("DRY RUN: claude not executed.\n", encoding="utf-8")
        return AgentResult(
            transcript_path=str(transcript_path),
            last_message_path=str(last_message_path),
            returncode=0,
            stdout="",
            stderr="",
        )

    completed = subprocess.run(
        ["claude", "-p", prompt_text],
        text=True,
        capture_output=True,
        check=False,
    )

    transcript = {
        "command": ["claude", "-p", "<prompt>"],
        "prompt_path": str(prompt_path),
        "returncode": completed.returncode,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
        "timestamp": datetime.now(UTC).isoformat(),
    }
    transcript_path = output_dir / "transcript.json"
    transcript_path.write_text(json.dumps(transcript, indent=2), encoding="utf-8")

    last_message_path = output_dir / "last_message.txt"
    last_line = ""
    for line in (completed.stdout or "").splitlines():
        if line.strip():
            last_line = line
    last_message_path.write_text(last_line + "\n", encoding="utf-8")

    return AgentResult(
        transcript_path=str(transcript_path),
        last_message_path=str(last_message_path),
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )
