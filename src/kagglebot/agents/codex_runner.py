from __future__ import annotations

from dataclasses import dataclass
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

    args = [
        "codex",
        "exec",
        "-a",
        "never",
        "--sandbox",
        "workspace-write",
        "--json",
        "--output-last-message",
        str(last_message_path),
        "-",
    ]
    result = run_command(args, input_text=prompt_text)
    transcript_path.write_text(result.stdout, encoding="utf-8")
    if not last_message_path.exists():
        last_message_path.write_text("", encoding="utf-8")
    return CodexResult(
        transcript_path=transcript_path,
        last_message_path=last_message_path,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
