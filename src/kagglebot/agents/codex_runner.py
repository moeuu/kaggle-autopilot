from __future__ import annotations

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
