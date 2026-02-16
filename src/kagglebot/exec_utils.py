from __future__ import annotations

import logging
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommandResult:
    """
    Result of an external command execution.
    """

    args: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_sec: float

    @property
    def output(self) -> str:
        return (self.stdout + self.stderr).strip()


def run_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
    timeout: float | None = None,
    dry_run: bool = False,
    stream_output: bool = False,
    line_callback: Callable[[str], None] | None = None,
) -> CommandResult:
    """
    Execute an external command safely (no shell).

    Args:
        args: Command arguments (list form).
        cwd: Optional working directory.
        env: Optional environment overrides.
        input_text: Optional stdin text.
        timeout: Optional timeout in seconds.
        dry_run: If True, do not execute; return a zero-code result.
        stream_output: If True, stream stdout while capturing output.

    Returns:
        CommandResult with stdout/stderr and return code.

    Raises:
        FileNotFoundError: If the executable is missing.
        subprocess.TimeoutExpired: If the command times out.
    """
    if dry_run:
        logger.info("DRY RUN: %s", " ".join(args))
        return CommandResult(args=args, returncode=0, stdout="", stderr="", duration_sec=0.0)

    start = time.monotonic()
    if stream_output:
        proc = subprocess.Popen(
            args,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdin=subprocess.PIPE if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        stdout_chunks: list[str] = []
        if input_text is not None and proc.stdin is not None:
            proc.stdin.write(input_text)
            proc.stdin.close()
        if proc.stdout is not None:
            for line in proc.stdout:
                stdout_chunks.append(line)
                if hasattr(sys.stdout, "write"):
                    sys.stdout.write(line)
                    sys.stdout.flush()
                if line_callback is not None:
                    try:
                        line_callback(line)
                    except Exception:  # noqa: BLE001
                        logger.debug("line callback failed", exc_info=True)
        returncode = proc.wait(timeout=timeout)
        stdout = "".join(stdout_chunks)
        duration = time.monotonic() - start
        return CommandResult(args=args, returncode=returncode, stdout=stdout, stderr="", duration_sec=duration)

    completed = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        env=env,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    duration = time.monotonic() - start
    return CommandResult(
        args=args,
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        duration_sec=duration,
    )
