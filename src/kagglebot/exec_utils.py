from __future__ import annotations

import codecs
import logging
import os
import selectors
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)
_STREAM_OUTPUT_POLL_INTERVAL_SEC = 0.2
_STREAM_OUTPUT_EXIT_PIPE_DRAIN_SEC = 1.0


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


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except (OSError, ProcessLookupError):
        return

    deadline = time.monotonic() + 1.0
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)

    if proc.poll() is not None:
        return

    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (OSError, ProcessLookupError):
        return


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
            start_new_session=os.name != "nt",
        )
        stdout_chunks: list[str] = []
        if input_text is not None and proc.stdin is not None:
            proc.stdin.write(input_text)
            proc.stdin.close()
        proc_stdout = proc.stdout
        if proc_stdout is not None:
            stdout_fd = proc_stdout.fileno()
            os.set_blocking(stdout_fd, False)
            selector = selectors.DefaultSelector()
            selector.register(stdout_fd, selectors.EVENT_READ)
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            pending = ""
            deadline = None if timeout is None else start + timeout
            last_data_at = start
            process_exited_at: float | None = None

            def _emit_text(text: str, *, final: bool) -> None:
                nonlocal pending
                if not text and not final:
                    return
                pending += text
                if not final:
                    lines = pending.splitlines(keepends=True)
                    if lines and not lines[-1].endswith(("\n", "\r")):
                        pending = lines.pop()
                    else:
                        pending = ""
                else:
                    lines = pending.splitlines(keepends=True) if pending else []
                    pending = ""
                for line in lines:
                    stdout_chunks.append(line)
                    if hasattr(sys.stdout, "write"):
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    if line_callback is not None:
                        try:
                            line_callback(line)
                        except Exception:  # noqa: BLE001
                            logger.debug("line callback failed", exc_info=True)

            try:
                while True:
                    now = time.monotonic()
                    if deadline is not None and now >= deadline:
                        _terminate_process(proc)
                        raise subprocess.TimeoutExpired(args, timeout)

                    wait_timeout = _STREAM_OUTPUT_POLL_INTERVAL_SEC
                    if deadline is not None:
                        wait_timeout = min(wait_timeout, max(0.0, deadline - now))
                    events = selector.select(timeout=wait_timeout)
                    saw_data = False
                    if events:
                        while True:
                            try:
                                chunk = os.read(stdout_fd, 65536)
                            except BlockingIOError:
                                break
                            if not chunk:
                                _emit_text(decoder.decode(b"", final=True), final=True)
                                proc_stdout.close()
                                proc_stdout = None
                                returncode = proc.wait(timeout=0)
                                break
                            saw_data = True
                            last_data_at = time.monotonic()
                            _emit_text(decoder.decode(chunk), final=False)
                        if proc_stdout is None:
                            break

                    if proc.poll() is not None:
                        if process_exited_at is None:
                            process_exited_at = time.monotonic()
                        if saw_data:
                            process_exited_at = time.monotonic()
                            continue
                        if (
                            time.monotonic() - max(process_exited_at, last_data_at)
                            >= _STREAM_OUTPUT_EXIT_PIPE_DRAIN_SEC
                        ):
                            _terminate_process(proc)
                            _emit_text(decoder.decode(b"", final=True), final=True)
                            proc_stdout.close()
                            proc_stdout = None
                            returncode = proc.wait(timeout=0)
                            break
                    else:
                        process_exited_at = None
            finally:
                selector.close()
                if proc_stdout is not None:
                    proc_stdout.close()
        else:
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
