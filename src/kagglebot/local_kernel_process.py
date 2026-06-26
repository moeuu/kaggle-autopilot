from __future__ import annotations

import codecs
import os
import selectors
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kagglebot import local_kernel_limits as _local_kernel_limits
from kagglebot import local_kernel_progress as _local_kernel_progress
from kagglebot.exec_utils import CommandResult

MEMORY_POLL_INTERVAL_SEC = 1.0
STDOUT_POLL_INTERVAL_SEC = 0.2
EXIT_PIPE_DRAIN_SEC = 1.0


@dataclass(frozen=True)
class LocalKernelExecResult:
    command_result: CommandResult
    peak_rss_bytes: int
    memory_cap_bytes: int | None
    killed_for_memory: bool = False
    memory_kill_message: str | None = None
    killed_for_stall: bool = False
    stall_kill_message: str | None = None


@dataclass
class LocalKernelLogFilterState:
    suppress_next_fragment_source_line: bool = False


_SUPPRESSED_LOCAL_KERNEL_LOG_MARKERS = (
    "PerformanceWarning: DataFrame is highly fragmented.",
    "This is usually the result of calling `frame.insert` many times, which has poor performance.",
    "Consider joining all columns at once using pd.concat(axis=1) instead.",
    "To get a de-fragmented frame, use `newframe = frame.copy()`",
    "-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html",
    "Default metric period is 5 because BrierScore is/are not implemented for GPU",
)


def should_suppress_local_kernel_log_line(line: str, *, state: LocalKernelLogFilterState) -> bool:
    if "PerformanceWarning: DataFrame is highly fragmented." in line:
        state.suppress_next_fragment_source_line = True
        return True

    if state.suppress_next_fragment_source_line:
        stripped = line.strip()
        if stripped:
            state.suppress_next_fragment_source_line = False
            return True

    return any(marker in line for marker in _SUPPRESSED_LOCAL_KERNEL_LOG_MARKERS)


def terminate_local_kernel_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return

    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except (OSError, ProcessLookupError):
        return

    deadline = time.monotonic() + 2.0
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


def terminate_local_kernel_process_group(proc: subprocess.Popen[str]) -> None:
    if os.name == "nt":
        terminate_local_kernel_process(proc)
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        try:
            os.killpg(proc.pid, 0)
        except (OSError, ProcessLookupError):
            return
        time.sleep(0.05)

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        return


def run_local_kernel_once(
    *,
    kernel_path: Path,
    kernel_stage_dir: Path,
    current_env: dict[str, str],
    timeout_sec: int | None,
    line_callback: Callable[[str], None] | None,
    progress_tracker: _local_kernel_progress.LocalKernelProgressTracker | None,
) -> LocalKernelExecResult:
    args = [sys.executable, "-u", str(kernel_path)]
    start = time.monotonic()
    proc = subprocess.Popen(
        args,
        cwd=str(kernel_stage_dir),
        env=current_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=os.name != "nt",
    )

    memory_cap_bytes = _local_kernel_limits.resolve_memory_cap_bytes(current_env)
    memory_state = {
        "peak_rss_bytes": 0,
        "killed_for_memory": False,
        "memory_kill_message": None,
    }
    stall_state = {
        "killed_for_stall": False,
        "stall_kill_message": None,
    }
    stall_timeout_sec = _local_kernel_limits.resolve_stall_timeout_sec(current_env)
    memory_stop = threading.Event()

    def _watch_memory() -> None:
        while not memory_stop.wait(MEMORY_POLL_INTERVAL_SEC):
            if proc.poll() is not None:
                break
            rss_bytes = _local_kernel_limits.process_tree_rss_bytes(proc.pid)
            memory_state["peak_rss_bytes"] = max(memory_state["peak_rss_bytes"], rss_bytes)
            if memory_cap_bytes is None or rss_bytes <= memory_cap_bytes:
                continue
            memory_state["killed_for_memory"] = True
            memory_state["memory_kill_message"] = (
                "Local kernel exceeded host memory guard "
                f"({rss_bytes // (1024 * 1024)} MiB RSS > {memory_cap_bytes // (1024 * 1024)} MiB cap)."
            )
            terminate_local_kernel_process_group(proc)
            break

    memory_thread = threading.Thread(target=_watch_memory, daemon=True, name="kb-local-kernel-memory-watchdog")
    memory_thread.start()
    stall_stop = threading.Event()

    def _watch_stall() -> None:
        if progress_tracker is None or stall_timeout_sec is None:
            return
        poll_interval = min(30.0, max(1.0, stall_timeout_sec / 10.0))
        while not stall_stop.wait(poll_interval):
            if proc.poll() is not None:
                break
            if bool(stall_state["killed_for_stall"]):
                break
            stall_message = _local_kernel_progress.detect_local_kernel_stall(
                progress_tracker=progress_tracker,
                stall_timeout_sec=stall_timeout_sec,
            )
            if stall_message is None:
                continue
            stall_state["killed_for_stall"] = True
            stall_state["stall_kill_message"] = stall_message
            terminate_local_kernel_process_group(proc)
            break

    stall_thread = threading.Thread(target=_watch_stall, daemon=True, name="kb-local-kernel-stall-watchdog")
    stall_thread.start()

    stdout_chunks: list[str] = []
    proc_stdout = proc.stdout
    try:
        if proc_stdout is not None:
            stdout_fd = proc_stdout.fileno()
            os.set_blocking(stdout_fd, False)
            selector = selectors.DefaultSelector()
            selector.register(stdout_fd, selectors.EVENT_READ)
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            pending = ""
            log_filter_state = LocalKernelLogFilterState()
            deadline = None if timeout_sec is None else start + timeout_sec
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
                    if should_suppress_local_kernel_log_line(line, state=log_filter_state):
                        continue
                    stdout_chunks.append(line)
                    if hasattr(sys.stdout, "write"):
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    if line_callback is not None:
                        try:
                            line_callback(line)
                        except Exception:
                            pass

            while True:
                now = time.monotonic()
                if deadline is not None and now >= deadline:
                    terminate_local_kernel_process_group(proc)
                    raise subprocess.TimeoutExpired(args, timeout_sec)

                wait_timeout = STDOUT_POLL_INTERVAL_SEC
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
                            selector.unregister(stdout_fd)
                            selector.close()
                            proc_stdout.close()
                            proc_stdout = None
                            remaining_timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
                            returncode = proc.wait(timeout=remaining_timeout)
                            break
                        saw_data = True
                        last_data_at = time.monotonic()
                        if progress_tracker is not None:
                            progress_tracker.observe_output_activity()
                        _emit_text(decoder.decode(chunk), final=False)
                    if proc_stdout is None:
                        break

                if proc.poll() is not None:
                    if process_exited_at is None:
                        process_exited_at = time.monotonic()
                    if saw_data:
                        process_exited_at = time.monotonic()
                        continue
                    if time.monotonic() - max(process_exited_at, last_data_at) >= EXIT_PIPE_DRAIN_SEC:
                        terminate_local_kernel_process_group(proc)
                        _emit_text(decoder.decode(b"", final=True), final=True)
                        selector.unregister(stdout_fd)
                        selector.close()
                        proc_stdout.close()
                        proc_stdout = None
                        returncode = proc.wait(timeout=1.0)
                        break
                else:
                    process_exited_at = None
                    if (
                        progress_tracker is not None
                        and stall_timeout_sec is not None
                        and not bool(stall_state["killed_for_stall"])
                    ):
                        stall_message = _local_kernel_progress.detect_local_kernel_stall(
                            progress_tracker=progress_tracker,
                            stall_timeout_sec=stall_timeout_sec,
                        )
                        if stall_message is not None:
                            stall_state["killed_for_stall"] = True
                            stall_state["stall_kill_message"] = stall_message
                            terminate_local_kernel_process_group(proc)
                            process_exited_at = time.monotonic()
        else:
            returncode = proc.wait(timeout=timeout_sec)
    finally:
        if proc_stdout is not None:
            try:
                proc_stdout.close()
            except OSError:
                pass
        stall_stop.set()
        stall_thread.join(timeout=1.0)
        memory_stop.set()
        memory_thread.join(timeout=1.0)
        memory_state["peak_rss_bytes"] = max(
            memory_state["peak_rss_bytes"], _local_kernel_limits.process_tree_rss_bytes(proc.pid)
        )

    duration = time.monotonic() - start
    return LocalKernelExecResult(
        command_result=CommandResult(
            args=args,
            returncode=returncode,
            stdout="".join(stdout_chunks),
            stderr="",
            duration_sec=duration,
        ),
        peak_rss_bytes=int(memory_state["peak_rss_bytes"]),
        memory_cap_bytes=memory_cap_bytes,
        killed_for_memory=bool(memory_state["killed_for_memory"]),
        memory_kill_message=(
            str(memory_state["memory_kill_message"]) if memory_state["memory_kill_message"] is not None else None
        ),
        killed_for_stall=bool(stall_state["killed_for_stall"]),
        stall_kill_message=(str(stall_state["stall_kill_message"]) if stall_state["stall_kill_message"] else None),
    )
