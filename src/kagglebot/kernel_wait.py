from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich import print

from kagglebot import kernel_logs as _kernel_logs
from kagglebot.exceptions import (
    KaggleCliError,
    KaggleNetworkError,
    KernelFailedError,
    KernelStillRunningError,
    KernelTimeoutError,
)
from kagglebot.kernel_status import (
    is_kernel_status_complete,
    is_kernel_status_failed,
    is_kernel_status_queued,
    is_kernel_status_running,
    parse_kernel_status,
)
from kagglebot.logging_utils import truncate_lines

LOG_POLL_INTERVAL = 2.0
HEARTBEAT_INTERVAL = 30.0
STATUS_ERROR_SLEEP = 10.0
MAX_STATUS_ERRORS = 6


@dataclass(frozen=True)
class KernelWaitDependencies:
    kernels_status: Callable[..., str]
    try_fetch_kernel_output: Callable[..., object]
    print_kernel_logs: Callable[[Path, _kernel_logs.KernelLogState], bool]
    detect_failure_in_logs: Callable[[Path], str | None]
    collect_log_tail: Callable[[Path], str]
    monotonic: Callable[[], float]
    sleep: Callable[[float], object]
    remote_kernel_queued_timeout_sec: Callable[[], float | None]
    raise_kernel_queued_timeout: Callable[[str, float, float], object]


@dataclass(frozen=True)
class KernelWaitLimits:
    log_poll_interval: float = LOG_POLL_INTERVAL
    heartbeat_interval: float = HEARTBEAT_INTERVAL
    status_error_sleep: float = STATUS_ERROR_SLEEP
    max_status_errors: int | None = MAX_STATUS_ERRORS


def raise_kernel_timeout(kernel_id: str, last_status: str | None) -> None:
    status = (last_status or "unknown").lower()
    if is_kernel_status_running(status):
        raise KernelStillRunningError(
            f"Kaggle kernel {kernel_id} is still {status} after the local wait budget; "
            "leaving the remote run active and refusing to push a duplicate version."
        )
    raise KernelTimeoutError(
        f"Kaggle kernel {kernel_id} did not complete within the local wait budget; last status was {status}."
    )


def wait_for_kernel(
    kernel_id: str,
    slug: str,
    timeout_minutes: int | None,
    *,
    output_dir: Path,
    initial_queued_since: float | None = None,
    deps: KernelWaitDependencies,
    limits: KernelWaitLimits | None = None,
) -> None:
    resolved_limits = limits or KernelWaitLimits()
    deadline = None
    if timeout_minutes is not None:
        deadline = deps.monotonic() + max(timeout_minutes, 1) * 60
    started_at = deps.monotonic()
    last_status = None
    last_log_fetch = 0.0
    log_state = _kernel_logs.KernelLogState()
    status_errors = 0
    queued_since = initial_queued_since
    queued_timeout_sec = deps.remote_kernel_queued_timeout_sec()
    while True:
        try:
            output = deps.kernels_status(kernel_id, slug=slug, dry_run=False)
            status_errors = 0
        except KaggleCliError as exc:
            status_errors += 1
            detail = (exc.output or str(exc)).strip()
            if detail:
                detail = detail.replace("\n", " ")
            if isinstance(exc, KaggleNetworkError):
                message = (
                    f"[yellow]kernel status network error[/yellow]: {detail or 'unknown error'} "
                    f"(attempt {status_errors})"
                )
                print(message)
                if deadline is not None and deps.monotonic() > deadline:
                    raise_kernel_timeout(kernel_id, last_status)
                if resolved_limits.max_status_errors is not None and status_errors >= resolved_limits.max_status_errors:
                    kernel_url = f"https://www.kaggle.com/code/{kernel_id}"
                    raise KaggleNetworkError(
                        "Kaggle API unreachable while polling kernel status. "
                        f"Check network/DNS and monitor the kernel at {kernel_url}.",
                        getattr(exc, "command", None),
                        exit_code=getattr(exc, "exit_code", None),
                        output=getattr(exc, "output", ""),
                    ) from exc
                deps.sleep(resolved_limits.status_error_sleep)
                continue
            message = f"[yellow]kernel status failed[/yellow]: {detail or 'unknown error'} (attempt {status_errors})"
            print(message)
            if deadline is not None and deps.monotonic() > deadline:
                raise_kernel_timeout(kernel_id, last_status)
            if resolved_limits.max_status_errors is not None and status_errors >= resolved_limits.max_status_errors:
                raise KernelFailedError(
                    f"Kaggle kernel status failed {status_errors} times. Last error: {detail or 'unknown error'}"
                ) from exc
            deps.sleep(resolved_limits.status_error_sleep)
            continue
        status = parse_kernel_status(output)
        if status != last_status:
            print(f"[cyan]kernel status[/cyan]: {status}")
            last_status = status
        now = deps.monotonic()
        if is_kernel_status_queued(status):
            if queued_since is None:
                queued_since = now
            if queued_timeout_sec is not None and now - queued_since >= queued_timeout_sec:
                deps.raise_kernel_queued_timeout(kernel_id, now - queued_since, queued_timeout_sec)
        else:
            queued_since = None
        if now - last_log_fetch >= resolved_limits.log_poll_interval:
            deps.try_fetch_kernel_output(kernel_id, output_dir=output_dir, slug=slug)
            had_logs = deps.print_kernel_logs(output_dir, log_state)
            if had_logs:
                log_state.last_log_at = now
            last_log_fetch = now
            log_failure = deps.detect_failure_in_logs(output_dir)
            if log_failure:
                log_failure = truncate_lines(log_failure, max_lines=5)
                message = f"Kaggle kernel error detected in logs.\n\n--- kernel log tail ---\n{log_failure}"
                raise KernelFailedError(message)
        if is_kernel_status_running(status):
            if log_state.last_heartbeat == 0.0 or now - log_state.last_heartbeat >= resolved_limits.heartbeat_interval:
                elapsed = max(0, int(now - started_at))
                timeout_hint = ""
                if deadline is not None:
                    timeout_hint = f", timeout in <= {max(0, int(deadline - now))}s"
                since = now - log_state.last_log_at if log_state.last_log_at is not None else None
                if since is None:
                    print(f"[cyan]kernel[/cyan]: still running (elapsed={elapsed}s{timeout_hint}, no logs yet)")
                else:
                    print(
                        f"[cyan]kernel[/cyan]: still running "
                        f"(elapsed={elapsed}s{timeout_hint}, no new logs for {since:.0f}s)"
                    )
                log_state.last_heartbeat = now
        if is_kernel_status_complete(status):
            return
        if is_kernel_status_failed(status):
            deps.try_fetch_kernel_output(kernel_id, output_dir=output_dir, slug=slug)
            log_tail = deps.collect_log_tail(output_dir)
            message = f"Kaggle kernel failed: {output}"
            if log_tail:
                log_tail = truncate_lines(log_tail, max_lines=5)
                message = f"{message}\n\n--- kernel log tail ---\n{log_tail}"
            raise KernelFailedError(message)
        deps.sleep(resolved_limits.status_error_sleep)
        if deadline is not None and deps.monotonic() > deadline:
            raise_kernel_timeout(kernel_id, last_status)
