from __future__ import annotations

import codecs
import os
import re
import selectors
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich import print

from kagglebot import kernel_bootstrap as _kernel_bootstrap
from kagglebot import kernel_contracts as _kernel_contracts
from kagglebot import kernel_logs as _kernel_logs
from kagglebot import kernel_metadata as _kernel_metadata
from kagglebot import kernel_module_inliner as _kernel_module_inliner
from kagglebot import kernel_package_files as _kernel_package_files
from kagglebot import kernel_plan_validation as _kernel_plan_validation
from kagglebot import local_kernel_aux_inputs as _local_kernel_aux_inputs
from kagglebot import local_kernel_context as _local_kernel_context
from kagglebot import local_kernel_data_resolver as _local_kernel_data_resolver
from kagglebot import local_kernel_drift_guard as _local_kernel_drift_guard
from kagglebot import local_kernel_duration as _local_kernel_duration
from kagglebot import local_kernel_limits as _local_kernel_limits
from kagglebot import local_kernel_metrics_normalization as _local_kernel_metrics_normalization
from kagglebot import local_kernel_models as _local_kernel_models
from kagglebot import local_kernel_pipeline_cfg as _local_kernel_pipeline_cfg
from kagglebot import local_kernel_progress as _local_kernel_progress
from kagglebot import local_kernel_runtime_env as _local_kernel_runtime_env
from kagglebot import local_kernel_shims as _local_kernel_shims
from kagglebot import local_sample_submission as _local_sample_submission
from kagglebot import remote_kernel_state as _remote_kernel_state
from kagglebot.artifact_io import copy_artifact_if_needed as _copy_artifact_if_needed
from kagglebot.compute import detect_local_gpu
from kagglebot.exceptions import (
    KaggleCliError,
    KaggleNetworkError,
    KernelFailedError,
    KernelStillRunningError,
    KernelTimeoutError,
    RulesNotAcceptedError,
)
from kagglebot.exec_utils import CommandResult
from kagglebot.hardware import hardware_env, resolve_hardware_profile
from kagglebot.json_utils import load_json_object
from kagglebot.kaggle_api import (
    check_rules_accepted,
    kernel_exists,
    kernel_id_by_title,
    kernels_init,
    kernels_output,
    kernels_push,
    kernels_status,
)
from kagglebot.kernel_outputs import find_output_file as _find_output_file
from kagglebot.kernel_outputs import find_submission_file
from kagglebot.kernel_outputs import resolve_local_kernel_artifact_file as _resolve_local_kernel_artifact_file
from kagglebot.kernel_outputs import resolve_local_kernel_artifacts as _resolve_local_kernel_artifacts
from kagglebot.kernel_push_validation import (
    raise_for_invalid_kernel_push_sources as _raise_for_invalid_kernel_push_sources,
)
from kagglebot.kernel_sources import load_kernel_source_config
from kagglebot.kernel_status import (
    is_kernel_status_complete,
    is_kernel_status_failed,
    is_kernel_status_queued,
    is_kernel_status_running,
    parse_kernel_status,
)
from kagglebot.kernel_submit_inference import (
    sanitize_submit_inference_output_roots as _sanitize_submit_inference_output_roots,
)
from kagglebot.kernel_submit_inference import validate_inference_submit_kernel as _validate_inference_submit_kernel
from kagglebot.kernel_submit_wrapper import (
    reject_static_tiny_code_competition_submission as _reject_static_tiny_code_competition_submission,
)
from kagglebot.kernel_submit_wrapper import render_submission_kernel_script as _render_submission_kernel_script
from kagglebot.logging_utils import truncate_lines
from kagglebot.solution_guard import ensure_solution_path_allowed
from kagglebot.validators import ensure_kernel_sources_valid, validate_kernel_package

_LOCAL_KERNEL_HEARTBEAT_INTERVAL_SEC = 30.0
_LOCAL_KERNEL_MEMORY_POLL_INTERVAL_SEC = 1.0
_LOCAL_KERNEL_STDOUT_POLL_INTERVAL_SEC = 0.2
_LOCAL_KERNEL_EXIT_PIPE_DRAIN_SEC = 1.0
_SUBMIT_KERNEL_ACCELERATOR_ENV = "KAGGLEBOT_SUBMIT_KERNEL_ACCELERATOR"
_BASELINE_SCORE_ASSIGNMENT_RE = re.compile(
    r"\b(?P<name>[a-z_][a-z0-9_]*?(?:score|auc|rmse|mae|mse|f1|loss|accuracy|acc|precision|recall|map|ndcg|logloss|brier|gini))\s*=\s*"
    r"(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _LocalKernelExecResult:
    command_result: CommandResult
    peak_rss_bytes: int
    memory_cap_bytes: int | None
    killed_for_memory: bool = False
    memory_kill_message: str | None = None
    killed_for_stall: bool = False
    stall_kill_message: str | None = None


@dataclass
class _LocalKernelLogFilterState:
    suppress_next_fragment_source_line: bool = False


_SUPPRESSED_LOCAL_KERNEL_LOG_MARKERS = (
    "PerformanceWarning: DataFrame is highly fragmented.",
    "This is usually the result of calling `frame.insert` many times, which has poor performance.",
    "Consider joining all columns at once using pd.concat(axis=1) instead.",
    "To get a de-fragmented frame, use `newframe = frame.copy()`",
    "-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html",
    "Default metric period is 5 because BrierScore is/are not implemented for GPU",
)


def _should_suppress_local_kernel_log_line(line: str, *, state: _LocalKernelLogFilterState) -> bool:
    if "PerformanceWarning: DataFrame is highly fragmented." in line:
        state.suppress_next_fragment_source_line = True
        return True

    if state.suppress_next_fragment_source_line:
        stripped = line.strip()
        if stripped:
            state.suppress_next_fragment_source_line = False
            return True

    return any(marker in line for marker in _SUPPRESSED_LOCAL_KERNEL_LOG_MARKERS)


def _terminate_local_kernel_process(proc: subprocess.Popen[str]) -> None:
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


def _terminate_local_kernel_process_group(proc: subprocess.Popen[str]) -> None:
    if os.name == "nt":
        _terminate_local_kernel_process(proc)
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


def _run_local_kernel_once(
    *,
    kernel_path: Path,
    kernel_stage_dir: Path,
    current_env: dict[str, str],
    timeout_sec: int | None,
    line_callback: Callable[[str], None] | None,
    progress_tracker: _local_kernel_progress.LocalKernelProgressTracker | None,
) -> _LocalKernelExecResult:
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
        while not memory_stop.wait(_LOCAL_KERNEL_MEMORY_POLL_INTERVAL_SEC):
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
            _terminate_local_kernel_process_group(proc)
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
            _terminate_local_kernel_process_group(proc)
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
            log_filter_state = _LocalKernelLogFilterState()
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
                    if _should_suppress_local_kernel_log_line(line, state=log_filter_state):
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
                    _terminate_local_kernel_process_group(proc)
                    raise subprocess.TimeoutExpired(args, timeout_sec)

                wait_timeout = _LOCAL_KERNEL_STDOUT_POLL_INTERVAL_SEC
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
                    if time.monotonic() - max(process_exited_at, last_data_at) >= _LOCAL_KERNEL_EXIT_PIPE_DRAIN_SEC:
                        _terminate_local_kernel_process_group(proc)
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
                            _terminate_local_kernel_process_group(proc)
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
    return _LocalKernelExecResult(
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


@dataclass(frozen=True)
class KernelRunResult:
    kernel_id: str
    output_dir: Path
    submission_path: Path | None
    metrics_path: Path | None


@dataclass(frozen=True)
class KernelPreparation:
    kernel_dir: Path
    output_dir: Path
    logs_dir: Path
    kernel_slug: str
    kernel_id: str
    runtime_bootstrap_mode: str = "force_train"
    supersede_stale_queued: bool = False


@dataclass(frozen=True)
class KernelBuildConfig:
    slug: str
    run_id: str
    iteration: int
    base_dir: Path
    kaggle_username: str
    kernel_name: str | None
    accelerator: str
    enable_internet: bool
    score_source: str
    metric: str
    direction: str
    holdout_frac: float
    cv_folds: int
    seed: int
    dry_run: bool
    hardware_profile: str | None = "auto"


@dataclass(frozen=True)
class KernelSubmitBuildConfig:
    slug: str
    run_id: str
    iteration: int
    base_dir: Path
    kaggle_username: str
    kernel_name: str | None
    accelerator: str
    enable_internet: bool
    submission_path: Path
    mode: str
    dry_run: bool
    hardware_profile: str | None = "auto"


def _resolve_submit_kernel_accelerator(requested: str) -> str:
    override = os.getenv(_SUBMIT_KERNEL_ACCELERATOR_ENV)
    value = str(override if override is not None else "cpu").strip().lower()
    if value in {"none", "no", "false", "0"}:
        return "cpu"
    if value in {"cpu", "gpu", "tpu"}:
        return value
    if override is not None:
        raise ValueError(f"{_SUBMIT_KERNEL_ACCELERATOR_ENV} must be one of cpu, gpu, or tpu; got {override!r}.")
    requested_value = str(requested or "cpu").strip().lower()
    return requested_value if requested_value in {"cpu", "gpu", "tpu"} else "cpu"


@dataclass(frozen=True)
class KernelPackageBuilder:
    def prepare(self, config: KernelBuildConfig) -> KernelPreparation:
        kernel_dir = config.base_dir / config.slug / "kernels" / config.run_id
        output_dir = config.base_dir / config.slug / "runs" / config.run_id / f"iter-{config.iteration}" / "output"
        logs_dir = config.base_dir / config.slug / "runs" / config.run_id / f"iter-{config.iteration}" / "logs"
        context_dir = config.base_dir / config.slug / "context"
        kernel_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        if not config.dry_run and not check_rules_accepted(config.slug, dry_run=False):
            raise RulesNotAcceptedError("Competition rules not accepted.")

        if not config.dry_run:
            print(f"[cyan]kernel init[/cyan]: {kernel_dir}")
            kernels_init(kernel_dir, dry_run=False)

        kernel_slug = _kernel_metadata.resolve_kernel_slug(
            config.kernel_name,
            config.slug,
            config.run_id,
            config.iteration,
        )
        kernel_id = f"{config.kaggle_username}/{kernel_slug}"
        custom_kernel_dir = config.base_dir / config.slug / "kernel"
        custom_kernel_path = custom_kernel_dir / "kernel.py"
        source_config = load_kernel_source_config(config.base_dir / config.slug / "plan.json")
        ensure_solution_path_allowed(custom_kernel_dir, artifacts_dir=config.base_dir, slug=config.slug)
        if not custom_kernel_path.exists():
            raise KernelFailedError(
                "Authoritative kernel entrypoint is missing. "
                f"Expected: {custom_kernel_path}. "
                "Generate/update artifacts/<slug>/kernel/kernel.py before running training."
            )
        _kernel_package_files.copy_kernel_sources(custom_kernel_dir, kernel_dir)
        _kernel_package_files.copy_shared_kernel_runtime_modules(kernel_dir)
        _kernel_package_files.copy_competition_external_assets(
            base_dir=config.base_dir,
            slug=config.slug,
            kernel_dir=kernel_dir,
        )
        _kernel_package_files.sync_plan_snapshot(
            plan_path=config.base_dir / config.slug / "plan.json",
            targets=[kernel_dir / "plan.json"],
        )
        _kernel_bootstrap.ensure_kernel_import_path(kernel_dir)
        _kernel_bootstrap.inject_competition_slug_env(kernel_dir, config.slug)
        _kernel_bootstrap.inject_hardware_profile_env(
            kernel_dir,
            config.hardware_profile,
            compute="kaggle_gpu" if config.accelerator == "gpu" else "kaggle_tpu",
        )
        _kernel_module_inliner.inline_kernel_modules(kernel_dir)
        _local_kernel_data_resolver.inject_data_dir_resolver(kernel_dir)
        _local_kernel_pipeline_cfg.inject_pipeline_cfg_fallback(kernel_dir)
        _local_kernel_shims.inject_context_io_shims(kernel_dir, context_dir)
        _local_kernel_shims.inject_training_compat_shims(kernel_dir)
        _local_kernel_drift_guard.prepare_zero_overlap_drift_guard(
            base_dir=config.base_dir,
            slug=config.slug,
            context_dir=context_dir,
        )
        _local_kernel_shims.inject_zero_overlap_drift_shim(kernel_dir, context_dir)
        _kernel_bootstrap.inject_competition_slug_env(kernel_dir, config.slug)
        _kernel_bootstrap.inject_force_train_env(kernel_dir)
        _local_kernel_shims.ensure_training_progress_shim(kernel_dir)
        ensure_kernel_sources_valid(kernel_dir)
        _kernel_metadata.write_kernel_metadata(
            kernel_dir=kernel_dir,
            kernel_id=kernel_id,
            title=kernel_slug,
            code_file="kernel.py",
            kernel_type="script",
            accelerator=config.accelerator,
            enable_internet=config.enable_internet,
            competition_slug=config.slug,
            source_config=source_config,
        )
        validate_kernel_package(kernel_dir)
        return KernelPreparation(
            kernel_dir=kernel_dir,
            output_dir=output_dir,
            logs_dir=logs_dir,
            kernel_slug=kernel_slug,
            kernel_id=kernel_id,
        )


@dataclass(frozen=True)
class KernelSubmitPackageBuilder:
    def prepare(self, config: KernelSubmitBuildConfig) -> KernelPreparation:
        kernel_dir = config.base_dir / config.slug / "kernels" / config.run_id / f"submit-iter-{config.iteration}"
        output_dir = config.base_dir / config.slug / "runs" / config.run_id / f"iter-{config.iteration}" / "output"
        logs_dir = config.base_dir / config.slug / "runs" / config.run_id / f"iter-{config.iteration}" / "logs"
        output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        if kernel_dir.exists():
            shutil.rmtree(kernel_dir)
        kernel_dir.mkdir(parents=True, exist_ok=True)

        if not config.dry_run and not check_rules_accepted(config.slug, dry_run=False):
            raise RulesNotAcceptedError("Competition rules not accepted.")

        if not config.submission_path.exists() or not config.submission_path.is_file():
            raise KernelFailedError(f"Submission artifact not found: {config.submission_path}")

        submit_mode = str(config.mode or "wrapper").strip().lower()
        if submit_mode != "inference":
            _reject_static_tiny_code_competition_submission(
                slug=config.slug,
                base_dir=config.base_dir,
                submission_path=config.submission_path,
            )

        if not config.dry_run:
            print(f"[cyan]kernel init[/cyan]: {kernel_dir}")
            kernels_init(kernel_dir, dry_run=False)

        kernel_slug = _kernel_metadata.resolve_submit_kernel_slug(
            config.kernel_name,
            config.slug,
            config.run_id,
            config.iteration,
        )
        kernel_id = f"{config.kaggle_username}/{kernel_slug}"
        if submit_mode == "inference":
            custom_kernel_dir = config.base_dir / config.slug / "kernel"
            custom_kernel_path = custom_kernel_dir / "kernel.py"
            context_dir = config.base_dir / config.slug / "context"
            source_config = load_kernel_source_config(config.base_dir / config.slug / "plan.json")
            ensure_solution_path_allowed(custom_kernel_dir, artifacts_dir=config.base_dir, slug=config.slug)
            if not custom_kernel_path.exists():
                raise KernelFailedError(
                    f"Authoritative kernel entrypoint is missing for notebook submit. Expected: {custom_kernel_path}."
                )
            _kernel_package_files.copy_kernel_sources(custom_kernel_dir, kernel_dir)
            _kernel_package_files.copy_shared_kernel_runtime_modules(kernel_dir)
            _kernel_package_files.copy_competition_external_assets(
                base_dir=config.base_dir,
                slug=config.slug,
                kernel_dir=kernel_dir,
            )
            _kernel_package_files.sync_plan_snapshot(
                plan_path=config.base_dir / config.slug / "plan.json",
                targets=[kernel_dir / "plan.json"],
            )
            _kernel_bootstrap.ensure_kernel_import_path(kernel_dir)
            _kernel_bootstrap.inject_competition_slug_env(kernel_dir, config.slug)
            _kernel_module_inliner.inline_kernel_modules(kernel_dir)
            _local_kernel_data_resolver.inject_data_dir_resolver(kernel_dir)
            _local_kernel_pipeline_cfg.inject_pipeline_cfg_fallback(kernel_dir)
            _local_kernel_shims.inject_context_io_shims(kernel_dir, context_dir)
            _local_kernel_shims.inject_training_compat_shims(kernel_dir)
            _local_kernel_drift_guard.prepare_zero_overlap_drift_guard(
                base_dir=config.base_dir,
                slug=config.slug,
                context_dir=context_dir,
            )
            _local_kernel_shims.inject_zero_overlap_drift_shim(kernel_dir, context_dir)
            _kernel_bootstrap.inject_submit_inference_env(kernel_dir)
            _sanitize_submit_inference_output_roots(kernel_dir)
            _validate_inference_submit_kernel(kernel_dir)
            ensure_kernel_sources_valid(kernel_dir)
        else:
            source_config = None
            (kernel_dir / "kernel.py").write_text(
                _render_submission_kernel_script(config.submission_path),
                encoding="utf-8",
            )
            _kernel_bootstrap.ensure_kernel_import_path(kernel_dir)
            _kernel_bootstrap.inject_competition_slug_env(kernel_dir, config.slug)
            ensure_kernel_sources_valid(kernel_dir, require_kaggle_input=True)
        submit_accelerator = _resolve_submit_kernel_accelerator(config.accelerator)
        _kernel_metadata.write_kernel_metadata(
            kernel_dir=kernel_dir,
            kernel_id=kernel_id,
            title=kernel_slug,
            code_file="kernel.py",
            kernel_type="script",
            accelerator=submit_accelerator,
            enable_internet=config.enable_internet,
            competition_slug=config.slug,
            source_config=source_config,
        )
        validate_kernel_package(kernel_dir)
        return KernelPreparation(
            kernel_dir=kernel_dir,
            output_dir=output_dir,
            logs_dir=logs_dir,
            kernel_slug=kernel_slug,
            kernel_id=kernel_id,
            runtime_bootstrap_mode="submit_inference" if submit_mode == "inference" else "none",
            supersede_stale_queued=True,
        )


@dataclass(frozen=True)
class KernelJobMonitor:
    def push_and_wait(
        self,
        *,
        preparation: KernelPreparation,
        slug: str,
        timeout_minutes: int | None,
    ) -> str:
        _kernel_bootstrap.ensure_kernel_competition_slug_env(preparation.kernel_dir, slug)
        if preparation.runtime_bootstrap_mode == "force_train":
            _kernel_bootstrap.ensure_kernel_force_train_env(preparation.kernel_dir)
        elif preparation.runtime_bootstrap_mode == "submit_inference":
            _kernel_bootstrap.ensure_kernel_submit_inference_env(preparation.kernel_dir)
        _clear_stale_kernel_output(preparation.output_dir)
        push_attempt = 1
        kernel_id = preparation.kernel_id
        pending_kernel_id = _remote_kernel_state.read_pending_remote_kernel_id(
            preparation.logs_dir
        ) or _remote_kernel_state.last_pushed_kernel_id(
            preparation.logs_dir,
            kernel_id,
        )
        if pending_kernel_id:
            resumed_kernel_id = _resume_prior_kernel_if_active(
                preparation=preparation,
                kernel_id=pending_kernel_id,
                slug=slug,
                timeout_minutes=timeout_minutes,
            )
            if resumed_kernel_id:
                return resumed_kernel_id

        print(f"[cyan]kernel push[/cyan]: {preparation.kernel_dir}")
        push_output = kernels_push(preparation.kernel_dir, slug=slug, dry_run=False)
        _write_push_log(preparation.logs_dir, push_attempt, push_output)
        _raise_for_invalid_kernel_push_sources(push_output, kernel_dir=preparation.kernel_dir)
        pushed_kernel_id = _remote_kernel_state.extract_kernel_id_from_push(push_output)
        if pushed_kernel_id and pushed_kernel_id != kernel_id:
            print(f"[cyan]kernel id[/cyan]: {pushed_kernel_id}")
            kernel_id = pushed_kernel_id
        kernel_id = _resolve_kernel_id(kernel_id, preparation.kernel_slug)
        resolved_id = _wait_for_kernel_registration(kernel_id, preparation.kernel_slug)
        if not resolved_id:
            print("[yellow]kernel not found after push[/yellow]: retrying once")
            push_attempt += 1
            push_output = kernels_push(preparation.kernel_dir, slug=slug, dry_run=False)
            _write_push_log(preparation.logs_dir, push_attempt, push_output)
            _raise_for_invalid_kernel_push_sources(push_output, kernel_dir=preparation.kernel_dir)
            pushed_kernel_id = _remote_kernel_state.extract_kernel_id_from_push(push_output)
            if pushed_kernel_id and pushed_kernel_id != kernel_id:
                print(f"[cyan]kernel id[/cyan]: {pushed_kernel_id}")
                kernel_id = pushed_kernel_id
            kernel_id = _resolve_kernel_id(kernel_id, preparation.kernel_slug)
            resolved_id = _wait_for_kernel_registration(kernel_id, preparation.kernel_slug)
            if not resolved_id:
                raise KernelFailedError("Kaggle kernel not found after push; aborting.")
            kernel_id = resolved_id
        else:
            kernel_id = resolved_id

        print(f"[cyan]kernel status[/cyan]: {kernel_id}")
        _wait_for_kernel_and_record_pending(
            preparation=preparation,
            kernel_id=kernel_id,
            slug=slug,
            timeout_minutes=timeout_minutes,
        )
        _remote_kernel_state.clear_pending_remote_kernel(preparation.logs_dir)
        print(f"[cyan]kernel output[/cyan]: {preparation.output_dir}")
        kernels_output(kernel_id, preparation.output_dir, slug=slug, dry_run=False)
        return kernel_id


@dataclass(frozen=True)
class KernelLogParser:
    @staticmethod
    def collect_tail(output_dir: Path, max_lines: int = 50) -> str | None:
        return _kernel_logs.collect_log_tail(output_dir, max_lines=max_lines)


def resolve_kaggle_username(explicit: str | None) -> str:
    if explicit:
        return explicit
    env_user = os.getenv("KAGGLE_USERNAME")
    if env_user:
        return env_user

    config_dir_env = os.getenv("KAGGLE_CONFIG_DIR")
    candidates: list[Path] = []
    if config_dir_env:
        config_path = Path(config_dir_env).expanduser()
        # Support KAGGLE_CONFIG_DIR pointing to either a directory or kaggle.json file.
        if config_path.suffix.lower() == ".json":
            candidates.append(config_path)
        else:
            candidates.extend([config_path / "kaggle.json", config_path / "kaggle" / "kaggle.json"])
    else:
        candidates.append(Path("~/.kaggle/kaggle.json").expanduser())

    candidates.extend(
        [
            Path.home() / ".kaggle" / "kaggle.json",
            Path.home() / ".config" / "kaggle" / "kaggle.json",
        ]
    )
    candidates = list(dict.fromkeys(candidates))

    for kaggle_json in candidates:
        data = load_json_object(kaggle_json)
        if data is None:
            continue
        username = data.get("username")
        if username:
            return str(username)
    raise ValueError(
        "Kaggle username is required for kaggle_* compute modes. "
        "Set --kaggle-username, KAGGLE_USERNAME, or point KAGGLE_CONFIG_DIR "
        "to a directory (or kaggle.json file) containing a username."
    )


def run_kernel(
    *,
    slug: str,
    run_id: str,
    iteration: int,
    base_dir: Path,
    kaggle_username: str,
    kernel_name: str | None,
    accelerator: str,
    enable_internet: bool,
    score_source: str,
    metric: str,
    direction: str,
    holdout_frac: float,
    cv_folds: int,
    seed: int,
    dry_run: bool,
    timeout_minutes: int | None,
    hardware_profile: str | None = "auto",
) -> KernelRunResult:
    build_config = KernelBuildConfig(
        slug=slug,
        run_id=run_id,
        iteration=iteration,
        base_dir=base_dir,
        kaggle_username=kaggle_username,
        kernel_name=kernel_name,
        accelerator=accelerator,
        enable_internet=enable_internet,
        score_source=score_source,
        metric=metric,
        direction=direction,
        holdout_frac=holdout_frac,
        cv_folds=cv_folds,
        seed=seed,
        dry_run=dry_run,
        hardware_profile=hardware_profile,
    )
    preparation = KernelPackageBuilder().prepare(build_config)

    if dry_run:
        return KernelRunResult(
            kernel_id=preparation.kernel_id,
            output_dir=preparation.output_dir,
            submission_path=None,
            metrics_path=None,
        )

    kernel_id = KernelJobMonitor().push_and_wait(
        preparation=preparation,
        slug=slug,
        timeout_minutes=timeout_minutes,
    )
    submission_path = find_submission_file(preparation.output_dir)
    metrics_path = _find_output_file(preparation.output_dir, "metrics.json")
    return KernelRunResult(
        kernel_id=kernel_id,
        output_dir=preparation.output_dir,
        submission_path=submission_path,
        metrics_path=metrics_path,
    )


def run_submit_kernel(
    *,
    slug: str,
    run_id: str,
    iteration: int,
    base_dir: Path,
    kaggle_username: str,
    kernel_name: str | None,
    accelerator: str,
    enable_internet: bool,
    submission_path: Path,
    mode: str = "wrapper",
    dry_run: bool,
    timeout_minutes: int | None,
) -> KernelRunResult:
    build_config = KernelSubmitBuildConfig(
        slug=slug,
        run_id=run_id,
        iteration=iteration,
        base_dir=base_dir,
        kaggle_username=kaggle_username,
        kernel_name=kernel_name,
        accelerator=accelerator,
        enable_internet=enable_internet,
        submission_path=submission_path,
        mode=mode,
        dry_run=dry_run,
    )
    preparation = KernelSubmitPackageBuilder().prepare(build_config)
    if dry_run:
        return KernelRunResult(
            kernel_id=preparation.kernel_id,
            output_dir=preparation.output_dir,
            submission_path=None,
            metrics_path=None,
        )

    kernel_id = KernelJobMonitor().push_and_wait(
        preparation=preparation,
        slug=slug,
        timeout_minutes=timeout_minutes,
    )
    resolved_submission_path = find_submission_file(preparation.output_dir)
    metrics_path = _find_output_file(preparation.output_dir, "metrics.json")
    return KernelRunResult(
        kernel_id=kernel_id,
        output_dir=preparation.output_dir,
        submission_path=resolved_submission_path,
        metrics_path=metrics_path,
    )


def run_kernel_local(
    *,
    slug: str,
    run_id: str,
    iteration: int,
    base_dir: Path,
    accelerator: str,
    score_source: str,
    metric: str,
    direction: str,
    holdout_frac: float,
    cv_folds: int,
    seed: int,
    dry_run: bool,
    timeout_minutes: int | None,
    strict_accelerator: bool = False,
    hardware_profile: str | None = "auto",
) -> KernelRunResult:
    del metric, direction, holdout_frac, cv_folds, seed

    kernel_source_dir = base_dir / slug / "kernel"
    kernel_stage_dir = base_dir / slug / "kernels" / run_id / f"local-iter-{iteration}"
    run_dir = kernel_stage_dir.parent
    context_dir = base_dir / slug / "context"
    output_dir = base_dir / slug / "runs" / run_id / f"iter-{iteration}" / "output"
    logs_dir = base_dir / slug / "runs" / run_id / f"iter-{iteration}" / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    _local_sample_submission.ensure_local_sample_submission_file(base_dir=base_dir, slug=slug)
    _local_kernel_context.stage_local_kernel_data_dir(base_dir=base_dir, slug=slug, run_dir=run_dir)
    _local_kernel_context.stage_local_kernel_context_profile(base_dir=base_dir, slug=slug, run_dir=run_dir)

    ensure_solution_path_allowed(kernel_source_dir, artifacts_dir=base_dir, slug=slug)
    kernel_path = kernel_source_dir / "kernel.py"
    if not kernel_path.exists():
        raise KernelFailedError(f"Local kernel execution requires {kernel_path} to exist.")
    if kernel_stage_dir.exists():
        shutil.rmtree(kernel_stage_dir)
    kernel_stage_dir.mkdir(parents=True, exist_ok=True)
    _kernel_package_files.copy_kernel_sources(kernel_source_dir, kernel_stage_dir)
    _kernel_package_files.copy_shared_kernel_runtime_modules(kernel_stage_dir)
    _kernel_package_files.copy_competition_external_assets(base_dir=base_dir, slug=slug, kernel_dir=kernel_stage_dir)
    _kernel_package_files.sync_plan_snapshot(
        plan_path=base_dir / slug / "plan.json",
        targets=[
            kernel_stage_dir / "plan.json",
            kernel_stage_dir.parent / "plan.json",
        ],
    )
    kernel_path = kernel_stage_dir / "kernel.py"
    _kernel_plan_validation.validate_local_kernel_plan_runtime_hyperparameters(kernel_stage_dir / "plan.json")

    if strict_accelerator and accelerator == "gpu":
        availability = detect_local_gpu()
        if not availability.any:
            raise KernelFailedError("No local GPU detected while --strict-accelerator is enabled for local_gpu.")

    # Mirror packaging shims so local and kaggle kernel behavior are aligned.
    _kernel_bootstrap.ensure_kernel_import_path(kernel_stage_dir)
    _kernel_bootstrap.inject_competition_slug_env(kernel_stage_dir, slug)
    _kernel_bootstrap.inject_hardware_profile_env(kernel_stage_dir, hardware_profile, compute="local_gpu")
    _kernel_module_inliner.inline_kernel_modules(kernel_stage_dir)
    _local_kernel_data_resolver.inject_data_dir_resolver(kernel_stage_dir)
    _local_kernel_pipeline_cfg.inject_pipeline_cfg_fallback(kernel_stage_dir)
    _local_kernel_shims.inject_context_io_shims(kernel_stage_dir, context_dir)
    _local_kernel_shims.inject_local_runtime_shims(kernel_stage_dir)
    _local_kernel_shims.inject_training_compat_shims(kernel_stage_dir)
    _local_kernel_drift_guard.prepare_zero_overlap_drift_guard(base_dir=base_dir, slug=slug, context_dir=context_dir)
    _local_kernel_shims.inject_zero_overlap_drift_shim(kernel_stage_dir, context_dir)
    _kernel_bootstrap.inject_competition_slug_env(kernel_stage_dir, slug)
    _kernel_bootstrap.inject_force_train_env(kernel_stage_dir)
    _local_kernel_shims.ensure_training_progress_shim(kernel_stage_dir)
    ensure_kernel_sources_valid(kernel_stage_dir, require_kaggle_input=False)
    local_aux_env, local_aux_notes = _local_kernel_aux_inputs.stage_local_kernel_aux_inputs(
        base_dir=base_dir,
        slug=slug,
        kernel_stage_dir=kernel_stage_dir,
    )
    for note in local_aux_notes:
        print(f"[yellow]kernel local[/yellow]: {note}")
    local_model_env, local_model_notes = _local_kernel_models.stage_local_kernel_models(
        base_dir=base_dir,
        slug=slug,
        kernel_stage_dir=kernel_stage_dir,
    )
    for note in local_model_notes:
        print(f"[yellow]kernel local[/yellow]: {note}")

    if dry_run:
        return KernelRunResult(
            kernel_id=f"local/{slug}",
            output_dir=output_dir,
            submission_path=None,
            metrics_path=None,
        )

    timeout_sec = None if timeout_minutes is None else max(60, int(timeout_minutes * 60))
    eta_total_sec, eta_samples = _local_kernel_duration.estimate_local_kernel_duration_seconds(
        base_dir=base_dir,
        slug=slug,
    )
    started_at = time.time()
    monotonic_start = time.monotonic()
    progress_tracker = _local_kernel_progress.build_local_kernel_progress_tracker(
        base_dir=base_dir,
        slug=slug,
        watch_dirs=[output_dir, kernel_stage_dir / "outputs", base_dir / slug / "kernel_output"],
        started_at_wall=started_at,
        started_at_monotonic=monotonic_start,
    )
    _local_kernel_progress.print_local_kernel_progress(
        elapsed_sec=0.0,
        timeout_sec=timeout_sec,
        eta_total_sec=eta_total_sec,
        eta_samples=eta_samples,
        progress_tracker=progress_tracker,
        accelerator=accelerator,
    )
    env = os.environ.copy()
    env["KAGGLEBOT_OUTPUT_DIR"] = str(output_dir)
    env.setdefault("KAGGLEBOT_LOCAL_KERNEL", "1")
    env.setdefault("KAGGLEBOT_SLUG", slug)
    env.setdefault("KAGGLEBOT_COMPETITION_SLUG", slug)
    env.setdefault("KAGGLEBOT_RUN_ID", run_id)
    env.setdefault("KAGGLEBOT_ITERATION", str(iteration))
    env.setdefault("KAGGLEBOT_ACCELERATOR", accelerator)
    for key, value in hardware_env(resolve_hardware_profile(hardware_profile, compute="local_gpu")).items():
        env.setdefault(key, value)
    for key, value in local_aux_env.items():
        env[key] = value
    for key, value in local_model_env.items():
        env[key] = value
    env_notes = _local_kernel_runtime_env.apply_local_runtime_env_defaults(
        env=env,
        accelerator=accelerator,
        local_working_dir=kernel_stage_dir / "outputs" / "kaggle_working",
    )
    for note in env_notes:
        print(f"[yellow]kernel local[/yellow]: {note}")
    memory_cap_bytes = _local_kernel_limits.resolve_memory_cap_bytes(env)
    if memory_cap_bytes is not None:
        print(f"[yellow]kernel local[/yellow]: host memory guard active at {memory_cap_bytes // (1024 * 1024)} MiB RSS")

    heartbeat_stop = threading.Event()
    heartbeat = threading.Thread(
        target=_local_kernel_progress.local_kernel_heartbeat,
        kwargs={
            "stop_event": heartbeat_stop,
            "start_monotonic": monotonic_start,
            "timeout_sec": timeout_sec,
            "eta_total_sec": eta_total_sec,
            "eta_samples": eta_samples,
            "progress_tracker": progress_tracker,
            "accelerator": accelerator,
            "interval_sec": _LOCAL_KERNEL_HEARTBEAT_INTERVAL_SEC,
        },
        daemon=True,
    )
    heartbeat.start()
    try:

        def run_once_with_watchdog(*, current_env: dict[str, str]) -> _LocalKernelExecResult:
            return _run_local_kernel_once(
                kernel_path=kernel_path,
                kernel_stage_dir=kernel_stage_dir,
                current_env=current_env,
                timeout_sec=timeout_sec,
                line_callback=progress_tracker.observe_line,
                progress_tracker=progress_tracker,
            )

        exec_result = run_once_with_watchdog(current_env=env)
        result = exec_result.command_result
    except subprocess.TimeoutExpired as exc:
        raise KernelTimeoutError(f"Local kernel timed out after {timeout_sec}s.") from exc
    finally:
        heartbeat_stop.set()
        heartbeat.join(timeout=1.0)

    (logs_dir / "local_kernel_stdout.log").write_text(result.stdout, encoding="utf-8")
    (logs_dir / "local_kernel_stderr.log").write_text(result.stderr, encoding="utf-8")

    if exec_result.killed_for_memory:
        detail = ""
        stdout_tail = truncate_lines(result.stdout[-4000:], max_lines=80)
        if stdout_tail:
            detail = f"\n{stdout_tail}"
        raise KernelFailedError(
            f"{exec_result.memory_kill_message} Peak RSS={exec_result.peak_rss_bytes // (1024 * 1024)} MiB.{detail}"
        )

    if exec_result.killed_for_stall:
        detail = ""
        stdout_tail = truncate_lines(result.stdout[-4000:], max_lines=80)
        if stdout_tail:
            detail = f"\n{stdout_tail}"
        raise KernelFailedError(f"{exec_result.stall_kill_message or 'Local kernel stalled.'}{detail}")

    if result.returncode != 0:
        combined = "\n".join(part for part in [result.stdout, result.stderr] if part)
        enable_llm_env = env.get("ENABLE_LLM")
        llm_disabled_by_env = enable_llm_env is not None and not _local_kernel_runtime_env.env_truthy(enable_llm_env)
        if (
            accelerator == "gpu"
            and not strict_accelerator
            and not llm_disabled_by_env
            and _local_kernel_runtime_env.detect_cuda_oom(combined)
        ):
            retry_env = env.copy()
            retry_notes = _local_kernel_runtime_env.apply_local_kernel_oom_fallback_env(retry_env)
            for note in retry_notes:
                print(f"[yellow]kernel local[/yellow]: {note}")
            try:
                shutil.rmtree(run_dir / "outputs", ignore_errors=True)
            except Exception:
                pass
            try:
                shutil.rmtree(kernel_stage_dir / "outputs", ignore_errors=True)
            except Exception:
                pass
            time.sleep(2.0)

            retry_exec_result = run_once_with_watchdog(current_env=retry_env)
            retry_result = retry_exec_result.command_result
            (logs_dir / "local_kernel_stdout_oom_retry.log").write_text(retry_result.stdout, encoding="utf-8")
            (logs_dir / "local_kernel_stderr_oom_retry.log").write_text(retry_result.stderr, encoding="utf-8")
            if retry_exec_result.killed_for_memory:
                detail = ""
                stdout_tail = truncate_lines(retry_result.stdout[-4000:], max_lines=80)
                if stdout_tail:
                    detail = f"\n{stdout_tail}"
                raise KernelFailedError(
                    f"{retry_exec_result.memory_kill_message} "
                    f"Peak RSS={retry_exec_result.peak_rss_bytes // (1024 * 1024)} MiB.{detail}"
                )
            if retry_exec_result.killed_for_stall:
                detail = ""
                stdout_tail = truncate_lines(retry_result.stdout[-4000:], max_lines=80)
                if stdout_tail:
                    detail = f"\n{stdout_tail}"
                raise KernelFailedError(
                    f"{retry_exec_result.stall_kill_message or 'Local kernel stalled after CUDA OOM retry.'}{detail}"
                )
            if retry_result.returncode == 0:
                result = retry_result
                (logs_dir / "local_kernel_stdout.log").write_text(result.stdout, encoding="utf-8")
                (logs_dir / "local_kernel_stderr.log").write_text(result.stderr, encoding="utf-8")
            else:
                stdout_tail = truncate_lines(retry_result.stdout[-4000:], max_lines=80)
                stderr_tail = truncate_lines(retry_result.stderr[-4000:], max_lines=80)
                detail = "\n".join(part for part in [stdout_tail, stderr_tail] if part).strip()
                if detail:
                    detail = f"\n{detail}"
                raise KernelFailedError(
                    f"Local kernel execution failed with CUDA OOM, then failed again after disabling LLM.{detail}"
                )

        if result.returncode != 0:
            stdout_tail = truncate_lines(result.stdout[-4000:], max_lines=80)
            stderr_tail = truncate_lines(result.stderr[-4000:], max_lines=80)
            detail = "\n".join(part for part in [stdout_tail, stderr_tail] if part).strip()
            if detail:
                detail = f"\n{detail}"
            raise KernelFailedError(f"Local kernel execution failed with exit code {result.returncode}.{detail}")
    _local_kernel_duration.append_local_kernel_duration_history(
        base_dir=base_dir,
        slug=slug,
        run_id=run_id,
        iteration=iteration,
        duration_sec=result.duration_sec,
    )
    print(f"[cyan]kernel local complete[/cyan]: elapsed={result.duration_sec:.0f}s")

    submission_src, metrics_src = _resolve_local_kernel_artifacts(
        kernel_dir=kernel_stage_dir,
        output_dir=output_dir,
        started_at=started_at,
    )
    if submission_src is None:
        raise KernelFailedError("Local kernel completed but submission output was not found.")

    submission_dst = _copy_artifact_if_needed(
        source=submission_src,
        destination=output_dir / submission_src.name,
    )
    metrics_dst = None
    if metrics_src is not None:
        metrics_dst = _copy_artifact_if_needed(
            source=metrics_src,
            destination=output_dir / "metrics.json",
        )
        metrics_dst = _local_kernel_metrics_normalization.normalize_local_kernel_metrics(
            slug=slug,
            data_dir=base_dir / slug / "data",
            metrics_path=metrics_dst,
            score_source=score_source,
        )
    _kernel_contracts.enforce_competition_kernel_contract(
        slug=slug,
        logs_dir=logs_dir,
        metrics_path=metrics_dst,
    )
    for filename in (
        "oof_predictions.csv",
        "split_diagnostics.json",
        "feature_suspects.csv",
        "submission_manifest.json",
        "metrics_summary.json",
        "cv_results.json",
        "cv_summary.json",
        "pipeline_diagnostics.json",
    ):
        optional_src = _resolve_local_kernel_artifact_file(
            kernel_dir=kernel_stage_dir,
            output_dir=output_dir,
            started_at=started_at,
            filename=filename,
        )
        if optional_src is None:
            continue
        _copy_artifact_if_needed(
            source=optional_src,
            destination=output_dir / filename,
        )

    return KernelRunResult(
        kernel_id=f"local/{slug}",
        output_dir=output_dir,
        submission_path=submission_dst,
        metrics_path=metrics_dst,
    )


LOG_POLL_INTERVAL = 2.0
HEARTBEAT_INTERVAL = 30.0
STATUS_ERROR_SLEEP = 10.0
MAX_STATUS_ERRORS = 6
KERNEL_REGISTER_RETRIES = 24
KERNEL_REGISTER_SLEEP = 5.0


def _raise_kernel_timeout(kernel_id: str, last_status: str | None) -> None:
    status = (last_status or "unknown").lower()
    if is_kernel_status_running(status):
        raise KernelStillRunningError(
            f"Kaggle kernel {kernel_id} is still {status} after the local wait budget; "
            "leaving the remote run active and refusing to push a duplicate version."
        )
    raise KernelTimeoutError(
        f"Kaggle kernel {kernel_id} did not complete within the local wait budget; last status was {status}."
    )


def _wait_for_kernel_and_record_pending(
    *,
    preparation: KernelPreparation,
    kernel_id: str,
    slug: str,
    timeout_minutes: int | None,
    initial_queued_since: float | None = None,
) -> None:
    try:
        _wait_for_kernel(
            kernel_id,
            slug,
            timeout_minutes,
            output_dir=preparation.output_dir,
            initial_queued_since=initial_queued_since,
        )
    except KernelStillRunningError:
        _remote_kernel_state.write_pending_remote_kernel(
            preparation.logs_dir,
            kernel_id=kernel_id,
            kernel_slug=preparation.kernel_slug,
        )
        raise


def _resume_prior_kernel_if_active(
    *,
    preparation: KernelPreparation,
    kernel_id: str,
    slug: str,
    timeout_minutes: int | None,
) -> str | None:
    try:
        output = kernels_status(kernel_id, slug=slug, dry_run=False)
    except KaggleCliError as exc:
        _remote_kernel_state.write_pending_remote_kernel(
            preparation.logs_dir,
            kernel_id=kernel_id,
            kernel_slug=preparation.kernel_slug,
        )
        raise KernelStillRunningError(
            f"Kaggle kernel {kernel_id} has a prior push record, but its status could not be verified; "
            "refusing to push a duplicate version."
        ) from exc

    status = parse_kernel_status(output)
    if is_kernel_status_failed(status):
        _remote_kernel_state.clear_pending_remote_kernel(preparation.logs_dir)
        print(f"[yellow]kernel resume[/yellow]: prior remote kernel failed ({status}); pushing a new version")
        return None
    if not (is_kernel_status_running(status) or is_kernel_status_complete(status)):
        print(f"[yellow]kernel resume[/yellow]: prior remote kernel status is {status}; pushing a new version")
        return None

    initial_queued_since = (
        _remote_kernel_state.queued_since_from_push_logs(preparation.logs_dir)
        if is_kernel_status_queued(status)
        else None
    )
    if (
        is_kernel_status_queued(status)
        and preparation.supersede_stale_queued
        and _remote_kernel_state.is_remote_kernel_queue_stale(initial_queued_since)
    ):
        _remote_kernel_state.clear_pending_remote_kernel(preparation.logs_dir)
        print(f"[yellow]kernel resume[/yellow]: prior remote kernel is stale queued ({status}); pushing a new version")
        return None

    _clear_stale_kernel_output(preparation.output_dir)
    print(f"[yellow]kernel resume[/yellow]: waiting for existing remote kernel {kernel_id} ({status})")
    if is_kernel_status_running(status):
        _wait_for_kernel_and_record_pending(
            preparation=preparation,
            kernel_id=kernel_id,
            slug=slug,
            timeout_minutes=timeout_minutes,
            initial_queued_since=initial_queued_since,
        )
    _remote_kernel_state.clear_pending_remote_kernel(preparation.logs_dir)
    print(f"[cyan]kernel output[/cyan]: {preparation.output_dir}")
    kernels_output(kernel_id, preparation.output_dir, slug=slug, dry_run=False)
    return kernel_id


def _wait_for_kernel(
    kernel_id: str,
    slug: str,
    timeout_minutes: int | None,
    *,
    output_dir: Path,
    initial_queued_since: float | None = None,
) -> None:
    deadline = None
    if timeout_minutes is not None:
        deadline = time.monotonic() + max(timeout_minutes, 1) * 60
    started_at = time.monotonic()
    last_status = None
    last_log_fetch = 0.0
    log_state = _kernel_logs.KernelLogState()
    status_errors = 0
    queued_since = initial_queued_since
    queued_timeout_sec = _remote_kernel_state.remote_kernel_queued_timeout_sec()
    while True:
        try:
            output = kernels_status(kernel_id, slug=slug, dry_run=False)
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
                if deadline is not None and time.monotonic() > deadline:
                    _raise_kernel_timeout(kernel_id, last_status)
                if MAX_STATUS_ERRORS is not None and status_errors >= MAX_STATUS_ERRORS:
                    kernel_url = f"https://www.kaggle.com/code/{kernel_id}"
                    raise KaggleNetworkError(
                        "Kaggle API unreachable while polling kernel status. "
                        f"Check network/DNS and monitor the kernel at {kernel_url}.",
                        getattr(exc, "command", None),
                        exit_code=getattr(exc, "exit_code", None),
                        output=getattr(exc, "output", ""),
                    ) from exc
                time.sleep(STATUS_ERROR_SLEEP)
                continue
            message = f"[yellow]kernel status failed[/yellow]: {detail or 'unknown error'} (attempt {status_errors})"
            print(message)
            if deadline is not None and time.monotonic() > deadline:
                _raise_kernel_timeout(kernel_id, last_status)
            if MAX_STATUS_ERRORS is not None and status_errors >= MAX_STATUS_ERRORS:
                raise KernelFailedError(
                    f"Kaggle kernel status failed {status_errors} times. Last error: {detail or 'unknown error'}"
                ) from exc
            time.sleep(STATUS_ERROR_SLEEP)
            continue
        status = parse_kernel_status(output)
        if status != last_status:
            print(f"[cyan]kernel status[/cyan]: {status}")
            last_status = status
        now = time.monotonic()
        if is_kernel_status_queued(status):
            if queued_since is None:
                queued_since = now
            if queued_timeout_sec is not None and now - queued_since >= queued_timeout_sec:
                _remote_kernel_state.raise_kernel_queued_timeout(kernel_id, now - queued_since, queued_timeout_sec)
        else:
            queued_since = None
        if now - last_log_fetch >= LOG_POLL_INTERVAL:
            _try_fetch_kernel_output(kernel_id, output_dir=output_dir, slug=slug)
            had_logs = _kernel_logs.print_kernel_logs(output_dir, log_state)
            if had_logs:
                log_state.last_log_at = now
            last_log_fetch = now
            log_failure = _kernel_logs.detect_failure_in_logs(output_dir)
            if log_failure:
                log_failure = truncate_lines(log_failure, max_lines=5)
                message = f"Kaggle kernel error detected in logs.\n\n--- kernel log tail ---\n{log_failure}"
                raise KernelFailedError(message)
        if is_kernel_status_running(status):
            if log_state.last_heartbeat == 0.0 or now - log_state.last_heartbeat >= HEARTBEAT_INTERVAL:
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
            _try_fetch_kernel_output(kernel_id, output_dir=output_dir, slug=slug)
            log_tail = _kernel_logs.collect_log_tail(output_dir)
            message = f"Kaggle kernel failed: {output}"
            if log_tail:
                log_tail = truncate_lines(log_tail, max_lines=5)
                message = f"{message}\n\n--- kernel log tail ---\n{log_tail}"
            raise KernelFailedError(message)
        time.sleep(STATUS_ERROR_SLEEP)
        if deadline is not None and time.monotonic() > deadline:
            _raise_kernel_timeout(kernel_id, last_status)


def _wait_for_kernel_registration(kernel_id: str, kernel_slug: str) -> str | None:
    for attempt in range(1, KERNEL_REGISTER_RETRIES + 1):
        try:
            kernels_status(kernel_id, dry_run=False)
            return kernel_id
        except KaggleCliError as exc:
            detail = (exc.output or str(exc)).strip().replace("\n", " ")
            if detail:
                print(f"[yellow]kernel status unavailable[/yellow]: {detail} (attempt {attempt})")
        try:
            if kernel_exists(kernel_id):
                return kernel_id
            resolved = kernel_id_by_title(kernel_slug)
            if resolved:
                return resolved
        except KaggleCliError as exc:
            detail = (exc.output or str(exc)).strip().replace("\n", " ")
            if detail:
                print(f"[yellow]kernel list failed[/yellow]: {detail} (attempt {attempt})")
        time.sleep(KERNEL_REGISTER_SLEEP)
    return None


def _resolve_kernel_id(kernel_id: str, kernel_slug: str) -> str:
    try:
        resolved = kernel_id_by_title(kernel_slug)
    except KaggleCliError:
        return kernel_id
    if resolved and resolved != kernel_id:
        print(f"[cyan]kernel id[/cyan]: {resolved}")
        return resolved
    return kernel_id


def _write_push_log(logs_dir: Path, attempt: int, output: str) -> None:
    path = logs_dir / f"kernel_push-{attempt:02d}.txt"
    path.write_text(output.strip() + "\n", encoding="utf-8")


def _clear_stale_kernel_output(output_dir: Path) -> None:
    """Remove stale files from prior kernel runs in the same output directory."""
    if not output_dir.exists():
        return
    for path in output_dir.iterdir():
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
                continue
            path.unlink()
        except OSError:
            continue


def _try_fetch_kernel_output(kernel_id: str, *, output_dir: Path, slug: str) -> None:
    try:
        kernels_output(kernel_id, output_dir, slug=slug, dry_run=False, force=True, quiet=True)
    except KaggleCliError:
        return
