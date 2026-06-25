from __future__ import annotations

import ast
import base64
import codecs
import gzip
import json
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

from kagglebot import kernel_logs as _kernel_logs
from kagglebot import kernel_metadata as _kernel_metadata
from kagglebot import kernel_plan_validation as _kernel_plan_validation
from kagglebot import local_kernel_context as _local_kernel_context
from kagglebot import local_kernel_duration as _local_kernel_duration
from kagglebot import local_kernel_limits as _local_kernel_limits
from kagglebot import local_kernel_models as _local_kernel_models
from kagglebot import local_kernel_progress as _local_kernel_progress
from kagglebot import local_sample_submission as _local_sample_submission
from kagglebot import remote_kernel_state as _remote_kernel_state
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
from kagglebot.json_utils import load_json_object, read_json_object, write_json_object
from kagglebot.kaggle_api import (
    check_rules_accepted,
    kernel_exists,
    kernel_id_by_title,
    kernels_init,
    kernels_output,
    kernels_push,
    kernels_status,
)
from kagglebot.kernel_outputs import copy_artifact_if_needed as _copy_artifact_if_needed
from kagglebot.kernel_outputs import find_output_file as _find_output_file
from kagglebot.kernel_outputs import find_submission_file
from kagglebot.kernel_outputs import resolve_local_kernel_artifact_file as _resolve_local_kernel_artifact_file
from kagglebot.kernel_outputs import resolve_local_kernel_artifacts as _resolve_local_kernel_artifacts
from kagglebot.kernel_sources import load_kernel_source_config
from kagglebot.kernel_status import (
    is_kernel_status_complete,
    is_kernel_status_failed,
    is_kernel_status_queued,
    is_kernel_status_running,
    parse_kernel_status,
)
from kagglebot.logging_utils import truncate_lines
from kagglebot.paths import CompetitionPaths
from kagglebot.solution_guard import ensure_solution_path_allowed
from kagglebot.validators import ensure_kernel_sources_valid, validate_kernel_package
from kagglebot.writeup import infer_code_competition_from_paths

_COLUMN_MAP_FILENAME = "column_map.json"
_COLUMN_MAP_SHIM_MARKER = "# kagglebot: column-map-shim"
_COLUMN_FILL_FILENAME = "column_fill.json"
_COLUMN_FILL_SHIM_MARKER = "# kagglebot: column-fill-shim"
_OBJECT_COERCE_FILENAME = "object_coerce.json"
_OBJECT_COERCE_SHIM_MARKER = "# kagglebot: object-coerce-shim"
_DEVICE_COERCE_FILENAME = "device_coerce.json"
_DEVICE_COERCE_SHIM_MARKER = "# kagglebot: device-coerce-shim"
_ZERO_OVERLAP_DRIFT_GUARD_FILENAME = "zero_overlap_drift_guard.json"
_ZERO_OVERLAP_DRIFT_SHIM_MARKER = "# kagglebot: zero-overlap-drift-shim"
_KAGGLE_WORKING_REDIRECT_SHIM_MARKER = "# kagglebot: kaggle-working-redirect-shim"
_LGBM_GPU_GUARD_SHIM_MARKER = "# kagglebot: lgbm-gpu-guard-shim"
_TORCH_RUNTIME_GUARD_SHIM_MARKER = "# kagglebot: torch-runtime-guard-shim"
_TRAIN_PROGRESS_SHIM_MARKER = "# kagglebot: train-progress-shim"
_TRANSFORMERS_EVAL_STRATEGY_SHIM_MARKER = "# kagglebot: transformers-eval-strategy-shim"
_KERNEL_FORCE_TRAIN_MARKER = "# kagglebot:force_train"
_KERNEL_SUBMIT_INFERENCE_MARKER = "# kagglebot:submit_inference"
_LOCAL_KERNEL_HEARTBEAT_INTERVAL_SEC = 30.0
_LOCAL_KERNEL_MEMORY_POLL_INTERVAL_SEC = 1.0
_LOCAL_KERNEL_STDOUT_POLL_INTERVAL_SEC = 0.2
_LOCAL_KERNEL_EXIT_PIPE_DRAIN_SEC = 1.0
_LOCAL_LGBM_GPU_PROBE_OK: bool | None = None
_SUBMIT_KERNEL_ACCELERATOR_ENV = "KAGGLEBOT_SUBMIT_KERNEL_ACCELERATOR"
_ZERO_OVERLAP_DRIFT_MIN_TVD = 0.20
_ZERO_OVERLAP_DRIFT_MIN_ABS_CORR = 0.08
_ZERO_OVERLAP_DRIFT_MIN_ZERO_OVERLAP_RATIO = 0.50
_ZERO_OVERLAP_DRIFT_MAX_CAT_UNIQUE_RATIO = 0.98
_BVS_KERNEL_CONTRACT_SLUG_PREFIX = "beyond-visible-spectrum-ai-for-agriculture-2026"
_BVS_TIMM_FAILURE_MARKERS = (
    "timm is unavailable",
    "timm.create_model is missing",
    "skipping tri_branch_timm_gated because timm is unavailable",
    "falling back to smallspectralencoder for rgb",
)
_TRUSTED_KERNEL_SCORE_SOURCES = frozenset({"cv", "holdout", "consensus"})
_URBAN_FLOOD_SAMPLEISH_SCORE_SOURCES = frozenset(
    {
        "sample_diagnostic",
        "sample_mode_smoke_cv",
        "sample",
        "fallback",
    }
)
_URBAN_FLOOD_FLAT_FULL_REQUIRED_FILES = frozenset(
    {
        "1d_nodes_dynamic_all.csv",
        "2d_nodes_dynamic_all.csv",
        "test_1d_nodes_dynamic_all.csv",
        "test_2d_nodes_dynamic_all.csv",
        "timesteps.csv",
        "test_timesteps.csv",
        "sample_submission.csv",
    }
)

_BASELINE_SCORE_ASSIGNMENT_RE = re.compile(
    r"\b(?P<name>[a-z_][a-z0-9_]*?(?:score|auc|rmse|mae|mse|f1|loss|accuracy|acc|precision|recall|map|ndcg|logloss|brier|gini))\s*=\s*"
    r"(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
_INVALID_KERNEL_SOURCE_RE = re.compile(
    r"The following are not valid (?P<kind>dataset|model|kernel) sources "
    r"and could not be added to the kernel:\s*(?P<items>\[[^\n]+\])",
    re.IGNORECASE,
)


def _requires_bvs_kernel_contract(slug: str) -> bool:
    return slug.strip().lower().startswith(_BVS_KERNEL_CONTRACT_SLUG_PREFIX)


def _normalize_kernel_score_source(value: object) -> str:
    return str(value or "").strip().lower()


def _looks_like_urban_flood_flat_full_root(data_dir: Path) -> bool:
    if not data_dir.exists() or not data_dir.is_dir():
        return False
    names = {child.name for child in data_dir.iterdir() if child.is_file()}
    return _URBAN_FLOOD_FLAT_FULL_REQUIRED_FILES.issubset(names)


def _normalize_local_kernel_metrics(
    *,
    slug: str,
    data_dir: Path,
    metrics_path: Path | None,
    score_source: str,
) -> Path | None:
    if slug.strip().lower() != "urban-flood-modelling":
        return metrics_path
    if metrics_path is None or not metrics_path.exists():
        return metrics_path
    if not _looks_like_urban_flood_flat_full_root(data_dir):
        return metrics_path

    payload = load_json_object(metrics_path)
    if payload is None:
        return metrics_path

    normalized_payload_source = _normalize_kernel_score_source(payload.get("score_source"))
    requested_source = _normalize_kernel_score_source(score_source)
    if requested_source not in _TRUSTED_KERNEL_SCORE_SOURCES:
        requested_source = "cv"

    if normalized_payload_source not in _URBAN_FLOOD_SAMPLEISH_SCORE_SOURCES and bool(
        payload.get("full_dataset_resolved")
    ):
        return metrics_path

    payload["score_source"] = requested_source
    payload["dataset_kind"] = "full"
    payload["dataset_mode"] = "full"
    payload["full_dataset_resolved"] = True
    payload["data_root_layout"] = "flat_full"
    payload["metrics_normalized_by"] = "kernel_runner.local_full_data_guard"
    try:
        write_json_object(metrics_path, payload)
    except OSError:
        return metrics_path
    return metrics_path


def _collect_local_kernel_log_text(logs_dir: Path) -> str:
    chunks: list[str] = []
    for name in (
        "local_kernel_stdout.log",
        "local_kernel_stderr.log",
        "local_kernel_stdout_oom_retry.log",
        "local_kernel_stderr_oom_retry.log",
    ):
        path = logs_dir / name
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if text:
            chunks.append(text)
    return "\n".join(chunks)


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


def _extract_kernel_size_markers(log_text: str) -> list[int]:
    pattern = re.compile(r"\b(?:load_size|img_size)\s*=\s*(\d+)\b")
    values: list[int] = []
    for match in pattern.finditer(log_text):
        try:
            values.append(int(match.group(1)))
        except ValueError:
            continue
    return values


def _enforce_competition_kernel_contract(
    *,
    slug: str,
    logs_dir: Path,
    metrics_path: Path | None,
) -> None:
    """Enforce competition-specific quality contracts to prevent silent regressions."""
    if not _requires_bvs_kernel_contract(slug):
        return

    errors: list[str] = []
    payload: dict[str, object] = {}
    if metrics_path is None or not metrics_path.exists():
        errors.append("metrics.json is missing; cannot validate BVS kernel contract.")
    else:
        try:
            payload = read_json_object(metrics_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"metrics.json is unreadable: {exc}")
        except ValueError:
            errors.append("metrics.json payload must be a JSON object.")

    log_text = _collect_local_kernel_log_text(logs_dir)
    lowered_log = log_text.lower()
    for marker in _BVS_TIMM_FAILURE_MARKERS:
        if marker in lowered_log:
            errors.append(f"timm/ConvNeXt fallback marker detected in logs: {marker}")

    size_markers = _extract_kernel_size_markers(log_text)
    if not size_markers:
        errors.append("No img_size/load_size markers found in local kernel logs.")
    else:
        undersized = sorted({value for value in size_markers if value < 128})
        if undersized:
            errors.append(f"Detected img_size/load_size below 128 in logs: {undersized}")

    if payload:
        model_name = str(payload.get("model_name") or "").strip().lower()
        if model_name in {"resnet50", "small_rgb_encoder", "none"}:
            errors.append(f"Weak fallback backbone detected in metrics model_name={model_name!r}.")

        pipelines = payload.get("pipelines")
        if not isinstance(pipelines, list) or len(pipelines) < 2:
            errors.append("metrics.json must report at least two pipeline candidates for ensemble selection.")

        chosen_pipeline = str(payload.get("chosen_pipeline") or "").strip().lower()
        if not chosen_pipeline:
            errors.append("metrics.json must include chosen_pipeline.")
        elif "ensemble" not in chosen_pipeline:
            errors.append(f"chosen_pipeline must be ensemble-based, got: {chosen_pipeline!r}.")

    if errors:
        issue_text = "\n".join(f"- {message}" for message in errors)
        raise KernelFailedError(f"BVS kernel contract failed (timm/size/ensemble guard):\n{issue_text}")


def _env_truthy(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _module_available(module_name: str) -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _local_lightgbm_gpu_probe_usable() -> bool:
    global _LOCAL_LGBM_GPU_PROBE_OK
    if _LOCAL_LGBM_GPU_PROBE_OK is not None:
        return _LOCAL_LGBM_GPU_PROBE_OK
    if _env_truthy(os.environ.get("KAGGLEBOT_SKIP_LGBM_GPU_PROBE")):
        _LOCAL_LGBM_GPU_PROBE_OK = False
        return False
    try:
        import lightgbm as lgb
        import numpy as np
    except Exception:
        _LOCAL_LGBM_GPU_PROBE_OK = False
        return False

    rng = np.random.default_rng(42)
    x = rng.normal(size=(128, 4)).astype(np.float32)
    y = (0.4 * x[:, 0] - 0.3 * x[:, 1] + 0.2 * x[:, 2]).astype(np.float32)
    try:
        model = lgb.LGBMRegressor(
            n_estimators=16,
            learning_rate=0.1,
            num_leaves=15,
            max_depth=5,
            min_data_in_leaf=1,
            min_data_in_bin=1,
            device_type="gpu",
            verbosity=-1,
        )
        model.fit(x, y)
    except Exception:
        _LOCAL_LGBM_GPU_PROBE_OK = False
        return False
    _LOCAL_LGBM_GPU_PROBE_OK = True
    return True


def _apply_local_runtime_env_defaults(
    *,
    env: dict[str, str],
    accelerator: str,
    local_working_dir: Path,
) -> list[str]:
    """Apply local execution defaults and force training to stay enabled."""
    notes: list[str] = []
    env.setdefault("KAGGLEBOT_LOCAL_WORKING_DIR", str(local_working_dir))
    env.setdefault("KAGGLEBOT_DISABLE_KAGGLE_WORKING_WRITES", "1")
    env.setdefault("KAGGLEBOT_NUM_WORKERS", "0")
    env.setdefault("KAGGLEBOT_TORCH_SHARING_STRATEGY", "file_system")
    env.setdefault("KAGGLEBOT_LOCAL_NOFILE", "4096")
    env.setdefault(_local_kernel_limits.STALL_ENV, str(int(_local_kernel_limits.DEFAULT_STALL_SEC)))
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["KAGGLEBOT_DO_TRAIN"] = "1"
    env["KAGGLEBOT_FORCE_TRAIN"] = "1"
    env["KAGGLEBOT_ALLOW_MODEL_DOWNLOAD"] = "1"
    notes.append("forcing KAGGLEBOT_DO_TRAIN=1 and KAGGLEBOT_FORCE_TRAIN=1")
    notes.append("forcing KAGGLEBOT_ALLOW_MODEL_DOWNLOAD=1")
    notes.append(f"defaulting KAGGLEBOT_NUM_WORKERS={env['KAGGLEBOT_NUM_WORKERS']} for local kernels")
    notes.append(f"defaulting KAGGLEBOT_TORCH_SHARING_STRATEGY={env['KAGGLEBOT_TORCH_SHARING_STRATEGY']}")
    notes.append(f"defaulting KAGGLEBOT_LOCAL_NOFILE={env['KAGGLEBOT_LOCAL_NOFILE']}")
    notes.append(f"defaulting {_local_kernel_limits.STALL_ENV}={env[_local_kernel_limits.STALL_ENV]}")
    notes.append(f"defaulting PYTHONUNBUFFERED={env['PYTHONUNBUFFERED']}")

    if not _module_available("xgboost"):
        env.setdefault("USE_XGB", "0")
        env.setdefault("KAGGLEBOT_DISABLE_XGBOOST", "1")
        notes.append("xgboost unavailable; forcing USE_XGB=0")

    force_lgbm_gpu = _env_truthy(os.environ.get("KAGGLEBOT_FORCE_LGBM_GPU"))
    if accelerator == "gpu" and not force_lgbm_gpu and not _local_lightgbm_gpu_probe_usable():
        env.setdefault("USE_LGBM_GPU", "0")
        env.setdefault("KAGGLEBOT_DISABLE_LGBM_GPU", "1")
        notes.append("LightGBM GPU probe failed; forcing CPU LightGBM")
    return notes


def _detect_cuda_oom(text: str) -> bool:
    lowered = text.lower()
    if "out of memory" not in lowered:
        return False
    if "cuda" in lowered:
        return True
    if "cublas_status_alloc_failed" in lowered:
        return True
    if "hiperroroutofmemory" in lowered:
        return True
    if "mps" in lowered and "out of memory" in lowered:
        return True
    return False


def _apply_local_kernel_oom_fallback_env(env: dict[str, str]) -> list[str]:
    notes: list[str] = []
    env["ENABLE_LLM"] = "0"
    env["PIPELINE_NAME"] = "retrieval_only_baseline"
    env["ENABLE_SELF_CONSIST"] = "0"
    env["SAVE_INTERMEDIATE"] = "0"
    notes.append("CUDA OOM detected; retrying with ENABLE_LLM=0 and retrieval_only_baseline")
    return notes


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


_SUBMISSION_KERNEL_TEMPLATE = """\
from __future__ import annotations

import base64
import gzip
import io
import os
from pathlib import Path

# This kernel exists to satisfy notebook-only competitions: it emits a prepared
# `submission.csv` artifact that is already validated locally by Kagglebot.
# Training metrics.json is preserved by the runner; this submit-only wrapper
# must not overwrite it with an unscored placeholder.
#
# NOTE: We still reference `/kaggle/input` to satisfy source validators and to
# make debugging easier in the Kaggle runtime.
KAGGLE_INPUT_ROOT = "/kaggle/input"
SUBMISSION_GZIP_B64 = "__SUBMISSION_GZIP_B64__"


def _resolve_kernel_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path(os.getcwd()).resolve()


def _candidate_sample_paths() -> list[Path]:
    root = Path(os.environ.get("KAGGLEBOT_INPUT_ROOT", KAGGLE_INPUT_ROOT))
    slug = os.environ.get("KAGGLEBOT_COMPETITION_SLUG") or os.environ.get("KAGGLEBOT_SLUG") or ""
    slug_variants = [slug, slug.replace("-", "_")] if slug else []
    candidates: list[Path] = []
    for item in slug_variants:
        if not item:
            continue
        candidates.extend(
            [
                root / item / "sample_submission.csv",
                root / "competitions" / item / "sample_submission.csv",
            ]
        )
    candidates.append(root / "sample_submission.csv")
    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    if root.exists():
        for candidate in sorted(root.rglob("sample_submission.csv")):
            if candidate.is_file() and candidate not in seen:
                ordered.append(candidate)
                seen.add(candidate)
    return ordered


def _find_sample_submission() -> Path | None:
    for candidate in _candidate_sample_paths():
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _read_embedded_submission(payload: bytes):
    import pandas as pd

    return pd.read_csv(io.BytesIO(payload))


def _numeric_frame(frame):
    import pandas as pd

    converted = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        converted[column] = pd.to_numeric(frame[column], errors="coerce")
    return converted


def _looks_like_probability_matrix(values) -> bool:
    if values.empty:
        return False
    row_sums = values.sum(axis=1)
    finite = row_sums.notna() & (row_sums > 0)
    if not finite.any():
        return False
    return bool((row_sums[finite] - 1.0).abs().median() <= 1e-4)


def _normalize_probability_rows(values):
    clipped = values.clip(lower=1e-12)
    row_sums = clipped.sum(axis=1).replace(0, 1.0)
    return clipped.div(row_sums, axis=0)


def _aligned_submission_bytes(payload: bytes) -> bytes:
    sample_path = _find_sample_submission()
    if sample_path is None:
        return payload
    try:
        import pandas as pd

        sample = pd.read_csv(sample_path)
        submission = _read_embedded_submission(payload)
    except Exception as exc:
        print(f"Runtime sample alignment skipped: {exc}")
        return payload

    if sample.empty or len(sample.columns) < 2:
        return payload

    sample_cols = [str(col) for col in sample.columns]
    submission.columns = [str(col) for col in submission.columns]
    id_col = sample_cols[0]
    target_cols = [col for col in sample_cols if col != id_col]
    if id_col not in submission.columns:
        if len(submission) == len(sample) and all(col in submission.columns for col in target_cols):
            out = sample.copy()
            out[target_cols] = submission[target_cols].to_numpy()
            return out.to_csv(index=False).encode("utf-8")
        return payload

    out = sample.copy()
    common_targets = [col for col in target_cols if col in submission.columns]
    if not common_targets:
        return payload

    submission_ids = submission[id_col].astype(str)
    sample_ids = sample[id_col].astype(str)
    sub_targets = submission[common_targets].copy()
    numeric_targets = _numeric_frame(sub_targets)
    all_numeric = bool(numeric_targets.notna().any().all())
    probability_matrix = all_numeric and len(common_targets) > 1 and _looks_like_probability_matrix(numeric_targets)

    if all_numeric:
        fallback = numeric_targets.mean(axis=0).fillna(0.0)
        if probability_matrix:
            total = float(fallback.sum())
            fallback = (fallback.clip(lower=1e-12) / total) if total > 0 else fallback + (1.0 / len(fallback))
        lookup_values = _normalize_probability_rows(numeric_targets) if probability_matrix else numeric_targets
        lookup = {
            key: lookup_values.iloc[idx]
            for idx, key in enumerate(submission_ids)
            if idx < len(lookup_values) and lookup_values.iloc[idx].notna().all()
        }
        aligned_rows = [lookup.get(key, fallback) for key in sample_ids]
        aligned = pd.DataFrame(aligned_rows, columns=common_targets).reset_index(drop=True)
        for col in common_targets:
            out[col] = aligned[col].astype(float).to_numpy()
    else:
        fallback_values = {}
        for col in common_targets:
            non_null = submission[col].dropna()
            if len(non_null):
                fallback_values[col] = non_null.iloc[0]
            else:
                sample_non_null = sample[col].dropna() if col in sample.columns else []
                fallback_values[col] = sample_non_null.iloc[0] if len(sample_non_null) else ""
        lookup = {
            key: submission.loc[idx, common_targets]
            for idx, key in enumerate(submission_ids)
            if idx < len(submission)
        }
        aligned_rows = [lookup.get(key, fallback_values) for key in sample_ids]
        aligned = pd.DataFrame(aligned_rows, columns=common_targets).reset_index(drop=True)
        for col in common_targets:
            out[col] = aligned[col].to_numpy()

    missing_targets = [col for col in target_cols if col not in common_targets]
    if missing_targets:
        print(f"Runtime sample alignment kept sample defaults for missing target columns: {missing_targets}")
    return out[sample_cols].to_csv(index=False).encode("utf-8")


def main() -> None:
    dst = Path(os.environ.get("KAGGLEBOT_WORKING_DIR", "/kaggle/working")) / "submission.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = gzip.decompress(base64.b64decode(SUBMISSION_GZIP_B64.encode("ascii")))
    except Exception as exc:
        raise RuntimeError("Failed to decode embedded submission payload.") from exc
    payload = _aligned_submission_bytes(payload)
    dst.write_bytes(payload)
    print(f\"Wrote {dst} (bytes={dst.stat().st_size})\")


if __name__ == \"__main__\":
    main()
"""


def _render_submission_kernel_script(submission_path: Path) -> str:
    """Render a self-contained submit-only kernel script with embedded submission bytes."""
    submission_bytes = submission_path.read_bytes()
    compressed = gzip.compress(submission_bytes, compresslevel=9)
    payload_b64 = base64.b64encode(compressed).decode("ascii")
    return _SUBMISSION_KERNEL_TEMPLATE.replace("__SUBMISSION_GZIP_B64__", payload_b64)


def _reject_static_tiny_code_competition_submission(
    *,
    slug: str,
    base_dir: Path,
    submission_path: Path,
    tiny_row_limit: int = 10,
) -> None:
    """Fail fast before embedding tiny public-test submissions for code competitions."""
    if _count_csv_data_rows_at_most(submission_path, limit=tiny_row_limit) is not True:
        return
    paths = CompetitionPaths(slug=slug, artifacts_dir=base_dir)
    if not infer_code_competition_from_paths(paths):
        return
    raise KernelFailedError(
        "Refusing to build a static wrapper submit kernel for a code/notebook competition "
        f"with only {tiny_row_limit} or fewer submission rows. "
        "Use notebook submit artifact mode 'inference' so Kaggle reruns the authoritative kernel "
        "against the hidden/full test set."
    )


def _count_csv_data_rows_at_most(path: Path, *, limit: int) -> bool | None:
    if path.suffix.lower() != ".csv":
        return None
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            next(handle, None)
            count = 0
            for line in handle:
                if not line.strip():
                    continue
                count += 1
                if count > limit:
                    return False
            return True
    except OSError:
        return None


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
        _copy_kernel_sources(custom_kernel_dir, kernel_dir)
        _copy_shared_kernel_runtime_modules(kernel_dir)
        _copy_competition_external_assets(base_dir=config.base_dir, slug=config.slug, kernel_dir=kernel_dir)
        _sync_plan_snapshot(
            plan_path=config.base_dir / config.slug / "plan.json",
            targets=[kernel_dir / "plan.json"],
        )
        _ensure_kernel_import_path(kernel_dir)
        _inject_competition_slug_env(kernel_dir, config.slug)
        _inject_hardware_profile_env(
            kernel_dir,
            config.hardware_profile,
            compute="kaggle_gpu" if config.accelerator == "gpu" else "kaggle_tpu",
        )
        _inline_kernel_modules(kernel_dir)
        _inject_data_dir_resolver(kernel_dir)
        _inject_pipeline_cfg_fallback(kernel_dir)
        _inject_column_map_shim(kernel_dir, context_dir)
        _inject_column_fill_shim(kernel_dir, context_dir)
        _inject_object_coerce_shim(kernel_dir, context_dir)
        _inject_device_coerce_shim(kernel_dir, context_dir)
        _inject_training_progress_shim(kernel_dir)
        _inject_transformers_eval_strategy_shim(kernel_dir)
        _prepare_zero_overlap_drift_guard(base_dir=config.base_dir, slug=config.slug, context_dir=context_dir)
        _inject_zero_overlap_drift_shim(kernel_dir, context_dir)
        _inject_competition_slug_env(kernel_dir, config.slug)
        _inject_force_train_env(kernel_dir)
        _ensure_training_progress_shim(kernel_dir)
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
            _copy_kernel_sources(custom_kernel_dir, kernel_dir)
            _copy_shared_kernel_runtime_modules(kernel_dir)
            _copy_competition_external_assets(base_dir=config.base_dir, slug=config.slug, kernel_dir=kernel_dir)
            _sync_plan_snapshot(
                plan_path=config.base_dir / config.slug / "plan.json",
                targets=[kernel_dir / "plan.json"],
            )
            _ensure_kernel_import_path(kernel_dir)
            _inject_competition_slug_env(kernel_dir, config.slug)
            _inline_kernel_modules(kernel_dir)
            _inject_data_dir_resolver(kernel_dir)
            _inject_pipeline_cfg_fallback(kernel_dir)
            _inject_column_map_shim(kernel_dir, context_dir)
            _inject_column_fill_shim(kernel_dir, context_dir)
            _inject_object_coerce_shim(kernel_dir, context_dir)
            _inject_device_coerce_shim(kernel_dir, context_dir)
            _inject_training_progress_shim(kernel_dir)
            _inject_transformers_eval_strategy_shim(kernel_dir)
            _prepare_zero_overlap_drift_guard(base_dir=config.base_dir, slug=config.slug, context_dir=context_dir)
            _inject_zero_overlap_drift_shim(kernel_dir, context_dir)
            _inject_submit_inference_env(kernel_dir)
            _sanitize_submit_inference_output_roots(kernel_dir)
            _validate_inference_submit_kernel(kernel_dir)
            ensure_kernel_sources_valid(kernel_dir)
        else:
            source_config = None
            (kernel_dir / "kernel.py").write_text(
                _render_submission_kernel_script(config.submission_path),
                encoding="utf-8",
            )
            _ensure_kernel_import_path(kernel_dir)
            _inject_competition_slug_env(kernel_dir, config.slug)
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
        _ensure_kernel_competition_slug_env(preparation.kernel_dir, slug)
        if preparation.runtime_bootstrap_mode == "force_train":
            _ensure_kernel_force_train_env(preparation.kernel_dir)
        elif preparation.runtime_bootstrap_mode == "submit_inference":
            _ensure_kernel_submit_inference_env(preparation.kernel_dir)
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


def _extract_invalid_kernel_push_sources(output: str) -> dict[str, list[str]]:
    invalid_sources: dict[str, list[str]] = {}
    if not output:
        return invalid_sources

    for match in _INVALID_KERNEL_SOURCE_RE.finditer(output):
        kind = str(match.group("kind") or "").strip().lower()
        raw_items = str(match.group("items") or "").strip()
        if not kind or not raw_items:
            continue
        try:
            parsed_items = ast.literal_eval(raw_items)
        except (SyntaxError, ValueError):
            parsed_items = []
        if not isinstance(parsed_items, list):
            continue
        cleaned = [str(item).strip() for item in parsed_items if str(item).strip()]
        if cleaned:
            invalid_sources.setdefault(kind, []).extend(cleaned)

    for kind, refs in list(invalid_sources.items()):
        invalid_sources[kind] = list(dict.fromkeys(refs))
    return invalid_sources


def _raise_for_invalid_kernel_push_sources(output: str, *, kernel_dir: Path) -> None:
    invalid_sources = _extract_invalid_kernel_push_sources(output)
    if not invalid_sources:
        return
    details = ", ".join(f"{kind}={','.join(refs)}" for kind, refs in sorted(invalid_sources.items()) if refs)
    raise KernelFailedError(
        "Kaggle kernel push rejected source references: "
        f"{details}. Fix {kernel_dir / 'kernel-metadata.json'} before retrying."
    )


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
    _copy_kernel_sources(kernel_source_dir, kernel_stage_dir)
    _copy_shared_kernel_runtime_modules(kernel_stage_dir)
    _copy_competition_external_assets(base_dir=base_dir, slug=slug, kernel_dir=kernel_stage_dir)
    _sync_plan_snapshot(
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
    _ensure_kernel_import_path(kernel_stage_dir)
    _inject_competition_slug_env(kernel_stage_dir, slug)
    _inject_hardware_profile_env(kernel_stage_dir, hardware_profile, compute="local_gpu")
    _inline_kernel_modules(kernel_stage_dir)
    _inject_data_dir_resolver(kernel_stage_dir)
    _inject_pipeline_cfg_fallback(kernel_stage_dir)
    _inject_column_map_shim(kernel_stage_dir, context_dir)
    _inject_column_fill_shim(kernel_stage_dir, context_dir)
    _inject_object_coerce_shim(kernel_stage_dir, context_dir)
    _inject_device_coerce_shim(kernel_stage_dir, context_dir)
    _inject_kaggle_working_redirect_shim(kernel_stage_dir)
    _inject_lgbm_gpu_guard_shim(kernel_stage_dir)
    _inject_torch_runtime_guard_shim(kernel_stage_dir)
    _inject_training_progress_shim(kernel_stage_dir)
    _inject_transformers_eval_strategy_shim(kernel_stage_dir)
    _prepare_zero_overlap_drift_guard(base_dir=base_dir, slug=slug, context_dir=context_dir)
    _inject_zero_overlap_drift_shim(kernel_stage_dir, context_dir)
    _inject_competition_slug_env(kernel_stage_dir, slug)
    _inject_force_train_env(kernel_stage_dir)
    _ensure_training_progress_shim(kernel_stage_dir)
    ensure_kernel_sources_valid(kernel_stage_dir, require_kaggle_input=False)
    local_aux_env, local_aux_notes = _stage_local_kernel_aux_inputs(
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
    env_notes = _apply_local_runtime_env_defaults(
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
        llm_disabled_by_env = enable_llm_env is not None and not _env_truthy(enable_llm_env)
        if accelerator == "gpu" and not strict_accelerator and not llm_disabled_by_env and _detect_cuda_oom(combined):
            retry_env = env.copy()
            retry_notes = _apply_local_kernel_oom_fallback_env(retry_env)
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
        metrics_dst = _normalize_local_kernel_metrics(
            slug=slug,
            data_dir=base_dir / slug / "data",
            metrics_path=metrics_dst,
            score_source=score_source,
        )
    _enforce_competition_kernel_contract(
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


def _stage_local_kernel_aux_inputs(
    *,
    base_dir: Path,
    slug: str,
    kernel_stage_dir: Path,
) -> tuple[dict[str, str], list[str]]:
    source_config = load_kernel_source_config(base_dir / slug / "plan.json")
    text_runtime = source_config.text_runtime
    if not text_runtime.active and not source_config.domain_adaptation.allow_kernel_finetune:
        return {}, []

    env_updates: dict[str, str] = {}
    notes: list[str] = []
    if source_config.domain_adaptation.allow_kernel_finetune:
        env_updates["KAGGLEBOT_ALLOW_KERNEL_FINETUNE"] = "1"
    if text_runtime.metadata_supervision:
        env_updates["KAGGLEBOT_TEXT_METADATA_SUPERVISION"] = text_runtime.metadata_supervision
    if text_runtime.constraint_rewrite_mode:
        env_updates["KAGGLEBOT_TEXT_CONSTRAINT_REWRITE_MODE"] = text_runtime.constraint_rewrite_mode
    if text_runtime.group_key_columns:
        env_updates["KAGGLEBOT_TEXT_GROUP_KEYS"] = ",".join(text_runtime.group_key_columns)

    if not text_runtime.required_aux_inputs:
        return env_updates, notes

    competition_dir = base_dir / slug
    aux_root = kernel_stage_dir / "aux_inputs"
    staged_relpaths: list[str] = []
    missing: list[str] = []
    for spec in text_runtime.required_aux_inputs:
        resolved = _resolve_required_aux_input(competition_dir=competition_dir, spec=spec)
        if resolved is None:
            missing.append(spec)
            continue
        relpath = _relative_aux_stage_path(competition_dir=competition_dir, source_path=resolved, spec=spec)
        target_path = aux_root / relpath
        _stage_local_path_alias(source_path=resolved, target_path=target_path)
        staged_relpaths.append(relpath.as_posix())

    if missing:
        missing_text = ", ".join(sorted(missing))
        raise KernelFailedError(
            "Required text runtime aux inputs could not be resolved: "
            f"{missing_text}. Checked competition root, data/, and context/."
        )

    env_updates["KAGGLEBOT_AUX_INPUT_ROOT"] = str(aux_root)
    env_updates["KAGGLEBOT_REQUIRED_AUX_INPUTS"] = ",".join(staged_relpaths)
    notes.append(f"staged {len(staged_relpaths)} text aux input(s)")
    return env_updates, notes


def _resolve_required_aux_input(*, competition_dir: Path, spec: str) -> Path | None:
    raw = str(spec).strip().strip("/")
    if not raw:
        return None
    candidates: list[Path] = []
    raw_path = competition_dir / raw
    if "/" in raw or "\\" in raw:
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                competition_dir / "data" / raw,
                competition_dir / "context" / raw,
                competition_dir / raw,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _relative_aux_stage_path(*, competition_dir: Path, source_path: Path, spec: str) -> Path:
    try:
        relative = source_path.resolve().relative_to(competition_dir.resolve())
    except ValueError:
        relative = Path(str(spec).strip().strip("/")).name
    return Path(relative)


def _stage_local_path_alias(*, source_path: Path, target_path: Path) -> None:
    if source_path.is_dir():
        _local_kernel_context.stage_local_data_alias(source_dir=source_path, target_dir=target_path)
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() or target_path.is_symlink():
        if target_path.is_dir() and not target_path.is_symlink():
            shutil.rmtree(target_path, ignore_errors=True)
        else:
            target_path.unlink(missing_ok=True)
    try:
        target_path.symlink_to(source_path)
    except Exception:
        shutil.copy2(source_path, target_path)


def _copy_kernel_sources(source_dir: Path, dest_dir: Path) -> None:
    for path in source_dir.iterdir():
        if path.name in {"output", "outputs", "__pycache__"}:
            continue
        dest_path = dest_dir / path.name
        if path.is_dir():
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(path, dest_path)
        elif path.is_file():
            if path.suffix == ".pyc":
                continue
            shutil.copy2(path, dest_path)


def _copy_competition_external_assets(*, base_dir: Path, slug: str, kernel_dir: Path) -> None:
    external_dir = base_dir / slug / "external"
    if not external_dir.exists():
        return
    for path in external_dir.iterdir():
        if not path.is_file():
            continue
        shutil.copy2(path, kernel_dir / path.name)


def _copy_shared_kernel_runtime_modules(kernel_dir: Path) -> None:
    runtime_dir = Path(__file__).resolve().parent / "kernel_runtime"
    if not runtime_dir.exists():
        return
    for path in sorted(runtime_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        shutil.copy2(path, kernel_dir / path.name)


def _sync_plan_snapshot(*, plan_path: Path, targets: list[Path]) -> None:
    if not plan_path.exists():
        return
    for target in targets:
        if target.resolve() == plan_path.resolve():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plan_path, target)


def _infer_target_column_from_frames(*, train_columns: list[str], test_columns: list[str]) -> str | None:
    test_set = set(test_columns)
    candidates = [col for col in train_columns if col not in test_set]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return candidates[-1]
    for name in ("target", "label", "y"):
        if name in train_columns and name not in test_set:
            return name
    return None


def _is_categorical_like_series(*, series, n_rows: int) -> bool:
    dtype_name = str(getattr(series, "dtype", "")).lower()
    if any(token in dtype_name for token in ("object", "category", "string", "bool")):
        return True
    try:
        nunique = int(series.nunique(dropna=True))
    except Exception:
        return False
    if nunique <= 0:
        return False
    unique_ratio = nunique / max(1, n_rows)
    return unique_ratio <= _ZERO_OVERLAP_DRIFT_MAX_CAT_UNIQUE_RATIO


def _categorical_tvd(*, train_series, test_series) -> float:
    train_values = train_series.fillna("__nan__").astype(str).value_counts(normalize=True)
    test_values = test_series.fillna("__nan__").astype(str).value_counts(normalize=True)
    keys = set(train_values.index) | set(test_values.index)
    if not keys:
        return 0.0
    total_variation = 0.0
    for key in keys:
        total_variation += abs(float(train_values.get(key, 0.0)) - float(test_values.get(key, 0.0)))
    return 0.5 * total_variation


def _abs_corr_with_target(*, feature_series, target_series) -> float:
    try:
        target_numeric = target_series.astype(float)
    except Exception:
        return 0.0
    if target_numeric.nunique(dropna=True) <= 1:
        return 0.0
    dtype_name = str(getattr(feature_series, "dtype", "")).lower()
    try:
        if any(token in dtype_name for token in ("object", "string", "category", "bool")):
            encoded = feature_series.fillna("__nan__").astype(str).factorize()[0]
            encoded_series = target_numeric.__class__(encoded, index=target_numeric.index)
            corr = target_numeric.corr(encoded_series)
        else:
            corr = target_numeric.corr(feature_series.astype(float))
    except Exception:
        return 0.0
    if corr is None:
        return 0.0
    try:
        value = abs(float(corr))
    except Exception:
        return 0.0
    if value != value:
        return 0.0
    return value


def _build_zero_overlap_drift_guard_payload(
    *,
    train_df,
    test_df,
    target_col: str | None,
    id_col: str | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "enabled": False,
        "drop_columns": [],
        "reason": "guard_not_triggered",
        "thresholds": {
            "min_tvd": _ZERO_OVERLAP_DRIFT_MIN_TVD,
            "min_abs_corr": _ZERO_OVERLAP_DRIFT_MIN_ABS_CORR,
            "min_zero_overlap_ratio": _ZERO_OVERLAP_DRIFT_MIN_ZERO_OVERLAP_RATIO,
        },
        "suspects": [],
    }
    if target_col is None or target_col not in train_df.columns:
        payload["reason"] = "missing_target_column"
        return payload

    feature_cols = [col for col in train_df.columns if col != target_col and col in test_df.columns]
    if not feature_cols:
        payload["reason"] = "no_common_feature_columns"
        return payload

    n_rows = int(len(train_df))
    categorical_checked = 0
    zero_overlap_checked = 0
    suspects: list[dict[str, object]] = []
    drop_columns: list[str] = []
    target_series = train_df[target_col]
    for column in feature_cols:
        if id_col is not None and column == id_col:
            continue
        train_series = train_df[column]
        test_series = test_df[column]
        if not _is_categorical_like_series(series=train_series, n_rows=n_rows):
            continue
        categorical_checked += 1
        train_keys = set(train_series.dropna().astype(str).unique().tolist())
        test_keys = set(test_series.dropna().astype(str).unique().tolist())
        if not train_keys or not test_keys:
            continue
        overlap = len(train_keys & test_keys)
        if overlap != 0:
            continue
        zero_overlap_checked += 1
        drift = _categorical_tvd(train_series=train_series, test_series=test_series)
        corr = _abs_corr_with_target(feature_series=train_series, target_series=target_series)
        candidate = {
            "column": column,
            "overlap_unique_count": overlap,
            "train_unique": len(train_keys),
            "test_unique": len(test_keys),
            "drift_tvd": drift,
            "abs_corr_target": corr,
        }
        suspects.append(candidate)
        if drift >= _ZERO_OVERLAP_DRIFT_MIN_TVD and corr >= _ZERO_OVERLAP_DRIFT_MIN_ABS_CORR:
            drop_columns.append(column)

    zero_overlap_ratio = zero_overlap_checked / categorical_checked if categorical_checked > 0 else 0.0
    payload["suspects"] = sorted(
        suspects,
        key=lambda item: float(item.get("drift_tvd", 0.0)) * float(item.get("abs_corr_target", 0.0)),
        reverse=True,
    )
    payload["stats"] = {
        "categorical_checked": categorical_checked,
        "zero_overlap_checked": zero_overlap_checked,
        "zero_overlap_ratio": zero_overlap_ratio,
    }
    payload["drop_columns"] = sorted(set(drop_columns))
    if payload["drop_columns"] and zero_overlap_ratio >= _ZERO_OVERLAP_DRIFT_MIN_ZERO_OVERLAP_RATIO:
        payload["enabled"] = True
        payload["reason"] = "zero_overlap_high_drift_detected"
    return payload


def _prepare_zero_overlap_drift_guard(*, base_dir: Path, slug: str, context_dir: Path) -> Path | None:
    enabled_env = os.getenv("KAGGLEBOT_ENABLE_ZERO_OVERLAP_DRIFT_GUARD")
    if enabled_env is not None and not _env_truthy(enabled_env):
        return None
    try:
        import pandas as pd  # noqa: PLC0415
    except Exception:
        return None

    data_dir = base_dir / slug / "data"
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        return None
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except Exception:
        return None
    if train_df.empty or test_df.empty:
        return None

    target_col, id_col = _local_kernel_context.load_dataset_profile_identity(context_dir=context_dir)
    if target_col is None:
        target_col = _infer_target_column_from_frames(
            train_columns=[str(col) for col in train_df.columns],
            test_columns=[str(col) for col in test_df.columns],
        )
    payload = _build_zero_overlap_drift_guard_payload(
        train_df=train_df,
        test_df=test_df,
        target_col=target_col,
        id_col=id_col,
    )
    payload["target_column"] = target_col
    payload["id_column"] = id_col
    payload["generated_at_epoch"] = int(time.time())

    guard_path = context_dir / _ZERO_OVERLAP_DRIFT_GUARD_FILENAME
    write_json_object(guard_path, payload)
    return guard_path


_KERNEL_BOOTSTRAP_MARKER = "# kagglebot:kernel_sys_path"
_KERNEL_BOOTSTRAP_END = "del _os, _sys, _KROOT, _KWORK"
_KERNEL_DATA_RESOLVER_MARKER = "# kagglebot:data_resolver"
_KERNEL_PIPELINE_CFG_MARKER = "# kagglebot:pipeline_cfg_fallback"
_KERNEL_COMPETITION_SLUG_MARKER = "# kagglebot:competition_slug"
_KERNEL_HARDWARE_PROFILE_MARKER = "# kagglebot:hardware_profile"
_DATA_DIR_JOIN_RE = re.compile(r"(\bdata_dir\s*/\s*)(['\"])([^'\"]+)\2")
_DATA_DIR_REQUIRED_RE = re.compile(r"all\(\(cand\s*/\s*name\)\.exists\(\)\s*for\s*name\s*in\s*required\)")
_DATA_DIR_LOCATE_FALLBACK_MARKER = "# kagglebot:data-dir-fallback-scan"
_DATA_DIR_RAISE_RE = re.compile(
    r"^\s*raise FileNotFoundError\(f\"Could not find required csv files for slug='\{slug\}'\"\)\s*$",
    re.MULTILINE,
)


def _strip_kernel_bootstrap(lines: list[str]) -> list[str]:
    stripped = lines
    while _KERNEL_BOOTSTRAP_MARKER in stripped:
        start = stripped.index(_KERNEL_BOOTSTRAP_MARKER)
        end = None
        search_end = min(start + 60, len(stripped))
        for idx in range(start + 1, search_end):
            if stripped[idx].strip() == _KERNEL_BOOTSTRAP_END:
                end = idx + 1
                break
        if end is None:
            stripped = stripped[:start] + stripped[start + 1 :]
        else:
            stripped = stripped[:start] + stripped[end:]
    return stripped


def _ensure_kernel_import_path(kernel_dir: Path) -> None:
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    bootstrap = (
        f"{_KERNEL_BOOTSTRAP_MARKER}\n"
        "import os as _os\n"
        "import sys as _sys\n"
        "try:\n"
        "    _KROOT = _os.path.dirname(_os.path.abspath(__file__))\n"
        "except NameError:\n"
        "    _KROOT = _os.getcwd()\n"
        "if _KROOT not in _sys.path:\n"
        "    _sys.path.insert(0, _KROOT)\n"
        "_KWORK = '/kaggle/working'\n"
        "if _KWORK not in _sys.path:\n"
        "    _sys.path.insert(0, _KWORK)\n"
        "try:\n"
        "    _KSC = _os.path.join(_KROOT, 'sitecustomize.py')\n"
        "    if _os.path.exists(_KSC):\n"
        "        with open(_KSC, 'rb') as _kb_f:\n"
        "            exec(\n"
        "                compile(_kb_f.read(), _KSC, 'exec'),\n"
        "                {'__file__': _KSC, '__name__': 'kagglebot_sitecustomize'},\n"
        "            )\n"
        "except Exception:\n"
        "    pass\n"
        "del _os, _sys, _KROOT, _KWORK\n"
    )
    lines = _strip_kernel_bootstrap(text.splitlines())
    insert_at = _find_bootstrap_insertion_index(lines)
    bootstrap_lines = bootstrap.splitlines()
    new_lines = lines[:insert_at] + bootstrap_lines + lines[insert_at:]
    new_text = "\n".join(new_lines)
    if text.endswith("\n"):
        new_text += "\n"
    kernel_path.write_text(new_text, encoding="utf-8")


def _inject_data_dir_resolver(kernel_dir: Path) -> None:
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if not _DATA_DIR_JOIN_RE.search(text):
        return
    lines = text.splitlines()
    if _KERNEL_DATA_RESOLVER_MARKER not in text:
        resolver_block = [
            _KERNEL_DATA_RESOLVER_MARKER,
            "from pathlib import Path as _KBPath",
            "",
            "def _kb_find_file(base: _KBPath, name: str) -> _KBPath:",
            "    candidate = base / name",
            "    if candidate.exists():",
            "        return candidate",
            "    try:",
            "        matches = list(base.rglob(name))",
            "    except Exception:",
            "        matches = []",
            "    if matches:",
            "        return matches[0]",
            "    return candidate",
            "",
        ]
        insert_at = _find_bootstrap_block_end(lines)
        if insert_at is None:
            insert_at = _find_bootstrap_insertion_index(lines)
        lines = lines[:insert_at] + resolver_block + lines[insert_at:]
    updated = "\n".join(lines)
    updated = _DATA_DIR_JOIN_RE.sub(r"_kb_find_file(data_dir, '\3')", updated)
    updated = _DATA_DIR_REQUIRED_RE.sub(
        "all(_kb_find_file(cand, name).exists() for name in required)",
        updated,
    )
    if _DATA_DIR_LOCATE_FALLBACK_MARKER not in updated:
        fallback_block = (
            "    input_root = _KBPath('/kaggle/input')\n"
            "    if input_root.exists() and input_root.is_dir():\n"
            f"        {_DATA_DIR_LOCATE_FALLBACK_MARKER}\n"
            "        for cand in sorted(input_root.iterdir(), key=lambda p: p.name):\n"
            "            if not cand.is_dir():\n"
            "                continue\n"
            "            if all(_kb_find_file(cand, name).exists() for name in required):\n"
            "                return cand\n"
            "    raise FileNotFoundError(f\"Could not find required csv files for slug='{slug}'\")"
        )
        updated = _DATA_DIR_RAISE_RE.sub(fallback_block, updated, count=1)
    if text.endswith("\n"):
        updated += "\n"
    kernel_path.write_text(updated, encoding="utf-8")


def _inject_competition_slug_env(kernel_dir: Path, competition_slug: str) -> None:
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if _KERNEL_COMPETITION_SLUG_MARKER in text:
        return

    slug_literal = json.dumps(str(competition_slug))
    resolver_block = [
        _KERNEL_COMPETITION_SLUG_MARKER,
        "import os as _kb_os",
        f"_kb_os.environ['KAGGLEBOT_COMPETITION_SLUG'] = {slug_literal}",
        f"_kb_os.environ['KAGGLEBOT_SLUG'] = {slug_literal}",
        "del _kb_os",
        "",
    ]
    lines = text.splitlines()
    insert_at = _find_bootstrap_block_end(lines)
    if insert_at is None:
        insert_at = _find_bootstrap_insertion_index(lines)
    lines = lines[:insert_at] + resolver_block + lines[insert_at:]
    updated = "\n".join(lines)
    if text.endswith("\n"):
        updated += "\n"
    kernel_path.write_text(updated, encoding="utf-8")


def _inject_hardware_profile_env(kernel_dir: Path, hardware_profile: str | None, *, compute: str) -> None:
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if _KERNEL_HARDWARE_PROFILE_MARKER in text:
        return

    profile = resolve_hardware_profile(hardware_profile, compute=compute)
    env_payload = hardware_env(profile)
    resolver_block = [
        _KERNEL_HARDWARE_PROFILE_MARKER,
        "import os as _kb_os",
    ]
    for key, value in sorted(env_payload.items()):
        resolver_block.append(f"_kb_os.environ.setdefault({json.dumps(key)}, {json.dumps(value)})")
    resolver_block.extend(["del _kb_os", ""])
    lines = text.splitlines()
    insert_at = _find_bootstrap_block_end(lines)
    if insert_at is None:
        insert_at = _find_bootstrap_insertion_index(lines)
    lines = lines[:insert_at] + resolver_block + lines[insert_at:]
    updated = "\n".join(lines)
    if text.endswith("\n"):
        updated += "\n"
    kernel_path.write_text(updated, encoding="utf-8")


def _inject_force_train_env(kernel_dir: Path) -> None:
    """Inject environment bootstrap that keeps training enabled in staged kernels."""
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if _KERNEL_FORCE_TRAIN_MARKER in text:
        return

    resolver_block = [
        _KERNEL_FORCE_TRAIN_MARKER,
        "import os as _kb_os",
        "_kb_os.environ['KAGGLEBOT_DO_TRAIN'] = '1'",
        "_kb_os.environ['KAGGLEBOT_FORCE_TRAIN'] = '1'",
        "del _kb_os",
        "",
    ]
    lines = text.splitlines()
    insert_at = _find_bootstrap_block_end(lines)
    if insert_at is None:
        insert_at = _find_bootstrap_insertion_index(lines)
    lines = lines[:insert_at] + resolver_block + lines[insert_at:]
    updated = "\n".join(lines)
    if text.endswith("\n"):
        updated += "\n"
    kernel_path.write_text(updated, encoding="utf-8")


def _strip_competition_slug_bootstrap(lines: list[str]) -> list[str]:
    stripped = lines
    while _KERNEL_COMPETITION_SLUG_MARKER in stripped:
        start = stripped.index(_KERNEL_COMPETITION_SLUG_MARKER)
        end = None
        search_end = min(start + 12, len(stripped))
        for idx in range(start + 1, search_end):
            if stripped[idx].strip() == "del _kb_os":
                end = idx + 1
                if end < len(stripped) and stripped[end].strip() == "":
                    end += 1
                break
        if end is None:
            stripped = stripped[:start] + stripped[start + 1 :]
        else:
            stripped = stripped[:start] + stripped[end:]
    return stripped


def _strip_force_train_bootstrap(lines: list[str]) -> list[str]:
    """Remove injected force-train bootstrap blocks from kernel text lines."""
    stripped = lines
    while _KERNEL_FORCE_TRAIN_MARKER in stripped:
        start = stripped.index(_KERNEL_FORCE_TRAIN_MARKER)
        end = None
        search_end = min(start + 12, len(stripped))
        for idx in range(start + 1, search_end):
            if stripped[idx].strip() == "del _kb_os":
                end = idx + 1
                if end < len(stripped) and stripped[end].strip() == "":
                    end += 1
                break
        if end is None:
            stripped = stripped[:start] + stripped[start + 1 :]
        else:
            stripped = stripped[:start] + stripped[end:]
    return stripped


def _inject_submit_inference_env(kernel_dir: Path) -> None:
    """Inject environment bootstrap that disables training and forces inference-only submit notebooks."""
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if _KERNEL_SUBMIT_INFERENCE_MARKER in text:
        return

    resolver_block = [
        _KERNEL_SUBMIT_INFERENCE_MARKER,
        "import os as _kb_os",
        "_kb_os.environ['KAGGLEBOT_DO_TRAIN'] = '0'",
        "_kb_os.environ['KAGGLEBOT_FORCE_TRAIN'] = '0'",
        "_kb_os.environ['KAGGLEBOT_DO_INFER'] = '1'",
        "_kb_os.environ['KAGGLEBOT_SUBMIT_NOTEBOOK'] = '1'",
        "_kb_os.environ['KAGGLEBOT_SUBMIT_SKIP_CV'] = '1'",
        "del _kb_os",
        "",
    ]
    lines = _strip_force_train_bootstrap(text.splitlines())
    insert_at = _find_bootstrap_block_end(lines)
    if insert_at is None:
        insert_at = _find_bootstrap_insertion_index(lines)
    lines = lines[:insert_at] + resolver_block + lines[insert_at:]
    updated = "\n".join(lines)
    if text.endswith("\n"):
        updated += "\n"
    kernel_path.write_text(updated, encoding="utf-8")


def _strip_submit_inference_bootstrap(lines: list[str]) -> list[str]:
    stripped = lines
    while _KERNEL_SUBMIT_INFERENCE_MARKER in stripped:
        start = stripped.index(_KERNEL_SUBMIT_INFERENCE_MARKER)
        end = None
        search_end = min(start + 12, len(stripped))
        for idx in range(start + 1, search_end):
            if stripped[idx].strip() == "del _kb_os":
                end = idx + 1
                if end < len(stripped) and stripped[end].strip() == "":
                    end += 1
                break
        if end is None:
            stripped = stripped[:start] + stripped[start + 1 :]
        else:
            stripped = stripped[:start] + stripped[end:]
    return stripped


def _ensure_kernel_competition_slug_env(kernel_dir: Path, competition_slug: str) -> None:
    """Ensure the kernel runtime can resolve the competition slug on Kaggle.

    Kaggle script kernels run from `/kaggle/working`, so naive filesystem-based defaults
    (e.g. using parent directory names) often resolve to "kaggle" instead of the
    competition slug. Inject a tiny env bootstrap into kernel.py so the runtime uses
    the correct slug regardless of working directory layout.
    """
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    slug_literal = json.dumps(str(competition_slug))
    expected_slug_line = f"_kb_os.environ['KAGGLEBOT_COMPETITION_SLUG'] = {slug_literal}"
    expected_alias_line = f"_kb_os.environ['KAGGLEBOT_SLUG'] = {slug_literal}"
    if _KERNEL_COMPETITION_SLUG_MARKER in text and expected_slug_line in text and expected_alias_line in text:
        return
    if _KERNEL_COMPETITION_SLUG_MARKER in text:
        stripped_lines = _strip_competition_slug_bootstrap(text.splitlines())
        stripped_text = "\n".join(stripped_lines)
        if text.endswith("\n"):
            stripped_text += "\n"
        kernel_path.write_text(stripped_text, encoding="utf-8")
    _inject_competition_slug_env(kernel_dir, competition_slug)
    updated = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if (
        _KERNEL_COMPETITION_SLUG_MARKER not in updated
        or expected_slug_line not in updated
        or expected_alias_line not in updated
    ):
        raise KernelFailedError(
            "Failed to inject competition slug bootstrap into kernel.py. "
            "Refusing to push a kernel that may mis-resolve /kaggle/input paths."
        )


def _ensure_kernel_force_train_env(kernel_dir: Path) -> None:
    """Ensure staged kernel runtime has force-train env injection."""
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    expected_train_line = "_kb_os.environ['KAGGLEBOT_DO_TRAIN'] = '1'"
    expected_force_line = "_kb_os.environ['KAGGLEBOT_FORCE_TRAIN'] = '1'"
    if _KERNEL_FORCE_TRAIN_MARKER in text and expected_train_line in text and expected_force_line in text:
        return
    if _KERNEL_FORCE_TRAIN_MARKER in text:
        stripped_lines = _strip_force_train_bootstrap(text.splitlines())
        stripped_text = "\n".join(stripped_lines)
        if text.endswith("\n"):
            stripped_text += "\n"
        kernel_path.write_text(stripped_text, encoding="utf-8")
    _inject_force_train_env(kernel_dir)
    updated = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if (
        _KERNEL_FORCE_TRAIN_MARKER not in updated
        or expected_train_line not in updated
        or expected_force_line not in updated
    ):
        raise KernelFailedError(
            "Failed to inject force-train bootstrap into kernel.py. "
            "Refusing to push a kernel that may auto-disable training."
        )


def _ensure_kernel_submit_inference_env(kernel_dir: Path) -> None:
    """Ensure staged notebook submit kernel disables training and keeps inference on."""
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    expected_train_line = "_kb_os.environ['KAGGLEBOT_DO_TRAIN'] = '0'"
    expected_force_line = "_kb_os.environ['KAGGLEBOT_FORCE_TRAIN'] = '0'"
    expected_infer_line = "_kb_os.environ['KAGGLEBOT_DO_INFER'] = '1'"
    expected_submit_line = "_kb_os.environ['KAGGLEBOT_SUBMIT_NOTEBOOK'] = '1'"
    expected_skip_cv_line = "_kb_os.environ['KAGGLEBOT_SUBMIT_SKIP_CV'] = '1'"
    if (
        _KERNEL_SUBMIT_INFERENCE_MARKER in text
        and expected_train_line in text
        and expected_force_line in text
        and expected_infer_line in text
        and expected_submit_line in text
        and expected_skip_cv_line in text
    ):
        return
    stripped_lines = _strip_submit_inference_bootstrap(_strip_force_train_bootstrap(text.splitlines()))
    stripped_text = "\n".join(stripped_lines)
    if text.endswith("\n"):
        stripped_text += "\n"
    kernel_path.write_text(stripped_text, encoding="utf-8")
    _inject_submit_inference_env(kernel_dir)
    updated = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if (
        _KERNEL_SUBMIT_INFERENCE_MARKER not in updated
        or expected_train_line not in updated
        or expected_force_line not in updated
        or expected_infer_line not in updated
        or expected_submit_line not in updated
        or expected_skip_cv_line not in updated
    ):
        raise KernelFailedError(
            "Failed to inject submit-inference bootstrap into kernel.py. "
            "Refusing to push a notebook submit kernel that may still force training."
        )


def _sanitize_submit_inference_output_roots(kernel_dir: Path) -> None:
    """Rewrite staged submit-kernel output roots from the source tree into `/kaggle/working`."""
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    working_root = "Path('/kaggle/working')"
    updated = text
    patterns = (
        re.compile(r"\b(?:KERNEL_DIR|ARTIFACT_DIR|ARTIFACT_ROOT)\s*/\s*(['\"])(?:output|outputs)\1"),
        re.compile(r"\b(?:KERNEL_DIR|ARTIFACT_DIR|ARTIFACT_ROOT)\.joinpath\(\s*(['\"])(?:output|outputs)\1\s*\)"),
        re.compile(
            r"(^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*)"
            r"(?:KERNEL_DIR|ARTIFACT_DIR|ARTIFACT_ROOT)\s*/\s*(['\"])(?:output|outputs)\2",
            re.MULTILINE,
        ),
        re.compile(
            r"(^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*)"
            r"(?:KERNEL_DIR|ARTIFACT_DIR|ARTIFACT_ROOT)\.joinpath\(\s*(['\"])(?:output|outputs)\2\s*\)",
            re.MULTILINE,
        ),
    )
    for pattern in patterns:
        if pattern.groups >= 2:
            updated = pattern.sub(rf"\1{working_root}", updated)
        else:
            updated = pattern.sub(working_root, updated)
    if updated != text:
        kernel_path.write_text(updated, encoding="utf-8")


def _validate_inference_submit_kernel(kernel_dir: Path) -> None:
    """Reject notebook submit kernels that still look like local wrapper artifacts or write to read-only paths."""
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        raise KernelFailedError("Notebook submit kernel is missing kernel.py.")
    if (kernel_dir / "output").exists():
        raise KernelFailedError(
            "Invalid notebook submit artifact for code competition inference mode: "
            "found staged output directory in notebook package."
        )
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    lowered = text.lower()
    suspicious_fragments = (
        ("submit_only metrics payload", '"kind": "submit_only"'),
        ("embedded submission wrapper payload", "submission_gzip_b64"),
        ("read-only kaggle source output path", "/kaggle/src/output"),
        ("read-only kaggle source outputs path", "/kaggle/src/outputs"),
    )
    for label, fragment in suspicious_fragments:
        if fragment in lowered:
            raise KernelFailedError(
                f"Invalid notebook submit artifact for code competition inference mode: found {label} in staged kernel."
            )
    direct_readonly_patterns = (
        (
            "read-only kaggle source joinpath output path",
            re.compile(r"/kaggle/src['\"]?\s*\)?\s*\.joinpath\(\s*[\"']outputs?[\"']\s*\)", re.IGNORECASE),
        ),
        (
            "staged output root assignment under kaggle source",
            re.compile(
                r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*"
                r"(?:KERNEL_DIR|ARTIFACT_DIR|ARTIFACT_ROOT)\s*(?:/\s*[\"']outputs?[\"']|\.joinpath\(\s*[\"']outputs?[\"']\s*\))",
                re.IGNORECASE | re.MULTILINE,
            ),
        ),
    )
    for label, pattern in direct_readonly_patterns:
        if pattern.search(text):
            raise KernelFailedError(
                f"Invalid notebook submit artifact for code competition inference mode: found {label} in staged kernel."
            )
    readonly_root_patterns = {
        var_name: re.compile(rf"\b{var_name}\s*=.*?/kaggle/src\b", re.IGNORECASE)
        for var_name in ("kernel_dir", "artifact_dir", "artifact_root")
    }
    output_usage_templates = (
        ("output mirror path", r"\b{var}\s*/\s*[\"']outputs?[\"']"),
        ("output mirror joinpath", r"\b{var}\.joinpath\(\s*[\"']outputs?[\"']\s*\)"),
    )
    for var_name, root_pattern in readonly_root_patterns.items():
        if not root_pattern.search(text):
            continue
        for label_suffix, usage_template in output_usage_templates:
            if re.search(usage_template.format(var=var_name), text, re.IGNORECASE):
                raise KernelFailedError(
                    "Invalid notebook submit artifact for code competition inference mode: "
                    f"found staged {var_name} {label_suffix} in staged kernel."
                )
    if "/kaggle/working" not in lowered and "kaggle_working" not in lowered:
        raise KernelFailedError(
            "Invalid notebook submit artifact for code competition inference mode: "
            "kernel does not appear to write outputs under /kaggle/working."
        )


def _inject_pipeline_cfg_fallback(kernel_dir: Path) -> None:
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    if _KERNEL_PIPELINE_CFG_MARKER in text:
        return

    lines = text.splitlines()
    changed = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("raise KeyError("):
            continue
        if "Pipeline not found in plan" not in stripped:
            continue
        indent = line[: len(line) - len(line.lstrip())]
        replacement = [
            f"{indent}{_KERNEL_PIPELINE_CFG_MARKER}",
            f"{indent}return {{",
            f'{indent}    "name": str(name),',
            f'{indent}    "features": [],',
            f'{indent}    "models": [str(name)],',
            f'{indent}    "key_hyperparameters": {{}},',
            f'{indent}    "runtime_memory": "unknown",',
            f'{indent}    "failure_modes": ["missing_pipeline_in_plan"],',
            f'{indent}    "fallbacks": ["use_default_pipeline_behavior"],',
            f"{indent}}}",
        ]
        lines[idx : idx + 1] = replacement
        changed = True
        break
    if not changed:
        return

    updated = "\n".join(lines)
    if text.endswith("\n"):
        updated += "\n"
    kernel_path.write_text(updated, encoding="utf-8")


def _inject_column_map_shim(kernel_dir: Path, context_dir: Path) -> None:
    map_path = context_dir / _COLUMN_MAP_FILENAME
    if not map_path.exists():
        return
    kernel_map_path = kernel_dir / _COLUMN_MAP_FILENAME
    shutil.copy2(map_path, kernel_map_path)
    site_path = kernel_dir / "sitecustomize.py"
    shim = [
        _COLUMN_MAP_SHIM_MARKER,
        "import json",
        "from pathlib import Path",
        "",
        "def _kb_load_map() -> dict:",
        "    candidates = [",
        f"        Path(__file__).with_name('{_COLUMN_MAP_FILENAME}'),",
        f"        Path('/kaggle/working/{_COLUMN_MAP_FILENAME}'),",
        "    ]",
        "    for path in candidates:",
        "        if path.exists():",
        "            try:",
        "                payload = json.loads(path.read_text(encoding='utf-8'))",
        "            except Exception:",
        "                continue",
        "            mapping = payload.get('mapping') if isinstance(payload, dict) else None",
        "            if isinstance(mapping, dict) and mapping:",
        "                return mapping",
        "    return {}",
        "",
        "def _kb_patch_pandas() -> None:",
        "    try:",
        "        import pandas as _pd",
        "    except Exception:",
        "        return",
        "    mapping = _kb_load_map()",
        "    if not mapping:",
        "        return",
        "    _orig = _pd.read_csv",
        "    def _patched(*args, **kwargs):",
        "        df = _orig(*args, **kwargs)",
        "        try:",
        "            return df.rename(columns=mapping)",
        "        except Exception:",
        "            return df",
        "    _pd.read_csv = _patched",
        "",
        "_kb_patch_pandas()",
        "",
    ]
    if site_path.exists():
        text = site_path.read_text(encoding="utf-8", errors="ignore")
        if _COLUMN_MAP_SHIM_MARKER in text:
            return
        site_path.write_text(text.rstrip("\n") + "\n\n" + "\n".join(shim), encoding="utf-8")
        return
    site_path.write_text("\n".join(shim), encoding="utf-8")


def _inject_zero_overlap_drift_shim(kernel_dir: Path, context_dir: Path) -> None:
    guard_path = context_dir / _ZERO_OVERLAP_DRIFT_GUARD_FILENAME
    if not guard_path.exists():
        return
    kernel_guard_path = kernel_dir / _ZERO_OVERLAP_DRIFT_GUARD_FILENAME
    shutil.copy2(guard_path, kernel_guard_path)
    site_path = kernel_dir / "sitecustomize.py"
    shim = [
        _ZERO_OVERLAP_DRIFT_SHIM_MARKER,
        "import json",
        "from pathlib import Path",
        "",
        "def _kb_load_zero_overlap_drift_guard() -> dict:",
        "    candidates = [",
        f"        Path(__file__).with_name('{_ZERO_OVERLAP_DRIFT_GUARD_FILENAME}'),",
        f"        Path('/kaggle/working/{_ZERO_OVERLAP_DRIFT_GUARD_FILENAME}'),",
        "    ]",
        "    for path in candidates:",
        "        if not path.exists():",
        "            continue",
        "        try:",
        "            payload = json.loads(path.read_text(encoding='utf-8'))",
        "        except Exception:",
        "            continue",
        "        if isinstance(payload, dict):",
        "            return payload",
        "    return {}",
        "",
        "def _kb_is_train_or_test_csv(path_value: object) -> bool:",
        "    try:",
        "        name = Path(str(path_value)).name.lower()",
        "    except Exception:",
        "        return False",
        "    return name in {'train.csv', 'test.csv'}",
        "",
        "def _kb_patch_zero_overlap_drift_drop() -> None:",
        "    try:",
        "        import pandas as _pd",
        "    except Exception:",
        "        return",
        "    guard = _kb_load_zero_overlap_drift_guard()",
        "    if not guard or not bool(guard.get('enabled')):",
        "        return",
        "    raw_cols = guard.get('drop_columns')",
        "    if not isinstance(raw_cols, list):",
        "        return",
        "    drop_columns = [str(col) for col in raw_cols if str(col).strip()]",
        "    if not drop_columns:",
        "        return",
        "    _orig = _pd.read_csv",
        "",
        "    def _patched(*args, **kwargs):",
        "        df = _orig(*args, **kwargs)",
        "        path_value = args[0] if args else kwargs.get('filepath_or_buffer')",
        "        if not _kb_is_train_or_test_csv(path_value):",
        "            return df",
        "        try:",
        "            cols = [col for col in drop_columns if col in df.columns]",
        "        except Exception:",
        "            cols = []",
        "        if not cols:",
        "            return df",
        "        try:",
        "            return df.drop(columns=cols)",
        "        except Exception:",
        "            return df",
        "",
        "    _pd.read_csv = _patched",
        "",
        "_kb_patch_zero_overlap_drift_drop()",
        "",
    ]
    if site_path.exists():
        text = site_path.read_text(encoding="utf-8", errors="ignore")
        if _ZERO_OVERLAP_DRIFT_SHIM_MARKER in text:
            return
        site_path.write_text(text.rstrip("\n") + "\n\n" + "\n".join(shim), encoding="utf-8")
        return
    site_path.write_text("\n".join(shim), encoding="utf-8")


def _inject_column_fill_shim(kernel_dir: Path, context_dir: Path) -> None:
    fill_path = context_dir / _COLUMN_FILL_FILENAME
    if not fill_path.exists():
        return
    kernel_fill_path = kernel_dir / _COLUMN_FILL_FILENAME
    shutil.copy2(fill_path, kernel_fill_path)
    site_path = kernel_dir / "sitecustomize.py"
    shim = [
        _COLUMN_FILL_SHIM_MARKER,
        "import json",
        "import re",
        "from pathlib import Path",
        "",
        "def _kb_load_fill() -> dict:",
        "    candidates = [",
        f"        Path(__file__).with_name('{_COLUMN_FILL_FILENAME}'),",
        f"        Path('/kaggle/working/{_COLUMN_FILL_FILENAME}'),",
        "    ]",
        "    for path in candidates:",
        "        if path.exists():",
        "            try:",
        "                payload = json.loads(path.read_text(encoding='utf-8'))",
        "            except Exception:",
        "                continue",
        "            if isinstance(payload, dict):",
        "                return payload",
        "    return {}",
        "",
        "def _kb_missing_columns_for(path_value) -> list[str]:",
        "    payload = _kb_load_fill()",
        "    if not payload:",
        "        return []",
        "    file_map = payload.get('files') if isinstance(payload, dict) else None",
        "    try:",
        "        name = Path(str(path_value)).name",
        "    except Exception:",
        "        name = ''",
        "    if isinstance(file_map, dict) and name in file_map:",
        "        cols = file_map.get(name)",
        "        if isinstance(cols, list):",
        "            return [str(c) for c in cols if str(c).strip()]",
        "    cols = payload.get('missing_columns') if isinstance(payload, dict) else None",
        "    if isinstance(cols, list):",
        "        return [str(c) for c in cols if str(c).strip()]",
        "    return []",
        "",
        "def _kb_global_missing_columns() -> set[str]:",
        "    payload = _kb_load_fill()",
        "    if not payload:",
        "        return set()",
        "    allowed: set[str] = set()",
        "    cols = payload.get('missing_columns') if isinstance(payload, dict) else None",
        "    if isinstance(cols, list):",
        "        for col in cols:",
        "            name = str(col).strip()",
        "            if name:",
        "                allowed.add(name)",
        "    file_map = payload.get('files') if isinstance(payload, dict) else None",
        "    if isinstance(file_map, dict):",
        "        for value in file_map.values():",
        "            if not isinstance(value, list):",
        "                continue",
        "            for col in value:",
        "                name = str(col).strip()",
        "                if name:",
        "                    allowed.add(name)",
        "    return allowed",
        "",
        "def _kb_add_missing_columns(df, columns: list[str]) -> bool:",
        "    added = False",
        "    for col in columns:",
        "        if col in df.columns:",
        "            continue",
        "        try:",
        "            df[col] = float('nan')",
        "            added = True",
        "        except Exception:",
        "            continue",
        "    return added",
        "",
        "def _kb_parse_missing_from_keyerror(exc: Exception) -> list[str]:",
        "    text = str(exc)",
        '    match = re.search(r"\\[([^\\]]+)\\]\\s*not in index", text, flags=re.IGNORECASE)',
        "    if not match:",
        "        return []",
        "    raw = match.group(1).strip()",
        "    if not raw:",
        "        return []",
        "    values = []",
        "    for token in raw.split(','):",
        '        name = token.strip().strip("\'\\"")',
        "        if name:",
        "            values.append(name)",
        "    return values",
        "",
        "def _kb_patch_pandas_fill() -> None:",
        "    try:",
        "        import pandas as _pd",
        "    except Exception:",
        "        return",
        "    _orig = _pd.read_csv",
        "    _orig_getitem = _pd.DataFrame.__getitem__",
        "    def _patched(*args, **kwargs):",
        "        df = _orig(*args, **kwargs)",
        "        try:",
        "            path_value = args[0] if args else kwargs.get('filepath_or_buffer')",
        "            missing_cols = _kb_missing_columns_for(path_value)",
        "            _kb_add_missing_columns(df, missing_cols)",
        "        except Exception:",
        "            return df",
        "        return df",
        "    def _patched_getitem(df, key):",
        "        try:",
        "            return _orig_getitem(df, key)",
        "        except KeyError as exc:",
        "            if not isinstance(key, (list, tuple)):",
        "                raise",
        "            requested = [str(item) for item in key if isinstance(item, str)]",
        "            if not requested:",
        "                raise",
        "            allowed = _kb_global_missing_columns()",
        "            missing = [col for col in requested if col not in df.columns and (not allowed or col in allowed)]",
        "            if not missing:",
        "                parsed = _kb_parse_missing_from_keyerror(exc)",
        "                missing = [col for col in parsed if col in requested and (not allowed or col in allowed)]",
        "            if not missing:",
        "                raise",
        "            if not _kb_add_missing_columns(df, missing):",
        "                raise",
        "            return _orig_getitem(df, key)",
        "    _pd.read_csv = _patched",
        "    _pd.DataFrame.__getitem__ = _patched_getitem",
        "",
        "_kb_patch_pandas_fill()",
        "",
    ]
    if site_path.exists():
        text = site_path.read_text(encoding="utf-8", errors="ignore")
        if _COLUMN_FILL_SHIM_MARKER in text:
            return
        site_path.write_text(text.rstrip("\n") + "\n\n" + "\n".join(shim), encoding="utf-8")
        return
    site_path.write_text("\n".join(shim), encoding="utf-8")


def _inject_object_coerce_shim(kernel_dir: Path, context_dir: Path) -> None:
    coerce_path = context_dir / _OBJECT_COERCE_FILENAME
    if not coerce_path.exists():
        return
    kernel_coerce_path = kernel_dir / _OBJECT_COERCE_FILENAME
    shutil.copy2(coerce_path, kernel_coerce_path)
    site_path = kernel_dir / "sitecustomize.py"
    shim = [
        _OBJECT_COERCE_SHIM_MARKER,
        "import json",
        "from pathlib import Path",
        "",
        "def _kb_object_coerce_enabled() -> bool:",
        "    candidates = [",
        f"        Path(__file__).with_name('{_OBJECT_COERCE_FILENAME}'),",
        f"        Path('/kaggle/working/{_OBJECT_COERCE_FILENAME}'),",
        "    ]",
        "    for path in candidates:",
        "        if path.exists():",
        "            try:",
        "                payload = json.loads(path.read_text(encoding='utf-8'))",
        "            except Exception:",
        "                return True",
        "            if isinstance(payload, dict):",
        "                return bool(payload.get('enabled', True))",
        "            return True",
        "    return False",
        "",
        "def _kb_coerce_ndarray(value):",
        "    try:",
        "        import numpy as _np",
        "    except Exception:",
        "        return value",
        "    if not isinstance(value, _np.ndarray) or value.dtype != object:",
        "        return value",
        "    try:",
        "        return value.astype('float32')",
        "    except Exception:",
        "        try:",
        "            import pandas as _pd",
        "            flat = _pd.to_numeric(value.ravel(), errors='coerce').to_numpy()",
        "            flat = _np.nan_to_num(flat, nan=0.0)",
        "            return flat.reshape(value.shape).astype('float32')",
        "        except Exception:",
        "            try:",
        "                flat = _np.array([0.0 if v is None else v for v in value.ravel()], dtype='float32')",
        "                return flat.reshape(value.shape)",
        "            except Exception:",
        "                return value",
        "",
        "def _kb_patch_torch() -> None:",
        "    if not _kb_object_coerce_enabled():",
        "        return",
        "    try:",
        "        import torch as _torch",
        "    except Exception:",
        "        return",
        "    _orig_tensor = _torch.tensor",
        "    def _tensor(data, *args, **kwargs):",
        "        return _orig_tensor(_kb_coerce_ndarray(data), *args, **kwargs)",
        "    _torch.tensor = _tensor",
        "    try:",
        "        _orig_as_tensor = _torch.as_tensor",
        "    except Exception:",
        "        _orig_as_tensor = None",
        "    if _orig_as_tensor is not None:",
        "        def _as_tensor(data, *args, **kwargs):",
        "            return _orig_as_tensor(_kb_coerce_ndarray(data), *args, **kwargs)",
        "        _torch.as_tensor = _as_tensor",
        "    try:",
        "        _orig_from_numpy = _torch.from_numpy",
        "    except Exception:",
        "        _orig_from_numpy = None",
        "    if _orig_from_numpy is not None:",
        "        def _from_numpy(arr):",
        "            return _orig_from_numpy(_kb_coerce_ndarray(arr))",
        "        _torch.from_numpy = _from_numpy",
        "",
        "_kb_patch_torch()",
        "",
    ]
    if site_path.exists():
        text = site_path.read_text(encoding="utf-8", errors="ignore")
        if _OBJECT_COERCE_SHIM_MARKER in text:
            return
        site_path.write_text(text.rstrip("\n") + "\n\n" + "\n".join(shim), encoding="utf-8")
        return
    site_path.write_text("\n".join(shim), encoding="utf-8")


def _inject_device_coerce_shim(kernel_dir: Path, context_dir: Path) -> None:
    coerce_path = context_dir / _DEVICE_COERCE_FILENAME
    if not coerce_path.exists():
        return
    kernel_coerce_path = kernel_dir / _DEVICE_COERCE_FILENAME
    shutil.copy2(coerce_path, kernel_coerce_path)
    site_path = kernel_dir / "sitecustomize.py"
    shim = [
        _DEVICE_COERCE_SHIM_MARKER,
        "import json",
        "from pathlib import Path",
        "",
        "def _kb_device_coerce_enabled() -> bool:",
        "    candidates = [",
        f"        Path(__file__).with_name('{_DEVICE_COERCE_FILENAME}'),",
        f"        Path('/kaggle/working/{_DEVICE_COERCE_FILENAME}'),",
        "    ]",
        "    for path in candidates:",
        "        if path.exists():",
        "            try:",
        "                payload = json.loads(path.read_text(encoding='utf-8'))",
        "            except Exception:",
        "                return True",
        "            if isinstance(payload, dict):",
        "                return bool(payload.get('enabled', True))",
        "            return True",
        "    return False",
        "",
        "def _kb_default_device():",
        "    try:",
        "        import torch as _torch",
        "    except Exception:",
        "        return None",
        "    if _torch.cuda.is_available():",
        "        return _torch.device('cuda')",
        "    return None",
        "",
        "def _kb_patch_torch_device() -> None:",
        "    if not _kb_device_coerce_enabled():",
        "        return",
        "    try:",
        "        import torch as _torch",
        "    except Exception:",
        "        return",
        "    device = _kb_default_device()",
        "    if device is None:",
        "        return",
        "    def _wrap_factory(fn):",
        "        def _wrapped(*args, **kwargs):",
        "            if 'device' not in kwargs:",
        "                kwargs['device'] = device",
        "            return fn(*args, **kwargs)",
        "        return _wrapped",
        "    factories = (",
        "        'tensor', 'as_tensor', 'from_numpy', 'zeros', 'ones', 'full', 'rand',",
        "        'randn', 'arange', 'zeros_like', 'ones_like', 'full_like',",
        "    )",
        "    for name in factories:",
        "        fn = getattr(_torch, name, None)",
        "        if fn is None:",
        "            continue",
        "        if name == 'from_numpy':",
        "            def _from_numpy(arr, _fn=fn):",
        "                out = _fn(arr)",
        "                try:",
        "                    return out.to(device)",
        "                except Exception:",
        "                    return out",
        "            setattr(_torch, name, _from_numpy)",
        "        else:",
        "            setattr(_torch, name, _wrap_factory(fn))",
        "",
        "    _orig_setattr = _torch.nn.Module.__setattr__",
        "    def _module_setattr(self, name, value):",
        "        if isinstance(value, _torch.Tensor):",
        "            try:",
        "                if value.device.type == 'cpu':",
        "                    value = value.to(device)",
        "            except Exception:",
        "                pass",
        "        return _orig_setattr(self, name, value)",
        "    _torch.nn.Module.__setattr__ = _module_setattr",
        "",
        "_kb_patch_torch_device()",
        "",
    ]
    if site_path.exists():
        text = site_path.read_text(encoding="utf-8", errors="ignore")
        if _DEVICE_COERCE_SHIM_MARKER in text:
            return
        site_path.write_text(text.rstrip("\n") + "\n\n" + "\n".join(shim), encoding="utf-8")
        return
    site_path.write_text("\n".join(shim), encoding="utf-8")


def _inject_kaggle_working_redirect_shim(kernel_dir: Path) -> None:
    site_path = kernel_dir / "sitecustomize.py"
    shim = [
        _KAGGLE_WORKING_REDIRECT_SHIM_MARKER,
        "import builtins",
        "import io",
        "import os",
        "from pathlib import Path",
        "",
        "def _kb_local_kernel_mode() -> bool:",
        "    value = str(os.environ.get('KAGGLEBOT_LOCAL_KERNEL', '0')).strip().lower()",
        "    return value in {'1', 'true', 'yes', 'on'}",
        "",
        "def _kb_redirect_root() -> Path | None:",
        "    root = str(os.environ.get('KAGGLEBOT_LOCAL_WORKING_DIR', '')).strip()",
        "    if not root:",
        "        return None",
        "    return Path(root)",
        "",
        "def _kb_remap_path(path_value):",
        "    try:",
        "        raw = os.fspath(path_value)",
        "    except Exception:",
        "        return path_value",
        "    if not isinstance(raw, str):",
        "        return path_value",
        "    if raw == '/kaggle/working':",
        "        root = _kb_redirect_root()",
        "        return str(root) if root is not None else path_value",
        "    if raw.startswith('/kaggle/working/'):",
        "        root = _kb_redirect_root()",
        "        if root is None:",
        "            return path_value",
        "        suffix = raw[len('/kaggle/working/'):].lstrip('/')",
        "        return str(root / suffix)",
        "    return path_value",
        "",
        "def _kb_prepare_parent(path_value, mode: str) -> None:",
        "    if not any(flag in mode for flag in ('w', 'a', 'x', '+')):",
        "        return",
        "    try:",
        "        parent = Path(os.fspath(path_value)).parent",
        "        parent.mkdir(parents=True, exist_ok=True)",
        "    except Exception:",
        "        return",
        "",
        "def _kb_patch_open_redirect() -> None:",
        "    if not _kb_local_kernel_mode():",
        "        return",
        "    _orig_builtin_open = builtins.open",
        "    _orig_io_open = io.open",
        "",
        "    def _open_builtin(file, mode='r', *args, **kwargs):",
        "        mapped = _kb_remap_path(file)",
        "        _kb_prepare_parent(mapped, mode)",
        "        return _orig_builtin_open(mapped, mode, *args, **kwargs)",
        "",
        "    def _open_io(file, mode='r', *args, **kwargs):",
        "        mapped = _kb_remap_path(file)",
        "        _kb_prepare_parent(mapped, mode)",
        "        return _orig_io_open(mapped, mode, *args, **kwargs)",
        "",
        "    builtins.open = _open_builtin",
        "    io.open = _open_io",
        "",
        "_kb_patch_open_redirect()",
        "",
    ]
    if site_path.exists():
        text = site_path.read_text(encoding="utf-8", errors="ignore")
        if _KAGGLE_WORKING_REDIRECT_SHIM_MARKER in text:
            return
        site_path.write_text(text.rstrip("\n") + "\n\n" + "\n".join(shim), encoding="utf-8")
        return
    site_path.write_text("\n".join(shim), encoding="utf-8")


def _inject_lgbm_gpu_guard_shim(kernel_dir: Path) -> None:
    site_path = kernel_dir / "sitecustomize.py"
    shim = [
        _LGBM_GPU_GUARD_SHIM_MARKER,
        "import os",
        "",
        "def _kb_disable_lgbm_gpu_enabled() -> bool:",
        "    value = str(os.environ.get('KAGGLEBOT_DISABLE_LGBM_GPU', '0')).strip().lower()",
        "    return value in {'1', 'true', 'yes', 'on'}",
        "",
        "def _kb_patch_lgbm_gpu_guard() -> None:",
        "    if not _kb_disable_lgbm_gpu_enabled():",
        "        return",
        "    try:",
        "        import lightgbm as _lgb",
        "    except Exception:",
        "        return",
        "",
        "    def _force_cpu(estimator) -> None:",
        "        for key in ('device', 'device_type'):",
        "            try:",
        "                estimator.set_params(**{key: 'cpu'})",
        "            except Exception:",
        "                continue",
        "",
        "    targets = ('LGBMModel', 'LGBMRegressor', 'LGBMClassifier', 'LGBMRanker')",
        "    for cls_name in targets:",
        "        cls = getattr(_lgb, cls_name, None)",
        "        if cls is None:",
        "            continue",
        "        fit = getattr(cls, 'fit', None)",
        "        if fit is None or not callable(fit) or getattr(fit, '__kb_lgbm_cpu_wrapped__', False):",
        "            continue",
        "        def _wrapped(self, *args, _fit=fit, **kwargs):",
        "            _force_cpu(self)",
        "            return _fit(self, *args, **kwargs)",
        "        _wrapped.__kb_lgbm_cpu_wrapped__ = True",
        "        setattr(cls, 'fit', _wrapped)",
        "",
        "    train_fn = getattr(_lgb, 'train', None)",
        "    if callable(train_fn) and not getattr(train_fn, '__kb_lgbm_cpu_wrapped__', False):",
        "        def _train(params, *args, _train=train_fn, **kwargs):",
        "            if isinstance(params, dict):",
        "                updated = dict(params)",
        "                updated['device'] = 'cpu'",
        "                updated['device_type'] = 'cpu'",
        "                params = updated",
        "            return _train(params, *args, **kwargs)",
        "        _train.__kb_lgbm_cpu_wrapped__ = True",
        "        _lgb.train = _train",
        "",
        "_kb_patch_lgbm_gpu_guard()",
        "",
    ]
    if site_path.exists():
        text = site_path.read_text(encoding="utf-8", errors="ignore")
        if _LGBM_GPU_GUARD_SHIM_MARKER in text:
            return
        site_path.write_text(text.rstrip("\n") + "\n\n" + "\n".join(shim), encoding="utf-8")
        return
    site_path.write_text("\n".join(shim), encoding="utf-8")


def _inject_torch_runtime_guard_shim(kernel_dir: Path) -> None:
    site_path = kernel_dir / "sitecustomize.py"
    shim = [
        _TORCH_RUNTIME_GUARD_SHIM_MARKER,
        "import os",
        "",
        "def _kb_local_kernel_mode() -> bool:",
        "    value = str(os.environ.get('KAGGLEBOT_LOCAL_KERNEL', '0')).strip().lower()",
        "    return value in {'1', 'true', 'yes', 'on'}",
        "",
        "def _kb_patch_torch_runtime_guard() -> None:",
        "    if not _kb_local_kernel_mode():",
        "        return",
        "    target_nofile = str(os.environ.get('KAGGLEBOT_LOCAL_NOFILE', '')).strip()",
        "    if target_nofile:",
        "        try:",
        "            import resource",
        "            desired = max(256, int(target_nofile))",
        "            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)",
        "            hard_cap = desired if hard is None or int(hard) < 0 else int(hard)",
        "            new_soft = min(max(int(soft), desired), hard_cap)",
        "            if new_soft > int(soft):",
        "                resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))",
        "        except Exception:",
        "            pass",
        "    strategy = str(os.environ.get('KAGGLEBOT_TORCH_SHARING_STRATEGY', '')).strip()",
        "    if strategy:",
        "        try:",
        "            import torch.multiprocessing as _kb_tmp",
        "            getter = getattr(_kb_tmp, 'get_sharing_strategy', None)",
        "            current = getter() if callable(getter) else None",
        "            if current != strategy:",
        "                _kb_tmp.set_sharing_strategy(strategy)",
        "        except Exception:",
        "            pass",
        "",
        "_kb_patch_torch_runtime_guard()",
        "",
    ]
    if site_path.exists():
        text = site_path.read_text(encoding="utf-8", errors="ignore")
        if _TORCH_RUNTIME_GUARD_SHIM_MARKER in text:
            return
        site_path.write_text(text.rstrip("\n") + "\n\n" + "\n".join(shim), encoding="utf-8")
        return
    site_path.write_text("\n".join(shim), encoding="utf-8")


def _inject_training_progress_shim(kernel_dir: Path) -> None:
    site_path = kernel_dir / "sitecustomize.py"
    shim = (
        (
            f"""
{_TRAIN_PROGRESS_SHIM_MARKER}
import importlib
import os
import threading
import time

_KB_PROGRESS = {{
    "started_at": time.monotonic(),
    "last_event_at": time.monotonic(),
    "watchdog_started": False,
}}

def _kb_progress_enabled() -> bool:
    value = str(os.environ.get("KAGGLEBOT_TRAIN_PROGRESS", "1")).strip().lower()
    return value not in {{"0", "false", "off", "no"}}

def _kb_int_env(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except Exception:
        value = default
    return max(minimum, value)

def _kb_float_env(name: str, default: float, minimum: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except Exception:
        value = default
    return max(minimum, value)

def _kb_emit(msg: str) -> None:
    _KB_PROGRESS["last_event_at"] = time.monotonic()
    print(f"[kernel] {{msg}}", flush=True)

def _kb_get_shape(args):
    if not args:
        return None, None
    x = args[0]
    rows = None
    cols = None
    try:
        rows = int(len(x))
    except Exception:
        rows = None
    try:
        shape = getattr(x, "shape", None)
        if shape is not None and len(shape) >= 2:
            cols = int(shape[1])
    except Exception:
        cols = None
    return rows, cols

def _kb_estimator_iter_budget(estimator) -> int | None:
    params = {{}}
    try:
        params = estimator.get_params(deep=False)
    except Exception:
        params = {{}}
    for key in ("iterations", "n_estimators", "max_iter", "num_iterations"):
        value = params.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None

def _kb_resolve_boosting_log_every(estimator) -> int:
    forced = _kb_int_env("KAGGLEBOT_BOOSTING_LOG_EVERY", 0, 0)
    if forced > 0:
        return forced
    budget = _kb_estimator_iter_budget(estimator)
    if budget is None:
        return 100
    # Target around 20-30 evaluation points across a full fit.
    period = max(1, budget // 25)
    return min(max(period, 10), 200)

def _kb_choose_fit_tick_interval(label: str, rows: int | None) -> float:
    base = _kb_float_env("KAGGLEBOT_MODEL_PROGRESS_INTERVAL_SEC", 12.0, 5.0)
    if label in {{"catboost", "lightgbm", "xgboost"}}:
        # Boosting models also emit iteration logs; keep timer sparse.
        return max(base, 30.0)
    if rows is None:
        return base
    if rows >= 200000:
        return max(base, 30.0)
    if rows >= 50000:
        return max(base, 20.0)
    if rows >= 10000:
        return max(base, 12.0)
    return base

def _kb_start_watchdog_thread() -> None:
    if not _kb_progress_enabled():
        return
    if bool(_KB_PROGRESS.get("watchdog_started", False)):
        return
    _KB_PROGRESS["watchdog_started"] = True
    silence_sec = _kb_float_env("KAGGLEBOT_PROGRESS_INTERVAL_SEC", 45.0, 10.0)
    poll_sec = max(1.0, min(5.0, silence_sec / 6.0))
    def _run():
        while True:
            time.sleep(poll_sec)
            now = time.monotonic()
            last = float(_KB_PROGRESS.get("last_event_at", now))
            if now - last < silence_sec:
                continue
            elapsed = int(max(0.0, now - float(_KB_PROGRESS.get("started_at", now))))
            quiet = int(max(0.0, now - last))
            _kb_emit(f"train watchdog: elapsed={{elapsed}}s no_new_logs_for={{quiet}}s")
    t = threading.Thread(target=_run, daemon=True, name="kb-train-watchdog")
    t.start()

def _kb_wrap_splitter(module_name: str, class_name: str) -> None:
    if not _kb_progress_enabled():
        return
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return
    cls = getattr(mod, class_name, None)
    if cls is None:
        return
    split = getattr(cls, "split", None)
    if split is None or not callable(split):
        return
    if getattr(split, "__kb_progress_wrapped__", False):
        return
    def _wrapped(self, *args, **kwargs):
        iterator = split(self, *args, **kwargs)
        total = getattr(self, "n_splits", None)
        idx = 0
        for item in iterator:
            idx += 1
            train_n = "?"
            valid_n = "?"
            if isinstance(item, tuple) and len(item) >= 2:
                try:
                    train_n = str(len(item[0]))
                except Exception:
                    pass
                try:
                    valid_n = str(len(item[1]))
                except Exception:
                    pass
            fold_part = f"{{idx}}/{{total}}" if isinstance(total, int) and total > 0 else str(idx)
            _kb_emit(
                f"cv fold start: splitter={{class_name}} fold={{fold_part}} train={{train_n}} valid={{valid_n}}"
            )
            yield item
        if idx > 0:
            _kb_emit(f"cv split done: splitter={{class_name}} folds={{idx}}")
    _wrapped.__kb_progress_wrapped__ = True
    setattr(cls, "split", _wrapped)

def _kb_wrap_fit(module_name: str, class_name: str, label: str) -> None:
    if not _kb_progress_enabled():
        return
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return
    cls = getattr(mod, class_name, None)
    if cls is None:
        return
    fit = getattr(cls, "fit", None)
    if fit is None or not callable(fit):
        return
    if getattr(fit, "__kb_progress_wrapped__", False):
        return
    def _wrapped(self, *args, **kwargs):
        model_name = self.__class__.__name__
        rows, cols = _kb_get_shape(args)
        iter_budget = _kb_estimator_iter_budget(self)
        log_every = None
        if label in {{"catboost", "lightgbm", "xgboost"}}:
            log_every = _kb_resolve_boosting_log_every(self)
        summary = [f"train start: model={{label}}.{{model_name}}"]
        if rows is not None:
            summary.append(f"rows={{rows}}")
        if cols is not None:
            summary.append(f"cols={{cols}}")
        if iter_budget is not None:
            summary.append(f"iter_budget={{iter_budget}}")
        if log_every is not None:
            summary.append(f"log_every={{log_every}}")
        _kb_emit(" ".join(summary))
        try:
            if label == "lightgbm":
                import lightgbm as _lgb
                callbacks = list(kwargs.get("callbacks") or [])
                callbacks.append(_lgb.log_evaluation(period=log_every))
                kwargs["callbacks"] = callbacks
            elif label == "xgboost":
                if kwargs.get("eval_set"):
                    kwargs["verbose"] = log_every
            elif label == "catboost":
                try:
                    self.set_params(verbose=log_every)
                except Exception:
                    pass
                kwargs.setdefault("verbose", log_every)
        except Exception:
            pass
        started = time.monotonic()
        interval = _kb_choose_fit_tick_interval(label, rows)
        stop = threading.Event()
        def _ticker():
            while not stop.wait(interval):
                elapsed = int(max(0.0, time.monotonic() - started))
                _kb_emit(f"train running: model={{label}}.{{model_name}} elapsed={{elapsed}}s")
        thread = threading.Thread(target=_ticker, daemon=True, name=f"kb-fit-{{label}}")
        thread.start()
        try:
            return fit(self, *args, **kwargs)
        finally:
            stop.set()
            thread.join(timeout=0.2)
            elapsed = int(max(0.0, time.monotonic() - started))
            _kb_emit(f"train done: model={{label}}.{{model_name}} elapsed={{elapsed}}s")
    _wrapped.__kb_progress_wrapped__ = True
    setattr(cls, "fit", _wrapped)

def _kb_patch_training_progress() -> None:
    if not _kb_progress_enabled():
        return
    _kb_start_watchdog_thread()
    splitters = [
        ("sklearn.model_selection", "KFold"),
        ("sklearn.model_selection", "StratifiedKFold"),
        ("sklearn.model_selection", "GroupKFold"),
        ("sklearn.model_selection", "TimeSeriesSplit"),
    ]
    for module_name, class_name in splitters:
        _kb_wrap_splitter(module_name, class_name)
    targets = [
        ("catboost", "CatBoostRegressor", "catboost"),
        ("lightgbm", "LGBMRegressor", "lightgbm"),
        ("xgboost", "XGBRegressor", "xgboost"),
        ("sklearn.ensemble", "HistGradientBoostingRegressor", "sklearn"),
        ("sklearn.linear_model", "ElasticNet", "sklearn"),
        ("sklearn.linear_model", "Ridge", "sklearn"),
        ("sklearn.linear_model", "SGDRegressor", "sklearn"),
        ("sklearn.kernel_ridge", "KernelRidge", "sklearn"),
    ]
    for module_name, class_name, label in targets:
        _kb_wrap_fit(module_name, class_name, label)

_kb_patch_training_progress()
"""
        )
        .strip("\n")
        .splitlines()
    )
    if site_path.exists():
        text = site_path.read_text(encoding="utf-8", errors="ignore")
        if _TRAIN_PROGRESS_SHIM_MARKER in text:
            return
        site_path.write_text(text.rstrip("\n") + "\n\n" + "\n".join(shim), encoding="utf-8")
        return
    site_path.write_text("\n".join(shim), encoding="utf-8")


def _inject_transformers_eval_strategy_shim(kernel_dir: Path) -> None:
    """Patch transformers API drift for Seq2SeqTrainingArguments eval strategy naming."""
    site_path = kernel_dir / "sitecustomize.py"
    shim = [
        _TRANSFORMERS_EVAL_STRATEGY_SHIM_MARKER,
        "import inspect",
        "",
        "def _kb_patch_transformers_eval_strategy_alias() -> None:",
        "    try:",
        "        import transformers as _tf",
        "    except Exception:",
        "        return",
        "    args_cls = getattr(_tf, 'Seq2SeqTrainingArguments', None)",
        "    if args_cls is None:",
        "        return",
        "    try:",
        "        params = inspect.signature(args_cls.__init__).parameters",
        "    except Exception:",
        "        return",
        "    if 'evaluation_strategy' in params:",
        "        return",
        "    if 'eval_strategy' not in params:",
        "        return",
        "    _orig_init = args_cls.__init__",
        "    def _patched_init(self, *args, **kwargs):",
        "        if 'evaluation_strategy' in kwargs and 'eval_strategy' not in kwargs:",
        "            kwargs['eval_strategy'] = kwargs.pop('evaluation_strategy')",
        "        return _orig_init(self, *args, **kwargs)",
        "    args_cls.__init__ = _patched_init",
        "",
        "_kb_patch_transformers_eval_strategy_alias()",
        "",
    ]
    if site_path.exists():
        text = site_path.read_text(encoding="utf-8", errors="ignore")
        if _TRANSFORMERS_EVAL_STRATEGY_SHIM_MARKER in text:
            return
        site_path.write_text(text.rstrip("\n") + "\n\n" + "\n".join(shim), encoding="utf-8")
        return
    site_path.write_text("\n".join(shim), encoding="utf-8")


def _ensure_training_progress_shim(kernel_dir: Path) -> None:
    site_path = kernel_dir / "sitecustomize.py"
    if not site_path.exists():
        raise KernelFailedError(
            f"Training progress shim missing: {site_path}. Refusing to run a kernel without mandatory progress logging."
        )
    text = site_path.read_text(encoding="utf-8", errors="ignore")
    if _TRAIN_PROGRESS_SHIM_MARKER not in text:
        raise KernelFailedError(
            f"Training progress shim marker not found in {site_path}. "
            "Refusing to run a kernel without mandatory progress logging."
        )


def _find_bootstrap_block_end(lines: list[str]) -> int | None:
    if _KERNEL_BOOTSTRAP_MARKER not in lines:
        return None
    start = lines.index(_KERNEL_BOOTSTRAP_MARKER)
    search_end = min(start + 30, len(lines))
    for idx in range(start + 1, search_end):
        if lines[idx].strip() == _KERNEL_BOOTSTRAP_END:
            return idx + 1
    return None


def _find_bootstrap_insertion_index(lines: list[str]) -> int:
    idx = 0
    if idx < len(lines) and lines[idx].startswith("#!"):
        idx += 1
    for _ in range(2):
        if idx < len(lines) and re.match(r"^#.*coding[:=]\s*[-\w.]+", lines[idx]):
            idx += 1
    while idx < len(lines) and (lines[idx].strip() == "" or lines[idx].lstrip().startswith("#")):
        idx += 1
    if idx < len(lines):
        stripped = lines[idx].lstrip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote = '"""' if stripped.startswith('"""') else "'''"
            if stripped.count(quote) >= 2:
                idx += 1
            else:
                idx += 1
                while idx < len(lines) and quote not in lines[idx]:
                    idx += 1
                if idx < len(lines):
                    idx += 1
    while idx < len(lines) and (lines[idx].strip() == "" or lines[idx].lstrip().startswith("#")):
        idx += 1
    while idx < len(lines) and re.match(r"^\s*from\s+__future__\s+import\s+", lines[idx]):
        idx += 1
    return idx


def _inline_kernel_modules(kernel_dir: Path, modules: tuple[str, ...] | None = None) -> None:
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return
    text = kernel_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    if modules is None:
        modules = _discover_inline_modules(kernel_dir, lines)
    if not modules or not _kernel_imports_local_modules(lines, modules):
        return
    alias_modules = _modules_with_alias_imports(lines, modules)
    if alias_modules:
        modules = tuple(module for module in modules if module not in alias_modules)
        if not modules:
            return

    stripped = lines
    for module in modules:
        stripped = _strip_module_import(stripped, module)

    module_blocks: list[str] = []
    for module in modules:
        module_path = kernel_dir / f"{module}.py"
        if not module_path.exists():
            continue
        module_lines = module_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        cleaned = _strip_module_headers(module_lines)
        cleaned = _strip_local_module_imports(cleaned, modules)
        if not cleaned:
            continue
        module_blocks.append(f"# --- Begin inlined module: {module}.py ---")
        module_blocks.extend(cleaned)
        module_blocks.append(f"# --- End inlined module: {module}.py ---")

    if not module_blocks:
        return

    insert_at = _find_main_guard_index(stripped)
    new_lines = stripped[:insert_at] + [""] + module_blocks + [""] + stripped[insert_at:]
    new_text = "\n".join(new_lines)
    if text.endswith("\n"):
        new_text += "\n"
    kernel_path.write_text(new_text, encoding="utf-8")


def _kernel_imports_local_modules(lines: list[str], modules: tuple[str, ...]) -> bool:
    for line in lines:
        for module in modules:
            if re.match(rf"^\s*from\s+\.?{re.escape(module)}\s+import\b", line):
                return True
            if re.match(rf"^\s*import\s+{re.escape(module)}\b", line):
                return True
    return False


def _modules_with_alias_imports(lines: list[str], modules: tuple[str, ...]) -> set[str]:
    if not modules:
        return set()
    text = "\n".join(lines)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _modules_with_alias_imports_fallback(lines, modules)

    alias_modules: set[str] = set()
    module_set = set(modules)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base = alias.name.split(".", 1)[0]
                if base in module_set and alias.asname:
                    alias_modules.add(base)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            base = node.module.split(".", 1)[0]
            if base not in module_set:
                continue
            for alias in node.names:
                if alias.asname:
                    alias_modules.add(base)
                    break
    return alias_modules


def _modules_with_alias_imports_fallback(lines: list[str], modules: tuple[str, ...]) -> set[str]:
    alias_modules: set[str] = set()
    for line in lines:
        for module in modules:
            if re.match(rf"^\s*import\s+{re.escape(module)}\s+as\s+\w+", line):
                alias_modules.add(module)
                continue
            if re.match(rf"^\s*from\s+\.?{re.escape(module)}\s+import\b", line):
                if " as " in line:
                    alias_modules.add(module)
    return alias_modules


def _strip_module_import(lines: list[str], module: str) -> list[str]:
    output: list[str] = []
    skipping = False
    paren_depth = 0
    for line in lines:
        if not skipping:
            if re.match(rf"^\s*from\s+\.?{re.escape(module)}\s+import\b", line):
                skipping = True
                paren_depth = line.count("(") - line.count(")")
                if paren_depth <= 0 and not line.rstrip().endswith("\\"):
                    skipping = False
                continue
            if re.match(rf"^\s*import\s+{re.escape(module)}\b", line):
                continue
            output.append(line)
            continue
        paren_depth += line.count("(") - line.count(")")
        if paren_depth <= 0 and not line.rstrip().endswith("\\"):
            skipping = False
        continue
    return output


def _discover_inline_modules(kernel_dir: Path, lines: list[str]) -> tuple[str, ...]:
    module_names: list[str] = []
    for path in kernel_dir.glob("*.py"):
        if path.name == "kernel.py":
            continue
        name = path.stem
        if name.isidentifier():
            module_names.append(name)
    if not module_names:
        return ()
    used: list[str] = []
    for name in module_names:
        if _kernel_imports_local_modules(lines, (name,)):
            used.append(name)
    return tuple(used)


def _strip_module_headers(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        if not cleaned and line.startswith("#!"):
            continue
        if not cleaned and re.match(r"^#.*coding[:=]\s*[-\w.]+", line):
            continue
        if re.match(r"^\s*from\s+__future__\s+import\s+", line):
            continue
        cleaned.append(line)
    while cleaned and cleaned[0].strip() == "":
        cleaned.pop(0)
    return cleaned


def _strip_local_module_imports(lines: list[str], modules: tuple[str, ...]) -> list[str]:
    cleaned = lines
    for module in modules:
        cleaned = _strip_module_import(cleaned, module)
    return cleaned


def _find_main_guard_index(lines: list[str]) -> int:
    for idx, line in enumerate(lines):
        if re.match(r"^\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", line):
            return idx
    return len(lines)


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
