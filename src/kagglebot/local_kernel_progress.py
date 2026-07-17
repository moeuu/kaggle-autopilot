from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich import print

from kagglebot.json_utils import load_json_object
from kagglebot.kernel_progress import (
    extract_catboost_fallback_reason_from_line,
    extract_pipeline_done_from_line,
    extract_pipeline_start_from_line,
    extract_pipeline_suite_from_line,
    extract_train_model_start_from_line,
    extract_training_stage_from_line,
    resolve_fold_current,
    resolve_seed_current,
)
from kagglebot.logging_utils import truncate_lines

LOCAL_KERNEL_STARTUP_STALL_GRACE_SEC = 30.0
_LOCAL_KERNEL_CONTROL_FILENAMES = frozenset({"local_launch_manifest.json"})


@dataclass
class LocalKernelProgressTracker:
    """Track local-kernel textual and artifact-level progress signals."""

    expected_folds: int | None
    expected_seeds: list[int]
    watch_dirs: tuple[Path, ...] = field(default_factory=tuple)
    started_at_monotonic: float = field(default_factory=time.monotonic)
    started_at_wall: float = field(default_factory=time.time)
    zero_based_folds: bool = False
    seen_triplets: set[tuple[str, int, int]] = field(default_factory=set)
    lines_seen: int = 0
    last_output_monotonic: float | None = None
    current_pipeline: str | None = None
    current_suite: str | None = None
    current_model: str | None = None
    last_fallback_reason: str | None = None
    completed_pipelines: set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def observe_output_activity(self) -> None:
        """Record raw stdout activity even when progress text has no newline yet."""
        with self._lock:
            self.last_output_monotonic = time.monotonic()

    def observe_line(self, line: str) -> None:
        now = time.monotonic()
        with self._lock:
            self.lines_seen += 1
            self.last_output_monotonic = now

        started_pipeline = extract_pipeline_start_from_line(line)
        if started_pipeline is not None:
            with self._lock:
                self.current_pipeline = started_pipeline

        current_suite = extract_pipeline_suite_from_line(line)
        if current_suite is not None:
            with self._lock:
                self.current_suite = current_suite

        current_model = extract_train_model_start_from_line(line)
        if current_model is not None:
            with self._lock:
                self.current_model = current_model

        fallback_reason = extract_catboost_fallback_reason_from_line(line)
        if fallback_reason is not None:
            with self._lock:
                self.last_fallback_reason = fallback_reason

        completed_pipeline = extract_pipeline_done_from_line(line)
        if completed_pipeline is not None:
            with self._lock:
                self.completed_pipelines.add(completed_pipeline)

        parsed = extract_training_stage_from_line(line)
        if parsed is None:
            return
        pipeline, seed, fold_raw = parsed
        key = (pipeline, seed, fold_raw)
        if key in self.seen_triplets:
            return
        self.seen_triplets.add(key)
        if fold_raw == 0:
            self.zero_based_folds = True

        fold_current = resolve_fold_current(
            fold_raw=fold_raw,
            expected_folds=self.expected_folds,
            zero_based=self.zero_based_folds,
        )
        seed_current = resolve_seed_current(seed=seed, expected_seeds=self.expected_seeds)
        elapsed_min = max(0.0, (time.monotonic() - self.started_at_monotonic) / 60.0)

        seed_part = (
            f"{seed_current}/{len(self.expected_seeds)}"
            if seed_current is not None and self.expected_seeds
            else str(seed)
        )
        fold_total = str(self.expected_folds) if self.expected_folds is not None else "?"
        fold_part = str(fold_current) if fold_current is not None else str(fold_raw)

        step_part = ""
        if (
            self.expected_folds is not None
            and self.expected_seeds
            and seed_current is not None
            and fold_current is not None
        ):
            step_current = ((seed_current - 1) * self.expected_folds) + fold_current
            step_total = self.expected_folds * len(self.expected_seeds)
            step_part = f" step={step_current}/{step_total}"

        print(
            "[cyan]kernel local stage[/cyan]: "
            f"pipeline={pipeline} seed={seed_part} fold={fold_part}/{fold_total}{step_part} "
            f"(elapsed={elapsed_min:.1f}m)"
        )

    def snapshot(self, now_monotonic: float | None = None) -> dict[str, object]:
        """Return a point-in-time progress snapshot for heartbeat rendering."""
        now = time.monotonic() if now_monotonic is None else now_monotonic
        now_wall = time.time()
        with self._lock:
            lines_seen = self.lines_seen
            last_output = self.last_output_monotonic
            current_pipeline = self.current_pipeline
            current_suite = self.current_suite
            current_model = self.current_model
            last_fallback_reason = self.last_fallback_reason
            completed_count = len(self.completed_pipelines)
        last_log_age_sec = None if last_output is None else max(0.0, now - last_output)
        artifact_count, last_artifact_age_sec = scan_watch_dirs_activity(
            watch_dirs=self.watch_dirs,
            now_wall=now_wall,
            min_mtime=self.started_at_wall - 1.0,
        )
        return {
            "lines_seen": lines_seen,
            "last_log_age_sec": last_log_age_sec,
            "current_pipeline": current_pipeline,
            "current_suite": current_suite,
            "current_model": current_model,
            "last_fallback_reason": last_fallback_reason,
            "completed_pipeline_count": completed_count,
            "artifact_count": artifact_count,
            "last_artifact_age_sec": last_artifact_age_sec,
        }


def scan_watch_dirs_activity(
    *,
    watch_dirs: tuple[Path, ...],
    now_wall: float,
    min_mtime: float | None = None,
) -> tuple[int, float | None]:
    """Scan watched directories and return (artifact_count, age_of_latest_artifact_sec)."""
    artifact_count = 0
    latest_mtime: float | None = None
    for root in watch_dirs:
        if not root.exists():
            continue
        try:
            paths = root.rglob("*")
        except OSError:
            continue
        for path in paths:
            if not path.is_file():
                continue
            if path.name in _LOCAL_KERNEL_CONTROL_FILENAMES:
                continue
            try:
                mtime = float(path.stat().st_mtime)
            except OSError:
                continue
            if min_mtime is not None and mtime < min_mtime:
                continue
            artifact_count += 1
            if latest_mtime is None or mtime > latest_mtime:
                latest_mtime = mtime
    if latest_mtime is None:
        return artifact_count, None
    return artifact_count, max(0.0, now_wall - latest_mtime)


def build_local_kernel_progress_tracker(
    *,
    base_dir: Path,
    slug: str,
    watch_dirs: list[Path] | None = None,
    started_at_wall: float | None = None,
    started_at_monotonic: float | None = None,
) -> LocalKernelProgressTracker:
    """Build a local-kernel progress tracker from plan metadata and watch dirs."""
    expected_folds: int | None = None
    expected_seeds: list[int] = []
    plan_path = base_dir / slug / "plan.json"
    payload = load_json_object(plan_path)
    if payload is not None:
        raw_folds = payload.get("cv_folds")
        if isinstance(raw_folds, int) and raw_folds > 0:
            expected_folds = raw_folds
        raw_eval_seeds = payload.get("eval_seeds")
        if isinstance(raw_eval_seeds, list):
            expected_seeds = [int(seed) for seed in raw_eval_seeds if isinstance(seed, int)]
        if not expected_seeds:
            raw_seed = payload.get("seed")
            if isinstance(raw_seed, int):
                expected_seeds = [raw_seed]
    watch_dir_tuple = tuple(watch_dirs or [])
    return LocalKernelProgressTracker(
        expected_folds=expected_folds,
        expected_seeds=expected_seeds,
        watch_dirs=watch_dir_tuple,
        started_at_wall=time.time() if started_at_wall is None else started_at_wall,
        started_at_monotonic=time.monotonic() if started_at_monotonic is None else started_at_monotonic,
    )


def detect_local_kernel_stall(
    *,
    progress_tracker: LocalKernelProgressTracker,
    stall_timeout_sec: float,
) -> str | None:
    snapshot = progress_tracker.snapshot()
    ages: list[float] = []
    for key in ("last_log_age_sec", "last_artifact_age_sec"):
        value = snapshot.get(key)
        if isinstance(value, (int, float)):
            ages.append(float(value))
    if ages:
        newest_activity_age_sec = min(ages)
    else:
        newest_activity_age_sec = max(0.0, time.monotonic() - progress_tracker.started_at_monotonic)
        stall_timeout_sec = max(stall_timeout_sec, LOCAL_KERNEL_STARTUP_STALL_GRACE_SEC)
    if newest_activity_age_sec < stall_timeout_sec:
        return None

    lines_seen = int(snapshot.get("lines_seen", 0))
    artifact_count = int(snapshot.get("artifact_count", 0))
    last_log_age_sec = snapshot.get("last_log_age_sec")
    last_artifact_age_sec = snapshot.get("last_artifact_age_sec")
    pipeline = str(snapshot.get("current_pipeline") or "unknown")
    model = str(snapshot.get("current_model") or "unknown")
    last_log_text = "none"
    if isinstance(last_log_age_sec, (int, float)):
        last_log_text = f"{int(last_log_age_sec)}s ago"
    last_artifact_text = "none"
    if isinstance(last_artifact_age_sec, (int, float)):
        last_artifact_text = f"{int(last_artifact_age_sec)}s ago"
    return (
        "Local kernel stalled: no stdout or artifact activity within the local watchdog budget "
        f"({int(stall_timeout_sec)}s). lines={lines_seen}, last_log={last_log_text}, "
        f"artifacts={artifact_count}, last_artifact={last_artifact_text}, "
        f"pipeline={pipeline}, model={model}."
    )


def local_kernel_heartbeat(
    *,
    stop_event: threading.Event,
    start_monotonic: float,
    timeout_sec: int | None,
    eta_total_sec: float | None,
    eta_samples: int,
    progress_tracker: LocalKernelProgressTracker,
    accelerator: str,
    interval_sec: float,
) -> None:
    """Emit periodic local-kernel progress heartbeats while execution is running."""
    while not stop_event.wait(interval_sec):
        elapsed = max(0.0, time.monotonic() - start_monotonic)
        print_local_kernel_progress(
            elapsed_sec=elapsed,
            timeout_sec=timeout_sec,
            eta_total_sec=eta_total_sec,
            eta_samples=eta_samples,
            progress_tracker=progress_tracker,
            accelerator=accelerator,
        )


def print_local_kernel_progress(
    *,
    elapsed_sec: float,
    timeout_sec: int | None,
    eta_total_sec: float | None,
    eta_samples: int,
    progress_tracker: LocalKernelProgressTracker | None,
    accelerator: str,
) -> None:
    """Render a single local-kernel heartbeat line."""
    activity_suffix = format_local_kernel_activity_suffix(progress_tracker)
    gpu_suffix = format_local_gpu_activity_suffix(accelerator=accelerator)
    elapsed = max(0, int(elapsed_sec))
    if eta_total_sec is not None and eta_total_sec > elapsed_sec:
        remaining = max(0, int(eta_total_sec - elapsed_sec))
        print(
            "[cyan]kernel local running[/cyan]: "
            f"elapsed={elapsed}s eta~{remaining}s (expected~{int(eta_total_sec)}s from {eta_samples} runs)"
            f"{activity_suffix}{gpu_suffix}"
        )
        return
    historical_suffix = ""
    if eta_total_sec is not None and eta_total_sec > 0:
        historical_suffix = f"; exact-source historical median~{int(eta_total_sec)}s exceeded"
    if timeout_sec is not None:
        timeout_remaining = max(0, int(timeout_sec - elapsed_sec))
        print(
            f"[cyan]kernel local running[/cyan]: "
            f"elapsed={elapsed}s eta=unknown (timeout in <= {timeout_remaining}s{historical_suffix})"
            f"{activity_suffix}{gpu_suffix}"
        )
        return
    exceeded_suffix = f" ({historical_suffix.removeprefix('; ')})" if historical_suffix else ""
    print(
        f"[cyan]kernel local running[/cyan]: elapsed={elapsed}s eta=unknown"
        f"{exceeded_suffix}{activity_suffix}{gpu_suffix}"
    )


def format_local_kernel_activity_suffix(progress_tracker: LocalKernelProgressTracker | None) -> str:
    """Format tracker activity details for local-kernel heartbeat lines."""
    if progress_tracker is None:
        return ""
    snapshot = progress_tracker.snapshot()
    lines_seen = int(snapshot.get("lines_seen", 0))
    last_log_age_sec = snapshot.get("last_log_age_sec")
    current_pipeline = snapshot.get("current_pipeline")
    current_suite = snapshot.get("current_suite")
    current_model = snapshot.get("current_model")
    last_fallback_reason = snapshot.get("last_fallback_reason")
    completed_pipeline_count = int(snapshot.get("completed_pipeline_count", 0))
    artifact_count = int(snapshot.get("artifact_count", 0))
    last_artifact_age_sec = snapshot.get("last_artifact_age_sec")
    last_log_text = "none"
    if isinstance(last_log_age_sec, (int, float)):
        last_log_text = f"{int(last_log_age_sec)}s ago"
    last_artifact_text = "none"
    if isinstance(last_artifact_age_sec, (int, float)):
        last_artifact_text = f"{int(last_artifact_age_sec)}s ago"
    pipeline_text = str(current_pipeline) if current_pipeline else "unknown"
    parts = [
        f"logs={lines_seen}",
        f"last_log={last_log_text}",
        f"pipeline={pipeline_text}",
    ]
    if current_suite:
        parts.append(f"suite={current_suite}")
    if current_model:
        parts.append(f"model={current_model}")
    parts.append(f"pipelines_done={completed_pipeline_count}")
    parts.append(f"artifacts={artifact_count}")
    parts.append(f"last_artifact={last_artifact_text}")
    if current_model and last_fallback_reason:
        parts.append(f"fallback={truncate_lines(last_fallback_reason, max_lines=1, max_chars=120)}")
    return " (" + ", ".join(parts) + ")"


def format_local_gpu_activity_suffix(*, accelerator: str) -> str:
    """Return a short GPU utilization suffix for heartbeat lines when available."""
    if accelerator != "gpu":
        return ""
    if shutil.which("nvidia-smi") is None:
        return ""
    try:
        probe = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
    except Exception:
        return ""
    if probe.returncode != 0:
        return ""
    first_line = ""
    for raw in probe.stdout.splitlines():
        stripped = raw.strip()
        if stripped:
            first_line = stripped
            break
    if not first_line:
        return ""
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) < 3:
        return ""
    util, mem_used, mem_total = parts[0], parts[1], parts[2]
    return f" (gpu={util}%, mem={mem_used}/{mem_total}MiB)"
