from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich import print

from kagglebot.compute import detect_local_gpu
from kagglebot.exceptions import (
    KaggleCliError,
    KaggleNetworkError,
    KernelFailedError,
    KernelTimeoutError,
    RulesNotAcceptedError,
)
from kagglebot.exec_utils import run_command
from kagglebot.kaggle_api import (
    check_rules_accepted,
    kernel_exists,
    kernel_id_by_title,
    kernels_init,
    kernels_output,
    kernels_push,
    kernels_status,
)
from kagglebot.logging_utils import truncate_lines
from kagglebot.solution_guard import ensure_solution_path_allowed
from kagglebot.validators import ensure_kernel_sources_valid, validate_kernel_package

_COLUMN_MAP_FILENAME = "column_map.json"
_COLUMN_MAP_SHIM_MARKER = "# kagglebot: column-map-shim"
_COLUMN_FILL_FILENAME = "column_fill.json"
_COLUMN_FILL_SHIM_MARKER = "# kagglebot: column-fill-shim"
_OBJECT_COERCE_FILENAME = "object_coerce.json"
_OBJECT_COERCE_SHIM_MARKER = "# kagglebot: object-coerce-shim"
_DEVICE_COERCE_FILENAME = "device_coerce.json"
_DEVICE_COERCE_SHIM_MARKER = "# kagglebot: device-coerce-shim"
_LOCAL_KERNEL_HEARTBEAT_INTERVAL_SEC = 30.0
_LOCAL_KERNEL_DURATION_HISTORY_LIMIT = 20

_PIPELINE_SEED_FOLD_RE = re.compile(r"(?P<pipeline>[A-Za-z0-9][A-Za-z0-9_.-]*)_seed(?P<seed>\d+)_fold(?P<fold>\d+)")
_PIPELINE_SEED_FOLD_INLINE_RE = re.compile(
    r"\b(?P<pipeline>[A-Za-z0-9][A-Za-z0-9_.-]*)\s*:\s*seed=(?P<seed>\d+)\s+fold=(?P<fold>\d+)\b"
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


@dataclass(frozen=True)
class KernelPackageBuilder:
    def prepare(self, config: KernelBuildConfig) -> KernelPreparation:
        kernel_dir = config.base_dir / config.slug / "kernels" / config.run_id
        output_dir = config.base_dir / config.slug / "runs" / config.run_id / f"iter-{config.iteration}" / "output"
        logs_dir = config.base_dir / config.slug / "runs" / config.run_id / f"iter-{config.iteration}" / "logs"
        kernel_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        if not config.dry_run and not check_rules_accepted(config.slug, dry_run=False):
            raise RulesNotAcceptedError("Competition rules not accepted.")

        if not config.dry_run:
            print(f"[cyan]kernel init[/cyan]: {kernel_dir}")
            kernels_init(kernel_dir, dry_run=False)

        kernel_slug = _resolve_kernel_slug(config.kernel_name, config.slug, config.run_id, config.iteration)
        kernel_id = f"{config.kaggle_username}/{kernel_slug}"
        custom_kernel_dir = config.base_dir / config.slug / "kernel"
        custom_kernel_path = custom_kernel_dir / "kernel.py"
        ensure_solution_path_allowed(custom_kernel_dir, artifacts_dir=config.base_dir, slug=config.slug)
        if not custom_kernel_path.exists():
            raise KernelFailedError(
                "Authoritative kernel entrypoint is missing. "
                f"Expected: {custom_kernel_path}. "
                "Generate/update artifacts/<slug>/kernel/kernel.py before running training."
            )
        _copy_kernel_sources(custom_kernel_dir, kernel_dir)
        _sync_plan_snapshot(
            plan_path=config.base_dir / config.slug / "plan.json",
            targets=[kernel_dir / "plan.json"],
        )
        _ensure_kernel_import_path(kernel_dir)
        _inline_kernel_modules(kernel_dir)
        _inject_data_dir_resolver(kernel_dir)
        _inject_pipeline_cfg_fallback(kernel_dir)
        _inject_column_map_shim(kernel_dir, config.base_dir / config.slug / "context")
        _inject_column_fill_shim(kernel_dir, config.base_dir / config.slug / "context")
        _inject_object_coerce_shim(kernel_dir, config.base_dir / config.slug / "context")
        _inject_device_coerce_shim(kernel_dir, config.base_dir / config.slug / "context")
        ensure_kernel_sources_valid(kernel_dir)
        _write_kernel_metadata(
            kernel_dir=kernel_dir,
            kernel_id=kernel_id,
            title=kernel_slug,
            code_file="kernel.py",
            accelerator=config.accelerator,
            enable_internet=config.enable_internet,
            competition_slug=config.slug,
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
class KernelJobMonitor:
    def push_and_wait(
        self,
        *,
        preparation: KernelPreparation,
        slug: str,
        timeout_minutes: int | None,
    ) -> str:
        print(f"[cyan]kernel push[/cyan]: {preparation.kernel_dir}")
        push_attempt = 1
        kernel_id = preparation.kernel_id
        push_output = kernels_push(preparation.kernel_dir, slug=slug, dry_run=False)
        _write_push_log(preparation.logs_dir, push_attempt, push_output)
        pushed_kernel_id = _extract_kernel_id_from_push(push_output)
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
            pushed_kernel_id = _extract_kernel_id_from_push(push_output)
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
        _wait_for_kernel(kernel_id, slug, timeout_minutes, output_dir=preparation.output_dir)
        print(f"[cyan]kernel output[/cyan]: {preparation.output_dir}")
        kernels_output(kernel_id, preparation.output_dir, slug=slug, dry_run=False)
        return kernel_id


@dataclass(frozen=True)
class KernelLogParser:
    @staticmethod
    def collect_tail(output_dir: Path, max_lines: int = 50) -> str | None:
        return _collect_log_tail(output_dir, max_lines=max_lines)


def sanitize_kernel_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    return cleaned[:50]


_KERNEL_URL_RE = re.compile(
    r"https?://(?:www\.)?kaggle\.com/(?:code/)?(?P<user>[A-Za-z0-9_-]+)/(?P<slug>[A-Za-z0-9_.-]+)"
)
_KERNEL_ID_RE = re.compile(r"(?P<user>[A-Za-z0-9_-]+)/(?P<slug>[A-Za-z0-9_.-]+)")


def _extract_kernel_id_from_push(output: str) -> str | None:
    if not output:
        return None
    match = _KERNEL_URL_RE.search(output)
    if match:
        return f"{match.group('user')}/{match.group('slug')}"
    for line in output.splitlines():
        if "kernel" not in line.lower():
            continue
        match = _KERNEL_ID_RE.search(line)
        if match:
            return f"{match.group('user')}/{match.group('slug')}"
    return None


def find_submission_file(output_dir: Path) -> Path | None:
    candidate = _find_output_file(output_dir, "submission.csv")
    if candidate:
        return candidate
    return _find_submission_by_extension(output_dir)


def resolve_kaggle_username(explicit: str | None) -> str:
    if explicit:
        return explicit
    env_user = os.getenv("KAGGLE_USERNAME")
    if env_user:
        return env_user
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        data = json.loads(kaggle_json.read_text(encoding="utf-8"))
        if "username" in data:
            return str(data["username"])
    raise ValueError("Kaggle username not found. Provide --kaggle-username or set KAGGLE_USERNAME.")


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
) -> KernelRunResult:
    del score_source, metric, direction, holdout_frac, cv_folds, seed

    kernel_source_dir = base_dir / slug / "kernel"
    kernel_stage_dir = base_dir / slug / "kernels" / run_id / f"local-iter-{iteration}"
    run_dir = kernel_stage_dir.parent
    context_dir = base_dir / slug / "context"
    output_dir = base_dir / slug / "runs" / run_id / f"iter-{iteration}" / "output"
    logs_dir = base_dir / slug / "runs" / run_id / f"iter-{iteration}" / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    _ensure_local_sample_submission_file(base_dir=base_dir, slug=slug)
    _stage_local_kernel_data_dir(base_dir=base_dir, slug=slug, run_dir=run_dir)

    ensure_solution_path_allowed(kernel_source_dir, artifacts_dir=base_dir, slug=slug)
    kernel_path = kernel_source_dir / "kernel.py"
    if not kernel_path.exists():
        raise KernelFailedError(f"Local kernel execution requires {kernel_path} to exist.")
    if kernel_stage_dir.exists():
        shutil.rmtree(kernel_stage_dir)
    shutil.copytree(kernel_source_dir, kernel_stage_dir)
    _sync_plan_snapshot(
        plan_path=base_dir / slug / "plan.json",
        targets=[
            kernel_stage_dir / "plan.json",
            kernel_stage_dir.parent / "plan.json",
        ],
    )
    kernel_path = kernel_stage_dir / "kernel.py"

    if strict_accelerator and accelerator == "gpu":
        availability = detect_local_gpu()
        if not availability.any:
            raise KernelFailedError("No local GPU detected while --strict-accelerator is enabled for local_gpu.")

    # Mirror packaging shims so local and kaggle kernel behavior are aligned.
    _ensure_kernel_import_path(kernel_stage_dir)
    _inline_kernel_modules(kernel_stage_dir)
    _inject_data_dir_resolver(kernel_stage_dir)
    _inject_pipeline_cfg_fallback(kernel_stage_dir)
    _inject_column_map_shim(kernel_stage_dir, context_dir)
    _inject_column_fill_shim(kernel_stage_dir, context_dir)
    _inject_object_coerce_shim(kernel_stage_dir, context_dir)
    _inject_device_coerce_shim(kernel_stage_dir, context_dir)
    ensure_kernel_sources_valid(kernel_stage_dir, require_kaggle_input=False)

    if dry_run:
        return KernelRunResult(
            kernel_id=f"local/{slug}",
            output_dir=output_dir,
            submission_path=None,
            metrics_path=None,
        )

    timeout_sec = None if timeout_minutes is None else max(60, int(timeout_minutes * 60))
    eta_total_sec, eta_samples = _estimate_local_kernel_duration_seconds(base_dir=base_dir, slug=slug)
    progress_tracker = _build_local_kernel_progress_tracker(base_dir=base_dir, slug=slug)
    _print_local_kernel_progress(
        elapsed_sec=0.0,
        timeout_sec=timeout_sec,
        eta_total_sec=eta_total_sec,
        eta_samples=eta_samples,
    )
    started_at = time.time()
    monotonic_start = time.monotonic()
    env = os.environ.copy()
    env.setdefault("KAGGLEBOT_LOCAL_KERNEL", "1")
    env.setdefault("KAGGLEBOT_SLUG", slug)
    env.setdefault("KAGGLEBOT_RUN_ID", run_id)
    env.setdefault("KAGGLEBOT_ITERATION", str(iteration))
    env.setdefault("KAGGLEBOT_ACCELERATOR", accelerator)

    heartbeat_stop = threading.Event()
    heartbeat = threading.Thread(
        target=_local_kernel_heartbeat,
        kwargs={
            "stop_event": heartbeat_stop,
            "start_monotonic": monotonic_start,
            "timeout_sec": timeout_sec,
            "eta_total_sec": eta_total_sec,
            "eta_samples": eta_samples,
            "interval_sec": _LOCAL_KERNEL_HEARTBEAT_INTERVAL_SEC,
        },
        daemon=True,
    )
    heartbeat.start()
    try:
        result = run_command(
            [sys.executable, str(kernel_path)],
            cwd=kernel_stage_dir,
            env=env,
            timeout=timeout_sec,
            stream_output=True,
            line_callback=progress_tracker.observe_line,
        )
    except subprocess.TimeoutExpired as exc:
        raise KernelTimeoutError(f"Local kernel timed out after {timeout_sec}s.") from exc
    finally:
        heartbeat_stop.set()
        heartbeat.join(timeout=1.0)

    (logs_dir / "local_kernel_stdout.log").write_text(result.stdout, encoding="utf-8")
    (logs_dir / "local_kernel_stderr.log").write_text(result.stderr, encoding="utf-8")

    if result.returncode != 0:
        stdout_tail = truncate_lines(result.stdout[-4000:], max_lines=80)
        stderr_tail = truncate_lines(result.stderr[-4000:], max_lines=80)
        detail = "\n".join(part for part in [stdout_tail, stderr_tail] if part).strip()
        if detail:
            detail = f"\n{detail}"
        raise KernelFailedError(f"Local kernel execution failed with exit code {result.returncode}.{detail}")
    _append_local_kernel_duration_history(
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

    return KernelRunResult(
        kernel_id=f"local/{slug}",
        output_dir=output_dir,
        submission_path=submission_dst,
        metrics_path=metrics_dst,
    )


def _stage_local_kernel_data_dir(*, base_dir: Path, slug: str, run_dir: Path) -> None:
    competition_dir = base_dir / slug
    source_dir = (competition_dir / "data").resolve()
    if not source_dir.exists():
        return

    target_dir = run_dir / "data"
    if target_dir.is_symlink():
        try:
            if target_dir.resolve() == source_dir:
                return
        except Exception:
            pass
        try:
            target_dir.unlink()
        except OSError:
            pass
    elif target_dir.exists():
        if target_dir.is_dir():
            shutil.rmtree(target_dir, ignore_errors=True)
        else:
            try:
                target_dir.unlink()
            except OSError:
                return

    try:
        target_dir.symlink_to(source_dir, target_is_directory=True)
        return
    except Exception:
        pass

    # Fallback for filesystems where directory symlink is unavailable.
    shutil.copytree(source_dir, target_dir, symlinks=True, dirs_exist_ok=True)


def _ensure_local_sample_submission_file(*, base_dir: Path, slug: str) -> Path | None:
    competition_dir = base_dir / slug
    data_dir = competition_dir / "data"
    canonical_path = data_dir / "sample_submission.csv"
    if canonical_path.exists():
        return canonical_path
    source_path = _resolve_sample_submission_source(
        context_dir=competition_dir / "context",
        data_dir=data_dir,
    )
    if source_path is None:
        return None
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, canonical_path)
    return canonical_path


def _resolve_sample_submission_source(*, context_dir: Path, data_dir: Path) -> Path | None:
    context_sample = context_dir / "sample_submission.csv"
    if context_sample.exists():
        return context_sample
    if not data_dir.exists():
        return None
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        if "sample_submission" not in name:
            continue
        if path.suffix.lower() != ".csv":
            continue
        return path
    return None


@dataclass
class _LocalKernelProgressTracker:
    expected_folds: int | None
    expected_seeds: list[int]
    started_at_monotonic: float = field(default_factory=time.monotonic)
    zero_based_folds: bool = False
    seen_triplets: set[tuple[str, int, int]] = field(default_factory=set)

    def observe_line(self, line: str) -> None:
        parsed = _extract_training_stage_from_line(line)
        if parsed is None:
            return
        pipeline, seed, fold_raw = parsed
        key = (pipeline, seed, fold_raw)
        if key in self.seen_triplets:
            return
        self.seen_triplets.add(key)
        if fold_raw == 0:
            self.zero_based_folds = True

        fold_current = _resolve_fold_current(
            fold_raw=fold_raw,
            expected_folds=self.expected_folds,
            zero_based=self.zero_based_folds,
        )
        seed_current = _resolve_seed_current(seed=seed, expected_seeds=self.expected_seeds)
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


def _build_local_kernel_progress_tracker(*, base_dir: Path, slug: str) -> _LocalKernelProgressTracker:
    expected_folds: int | None = None
    expected_seeds: list[int] = []
    plan_path = base_dir / slug / "plan.json"
    if plan_path.exists():
        try:
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
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
    return _LocalKernelProgressTracker(expected_folds=expected_folds, expected_seeds=expected_seeds)


def _extract_training_stage_from_line(line: str) -> tuple[str, int, int] | None:
    inline_match = _PIPELINE_SEED_FOLD_INLINE_RE.search(line)
    if inline_match:
        return _match_to_stage_tuple(inline_match)
    path_match = _PIPELINE_SEED_FOLD_RE.search(line)
    if path_match:
        return _match_to_stage_tuple(path_match)
    return None


def _match_to_stage_tuple(match: re.Match[str]) -> tuple[str, int, int] | None:
    try:
        pipeline = str(match.group("pipeline")).strip()
        seed = int(match.group("seed"))
        fold = int(match.group("fold"))
    except Exception:  # noqa: BLE001
        return None
    if not pipeline:
        return None
    return pipeline, seed, fold


def _resolve_seed_current(*, seed: int, expected_seeds: list[int]) -> int | None:
    if not expected_seeds:
        return None
    try:
        return expected_seeds.index(seed) + 1
    except ValueError:
        return None


def _resolve_fold_current(*, fold_raw: int, expected_folds: int | None, zero_based: bool) -> int | None:
    if expected_folds is None:
        return None
    if zero_based:
        value = fold_raw + 1
        if 1 <= value <= expected_folds:
            return value
    if 1 <= fold_raw <= expected_folds:
        return fold_raw
    if 0 <= fold_raw < expected_folds:
        return fold_raw + 1
    return None


def _local_kernel_history_path(*, base_dir: Path, slug: str) -> Path:
    return base_dir / slug / "context" / "local_kernel_duration_history.jsonl"


def _estimate_local_kernel_duration_seconds(*, base_dir: Path, slug: str) -> tuple[float | None, int]:
    path = _local_kernel_history_path(base_dir=base_dir, slug=slug)
    if not path.exists():
        return None, 0
    durations: list[float] = []
    for raw in reversed(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        value = payload.get("duration_sec")
        if isinstance(value, (int, float)) and value > 0:
            durations.append(float(value))
        if len(durations) >= _LOCAL_KERNEL_DURATION_HISTORY_LIMIT:
            break
    if not durations:
        return None, 0
    durations_sorted = sorted(durations)
    mid = len(durations_sorted) // 2
    if len(durations_sorted) % 2 == 1:
        median = durations_sorted[mid]
    else:
        median = (durations_sorted[mid - 1] + durations_sorted[mid]) / 2.0
    return median, len(durations_sorted)


def _append_local_kernel_duration_history(
    *,
    base_dir: Path,
    slug: str,
    run_id: str,
    iteration: int,
    duration_sec: float,
) -> None:
    path = _local_kernel_history_path(base_dir=base_dir, slug=slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "iteration": int(iteration),
        "duration_sec": float(duration_sec),
        "recorded_at": int(time.time()),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _local_kernel_heartbeat(
    *,
    stop_event: threading.Event,
    start_monotonic: float,
    timeout_sec: int | None,
    eta_total_sec: float | None,
    eta_samples: int,
    interval_sec: float,
) -> None:
    while not stop_event.wait(interval_sec):
        elapsed = max(0.0, time.monotonic() - start_monotonic)
        _print_local_kernel_progress(
            elapsed_sec=elapsed,
            timeout_sec=timeout_sec,
            eta_total_sec=eta_total_sec,
            eta_samples=eta_samples,
        )


def _print_local_kernel_progress(
    *,
    elapsed_sec: float,
    timeout_sec: int | None,
    eta_total_sec: float | None,
    eta_samples: int,
) -> None:
    elapsed = max(0, int(elapsed_sec))
    if eta_total_sec is not None and eta_total_sec > 0:
        remaining = max(0, int(eta_total_sec - elapsed_sec))
        print(
            "[cyan]kernel local running[/cyan]: "
            f"elapsed={elapsed}s eta~{remaining}s (expected~{int(eta_total_sec)}s from {eta_samples} runs)"
        )
        return
    if timeout_sec is not None:
        timeout_remaining = max(0, int(timeout_sec - elapsed_sec))
        print(f"[cyan]kernel local running[/cyan]: elapsed={elapsed}s eta=unknown (timeout in <= {timeout_remaining}s)")
        return
    print(f"[cyan]kernel local running[/cyan]: elapsed={elapsed}s eta=unknown")


def _resolve_local_kernel_artifacts(
    *,
    kernel_dir: Path,
    output_dir: Path,
    started_at: float,
) -> tuple[Path | None, Path | None]:
    candidates: list[Path] = [
        output_dir,
        # Many kernels treat the parent of the staged copy (run_dir) as the
        # "challenge dir" and write artifacts under run_dir/outputs.
        kernel_dir.parent / "outputs",
        kernel_dir.parent,
        kernel_dir / "outputs",
        Path("/kaggle/working"),
        kernel_dir,
    ]
    submission_candidates: list[Path] = []
    metrics_candidates: list[Path] = []
    for root in candidates:
        if not root.exists():
            continue
        sub = find_submission_file(root)
        if sub is not None and sub.exists():
            submission_candidates.append(sub)
        metric_path = _find_output_file(root, "metrics.json")
        if metric_path is not None and metric_path.exists():
            metrics_candidates.append(metric_path)

    min_mtime = started_at - 1.0
    submission_path = _pick_latest_artifact(submission_candidates, min_mtime=min_mtime)
    metrics_path = _pick_latest_artifact(metrics_candidates, min_mtime=min_mtime)
    return submission_path, metrics_path


def _pick_latest_artifact(paths: list[Path], *, min_mtime: float) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    fresh = [path for path in existing if path.stat().st_mtime >= min_mtime]
    pool = fresh or existing
    return max(pool, key=lambda path: path.stat().st_mtime)


def _copy_artifact_if_needed(*, source: Path, destination: Path) -> Path:
    source_resolved = source.resolve()
    destination_resolved = destination.resolve()
    if source_resolved == destination_resolved:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _resolve_kernel_slug(kernel_name: str | None, slug: str, run_id: str, iteration: int) -> str:
    if kernel_name:
        return sanitize_kernel_slug(kernel_name)
    suffix = f"{run_id[-6:]}-i{iteration}"
    prefix = f"kagglebot-{slug}"
    max_len = 50
    allowed_prefix_len = max_len - len(suffix) - 1
    if allowed_prefix_len < 1:
        prefix = "kagglebot"
    else:
        prefix = prefix[:allowed_prefix_len].rstrip("-")
    base = f"{prefix}-{suffix}"
    return sanitize_kernel_slug(base)


def _write_kernel_metadata(
    *,
    kernel_dir: Path,
    kernel_id: str,
    title: str,
    code_file: str,
    accelerator: str,
    enable_internet: bool,
    competition_slug: str,
) -> None:
    meta_path = kernel_dir / "kernel-metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {}
    meta.update(
        {
            "id": kernel_id,
            "title": title,
            "code_file": code_file,
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": accelerator == "gpu",
            "enable_tpu": accelerator == "tpu",
            "enable_internet": bool(enable_internet),
            "competition_sources": [competition_slug],
            "dataset_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
    )
    if meta["enable_gpu"] and meta["enable_tpu"]:
        raise ValueError("kernel-metadata.json cannot enable both GPU and TPU.")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _copy_kernel_sources(source_dir: Path, dest_dir: Path) -> None:
    for path in source_dir.iterdir():
        dest_path = dest_dir / path.name
        if path.is_dir():
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(path, dest_path)
        elif path.is_file():
            shutil.copy2(path, dest_path)


def _sync_plan_snapshot(*, plan_path: Path, targets: list[Path]) -> None:
    if not plan_path.exists():
        return
    for target in targets:
        if target.resolve() == plan_path.resolve():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plan_path, target)


_KERNEL_BOOTSTRAP_MARKER = "# kagglebot:kernel_sys_path"
_KERNEL_BOOTSTRAP_END = "del _os, _sys, _KROOT, _KWORK"
_KERNEL_DATA_RESOLVER_MARKER = "# kagglebot:data_resolver"
_KERNEL_PIPELINE_CFG_MARKER = "# kagglebot:pipeline_cfg_fallback"
_DATA_DIR_JOIN_RE = re.compile(r"(\bdata_dir\s*/\s*)(['\"])([^'\"]+)\2")


def _strip_kernel_bootstrap(lines: list[str]) -> list[str]:
    stripped = lines
    while _KERNEL_BOOTSTRAP_MARKER in stripped:
        start = stripped.index(_KERNEL_BOOTSTRAP_MARKER)
        end = None
        search_end = min(start + 20, len(stripped))
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
    if _KERNEL_DATA_RESOLVER_MARKER in text:
        return
    if not _DATA_DIR_JOIN_RE.search(text):
        return
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
    lines = text.splitlines()
    insert_at = _find_bootstrap_block_end(lines)
    if insert_at is None:
        insert_at = _find_bootstrap_insertion_index(lines)
    lines = lines[:insert_at] + resolver_block + lines[insert_at:]
    updated = "\n".join(lines)
    updated = _DATA_DIR_JOIN_RE.sub(r"_kb_find_file(data_dir, '\3')", updated)
    if text.endswith("\n"):
        updated += "\n"
    kernel_path.write_text(updated, encoding="utf-8")


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
        "def _kb_patch_pandas_fill() -> None:",
        "    try:",
        "        import pandas as _pd",
        "    except Exception:",
        "        return",
        "    _orig = _pd.read_csv",
        "    def _patched(*args, **kwargs):",
        "        df = _orig(*args, **kwargs)",
        "        try:",
        "            path_value = args[0] if args else kwargs.get('filepath_or_buffer')",
        "            missing_cols = _kb_missing_columns_for(path_value)",
        "            for col in missing_cols:",
        "                if col not in df.columns:",
        "                    df[col] = _pd.NA",
        "        except Exception:",
        "            return df",
        "        return df",
        "    _pd.read_csv = _patched",
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
        site_path.write_text(text.rstrip("\\n") + "\\n\\n" + "\\n".join(shim), encoding="utf-8")
        return
    site_path.write_text("\\n".join(shim), encoding="utf-8")


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


def _wait_for_kernel(kernel_id: str, slug: str, timeout_minutes: int | None, *, output_dir: Path) -> None:
    deadline = None
    if timeout_minutes is not None:
        deadline = time.monotonic() + max(timeout_minutes, 1) * 60
    started_at = time.monotonic()
    last_status = None
    last_log_fetch = 0.0
    log_state = _KernelLogState()
    status_errors = 0
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
                    raise KernelTimeoutError("Kaggle kernel did not complete within timeout.") from exc
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
                raise KernelTimeoutError("Kaggle kernel did not complete within timeout.") from exc
            if MAX_STATUS_ERRORS is not None and status_errors >= MAX_STATUS_ERRORS:
                raise KernelFailedError(
                    f"Kaggle kernel status failed {status_errors} times. Last error: {detail or 'unknown error'}"
                ) from exc
            time.sleep(STATUS_ERROR_SLEEP)
            continue
        status = _parse_kernel_status(output).lower()
        if status != last_status:
            print(f"[cyan]kernel status[/cyan]: {status}")
            last_status = status
        now = time.monotonic()
        if now - last_log_fetch >= LOG_POLL_INTERVAL:
            _try_fetch_kernel_output(kernel_id, output_dir=output_dir, slug=slug)
            had_logs = _print_kernel_logs(output_dir, log_state)
            if had_logs:
                log_state.last_log_at = now
            last_log_fetch = now
            log_failure = _detect_failure_in_logs(output_dir)
            if log_failure:
                log_failure = truncate_lines(log_failure, max_lines=5)
                message = f"Kaggle kernel error detected in logs.\n\n--- kernel log tail ---\n{log_failure}"
                raise KernelFailedError(message)
        if status in {"running", "queued"}:
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
        if "complete" in status:
            return
        if "error" in status or "fail" in status:
            _try_fetch_kernel_output(kernel_id, output_dir=output_dir, slug=slug)
            log_tail = _collect_log_tail(output_dir)
            message = f"Kaggle kernel failed: {output}"
            if log_tail:
                log_tail = truncate_lines(log_tail, max_lines=5)
                message = f"{message}\n\n--- kernel log tail ---\n{log_tail}"
            raise KernelFailedError(message)
        time.sleep(STATUS_ERROR_SLEEP)
        if deadline is not None and time.monotonic() > deadline:
            raise KernelTimeoutError("Kaggle kernel did not complete within timeout.")


@dataclass
class _KernelLogState:
    seen_lines: dict[Path, int] = field(default_factory=dict)
    seen_json: dict[Path, int] = field(default_factory=dict)
    seen_size: dict[Path, int] = field(default_factory=dict)
    last_log_at: float | None = None
    last_heartbeat: float = 0.0


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


def _parse_kernel_status(output: str) -> str:
    match = re.search(r"status\\s+\\\"?([A-Za-z0-9_.-]+)\\\"?", output)
    if match:
        return match.group(1)
    return output.strip() or "unknown"


def _try_fetch_kernel_output(kernel_id: str, *, output_dir: Path, slug: str) -> None:
    try:
        kernels_output(kernel_id, output_dir, slug=slug, dry_run=False, force=True, quiet=True)
    except KaggleCliError:
        return


def _log_candidates(output_dir: Path) -> list[Path]:
    candidates = []
    for name in ("stdout.txt", "stderr.txt", "output.log", "log.txt", "logs.txt"):
        path = output_dir / name
        if path.exists():
            candidates.append(path)
    candidates.extend(sorted(output_dir.rglob("*.log")))
    return candidates


def _print_kernel_logs(output_dir: Path, state: _KernelLogState) -> bool:
    printed = False
    for path in _log_candidates(output_dir):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        size = len(text)
        prev_size = state.seen_size.get(path, 0)
        if size < prev_size:
            state.seen_lines[path] = 0
            state.seen_json[path] = 0
        state.seen_size[path] = size

        json_events = _parse_json_log(text)
        if json_events is not None:
            last = state.seen_json.get(path, 0)
            if len(json_events) <= last:
                continue
            new_events = json_events[last:]
            state.seen_json[path] = len(json_events)
            formatted = _format_log_events(new_events)
            if not formatted:
                continue
            print(f"[cyan]kernel log[/cyan]: {path.name}")
            print(truncate_lines("\n".join(formatted), max_lines=5))
            printed = True
            continue

        lines = text.splitlines()
        last = state.seen_lines.get(path, 0)
        if len(lines) <= last:
            continue
        new_lines = lines[last:]
        state.seen_lines[path] = len(lines)
        print(f"[cyan]kernel log[/cyan]: {path.name}")
        print(truncate_lines("\n".join(new_lines), max_lines=5))
        printed = True
    return printed


def _detect_failure_in_logs(output_dir: Path) -> str | None:
    for path in _log_candidates(output_dir):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "Traceback (most recent call last)" not in text:
            continue
        tail = _collect_log_tail_from_text(path, text)
        if tail:
            return tail
        return f"{path.name}\nTraceback detected"
    return None


def _collect_log_tail(output_dir: Path, max_lines: int = 50) -> str | None:
    candidates = _log_candidates(output_dir)
    if not candidates:
        return None
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "Traceback (most recent call last)" in text:
            return _collect_log_tail_from_text(path, text, max_lines=max_lines)
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "Error" in text or "Exception" in text:
            return _collect_log_tail_from_text(path, text, max_lines=max_lines)
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        tail = _collect_log_tail_from_text(path, text, max_lines=max_lines)
        if tail:
            return tail
    return None


def _collect_log_tail_from_text(path: Path, text: str, max_lines: int = 50) -> str | None:
    json_events = _parse_json_log(text)
    if json_events is not None:
        formatted = _format_log_events(json_events)
        if not formatted:
            return None
        start = _find_error_marker_index(formatted)
        if start is None:
            start = max(len(formatted) - max_lines, 0)
        else:
            if len(formatted) - start > max_lines:
                start = max(len(formatted) - max_lines, start)
        tail = "\n".join(formatted[start:])
        return f"{path.name}\n{tail}".strip()
    lines = text.splitlines()
    if not lines:
        return None
    start = _find_error_marker_index(lines)
    if start is None:
        start = max(len(lines) - max_lines, 0)
    else:
        if len(lines) - start > max_lines:
            start = max(len(lines) - max_lines, start)
    tail = "\n".join(lines[start:])
    return f"{path.name}\n{tail}".strip()


def _find_error_marker_index(lines: list[str]) -> int | None:
    markers = ("Traceback", "Error", "Exception")
    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx]
        if any(marker in line for marker in markers):
            return idx
    return None


def _parse_json_log(text: str) -> list[dict[str, object]] | None:
    stripped = text.lstrip()
    if not stripped:
        return None
    if stripped.startswith("["):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return None
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict) and isinstance(payload.get("logs"), list):
            return [item for item in payload["logs"] if isinstance(item, dict)]
        return None
    return None


def _format_log_events(events: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for event in events:
        data = event.get("data")
        if not isinstance(data, str) or not data:
            continue
        stream = event.get("stream_name")
        prefix = f"[{stream}] " if isinstance(stream, str) and stream else ""
        for line in data.splitlines():
            lines.append(f"{prefix}{line}")
    return lines


def _find_output_file(output_dir: Path, filename: str) -> Path | None:
    """Find the newest matching artifact within an output tree.

    Local kernels can be executed repeatedly for the same run/iteration while
    iterating on fixes. In that scenario, stale artifacts may exist alongside
    fresh ones (or nested under additional run directories). Prefer the most
    recently modified match to avoid accidentally reusing stale outputs.
    """

    candidates: list[Path] = []
    direct = output_dir / filename
    if direct.exists():
        candidates.append(direct)
    try:
        candidates.extend(path for path in output_dir.rglob(filename) if path.exists())
    except OSError:
        # Best-effort discovery; callers handle missing artifacts.
        pass
    files = [path for path in candidates if path.is_file()]
    if not files:
        return None
    # Deterministic tie-breaker: path string.
    return max(files, key=lambda path: (path.stat().st_mtime, str(path)))


def _find_submission_by_extension(output_dir: Path) -> Path | None:
    suffixes = [".csv", ".tsv", ".txt", ".parquet", ".json", ".jsonl"]
    for suffix in suffixes:
        candidate = output_dir / f"submission{suffix}"
        if candidate.exists():
            return candidate
    for path in output_dir.rglob("submission.*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in suffixes:
            return path
    return None
