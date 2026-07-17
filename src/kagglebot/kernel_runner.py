from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
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
from kagglebot import kernel_remote_ops as _kernel_remote_ops
from kagglebot import kernel_wait as _kernel_wait
from kagglebot import local_kernel_aux_inputs as _local_kernel_aux_inputs
from kagglebot import local_kernel_context as _local_kernel_context
from kagglebot import local_kernel_data_resolver as _local_kernel_data_resolver
from kagglebot import local_kernel_drift_guard as _local_kernel_drift_guard
from kagglebot import local_kernel_duration as _local_kernel_duration
from kagglebot import local_kernel_limits as _local_kernel_limits
from kagglebot import local_kernel_metrics_normalization as _local_kernel_metrics_normalization
from kagglebot import local_kernel_models as _local_kernel_models
from kagglebot import local_kernel_pipeline_cfg as _local_kernel_pipeline_cfg
from kagglebot import local_kernel_process as _local_kernel_process
from kagglebot import local_kernel_progress as _local_kernel_progress
from kagglebot import local_kernel_resume as _local_kernel_resume
from kagglebot import local_kernel_runtime_env as _local_kernel_runtime_env
from kagglebot import local_kernel_shims as _local_kernel_shims
from kagglebot import local_sample_submission as _local_sample_submission
from kagglebot import remote_kernel_state as _remote_kernel_state
from kagglebot import submit_kernel_fidelity as _submit_kernel_fidelity
from kagglebot.competition_policy import load_competition_policy
from kagglebot.compute import detect_local_gpu
from kagglebot.exceptions import (
    KaggleCliError,
    KernelFailedError,
    KernelStillRunningError,
    KernelTimeoutError,
    RulesNotAcceptedError,
)
from kagglebot.hardware import hardware_env, resolve_hardware_profile
from kagglebot.hashing import sha256_path, sha256_text
from kagglebot.json_utils import load_json_object_or_empty
from kagglebot.kaggle_api import (
    check_rules_accepted,
    kernel_exists,
    kernel_id_by_title,
    kernels_init,
    kernels_output,
    kernels_push,
    kernels_status,
)
from kagglebot.kaggle_credentials import resolve_kaggle_username as _resolve_kaggle_username
from kagglebot.kernel_outputs import copy_local_kernel_primary_artifacts as _copy_local_kernel_primary_artifacts
from kagglebot.kernel_outputs import copy_optional_local_kernel_artifacts as _copy_optional_local_kernel_artifacts
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
from kagglebot.kernel_submit_accelerator import machine_shape_requires_offline as _machine_shape_requires_offline
from kagglebot.kernel_submit_accelerator import resolve_submit_kernel_accelerator as _resolve_submit_accelerator
from kagglebot.kernel_submit_accelerator import (
    resolve_submit_kernel_machine_shape_decision as _resolve_submit_machine_shape_decision,
)
from kagglebot.kernel_submit_accelerator import (
    submit_hardware_profile_for_machine_shape as _submit_hardware_profile_for_machine_shape,
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
from kagglebot.paths import CompetitionPaths
from kagglebot.runtime_policy import local_gpu_time_budget_limit_min
from kagglebot.solution_guard import ensure_solution_path_allowed
from kagglebot.submission_format import load_submission_format_hint
from kagglebot.submission_output_naming import (
    all_submission_output_suffixes,
    output_filename_from_format_text,
    tabular_submission_output_suffixes,
)
from kagglebot.submission_sample_discovery import tabular_suffix
from kagglebot.training_route import plan_requests_non_training, validate_non_training_source
from kagglebot.validators import ensure_kernel_sources_valid, validate_kernel_package

_LOCAL_KERNEL_HEARTBEAT_INTERVAL_SEC = 30.0
_LOCAL_FORMAT_SUBMISSION_OUTPUT_SUFFIXES = all_submission_output_suffixes()
_LOCAL_SAMPLE_SUBMISSION_OUTPUT_SUFFIXES = tabular_submission_output_suffixes()
_KERNEL_LOG_OUTPUT_FILE_PATTERN = r"(^|/)(stdout\.txt|stderr\.txt|output\.log|log\.txt|logs\.txt|[^/]+\.log)$"
_LOCAL_EXECUTION_MANIFEST = "local_launch_manifest.json"
_LOCAL_EXECUTION_ENV_EXCLUDED = {
    "KAGGLEBOT_COMPETITION_SLUG",
    "KAGGLEBOT_ITERATION",
    "KAGGLEBOT_LOCAL_KERNEL",
    "KAGGLEBOT_LOCAL_KERNEL_MAX_RSS_MB",
    "KAGGLEBOT_LOCAL_KERNEL_STALL_SEC",
    "KAGGLEBOT_LOCAL_NOFILE",
    "KAGGLEBOT_LOCAL_WORKING_DIR",
    "KAGGLEBOT_OUTPUT_DIR",
    "KAGGLEBOT_RESUME",
    "KAGGLEBOT_RESUME_RUN_ID",
    "KAGGLEBOT_RESUME_SLUG",
    "KAGGLEBOT_RUN_ID",
    "KAGGLEBOT_SLUG",
    "KAGGLEBOT_SUBMISSION_FILENAME",
    "KAGGLEBOT_TORCH_SHARING_STRATEGY",
}
_LOCAL_EXECUTION_ENV_EXCLUDED_PREFIXES = (
    "KAGGLEBOT_KAGGLE_",
    "KAGGLEBOT_ORACLE_",
    "KAGGLEBOT_WATCH_",
)
_LOCAL_EXECUTION_ENV_SECRET_SEGMENTS = {
    "AUTH",
    "COOKIE",
    "CREDENTIAL",
    "CREDENTIALS",
    "KEY",
    "PASSWORD",
    "SECRET",
    "TOKEN",
}


def _local_execution_manifest_matches(manifest: dict[str, object], launch_signature: str) -> bool:
    return manifest.get("launch_signature") == launch_signature


def _local_execution_dir_is_empty(path: Path) -> bool:
    return not path.exists() or not any(path.iterdir())


def _training_kagglebot_env(env: dict[str, str]) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    for key in sorted(env):
        if not key.startswith("KAGGLEBOT_"):
            continue
        if key in _LOCAL_EXECUTION_ENV_EXCLUDED or key.startswith(_LOCAL_EXECUTION_ENV_EXCLUDED_PREFIXES):
            continue
        if _LOCAL_EXECUTION_ENV_SECRET_SEGMENTS.intersection(key.split("_")):
            continue
        selected.append((key, env[key]))
    return selected


def _latest_local_autofix_attempt(run_dir: Path) -> int | None:
    autofix_dir = run_dir / "autofix"
    if not autofix_dir.is_dir():
        return None
    attempts = []
    for path in autofix_dir.iterdir():
        match = re.fullmatch(r"attempt-(\d+)", path.name)
        if path.is_dir() and match:
            attempts.append(int(match.group(1)))
    return max(attempts, default=None)


def _atomic_write_local_execution_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _prepare_local_execution_output(
    *,
    base_output_dir: Path,
    kernel_path: Path,
    plan_path: Path,
    hardware_profile: str,
    accelerator: str,
    iteration: int,
    autofix_attempt: int | None,
    env: dict[str, str],
) -> tuple[Path, str, bool]:
    """Select a checkpoint-compatible output namespace for one local execution."""
    training_env = _training_kagglebot_env(env)
    training_env_sha256 = sha256_text(json.dumps(training_env, ensure_ascii=True, separators=(",", ":")))
    signature_inputs: dict[str, object] = {
        "schema_version": 1,
        "kernel_sha256": sha256_path(kernel_path),
        "plan_sha256": sha256_path(plan_path) if plan_path.is_file() else None,
        "hardware_profile": hardware_profile,
        "accelerator": accelerator,
        "training_env_sha256": training_env_sha256,
    }
    launch_signature = sha256_text(
        json.dumps(signature_inputs, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    )

    base_output_dir.mkdir(parents=True, exist_ok=True)
    base_manifest = load_json_object_or_empty(base_output_dir / _LOCAL_EXECUTION_MANIFEST)
    base_matches = _local_execution_manifest_matches(base_manifest, launch_signature)
    selected_output_dir = base_output_dir
    if not base_matches and not _local_execution_dir_is_empty(base_output_dir):
        candidate_stem = f"launch-{launch_signature[:16]}"
        collision = 1
        while True:
            suffix = "" if collision == 1 else f"-{collision}"
            candidate = base_output_dir / f"{candidate_stem}{suffix}"
            candidate_manifest = load_json_object_or_empty(candidate / _LOCAL_EXECUTION_MANIFEST)
            candidate_matches = _local_execution_manifest_matches(candidate_manifest, launch_signature)
            if candidate_matches or _local_execution_dir_is_empty(candidate):
                selected_output_dir = candidate
                break
            collision += 1

    selected_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = selected_output_dir / _LOCAL_EXECUTION_MANIFEST
    previous_manifest = load_json_object_or_empty(manifest_path)
    same_identity = _local_execution_manifest_matches(previous_manifest, launch_signature)
    try:
        previous_launch_count = int(previous_manifest.get("launch_count", 0))
    except (TypeError, ValueError):
        previous_launch_count = 0
    manifest = {
        **signature_inputs,
        "launch_signature": launch_signature,
        "kernel_path": str(kernel_path.resolve()),
        "plan_path": str(plan_path.resolve()),
        "iteration": int(iteration),
        "autofix_attempt": autofix_attempt,
        "training_env_keys": [key for key, _ in training_env],
        "launch_count": previous_launch_count + 1 if same_identity else 1,
        "resume_enabled_for_launch": same_identity,
        "resume_policy": "same_launch_signature_only",
    }
    _atomic_write_local_execution_manifest(manifest_path, manifest)
    return selected_output_dir, launch_signature, same_identity


@dataclass(frozen=True)
class KernelRunResult:
    kernel_id: str
    output_dir: Path
    submission_path: Path | None
    metrics_path: Path | None
    fidelity_expected_path: Path | None = None
    fidelity_runtime_path: Path | None = None
    fidelity_report_path: Path | None = None
    source_fingerprint: str | None = None
    requested_accelerator: str | None = None
    executed_accelerator: str | None = None
    capacity_fallback_used: bool = False


@dataclass(frozen=True)
class KernelPreparation:
    kernel_dir: Path
    output_dir: Path
    logs_dir: Path
    kernel_slug: str
    kernel_id: str
    runtime_bootstrap_mode: str = "force_train"
    supersede_stale_queued: bool = False
    source_fingerprint: str | None = None
    supersede_completed_on_source_change: bool = False
    machine_shape: str | None = None
    output_file_pattern: str | None = None
    fidelity_expected_path: Path | None = None
    requested_accelerator: str | None = None
    executed_accelerator: str | None = None
    capacity_fallback_used: bool = False


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
    expected_output_file: str | None = None
    requested_accelerator: str | None = None
    capacity_fallback_used: bool = False


@dataclass(frozen=True)
class KernelPackageBuilder:
    def prepare(self, config: KernelBuildConfig) -> KernelPreparation:
        kernel_dir = config.base_dir / config.slug / "kernels" / config.run_id
        output_dir = config.base_dir / config.slug / "runs" / config.run_id / f"iter-{config.iteration}" / "output"
        logs_dir = config.base_dir / config.slug / "runs" / config.run_id / f"iter-{config.iteration}" / "logs"
        context_dir = config.base_dir / config.slug / "context"
        if kernel_dir.exists():
            shutil.rmtree(kernel_dir)
        kernel_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        if not config.dry_run and not check_rules_accepted(config.slug, dry_run=False):
            raise RulesNotAcceptedError("Competition rules not accepted.")

        plan_path = config.base_dir / config.slug / "plan.json"
        plan_payload = load_json_object_or_empty(plan_path)
        machine_shape_decision = _resolve_submit_machine_shape_decision(
            env_get=os.getenv,
            accelerator=config.accelerator,
            hardware_profile=config.hardware_profile,
            plan=plan_payload,
            competition_slug=config.slug,
        )
        machine_shape = machine_shape_decision.machine_shape
        enable_internet = bool(config.enable_internet)
        if _machine_shape_requires_offline(machine_shape, competition_slug=config.slug) and enable_internet:
            enable_internet = False
            print(
                "[yellow]kernel internet[/yellow]: forced off because "
                f"{machine_shape} sessions must not enable internet"
            )
        print(
            "[cyan]kernel accelerator[/cyan]: "
            f"accelerator={config.accelerator} "
            f"machine_shape={machine_shape or 'KaggleDefault'} "
            f"source={machine_shape_decision.source}"
        )

        if not config.dry_run:
            print(f"[cyan]kernel init[/cyan]: {kernel_dir}")
            kernels_init(kernel_dir, dry_run=False)

        kernel_slug = _kernel_metadata.resolve_kernel_slug(
            config.kernel_name,
            config.slug,
            config.run_id,
            config.iteration,
            machine_shape=machine_shape,
        )
        kernel_id = f"{config.kaggle_username}/{kernel_slug}"
        custom_kernel_dir = config.base_dir / config.slug / "kernel"
        custom_kernel_path = custom_kernel_dir / "kernel.py"
        source_config = load_kernel_source_config(plan_path)
        ensure_solution_path_allowed(custom_kernel_dir, artifacts_dir=config.base_dir, slug=config.slug)
        if not custom_kernel_path.exists():
            raise KernelFailedError(
                "Authoritative kernel entrypoint is missing. "
                f"Expected: {custom_kernel_path}. "
                "Generate/update artifacts/<slug>/kernel/kernel.py before running training."
            )
        if plan_requests_non_training(plan_payload):
            source_issues = validate_non_training_source(
                custom_kernel_path.read_text(encoding="utf-8", errors="ignore")
            )
            if source_issues:
                raise KernelFailedError("; ".join(source_issues))
        _kernel_package_files.copy_kernel_sources(custom_kernel_dir, kernel_dir)
        _kernel_package_files.copy_shared_kernel_runtime_modules(kernel_dir)
        _kernel_package_files.copy_competition_external_assets(
            base_dir=config.base_dir,
            slug=config.slug,
            kernel_dir=kernel_dir,
        )
        _kernel_package_files.sync_plan_snapshot(
            plan_path=plan_path,
            targets=[kernel_dir / "plan.json"],
        )
        _kernel_bootstrap.ensure_kernel_import_path(kernel_dir)
        _kernel_bootstrap.inject_competition_slug_env(kernel_dir, config.slug)
        kernel_hardware_profile = _submit_hardware_profile_for_machine_shape(machine_shape) or config.hardware_profile
        _kernel_bootstrap.inject_hardware_profile_env(
            kernel_dir,
            kernel_hardware_profile,
            compute="kaggle_gpu" if config.accelerator == "gpu" else "kaggle_tpu",
        )
        _kernel_module_inliner.inline_kernel_modules(kernel_dir)
        _local_kernel_data_resolver.inject_data_dir_resolver(kernel_dir)
        _local_kernel_pipeline_cfg.inject_staged_plan_payload_fallback(kernel_dir)
        _local_kernel_pipeline_cfg.inject_staged_plan_path_fallback(kernel_dir)
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
        if plan_requests_non_training(plan_payload):
            _kernel_bootstrap.ensure_kernel_non_training_env(kernel_dir)
        else:
            _kernel_bootstrap.ensure_kernel_force_train_env(kernel_dir)
        _local_kernel_shims.ensure_training_progress_shim(kernel_dir)
        ensure_kernel_sources_valid(kernel_dir)
        _kernel_metadata.write_kernel_metadata(
            kernel_dir=kernel_dir,
            kernel_id=kernel_id,
            title=kernel_slug,
            code_file="kernel.py",
            kernel_type="script",
            accelerator=config.accelerator,
            enable_internet=enable_internet,
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
            machine_shape=machine_shape,
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

        if not config.submission_path.exists() or not (
            config.submission_path.is_file() or config.submission_path.is_dir()
        ):
            raise KernelFailedError(f"Submission artifact not found: {config.submission_path}")

        submit_mode = str(config.mode or "wrapper").strip().lower()
        code_output_mode = submit_mode in {"gateway", "inference"}
        resolved_expected_output_file = (
            config.expected_output_file or config.submission_path.name
            if code_output_mode
            else config.expected_output_file
        )
        plan_path = config.base_dir / config.slug / "plan.json"
        plan_payload = load_json_object_or_empty(plan_path)
        submit_accelerator = _resolve_submit_accelerator(config.accelerator, env_get=os.getenv)
        machine_shape_decision = _resolve_submit_machine_shape_decision(
            env_get=os.getenv,
            accelerator=submit_accelerator,
            hardware_profile=config.hardware_profile,
            plan=plan_payload,
            competition_slug=config.slug,
        )
        submit_machine_shape = machine_shape_decision.machine_shape
        if not code_output_mode:
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
            machine_shape=submit_machine_shape,
        )
        kernel_id = f"{config.kaggle_username}/{kernel_slug}"
        print(
            "[cyan]kernel accelerator[/cyan]: "
            f"accelerator={submit_accelerator} "
            f"machine_shape={submit_machine_shape or 'KaggleDefault'} "
            f"source={machine_shape_decision.source}"
        )
        expected_metrics: dict[str, object] | None = None
        if code_output_mode:
            custom_kernel_dir = config.base_dir / config.slug / "kernel"
            custom_kernel_path = custom_kernel_dir / "kernel.py"
            context_dir = config.base_dir / config.slug / "context"
            source_config = load_kernel_source_config(plan_path)
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
                plan_path=plan_path,
                targets=[kernel_dir / "plan.json"],
            )
            _kernel_bootstrap.ensure_kernel_import_path(kernel_dir)
            _kernel_bootstrap.inject_competition_slug_env(kernel_dir, config.slug)
            submit_hardware_profile = _submit_hardware_profile_for_machine_shape(submit_machine_shape)
            if submit_hardware_profile is not None:
                _kernel_bootstrap.inject_hardware_profile_env(
                    kernel_dir,
                    submit_hardware_profile,
                    compute="kaggle_gpu",
                )
            _kernel_module_inliner.inline_kernel_modules(kernel_dir)
            _local_kernel_data_resolver.inject_data_dir_resolver(kernel_dir)
            _local_kernel_pipeline_cfg.inject_staged_plan_payload_fallback(kernel_dir)
            _local_kernel_pipeline_cfg.inject_staged_plan_path_fallback(kernel_dir)
            _local_kernel_pipeline_cfg.inject_pipeline_cfg_fallback(kernel_dir)
            _local_kernel_shims.inject_context_io_shims(kernel_dir, context_dir)
            _local_kernel_shims.inject_submit_inference_compat_shims(kernel_dir)
            _local_kernel_drift_guard.prepare_zero_overlap_drift_guard(
                base_dir=config.base_dir,
                slug=config.slug,
                context_dir=context_dir,
            )
            _local_kernel_shims.inject_zero_overlap_drift_shim(kernel_dir, context_dir)
            expected_metrics = _submit_kernel_fidelity.load_expected_submit_metrics_snapshot(
                [
                    config.base_dir
                    / config.slug
                    / "runs"
                    / config.run_id
                    / f"iter-{config.iteration}"
                    / "metrics.json",
                    config.base_dir
                    / config.slug
                    / "kernels"
                    / config.run_id
                    / f"local-iter-{config.iteration}"
                    / "outputs"
                    / "metrics.json",
                    output_dir / "metrics.json",
                ]
            )
            _submit_kernel_fidelity.validate_reference_submission_readiness(
                reproduction_report_path=context_dir / "reference_reproduction_report.json",
                expected_metrics=expected_metrics,
            )
            runtime_env = _submit_kernel_fidelity.build_submit_runtime_env(expected_metrics)
            runtime_env.update(
                {
                    "KAGGLEBOT_FIDELITY_REQUESTED_ACCELERATOR": config.requested_accelerator or config.accelerator,
                    "KAGGLEBOT_FIDELITY_EXECUTED_ACCELERATOR": submit_accelerator,
                    "KAGGLEBOT_FIDELITY_MACHINE_SHAPE": submit_machine_shape or "",
                    "KAGGLEBOT_FIDELITY_CAPACITY_FALLBACK_USED": ("1" if config.capacity_fallback_used else "0"),
                }
            )
            _kernel_bootstrap.inject_submit_inference_env(kernel_dir, runtime_env=runtime_env)
            _kernel_bootstrap.inject_submit_runtime_fidelity(kernel_dir)
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
        _kernel_metadata.write_kernel_metadata(
            kernel_dir=kernel_dir,
            kernel_id=kernel_id,
            title=kernel_slug,
            code_file="kernel.py",
            kernel_type="script",
            accelerator=submit_accelerator,
            # Submission notebooks must remain offline. In particular, code
            # competitions reject otherwise valid notebook versions that have
            # internet enabled in their metadata.
            enable_internet=False,
            competition_slug=config.slug,
            source_config=source_config,
        )
        validate_kernel_package(kernel_dir)
        # Source validation compiles Python files and therefore creates timestamped
        # bytecode. Those local caches are neither part of the executable source
        # contract nor stable across otherwise identical preparation attempts.
        _kernel_package_files.remove_generated_kernel_cache_files(kernel_dir)
        fidelity_expected_path = None
        if code_output_mode:
            fidelity_expected_path = _submit_kernel_fidelity.stage_submit_fidelity_expected_contract(
                package_dir=kernel_dir,
                slug=config.slug,
                run_id=config.run_id,
                iteration=config.iteration,
                kernel_id=kernel_id,
                artifact_mode=submit_mode,
                expected_output_file=str(resolved_expected_output_file),
                expected_metrics=expected_metrics,
                selected_candidate_path=config.submission_path,
                requested_accelerator=config.requested_accelerator or config.accelerator,
                executed_accelerator=submit_accelerator,
                machine_shape=submit_machine_shape,
                capacity_fallback_used=config.capacity_fallback_used,
            )
        source_fingerprint = _kernel_source_fingerprint(
            kernel_dir,
            mode=submit_mode,
            expected_output_file=resolved_expected_output_file,
            machine_shape=submit_machine_shape,
        )
        return KernelPreparation(
            kernel_dir=kernel_dir,
            output_dir=output_dir,
            logs_dir=logs_dir,
            kernel_slug=kernel_slug,
            kernel_id=kernel_id,
            runtime_bootstrap_mode="submit_inference" if code_output_mode else "none",
            supersede_stale_queued=True,
            source_fingerprint=source_fingerprint,
            supersede_completed_on_source_change=code_output_mode,
            machine_shape=submit_machine_shape,
            output_file_pattern=_submit_kernel_output_file_pattern(resolved_expected_output_file),
            fidelity_expected_path=fidelity_expected_path,
            requested_accelerator=config.requested_accelerator or config.accelerator,
            executed_accelerator=submit_accelerator,
            capacity_fallback_used=config.capacity_fallback_used,
        )


@dataclass(frozen=True)
class KernelJobMonitor:
    def push_and_wait(
        self,
        *,
        preparation: KernelPreparation,
        slug: str,
        timeout_minutes: int | None,
        on_remote_started: Callable[[str], None] | None = None,
    ) -> str:
        _kernel_bootstrap.ensure_kernel_competition_slug_env(preparation.kernel_dir, slug)
        if preparation.runtime_bootstrap_mode == "force_train":
            _kernel_bootstrap.ensure_kernel_force_train_env(preparation.kernel_dir)
        elif preparation.runtime_bootstrap_mode == "submit_inference":
            _kernel_bootstrap.ensure_kernel_submit_inference_env(preparation.kernel_dir)
        _kernel_remote_ops.clear_stale_kernel_output(preparation.output_dir)
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
                on_remote_started=on_remote_started,
            )
            if resumed_kernel_id:
                return resumed_kernel_id

        print(f"[cyan]kernel push[/cyan]: {preparation.kernel_dir}")
        push_output = _push_kernel_preparation(preparation, slug=slug)
        _kernel_remote_ops.write_push_log(preparation.logs_dir, push_attempt, push_output)
        _raise_for_invalid_kernel_push_sources(push_output, kernel_dir=preparation.kernel_dir)
        pushed_kernel_id = _remote_kernel_state.extract_kernel_id_from_push(push_output)
        if pushed_kernel_id and pushed_kernel_id != kernel_id:
            print(f"[cyan]kernel id[/cyan]: {pushed_kernel_id}")
            kernel_id = pushed_kernel_id
        _record_remote_kernel_source(preparation, kernel_id=kernel_id)
        kernel_id = _kernel_remote_ops.resolve_kernel_id(
            kernel_id,
            preparation.kernel_slug,
            kernel_id_by_title_func=kernel_id_by_title,
        )
        resolved_id = _wait_for_kernel_registration(kernel_id, preparation.kernel_slug)
        if not resolved_id:
            print("[yellow]kernel not found after push[/yellow]: retrying once")
            push_attempt += 1
            push_output = _push_kernel_preparation(preparation, slug=slug)
            _kernel_remote_ops.write_push_log(preparation.logs_dir, push_attempt, push_output)
            _raise_for_invalid_kernel_push_sources(push_output, kernel_dir=preparation.kernel_dir)
            pushed_kernel_id = _remote_kernel_state.extract_kernel_id_from_push(push_output)
            if pushed_kernel_id and pushed_kernel_id != kernel_id:
                print(f"[cyan]kernel id[/cyan]: {pushed_kernel_id}")
                kernel_id = pushed_kernel_id
            _record_remote_kernel_source(preparation, kernel_id=kernel_id)
            kernel_id = _kernel_remote_ops.resolve_kernel_id(
                kernel_id,
                preparation.kernel_slug,
                kernel_id_by_title_func=kernel_id_by_title,
            )
            resolved_id = _wait_for_kernel_registration(kernel_id, preparation.kernel_slug)
            if not resolved_id:
                raise KernelFailedError("Kaggle kernel not found after push; aborting.")
            kernel_id = resolved_id
        else:
            kernel_id = resolved_id
        _record_remote_kernel_source(preparation, kernel_id=kernel_id)

        print(f"[cyan]kernel status[/cyan]: {kernel_id}")
        if on_remote_started is not None:
            on_remote_started(kernel_id)
        _wait_for_kernel_and_record_pending(
            preparation=preparation,
            kernel_id=kernel_id,
            slug=slug,
            timeout_minutes=timeout_minutes,
        )
        _remote_kernel_state.clear_pending_remote_kernel(preparation.logs_dir)
        print(f"[cyan]kernel output[/cyan]: {preparation.output_dir}")
        kernels_output(
            kernel_id,
            preparation.output_dir,
            slug=slug,
            dry_run=False,
            file_pattern=preparation.output_file_pattern,
        )
        return kernel_id


@dataclass(frozen=True)
class KernelLogParser:
    @staticmethod
    def collect_tail(output_dir: Path, max_lines: int = 50) -> str | None:
        return _kernel_logs.collect_log_tail(output_dir, max_lines=max_lines)


def resolve_kaggle_username(explicit: str | None) -> str:
    return _resolve_kaggle_username(explicit)


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
    on_remote_started: Callable[[str], None] | None = None,
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
        on_remote_started=on_remote_started,
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
    expected_output_file: str | None = None,
    hardware_profile: str | None = "auto",
    requested_accelerator: str | None = None,
    capacity_fallback_used: bool = False,
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
        hardware_profile=hardware_profile,
        expected_output_file=expected_output_file,
        requested_accelerator=requested_accelerator,
        capacity_fallback_used=capacity_fallback_used,
    )
    preparation = KernelSubmitPackageBuilder().prepare(build_config)
    if dry_run:
        return KernelRunResult(
            kernel_id=preparation.kernel_id,
            output_dir=preparation.output_dir,
            submission_path=None,
            metrics_path=None,
            fidelity_expected_path=preparation.fidelity_expected_path,
            source_fingerprint=preparation.source_fingerprint,
            requested_accelerator=preparation.requested_accelerator,
            executed_accelerator=preparation.executed_accelerator,
            capacity_fallback_used=preparation.capacity_fallback_used,
        )

    kernel_id = KernelJobMonitor().push_and_wait(
        preparation=preparation,
        slug=slug,
        timeout_minutes=timeout_minutes,
    )
    resolved_submission_path = (
        _find_output_file(preparation.output_dir, expected_output_file)
        if expected_output_file
        else find_submission_file(preparation.output_dir)
    )
    metrics_path = _find_output_file(preparation.output_dir, "metrics.json")
    fidelity_runtime_path = _find_output_file(
        preparation.output_dir,
        _submit_kernel_fidelity.RUNTIME_FILE_NAME,
    )
    return KernelRunResult(
        kernel_id=kernel_id,
        output_dir=preparation.output_dir,
        submission_path=resolved_submission_path,
        metrics_path=metrics_path,
        fidelity_expected_path=preparation.fidelity_expected_path,
        fidelity_runtime_path=fidelity_runtime_path,
        source_fingerprint=preparation.source_fingerprint,
        requested_accelerator=preparation.requested_accelerator,
        executed_accelerator=preparation.executed_accelerator,
        capacity_fallback_used=preparation.capacity_fallback_used,
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
    competition_policy = load_competition_policy(CompetitionPaths(slug=slug, artifacts_dir=base_dir))
    policy_kernel_contract = competition_policy.execution_hint("kernel_contract")
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    _local_sample_submission.ensure_local_sample_submission_file(base_dir=base_dir, slug=slug)
    _local_kernel_context.stage_local_kernel_data_dir(base_dir=base_dir, slug=slug, run_dir=run_dir)
    _local_kernel_context.stage_local_kernel_context_profile(base_dir=base_dir, slug=slug, run_dir=run_dir)
    _local_kernel_context.stage_local_kernel_reference_inputs(base_dir=base_dir, slug=slug, run_dir=run_dir)

    ensure_solution_path_allowed(kernel_source_dir, artifacts_dir=base_dir, slug=slug)
    kernel_path = kernel_source_dir / "kernel.py"
    if not kernel_path.exists():
        raise KernelFailedError(f"Local kernel execution requires {kernel_path} to exist.")
    source_plan = load_json_object_or_empty(base_dir / slug / "plan.json")
    if plan_requests_non_training(source_plan):
        source_issues = validate_non_training_source(kernel_path.read_text(encoding="utf-8", errors="ignore"))
        if source_issues:
            raise KernelFailedError("; ".join(source_issues))
    durable_resume_root = _local_kernel_resume.durable_state_root(
        base_dir=base_dir,
        slug=slug,
        run_id=run_id,
        iteration=iteration,
    )
    if kernel_stage_dir.exists():
        preserved = _local_kernel_resume.preserve_local_kernel_checkpoints(
            kernel_stage_dir=kernel_stage_dir,
            durable_root=durable_resume_root,
        )
        if preserved is not None:
            print(f"[cyan]kernel local resume[/cyan]: preserved checkpoints at {preserved}")
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
    _local_kernel_pipeline_cfg.inject_staged_plan_path_fallback(kernel_stage_dir)
    _local_kernel_pipeline_cfg.inject_pipeline_cfg_fallback(kernel_stage_dir)
    _local_kernel_shims.inject_context_io_shims(kernel_stage_dir, context_dir)
    _local_kernel_shims.inject_local_runtime_shims(kernel_stage_dir)
    _local_kernel_shims.inject_training_compat_shims(kernel_stage_dir)
    _local_kernel_drift_guard.prepare_zero_overlap_drift_guard(base_dir=base_dir, slug=slug, context_dir=context_dir)
    _local_kernel_shims.inject_zero_overlap_drift_shim(kernel_stage_dir, context_dir)
    _kernel_bootstrap.inject_competition_slug_env(kernel_stage_dir, slug)
    staged_plan = load_json_object_or_empty(kernel_stage_dir / "plan.json")
    if plan_requests_non_training(staged_plan):
        _kernel_bootstrap.ensure_kernel_non_training_env(kernel_stage_dir)
    else:
        _kernel_bootstrap.ensure_kernel_force_train_env(kernel_stage_dir)
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

    effective_timeout_minutes = timeout_minutes
    if effective_timeout_minutes is None:
        effective_timeout_minutes = local_gpu_time_budget_limit_min()
    timeout_sec = None if effective_timeout_minutes is None else max(60, int(effective_timeout_minutes * 60))
    if effective_timeout_minutes is not None:
        print(f"[cyan]kernel local budget[/cyan]: hard timeout={effective_timeout_minutes}m")
    kernel_fingerprint = sha256_path(kernel_path)
    eta_total_sec, eta_samples = _local_kernel_duration.estimate_local_kernel_duration_seconds(
        base_dir=base_dir,
        slug=slug,
        kernel_fingerprint=kernel_fingerprint,
    )
    exact_source_timeout_count = _local_kernel_duration.exact_source_timeout_count(
        base_dir=base_dir,
        slug=slug,
        kernel_fingerprint=kernel_fingerprint,
    )
    if _local_kernel_duration.exact_source_exceeds_timeout(
        estimated_duration_sec=eta_total_sec,
        sample_count=eta_samples,
        timeout_sec=timeout_sec,
        timeout_count=exact_source_timeout_count,
    ):
        evidence = (
            f"timed out {exact_source_timeout_count} times"
            if exact_source_timeout_count >= 2
            else f"has historical median {eta_total_sec / 60:.1f}m across {eta_samples} completed runs"
        )
        raise KernelFailedError(
            f"Local kernel preflight rejected an exact source version that {evidence}; "
            f"the hard timeout is {timeout_sec / 60:.1f}m. Keep the accuracy strategy, but fix its runtime contract or "
            "checkpoint/resume behavior before retrying."
        )
    env = os.environ.copy()
    env.setdefault("KAGGLEBOT_LOCAL_KERNEL", "1")
    env.setdefault("KAGGLEBOT_SLUG", slug)
    env.setdefault("KAGGLEBOT_COMPETITION_SLUG", slug)
    env.setdefault("KAGGLEBOT_RUN_ID", run_id)
    env.setdefault("KAGGLEBOT_ITERATION", str(iteration))
    env.setdefault("KAGGLEBOT_ACCELERATOR", accelerator)
    local_submission_filename = _local_submission_filename_from_sample(base_dir=base_dir, slug=slug)
    if local_submission_filename is not None:
        env.setdefault("KAGGLEBOT_SUBMISSION_FILENAME", local_submission_filename)
    resolved_hardware_profile = resolve_hardware_profile(hardware_profile, compute="local_gpu")
    for key, value in hardware_env(resolved_hardware_profile).items():
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
    output_dir, launch_signature, resume_enabled = _prepare_local_execution_output(
        base_output_dir=output_dir,
        kernel_path=kernel_path,
        plan_path=kernel_stage_dir / "plan.json",
        hardware_profile=resolved_hardware_profile.key,
        accelerator=accelerator,
        iteration=iteration,
        autofix_attempt=_latest_local_autofix_attempt(output_dir.parent.parent),
        env=env,
    )
    env["KAGGLEBOT_OUTPUT_DIR"] = str(output_dir)
    env["KAGGLEBOT_RESUME"] = "1" if resume_enabled else "0"
    if resume_enabled:
        restored = _local_kernel_resume.restore_local_kernel_checkpoints(
            kernel_stage_dir=kernel_stage_dir,
            durable_root=durable_resume_root,
        )
        if restored is not None:
            print(f"[cyan]kernel local resume[/cyan]: restored exact-source checkpoints into {restored}")
    print(
        "[cyan]kernel local execution[/cyan]: "
        f"output={output_dir} signature={launch_signature[:16]} "
        f"resume={'enabled' if resume_enabled else 'disabled'}"
    )
    memory_cap_bytes = _local_kernel_limits.resolve_memory_cap_bytes(env)
    if memory_cap_bytes is not None:
        print(f"[yellow]kernel local[/yellow]: host memory guard active at {memory_cap_bytes // (1024 * 1024)} MiB RSS")

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

        def run_once_with_watchdog(*, current_env: dict[str, str]) -> _local_kernel_process.LocalKernelExecResult:
            return _local_kernel_process.run_local_kernel_once(
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
        _local_kernel_duration.append_local_kernel_duration_history(
            base_dir=base_dir,
            slug=slug,
            run_id=run_id,
            iteration=iteration,
            duration_sec=max(0.0, time.monotonic() - monotonic_start),
            kernel_fingerprint=kernel_fingerprint,
            outcome="timeout",
        )
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
        kernel_fingerprint=kernel_fingerprint,
    )
    print(f"[cyan]kernel local complete[/cyan]: elapsed={result.duration_sec:.0f}s")

    explicit_submission_src = None
    if local_submission_filename is not None:
        explicit_submission_src = _resolve_local_kernel_artifact_file(
            kernel_dir=kernel_stage_dir,
            output_dir=output_dir,
            started_at=started_at,
            filename=local_submission_filename,
        )
    submission_src, metrics_src = _resolve_local_kernel_artifacts(
        kernel_dir=kernel_stage_dir,
        output_dir=output_dir,
        started_at=started_at,
    )
    if explicit_submission_src is not None:
        submission_src = explicit_submission_src
    if submission_src is None:
        raise KernelFailedError("Local kernel completed but submission output was not found.")

    submission_dst, metrics_dst = _copy_local_kernel_primary_artifacts(
        submission_path=submission_src,
        metrics_path=metrics_src,
        output_dir=output_dir,
    )
    if metrics_dst is not None:
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
        policy_contract=policy_kernel_contract,
    )
    _copy_optional_local_kernel_artifacts(
        kernel_dir=kernel_stage_dir,
        output_dir=output_dir,
        started_at=started_at,
    )

    return KernelRunResult(
        kernel_id=f"local/{slug}",
        output_dir=output_dir,
        submission_path=submission_dst,
        metrics_path=metrics_dst,
    )


def _local_submission_filename_from_sample(*, base_dir: Path, slug: str) -> str | None:
    format_filename = _local_submission_filename_from_format(base_dir=base_dir, slug=slug)
    if format_filename is not None:
        return format_filename
    source = _local_sample_submission.resolve_sample_submission_source(
        context_dir=base_dir / slug / "context",
        data_dir=base_dir / slug / "data",
    )
    if source is None:
        return None
    suffix = tabular_suffix(source)
    if suffix not in _LOCAL_SAMPLE_SUBMISSION_OUTPUT_SUFFIXES:
        return None
    return f"submission{suffix}"


def _local_submission_filename_from_format(*, base_dir: Path, slug: str) -> str | None:
    format_path = base_dir / slug / "context" / "submission_format.md"
    hint = load_submission_format_hint(format_path)
    if hint is None or not hint.expected_suffixes:
        return None
    try:
        text = format_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        text = ""
    return output_filename_from_format_text(
        text,
        expected_suffixes=hint.expected_suffixes,
        allowed_suffixes=_LOCAL_FORMAT_SUBMISSION_OUTPUT_SUFFIXES,
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
            log_output_file_pattern=_KERNEL_LOG_OUTPUT_FILE_PATTERN,
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
    on_remote_started: Callable[[str], None] | None = None,
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

    source_changed = _remote_kernel_source_changed(preparation, kernel_id=kernel_id)
    if source_changed and is_kernel_status_complete(status):
        _remote_kernel_state.clear_pending_remote_kernel(preparation.logs_dir)
        print(
            "[yellow]kernel resume[/yellow]: prior completed kernel used stale sources or output contract; "
            "pushing a new version"
        )
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

    _kernel_remote_ops.clear_stale_kernel_output(preparation.output_dir)
    print(f"[yellow]kernel resume[/yellow]: waiting for existing remote kernel {kernel_id} ({status})")
    if on_remote_started is not None:
        on_remote_started(kernel_id)
    if is_kernel_status_running(status):
        _wait_for_kernel_and_record_pending(
            preparation=preparation,
            kernel_id=kernel_id,
            slug=slug,
            timeout_minutes=timeout_minutes,
            initial_queued_since=initial_queued_since,
        )
    _remote_kernel_state.clear_pending_remote_kernel(preparation.logs_dir)
    if source_changed:
        print(
            "[yellow]kernel resume[/yellow]: prior running kernel finished with stale sources or output contract; "
            "pushing a new version"
        )
        return None
    print(f"[cyan]kernel output[/cyan]: {preparation.output_dir}")
    kernels_output(
        kernel_id,
        preparation.output_dir,
        slug=slug,
        dry_run=False,
        file_pattern=preparation.output_file_pattern,
    )
    return kernel_id


def _kernel_source_fingerprint(
    kernel_dir: Path,
    *,
    mode: str,
    expected_output_file: str | None,
    machine_shape: str | None,
) -> str:
    package_digest = sha256_path(kernel_dir)
    contract = (
        f"kagglebot-submit-kernel-v3\0{mode}\0{expected_output_file or ''}\0{machine_shape or ''}\0{package_digest}"
    )
    return sha256_text(contract)


def _push_kernel_preparation(preparation: KernelPreparation, *, slug: str) -> str:
    if preparation.machine_shape:
        return kernels_push(
            preparation.kernel_dir,
            slug=slug,
            dry_run=False,
            accelerator=preparation.machine_shape,
        )
    return kernels_push(preparation.kernel_dir, slug=slug, dry_run=False)


def _record_remote_kernel_source(preparation: KernelPreparation, *, kernel_id: str) -> None:
    if not preparation.source_fingerprint:
        return
    _remote_kernel_state.write_remote_kernel_source(
        preparation.logs_dir,
        kernel_id=kernel_id,
        source_fingerprint=preparation.source_fingerprint,
    )


def _remote_kernel_source_changed(preparation: KernelPreparation, *, kernel_id: str) -> bool:
    if not preparation.supersede_completed_on_source_change or not preparation.source_fingerprint:
        return False
    return not _remote_kernel_state.remote_kernel_source_matches(
        preparation.logs_dir,
        kernel_id=kernel_id,
        source_fingerprint=preparation.source_fingerprint,
    )


def _wait_for_kernel(
    kernel_id: str,
    slug: str,
    timeout_minutes: int | None,
    *,
    output_dir: Path,
    log_output_file_pattern: str | None = _KERNEL_LOG_OUTPUT_FILE_PATTERN,
    initial_queued_since: float | None = None,
) -> None:
    _kernel_wait.wait_for_kernel(
        kernel_id,
        slug,
        timeout_minutes,
        output_dir=output_dir,
        initial_queued_since=initial_queued_since,
        deps=_kernel_wait.KernelWaitDependencies(
            kernels_status=kernels_status,
            try_fetch_kernel_output=lambda kernel_id, *, output_dir, slug: _try_fetch_kernel_output(
                kernel_id,
                output_dir=output_dir,
                slug=slug,
                file_pattern=log_output_file_pattern,
            ),
            print_kernel_logs=_kernel_logs.print_kernel_logs,
            detect_failure_in_logs=_kernel_logs.detect_failure_in_logs,
            collect_log_tail=_kernel_logs.collect_log_tail,
            monotonic=time.monotonic,
            sleep=time.sleep,
            remote_kernel_queued_timeout_sec=_remote_kernel_state.remote_kernel_queued_timeout_sec,
            raise_kernel_queued_timeout=_remote_kernel_state.raise_kernel_queued_timeout,
        ),
    )


def _wait_for_kernel_registration(kernel_id: str, kernel_slug: str) -> str | None:
    return _kernel_remote_ops.wait_for_kernel_registration(
        kernel_id,
        kernel_slug,
        deps=_kernel_remote_ops.KernelRegistrationDependencies(
            kernels_status=kernels_status,
            kernel_exists=kernel_exists,
            kernel_id_by_title=kernel_id_by_title,
            sleep=time.sleep,
        ),
    )


def _try_fetch_kernel_output(
    kernel_id: str,
    *,
    output_dir: Path,
    slug: str,
    file_pattern: str | None = _KERNEL_LOG_OUTPUT_FILE_PATTERN,
) -> None:
    _kernel_remote_ops.try_fetch_kernel_output(
        kernel_id,
        output_dir=output_dir,
        slug=slug,
        kernels_output_func=kernels_output,
        file_pattern=file_pattern,
    )


def _submit_kernel_output_file_pattern(expected_output_file: str | None) -> str | None:
    expected_name = Path(str(expected_output_file or "").strip()).name
    if not expected_name:
        return None
    return (
        rf"(^|/)({re.escape(expected_name)}|metrics\.json|submit_fidelity_runtime\.json|"
        r"kernel_error[^/]*\.txt|stderr\.txt|error[^/]*\.(txt|log)|[^/]+\.log)$"
    )
