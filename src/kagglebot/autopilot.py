from __future__ import annotations

import inspect
import math
import os
import re
import time
import traceback
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING

from rich import print

from kagglebot import agent_io as _agent_io
from kagglebot import agent_prompts as _agent_prompts
from kagglebot import agent_strategy as _agent_strategy
from kagglebot import autofix_context as _autofix_context
from kagglebot import autofix_restart as _autofix_restart
from kagglebot import autopilot_loop_settings as _autopilot_loop_settings
from kagglebot import autopilot_state as _autopilot_state
from kagglebot import autopilot_submit as _autopilot_submit
from kagglebot import campaign_metrics as _campaign_metrics
from kagglebot import code_reference as _code_reference
from kagglebot import compute_handoff as _compute_handoff
from kagglebot import diagnostics as _diagnostics
from kagglebot import env_utils as _env_utils
from kagglebot import improvement_context as _improvement_context
from kagglebot import iteration_evidence as _iteration_evidence
from kagglebot import iteration_metrics as _iteration_metrics
from kagglebot import iteration_signals as _iteration_signals
from kagglebot import json_utils as _json_utils
from kagglebot import kernel_errors as _kernel_errors
from kagglebot import kernel_fix_context as _kernel_fix_context
from kagglebot import kernel_metrics as _kernel_metrics
from kagglebot import kernel_plan_validation as _kernel_plan_validation
from kagglebot import kernel_preflight as _kernel_preflight
from kagglebot import kernel_quality as _kernel_quality
from kagglebot import kernel_snapshot as _kernel_snapshot
from kagglebot import leaderboard_anomaly as _leaderboard_anomaly
from kagglebot import leaderboard_policy as _leaderboard_policy
from kagglebot import loop_control as _loop_control
from kagglebot import method_scout as _method_scout
from kagglebot import metric_fix as _metric_fix
from kagglebot import metric_matching as _metric_matching
from kagglebot import metric_recheck as _metric_recheck
from kagglebot import oracle_workflow_state as _oracle_workflow_state
from kagglebot import plan_policy as _plan_policy
from kagglebot import plan_resolution as _plan_resolution
from kagglebot import planning_runner as _planning_runner
from kagglebot import runtime_fixes as _runtime_fixes
from kagglebot import score_progress as _score_progress
from kagglebot import score_utils as _score_utils
from kagglebot import submission_fidelity as _submission_fidelity
from kagglebot import submission_history as _submission_history
from kagglebot import submission_policy as _submission_policy
from kagglebot import submit_attempts as _submit_attempts
from kagglebot import submit_failure_context as _submit_failure_context
from kagglebot import submit_failure_policy as _submit_failure_policy  # noqa: F401
from kagglebot import submit_gate as _submit_gate
from kagglebot import submit_knowledge as _submit_knowledge
from kagglebot import submit_rank as _submit_rank
from kagglebot import submit_stage_duplicate as _submit_stage_duplicate
from kagglebot import submit_stage_messages as _submit_stage_messages
from kagglebot import submit_stage_modes as _submit_stage_modes
from kagglebot import submit_tracking as _submit_tracking
from kagglebot import verify_artifacts as _verify_artifacts
from kagglebot import watch_state as _watch_state
from kagglebot.agents.codex_runner import run_codex
from kagglebot.agents.identity import (
    IMPLEMENTATION_AGENT,
    ORACLE_IMPLEMENTATION_AGENT,
    STRATEGY_AGENT,
    prompt_identity_format_args,
)
from kagglebot.agents.strategy_runner import run_strategy
from kagglebot.artifact_io import copy_artifact_if_needed
from kagglebot.autopilot_session import AutopilotSession, SubmissionPhase
from kagglebot.campaign import (
    allocate_submission,
    build_campaign_candidate,
    campaign_state_path,
    candidate_registry_path,
    update_campaign_state,
    upsert_candidate,
)
from kagglebot.competition_policy import load_competition_policy
from kagglebot.deliverable_artifacts import resolve_deliverable_artifact_contract
from kagglebot.exceptions import (
    KaggleNetworkError,
    KernelCapacityError,
    KernelFailedError,
    KernelStillRunningError,
    MissingCompetitionDataError,
    OracleStrategyError,
    RulesNotAcceptedError,
    SubmitAbortedError,
)
from kagglebot.exec_utils import run_command
from kagglebot.experiment_executor import execute_experiment_graph
from kagglebot.experiment_graph import (
    append_campaign_outcome,
    build_experiment_graph,
    write_allocator_decision,
)
from kagglebot.hardware import render_hardware_constraints, resolve_hardware_profile
from kagglebot.hashing import sha256_file_or_none as _sha256_or_none
from kagglebot.history import new_run_id
from kagglebot.kaggle_api import (
    leaderboard_rank_for_score,
    leaderboard_top1,
)
from kagglebot.kernel_runner import KernelRunResult, run_kernel, run_kernel_local
from kagglebot.knowledge import (
    record_error_fix_insight,
    record_improvement,
    record_iteration,
    record_problem_type_insight,
    record_run,
)
from kagglebot.knowledge_phase import KnowledgePhase
from kagglebot.medals import DEFAULT_TARGET_MEDAL
from kagglebot.planning_phase import PlanningPhase
from kagglebot.runners.base import RunContext
from kagglebot.runners.local_kernel import LocalKernelRunner
from kagglebot.scalar_utils import tolerant_finite_float, tolerant_int
from kagglebot.submission_format import load_submission_format_hint
from kagglebot.submission_output_naming import (
    all_submission_output_suffixes,
    first_allowed_expected_output_suffix,
    output_filename_from_format_text,
    tabular_submission_output_suffixes,
)
from kagglebot.submission_sample_discovery import tabular_suffix
from kagglebot.top1_campaign import (
    build_blend_report,
    build_candidate_portfolio_plan,
    build_reference_reproduction_report,
    private_robustness_score,
    select_method_id_for_category,
)
from kagglebot.top1_exhaustive import (
    build_portfolio_optimizer_report,
    build_private_robustness_report,
    build_top1_exhaustion_report,
    build_win_contract,
    format_top1_public_score_message,
    write_top1_public_snapshot,
)
from kagglebot.training_route import (
    TrainingRouteDecision,
    decide_training_route,
    is_unscored_non_training_diagnostic,
    resolve_non_training_validation_blockers,
    validate_non_training_metrics,
)
from kagglebot.validation_lab import run_validation_lab
from kagglebot.write_guard import (
    _backup_guarded_files,
    _diff_snapshots,
    _enforce_allowlist_changes,
    _snapshot_tree,
    build_repair_write_policy,
)
from kagglebot.writeup import attach_published_writeup_notebook, build_writeup_bundle
from kagglebot.writeup_submission import WriteupSubmissionRequest, submit_validated_writeup

if TYPE_CHECKING:
    from kagglebot.paths import CompetitionPaths, KnowledgePaths
    from kagglebot.solver.evaluate import EvaluationResult


def _run_required_oracle_strategy(prompt_path: Path, output_dir: Path, *, dry_run: bool) -> object:
    parameters = inspect.signature(run_strategy).parameters
    if "engine" in parameters:
        return run_strategy(prompt_path, output_dir, dry_run=dry_run, engine="oracle")
    return run_strategy(prompt_path, output_dir, dry_run=dry_run)


@dataclass(frozen=True)
class AutopilotConfig:
    run_id: str | None
    slug: str
    competition_url: str | None
    paths: CompetitionPaths
    knowledge_paths: KnowledgePaths
    agent: str
    compute: str
    accelerator: str
    strict_accelerator: bool
    kaggle_username: str | None
    kernel_name: str | None
    internet: str | None
    time_budget_min: int | None
    seed: int | None
    score_source: str | None
    holdout_frac: float | None
    cv_folds: int | None
    target_metric: str | None
    target_score: float | None
    target_direction: str | None
    max_iterations: int | None
    max_total_min: int | None
    patience: int | None
    min_improvement: float | None
    submit: bool
    force_submit: bool
    message: str | None
    verify_cmd: str
    dry_run: bool
    submit_policy: str | None = None
    campaign_mode: str | None = "baseline"
    method_scout: str | None = "auto"
    research_scout: str | None = "auto"
    method_scout_max_sources: int = 12
    portfolio_execution: str | None = "serial"
    validation_lab: str | None = "auto"
    candidate_budget_min: int | None = None
    max_candidates_per_iteration: int | None = None
    top1_exhaustive: bool = False
    top1_submit_policy: str | None = "value_only"
    hardware_profile: str | None = "auto"


@dataclass(frozen=True)
class _ValidationBlockedKernelResult:
    blockers: tuple[str, ...]
    route_result: dict[str, object]
    submission_contract: dict[str, object]


@dataclass(frozen=True)
class _UnscoredDiagnosticKernelResult:
    route_result: dict[str, object]
    submission_contract: dict[str, object]
    diagnostic_archive_path: Path


MAX_KERNEL_FIX_ATTEMPTS: int | None = 8
MAX_KERNEL_CAPACITY_RETRIES = 3
KERNEL_CAPACITY_RETRY_SLEEP = 30.0
MAX_KERNEL_CAPACITY_REPEAT = 6
MAX_KERNEL_REGISTRATION_RETRIES = 2
KERNEL_REGISTRATION_RETRY_SLEEP = 15.0
KERNEL_STILL_RUNNING_RETRY_SLEEP = 60.0
MAX_AUTOFIX_ATTEMPTS = 2
MAX_AUTOFIX_RESTARTS = 1
MAX_AUTOFIX_CODEX_PASSES = 3
MAX_KERNEL_FIX_CODEX_PASSES = 3
MAX_AGENT_CAPACITY_ATTEMPTS = 3
AGENT_CAPACITY_RETRY_SLEEP = 5.0
_ERROR_FIX_CODEX_MODEL = ORACLE_IMPLEMENTATION_AGENT.model
_ERROR_FIX_REASONING_EFFORT = ORACLE_IMPLEMENTATION_AGENT.reasoning_effort
_ERROR_STRATEGY_MODEL = STRATEGY_AGENT.model
_ERROR_STRATEGY_REASONING_EFFORT = STRATEGY_AGENT.reasoning_effort
_METRIC_FIX_CODEX_MODEL = ORACLE_IMPLEMENTATION_AGENT.model
_METRIC_FIX_REASONING_EFFORT = ORACLE_IMPLEMENTATION_AGENT.reasoning_effort
_MAX_METRIC_FIX_ATTEMPTS = 3
_MAX_METRIC_FIX_CODEX_PASSES = 4
_FORCED_INITIAL_SUBMIT_REASON = "initial_submit_contract_probe"
_SPARE_DAILY_SUBMIT_REASON = "spare_daily_submission_slot"
_SUBMIT_FAILED_DEFERRED_STATE = "submit_failed_deferred"
_DEFAULT_EVAL_SEEDS = list(_plan_policy.DEFAULT_EVAL_SEEDS)
_DEFAULT_MAX_ITERATIONS = 5
_LONG_LOCAL_GPU_ITERATION_BUDGET_MIN = 12 * 60
_LONG_LOCAL_GPU_MAX_ITERATIONS = 3
_HEAVY_LOCAL_GPU_MAX_CV_FOLDS = 3
_DEFAULT_FORCE_MAJOR_RANK_MAX_PERCENTILE = 0.35
_DEFAULT_FORCE_MAJOR_RANK_MIN_TEAMS = 200
_DEFAULT_TARGET_MEDAL = DEFAULT_TARGET_MEDAL
_DEFAULT_LIMITED_SUBMISSION_GATE = "readiness_or_final"
_DEFAULT_STRICT_COMPETITION_METRIC = True
_DEFAULT_REQUIRE_SUBMIT_IMPROVEMENT = True
_DEFAULT_FORCE_MAJOR_ON_NO_IMPROVE = True
_MAX_KERNEL_PREFLIGHT_FIX_ATTEMPTS = 2
_ITERATION_SUBMISSION_SUFFIXES = all_submission_output_suffixes()
_ITERATION_SAMPLE_SUBMISSION_SUFFIXES = tabular_submission_output_suffixes()


def _first_nonempty_string(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalized_score_source(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _is_finite_score(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _declares_unscored_official_metric(metrics: dict[str, object]) -> bool:
    """Return whether metrics explicitly decline to claim an official score."""
    return bool(
        metrics.get("result_kind") == "unscored_diagnostic_artifact"
        or metrics.get("score_status") == "official_paired_evaluation_unavailable"
    )


def _is_strict_unscored_diagnostic_payload(metrics: dict[str, object]) -> bool:
    """Return whether metrics consistently describe a finite diagnostic-only score."""
    return bool(
        metrics.get("result_kind") == "unscored_diagnostic_artifact"
        and metrics.get("score_status") == "official_paired_evaluation_unavailable"
        and metrics.get("primary_metric_available") is False
        and metrics.get("submission_ready") is False
        and metrics.get("final_ready") is False
        and metrics.get("validation_status") in {"validation_unavailable", "paired_validation_unavailable"}
        and "metric_value" in metrics
        and metrics.get("metric_value") is None
        and "primary_metric_value" in metrics
        and metrics.get("primary_metric_value") is None
        and "primary_score" in metrics
        and metrics.get("primary_score") is None
        and metrics.get("cv_metric_is_primary_competition_metric") is False
        and _is_finite_score(metrics.get("cv_metric_value"))
    )


def _resolve_local_score(
    metrics: dict[str, object],
    score_source: str,
) -> tuple[float | None, str | None]:
    """Resolve the finite local-selection score without promoting nullable official scores."""
    normalized_source = _normalized_score_source(score_source)
    if normalized_source == "cv":
        candidate_keys = ("cv_score", "cv_metric_value", "metric_value", "score")
    elif normalized_source == "holdout":
        candidate_keys = (
            "holdout_score",
            "primary_metric_value",
            "primary_score",
            "metric_value",
            "score",
        )
    else:
        candidate_keys = ("metric_value", "score")

    for key in candidate_keys:
        value = metrics.get(key)
        if _is_finite_score(value):
            return float(value), key
    return None, None


def _metric_names_compatible(actual: object, expected: object) -> bool:
    if not isinstance(actual, str) or not actual.strip() or not isinstance(expected, str) or not expected.strip():
        return False
    if _metric_matching.metrics_equivalent(actual, expected):
        return True
    actual_tokens = set(re.findall(r"[a-z0-9]+", actual.lower()))
    expected_tokens = set(re.findall(r"[a-z0-9]+", expected.lower()))
    paired_lift = {"paired", "lift"}
    proxy_markers = {"routing", "screening", "lint"}
    return paired_lift <= actual_tokens and paired_lift <= expected_tokens and not (actual_tokens & proxy_markers)


def _resolve_authoritative_evaluation_contract(
    *,
    run_payload: dict[str, object],
    frozen_plan: dict[str, object],
    fallback_metric: str,
    fallback_direction: str,
    fallback_score_source: str,
) -> tuple[dict[str, object], str]:
    run_config = run_payload.get("config") if isinstance(run_payload.get("config"), dict) else {}
    run_contract = (
        run_config.get("evaluation_contract") if isinstance(run_config.get("evaluation_contract"), dict) else {}
    )
    plan_contract = (
        frozen_plan.get("evaluation_contract") if isinstance(frozen_plan.get("evaluation_contract"), dict) else {}
    )
    if run_contract:
        contract = dict(run_contract)
        source = "run.json.config.evaluation_contract"
    elif plan_contract:
        contract = dict(plan_contract)
        source = "plan.json.evaluation_contract"
    else:
        contract = {}
        source = "plan.json"

    expected_metric = _first_nonempty_string(
        contract.get("expected_metric"),
        contract.get("metric_name"),
        contract.get("metric"),
        frozen_plan.get("target_metric"),
        fallback_metric,
    )
    expected_direction = _first_nonempty_string(
        contract.get("expected_direction"),
        contract.get("direction"),
        frozen_plan.get("target_direction"),
        fallback_direction,
    )
    expected_split = _first_nonempty_string(
        contract.get("expected_split_strategy"),
        contract.get("split_strategy"),
        frozen_plan.get("split_strategy"),
    )
    raw_score_sources = contract.get("accepted_score_sources")
    accepted_score_sources = (
        [_normalized_score_source(item) for item in raw_score_sources]
        if isinstance(raw_score_sources, (list, tuple))
        else []
    )
    accepted_score_sources = [item for item in accepted_score_sources if item]
    if not accepted_score_sources:
        plan_score_source = _normalized_score_source(frozen_plan.get("score_source"))
        accepted_score_sources = [plan_score_source or _normalized_score_source(fallback_score_source)]

    contract["expected_metric"] = expected_metric
    contract["expected_direction"] = (
        expected_direction.lower()
        if expected_direction and expected_direction.lower() in {"minimize", "maximize"}
        else None
    )
    contract["expected_split_strategy"] = expected_split
    contract["accepted_score_sources"] = list(dict.fromkeys(accepted_score_sources))
    return contract, source


def _evaluation_spec_conflict_warnings(
    *,
    authoritative_contract: dict[str, object],
    evaluation_spec: dict[str, object],
) -> tuple[str, ...]:
    warnings: list[str] = []
    expected_metric = authoritative_contract.get("expected_metric")
    spec_metric = _first_nonempty_string(evaluation_spec.get("metric_name"), evaluation_spec.get("metric"))
    if spec_metric and expected_metric and not _metric_names_compatible(spec_metric, expected_metric):
        warnings.append(
            "context/evaluation_spec.json metric "
            f"'{spec_metric}' conflicts with authoritative metric '{expected_metric}' and was ignored"
        )
    expected_direction = _first_nonempty_string(authoritative_contract.get("expected_direction"))
    spec_direction = _first_nonempty_string(evaluation_spec.get("direction"))
    if (
        spec_direction
        and expected_direction
        and spec_direction.lower() in {"minimize", "maximize"}
        and spec_direction.lower() != expected_direction.lower()
    ):
        warnings.append(
            "context/evaluation_spec.json direction "
            f"'{spec_direction}' conflicts with authoritative direction '{expected_direction}' and was ignored"
        )
    return tuple(warnings)


def _kernel_output_reports_failure(output_dir: Path) -> bool:
    try:
        return any(path.is_file() for path in output_dir.rglob("failure.json"))
    except OSError:
        return True


def _resolve_diagnostic_archive_path(
    *,
    kernel_output_dir: Path,
    route_result: dict[str, object],
    allow_submission_archive: bool = False,
) -> Path | None:
    manifest = route_result.get("manifest")
    if not isinstance(manifest, dict):
        return None
    archive_name = manifest.get("archive")
    if not isinstance(archive_name, str) or not archive_name.strip():
        return None
    archive_name = archive_name.strip()
    if Path(archive_name).name != archive_name:
        return None

    artifact_paths = route_result.get("artifact_paths")
    if not isinstance(artifact_paths, list) or not artifact_paths:
        return None
    resolved_artifact_paths: list[Path] = []
    for raw_path in artifact_paths:
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None
        candidate = Path(raw_path.strip())
        candidate = candidate if candidate.is_absolute() else kernel_output_dir / candidate
        if not candidate.is_file() or (candidate.name.lower() == "submission.zip" and not allow_submission_archive):
            return None
        resolved_artifact_paths.append(candidate)

    candidates = [kernel_output_dir / archive_name]
    candidates.extend(candidate for candidate in resolved_artifact_paths if candidate.name == archive_name)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _is_validated_unscored_training_artifact(
    *,
    kernel_output_dir: Path,
    metrics: dict[str, object],
    route_result: dict[str, object],
    submission_contract: dict[str, object],
) -> bool:
    """Return whether a trained artifact is valid but honestly unscored."""
    unavailable_reason = metrics.get("primary_metric_unavailable_reason")
    readiness_blockers = metrics.get("readiness_blockers")
    manifest = route_result.get("manifest")
    artifact_paths = route_result.get("artifact_paths")
    archive_name = manifest.get("archive") if isinstance(manifest, dict) else None
    if (
        not _is_strict_unscored_diagnostic_payload(metrics)
        or metrics.get("execution_mode") != "train_and_validate"
        or metrics.get("training_performed") is not True
        or metrics.get("execution_status")
        not in {
            "success",
            "unavailable",
            "validation_unavailable",
            "paired_validation_unavailable",
            "validated_unscored_artifact",
        }
        or metrics.get("result_kind") != "unscored_diagnostic_artifact"
        or metrics.get("score_status") != "official_paired_evaluation_unavailable"
        or metrics.get("offline_artifact_validation_passed") is not True
        or metrics.get("official_paired_validation_passed") is not False
        or metrics.get("primary_metric_available") is not False
        or "primary_score" not in metrics
        or metrics.get("primary_score") is not None
        or "primary_metric_value" not in metrics
        or any(metrics.get(key) is not None for key in ("primary_metric_value", "offline_value", "value"))
        or metrics.get("cv_metric_is_primary_competition_metric") is not False
        or metrics.get("submission_ready") is not False
        or not isinstance(unavailable_reason, str)
        or not unavailable_reason.strip()
        or not isinstance(readiness_blockers, list)
        or not readiness_blockers
        or any(not isinstance(blocker, str) or not blocker.strip() for blocker in readiness_blockers)
        or _first_nonempty_string(metrics.get("selected_pipeline")) is None
        or _first_nonempty_string(metrics.get("selected_candidate_hash")) is None
        or route_result.get("mode") != "skill_artifact"
        or route_result.get("validation_status")
        not in {
            "success",
            "unavailable",
            "validation_unavailable",
            "paired_validation_unavailable",
            "validated_unscored_artifact",
        }
        or not isinstance(manifest, dict)
        or not isinstance(archive_name, str)
        or not archive_name.strip()
        or Path(archive_name.strip()).name != archive_name.strip()
        or Path(archive_name.strip()).suffix.lower() != ".zip"
        or manifest.get("submission_ready") is not False
        or not isinstance(artifact_paths, list)
        or not artifact_paths
    ):
        return False

    if any(not isinstance(path, str) or not path.strip() for path in artifact_paths):
        return False
    if not any(Path(path.strip()).name == archive_name.strip() for path in artifact_paths):
        return False

    if not bool(
        submission_contract.get("archive_name") == archive_name.strip()
        and submission_contract.get("submission_ready") is False
        and submission_contract.get("canonical_submission_ready") is False
        and submission_contract.get("offline_artifact_validation_passed") is True
        and submission_contract.get("official_paired_validation_passed") is False
    ):
        return False

    archive_path = _resolve_diagnostic_archive_path(
        kernel_output_dir=kernel_output_dir,
        route_result=route_result,
        allow_submission_archive=True,
    )
    if archive_path is None:
        return False
    archive_sha256 = _sha256_or_none(archive_path)
    if not isinstance(archive_sha256, str):
        return False

    asset_hashes = metrics.get("asset_hashes")
    metrics_archive_sha256 = asset_hashes.get("archive") if isinstance(asset_hashes, dict) else None
    final_manifest = _json_utils.load_json_object(kernel_output_dir / "final_artifact_manifest.json") or {}
    deliverables = final_manifest.get("deliverables")
    if not isinstance(deliverables, list):
        return False
    archive_entries = [
        entry for entry in deliverables if isinstance(entry, dict) and entry.get("path") == archive_name.strip()
    ]
    if len(archive_entries) != 1:
        return False
    archive_entry = archive_entries[0]

    submission_validation = _json_utils.load_json_object(kernel_output_dir / "submission_validation.json") or {}
    claimed_hashes = [
        metrics.get("archive_hash"),
        metrics_archive_sha256,
        manifest.get("archive_sha256"),
        submission_contract.get("archive_sha256"),
        archive_entry.get("sha256"),
        submission_validation.get("archive_sha256"),
    ]

    return bool(
        re.fullmatch(r"[0-9a-f]{64}", archive_sha256)
        and all(claimed_hash == archive_sha256 for claimed_hash in claimed_hashes)
        and final_manifest.get("submission_ready") is False
        and archive_entry.get("size") == archive_path.stat().st_size
        and archive_entry.get("validation_status") == "validated"
        and archive_entry.get("intended_for_kaggle_submission") is False
        and submission_validation.get("archive_name") == archive_name.strip()
        and submission_validation.get("archive_static_valid") is True
        and submission_validation.get("passed") is False
    )


def _unscored_metrics_match_contract(
    *,
    metrics: dict[str, object],
    authoritative_contract: dict[str, object],
) -> bool:
    if not _metric_names_compatible(metrics.get("primary_metric"), authoritative_contract.get("expected_metric")):
        return False
    score_source = _normalized_score_source(metrics.get("score_source"))
    raw_accepted_sources = authoritative_contract.get("accepted_score_sources")
    accepted_sources = (
        {_normalized_score_source(item) for item in raw_accepted_sources}
        if isinstance(raw_accepted_sources, (list, tuple))
        else set()
    )
    if not score_source or score_source not in accepted_sources:
        return False
    actual_direction = _first_nonempty_string(
        metrics.get("primary_direction"),
        metrics.get("metric_direction"),
        metrics.get("direction"),
    )
    expected_direction = _first_nonempty_string(authoritative_contract.get("expected_direction"))
    return not actual_direction or not expected_direction or actual_direction.lower() == expected_direction.lower()


def _scored_primary_contract_issues(
    *,
    metrics: dict[str, object],
    authoritative_contract: dict[str, object],
) -> tuple[str, ...]:
    issues: list[str] = []
    if not _metric_names_compatible(metrics.get("primary_metric"), authoritative_contract.get("expected_metric")):
        issues.append("primary_metric does not match the authoritative evaluation metric")

    expected_direction = _first_nonempty_string(authoritative_contract.get("expected_direction"))
    actual_direction = _first_nonempty_string(
        metrics.get("primary_direction"),
        metrics.get("metric_direction"),
        metrics.get("direction"),
    )
    if expected_direction and (not actual_direction or actual_direction.lower() != expected_direction.lower()):
        issues.append("metric direction does not match the authoritative evaluation direction")

    expected_split = _first_nonempty_string(authoritative_contract.get("expected_split_strategy"))
    actual_split = _first_nonempty_string(metrics.get("split_strategy"), metrics.get("validation_split"))
    if expected_split and (
        not actual_split
        or actual_split.strip().lower().replace("-", "_") != expected_split.strip().lower().replace("-", "_")
    ):
        issues.append("split strategy does not match the authoritative evaluation split")

    score_source = _normalized_score_source(metrics.get("score_source"))
    raw_accepted_sources = authoritative_contract.get("accepted_score_sources")
    accepted_sources = (
        {_normalized_score_source(item) for item in raw_accepted_sources}
        if isinstance(raw_accepted_sources, (list, tuple))
        else set()
    )
    if not score_source or score_source not in accepted_sources:
        issues.append("score_source is not accepted by the authoritative evaluation contract")

    if metrics.get("execution_mode") == "non_training_submission" and (
        metrics.get("offline_artifact_validation_passed") is not True
        or metrics.get("official_paired_validation_passed") is not True
    ):
        issues.append("non-training primary_score requires successful offline and official paired validation")
    return tuple(issues)


def _canonical_metric_contract_issues(
    *,
    metrics: dict[str, object],
    authoritative_contract: dict[str, object],
) -> tuple[str, ...]:
    issues: list[str] = []
    actual_metric = _first_nonempty_string(
        metrics.get("metric_name"),
        metrics.get("metric"),
        metrics.get("target_metric"),
    )
    expected_metric = authoritative_contract.get("expected_metric")
    if not _metric_names_compatible(actual_metric, expected_metric):
        issues.append("metric name does not match the authoritative evaluation metric")

    actual_direction = _first_nonempty_string(
        metrics.get("direction"),
        metrics.get("metric_direction"),
        metrics.get("target_direction"),
    )
    expected_direction = _first_nonempty_string(authoritative_contract.get("expected_direction"))
    if expected_direction and (not actual_direction or actual_direction.lower() != expected_direction.lower()):
        issues.append("metric direction does not match the authoritative evaluation direction")

    score_source = _normalized_score_source(metrics.get("score_source"))
    raw_accepted_sources = authoritative_contract.get("accepted_score_sources")
    accepted_sources = (
        {_normalized_score_source(item) for item in raw_accepted_sources}
        if isinstance(raw_accepted_sources, (list, tuple))
        else set()
    )
    if not score_source or score_source not in accepted_sources:
        issues.append("score_source is not accepted by the authoritative evaluation contract")

    if (
        authoritative_contract.get("require_trusted_score_source") is True
        and metrics.get("model_selection_trusted") is not True
    ):
        issues.append("model-selection score is not explicitly trusted")
    return tuple(issues)


def _kernel_metric_ingestion_error(
    *,
    reason: str,
    metrics_path: Path,
    metrics: dict[str, object],
    candidate_key: str | None,
    contract_issues: tuple[str, ...] = (),
) -> KernelFailedError:
    candidate_value = metrics.get(candidate_key) if candidate_key is not None else None
    detail = (
        f"reason={reason}; metrics_path={metrics_path}; metrics_file_exists={metrics_path.is_file()}; "
        f"top_level_keys={sorted(str(key) for key in metrics)}; "
        f"candidate_score_key={candidate_key or '(missing)'}; "
        f"candidate_score_type={type(candidate_value).__name__ if candidate_key is not None else '(missing)'}; "
        f"failure_json_exists={(metrics_path.parent / 'failure.json').is_file()}"
    )
    if contract_issues:
        detail += "; contract_issues=" + ", ".join(contract_issues)
    return KernelFailedError(f"Kernel metrics rejected: {detail}")


def _load_contract_aware_kernel_metrics(
    *,
    metrics_path: Path,
    metrics: dict[str, object],
    direction: str,
    target_metric: str,
    authoritative_contract: dict[str, object],
) -> EvaluationResult | None:
    # The caller classifies and preserves the corresponding artifacts before it
    # reaches this loader. Keep the diagnostic value out of score comparisons.
    if _is_strict_unscored_diagnostic_payload(metrics):
        return None
    if _declares_unscored_official_metric(metrics):
        raise _kernel_metric_ingestion_error(
            reason="unscored_diagnostic_contract_rejected",
            metrics_path=metrics_path,
            metrics=metrics,
            candidate_key="metric_value",
        )
    score_source = _normalized_score_source(metrics.get("score_source"))
    local_score, score_key = _resolve_local_score(metrics, score_source)
    if score_key in {"cv_score", "cv_metric_value", "holdout_score"}:
        if score_key.startswith("cv_"):
            metric_name = _first_nonempty_string(
                metrics.get("cv_metric_name"),
                metrics.get("cv_metric"),
                metrics.get("metric_name"),
                metrics.get("metric"),
            )
            metric_direction = _first_nonempty_string(
                metrics.get("cv_metric_direction"),
                metrics.get("direction"),
                metrics.get("metric_direction"),
            )
        else:
            metric_name = _first_nonempty_string(
                metrics.get("holdout_metric_name"),
                metrics.get("holdout_metric"),
                metrics.get("metric_name"),
                metrics.get("metric"),
            )
            metric_direction = _first_nonempty_string(
                metrics.get("holdout_metric_direction"),
                metrics.get("direction"),
                metrics.get("metric_direction"),
            )
        metrics["score_key_used"] = score_key
        scored_payload = dict(metrics)
        scored_payload["offline_value"] = local_score
        scored_payload["metric"] = metric_name or target_metric
        scored_payload["direction"] = metric_direction or direction
        return _kernel_metrics.evaluation_from_kernel_metrics_payload(
            scored_payload,
            direction=direction,
            target_metric=target_metric,
        )
    if "metric_value" in metrics:
        raw_metric_value = metrics["metric_value"]
        if isinstance(raw_metric_value, bool) or not isinstance(raw_metric_value, Real):
            raise _kernel_metric_ingestion_error(
                reason="metric_value_not_numeric",
                metrics_path=metrics_path,
                metrics=metrics,
                candidate_key="metric_value",
            )
        if not math.isfinite(float(raw_metric_value)):
            raise _kernel_metric_ingestion_error(
                reason="metric_value_non_finite",
                metrics_path=metrics_path,
                metrics=metrics,
                candidate_key="metric_value",
            )
        contract_issues = _canonical_metric_contract_issues(
            metrics=metrics,
            authoritative_contract=authoritative_contract,
        )
        if contract_issues:
            raise _kernel_metric_ingestion_error(
                reason="metric_contract_rejected",
                metrics_path=metrics_path,
                metrics=metrics,
                candidate_key="metric_value",
                contract_issues=contract_issues,
            )
        metric_name = _first_nonempty_string(
            metrics.get("metric_name"),
            metrics.get("metric"),
            metrics.get("target_metric"),
        )
        metric_direction = _first_nonempty_string(
            metrics.get("direction"),
            metrics.get("metric_direction"),
            metrics.get("target_direction"),
        )
        scored_payload = dict(metrics)
        scored_payload["offline_value"] = float(raw_metric_value)
        scored_payload["metric"] = metric_name or target_metric
        scored_payload["direction"] = metric_direction or direction
        scored_payload["score_key_used"] = "metric_value"
        metrics["score_key_used"] = "metric_value"
        return _kernel_metrics.evaluation_from_kernel_metrics_payload(
            scored_payload,
            direction=direction,
            target_metric=target_metric,
        )
    if score_key == "score":
        metrics["score_key_used"] = "score"
        return _kernel_metrics.load_kernel_metrics(metrics_path, direction, target_metric)
    if score_key in {"primary_metric_value", "primary_score"}:
        raw_primary_score = local_score
        primary_score_key = score_key
    elif "primary_score" not in metrics:
        return _kernel_metrics.load_kernel_metrics(metrics_path, direction, target_metric)
    else:
        raw_primary_score = metrics.get("primary_score")
        primary_score_key = "primary_score"
    if not _is_finite_score(raw_primary_score):
        return None
    primary_score = float(raw_primary_score)
    contract_issues = _scored_primary_contract_issues(
        metrics=metrics,
        authoritative_contract=authoritative_contract,
    )
    if contract_issues:
        raise KernelFailedError(
            "Kernel primary_score violates the authoritative evaluation contract: " + "; ".join(contract_issues)
        )
    scored_payload = dict(metrics)
    scored_payload["offline_value"] = primary_score
    scored_payload["direction"] = direction
    scored_payload["score_key_used"] = primary_score_key
    metrics["score_key_used"] = primary_score_key
    return _kernel_metrics.evaluation_from_kernel_metrics_payload(
        scored_payload,
        direction=direction,
        target_metric=target_metric,
    )


def _add_kernel_score_provenance(
    payload: dict[str, object],
    kernel_metrics: dict[str, object] | None,
) -> None:
    if not isinstance(kernel_metrics, dict):
        return
    for key in (
        "score_key_used",
        "cv_score",
        "cv_metric_name",
        "cv_metric_is_primary_competition_metric",
        "primary_score",
        "primary_metric_value",
        "primary_metric_available",
        "official_paired_validation_passed",
        "submission_ready",
    ):
        if key in kernel_metrics:
            payload[key] = kernel_metrics[key]


def _resolve_validation_blocked_kernel_result(
    *,
    kernel_result: KernelRunResult,
    kernel_metrics: dict[str, object],
    training_route_decision: TrainingRouteDecision,
) -> _ValidationBlockedKernelResult | None:
    route_result = _json_utils.load_json_object(kernel_result.output_dir / "route_result.json") or {}
    submission_contract = _json_utils.load_json_object(kernel_result.output_dir / "submission_contract.json") or {}
    canonical_submission_emitted = any(path.is_file() for path in kernel_result.output_dir.rglob("submission.zip"))
    blockers = resolve_non_training_validation_blockers(
        metrics=kernel_metrics,
        route_result=route_result,
        submission_contract=submission_contract,
        decision=training_route_decision,
        canonical_submission_emitted=canonical_submission_emitted,
    )
    if blockers is None:
        return None
    return _ValidationBlockedKernelResult(
        blockers=blockers,
        route_result=route_result,
        submission_contract=submission_contract,
    )


def _resolve_unscored_diagnostic_kernel_result(
    *,
    kernel_result: KernelRunResult,
    kernel_metrics: dict[str, object],
    training_route_decision: TrainingRouteDecision,
    authoritative_evaluation_contract: dict[str, object],
    deliverable_mode: str,
    submit_mode: str,
    code_competition: bool = False,
) -> _UnscoredDiagnosticKernelResult | None:
    if deliverable_mode != "writeup" or submit_mode != "file" or code_competition:
        return None
    route_result = _json_utils.load_json_object(kernel_result.output_dir / "route_result.json") or {}
    submission_contract = _json_utils.load_json_object(kernel_result.output_dir / "submission_contract.json") or {}
    canonical_submission_emitted = any(path.is_file() for path in kernel_result.output_dir.rglob("submission.zip"))
    unscored_non_training_diagnostic = is_unscored_non_training_diagnostic(
        metrics=kernel_metrics,
        route_result=route_result,
        submission_contract=submission_contract,
        decision=training_route_decision,
        canonical_submission_emitted=canonical_submission_emitted,
    )
    validated_unscored_training_artifact = _is_validated_unscored_training_artifact(
        kernel_output_dir=kernel_result.output_dir,
        metrics=kernel_metrics,
        route_result=route_result,
        submission_contract=submission_contract,
    )
    if not unscored_non_training_diagnostic and not validated_unscored_training_artifact:
        return None
    if _kernel_output_reports_failure(kernel_result.output_dir):
        return None
    if not _unscored_metrics_match_contract(
        metrics=kernel_metrics,
        authoritative_contract=authoritative_evaluation_contract,
    ):
        return None
    if _first_nonempty_string(kernel_metrics.get("selected_pipeline")) is None:
        return None
    diagnostic_archive_path = _resolve_diagnostic_archive_path(
        kernel_output_dir=kernel_result.output_dir,
        route_result=route_result,
        allow_submission_archive=validated_unscored_training_artifact,
    )
    if diagnostic_archive_path is None:
        return None
    return _UnscoredDiagnosticKernelResult(
        route_result=route_result,
        submission_contract=submission_contract,
        diagnostic_archive_path=diagnostic_archive_path,
    )


def _preserve_diagnostic_artifacts(
    *,
    kernel_output_dir: Path,
    iter_dir: Path,
    route_result: dict[str, object],
) -> list[str]:
    artifact_names = {
        "candidate_ledger.jsonl",
        "final_artifact_manifest.json",
        "metrics.json",
        "route_result.json",
        "submission_contract.json",
        "submission_validation.json",
        "writeup.md",
        "writeup_evidence.json",
        "writeup_word_count.json",
    }
    manifest = route_result.get("manifest")
    if isinstance(manifest, dict):
        archive_name = manifest.get("archive")
        if isinstance(archive_name, str) and archive_name.strip():
            safe_name = Path(archive_name.strip()).name
            if safe_name == archive_name.strip():
                artifact_names.add(safe_name)

    routed_artifacts: dict[str, Path] = {}
    artifact_paths = route_result.get("artifact_paths")
    if isinstance(artifact_paths, list):
        for raw_path in artifact_paths:
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            candidate = Path(raw_path.strip())
            if candidate.name in artifact_names and candidate.is_file():
                routed_artifacts[candidate.name] = candidate

    destination_dir = iter_dir / "output"
    preserved: list[str] = []
    for name in sorted(artifact_names):
        source = kernel_output_dir / name
        if not source.is_file():
            source = routed_artifacts.get(name, source)
        if not source.is_file():
            continue
        destination = destination_dir / name
        copy_artifact_if_needed(source=source, destination=destination)
        preserved.append(str(destination.relative_to(iter_dir)))
    return preserved


def check_rules_accepted(*args, **kwargs):
    return _autopilot_submit.check_rules_accepted(*args, **kwargs)


check_rules_accepted._kagglebot_default_wrapper = True


def resolve_kaggle_username(*args, **kwargs):
    return _autopilot_submit.resolve_kaggle_username(*args, **kwargs)


resolve_kaggle_username._kagglebot_default_wrapper = True


def run_submit_kernel(**kwargs):
    return _autopilot_submit.run_submit_kernel(**kwargs)


run_submit_kernel._kagglebot_default_wrapper = True


def run_kaggle_submit_kernel(**kwargs):
    return _autopilot_submit.run_kaggle_submit_kernel(**kwargs)


run_kaggle_submit_kernel._kagglebot_default_wrapper = True


def classify_submit_error(*args, **kwargs):
    return _autopilot_submit.classify_submit_error(*args, **kwargs)


classify_submit_error._kagglebot_default_wrapper = True


def list_competition_submissions(*args, **kwargs):
    return _autopilot_submit.list_competition_submissions(*args, **kwargs)


list_competition_submissions._kagglebot_default_wrapper = True


def _iteration_submission_path(
    *,
    iter_dir: Path,
    sample_submission_path: Path,
    submission_format_path: Path | None = None,
) -> Path:
    filename = _iteration_submission_filename_from_format(submission_format_path)
    if filename is not None:
        return iter_dir / filename
    suffix = _iteration_submission_suffix_from_format(submission_format_path)
    if suffix is not None:
        return iter_dir / f"submission{suffix}"
    suffix = tabular_suffix(sample_submission_path)
    if suffix not in _ITERATION_SAMPLE_SUBMISSION_SUFFIXES:
        suffix = ".csv"
    return iter_dir / f"submission{suffix}"


def _iteration_submission_filename_from_format(submission_format_path: Path | None) -> str | None:
    if submission_format_path is None or not submission_format_path.exists():
        return None
    hint = load_submission_format_hint(submission_format_path)
    if hint is None or not hint.expected_suffixes:
        return None
    try:
        text = submission_format_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        text = ""
    return output_filename_from_format_text(
        text,
        expected_suffixes=hint.expected_suffixes,
        allowed_suffixes=_ITERATION_SUBMISSION_SUFFIXES,
    )


def _iteration_submission_suffix_from_format(submission_format_path: Path | None) -> str | None:
    if submission_format_path is None or not submission_format_path.exists():
        return None
    hint = load_submission_format_hint(submission_format_path)
    if hint is None or not hint.expected_suffixes:
        return None
    return first_allowed_expected_output_suffix(
        hint.expected_suffixes,
        allowed_suffixes=_ITERATION_SUBMISSION_SUFFIXES,
    )


def _run_local_to_kaggle_gpu_handoff(
    *,
    config: AutopilotConfig,
    run_id: str,
    iteration: int,
    iter_dir: Path,
    logs_dir: Path,
    output_dir: Path,
    kernel_name: str | None,
    enable_internet: bool,
    score_source: str,
    target_metric: str,
    metric_direction: str,
    holdout_frac: float,
    cv_folds: int,
    seed: int,
    time_budget_min: int | None,
    local_error_text: str,
    pending_error_fixes: list[dict[str, object]],
) -> tuple[AutopilotConfig, KernelRunResult]:
    run_dir = config.paths.run_dir(run_id)
    hardware_profile = _compute_handoff.kaggle_gpu_handoff_profile()
    remote_config = replace(
        config,
        compute="kaggle_gpu",
        accelerator="gpu",
        strict_accelerator=False,
        hardware_profile=hardware_profile,
    )
    handoff_payload = _compute_handoff.begin_handoff(
        run_dir=run_dir,
        iter_dir=iter_dir,
        run_id=run_id,
        iteration=iteration,
        error_text=local_error_text,
        to_hardware_profile=hardware_profile,
    )
    _watch_state.update_watch_phase(
        remote_config,
        run_id,
        "local_to_kaggle_gpu_handoff",
        detail="Local training exceeded available resources; continuing the same iteration on Kaggle GPU.",
        iteration=iteration,
    )
    print(
        "[yellow]compute handoff[/yellow]: local_gpu resource limit detected; "
        f"continuing run={run_id} iter={iteration} on kaggle_gpu ({hardware_profile})"
    )
    kaggle_user = resolve_kaggle_username(remote_config.kaggle_username)
    kernel_attempts = 0
    error_fingerprints: dict[str, int] = {}

    def mark_remote_started(kernel_id: str) -> None:
        nonlocal handoff_payload
        handoff_payload = _compute_handoff.finish_handoff(
            run_dir=run_dir,
            iter_dir=iter_dir,
            payload=handoff_payload,
            status="kaggle_gpu_running",
            kernel_id=kernel_id,
        )
        _watch_state.update_watch_phase(
            remote_config,
            run_id,
            "kaggle_kernel_running",
            detail=f"Kaggle kernel {kernel_id} was accepted and is running or queued.",
            iteration=iteration,
        )

    try:
        while True:
            _watch_state.update_watch_phase(
                remote_config,
                run_id,
                "kaggle_kernel_preparing",
                detail="Building and validating the Kaggle kernel package before push.",
                iteration=iteration,
            )
            try:
                result = run_kernel(
                    slug=remote_config.slug,
                    run_id=run_id,
                    iteration=iteration,
                    base_dir=remote_config.paths.base_dir.parent,
                    kaggle_username=kaggle_user,
                    kernel_name=kernel_name,
                    accelerator="gpu",
                    enable_internet=enable_internet,
                    score_source=score_source,
                    metric=target_metric,
                    direction=metric_direction,
                    holdout_frac=holdout_frac,
                    cv_folds=cv_folds,
                    seed=seed,
                    dry_run=remote_config.dry_run,
                    timeout_minutes=time_budget_min,
                    hardware_profile=hardware_profile,
                    on_remote_started=mark_remote_started,
                )
                handoff_payload = _compute_handoff.finish_handoff(
                    run_dir=run_dir,
                    iter_dir=iter_dir,
                    payload=handoff_payload,
                    status="completed",
                    kernel_id=result.kernel_id,
                )
                return remote_config, result
            except RulesNotAcceptedError:
                raise
            except KaggleNetworkError:
                raise
            except KernelStillRunningError as exc:
                handoff_payload = _compute_handoff.finish_handoff(
                    run_dir=run_dir,
                    iter_dir=iter_dir,
                    payload=handoff_payload,
                    status="kaggle_gpu_running",
                )
                error_text = _kernel_errors.format_kernel_error(exc)
                (logs_dir / "kernel_remote_still_running.txt").write_text(error_text + "\n", encoding="utf-8")
                _watch_state.update_watch_phase(
                    remote_config,
                    run_id,
                    "kaggle_kernel_still_running",
                    detail="The handed-off Kaggle notebook is still running; waiting without pushing a duplicate.",
                    iteration=iteration,
                )
                time.sleep(KERNEL_STILL_RUNNING_RETRY_SLEEP)
            except KernelCapacityError as exc:
                kernel_attempts += 1
                error_text = _kernel_errors.format_kernel_error(exc)
                _kernel_errors.record_kernel_error(
                    logs_dir=logs_dir,
                    attempt=kernel_attempts,
                    error_text=error_text,
                    error_fingerprints=error_fingerprints,
                    max_repeats=MAX_KERNEL_CAPACITY_REPEAT,
                    output_dir=output_dir,
                )
                capacity_retries = _env_utils.env_int(
                    "KAGGLEBOT_KERNEL_CAPACITY_RETRIES",
                    default=MAX_KERNEL_CAPACITY_RETRIES,
                )
                _watch_state.update_watch_phase(
                    remote_config,
                    run_id,
                    "kaggle_gpu_no_capacity",
                    detail="Kaggle GPU capacity is unavailable for the local-training handoff.",
                    iteration=iteration,
                )
                if kernel_attempts > capacity_retries:
                    raise
                time.sleep(KERNEL_CAPACITY_RETRY_SLEEP * kernel_attempts)
            except Exception as exc:  # noqa: BLE001
                kernel_attempts += 1
                error_text = _kernel_errors.format_kernel_error(exc)
                if _kernel_errors.is_kernel_registration_error(exc):
                    _kernel_errors.record_kernel_error(
                        logs_dir=logs_dir,
                        attempt=kernel_attempts,
                        error_text=error_text,
                        error_fingerprints=error_fingerprints,
                        output_dir=output_dir,
                    )
                    if kernel_attempts > MAX_KERNEL_REGISTRATION_RETRIES:
                        raise
                    time.sleep(KERNEL_REGISTRATION_RETRY_SLEEP * kernel_attempts)
                    continue
                try:
                    _kernel_errors.record_kernel_error(
                        logs_dir=logs_dir,
                        attempt=kernel_attempts,
                        error_text=error_text,
                        error_fingerprints=error_fingerprints,
                        output_dir=output_dir,
                    )
                except KernelFailedError:
                    if _autofix_restart.maybe_regenerate_kernel_sources_once(
                        dry_run=remote_config.dry_run,
                        agent_dir=iter_dir / "agent",
                        run_id=run_id,
                        iteration=iteration,
                        attempt=kernel_attempts,
                        trigger_reason="handoff_repeated_error_fingerprint",
                        regenerate_kernel_sources=lambda: _planning_runner.run_plan_and_initial(remote_config, run_id),
                    ):
                        error_fingerprints.clear()
                        continue
                    raise
                if remote_config.dry_run:
                    raise
                if MAX_KERNEL_FIX_ATTEMPTS is not None and kernel_attempts > MAX_KERNEL_FIX_ATTEMPTS:
                    raise
                print(
                    "[yellow]handoff kernel failed[/yellow]: invoking "
                    f"{IMPLEMENTATION_AGENT.log_alias} to repair the Kaggle GPU run (attempt {kernel_attempts})"
                )
                _run_kernel_fix(
                    config=remote_config,
                    run_id=run_id,
                    iteration=iteration,
                    iter_dir=iter_dir,
                    error_message=error_text,
                    attempt=kernel_attempts,
                    pending_error_fixes=pending_error_fixes,
                    original_error=exc if isinstance(exc, KernelFailedError) else None,
                )
    except BaseException as exc:
        _compute_handoff.finish_handoff(
            run_dir=run_dir,
            iter_dir=iter_dir,
            payload=handoff_payload,
            status="failed",
            error_text=_kernel_errors.format_kernel_error(exc),
        )
        raise


def _config_for_committed_compute_handoff(config: AutopilotConfig, run_id: str) -> AutopilotConfig:
    if config.compute != "local_gpu":
        return config
    committed_handoff = _compute_handoff.load_committed_handoff(config.paths.run_dir(run_id))
    if committed_handoff is None:
        return config
    resumed_profile = str(committed_handoff.get("to_hardware_profile") or _compute_handoff.kaggle_gpu_handoff_profile())
    return replace(
        config,
        compute="kaggle_gpu",
        accelerator="gpu",
        strict_accelerator=False,
        hardware_profile=resumed_profile,
    )


def _recover_pending_oracle_workflow(*, config: AutopilotConfig, run_id: str) -> None:
    pending = _oracle_workflow_state.load_pending_oracle_workflow(config.paths.run_dir(run_id))
    if pending is None:
        return
    workflow_id = str(pending.get("workflow_id") or "unknown")
    workflow_kind = str(pending.get("workflow_kind") or "")
    raw_payload = pending.get("recovery_payload")
    if not isinstance(raw_payload, dict):
        raise OracleStrategyError(f"Interrupted Oracle workflow {workflow_id} has no valid recovery payload.")
    payload: dict[str, object] = raw_payload
    _watch_state.update_watch_phase(
        config,
        run_id,
        "oracle_workflow_recovering",
        detail=f"Recovering interrupted {workflow_kind} Oracle-to-Codex workflow before resuming kernels.",
        iteration=_workflow_optional_int(payload.get("iteration")),
    )
    print(
        "[yellow]resume[/yellow]: recovering interrupted Oracle workflow "
        f"{workflow_id} ({pending.get('status')}) before planning or kernel execution"
    )
    if workflow_kind == "kernel_fix":
        if (
            _complete_obsolete_kernel_source_fix_recovery(
                config=config,
                run_id=run_id,
                workflow_id=workflow_id,
                payload=payload,
            )
            or _complete_obsolete_kernel_runtime_source_fix_recovery(
                config=config,
                run_id=run_id,
                workflow_id=workflow_id,
                payload=payload,
            )
            or _complete_obsolete_kernel_metrics_fix_recovery(
                config=config,
                run_id=run_id,
                workflow_id=workflow_id,
                payload=payload,
            )
        ):
            return
        iteration = _workflow_required_int(payload, "iteration", workflow_id=workflow_id)
        attempt = _workflow_required_int(payload, "attempt", workflow_id=workflow_id)
        _run_kernel_fix(
            config=config,
            run_id=run_id,
            iteration=iteration,
            iter_dir=config.paths.iter_dir(run_id, iteration),
            error_message=str(payload.get("error_message") or "Interrupted kernel failure"),
            attempt=attempt,
            pending_error_fixes=[],
            use_gpt_strategy=bool(payload.get("use_gpt_strategy", True)),
            codex_model=_workflow_optional_str(payload.get("codex_model")),
            codex_reasoning_effort=_workflow_optional_str(payload.get("codex_reasoning_effort")),
            prompt_prefix=str(payload.get("prompt_prefix") or ""),
            max_codex_passes=_workflow_optional_int(payload.get("max_codex_passes")),
            failure_stage=str(payload.get("failure_stage") or "kernel_runtime"),
        )
    elif workflow_kind == "improvement":
        from kagglebot.solver.evaluate import EvaluationResult

        iteration = _workflow_required_int(payload, "iteration", workflow_id=workflow_id)
        raw_evaluation = payload.get("evaluation")
        if not isinstance(raw_evaluation, dict):
            raise OracleStrategyError(f"Interrupted Oracle workflow {workflow_id} has no evaluation payload.")
        fold_scores_raw = raw_evaluation.get("fold_scores")
        fold_scores = (
            [float(value) for value in fold_scores_raw if isinstance(value, (int, float))]
            if isinstance(fold_scores_raw, list)
            else None
        )
        evaluation = EvaluationResult(
            score_source=str(raw_evaluation.get("score_source") or "offline"),
            metric=str(raw_evaluation.get("metric") or "unknown"),
            direction=str(raw_evaluation.get("direction") or "maximize"),  # type: ignore[arg-type]
            value=_workflow_required_float(raw_evaluation, "value", workflow_id=workflow_id),
            std=_workflow_optional_float(raw_evaluation.get("std")),
            train_score=_workflow_optional_float(raw_evaluation.get("train_score")),
            val_score=_workflow_optional_float(raw_evaluation.get("val_score")),
            fold_scores=fold_scores,
        )
        top1_info = payload.get("top1_info") if isinstance(payload.get("top1_info"), dict) else {}
        pending_insights_raw = payload.get("pending_problem_insights")
        pending_insights = (
            [item for item in pending_insights_raw if isinstance(item, dict)]
            if isinstance(pending_insights_raw, list)
            else []
        )
        extra_notes_raw = payload.get("extra_policy_notes")
        extra_notes = [str(item) for item in extra_notes_raw] if isinstance(extra_notes_raw, list) else None
        previous_history = (
            payload.get("previous_submission_history")
            if isinstance(payload.get("previous_submission_history"), dict)
            else None
        )
        iteration_evidence_path = _workflow_optional_str(payload.get("iteration_evidence_path"))
        iteration_evidence_sha256 = _workflow_optional_str(payload.get("iteration_evidence_sha256"))
        _run_improvement(
            config=config,
            run_id=run_id,
            iteration=iteration,
            iter_dir=config.paths.iter_dir(run_id, iteration),
            evaluation=evaluation,
            top1_info=top1_info,
            target_score=_workflow_required_float(payload, "target_score", workflow_id=workflow_id),
            delta_offline=_workflow_optional_float(payload.get("delta_offline")),
            pending_problem_insights=pending_insights,
            current_score=_workflow_optional_float(payload.get("current_score")),
            current_score_source=str(payload.get("current_score_source") or "offline"),
            minimum_improvement_mode=_workflow_optional_str(payload.get("minimum_improvement_mode")),
            minimum_improvement_reason=_workflow_optional_str(payload.get("minimum_improvement_reason")),
            target_medal=_workflow_optional_str(payload.get("target_medal")),
            target_rank_percentile=_workflow_optional_float(payload.get("target_rank_percentile")),
            forced_improvement_mode=_workflow_optional_str(payload.get("forced_improvement_mode")),
            forced_improvement_reason=_workflow_optional_str(payload.get("forced_improvement_reason")),
            extra_policy_notes=extra_notes,
            enforce_code_reference_implementation=bool(payload.get("enforce_code_reference_implementation")),
            code_reference_enforcement_reason=_workflow_optional_str(payload.get("code_reference_enforcement_reason")),
            best_score_so_far=_workflow_optional_float(payload.get("best_score_so_far")),
            previous_submission_history=previous_history,
            expected_iteration_evidence_path=(
                Path(iteration_evidence_path) if iteration_evidence_path is not None else None
            ),
            expected_iteration_evidence_sha256=iteration_evidence_sha256,
        )
    elif workflow_kind == "autofix":
        attempt = _workflow_required_int(payload, "attempt", workflow_id=workflow_id)
        error_text = str(payload.get("error_text") or "Interrupted autofix failure")
        error: Exception
        if bool(payload.get("submit_autofix")):
            error = SubmitAbortedError(error_text)
        else:
            error = RuntimeError(error_text)
        _run_autofix(config=config, run_id=run_id, attempt=attempt, error=error)
    else:
        raise OracleStrategyError(
            f"Interrupted Oracle workflow {workflow_id} has unsupported kind {workflow_kind!r}; refusing to skip it."
        )
    print(f"[green]resume[/green]: recovered Oracle workflow {workflow_id}; normal autopilot resume may continue")


def _complete_obsolete_kernel_source_fix_recovery(
    *,
    config: AutopilotConfig,
    run_id: str,
    workflow_id: str,
    payload: dict[str, object],
) -> bool:
    """Skip an interrupted source fix when the current source contract already passes."""
    failure_stage = str(payload.get("failure_stage") or "").strip().lower()
    error_message = str(payload.get("error_message") or "")
    legacy_source_failure = (
        "kernel source validation failed" in error_message.lower() or "kernel_source_preflight_error" in error_message
    )
    if failure_stage != "kernel_source_preflight" and not legacy_source_failure:
        return False

    deliverable_contract = resolve_deliverable_artifact_contract(config.paths.base_dir)
    current_error = _kernel_preflight.kernel_source_preflight_error(
        config.paths.kernel_source_dir,
        require_kaggle_input=False,
        deliverable_mode=deliverable_contract.deliverable_mode,
        required_output_names=deliverable_contract.required_output_names,
        format_error=_kernel_errors.format_kernel_error,
    )
    if current_error is not None:
        return False

    checkpoint = _oracle_workflow_state.OracleWorkflowCheckpoint(
        path=_oracle_workflow_state.oracle_workflow_state_path(config.paths.run_dir(run_id)),
        workflow_id=workflow_id,
    )
    checkpoint.mark_completed()
    print(
        "[yellow]resume[/yellow]: interrupted kernel source fix is obsolete; "
        "the current deliverable contract already passes, so kernel execution can resume"
    )
    return True


def _complete_obsolete_kernel_runtime_source_fix_recovery(
    *,
    config: AutopilotConfig,
    run_id: str,
    workflow_id: str,
    payload: dict[str, object],
) -> bool:
    """Skip an interrupted runtime repair after the failed kernel source changed."""
    if str(payload.get("failure_stage") or "").strip().lower() != "kernel_runtime":
        return False
    iteration = _workflow_optional_int(payload.get("iteration"))
    if iteration is None:
        return False
    failed_sha = _workflow_optional_str(payload.get("failed_kernel_source_sha256"))
    if failed_sha is None:
        iter_dir = config.paths.iter_dir(run_id, iteration)
        manifests = sorted(
            iter_dir.glob("output/**/local_launch_manifest.json"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for manifest_path in manifests:
            manifest = _json_utils.load_json_object_or_empty(manifest_path)
            failed_sha = _workflow_optional_str(manifest.get("kernel_sha256"))
            if failed_sha is not None:
                break
    current_sha = _kernel_preflight.kernel_source_sha256(config.paths.kernel_source_dir)
    if failed_sha is None or current_sha is None or failed_sha == current_sha:
        return False
    checkpoint = _oracle_workflow_state.OracleWorkflowCheckpoint(
        path=_oracle_workflow_state.oracle_workflow_state_path(config.paths.run_dir(run_id)),
        workflow_id=workflow_id,
    )
    checkpoint.mark_completed()
    print(
        "[yellow]resume[/yellow]: interrupted kernel runtime fix is obsolete; "
        "kernel.py changed after the failed launch, so the new source will run before any further repair"
    )
    return True


def _complete_obsolete_kernel_metrics_fix_recovery(
    *,
    config: AutopilotConfig,
    run_id: str,
    workflow_id: str,
    payload: dict[str, object],
) -> bool:
    """Skip a nullable-metric repair after the loader learned its strict contract."""
    if str(payload.get("failure_stage") or "").strip().lower() != "kernel_runtime":
        return False
    error_message = str(payload.get("error_message") or "")
    if "reason=metric_value_not_numeric" not in error_message:
        return False
    match = re.search(r"(?:^|;\s*)metrics_path=([^;]+)", error_message)
    if match is None:
        return False
    metrics_path = Path(match.group(1).strip()).resolve()
    run_dir = config.paths.run_dir(run_id).resolve()
    if not metrics_path.is_relative_to(run_dir) or not metrics_path.is_file():
        return False
    metrics = _json_utils.load_json_object_or_empty(metrics_path)
    if not _is_strict_unscored_diagnostic_payload(metrics):
        return False

    checkpoint = _oracle_workflow_state.OracleWorkflowCheckpoint(
        path=_oracle_workflow_state.oracle_workflow_state_path(run_dir),
        workflow_id=workflow_id,
    )
    checkpoint.mark_completed()
    print(
        "[yellow]resume[/yellow]: interrupted nullable-metric repair is obsolete; "
        "the current loader accepts this strict diagnostic-only contract"
    )
    return True


def _workflow_optional_str(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _workflow_optional_int(value: object) -> int | None:
    return tolerant_int(value)


def _workflow_required_int(payload: dict[str, object], key: str, *, workflow_id: str) -> int:
    value = _workflow_optional_int(payload.get(key))
    if value is None:
        raise OracleStrategyError(f"Interrupted Oracle workflow {workflow_id} has invalid {key}.")
    return value


def _workflow_optional_float(value: object) -> float | None:
    return tolerant_finite_float(value)


def _workflow_required_float(payload: dict[str, object], key: str, *, workflow_id: str) -> float:
    value = _workflow_optional_float(payload.get(key))
    if value is None:
        raise OracleStrategyError(f"Interrupted Oracle workflow {workflow_id} has invalid {key}.")
    return value


def run_autopilot(config: AutopilotConfig) -> None:
    resume_id = os.environ.get("KAGGLEBOT_RESUME_RUN_ID")
    resume_slug = os.environ.get("KAGGLEBOT_RESUME_SLUG")
    resume_run = bool(config.run_id is None and resume_id and resume_slug == config.slug)
    if resume_run:
        run_id = resume_id
    else:
        run_id = config.run_id or new_run_id()
    if resume_id:
        os.environ.pop("KAGGLEBOT_RESUME_RUN_ID", None)
        os.environ.pop("KAGGLEBOT_RESUME_SLUG", None)
    resume_after_failure = resume_run
    attempt = 0
    submit_force_override = False
    try:
        while True:
            effective_config = _config_for_committed_compute_handoff(config, run_id)
            _recover_pending_oracle_workflow(config=effective_config, run_id=run_id)
            session = AutopilotSession(config=effective_config, run_id=run_id, resume_run=resume_after_failure)
            try:
                return session.run()
            except RulesNotAcceptedError:
                raise
            except SubmitAbortedError as exc:
                if config.dry_run:
                    raise
                run_dir = config.paths.run_dir(run_id)
                submit_abort_autofix = _submit_failure_context.resolve_submit_abort_autofixability_for_run(
                    run_dir=run_dir,
                    load_run_state=_autopilot_state.load_run_state,
                )
                if submit_abort_autofix.message:
                    print(submit_abort_autofix.message)
                if not submit_abort_autofix.autofixable:
                    raise
                attempt += 1
                if attempt > MAX_AUTOFIX_ATTEMPTS:
                    raise
                print(
                    f"[yellow]autofix[/yellow]: submit stage failed; invoking "
                    f"{IMPLEMENTATION_AGENT.log_alias} to repair and retry submit"
                )
                if _submit_failure_context.should_force_resubmit_after_submit_abort_for_run(
                    run_dir=run_dir,
                    load_run_state=_autopilot_state.load_run_state,
                    has_successful_submit_attempt=_submit_attempts.has_successful_submit_attempt,
                ):
                    os.environ["KAGGLEBOT_FORCE_RESUBMIT"] = "1"
                    submit_force_override = True
                _run_autofix(config=effective_config, run_id=run_id, attempt=attempt, error=exc)
                resume_after_failure = True
            except KernelCapacityError:
                raise
            except MissingCompetitionDataError as exc:
                print(f"[yellow]blocked on data[/yellow]: {exc}")
                raise
            except OracleStrategyError:
                raise
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                if config.dry_run:
                    raise
                if _runtime_fixes.is_non_autofixable_runtime_error(exc):
                    raise
                attempt += 1
                if attempt > MAX_AUTOFIX_ATTEMPTS:
                    raise
                print(f"[yellow]autofix[/yellow]: invoking {IMPLEMENTATION_AGENT.log_alias} to repair error")
                _run_autofix(config=effective_config, run_id=run_id, attempt=attempt, error=exc)
                resume_after_failure = True
    finally:
        if submit_force_override:
            os.environ.pop("KAGGLEBOT_FORCE_RESUBMIT", None)


def run_autopilot_core(config: AutopilotConfig, run_id: str, *, resume_run: bool = False) -> None:
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    resumed_config = _config_for_committed_compute_handoff(config, run_id)
    if resumed_config != config:
        config = resumed_config
        print(
            "[yellow]compute handoff resume[/yellow]: "
            f"continuing run={run_id} on kaggle_gpu ({config.hardware_profile})"
        )
    _watch_state.update_watch_phase(config, run_id, "autopilot_starting")
    print(f"[green]run started[/green]: {run_id}")
    planning_phase = PlanningPhase(config=config, run_id=run_id, resume_run=resume_run)
    knowledge_phase = KnowledgePhase(config=config)
    plan = _plan_policy.load_plan_config(config.paths)
    if not config.paths.plan_path.exists():
        _plan_policy.write_plan_config(config.paths, plan)

    _watch_state.update_watch_phase(config, run_id, "leaderboard_fetching")
    print(f"[cyan]fetching leaderboard[/cyan]: {config.slug}")
    metric_hint = config.target_metric or plan.target_metric
    top1_info = leaderboard_top1(
        config.slug,
        config.paths.context_dir,
        dry_run=config.dry_run,
        metric_hint=metric_hint,
    )
    write_top1_public_snapshot(config.paths.top1_public_path, top1_info)
    print(format_top1_public_score_message(top1_info))
    _watch_state.update_watch_phase(config, run_id, "knowledge_refreshing")
    knowledge_phase.refresh()
    plan = planning_phase.execute(plan)
    _planning_runner.ensure_local_training_data_ready(config, run_id)

    _watch_state.update_watch_phase(config, run_id, "resolving_plan")
    resolved = _plan_resolution.resolve_plan_for_autopilot_config(
        plan=plan,
        config=config,
        defaults=_plan_resolution.AutopilotPlanResolutionDefaults(
            strict_competition_metric=_DEFAULT_STRICT_COMPETITION_METRIC,
            target_medal=_DEFAULT_TARGET_MEDAL,
            limited_submission_gate=_DEFAULT_LIMITED_SUBMISSION_GATE,
            max_iterations=_DEFAULT_MAX_ITERATIONS,
            heavy_local_gpu_max_cv_folds=_HEAVY_LOCAL_GPU_MAX_CV_FOLDS,
            long_local_gpu_iteration_budget_min=_LONG_LOCAL_GPU_ITERATION_BUDGET_MIN,
            long_local_gpu_max_iterations=_LONG_LOCAL_GPU_MAX_ITERATIONS,
            force_major_rank_max_percentile=_DEFAULT_FORCE_MAJOR_RANK_MAX_PERCENTILE,
            force_major_rank_min_teams=_DEFAULT_FORCE_MAJOR_RANK_MIN_TEAMS,
        ),
        on_message=print,
    )
    target_metric = resolved["target_metric"]
    target_score = resolved["target_score"]
    if target_metric is None or target_score is None:
        run_payload = _autopilot_state.build_run_payload(
            run_id=run_id,
            config=config,
            resolved=resolved,
            status="missing_target",
        )
        _autopilot_state.write_run_payload(run_dir, run_payload)
        return

    loop_settings = _autopilot_loop_settings.resolve_autopilot_loop_settings(
        config=config,
        resolved=resolved,
        target_metric=target_metric,
        target_score=target_score,
        defaults=_autopilot_loop_settings.AutopilotLoopSettingsDefaults(
            strict_competition_metric=_DEFAULT_STRICT_COMPETITION_METRIC,
            require_submit_improvement=_DEFAULT_REQUIRE_SUBMIT_IMPROVEMENT,
            force_major_on_no_improve=_DEFAULT_FORCE_MAJOR_ON_NO_IMPROVE,
            force_major_rank_max_percentile=_DEFAULT_FORCE_MAJOR_RANK_MAX_PERCENTILE,
            force_major_rank_min_teams=_DEFAULT_FORCE_MAJOR_RANK_MIN_TEAMS,
        ),
    )
    metric_direction = loop_settings.metric_direction
    deliverable_mode = loop_settings.deliverable_mode
    campaign_mode = loop_settings.campaign_mode
    portfolio_execution = loop_settings.portfolio_execution
    validation_lab_mode = loop_settings.validation_lab_mode
    research_scout_mode = loop_settings.research_scout_mode
    top1_submit_policy = loop_settings.top1_submit_policy
    submit_mode = loop_settings.submit_mode
    code_competition = bool(resolved.get("code_competition"))
    writeup_mode = loop_settings.writeup_mode
    submit_enabled = loop_settings.submit_enabled
    writeup_submit_enabled = loop_settings.writeup_submit_enabled
    deliverable_artifact_contract = resolve_deliverable_artifact_contract(
        config.paths.base_dir,
        deliverable_mode=deliverable_mode,
        submit_mode=submit_mode,
    )
    strict_competition_metric = loop_settings.strict_competition_metric
    require_submit_improvement = loop_settings.require_submit_improvement
    submit_improved_only = loop_settings.submit_improved_only
    force_major_on_no_improve = loop_settings.force_major_on_no_improve

    _plan_policy.write_resolved_plan_config(
        config.paths,
        resolved,
        default_max_iterations=_DEFAULT_MAX_ITERATIONS,
        default_force_major_rank_max_percentile=_DEFAULT_FORCE_MAJOR_RANK_MAX_PERCENTILE,
        default_force_major_rank_min_teams=_DEFAULT_FORCE_MAJOR_RANK_MIN_TEAMS,
    )
    _watch_state.update_watch_phase(config, run_id, "initializing_iterations")
    run_payload = _autopilot_state.build_run_payload(
        run_id=run_id,
        config=config,
        resolved=resolved,
        status="running",
    )
    frozen_plan_payload = _json_utils.load_json_object_or_empty(config.paths.plan_path)
    evaluation_contract, evaluation_contract_source = _resolve_authoritative_evaluation_contract(
        run_payload=run_payload,
        frozen_plan=frozen_plan_payload,
        fallback_metric=str(target_metric),
        fallback_direction=metric_direction,
        fallback_score_source=loop_settings.score_source,
    )
    run_config_payload = run_payload.get("config")
    if isinstance(run_config_payload, dict):
        run_config_payload["evaluation_contract"] = evaluation_contract
    run_payload["evaluation_contract_source"] = evaluation_contract_source
    evaluation_contract_warnings = _evaluation_spec_conflict_warnings(
        authoritative_contract=evaluation_contract,
        evaluation_spec=_json_utils.load_json_object_or_empty(config.paths.context_dir / "evaluation_spec.json"),
    )
    if evaluation_contract_warnings:
        run_payload["evaluation_contract_warnings"] = list(evaluation_contract_warnings)
        for warning in evaluation_contract_warnings:
            print(f"[yellow]evaluation contract warning[/yellow]: {warning}.")
    _autopilot_state.write_run_payload(run_dir, run_payload)
    _kernel_snapshot.ensure_best_kernel_snapshot(paths=config.paths, run_dir=run_dir)

    record_run(
        knowledge_paths=config.knowledge_paths,
        run_id=run_id,
        slug=config.slug,
        compute=config.compute,
        goal_metric=target_metric,
        goal_score=target_score,
        direction=metric_direction,
    )
    dataset_profile = knowledge_phase.load_dataset_profile()
    problem_types = knowledge_phase.derive_problem_types()
    training_route_decision = decide_training_route(
        _json_utils.load_json_object_or_empty(config.paths.plan_path),
        compute=config.compute,
        deliverable_mode=deliverable_mode,
        submit_mode=submit_mode,
        code_competition=code_competition,
    )
    direct_notebook_execution = bool(training_route_decision.direct_notebook and submit_enabled)
    training_route_payload = training_route_decision.to_dict()
    training_route_payload["direct_notebook_execution"] = direct_notebook_execution
    _json_utils.write_json_object(run_dir / "training_route.json", training_route_payload)
    run_payload["training_route"] = training_route_payload
    kernel_execution_config = config
    if direct_notebook_execution:
        kernel_execution_config = replace(
            config,
            compute="kaggle_gpu",
            accelerator="gpu",
            strict_accelerator=False,
            hardware_profile=_compute_handoff.kaggle_gpu_handoff_profile(),
        )
        print(
            "[cyan]execution route[/cyan]: skipping very heavy optional local training; "
            "running the implemented non-training solution as a guarded Kaggle notebook."
        )
    run_config_payload = run_payload.get("config")
    if isinstance(run_config_payload, dict):
        run_config_payload["execution_compute"] = kernel_execution_config.compute
        run_config_payload["execution_hardware_profile"] = kernel_execution_config.hardware_profile
    _autopilot_state.write_run_payload(run_dir, run_payload)
    submission_phase = (
        SubmissionPhase(
            config=config,
            run_id=run_id,
            problem_types=problem_types,
            submit_mode=submit_mode,
            notebook_submit_artifact_mode=str(resolved.get("notebook_submit_artifact_mode") or "wrapper"),
        )
        if submit_enabled
        else None
    )
    best_score = None
    best_submission: Path | None = None
    best_submittable_score: float | None = None
    best_submittable_submission: Path | None = None
    best_high_potential_score: float | None = None
    best_high_potential_submission: Path | None = None
    best_high_potential_iteration: int | None = None
    best_high_potential_meta: dict[str, object] | None = None
    submitted = False
    pending_problem_insights: list[dict[str, object]] = []
    pending_error_fixes: list[dict[str, object]] = []
    last_submission_result: dict[str, object] | None = None
    fallback_submit_blocked_reason: str | None = None
    writeup_bundle_meta: dict[str, object] | None = None

    max_iterations = loop_settings.max_iterations
    iteration_phase = _score_progress.IterationPhase(metric_direction=metric_direction)
    holdout_frac = loop_settings.holdout_frac
    cv_folds = loop_settings.cv_folds
    split_strategy = loop_settings.split_strategy
    seed = loop_settings.seed
    eval_seeds = loop_settings.eval_seeds
    eval_repeats = loop_settings.eval_repeats
    score_source = loop_settings.score_source
    max_total_min = loop_settings.max_total_min
    time_budget_min = loop_settings.time_budget_min
    kernel_name = loop_settings.kernel_name
    enable_internet = loop_settings.enable_internet
    submission_gate = loop_settings.submission_gate
    submission_limit_per_day = loop_settings.submission_limit_per_day
    loop_metric = loop_settings.loop_metric
    readiness_target = loop_settings.readiness_target
    readiness_method = loop_settings.readiness_method
    readiness_k = loop_settings.readiness_k
    ci_method = loop_settings.ci_method
    ci_alpha = loop_settings.ci_alpha
    target_medal = loop_settings.target_medal
    target_rank_percentile = loop_settings.target_rank_percentile
    drift_check_enabled = loop_settings.drift_check_enabled
    drift_weight = loop_settings.drift_weight
    stop_min_delta = loop_settings.stop_min_delta
    stop_no_improve_patience = loop_settings.stop_no_improve_patience
    stop_same_config_patience = loop_settings.stop_same_config_patience
    rank_force_major_max_percentile = loop_settings.rank_force_major_max_percentile
    rank_force_major_min_teams = loop_settings.rank_force_major_min_teams
    no_improve_streak = 0
    frontier_no_improve_streak = 0
    same_config_streak = 0
    last_config_hash: str | None = None
    eval_data_cache: dict[str, object] | None = None
    previous_readiness_score, noise_limited_streak = _iteration_metrics.resume_noise_guard_state(
        run_dir=config.paths.run_dir(run_id),
        max_iterations=max_iterations,
    )
    start_iteration, best_score, best_submission = _autopilot_state.resume_iteration_state(
        paths=config.paths,
        run_id=run_id,
        metric_direction=metric_direction,
        target_metric=target_metric,
        max_iterations=max_iterations,
        require_submit_phase=submit_enabled and not config.dry_run,
        load_kernel_metrics=_kernel_metrics.load_kernel_metrics,
        infer_iteration_from_submission_path=_submit_stage_duplicate.infer_iteration_from_submission_path,
    )
    best_submitted_score = _autopilot_state.resume_best_submitted_offline_score(
        paths=config.paths,
        run_id=run_id,
        metric_direction=metric_direction,
        target_metric=target_metric,
        max_iterations=max_iterations,
        load_kernel_metrics=_kernel_metrics.load_kernel_metrics,
    )
    best_online_submission_score = _leaderboard_policy.resume_best_online_submission_score(
        paths=config.paths,
        run_id=run_id,
        direction=metric_direction,
        max_iterations=max_iterations,
    )
    previous_submission_history = _submission_history.load_previous_submission_history(
        slug=config.slug,
        history_path=config.paths.context_dir / "submission_history.json",
        direction=metric_direction,
        dry_run=config.dry_run,
        fetch_submission_rows=lambda current_slug: list_competition_submissions(current_slug, dry_run=False),
        on_message=print,
    )
    historical_best_submission_score = tolerant_finite_float(previous_submission_history.get("best_score"))
    if historical_best_submission_score is not None:
        if _score_utils.should_update_best_score(
            best_submitted_score,
            historical_best_submission_score,
            metric_direction,
            0.0,
        ):
            best_submitted_score = historical_best_submission_score
        best_online_submission_score = _leaderboard_policy.update_best_online_submission_score(
            current_best_score=best_online_submission_score,
            candidate_score=historical_best_submission_score,
            direction=metric_direction,
        )
        print(
            "[cyan]submission history[/cyan]: "
            f"best public score={historical_best_submission_score:.6f} "
            f"source={previous_submission_history.get('source') or 'unknown'}"
        )
    campaign_state: dict[str, object] = {}
    campaign_state_file = campaign_state_path(config.paths.context_dir)
    campaign_registry_file = candidate_registry_path(config.paths.context_dir)
    if campaign_mode == "top1":
        campaign_state = update_campaign_state(
            state_path=campaign_state_file,
            registry_path=campaign_registry_file,
            slug=config.slug,
            run_id=run_id,
            mode=campaign_mode,
            direction=metric_direction,
            top1_info=top1_info if isinstance(top1_info, dict) else {},
            submission_history=previous_submission_history,
        )
        print(f"[cyan]campaign[/cyan]: top1 mode active; state={campaign_state_file}")
    effective_method_scout = _method_scout.effective_method_scout_mode(
        requested_mode=config.method_scout,
        campaign_mode=campaign_mode,
    )
    method_registry: dict[str, object] = {}
    source_registry: dict[str, object] = {}
    validation_lab_report: dict[str, object] | None = None
    win_contract: dict[str, object] | None = None
    if effective_method_scout != "off":
        method_registry = _method_scout.run_method_scout(
            paths=config.paths,
            slug=config.slug,
            problem_types=problem_types,
            dataset_profile=dataset_profile,
            metric=target_metric,
            campaign_state=campaign_state,
            mode=effective_method_scout,
            research_mode=research_scout_mode,
            max_sources=int(config.method_scout_max_sources or 12),
        )
        source_registry = _method_scout.load_source_registry(config.paths.source_registry_path)
        if campaign_mode == "top1":
            campaign_state = update_campaign_state(
                state_path=campaign_state_file,
                registry_path=campaign_registry_file,
                slug=config.slug,
                run_id=run_id,
                mode=campaign_mode,
                direction=metric_direction,
                top1_info=top1_info if isinstance(top1_info, dict) else {},
                submission_history=previous_submission_history,
                method_registry=method_registry,
            )
            validation_registry = _method_scout.load_validation_registry(config.paths.validation_registry_path)
            validation_lab_report = run_validation_lab(
                context_dir=config.paths.context_dir,
                validation_registry=validation_registry,
                candidate_registry_path=campaign_registry_file,
                campaign_state=campaign_state,
                mode=validation_lab_mode,
            )
            if isinstance(validation_lab_report.get("registry"), dict):
                method_registry["active_validation_profile"] = validation_lab_report["registry"].get("active_profile")
        print(f"[cyan]method scout[/cyan]: {config.paths.method_registry_path}")
    elif campaign_mode == "top1":
        method_registry = _method_scout.load_method_registry(config.paths.method_registry_path)
        source_registry = _method_scout.load_source_registry(config.paths.source_registry_path)
    if campaign_mode == "top1":
        validation_registry_for_contract = _method_scout.load_validation_registry(config.paths.validation_registry_path)
        win_contract = build_win_contract(
            context_dir=config.paths.context_dir,
            slug=config.slug,
            direction=metric_direction,
            campaign_state=campaign_state,
            top1_info=top1_info if isinstance(top1_info, dict) else {},
            submission_history=previous_submission_history,
            method_registry=method_registry,
            source_registry=source_registry,
            validation_registry=validation_registry_for_contract,
            submission_limit_per_day=submission_limit_per_day,
        )
    resumed_best_readiness = _iteration_metrics.resume_best_readiness_score(
        run_dir=config.paths.run_dir(run_id),
        direction=metric_direction,
        max_iterations=max_iterations,
    )
    if loop_metric == "rubric_readiness_score_0_100":
        resumed_rubric_score = _iteration_metrics.resume_best_kernel_metric_score(
            run_dir=config.paths.run_dir(run_id),
            metric_key=loop_metric,
            direction=metric_direction,
            max_iterations=max_iterations,
        )
        if resumed_rubric_score is not None:
            best_score = resumed_rubric_score
    elif resumed_best_readiness is not None and best_score is None:
        best_score = resumed_best_readiness
    best_submittable_score, best_submittable_submission = _autopilot_state.resume_best_submittable_iteration_state(
        paths=config.paths,
        run_id=run_id,
        metric_direction=metric_direction,
        target_metric=target_metric,
        max_iterations=max_iterations,
        load_kernel_metrics=_kernel_metrics.load_kernel_metrics,
        iteration_metrics_allow_submit=_iteration_metrics.iteration_metrics_allow_submit,
    )
    if start_iteration > 1:
        print(f"[yellow]resume[/yellow]: found completed iterations; resuming at {start_iteration}/{max_iterations}")
    loop_started_at = time.monotonic()
    last_completed_iteration = start_iteration - 1
    non_submittable_diagnostic_reason: str | None = None
    external_writeup_submission_allowed = False

    try:
        for iteration in range(start_iteration, max_iterations + 1):
            _watch_state.update_watch_phase(config, run_id, "iteration_starting", iteration=iteration)
            last_completed_iteration = iteration
            elapsed_total_min = (time.monotonic() - loop_started_at) / 60.0
            max_total_stop = _loop_control.decide_max_total_time_stop(
                elapsed_total_min=elapsed_total_min,
                max_total_min=max_total_min,
            )
            if max_total_stop.should_stop:
                _autopilot_state.apply_run_status(
                    run_payload,
                    status=max_total_stop.status,
                    stop_reason=max_total_stop.stop_reason,
                )
                print(max_total_stop.message)
                break
            iter_dir = config.paths.iter_dir(run_id, iteration)
            logs_dir = iter_dir / "logs"
            agent_dir = iter_dir / "agent"
            output_dir = iter_dir / "output"
            iter_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)
            agent_dir.mkdir(parents=True, exist_ok=True)

            print(f"[cyan]iteration[/cyan]: {iteration}/{max_iterations}")
            knowledge_phase.refresh()

            _watch_state.update_watch_phase(config, run_id, "verifying", iteration=iteration)
            _verify_artifacts.run_repo_verify(
                config.verify_cmd,
                dry_run=config.dry_run,
                artifacts_dir=config.paths.artifacts_dir,
                run_command_fn=run_command,
            )

            if config.dry_run:
                _watch_state.update_watch_phase(config, run_id, "dry_run_complete", iteration=iteration)
                _autopilot_state.apply_run_status(
                    run_payload,
                    status="completed",
                    stop_reason="dry_run_preview",
                )
                print(
                    "[yellow]DRY RUN[/yellow]: would validate and run the generated kernel, "
                    "evaluate its metrics, validate the submission artifact, and submit only if all guardrails pass."
                )
                break

            submission_path = _iteration_submission_path(
                iter_dir=iter_dir,
                sample_submission_path=config.paths.sample_submission_path,
                submission_format_path=config.paths.submission_format_md_path,
            )
            metrics_path = iter_dir / "metrics.json"
            evaluation_report_path = iter_dir / "evaluation_report.json"
            evaluation = None
            validation_blocked_result: _ValidationBlockedKernelResult | None = None
            validation_blocked_kernel_output_dir: Path | None = None
            unscored_diagnostic_result: _UnscoredDiagnosticKernelResult | None = None
            unscored_diagnostic_kernel_output_dir: Path | None = None
            kernel_metrics_payload: dict[str, object] | None = None
            kernel_metrics_artifact_path: Path | None = None
            iteration_kernel_id: str | None = None
            evaluation_by_source: dict[str, EvaluationResult] = {}
            model_summary = {}
            accelerator_used = config.accelerator
            submit_retry_resume = _autopilot_state.load_submit_retry_artifacts(
                run_dir=run_dir,
                iter_dir=iter_dir,
                iteration=iteration,
                max_iterations=max_iterations,
                metric_direction=metric_direction,
                target_metric=target_metric,
                require_submit_phase=submit_enabled and not config.dry_run,
                load_kernel_metrics=_kernel_metrics.load_kernel_metrics,
            )
            if submit_retry_resume is not None:
                resume_submission_path, resume_metrics_path, resume_evaluation = submit_retry_resume
                if resume_submission_path != submission_path:
                    submission_path = _autopilot_state.copy_submission_artifact_to_iteration_dir(
                        source=resume_submission_path,
                        iter_dir=iter_dir,
                    )
                if resume_metrics_path != metrics_path:
                    metrics_path.write_bytes(resume_metrics_path.read_bytes())
                evaluation = resume_evaluation
                kernel_metrics_payload = _json_utils.load_json_object(resume_metrics_path)
                kernel_metrics_artifact_path = resume_metrics_path
                print(
                    "[yellow]resume[/yellow]: "
                    f"iter-{iteration} has completed training artifacts; retrying submit without retraining."
                )

            if evaluation is None:
                _watch_state.update_watch_phase(config, run_id, "kernel_preflight", iteration=iteration)
                _kernel_preflight.run_kernel_source_preflight_fixes(
                    kernel_source_dir=config.paths.kernel_source_dir,
                    dry_run=config.dry_run,
                    max_attempts=_MAX_KERNEL_PREFLIGHT_FIX_ATTEMPTS,
                    format_error=_kernel_errors.format_kernel_error,
                    deliverable_mode=deliverable_artifact_contract.deliverable_mode,
                    required_output_names=deliverable_artifact_contract.required_output_names,
                    diagnostics_dir=config.paths.run_dir(run_id) / "autofix",
                    implementation_agent_alias=IMPLEMENTATION_AGENT.log_alias,
                    run_kernel_fix=lambda preflight_error, attempt: _run_kernel_fix(
                        config=config,
                        run_id=run_id,
                        iteration=iteration,
                        iter_dir=iter_dir,
                        error_message=preflight_error,
                        attempt=attempt,
                        pending_error_fixes=pending_error_fixes,
                        failure_stage="kernel_source_preflight",
                    ),
                )

            if evaluation is None and kernel_execution_config.compute.startswith("kaggle_"):
                kaggle_user = resolve_kaggle_username(kernel_execution_config.kaggle_username)
                _watch_state.update_watch_phase(
                    config,
                    run_id,
                    "kaggle_kernel_preparing",
                    detail="Building and validating the Kaggle kernel package before push.",
                    iteration=iteration,
                )
                print(f"[cyan]kernel run[/cyan]: {kernel_execution_config.compute}")
                kernel_attempts = 0
                error_fingerprints: dict[str, int] = {}

                def mark_kaggle_kernel_started(kernel_id: str) -> None:
                    _watch_state.update_watch_phase(
                        config,
                        run_id,
                        "kaggle_kernel_running",
                        detail=f"Kaggle kernel {kernel_id} was accepted and is running or queued.",
                        iteration=iteration,
                    )

                while True:
                    try:
                        kernel_result = run_kernel(
                            slug=config.slug,
                            run_id=run_id,
                            iteration=iteration,
                            base_dir=config.paths.base_dir.parent,
                            kaggle_username=kaggle_user,
                            kernel_name=kernel_name,
                            accelerator=kernel_execution_config.accelerator,
                            enable_internet=enable_internet,
                            score_source=score_source,
                            metric=target_metric,
                            direction=metric_direction,
                            holdout_frac=holdout_frac,
                            cv_folds=cv_folds,
                            seed=seed,
                            dry_run=config.dry_run,
                            timeout_minutes=(
                                min(time_budget_min or 12 * 60, 12 * 60)
                                if direct_notebook_execution
                                else time_budget_min
                            ),
                            hardware_profile=kernel_execution_config.hardware_profile,
                            on_remote_started=mark_kaggle_kernel_started,
                        )
                        iteration_kernel_id = kernel_result.kernel_id
                        if kernel_result.submission_path:
                            submission_path = _autopilot_state.copy_submission_artifact_to_iteration_dir(
                                source=kernel_result.submission_path,
                                iter_dir=iter_dir,
                            )
                        _autopilot_state.copy_kernel_support_artifacts_to_iteration_dir(
                            kernel_output_dir=kernel_result.output_dir,
                            iter_dir=iter_dir,
                        )
                        if kernel_result.metrics_path and kernel_result.metrics_path.exists():
                            kernel_metrics_artifact_path = kernel_result.metrics_path
                            kernel_metrics_payload = _json_utils.load_json_object(kernel_result.metrics_path)
                            non_training_issues = validate_non_training_metrics(
                                kernel_metrics_payload,
                                training_route_decision,
                            )
                            if non_training_issues:
                                validation_blocked_result = _resolve_validation_blocked_kernel_result(
                                    kernel_result=kernel_result,
                                    kernel_metrics=kernel_metrics_payload,
                                    training_route_decision=training_route_decision,
                                )
                                if validation_blocked_result is None:
                                    raise KernelFailedError("; ".join(non_training_issues))
                                validation_blocked_kernel_output_dir = kernel_result.output_dir
                            else:
                                unscored_diagnostic_result = _resolve_unscored_diagnostic_kernel_result(
                                    kernel_result=kernel_result,
                                    kernel_metrics=kernel_metrics_payload,
                                    training_route_decision=training_route_decision,
                                    authoritative_evaluation_contract=evaluation_contract,
                                    deliverable_mode=deliverable_mode,
                                    submit_mode=submit_mode,
                                    code_competition=code_competition,
                                )
                                if unscored_diagnostic_result is not None:
                                    unscored_diagnostic_kernel_output_dir = kernel_result.output_dir
                                else:
                                    evaluation = _load_contract_aware_kernel_metrics(
                                        metrics_path=kernel_result.metrics_path,
                                        metrics=kernel_metrics_payload,
                                        direction=metric_direction,
                                        target_metric=target_metric,
                                        authoritative_contract=evaluation_contract,
                                    )
                        if validation_blocked_result is not None or unscored_diagnostic_result is not None:
                            break
                        if evaluation is None:
                            raise KernelFailedError(
                                "Kernel metrics missing expected score; "
                                "ensure metrics.json includes a numeric metric value."
                            )
                        break
                    except RulesNotAcceptedError:
                        raise
                    except KaggleNetworkError as exc:
                        kernel_attempts += 1
                        error_text = _kernel_errors.format_kernel_error(exc)
                        _kernel_errors.record_kernel_error(
                            logs_dir=logs_dir,
                            attempt=kernel_attempts,
                            error_text=error_text,
                            error_fingerprints=error_fingerprints,
                            output_dir=output_dir,
                        )
                        raise
                    except KernelStillRunningError as exc:
                        error_text = _kernel_errors.format_kernel_error(exc)
                        logs_dir.mkdir(parents=True, exist_ok=True)
                        (logs_dir / "kernel_remote_still_running.txt").write_text(error_text + "\n", encoding="utf-8")
                        _watch_state.update_watch_phase(
                            config,
                            run_id,
                            "kaggle_kernel_still_running",
                            detail=(
                                "Kaggle notebook is still running remotely; "
                                "waiting instead of pushing a duplicate version."
                            ),
                            iteration=iteration,
                        )
                        print(
                            "[yellow]kernel still running[/yellow]: "
                            f"retrying status in {KERNEL_STILL_RUNNING_RETRY_SLEEP:.0f}s without pushing a new version"
                        )
                        time.sleep(KERNEL_STILL_RUNNING_RETRY_SLEEP)
                        continue
                    except KernelCapacityError as exc:
                        kernel_attempts += 1
                        error_text = _kernel_errors.format_kernel_error(exc)
                        _kernel_errors.record_kernel_error(
                            logs_dir=logs_dir,
                            attempt=kernel_attempts,
                            error_text=error_text,
                            error_fingerprints=error_fingerprints,
                            max_repeats=MAX_KERNEL_CAPACITY_REPEAT,
                            output_dir=output_dir,
                        )
                        capacity_retries = _env_utils.env_int(
                            "KAGGLEBOT_KERNEL_CAPACITY_RETRIES",
                            default=MAX_KERNEL_CAPACITY_RETRIES,
                        )
                        _watch_state.update_watch_phase(
                            config,
                            run_id,
                            "kaggle_gpu_no_capacity",
                            detail="Kaggle GPU session capacity is unavailable.",
                            iteration=iteration,
                        )
                        if kernel_attempts > capacity_retries:
                            raise
                        wait_seconds = KERNEL_CAPACITY_RETRY_SLEEP * kernel_attempts
                        print(
                            "[yellow]kaggle gpu limit reached[/yellow]: "
                            f"retrying in {wait_seconds:.0f}s (attempt {kernel_attempts})"
                        )
                        time.sleep(wait_seconds)
                        continue
                    except Exception as exc:  # noqa: BLE001
                        if _kernel_errors.is_kernel_registration_error(exc):
                            kernel_attempts += 1
                            error_text = _kernel_errors.format_kernel_error(exc)
                            _kernel_errors.record_kernel_error(
                                logs_dir=logs_dir,
                                attempt=kernel_attempts,
                                error_text=error_text,
                                error_fingerprints=error_fingerprints,
                                output_dir=output_dir,
                            )
                            if kernel_attempts > MAX_KERNEL_REGISTRATION_RETRIES:
                                raise
                            wait_seconds = KERNEL_REGISTRATION_RETRY_SLEEP * kernel_attempts
                            print(
                                "[yellow]kernel registration pending[/yellow]: "
                                f"retrying in {wait_seconds:.0f}s (attempt {kernel_attempts})"
                            )
                            time.sleep(wait_seconds)
                            continue
                        kernel_attempts += 1
                        error_text = _kernel_errors.format_kernel_error(exc)
                        try:
                            _kernel_errors.record_kernel_error(
                                logs_dir=logs_dir,
                                attempt=kernel_attempts,
                                error_text=error_text,
                                error_fingerprints=error_fingerprints,
                                output_dir=output_dir,
                            )
                        except KernelFailedError:
                            if _autofix_restart.maybe_regenerate_kernel_sources_once(
                                dry_run=config.dry_run,
                                agent_dir=iter_dir / "agent",
                                run_id=run_id,
                                iteration=iteration,
                                attempt=kernel_attempts,
                                trigger_reason="repeated_error_fingerprint",
                                regenerate_kernel_sources=lambda: _planning_runner.run_plan_and_initial(config, run_id),
                            ):
                                error_fingerprints.clear()
                                continue
                            raise
                        if config.dry_run:
                            raise
                        if MAX_KERNEL_FIX_ATTEMPTS is not None and kernel_attempts > MAX_KERNEL_FIX_ATTEMPTS:
                            raise
                        print(
                            f"[yellow]kernel failed[/yellow]: invoking "
                            f"{IMPLEMENTATION_AGENT.log_alias} to fix (attempt {kernel_attempts})"
                        )
                        _run_kernel_fix(
                            config=config,
                            run_id=run_id,
                            iteration=iteration,
                            iter_dir=iter_dir,
                            error_message=error_text,
                            attempt=kernel_attempts,
                            pending_error_fixes=pending_error_fixes,
                            original_error=exc if isinstance(exc, KernelFailedError) else None,
                        )
            elif evaluation is None:
                kernel_path = config.paths.kernel_source_dir / "kernel.py"
                if not kernel_path.exists():
                    raise RuntimeError(
                        "Local autopilot requires kernel.py, but "
                        f"{kernel_path} was not found. Run planning/implement to generate kernel.py first."
                    )
                _watch_state.update_watch_phase(config, run_id, "local_kernel_running", iteration=iteration)
                print(f"[cyan]kernel local run[/cyan]: {config.compute}")
                kernel_attempts = 0
                error_fingerprints = {}
                consecutive_resource_failures = 0
                previous_resource_failure_kind: str | None = None
                while True:
                    try:
                        kernel_result = run_kernel_local(
                            slug=config.slug,
                            run_id=run_id,
                            iteration=iteration,
                            base_dir=config.paths.base_dir.parent,
                            accelerator=config.accelerator,
                            score_source=score_source,
                            metric=target_metric,
                            direction=metric_direction,
                            holdout_frac=holdout_frac,
                            cv_folds=cv_folds,
                            seed=seed,
                            dry_run=config.dry_run,
                            timeout_minutes=time_budget_min,
                            strict_accelerator=config.strict_accelerator,
                            hardware_profile=config.hardware_profile,
                            plan_path=config.paths.plan_path,
                        )
                        iteration_kernel_id = kernel_result.kernel_id
                        if kernel_result.submission_path:
                            submission_path = _autopilot_state.copy_submission_artifact_to_iteration_dir(
                                source=kernel_result.submission_path,
                                iter_dir=iter_dir,
                            )
                        _autopilot_state.copy_kernel_support_artifacts_to_iteration_dir(
                            kernel_output_dir=kernel_result.output_dir,
                            iter_dir=iter_dir,
                        )
                        if kernel_result.metrics_path and kernel_result.metrics_path.exists():
                            kernel_metrics_artifact_path = kernel_result.metrics_path
                            kernel_metrics_payload = _json_utils.load_json_object(kernel_result.metrics_path)
                            non_training_issues = validate_non_training_metrics(
                                kernel_metrics_payload,
                                training_route_decision,
                            )
                            if non_training_issues:
                                validation_blocked_result = _resolve_validation_blocked_kernel_result(
                                    kernel_result=kernel_result,
                                    kernel_metrics=kernel_metrics_payload,
                                    training_route_decision=training_route_decision,
                                )
                                if validation_blocked_result is None:
                                    raise KernelFailedError("; ".join(non_training_issues))
                                validation_blocked_kernel_output_dir = kernel_result.output_dir
                            else:
                                unscored_diagnostic_result = _resolve_unscored_diagnostic_kernel_result(
                                    kernel_result=kernel_result,
                                    kernel_metrics=kernel_metrics_payload,
                                    training_route_decision=training_route_decision,
                                    authoritative_evaluation_contract=evaluation_contract,
                                    deliverable_mode=deliverable_mode,
                                    submit_mode=submit_mode,
                                    code_competition=code_competition,
                                )
                                if unscored_diagnostic_result is not None:
                                    unscored_diagnostic_kernel_output_dir = kernel_result.output_dir
                                else:
                                    evaluation = _load_contract_aware_kernel_metrics(
                                        metrics_path=kernel_result.metrics_path,
                                        metrics=kernel_metrics_payload,
                                        direction=metric_direction,
                                        target_metric=target_metric,
                                        authoritative_contract=evaluation_contract,
                                    )
                        if validation_blocked_result is not None or unscored_diagnostic_result is not None:
                            break
                        if evaluation is None:
                            raise KernelFailedError(
                                "Local kernel metrics missing expected score; "
                                "ensure metrics.json includes a numeric metric value."
                            )
                        break
                    except Exception as exc:  # noqa: BLE001
                        kernel_attempts += 1
                        error_text = _kernel_errors.format_kernel_error(exc)
                        try:
                            _kernel_errors.record_kernel_error(
                                logs_dir=logs_dir,
                                attempt=kernel_attempts,
                                error_text=error_text,
                                error_fingerprints=error_fingerprints,
                                output_dir=output_dir,
                            )
                        except KernelFailedError:
                            if _autofix_restart.maybe_regenerate_kernel_sources_once(
                                dry_run=config.dry_run,
                                agent_dir=iter_dir / "agent",
                                run_id=run_id,
                                iteration=iteration,
                                attempt=kernel_attempts,
                                trigger_reason="repeated_error_fingerprint",
                                regenerate_kernel_sources=lambda: _planning_runner.run_plan_and_initial(config, run_id),
                            ):
                                error_fingerprints.clear()
                                continue
                            raise
                        resource_failure_kind = _compute_handoff.local_resource_failure_kind(exc)
                        if resource_failure_kind is None:
                            consecutive_resource_failures = 0
                            previous_resource_failure_kind = None
                        elif resource_failure_kind == previous_resource_failure_kind:
                            consecutive_resource_failures += 1
                        else:
                            consecutive_resource_failures = 1
                            previous_resource_failure_kind = resource_failure_kind
                        if _compute_handoff.should_handoff_local_failure(
                            exc,
                            consecutive_failures=consecutive_resource_failures,
                        ):
                            quota = _compute_handoff.evaluate_kaggle_gpu_handoff_quota(
                                artifact_root=config.paths.base_dir.parent,
                                time_budget_minutes=time_budget_min,
                            )
                            if quota.allowed:
                                config, kernel_result = _run_local_to_kaggle_gpu_handoff(
                                    config=config,
                                    run_id=run_id,
                                    iteration=iteration,
                                    iter_dir=iter_dir,
                                    logs_dir=logs_dir,
                                    output_dir=output_dir,
                                    kernel_name=kernel_name,
                                    enable_internet=enable_internet,
                                    score_source=score_source,
                                    target_metric=target_metric,
                                    metric_direction=metric_direction,
                                    holdout_frac=holdout_frac,
                                    cv_folds=cv_folds,
                                    seed=seed,
                                    time_budget_min=time_budget_min,
                                    local_error_text=error_text,
                                    pending_error_fixes=pending_error_fixes,
                                )
                                iteration_kernel_id = kernel_result.kernel_id
                                accelerator_used = config.accelerator
                                run_config_payload = run_payload.get("config")
                                if isinstance(run_config_payload, dict):
                                    run_config_payload.update(
                                        {
                                            "compute": config.compute,
                                            "accelerator": config.accelerator,
                                            "hardware_profile": config.hardware_profile,
                                        }
                                    )
                                    _autopilot_state.write_run_payload(run_dir, run_payload)
                                if kernel_result.submission_path:
                                    submission_path = _autopilot_state.copy_submission_artifact_to_iteration_dir(
                                        source=kernel_result.submission_path,
                                        iter_dir=iter_dir,
                                    )
                                _autopilot_state.copy_kernel_support_artifacts_to_iteration_dir(
                                    kernel_output_dir=kernel_result.output_dir,
                                    iter_dir=iter_dir,
                                )
                                if kernel_result.metrics_path and kernel_result.metrics_path.exists():
                                    kernel_metrics_artifact_path = kernel_result.metrics_path
                                    kernel_metrics_payload = _json_utils.load_json_object(kernel_result.metrics_path)
                                    non_training_issues = validate_non_training_metrics(
                                        kernel_metrics_payload,
                                        training_route_decision,
                                    )
                                    if non_training_issues:
                                        validation_blocked_result = _resolve_validation_blocked_kernel_result(
                                            kernel_result=kernel_result,
                                            kernel_metrics=kernel_metrics_payload,
                                            training_route_decision=training_route_decision,
                                        )
                                        if validation_blocked_result is None:
                                            raise KernelFailedError("; ".join(non_training_issues))
                                        validation_blocked_kernel_output_dir = kernel_result.output_dir
                                    else:
                                        unscored_diagnostic_result = _resolve_unscored_diagnostic_kernel_result(
                                            kernel_result=kernel_result,
                                            kernel_metrics=kernel_metrics_payload,
                                            training_route_decision=training_route_decision,
                                            authoritative_evaluation_contract=evaluation_contract,
                                            deliverable_mode=deliverable_mode,
                                            submit_mode=submit_mode,
                                            code_competition=code_competition,
                                        )
                                        if unscored_diagnostic_result is not None:
                                            unscored_diagnostic_kernel_output_dir = kernel_result.output_dir
                                        else:
                                            evaluation = _load_contract_aware_kernel_metrics(
                                                metrics_path=kernel_result.metrics_path,
                                                metrics=kernel_metrics_payload,
                                                direction=metric_direction,
                                                target_metric=target_metric,
                                                authoritative_contract=evaluation_contract,
                                            )
                                if validation_blocked_result is not None or unscored_diagnostic_result is not None:
                                    break
                                if evaluation is None:
                                    raise KernelFailedError(
                                        "Handed-off Kaggle kernel metrics missing expected score; "
                                        "ensure metrics.json includes a numeric metric value."
                                    )
                                break
                            available = (
                                f"{quota.available_minutes}m available"
                                if quota.available_minutes is not None
                                else "quota unavailable"
                            )
                            print(
                                "[yellow]compute handoff deferred[/yellow]: "
                                f"{available}; {quota.required_minutes}m required. "
                                "Repairing and retrying on local_gpu."
                            )
                        if config.dry_run:
                            raise
                        if MAX_KERNEL_FIX_ATTEMPTS is not None and kernel_attempts > MAX_KERNEL_FIX_ATTEMPTS:
                            raise
                        print(
                            f"[yellow]local kernel failed[/yellow]: invoking "
                            f"{IMPLEMENTATION_AGENT.log_alias} to fix (attempt {kernel_attempts})"
                        )
                        _run_kernel_fix(
                            config=config,
                            run_id=run_id,
                            iteration=iteration,
                            iter_dir=iter_dir,
                            error_message=error_text,
                            attempt=kernel_attempts,
                            pending_error_fixes=pending_error_fixes,
                            original_error=exc if isinstance(exc, KernelFailedError) else None,
                        )

            if validation_blocked_result is not None or unscored_diagnostic_result is not None:
                if validation_blocked_result is not None:
                    diagnostic_status = "validation_blocked"
                    diagnostic_reason = "; ".join(validation_blocked_result.blockers)
                    diagnostic_route_result = validation_blocked_result.route_result
                    diagnostic_output_dir = validation_blocked_kernel_output_dir
                else:
                    diagnostic_status = "validated_unscored_artifact"
                    diagnostic_reason = "competition-faithful score unavailable"
                    if unscored_diagnostic_result is None:
                        raise RuntimeError("Unscored diagnostic result was not recorded.")
                    diagnostic_route_result = unscored_diagnostic_result.route_result
                    diagnostic_output_dir = unscored_diagnostic_kernel_output_dir
                if diagnostic_output_dir is None:
                    raise RuntimeError("Diagnostic kernel output directory was not recorded.")
                preserved_artifacts = _preserve_diagnostic_artifacts(
                    kernel_output_dir=diagnostic_output_dir,
                    iter_dir=iter_dir,
                    route_result=diagnostic_route_result,
                )
                non_submittable_diagnostic_reason = diagnostic_reason
                diagnostic_metrics_payload = dict(kernel_metrics_payload or {})
                diagnostic_score_status = (
                    _first_nonempty_string(diagnostic_metrics_payload.get("score_status")) or "unavailable"
                )
                asset_hashes = diagnostic_metrics_payload.get("asset_hashes")
                archive_hash = asset_hashes.get("archive") if isinstance(asset_hashes, dict) else None
                candidate_hash = _first_nonempty_string(
                    diagnostic_metrics_payload.get("selected_candidate_hash"),
                    diagnostic_metrics_payload.get("selected_packaged_hash"),
                    archive_hash,
                )
                diagnostic_metrics_payload.update(
                    {
                        "run_id": run_id,
                        "iter": iteration,
                        "iteration_status": diagnostic_status,
                        "metric_available": False,
                        "score_available": False,
                        "score_status": diagnostic_score_status,
                        "offline_value": None,
                        "value": None,
                        "improved": False,
                        "submission_ready": False,
                        "submission_attempted": False,
                        "submission_eligible": False,
                        "scored_candidate": False,
                        "diagnostic_only": True,
                        "competition_faithful": False,
                        "candidate_hash": candidate_hash,
                        "diagnostic_artifact_paths": preserved_artifacts,
                        "validation_status": diagnostic_route_result.get("validation_status"),
                        "preserved_artifacts": preserved_artifacts,
                    }
                )
                if validation_blocked_result is not None:
                    diagnostic_metrics_payload["validation_blockers"] = list(validation_blocked_result.blockers)
                if writeup_mode and unscored_diagnostic_result is not None:
                    source_report_path = diagnostic_output_dir / "writeup.md"
                    writeup_bundle_meta = build_writeup_bundle(
                        paths=config.paths,
                        run_id=run_id,
                        iteration=iteration,
                        resolved=resolved,
                        evaluation=None,
                        metrics_payload=diagnostic_metrics_payload,
                        top1_info=top1_info if isinstance(top1_info, dict) else None,
                        notebook_id=iteration_kernel_id,
                        source_report_path=source_report_path,
                    )
                    diagnostic_metrics_payload["deliverable_mode"] = "writeup"
                    diagnostic_metrics_payload["writeup_bundle"] = writeup_bundle_meta
                    run_payload["writeup_bundle"] = writeup_bundle_meta
                    external_writeup_submission_allowed = bool(
                        config.force_submit
                        and writeup_bundle_meta.get("status") == "ready_for_submit"
                        and writeup_bundle_meta.get("external_evaluation_required") is True
                    )
                _iteration_metrics.write_iteration_metrics_payload(metrics_path, diagnostic_metrics_payload)
                _autopilot_state.write_iteration_state_marker(
                    iter_dir=iter_dir,
                    run_id=run_id,
                    iteration=iteration,
                    submission_path=iter_dir / "submission.zip",
                    metrics_path=metrics_path,
                    evaluation_report_path=evaluation_report_path,
                    submit_phase_required=submit_enabled and not config.dry_run,
                    submit_phase_finished=True,
                    submit_allowed_by_gate=False,
                    submit_phase_state=diagnostic_status,
                    submitted=False,
                    readiness_score=None,
                    trained=diagnostic_metrics_payload.get("training_performed") is True,
                    iteration_status=diagnostic_status,
                )
                _autopilot_state.apply_run_status(
                    run_payload,
                    status=diagnostic_status,
                    stop_reason=diagnostic_reason,
                )
                _watch_state.update_watch_phase(
                    config,
                    run_id,
                    diagnostic_status,
                    detail=diagnostic_reason,
                    iteration=iteration,
                )
                print(f"[yellow]{diagnostic_status.replace('_', ' ')}[/yellow]: {diagnostic_reason}.")
                break

            if evaluation is None:
                raise RuntimeError("No evaluation metrics produced.")
            metric_mismatch_detected = False
            metric_mismatch_reason: str | None = None
            metric_fix_attempts = 0
            metric_recheck_attempted = False
            local_score_key = (
                kernel_metrics_payload.get("score_key_used") if isinstance(kernel_metrics_payload, dict) else None
            )
            non_primary_local_selection_score = bool(
                local_score_key in {"cv_score", "cv_metric_value"}
                and isinstance(kernel_metrics_payload, dict)
                and kernel_metrics_payload.get("cv_metric_is_primary_competition_metric") is False
            )
            if (
                non_primary_local_selection_score
                and evaluation.metric
                and target_metric
                and not _metric_matching.metrics_equivalent(evaluation.metric, target_metric)
            ):
                metric_mismatch_detected = True
                metric_mismatch_reason = (
                    f"local_selection={evaluation.metric}, official_target={target_metric}, score_key={local_score_key}"
                )
                print(
                    "[yellow]local selection score[/yellow]: "
                    f"using {local_score_key}={evaluation.value:.6f} for iteration ranking; "
                    "official competition validation remains unavailable."
                )
            while (
                not non_primary_local_selection_score
                and evaluation.metric
                and target_metric
                and (not _metric_matching.metrics_equivalent(evaluation.metric, target_metric))
            ):
                corrected_direction, confident = _metric_matching.infer_metric_direction_for_mismatch(
                    evaluation.metric,
                    metric_direction,
                )
                confidence_text = "high" if confident else "fallback"
                official_metric_override = _score_progress.resolve_explicit_official_metric_override(
                    kernel_metrics_payload,
                    target_metric=target_metric,
                    evaluation_metric=evaluation.metric,
                )
                if official_metric_override:
                    metric_direction, _ = _metric_matching.infer_metric_direction_for_mismatch(
                        official_metric_override,
                        corrected_direction,
                    )
                    print(
                        "[yellow]metric mismatch[/yellow]: "
                        f"plan={target_metric}/{metric_direction}, "
                        f"kernel={evaluation.metric}/{corrected_direction}. "
                        "Kernel metrics.json declares an explicit official competition metric; "
                        "updating the run target to match it."
                    )
                    target_metric = official_metric_override
                    resolved["target_metric"] = target_metric
                    resolved["target_direction"] = metric_direction
                    _plan_policy.write_resolved_plan_config(
                        config.paths,
                        resolved,
                        default_max_iterations=_DEFAULT_MAX_ITERATIONS,
                        default_force_major_rank_max_percentile=_DEFAULT_FORCE_MAJOR_RANK_MAX_PERCENTILE,
                        default_force_major_rank_min_teams=_DEFAULT_FORCE_MAJOR_RANK_MIN_TEAMS,
                    )
                    from kagglebot.solver.evaluate import EvaluationResult

                    evaluation = EvaluationResult(
                        score_source=evaluation.score_source,
                        metric=target_metric,
                        direction=metric_direction,  # type: ignore[arg-type]
                        value=evaluation.value,
                        std=evaluation.std,
                        train_score=evaluation.train_score,
                        val_score=evaluation.val_score,
                        fold_scores=evaluation.fold_scores,
                    )
                    continue
                if strict_competition_metric:
                    metric_mismatch_reason = (
                        f"target={target_metric}/{metric_direction}, kernel={evaluation.metric}/{corrected_direction}"
                    )
                    if (
                        not metric_recheck_attempted
                        and kernel_metrics_artifact_path is not None
                        and kernel_metrics_artifact_path.exists()
                    ):
                        metric_recheck_attempted = True
                        print(
                            "[yellow]metric mismatch[/yellow]: "
                            f"{metric_mismatch_reason} "
                            f"(direction_confidence={confidence_text}). "
                            "Strict competition metric mode is enabled; attempting same-iteration metric recheck "
                            f"before invoking {IMPLEMENTATION_AGENT.display_name}."
                        )
                        evaluation, kernel_metrics_payload, submission_path = (
                            _metric_recheck.recheck_kernel_metrics_from_artifacts(
                                submission_path=submission_path,
                                iter_dir=iter_dir,
                                metrics_artifact_path=kernel_metrics_artifact_path,
                                target_metric=target_metric,
                                metric_direction=metric_direction,
                            )
                        )
                        continue
                    metric_fix_attempts += 1
                    if metric_fix_attempts > _MAX_METRIC_FIX_ATTEMPTS:
                        metric_mismatch_detected = True
                        raise RuntimeError(
                            "Competition metric mismatch persisted after metric-only repairs "
                            f"(attempts={_MAX_METRIC_FIX_ATTEMPTS}, {metric_mismatch_reason})."
                        )
                    print(
                        "[yellow]metric mismatch[/yellow]: "
                        f"{metric_mismatch_reason} "
                        f"(direction_confidence={confidence_text}). "
                        "Strict competition metric mode is enabled; applying "
                        f"metric-only {IMPLEMENTATION_AGENT.display_name} fix "
                        f"(attempt {metric_fix_attempts}/{_MAX_METRIC_FIX_ATTEMPTS}) and re-running evaluation."
                    )
                    _metric_fix.run_metric_only_competition_metric_fix(
                        mismatch_reason=metric_mismatch_reason,
                        attempt=metric_fix_attempts,
                        codex_model=_METRIC_FIX_CODEX_MODEL,
                        codex_reasoning_effort=_METRIC_FIX_REASONING_EFFORT,
                        max_codex_passes=_MAX_METRIC_FIX_CODEX_PASSES,
                        run_kernel_fix=lambda **kwargs: _run_kernel_fix(
                            config=config,
                            run_id=run_id,
                            iteration=iteration,
                            iter_dir=iter_dir,
                            pending_error_fixes=pending_error_fixes,
                            **kwargs,
                        ),
                    )
                    evaluation, kernel_metrics_payload, submission_path = (
                        _metric_recheck.recheck_kernel_metrics_from_artifacts(
                            submission_path=submission_path,
                            iter_dir=iter_dir,
                            metrics_artifact_path=kernel_metrics_artifact_path,
                            target_metric=target_metric,
                            metric_direction=metric_direction,
                        )
                    )
                    metric_still_mismatched = bool(
                        evaluation.metric
                        and target_metric
                        and (not _metric_matching.metrics_equivalent(evaluation.metric, target_metric))
                    )
                    if metric_still_mismatched and (not config.dry_run) and (not config.compute.startswith("kaggle_")):
                        print(
                            "[yellow]metric mismatch[/yellow]: "
                            f"{metric_mismatch_reason}. "
                            "Metric-only fix was applied but metrics.json is still stale; "
                            "re-running local kernel once to materialize updated metric outputs."
                        )
                        kernel_result = run_kernel_local(
                            slug=config.slug,
                            run_id=run_id,
                            iteration=iteration,
                            base_dir=config.paths.base_dir.parent,
                            accelerator=config.accelerator,
                            score_source=score_source,
                            metric=target_metric,
                            direction=metric_direction,
                            holdout_frac=holdout_frac,
                            cv_folds=cv_folds,
                            seed=seed,
                            dry_run=config.dry_run,
                            timeout_minutes=time_budget_min,
                            strict_accelerator=config.strict_accelerator,
                            hardware_profile=config.hardware_profile,
                            plan_path=config.paths.plan_path,
                        )
                        iteration_kernel_id = kernel_result.kernel_id
                        if kernel_result.submission_path:
                            submission_path = _autopilot_state.copy_submission_artifact_to_iteration_dir(
                                source=kernel_result.submission_path,
                                iter_dir=iter_dir,
                            )
                        _autopilot_state.copy_kernel_support_artifacts_to_iteration_dir(
                            kernel_output_dir=kernel_result.output_dir,
                            iter_dir=iter_dir,
                        )
                        if kernel_result.metrics_path and kernel_result.metrics_path.exists():
                            kernel_metrics_artifact_path = kernel_result.metrics_path
                            kernel_metrics_payload = _json_utils.load_json_object(kernel_result.metrics_path)
                            evaluation = _load_contract_aware_kernel_metrics(
                                metrics_path=kernel_result.metrics_path,
                                metrics=kernel_metrics_payload,
                                direction=metric_direction,
                                target_metric=target_metric,
                                authoritative_contract=evaluation_contract,
                            )
                        if evaluation is None:
                            raise KernelFailedError(
                                "Metric-only repair rerun failed: local kernel metrics missing expected score; "
                                "ensure metrics.json includes a numeric metric value."
                            )
                    continue
                if corrected_direction != metric_direction or evaluation.metric != target_metric:
                    print(
                        "[yellow]metric mismatch[/yellow]: "
                        f"plan={target_metric}/{metric_direction}, "
                        f"kernel={evaluation.metric}/{corrected_direction} "
                        f"(direction_confidence={confidence_text}). "
                        "Updating plan to match kernel metric."
                    )
                    metric_direction = corrected_direction
                    target_metric = evaluation.metric
                    resolved["target_metric"] = target_metric
                    resolved["target_direction"] = metric_direction
                    if isinstance(top1_info, dict) and isinstance(top1_info.get("score"), (int, float)):
                        target_score = float(top1_info["score"])
                        resolved["target_score"] = target_score
                    _plan_policy.write_resolved_plan_config(
                        config.paths,
                        resolved,
                        default_max_iterations=_DEFAULT_MAX_ITERATIONS,
                        default_force_major_rank_max_percentile=_DEFAULT_FORCE_MAJOR_RANK_MAX_PERCENTILE,
                        default_force_major_rank_min_teams=_DEFAULT_FORCE_MAJOR_RANK_MIN_TEAMS,
                    )
                    from kagglebot.solver.evaluate import EvaluationResult

                    evaluation = EvaluationResult(
                        score_source=evaluation.score_source,
                        metric=target_metric,
                        direction=metric_direction,  # type: ignore[arg-type]
                        value=evaluation.value,
                        std=evaluation.std,
                        train_score=evaluation.train_score,
                        val_score=evaluation.val_score,
                        fold_scores=evaluation.fold_scores,
                    )
            _watch_state.update_watch_phase(config, run_id, "evaluating_iteration", iteration=iteration)
            report, report_payload, eval_data_cache = _iteration_metrics.build_iteration_evaluation_report(
                config=config,
                run_id=run_id,
                iteration=iteration,
                evaluation=evaluation,
                evaluation_by_source=evaluation_by_source,
                metric_direction=metric_direction,
                cv_folds=cv_folds,
                split_strategy=split_strategy,
                seed=seed,
                eval_seeds=eval_seeds,
                eval_repeats=eval_repeats,
                score_source=score_source,
                ci_method=ci_method,
                ci_alpha=ci_alpha,
                readiness_method=readiness_method,
                readiness_k=readiness_k,
                drift_check_enabled=drift_check_enabled,
                drift_weight=drift_weight,
                eval_data_cache=eval_data_cache,
            )
            report_payload = _iteration_metrics.enrich_evaluation_report_with_kernel_provenance(
                report_payload,
                kernel_metrics=kernel_metrics_payload,
            )
            _iteration_metrics.append_run_evaluation_report(
                run_dir=run_dir, iteration=iteration, payload=report_payload
            )
            evaluation_report_path = iter_dir / "evaluation_report.json"
            _iteration_metrics.write_iteration_evaluation_report(evaluation_report_path, report_payload)

            readiness_score = report.readiness_score
            print(
                f"[green]iteration complete[/green]: {evaluation.metric}={evaluation.value:.6f} "
                f"(SRS={readiness_score:.6f})"
            )
            if evaluation_by_source:
                values_line = ", ".join(
                    f"{source}={result.value:.6f}" for source, result in evaluation_by_source.items()
                )
                print(f"[cyan]evaluation sources[/cyan]: {values_line}")
            top1_score = top1_info.get("score") if isinstance(top1_info, dict) else None
            if loop_metric == "rubric_readiness_score_0_100":
                rubric_value = (
                    tolerant_finite_float(kernel_metrics_payload.get("rubric_readiness_score_0_100"))
                    if isinstance(kernel_metrics_payload, dict)
                    else None
                )
                if rubric_value is None or not 0.0 <= rubric_value <= 100.0:
                    raise KernelFailedError(
                        "Plan declares loop_metric=rubric_readiness_score_0_100, but kernel metrics "
                        "do not contain a valid 0-100 rubric readiness value."
                    )
                if rubric_value <= 1.0 and math.isclose(
                    rubric_value,
                    float(evaluation.value),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise KernelFailedError(
                        "Measurement provenance failure: the 0-1 technical proxy was copied into "
                        "the 0-100 rubric readiness loop metric."
                    )
                decision_score = float(rubric_value)
                decision_source = "offline_artifact_rubric"
                decision_target_score = float(readiness_target)
            else:
                decision_score = float(evaluation.value)
                decision_source = str(evaluation.score_source or "offline")
                decision_target_score = float(target_score)
            top1_score_value = float(top1_score) if isinstance(top1_score, (int, float)) else None
            effective_best_score, best_score_guard = _score_progress.effective_best_score_for_progress(
                prev_best=best_score,
                current_score=decision_score,
                top1_score=top1_score_value,
                direction=metric_direction,
            )
            if best_score_guard is not None and effective_best_score is not None:
                print(
                    "[yellow]best-score guard[/yellow]: "
                    f"clipped previous best from {float(best_score_guard['prev_best']):.6f} "
                    f"to {float(best_score_guard['effective_best']):.6f} "
                    f"(top1={float(best_score_guard['top1_score']):.6f}, "
                    f"margin={float(best_score_guard['margin']):.6f})."
                )
                best_score = effective_best_score
            top1_tier_by_submission = False
            submission_rank: int | None = None
            submission_total_teams: int | None = None
            submission_rank_percentile: float | None = None
            submission_rank_source: str | None = None
            submission_rank_estimate: int | None = None
            submission_total_teams_estimate: int | None = None
            submission_rank_percentile_estimate: float | None = None
            submission_rank_estimate_source: str | None = None
            medal_target_met = False
            medal_minimum_improvement_mode: str | None = None
            medal_policy_reason: str | None = None
            rank_forced_major_overhaul = False
            rank_force_reason: str | None = None
            leaderboard_anomaly_payload: dict[str, object] | None = None
            code_reference_score, code_reference_source = _code_reference.extract_code_reference_score(config.paths)
            code_reference_comparison_score = _score_progress.normalize_code_reference_score_for_comparison(
                current=decision_score,
                reference=code_reference_score,
                metric=evaluation.metric,
            )
            code_reference_delta_vs_current = (
                _score_progress.score_delta_vs_reference(
                    decision_score,
                    code_reference_comparison_score,
                    metric_direction,
                )
                if code_reference_comparison_score is not None
                else None
            )
            first_iteration_below_code_reference = bool(
                iteration == 1 and code_reference_delta_vs_current is not None and code_reference_delta_vs_current < 0.0
            )
            score_drop_vs_best = _score_progress.score_drop_vs_best(
                best_score=best_score,
                current_score=decision_score,
                direction=metric_direction,
            )
            severe_regression_detected = _score_progress.is_severe_regression_vs_best(
                metric=evaluation.metric,
                direction=metric_direction,
                best_score=best_score,
                current_score=decision_score,
            )
            conservative_feature_collapse = _score_progress.is_conservative_feature_collapse(kernel_metrics_payload)
            conservative_regression_detected = bool(severe_regression_detected and conservative_feature_collapse)

            quality_guard = _kernel_quality.build_kernel_quality_guard(
                evaluation=evaluation,
                kernel_metrics_payload=kernel_metrics_payload,
                evaluation_report=report,
                evaluation_contract=evaluation_contract,
                logs_dir=logs_dir,
                direction=metric_direction,
                iteration=iteration,
                max_iterations=max_iterations,
                force_submit=config.force_submit,
                code_reference_score=code_reference_score,
                code_reference_source=code_reference_source,
                metric_mismatch_detected=metric_mismatch_detected,
                metric_mismatch_reason=metric_mismatch_reason,
            )
            quality_allows_submit = bool(quality_guard.get("allow_submit", True))
            quality_reasons_raw = quality_guard.get("reasons")
            quality_reasons = (
                [str(item) for item in quality_reasons_raw if isinstance(item, str)]
                if isinstance(quality_reasons_raw, list)
                else []
            )
            accuracy_potential = _kernel_quality.build_accuracy_potential(
                score_source=evaluation.score_source,
                kernel_metrics_payload=kernel_metrics_payload,
                model_summary=model_summary,
                quality_guard=quality_guard,
                evaluation_contract=evaluation_contract,
            )
            non_generalizable_eval_detected = any(
                reason
                in {
                    "untrusted_score_source",
                    "oracle_override_detected",
                    "competition_metric_mismatch",
                    "competition_split_mismatch",
                    "competition_score_source_mismatch",
                    "competition_evaluation_unfaithful",
                    "missing_competitive_data",
                    "external_test_label_transfer_detected",
                }
                for reason in quality_reasons
            )
            quality_forced_major_overhaul = "below_code_reference_baseline" in quality_reasons
            quality_force_reason: str | None = None
            if quality_forced_major_overhaul:
                if code_reference_comparison_score is not None:
                    code_delta = _score_progress.score_delta_vs_reference(
                        decision_score,
                        code_reference_comparison_score,
                        metric_direction,
                    )
                    quality_force_reason = (
                        "Offline score is materially below code reference baseline: "
                        f"current={decision_score:.6f}, code_ref={code_reference_score:.6f}, "
                        f"comparison_ref={code_reference_comparison_score:.6f}, "
                        f"delta={code_delta:+.6f}, source={code_reference_source or 'unknown'}."
                    )
                else:
                    quality_force_reason = (
                        "Offline score is materially below code reference baseline detected by quality guard."
                    )
            force_initial_submit = _submission_policy.should_force_initial_submit(
                deliverable_mode=deliverable_mode,
                iteration=iteration,
                submit_enabled=submit_enabled,
                dry_run=config.dry_run,
                has_successful_submission=_submit_attempts.has_successful_submit_attempt(run_dir),
                submit_policy=str(resolved.get("submit_policy") or ""),
                submission_limit_per_day=submission_limit_per_day,
            )
            is_final_iteration = iteration >= max_iterations
            successful_submit_count = _submission_policy.submission_count_for_daily_limit(
                slug=config.slug,
                fallback_count=_submit_attempts.count_successful_submit_attempts(run_dir),
                submission_limit_per_day=submission_limit_per_day,
                dry_run=config.dry_run,
                fetch_submission_rows=lambda current_slug, dry_run: list_competition_submissions(
                    current_slug,
                    dry_run=dry_run,
                ),
                on_warning=print,
            )
            spare_daily_submission_slot = _submission_policy.has_spare_daily_submission_slot(
                submission_limit_per_day=submission_limit_per_day,
                submissions_used_today=successful_submit_count,
                iteration=iteration,
                max_iterations=max_iterations,
            )
            extra_daily_submission_slot = False
            if isinstance(submission_limit_per_day, int) and submission_limit_per_day > 0:
                remaining_daily_slots = max(0, submission_limit_per_day - max(0, int(successful_submit_count)))
                remaining_iterations = max(1, int(max_iterations) - int(iteration) + 1)
                extra_daily_submission_slot = remaining_daily_slots > remaining_iterations
            else:
                remaining_daily_slots = None
            campaign_candidate = None
            campaign_allocation = None
            reference_reproduction_report: dict[str, object] | None = None
            portfolio_plan: dict[str, object] | None = None
            blend_report: dict[str, object] | None = None
            experiment_graph: dict[str, object] | None = None
            allocator_decision: dict[str, object] | None = None
            graph_execution_report: dict[str, object] | None = None
            validation_lab_report: dict[str, object] | None = None
            private_robustness_report: dict[str, object] | None = None
            portfolio_optimizer_report: dict[str, object] | None = None
            top1_exhaustion_report: dict[str, object] | None = None
            if campaign_mode == "top1":
                campaign_category = _campaign_metrics.infer_campaign_candidate_category(
                    iteration=iteration,
                    kernel_metrics_payload=kernel_metrics_payload,
                    quality_reasons=quality_reasons,
                )
                candidate_offline_std = tolerant_finite_float(evaluation.std)
                campaign_candidate = build_campaign_candidate(
                    run_id=run_id,
                    iteration=iteration,
                    direction=metric_direction,
                    category=campaign_category,
                    offline_score=decision_score,
                    offline_std=candidate_offline_std,
                    score_source=decision_source,
                    submission_path=submission_path,
                    metrics_path=metrics_path,
                    oof_path=_campaign_metrics.extract_campaign_artifact_path(kernel_metrics_payload, "oof"),
                    prediction_path=_campaign_metrics.extract_campaign_artifact_path(
                        kernel_metrics_payload, "prediction"
                    ),
                    model_family=_campaign_metrics.infer_campaign_model_family(model_summary, kernel_metrics_payload),
                    feature_set=_campaign_metrics.infer_campaign_feature_set(model_summary, kernel_metrics_payload),
                    method_id=_campaign_metrics.extract_campaign_method_id(kernel_metrics_payload)
                    or select_method_id_for_category(method_registry, campaign_category),
                    validation_profile_id=_campaign_metrics.extract_campaign_validation_profile_id(
                        kernel_metrics_payload
                    )
                    or str(method_registry.get("active_validation_profile") or "default_cv"),
                    fold_scores=_campaign_metrics.extract_campaign_fold_scores(kernel_metrics_payload),
                    prediction_correlation=_campaign_metrics.extract_campaign_prediction_correlation(
                        kernel_metrics_payload
                    ),
                    metadata={
                        "metric": evaluation.metric,
                        "readiness_score": readiness_score,
                        "quality_reasons": quality_reasons,
                    },
                )
                campaign_candidate = replace(
                    campaign_candidate,
                    private_robustness_score=private_robustness_score(
                        campaign_candidate,
                        campaign_state=campaign_state,
                    ),
                )
                upsert_candidate(campaign_registry_file, campaign_candidate)
                campaign_state = update_campaign_state(
                    state_path=campaign_state_file,
                    registry_path=campaign_registry_file,
                    slug=config.slug,
                    run_id=run_id,
                    mode=campaign_mode,
                    direction=metric_direction,
                    top1_info=top1_info if isinstance(top1_info, dict) else {},
                    submission_history=previous_submission_history,
                    remaining_daily_slots=remaining_daily_slots,
                    method_registry=method_registry,
                )
                reference_reproduction_report = build_reference_reproduction_report(
                    context_dir=config.paths.context_dir,
                    campaign_state=campaign_state,
                    method_registry=method_registry,
                    direction=metric_direction,
                    current_candidate=campaign_candidate,
                    code_reference_score=code_reference_comparison_score,
                    code_reference_source=code_reference_source,
                )
                validation_registry = _method_scout.load_validation_registry(config.paths.validation_registry_path)
                validation_lab_report = run_validation_lab(
                    context_dir=config.paths.context_dir,
                    validation_registry=validation_registry,
                    candidate_registry_path=campaign_registry_file,
                    campaign_state=campaign_state,
                    mode=validation_lab_mode,
                )
                validation_registry = (
                    validation_lab_report.get("registry")
                    if isinstance(validation_lab_report.get("registry"), dict)
                    else validation_registry
                )
                if isinstance(validation_registry, dict):
                    campaign_state["active_validation_profile"] = validation_registry.get("active_profile")
                portfolio_plan = build_candidate_portfolio_plan(
                    iter_dir=iter_dir,
                    registry_path=campaign_registry_file,
                    method_registry=method_registry,
                    validation_registry=validation_registry,
                    campaign_state=campaign_state,
                    run_id=run_id,
                    iteration=iteration,
                    direction=metric_direction,
                )
                blend_report = build_blend_report(
                    iter_dir=iter_dir,
                    registry_path=campaign_registry_file,
                    campaign_state=campaign_state,
                    validation_registry=validation_registry,
                    direction=metric_direction,
                )
                experiment_graph = build_experiment_graph(
                    context_dir=config.paths.context_dir,
                    iter_dir=iter_dir,
                    run_id=run_id,
                    iteration=iteration,
                    portfolio_execution=portfolio_execution,
                    portfolio_plan=portfolio_plan,
                    reference_report=reference_reproduction_report,
                    blend_report=blend_report,
                    validation_registry=validation_registry,
                    method_registry=method_registry,
                    campaign_state=campaign_state,
                )
                if portfolio_execution != "off":
                    graph_execution = execute_experiment_graph(
                        graph=experiment_graph,
                        context=RunContext(
                            competition=config.competition_url or config.slug,
                            slug=config.slug,
                            run_id=run_id,
                            paths=config.paths,
                            workdir=config.paths.repo_root,
                            dry_run=config.dry_run,
                            force=False,
                            force_submit=config.force_submit,
                            message=config.message or f"kagglebot campaign {run_id}",
                            time_budget_minutes=int(config.time_budget_min or 60),
                            cv_folds=max(2, int(cv_folds)),
                            model_names=None,
                            use_stacking=False,
                            compute=config.compute,
                            accelerator=accelerator_used,
                            enable_internet=str(config.internet or "off").lower() == "on",
                            kaggle_username=config.kaggle_username,
                            strict_accelerator=config.strict_accelerator,
                            candidate_budget_minutes=config.candidate_budget_min,
                            max_candidates_per_iteration=config.max_candidates_per_iteration,
                        ),
                        runner=LocalKernelRunner(),
                        iter_dir=iter_dir,
                    )
                    graph_execution_report = graph_execution.to_payload()
                    experiment_graph = (
                        _json_utils.load_json_object_or_empty(iter_dir / "experiment_graph.json") or experiment_graph
                    )
                private_robustness_report = build_private_robustness_report(
                    context_dir=config.paths.context_dir,
                    registry_path=campaign_registry_file,
                    campaign_state=campaign_state,
                    validation_lab_report=validation_lab_report,
                    direction=metric_direction,
                )
                portfolio_optimizer_report = build_portfolio_optimizer_report(
                    iter_dir=iter_dir,
                    registry_path=campaign_registry_file,
                    campaign_state=campaign_state,
                    validation_registry=validation_registry,
                    private_robustness_report=private_robustness_report,
                    remaining_daily_slots=remaining_daily_slots,
                    submit_policy=top1_submit_policy,
                    direction=metric_direction,
                )
                source_registry = (
                    _method_scout.load_source_registry(config.paths.source_registry_path) or source_registry
                )
                top1_exhaustion_report = build_top1_exhaustion_report(
                    context_dir=config.paths.context_dir,
                    run_id=run_id,
                    iteration=iteration,
                    campaign_state=campaign_state,
                    win_contract=win_contract,
                    method_registry=method_registry,
                    source_registry=source_registry,
                    validation_lab_report=validation_lab_report,
                    private_robustness_report=private_robustness_report,
                    portfolio_optimizer_report=portfolio_optimizer_report,
                    experiment_graph=experiment_graph,
                )
            forced_submit_reason: str | None = None
            quality_submit_override = _submission_policy.decide_quality_submit_override(
                submit_enabled=submit_enabled,
                quality_allows_submit=quality_allows_submit,
                force_submit=config.force_submit,
                force_initial_submit=force_initial_submit,
                spare_daily_submission_slot=spare_daily_submission_slot,
                quality_reasons=quality_reasons,
                spare_reason=_SPARE_DAILY_SUBMIT_REASON,
            )
            quality_allows_submit = quality_submit_override.quality_allows_submit
            forced_submit_reason = quality_submit_override.forced_submit_reason
            if quality_submit_override.override_reason == _SPARE_DAILY_SUBMIT_REASON:
                print(
                    "[yellow]submit override[/yellow]: spare daily submission slots remain; "
                    "allowing submit through soft quality guard reasons."
                )
            if quality_submit_override.blocked_reason is not None:
                print(
                    "[yellow]submit blocked[/yellow]: kernel quality guard detected unstable evaluation "
                    f"({quality_submit_override.blocked_reason}); submission is deferred to a later iteration."
                )
            high_potential_improved = False
            if accuracy_potential.get("eligible"):
                if _score_progress.should_update_best_accuracy_candidate(
                    current_potential=accuracy_potential,
                    best_potential=best_high_potential_meta,
                    current_score=decision_score,
                    best_score=best_high_potential_score,
                    direction=metric_direction,
                ):
                    best_high_potential_score = decision_score
                    best_high_potential_submission = submission_path
                    best_high_potential_iteration = iteration
                    best_high_potential_meta = dict(accuracy_potential)
                    frontier_no_improve_streak = 0
                    high_potential_improved = True
                else:
                    frontier_no_improve_streak += 1
            if submit_enabled and (quality_allows_submit or config.force_submit):
                if _score_utils.should_update_best_score(best_submittable_score, decision_score, metric_direction, 0.0):
                    best_submittable_score = decision_score
                    best_submittable_submission = submission_path

            submit_improvement_allowed = True
            submit_non_improving = False
            defer_submit_for_accuracy_frontier = False
            submit_improvement_gate = _submit_gate.decide_iteration_submit_improvement_gate(
                submit_improved_only=submit_improved_only,
                force_submit=config.force_submit,
                require_submit_improvement=require_submit_improvement,
                best_submitted_score=best_submitted_score,
                current_score=decision_score,
                direction=metric_direction,
                min_improvement=stop_min_delta,
                final_iteration=is_final_iteration,
                submit_enabled=submit_enabled,
                quality_allows_submit=quality_allows_submit,
                spare_daily_submission_slot=spare_daily_submission_slot,
                submission_limit_per_day=submission_limit_per_day,
                forced_submit_reason=forced_submit_reason,
                spare_submit_reason=_SPARE_DAILY_SUBMIT_REASON,
            )
            submit_improvement_allowed = submit_improvement_gate.submit_improvement_allowed
            submit_non_improving = submit_improvement_gate.submit_non_improving
            forced_submit_reason = submit_improvement_gate.forced_submit_reason
            if submit_improvement_gate.message:
                print(submit_improvement_gate.message)
            if submit_enabled and isinstance(best_high_potential_meta, dict):
                current_priority = tolerant_int(accuracy_potential.get("frontier_priority")) or 0
                best_priority = tolerant_int(best_high_potential_meta.get("frontier_priority")) or 0
                if (
                    best_high_potential_submission is not None
                    and best_high_potential_submission != submission_path
                    and best_priority > current_priority
                    and (
                        not bool(best_high_potential_meta.get("faithful", False))
                        or not bool(best_high_potential_meta.get("trusted", False))
                    )
                ):
                    defer_submit_for_accuracy_frontier = True
                    print(
                        "[yellow]submit deferred[/yellow]: preserving a higher-potential unsubmitted candidate "
                        "instead of auto-submitting a weaker artifact."
                    )
            if (
                defer_submit_for_accuracy_frontier
                and extra_daily_submission_slot
                and quality_allows_submit
                and submit_improvement_allowed
            ):
                defer_submit_for_accuracy_frontier = False
                forced_submit_reason = forced_submit_reason or _SPARE_DAILY_SUBMIT_REASON
                print(
                    "[yellow]submit override[/yellow]: spare daily submission slots remain; "
                    "not preserving a higher-potential candidate for later."
                )
            allow_submit = _submission_policy.should_attempt_submit_for_readiness(
                gate=submission_gate,
                readiness_score=decision_score,
                readiness_target=decision_target_score,
                direction=metric_direction,
                iteration=iteration,
                max_iterations=max_iterations,
                submission_limit_per_day=submission_limit_per_day,
                successful_submissions=successful_submit_count,
                top1_score=top1_score if isinstance(top1_score, (int, float)) else None,
            )
            if not submit_improvement_allowed:
                allow_submit = False
            if defer_submit_for_accuracy_frontier:
                allow_submit = False
            if (not quality_allows_submit) and (not config.force_submit):
                allow_submit = False
            submit_non_improving = submit_enabled and submit_non_improving
            daily_submission_limit_reached = (
                submit_enabled
                and isinstance(submission_limit_per_day, int)
                and submission_limit_per_day > 0
                and max(0, int(successful_submit_count)) >= submission_limit_per_day
            )
            limited_holdback_decision = _submission_policy.decide_limited_submission_holdback(
                submit_enabled=submit_enabled,
                submission_limit_per_day=submission_limit_per_day,
                quality_allows_submit=quality_allows_submit,
                submit_improvement_allowed=submit_improvement_allowed,
                successful_submit_count=successful_submit_count,
                max_iterations=max_iterations,
                allow_submit=allow_submit,
            )
            submit_limited_holdback = limited_holdback_decision.holdback
            if limited_holdback_decision.reason == "reserved_final_slot":
                print(
                    "[yellow]submit deferred[/yellow]: reserved final submission slot "
                    "until offline score reaches top1-tier, readiness target, or final iteration."
                )
            elif limited_holdback_decision.reason == "strict_limited_cadence":
                print(
                    "[yellow]submit deferred[/yellow]: strict limited-submission cadence "
                    "is active because daily limit is lower than max iterations."
                )
            initial_probe_decision = _submission_policy.decide_initial_submit_probe(
                force_initial_submit=force_initial_submit,
                quality_allows_submit=quality_allows_submit,
                force_submit=config.force_submit,
                quality_reasons=quality_reasons,
                allow_submit=allow_submit,
                forced_submit_reason=forced_submit_reason,
                probe_reason=_FORCED_INITIAL_SUBMIT_REASON,
            )
            force_initial_submit = initial_probe_decision.force_initial_submit
            quality_allows_submit = initial_probe_decision.quality_allows_submit
            allow_submit = initial_probe_decision.allow_submit
            forced_submit_reason = initial_probe_decision.forced_submit_reason
            if initial_probe_decision.soft_probe_override:
                print(
                    "[yellow]submit override[/yellow]: iter 1 only failed a soft baseline guard; "
                    "submitting the trained/validated artifact to probe the Kaggle contract."
                )
            if initial_probe_decision.skipped_reason == "quality_guard":
                print(
                    "[yellow]submit override skipped[/yellow]: "
                    "iter 1 artifact failed training/validation quality guard; "
                    "not probing with an untrusted output."
                )
            if initial_probe_decision.probe_forced:
                submit_non_improving = False
                defer_submit_for_accuracy_frontier = False
                submit_limited_holdback = False
                print(
                    "[yellow]submit override[/yellow]: forcing the first successful submit "
                    "to probe the Kaggle submission contract."
                )
            if campaign_mode == "top1" and campaign_candidate is not None:
                campaign_allocation = allocate_submission(
                    candidate=campaign_candidate,
                    campaign_state=campaign_state,
                    remaining_daily_slots=remaining_daily_slots,
                    novelty=0.6 if campaign_candidate.category in {"blend", "validation_variant"} else 0.4,
                    calibration_exception=campaign_candidate.category == "calibration",
                    force=config.force_submit or force_initial_submit,
                )
                if not campaign_allocation.allow_submit and not config.force_submit:
                    allow_submit = False
                    submit_non_improving = False
                    defer_submit_for_accuracy_frontier = False
                    submit_limited_holdback = False
                    print(f"[yellow]campaign submit deferred[/yellow]: {campaign_allocation.reason}.")
                allocator_decision = write_allocator_decision(
                    iter_dir=iter_dir,
                    candidate=campaign_candidate,
                    allocation=campaign_allocation,
                    campaign_state=campaign_state,
                    experiment_graph=experiment_graph,
                )
                append_campaign_outcome(
                    context_dir=config.paths.context_dir,
                    run_id=run_id,
                    iteration=iteration,
                    phase="pre_submit",
                    candidate=campaign_candidate,
                    allocation=campaign_allocation,
                    campaign_state=campaign_state,
                    experiment_graph=experiment_graph,
                )
            if daily_submission_limit_reached:
                forced_submit_reason = None
                allow_submit = False
                submit_non_improving = False
                defer_submit_for_accuracy_frontier = False
                submit_limited_holdback = False
                print(
                    "[yellow]submit skipped[/yellow]: daily submission limit reached "
                    f"({successful_submit_count}/{submission_limit_per_day} used in the current 24h window)."
                )
            submit_phase_required = submit_enabled and not config.dry_run
            submit_allowed_by_gate = submit_enabled and allow_submit
            pre_submit_phase_state = _submit_stage_modes.resolve_iteration_submit_phase_state(
                submit_enabled=submit_enabled,
                daily_submission_limit_reached=daily_submission_limit_reached,
                force_initial_submit=force_initial_submit,
                quality_allows_submit=quality_allows_submit,
                force_submit=config.force_submit,
                submit_non_improving=submit_non_improving,
                defer_submit_for_accuracy_frontier=defer_submit_for_accuracy_frontier,
                submit_limited_holdback=submit_limited_holdback,
            )
            pre_submit_phase_finished = (not submit_phase_required) or (not submit_allowed_by_gate)
            submit_status_message = _submit_stage_messages.format_iteration_submit_status_message(
                iteration=iteration,
                max_iterations=max_iterations,
                submit_enabled=submit_enabled,
                submit_allowed_by_gate=submit_allowed_by_gate,
                submit_phase_state=pre_submit_phase_state,
                quality_reasons=quality_reasons,
                competition_faithfulness=quality_guard.get("competition_faithfulness")
                if isinstance(quality_guard.get("competition_faithfulness"), dict)
                else None,
            )
            if submit_status_message:
                print(submit_status_message)
            pre_submit_metrics_payload = _iteration_metrics.build_metrics_payload(
                run_id=run_id,
                iteration=iteration,
                evaluation=evaluation,
                target_score=target_score,
                met_target=_submission_policy.meets_target(decision_score, decision_target_score, metric_direction),
                top1_info=top1_info if isinstance(top1_info, dict) else {},
                compute=config.compute,
                accelerator=accelerator_used,
                holdout_frac=holdout_frac,
                cv_folds=cv_folds,
                seed=seed,
                evaluation_by_source=evaluation_by_source,
                evaluation_report=report,
                readiness_target=readiness_target,
                evaluation_contract=evaluation_contract,
                competition_faithfulness=quality_guard.get("competition_faithfulness")
                if isinstance(quality_guard.get("competition_faithfulness"), dict)
                else None,
                accuracy_potential=accuracy_potential,
            )
            _add_kernel_score_provenance(pre_submit_metrics_payload, kernel_metrics_payload)
            pre_submit_metrics_payload["checkpoint_phase"] = "pre_submit"
            pre_submit_metrics_payload["quality_guard"] = quality_guard
            pre_submit_metrics_payload["forced_submit_reason"] = forced_submit_reason or ""
            if campaign_mode == "top1":
                pre_submit_metrics_payload["campaign"] = {
                    "state_path": str(campaign_state_file),
                    "registry_path": str(campaign_registry_file),
                    "state": campaign_state,
                    "candidate": campaign_candidate.to_payload() if campaign_candidate is not None else None,
                    "allocation": campaign_allocation.to_payload() if campaign_allocation is not None else None,
                    "reference_reproduction_report": reference_reproduction_report,
                    "portfolio_plan": portfolio_plan,
                    "blend_report": blend_report,
                    "validation_lab_report": validation_lab_report,
                    "win_contract": win_contract,
                    "private_robustness_report": private_robustness_report,
                    "portfolio_optimizer_report": portfolio_optimizer_report,
                    "top1_exhaustion_report": top1_exhaustion_report,
                    "experiment_graph": experiment_graph,
                    "allocator_decision": allocator_decision,
                    "graph_execution_report": graph_execution_report,
                }
            _iteration_metrics.write_iteration_metrics_payload(metrics_path, pre_submit_metrics_payload)
            _autopilot_state.write_iteration_state_marker(
                iter_dir=iter_dir,
                run_id=run_id,
                iteration=iteration,
                submission_path=submission_path,
                metrics_path=metrics_path,
                evaluation_report_path=evaluation_report_path,
                submit_phase_required=submit_phase_required,
                submit_phase_finished=pre_submit_phase_finished,
                submit_allowed_by_gate=submit_allowed_by_gate,
                submit_phase_state=pre_submit_phase_state,
                forced_submit_reason=forced_submit_reason,
                submitted=False,
                readiness_score=readiness_score,
            )
            submission_result: dict[str, object] | None = None
            submission_skipped = False
            submit_failed_deferred = False
            online_score: float | None = None
            current_submission_outcome: dict[str, object] | None = None
            submit_phase_state = pre_submit_phase_state
            if submit_enabled and allow_submit and submission_phase is not None:
                try:
                    submission_result = submission_phase.attempt(
                        submission_path=submission_path,
                        best_score=decision_score,
                    )
                except SubmitAbortedError:
                    if _submit_failure_context.should_defer_submit_abort_to_next_iteration_for_run(
                        run_dir=run_dir,
                        compute=config.compute,
                        iteration=iteration,
                        max_iterations=max_iterations,
                    ):
                        submit_failed_deferred = True
                        submit_phase_state = _SUBMIT_FAILED_DEFERRED_STATE
                        print(
                            "[yellow]submit deferred[/yellow]: non-final Kaggle GPU submit failed; "
                            "carrying the submit contract failure into the next iteration."
                        )
                    else:
                        run_payload["status"] = "submit_failed"
                        _autopilot_state.write_run_payload(run_dir, run_payload)
                        raise
                if submission_result:
                    if bool(submission_result.get("skipped")):
                        submission_skipped = True
                        submit_phase_state = str(submission_result.get("reason") or "skipped")
                    else:
                        submit_phase_state = "submitted"
                        submitted = True
                        last_submission_result = submission_result
                        outcome_payload = submission_result.get("outcome")
                        if isinstance(outcome_payload, dict):
                            current_submission_outcome = outcome_payload
                            previous_submission_history = _submission_history.merge_current_submission_outcome(
                                history=previous_submission_history,
                                outcome=outcome_payload,
                                direction=metric_direction,
                                history_path=config.paths.context_dir / "submission_history.json",
                            )
                            online_score = tolerant_finite_float(outcome_payload.get("score"))
                            if online_score is not None:
                                print(f"[cyan]submission score[/cyan]: {online_score:.6f}")
                                if isinstance(top1_score, (int, float)):
                                    top1_tier_by_submission = _submission_policy.is_top1_tier(
                                        float(online_score),
                                        float(top1_score),
                                        metric_direction,
                                    )
                            rank_payload = _submit_rank.resolve_submission_rank_payload(
                                slug=config.slug,
                                context_dir=config.paths.context_dir,
                                direction=metric_direction,
                                outcome=outcome_payload,
                                dry_run=config.dry_run,
                                leaderboard_rank_for_score=leaderboard_rank_for_score,
                            )
                            if rank_payload:
                                rank_state = _submit_rank.resolve_submission_rank_state(
                                    rank_payload=rank_payload,
                                    rank_force_major_max_percentile=rank_force_major_max_percentile,
                                    rank_force_major_min_teams=rank_force_major_min_teams,
                                    should_force_major_overhaul_by_rank=(
                                        _leaderboard_policy.should_force_major_overhaul_by_rank
                                    ),
                                )
                                outcome_payload.update(rank_state.rank_payload)
                                submission_rank = rank_state.rank
                                submission_total_teams = rank_state.total_teams
                                submission_rank_percentile = rank_state.rank_percentile
                                submission_rank_source = rank_state.rank_source
                                submission_rank_estimate = rank_state.estimated_rank
                                submission_total_teams_estimate = rank_state.estimated_total_teams
                                submission_rank_percentile_estimate = rank_state.estimated_rank_percentile
                                submission_rank_estimate_source = rank_state.rank_estimate_source
                                rank_forced_major_overhaul = rank_state.force_major_overhaul
                                rank_force_reason = rank_state.force_reason
                                for message in rank_state.messages:
                                    print(message)
                            anomaly = _leaderboard_anomaly.assess_leaderboard_anomaly(
                                direction=metric_direction,
                                online_score=online_score,
                                offline_score=decision_score,
                                top1_score=top1_score,
                                rank=submission_rank,
                                total_teams=submission_total_teams,
                                rank_percentile=submission_rank_percentile,
                                estimated_rank=submission_rank_estimate,
                                estimated_total_teams=submission_total_teams_estimate,
                                estimated_rank_percentile=submission_rank_percentile_estimate,
                            )
                            if anomaly is not None:
                                leaderboard_anomaly_payload = anomaly.to_payload()
                                outcome_payload["leaderboard_anomaly"] = leaderboard_anomaly_payload
                                print(
                                    "[red]leaderboard implementation anomaly[/red]: "
                                    f"{leaderboard_anomaly_payload['note']}"
                                )
                            if online_score is not None:
                                quarantine_action = _submission_fidelity.persist_leaderboard_outcome_quarantine(
                                    slug=config.slug,
                                    run_id=run_id,
                                    run_state=_autopilot_state.load_run_state(run_dir),
                                    latest_submit_attempt=_submit_attempts.load_latest_submit_attempt(run_dir),
                                    anomaly=leaderboard_anomaly_payload,
                                    submission_ledger_path=config.paths.submission_ledger_path,
                                    save_run_state=lambda updates: _autopilot_state.save_run_state(run_dir, updates),
                                )
                                if quarantine_action is not None:
                                    print(f"[yellow]submission fidelity quarantine[/yellow]: {quarantine_action}")
                        submitted_tracking_score, submitted_tracking_source = (
                            _submit_tracking.submission_score_for_tracking(
                                offline_score=decision_score,
                                online_score=online_score,
                            )
                        )
                        if _score_utils.should_update_best_score(
                            best_submitted_score,
                            submitted_tracking_score,
                            metric_direction,
                            0.0,
                        ):
                            best_submitted_score = submitted_tracking_score
                            if submitted_tracking_source != "offline":
                                print(
                                    "[cyan]submit tracking[/cyan]: "
                                    f"updated best submitted score from {submitted_tracking_source}."
                                )
                else:
                    submit_phase_state = "dry_run" if config.dry_run else "attempted_no_result"
            if target_rank_percentile is not None and deliverable_mode == "leaderboard":
                medal_target_met = _leaderboard_policy.meets_rank_percentile_target(
                    rank_percentile=submission_rank_percentile,
                    estimated_rank_percentile=submission_rank_percentile_estimate,
                    target_rank_percentile=target_rank_percentile,
                )
                if not medal_target_met:
                    medal_minimum_improvement_mode = "moderate_update"
                    medal_policy_reason = _leaderboard_policy.build_medal_target_reason(
                        target_medal=target_medal,
                        target_rank_percentile=target_rank_percentile,
                        rank_percentile=submission_rank_percentile,
                        estimated_rank_percentile=submission_rank_percentile_estimate,
                    )
                    if medal_policy_reason:
                        print(f"[yellow]medal policy[/yellow]: {medal_policy_reason}")
            met_target = _submission_policy.meets_target(decision_score, decision_target_score, metric_direction)
            top1_tier = _submission_policy.is_top1_tier(decision_score, top1_score, metric_direction)
            top1_tier_by_readiness = _submission_policy.is_top1_tier(readiness_score, top1_score, metric_direction)
            noise_guard_decision = _loop_control.update_readiness_noise_guard(
                previous_readiness_score=previous_readiness_score,
                readiness_score=readiness_score,
                report_std=report.std,
                noise_limited_streak=noise_limited_streak,
            )
            previous_readiness_score = noise_guard_decision.previous_readiness_score
            delta_srs_vs_prev = noise_guard_decision.delta_srs_vs_prev
            noise_threshold = noise_guard_decision.noise_threshold
            noise_limited_streak = noise_guard_decision.noise_limited_streak
            noise_forced_major_overhaul = noise_guard_decision.force_major_overhaul
            code_reference_forced_reproduction = bool(
                first_iteration_below_code_reference or conservative_regression_detected
            )
            code_reference_force_reason: str | None = None
            if first_iteration_below_code_reference and code_reference_score is not None:
                code_reference_force_reason = (
                    "First iteration is below /code reference baseline; "
                    f"current={decision_score:.6f}, code_ref={code_reference_score:.6f}, "
                    f"comparison_ref={code_reference_comparison_score:.6f}, "
                    f"delta={float(code_reference_delta_vs_current):+.6f}. "
                    "Next iteration must implement the required reference notebook path."
                )
            elif conservative_regression_detected:
                drop_text = (
                    f"{float(score_drop_vs_best):.6f}" if isinstance(score_drop_vs_best, (int, float)) else "unknown"
                )
                code_reference_force_reason = (
                    "Detected severe regression with conservative feature collapse "
                    f"(drop_vs_best={drop_text}, max_features={_score_progress.CONSERVATIVE_COLLAPSE_MAX_FEATURES}). "
                    "Next iteration must recover from code reference baseline instead of keeping the collapsed path."
                )
            major_overhaul_policy = _submission_policy.decide_major_overhaul_policy(
                noise_forced_major_overhaul=noise_forced_major_overhaul,
                rank_forced_major_overhaul=rank_forced_major_overhaul,
                quality_forced_major_overhaul=quality_forced_major_overhaul,
                code_reference_forced_reproduction=code_reference_forced_reproduction,
                noise_limited_streak=noise_limited_streak,
                rank_force_reason=rank_force_reason,
                quality_force_reason=quality_force_reason,
                code_reference_force_reason=code_reference_force_reason,
                quality_reasons=quality_reasons,
            )
            force_major_overhaul_next = major_overhaul_policy.force_major_overhaul
            forced_major_overhaul_reason = major_overhaul_policy.forced_major_overhaul_reason
            fallback_submit_blocked_reason = major_overhaul_policy.fallback_submit_blocked_reason

            metrics_payload = _iteration_metrics.build_metrics_payload(
                run_id=run_id,
                iteration=iteration,
                evaluation=evaluation,
                target_score=target_score,
                met_target=met_target,
                top1_info=top1_info,
                compute=config.compute,
                accelerator=accelerator_used,
                holdout_frac=holdout_frac,
                cv_folds=cv_folds,
                seed=seed,
                evaluation_by_source=evaluation_by_source,
                evaluation_report=report,
                readiness_target=readiness_target,
                evaluation_contract=evaluation_contract,
                competition_faithfulness=quality_guard.get("competition_faithfulness")
                if isinstance(quality_guard.get("competition_faithfulness"), dict)
                else None,
                accuracy_potential=accuracy_potential,
            )
            _add_kernel_score_provenance(metrics_payload, kernel_metrics_payload)
            campaign_metrics_payload = None
            if campaign_mode == "top1":
                campaign_metrics_payload = {
                    "state_path": str(campaign_state_file),
                    "registry_path": str(campaign_registry_file),
                    "state": campaign_state,
                    "candidate": campaign_candidate.to_payload() if campaign_candidate is not None else None,
                    "allocation": campaign_allocation.to_payload() if campaign_allocation is not None else None,
                    "reference_reproduction_report": reference_reproduction_report,
                    "portfolio_plan": portfolio_plan,
                    "blend_report": blend_report,
                    "validation_lab_report": validation_lab_report,
                    "win_contract": win_contract,
                    "private_robustness_report": private_robustness_report,
                    "portfolio_optimizer_report": portfolio_optimizer_report,
                    "top1_exhaustion_report": top1_exhaustion_report,
                    "experiment_graph": experiment_graph,
                    "allocator_decision": allocator_decision,
                    "graph_execution_report": graph_execution_report,
                }
            metrics_payload = _iteration_metrics.build_final_metrics_payload(
                base_payload=metrics_payload,
                loop_decision_source=decision_source,
                loop_decision_value=decision_score,
                noise_guard=_iteration_metrics.build_noise_guard_payload(
                    delta_srs_vs_prev=delta_srs_vs_prev,
                    noise_threshold=noise_threshold,
                    noise_limited_streak=noise_limited_streak,
                    force_major_overhaul_next=force_major_overhaul_next,
                ),
                rank_guard=_iteration_metrics.build_rank_guard_payload(
                    target_medal=target_medal,
                    target_rank_percentile=target_rank_percentile,
                    target_rank_met=medal_target_met,
                    minimum_improvement_mode=medal_minimum_improvement_mode,
                    rank=submission_rank,
                    total_teams=submission_total_teams,
                    rank_percentile=submission_rank_percentile,
                    rank_source=submission_rank_source,
                    estimated_rank=submission_rank_estimate,
                    estimated_total_teams=submission_total_teams_estimate,
                    estimated_rank_percentile=submission_rank_percentile_estimate,
                    rank_estimate_source=submission_rank_estimate_source,
                    max_percentile=rank_force_major_max_percentile,
                    min_teams=rank_force_major_min_teams,
                    force_major_overhaul_next=rank_forced_major_overhaul,
                ),
                top1_tier_offline_decision=top1_tier,
                top1_tier_by_readiness=top1_tier_by_readiness,
                top1_tier_by_submission=top1_tier_by_submission,
                forced_submit_reason=forced_submit_reason,
                online_score=online_score,
                campaign_payload=campaign_metrics_payload,
                best_score_guard=best_score_guard,
                quality_guard=quality_guard,
                regression_guard=_iteration_metrics.build_regression_guard_payload(
                    best_score_before_iteration=best_score,
                    score_drop_vs_best=score_drop_vs_best,
                    severe_regression_detected=severe_regression_detected,
                    conservative_feature_collapse=conservative_feature_collapse,
                    conservative_regression_detected=conservative_regression_detected,
                    first_iteration_below_code_reference=first_iteration_below_code_reference,
                    code_reference_score=code_reference_score,
                    code_reference_comparison_score=code_reference_comparison_score,
                    code_reference_delta_vs_current=code_reference_delta_vs_current,
                    code_reference_forced_reproduction=code_reference_forced_reproduction,
                ),
                model_selection_decision=(
                    report_payload.get("model_selection_decision")
                    if isinstance(report_payload.get("model_selection_decision"), dict)
                    else None
                ),
            )
            _iteration_metrics.write_iteration_metrics_payload(metrics_path, metrics_payload)

            diff_summary = "Diff tracking disabled (git integration removed)."
            diagnostics = _diagnostics.build_diagnostics(
                evaluation=evaluation,
                model_summary=model_summary,
                best_score=best_score,
                target_score=decision_target_score,
                dataset_profile=dataset_profile,
                top1_score=top1_score,
                top1_tier=top1_tier,
                diff_summary=diff_summary,
                evaluation_by_source=evaluation_by_source,
                loop_decision_score=decision_score,
                loop_decision_source=decision_source,
                quality_guard=quality_guard,
                accuracy_potential=accuracy_potential,
            )
            _diagnostics.write_iteration_diagnostics(iter_dir=iter_dir, diagnostics=diagnostics)

            competition_policy = load_competition_policy(config.paths)
            reference_inputs_manifest_payload = _json_utils.load_json_object(
                config.paths.reference_inputs_manifest_path
            )
            repair_signals = _iteration_signals.collect_iteration_repair_signals(
                kernel_metrics_payload=kernel_metrics_payload,
                diagnostics_text=diagnostics,
                reference_inputs_manifest_payload=reference_inputs_manifest_payload,
                enable_missing_ensemble_signal=competition_policy.repair.missing_ensemble_signal,
                enable_original_data_unused_signal=competition_policy.repair.original_data_unused_signal,
                enable_same_family_plateau_signal=competition_policy.repair.same_family_plateau_signal,
                direction=metric_direction,
                previous_best_offline=best_score,
                current_offline=decision_score,
                previous_best_online=best_online_submission_score,
                current_online=online_score,
                previous_submission_history=previous_submission_history,
                detect_subgroup_collapse_signal=_kernel_quality.detect_subgroup_collapse_signal,
                detect_online_history_regression_signal=(
                    _submission_history.detect_online_regression_vs_submission_history
                ),
                leaderboard_anomaly_signal=leaderboard_anomaly_payload,
            )
            best_online_submission_score = _leaderboard_policy.update_best_online_submission_score(
                current_best_score=best_online_submission_score,
                candidate_score=online_score,
                direction=metric_direction,
            )
            if campaign_mode == "top1" and campaign_candidate is not None:
                campaign_submission_succeeded = _campaign_metrics.campaign_submission_succeeded(
                    submission_result=submission_result,
                    submission_skipped=submission_skipped,
                )
                if campaign_submission_succeeded:
                    campaign_candidate = replace(
                        campaign_candidate,
                        submitted=True,
                        public_score=_campaign_metrics.campaign_public_score_from_online_score(online_score),
                    )
                    upsert_candidate(campaign_registry_file, campaign_candidate)
                campaign_state = update_campaign_state(
                    state_path=campaign_state_file,
                    registry_path=campaign_registry_file,
                    slug=config.slug,
                    run_id=run_id,
                    mode=campaign_mode,
                    direction=metric_direction,
                    top1_info=top1_info if isinstance(top1_info, dict) else {},
                    submission_history=previous_submission_history,
                    latest_public_score=online_score,
                    remaining_daily_slots=remaining_daily_slots,
                    method_registry=method_registry,
                )
                campaign_outcome_phase = _campaign_metrics.campaign_outcome_phase(
                    submission_result=submission_result,
                    submission_skipped=submission_skipped,
                )
                append_campaign_outcome(
                    context_dir=config.paths.context_dir,
                    run_id=run_id,
                    iteration=iteration,
                    phase=campaign_outcome_phase,
                    candidate=campaign_candidate,
                    allocation=campaign_allocation,
                    campaign_state=campaign_state,
                    experiment_graph=experiment_graph,
                )

            prefer_validation_redesign = (
                campaign_mode == "top1"
                and _campaign_metrics.campaign_prefers_validation_redesign(campaign_state, method_registry)
            )
            repair_signal_policy = _iteration_signals.apply_iteration_repair_signal_policy(
                iteration=iteration,
                orig_proba_signal=repair_signals.orig_proba_signal,
                original_data_unused_signal=repair_signals.original_data_unused_signal,
                pseudo_label_signal=repair_signals.pseudo_label_signal,
                missing_ensemble_signal=repair_signals.missing_ensemble_signal,
                same_family_plateau_signal=repair_signals.same_family_plateau_signal,
                subgroup_collapse_signal=repair_signals.subgroup_collapse_signal,
                online_mismatch_signal=repair_signals.online_mismatch_signal,
                online_history_regression_signal=repair_signals.online_history_regression_signal,
                leaderboard_anomaly_signal=repair_signals.leaderboard_anomaly_signal,
                minimum_improvement_mode=medal_minimum_improvement_mode,
                minimum_improvement_reason=medal_policy_reason,
                force_major_overhaul=force_major_overhaul_next,
                forced_major_overhaul_reason=forced_major_overhaul_reason,
                prefer_validation_redesign=prefer_validation_redesign,
                upgrade_improvement_mode=_plan_policy.upgrade_improvement_mode,
            )
            extra_policy_notes = repair_signal_policy.extra_policy_notes
            minimum_improvement_mode_next = repair_signal_policy.minimum_improvement_mode
            minimum_improvement_reason_next = repair_signal_policy.minimum_improvement_reason
            force_major_overhaul_next = repair_signal_policy.force_major_overhaul
            forced_major_overhaul_reason = repair_signal_policy.forced_major_overhaul_reason
            forced_validation_redesign_reason = repair_signal_policy.forced_validation_redesign_reason
            loop_signal_errors = repair_signal_policy.loop_signal_errors
            loop_signal_problems = repair_signal_policy.loop_signal_problems
            if repair_signal_policy.repair_signals is not None:
                metrics_payload["repair_signals"] = repair_signal_policy.repair_signals
            if leaderboard_anomaly_payload is not None:
                metrics_payload["leaderboard_anomaly"] = leaderboard_anomaly_payload
            metrics_payload["previous_submission_history"] = previous_submission_history
            metrics_payload["next_iteration_policy"] = repair_signal_policy.next_iteration_policy
            _iteration_metrics.write_iteration_metrics_payload(metrics_path, metrics_payload)

            _iteration_signals.record_iteration_repair_signal_knowledge(
                knowledge_paths=config.knowledge_paths,
                slug=config.slug,
                run_id=run_id,
                iteration=iteration,
                problem_types=problem_types,
                loop_signal_errors=loop_signal_errors,
                loop_signal_problems=loop_signal_problems,
                submission_score=online_score,
                record_error_fix_insight=record_error_fix_insight,
                record_problem_type_insight=record_problem_type_insight,
            )

            if writeup_mode:
                writeup_bundle_meta = build_writeup_bundle(
                    paths=config.paths,
                    run_id=run_id,
                    iteration=iteration,
                    resolved=resolved,
                    evaluation=evaluation,
                    metrics_payload=metrics_payload,
                    top1_info=top1_info if isinstance(top1_info, dict) else None,
                    notebook_id=iteration_kernel_id,
                )
                metrics_payload["deliverable_mode"] = "writeup"
                metrics_payload["writeup_bundle"] = writeup_bundle_meta
                _iteration_metrics.write_iteration_metrics_payload(metrics_path, metrics_payload)

            submit_phase_completion = _iteration_metrics.resolve_iteration_submit_phase_completion(
                submit_enabled=submit_enabled,
                allow_submit=allow_submit,
                submit_phase_required=submit_phase_required,
                submission_result=submission_result,
                submit_failed_deferred=submit_failed_deferred,
            )

            iteration_record_kwargs = _iteration_metrics.build_iteration_record_kwargs(
                knowledge_paths=config.knowledge_paths,
                run_id=run_id,
                iteration=iteration,
                evaluation=evaluation,
                top1_info=top1_info if isinstance(top1_info, dict) else {},
                met_target=met_target,
            )
            record_iteration(**iteration_record_kwargs)
            _autopilot_state.write_iteration_state_marker(
                iter_dir=iter_dir,
                run_id=run_id,
                iteration=iteration,
                submission_path=submission_path,
                metrics_path=metrics_path,
                evaluation_report_path=evaluation_report_path,
                submit_phase_required=submit_phase_required,
                submit_phase_finished=submit_phase_completion.submit_phase_finished,
                submit_allowed_by_gate=submit_phase_completion.submit_allowed_by_gate,
                submit_phase_state=submit_phase_state,
                forced_submit_reason=forced_submit_reason,
                submitted=submission_result is not None and not submission_skipped,
                readiness_score=readiness_score,
            )

            prev_best = best_score
            score_update_decision = _loop_control.decide_iteration_score_update(
                metric_mismatch_detected=metric_mismatch_detected,
                non_generalizable_eval_detected=non_generalizable_eval_detected,
                previous_best_score=prev_best,
                current_score=decision_score,
                submission_path=submission_path,
                no_improve_streak=no_improve_streak,
                stop_min_delta=stop_min_delta,
                conservative_regression_detected=conservative_regression_detected,
                delta_from_best=iteration_phase.delta_from_best,
                should_update_best=iteration_phase.should_update_best,
            )
            delta_offline = score_update_decision.delta_offline
            improved = score_update_decision.improved
            no_improve_streak = score_update_decision.no_improve_streak
            if score_update_decision.best_score is not None:
                best_score = score_update_decision.best_score
            if score_update_decision.best_submission is not None:
                best_submission = score_update_decision.best_submission
            if score_update_decision.capture_best_snapshot:
                _kernel_snapshot.capture_best_kernel_snapshot(paths=config.paths, run_dir=run_dir)
            if score_update_decision.restore_regression_snapshot:
                restored = _kernel_snapshot.restore_best_kernel_snapshot(paths=config.paths, run_dir=run_dir)
                if restored:
                    print(
                        "[yellow]kernel regression guard[/yellow]: "
                        "restored best-known kernel source after severe conservative regression."
                    )

            major_overhaul_decision = _loop_control.decide_no_improve_major_overhaul(
                force_enabled=force_major_on_no_improve,
                improved=improved,
                high_potential_improved=high_potential_improved,
                best_score_guarded=best_score_guard is not None,
                metric_name=evaluation.metric,
                current_score=decision_score,
                previous_best_score=float(prev_best) if prev_best is not None else None,
            )
            if major_overhaul_decision.skip_message:
                print(major_overhaul_decision.skip_message)
            if major_overhaul_decision.force_major_overhaul:
                force_major_overhaul_next = True
                forced_major_overhaul_reason = _loop_control.append_policy_reason(
                    forced_major_overhaul_reason,
                    major_overhaul_decision.reason,
                )

            current_config_hash = _diagnostics.pipeline_config_hash(
                model_summary=model_summary,
                metric=evaluation.metric,
                accelerator=accelerator_used,
            )
            config_streak = _loop_control.update_same_config_streak(
                current_config_hash=current_config_hash,
                last_config_hash=last_config_hash,
                same_config_streak=same_config_streak,
            )
            same_config_streak = config_streak.same_config_streak
            last_config_hash = config_streak.last_config_hash

            stagnation_track = _loop_control.select_stagnation_track(
                best_high_potential_score=best_high_potential_score,
                no_improve_streak=no_improve_streak,
                frontier_no_improve_streak=frontier_no_improve_streak,
            )
            stagnation_stop = _loop_control.decide_stagnation_stop(
                stop_allowed=not submit_enabled,
                no_improve_streak=stagnation_track.no_improve_streak,
                no_improve_patience=stop_no_improve_patience,
                stop_min_delta=stop_min_delta,
                track_label=stagnation_track.label,
                same_config_streak=same_config_streak,
                same_config_patience=stop_same_config_patience,
            )
            if stagnation_stop.should_stop:
                _autopilot_state.apply_run_status(run_payload, status="stopped", stop_reason=stagnation_stop.reason)
                print(f"[yellow]stop[/yellow]: {run_payload['stop_reason']}")
                break

            terminal_stop = _loop_control.decide_terminal_iteration_stop(
                confirmed_first_place=_score_progress.is_confirmed_first_place(
                    submission_rank,
                    submission_rank_source,
                ),
                iteration=iteration,
                max_iterations=max_iterations,
                submitted=submitted,
                allow_max_iteration_stop=False,
            )
            if terminal_stop.should_stop:
                _autopilot_state.apply_run_status(
                    run_payload,
                    status=terminal_stop.status,
                    stop_reason=terminal_stop.stop_reason,
                )
                if terminal_stop.message:
                    print(terminal_stop.message)
                break

            if top1_tier:
                print("[yellow]note[/yellow]: offline top1-tier reached; awaiting submission-score confirmation")

            terminal_stop = _loop_control.decide_terminal_iteration_stop(
                confirmed_first_place=False,
                iteration=iteration,
                max_iterations=max_iterations,
                submitted=submitted,
            )
            if terminal_stop.should_stop:
                _autopilot_state.apply_run_status(
                    run_payload,
                    status=terminal_stop.status,
                    stop_reason=terminal_stop.stop_reason,
                )
                break

            if not config.dry_run:
                previous_submission_history = _submission_history.load_previous_submission_history(
                    slug=config.slug,
                    history_path=config.paths.context_dir / "submission_history.json",
                    direction=metric_direction,
                    dry_run=False,
                    fetch_submission_rows=lambda current_slug: list_competition_submissions(
                        current_slug,
                        dry_run=False,
                    ),
                    on_message=print,
                )
                if current_submission_outcome is not None:
                    previous_submission_history = _submission_history.merge_current_submission_outcome(
                        history=previous_submission_history,
                        outcome=current_submission_outcome,
                        direction=metric_direction,
                        history_path=config.paths.context_dir / "submission_history.json",
                    )
                refreshed_best_public = tolerant_finite_float(previous_submission_history.get("best_score"))
                if refreshed_best_public is not None:
                    if _score_utils.should_update_best_score(
                        best_submitted_score,
                        refreshed_best_public,
                        metric_direction,
                        0.0,
                    ):
                        best_submitted_score = refreshed_best_public
                    best_online_submission_score = _leaderboard_policy.update_best_online_submission_score(
                        current_best_score=best_online_submission_score,
                        candidate_score=refreshed_best_public,
                        direction=metric_direction,
                    )
                feedback = _submission_history.build_public_score_feedback(previous_submission_history)
                if feedback is not None and feedback.get("latest_public_score") is not None:
                    print(
                        "[cyan]submission feedback[/cyan]: "
                        f"latest={float(feedback['latest_public_score']):.6f} "
                        f"best={float(feedback['best_public_score']):.6f} "
                        f"result={feedback.get('result')}"
                    )

            print("[cyan]improve[/cyan]: generating next iteration plan")
            _run_improvement(
                config=config,
                run_id=run_id,
                iteration=iteration,
                iter_dir=iter_dir,
                evaluation=evaluation,
                top1_info=top1_info,
                target_score=target_score,
                delta_offline=delta_offline,
                pending_problem_insights=pending_problem_insights,
                current_score=decision_score,
                current_score_source=decision_source,
                minimum_improvement_mode=minimum_improvement_mode_next,
                minimum_improvement_reason=minimum_improvement_reason_next,
                target_medal=target_medal,
                target_rank_percentile=target_rank_percentile,
                forced_improvement_mode=(
                    "implementation_audit"
                    if repair_signal_policy.implementation_audit_required
                    else "validation_redesign"
                    if forced_validation_redesign_reason and not force_major_overhaul_next
                    else "major_overhaul"
                    if force_major_overhaul_next
                    else None
                ),
                forced_improvement_reason=forced_major_overhaul_reason or forced_validation_redesign_reason,
                extra_policy_notes=extra_policy_notes,
                enforce_code_reference_implementation=code_reference_forced_reproduction,
                code_reference_enforcement_reason=code_reference_force_reason,
                best_score_so_far=best_score,
                previous_submission_history=previous_submission_history,
            )
    except KeyboardInterrupt:
        run_payload["status"] = "interrupted"
        _autopilot_state.write_run_payload(run_dir, run_payload)
        print("[yellow]run interrupted[/yellow]")
        return

    fallback_submit_blocked_reason = _submission_policy.resolve_fallback_submit_blocked_reason(
        current_reason=fallback_submit_blocked_reason,
        best_high_potential_meta=best_high_potential_meta if isinstance(best_high_potential_meta, dict) else None,
        best_high_potential_submission=best_high_potential_submission,
        best_submittable_submission=best_submittable_submission,
    )

    if non_submittable_diagnostic_reason is not None:
        print(
            "[yellow]submit skipped[/yellow]: validation assets were unavailable; "
            "diagnostic artifacts are not eligible for submission."
        )
    elif (
        submit_enabled
        and not submitted
        and best_submittable_submission is not None
        and fallback_submit_blocked_reason is None
    ):
        final_iteration_reached = last_completed_iteration >= max_iterations
        fallback_submit_gate = _submit_gate.decide_fallback_submit_gate(
            submit_improved_only=submit_improved_only,
            force_submit=config.force_submit,
            require_submit_improvement=require_submit_improvement,
            best_submittable_score=best_submittable_score,
            best_submitted_score=best_submitted_score,
            direction=metric_direction,
            min_improvement=stop_min_delta,
            final_iteration_reached=final_iteration_reached,
        )
        allow_fallback_submit = fallback_submit_gate.allow_submit
        if fallback_submit_gate.message:
            print(fallback_submit_gate.message)
        if allow_fallback_submit:
            fallback_iteration = _submit_stage_duplicate.infer_iteration_from_submission_path(
                best_submittable_submission
            )
            score_text = (
                f" score={best_submittable_score:.6f}" if isinstance(best_submittable_score, (int, float)) else ""
            )
            if fallback_iteration is not None:
                fallback_label = (
                    f"best competition-faithful artifact from iter {fallback_iteration}/{max_iterations}{score_text}"
                )
                print(f"[cyan]submit[/cyan]: using {fallback_label}.")
            else:
                print(f"[cyan]submit[/cyan]: using best competition-faithful artifact{score_text}.")
            try:
                fallback_result = submission_phase.attempt(
                    submission_path=best_submittable_submission,
                    best_score=best_submittable_score,
                )
            except SubmitAbortedError:
                run_payload["status"] = "submit_failed"
                _autopilot_state.write_run_payload(run_dir, run_payload)
                raise
            if fallback_result:
                if bool(fallback_result.get("skipped")):
                    allow_fallback_submit = False
                    print(
                        "[yellow]submit skipped[/yellow]: "
                        f"{fallback_result.get('reason') or 'duplicate submission skipped'}."
                    )
                    fallback_result = None
                else:
                    submitted = True
                    last_submission_result = fallback_result
                    tracking_decision = _submit_tracking.decide_submitted_tracking_score_update(
                        submission_result=fallback_result,
                        offline_score=best_submittable_score,
                        previous_best_score=best_submitted_score,
                        direction=metric_direction,
                    )
                    if tracking_decision.update_best_submitted_score:
                        best_submitted_score = tracking_decision.best_submitted_score
        else:
            print(
                "[yellow]submit skipped[/yellow]: fallback artifact is not better "
                "than previously submitted checkpoint score."
            )
    elif submit_enabled and not submitted and fallback_submit_blocked_reason is not None:
        print(
            "[yellow]submit skipped[/yellow]: latest iteration was not competition-faithful "
            f"({fallback_submit_blocked_reason}); refusing fallback submit from an older artifact."
        )
    elif submit_enabled and not submitted and best_submission is not None and best_submittable_submission is None:
        print(
            "[yellow]submit skipped[/yellow]: no competition-faithful fallback artifact "
            "(all candidates were blocked by quality guard)."
        )

    if (
        (non_submittable_diagnostic_reason is None or external_writeup_submission_allowed)
        and writeup_submit_enabled
        and writeup_bundle_meta is not None
    ):
        notebook_payload = writeup_bundle_meta.get("notebook")
        notebook_publish_required = (
            isinstance(notebook_payload, dict)
            and notebook_payload.get("required") is True
            and notebook_payload.get("status") != "ready"
        )
        if notebook_publish_required and config.force_submit and not config.dry_run:
            notebook_iteration = int(writeup_bundle_meta.get("iteration") or last_completed_iteration or 1)
            _watch_state.update_watch_phase(
                config,
                run_id,
                "writeup_notebook_publishing",
                detail="Publishing and verifying the required private Kaggle notebook before writeup submission.",
                iteration=notebook_iteration,
            )
            try:
                kaggle_user = resolve_kaggle_username(config.kaggle_username)
                published_notebook = run_kernel(
                    slug=config.slug,
                    run_id=run_id,
                    iteration=notebook_iteration,
                    base_dir=config.paths.base_dir.parent,
                    kaggle_username=kaggle_user,
                    kernel_name=kernel_name,
                    accelerator=config.accelerator,
                    enable_internet=enable_internet,
                    score_source=score_source,
                    metric=target_metric,
                    direction=metric_direction,
                    holdout_frac=holdout_frac,
                    cv_folds=cv_folds,
                    seed=seed,
                    dry_run=False,
                    timeout_minutes=min(time_budget_min or 12 * 60, 12 * 60),
                    hardware_profile=config.hardware_profile,
                    on_remote_started=lambda kernel_id: _watch_state.update_watch_phase(
                        config,
                        run_id,
                        "writeup_notebook_running",
                        detail=f"Required private notebook {kernel_id} is running on Kaggle.",
                        iteration=notebook_iteration,
                    ),
                )
                writeup_bundle_meta = attach_published_writeup_notebook(
                    writeup_bundle_meta,
                    kernel_id=published_notebook.kernel_id,
                    output_dir=published_notebook.output_dir,
                )
            except Exception as exc:  # noqa: BLE001
                writeup_bundle_meta["notebook_publication"] = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            _json_utils.write_json_object(
                run_dir / "writeup" / "writeup_metadata.json",
                writeup_bundle_meta,
            )
        writeup_submission = submit_validated_writeup(
            WriteupSubmissionRequest(
                slug=config.slug,
                metadata=writeup_bundle_meta,
                attempts_path=run_dir / "writeup_submit_attempts.jsonl",
                force=config.force_submit,
                dry_run=config.dry_run,
            )
        )
        writeup_bundle_meta["submission"] = writeup_submission
        writeup_bundle_meta["status"] = (
            "submitted" if writeup_submission.get("status") == "submitted" else "submit_blocked"
        )
        _json_utils.write_json_object(
            run_dir / "writeup" / "writeup_metadata.json",
            writeup_bundle_meta,
        )

    if submitted and last_submission_result:
        top1_score = top1_info.get("score") if isinstance(top1_info, dict) else None

        def load_submission_diagnostics(iteration: int) -> str:
            return _diagnostics.load_iteration_diagnostics_text(config.paths.iter_dir(run_id, iteration))

        _submit_knowledge.record_submission_knowledge(
            knowledge_paths=config.knowledge_paths,
            slug=config.slug,
            run_id=run_id,
            problem_types=problem_types,
            pending_problem_insights=pending_problem_insights,
            pending_error_fixes=pending_error_fixes,
            submission_result=last_submission_result,
            metric_direction=metric_direction,
            target_score=target_score,
            top1_score=top1_score if isinstance(top1_score, (int, float)) else None,
            load_diagnostics_text=load_submission_diagnostics,
            record_problem_type_insight=record_problem_type_insight,
            record_error_fix_insight=record_error_fix_insight,
        )
    has_successful_submission = _submit_attempts.has_successful_submit_attempt(run_dir)
    submit_obligation_satisfied = (submitted and bool(last_submission_result)) or (
        _submit_attempts.has_satisfied_submit_obligation(run_dir)
    )
    terminal_submit_contract_failed = (
        non_submittable_diagnostic_reason is None
        and submit_enabled
        and not config.dry_run
        and not submit_obligation_satisfied
    )
    if non_submittable_diagnostic_reason is None or external_writeup_submission_allowed:
        _autopilot_state.apply_final_run_status(
            run_payload,
            submitted=submitted,
            has_submission_result=bool(last_submission_result),
            has_successful_submission=has_successful_submission,
            submit_required=submit_enabled and not config.dry_run,
            submit_obligation_satisfied=submit_obligation_satisfied,
            writeup_mode=writeup_mode,
            writeup_bundle_meta=writeup_bundle_meta,
        )

    run_payload["summary"] = _autopilot_state.build_run_summary_payload(
        best_score=best_score,
        best_submission=best_submission,
        best_submittable_score=best_submittable_score,
        best_submittable_submission=best_submittable_submission,
        best_high_potential_score=best_high_potential_score,
        best_high_potential_submission=best_high_potential_submission,
        best_high_potential_iteration=best_high_potential_iteration,
        best_high_potential_meta=best_high_potential_meta,
        fallback_submit_blocked_reason=fallback_submit_blocked_reason,
    )

    _autopilot_state.write_run_payload(run_dir, run_payload)
    if terminal_submit_contract_failed:
        raise SubmitAbortedError(
            "Leaderboard submission was required, but the run reached its terminal state without a successful "
            "submission or a duplicate-submission guard match."
        )


def _run_improvement(
    config: AutopilotConfig,
    run_id: str,
    iteration: int,
    iter_dir: Path,
    evaluation: EvaluationResult,
    top1_info: dict[str, object],
    target_score: float,
    delta_offline: float | None,
    pending_problem_insights: list[dict[str, object]],
    current_score: float | None = None,
    current_score_source: str = "offline",
    minimum_improvement_mode: str | None = None,
    minimum_improvement_reason: str | None = None,
    target_medal: str | None = None,
    target_rank_percentile: float | None = None,
    forced_improvement_mode: str | None = None,
    forced_improvement_reason: str | None = None,
    extra_policy_notes: list[str] | None = None,
    enforce_code_reference_implementation: bool = False,
    code_reference_enforcement_reason: str | None = None,
    best_score_so_far: float | None = None,
    previous_submission_history: dict[str, object] | None = None,
    expected_iteration_evidence_path: Path | None = None,
    expected_iteration_evidence_sha256: str | None = None,
) -> None:
    evidence_bundle = _iteration_evidence.prepare_iteration_evidence(
        paths=config.paths,
        slug=config.slug,
        run_id=run_id,
        iteration=iteration,
        evaluation=evaluation,
        target_score=target_score,
        current_score=current_score,
        current_score_source=current_score_source,
        delta_offline=delta_offline,
        pending_problem_insights=pending_problem_insights,
        previous_submission_history=previous_submission_history,
        expected_path=expected_iteration_evidence_path,
        expected_sha256=expected_iteration_evidence_sha256,
    )
    if config.dry_run:
        _run_improvement_body(
            config=config,
            run_id=run_id,
            iteration=iteration,
            iter_dir=iter_dir,
            evaluation=evaluation,
            top1_info=top1_info,
            target_score=target_score,
            delta_offline=delta_offline,
            pending_problem_insights=pending_problem_insights,
            current_score=current_score,
            current_score_source=current_score_source,
            minimum_improvement_mode=minimum_improvement_mode,
            minimum_improvement_reason=minimum_improvement_reason,
            target_medal=target_medal,
            target_rank_percentile=target_rank_percentile,
            forced_improvement_mode=forced_improvement_mode,
            forced_improvement_reason=forced_improvement_reason,
            extra_policy_notes=extra_policy_notes,
            enforce_code_reference_implementation=enforce_code_reference_implementation,
            code_reference_enforcement_reason=code_reference_enforcement_reason,
            best_score_so_far=best_score_so_far,
            previous_submission_history=previous_submission_history,
            iteration_evidence=evidence_bundle,
            workflow_checkpoint=None,
        )
        return
    recovery_payload: dict[str, object] = {
        "iteration": int(iteration),
        "evaluation": {
            "score_source": evaluation.score_source,
            "metric": evaluation.metric,
            "direction": evaluation.direction,
            "value": evaluation.value,
            "std": evaluation.std,
            "train_score": evaluation.train_score,
            "val_score": evaluation.val_score,
            "fold_scores": evaluation.fold_scores,
        },
        "top1_info": top1_info,
        "target_score": target_score,
        "delta_offline": delta_offline,
        "pending_problem_insights": pending_problem_insights,
        "current_score": current_score,
        "current_score_source": current_score_source,
        "minimum_improvement_mode": minimum_improvement_mode,
        "minimum_improvement_reason": minimum_improvement_reason,
        "target_medal": target_medal,
        "target_rank_percentile": target_rank_percentile,
        "forced_improvement_mode": forced_improvement_mode,
        "forced_improvement_reason": forced_improvement_reason,
        "extra_policy_notes": extra_policy_notes,
        "enforce_code_reference_implementation": enforce_code_reference_implementation,
        "code_reference_enforcement_reason": code_reference_enforcement_reason,
        "best_score_so_far": best_score_so_far,
        "previous_submission_history": previous_submission_history,
        "iteration_evidence_path": str(evidence_bundle.path),
        "iteration_evidence_sha256": evidence_bundle.sha256,
    }
    with _oracle_workflow_state.oracle_workflow_checkpoint(
        run_dir=config.paths.run_dir(run_id),
        workflow_id=f"improvement-iter-{iteration}",
        workflow_kind="improvement",
        recovery_payload=recovery_payload,
    ) as checkpoint:
        _run_improvement_body(
            config=config,
            run_id=run_id,
            iteration=iteration,
            iter_dir=iter_dir,
            evaluation=evaluation,
            top1_info=top1_info,
            target_score=target_score,
            delta_offline=delta_offline,
            pending_problem_insights=pending_problem_insights,
            current_score=current_score,
            current_score_source=current_score_source,
            minimum_improvement_mode=minimum_improvement_mode,
            minimum_improvement_reason=minimum_improvement_reason,
            target_medal=target_medal,
            target_rank_percentile=target_rank_percentile,
            forced_improvement_mode=forced_improvement_mode,
            forced_improvement_reason=forced_improvement_reason,
            extra_policy_notes=extra_policy_notes,
            enforce_code_reference_implementation=enforce_code_reference_implementation,
            code_reference_enforcement_reason=code_reference_enforcement_reason,
            best_score_so_far=best_score_so_far,
            previous_submission_history=previous_submission_history,
            iteration_evidence=evidence_bundle,
            workflow_checkpoint=checkpoint,
        )


def _run_improvement_body(
    config: AutopilotConfig,
    run_id: str,
    iteration: int,
    iter_dir: Path,
    evaluation: EvaluationResult,
    top1_info: dict[str, object],
    target_score: float,
    delta_offline: float | None,
    pending_problem_insights: list[dict[str, object]],
    current_score: float | None,
    current_score_source: str,
    minimum_improvement_mode: str | None,
    minimum_improvement_reason: str | None,
    target_medal: str | None,
    target_rank_percentile: float | None,
    forced_improvement_mode: str | None,
    forced_improvement_reason: str | None,
    extra_policy_notes: list[str] | None,
    enforce_code_reference_implementation: bool,
    code_reference_enforcement_reason: str | None,
    best_score_so_far: float | None,
    previous_submission_history: dict[str, object] | None,
    iteration_evidence: _iteration_evidence.IterationEvidenceBundle,
    workflow_checkpoint: _oracle_workflow_state.OracleWorkflowCheckpoint | None,
) -> None:
    agent_dir = iter_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)

    prompt_plan = _improvement_context.build_improvement_prompt_plan(
        config=config,
        run_id=run_id,
        iteration=iteration,
        iter_dir=iter_dir,
        agent_dir=agent_dir,
        evaluation=evaluation,
        top1_info=top1_info,
        target_score=target_score,
        delta_offline=delta_offline,
        current_score=current_score,
        current_score_source=current_score_source,
        minimum_improvement_mode=minimum_improvement_mode,
        minimum_improvement_reason=minimum_improvement_reason,
        target_medal=target_medal,
        target_rank_percentile=target_rank_percentile,
        forced_improvement_mode=forced_improvement_mode,
        forced_improvement_reason=forced_improvement_reason,
        extra_policy_notes=extra_policy_notes,
        enforce_code_reference_implementation=enforce_code_reference_implementation,
        code_reference_enforcement_reason=code_reference_enforcement_reason,
        best_score_so_far=best_score_so_far,
        previous_submission_history=previous_submission_history,
        iteration_evidence_path=iteration_evidence.path,
        iteration_evidence_sha256=iteration_evidence.sha256,
        iteration_evidence_summary=iteration_evidence.prompt_summary,
        prompt_identity_args=prompt_identity_format_args(),
    )
    for notice in prompt_plan.mode_notices:
        label = "improve mode floor" if notice.kind == "floor" else "improve mode override"
        print(f"[yellow]{label}[/yellow]: {notice.previous_mode} -> {notice.new_mode} ({notice.reason})")
    prompt_path = prompt_plan.prompt_path
    base_prompt_text = prompt_plan.base_prompt_text
    strategy_prompt = prompt_plan.strategy_prompt
    strategy_dir = prompt_plan.strategy_dir
    code_reference_mandatory = prompt_plan.code_reference_mandatory
    required_reference_notebook = prompt_plan.required_reference_notebook
    _watch_state.update_watch_phase(
        config,
        run_id,
        "gpt_improvement_thinking",
        detail="GPT is drafting the next improvement strategy.",
        iteration=iteration,
    )
    strategy_text = _agent_strategy.run_improvement_strategy_prompt(
        prompt_text=strategy_prompt,
        output_dir=strategy_dir,
        dry_run=config.dry_run,
        implementation_agent_alias=IMPLEMENTATION_AGENT.log_alias,
        run_strategy_func=_run_required_oracle_strategy,
    )
    if not config.dry_run and not strategy_text.strip():
        raise OracleStrategyError("Oracle improvement strategy is required before Codex implementation.")
    if workflow_checkpoint is not None:
        workflow_checkpoint.mark_oracle_complete(response_path=strategy_dir / "strategy_last_message.txt")

    prompt_text = _improvement_context.build_improvement_implementation_prompt(
        base_prompt_text=base_prompt_text,
        strategy_text=strategy_text,
    )

    _agent_io.write_agent_prompt(prompt_path, prompt_text)
    _agent_io.print_agent_prompt(
        log_alias=IMPLEMENTATION_AGENT.log_alias,
        prompt_path=prompt_path,
        prompt_text=prompt_text,
    )

    print(f"[cyan]improve[/cyan]: running {IMPLEMENTATION_AGENT.log_alias} implementer")
    _watch_state.update_watch_phase(
        config,
        run_id,
        "gpt_improvement_fixing",
        detail=f"{IMPLEMENTATION_AGENT.display_name} is applying the improvement strategy.",
        iteration=iteration,
    )
    # Codex runner always writes execution logs (codex_exec.jsonl / codex_last_message.txt)
    # under the provided output_dir (agent_dir). Include it in the allowlist so the guard
    # does not fail on its own transcripts.
    #
    # During improvement iterations we also update competition context (e.g. leaderboard snapshots,
    # eval advisor status) and run-scoped metadata (run.json, evaluation_report.json). Those are
    # side effects of Kagglebot itself rather than agent edits, so they must be allowlisted here
    # to avoid spurious write-guard failures.
    allowed_prefixes = build_repair_write_policy(
        repo_root=config.paths.repo_root,
        data_dir=config.paths.data_dir,
        kernels_dir=config.paths.kernels_dir / run_id,
        module_file=Path(__file__),
        extra_allowed_prefixes=[agent_dir],
    )

    def _run_improve_codex_pass(*, current_prompt_path: Path, stage_suffix: str) -> tuple[str, Path]:
        capacity_attempts = max(
            1,
            _env_utils.env_int("KAGGLEBOT_AGENT_CAPACITY_ATTEMPTS", default=MAX_AGENT_CAPACITY_ATTEMPTS),
        )
        for capacity_attempt in range(1, capacity_attempts + 1):
            pass_output_dir = (
                agent_dir
                if capacity_attempt == 1
                else agent_dir / f"improve_capacity_retry{stage_suffix}-{capacity_attempt:02d}"
            )
            guard_snapshot = _backup_guarded_files(config.paths.repo_root, allowed_prefixes)
            before = _snapshot_tree(config.paths.repo_root, allowed_prefixes)
            result = run_codex(
                current_prompt_path,
                pass_output_dir,
                dry_run=config.dry_run,
                model=ORACLE_IMPLEMENTATION_AGENT.model,
                reasoning_effort=ORACLE_IMPLEMENTATION_AGENT.reasoning_effort,
                reasoning_profile=ORACLE_IMPLEMENTATION_AGENT.reasoning_profile,
                cli_profile=ORACLE_IMPLEMENTATION_AGENT.cli_profile,
                cwd=config.paths.repo_root,
            )
            after = _snapshot_tree(config.paths.repo_root, allowed_prefixes)
            _enforce_allowlist_changes(
                root=config.paths.repo_root,
                before=before,
                after=after,
                allowed_prefixes=allowed_prefixes,
                stage=f"improve_iteration_{iteration}{stage_suffix}",
                guard_snapshot=guard_snapshot,
                auto_repair=True,
            )
            response = _agent_io.read_agent_response(result.last_message_path)
            _agent_io.print_agent_response(
                log_alias=IMPLEMENTATION_AGENT.log_alias,
                response_path=result.last_message_path,
                response_text=response,
            )
            _agent_io.log_codex_sandbox_fallback(stage_label="improve", result=result)
            if result.returncode == 0:
                return response, result.last_message_path

            detail = _agent_io.agent_failure_detail(result, response)
            if _agent_io.is_agent_capacity_failure(result, response):
                if capacity_attempt < capacity_attempts:
                    wait_seconds = AGENT_CAPACITY_RETRY_SLEEP * capacity_attempt
                    print(
                        "[yellow]improve[/yellow]: "
                        f"{IMPLEMENTATION_AGENT.log_alias} capacity unavailable; "
                        f"retrying in {wait_seconds:.0f}s "
                        f"(attempt {capacity_attempt + 1}/{capacity_attempts})."
                    )
                    time.sleep(wait_seconds)
                    continue
                raise RuntimeError(
                    f"{IMPLEMENTATION_AGENT.display_name} improvement failed: transient_agent_capacity\n{detail}"
                )
            raise RuntimeError(f"{IMPLEMENTATION_AGENT.display_name} improvement failed.\n{detail}")
        raise RuntimeError(f"{IMPLEMENTATION_AGENT.display_name} improvement failed.")

    _watch_state.update_watch_phase(
        config,
        run_id,
        "gpt_improvement_fixing",
        detail=f"{IMPLEMENTATION_AGENT.display_name} is editing kernel code for the next iteration.",
        iteration=iteration,
    )
    response_text, _ = _run_improve_codex_pass(current_prompt_path=prompt_path, stage_suffix="")
    _iteration_evidence.verify_iteration_evidence_bundle(iteration_evidence)

    if code_reference_mandatory and required_reference_notebook is not None and not config.dry_run:
        kernel_path = config.paths.kernel_source_dir / "kernel.py"
        implementation_issues = _code_reference.validate_code_reference_implementation(
            kernel_path=kernel_path,
            reference=required_reference_notebook,
        )
        if implementation_issues:
            print(
                "[yellow]code reference guard[/yellow]: "
                "required reference implementation missing; rerunning "
                f"{IMPLEMENTATION_AGENT.log_alias} with strict repair prompt."
            )
            repair_prompt_path = agent_dir / f"code_reference_repair_prompt-{iteration:02d}.md"
            repair_prompt_text = _agent_prompts.build_code_reference_repair_prompt(
                base_prompt_text=base_prompt_text,
                reference=required_reference_notebook,
                issues=implementation_issues,
                kernel_path=kernel_path,
            )
            _agent_io.write_agent_prompt(repair_prompt_path, repair_prompt_text)
            _agent_io.print_agent_prompt(
                log_alias=IMPLEMENTATION_AGENT.log_alias,
                prompt_path=repair_prompt_path,
                prompt_text=repair_prompt_text,
            )
            repair_response, _ = _run_improve_codex_pass(
                current_prompt_path=repair_prompt_path,
                stage_suffix="_code_reference_repair",
            )
            _iteration_evidence.verify_iteration_evidence_bundle(iteration_evidence)
            implementation_issues = _code_reference.validate_code_reference_implementation(
                kernel_path=kernel_path,
                reference=required_reference_notebook,
            )
            if implementation_issues:
                issues_text = ", ".join(implementation_issues)
                raise RuntimeError(
                    f"Code reference implementation requirement not satisfied after repair pass (issues={issues_text})."
                )
            response_text = f"{response_text}\n\n{repair_response}".strip()

    _verify_artifacts.run_repo_verify(
        config.verify_cmd,
        dry_run=config.dry_run,
        artifacts_dir=config.paths.artifacts_dir,
        run_command_fn=run_command,
    )
    summary = response_text
    diagnostics_text = _diagnostics.load_iteration_diagnostics_text(iter_dir)
    record_improvement(
        knowledge_paths=config.knowledge_paths,
        run_id=run_id,
        iteration=iteration,
        summary=summary.strip(),
        delta_offline=delta_offline,
    )
    pending_problem_insights.append(
        {
            "iteration": iteration,
            "why_poor": diagnostics_text,
            "how_improved": strategy_text or summary,
            "delta_offline": delta_offline,
        }
    )


_LOCAL_STAGED_PLAN_SEQUENCE_ERROR = "Local kernel staged plan contains unresolved hyperparameter sequences"


def _authoritative_plan_supersedes_failed_local_stage(
    *,
    config: AutopilotConfig,
    run_id: str,
    iteration: int,
    error_message: str,
) -> bool:
    """Return whether regeneration replaced the plan that failed local staging."""
    if _LOCAL_STAGED_PLAN_SEQUENCE_ERROR not in error_message:
        return False
    source_plan_path = config.paths.plan_path
    failed_stage_plan_path = config.paths.kernels_dir / run_id / f"local-iter-{iteration}" / "plan.json"
    source_hash = _sha256_or_none(source_plan_path)
    failed_stage_hash = _sha256_or_none(failed_stage_plan_path)
    if source_hash is None or failed_stage_hash is None or source_hash == failed_stage_hash:
        return False
    try:
        _kernel_plan_validation.validate_local_kernel_plan_runtime_hyperparameters(source_plan_path)
    except KernelFailedError:
        return False
    print(
        "[yellow]kernel fix[/yellow]: authoritative plan now passes local runtime validation "
        f"and supersedes the failed staged snapshot (source_sha256={source_hash}, "
        f"failed_staged_sha256={failed_stage_hash}); retrying with a fresh stage."
    )
    return True


def _run_kernel_fix(
    *,
    config: AutopilotConfig,
    run_id: str,
    iteration: int,
    iter_dir: Path,
    error_message: str,
    attempt: int,
    pending_error_fixes: list[dict[str, object]] | None = None,
    use_gpt_strategy: bool = True,
    codex_model: str | None = None,
    codex_reasoning_effort: str | None = None,
    prompt_prefix: str = "",
    max_codex_passes: int | None = None,
    failure_stage: str = "kernel_runtime",
    original_error: KernelFailedError | None = None,
) -> _kernel_preflight.KernelFixResult:
    if config.dry_run:
        return _run_kernel_fix_body(
            config=config,
            run_id=run_id,
            iteration=iteration,
            iter_dir=iter_dir,
            error_message=error_message,
            attempt=attempt,
            pending_error_fixes=pending_error_fixes,
            use_gpt_strategy=use_gpt_strategy,
            codex_model=codex_model,
            codex_reasoning_effort=codex_reasoning_effort,
            prompt_prefix=prompt_prefix,
            max_codex_passes=max_codex_passes,
            workflow_checkpoint=None,
            failure_stage=failure_stage,
            original_error=original_error,
        )
    recovery_payload: dict[str, object] = {
        "iteration": int(iteration),
        "attempt": int(attempt),
        "error_message": error_message,
        "use_gpt_strategy": bool(use_gpt_strategy),
        "codex_model": codex_model,
        "codex_reasoning_effort": codex_reasoning_effort,
        "prompt_prefix": prompt_prefix,
        "max_codex_passes": max_codex_passes,
        "failure_stage": failure_stage,
        "failed_kernel_source_sha256": _kernel_preflight.kernel_source_sha256(config.paths.kernel_source_dir),
    }
    with _oracle_workflow_state.oracle_workflow_checkpoint(
        run_dir=config.paths.run_dir(run_id),
        workflow_id=f"kernel-fix-iter-{iteration}-attempt-{attempt}",
        workflow_kind="kernel_fix",
        recovery_payload=recovery_payload,
    ) as checkpoint:
        return _run_kernel_fix_body(
            config=config,
            run_id=run_id,
            iteration=iteration,
            iter_dir=iter_dir,
            error_message=error_message,
            attempt=attempt,
            pending_error_fixes=pending_error_fixes,
            use_gpt_strategy=use_gpt_strategy,
            codex_model=codex_model,
            codex_reasoning_effort=codex_reasoning_effort,
            prompt_prefix=prompt_prefix,
            max_codex_passes=max_codex_passes,
            workflow_checkpoint=checkpoint,
            failure_stage=failure_stage,
            original_error=original_error,
        )


def _run_kernel_fix_body(
    *,
    config: AutopilotConfig,
    run_id: str,
    iteration: int,
    iter_dir: Path,
    error_message: str,
    attempt: int,
    pending_error_fixes: list[dict[str, object]] | None,
    use_gpt_strategy: bool,
    codex_model: str | None,
    codex_reasoning_effort: str | None,
    prompt_prefix: str,
    max_codex_passes: int | None,
    workflow_checkpoint: _oracle_workflow_state.OracleWorkflowCheckpoint | None,
    failure_stage: str,
    original_error: KernelFailedError | None,
) -> _kernel_preflight.KernelFixResult:
    agent_dir = iter_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    kernel_sha_before = _kernel_preflight.kernel_source_sha256(config.paths.kernel_source_dir)
    lightweight_fix = _kernel_fix_context.prepare_lightweight_kernel_fix(
        config=config,
        iter_dir=iter_dir,
        attempt=attempt,
        error_text=error_message,
    )
    if lightweight_fix:
        print(
            f"[yellow]kernel fix[/yellow]: wrote {lightweight_fix.artifact_name}; "
            f"retrying without {IMPLEMENTATION_AGENT.log_alias} edits"
        )
        if pending_error_fixes is not None:
            pending_error_fixes.append(
                {
                    "iteration": iteration,
                    "error_message": error_message,
                    "fix_summary": f"Applied lightweight runtime autofix: {lightweight_fix.artifact_name}",
                    "resolved": True,
                }
            )
        return _kernel_preflight.KernelFixResult(
            kernel_sha_before=kernel_sha_before,
            kernel_sha_after=_kernel_preflight.kernel_source_sha256(config.paths.kernel_source_dir),
        )

    if codex_model not in {None, ORACLE_IMPLEMENTATION_AGENT.model}:
        raise ValueError(
            "Oracle-mediated kernel fixes must use the configured oracle_implementation_model "
            f"({ORACLE_IMPLEMENTATION_AGENT.model}), not {codex_model}."
        )
    if codex_reasoning_effort not in {None, ORACLE_IMPLEMENTATION_AGENT.reasoning_effort}:
        raise ValueError(
            "Oracle-mediated kernel fixes must use the configured oracle_implementation_reasoning_effort "
            f"({ORACLE_IMPLEMENTATION_AGENT.reasoning_effort}), not {codex_reasoning_effort}."
        )

    prompt_plan = _kernel_fix_context.build_kernel_fix_prompt_plan(
        config=config,
        run_id=run_id,
        iteration=iteration,
        iter_dir=iter_dir,
        agent_dir=agent_dir,
        error_message=error_message,
        attempt=attempt,
        prompt_prefix=prompt_prefix,
        use_gpt_strategy=use_gpt_strategy,
        prompt_identity_args=prompt_identity_format_args(),
        hardware_constraints=render_hardware_constraints(
            resolve_hardware_profile(config.hardware_profile, compute=config.compute),
            compute=config.compute,
            time_budget_min=config.time_budget_min,
        ),
        failure_stage=failure_stage,
    )

    strategy_text = ""
    if prompt_plan.strategy_skip_reason:
        print(
            "[yellow]kernel fix[/yellow]: "
            f"skipping gpt strategy ({prompt_plan.strategy_skip_reason}); "
            f"invoking {IMPLEMENTATION_AGENT.log_alias} fixer directly."
        )
    else:
        _watch_state.update_watch_phase(
            config,
            run_id,
            "gpt_kernel_fix_thinking",
            detail="GPT is analyzing the kernel failure and drafting a fix strategy.",
            iteration=iteration,
        )
        strategy_text = _agent_strategy.run_error_strategy_prompt(
            prompt_text=str(prompt_plan.strategy_prompt or ""),
            output_dir=prompt_plan.strategy_dir,
            dry_run=config.dry_run,
            stage_label="kernel fix",
            implementation_agent_alias=IMPLEMENTATION_AGENT.log_alias,
            strategy_model=_ERROR_STRATEGY_MODEL,
            reasoning_effort=_ERROR_STRATEGY_REASONING_EFFORT,
            run_strategy_func=_run_required_oracle_strategy,
        )
        if not config.dry_run and not strategy_text.strip():
            raise OracleStrategyError("Oracle kernel-fix strategy is required before Codex implementation.")
    if workflow_checkpoint is not None:
        response_path = prompt_plan.strategy_dir / "strategy_last_message.txt" if strategy_text.strip() else None
        workflow_checkpoint.mark_oracle_complete(response_path=response_path)
    prompt_text = _kernel_fix_context.append_kernel_fix_strategy(
        prompt_text=prompt_plan.prompt_text,
        strategy_text=strategy_text,
        strategy_agent_display_name=STRATEGY_AGENT.display_name,
    )

    base_prompt_text = f"Kernel fix attempt: {attempt}\n\n{prompt_text}"
    prompt_path = prompt_plan.prompt_path
    _agent_io.write_agent_prompt(prompt_path, base_prompt_text)
    _agent_io.write_agent_prompt(prompt_plan.attempt_path, base_prompt_text)
    _agent_io.print_agent_prompt(
        log_alias=IMPLEMENTATION_AGENT.log_alias,
        prompt_path=prompt_path,
        prompt_text=base_prompt_text,
    )

    allowed_prefixes = build_repair_write_policy(
        repo_root=config.paths.repo_root,
        data_dir=config.paths.data_dir,
        kernels_dir=config.paths.kernels_dir / run_id,
        module_file=Path(__file__),
    )
    guard_snapshot = _backup_guarded_files(config.paths.repo_root, allowed_prefixes)
    codex_pass_limit = max(1, int(max_codex_passes or MAX_KERNEL_FIX_CODEX_PASSES))
    retry_feedback = ""
    last_response_text = ""
    repo_changed_paths: set[str] = set()
    for codex_pass in range(1, codex_pass_limit + 1):
        pass_prompt_text = (
            base_prompt_text
            if not retry_feedback
            else _agent_io.append_fix_retry_feedback(
                base_prompt=base_prompt_text,
                stage_label="kernel_fix",
                codex_pass=codex_pass - 1,
                failure_text=retry_feedback,
            )
        )
        pass_prompt_path = (
            prompt_path if codex_pass == 1 else agent_dir / f"kernel_fix_prompt-{attempt:02d}-pass-{codex_pass:02d}.md"
        )
        _agent_io.write_agent_prompt(pass_prompt_path, pass_prompt_text)
        if codex_pass > 1:
            print(
                "[yellow]kernel fix[/yellow]: "
                f"retrying {IMPLEMENTATION_AGENT.log_alias} pass "
                f"{codex_pass}/{codex_pass_limit} with previous failure context."
            )
        before = _snapshot_tree(config.paths.repo_root, allowed_prefixes)
        pass_output_dir = (
            agent_dir if codex_pass == 1 else agent_dir / f"kernel_fix_pass-{attempt:02d}-{codex_pass:02d}"
        )
        print(f"[cyan]kernel fix[/cyan]: running {IMPLEMENTATION_AGENT.log_alias} fixer")
        _watch_state.update_watch_phase(
            config,
            run_id,
            "gpt_kernel_fix_fixing",
            detail=f"{IMPLEMENTATION_AGENT.display_name} is repairing the kernel failure.",
            iteration=iteration,
        )
        result = run_codex(
            pass_prompt_path,
            pass_output_dir,
            dry_run=config.dry_run,
            heartbeat_label="fixing error",
            model=ORACLE_IMPLEMENTATION_AGENT.model,
            reasoning_effort=ORACLE_IMPLEMENTATION_AGENT.reasoning_effort,
            reasoning_profile=ORACLE_IMPLEMENTATION_AGENT.reasoning_profile,
            cli_profile=ORACLE_IMPLEMENTATION_AGENT.cli_profile,
            cwd=config.paths.repo_root,
        )
        after = _snapshot_tree(config.paths.repo_root, allowed_prefixes)
        changed = _diff_snapshots(before, after)
        repo_changed_paths.update(changed)
        if changed:
            _enforce_allowlist_changes(
                root=config.paths.repo_root,
                before=before,
                after=after,
                allowed_prefixes=allowed_prefixes,
                stage=f"kernel_fix_attempt_{attempt}",
                guard_snapshot=guard_snapshot,
                auto_repair=True,
            )
            if not config.dry_run and any(path.startswith("src/") for path in changed):
                # The edit itself is durable.  Mark this Oracle-to-Codex step
                # complete before exec so the new process resumes the failed
                # iteration instead of asking another agent to repeat the same
                # source repair.
                if workflow_checkpoint is not None:
                    workflow_checkpoint.mark_completed()
            _autofix_restart.maybe_restart_for_src_changes(
                dry_run=config.dry_run,
                run_dir=config.paths.run_dir(run_id),
                repo_root=config.paths.repo_root,
                run_id=run_id,
                slug=config.slug,
                changed=changed,
                stage=f"kernel_fix_attempt_{attempt}",
                max_restarts=MAX_AUTOFIX_RESTARTS,
            )
        response_text = _agent_io.read_agent_response(result.last_message_path)
        _agent_io.print_agent_response(
            log_alias=IMPLEMENTATION_AGENT.log_alias,
            response_path=result.last_message_path,
            response_text=response_text,
        )
        _agent_io.log_codex_sandbox_fallback(stage_label="kernel fix", result=result)
        last_response_text = response_text
        if result.returncode != 0:
            retry_feedback = (
                f"{IMPLEMENTATION_AGENT.display_name} kernel-fix step failed with non-zero exit status.\n"
                f"returncode={result.returncode}\n"
                f"pass={codex_pass}/{codex_pass_limit}\n"
                f"response={response_text}"
            )
            if codex_pass < codex_pass_limit:
                continue
            raise RuntimeError(f"{IMPLEMENTATION_AGENT.display_name} kernel-fix step failed.")

        if not changed:
            kernel_sha_after_agent = _kernel_preflight.kernel_source_sha256(config.paths.kernel_source_dir)
            kernel_changed = kernel_sha_after_agent != kernel_sha_before
            regeneration_already_used = False
            regeneration_attempted = False
            regenerated = False
            if not kernel_changed:
                regeneration_already_used = _autofix_restart.kernel_regeneration_already_marked(
                    agent_dir,
                    kernel_sha256=kernel_sha_after_agent,
                )
                if failure_stage == "kernel_source_preflight":
                    deliverable_contract = resolve_deliverable_artifact_contract(config.paths.base_dir)
                    current_failure = _kernel_preflight.check_kernel_source_preflight(
                        kernel_source_dir=config.paths.kernel_source_dir,
                        format_error=_kernel_errors.format_kernel_error,
                        deliverable_mode=deliverable_contract.deliverable_mode,
                        required_output_names=deliverable_contract.required_output_names,
                    )
                    if current_failure is None:
                        print(
                            "[yellow]kernel fix[/yellow]: no repository diff, but the current "
                            "kernel source preflight passes"
                        )
                        return _kernel_preflight.KernelFixResult(
                            agent_exit_code=result.returncode,
                            repo_changed=bool(repo_changed_paths),
                            changed_paths=tuple(sorted(repo_changed_paths)),
                            kernel_sha_before=kernel_sha_before,
                            kernel_sha_after=kernel_sha_after_agent,
                            regeneration_already_used=regeneration_already_used,
                        )
                regeneration_attempted = not config.dry_run and not regeneration_already_used
                regenerated = _autofix_restart.maybe_regenerate_kernel_sources_once(
                    dry_run=config.dry_run,
                    agent_dir=agent_dir,
                    run_id=run_id,
                    iteration=iteration,
                    attempt=attempt,
                    trigger_reason="codex_no_changes",
                    regenerate_kernel_sources=lambda: _planning_runner.run_plan_and_initial(config, run_id),
                    get_kernel_sha256=lambda: _kernel_preflight.kernel_source_sha256(config.paths.kernel_source_dir),
                )
            if not kernel_changed and not regenerated:
                if _authoritative_plan_supersedes_failed_local_stage(
                    config=config,
                    run_id=run_id,
                    iteration=iteration,
                    error_message=error_message,
                ):
                    if pending_error_fixes is not None:
                        pending_error_fixes.append(
                            {
                                "iteration": iteration,
                                "error_message": error_message,
                                "fix_summary": (
                                    "Authoritative plan superseded the failed staged snapshot and passed "
                                    "local runtime-plan validation; retrying fresh staging."
                                ),
                                "resolved": True,
                            }
                        )
                    return _kernel_preflight.KernelFixResult(
                        agent_exit_code=result.returncode,
                        repo_changed=bool(repo_changed_paths),
                        changed_paths=tuple(sorted(repo_changed_paths)),
                        kernel_sha_before=kernel_sha_before,
                        kernel_sha_after=_kernel_preflight.kernel_source_sha256(config.paths.kernel_source_dir),
                        regeneration_attempted=regeneration_attempted,
                        regeneration_already_used=regeneration_already_used,
                    )
                if failure_stage == "kernel_source_preflight":
                    raise KernelFailedError(
                        "Kernel fix made no repository changes and kernel source preflight still fails.\n"
                        "Original preflight failure:\n"
                        f"{error_message}\n\n"
                        "Current preflight failure:\n"
                        f"{_kernel_preflight.format_kernel_preflight_failure(current_failure)}"
                    )
                no_change_error = KernelFailedError(
                    "Kernel fix agent produced no file changes and regeneration fallback was already used."
                )
                if original_error is not None:
                    raise original_error from no_change_error
                raise no_change_error
            try:
                _verify_artifacts.run_repo_verify(
                    config.verify_cmd,
                    dry_run=config.dry_run,
                    artifacts_dir=config.paths.artifacts_dir,
                    run_command_fn=run_command,
                )
            except Exception as exc:  # noqa: BLE001
                retry_feedback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
                if codex_pass < codex_pass_limit:
                    continue
                raise
            if pending_error_fixes is not None:
                pending_error_fixes.append(
                    {
                        "iteration": iteration,
                        "error_message": error_message,
                        "fix_summary": (
                            f"{IMPLEMENTATION_AGENT.display_name} kernel-fix changed the generated kernel directly; "
                            "verification passed."
                            if kernel_changed
                            else f"{IMPLEMENTATION_AGENT.display_name} kernel-fix made no edits; "
                            "regenerated kernel sources once and verification passed."
                        ),
                        "resolved": True,
                    }
                )
            return _kernel_preflight.KernelFixResult(
                agent_exit_code=result.returncode,
                repo_changed=bool(repo_changed_paths),
                changed_paths=tuple(sorted(repo_changed_paths)),
                kernel_sha_before=kernel_sha_before,
                kernel_sha_after=_kernel_preflight.kernel_source_sha256(config.paths.kernel_source_dir),
                regeneration_attempted=regeneration_attempted,
                regeneration_used=regenerated,
                regeneration_already_used=regeneration_already_used,
            )

        try:
            _verify_artifacts.run_repo_verify(
                config.verify_cmd,
                dry_run=config.dry_run,
                artifacts_dir=config.paths.artifacts_dir,
                run_command_fn=run_command,
            )
        except Exception as exc:  # noqa: BLE001
            retry_feedback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
            if codex_pass < codex_pass_limit:
                continue
            raise
        if pending_error_fixes is not None:
            pending_error_fixes.append(
                {
                    "iteration": iteration,
                    "error_message": error_message,
                    "fix_summary": strategy_text or last_response_text,
                    "resolved": True,
                }
            )
        return _kernel_preflight.KernelFixResult(
            agent_exit_code=result.returncode,
            repo_changed=bool(repo_changed_paths),
            changed_paths=tuple(sorted(repo_changed_paths)),
            kernel_sha_before=kernel_sha_before,
            kernel_sha_after=_kernel_preflight.kernel_source_sha256(config.paths.kernel_source_dir),
        )

    raise RuntimeError(
        f"Kernel fix exhausted {IMPLEMENTATION_AGENT.log_alias} retry passes without resolving the error."
    )


def _run_autofix(*, config: AutopilotConfig, run_id: str, attempt: int, error: Exception) -> None:
    if config.dry_run:
        _run_autofix_body(
            config=config,
            run_id=run_id,
            attempt=attempt,
            error=error,
            workflow_checkpoint=None,
        )
        return
    error_text = "".join(traceback.format_exception(type(error), error, error.__traceback__)).strip()
    recovery_payload: dict[str, object] = {
        "attempt": int(attempt),
        "error_text": error_text[: 1024 * 1024],
        "submit_autofix": isinstance(error, SubmitAbortedError),
    }
    with _oracle_workflow_state.oracle_workflow_checkpoint(
        run_dir=config.paths.run_dir(run_id),
        workflow_id=f"autofix-attempt-{attempt}",
        workflow_kind="autofix",
        recovery_payload=recovery_payload,
    ) as checkpoint:
        _run_autofix_body(
            config=config,
            run_id=run_id,
            attempt=attempt,
            error=error,
            workflow_checkpoint=checkpoint,
        )


def _run_autofix_body(
    *,
    config: AutopilotConfig,
    run_id: str,
    attempt: int,
    error: Exception,
    workflow_checkpoint: _oracle_workflow_state.OracleWorkflowCheckpoint | None,
) -> None:
    prepared_context = _autofix_context.prepare_autofix_context(
        config=config,
        run_id=run_id,
        attempt=attempt,
        error=error,
        max_search_iteration=MAX_AUTOFIX_ATTEMPTS + int(MAX_KERNEL_FIX_ATTEMPTS or 0) + MAX_AUTOFIX_CODEX_PASSES,
        sha256_or_none=_sha256_or_none,
    )
    run_dir = prepared_context.run_dir
    autofix_dir = prepared_context.autofix_dir
    error_text = prepared_context.error_text
    submit_file_fix_required = prepared_context.submit_file_fix_required
    submit_file_fix_baseline_path = prepared_context.submit_file_fix_baseline_path
    submit_file_fix_baseline_sha256 = prepared_context.submit_file_fix_baseline_sha256

    allowed_prefixes = build_repair_write_policy(
        repo_root=config.paths.repo_root,
        data_dir=config.paths.data_dir,
        kernels_dir=config.paths.kernels_dir / run_id,
        module_file=Path(__file__),
    )
    prompt_plan = _autofix_context.build_autofix_prompt_plan(
        config=config,
        run_id=run_id,
        attempt=attempt,
        prepared_context=prepared_context,
        allowed_prefixes=allowed_prefixes,
        autopilot_path=Path(__file__).resolve(),
    )
    prompt_text = prompt_plan.prompt_text
    strategy_label = prompt_plan.strategy_label
    print(
        f"[cyan]{strategy_label}[/cyan]: strategy={_ERROR_STRATEGY_MODEL}({_ERROR_STRATEGY_REASONING_EFFORT}) "
        f"-> fixer={_ERROR_FIX_CODEX_MODEL}({_ERROR_FIX_REASONING_EFFORT})"
    )
    _watch_state.update_watch_phase(
        config,
        run_id,
        prompt_plan.thinking_phase,
        detail=f"GPT is analyzing the {strategy_label} failure and drafting a fix strategy.",
    )
    strategy_prompt = _autofix_context.build_autofix_strategy_prompt(
        config=config,
        run_id=run_id,
        attempt=attempt,
        prompt_plan=prompt_plan,
        hardware_constraints=render_hardware_constraints(
            resolve_hardware_profile(config.hardware_profile, compute=config.compute),
            compute=config.compute,
            time_budget_min=config.time_budget_min,
        ),
        error_text=error_text,
    )
    strategy_text = _agent_strategy.run_error_strategy_prompt(
        prompt_text=strategy_prompt,
        output_dir=autofix_dir / "gpt_strategy",
        dry_run=config.dry_run,
        stage_label=strategy_label,
        implementation_agent_alias=IMPLEMENTATION_AGENT.log_alias,
        strategy_model=_ERROR_STRATEGY_MODEL,
        reasoning_effort=_ERROR_STRATEGY_REASONING_EFFORT,
        run_strategy_func=_run_required_oracle_strategy,
    )
    if not config.dry_run and not strategy_text.strip():
        raise OracleStrategyError(f"Oracle {strategy_label} strategy is required before Codex implementation.")
    if workflow_checkpoint is not None:
        workflow_checkpoint.mark_oracle_complete(
            response_path=autofix_dir / "gpt_strategy" / "strategy_last_message.txt"
        )
    if strategy_text.strip():
        prompt_text += (
            f"\n\n## {STRATEGY_AGENT.display_name} Extra-High Error-Fix Strategy\n"
            "Use the strategy below as guidance, then apply minimal targeted edits.\n\n"
            f"{strategy_text}\n"
        )
    prompt_path = autofix_dir / "prompt.md"
    _agent_io.write_agent_prompt(prompt_path, prompt_text)
    _agent_io.print_agent_prompt(
        log_alias=IMPLEMENTATION_AGENT.log_alias,
        prompt_path=prompt_path,
        prompt_text=prompt_text,
    )

    retry_feedback = ""
    for codex_pass in range(1, MAX_AUTOFIX_CODEX_PASSES + 1):
        pass_prompt_text = (
            prompt_text
            if not retry_feedback
            else _agent_io.append_fix_retry_feedback(
                base_prompt=prompt_text,
                stage_label="autofix",
                codex_pass=codex_pass - 1,
                failure_text=retry_feedback,
            )
        )
        pass_prompt_path = prompt_path if codex_pass == 1 else autofix_dir / f"prompt-pass-{codex_pass:02d}.md"
        _agent_io.write_agent_prompt(pass_prompt_path, pass_prompt_text)
        if codex_pass > 1:
            print(
                "[yellow]autofix[/yellow]: "
                f"retrying {IMPLEMENTATION_AGENT.log_alias} pass "
                f"{codex_pass}/{MAX_AUTOFIX_CODEX_PASSES} with previous failure context."
            )

        before = _snapshot_tree(config.paths.repo_root)
        pass_output_dir = autofix_dir if codex_pass == 1 else autofix_dir / f"pass-{codex_pass:02d}"
        _watch_state.update_watch_phase(
            config,
            run_id,
            prompt_plan.fixing_phase,
            detail=f"{IMPLEMENTATION_AGENT.display_name} is applying the {strategy_label} fix.",
        )
        result = run_codex(
            pass_prompt_path,
            pass_output_dir,
            dry_run=config.dry_run,
            heartbeat_label="fixing error",
            model=ORACLE_IMPLEMENTATION_AGENT.model,
            reasoning_effort=ORACLE_IMPLEMENTATION_AGENT.reasoning_effort,
            reasoning_profile=ORACLE_IMPLEMENTATION_AGENT.reasoning_profile,
            cli_profile=ORACLE_IMPLEMENTATION_AGENT.cli_profile,
            cwd=config.paths.repo_root,
        )
        after = _snapshot_tree(config.paths.repo_root)
        changed = _diff_snapshots(before, after)
        # Autofix often needs to regenerate staged outputs under artifacts/*/kernels and
        # run-level state files; do not apply write-guard restrictions in this stage.
        if not config.dry_run and any(path.startswith("src/") for path in changed):
            if workflow_checkpoint is not None:
                workflow_checkpoint.mark_completed()
        _autofix_restart.maybe_restart_for_src_changes(
            dry_run=config.dry_run,
            run_dir=config.paths.run_dir(run_id),
            repo_root=config.paths.repo_root,
            run_id=run_id,
            slug=config.slug,
            changed=changed,
            stage=f"autofix_attempt_{attempt}",
            max_restarts=MAX_AUTOFIX_RESTARTS,
        )
        response_text = _agent_io.read_agent_response(result.last_message_path)
        _agent_io.print_agent_response(
            log_alias=IMPLEMENTATION_AGENT.log_alias,
            response_path=result.last_message_path,
            response_text=response_text,
        )
        _agent_io.log_codex_sandbox_fallback(stage_label=strategy_label, result=result)
        if result.returncode != 0:
            retry_feedback = (
                f"{IMPLEMENTATION_AGENT.display_name} autofix step failed with non-zero exit status.\n"
                f"returncode={result.returncode}\n"
                f"pass={codex_pass}/{MAX_AUTOFIX_CODEX_PASSES}\n"
                f"response={response_text}"
            )
            if codex_pass < MAX_AUTOFIX_CODEX_PASSES:
                continue
            raise RuntimeError(f"{IMPLEMENTATION_AGENT.display_name} autofix step failed.")

        if submit_file_fix_required and not _submit_failure_context.submit_file_fix_contract_satisfied_for_run(
            run_dir=run_dir,
            load_run_state=_autopilot_state.load_run_state,
            baseline_path=submit_file_fix_baseline_path,
            baseline_sha256=submit_file_fix_baseline_sha256,
            sha256_or_none=_sha256_or_none,
        ):
            retry_feedback = _submit_failure_context.format_submit_file_repair_contract_retry_feedback(
                baseline_path=submit_file_fix_baseline_path,
                baseline_sha256=submit_file_fix_baseline_sha256,
            )
            if codex_pass < MAX_AUTOFIX_CODEX_PASSES:
                continue
            raise RuntimeError("Submit autofix did not repair the submission artifact required by Kaggle.")

        try:
            _verify_artifacts.run_repo_verify(
                config.verify_cmd,
                dry_run=config.dry_run,
                artifacts_dir=config.paths.artifacts_dir,
                run_command_fn=run_command,
            )
        except Exception as exc:  # noqa: BLE001
            retry_feedback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
            if codex_pass < MAX_AUTOFIX_CODEX_PASSES:
                continue
            raise
        return

    raise RuntimeError(f"Autofix exhausted {IMPLEMENTATION_AGENT.log_alias} retry passes without resolving the error.")
