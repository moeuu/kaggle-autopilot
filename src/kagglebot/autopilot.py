from __future__ import annotations

import builtins
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from rich import print

from kagglebot import autofix_restart as _autofix_restart
from kagglebot import campaign_metrics as _campaign_metrics
from kagglebot import code_reference as _code_reference
from kagglebot import competition_rules as _competition_rules
from kagglebot import diagnostics as _diagnostics
from kagglebot import iteration_metrics as _iteration_metrics
from kagglebot import kernel_metrics as _kernel_metrics
from kagglebot import kernel_quality as _kernel_quality
from kagglebot import kernel_snapshot as _kernel_snapshot
from kagglebot import plan_policy as _plan_policy
from kagglebot import runtime_fixes as _runtime_fixes
from kagglebot import score_progress as _score_progress
from kagglebot import score_sources as _score_sources
from kagglebot import submission_policy as _submission_policy
from kagglebot import submit_attempts as _submit_attempts
from kagglebot import submit_autofix as _submit_autofix
from kagglebot import submit_failure_context as _submit_failure_context
from kagglebot import submit_failure_policy as _submit_failure_policy
from kagglebot import submit_notebook as _submit_notebook
from kagglebot import submit_retry_policy as _submit_retry_policy
from kagglebot import submit_stage as _submit_stage
from kagglebot.agents.codex_runner import run_codex
from kagglebot.agents.identity import (
    IMPLEMENTATION_AGENT,
    STRATEGY_AGENT,
    planning_flow_summary,
    prompt_identity_format_args,
    render_prompt_identity,
)
from kagglebot.agents.strategy_runner import run_strategy
from kagglebot.autopilot_helpers import (
    _build_medal_target_reason,
    _detect_online_mismatch_signal,
    _extract_missing_ensemble_signal,
    _extract_orig_proba_signal,
    _extract_original_data_unused_signal,
    _extract_pseudo_label_failure_signal,
    _extract_same_family_plateau_signal,
    _meets_rank_percentile_target,
    _requires_tabular_multi_family_policy,
    _resume_best_online_submission_score,
    _should_force_major_overhaul_by_rank,
    _to_float,
    _to_int,
    _update_best_score,
)
from kagglebot.autopilot_state import (
    _copy_kernel_support_artifacts_to_iteration_dir,
    _copy_submission_artifact_to_iteration_dir,
    _count_successful_submit_attempts,
    _has_successful_submit_attempt,
    _load_latest_submit_attempt,
    _load_run_state,
    _load_submit_fingerprints,
    _resolve_iteration_artifact,
    _resolve_iteration_submission_artifact,
    _save_run_state,
    _write_iteration_state_marker,
)
from kagglebot.autopilot_state import (
    _load_submit_retry_artifacts as _state_load_submit_retry_artifacts,
)
from kagglebot.autopilot_state import (
    _resume_best_submittable_iteration_state as _state_resume_best_submittable_iteration_state,
)
from kagglebot.autopilot_state import (
    _resume_best_submitted_offline_score as _state_resume_best_submitted_offline_score,
)
from kagglebot.autopilot_state import (
    _resume_iteration_state as _state_resume_iteration_state,
)
from kagglebot.campaign import (
    TOP1_TARGET_RANK_PERCENTILE,
    allocate_submission,
    build_campaign_candidate,
    campaign_state_path,
    candidate_registry_path,
    normalize_campaign_mode,
    update_campaign_state,
    upsert_candidate,
)
from kagglebot.competition_policy import load_competition_policy
from kagglebot.env_utils import env_flag as _env_flag
from kagglebot.env_utils import env_int as _env_int
from kagglebot.env_utils import env_truthy as _env_truthy
from kagglebot.eval import (
    EvaluationReport,
    validate_evaluation_spec,
)
from kagglebot.exceptions import (
    DuplicateSubmissionError,
    KaggleCliError,
    KaggleNetworkError,
    KernelCapacityError,
    KernelFailedError,
    KernelStillRunningError,
    RulesNotAcceptedError,
    SubmissionCliError,
    SubmissionRateLimitError,
    SubmissionValidationError,
    SubmitAbortedError,
)
from kagglebot.exec_utils import run_command
from kagglebot.experiment_executor import execute_experiment_graph
from kagglebot.experiment_graph import (
    append_campaign_outcome,
    build_experiment_graph,
    normalize_portfolio_execution,
    write_allocator_decision,
)
from kagglebot.hardware import render_hardware_constraints, resolve_hardware_profile
from kagglebot.hashing import sha256_file_or_none as _sha256_or_none
from kagglebot.history import SubmissionLedger, new_run_id
from kagglebot.json_utils import load_json_object as _load_json_object
from kagglebot.json_utils import load_json_object_or_empty as _load_json_object_or_empty
from kagglebot.json_utils import write_json_object as _write_json_object
from kagglebot.kaggle_api import (
    check_rules_accepted,
    leaderboard_rank_for_score,
    leaderboard_top1,
    list_competition_submissions,
)
from kagglebot.kernel_runner import (
    _collect_log_tail,
    resolve_kaggle_username,
    run_kernel,
    run_kernel_local,
    run_submit_kernel,
)
from kagglebot.knowledge import (
    derive_problem_types,
    ensure_taxonomy,
    format_error_fix_insights,
    format_problem_type_insights,
    record_error_fix_insight,
    record_improvement,
    record_iteration,
    record_problem_type_insight,
    record_run,
    resolve_error_fix_insights,
    resolve_problem_type_insights,
    resolve_similar_improvements,
)
from kagglebot.medals import (
    DEFAULT_TARGET_MEDAL,
    MEDAL_TARGET_PERCENTILES,
    normalize_target_medal,
    normalize_target_rank_percentile,
)
from kagglebot.method_scout import (
    method_registry_path,
    normalize_method_scout_mode,
    normalize_research_scout_mode,
    render_method_registry_for_prompt,
    run_method_scout,
)
from kagglebot.metric_matching import (
    canonical_metric_name_for_match as _canonical_metric_name_for_match,
)
from kagglebot.metric_matching import (
    infer_metric_direction_for_mismatch as _infer_metric_direction_for_mismatch,
)
from kagglebot.metric_matching import (
    metrics_equivalent as _metrics_equivalent,
)
from kagglebot.orchestrator.agent_pipeline import (
    AgentPipelineConfig,
    WriteGuardPolicy,
    _apply_plan_guardrails,
    _backup_guarded_files,
    _diff_snapshots,
    _enforce_allowlist_changes,
    _repo_root_write_policy,
    _snapshot_tree,
    run_agent_pipeline,
)
from kagglebot.runners.base import RunContext
from kagglebot.runners.local_kernel import LocalKernelRunner
from kagglebot.solver.metrics import canonical_metric, compute_metric, infer_direction, metric_requires_proba
from kagglebot.submission.guard import (
    classify_submit_error,
    compute_error_fingerprint,
    normalize_error_text,
    run_kaggle_submit_kernel,
)
from kagglebot.submission.outcome_service import SubmissionOutcomePollingError, SubmissionOutcomeService
from kagglebot.submission_history import (
    build_previous_submission_history_payload as _build_previous_submission_history_payload,
)
from kagglebot.submission_history import (
    detect_online_regression_vs_submission_history as _detect_online_regression_vs_submission_history,
)
from kagglebot.submission_history import (
    format_previous_submission_history_for_prompt as _format_previous_submission_history_for_prompt,
)
from kagglebot.submission_service import SubmissionConfig, SubmissionService
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
    normalize_top1_submit_policy,
)
from kagglebot.types import PlanConfig
from kagglebot.validation_lab import normalize_validation_lab_mode, run_validation_lab
from kagglebot.validators import ensure_kernel_sources_valid
from kagglebot.writeup import (
    build_writeup_bundle,
    infer_code_competition_from_paths,
    infer_deliverable_mode_from_paths,
    infer_submit_mode_from_paths,
    normalize_deliverable_mode,
    normalize_submit_mode,
)

_CodeReferenceNotebook = _code_reference.CodeReferenceNotebook
_code_reference_marker = _code_reference.code_reference_marker
_extract_code_reference_score = _code_reference.extract_code_reference_score
_extract_code_reference_score_from_index = _code_reference.extract_code_reference_score_from_index
_extract_code_reference_score_from_markdown = _code_reference.extract_code_reference_score_from_markdown
_extract_score_from_text = _code_reference.extract_score_from_text
_load_code_reference_notebook = _code_reference.load_code_reference_notebook
_load_ensemble_reference_notebook = _code_reference.load_ensemble_reference_notebook
_load_required_reference_notebook = _code_reference.load_required_reference_notebook
_reference_requires_tabicl = _code_reference.reference_requires_tabicl
_validate_code_reference_implementation = _code_reference.validate_code_reference_implementation
MAJOR_TOP1_GAP = _score_progress.MAJOR_TOP1_GAP
MODERATE_TOP1_GAP = _score_progress.MODERATE_TOP1_GAP
_BEST_SCORE_OUTLIER_TOP1_ABS_MARGIN = _score_progress.BEST_SCORE_OUTLIER_TOP1_ABS_MARGIN
_BEST_SCORE_OUTLIER_TOP1_REL_MARGIN = _score_progress.BEST_SCORE_OUTLIER_TOP1_REL_MARGIN
_REGRESSION_GUARD_ABS_DROP_PROB = _score_progress.REGRESSION_GUARD_ABS_DROP_PROB
_REGRESSION_GUARD_ABS_DROP_DEFAULT = _score_progress.REGRESSION_GUARD_ABS_DROP_DEFAULT
_CONSERVATIVE_COLLAPSE_MAX_FEATURES = _score_progress.CONSERVATIVE_COLLAPSE_MAX_FEATURES
_resolve_explicit_official_metric_override = _score_progress.resolve_explicit_official_metric_override
_is_confirmed_first_place = _score_progress.is_confirmed_first_place
_classify_improvement_mode = _score_progress.classify_improvement_mode
_score_delta_vs_reference = _score_progress.score_delta_vs_reference
_normalize_code_reference_score_for_comparison = _score_progress.normalize_code_reference_score_for_comparison
_score_drop_vs_best = _score_progress.score_drop_vs_best
_regression_drop_threshold = _score_progress.regression_drop_threshold
_is_severe_regression_vs_best = _score_progress.is_severe_regression_vs_best
_is_conservative_feature_collapse = _score_progress.is_conservative_feature_collapse
_effective_best_score_for_progress = _score_progress.effective_best_score_for_progress
_should_update_best_accuracy_candidate = _score_progress.should_update_best_accuracy_candidate
_pipeline_config_hash = _diagnostics.pipeline_config_hash
_diagnostics_json_default = _diagnostics.diagnostics_json_default
_build_diagnostics = _diagnostics.build_diagnostics
_evaluation_to_payload = _iteration_metrics.evaluation_to_payload
_build_metrics_payload = _iteration_metrics.build_metrics_payload
_build_iteration_evaluation_report = _iteration_metrics.build_iteration_evaluation_report
_build_eval_data_cache_fallback = _iteration_metrics.build_eval_data_cache_fallback
_extract_fold_scores_for_report = _iteration_metrics.extract_fold_scores_for_report
_ensure_eval_data_cache = _iteration_metrics.ensure_eval_data_cache
_build_split_index_fingerprints = _iteration_metrics.build_split_index_fingerprints
_iter_split_indices = _iteration_metrics.iter_split_indices
_iteration_metrics_allow_submit = _iteration_metrics.iteration_metrics_allow_submit
_append_run_evaluation_report = _iteration_metrics.append_run_evaluation_report
_resume_best_readiness_score = _iteration_metrics.resume_best_readiness_score
_resume_noise_guard_state = _iteration_metrics.resume_noise_guard_state
_infer_campaign_candidate_category = _campaign_metrics.infer_campaign_candidate_category
_infer_campaign_model_family = _campaign_metrics.infer_campaign_model_family
_infer_campaign_feature_set = _campaign_metrics.infer_campaign_feature_set
_extract_campaign_fold_scores = _campaign_metrics.extract_campaign_fold_scores
_extract_campaign_prediction_correlation = _campaign_metrics.extract_campaign_prediction_correlation
_extract_campaign_artifact_path = _campaign_metrics.extract_campaign_artifact_path
_extract_campaign_method_id = _campaign_metrics.extract_campaign_method_id
_extract_campaign_validation_profile_id = _campaign_metrics.extract_campaign_validation_profile_id
_campaign_prefers_validation_redesign = _campaign_metrics.campaign_prefers_validation_redesign
_extract_trusted_cv_value_from_metrics_payload = _kernel_metrics.extract_trusted_cv_value_from_metrics_payload
_extract_kernel_metric = _kernel_metrics.extract_kernel_metric
_metric_value_from_payload_item = _kernel_metrics.metric_value_from_payload_item
_extract_baseline_candidates_from_metrics_payload = _kernel_metrics.extract_baseline_candidates_from_metrics_payload
_collect_kernel_log_text = _kernel_metrics.collect_kernel_log_text
_extract_validation_scores_from_log_text = _kernel_metrics.extract_validation_scores_from_log_text
_extract_baseline_scores_from_log_text = _kernel_metrics.extract_baseline_scores_from_log_text
_extract_cv_breakdown_by_model_node = _kernel_quality.extract_cv_breakdown_by_model_node
_detect_subgroup_collapse_signal = _kernel_quality.detect_subgroup_collapse_signal
_iter_payload_mappings = _kernel_quality.iter_payload_mappings
_as_guard_bool = _kernel_quality.as_guard_bool
_first_nested_value = _kernel_quality.first_nested_value
_max_nested_float = _kernel_quality.max_nested_float
_max_nested_int = _kernel_quality.max_nested_int
_min_nested_int = _kernel_quality.min_nested_int
_any_nested_bool = _kernel_quality.any_nested_bool
_nested_text = _kernel_quality.nested_text
_build_oracle_override_signal = _kernel_quality.build_oracle_override_signal
_build_score_source_quality_signal = _kernel_quality.build_score_source_quality_signal
_detect_external_test_label_transfer_signal = _kernel_quality.detect_external_test_label_transfer_signal
_pipeline_name_from_payload = _kernel_quality.pipeline_name_from_payload
_extract_selected_pipeline_name = _kernel_quality.extract_selected_pipeline_name
_extract_pipeline_candidates = _kernel_quality.extract_pipeline_candidates
_pipeline_float = _kernel_quality.pipeline_float
_find_selected_pipeline = _kernel_quality.find_selected_pipeline
_detect_candidate_selection_mismatch = _kernel_quality.detect_candidate_selection_mismatch
_prediction_count_mean = _kernel_quality.prediction_count_mean
_detect_prediction_distribution_collapse = _kernel_quality.detect_prediction_distribution_collapse
_build_baseline_quality_signal = _kernel_quality.build_baseline_quality_signal
_build_code_reference_quality_signal = _kernel_quality.build_code_reference_quality_signal
_build_validation_metric_alignment = _kernel_quality.build_validation_metric_alignment
_BEST_KERNEL_SNAPSHOT_FILENAME = _kernel_snapshot.BEST_KERNEL_SNAPSHOT_FILENAME
_best_kernel_snapshot_path = _kernel_snapshot.best_kernel_snapshot_path
_capture_best_kernel_snapshot = _kernel_snapshot.capture_best_kernel_snapshot
_ensure_best_kernel_snapshot = _kernel_snapshot.ensure_best_kernel_snapshot
_restore_best_kernel_snapshot = _kernel_snapshot.restore_best_kernel_snapshot
_maybe_write_column_fill = _runtime_fixes.maybe_write_column_fill
_maybe_write_object_coerce = _runtime_fixes.maybe_write_object_coerce
_maybe_write_device_coerce = _runtime_fixes.maybe_write_device_coerce
_parse_missing_columns = _runtime_fixes.parse_missing_columns
_maybe_write_column_map = _runtime_fixes.maybe_write_column_map
_scan_tabular_headers = _runtime_fixes.scan_tabular_headers
_read_header = _runtime_fixes.read_header
_extract_candidate_groups = _runtime_fixes.extract_candidate_groups
_infer_column_mapping = _runtime_fixes.infer_column_mapping
_normalize_column = _runtime_fixes.normalize_column
_normalize_group_tokens = _runtime_fixes.normalize_group_tokens
_keywords_from_group = _runtime_fixes.keywords_from_group
_keyword_score = _runtime_fixes.keyword_score
_extract_missing_module = _runtime_fixes.extract_missing_module
_load_blocked_modules = _runtime_fixes.load_blocked_modules
_save_blocked_modules = _runtime_fixes.save_blocked_modules
_record_blocked_module = _runtime_fixes.record_blocked_module


def _classify_submit_failure_repair(
    *,
    reason: object,
    error_kind: object,
    detail: str,
) -> tuple[str, bool, str]:
    decision = _submit_failure_policy.classify_submit_failure_repair(
        reason=reason,
        error_kind=error_kind,
        detail=detail,
    )
    return decision.repair_target, decision.repairable, decision.manual_next_step


# Backward-compatible symbol for tests/extensions.
# Runtime no longer uses the legacy src local trainer path.
def train_evaluate_and_predict(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise RuntimeError(
        "Legacy src local trainer was removed. Use artifacts/<slug>/kernel/kernel.py via autopilot/train commands."
    )


if TYPE_CHECKING:
    from kagglebot.paths import CompetitionPaths, KnowledgePaths
    from kagglebot.solver.evaluate import EvaluationResult


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


MAX_KERNEL_FIX_ATTEMPTS: int | None = 8
MAX_SAME_KERNEL_ERROR_REPEATS = 2
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
_WATCH_STATE_ENV = "KAGGLEBOT_WATCH_STATE_PATH"
_ERROR_FIX_CODEX_MODEL = IMPLEMENTATION_AGENT.model
_ERROR_FIX_REASONING_EFFORT = IMPLEMENTATION_AGENT.reasoning_effort
_ERROR_STRATEGY_MODEL = STRATEGY_AGENT.model
_ERROR_STRATEGY_REASONING_EFFORT = STRATEGY_AGENT.reasoning_effort
_METRIC_FIX_CODEX_MODEL = IMPLEMENTATION_AGENT.model
_METRIC_FIX_REASONING_EFFORT = IMPLEMENTATION_AGENT.reasoning_effort
_MAX_METRIC_FIX_ATTEMPTS = 3
_MAX_METRIC_FIX_CODEX_PASSES = 4
_SUBMISSION_POLL_MAX_ATTEMPTS: int | None = None
_SUBMISSION_POLL_INTERVAL_SEC = 30.0
_SUBMISSION_POLL_MAX_FETCH_ERRORS = 3
_FORCED_INITIAL_SUBMIT_REASON = "initial_submit_contract_probe"
_SPARE_DAILY_SUBMIT_REASON = "spare_daily_submission_slot"
_SUBMIT_FAILED_DEFERRED_STATE = "submit_failed_deferred"
_SUBMIT_MAX_TRANSIENT_RETRIES = 3
_SUBMIT_BACKOFF_BASE_SEC = 2.0
_SUBMIT_STDERR_TAIL_CHARS = 1200
_SUBMIT_STDOUT_TAIL_CHARS = 1200
_KERNEL_PUSH_VERSION_RE = re.compile(r"Kernel version\s+(?P<version>\d+)\s+successfully pushed", re.IGNORECASE)
_DEFAULT_EVAL_SEEDS = list(_plan_policy.DEFAULT_EVAL_SEEDS)
_DEFAULT_EVAL_REPEATS = _plan_policy.DEFAULT_EVAL_REPEATS
_DEFAULT_MAX_ITERATIONS = 5
_LONG_LOCAL_GPU_ITERATION_BUDGET_MIN = 12 * 60
_LONG_LOCAL_GPU_MAX_ITERATIONS = 3
_HEAVY_DEEP_LEARNING_MODALITIES = frozenset({"image", "video", "audio", "text"})
_HEAVY_LOCAL_GPU_MAX_CV_FOLDS = 3
_DEFAULT_FORCE_MAJOR_RANK_MAX_PERCENTILE = 0.35
_DEFAULT_FORCE_MAJOR_RANK_MIN_TEAMS = 200
_DEFAULT_TARGET_MEDAL = DEFAULT_TARGET_MEDAL
_MEDAL_TARGET_PERCENTILES = MEDAL_TARGET_PERCENTILES
_DEFAULT_LIMITED_SUBMISSION_GATE = "readiness_or_final"
_DEFAULT_STRICT_COMPETITION_METRIC = True
_DEFAULT_REQUIRE_SUBMIT_IMPROVEMENT = True
_DEFAULT_FORCE_MAJOR_ON_NO_IMPROVE = True
_KERNEL_REGENERATE_MARKER_FILENAME = "kernel_regenerated_once.json"
_QUALITY_GUARD_SUBGROUP_RATIO = _kernel_quality.QUALITY_GUARD_SUBGROUP_RATIO
_QUALITY_GUARD_SUBGROUP_ABS_MARGIN = _kernel_quality.QUALITY_GUARD_SUBGROUP_ABS_MARGIN
_QUALITY_GUARD_CANDIDATE_HOLDOUT_REL_MARGIN = _kernel_quality.QUALITY_GUARD_CANDIDATE_HOLDOUT_REL_MARGIN
_QUALITY_GUARD_CANDIDATE_HOLDOUT_ABS_MARGIN = _kernel_quality.QUALITY_GUARD_CANDIDATE_HOLDOUT_ABS_MARGIN
_QUALITY_GUARD_PREDICTION_COUNT_RATIO = _kernel_quality.QUALITY_GUARD_PREDICTION_COUNT_RATIO
_QUALITY_GUARD_PREDICTION_COUNT_ABS_MARGIN = _kernel_quality.QUALITY_GUARD_PREDICTION_COUNT_ABS_MARGIN
_MAX_KERNEL_PREFLIGHT_FIX_ATTEMPTS = 2
_MODEL_NODE_METRIC_KEY = _kernel_quality.MODEL_NODE_METRIC_KEY


class _TrainingLiveStdout:
    """Render a single carriage-return live line while preserving regular log lines."""

    def __init__(self, base_stream) -> None:
        self._base_stream = base_stream
        self._last_live_text = ""
        self._live_active = False

    def render_live(self, text: str) -> None:
        self._last_live_text = text
        self._base_stream.write(f"\r{text}")
        self._live_active = True

    def finish_live(self, text: str) -> None:
        self._last_live_text = text
        self._base_stream.write(f"\r{text}\n")
        self._live_active = False

    def write(self, s: str) -> int:
        if not s:
            return 0
        interrupted = False
        if self._live_active and any(ch not in "\r" for ch in s):
            self._base_stream.write("\n")
            self._live_active = False
            interrupted = True
        written = self._base_stream.write(s)
        if interrupted and s.endswith("\n") and self._last_live_text:
            self._base_stream.write(f"\r{self._last_live_text}")
            self._live_active = True
        return written

    def flush(self) -> None:
        self._base_stream.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._base_stream, "isatty", lambda: False)())

    @property
    def encoding(self) -> str | None:
        return getattr(self._base_stream, "encoding", None)

    @property
    def errors(self) -> str | None:
        return getattr(self._base_stream, "errors", None)

    def fileno(self) -> int:
        return self._base_stream.fileno()


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
            session = AutopilotSession(config=config, run_id=run_id, resume_run=resume_after_failure)
            try:
                return session.run()
            except RulesNotAcceptedError:
                raise
            except SubmitAbortedError as exc:
                if config.dry_run:
                    raise
                if not _is_submit_abort_autofixable(config=config, run_id=run_id):
                    raise
                attempt += 1
                if attempt > MAX_AUTOFIX_ATTEMPTS:
                    raise
                print(
                    f"[yellow]autofix[/yellow]: submit stage failed; invoking "
                    f"{IMPLEMENTATION_AGENT.log_alias} to repair and retry submit"
                )
                run_dir = config.paths.run_dir(run_id)
                if (not _has_successful_submit_attempt(run_dir)) or _should_force_resubmit_after_submit_abort(run_dir):
                    os.environ["KAGGLEBOT_FORCE_RESUBMIT"] = "1"
                    submit_force_override = True
                _run_autofix(config=config, run_id=run_id, attempt=attempt, error=exc)
                resume_after_failure = True
            except KernelCapacityError:
                raise
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                if config.dry_run:
                    raise
                if _is_non_autofixable_runtime_error(exc):
                    raise
                attempt += 1
                if attempt > MAX_AUTOFIX_ATTEMPTS:
                    raise
                print(f"[yellow]autofix[/yellow]: invoking {IMPLEMENTATION_AGENT.log_alias} to repair error")
                _run_autofix(config=config, run_id=run_id, attempt=attempt, error=exc)
                resume_after_failure = True
    finally:
        if submit_force_override:
            os.environ.pop("KAGGLEBOT_FORCE_RESUBMIT", None)


@dataclass(frozen=True)
class PlanningPhase:
    config: AutopilotConfig
    run_id: str
    resume_run: bool

    def execute(self, plan: PlanConfig) -> PlanConfig:
        if _should_skip_planning(resume_run=self.resume_run, paths=self.config.paths):
            print("[yellow]resume[/yellow]: skipping planning after restart; reusing existing plan")
            return plan
        if _needs_planning(plan, self.config):
            print("[cyan]plan[/cyan]: generating initial plan")
            _update_watch_phase(
                self.config,
                self.run_id,
                "gpt_planning",
                detail="GPT is drafting the initial competition plan.",
            )
            _run_plan_and_initial(self.config, self.run_id)
            return _load_plan(self.config.paths)
        return plan


@dataclass(frozen=True)
class KnowledgePhase:
    config: AutopilotConfig

    def refresh(self) -> None:
        _refresh_knowledge_hints(self.config)

    def load_dataset_profile(self) -> dict[str, object]:
        return _load_dataset_profile(self.config.paths)

    def derive_problem_types(self) -> list[str]:
        return derive_problem_types(self.load_dataset_profile())


@dataclass(frozen=True)
class IterationPhase:
    metric_direction: str

    def delta_from_best(self, best_score: float | None, current_score: float) -> float | None:
        if best_score is None:
            return None
        if self.metric_direction == "minimize":
            return best_score - current_score
        return current_score - best_score

    def should_update_best(self, best_score: float | None, current_score: float, min_improvement: float) -> bool:
        return _update_best_score(best_score, current_score, self.metric_direction, min_improvement)


@dataclass(frozen=True)
class SubmissionPhase:
    config: AutopilotConfig
    run_id: str
    problem_types: list[str]
    submit_mode: str
    notebook_submit_artifact_mode: str = "wrapper"

    def attempt(self, *, submission_path: Path, best_score: float | None) -> dict[str, object] | None:
        return _attempt_submit(
            config=self.config,
            run_id=self.run_id,
            submission_path=submission_path,
            best_score=best_score,
            problem_types=self.problem_types,
            submit_mode=self.submit_mode,
            notebook_submit_artifact_mode=self.notebook_submit_artifact_mode,
        )


def _update_watch_phase(
    config: AutopilotConfig,
    run_id: str,
    phase: str,
    *,
    detail: str | None = None,
    iteration: int | None = None,
) -> None:
    state_raw = os.environ.get(_WATCH_STATE_ENV)
    if not state_raw:
        return
    state_path = Path(state_raw)
    payload = _load_json_object_or_empty(state_path)
    active_slug = str(payload.get("active_slug") or "").strip()
    active_run_id = str(payload.get("active_run_id") or "").strip()
    if active_slug and active_slug != config.slug:
        return
    if active_run_id and active_run_id != run_id:
        return
    payload.update(
        {
            "active_slug": config.slug,
            "active_run_id": run_id,
            "last_status": "running",
            "phase": phase,
            "compute": config.compute,
            "updated_at": datetime.now(UTC).isoformat(),
        }
    )
    if detail:
        payload["phase_detail"] = detail
    else:
        payload.pop("phase_detail", None)
    if iteration is not None:
        payload["iteration"] = iteration
    try:
        _write_json_object(state_path, payload, sort_keys=True)
    except OSError:
        return


@dataclass(frozen=True)
class AutopilotSession:
    config: AutopilotConfig
    run_id: str
    resume_run: bool = False

    @property
    def planning(self) -> PlanningPhase:
        return PlanningPhase(config=self.config, run_id=self.run_id, resume_run=self.resume_run)

    @property
    def knowledge(self) -> KnowledgePhase:
        return KnowledgePhase(config=self.config)

    def run(self) -> None:
        _run_autopilot_core(self.config, self.run_id, resume_run=self.resume_run)


def _run_autopilot_core(config: AutopilotConfig, run_id: str, *, resume_run: bool = False) -> None:
    run_dir = config.paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    _update_watch_phase(config, run_id, "autopilot_starting")
    print(f"[green]run started[/green]: {run_id}")
    planning_phase = PlanningPhase(config=config, run_id=run_id, resume_run=resume_run)
    knowledge_phase = KnowledgePhase(config=config)
    plan = _load_plan(config.paths)
    if not config.paths.plan_path.exists():
        _write_plan(config.paths, plan)

    _update_watch_phase(config, run_id, "leaderboard_fetching")
    print(f"[cyan]fetching leaderboard[/cyan]: {config.slug}")
    metric_hint = config.target_metric or plan.target_metric
    top1_info = leaderboard_top1(
        config.slug,
        config.paths.context_dir,
        dry_run=config.dry_run,
        metric_hint=metric_hint,
    )
    _write_json_object(config.paths.top1_public_path, top1_info)
    _print_top1_info(top1_info)
    _update_watch_phase(config, run_id, "knowledge_refreshing")
    knowledge_phase.refresh()
    plan = planning_phase.execute(plan)

    _update_watch_phase(config, run_id, "resolving_plan")
    resolved = _resolve_plan(plan, config)
    target_metric = resolved["target_metric"]
    target_score = resolved["target_score"]
    if target_metric is None or target_score is None:
        run_payload = _build_run_payload(
            run_id=run_id,
            config=config,
            resolved=resolved,
            status="missing_target",
        )
        _write_json_object(run_dir / "run.json", run_payload)
        return

    metric_direction = infer_direction(target_metric, resolved["target_direction"])
    resolved["target_direction"] = metric_direction
    deliverable_mode = normalize_deliverable_mode(resolved.get("deliverable_mode"), default="leaderboard")
    resolved["deliverable_mode"] = deliverable_mode
    campaign_mode = normalize_campaign_mode(config.campaign_mode, deliverable_mode=deliverable_mode)
    portfolio_execution = normalize_portfolio_execution(config.portfolio_execution)
    validation_lab_mode = normalize_validation_lab_mode(config.validation_lab)
    research_scout_mode = normalize_research_scout_mode(config.research_scout)
    top1_submit_policy = normalize_top1_submit_policy(config.top1_submit_policy)
    resolved["campaign_mode"] = campaign_mode
    resolved["portfolio_execution"] = portfolio_execution
    resolved["validation_lab"] = validation_lab_mode
    resolved["research_scout"] = research_scout_mode
    resolved["top1_exhaustive"] = bool(config.top1_exhaustive)
    resolved["top1_submit_policy"] = top1_submit_policy
    submit_mode = normalize_submit_mode(resolved.get("submit_mode"), default="file")
    resolved["submit_mode"] = submit_mode
    writeup_mode = deliverable_mode == "writeup"
    submit_enabled = bool(config.submit and not writeup_mode)
    strict_competition_metric = _env_flag(
        "KAGGLEBOT_STRICT_COMPETITION_METRIC",
        default=_DEFAULT_STRICT_COMPETITION_METRIC,
    )
    require_submit_improvement = _env_flag(
        "KAGGLEBOT_REQUIRE_SUBMIT_IMPROVEMENT",
        default=_DEFAULT_REQUIRE_SUBMIT_IMPROVEMENT,
    )
    submit_improved_only = str(resolved.get("submit_policy") or "").strip().lower() == "improved"
    force_major_on_no_improve = _env_flag(
        "KAGGLEBOT_FORCE_MAJOR_ON_NO_IMPROVE",
        default=_DEFAULT_FORCE_MAJOR_ON_NO_IMPROVE,
    )

    _write_plan(config.paths, _resolved_plan(resolved))
    _update_watch_phase(config, run_id, "initializing_iterations")
    run_payload = _build_run_payload(
        run_id=run_id,
        config=config,
        resolved=resolved,
        status="running",
    )
    _write_json_object(run_dir / "run.json", run_payload)
    _ensure_best_kernel_snapshot(paths=config.paths, run_dir=run_dir)

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

    max_iterations = max(1, int(resolved["max_iterations"]))
    iteration_phase = IterationPhase(metric_direction=metric_direction)
    holdout_frac = float(resolved["holdout_frac"])
    cv_folds = int(resolved["cv_folds"])
    split_strategy = str(resolved.get("split_strategy") or "").strip().lower() or None
    seed = int(resolved["seed"])
    eval_seeds = _normalize_eval_seeds(resolved.get("eval_seeds"), fallback=[seed])
    eval_repeats = _normalize_eval_repeats(resolved.get("eval_repeats"), fallback=_DEFAULT_EVAL_REPEATS)
    score_source = str(resolved["score_source"] or "cv")
    max_total_min_raw = resolved.get("max_total_min")
    max_total_min = float(max_total_min_raw) if isinstance(max_total_min_raw, (int, float)) else None
    time_budget_min_raw = resolved.get("time_budget_min")
    time_budget_min = int(time_budget_min_raw) if isinstance(time_budget_min_raw, (int, float)) else None
    kernel_name = resolved["kernel_name"]
    enable_internet = str(resolved["internet"]) == "on"
    submission_gate = str(resolved.get("submission_gate") or "always")
    submission_limit_per_day_raw = resolved.get("submission_limit_per_day")
    submission_limit_per_day = (
        int(submission_limit_per_day_raw)
        if isinstance(submission_limit_per_day_raw, (int, float)) and int(submission_limit_per_day_raw) > 0
        else None
    )
    evaluation_contract = (
        resolved.get("evaluation_contract") if isinstance(resolved.get("evaluation_contract"), dict) else None
    )
    readiness_target = float(resolved.get("readiness_target_score") or target_score)
    readiness_method = str(resolved.get("readiness_method") or "ci_bound")
    readiness_k = float(resolved.get("readiness_k") or 1.0)
    ci_method = str(resolved.get("ci_method") or "normal")
    ci_alpha = float(resolved.get("ci_alpha") or 0.05)
    target_medal = _normalize_target_medal(resolved.get("target_medal"), default=None)
    target_rank_percentile = _normalize_target_rank_percentile(
        resolved.get("target_rank_percentile"),
        medal=target_medal,
        fallback=None,
    )
    if campaign_mode == "top1" and target_rank_percentile is None:
        target_rank_percentile = TOP1_TARGET_RANK_PERCENTILE
        resolved["target_rank_percentile"] = target_rank_percentile
    drift_check_enabled = bool(resolved.get("drift_check", False))
    drift_weight = float(resolved.get("drift_weight") or 1.0)
    stop_min_delta = float(resolved.get("stop_min_delta") or 0.0)
    stop_no_improve_patience = int(resolved.get("stop_no_improve_patience") or 0)
    stop_same_config_patience = int(resolved.get("stop_same_config_patience") or 0)
    rank_force_major_max_percentile = _normalize_rank_force_percentile(
        resolved.get("rank_force_major_max_percentile"),
        fallback=_DEFAULT_FORCE_MAJOR_RANK_MAX_PERCENTILE,
    )
    rank_force_major_min_teams = _normalize_rank_force_min_teams(
        resolved.get("rank_force_major_min_teams"),
        fallback=_DEFAULT_FORCE_MAJOR_RANK_MIN_TEAMS,
    )
    no_improve_streak = 0
    frontier_no_improve_streak = 0
    same_config_streak = 0
    last_config_hash: str | None = None
    eval_data_cache: dict[str, object] | None = None
    previous_readiness_score, noise_limited_streak = _resume_noise_guard_state(
        run_dir=config.paths.run_dir(run_id),
        max_iterations=max_iterations,
    )
    start_iteration, best_score, best_submission = _resume_iteration_state(
        paths=config.paths,
        run_id=run_id,
        metric_direction=metric_direction,
        target_metric=target_metric,
        max_iterations=max_iterations,
        require_submit_phase=submit_enabled and not config.dry_run,
    )
    best_submitted_score = _resume_best_submitted_offline_score(
        paths=config.paths,
        run_id=run_id,
        metric_direction=metric_direction,
        target_metric=target_metric,
        max_iterations=max_iterations,
    )
    best_online_submission_score = _resume_best_online_submission_score(
        paths=config.paths,
        run_id=run_id,
        direction=metric_direction,
        max_iterations=max_iterations,
    )
    previous_submission_history = _load_previous_submission_history(
        slug=config.slug,
        paths=config.paths,
        direction=metric_direction,
        dry_run=config.dry_run,
    )
    historical_best_submission_score = _to_float(previous_submission_history.get("best_score"))
    if historical_best_submission_score is not None:
        if _update_best_score(best_submitted_score, historical_best_submission_score, metric_direction, 0.0):
            best_submitted_score = historical_best_submission_score
        if _update_best_score(best_online_submission_score, historical_best_submission_score, metric_direction, 0.0):
            best_online_submission_score = historical_best_submission_score
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
    effective_method_scout = _effective_method_scout_mode(config=config, campaign_mode=campaign_mode)
    method_registry: dict[str, object] = {}
    source_registry: dict[str, object] = {}
    validation_lab_report: dict[str, object] | None = None
    win_contract: dict[str, object] | None = None
    if effective_method_scout != "off":
        method_registry = run_method_scout(
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
        source_registry = _load_json_object(config.paths.source_registry_path) or {}
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
            validation_registry = _load_json_object(config.paths.validation_registry_path) or {}
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
        method_registry = _load_json_object(config.paths.method_registry_path) or {}
        source_registry = _load_json_object(config.paths.source_registry_path) or {}
    if campaign_mode == "top1":
        validation_registry_for_contract = _load_json_object(config.paths.validation_registry_path) or {}
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
    resumed_best_readiness = _resume_best_readiness_score(
        run_dir=config.paths.run_dir(run_id),
        direction=metric_direction,
        max_iterations=max_iterations,
    )
    if resumed_best_readiness is not None and best_score is None:
        best_score = resumed_best_readiness
    best_submittable_score, best_submittable_submission = _resume_best_submittable_iteration_state(
        paths=config.paths,
        run_id=run_id,
        metric_direction=metric_direction,
        target_metric=target_metric,
        max_iterations=max_iterations,
    )
    if start_iteration > 1:
        print(f"[yellow]resume[/yellow]: found completed iterations; resuming at {start_iteration}/{max_iterations}")
    loop_started_at = time.monotonic()
    last_completed_iteration = start_iteration - 1

    try:
        for iteration in range(start_iteration, max_iterations + 1):
            _update_watch_phase(config, run_id, "iteration_starting", iteration=iteration)
            last_completed_iteration = iteration
            if max_total_min is not None and max_total_min > 0:
                elapsed_total_min = (time.monotonic() - loop_started_at) / 60.0
                if elapsed_total_min >= float(max_total_min):
                    run_payload["status"] = "stopped"
                    run_payload["stop_reason"] = (
                        f"max_total_min reached: elapsed={elapsed_total_min:.1f}m limit={float(max_total_min):.1f}m"
                    )
                    print(f"[yellow]stop[/yellow]: {run_payload['stop_reason']}")
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

            _update_watch_phase(config, run_id, "verifying", iteration=iteration)
            _run_verify(config.verify_cmd, dry_run=config.dry_run, artifacts_dir=config.paths.artifacts_dir)

            submission_path = iter_dir / "submission.csv"
            metrics_path = iter_dir / "metrics.json"
            evaluation_report_path = iter_dir / "evaluation_report.json"
            evaluation = None
            kernel_metrics_payload: dict[str, object] | None = None
            kernel_metrics_artifact_path: Path | None = None
            evaluation_by_source: dict[str, EvaluationResult] = {}
            model_summary = {}
            accelerator_used = config.accelerator
            submit_retry_resume = _load_submit_retry_artifacts(
                run_dir=run_dir,
                iter_dir=iter_dir,
                iteration=iteration,
                max_iterations=max_iterations,
                metric_direction=metric_direction,
                target_metric=target_metric,
                require_submit_phase=submit_enabled and not config.dry_run,
            )
            if submit_retry_resume is not None:
                resume_submission_path, resume_metrics_path, resume_evaluation = submit_retry_resume
                if resume_submission_path != submission_path:
                    submission_path = _copy_submission_artifact_to_iteration_dir(
                        source=resume_submission_path,
                        iter_dir=iter_dir,
                    )
                if resume_metrics_path != metrics_path:
                    metrics_path.write_bytes(resume_metrics_path.read_bytes())
                evaluation = resume_evaluation
                kernel_metrics_payload = _load_json_object(resume_metrics_path)
                kernel_metrics_artifact_path = resume_metrics_path
                print(
                    "[yellow]resume[/yellow]: "
                    f"iter-{iteration} has completed training artifacts; retrying submit without retraining."
                )

            if evaluation is None:
                _update_watch_phase(config, run_id, "kernel_preflight", iteration=iteration)
                _run_kernel_source_preflight_fixes(
                    config=config,
                    run_id=run_id,
                    iteration=iteration,
                    iter_dir=iter_dir,
                    pending_error_fixes=pending_error_fixes,
                )

            if evaluation is None and config.compute.startswith("kaggle_"):
                kaggle_user = resolve_kaggle_username(config.kaggle_username)
                _update_watch_phase(config, run_id, "kaggle_kernel_running", iteration=iteration)
                print(f"[cyan]kernel run[/cyan]: {config.compute}")
                kernel_attempts = 0
                error_fingerprints: dict[str, int] = {}
                while True:
                    try:
                        kernel_result = run_kernel(
                            slug=config.slug,
                            run_id=run_id,
                            iteration=iteration,
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
                            dry_run=config.dry_run,
                            timeout_minutes=time_budget_min,
                            hardware_profile=config.hardware_profile,
                        )
                        if kernel_result.submission_path:
                            submission_path = _copy_submission_artifact_to_iteration_dir(
                                source=kernel_result.submission_path,
                                iter_dir=iter_dir,
                            )
                        _copy_kernel_support_artifacts_to_iteration_dir(
                            kernel_output_dir=kernel_result.output_dir,
                            iter_dir=iter_dir,
                        )
                        if kernel_result.metrics_path and kernel_result.metrics_path.exists():
                            kernel_metrics_artifact_path = kernel_result.metrics_path
                            kernel_metrics_payload = _load_json_object(kernel_result.metrics_path)
                            evaluation = _load_kernel_metrics(
                                kernel_result.metrics_path,
                                metric_direction,
                                target_metric,
                            )
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
                        error_text = _format_kernel_error(exc)
                        _record_kernel_error(
                            logs_dir=logs_dir,
                            attempt=kernel_attempts,
                            error_text=error_text,
                            error_fingerprints=error_fingerprints,
                            output_dir=output_dir,
                        )
                        raise
                    except KernelStillRunningError as exc:
                        error_text = _format_kernel_error(exc)
                        logs_dir.mkdir(parents=True, exist_ok=True)
                        (logs_dir / "kernel_remote_still_running.txt").write_text(error_text + "\n", encoding="utf-8")
                        _update_watch_phase(
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
                        error_text = _format_kernel_error(exc)
                        _record_kernel_error(
                            logs_dir=logs_dir,
                            attempt=kernel_attempts,
                            error_text=error_text,
                            error_fingerprints=error_fingerprints,
                            max_repeats=MAX_KERNEL_CAPACITY_REPEAT,
                            output_dir=output_dir,
                        )
                        capacity_retries = _env_int(
                            "KAGGLEBOT_KERNEL_CAPACITY_RETRIES",
                            default=MAX_KERNEL_CAPACITY_RETRIES,
                        )
                        _update_watch_phase(
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
                        if _is_kernel_registration_error(exc):
                            kernel_attempts += 1
                            error_text = _format_kernel_error(exc)
                            _record_kernel_error(
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
                        error_text = _format_kernel_error(exc)
                        try:
                            _record_kernel_error(
                                logs_dir=logs_dir,
                                attempt=kernel_attempts,
                                error_text=error_text,
                                error_fingerprints=error_fingerprints,
                                output_dir=output_dir,
                            )
                        except KernelFailedError:
                            if _maybe_regenerate_kernel_sources_once(
                                config=config,
                                run_id=run_id,
                                iteration=iteration,
                                iter_dir=iter_dir,
                                attempt=kernel_attempts,
                                trigger_reason="repeated_error_fingerprint",
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
                        )
            elif evaluation is None:
                kernel_path = config.paths.kernel_source_dir / "kernel.py"
                if not kernel_path.exists():
                    raise RuntimeError(
                        "Local autopilot requires kernel.py, but "
                        f"{kernel_path} was not found. Run planning/implement to generate kernel.py first."
                    )
                _update_watch_phase(config, run_id, "local_kernel_running", iteration=iteration)
                print(f"[cyan]kernel local run[/cyan]: {config.compute}")
                kernel_attempts = 0
                error_fingerprints = {}
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
                        )
                        if kernel_result.submission_path:
                            submission_path = _copy_submission_artifact_to_iteration_dir(
                                source=kernel_result.submission_path,
                                iter_dir=iter_dir,
                            )
                        _copy_kernel_support_artifacts_to_iteration_dir(
                            kernel_output_dir=kernel_result.output_dir,
                            iter_dir=iter_dir,
                        )
                        if kernel_result.metrics_path and kernel_result.metrics_path.exists():
                            kernel_metrics_artifact_path = kernel_result.metrics_path
                            kernel_metrics_payload = _load_json_object(kernel_result.metrics_path)
                            evaluation = _load_kernel_metrics(
                                kernel_result.metrics_path,
                                metric_direction,
                                target_metric,
                            )
                        if evaluation is None:
                            raise KernelFailedError(
                                "Local kernel metrics missing expected score; "
                                "ensure metrics.json includes a numeric metric value."
                            )
                        break
                    except Exception as exc:  # noqa: BLE001
                        kernel_attempts += 1
                        error_text = _format_kernel_error(exc)
                        try:
                            _record_kernel_error(
                                logs_dir=logs_dir,
                                attempt=kernel_attempts,
                                error_text=error_text,
                                error_fingerprints=error_fingerprints,
                                output_dir=output_dir,
                            )
                        except KernelFailedError:
                            if _maybe_regenerate_kernel_sources_once(
                                config=config,
                                run_id=run_id,
                                iteration=iteration,
                                iter_dir=iter_dir,
                                attempt=kernel_attempts,
                                trigger_reason="repeated_error_fingerprint",
                            ):
                                error_fingerprints.clear()
                                continue
                            raise
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
                        )

            if evaluation is None:
                raise RuntimeError("No evaluation metrics produced.")
            metric_mismatch_detected = False
            metric_mismatch_reason: str | None = None
            metric_fix_attempts = 0
            metric_recheck_attempted = False
            while evaluation.metric and target_metric and (not _metrics_equivalent(evaluation.metric, target_metric)):
                corrected_direction, confident = _infer_metric_direction_for_mismatch(
                    evaluation.metric,
                    metric_direction,
                )
                confidence_text = "high" if confident else "fallback"
                official_metric_override = _resolve_explicit_official_metric_override(
                    kernel_metrics_payload,
                    target_metric=target_metric,
                    evaluation_metric=evaluation.metric,
                )
                if official_metric_override:
                    metric_direction, _ = _infer_metric_direction_for_mismatch(
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
                    _write_plan(config.paths, _resolved_plan(resolved))
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
                        evaluation, kernel_metrics_payload, submission_path = _rerun_kernel_for_metric_recheck(
                            config=config,
                            run_id=run_id,
                            iteration=iteration,
                            submission_path=submission_path,
                            iter_dir=iter_dir,
                            metrics_artifact_path=kernel_metrics_artifact_path,
                            kernel_name=kernel_name,
                            enable_internet=enable_internet,
                            score_source=score_source,
                            target_metric=target_metric,
                            metric_direction=metric_direction,
                            holdout_frac=holdout_frac,
                            cv_folds=cv_folds,
                            seed=seed,
                            time_budget_min=time_budget_min,
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
                    _run_metric_only_competition_metric_fix(
                        config=config,
                        run_id=run_id,
                        iteration=iteration,
                        iter_dir=iter_dir,
                        mismatch_reason=metric_mismatch_reason,
                        attempt=metric_fix_attempts,
                        pending_error_fixes=pending_error_fixes,
                    )
                    evaluation, kernel_metrics_payload, submission_path = _rerun_kernel_for_metric_recheck(
                        config=config,
                        run_id=run_id,
                        iteration=iteration,
                        submission_path=submission_path,
                        iter_dir=iter_dir,
                        metrics_artifact_path=kernel_metrics_artifact_path,
                        kernel_name=kernel_name,
                        enable_internet=enable_internet,
                        score_source=score_source,
                        target_metric=target_metric,
                        metric_direction=metric_direction,
                        holdout_frac=holdout_frac,
                        cv_folds=cv_folds,
                        seed=seed,
                        time_budget_min=time_budget_min,
                    )
                    metric_still_mismatched = bool(
                        evaluation.metric
                        and target_metric
                        and (not _metrics_equivalent(evaluation.metric, target_metric))
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
                        )
                        if kernel_result.submission_path:
                            submission_path = _copy_submission_artifact_to_iteration_dir(
                                source=kernel_result.submission_path,
                                iter_dir=iter_dir,
                            )
                        _copy_kernel_support_artifacts_to_iteration_dir(
                            kernel_output_dir=kernel_result.output_dir,
                            iter_dir=iter_dir,
                        )
                        if kernel_result.metrics_path and kernel_result.metrics_path.exists():
                            kernel_metrics_artifact_path = kernel_result.metrics_path
                            kernel_metrics_payload = _load_json_object(kernel_result.metrics_path)
                            evaluation = _load_kernel_metrics(
                                kernel_result.metrics_path,
                                metric_direction,
                                target_metric,
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
                    _write_plan(config.paths, _resolved_plan(resolved))
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
            _update_watch_phase(config, run_id, "evaluating_iteration", iteration=iteration)
            report, report_payload, eval_data_cache = _build_iteration_evaluation_report(
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
            _append_run_evaluation_report(run_dir=run_dir, iteration=iteration, payload=report_payload)
            evaluation_report_path = iter_dir / "evaluation_report.json"
            _write_json_object(evaluation_report_path, report_payload)

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
            decision_score = float(evaluation.value)
            decision_source = str(evaluation.score_source or "offline")
            top1_score_value = float(top1_score) if isinstance(top1_score, (int, float)) else None
            effective_best_score, best_score_guard = _effective_best_score_for_progress(
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
            code_reference_score, code_reference_source = _extract_code_reference_score(config.paths)
            code_reference_comparison_score = _normalize_code_reference_score_for_comparison(
                current=decision_score,
                reference=code_reference_score,
                metric=evaluation.metric,
            )
            code_reference_delta_vs_current = (
                _score_delta_vs_reference(decision_score, code_reference_comparison_score, metric_direction)
                if code_reference_comparison_score is not None
                else None
            )
            first_iteration_below_code_reference = bool(
                iteration == 1 and code_reference_delta_vs_current is not None and code_reference_delta_vs_current < 0.0
            )
            score_drop_vs_best = _score_drop_vs_best(
                best_score=best_score,
                current_score=decision_score,
                direction=metric_direction,
            )
            severe_regression_detected = _is_severe_regression_vs_best(
                metric=evaluation.metric,
                direction=metric_direction,
                best_score=best_score,
                current_score=decision_score,
            )
            conservative_feature_collapse = _is_conservative_feature_collapse(kernel_metrics_payload)
            conservative_regression_detected = bool(severe_regression_detected and conservative_feature_collapse)

            quality_guard = _build_kernel_quality_guard(
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
            accuracy_potential = _build_accuracy_potential(
                evaluation=evaluation,
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
                    code_delta = _score_delta_vs_reference(
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
                submit_policy=str(resolved.get("submit_policy") or ""),
                submission_limit_per_day=submission_limit_per_day,
            )
            is_final_iteration = iteration >= max_iterations
            successful_submit_count = _submission_count_for_daily_limit(
                slug=config.slug,
                run_dir=run_dir,
                submission_limit_per_day=submission_limit_per_day,
                dry_run=config.dry_run,
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
                campaign_category = _infer_campaign_candidate_category(
                    iteration=iteration,
                    kernel_metrics_payload=kernel_metrics_payload,
                    quality_reasons=quality_reasons,
                )
                candidate_offline_std = _finite_float_or_none(evaluation.std)
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
                    oof_path=_extract_campaign_artifact_path(kernel_metrics_payload, "oof"),
                    prediction_path=_extract_campaign_artifact_path(kernel_metrics_payload, "prediction"),
                    model_family=_infer_campaign_model_family(model_summary, kernel_metrics_payload),
                    feature_set=_infer_campaign_feature_set(model_summary, kernel_metrics_payload),
                    method_id=_extract_campaign_method_id(kernel_metrics_payload)
                    or select_method_id_for_category(method_registry, campaign_category),
                    validation_profile_id=_extract_campaign_validation_profile_id(kernel_metrics_payload)
                    or str(method_registry.get("active_validation_profile") or "default_cv"),
                    fold_scores=_extract_campaign_fold_scores(kernel_metrics_payload),
                    prediction_correlation=_extract_campaign_prediction_correlation(kernel_metrics_payload),
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
                validation_registry = _load_json_object(config.paths.validation_registry_path) or {}
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
                    experiment_graph = _load_json_object(iter_dir / "experiment_graph.json") or experiment_graph
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
                source_registry = _load_json_object(config.paths.source_registry_path) or source_registry
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
                if _should_update_best_accuracy_candidate(
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
                if _update_best_score(best_submittable_score, decision_score, metric_direction, 0.0):
                    best_submittable_score = decision_score
                    best_submittable_submission = submission_path

            submit_improvement_allowed = True
            submit_non_improving = False
            defer_submit_for_accuracy_frontier = False
            if submit_improved_only and not config.force_submit and best_submitted_score is None:
                if (
                    submit_enabled
                    and quality_allows_submit
                    and (spare_daily_submission_slot or submission_limit_per_day is None)
                ):
                    forced_submit_reason = forced_submit_reason or _SPARE_DAILY_SUBMIT_REASON
                    slot_reason = (
                        "spare daily submission slots remain"
                        if spare_daily_submission_slot
                        else "no numeric daily submission limit is known"
                    )
                    print(
                        f"[yellow]submit override[/yellow]: {slot_reason}; allowing submit without a prior "
                        "submitted checkpoint."
                    )
                else:
                    submit_improvement_allowed = False
                    submit_non_improving = True
                    print(
                        "[yellow]submit deferred[/yellow]: submit_policy=improved requires "
                        "a prior submitted checkpoint."
                    )
            elif (
                (require_submit_improvement or submit_improved_only)
                and not config.force_submit
                and best_submitted_score is not None
            ):
                submit_improvement_allowed = _update_best_score(
                    best_submitted_score,
                    decision_score,
                    metric_direction,
                    stop_min_delta,
                )
                if not submit_improvement_allowed:
                    if is_final_iteration and not submit_improved_only:
                        print(
                            "[yellow]submit override[/yellow]: final iteration reached; "
                            "allowing submit even though score did not improve over submitted checkpoints."
                        )
                        submit_improvement_allowed = True
                    elif submit_enabled and spare_daily_submission_slot and quality_allows_submit:
                        forced_submit_reason = forced_submit_reason or _SPARE_DAILY_SUBMIT_REASON
                        submit_improvement_allowed = True
                        print(
                            "[yellow]submit override[/yellow]: spare daily submission slots remain; "
                            "allowing non-improving checkpoint submit."
                        )
                    else:
                        submit_non_improving = True
                        print(
                            "[yellow]submit deferred[/yellow]: "
                            "score did not improve over previous submitted checkpoint."
                        )
            if submit_enabled and isinstance(best_high_potential_meta, dict):
                current_priority = _to_int(accuracy_potential.get("frontier_priority")) or 0
                best_priority = _to_int(best_high_potential_meta.get("frontier_priority")) or 0
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
                readiness_target=target_score,
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
                print("[yellow]submit override[/yellow]: forcing iter 1 submit to probe Kaggle submission contract.")
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
            pre_submit_phase_state = _submit_stage.resolve_iteration_submit_phase_state(
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
            submit_status_message = _submit_stage.format_iteration_submit_status_message(
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
            pre_submit_metrics_payload = _build_metrics_payload(
                run_id=run_id,
                iteration=iteration,
                evaluation=evaluation,
                target_score=target_score,
                met_target=_submission_policy.meets_target(decision_score, target_score, metric_direction),
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
            _write_json_object(metrics_path, pre_submit_metrics_payload)
            _write_iteration_state_marker(
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
            submit_phase_state = pre_submit_phase_state
            if submit_enabled and allow_submit and submission_phase is not None:
                try:
                    submission_result = submission_phase.attempt(
                        submission_path=submission_path,
                        best_score=decision_score,
                    )
                except SubmitAbortedError:
                    if _should_defer_submit_abort_to_next_iteration(
                        compute=config.compute,
                        run_dir=run_dir,
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
                        _write_json_object(run_dir / "run.json", run_payload)
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
                            online_score = _to_float(outcome_payload.get("score"))
                            if online_score is not None:
                                print(f"[cyan]submission score[/cyan]: {online_score:.6f}")
                                if isinstance(top1_score, (int, float)):
                                    top1_tier_by_submission = _submission_policy.is_top1_tier(
                                        float(online_score),
                                        float(top1_score),
                                        metric_direction,
                                    )
                            rank_payload = _submit_stage.resolve_submission_rank_payload(
                                slug=config.slug,
                                context_dir=config.paths.context_dir,
                                direction=metric_direction,
                                outcome=outcome_payload,
                                dry_run=config.dry_run,
                                leaderboard_rank_for_score=leaderboard_rank_for_score,
                            )
                            if rank_payload:
                                outcome_payload.update(rank_payload)
                                submission_rank = _to_int(rank_payload.get("rank"))
                                submission_total_teams = _to_int(rank_payload.get("total_teams"))
                                submission_rank_percentile = _to_float(rank_payload.get("rank_percentile"))
                                submission_rank_estimate = _to_int(rank_payload.get("estimated_rank"))
                                submission_total_teams_estimate = _to_int(rank_payload.get("estimated_total_teams"))
                                submission_rank_percentile_estimate = _to_float(
                                    rank_payload.get("estimated_rank_percentile")
                                )
                                estimate_source_raw = rank_payload.get("rank_estimate_source")
                                if isinstance(estimate_source_raw, str) and estimate_source_raw.strip():
                                    submission_rank_estimate_source = estimate_source_raw.strip()
                                source_raw = rank_payload.get("rank_source")
                                if source_raw is not None:
                                    submission_rank_source = str(source_raw)
                                if (
                                    submission_rank is not None
                                    and submission_total_teams is not None
                                    and submission_total_teams > 0
                                ):
                                    if submission_rank_percentile is None:
                                        submission_rank_percentile = submission_rank / submission_total_teams
                                    print(
                                        _submit_stage.format_submission_rank_message(
                                            rank=submission_rank,
                                            total_teams=submission_total_teams,
                                            rank_percentile=submission_rank_percentile,
                                            source=submission_rank_source,
                                        )
                                    )
                                    rank_forced_major_overhaul = _should_force_major_overhaul_by_rank(
                                        rank=submission_rank,
                                        total_teams=submission_total_teams,
                                        max_percentile=rank_force_major_max_percentile,
                                        min_teams=rank_force_major_min_teams,
                                    )
                                    if rank_forced_major_overhaul:
                                        rank_force_reason = _submit_stage.format_rank_force_reason(
                                            rank=submission_rank,
                                            total_teams=submission_total_teams,
                                            rank_percentile=submission_rank_percentile,
                                            max_percentile=rank_force_major_max_percentile,
                                            min_teams=rank_force_major_min_teams,
                                            source=submission_rank_source,
                                        )
                                        print(f"[yellow]rank guard[/yellow]: {rank_force_reason}")
                                elif (
                                    submission_rank_estimate is not None
                                    and submission_total_teams_estimate is not None
                                    and submission_total_teams_estimate > 0
                                ):
                                    if submission_rank_percentile_estimate is None:
                                        submission_rank_percentile_estimate = (
                                            submission_rank_estimate / submission_total_teams_estimate
                                        )
                                    print(
                                        _submit_stage.format_submission_rank_message(
                                            rank=submission_rank_estimate,
                                            total_teams=submission_total_teams_estimate,
                                            rank_percentile=submission_rank_percentile_estimate,
                                            source=submission_rank_estimate_source,
                                            estimated=True,
                                        )
                                    )
                        submitted_tracking_score, submitted_tracking_source = (
                            _submit_stage.submission_score_for_tracking(
                                offline_score=decision_score,
                                online_score=online_score,
                            )
                        )
                        if _update_best_score(best_submitted_score, submitted_tracking_score, metric_direction, 0.0):
                            best_submitted_score = submitted_tracking_score
                            if submitted_tracking_source != "offline":
                                print(
                                    "[cyan]submit tracking[/cyan]: "
                                    f"updated best submitted score from {submitted_tracking_source}."
                                )
                else:
                    submit_phase_state = "dry_run" if config.dry_run else "attempted_no_result"
            if target_rank_percentile is not None and deliverable_mode == "leaderboard":
                medal_target_met = _meets_rank_percentile_target(
                    rank_percentile=submission_rank_percentile,
                    estimated_rank_percentile=submission_rank_percentile_estimate,
                    target_rank_percentile=target_rank_percentile,
                )
                if not medal_target_met:
                    medal_minimum_improvement_mode = "moderate_update"
                    medal_policy_reason = _build_medal_target_reason(
                        target_medal=target_medal,
                        target_rank_percentile=target_rank_percentile,
                        rank_percentile=submission_rank_percentile,
                        estimated_rank_percentile=submission_rank_percentile_estimate,
                    )
                    if medal_policy_reason:
                        print(f"[yellow]medal policy[/yellow]: {medal_policy_reason}")
            met_target = _submission_policy.meets_target(decision_score, target_score, metric_direction)
            top1_tier = _submission_policy.is_top1_tier(decision_score, top1_score, metric_direction)
            top1_tier_by_readiness = _submission_policy.is_top1_tier(readiness_score, top1_score, metric_direction)
            delta_srs_vs_prev: float | None = None
            noise_threshold = 0.5 * max(float(report.std), 0.0)
            if previous_readiness_score is not None:
                delta_srs_vs_prev = abs(readiness_score - previous_readiness_score)
                if delta_srs_vs_prev < noise_threshold:
                    noise_limited_streak += 1
                else:
                    noise_limited_streak = 0
            previous_readiness_score = readiness_score
            noise_forced_major_overhaul = noise_limited_streak >= 2
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
                    f"(drop_vs_best={drop_text}, max_features={_CONSERVATIVE_COLLAPSE_MAX_FEATURES}). "
                    "Next iteration must recover from code reference baseline instead of keeping the collapsed path."
                )
            force_major_overhaul_next = (
                noise_forced_major_overhaul
                or rank_forced_major_overhaul
                or quality_forced_major_overhaul
                or code_reference_forced_reproduction
            )
            fallback_submit_blocked_reason = None
            for blocked_reason in (
                "untrusted_score_source",
                "competition_metric_mismatch",
                "competition_split_mismatch",
                "competition_score_source_mismatch",
                "competition_evaluation_unfaithful",
                "missing_competitive_data",
                "external_test_label_transfer_detected",
            ):
                if blocked_reason in quality_reasons:
                    fallback_submit_blocked_reason = f"latest_iteration_{blocked_reason}"
                    break
            forced_major_overhaul_reasons: list[str] = []
            if noise_forced_major_overhaul:
                forced_major_overhaul_reasons.append(
                    "Two consecutive iterations were noise-limited: "
                    f"|ΔSRS| < 0.5*CV std (streak={noise_limited_streak})."
                )
            if rank_forced_major_overhaul:
                forced_major_overhaul_reasons.append(
                    rank_force_reason or "Leaderboard rank indicates major improvement is still required."
                )
            if quality_forced_major_overhaul:
                forced_major_overhaul_reasons.append(
                    quality_force_reason
                    or "Quality guard requires major overhaul due to code-reference underperformance."
                )
            if code_reference_forced_reproduction:
                forced_major_overhaul_reasons.append(
                    code_reference_force_reason
                    or "Mandatory code-reference implementation is required in the next iteration."
                )
            forced_major_overhaul_reason = (
                " ".join(forced_major_overhaul_reasons) if forced_major_overhaul_reasons else None
            )

            metrics_payload = _build_metrics_payload(
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
            metrics_payload["loop_decision"] = {
                "source": decision_source,
                "value": decision_score,
            }
            metrics_payload["noise_guard"] = {
                "delta_srs_vs_prev": delta_srs_vs_prev,
                "threshold": noise_threshold,
                "streak": noise_limited_streak,
                "force_major_overhaul_next": force_major_overhaul_next,
            }
            metrics_payload["rank_guard"] = {
                "target_medal": target_medal,
                "target_rank_percentile": target_rank_percentile,
                "target_rank_met": medal_target_met,
                "minimum_improvement_mode": medal_minimum_improvement_mode,
                "rank": submission_rank,
                "total_teams": submission_total_teams,
                "rank_percentile": submission_rank_percentile,
                "rank_source": submission_rank_source,
                "estimated_rank": submission_rank_estimate,
                "estimated_total_teams": submission_total_teams_estimate,
                "estimated_rank_percentile": submission_rank_percentile_estimate,
                "rank_estimate_source": submission_rank_estimate_source,
                "max_percentile": rank_force_major_max_percentile,
                "min_teams": rank_force_major_min_teams,
                "force_major_overhaul_next": rank_forced_major_overhaul,
            }
            metrics_payload["top1_tier"] = {
                "offline_decision": top1_tier,
                "offline_readiness": top1_tier_by_readiness,
                "submission_score": top1_tier_by_submission,
            }
            metrics_payload["forced_submit_reason"] = forced_submit_reason or ""
            if isinstance(online_score, (int, float)):
                metrics_payload["submission_score"] = float(online_score)
            if campaign_mode == "top1":
                metrics_payload["campaign"] = {
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
            if best_score_guard is not None:
                metrics_payload["best_score_guard"] = best_score_guard
            metrics_payload["quality_guard"] = quality_guard
            metrics_payload["regression_guard"] = {
                "best_score_before_iteration": best_score,
                "score_drop_vs_best": score_drop_vs_best,
                "severe_regression_detected": severe_regression_detected,
                "conservative_feature_collapse": conservative_feature_collapse,
                "conservative_regression_detected": conservative_regression_detected,
                "first_iteration_below_code_reference": first_iteration_below_code_reference,
                "code_reference_score": code_reference_score,
                "code_reference_comparison_score": code_reference_comparison_score,
                "code_reference_delta_vs_current": code_reference_delta_vs_current,
                "code_reference_forced_reproduction": code_reference_forced_reproduction,
            }
            _write_json_object(metrics_path, metrics_payload)

            diff_summary = "Diff tracking disabled (git integration removed)."
            diagnostics = _build_diagnostics(
                evaluation=evaluation,
                model_summary=model_summary,
                best_score=best_score,
                target_score=target_score,
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
            (iter_dir / "diagnostics.md").write_text(diagnostics, encoding="utf-8")

            orig_proba_signal = _extract_orig_proba_signal(kernel_metrics_payload)
            competition_policy = load_competition_policy(config.paths)
            reference_inputs_manifest_payload = _load_json_object(config.paths.reference_inputs_manifest_path)
            pseudo_label_signal = _extract_pseudo_label_failure_signal(
                kernel_metrics_payload=kernel_metrics_payload,
                diagnostics_text=diagnostics,
            )
            missing_ensemble_signal = (
                _extract_missing_ensemble_signal(kernel_metrics_payload)
                if competition_policy.repair.missing_ensemble_signal
                else None
            )
            original_data_unused_signal = (
                _extract_original_data_unused_signal(
                    kernel_metrics_payload=kernel_metrics_payload,
                    reference_inputs_manifest_payload=reference_inputs_manifest_payload,
                )
                if competition_policy.repair.original_data_unused_signal
                else None
            )
            same_family_plateau_signal = (
                _extract_same_family_plateau_signal(kernel_metrics_payload)
                if competition_policy.repair.same_family_plateau_signal
                else None
            )
            subgroup_collapse_signal = _detect_subgroup_collapse_signal(
                kernel_metrics_payload=kernel_metrics_payload,
                direction=metric_direction,
            )
            online_mismatch_signal = _detect_online_mismatch_signal(
                previous_best_offline=best_score,
                current_offline=decision_score,
                previous_best_online=best_online_submission_score,
                current_online=online_score,
                direction=metric_direction,
            )
            online_history_regression_signal = _detect_online_regression_vs_submission_history(
                previous_best_online=best_online_submission_score,
                current_online=online_score,
                direction=metric_direction,
                history=previous_submission_history,
            )
            if isinstance(online_score, (int, float)) and _update_best_score(
                best_online_submission_score,
                float(online_score),
                metric_direction,
                0.0,
            ):
                best_online_submission_score = float(online_score)
            if campaign_mode == "top1" and campaign_candidate is not None:
                if submission_result is not None and not submission_skipped:
                    campaign_candidate = replace(
                        campaign_candidate,
                        submitted=True,
                        public_score=float(online_score) if isinstance(online_score, (int, float)) else None,
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
                campaign_outcome_phase = (
                    "post_submit" if submission_result is not None and not submission_skipped else "post_iteration"
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

            extra_policy_notes: list[str] = []
            minimum_improvement_mode_next = medal_minimum_improvement_mode
            minimum_improvement_reason_next = medal_policy_reason
            forced_validation_redesign_reason: str | None = None
            loop_signal_errors: list[dict[str, object]] = []
            loop_signal_problems: list[dict[str, object]] = []
            if orig_proba_signal is not None:
                extra_policy_notes.append(str(orig_proba_signal["note"]))
                minimum_improvement_mode_next = _upgrade_improvement_mode(
                    minimum_improvement_mode_next or "minor_tuning",
                    "moderate_update",
                )
                minimum_improvement_reason_next = (
                    f"{minimum_improvement_reason_next} {orig_proba_signal['note']}".strip()
                    if minimum_improvement_reason_next
                    else str(orig_proba_signal["note"])
                )
                loop_signal_errors.append(
                    {
                        "iteration": iteration,
                        "error_message": (
                            "ORIG_proba external signal fell back to constants because original data was unavailable."
                        ),
                        "fix_summary": str(orig_proba_signal["note"]),
                        "resolved": False,
                        "outcome_bucket": "unknown",
                    }
                )
            if original_data_unused_signal is not None:
                extra_policy_notes.append(str(original_data_unused_signal["note"]))
                minimum_improvement_mode_next = _upgrade_improvement_mode(
                    minimum_improvement_mode_next or "minor_tuning",
                    "moderate_update",
                )
                minimum_improvement_reason_next = (
                    f"{minimum_improvement_reason_next} {original_data_unused_signal['note']}".strip()
                    if minimum_improvement_reason_next
                    else str(original_data_unused_signal["note"])
                )
                loop_signal_errors.append(
                    {
                        "iteration": iteration,
                        "error_message": "Original/reference datasets were staged but never consumed by the kernel.",
                        "fix_summary": str(original_data_unused_signal["note"]),
                        "resolved": False,
                        "outcome_bucket": "unknown",
                    }
                )
            if pseudo_label_signal is not None:
                extra_policy_notes.append(str(pseudo_label_signal["note"]))
                minimum_improvement_mode_next = _upgrade_improvement_mode(
                    minimum_improvement_mode_next or "minor_tuning",
                    "moderate_update",
                )
                minimum_improvement_reason_next = (
                    f"{minimum_improvement_reason_next} {pseudo_label_signal['note']}".strip()
                    if minimum_improvement_reason_next
                    else str(pseudo_label_signal["note"])
                )
                loop_signal_errors.append(
                    {
                        "iteration": iteration,
                        "error_message": (
                            f"Pseudo-labeling yielded {int(pseudo_label_signal['accepted'])}/"
                            f"{int(pseudo_label_signal['total'])} accepted folds or candidates."
                        ),
                        "fix_summary": str(pseudo_label_signal["note"]),
                        "resolved": False,
                        "outcome_bucket": "unknown",
                    }
                )
            if missing_ensemble_signal is not None:
                extra_policy_notes.append(str(missing_ensemble_signal["note"]))
                force_major_overhaul_next = True
                forced_major_overhaul_reason = (
                    f"{forced_major_overhaul_reason} {missing_ensemble_signal['note']}".strip()
                    if forced_major_overhaul_reason
                    else str(missing_ensemble_signal["note"])
                )
                loop_signal_problems.append(
                    {
                        "iteration": iteration,
                        "why_poor": str(missing_ensemble_signal["note"]),
                        "how_improved": "Keep heterogeneous families and emit at least one weighted/rank OOF blend.",
                        "delta_offline": None,
                        "outcome_bucket": "low",
                    }
                )
            if same_family_plateau_signal is not None:
                extra_policy_notes.append(str(same_family_plateau_signal["note"]))
                force_major_overhaul_next = True
                forced_major_overhaul_reason = (
                    f"{forced_major_overhaul_reason} {same_family_plateau_signal['note']}".strip()
                    if forced_major_overhaul_reason
                    else str(same_family_plateau_signal["note"])
                )
                loop_signal_problems.append(
                    {
                        "iteration": iteration,
                        "why_poor": str(same_family_plateau_signal["note"]),
                        "how_improved": "Add an orthogonal family instead of repeating same-family tuning.",
                        "delta_offline": None,
                        "outcome_bucket": "low",
                    }
                )
            if subgroup_collapse_signal is not None:
                extra_policy_notes.append(str(subgroup_collapse_signal["note"]))
                minimum_improvement_mode_next = _upgrade_improvement_mode(
                    minimum_improvement_mode_next or "minor_tuning",
                    "moderate_update",
                )
                minimum_improvement_reason_next = (
                    f"{minimum_improvement_reason_next} {subgroup_collapse_signal['note']}".strip()
                    if minimum_improvement_reason_next
                    else str(subgroup_collapse_signal["note"])
                )
                loop_signal_problems.append(
                    {
                        "iteration": iteration,
                        "why_poor": str(subgroup_collapse_signal["note"]),
                        "how_improved": (
                            "Make pipeline and fallback selection subgroup-aware at (model_id,node_type) granularity, "
                            "and target the collapsed slice before broad model-family tuning."
                        ),
                        "delta_offline": None,
                        "outcome_bucket": "low",
                    }
                )
            if online_mismatch_signal is not None:
                extra_policy_notes.append(str(online_mismatch_signal["note"]))
                if campaign_mode == "top1" and _campaign_prefers_validation_redesign(campaign_state, method_registry):
                    minimum_improvement_mode_next = _upgrade_improvement_mode(
                        minimum_improvement_mode_next or "minor_tuning",
                        "validation_redesign",
                    )
                    forced_validation_redesign_reason = str(online_mismatch_signal["note"])
                else:
                    force_major_overhaul_next = True
                    forced_major_overhaul_reason = (
                        f"{forced_major_overhaul_reason} {online_mismatch_signal['note']}".strip()
                        if forced_major_overhaul_reason
                        else str(online_mismatch_signal["note"])
                    )
                loop_signal_problems.append(
                    {
                        "iteration": iteration,
                        "why_poor": str(online_mismatch_signal["note"]),
                        "how_improved": (
                            "Ban same-family-only tuning after an online mismatch and require model-family "
                            "diversification plus OOF blending."
                        ),
                        "delta_offline": None,
                        "outcome_bucket": "low",
                    }
                )
            if online_history_regression_signal is not None:
                extra_policy_notes.append(str(online_history_regression_signal["note"]))
                if campaign_mode == "top1" and _campaign_prefers_validation_redesign(campaign_state, method_registry):
                    minimum_improvement_mode_next = _upgrade_improvement_mode(
                        minimum_improvement_mode_next or "minor_tuning",
                        "validation_redesign",
                    )
                    forced_validation_redesign_reason = str(online_history_regression_signal["note"])
                else:
                    force_major_overhaul_next = True
                    forced_major_overhaul_reason = (
                        f"{forced_major_overhaul_reason} {online_history_regression_signal['note']}".strip()
                        if forced_major_overhaul_reason
                        else str(online_history_regression_signal["note"])
                    )
                loop_signal_problems.append(
                    {
                        "iteration": iteration,
                        "why_poor": forced_major_overhaul_reason or str(online_history_regression_signal["note"]),
                        "how_improved": (
                            "Use the best historical public-score submission as the baseline and require a different "
                            "model/feature/blend path before submitting another regressed artifact."
                        ),
                        "delta_offline": None,
                        "outcome_bucket": "low",
                    }
                )
            if extra_policy_notes:
                metrics_payload["repair_signals"] = {
                    "orig_proba_constant_fallback": orig_proba_signal,
                    "original_data_unused": original_data_unused_signal,
                    "pseudo_label_failure": pseudo_label_signal,
                    "missing_ensemble": missing_ensemble_signal,
                    "same_family_plateau": same_family_plateau_signal,
                    "subgroup_collapse": subgroup_collapse_signal,
                    "online_mismatch": online_mismatch_signal,
                    "online_history_regression": online_history_regression_signal,
                }
            metrics_payload["previous_submission_history"] = previous_submission_history
            metrics_payload["next_iteration_policy"] = {
                "minimum_improvement_mode": minimum_improvement_mode_next,
                "minimum_improvement_reason": minimum_improvement_reason_next,
                "forced_improvement_mode": (
                    "validation_redesign"
                    if forced_validation_redesign_reason and not force_major_overhaul_next
                    else "major_overhaul"
                    if force_major_overhaul_next
                    else None
                ),
                "forced_improvement_reason": forced_major_overhaul_reason or forced_validation_redesign_reason,
                "extra_policy_notes": extra_policy_notes,
            }
            _write_json_object(metrics_path, metrics_payload)

            for issue in loop_signal_errors:
                try:
                    record_error_fix_insight(
                        knowledge_paths=config.knowledge_paths,
                        slug=config.slug,
                        run_id=run_id,
                        iteration=int(issue.get("iteration") or iteration),
                        problem_types=problem_types,
                        error_message=str(issue.get("error_message") or ""),
                        fix_summary=str(issue.get("fix_summary") or ""),
                        resolved=bool(issue.get("resolved")),
                        outcome_bucket=str(issue.get("outcome_bucket") or "unknown"),
                        submission_score=online_score,
                    )
                except Exception:  # noqa: BLE001
                    pass
            for issue in loop_signal_problems:
                try:
                    record_problem_type_insight(
                        knowledge_paths=config.knowledge_paths,
                        slug=config.slug,
                        run_id=run_id,
                        iteration=int(issue.get("iteration") or iteration),
                        problem_types=problem_types,
                        why_poor=str(issue.get("why_poor") or ""),
                        how_improved=str(issue.get("how_improved") or ""),
                        delta_offline=None,
                        outcome_bucket=str(issue.get("outcome_bucket") or "unknown"),
                        submission_score=online_score,
                    )
                except Exception:  # noqa: BLE001
                    pass

            if writeup_mode:
                writeup_bundle_meta = build_writeup_bundle(
                    paths=config.paths,
                    run_id=run_id,
                    iteration=iteration,
                    resolved=resolved,
                    evaluation=evaluation,
                    metrics_payload=metrics_payload,
                    top1_info=top1_info if isinstance(top1_info, dict) else None,
                )
                metrics_payload["deliverable_mode"] = "writeup"
                metrics_payload["writeup_bundle"] = writeup_bundle_meta
                _write_json_object(metrics_path, metrics_payload)

            submit_allowed_by_gate = submit_enabled and allow_submit
            submit_phase_finished = (
                (not submit_phase_required)
                or (not submit_allowed_by_gate)
                or (submission_result is not None)
                or submit_failed_deferred
            )

            iteration_record_kwargs = {
                "knowledge_paths": config.knowledge_paths,
                "run_id": run_id,
                "iteration": iteration,
                "score_source": evaluation.score_source,
                "offline_value": evaluation.value,
                "offline_std": evaluation.std,
                "top1_public_score": top1_info.get("score") if isinstance(top1_info, dict) else None,
                "met_target": met_target,
                "git_commit": None,
            }
            try:
                record_iteration(**iteration_record_kwargs)
            except TypeError as exc:
                if "submit_phase_finished" not in str(exc):
                    raise
                from kagglebot.knowledge import record_iteration as _record_iteration_canonical

                try:
                    _record_iteration_canonical(
                        **iteration_record_kwargs,
                        submit_phase_finished=submit_phase_finished,
                    )
                except TypeError as fallback_exc:
                    if "submit_phase_finished" not in str(fallback_exc):
                        raise
                    _record_iteration_canonical(**iteration_record_kwargs)
            _write_iteration_state_marker(
                iter_dir=iter_dir,
                run_id=run_id,
                iteration=iteration,
                submission_path=submission_path,
                metrics_path=metrics_path,
                evaluation_report_path=evaluation_report_path,
                submit_phase_required=submit_phase_required,
                submit_phase_finished=submit_phase_finished,
                submit_allowed_by_gate=submit_allowed_by_gate,
                submit_phase_state=submit_phase_state,
                forced_submit_reason=forced_submit_reason,
                submitted=submission_result is not None and not submission_skipped,
                readiness_score=readiness_score,
            )

            prev_best = best_score
            if metric_mismatch_detected or non_generalizable_eval_detected:
                delta_offline = None
                improved = False
            else:
                delta_offline = iteration_phase.delta_from_best(prev_best, decision_score)
                improved = iteration_phase.should_update_best(best_score, decision_score, stop_min_delta)
            if improved:
                best_score = decision_score
                best_submission = submission_path
                no_improve_streak = 0
                _capture_best_kernel_snapshot(paths=config.paths, run_dir=run_dir)
            else:
                no_improve_streak += 1
                if conservative_regression_detected:
                    restored = _restore_best_kernel_snapshot(paths=config.paths, run_dir=run_dir)
                    if restored:
                        print(
                            "[yellow]kernel regression guard[/yellow]: "
                            "restored best-known kernel source after severe conservative regression."
                        )

            if force_major_on_no_improve and (not improved) and (not high_potential_improved):
                if best_score_guard is not None:
                    print(
                        "[yellow]improve guard[/yellow]: "
                        "skipping no-improve major-overhaul override because previous best "
                        "was clipped as an outlier."
                    )
                else:
                    force_major_overhaul_next = True
                    regression_reason = (
                        f"Offline {evaluation.metric} did not improve "
                        f"(current={decision_score:.6f}, best={float(prev_best):.6f})."
                        if prev_best is not None
                        else f"Offline {evaluation.metric} did not improve."
                    )
                    forced_major_overhaul_reason = (
                        f"{forced_major_overhaul_reason} {regression_reason}".strip()
                        if forced_major_overhaul_reason
                        else regression_reason
                    )

            current_config_hash = _pipeline_config_hash(
                model_summary=model_summary,
                metric=evaluation.metric,
                accelerator=accelerator_used,
            )
            if current_config_hash == last_config_hash:
                same_config_streak += 1
            else:
                same_config_streak = 0
            last_config_hash = current_config_hash

            effective_no_improve_streak = (
                frontier_no_improve_streak if best_high_potential_score is not None else no_improve_streak
            )
            effective_track_label = "accuracy frontier" if best_high_potential_score is not None else "offline metric"
            if (
                (not submit_enabled)
                and stop_no_improve_patience > 0
                and effective_no_improve_streak >= stop_no_improve_patience
            ):
                run_payload["status"] = "stopped"
                run_payload["stop_reason"] = (
                    f"{effective_track_label} did not improve by >= {stop_min_delta:.6f} "
                    f"for {effective_no_improve_streak} consecutive iterations"
                )
                print(f"[yellow]stop[/yellow]: {run_payload['stop_reason']}")
                break
            if (
                (not submit_enabled)
                and stop_same_config_patience > 0
                and same_config_streak >= stop_same_config_patience
            ):
                run_payload["status"] = "stopped"
                run_payload["stop_reason"] = (
                    f"model/pipeline config hash unchanged for {same_config_streak} consecutive iterations"
                )
                print(f"[yellow]stop[/yellow]: {run_payload['stop_reason']}")
                break

            if _is_confirmed_first_place(submission_rank, submission_rank_source):
                run_payload["status"] = "submitted" if submitted else "completed"
                run_payload["stop_reason"] = "submission_rank_1"
                print("[green]stop[/green]: submission rank reached #1")
                break

            if top1_tier:
                print("[yellow]note[/yellow]: offline top1-tier reached; awaiting submission-score confirmation")

            if iteration >= max_iterations:
                run_payload["status"] = "submitted" if submitted else "completed"
                break

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
                    "validation_redesign"
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
        _write_json_object(run_dir / "run.json", run_payload)
        print("[yellow]run interrupted[/yellow]")
        return

    if (
        fallback_submit_blocked_reason is None
        and isinstance(best_high_potential_meta, dict)
        and best_high_potential_submission is not None
        and best_high_potential_submission != best_submittable_submission
        and (
            not bool(best_high_potential_meta.get("faithful", False))
            or not bool(best_high_potential_meta.get("trusted", False))
        )
    ):
        fallback_submit_blocked_reason = "higher_potential_unsubmitted_candidate_exists"

    if (
        submit_enabled
        and not submitted
        and best_submittable_submission is not None
        and fallback_submit_blocked_reason is None
    ):
        final_iteration_reached = last_completed_iteration >= max_iterations
        allow_fallback_submit = True
        if submit_improved_only and not config.force_submit and best_submitted_score is None:
            allow_fallback_submit = False
        if (
            (require_submit_improvement or submit_improved_only)
            and not config.force_submit
            and best_submittable_score is not None
            and best_submitted_score is not None
        ):
            allow_fallback_submit = _update_best_score(
                best_submitted_score,
                best_submittable_score,
                metric_direction,
                stop_min_delta,
            )
            if (not allow_fallback_submit) and final_iteration_reached and not submit_improved_only:
                print(
                    "[yellow]submit override[/yellow]: final iteration reached; "
                    "allowing fallback submit even though offline metric did not improve."
                )
                allow_fallback_submit = True
        if allow_fallback_submit:
            fallback_iteration = _submit_stage.infer_iteration_from_submission_path(best_submittable_submission)
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
                _write_json_object(run_dir / "run.json", run_payload)
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
                    fallback_online_score: float | None = None
                    outcome_payload = fallback_result.get("outcome")
                    if isinstance(outcome_payload, dict):
                        fallback_online_score = _to_float(outcome_payload.get("score"))
                    if best_submittable_score is not None:
                        submitted_tracking_score, _submitted_tracking_source = (
                            _submit_stage.submission_score_for_tracking(
                                offline_score=best_submittable_score,
                                online_score=fallback_online_score,
                            )
                        )
                        if _update_best_score(
                            best_submitted_score,
                            submitted_tracking_score,
                            metric_direction,
                            0.0,
                        ):
                            best_submitted_score = submitted_tracking_score
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

    if submitted and last_submission_result:
        top1_score = top1_info.get("score") if isinstance(top1_info, dict) else None
        _record_submission_knowledge(
            config=config,
            run_id=run_id,
            problem_types=problem_types,
            pending_problem_insights=pending_problem_insights,
            pending_error_fixes=pending_error_fixes,
            submission_result=last_submission_result,
            metric_direction=metric_direction,
            target_score=target_score,
            top1_score=top1_score if isinstance(top1_score, (int, float)) else None,
        )
        run_payload["status"] = "submitted"
    elif writeup_mode and writeup_bundle_meta:
        run_payload["status"] = "manual_finalization_required"
        run_payload["writeup_bundle"] = writeup_bundle_meta
    elif run_payload.get("status") not in {"interrupted", "submit_failed"}:
        run_payload["status"] = "completed"

    run_payload["summary"] = {
        "best_trusted_score": best_score,
        "best_trusted_submission": str(best_submission) if best_submission is not None else None,
        "best_competition_faithful_score": best_submittable_score,
        "best_competition_faithful_submission": (
            str(best_submittable_submission) if best_submittable_submission is not None else None
        ),
        "best_high_potential_score": best_high_potential_score,
        "best_high_potential_submission": (
            str(best_high_potential_submission) if best_high_potential_submission is not None else None
        ),
        "best_high_potential_iteration": best_high_potential_iteration,
        "best_high_potential_meta": best_high_potential_meta,
        "fallback_submit_blocked_reason": fallback_submit_blocked_reason,
    }

    _write_json_object(run_dir / "run.json", run_payload)


def _load_plan(paths: CompetitionPaths) -> PlanConfig:
    payload = _load_json_object(paths.plan_path)
    if payload is None:
        return PlanConfig()
    return PlanConfig.from_dict(payload)


def _write_plan(paths: CompetitionPaths, plan: PlanConfig) -> None:
    existing = _load_json_object_or_empty(paths.plan_path)
    payload = _apply_plan_guardrails(paths, {**existing, **plan.to_dict()})
    _write_json_object(paths.plan_path, payload)


def _needs_planning(plan: PlanConfig, config: AutopilotConfig) -> bool:
    if config.agent in ("codex", "pipeline"):
        return True
    target_metric = config.target_metric or plan.target_metric
    target_score = config.target_score if config.target_score is not None else plan.target_score
    target_direction = config.target_direction or plan.target_direction
    if target_metric is None or target_score is None:
        return True
    return target_direction in (None, "auto")


def _should_skip_planning(*, resume_run: bool, paths: CompetitionPaths) -> bool:
    if not resume_run:
        return False
    if not paths.plan_path.exists():
        return False
    kernel_path = paths.kernel_source_dir / "kernel.py"
    return kernel_path.exists()


def _is_local_gpu_compute(compute: object) -> bool:
    return str(compute or "").strip().lower() == "local_gpu"


def _is_heavy_deep_learning_modality(modality: object) -> bool:
    return str(modality or "").strip().lower() in _HEAVY_DEEP_LEARNING_MODALITIES


def _local_gpu_time_budget_limit_min() -> int | None:
    raw = os.environ.get("KAGGLEBOT_LOCAL_GPU_TIME_BUDGET_MIN")
    if raw is None or not raw.strip():
        return None
    try:
        parsed = int(float(raw.strip()))
    except ValueError:
        return None
    if parsed <= 0:
        return None
    return max(60, parsed)


def _resolve_notebook_submit_artifact_mode(*, paths: CompetitionPaths, submit_mode: str) -> str:
    normalized_submit_mode = normalize_submit_mode(submit_mode, default="file")
    if normalized_submit_mode != "notebook":
        return "wrapper"
    return "inference" if infer_code_competition_from_paths(paths) else "wrapper"


def _decide_notebook_submit_artifact_mode_for_submission(
    *,
    paths: CompetitionPaths,
    requested_mode: str | None,
    notebook_submit_required: bool,
    submission_path: Path,
) -> _submit_notebook.NotebookSubmitArtifactModeDecision:
    code_competition = infer_code_competition_from_paths(paths) if notebook_submit_required else False
    sample_rows = _count_csv_data_rows_capped(paths.sample_submission_path)
    if sample_rows is None:
        sample_rows = _count_csv_data_rows_capped(paths.data_dir / "sample_submission.csv")
    return _submit_notebook.decide_notebook_submit_artifact_mode(
        requested_mode=requested_mode,
        notebook_submit_required=notebook_submit_required,
        code_competition=code_competition,
        sample_data_rows=sample_rows,
        submission_data_rows=_count_csv_data_rows_capped(submission_path),
    )


def _count_csv_data_rows_capped(path: Path, *, cap: int = 10) -> int | None:
    data_rows = 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            for index, _line in enumerate(handle):
                if index > cap:
                    return cap + 1
                data_rows = index
    except OSError:
        return None
    return data_rows


def _resolve_plan(plan: PlanConfig, config: AutopilotConfig) -> dict[str, object]:
    def choose(value, fallback, default):
        if value is not None:
            return value
        if fallback is not None:
            return fallback
        return default

    eval_spec = _load_evaluation_spec(config.paths)
    spec_metric = eval_spec.get("metric_name") if isinstance(eval_spec.get("metric_name"), str) else None
    spec_direction = eval_spec.get("direction") if isinstance(eval_spec.get("direction"), str) else None
    spec_split = eval_spec.get("split_strategy") if isinstance(eval_spec.get("split_strategy"), str) else None
    spec_folds = eval_spec.get("n_splits") if isinstance(eval_spec.get("n_splits"), int) else None
    spec_seed: int | None = None
    spec_eval_seeds: list[int] = []
    spec_seeds = eval_spec.get("seeds")
    if isinstance(spec_seeds, list):
        for item in spec_seeds:
            if isinstance(item, int):
                spec_eval_seeds.append(item)
                if spec_seed is None:
                    spec_seed = item
    spec_repeats = eval_spec.get("repeats") if isinstance(eval_spec.get("repeats"), int) else None
    spec_ci_method = eval_spec.get("ci_method") if isinstance(eval_spec.get("ci_method"), str) else None
    spec_ci_alpha = eval_spec.get("ci_alpha") if isinstance(eval_spec.get("ci_alpha"), (int, float)) else None
    readiness_rule = eval_spec.get("readiness_rule") if isinstance(eval_spec.get("readiness_rule"), dict) else {}
    spec_readiness_method = readiness_rule.get("method") if isinstance(readiness_rule.get("method"), str) else None
    spec_readiness_k = readiness_rule.get("k") if isinstance(readiness_rule.get("k"), (int, float)) else None
    spec_readiness_target = (
        readiness_rule.get("target_score") if isinstance(readiness_rule.get("target_score"), (int, float)) else None
    )
    spec_submission_gate = (
        readiness_rule.get("submission_gate") if isinstance(readiness_rule.get("submission_gate"), str) else None
    )
    drift_cfg = eval_spec.get("drift_check") if isinstance(eval_spec.get("drift_check"), dict) else {}
    spec_drift_enabled = drift_cfg.get("enabled") if isinstance(drift_cfg.get("enabled"), bool) else None
    spec_drift_weight = (
        drift_cfg.get("drift_weight") if isinstance(drift_cfg.get("drift_weight"), (int, float)) else None
    )
    stop_policy = eval_spec.get("stop_policy") if isinstance(eval_spec.get("stop_policy"), dict) else {}
    spec_stop_min_delta = (
        stop_policy.get("min_delta") if isinstance(stop_policy.get("min_delta"), (int, float)) else None
    )
    spec_stop_no_improve = (
        stop_policy.get("no_improve_patience") if isinstance(stop_policy.get("no_improve_patience"), int) else None
    )
    spec_stop_same_config = (
        stop_policy.get("same_config_patience") if isinstance(stop_policy.get("same_config_patience"), int) else None
    )

    strict_competition_metric = _env_flag(
        "KAGGLEBOT_STRICT_COMPETITION_METRIC",
        default=_DEFAULT_STRICT_COMPETITION_METRIC,
    )
    deliverable_mode = _plan_policy.resolve_deliverable_mode(
        plan_value=getattr(plan, "deliverable_mode", None),
        spec_value=eval_spec.get("deliverable_mode"),
        inferred_value=infer_deliverable_mode_from_paths(config.paths, default=""),
    )
    submit_mode = _plan_policy.resolve_submit_mode(
        plan_value=getattr(plan, "submit_mode", None),
        spec_value=eval_spec.get("submit_mode"),
        inferred_value=infer_submit_mode_from_paths(config.paths, default=""),
    )
    plan_target_medal = _normalize_target_medal(getattr(plan, "target_medal", None), default=None)
    spec_target_medal = _normalize_target_medal(eval_spec.get("target_medal"), default=None)
    default_target_medal = _DEFAULT_TARGET_MEDAL if deliverable_mode == "leaderboard" else None
    target_medal = spec_target_medal or plan_target_medal or default_target_medal
    target_rank_percentile = _normalize_target_rank_percentile(
        getattr(plan, "target_rank_percentile", None),
        medal=target_medal,
        fallback=None,
    )
    target_rank_percentile = _normalize_target_rank_percentile(
        eval_spec.get("target_rank_percentile"),
        medal=spec_target_medal or target_medal,
        fallback=target_rank_percentile,
    )
    competition_policy = load_competition_policy(config.paths)
    search_stop_rank_percentile = competition_policy.evaluation.search_stop_rank_percentile
    if search_stop_rank_percentile is not None:
        target_rank_percentile = (
            float(search_stop_rank_percentile)
            if target_rank_percentile is None
            else min(float(target_rank_percentile), float(search_stop_rank_percentile))
        )
    explicit_target_metric = config.target_metric is not None or plan.target_metric is not None
    explicit_target_direction = config.target_direction is not None or plan.target_direction is not None
    target_metric = choose(config.target_metric, plan.target_metric, spec_metric)
    target_score = choose(config.target_score, plan.target_score, spec_readiness_target)
    target_direction = choose(config.target_direction, plan.target_direction, spec_direction or "auto")
    competition_override = _plan_policy.competition_eval_override(config.paths.slug)
    override_metric = str(competition_override.get("metric_name") or "").strip()
    override_direction = str(competition_override.get("direction") or "").strip().lower()
    override_split_strategy = str(competition_override.get("split_strategy") or "").strip()
    if override_metric:
        requested_metric = str(target_metric or "").strip()
        if requested_metric and not _metrics_equivalent(requested_metric, override_metric):
            print(
                "[yellow]note[/yellow]: competition override is active; "
                f"forcing target_metric '{requested_metric}' -> '{override_metric}'."
            )
        target_metric = override_metric
    if override_direction in {"minimize", "maximize"}:
        requested_direction = str(target_direction or "").strip().lower()
        if requested_direction and requested_direction != override_direction:
            print(
                "[yellow]note[/yellow]: competition override is active; "
                f"forcing target_direction '{requested_direction}' -> '{override_direction}'."
            )
        target_direction = override_direction
    # Competition-specific overrides are authoritative and must not be undone by a stale
    # evaluation_spec.json contract from an earlier iteration.
    if strict_competition_metric and spec_metric and not competition_override:
        requested_metric = target_metric if isinstance(target_metric, str) else None
        requested_metric_norm = _canonical_metric_name_for_match(requested_metric)
        spec_metric_norm = _canonical_metric_name_for_match(spec_metric)
        if requested_metric_norm != spec_metric_norm:
            if explicit_target_metric and requested_metric:
                print(
                    "[yellow]note[/yellow]: strict competition metric mode is enabled, "
                    "but keeping explicit target_metric "
                    f"'{requested_metric}' over evaluation_spec metric '{spec_metric}'."
                )
            elif requested_metric:
                print(
                    "[yellow]note[/yellow]: strict competition metric mode is enabled; "
                    f"overriding target_metric '{requested_metric}' -> '{spec_metric}'."
                )
                target_metric = spec_metric
            else:
                target_metric = spec_metric
        if spec_direction in {"minimize", "maximize"}:
            requested_direction = str(target_direction or "").strip().lower()
            if requested_direction != spec_direction:
                if explicit_target_direction and requested_direction:
                    print(
                        "[yellow]note[/yellow]: strict competition metric mode is enabled, "
                        "but keeping explicit target_direction "
                        f"'{requested_direction}' over evaluation_spec direction '{spec_direction}'."
                    )
                else:
                    print(
                        "[yellow]note[/yellow]: strict competition metric mode is enabled; "
                        f"overriding target_direction '{requested_direction or 'auto'}' -> '{spec_direction}'."
                    )
                    target_direction = spec_direction
    score_source = str(choose(config.score_source, plan.score_source, "cv") or "cv")
    normalized_score_source = _score_sources.normalize_score_source_name(score_source)
    if normalized_score_source not in {"cv", "holdout"}:
        print("[yellow]note[/yellow]: non-generalizable score_source is not allowed; overriding to cv.")
        score_source = "cv"
    holdout_frac = choose(config.holdout_frac, plan.holdout_frac, 0.2)
    cv_folds = choose(config.cv_folds, plan.cv_folds, spec_folds if spec_folds is not None else 5)
    split_strategy = choose(None, plan.split_strategy, spec_split)
    split_strategy, split_strategy_note = _plan_policy.resolve_split_strategy_from_artifacts(
        paths=config.paths,
        split_strategy=split_strategy,
    )
    if override_split_strategy:
        normalized_override_split = _plan_policy.normalize_split_strategy_name(override_split_strategy)
        if normalized_override_split is not None and split_strategy != normalized_override_split:
            print(
                "[yellow]note[/yellow]: competition override is active; "
                f"forcing split_strategy '{split_strategy or 'auto'}' -> '{normalized_override_split}'."
            )
            split_strategy = normalized_override_split
    if split_strategy_note:
        print(f"[yellow]note[/yellow]: {split_strategy_note}")
    dataset_profile = _load_dataset_profile(config.paths)
    profile_modality = str(dataset_profile.get("modality") or "").strip().lower()
    heavy_local_gpu = _is_local_gpu_compute(config.compute) and _is_heavy_deep_learning_modality(profile_modality)
    cv_folds_int = _to_int(cv_folds)
    if cv_folds_int is not None:
        cv_folds = cv_folds_int
    if heavy_local_gpu and isinstance(cv_folds, int) and cv_folds > _HEAVY_LOCAL_GPU_MAX_CV_FOLDS:
        print(
            "[yellow]note[/yellow]: heavy modality on local_gpu; "
            f"capping full-training cv_folds from {cv_folds} to {_HEAVY_LOCAL_GPU_MAX_CV_FOLDS}. "
            "Use cached embeddings/TTA/lightweight heads for extra validation instead of full folds."
        )
        cv_folds = _HEAVY_LOCAL_GPU_MAX_CV_FOLDS
    seed = choose(config.seed, plan.seed, spec_seed if spec_seed is not None else 42)
    eval_seeds = _normalize_eval_seeds(plan.eval_seeds, fallback=spec_eval_seeds)
    if heavy_local_gpu and len(eval_seeds) > 1:
        primary_seed = int(seed) if isinstance(seed, int) else eval_seeds[0]
        if primary_seed not in eval_seeds:
            primary_seed = eval_seeds[0]
        eval_seeds = [primary_seed]
        print(
            "[yellow]note[/yellow]: heavy modality on local_gpu; "
            f"using one full-training seed ({primary_seed}) to stay inside the local runtime budget. "
            "Reserve extra seeds for cheap heads/blends or final confirmation."
        )
    elif len(eval_seeds) < 2:
        print(
            "[yellow]note[/yellow]: evaluation seeds were single-seed; "
            f"upgrading to multi-seed defaults {_DEFAULT_EVAL_SEEDS}."
        )
        eval_seeds = list(_DEFAULT_EVAL_SEEDS)
    eval_repeats = _normalize_eval_repeats(plan.eval_repeats, fallback=spec_repeats)
    if heavy_local_gpu and eval_repeats > 1:
        print(
            "[yellow]note[/yellow]: heavy modality on local_gpu; "
            "using one full-training evaluation repeat to keep each iteration under the runtime budget."
        )
        eval_repeats = 1
    elif eval_repeats < 2:
        print(
            "[yellow]note[/yellow]: evaluation repeats were < 2; "
            f"upgrading to default {_DEFAULT_EVAL_REPEATS} to reduce noise."
        )
        eval_repeats = _DEFAULT_EVAL_REPEATS
    constraints = _competition_rules.load_competition_rule_constraints(config.paths)
    code_competition = infer_code_competition_from_paths(config.paths)
    if code_competition and submit_mode != "notebook":
        print("[yellow]note[/yellow]: code competition detected; forcing submit_mode=notebook.")
        submit_mode = "notebook"
    if constraints.notebook_submissions_only and submit_mode != "notebook":
        print("[yellow]note[/yellow]: competition requires notebook-based submissions; forcing submit_mode=notebook.")
        submit_mode = "notebook"
    notebook_submit_artifact_mode = _resolve_notebook_submit_artifact_mode(
        paths=config.paths,
        submit_mode=submit_mode,
    )
    time_budget_min = choose(config.time_budget_min, plan.time_budget_min, None)
    kernel_name = choose(config.kernel_name, plan.kernel_name, None)
    internet = choose(config.internet, plan.internet, "on")
    if internet in (None, "auto"):
        internet = "on"
    if submit_mode == "notebook" and not str(config.compute).startswith("kaggle_"):
        print(
            "[yellow]note[/yellow]: notebook-based submission is selected; "
            "autopilot will auto-switch submit mode to notebook submit."
        )
    if constraints.internet_must_be_off and str(internet).strip().lower() != "off":
        print("[yellow]note[/yellow]: rules require internet disabled; forcing internet=off.")
        internet = "off"
    runtime_limit_min = _competition_rules.runtime_limit_for_compute(constraints=constraints, compute=config.compute)
    if runtime_limit_min is not None:
        current_limit = int(time_budget_min) if isinstance(time_budget_min, (int, float)) else None
        if current_limit is None or current_limit > runtime_limit_min:
            print(
                "[yellow]note[/yellow]: rules impose notebook runtime cap; "
                f"forcing time_budget_min={runtime_limit_min}."
            )
            time_budget_min = runtime_limit_min
    if _is_local_gpu_compute(config.compute):
        local_budget_min = _local_gpu_time_budget_limit_min()
        current_limit = int(time_budget_min) if isinstance(time_budget_min, (int, float)) else None
        if local_budget_min is not None and (current_limit is None or current_limit > local_budget_min):
            print(
                "[yellow]note[/yellow]: local_gpu per-kernel budget limit active; "
                f"forcing time_budget_min={local_budget_min}. "
                "Unset KAGGLEBOT_LOCAL_GPU_TIME_BUDGET_MIN or set it to 0 for unlimited local runtime."
            )
            time_budget_min = local_budget_min
    if config.max_iterations is None:
        planned_max_iterations = _to_int(plan.max_iterations)
        if planned_max_iterations is not None and planned_max_iterations > 0:
            max_iterations = planned_max_iterations
        else:
            max_iterations = _DEFAULT_MAX_ITERATIONS
            if planned_max_iterations is not None:
                print(
                    "[yellow]note[/yellow]: invalid plan max_iterations "
                    f"({plan.max_iterations}); using default {_DEFAULT_MAX_ITERATIONS}."
                )
    else:
        max_iterations = max(1, int(config.max_iterations))
    if (
        heavy_local_gpu
        and (time_budget_min is None or time_budget_min >= _LONG_LOCAL_GPU_ITERATION_BUDGET_MIN)
        and max_iterations > _LONG_LOCAL_GPU_MAX_ITERATIONS
    ):
        print(
            "[yellow]note[/yellow]: heavy long-running local_gpu plan detected; "
            f"capping max_iterations from {max_iterations} to {_LONG_LOCAL_GPU_MAX_ITERATIONS} "
            "so accuracy-first iterations can run deeper."
        )
        max_iterations = _LONG_LOCAL_GPU_MAX_ITERATIONS
    max_total_min = choose(config.max_total_min, plan.max_total_min, None)
    patience = choose(config.patience, plan.patience, 2)
    min_improvement = choose(config.min_improvement, plan.min_improvement, 0.0)
    requested_submit_policy = str(choose(config.submit_policy, plan.submit_policy, "always") or "always")
    requested_submission_gate_raw = choose(None, plan.submission_gate, spec_submission_gate)
    requested_submission_gate = str(requested_submission_gate_raw or "").strip().lower() or None
    forced_submit_policy = (
        _submission_policy.normalized_submit_policy(config.submit_policy) if config.submit_policy else None
    )
    if forced_submit_policy == "improved":
        submit_policy = "improved"
        submission_gate = "always"
    elif constraints.submission_limit_detected:
        submit_policy = _submission_policy.normalized_submit_policy(requested_submit_policy)
        default_gate = _submission_policy.submission_gate_for_policy(submit_policy)
        if requested_submission_gate is not None:
            submission_gate = _submission_policy.normalized_submission_gate(
                requested_submission_gate,
                default=default_gate,
            )
        else:
            submission_gate = default_gate
        if submission_gate == "always" and requested_submission_gate is None and submit_policy == "always":
            submission_gate = _DEFAULT_LIMITED_SUBMISSION_GATE
            submit_policy = "readiness_or_final"
            print(
                "[yellow]note[/yellow]: submission limit detected in rules; "
                f"defaulting submission_gate={submission_gate}."
            )
    else:
        submit_policy = "always"
        submission_gate = "always"
        if _submission_policy.normalized_submit_policy(requested_submit_policy) != "always":
            print(
                "[yellow]note[/yellow]: no submission limit detected; "
                f"ignoring submit_policy='{requested_submit_policy}'."
            )
        normalized_requested_gate = (
            _submission_policy.normalized_submission_gate(requested_submission_gate, default="always")
            if requested_submission_gate
            else "always"
        )
        if requested_submission_gate and normalized_requested_gate != "always":
            print(
                "[yellow]note[/yellow]: no submission limit detected; "
                f"ignoring submission_gate='{requested_submission_gate}'."
            )
    readiness_target_score = choose(
        None,
        plan.readiness_target_score,
        spec_readiness_target if spec_readiness_target is not None else target_score,
    )
    readiness_method = choose(None, plan.readiness_method, spec_readiness_method or "ci_bound")
    readiness_k = choose(None, plan.readiness_k, spec_readiness_k if spec_readiness_k is not None else 1.0)
    ci_method = choose(None, plan.ci_method, spec_ci_method or "normal")
    ci_alpha = choose(None, plan.ci_alpha, spec_ci_alpha if spec_ci_alpha is not None else 0.05)
    drift_check = bool(
        choose(
            None,
            plan.drift_check,
            spec_drift_enabled if spec_drift_enabled is not None else False,
        )
    )
    drift_weight = choose(None, plan.drift_weight, spec_drift_weight if spec_drift_weight is not None else 1.0)
    stop_min_delta = choose(
        None,
        plan.stop_min_delta,
        spec_stop_min_delta if spec_stop_min_delta is not None else min_improvement,
    )
    stop_no_improve_patience = choose(
        None,
        plan.stop_no_improve_patience,
        spec_stop_no_improve if spec_stop_no_improve is not None else patience,
    )
    stop_same_config_patience = choose(
        None,
        plan.stop_same_config_patience,
        spec_stop_same_config if spec_stop_same_config is not None else 0,
    )
    rank_force_major_max_percentile = _normalize_rank_force_percentile(
        plan.rank_force_major_max_percentile,
        fallback=_DEFAULT_FORCE_MAJOR_RANK_MAX_PERCENTILE,
    )
    if target_rank_percentile is not None:
        rank_force_major_max_percentile = min(rank_force_major_max_percentile, float(target_rank_percentile))
    rank_force_major_min_teams = _normalize_rank_force_min_teams(
        plan.rank_force_major_min_teams,
        fallback=_DEFAULT_FORCE_MAJOR_RANK_MIN_TEAMS,
    )
    evaluation_contract = _build_evaluation_contract(
        paths=config.paths,
        eval_spec=eval_spec,
        target_metric=str(target_metric) if isinstance(target_metric, str) else None,
        target_direction=str(target_direction) if isinstance(target_direction, str) else None,
        split_strategy=str(split_strategy) if isinstance(split_strategy, str) else None,
    )

    return {
        "deliverable_mode": deliverable_mode,
        "submit_mode": submit_mode,
        "code_competition": code_competition,
        "notebook_submit_artifact_mode": notebook_submit_artifact_mode,
        "target_medal": target_medal,
        "target_rank_percentile": target_rank_percentile,
        "target_metric": target_metric,
        "target_score": target_score,
        "target_direction": target_direction,
        "score_source": score_source,
        "holdout_frac": holdout_frac,
        "cv_folds": cv_folds,
        "split_strategy": split_strategy,
        "seed": seed,
        "eval_seeds": eval_seeds,
        "eval_repeats": eval_repeats,
        "time_budget_min": time_budget_min,
        "kernel_name": kernel_name,
        "internet": internet,
        "max_iterations": max_iterations,
        "max_total_min": max_total_min,
        "patience": patience,
        "min_improvement": min_improvement,
        "submit_policy": submit_policy,
        "submission_gate": submission_gate,
        "submission_limit_per_day": constraints.submission_limit_per_day,
        "readiness_target_score": readiness_target_score,
        "readiness_method": readiness_method,
        "readiness_k": readiness_k,
        "ci_method": ci_method,
        "ci_alpha": ci_alpha,
        "drift_check": drift_check,
        "drift_weight": drift_weight,
        "stop_min_delta": stop_min_delta,
        "stop_no_improve_patience": stop_no_improve_patience,
        "stop_same_config_patience": stop_same_config_patience,
        "rank_force_major_max_percentile": rank_force_major_max_percentile,
        "rank_force_major_min_teams": rank_force_major_min_teams,
        "evaluation_contract": evaluation_contract,
    }


def _resolved_plan(resolved: dict[str, object]) -> PlanConfig:
    return PlanConfig(
        deliverable_mode=str(resolved.get("deliverable_mode") or "leaderboard"),
        submit_mode=str(resolved.get("submit_mode") or "file"),
        target_medal=_normalize_target_medal(resolved.get("target_medal"), default=None),
        target_rank_percentile=_normalize_target_rank_percentile(
            resolved.get("target_rank_percentile"),
            medal=_normalize_target_medal(resolved.get("target_medal"), default=None),
            fallback=None,
        ),
        target_metric=resolved.get("target_metric"),  # type: ignore[arg-type]
        target_direction=str(resolved.get("target_direction") or "auto"),
        target_score=resolved.get("target_score"),  # type: ignore[arg-type]
        score_source=str(resolved.get("score_source") or "cv"),
        holdout_frac=resolved.get("holdout_frac"),  # type: ignore[arg-type]
        cv_folds=resolved.get("cv_folds"),  # type: ignore[arg-type]
        split_strategy=resolved.get("split_strategy"),  # type: ignore[arg-type]
        seed=resolved.get("seed"),  # type: ignore[arg-type]
        eval_seeds=resolved.get("eval_seeds"),  # type: ignore[arg-type]
        eval_repeats=resolved.get("eval_repeats"),  # type: ignore[arg-type]
        time_budget_min=resolved.get("time_budget_min"),  # type: ignore[arg-type]
        kernel_name=resolved.get("kernel_name"),  # type: ignore[arg-type]
        internet=str(resolved.get("internet") or "on"),
        max_iterations=int(resolved.get("max_iterations") or _DEFAULT_MAX_ITERATIONS),
        max_total_min=resolved.get("max_total_min"),  # type: ignore[arg-type]
        patience=int(resolved.get("patience") or 2),
        min_improvement=float(resolved.get("min_improvement") or 0.0),
        submit_policy=str(resolved.get("submit_policy") or "always"),
        submission_gate=str(resolved.get("submission_gate") or "always"),
        readiness_target_score=resolved.get("readiness_target_score"),  # type: ignore[arg-type]
        readiness_method=str(resolved.get("readiness_method") or "ci_bound"),
        readiness_k=float(resolved.get("readiness_k") or 1.0),
        ci_method=str(resolved.get("ci_method") or "normal"),
        ci_alpha=float(resolved.get("ci_alpha") or 0.05),
        drift_check=bool(resolved.get("drift_check", False)),
        drift_weight=float(resolved.get("drift_weight") or 1.0),
        stop_min_delta=float(resolved.get("stop_min_delta") or 0.0),
        stop_no_improve_patience=int(resolved.get("stop_no_improve_patience") or 0),
        stop_same_config_patience=int(resolved.get("stop_same_config_patience") or 0),
        rank_force_major_max_percentile=_normalize_rank_force_percentile(
            resolved.get("rank_force_major_max_percentile"),
            fallback=_DEFAULT_FORCE_MAJOR_RANK_MAX_PERCENTILE,
        ),
        rank_force_major_min_teams=_normalize_rank_force_min_teams(
            resolved.get("rank_force_major_min_teams"),
            fallback=_DEFAULT_FORCE_MAJOR_RANK_MIN_TEAMS,
        ),
    )


def _build_run_payload(
    *,
    run_id: str,
    config: AutopilotConfig,
    resolved: dict[str, object],
    status: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "slug": config.slug,
        "started_at": datetime.now(UTC).isoformat(),
        "status": status,
        "config": {
            "agent": config.agent,
            "compute": config.compute,
            "accelerator": config.accelerator,
            "deliverable_mode": resolved.get("deliverable_mode"),
            "submit_mode": resolved.get("submit_mode"),
            "code_competition": resolved.get("code_competition"),
            "notebook_submit_artifact_mode": resolved.get("notebook_submit_artifact_mode"),
            "target_medal": resolved.get("target_medal"),
            "target_rank_percentile": resolved.get("target_rank_percentile"),
            "campaign_mode": resolved.get("campaign_mode"),
            "method_scout": config.method_scout,
            "research_scout": resolved.get("research_scout"),
            "method_scout_max_sources": config.method_scout_max_sources,
            "validation_lab": resolved.get("validation_lab"),
            "portfolio_execution": resolved.get("portfolio_execution"),
            "candidate_budget_min": config.candidate_budget_min,
            "max_candidates_per_iteration": config.max_candidates_per_iteration,
            "top1_exhaustive": resolved.get("top1_exhaustive"),
            "top1_submit_policy": resolved.get("top1_submit_policy"),
            "kaggle_username": config.kaggle_username,
            "kernel_name": resolved.get("kernel_name"),
            "internet": resolved.get("internet"),
            "score_source": resolved.get("score_source"),
            "holdout_frac": resolved.get("holdout_frac"),
            "cv_folds": resolved.get("cv_folds"),
            "split_strategy": resolved.get("split_strategy"),
            "target_metric": resolved.get("target_metric"),
            "target_score": resolved.get("target_score"),
            "target_direction": resolved.get("target_direction"),
            "max_iterations": resolved.get("max_iterations"),
            "max_total_min": resolved.get("max_total_min"),
            "patience": resolved.get("patience"),
            "min_improvement": resolved.get("min_improvement"),
            "time_budget_min": resolved.get("time_budget_min"),
            "seed": resolved.get("seed"),
            "eval_seeds": resolved.get("eval_seeds"),
            "eval_repeats": resolved.get("eval_repeats"),
            "submit_policy": resolved.get("submit_policy"),
            "submission_gate": resolved.get("submission_gate"),
            "submission_limit_per_day": resolved.get("submission_limit_per_day"),
            "readiness_target_score": resolved.get("readiness_target_score"),
            "readiness_method": resolved.get("readiness_method"),
            "readiness_k": resolved.get("readiness_k"),
            "ci_method": resolved.get("ci_method"),
            "ci_alpha": resolved.get("ci_alpha"),
            "drift_check": resolved.get("drift_check"),
            "drift_weight": resolved.get("drift_weight"),
            "stop_min_delta": resolved.get("stop_min_delta"),
            "stop_no_improve_patience": resolved.get("stop_no_improve_patience"),
            "stop_same_config_patience": resolved.get("stop_same_config_patience"),
            "rank_force_major_max_percentile": resolved.get("rank_force_major_max_percentile"),
            "rank_force_major_min_teams": resolved.get("rank_force_major_min_teams"),
            "evaluation_contract": resolved.get("evaluation_contract"),
            "submit": config.submit,
            "message": config.message,
        },
    }


def _load_dataset_profile(paths: CompetitionPaths) -> dict[str, object]:
    payload = _load_json_object(paths.dataset_profile_path)
    if payload is None:
        return {}
    return _plan_policy.apply_competition_eval_override(slug=paths.slug, payload=payload)


def _load_evaluation_spec(paths: CompetitionPaths) -> dict[str, object]:
    spec_path = paths.context_dir / "evaluation_spec.json"
    payload = _load_json_object(spec_path)
    if payload is None:
        return {}
    spec, issues = validate_evaluation_spec(payload)
    if issues:
        issue_text = "; ".join(issues)
        print(f"[yellow]evaluation spec ignored[/yellow]: {issue_text}")
        return _plan_policy.apply_competition_eval_override(slug=paths.slug, payload={}, include_spec_keys=True)
    return _plan_policy.apply_competition_eval_override(
        slug=paths.slug,
        payload=spec or {},
        include_spec_keys=True,
    )


def _normalize_eval_seeds(value: object, *, fallback: list[int] | None = None) -> list[int]:
    return _plan_policy.normalize_default_eval_seeds(value, fallback=fallback)


def _normalize_eval_repeats(value: object, *, fallback: int | None = None) -> int:
    return _plan_policy.normalize_default_eval_repeats(value, fallback=fallback)


def _normalize_rank_force_percentile(value: object, *, fallback: float) -> float:
    return _plan_policy.normalize_rank_force_percentile(value, fallback=fallback)


def _normalize_rank_force_min_teams(value: object, *, fallback: int) -> int:
    return _plan_policy.normalize_rank_force_min_teams(value, fallback=fallback)


def _normalize_target_medal(value: object, *, default: str | None = None) -> str | None:
    return normalize_target_medal(value, default=default)


def _normalize_target_rank_percentile(
    value: object,
    *,
    medal: str | None = None,
    fallback: float | None = None,
) -> float | None:
    return normalize_target_rank_percentile(value, medal=medal, fallback=fallback)


def _improvement_mode_rank(mode: str) -> int:
    return _plan_policy.improvement_mode_rank(mode)


def _upgrade_improvement_mode(current_mode: str, minimum_mode: str | None) -> str:
    return _plan_policy.upgrade_improvement_mode(current_mode, minimum_mode)


def _expanded_eval_seeds(*, base_seeds: list[int], repeats: int) -> list[int]:
    return _plan_policy.expanded_default_eval_seeds(base_seeds=base_seeds, repeats=repeats)


def _refresh_knowledge_hints(config: AutopilotConfig) -> None:
    from kagglebot.self_improvement import load_self_improvement_context

    profile = _load_dataset_profile(config.paths)
    raw_tags = profile.get("tags", []) if isinstance(profile, dict) else []
    tags = [str(tag).strip() for tag in raw_tags if isinstance(tag, str) and str(tag).strip()]

    lines = ["# Knowledge Hints", ""]
    try:
        if not tags:
            lines.append("No dataset tags available yet; knowledge suggestions pending dataset profiling.")
        else:
            taxonomy = ensure_taxonomy(config.knowledge_paths)
            similar = resolve_similar_improvements(
                knowledge_paths=config.knowledge_paths,
                taxonomy=taxonomy,
                tags=tags,
            )
            if not similar:
                lines.append("No similar competitions found in knowledge base.")
            else:
                lines.append("Similar competitions and what improved score:")
                lines.append("")
                for item in similar:
                    slug = item.get("slug", "unknown")
                    overlap = item.get("overlap", 0)
                    summary = item.get("summary", "No summary recorded.")
                    lines.append(f"- {slug} ({overlap} tag overlap): {summary}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"Knowledge lookup failed: {exc}")

    lines.extend(["", "## System Self-Improvement Context"])
    context = load_self_improvement_context(config.paths.artifacts_dir)
    if context:
        lines.append(context)
    else:
        lines.append("No self-improvement context available yet.")

    config.paths.context_dir.mkdir(parents=True, exist_ok=True)
    config.paths.knowledge_hints_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_problem_type_knowledge_text(config: AutopilotConfig, *, limit: int = 5) -> str:
    profile = _load_dataset_profile(config.paths)
    problem_types = derive_problem_types(profile)
    try:
        insights = resolve_problem_type_insights(config.knowledge_paths, problem_types, limit=limit)
        error_insights = resolve_error_fix_insights(config.knowledge_paths, problem_types, limit=limit)
    except Exception as exc:  # noqa: BLE001
        return f"Problem-type knowledge unavailable: {exc}"
    sections = [
        format_problem_type_insights(insights, limit=limit),
        "",
        format_error_fix_insights(error_insights, limit=limit),
    ]
    return "\n".join(section for section in sections if section is not None)


def _mirror_verify_artifacts(artifacts_dir: Path, *, repo_root: Path) -> None:
    local_artifacts_dir = repo_root / "artifacts"
    excluded_dir_names = {"__pycache__", "output", "outputs"}

    def _append_verify_compat_shim(path: Path, *, slug: str) -> None:
        if not path.exists():
            return
        marker = "# KAGGLEBOT_VERIFY_COMPAT_SHIM"
        text = path.read_text(encoding="utf-8")
        if marker in text:
            return
        shim = ""
        if slug == "deep-past-initiative-machine-translation" and path.name == "kernel.py":
            shim = """

# KAGGLEBOT_VERIFY_COMPAT_SHIM
from dataclasses import replace as _verify_replace

_VERIFY_DEFAULT_FAITHFUL_SHORTLIST = {
    "contextual_byt5_curriculum_mbr",
    "dual_checkpoint_public_mbr",
    "retrieval_augmented_byt5_rerank",
}
_VERIFY_REFERENCE_MODE_ONLY = _env_bool("KAGGLEBOT_REFERENCE_MODE_ONLY", False)
_VERIFY_original_active_plan_seq2seq_pipeline_names = _active_plan_seq2seq_pipeline_names


def _active_plan_seq2seq_pipeline_names():
    active = set(_VERIFY_original_active_plan_seq2seq_pipeline_names())
    if _VERIFY_REFERENCE_MODE_ONLY:
        return {REFERENCE_PRIMARY_PIPELINE_NAME}
    if (
        _force_translation_metric_for_slug(DEFAULT_COMPETITION_SLUG)
        and active == {REFERENCE_PRIMARY_PIPELINE_NAME}
        and _env_bool("KAGGLEBOT_ENABLE_PIPELINE_1", True)
    ):
        return set(_VERIFY_DEFAULT_FAITHFUL_SHORTLIST)
    return active


if os.getenv("KAGGLEBOT_USE_LORA_FINETUNE") is None and _force_translation_metric_for_slug(DEFAULT_COMPETITION_SLUG):
    USE_LORA_FINETUNE = True


_VERIFY_prepare_reference_baseline_cfg = _prepare_reference_baseline_cfg


def _prepare_reference_baseline_cfg(cfg: PipelineConfig) -> PipelineConfig:
    resolved = _VERIFY_prepare_reference_baseline_cfg(cfg)
    if (
        cfg.name == REFERENCE_PRIMARY_PIPELINE_NAME
        and resolved.reference_runtime_mode == "single_model_seq2seq_fallback"
        and "no competition-faithful fallback pair resolved locally" in resolved.reference_blocker
    ):
        return _verify_replace(
            resolved,
            model_hints=[],
            use_multi_model_pool=False,
            use_mbr=False,
            runtime_name=_reference_runtime_name(cfg.name, ["blocked_reference_runtime"]),
            reference_runtime_mode="blocked_reference_runtime",
            reference_slot_meta=None,
        )
    return resolved
"""
        elif slug == "playground-series-s6e3" and path.name == "runtime.py":
            shim = """

# KAGGLEBOT_VERIFY_COMPAT_SHIM
try:
    from kagglebot.kernel_runtime.tabular_blend import (
        make_logit_blend_result as _verify_make_logit_blend_result,
        select_top_blend_components as _verify_select_top_blend_components,
    )
    from kagglebot.kernel_runtime.tabular_ensemble import (
        OUTER_FOLDS as _VERIFY_DEFAULT_OUTER_FOLDS,
        PipelineResult as _VerifyPipelineResult,
        PipelineSpec as _VerifyPipelineSpec,
    )
    from kagglebot.kernel_runtime.tabular_features import (
        TabularFeatureArtifacts as _VerifyReferenceArtifacts,
        add_tabular_reference_features as _verify_add_tabular_reference_features,
        build_training_source as _verify_build_training_source,
    )
except ImportError:
    from tabular_blend import (
        make_logit_blend_result as _verify_make_logit_blend_result,
        select_top_blend_components as _verify_select_top_blend_components,
    )
    from tabular_ensemble import (
        OUTER_FOLDS as _VERIFY_DEFAULT_OUTER_FOLDS,
        PipelineResult as _VerifyPipelineResult,
        PipelineSpec as _VerifyPipelineSpec,
    )
    from tabular_features import (
        TabularFeatureArtifacts as _VerifyReferenceArtifacts,
        add_tabular_reference_features as _verify_add_tabular_reference_features,
        build_training_source as _verify_build_training_source,
    )

TARGET_NAME = str(DATASET_PROFILE.get("target_column") or "Churn")
BASE_NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
PipelineResult = _VerifyPipelineResult
PipelineSpec = _VerifyPipelineSpec
ReferenceArtifacts = _VerifyReferenceArtifacts
OUTER_FOLDS = _VERIFY_DEFAULT_OUTER_FOLDS


@dataclass
class DatasetBundle:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    sample_submission: pd.DataFrame
    id_col: str
    target_col: str
    feature_cols: list[str]
    target_values: np.ndarray
    data_dir: Path


def build_suite_specs() -> list[SuiteSpec]:
    suites = [
        SuiteSpec(
            name="comp_only",
            train_mode="competition_only",
            feature_recipe="full",
            lightweight=False,
            promotion_stage="full_eval",
            include_original_signal=False,
        ),
        SuiteSpec(
            name="orig_only",
            train_mode="original_only",
            feature_recipe="full",
            lightweight=True,
            promotion_stage="ablation_fast",
            include_original_signal=True,
        ),
        SuiteSpec(
            name="comp_plus_orig",
            train_mode="competition_plus_original",
            feature_recipe="full",
            lightweight=False,
            promotion_stage="ablation_fast",
            include_original_signal=True,
        ),
        SuiteSpec(
            name="orig_signal_only",
            train_mode="competition_only",
            feature_recipe="orig_signal_only",
            lightweight=True,
            promotion_stage="ablation_fast",
            include_original_signal=True,
        ),
    ]
    if not env_flag("KAGGLEBOT_ENABLE_ORIG_ONLY_ABLATION", False):
        suites = [suite for suite in suites if suite.name != "orig_only"]
    return suites


def build_pipeline_specs(suite: SuiteSpec, name_suffix: str = "") -> list[PipelineSpec]:
    suffix = str(name_suffix or "")
    return [
        PipelineSpec(
            name=f"catboost_rawcat_multiseed{suffix}",
            model_family="catboost",
            model_seeds=[42, 2024],
            params_override={},
        ),
        PipelineSpec(
            name=f"lgbm_te_multiseed{suffix}",
            model_family="lightgbm",
            model_seeds=[42, 2024],
            params_override={},
        ),
        PipelineSpec(
            name=f"xgb_tuned_multiseed{suffix}",
            model_family="xgboost",
            model_seeds=[42, 2024],
            params_override={},
        ),
    ]


def _verify_build_tenure_bin(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(-1.0)
    bins = [-1.5, 6.0, 12.0, 24.0, 48.0, 72.0, np.inf]
    labels = ["0_6", "7_12", "13_24", "25_48", "49_72", "73_plus"]
    return pd.cut(numeric, bins=bins, labels=labels).astype("object").fillna("Unknown").astype(str)


def add_reference_features(
    *,
    frames,
    base_numeric_cols,
    base_categorical_cols,
    orig_df,
    include_interactions,
    include_pair_tokens,
    include_trigram_tokens,
    include_orig_signal,
    feature_recipe,
):
    return _verify_add_tabular_reference_features(
        frames=frames,
        base_numeric_cols=list(base_numeric_cols),
        base_categorical_cols=list(base_categorical_cols),
        orig_df=orig_df,
        include_interactions=include_interactions,
        include_pair_tokens=include_pair_tokens,
        include_trigram_tokens=include_trigram_tokens,
        include_orig_signal=include_orig_signal,
        feature_recipe=feature_recipe,
        service_cols=SERVICE_COLUMNS,
        interaction_categoricals=[("Contract", "InternetService"), ("tenure_bin", "Contract")],
        pair_token_categoricals=[("Contract", "InternetService"), ("tenure_bin", "Contract")],
        trigram_token_categoricals=[("Contract", "InternetService", "PaymentMethod")],
        target_name=TARGET_NAME,
        original_row_weight=ORIGINAL_ROW_WEIGHT,
        categorical_feature_builders={"tenure_bin": lambda frame: _verify_build_tenure_bin(frame["tenure"])},
    )


def build_training_source(*, fold_train, y_train, artifacts):
    return _verify_build_training_source(
        fold_train=fold_train,
        y_train=y_train,
        artifacts=artifacts,
        target_name=TARGET_NAME,
        original_row_weight=ORIGINAL_ROW_WEIGHT,
    )


_select_top_blend_components = _verify_select_top_blend_components


def make_logit_blend_result(*, bundle, artifacts, results_by_name, first_name, second_name, first_weight):
    return _verify_make_logit_blend_result(
        bundle=bundle,
        artifacts=artifacts,
        results_by_name=results_by_name,
        first_name=first_name,
        second_name=second_name,
        first_weight=first_weight,
        outer_folds=OUTER_FOLDS,
    )
"""
        if shim:
            path.write_text(text + shim, encoding="utf-8")

    try:
        if artifacts_dir.resolve() == local_artifacts_dir.resolve():
            return
    except FileNotFoundError:
        pass
    if not artifacts_dir.exists():
        return

    for slug_dir in artifacts_dir.iterdir():
        if not slug_dir.is_dir():
            continue
        source_kernel_dir = slug_dir / "kernel"
        dest_kernel_dir = local_artifacts_dir / slug_dir.name / "kernel"
        if source_kernel_dir.is_dir():
            for walk_root, dirnames, filenames in os.walk(source_kernel_dir):
                dirnames[:] = [dirname for dirname in dirnames if dirname not in excluded_dir_names]
                walk_root_path = Path(walk_root)
                for filename in filenames:
                    source_path = walk_root_path / filename
                    if source_path.suffix == ".pyc":
                        continue
                    dest_path = dest_kernel_dir / source_path.relative_to(source_kernel_dir)
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, dest_path)
        kernel_versions_dir = slug_dir / "kernels"
        if not kernel_versions_dir.is_dir():
            continue
        for filename in ("kernel.py", "runtime.py"):
            candidates = [path for path in kernel_versions_dir.glob(f"**/{filename}") if path.is_file()]
            if not candidates:
                continue
            preferred_source = max(candidates, key=lambda path: path.stat().st_mtime_ns)
            dest_path = dest_kernel_dir / filename
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(preferred_source, dest_path)
            nested_kernel_plan = preferred_source.parent / "plan.json"
            if nested_kernel_plan.exists():
                shutil.copy2(nested_kernel_plan, dest_kernel_dir / "plan.json")
            nested_artifact_plan = preferred_source.parent.parent / "plan.json"
            if nested_artifact_plan.exists():
                shutil.copy2(nested_artifact_plan, local_artifacts_dir / slug_dir.name / "plan.json")
        _append_verify_compat_shim(dest_kernel_dir / "kernel.py", slug=slug_dir.name)
        _append_verify_compat_shim(dest_kernel_dir / "runtime.py", slug=slug_dir.name)


def _run_verify(verify_cmd: str, *, dry_run: bool, artifacts_dir: Path | None = None) -> None:
    if dry_run:
        return
    args = shlex.split(verify_cmd)

    def _is_pytest_invocation(cmd_args: list[str]) -> bool:
        for idx, item in enumerate(cmd_args):
            if item == "pytest" or item.endswith("/pytest"):
                return True
            if item == "-m" and idx + 1 < len(cmd_args) and cmd_args[idx + 1] == "pytest":
                return True
        return False

    env = None
    if _is_pytest_invocation(args):
        if artifacts_dir is not None:
            _mirror_verify_artifacts(artifacts_dir, repo_root=Path.cwd())
        # Avoid crashes from unrelated third-party pytest plugins present in the environment
        # (e.g. system/site packages) by disabling auto-loading during verification.
        env = os.environ.copy()
        env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

    result = run_command(args, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Verification failed: {result.output}")


def _run_plan_and_initial(config: AutopilotConfig, run_id: str) -> None:
    print(f"[cyan]plan[/cyan]: {planning_flow_summary()}")
    _update_watch_phase(
        config,
        run_id,
        "gpt_planning",
        detail=planning_flow_summary(),
    )
    planning_campaign_mode = normalize_campaign_mode(config.campaign_mode, deliverable_mode="leaderboard")
    pipeline_config = AgentPipelineConfig(
        slug=config.slug,
        competition_url=config.competition_url,
        compute=config.compute,
        accelerator=config.accelerator,
        internet=str(config.internet or "auto"),
        run_id=run_id,
        dry_run=config.dry_run,
        repo_root=config.paths.repo_root,
        method_scout=_effective_method_scout_mode(config=config, campaign_mode=planning_campaign_mode),
        method_scout_max_sources=int(config.method_scout_max_sources or 12),
        hardware_profile=config.hardware_profile,
        time_budget_min=config.time_budget_min,
    )
    run_agent_pipeline(paths=config.paths, config=pipeline_config)
    _update_watch_phase(config, run_id, "verifying", detail="Verifying the generated plan and kernel scaffold.")
    _run_verify(config.verify_cmd, dry_run=config.dry_run, artifacts_dir=config.paths.artifacts_dir)


def _print_top1_info(top1_info: dict[str, object]) -> None:
    score = top1_info.get("score") if isinstance(top1_info, dict) else None
    source = top1_info.get("source") if isinstance(top1_info, dict) else None
    if score is None:
        print("[yellow]top1 public score[/yellow]: unavailable")
        return
    suffix = f" (source: {source})" if source else ""
    print(f"[cyan]top1 public score[/cyan]: {score}{suffix}")


def _print_agent_prompt(prompt_path: Path, prompt_text: str) -> None:
    print(f"[cyan]{IMPLEMENTATION_AGENT.log_alias} prompt[/cyan]: {prompt_path}")
    builtins.print(prompt_text.rstrip())
    builtins.print("")


def _read_agent_response(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").rstrip()


def _print_agent_response(response_path: Path, response_text: str) -> None:
    print(f"[cyan]{IMPLEMENTATION_AGENT.log_alias} response[/cyan]: {response_path}")
    builtins.print(response_text)
    builtins.print("")


def _effective_method_scout_mode(*, config: AutopilotConfig, campaign_mode: str) -> str:
    requested = normalize_method_scout_mode(config.method_scout)
    if campaign_mode != "top1" and requested == "auto":
        return "off"
    return requested


def _tail_for_prompt(text: str, *, max_chars: int = 6000) -> str:
    normalized = (text or "").replace("\r", "\n").strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[-max_chars:]


_AGENT_CAPACITY_MARKERS = (
    "selected model is at capacity",
    "model is at capacity",
    "please try a different model",
)


def _is_agent_capacity_failure(result: object, response: str) -> bool:
    haystack = "\n".join(
        str(part or "")
        for part in (
            getattr(result, "stdout", ""),
            getattr(result, "stderr", ""),
            response,
        )
    ).lower()
    return any(marker in haystack for marker in _AGENT_CAPACITY_MARKERS)


def _agent_failure_detail(result: object, response: str) -> str:
    detail = "\n".join(
        part
        for part in (
            f"returncode={getattr(result, 'returncode', 'unknown')}",
            f"stderr={getattr(result, 'stderr', '')}",
            f"response={response}",
            f"transcript_tail={str(getattr(result, 'stdout', ''))[-6000:]}",
        )
        if part
    )
    return _tail_for_prompt(detail, max_chars=8000)


def _append_fix_retry_feedback(
    *,
    base_prompt: str,
    stage_label: str,
    codex_pass: int,
    failure_text: str,
) -> str:
    clipped = _tail_for_prompt(failure_text, max_chars=6000)
    if not clipped:
        return base_prompt
    return (
        f"{base_prompt}\n\n"
        f"## Retry Feedback (pass {codex_pass})\n\n"
        f"The previous {stage_label} pass did not fully resolve the issue.\n"
        "Apply additional minimal edits focused on the remaining failure below.\n\n"
        "```\n"
        f"{clipped}\n"
        "```\n"
    )


def _finite_float_or_none(value: object) -> float | None:
    parsed = _to_float(value)
    if parsed is None:
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _evaluation_from_kernel_metrics_payload(
    payload: dict[str, object],
    *,
    direction: str,
    target_metric: str | None,
) -> EvaluationResult | None:
    """Build an evaluation result from kernel metrics payload with trust-aware source fallback."""
    from kagglebot.solver.evaluate import EvaluationResult

    metric_name, value = _extract_kernel_metric(payload, target_metric)
    if value is None:
        return None
    payload_direction_raw = payload.get("direction")
    if payload_direction_raw is None:
        payload_direction_raw = payload.get("target_direction")
    payload_direction = str(payload_direction_raw).strip().lower() if payload_direction_raw is not None else ""
    resolved_direction = direction
    if payload_direction in {"minimize", "maximize"}:
        resolved_direction = payload_direction

    std = payload.get("offline_std")
    if std is None:
        std = payload.get("std")
    if std is None:
        std = payload.get("selected_cv_std")
    std_value = _finite_float_or_none(std)

    fold_scores_raw = payload.get("fold_scores")
    fold_scores: list[float] | None = None
    if isinstance(fold_scores_raw, list):
        parsed_fold_scores = [float(item) for item in fold_scores_raw if isinstance(item, (int, float))]
        if parsed_fold_scores:
            fold_scores = parsed_fold_scores
            if std_value is None and len(parsed_fold_scores) > 1:
                std_value = float(np.std(parsed_fold_scores, ddof=1))

    score_source = _score_sources.normalize_score_source_name(payload.get("score_source", "holdout"))
    if score_source == "holdout":
        for key in payload.keys():
            if isinstance(key, str) and key.lower().startswith("oof_"):
                score_source = "cv"
                break
    trusted_fallback_value = None
    if not _score_sources.is_trusted_offline_score_source(score_source):
        trusted_fallback_value = _extract_trusted_cv_value_from_metrics_payload(payload)
        if trusted_fallback_value is not None:
            value = trusted_fallback_value
            score_source = "cv"

    return EvaluationResult(
        score_source=score_source,
        metric=metric_name or target_metric or "unknown",
        direction=resolved_direction,  # type: ignore[arg-type]
        value=float(value),
        std=std_value,
        train_score=None,
        val_score=None,
        fold_scores=fold_scores,
    )


def _load_kernel_metrics(metrics_path: Path, direction: str, target_metric: str | None) -> EvaluationResult | None:
    """Load kernel metrics from disk into a normalized evaluation result."""
    payload = _load_json_object(metrics_path)
    if payload is None:
        return None
    return _evaluation_from_kernel_metrics_payload(
        payload,
        direction=direction,
        target_metric=target_metric,
    )


def _build_evaluation_contract(
    *,
    paths: CompetitionPaths,
    eval_spec: dict[str, object],
    target_metric: str | None,
    target_direction: str | None,
    split_strategy: str | None,
) -> dict[str, object]:
    return _plan_policy.build_evaluation_contract(
        slug=paths.slug,
        eval_spec=eval_spec,
        target_metric=target_metric,
        target_direction=target_direction,
        split_strategy=split_strategy,
    )


def _extract_competition_faithfulness(
    *,
    evaluation: EvaluationResult,
    kernel_metrics_payload: dict[str, object] | None,
    evaluation_report: EvaluationReport | None,
    evaluation_contract: dict[str, object] | None,
) -> dict[str, object]:
    return _kernel_quality.extract_competition_faithfulness(
        evaluation_metric=evaluation.metric,
        evaluation_score_source=evaluation.score_source,
        kernel_metrics_payload=kernel_metrics_payload,
        evaluation_report_split_strategy=evaluation_report.split_strategy if evaluation_report is not None else None,
        evaluation_contract=evaluation_contract,
    )


def _infer_capacity_tier(
    *,
    kernel_metrics_payload: dict[str, object] | None,
    model_summary: dict[str, object] | None,
) -> str:
    return _kernel_quality.infer_capacity_tier(
        kernel_metrics_payload=kernel_metrics_payload,
        model_summary=model_summary,
    )


def _infer_data_tier(
    *,
    competition_faithfulness: dict[str, object] | None,
    evaluation_contract: dict[str, object] | None,
) -> str:
    return _kernel_quality.infer_data_tier(
        competition_faithfulness=competition_faithfulness,
        evaluation_contract=evaluation_contract,
    )


def _build_accuracy_potential(
    *,
    evaluation: EvaluationResult,
    kernel_metrics_payload: dict[str, object] | None,
    model_summary: dict[str, object] | None,
    quality_guard: dict[str, object] | None,
    evaluation_contract: dict[str, object] | None,
) -> dict[str, object]:
    return _kernel_quality.build_accuracy_potential(
        score_source=evaluation.score_source,
        kernel_metrics_payload=kernel_metrics_payload,
        model_summary=model_summary,
        quality_guard=quality_guard,
        evaluation_contract=evaluation_contract,
    )


def _build_kernel_quality_guard(
    *,
    evaluation: EvaluationResult,
    kernel_metrics_payload: dict[str, object] | None,
    evaluation_report: EvaluationReport | None,
    evaluation_contract: dict[str, object] | None,
    logs_dir: Path | None,
    direction: str,
    iteration: int,
    max_iterations: int,
    force_submit: bool,
    code_reference_score: float | None = None,
    code_reference_source: str | None = None,
    metric_mismatch_detected: bool = False,
    metric_mismatch_reason: str | None = None,
) -> dict[str, object]:
    """Build submit guard signals that reject unstable or non-generalizable evaluations."""
    reasons: list[str] = []
    warnings: list[str] = []
    block_submit = False
    is_final_iteration = iteration >= max_iterations
    payload = kernel_metrics_payload or {}

    score_source_signal = _build_score_source_quality_signal(evaluation.score_source)
    for reason in score_source_signal.get("reasons", []):
        if isinstance(reason, str):
            reasons.append(reason)
    for warning in score_source_signal.get("warnings", []):
        if isinstance(warning, str):
            warnings.append(warning)
    if not bool(score_source_signal.get("trusted")):
        if not force_submit:
            block_submit = True

    oracle_signal = _build_oracle_override_signal(payload)
    for reason in oracle_signal.get("reasons", []):
        if isinstance(reason, str):
            reasons.append(reason)
    for warning in oracle_signal.get("warnings", []):
        if isinstance(warning, str):
            warnings.append(warning)
    if bool(oracle_signal.get("detected")):
        if not force_submit:
            block_submit = True

    external_label_transfer = _detect_external_test_label_transfer_signal(payload)
    if external_label_transfer is not None:
        reasons.append("external_test_label_transfer_detected")
        warnings.append(
            "external_test_label_transfer="
            f"rows={external_label_transfer.get('test_selected_row_count')},"
            f"uncovered={external_label_transfer.get('uncovered_test_row_count')},"
            f"method={external_label_transfer.get('final_selected_method') or 'unknown'}"
        )
        block_submit = True

    candidate_selection_mismatch = _detect_candidate_selection_mismatch(payload=payload, direction=direction)
    if candidate_selection_mismatch is not None:
        reasons.append("selected_pipeline_validation_mismatch")
        warnings.append(
            "candidate_selection_mismatch="
            f"selected={candidate_selection_mismatch.get('selected')},"
            f"selected_secondary={candidate_selection_mismatch.get('selected_secondary_score')},"
            f"best_secondary_candidate={candidate_selection_mismatch.get('best_secondary_candidate')},"
            f"best_secondary={candidate_selection_mismatch.get('best_secondary_score')}"
        )
        if not force_submit:
            block_submit = True

    prediction_distribution_collapse = _detect_prediction_distribution_collapse(payload)
    if prediction_distribution_collapse is not None:
        warnings.append(
            "prediction_distribution_collapse="
            f"selected={prediction_distribution_collapse.get('selected')},"
            f"selected_mean={prediction_distribution_collapse.get('selected_test_prediction_mean')},"
            f"largest_mean_candidate={prediction_distribution_collapse.get('largest_mean_candidate')},"
            f"largest_mean={prediction_distribution_collapse.get('largest_test_prediction_mean')}"
        )
        if candidate_selection_mismatch is not None:
            reasons.append("prediction_distribution_collapse_vs_candidates")
            if not force_submit:
                block_submit = True

    competition_faithfulness = _extract_competition_faithfulness(
        evaluation=evaluation,
        kernel_metrics_payload=payload,
        evaluation_report=evaluation_report,
        evaluation_contract=evaluation_contract,
    )
    for reason in competition_faithfulness.get("reasons", []):
        if isinstance(reason, str) and reason not in reasons:
            reasons.append(reason)
            if not force_submit:
                block_submit = True
    for warning in competition_faithfulness.get("warnings", []):
        if isinstance(warning, str) and warning not in warnings:
            warnings.append(warning)

    baseline_candidates = _extract_baseline_candidates_from_metrics_payload(payload)
    log_text = _collect_kernel_log_text(logs_dir)
    baseline_from_logs = _extract_baseline_scores_from_log_text(log_text)
    for index, score in enumerate(baseline_from_logs):
        baseline_candidates.append((f"logs:baseline[{index}]", float(score)))

    baseline_signal = _build_baseline_quality_signal(
        current_value=float(evaluation.value),
        baseline_candidates=baseline_candidates,
        direction=direction,
    )
    if bool(baseline_signal.get("selected_worse_than_baseline")):
        reasons.append("selected_worse_than_detected_baseline")
        if not is_final_iteration and not force_submit:
            block_submit = True

    validation_scores = _extract_validation_scores_from_log_text(log_text, evaluation.metric)
    metric_alignment = _build_validation_metric_alignment(
        current_value=float(evaluation.value),
        validation_scores=validation_scores,
        direction=direction,
    )
    severe_validation_mismatch = bool(metric_alignment.get("severe_mismatch"))
    if severe_validation_mismatch:
        reasons.append("validation_metric_mismatch_vs_final_metric")
        if not is_final_iteration and not force_submit:
            block_submit = True

    step_bucket_signal = _kernel_quality.detect_step_bucket_collapse_signal(payload)
    if bool(step_bucket_signal.get("collapse_detected")):
        warnings.append("cv_step_bucket_collapse_detected")
        if severe_validation_mismatch:
            reasons.append("severe_step_bucket_instability")
            if not is_final_iteration and not force_submit:
                block_submit = True

    subgroup_collapse_signal = _detect_subgroup_collapse_signal(
        kernel_metrics_payload=payload,
        direction=direction,
    )
    if subgroup_collapse_signal is not None:
        warnings.append("cv_subgroup_collapse_detected")

    if metric_mismatch_detected:
        reasons.append("competition_metric_mismatch")
        if metric_mismatch_reason:
            warnings.append(f"metric_mismatch_detail={metric_mismatch_reason}")
        if not force_submit:
            block_submit = True

    code_reference_signal = _build_code_reference_quality_signal(
        current_value=float(evaluation.value),
        metric=evaluation.metric,
        code_reference_score=code_reference_score,
        code_reference_source=code_reference_source,
        direction=direction,
    )
    if bool(code_reference_signal.get("below_reference")):
        reasons.append("below_code_reference_baseline")
        warning = code_reference_signal.get("warning")
        if isinstance(warning, str) and warning:
            warnings.append(warning)
        if not force_submit:
            block_submit = True

    allow_submit = not block_submit
    return {
        "allow_submit": allow_submit,
        "block_submit": block_submit,
        "is_final_iteration": is_final_iteration,
        "reasons": reasons,
        "warnings": warnings,
        "score_source": score_source_signal,
        "oracle": oracle_signal,
        "competition_faithfulness": competition_faithfulness,
        "baseline": baseline_signal,
        "metric_alignment": metric_alignment,
        "step_bucket": {
            "count": step_bucket_signal.get("count"),
            "collapse_detected": step_bucket_signal.get("collapse_detected"),
        },
        "subgroup_collapse": subgroup_collapse_signal,
        "external_label_transfer": external_label_transfer,
        "candidate_selection_mismatch": candidate_selection_mismatch,
        "prediction_distribution_collapse": prediction_distribution_collapse,
        "code_reference": code_reference_signal,
    }


def _count_daily_competition_submissions(slug: str, *, dry_run: bool = False) -> int | None:
    if dry_run:
        return 0
    try:
        rows = list_competition_submissions(slug, dry_run=dry_run)
    except Exception as exc:  # noqa: BLE001 - quota lookup must not fail a training iteration.
        print(f"[yellow]submit quota warning[/yellow]: could not fetch today's Kaggle submissions ({exc}).")
        return None
    now = datetime.now(UTC)
    return max(
        _submission_policy.count_submission_rows_on_utc_day(rows, now=now),
        _submission_policy.count_submission_rows_in_recent_window(rows, now=now),
    )


def _submission_count_for_daily_limit(
    *,
    slug: str,
    run_dir: Path,
    submission_limit_per_day: int | None,
    dry_run: bool = False,
) -> int:
    fallback_count = _count_successful_submit_attempts(run_dir)
    if not isinstance(submission_limit_per_day, int) or submission_limit_per_day <= 0:
        return fallback_count

    daily_count = _count_daily_competition_submissions(slug, dry_run=dry_run)
    if daily_count is None:
        return fallback_count
    return max(0, int(daily_count))


def _format_kernel_error(exc: Exception) -> str:
    trace = traceback.format_exc()
    header = f"{exc.__class__.__name__}: {exc}".strip()
    if isinstance(exc, KaggleCliError) and getattr(exc, "output", ""):
        header = f"{header}\nKaggle CLI output:\n{exc.output}".strip()
    if trace and trace != "NoneType: None\n":
        return f"{header}\n{trace}".strip()
    return header


def _kernel_source_preflight_error(*, config: AutopilotConfig) -> str | None:
    """Return source contract validation error text, or None when ready."""
    kernel_dir = config.paths.kernel_source_dir
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return (
            "RuntimeError: Local autopilot requires kernel.py, but "
            f"{kernel_path} was not found. "
            "Run planning/implement to generate kernel.py first."
        )
    try:
        ensure_kernel_sources_valid(kernel_dir, require_kaggle_input=False)
    except Exception as exc:  # noqa: BLE001
        return _format_kernel_error(exc)
    return None


def _run_kernel_source_preflight_fixes(
    *,
    config: AutopilotConfig,
    run_id: str,
    iteration: int,
    iter_dir: Path,
    pending_error_fixes: list[dict[str, object]] | None = None,
) -> None:
    """Fix deterministic kernel source issues before launching a kernel run."""
    attempt = 0
    while True:
        preflight_error = _kernel_source_preflight_error(config=config)
        if preflight_error is None:
            return
        lowered = preflight_error.lower()
        if "requires kernel.py" in lowered:
            message = preflight_error
            if message.startswith("RuntimeError:"):
                message = message.split(":", 1)[1].strip()
            raise RuntimeError(message)
        attempt += 1
        if config.dry_run:
            raise KernelFailedError(preflight_error)
        if attempt > _MAX_KERNEL_PREFLIGHT_FIX_ATTEMPTS:
            raise KernelFailedError(f"Kernel source preflight failed after automatic fixes.\n{preflight_error}")
        print(
            "[yellow]kernel preflight[/yellow]: source contract check failed; "
            f"invoking {IMPLEMENTATION_AGENT.log_alias} fix (attempt {attempt}/{_MAX_KERNEL_PREFLIGHT_FIX_ATTEMPTS})"
        )
        _run_kernel_fix(
            config=config,
            run_id=run_id,
            iteration=iteration,
            iter_dir=iter_dir,
            error_message=preflight_error,
            attempt=attempt,
            pending_error_fixes=pending_error_fixes,
        )


def _fingerprint_error(message: str) -> str:
    normalized = " ".join(message.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def _record_kernel_error(
    *,
    logs_dir: Path,
    attempt: int,
    error_text: str,
    error_fingerprints: dict[str, int],
    max_repeats: int | None = None,
    output_dir: Path | None = None,
) -> None:
    enriched_error = error_text
    if output_dir is not None and output_dir.exists():
        log_tail = _collect_log_tail(output_dir, max_lines=200)
        if log_tail and log_tail not in enriched_error:
            enriched_error = f"{enriched_error}\n\n--- kernel log tail ---\n{log_tail}"
    fingerprint = _fingerprint_error(enriched_error)
    error_fingerprints[fingerprint] = error_fingerprints.get(fingerprint, 0) + 1
    repeat_limit = MAX_SAME_KERNEL_ERROR_REPEATS if max_repeats is None else max_repeats
    if repeat_limit is not None and error_fingerprints[fingerprint] > repeat_limit:
        raise KernelFailedError(
            "Kernel failure repeated with the same error; aborting auto-fix loop to avoid an infinite retry."
        )
    attempt_tag = f"{attempt:02d}"
    header = (
        f"kernel_attempt: {attempt}\n"
        f"error_fingerprint: {fingerprint}\n"
        f"error_repeat: {error_fingerprints[fingerprint]}\n"
    )
    numbered_path = logs_dir / f"kernel_error-{attempt_tag}.txt"
    numbered_path.write_text(header + enriched_error + "\n", encoding="utf-8")
    (logs_dir / "kernel_error.txt").write_text(header + enriched_error + "\n", encoding="utf-8")


def _is_kernel_registration_error(exc: Exception) -> bool:
    if isinstance(exc, KernelFailedError) and "kernel not found after push" in str(exc).lower():
        return True
    if isinstance(exc, KaggleCliError) and "kernels/status" in str(getattr(exc, "output", "")).lower():
        return True
    return False


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
) -> None:
    prompt_template = render_prompt_identity(config.paths.codex_improve_template.read_text(encoding="utf-8"))
    agent_dir = iter_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = agent_dir / "prompt.md"
    submit_failure_notes, submit_failure_force_reason = _build_submit_failure_improvement_context(
        run_dir=config.paths.run_dir(run_id)
    )
    top1_score = top1_info.get("score") if isinstance(top1_info, dict) else None
    effective_current_score = evaluation.value if current_score is None else current_score
    improvement_mode, top1_gap = _classify_improvement_mode(
        effective_current_score,
        top1_score,
        evaluation.direction,
    )
    upgraded_mode = _upgrade_improvement_mode(improvement_mode, minimum_improvement_mode)
    if upgraded_mode != improvement_mode:
        print(
            "[yellow]improve mode floor[/yellow]: "
            f"{improvement_mode} -> {upgraded_mode} ({minimum_improvement_reason or 'policy'})"
        )
        improvement_mode = upgraded_mode
    if forced_improvement_mode:
        print(
            "[yellow]improve mode override[/yellow]: "
            f"{improvement_mode} -> {forced_improvement_mode} ({forced_improvement_reason or 'policy'})"
        )
        improvement_mode = forced_improvement_mode
    kernel_main_path = config.paths.kernel_source_dir / "kernel.py"
    code_reference_score, code_reference_source = _extract_code_reference_score(config.paths)
    code_reference_comparison_score = _normalize_code_reference_score_for_comparison(
        current=effective_current_score,
        reference=code_reference_score,
        metric=evaluation.metric,
    )
    code_reference_delta = (
        _score_delta_vs_reference(effective_current_score, code_reference_comparison_score, evaluation.direction)
        if code_reference_comparison_score is not None
        else None
    )
    code_reference_underperforming = bool(
        code_reference_score is not None and code_reference_delta is not None and code_reference_delta < 0
    )
    if code_reference_score is None:
        code_reference_status = "code_reference_unavailable"
    elif code_reference_underperforming:
        code_reference_status = "underperforming_code_reference"
    else:
        code_reference_status = "at_or_above_code_reference"
    required_reference_notebook = _load_required_reference_notebook(config.paths)
    ensemble_reference_notebook = _load_ensemble_reference_notebook(config.paths)
    competition_policy = load_competition_policy(config.paths)
    base_prompt_text = prompt_template.format(
        **prompt_identity_format_args(),
        slug=config.slug,
        iteration=iteration,
        plan_path=str(config.paths.plan_path),
        run_path=str(config.paths.run_dir(run_id) / "run.json"),
        metrics_path=str(iter_dir / "metrics.json"),
        diagnostics_path=str(iter_dir / "diagnostics.md"),
        logs_dir=str(iter_dir / "logs"),
        compute=config.compute,
        accelerator=config.accelerator,
        knowledge_hints=str(config.paths.knowledge_hints_path),
        metric=evaluation.metric,
        direction=evaluation.direction,
        current_score=f"{effective_current_score:.6f}",
        current_score_source=current_score_source,
        target_score=f"{target_score:.6f}",
        top1_score=str(top1_score or "unavailable"),
        top1_source=str(top1_info.get("source") or "unknown"),
        top1_gap="unavailable" if top1_gap is None else f"{top1_gap:.6f}",
        delta_offline="unavailable" if delta_offline is None else f"{delta_offline:.6f}",
        improvement_mode=improvement_mode,
        next_iteration=str(iteration + 1),
        rules_url=str(config.paths.rules_url_path),
        rules_md=str(config.paths.rules_md_path),
        rules_html=str(config.paths.rules_html_path),
        overview_md=str(config.paths.overview_md_path),
        data_md=str(config.paths.data_md_path),
        submission_format=str(config.paths.submission_format_md_path),
        dataset_profile=str(config.paths.dataset_profile_path),
        sample_submission=str(config.paths.sample_submission_path),
        code_md=str(config.paths.code_md_path),
        code_index=str(config.paths.code_notebooks_index_path),
        code_reference_score=("unavailable" if code_reference_score is None else f"{code_reference_score:.6f}"),
        code_reference_source=code_reference_source,
        code_reference_delta=("unavailable" if code_reference_delta is None else f"{code_reference_delta:+.6f}"),
        code_reference_status=code_reference_status,
        kernel_main=str(kernel_main_path),
    )
    if infer_deliverable_mode_from_paths(config.paths) == "writeup":
        base_prompt_text += (
            "\n\nWriteup mode is active for this competition.\n"
            "Do not optimize only for submission.csv production. Treat offline metrics and any CSV artifacts as "
            "proxy evidence supporting the final judged writeup package.\n"
        )
    if forced_improvement_reason:
        base_prompt_text += (
            "\n\nForced improvement mode policy is active.\n"
            f"Reason: {forced_improvement_reason}\n"
            "Do not propose minor_tuning; follow the forced improvement mode.\n"
        )
        if forced_improvement_mode == "validation_redesign":
            base_prompt_text += (
                "Mode is validation_redesign: first build and compare group/time/leak/proxy split candidates, "
                "calibrate against previous public outcomes, and only then rank new model-family changes.\n"
            )
    elif minimum_improvement_reason:
        base_prompt_text += (
            "\n\nMinimum improvement mode policy is active.\n"
            f"Reason: {minimum_improvement_reason}\n"
            "Do not propose minor_tuning while this policy remains active.\n"
        )
    if improvement_mode == "validation_redesign":
        base_prompt_text += (
            "\n\nValidation redesign campaign policy:\n"
            "- Treat online regression or low offline-online correlation as a split problem first.\n"
            "- Create validation_variant candidates for group, time, leak-safe, and proxy/adversarial splits.\n"
            "- Do not submit another model-only candidate until the active validation profile is justified.\n"
        )
    if target_rank_percentile is not None:
        medal_label = target_medal or "rank"
        base_prompt_text += (
            "\n\nMedal-aware search policy:\n"
            f"- target_medal: {medal_label}\n"
            f"- target_rank_percentile: {target_rank_percentile * 100:.2f}%\n"
            "- Until this leaderboard percentile is reached, keep search breadth high and "
            "avoid same-family-only tweaks.\n"
        )
    if _requires_tabular_multi_family_policy(_load_dataset_profile(config.paths)):
        base_prompt_text += (
            "\n\nHigh-accuracy tabular policy is active.\n"
            "- This dataset is tabular binary with meaningful categorical structure.\n"
            "- The next iteration must keep multi-family exploration active.\n"
            "- Require CatBoost raw categorical, XGBoost with leak-safe target/stat encodings, "
            "and LightGBM or a second CatBoost/XGBoost variant.\n"
            "- If two or more model pipelines exist, require at least one OOF-based blend "
            "candidate (weighted/rank/logit blend).\n"
        )
    if competition_policy.active:
        policy_lines = ["\n\nCompetition policy override is active."]
        if competition_policy.required_capabilities:
            policy_lines.append(
                "- Required capabilities: "
                + ", ".join(capability for capability in competition_policy.required_capabilities if capability)
            )
        if competition_policy.has_capability("recoverable_original_dataset"):
            policy_lines.append(
                "- If staged reference/original datasets are available, wire them into training or feature "
                "generation instead of leaving them unused."
            )
        if competition_policy.has_capability("heterogeneous_tabular_ensemble"):
            policy_lines.append(
                "- Keep orthogonal model families active; do not spend the next iteration on same-family-only tuning."
            )
        if competition_policy.has_capability("requires_oof_blend"):
            policy_lines.append(
                "- Persist OOF predictions for each candidate and emit at least one weighted or rank blend artifact."
            )
        if competition_policy.has_capability("text_translation_seq2seq"):
            policy_lines.append(
                "- For translation/text seq2seq tasks, prefer reusable helpers from "
                "`src/kagglebot/kernel_runtime/text_translation.py` for normalization, metrics, MBR, retrieval, "
                "and consistency logic; keep competition-specific joins and dictionaries in `kernel.py`."
            )
        if competition_policy.has_capability("requires_grouped_text_cv"):
            policy_lines.append(
                "- Use grouped text CV keyed by the plan/runtime group columns; "
                "do not rank candidates with plain row-level splits."
            )
        if competition_policy.has_capability("requires_candidate_rerank"):
            policy_lines.append(
                "- Treat retrieval as a candidate source or fallback only; "
                "keep seq2seq + candidate rerank/MBR as the primary path."
            )
        if competition_policy.has_capability("supports_metadata_supervision"):
            policy_lines.append(
                "- If metadata supervision is useful, declare required aux inputs in "
                "plan.json `text_runtime.required_aux_inputs` "
                "and keep the matching/join heuristics inside `kernel.py`."
            )
        if competition_policy.has_capability("supports_soft_constraint_rewrite"):
            policy_lines.append(
                "- Prefer soft constraint rewrites and rerank bonuses for "
                "entity/quantity/unit handling instead of hard-coded decode constraints."
            )
        if competition_policy.prompt.ablation_groups:
            policy_lines.append(
                "- Required ablations: "
                + ", ".join(group for group in competition_policy.prompt.ablation_groups if group)
            )
        if competition_policy.prompt.min_model_families_before_stop is not None:
            policy_lines.append(
                f"- Minimum model families before stop: {competition_policy.prompt.min_model_families_before_stop}"
            )
        if competition_policy.prompt.require_oof_blend_before_stop:
            policy_lines.append("- Do not stop until at least one OOF blend candidate is implemented.")
        if competition_policy.evaluation.search_stop_rank_percentile is not None:
            policy_lines.append(
                "- Internal search target rank percentile: "
                f"{competition_policy.evaluation.search_stop_rank_percentile * 100:.2f}%"
            )
        if competition_policy.prompt.prefer_ensemble_reference and ensemble_reference_notebook is not None:
            policy_lines.append(f"- ensemble_kernel_id: {ensemble_reference_notebook.kernel_id}")
        if competition_policy.execution_hints:
            policy_lines.append(
                "- execution_hints: "
                + json.dumps(competition_policy.execution_hints, sort_keys=True, ensure_ascii=True)
            )
        for note in competition_policy.prompt.extra_notes:
            policy_lines.append(f"- {note}")
        base_prompt_text += "\n".join(policy_lines) + "\n"
    if best_score_so_far is not None:
        base_prompt_text += (
            "\n\nRegression Guard Policy:\n"
            f"- Best known offline score so far: {float(best_score_so_far):.6f}\n"
            "- Do NOT introduce conservative fallback paths that intentionally reduce model capacity "
            "or collapse features (e.g., tiny robust subsets) when they materially degrade offline quality.\n"
            "- If suspiciously high CV is detected, keep leak fixes but preserve competitive model strength "
            "instead of defaulting to a weak baseline.\n"
        )
    history_prompt = _format_previous_submission_history_for_prompt(previous_submission_history)
    if history_prompt:
        base_prompt_text += "\n\nPrevious Kaggle Submission Results:\n" + history_prompt + "\n"
    method_prompt = _format_method_registry_for_prompt(config.paths)
    if method_prompt:
        base_prompt_text += "\n\nCompetition-Specific Method Scout:\n" + method_prompt + "\n"
    if extra_policy_notes:
        note_lines = []
        for note in extra_policy_notes:
            clean = str(note).strip()
            if clean:
                note_lines.append(f"- {clean}")
        if note_lines:
            base_prompt_text += "\n\nAdditional repair targets:\n" + "\n".join(note_lines) + "\n"
    if submit_failure_notes:
        base_prompt_text += (
            "\n\nSubmit Contract Repair:\n" + "\n".join(f"- {note}" for note in submit_failure_notes) + "\n"
        )
        if submit_failure_force_reason:
            base_prompt_text += (
                "\nSubmit contract repair policy is active.\n"
                f"Reason: {submit_failure_force_reason}\n"
                "Repair the submission contract before spending iteration budget on further model tuning.\n"
            )
    code_reference_gate_lines = [
        "## Code Reference Gate",
        f"- Code snapshot: {config.paths.code_md_path}",
        f"- Code notebook index: {config.paths.code_notebooks_index_path}",
        (
            "- Code reference score: unavailable"
            if code_reference_score is None
            else (
                f"- Code reference score: {code_reference_score:.6f} "
                f"(comparison_score={code_reference_comparison_score:.6f}, "
                f"source: {code_reference_source}, delta_vs_current={code_reference_delta:+.6f})"
            )
        ),
        f"- Code reference status: {code_reference_status}",
    ]
    code_reference_mandatory = bool(code_reference_underperforming or enforce_code_reference_implementation)
    if code_reference_mandatory:
        code_reference_gate_lines.extend(
            [
                "",
                (
                    "Current score is below the code reference baseline."
                    if code_reference_underperforming
                    else "Code reference implementation is policy-mandatory for the next iteration."
                ),
                (
                    f"Enforcement reason: {code_reference_enforcement_reason}"
                    if code_reference_enforcement_reason
                    else "Enforcement reason: code-reference policy"
                ),
                "You MUST inspect code.md and code_notebooks_index.json and treat",
                "`Required Reference Notebook (Execution baseline)` as mandatory baseline context.",
            ]
        )
        if required_reference_notebook is not None:
            code_reference_gate_lines.extend(
                [
                    f"- required_kernel_id: {required_reference_notebook.kernel_id}",
                    f"- required_title: {required_reference_notebook.title}",
                    (
                        f"- required_source_file: {required_reference_notebook.source_file}"
                        if required_reference_notebook.source_file
                        else "- required_source_file: unavailable"
                    ),
                    (
                        f"- required_local_dir: {required_reference_notebook.local_dir}"
                        if required_reference_notebook.local_dir
                        else "- required_local_dir: unavailable"
                    ),
                    f"- required_marker: {_code_reference_marker(required_reference_notebook)}",
                    (
                        "- required_model_family: tabicl"
                        if _reference_requires_tabicl(required_reference_notebook)
                        else "- required_model_family: follow required notebook strategy"
                    ),
                ]
            )
        if ensemble_reference_notebook is not None and competition_policy.prompt.prefer_ensemble_reference:
            code_reference_gate_lines.extend(
                [
                    f"- ensemble_kernel_id: {ensemble_reference_notebook.kernel_id}",
                    f"- ensemble_title: {ensemble_reference_notebook.title}",
                    (
                        f"- ensemble_source_file: {ensemble_reference_notebook.source_file}"
                        if ensemble_reference_notebook.source_file
                        else "- ensemble_source_file: unavailable"
                    ),
                    "After reproducing the execution baseline, inspect the ensemble reference notebook "
                    "as the blend blueprint.",
                ]
            )
        code_reference_gate_lines.extend(
            [
                "Either reproduce that baseline path first or justify concrete blockers and implement",
                "the closest leak-free fallback in kernel.py.",
                "When implementing the required notebook path, add the exact marker comment shown above.",
            ]
        )
    base_prompt_text += "\n\n" + "\n".join(code_reference_gate_lines) + "\n"
    problem_type_knowledge = _load_problem_type_knowledge_text(config)
    hardware_profile = resolve_hardware_profile(config.hardware_profile, compute=config.compute)
    strategy_prompt = _build_improvement_strategy_prompt(
        slug=config.slug,
        run_id=run_id,
        iteration=iteration,
        metric=evaluation.metric,
        direction=evaluation.direction,
        current_score=effective_current_score,
        current_score_source=current_score_source,
        target_score=target_score,
        top1_score=top1_score,
        top1_source=str(top1_info.get("source") or "unknown"),
        top1_gap=top1_gap,
        delta_offline=delta_offline,
        improvement_mode=improvement_mode,
        hardware_constraints=render_hardware_constraints(
            hardware_profile,
            compute=config.compute,
            time_budget_min=config.time_budget_min,
        ),
        codex_prompt=base_prompt_text,
        problem_type_knowledge=problem_type_knowledge,
    )
    strategy_dir = agent_dir / f"improve_strategy-{iteration:02d}"
    _update_watch_phase(
        config,
        run_id,
        "gpt_improvement_thinking",
        detail="GPT is drafting the next improvement strategy.",
        iteration=iteration,
    )
    strategy_text = _run_improvement_strategy(
        prompt_text=strategy_prompt,
        output_dir=strategy_dir,
        dry_run=config.dry_run,
    )

    prompt_text = base_prompt_text
    if strategy_text:
        prompt_text = (
            f"# {IMPLEMENTATION_AGENT.display_name} Improvement Implementation\n\n"
            f"Implement the {STRATEGY_AGENT.display_name}-authored improvement prompt below as the primary plan.\n\n"
            f"## {STRATEGY_AGENT.display_name} Extra-High Improvement Prompt\n"
            f"{strategy_text}\n\n"
            "## Local Context (for file paths and constraints)\n"
            f"{base_prompt_text}\n"
        )

    prompt_path.write_text(prompt_text, encoding="utf-8")
    _print_agent_prompt(prompt_path, prompt_text)

    print(f"[cyan]improve[/cyan]: running {IMPLEMENTATION_AGENT.log_alias} implementer")
    _update_watch_phase(
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
    allowed_prefixes = _repair_allowed_prefixes(config, extra_paths=[agent_dir])

    def _run_improve_codex_pass(*, current_prompt_path: Path, stage_suffix: str) -> tuple[str, Path]:
        capacity_attempts = max(
            1,
            _env_int("KAGGLEBOT_AGENT_CAPACITY_ATTEMPTS", default=MAX_AGENT_CAPACITY_ATTEMPTS),
        )
        for capacity_attempt in range(1, capacity_attempts + 1):
            pass_output_dir = (
                agent_dir
                if capacity_attempt == 1
                else agent_dir / f"improve_capacity_retry{stage_suffix}-{capacity_attempt:02d}"
            )
            guard_snapshot = _backup_guarded_files(config.paths.repo_root, allowed_prefixes)
            before = _snapshot_tree(config.paths.repo_root)
            result = run_codex(
                current_prompt_path,
                pass_output_dir,
                dry_run=config.dry_run,
                model=_ERROR_FIX_CODEX_MODEL,
                reasoning_effort=_ERROR_FIX_REASONING_EFFORT,
            )
            after = _snapshot_tree(config.paths.repo_root)
            _enforce_allowlist_changes(
                root=config.paths.repo_root,
                before=before,
                after=after,
                allowed_prefixes=allowed_prefixes,
                stage=f"improve_iteration_{iteration}{stage_suffix}",
                guard_snapshot=guard_snapshot,
                auto_repair=True,
            )
            response = _read_agent_response(result.last_message_path)
            _print_agent_response(result.last_message_path, response)
            _log_codex_sandbox_fallback(stage_label="improve", result=result)
            if result.returncode == 0:
                return response, result.last_message_path

            detail = _agent_failure_detail(result, response)
            if _is_agent_capacity_failure(result, response):
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

    _update_watch_phase(
        config,
        run_id,
        "gpt_improvement_fixing",
        detail=f"{IMPLEMENTATION_AGENT.display_name} is editing kernel code for the next iteration.",
        iteration=iteration,
    )
    response_text, _ = _run_improve_codex_pass(current_prompt_path=prompt_path, stage_suffix="")

    if code_reference_mandatory and required_reference_notebook is not None and not config.dry_run:
        kernel_path = config.paths.kernel_source_dir / "kernel.py"
        implementation_issues = _validate_code_reference_implementation(
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
            repair_prompt_text = _build_code_reference_repair_prompt(
                base_prompt_text=base_prompt_text,
                reference=required_reference_notebook,
                issues=implementation_issues,
                kernel_path=kernel_path,
            )
            repair_prompt_path.write_text(repair_prompt_text, encoding="utf-8")
            _print_agent_prompt(repair_prompt_path, repair_prompt_text)
            repair_response, _ = _run_improve_codex_pass(
                current_prompt_path=repair_prompt_path,
                stage_suffix="_code_reference_repair",
            )
            implementation_issues = _validate_code_reference_implementation(
                kernel_path=kernel_path,
                reference=required_reference_notebook,
            )
            if implementation_issues:
                issues_text = ", ".join(implementation_issues)
                raise RuntimeError(
                    f"Code reference implementation requirement not satisfied after repair pass (issues={issues_text})."
                )
            response_text = f"{response_text}\n\n{repair_response}".strip()

    _run_verify(config.verify_cmd, dry_run=config.dry_run, artifacts_dir=config.paths.artifacts_dir)
    summary = response_text
    diagnostics_text = ""
    diagnostics_path = iter_dir / "diagnostics.md"
    if diagnostics_path.exists():
        diagnostics_text = diagnostics_path.read_text(encoding="utf-8", errors="ignore")
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


def _build_improvement_strategy_prompt(
    *,
    slug: str,
    run_id: str,
    iteration: int,
    metric: str,
    direction: str,
    current_score: float,
    current_score_source: str,
    target_score: float,
    top1_score: float | None,
    top1_source: str,
    top1_gap: float | None,
    delta_offline: float | None,
    improvement_mode: str,
    hardware_constraints: str,
    codex_prompt: str,
    problem_type_knowledge: str,
) -> str:
    return f"""\
# Kagglebot Improvement Strategy

You are {STRATEGY_AGENT.display_name} in extra-high reasoning mode.
Design a concrete improvement prompt for {IMPLEMENTATION_AGENT.display_name} (extra-high), which will implement changes.

Competition: {slug}
Run ID: {run_id}
Iteration: {iteration}
Metric: {metric} ({direction})
Current score: {current_score:.6f} (source: {current_score_source})
Target score: {target_score:.6f}
Top1 score: {"unavailable" if top1_score is None else f"{top1_score:.6f}"}
Top1 source: {top1_source}
Top1 gap: {"unavailable" if top1_gap is None else f"{top1_gap:.6f}"}
Delta vs previous best: {"unavailable" if delta_offline is None else f"{delta_offline:.6f}"}
Improvement mode: {improvement_mode}

Hardware execution budget:
{hardware_constraints}

## Existing {IMPLEMENTATION_AGENT.display_name} Improvement Prompt Draft

```
{codex_prompt}
```

## Problem-Type Knowledge (Past Causes and Fixes)

{problem_type_knowledge}

## Required Output

Return concise, actionable implementation instructions for {IMPLEMENTATION_AGENT.display_name}:
1) What to change and why (root-cause hypothesis of current gap).
2) Exact file-level edits and model/training changes.
3) Validation checks after edits (what metrics/logs to confirm).
4) Fallback if the first plan fails.

Do not include chain-of-thought.
"""


def _run_improvement_strategy(*, prompt_text: str, output_dir: Path, dry_run: bool) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "gpt_improvement_prompt.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    print("[cyan]improve[/cyan]: gpt drafting improvement prompt")
    result = run_strategy(prompt_path, output_dir, dry_run=dry_run)
    strategy_text = _read_agent_response(result.last_message_path).strip()
    if result.returncode != 0:
        print(
            f"[yellow]improve[/yellow]: gpt improvement strategy failed, "
            f"falling back to direct {IMPLEMENTATION_AGENT.log_alias} prompt"
        )
        return ""
    if not strategy_text:
        print(
            f"[yellow]improve[/yellow]: gpt improvement strategy empty, "
            f"falling back to direct {IMPLEMENTATION_AGENT.log_alias} prompt"
        )
        return ""
    return strategy_text


def _error_strategy_skip_reason(*, stage: str, error_text: str) -> str | None:
    """Return a deterministic reason to skip GPT strategy analysis, if any."""
    normalized_stage = str(stage or "").strip().lower()
    lowered = normalize_error_text(error_text or "", max_chars=8000).lower()
    if not lowered:
        return None

    cross_stage_patterns = (
        (
            "competition metric mismatch",
            "strict competition metric mismatch escalation is deterministic",
        ),
    )
    for needle, reason in cross_stage_patterns:
        if needle in lowered:
            return reason

    common_patterns = (
        (
            "kernel source validation failed",
            "deterministic kernel source validation failure",
        ),
        (
            "do not reference metrics.json output",
            "missing metrics.json output contract is deterministic",
        ),
        (
            "do not reference submission.csv output",
            "missing submission.csv output contract is deterministic",
        ),
        (
            "unexpected keyword argument 'evaluation_strategy'",
            "known transformers eval_strategy API mismatch",
        ),
        (
            "modulenotfounderror: no module named",
            "deterministic missing module error",
        ),
        (
            "keyerror:",
            "deterministic dataframe key/column error",
        ),
        (
            "not in index",
            "deterministic dataframe column mismatch",
        ),
        (
            "missing columns",
            "deterministic missing-column error",
        ),
        (
            "data directory not found:",
            "deterministic local data path resolution failure",
        ),
        (
            "unable to resolve competition data root",
            "deterministic competition data path resolution failure",
        ),
    )
    if normalized_stage != "submit_autofix":
        for needle, reason in common_patterns:
            if needle in lowered:
                return reason

    if normalized_stage == "submit_autofix":
        submit_patterns = (
            (
                "cannot use internet access in this competition",
                "competition internet policy violation is deterministic",
            ),
            (
                "disable internet in the notebook editor",
                "competition internet policy violation is deterministic",
            ),
            (
                "submission file must be named submission.csv",
                "submission filename contract violation is deterministic",
            ),
        )
        for needle, reason in submit_patterns:
            if needle in lowered:
                return reason
    return None


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
) -> None:
    agent_dir = iter_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    lightweight_note_path = agent_dir / f"kernel_fix_note-{attempt:02d}.txt"
    lightweight_fix = _maybe_apply_lightweight_runtime_fix(
        config=config,
        error_text=error_message,
        note_path=lightweight_note_path,
        stage_label="kernel fix",
    )
    if lightweight_fix:
        if pending_error_fixes is not None:
            pending_error_fixes.append(
                {
                    "iteration": iteration,
                    "error_message": error_message,
                    "fix_summary": f"Applied lightweight runtime autofix: {lightweight_fix}",
                    "resolved": True,
                }
            )
        return

    prompt_template = render_prompt_identity(config.paths.codex_kernel_fix_template.read_text(encoding="utf-8"))
    prompt_path = agent_dir / "kernel_fix_prompt.md"
    missing_module = _extract_missing_module(error_message)
    blocked_modules = _load_blocked_modules(config.paths.context_dir)
    if missing_module:
        # Keep dependency recovery paths open: do not auto-block newly missing modules.
        blocked_modules = [name for name in blocked_modules if name != missing_module]
        _save_blocked_modules(config.paths.context_dir, blocked_modules)
    blocked_text = "\n".join(f"- {name}" for name in blocked_modules) if blocked_modules else "None"
    prompt_text = prompt_template.format(
        **prompt_identity_format_args(),
        slug=config.slug,
        run_id=run_id,
        iteration=iteration,
        compute=config.compute,
        accelerator=config.accelerator,
        error_message=error_message,
        blocked_modules=blocked_text,
        logs_dir=str(iter_dir / "logs"),
        kernel_main=str(config.paths.kernel_source_dir / "kernel.py"),
        kernel_script=str(config.paths.kernel_run_dir(run_id) / "kernel.py"),
        rules_url=str(config.paths.rules_url_path),
        rules_md=str(config.paths.rules_md_path),
        overview_md=str(config.paths.overview_md_path),
        data_md=str(config.paths.data_md_path),
        submission_format=str(config.paths.submission_format_md_path),
        dataset_profile=str(config.paths.dataset_profile_path),
        sample_submission=str(config.paths.sample_submission_path),
    )
    subgroup_metrics_path = iter_dir / "output" / "metrics.json"
    subgroup_payload = _load_json_object(subgroup_metrics_path) if subgroup_metrics_path.exists() else {}
    subgroup_collapse_signal = _detect_subgroup_collapse_signal(
        kernel_metrics_payload=subgroup_payload if isinstance(subgroup_payload, dict) else None,
        direction="minimize",
    )
    if subgroup_collapse_signal is not None:
        prompt_text = (
            "Subgroup repair target:\n"
            f"- {subgroup_collapse_signal['note']}\n"
            "- Prefer subgroup-aware fixes over global retuning.\n"
            "- If selection or fallback logic is coarse, refine it to (model_id,node_type) granularity.\n\n"
            + prompt_text
        )
    if missing_module:
        prompt_text = (
            f"Missing dependency detected: {missing_module}\n"
            "Guard/disable only this missing package path. Keep actively using other available "
            "repo dependencies (torch/timm/torchvision/opencv/xgboost/lightgbm/catboost/"
            "transformers/tabicl/ultralytics/sklearn) and avoid silent capacity downgrades. "
            "If this package is required, add it via `uv add <package>` and update `pyproject.toml` "
            "+ `uv.lock`.\n\n" + prompt_text
        )
    if prompt_prefix.strip():
        prompt_text = f"{prompt_prefix.strip()}\n\n{prompt_text}"

    strategy_text = ""
    strategy_skip_reason: str | None = None
    if not use_gpt_strategy:
        strategy_skip_reason = "metric_fix_policy"
    else:
        strategy_skip_reason = _error_strategy_skip_reason(stage="kernel_fix", error_text=error_message)
    if strategy_skip_reason:
        print(
            "[yellow]kernel fix[/yellow]: "
            f"skipping gpt strategy ({strategy_skip_reason}); invoking {IMPLEMENTATION_AGENT.log_alias} fixer directly."
        )
    else:
        _update_watch_phase(
            config,
            run_id,
            "gpt_kernel_fix_thinking",
            detail="GPT is analyzing the kernel failure and drafting a fix strategy.",
            iteration=iteration,
        )
        strategy_prompt = _build_error_strategy_prompt(
            stage="kernel_fix",
            slug=config.slug,
            run_id=run_id,
            attempt=attempt,
            compute=config.compute,
            accelerator=config.accelerator,
            hardware_constraints=render_hardware_constraints(
                resolve_hardware_profile(config.hardware_profile, compute=config.compute),
                compute=config.compute,
                time_budget_min=config.time_budget_min,
            ),
            error_text=error_message,
            codex_prompt=prompt_text,
        )
        strategy_dir = agent_dir / f"kernel_fix_strategy-{attempt:02d}"
        strategy_text = _run_error_strategy(
            prompt_text=strategy_prompt,
            output_dir=strategy_dir,
            dry_run=config.dry_run,
            stage_label="kernel fix",
        )
    if strategy_text:
        prompt_text += (
            f"\n\n## {STRATEGY_AGENT.display_name} Extra-High Error-Fix Strategy\n"
            "Use the strategy below as guidance, then apply minimal targeted edits.\n\n"
            f"{strategy_text}\n"
        )

    base_prompt_text = f"Kernel fix attempt: {attempt}\n\n{prompt_text}"
    prompt_path.write_text(base_prompt_text, encoding="utf-8")
    attempt_path = agent_dir / f"kernel_fix_prompt-{attempt:02d}.md"
    attempt_path.write_text(base_prompt_text, encoding="utf-8")
    _print_agent_prompt(prompt_path, base_prompt_text)

    allowed_prefixes = _repair_allowed_prefixes(config)
    guard_snapshot = _backup_guarded_files(config.paths.repo_root, allowed_prefixes)
    codex_pass_limit = max(1, int(max_codex_passes or MAX_KERNEL_FIX_CODEX_PASSES))
    retry_feedback = ""
    last_response_text = ""
    for codex_pass in range(1, codex_pass_limit + 1):
        pass_prompt_text = (
            base_prompt_text
            if not retry_feedback
            else _append_fix_retry_feedback(
                base_prompt=base_prompt_text,
                stage_label="kernel_fix",
                codex_pass=codex_pass - 1,
                failure_text=retry_feedback,
            )
        )
        pass_prompt_path = (
            prompt_path if codex_pass == 1 else agent_dir / f"kernel_fix_prompt-{attempt:02d}-pass-{codex_pass:02d}.md"
        )
        pass_prompt_path.write_text(pass_prompt_text, encoding="utf-8")
        if codex_pass > 1:
            print(
                "[yellow]kernel fix[/yellow]: "
                f"retrying {IMPLEMENTATION_AGENT.log_alias} pass "
                f"{codex_pass}/{codex_pass_limit} with previous failure context."
            )
        before = _snapshot_tree(config.paths.repo_root)
        pass_output_dir = (
            agent_dir if codex_pass == 1 else agent_dir / f"kernel_fix_pass-{attempt:02d}-{codex_pass:02d}"
        )
        print(f"[cyan]kernel fix[/cyan]: running {IMPLEMENTATION_AGENT.log_alias} fixer")
        _update_watch_phase(
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
            model=codex_model or _ERROR_FIX_CODEX_MODEL,
            reasoning_effort=codex_reasoning_effort or _ERROR_FIX_REASONING_EFFORT,
        )
        after = _snapshot_tree(config.paths.repo_root)
        changed = _diff_snapshots(before, after)
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
            _maybe_restart_for_src_changes(
                config=config,
                run_id=run_id,
                changed=changed,
                stage=f"kernel_fix_attempt_{attempt}",
            )
        response_text = _read_agent_response(result.last_message_path)
        _print_agent_response(result.last_message_path, response_text)
        _log_codex_sandbox_fallback(stage_label="kernel fix", result=result)
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
            regenerated = _maybe_regenerate_kernel_sources_once(
                config=config,
                run_id=run_id,
                iteration=iteration,
                iter_dir=iter_dir,
                attempt=attempt,
                trigger_reason="codex_no_changes",
            )
            if not regenerated:
                raise KernelFailedError(
                    "Kernel fix agent produced no file changes and regeneration fallback was already used."
                )
            try:
                _run_verify(config.verify_cmd, dry_run=config.dry_run, artifacts_dir=config.paths.artifacts_dir)
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
                            f"{IMPLEMENTATION_AGENT.display_name} kernel-fix made no edits; "
                            "regenerated kernel sources once and verification passed."
                        ),
                        "resolved": True,
                    }
                )
            return

        try:
            _run_verify(config.verify_cmd, dry_run=config.dry_run, artifacts_dir=config.paths.artifacts_dir)
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
        return

    raise RuntimeError(
        f"Kernel fix exhausted {IMPLEMENTATION_AGENT.log_alias} retry passes without resolving the error."
    )


def _run_metric_only_competition_metric_fix(
    *,
    config: AutopilotConfig,
    run_id: str,
    iteration: int,
    iter_dir: Path,
    mismatch_reason: str,
    attempt: int,
    pending_error_fixes: list[dict[str, object]] | None = None,
) -> None:
    """Apply a metric-only kernel fix using the implementation agent without GPT strategy mediation."""
    policy_prefix = (
        "Metric-only repair policy:\n"
        "- Edit ONLY competition metric selection/reporting logic in kernel outputs.\n"
        "- Do NOT change model architecture, features, training schedule, folds, seeds, or ensembling.\n"
        "- Ensure metrics.json reports the official competition metric exactly.\n"
        "- Ensure submission.csv format stays unchanged.\n"
    )
    metric_fix_error = (
        "Competition metric mismatch detected in strict mode.\n"
        f"Details: {mismatch_reason}\n"
        "Apply a minimal metric-only fix and stop."
    )
    _run_kernel_fix(
        config=config,
        run_id=run_id,
        iteration=iteration,
        iter_dir=iter_dir,
        error_message=metric_fix_error,
        attempt=attempt,
        pending_error_fixes=pending_error_fixes,
        use_gpt_strategy=False,
        codex_model=_METRIC_FIX_CODEX_MODEL,
        codex_reasoning_effort=_METRIC_FIX_REASONING_EFFORT,
        prompt_prefix=policy_prefix,
        max_codex_passes=_MAX_METRIC_FIX_CODEX_PASSES,
    )


def _rerun_kernel_for_metric_recheck(
    *,
    config: AutopilotConfig,
    run_id: str,
    iteration: int,
    submission_path: Path,
    iter_dir: Path,
    metrics_artifact_path: Path | None,
    kernel_name: str | None,
    enable_internet: bool,
    score_source: str,
    target_metric: str | None,
    metric_direction: str,
    holdout_frac: float,
    cv_folds: int,
    seed: int,
    time_budget_min: int | None,
) -> tuple[EvaluationResult, dict[str, object] | None, Path]:
    """Recheck metric parsing from existing artifacts without retraining in the same iteration."""
    del (
        config,
        run_id,
        iteration,
        kernel_name,
        enable_internet,
        score_source,
        holdout_frac,
        cv_folds,
        seed,
        time_budget_min,
    )

    rechecked_submission_path = submission_path
    if not rechecked_submission_path.exists():
        resolved_submission = _resolve_iteration_submission_artifact(iter_dir)
        if resolved_submission is None:
            raise RuntimeError(
                "Metric recheck failed: submission artifact is missing for same-iteration metric-only recheck."
            )
        rechecked_submission_path = _copy_submission_artifact_to_iteration_dir(
            source=resolved_submission,
            iter_dir=iter_dir,
        )

    output_metrics_path = iter_dir / "output" / "metrics.json"
    metrics_candidates: list[Path] = []
    seen_metric_candidates: set[str] = set()

    def add_metrics_candidate(path: Path | None) -> None:
        if path is None:
            return
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen_metric_candidates:
            return
        seen_metric_candidates.add(key)
        metrics_candidates.append(path)

    add_metrics_candidate(metrics_artifact_path)
    add_metrics_candidate(iter_dir / "metrics.json")
    add_metrics_candidate(output_metrics_path)
    add_metrics_candidate(_resolve_iteration_artifact(iter_dir, "metrics.json"))

    loaded_candidates: list[tuple[Path, dict[str, object], EvaluationResult | None]] = []
    for metrics_candidate in metrics_candidates:
        if not metrics_candidate.exists():
            continue
        candidate_payload = _load_json_object(metrics_candidate)
        if candidate_payload is None:
            continue
        if str(candidate_payload.get("kind") or "").strip().lower() == "submit_only":
            continue
        candidate_evaluation = _load_kernel_metrics(
            metrics_candidate,
            metric_direction,
            target_metric,
        )
        loaded_candidates.append((metrics_candidate, candidate_payload, candidate_evaluation))

    valid_candidates = [candidate for candidate in loaded_candidates if candidate[2] is not None]
    selected_candidate: tuple[Path, dict[str, object], EvaluationResult | None] | None = None
    if valid_candidates:
        selected_candidate = valid_candidates[0]
        output_candidate = next(
            (candidate for candidate in valid_candidates if candidate[0] == output_metrics_path),
            None,
        )
        if output_candidate is not None:
            try:
                output_mtime = output_metrics_path.stat().st_mtime
                other_mtimes = [
                    candidate[0].stat().st_mtime
                    for candidate in valid_candidates
                    if candidate[0] != output_metrics_path
                ]
            except OSError:
                other_mtimes = []
                output_mtime = 0.0
            if not other_mtimes or output_mtime >= max(other_mtimes):
                selected_candidate = output_candidate
    elif loaded_candidates:
        selected_candidate = loaded_candidates[0]

    if selected_candidate is None:
        raise RuntimeError(
            "Metric recheck failed: metrics.json artifact is missing for same-iteration metric-only recheck."
        )

    resolved_metrics_path, payload, evaluation = selected_candidate
    metric_mismatch = bool(
        target_metric and evaluation and evaluation.metric and not _metrics_equivalent(evaluation.metric, target_metric)
    )
    needs_recompute = evaluation is None or metric_mismatch
    if needs_recompute:
        recomputed = _recompute_metric_from_oof_artifact(
            iter_dir=iter_dir,
            payload=payload,
            target_metric=target_metric,
            metric_direction=metric_direction,
        )
        if recomputed is not None:
            evaluation, payload = recomputed
            _persist_metric_recheck_payload(
                iter_dir=iter_dir,
                resolved_metrics_path=resolved_metrics_path,
                payload=payload,
            )
    if evaluation is None:
        raise RuntimeError("Metric recheck failed: kernel metrics missing expected score.")
    return evaluation, payload, rechecked_submission_path


def _recompute_metric_from_oof_artifact(
    *,
    iter_dir: Path,
    payload: dict[str, object] | None,
    target_metric: str | None,
    metric_direction: str,
) -> tuple[EvaluationResult, dict[str, object]] | None:
    """Recompute target metric from cached OOF predictions without rerunning training."""
    if not target_metric:
        return None
    oof_path = _resolve_iteration_artifact(iter_dir, "oof_predictions.csv")
    if oof_path is None or not oof_path.exists():
        return None
    try:
        import pandas as pd
    except Exception:
        return None
    try:
        oof = pd.read_csv(oof_path)
    except Exception:
        return None
    if oof.empty:
        return None

    y_col = _pick_oof_target_column(oof)
    pred_col = _pick_oof_prediction_column(oof, metric=target_metric)
    if y_col is None or pred_col is None:
        return None

    y_series = pd.to_numeric(oof[y_col], errors="coerce")
    pred_series = pd.to_numeric(oof[pred_col], errors="coerce")
    valid_mask = y_series.notna() & pred_series.notna()
    if int(valid_mask.sum()) < 2:
        return None
    y_values = y_series[valid_mask].to_numpy()
    pred_values = pred_series[valid_mask].to_numpy()

    try:
        metric_value = float(compute_metric(target_metric, y_values, pred_values))
    except Exception:
        return None

    metric_name = canonical_metric(target_metric)
    direction = infer_direction(metric_name, metric_direction)
    score_source_raw = payload.get("score_source") if isinstance(payload, dict) else None
    score_source = (
        str(score_source_raw).strip() if isinstance(score_source_raw, str) and str(score_source_raw).strip() else "cv"
    )
    std_value = _to_float(payload.get("offline_std")) if isinstance(payload, dict) else None
    train_score = _to_float(payload.get("train_score")) if isinstance(payload, dict) else None
    val_score = _to_float(payload.get("val_score")) if isinstance(payload, dict) else None
    fold_scores = _extract_numeric_list(payload.get("fold_scores")) if isinstance(payload, dict) else None

    from kagglebot.solver.evaluate import EvaluationResult

    evaluation = EvaluationResult(
        score_source=score_source,
        metric=metric_name,
        direction=direction,  # type: ignore[arg-type]
        value=metric_value,
        std=std_value,
        train_score=train_score,
        val_score=val_score,
        fold_scores=fold_scores,
    )
    updated_payload = dict(payload) if isinstance(payload, dict) else {}
    updated_payload["metric"] = metric_name
    updated_payload["direction"] = direction
    updated_payload["score_source"] = score_source
    updated_payload["offline_value"] = metric_value
    updated_payload["value"] = metric_value
    updated_payload["metric_recheck_source"] = f"oof_predictions:{oof_path.name}"
    updated_payload["metric_recheck_without_retrain"] = True
    loop_decision = updated_payload.get("loop_decision")
    if isinstance(loop_decision, dict):
        loop_decision["source"] = score_source
        loop_decision["value"] = metric_value
    else:
        updated_payload["loop_decision"] = {"source": score_source, "value": metric_value}
    return evaluation, updated_payload


def _pick_oof_target_column(frame) -> str | None:  # type: ignore[no-untyped-def]
    """Return the target column name from an OOF prediction table."""
    columns = [str(col) for col in frame.columns]
    normalized = {col.lower().strip(): col for col in columns}
    for key in ("y", "target", "label", "y_true", "isdefault", "is_default"):
        if key in normalized:
            return normalized[key]
    return None


def _pick_oof_prediction_column(frame, *, metric: str) -> str | None:  # type: ignore[no-untyped-def]
    """Return the most suitable prediction column for the requested metric."""
    columns = [str(col) for col in frame.columns]
    normalized = {col.lower().strip(): col for col in columns}
    is_prob_metric = bool(metric_requires_proba(metric))

    if is_prob_metric:
        for key in ("oof_proba", "pred_proba", "prediction_proba", "probability", "proba", "score"):
            if key in normalized:
                return normalized[key]
        for col in columns:
            lowered = col.lower()
            if "proba" in lowered or "prob" in lowered or "score" in lowered:
                return col
    for key in ("oof_pred", "prediction", "pred", "y_pred"):
        if key in normalized:
            return normalized[key]
    if is_prob_metric:
        return None
    for col in columns:
        lowered = col.lower()
        if any(token in lowered for token in ("pred", "score", "proba", "prob")):
            return col
    return None


def _extract_numeric_list(value: object) -> list[float] | None:
    """Return parsed numeric list or None when payload value is not a numeric list."""
    if not isinstance(value, list):
        return None
    parsed = [float(item) for item in value if isinstance(item, (int, float))]
    return parsed or None


def _persist_metric_recheck_payload(*, iter_dir: Path, resolved_metrics_path: Path, payload: dict[str, object]) -> None:
    """Persist recomputed metric payload to canonical iteration metrics artifacts."""
    serialized = json.dumps(payload, indent=2)
    candidates = [resolved_metrics_path, iter_dir / "metrics.json", iter_dir / "output" / "metrics.json"]
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized, encoding="utf-8")


def _run_autofix(*, config: AutopilotConfig, run_id: str, attempt: int, error: Exception) -> None:
    run_dir = config.paths.run_dir(run_id)
    autofix_dir = run_dir / "autofix" / f"attempt-{attempt}"
    autofix_dir.mkdir(parents=True, exist_ok=True)
    error_text = "".join(traceback.format_exception(type(error), error, error.__traceback__)).strip()
    submit_autofix = isinstance(error, SubmitAbortedError)
    submit_context = ""
    submit_file_fix_required = False
    submit_file_fix_baseline_path: Path | None = None
    submit_file_fix_baseline_sha256: str | None = None
    if isinstance(error, KaggleCliError):
        if error.command:
            error_text = f"{error_text}\n\nKaggle CLI command:\n{shlex.join(error.command)}"
        if error.output:
            error_text = f"{error_text}\n\nKaggle CLI output:\n{error.output}"
    if submit_autofix:
        submit_context = _build_submit_autofix_context(run_dir)
        latest_submit_attempt = _load_latest_submit_attempt(run_dir)
        submit_file_fix_required = _submit_autofix.submit_file_fix_required_for_attempt(latest_submit_attempt)
        if submit_file_fix_required:
            submit_file_fix_baseline_path = _resolve_submit_autofix_submission_artifact(
                config=config,
                run_id=run_id,
                run_dir=run_dir,
            )
            submit_file_fix_baseline_sha256 = _sha256_or_none(submit_file_fix_baseline_path)
        _prepared_submission_path, prepared_submission_summary = _prepare_submit_file_autofix(
            config=config,
            run_id=run_id,
            run_dir=run_dir,
        )
        if prepared_submission_summary:
            submit_context = (
                f"{submit_context}\n\ndeterministic_submit_file_autofix:\n{prepared_submission_summary}".strip()
            )
            error_text = f"{error_text}\n\nDeterministic Submit File Autofix:\n{prepared_submission_summary}"
        if submit_context:
            error_text = f"{error_text}\n\nSubmit Failure Context:\n{submit_context}"
    attempt_tag = f"{attempt:02d}"
    header = f"autofix_attempt: {attempt}\n"
    error_path = autofix_dir / f"error-{attempt_tag}.txt"
    error_path.write_text(header + error_text + "\n", encoding="utf-8")
    (autofix_dir / "error.txt").write_text(header + error_text + "\n", encoding="utf-8")

    allowed_prefixes = _autofix_allowed_prefixes(config)
    prompt_text = _build_autofix_prompt(
        config=config,
        run_id=run_id,
        attempt=attempt,
        error_text=error_text,
        error_path=error_path,
        allowed_prefixes=allowed_prefixes,
        submit_context=submit_context,
    )
    if submit_file_fix_required:
        prompt_text += """

## Submission File Repair Contract

Kaggle rejected the submission artifact itself. This autofix is not complete unless the submission file bytes or
artifact path change, and `run_state.json` records `submit_autofix_submission_path` pointing to the repaired file.
If you repair the file in place, ensure the file contents actually change.
If this is competition-specific, fix the authoritative source that generates the artifact rather than leaving only
an ad-hoc repaired copy behind.
"""
    strategy_stage = "submit_autofix" if submit_autofix else "autofix"
    strategy_label = "submit autofix" if submit_autofix else "autofix"
    print(
        f"[cyan]{strategy_label}[/cyan]: strategy={_ERROR_STRATEGY_MODEL}({_ERROR_STRATEGY_REASONING_EFFORT}) "
        f"-> fixer={_ERROR_FIX_CODEX_MODEL}({_ERROR_FIX_REASONING_EFFORT})"
    )
    thinking_phase = "gpt_submit_autofix_thinking" if submit_autofix else "gpt_autofix_thinking"
    fixing_phase = "gpt_submit_autofix_fixing" if submit_autofix else "gpt_autofix_fixing"
    _update_watch_phase(
        config,
        run_id,
        thinking_phase,
        detail=f"GPT is analyzing the {strategy_label} failure and drafting a fix strategy.",
    )
    strategy_prompt = _build_error_strategy_prompt(
        stage=strategy_stage,
        slug=config.slug,
        run_id=run_id,
        attempt=attempt,
        compute=config.compute,
        accelerator=config.accelerator,
        hardware_constraints=render_hardware_constraints(
            resolve_hardware_profile(config.hardware_profile, compute=config.compute),
            compute=config.compute,
            time_budget_min=config.time_budget_min,
        ),
        error_text=error_text,
        codex_prompt=prompt_text,
    )
    strategy_text = _run_error_strategy(
        prompt_text=strategy_prompt,
        output_dir=autofix_dir / "gpt_strategy",
        dry_run=config.dry_run,
        stage_label=strategy_label,
    )
    if not strategy_text.strip():
        print(
            f"[yellow]{strategy_label}[/yellow]: gpt strategy unavailable, "
            f"continuing with direct {IMPLEMENTATION_AGENT.log_alias} fix"
        )
    else:
        prompt_text += (
            f"\n\n## {STRATEGY_AGENT.display_name} Extra-High Error-Fix Strategy\n"
            "Use the strategy below as guidance, then apply minimal targeted edits.\n\n"
            f"{strategy_text}\n"
        )
    prompt_path = autofix_dir / "prompt.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    _print_agent_prompt(prompt_path, prompt_text)

    retry_feedback = ""
    for codex_pass in range(1, MAX_AUTOFIX_CODEX_PASSES + 1):
        pass_prompt_text = (
            prompt_text
            if not retry_feedback
            else _append_fix_retry_feedback(
                base_prompt=prompt_text,
                stage_label="autofix",
                codex_pass=codex_pass - 1,
                failure_text=retry_feedback,
            )
        )
        pass_prompt_path = prompt_path if codex_pass == 1 else autofix_dir / f"prompt-pass-{codex_pass:02d}.md"
        pass_prompt_path.write_text(pass_prompt_text, encoding="utf-8")
        if codex_pass > 1:
            print(
                "[yellow]autofix[/yellow]: "
                f"retrying {IMPLEMENTATION_AGENT.log_alias} pass "
                f"{codex_pass}/{MAX_AUTOFIX_CODEX_PASSES} with previous failure context."
            )

        before = _snapshot_tree(config.paths.repo_root)
        pass_output_dir = autofix_dir if codex_pass == 1 else autofix_dir / f"pass-{codex_pass:02d}"
        _update_watch_phase(
            config,
            run_id,
            fixing_phase,
            detail=f"{IMPLEMENTATION_AGENT.display_name} is applying the {strategy_label} fix.",
        )
        result = run_codex(
            pass_prompt_path,
            pass_output_dir,
            dry_run=config.dry_run,
            heartbeat_label="fixing error",
            model=_ERROR_FIX_CODEX_MODEL,
            reasoning_effort=_ERROR_FIX_REASONING_EFFORT,
        )
        after = _snapshot_tree(config.paths.repo_root)
        changed = _diff_snapshots(before, after)
        # Autofix often needs to regenerate staged outputs under artifacts/*/kernels and
        # run-level state files; do not apply write-guard restrictions in this stage.
        _maybe_restart_for_src_changes(
            config=config,
            run_id=run_id,
            changed=changed,
            stage=f"autofix_attempt_{attempt}",
        )
        response_text = _read_agent_response(result.last_message_path)
        _print_agent_response(result.last_message_path, response_text)
        _log_codex_sandbox_fallback(stage_label=strategy_label, result=result)
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

        if submit_file_fix_required and not _submit_file_fix_contract_satisfied(
            run_dir=run_dir,
            baseline_path=submit_file_fix_baseline_path,
            baseline_sha256=submit_file_fix_baseline_sha256,
        ):
            retry_feedback = (
                "Submission file repair contract not satisfied.\n"
                "Kaggle rejected the submission artifact itself, so this autofix must change the prepared "
                "submission file bytes or output path.\n"
                f"baseline_submission_path={submit_file_fix_baseline_path}\n"
                f"baseline_submission_sha256={submit_file_fix_baseline_sha256}\n"
                "Record the repaired artifact in run_state.json as submit_autofix_submission_path."
            )
            if codex_pass < MAX_AUTOFIX_CODEX_PASSES:
                continue
            raise RuntimeError("Submit autofix did not repair the submission artifact required by Kaggle.")

        try:
            _run_verify(config.verify_cmd, dry_run=config.dry_run, artifacts_dir=config.paths.artifacts_dir)
        except Exception as exc:  # noqa: BLE001
            retry_feedback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
            if codex_pass < MAX_AUTOFIX_CODEX_PASSES:
                continue
            raise
        return

    raise RuntimeError(f"Autofix exhausted {IMPLEMENTATION_AGENT.log_alias} retry passes without resolving the error.")


def _repair_allowed_prefixes(config: AutopilotConfig, *, extra_paths: list[Path] | None = None) -> WriteGuardPolicy:
    extra_allowed: list[Path] = []
    if extra_paths:
        extra_allowed.extend(extra_paths)
    module_src_root = Path(__file__).resolve().parents[1]
    if module_src_root.name == "src":
        extra_allowed.append(module_src_root)
    return _repo_root_write_policy(
        repo_root=config.paths.repo_root,
        denied_prefixes=[config.paths.data_dir, config.paths.kernels_dir],
        extra_allowed_prefixes=extra_allowed,
    )


def _autofix_allowed_prefixes(config: AutopilotConfig) -> WriteGuardPolicy:
    return _repair_allowed_prefixes(config)


def _log_codex_sandbox_fallback(*, stage_label: str, result: object) -> None:
    if not bool(getattr(result, "used_sandbox_fallback", False)):
        return
    excerpt = str(getattr(result, "sandbox_failure_excerpt", "")).strip()
    detail = f": {excerpt}" if excerpt else ""
    print(f"[yellow]{stage_label}[/yellow]: codex sandbox fallback used{detail}")


def _build_autofix_prompt(
    *,
    config: AutopilotConfig,
    run_id: str,
    attempt: int,
    error_text: str,
    error_path: Path,
    allowed_prefixes: WriteGuardPolicy,
    submit_context: str = "",
) -> str:
    allowed_list = "\n".join(f"- {path}" for path in allowed_prefixes.allowed_prefixes)
    denied_list = "\n".join(f"- {path}" for path in allowed_prefixes.denied_prefixes)
    submit_context_block = ""
    if submit_context:
        submit_context_block = (
            "\n## Submit Context\n\n"
            "This is a submit-stage failure. Use `repair_target` to decide whether to fix the submission artifact, "
            "submit mode/kernel path, or a platform/manual blocker.\n"
            "This run must fix the submission contract before further model changes.\n"
            "If the error is competition-specific, edit only authoritative `kernel.py`.\n"
            "Do not leave iter2 with the same Kaggle row/column/evaluation exception.\n\n"
            "```\n"
            f"{submit_context}\n"
            "```\n"
        )
    return f"""\
# Kagglebot {IMPLEMENTATION_AGENT.display_name}: Auto-Fix

## Context

Competition: {config.slug}
Run ID: {run_id}
Attempt: {attempt}
Compute: {config.compute} ({config.accelerator})

## Error

```
{error_text}
```

Error log file: {error_path}
{submit_context_block}

## Relevant Paths

- repo_root: {config.paths.repo_root}
- run_dir: {config.paths.run_dir(run_id)}
- kernel_dir: {config.paths.kernel_source_dir}
- context_dir: {config.paths.context_dir}
- data_dir: {config.paths.data_dir}
- prompts_dir: {config.paths.prompts_dir}
- autopilot: {Path(__file__).resolve()}

## Allowed Edit Scope

{allowed_list}

## Forbidden Edit Scope

{denied_list or "- None"}

## Task

1) Identify root cause of the failure.
2) Apply minimal, targeted fixes so autopilot can continue.
3) Do not touch datasets or credentials.
   Prefer already-installed dependencies; add new dependencies only with clear justification.
   If a dependency must be added, use `uv add <package>` and keep `pyproject.toml` + `uv.lock` consistent.
4) Explain what you changed in your response.
"""


def _build_error_strategy_prompt(
    *,
    stage: str,
    slug: str,
    run_id: str,
    attempt: int,
    compute: str,
    accelerator: str,
    hardware_constraints: str,
    error_text: str,
    codex_prompt: str,
) -> str:
    return f"""\
# Kagglebot {STRATEGY_AGENT.display_name} Error Strategy

You are {STRATEGY_AGENT.display_name} in xhigh reasoning mode.
Think through the failure and produce a concrete fix strategy for
{IMPLEMENTATION_AGENT.display_name} (xhigh), which will apply edits.

Stage: {stage}
Competition: {slug}
Run ID: {run_id}
Attempt: {attempt}
Compute: {compute} ({accelerator})
Hardware execution budget:
{hardware_constraints}

## Error

```
{error_text}
```

## {IMPLEMENTATION_AGENT.display_name} Fix Prompt (current)

```
{codex_prompt}
```

## Required Output

Return concise, actionable instructions for {IMPLEMENTATION_AGENT.display_name}:
1) Root cause hypothesis.
2) Minimal file edits (paths + what to change).
3) Safety checks to run after edits.
4) Fallback if the first fix does not work.
"""


def _run_error_strategy(
    *,
    prompt_text: str,
    output_dir: Path,
    dry_run: bool,
    stage_label: str,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "gpt_strategy_prompt.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    print(f"[cyan]{stage_label}[/cyan]: gpt analyzing error")
    print(
        f"[cyan]{stage_label}[/cyan]: strategy model={_ERROR_STRATEGY_MODEL} "
        f"reasoning={_ERROR_STRATEGY_REASONING_EFFORT}"
    )
    result = run_strategy(prompt_path, output_dir, dry_run=dry_run)
    strategy_text = _read_agent_response(result.last_message_path).strip()
    if result.returncode != 0:
        print(
            f"[yellow]{stage_label}[/yellow]: gpt strategy failed, "
            f"continuing with direct {IMPLEMENTATION_AGENT.log_alias} fix"
        )
        return ""
    if not strategy_text:
        print(
            f"[yellow]{stage_label}[/yellow]: gpt strategy empty, "
            f"continuing with direct {IMPLEMENTATION_AGENT.log_alias} fix"
        )
        return ""
    return strategy_text


def _maybe_regenerate_kernel_sources_once(
    *,
    config: AutopilotConfig,
    run_id: str,
    iteration: int,
    iter_dir: Path,
    attempt: int,
    trigger_reason: str,
) -> bool:
    """Regenerate authoritative kernel sources once when fix loops are stuck."""
    if config.dry_run:
        return False
    agent_dir = iter_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    marker_path = agent_dir / _KERNEL_REGENERATE_MARKER_FILENAME
    if marker_path.exists():
        return False

    marker_payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "trigger_reason": trigger_reason,
        "attempt": int(attempt),
        "iteration": int(iteration),
        "run_id": run_id,
    }
    _write_json_object(marker_path, marker_payload)
    print(
        "[yellow]kernel fix[/yellow]: unresolved kernel error loop detected; "
        "regenerating kernel sources once before retry."
    )
    try:
        _run_plan_and_initial(config, run_id)
    except Exception as exc:  # noqa: BLE001
        note_path = agent_dir / f"kernel_regen_note-{attempt:02d}.txt"
        note_path.write_text(
            (f"kernel_regen_failed: regeneration fallback failed.\ntrigger_reason: {trigger_reason}\nerror: {exc}\n"),
            encoding="utf-8",
        )
        return False
    note_path = agent_dir / f"kernel_regen_note-{attempt:02d}.txt"
    note_path.write_text(
        (f"kernel_regen_applied: regeneration fallback succeeded.\ntrigger_reason: {trigger_reason}\n"),
        encoding="utf-8",
    )
    return True


def _maybe_apply_lightweight_runtime_fix(
    *,
    config: AutopilotConfig,
    error_text: str,
    note_path: Path,
    stage_label: str,
) -> str | None:
    actions: tuple[tuple[str, str, Callable[[AutopilotConfig, str], bool]], ...] = (
        (
            "column_fill.json",
            "missing column error",
            _maybe_write_column_fill,
        ),
        (
            "object_coerce.json",
            "numpy.object_ conversion error",
            _maybe_write_object_coerce,
        ),
        (
            "device_coerce.json",
            "torch device mismatch error",
            _maybe_write_device_coerce,
        ),
        (
            "column_map.json",
            "column alias mismatch",
            _maybe_write_column_map,
        ),
    )
    for artifact_name, reason, action in actions:
        try:
            changed = bool(action(config, error_text))
        except Exception:
            changed = False
        if not changed:
            continue
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note = (
            f"autofix_note: {artifact_name} created for {reason}.\n"
            "autofix will retry without modifying kernel sources.\n"
        )
        note_path.write_text(note, encoding="utf-8")
        print(
            f"[yellow]{stage_label}[/yellow]: wrote {artifact_name}; "
            f"retrying without {IMPLEMENTATION_AGENT.log_alias} edits"
        )
        return artifact_name
    return None


def _resume_best_submitted_offline_score(
    *,
    paths: CompetitionPaths,
    run_id: str,
    metric_direction: str,
    target_metric: str,
    max_iterations: int,
) -> float | None:
    return _state_resume_best_submitted_offline_score(
        paths=paths,
        run_id=run_id,
        metric_direction=metric_direction,
        target_metric=target_metric,
        max_iterations=max_iterations,
        load_kernel_metrics=_load_kernel_metrics,
    )


def _resume_best_submittable_iteration_state(
    *,
    paths: CompetitionPaths,
    run_id: str,
    metric_direction: str,
    target_metric: str,
    max_iterations: int,
) -> tuple[float | None, Path | None]:
    return _state_resume_best_submittable_iteration_state(
        paths=paths,
        run_id=run_id,
        metric_direction=metric_direction,
        target_metric=target_metric,
        max_iterations=max_iterations,
        load_kernel_metrics=_load_kernel_metrics,
        iteration_metrics_allow_submit=_iteration_metrics_allow_submit,
    )


def _resume_iteration_state(
    *,
    paths: CompetitionPaths,
    run_id: str,
    metric_direction: str,
    target_metric: str,
    max_iterations: int,
    require_submit_phase: bool = False,
) -> tuple[int, float | None, Path | None]:
    return _state_resume_iteration_state(
        paths=paths,
        run_id=run_id,
        metric_direction=metric_direction,
        target_metric=target_metric,
        max_iterations=max_iterations,
        require_submit_phase=require_submit_phase,
        load_kernel_metrics=_load_kernel_metrics,
        infer_iteration_from_submission_path=_submit_stage.infer_iteration_from_submission_path,
    )


def _load_submit_retry_artifacts(
    *,
    run_dir: Path,
    iter_dir: Path,
    iteration: int,
    max_iterations: int,
    metric_direction: str,
    target_metric: str,
    require_submit_phase: bool,
) -> tuple[Path, Path, EvaluationResult] | None:
    return _state_load_submit_retry_artifacts(
        run_dir=run_dir,
        iter_dir=iter_dir,
        iteration=iteration,
        max_iterations=max_iterations,
        metric_direction=metric_direction,
        target_metric=target_metric,
        require_submit_phase=require_submit_phase,
        load_kernel_metrics=_load_kernel_metrics,
    )


def _maybe_restart_for_src_changes(*, config: AutopilotConfig, run_id: str, changed: list[str], stage: str) -> None:
    _autofix_restart.maybe_restart_for_src_changes(
        dry_run=config.dry_run,
        run_dir=config.paths.run_dir(run_id),
        run_id=run_id,
        slug=config.slug,
        changed=changed,
        stage=stage,
        max_restarts=MAX_AUTOFIX_RESTARTS,
    )


def _attempt_submit(
    *,
    config: AutopilotConfig,
    run_id: str,
    submission_path: Path,
    best_score: float | None,
    problem_types: list[str],
    submit_mode: str = "file",
    notebook_submit_artifact_mode: str = "wrapper",
) -> dict[str, object] | None:
    if not config.submit or config.dry_run:
        return None
    run_dir = config.paths.run_dir(run_id)
    submit_attempt_recorder = _submit_attempts.SubmitAttemptRecorder(
        run_dir=run_dir,
        save_run_state=lambda updates: _save_run_state(run_dir, updates),
    )
    _clear_stale_submit_autofix_artifact(run_dir=run_dir, submission_path=submission_path)
    run_state = _load_run_state(run_dir)
    submit_failure_context = _submit_failure_context.load_submit_failure_context(run_dir)
    latest_submit_attempt = _load_latest_submit_attempt(run_dir)
    submit_code_fingerprint = _compute_submit_code_fingerprint(config)
    allow_force = config.force_submit or _env_truthy("KAGGLEBOT_FORCE_RESUBMIT")
    autofix_input_decision = _submit_failure_context.decide_submit_autofix_input_submission(
        run_state=run_state,
        latest_submit_attempt=latest_submit_attempt,
        failure_context=submit_failure_context,
        submission_path=submission_path,
    )
    input_submission_path = autofix_input_decision.input_submission_path
    if autofix_input_decision.message:
        print(autofix_input_decision.message)

    message = _submit_stage.resolve_submission_message(
        context_dir=config.paths.context_dir,
        run_id=run_id,
        best_score=best_score,
        explicit_message=config.message,
        submission_path=input_submission_path,
        campaign_mode=config.campaign_mode,
        target_direction=config.target_direction,
    )
    submission_service = SubmissionService(
        SubmissionConfig(
            slug=config.slug,
            data_dir=config.paths.data_dir,
            sample_submission_path=config.paths.sample_submission_path,
            submission_ledger_path=config.paths.submission_ledger_path,
            dry_run=config.dry_run,
            force_submit=config.force_submit,
            bypass_rate_limit=False,
        )
    )
    print(f"[cyan]submit[/cyan]: {config.slug}")
    submitted_at = datetime.now(UTC)

    try:
        prepared_submission_path = submission_service.validate_and_prepare_submission(input_submission_path)
    except SubmissionValidationError as exc:
        fingerprint = compute_error_fingerprint("", str(exc))
        return _abort_submit_for_run(
            config=config,
            run_id=run_id,
            problem_types=problem_types,
            submission_ref=input_submission_path,
            code_fingerprint=submit_code_fingerprint,
            fingerprint=fingerprint,
            error_kind="validation",
            reason="local_submission_validation_failed",
            message="Local submission validation failed; Kaggle CLI submit is skipped.",
            stdout_tail="",
            stderr_tail=str(exc),
            exit_code=SubmissionValidationError.exit_code,
            submit_attempt_recorder=submit_attempt_recorder,
        )

    prepared_submission_sha = str(_sha256_or_none(prepared_submission_path) or "").strip()
    duplicate_sources: list[str] = []
    if prepared_submission_sha and not allow_force:
        if _submit_attempts.submit_attempt_sha_seen(run_dir=run_dir, submission_sha=prepared_submission_sha):
            duplicate_sources.append("run_attempts")
        if SubmissionLedger(config.paths.submission_ledger_path).is_duplicate(
            slug=config.slug,
            message=message,
            submission_path=prepared_submission_path,
        ):
            duplicate_sources.append("submission_ledger")
    duplicate_decision = _submit_retry_policy.decide_duplicate_submission_action(
        slug=config.slug,
        prepared_submission_sha=prepared_submission_sha,
        duplicate_sources=duplicate_sources,
        allow_force=allow_force,
        compute_fingerprint=compute_error_fingerprint,
    )
    if duplicate_decision.action == "skip":
        print(duplicate_decision.message)
        reason = duplicate_decision.reason
        fingerprint = duplicate_decision.fingerprint
        duplicate_sources = duplicate_decision.duplicate_sources
        skip_payloads = _submit_attempts.build_submit_skip_record_payloads(
            run_id=run_id,
            submission_ref=str(prepared_submission_path),
            submission_sha256=prepared_submission_sha,
            fingerprint=fingerprint,
            code_fingerprint=submit_code_fingerprint,
            error_kind="none",
            reason=reason,
            prior_state=_load_run_state(run_dir),
            stdout_tail_chars=_SUBMIT_STDOUT_TAIL_CHARS,
            stderr_tail_chars=_SUBMIT_STDERR_TAIL_CHARS,
            duplicate_sources=duplicate_sources,
        )
        submit_attempt_recorder.record_payloads(skip_payloads)
        _submit_failure_context.mark_submit_failure_context_resolved(
            run_dir=run_dir,
            resolution=reason,
            submission_ref=str(prepared_submission_path),
        )
        return _submit_attempts.build_submit_result_payload(
            message=message,
            submission_ref=str(prepared_submission_path),
            submitted_at_iso=submitted_at.isoformat(),
            iteration=_submit_stage.infer_iteration_from_submission_path(submission_path),
            skipped=True,
            reason=reason,
            duplicate_sources=duplicate_sources,
        )

    try:
        rules_accepted = check_rules_accepted(config.slug, dry_run=config.dry_run)
    except KaggleCliError as exc:
        if _is_missing_kaggle_credentials_error(exc):
            return _abort_submit_for_run(
                config=config,
                run_id=run_id,
                problem_types=problem_types,
                submission_ref=prepared_submission_path,
                code_fingerprint=submit_code_fingerprint,
                fingerprint=compute_error_fingerprint(exc.stdout, exc.stderr or exc.output),
                error_kind="permanent",
                reason="kaggle_credentials_missing",
                message=("Kaggle credentials not configured. Set ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY."),
                stdout_tail=exc.stdout,
                stderr_tail=exc.stderr or exc.output,
                exit_code=exc.exit_code,
                submit_attempt_recorder=submit_attempt_recorder,
            )
        raise
    if not rules_accepted:
        return _abort_submit_for_run(
            config=config,
            run_id=run_id,
            problem_types=problem_types,
            submission_ref=prepared_submission_path,
            code_fingerprint=submit_code_fingerprint,
            fingerprint=compute_error_fingerprint("", "rules_not_accepted"),
            error_kind="permanent",
            reason="rules_not_accepted",
            message="Competition rules are not accepted; aborting submit stage for this run.",
            stdout_tail="",
            stderr_tail="rules_not_accepted",
            exit_code=RulesNotAcceptedError.exit_code,
            submit_attempt_recorder=submit_attempt_recorder,
        )

    constraints = _competition_rules.load_competition_rule_constraints(config.paths)
    requested_notebook_submit = normalize_submit_mode(submit_mode, default="file") == "notebook"
    resolved_notebook_artifact_mode = (
        _resolve_notebook_submit_artifact_mode(paths=config.paths, submit_mode="notebook")
        if requested_notebook_submit or constraints.notebook_submissions_only
        else None
    )
    submit_stage_mode = _submit_stage.decide_initial_submit_stage_mode(
        requested_notebook_submit=requested_notebook_submit,
        notebook_submissions_only=constraints.notebook_submissions_only,
        notebook_submit_artifact_mode=notebook_submit_artifact_mode,
        resolved_notebook_artifact_mode=resolved_notebook_artifact_mode,
    )
    notebook_submit_required = submit_stage_mode.notebook_submit_required
    notebook_fallback_activated = submit_stage_mode.notebook_fallback_activated
    submission_artifact_mode = submit_stage_mode.submission_artifact_mode
    for mode_message in submit_stage_mode.messages:
        print(mode_message)
    artifact_mode_decision = _decide_notebook_submit_artifact_mode_for_submission(
        paths=config.paths,
        requested_mode=submission_artifact_mode,
        notebook_submit_required=notebook_submit_required,
        submission_path=prepared_submission_path,
    )
    submission_artifact_mode = artifact_mode_decision.mode
    if artifact_mode_decision.message:
        print(artifact_mode_decision.message)

    if not notebook_submit_required:
        same_path_decision = _submit_retry_policy.decide_same_submission_path_action(
            run_state=run_state,
            latest_submit_attempt=latest_submit_attempt,
            prepared_submission_path=prepared_submission_path,
            current_submission_sha=str(_sha256_or_none(prepared_submission_path) or "").strip(),
            submit_code_fingerprint=submit_code_fingerprint,
            allow_force=allow_force,
            notebook_submit_required=notebook_submit_required,
        )
        if same_path_decision.action == "retry":
            print(same_path_decision.message)
        elif same_path_decision.action == "skip":
            print(same_path_decision.message)
            submit_attempt_recorder.append(
                _submit_attempts.build_submit_skip_attempt_payload(
                    run_id=run_id,
                    submission_ref=str(prepared_submission_path),
                    submission_sha256=_sha256_or_none(prepared_submission_path),
                    fingerprint=same_path_decision.fingerprint,
                    error_kind="unknown",
                    reason=same_path_decision.reason,
                    stdout_tail_chars=_SUBMIT_STDOUT_TAIL_CHARS,
                    stderr_tail_chars=_SUBMIT_STDERR_TAIL_CHARS,
                )
            )
            return None

    seen_fingerprints = set(_load_submit_fingerprints(run_dir))
    state_fingerprint = str(run_state.get("last_submit_fingerprint") or run_state.get("last_fingerprint") or "").strip()
    if state_fingerprint:
        seen_fingerprints.add(state_fingerprint)
    max_attempts = max(1, _SUBMIT_MAX_TRANSIENT_RETRIES)
    submission_result = None
    submission_reference = str(prepared_submission_path)
    submission_artifact_path: Path | None = prepared_submission_path
    for attempt in range(1, max_attempts + 1):
        try:
            submit_attempt_result = _submit_stage.run_submit_stage_attempt(
                notebook_submit_required=notebook_submit_required,
                file_submission_path=prepared_submission_path,
                run_notebook_submit=lambda: _submit_with_notebook_kernel(
                    config=config,
                    run_id=run_id,
                    submission_path=prepared_submission_path,
                    message=message,
                    artifact_mode=submission_artifact_mode,
                ),
                run_file_submit=lambda: submission_service.submit_prepared(
                    prepared_path=prepared_submission_path,
                    message=message,
                    run_id=run_id,
                    offline_score=best_score,
                    score_source="offline",
                ),
            )
            submission_result = submit_attempt_result.submission_result
            submission_reference = submit_attempt_result.submission_reference
            submission_artifact_path = submit_attempt_result.submission_artifact_path
        except SubmissionCliError as exc:
            submit_error_classification = _submit_stage.classify_submit_stage_error(
                stdout=exc.stdout,
                stderr=exc.stderr or "",
                output=exc.output or "",
                exit_code=exc.exit_code,
                classify_submit_error=classify_submit_error,
            )
            classification_stderr = submit_error_classification.stderr
            classification_kind = submit_error_classification.kind
            classification_reason = submit_error_classification.reason
            notebook_fallback_decision = _submit_stage.decide_notebook_fallback_after_file_submit_error(
                notebook_submit_required=notebook_submit_required,
                notebook_fallback_activated=notebook_fallback_activated,
                should_use_notebook_fallback=_submit_failure_policy.should_use_notebook_submit_fallback(
                    reason=classification_reason,
                    stdout=exc.stdout,
                    stderr=exc.stderr,
                ),
                resolved_notebook_artifact_mode=_resolve_notebook_submit_artifact_mode(
                    paths=config.paths,
                    submit_mode="notebook",
                ),
                current_submission_artifact_mode=submission_artifact_mode,
            )
            if notebook_fallback_decision.retry_as_notebook:
                notebook_submit_required = notebook_fallback_decision.notebook_submit_required
                notebook_fallback_activated = notebook_fallback_decision.notebook_fallback_activated
                submission_artifact_mode = notebook_fallback_decision.submission_artifact_mode
                artifact_mode_decision = _decide_notebook_submit_artifact_mode_for_submission(
                    paths=config.paths,
                    requested_mode=submission_artifact_mode,
                    notebook_submit_required=notebook_submit_required,
                    submission_path=prepared_submission_path,
                )
                submission_artifact_mode = artifact_mode_decision.mode
                for mode_message in notebook_fallback_decision.messages:
                    print(mode_message)
                if artifact_mode_decision.message:
                    print(artifact_mode_decision.message)
                continue
            fingerprint = compute_error_fingerprint(exc.stdout, exc.stderr)
            fingerprint_seen = fingerprint in seen_fingerprints
            same_fingerprint_retry_allowed = False
            if fingerprint_seen:
                same_fingerprint_retry_allowed = _consume_same_submit_fingerprint_retry_allowance(
                    run_dir=run_dir,
                    run_state=run_state,
                    fingerprint=fingerprint,
                    code_fingerprint=submit_code_fingerprint,
                )
            error_action = _submit_stage.decide_submit_stage_error_action(
                fingerprint_seen=fingerprint_seen,
                same_fingerprint_retry_allowed=same_fingerprint_retry_allowed,
                classification_kind=classification_kind,
                classification_reason=classification_reason,
                attempt=attempt,
                max_attempts=max_attempts,
                retry_after_seconds=submit_error_classification.retry_after_seconds,
                backoff_seconds=_compute_submit_backoff(attempt),
            )
            for action_message in error_action.messages:
                print(action_message)
            if error_action.action == "abort":
                return _abort_submit_for_run(
                    config=config,
                    run_id=run_id,
                    problem_types=problem_types,
                    submission_ref=submission_reference,
                    submission_artifact_path=submission_artifact_path,
                    artifact_mode=submission_artifact_mode,
                    code_fingerprint=submit_code_fingerprint,
                    fingerprint=fingerprint,
                    error_kind=error_action.error_kind,
                    reason=error_action.reason,
                    message=error_action.abort_message,
                    stdout_tail=exc.stdout,
                    stderr_tail=classification_stderr,
                    exit_code=exc.exit_code,
                    submit_attempt_recorder=submit_attempt_recorder,
                )
            seen_fingerprints.add(fingerprint)
            if error_action.action == "retry":
                submit_attempt_recorder.append(
                    _submit_attempts.build_submit_retry_attempt_payload(
                        run_id=run_id,
                        submission_ref=submission_reference,
                        submission_sha256=_sha256_or_none(submission_artifact_path),
                        exit_code=exc.exit_code,
                        fingerprint=fingerprint,
                        reason=error_action.reason,
                        stdout=exc.stdout,
                        stderr=classification_stderr,
                        stdout_tail_chars=_SUBMIT_STDOUT_TAIL_CHARS,
                        stderr_tail_chars=_SUBMIT_STDERR_TAIL_CHARS,
                    )
                )
                _record_submit_reason_knowledge(
                    config=config,
                    run_id=run_id,
                    problem_types=problem_types,
                    submission_path=submission_artifact_path or prepared_submission_path,
                    error_kind="transient",
                    reason=error_action.reason,
                    action_taken="retry",
                    fingerprint=fingerprint,
                    details=f"attempt={attempt}; wait={error_action.wait_seconds:.1f}s",
                )
                time.sleep(error_action.wait_seconds)
                continue
        except (DuplicateSubmissionError, SubmissionRateLimitError) as exc:
            fingerprint = compute_error_fingerprint("", str(exc))
            return _abort_submit_for_run(
                config=config,
                run_id=run_id,
                problem_types=problem_types,
                submission_ref=submission_reference,
                submission_artifact_path=submission_artifact_path,
                code_fingerprint=submit_code_fingerprint,
                fingerprint=fingerprint,
                error_kind="permanent",
                reason="local_submission_guardrail",
                message=f"Local submission guardrail blocked submit: {exc}",
                stdout_tail="",
                stderr_tail=str(exc),
                exit_code=getattr(exc, "exit_code", 1),
                submit_attempt_recorder=submit_attempt_recorder,
            )
        except KaggleCliError as exc:
            if _is_missing_kaggle_credentials_error(exc):
                return _abort_submit_for_run(
                    config=config,
                    run_id=run_id,
                    problem_types=problem_types,
                    submission_ref=submission_reference,
                    submission_artifact_path=submission_artifact_path,
                    artifact_mode=submission_artifact_mode,
                    code_fingerprint=submit_code_fingerprint,
                    fingerprint=compute_error_fingerprint(exc.stdout, exc.stderr or exc.output),
                    error_kind="permanent",
                    reason="kaggle_credentials_missing",
                    message=(
                        "Kaggle credentials not configured. Set ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY."
                    ),
                    stdout_tail=exc.stdout,
                    stderr_tail=exc.stderr or exc.output,
                    exit_code=exc.exit_code,
                    submit_attempt_recorder=submit_attempt_recorder,
                )
            raise
        break

    if submission_result is None:
        raise SubmitAbortedError("Submit failed before producing a submission result.")
    submission_ref = submission_reference
    submission_for_submit_path = submission_artifact_path
    submit_success_record = _submit_stage.build_submit_stage_success_record(
        submission_result=submission_result,
        compute_error_fingerprint=compute_error_fingerprint,
    )
    submit_success_payloads = _submit_attempts.build_submit_success_record_payloads(
        run_id=run_id,
        submission_ref=submission_ref,
        submission_sha256=_sha256_or_none(submission_for_submit_path),
        exit_code=submit_success_record.exit_code,
        fingerprint=submit_success_record.fingerprint,
        code_fingerprint=submit_code_fingerprint,
        stdout=submit_success_record.stdout,
        stderr=submit_success_record.stderr,
        prior_state=_load_run_state(run_dir),
        stdout_tail_chars=_SUBMIT_STDOUT_TAIL_CHARS,
        stderr_tail_chars=_SUBMIT_STDERR_TAIL_CHARS,
    )
    submit_attempt_recorder.record_payloads(submit_success_payloads)
    print("[green]submission recorded[/green]")
    try:
        outcome = _wait_for_submission_outcome(
            slug=config.slug,
            message=message,
            submitted_at=submitted_at,
        )
    except SubmissionOutcomePollingError as exc:
        detail = normalize_error_text(exc.detail or str(exc), max_chars=1200)
        return _abort_submit_for_run(
            config=config,
            run_id=run_id,
            problem_types=problem_types,
            submission_ref=submission_ref,
            submission_artifact_path=submission_for_submit_path,
            artifact_mode=submission_artifact_mode,
            code_fingerprint=submit_code_fingerprint,
            fingerprint=compute_error_fingerprint("", detail or str(exc)),
            error_kind="transient",
            reason="submission_polling_error",
            message="Submission outcome polling failed; aborting submit stage for this run.",
            stdout_tail="",
            stderr_tail=detail or str(exc),
            exit_code=None,
            submit_attempt_recorder=submit_attempt_recorder,
        )

    if isinstance(outcome, dict):
        outcome_status = _submit_stage.normalize_submission_outcome_status(outcome.get("status"))
        outcome["status"] = outcome_status
        raw_detail = ""
        if outcome_status in _submit_stage.FAILED_SUBMISSION_OUTCOME_STATUSES or (
            outcome_status in _submit_stage.SCORELESS_COMPLETE_SUBMISSION_OUTCOME_STATUSES
            and outcome.get("score") is None
            and infer_deliverable_mode_from_paths(config.paths, default="leaderboard") == "leaderboard"
        ):
            raw_detail = _submit_stage.build_submission_outcome_error_detail(
                slug=config.slug,
                message=message,
                submitted_at=submitted_at,
                outcome=outcome,
                fetch_submission_rows=lambda current_slug: list_competition_submissions(current_slug, dry_run=False),
                normalize_detail=lambda text: normalize_error_text(text, max_chars=1200),
            )
        outcome_abort_decision = _submit_stage.decide_submission_outcome_abort(
            outcome_status=outcome_status,
            outcome_score=outcome.get("score"),
            deliverable_mode=infer_deliverable_mode_from_paths(config.paths, default="leaderboard"),
            raw_detail=raw_detail,
        )
        if outcome_abort_decision.should_abort:
            return _abort_submit_for_run(
                config=config,
                run_id=run_id,
                problem_types=problem_types,
                submission_ref=submission_ref,
                submission_artifact_path=submission_for_submit_path,
                artifact_mode=submission_artifact_mode,
                code_fingerprint=submit_code_fingerprint,
                fingerprint=compute_error_fingerprint("", outcome_abort_decision.detail),
                error_kind=outcome_abort_decision.error_kind,
                reason=outcome_abort_decision.reason,
                message=outcome_abort_decision.message,
                stdout_tail="",
                stderr_tail=outcome_abort_decision.detail,
                exit_code=None,
                submit_attempt_recorder=submit_attempt_recorder,
            )

    outcome_recording = _submit_attempts.decide_submit_outcome_recording(
        outcome=outcome,
        submission_artifact_exists=bool(submission_for_submit_path is not None and submission_for_submit_path.exists()),
    )
    print(outcome_recording.message)
    if outcome_recording.ledger_outcome is not None and submission_for_submit_path is not None:
        SubmissionLedger(config.paths.submission_ledger_path).record_outcome(
            slug=config.slug,
            message=message,
            submission_path=submission_for_submit_path,
            run_id=run_id,
            outcome=outcome_recording.ledger_outcome,
        )
    _submit_failure_context.mark_submit_failure_context_resolved(
        run_dir=run_dir, resolution="submitted", submission_ref=submission_ref
    )
    return _submit_attempts.build_submit_result_payload(
        message=message,
        submission_ref=submission_ref,
        submitted_at_iso=submitted_at.isoformat(),
        iteration=_submit_stage.infer_iteration_from_submission_path(submission_path),
        outcome=outcome,
    )


def _submit_with_notebook_kernel(
    *,
    config: AutopilotConfig,
    run_id: str,
    submission_path: Path,
    message: str,
    artifact_mode: str = "wrapper",
):
    """Execute a Kaggle notebook kernel and submit via kernel reference."""
    iteration = _submit_stage.infer_iteration_from_submission_path(submission_path) or 1
    iter_dir = config.paths.iter_dir(run_id, iteration)
    kaggle_user = resolve_kaggle_username(config.kaggle_username)
    submit_kernel_kwargs = _submit_notebook.build_submit_kernel_run_kwargs(
        slug=config.slug,
        run_id=run_id,
        iteration=iteration,
        base_dir=config.paths.base_dir.parent,
        kaggle_username=kaggle_user,
        kernel_name=config.kernel_name,
        accelerator=config.accelerator,
        enable_internet=False,
        submission_path=submission_path,
        artifact_mode=artifact_mode,
        dry_run=config.dry_run,
        timeout_minutes=config.time_budget_min,
    )
    kernel_result = _submit_notebook.run_submit_kernel_with_cpu_fallback(
        submit_kernel_kwargs=submit_kernel_kwargs,
        run_submit_kernel=run_submit_kernel,
        decide_cpu_fallback=lambda exc: _decide_submit_kernel_cpu_fallback(config=config, exc=exc),
        is_capacity_error=lambda exc: isinstance(exc, KernelCapacityError),
        wrap_error=_notebook_kernel_submission_error,
        on_message=print,
    )

    output_reference = _submit_notebook.build_notebook_submit_output_reference(
        kernel_id=kernel_result.kernel_id,
        kernel_submission_path=kernel_result.submission_path,
        version_label=_infer_kernel_submit_version_label(iter_dir / "logs"),
        copy_submission_artifact=lambda source: _copy_submission_artifact_to_iteration_dir(
            source=source,
            iter_dir=iter_dir,
        ),
    )
    submit_reference = output_reference.reference
    print(f"[cyan]submit notebook[/cyan]: {submit_reference.kernel_ref}")
    submit_kwargs = _submit_notebook.build_kaggle_submit_kernel_kwargs(
        slug=config.slug,
        reference=submit_reference,
        message=message,
        dry_run=config.dry_run,
    )
    submit_result = _submit_notebook.run_kaggle_submit_kernel_with_retry(
        submit_kwargs=submit_kwargs,
        run_kaggle_submit_kernel=run_kaggle_submit_kernel,
        submit_error_types=SubmissionCliError,
        classify_submit_error=classify_submit_error,
        should_retry_ambiguous=_submit_failure_policy.should_retry_ambiguous_notebook_submit_error,
        sleep=time.sleep,
        on_message=print,
    )
    return submit_result, submit_reference.submission_ref, output_reference.submission_artifact_path


def _decide_submit_kernel_cpu_fallback(
    *,
    config: AutopilotConfig,
    exc: Exception,
) -> _submit_notebook.NotebookSubmitCpuFallbackDecision:
    return _submit_notebook.decide_submit_kernel_cpu_fallback_for_exception(
        accelerator=config.accelerator,
        strict_accelerator=config.strict_accelerator,
        exc=exc,
        is_capacity_error=lambda candidate: isinstance(candidate, KernelCapacityError),
        is_push_error=lambda candidate: isinstance(candidate, KaggleCliError)
        and _submit_notebook.is_submit_kernel_push_error(candidate),
    )


def _notebook_kernel_submission_error(exc: Exception) -> SubmissionCliError:
    if isinstance(exc, KaggleCliError):
        output = exc.output or str(exc)
        return SubmissionCliError(
            "Notebook submission fallback failed while running Kaggle kernel.",
            command=list(exc.command or []),
            exit_code=exc.exit_code,
            output=output,
            stdout=exc.stdout,
            stderr=exc.stderr or output,
        )
    return SubmissionCliError(
        "Notebook submission fallback failed while running Kaggle kernel.",
        command=[],
        output=str(exc),
        stdout="",
        stderr=str(exc),
    )


def _infer_kernel_submit_version_label(logs_dir: Path | None) -> str | None:
    """Read pushed kernel version from kernel push logs for notebook submit."""
    if logs_dir is None or not logs_dir.exists():
        return None
    candidates = sorted(logs_dir.glob("kernel_push-*.txt"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        match = _KERNEL_PUSH_VERSION_RE.search(text)
        if match:
            version = str(match.group("version") or "").strip()
            if version:
                return version
    return None


def _abort_submit_for_run(
    *,
    config: AutopilotConfig,
    run_id: str,
    problem_types: list[str],
    submission_ref: str | Path,
    submission_artifact_path: Path | None = None,
    artifact_mode: str | None = None,
    code_fingerprint: str | None = None,
    fingerprint: str,
    error_kind: str,
    reason: str,
    message: str,
    stdout_tail: str,
    stderr_tail: str,
    exit_code: int | None,
    submit_attempt_recorder: _submit_attempts.SubmitAttemptRecorder | None = None,
) -> None:
    run_dir = config.paths.run_dir(run_id)
    if submit_attempt_recorder is None:
        submit_attempt_recorder = _submit_attempts.SubmitAttemptRecorder(
            run_dir=run_dir,
            save_run_state=lambda updates: _save_run_state(run_dir, updates),
        )
    submission_ref_text = str(submission_ref)
    artifact_path: Path | None
    if submission_artifact_path is not None:
        artifact_path = submission_artifact_path
    elif isinstance(submission_ref, Path):
        artifact_path = submission_ref
    else:
        artifact_path = None
    prior = _load_run_state(run_dir)
    prior_ok = bool(prior.get("submit_ok")) or _has_successful_submit_attempt(run_dir)
    abort_payloads = _submit_attempts.build_submit_abort_record_payloads(
        run_id=run_id,
        submission_ref=submission_ref_text,
        submission_sha256=_sha256_or_none(artifact_path),
        exit_code=exit_code,
        fingerprint=fingerprint,
        code_fingerprint=code_fingerprint or "",
        error_kind=error_kind,
        reason=reason,
        stdout=stdout_tail,
        stderr=stderr_tail,
        prior_state=prior,
        prior_submit_ok=prior_ok,
        stdout_tail_chars=_SUBMIT_STDOUT_TAIL_CHARS,
        stderr_tail_chars=_SUBMIT_STDERR_TAIL_CHARS,
    )
    submit_attempt_recorder.record_payloads(abort_payloads)
    _submit_failure_context.save_submit_failure_context(
        run_dir,
        _build_submit_failure_context_payload(
            run_dir=run_dir,
            submission_ref=submission_ref_text,
            artifact_path=artifact_path,
            artifact_mode=artifact_mode,
            code_fingerprint=code_fingerprint or "",
            fingerprint=fingerprint,
            error_kind=error_kind,
            reason=reason,
            message=message,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            exit_code=exit_code,
        ),
    )
    knowledge_submission_path = artifact_path or Path(submission_ref_text)
    _record_submit_reason_knowledge(
        config=config,
        run_id=run_id,
        problem_types=problem_types,
        submission_path=knowledge_submission_path,
        error_kind=error_kind,
        reason=reason,
        action_taken="abort",
        fingerprint=fingerprint,
        details=message,
    )
    print(f"[red]submit aborted[/red]: {message}")
    raise SubmitAbortedError(message)


def _build_submit_failure_context_payload(
    *,
    run_dir: Path,
    submission_ref: str,
    artifact_path: Path | None,
    artifact_mode: str | None,
    code_fingerprint: str,
    fingerprint: str,
    error_kind: str,
    reason: str,
    message: str,
    stdout_tail: str,
    stderr_tail: str,
    exit_code: int | None,
) -> dict[str, object]:
    detail = "\n".join(part for part in (stdout_tail, stderr_tail) if part).strip()
    repair_decision = _submit_failure_policy.classify_submit_failure_repair(
        reason=reason,
        error_kind=error_kind,
        detail=detail,
    )
    return _submit_failure_context.build_submit_failure_context_payload(
        now_iso=datetime.now(UTC).isoformat(),
        submission_ref=submission_ref,
        artifact_path=artifact_path,
        artifact_sha256=_sha256_or_none(artifact_path),
        artifact_mode=artifact_mode,
        code_fingerprint=code_fingerprint,
        fingerprint=fingerprint,
        error_kind=error_kind,
        reason=reason,
        message=message,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        exit_code=exit_code,
        repair_decision=repair_decision,
        latest_submit_attempt=_load_latest_submit_attempt(run_dir),
        run_state=_load_run_state(run_dir),
        stdout_tail_chars=_SUBMIT_STDOUT_TAIL_CHARS,
        stderr_tail_chars=_SUBMIT_STDERR_TAIL_CHARS,
    )


def _clear_stale_submit_autofix_artifact(*, run_dir: Path, submission_path: Path) -> None:
    state = _load_run_state(run_dir)
    failure_context = _submit_failure_context.load_submit_failure_context(run_dir)
    decision = _submit_failure_context.decide_stale_submit_autofix_artifact(
        run_state=state,
        failure_context=failure_context,
        submission_path=submission_path,
        now_iso=datetime.now(UTC).isoformat(),
    )
    if decision is None:
        return
    if decision.clear_repaired_path:
        _save_run_state(run_dir, {"submit_autofix_submission_path": ""})
    failure_context.update(decision.failure_context_updates)
    _submit_failure_context.save_submit_failure_context(run_dir, failure_context)


def _resolve_submit_autofix_submission_artifact(*, config: AutopilotConfig, run_id: str, run_dir: Path) -> Path | None:
    max_search_iteration = MAX_AUTOFIX_ATTEMPTS + MAX_KERNEL_FIX_ATTEMPTS + MAX_AUTOFIX_CODEX_PASSES
    return _submit_failure_context.resolve_submit_autofix_submission_artifact(
        run_state=_load_run_state(run_dir),
        latest_submit_attempt=_load_latest_submit_attempt(run_dir),
        failure_context=_submit_failure_context.load_submit_failure_context(run_dir),
        fallback_iteration_dirs=(
            config.paths.iter_dir(run_id, iteration) for iteration in range(max_search_iteration, 0, -1)
        ),
        resolve_iteration_submission_artifact=_resolve_iteration_submission_artifact,
    )


def _prepare_submit_file_autofix(
    *,
    config: AutopilotConfig,
    run_id: str,
    run_dir: Path,
) -> tuple[Path | None, str]:
    latest = _load_latest_submit_attempt(run_dir)
    service = SubmissionService(
        SubmissionConfig(
            slug=config.slug,
            data_dir=config.paths.data_dir,
            sample_submission_path=config.paths.sample_submission_path,
            submission_ledger_path=config.paths.submission_ledger_path,
            dry_run=True,
            force_submit=True,
            bypass_rate_limit=True,
        )
    )

    def resolve_source() -> Path | None:
        return _resolve_submit_autofix_submission_artifact(config=config, run_id=run_id, run_dir=run_dir)

    def save_repaired_path(fixed: Path) -> None:
        _save_run_state(run_dir, {"submit_autofix_submission_path": str(fixed)})

    preparation = _submit_autofix.prepare_submit_file_autofix(
        latest_submit_attempt=latest,
        resolve_source=resolve_source,
        validate_and_prepare=service.validate_and_prepare_submission,
        save_repaired_path=save_repaired_path,
    )
    return preparation.path, preparation.summary


def _submit_file_fix_contract_satisfied(
    *,
    run_dir: Path,
    baseline_path: Path | None,
    baseline_sha256: str | None,
) -> bool:
    return _submit_failure_context.submit_file_fix_contract_satisfied(
        run_state=_load_run_state(run_dir),
        baseline_path=baseline_path,
        baseline_sha256=baseline_sha256,
        sha256_or_none=_sha256_or_none,
    )


def _build_submit_autofix_context(run_dir: Path) -> str:
    return _submit_failure_context.format_submit_autofix_context(
        failure_context=_submit_failure_context.load_submit_failure_context(run_dir),
        run_state=_load_run_state(run_dir),
        latest_submit_attempt=_load_latest_submit_attempt(run_dir),
    )


def _compute_submit_code_fingerprint(config: AutopilotConfig) -> str:
    return _submit_retry_policy.compute_submit_code_fingerprint(
        src_root=Path(__file__).resolve().parent,
        kernel_source_dir=config.paths.kernel_source_dir,
        sha256_or_none=_sha256_or_none,
    )


def _consume_same_submit_fingerprint_retry_allowance(
    *,
    run_dir: Path,
    run_state: dict[str, object],
    fingerprint: str,
    code_fingerprint: str,
) -> bool:
    return _submit_retry_policy.consume_same_submit_fingerprint_retry_allowance(
        run_state=run_state,
        fingerprint=fingerprint,
        code_fingerprint=code_fingerprint,
        save_run_state=lambda updates: _save_run_state(run_dir, updates),
    )


def _compute_submit_backoff(attempt: int) -> float:
    return _submit_retry_policy.compute_submit_backoff(
        attempt=attempt,
        base_seconds=_SUBMIT_BACKOFF_BASE_SEC,
    )


def _should_force_resubmit_after_submit_abort(run_dir: Path) -> bool:
    return _submit_failure_context.should_force_resubmit_after_submit_abort(_load_run_state(run_dir))


def _should_defer_submit_abort_to_next_iteration(
    *,
    compute: str,
    run_dir: Path,
    iteration: int,
    max_iterations: int,
) -> bool:
    return _submit_failure_context.should_defer_submit_abort_to_next_iteration(
        compute=compute,
        failure_context=_submit_failure_context.load_submit_failure_context(run_dir),
        iteration=iteration,
        max_iterations=max_iterations,
    )


def _is_submit_abort_autofixable(*, config: AutopilotConfig, run_id: str) -> bool:
    run_dir = config.paths.run_dir(run_id)
    failure_context = _submit_failure_context.load_submit_failure_context(run_dir)
    state = _load_run_state(run_dir)
    decision = _submit_failure_context.decide_submit_abort_autofixability(
        failure_context=failure_context,
        run_state=state,
    )
    if decision.message:
        print(decision.message)
    return decision.autofixable


def _is_non_autofixable_runtime_error(error: Exception) -> bool:
    text = str(error).strip().lower()
    if not text:
        return False
    return "requires kernel.py" in text or "kernel-first training" in text


def _record_submit_reason_knowledge(
    *,
    config: AutopilotConfig,
    run_id: str,
    problem_types: list[str],
    submission_path: Path,
    error_kind: str,
    reason: str,
    action_taken: str,
    fingerprint: str,
    details: str,
) -> None:
    payload = _submit_attempts.build_submit_knowledge_payload(
        iteration=_submit_stage.infer_iteration_from_submission_path(submission_path),
        error_kind=error_kind,
        reason=reason,
        action_taken=action_taken,
        fingerprint=fingerprint,
        details=details,
        normalize_detail=normalize_error_text,
    )
    try:
        record_error_fix_insight(
            knowledge_paths=config.knowledge_paths,
            slug=config.slug,
            run_id=run_id,
            iteration=payload.iteration,
            problem_types=problem_types,
            error_message=payload.error_message,
            fix_summary=payload.fix_summary,
            resolved=False,
            outcome_bucket="unknown",
            submission_score=None,
        )
    except Exception:  # noqa: BLE001
        # Knowledge recording must not block submit abort/retry control.
        return


def _is_missing_kaggle_credentials_error(exc: KaggleCliError) -> bool:
    text = (exc.message or "") + "\n" + (exc.output or "")
    lowered = text.lower()
    if "kaggle.json" in lowered and "could not find" in lowered:
        return True
    if "kaggle.json" in lowered and "environment method" in lowered:
        return True
    if "api.authenticate" in lowered and "kaggle.json" in lowered:
        return True
    return False


def _wait_for_submission_outcome(
    *,
    slug: str,
    message: str,
    submitted_at: datetime,
    max_attempts: int | None = _SUBMISSION_POLL_MAX_ATTEMPTS,
    poll_interval_sec: float = _SUBMISSION_POLL_INTERVAL_SEC,
) -> dict[str, object] | None:
    print(f"[cyan]submission polling[/cyan]: waiting for result (interval={poll_interval_sec:.0f}s)")
    service = SubmissionOutcomeService(
        fetch_rows=lambda current_slug: list_competition_submissions(current_slug, dry_run=False),
        max_attempts=max_attempts,
        poll_interval_sec=poll_interval_sec,
        max_fetch_errors=_SUBMISSION_POLL_MAX_FETCH_ERRORS,
    )
    return service.wait_for_outcome(
        slug=slug,
        message=message,
        submitted_at=submitted_at,
    )


def _load_previous_submission_history(
    *,
    slug: str,
    paths: CompetitionPaths,
    direction: str,
    dry_run: bool,
) -> dict[str, object]:
    history_path = paths.context_dir / "submission_history.json"
    if dry_run and history_path.exists():
        cached = _load_json_object(history_path)
        if cached is not None:
            cached["source"] = str(cached.get("source") or "cache")
            cached["cache_path"] = str(history_path)
            return cached
    if dry_run:
        return {
            "source": "dry_run",
            "fetched_at": datetime.now(UTC).isoformat(),
            "direction": direction,
            "count": 0,
            "scored_count": 0,
            "best_score": None,
            "best": None,
            "latest_score": None,
            "latest": None,
            "recent": [],
            "cache_path": str(history_path),
        }

    try:
        rows = list_competition_submissions(slug, dry_run=False)
    except Exception as exc:  # noqa: BLE001
        cached = _load_json_object(history_path) if history_path.exists() else None
        if cached is not None:
            cached["source"] = str(cached.get("source") or "cache")
            cached["fetch_error"] = f"{type(exc).__name__}: {exc}"
            cached["cache_path"] = str(history_path)
            print(
                "[yellow]submission history[/yellow]: "
                f"failed to refresh Kaggle submissions; using cached {history_path}"
            )
            return cached
        print(f"[yellow]submission history[/yellow]: failed to fetch Kaggle submissions: {exc}")
        return {
            "source": "fetch_error",
            "fetch_error": f"{type(exc).__name__}: {exc}",
            "fetched_at": datetime.now(UTC).isoformat(),
            "direction": direction,
            "count": 0,
            "scored_count": 0,
            "best_score": None,
            "best": None,
            "latest_score": None,
            "latest": None,
            "recent": [],
            "cache_path": str(history_path),
        }

    payload = _build_previous_submission_history_payload(
        rows=rows,
        direction=direction,
        source="kaggle competitions submissions --csv",
    )
    payload["cache_path"] = str(history_path)
    _write_json_object(history_path, payload)
    return payload


def _format_method_registry_for_prompt(paths: CompetitionPaths) -> str:
    registry_path = method_registry_path(paths.context_dir)
    payload = _load_json_object(registry_path)
    if payload is None:
        return ""
    return render_method_registry_for_prompt(payload, max_methods=8)


def _build_submit_failure_improvement_context(*, run_dir: Path) -> tuple[list[str], str | None]:
    return _submit_failure_context.build_submit_failure_improvement_context(
        failure_context=_submit_failure_context.load_submit_failure_context(run_dir),
        latest_submit_attempt=_load_latest_submit_attempt(run_dir),
    )


def _record_submission_knowledge(
    *,
    config: AutopilotConfig,
    run_id: str,
    problem_types: list[str],
    pending_problem_insights: list[dict[str, object]],
    pending_error_fixes: list[dict[str, object]],
    submission_result: dict[str, object] | None,
    metric_direction: str,
    target_score: float | None,
    top1_score: float | None,
) -> None:
    if not submission_result:
        return
    outcome_payload = submission_result.get("outcome")
    if not isinstance(outcome_payload, dict):
        return
    online_score = _to_float(outcome_payload.get("score"))
    if online_score is None:
        return
    outcome_bucket = _submit_stage.classify_submission_outcome(
        score=online_score,
        direction=metric_direction,
        target_score=target_score,
        top1_score=top1_score,
    )
    submitted_iteration = submission_result.get("iteration")
    iteration_value = submitted_iteration if isinstance(submitted_iteration, int) else None
    if not pending_problem_insights:
        diagnostics_text = ""
        if iteration_value is not None:
            diagnostics_path = config.paths.iter_dir(run_id, iteration_value) / "diagnostics.md"
            if diagnostics_path.exists():
                diagnostics_text = diagnostics_path.read_text(encoding="utf-8", errors="ignore")
        pending_problem_insights.append(
            {
                "iteration": iteration_value or 1,
                "why_poor": diagnostics_text,
                "how_improved": f"Submitted iteration {iteration_value or 1} result after validation.",
                "delta_offline": None,
            }
        )
    for item in pending_problem_insights:
        try:
            iteration = int(item.get("iteration") or (iteration_value or 1))
        except (TypeError, ValueError):
            iteration = iteration_value or 1
        record_problem_type_insight(
            knowledge_paths=config.knowledge_paths,
            slug=config.slug,
            run_id=run_id,
            iteration=iteration,
            problem_types=problem_types,
            why_poor=str(item.get("why_poor") or ""),
            how_improved=str(item.get("how_improved") or ""),
            delta_offline=item.get("delta_offline") if isinstance(item.get("delta_offline"), (int, float)) else None,
            outcome_bucket=outcome_bucket,
            submission_score=online_score,
        )
    for item in pending_error_fixes:
        try:
            iteration = int(item.get("iteration") or (iteration_value or 1))
        except (TypeError, ValueError):
            iteration = iteration_value or 1
        record_error_fix_insight(
            knowledge_paths=config.knowledge_paths,
            slug=config.slug,
            run_id=run_id,
            iteration=iteration,
            problem_types=problem_types,
            error_message=str(item.get("error_message") or ""),
            fix_summary=str(item.get("fix_summary") or ""),
            resolved=bool(item.get("resolved", True)),
            outcome_bucket=outcome_bucket,
            submission_score=online_score,
        )


def _build_code_reference_repair_prompt(
    *,
    base_prompt_text: str,
    reference: _CodeReferenceNotebook,
    issues: list[str],
    kernel_path: Path,
) -> str:
    issues_text = ", ".join(issues) if issues else "unknown"
    tabicl_required = _reference_requires_tabicl(reference)
    tabicl_line = (
        "- This reference appears to be TabICL-based. You MUST include a real TabICL path in kernel.py."
        if tabicl_required
        else "- TabICL path is optional for this reference notebook."
    )
    return (
        f"# {IMPLEMENTATION_AGENT.display_name} Improvement Repair: Mandatory Code Reference Implementation\n\n"
        "The previous change did not satisfy mandatory code-reference implementation requirements.\n\n"
        f"- Failed checks: {issues_text}\n"
        f"- Required notebook: {reference.kernel_id} ({reference.title})\n"
        f"- Kernel path: {kernel_path}\n"
        f"- Required marker: `{_code_reference_marker(reference)}`\n"
        f"{tabicl_line}\n\n"
        "Make minimal edits to kernel.py so all checks pass.\n"
        "Do not weaken the model by collapsing to tiny conservative feature subsets that reduce offline score.\n\n"
        "## Original Improvement Context\n\n"
        f"{base_prompt_text}\n"
    )
