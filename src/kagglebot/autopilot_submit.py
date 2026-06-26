from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from rich import print

from kagglebot import autopilot_state as _autopilot_state
from kagglebot import competition_rules as _competition_rules
from kagglebot import context_artifacts as _context_artifacts
from kagglebot import env_utils as _env_utils
from kagglebot import kaggle_cli_errors as _kaggle_cli_errors
from kagglebot import submit_failure_policy as _submit_failure_policy
from kagglebot import submit_notebook as _submit_notebook
from kagglebot import submit_retry_policy as _submit_retry_policy
from kagglebot import submit_runner as _submit_runner
from kagglebot.exceptions import SubmitAbortedError
from kagglebot.hashing import sha256_file_or_none as _sha256_or_none
from kagglebot.kaggle_api import check_rules_accepted, list_competition_submissions
from kagglebot.kernel_runner import resolve_kaggle_username, run_submit_kernel
from kagglebot.knowledge import record_error_fix_insight
from kagglebot.submission.guard import (
    classify_submit_error,
    compute_error_fingerprint,
    normalize_error_text,
    run_kaggle_submit_kernel,
)
from kagglebot.writeup import infer_code_competition_from_paths, infer_deliverable_mode_from_paths

_SUBMISSION_POLL_MAX_ATTEMPTS: int | None = None
_SUBMISSION_POLL_INTERVAL_SEC = 30.0
_SUBMISSION_POLL_MAX_FETCH_ERRORS = 3
_SUBMIT_MAX_TRANSIENT_RETRIES = 3
_SUBMIT_BACKOFF_BASE_SEC = 2.0
_SUBMIT_STDERR_TAIL_CHARS = 1200
_SUBMIT_STDOUT_TAIL_CHARS = 1200


class AutopilotSubmitConfig(Protocol):
    submit: bool
    dry_run: bool
    slug: str
    paths: object
    knowledge_paths: object
    force_submit: bool
    message: str | None
    campaign_mode: str
    target_direction: str | None
    kaggle_username: str | None
    kernel_name: str | None
    accelerator: str
    strict_accelerator: bool
    time_budget_min: int | None


def build_autopilot_submit_dependencies(
    *,
    check_rules_accepted_func: Callable[..., bool] = check_rules_accepted,
    infer_code_competition_from_paths_func: Callable[..., bool] = infer_code_competition_from_paths,
    resolve_kaggle_username_func: Callable[..., str] = resolve_kaggle_username,
    run_submit_kernel_func: Callable[..., object] = run_submit_kernel,
    run_kaggle_submit_kernel_func: Callable[..., object] = run_kaggle_submit_kernel,
    classify_submit_error_func: Callable[..., dict[str, object]] = classify_submit_error,
    compute_error_fingerprint_func: Callable[..., str] = compute_error_fingerprint,
    normalize_error_text_func: Callable[..., str] = normalize_error_text,
    record_error_fix_insight_func: Callable[..., object] = record_error_fix_insight,
    list_competition_submissions_func: Callable[..., list[dict[str, object]]] = list_competition_submissions,
    deliverable_mode_func: Callable[..., str] = infer_deliverable_mode_from_paths,
) -> _submit_runner.SubmitRunnerDependencies:
    """Build concrete side-effect dependencies for the submit runner."""
    return _submit_runner.SubmitRunnerDependencies(
        load_competition_rule_constraints=_competition_rules.load_competition_rule_constraints,
        env_truthy=_env_utils.env_truthy,
        load_run_state=_autopilot_state.load_run_state,
        save_run_state=_autopilot_state.save_run_state,
        compute_submit_code_fingerprint=_submit_retry_policy.compute_submit_code_fingerprint,
        compute_submission_sha256=_sha256_or_none,
        now_iso=lambda: datetime.now(UTC).isoformat(),
        now_datetime=lambda: datetime.now(UTC),
        normalize_error_text=normalize_error_text_func,
        record_error_fix_insight=record_error_fix_insight_func,
        build_error=SubmitAbortedError,
        check_rules_accepted=check_rules_accepted_func,
        infer_code_competition_from_paths=infer_code_competition_from_paths_func,
        collect_duplicate_submission_sources=_submit_retry_policy.collect_duplicate_submission_sources,
        decide_duplicate_submission_action=_submit_retry_policy.decide_duplicate_submission_action,
        decide_same_submission_path_action=_submit_retry_policy.decide_same_submission_path_action,
        resolve_notebook_submit_artifact_mode=_submit_notebook.resolve_notebook_submit_artifact_mode,
        decide_notebook_submit_artifact_mode_for_paths=_submit_notebook.decide_notebook_submit_artifact_mode_for_paths,
        count_csv_data_rows=_context_artifacts.count_csv_data_rows_capped,
        resolve_kaggle_username=resolve_kaggle_username_func,
        run_submit_kernel=run_submit_kernel_func,
        run_kaggle_submit_kernel=run_kaggle_submit_kernel_func,
        copy_submission_artifact_to_iteration_dir=_autopilot_state.copy_submission_artifact_to_iteration_dir,
        classify_submit_error=classify_submit_error_func,
        should_retry_ambiguous_notebook_submit_error=(
            _submit_failure_policy.should_retry_ambiguous_notebook_submit_error
        ),
        should_use_notebook_submit_fallback=_submit_failure_policy.should_use_notebook_submit_fallback,
        compute_error_fingerprint=compute_error_fingerprint_func,
        decide_submit_fingerprint_reuse=_submit_retry_policy.decide_submit_fingerprint_reuse,
        compute_submit_backoff=_submit_retry_policy.compute_submit_backoff,
        is_missing_kaggle_credentials_error=_kaggle_cli_errors.is_missing_kaggle_credentials_error,
        deliverable_mode=lambda paths: deliverable_mode_func(paths, default="leaderboard"),
        list_competition_submissions=list_competition_submissions_func,
        sleep=time.sleep,
        on_message=print,
    )


def build_autopilot_submit_limits() -> _submit_runner.SubmitRunnerLimits:
    """Build the default submit retry/poll limits used by autopilot."""
    return _submit_runner.SubmitRunnerLimits(
        stdout_tail_chars=_SUBMIT_STDOUT_TAIL_CHARS,
        stderr_tail_chars=_SUBMIT_STDERR_TAIL_CHARS,
        max_transient_retries=_SUBMIT_MAX_TRANSIENT_RETRIES,
        backoff_base_sec=_SUBMIT_BACKOFF_BASE_SEC,
        poll_max_attempts=_SUBMISSION_POLL_MAX_ATTEMPTS,
        poll_interval_sec=_SUBMISSION_POLL_INTERVAL_SEC,
        poll_max_fetch_errors=_SUBMISSION_POLL_MAX_FETCH_ERRORS,
    )


def attempt_submit_for_autopilot_run(
    *,
    config: AutopilotSubmitConfig,
    run_id: str,
    submission_path: Path,
    best_score: float | None,
    problem_types: list[str],
    submit_mode: str = "file",
    notebook_submit_artifact_mode: str = "wrapper",
    deps: _submit_runner.SubmitRunnerDependencies | None = None,
    limits: _submit_runner.SubmitRunnerLimits | None = None,
) -> dict[str, object] | None:
    """Run autopilot submission with concrete production dependencies."""
    return _submit_runner.attempt_submit_for_run(
        config=config,
        run_id=run_id,
        submission_path=submission_path,
        best_score=best_score,
        problem_types=problem_types,
        submit_mode=submit_mode,
        notebook_submit_artifact_mode=notebook_submit_artifact_mode,
        deps=deps or build_autopilot_submit_dependencies(),
        limits=limits or build_autopilot_submit_limits(),
    )
