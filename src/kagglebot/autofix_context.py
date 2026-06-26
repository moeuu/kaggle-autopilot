from __future__ import annotations

import shlex
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from kagglebot import agent_io as _agent_io
from kagglebot import autopilot_state as _autopilot_state
from kagglebot import submit_autofix as _submit_autofix
from kagglebot import submit_failure_context as _submit_failure_context
from kagglebot.exceptions import KaggleCliError, SubmitAbortedError
from kagglebot.submission_service import SubmissionConfig, SubmissionService


class AutofixContextConfig(Protocol):
    slug: str
    paths: object


@dataclass(frozen=True)
class AutofixPreparedContext:
    run_dir: Path
    autofix_dir: Path
    error_text: str
    error_path: Path
    submit_autofix: bool
    submit_context: str
    submit_file_fix_required: bool
    submit_file_fix_baseline_path: Path | None
    submit_file_fix_baseline_sha256: str | None


def prepare_autofix_context(
    *,
    config: AutofixContextConfig,
    run_id: str,
    attempt: int,
    error: Exception,
    max_search_iteration: int,
    sha256_or_none: Callable[[Path | None], str | None],
) -> AutofixPreparedContext:
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
        submit_autofix_context = _submit_failure_context.load_submit_autofix_run_context(
            run_dir=run_dir,
            load_run_state=_autopilot_state.load_run_state,
        )
        failure_context = submit_autofix_context.failure_context
        run_state = submit_autofix_context.run_state
        latest_submit_attempt = submit_autofix_context.latest_submit_attempt
        submit_context = submit_autofix_context.formatted_context
        submit_file_fix_required = _submit_autofix.submit_file_fix_required_for_attempt(latest_submit_attempt)

        def fallback_iteration_dirs():
            return (config.paths.iter_dir(run_id, iteration) for iteration in range(max_search_iteration, 0, -1))

        def save_repaired_submit_path(fixed: Path) -> None:
            _submit_failure_context.save_submit_autofix_repaired_path_for_run(
                run_dir=run_dir,
                repaired_path=fixed,
                save_run_state_for_run=_autopilot_state.save_run_state,
            )

        if submit_file_fix_required:
            submit_file_fix_baseline_path = _submit_failure_context.resolve_submit_autofix_submission_artifact(
                run_state=run_state,
                latest_submit_attempt=latest_submit_attempt,
                failure_context=failure_context,
                fallback_iteration_dirs=fallback_iteration_dirs(),
                resolve_iteration_submission_artifact=_autopilot_state.resolve_iteration_submission_artifact,
            )
            submit_file_fix_baseline_sha256 = sha256_or_none(submit_file_fix_baseline_path)
        repair_service = SubmissionService(
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
        preparation = _submit_autofix.prepare_submit_file_autofix_for_run(
            latest_submit_attempt=latest_submit_attempt,
            run_state=run_state,
            failure_context=failure_context,
            fallback_iteration_dirs=fallback_iteration_dirs,
            resolve_iteration_submission_artifact=_autopilot_state.resolve_iteration_submission_artifact,
            validate_and_prepare=repair_service.validate_and_prepare_submission,
            save_repaired_path=save_repaired_submit_path,
        )
        _prepared_submission_path, prepared_submission_summary = preparation.path, preparation.summary
        if prepared_submission_summary:
            submit_context = (
                f"{submit_context}\n\ndeterministic_submit_file_autofix:\n{prepared_submission_summary}".strip()
            )
            error_text = f"{error_text}\n\nDeterministic Submit File Autofix:\n{prepared_submission_summary}"
        if submit_context:
            error_text = f"{error_text}\n\nSubmit Failure Context:\n{submit_context}"

    error_path = _agent_io.write_autofix_error_transcript(
        autofix_dir=autofix_dir,
        attempt=attempt,
        error_text=error_text,
    )
    return AutofixPreparedContext(
        run_dir=run_dir,
        autofix_dir=autofix_dir,
        error_text=error_text,
        error_path=error_path,
        submit_autofix=submit_autofix,
        submit_context=submit_context,
        submit_file_fix_required=submit_file_fix_required,
        submit_file_fix_baseline_path=submit_file_fix_baseline_path,
        submit_file_fix_baseline_sha256=submit_file_fix_baseline_sha256,
    )
