from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kagglebot import submit_attempts as _submit_attempts
from kagglebot import submit_failure_context as _submit_failure_context
from kagglebot import submit_stage_duplicate as _submit_stage_duplicate
from kagglebot.submit_cli_error_resolution import SubmitStageErrorActionDecision


@dataclass(frozen=True)
class SubmitRunAborter:
    run_dir: Path
    run_id: str
    slug: str
    knowledge_paths: object
    problem_types: list[str]
    save_run_state_for_run: Callable[[Path, dict[str, object]], object]
    resolve_submit_abort_artifact_path: Callable[..., Path | None]
    persist_submit_abort_failure: Callable[..., object]
    load_run_state: Callable[[Path], dict[str, object]]
    load_latest_submit_attempt: Callable[[Path], dict[str, object]]
    has_successful_submit_attempt: Callable[[Path], bool]
    compute_submission_sha256: Callable[[Path | None], str | None]
    stdout_tail_chars: int
    stderr_tail_chars: int
    now_iso: Callable[[], str]
    normalize_detail: Callable[..., str]
    record_error_fix_insight: Callable[..., object]
    on_message: Callable[[str], object]
    build_error: Callable[[str], BaseException]

    def abort(
        self,
        *,
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
        submit_attempt_recorder: object | None,
    ) -> None:
        abort_submit_for_run(
            run_dir=self.run_dir,
            run_id=self.run_id,
            slug=self.slug,
            knowledge_paths=self.knowledge_paths,
            problem_types=self.problem_types,
            submission_ref=submission_ref,
            submission_artifact_path=submission_artifact_path,
            artifact_mode=artifact_mode,
            code_fingerprint=code_fingerprint,
            fingerprint=fingerprint,
            error_kind=error_kind,
            reason=reason,
            message=message,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            exit_code=exit_code,
            submit_attempt_recorder=submit_attempt_recorder,
            save_run_state=lambda updates: self.save_run_state_for_run(self.run_dir, updates),
            resolve_submit_abort_artifact_path=self.resolve_submit_abort_artifact_path,
            persist_submit_abort_failure=self.persist_submit_abort_failure,
            load_run_state=self.load_run_state,
            load_latest_submit_attempt=self.load_latest_submit_attempt,
            has_successful_submit_attempt=self.has_successful_submit_attempt,
            compute_submission_sha256=self.compute_submission_sha256,
            stdout_tail_chars=self.stdout_tail_chars,
            stderr_tail_chars=self.stderr_tail_chars,
            now_iso=self.now_iso(),
            normalize_detail=self.normalize_detail,
            record_error_fix_insight=self.record_error_fix_insight,
            on_message=self.on_message,
            build_error=self.build_error,
        )


def build_submit_run_aborter_for_run(
    *,
    run_dir: Path,
    run_id: str,
    slug: str,
    knowledge_paths: object,
    problem_types: list[str],
    save_run_state_for_run: Callable[[Path, dict[str, object]], object],
    load_run_state: Callable[[Path], dict[str, object]],
    compute_submission_sha256: Callable[[Path | None], str | None],
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    now_iso: Callable[[], str],
    normalize_detail: Callable[..., str],
    record_error_fix_insight: Callable[..., object],
    on_message: Callable[[str], object],
    build_error: Callable[[str], BaseException],
) -> SubmitRunAborter:
    return SubmitRunAborter(
        run_dir=run_dir,
        run_id=run_id,
        slug=slug,
        knowledge_paths=knowledge_paths,
        problem_types=problem_types,
        save_run_state_for_run=save_run_state_for_run,
        resolve_submit_abort_artifact_path=_submit_failure_context.resolve_submit_abort_artifact_path,
        persist_submit_abort_failure=_submit_failure_context.persist_submit_abort_failure,
        load_run_state=load_run_state,
        load_latest_submit_attempt=_submit_attempts.load_latest_submit_attempt,
        has_successful_submit_attempt=_submit_attempts.has_successful_submit_attempt,
        compute_submission_sha256=compute_submission_sha256,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
        now_iso=now_iso,
        normalize_detail=normalize_detail,
        record_error_fix_insight=record_error_fix_insight,
        on_message=on_message,
        build_error=build_error,
    )


@dataclass(frozen=True)
class SubmitRunRetryRecorder:
    submit_attempt_recorder: object
    run_id: str
    slug: str
    problem_types: list[str]
    knowledge_paths: object
    compute_submission_sha256: Callable[[Path | None], str | None]
    stdout_tail_chars: int
    stderr_tail_chars: int
    normalize_detail: Callable[..., str]
    record_error_fix_insight: Callable[..., object]

    def record(
        self,
        *,
        submission_ref: str,
        submission_artifact_path: Path | None,
        fallback_submission_path: Path,
        exit_code: int | None,
        fingerprint: str,
        action: SubmitStageErrorActionDecision,
        stdout: str,
        stderr: str,
        attempt: int,
    ) -> bool:
        return record_submit_stage_retry_attempt(
            submit_attempt_recorder=self.submit_attempt_recorder,
            run_id=self.run_id,
            slug=self.slug,
            problem_types=self.problem_types,
            submission_ref=submission_ref,
            submission_artifact_path=submission_artifact_path,
            fallback_submission_path=fallback_submission_path,
            compute_submission_sha256=self.compute_submission_sha256,
            exit_code=exit_code,
            fingerprint=fingerprint,
            action=action,
            stdout=stdout,
            stderr=stderr,
            attempt=attempt,
            stdout_tail_chars=self.stdout_tail_chars,
            stderr_tail_chars=self.stderr_tail_chars,
            knowledge_paths=self.knowledge_paths,
            normalize_detail=self.normalize_detail,
            record_error_fix_insight=self.record_error_fix_insight,
        )


def record_submit_stage_retry_attempt(
    *,
    submit_attempt_recorder: object,
    run_id: str,
    slug: str,
    problem_types: list[str],
    submission_ref: str,
    submission_artifact_path: Path | None,
    fallback_submission_path: Path,
    compute_submission_sha256: Callable[[Path | None], str | None],
    exit_code: int | None,
    fingerprint: str,
    action: SubmitStageErrorActionDecision,
    stdout: str,
    stderr: str,
    attempt: int,
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    knowledge_paths: object,
    normalize_detail: Callable[..., str],
    record_error_fix_insight: Callable[..., object],
) -> bool:
    return _submit_attempts.record_submit_retry_attempt_and_knowledge(
        submit_attempt_recorder=submit_attempt_recorder,
        run_id=run_id,
        slug=slug,
        problem_types=problem_types,
        submission_ref=submission_ref,
        submission_path=submission_artifact_path or fallback_submission_path,
        submission_sha256=compute_submission_sha256(submission_artifact_path),
        exit_code=exit_code,
        fingerprint=fingerprint,
        reason=action.reason,
        stdout=stdout,
        stderr=stderr,
        attempt=attempt,
        wait_seconds=action.wait_seconds,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
        knowledge_paths=knowledge_paths,
        infer_iteration=_submit_stage_duplicate.infer_iteration_from_submission_path,
        normalize_detail=normalize_detail,
        record_error_fix_insight=record_error_fix_insight,
    )


def record_submit_abort_for_run(
    *,
    run_dir: Path,
    run_id: str,
    slug: str,
    knowledge_paths: object,
    problem_types: list[str],
    submission_ref: str | Path,
    submission_artifact_path: Path | None,
    artifact_mode: str | None,
    code_fingerprint: str,
    fingerprint: str,
    error_kind: str,
    reason: str,
    message: str,
    stdout_tail: str,
    stderr_tail: str,
    exit_code: int | None,
    submit_attempt_recorder: object,
    resolve_submit_abort_artifact_path: Callable[..., Path | None],
    persist_submit_abort_failure: Callable[..., object],
    load_run_state: Callable[[Path], dict[str, object]],
    load_latest_submit_attempt: Callable[[Path], dict[str, object]],
    has_successful_submit_attempt: Callable[[Path], bool],
    compute_submission_sha256: Callable[[Path | None], str | None],
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    now_iso: str,
    normalize_detail: Callable[..., str],
    record_error_fix_insight: Callable[..., object],
    on_message: Callable[[str], object],
) -> None:
    submission_ref_text = str(submission_ref)
    artifact_path = resolve_submit_abort_artifact_path(
        submission_ref=submission_ref,
        submission_artifact_path=submission_artifact_path,
    )
    prior_state = load_run_state(run_dir)
    prior_submit_ok = bool(prior_state.get("submit_ok")) or has_successful_submit_attempt(run_dir)
    persist_submit_abort_failure(
        run_dir=run_dir,
        run_id=run_id,
        submission_ref=submission_ref_text,
        submission_sha256=compute_submission_sha256(artifact_path),
        artifact_path=artifact_path,
        artifact_mode=artifact_mode,
        code_fingerprint=code_fingerprint,
        fingerprint=fingerprint,
        error_kind=error_kind,
        reason=reason,
        message=message,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        exit_code=exit_code,
        prior_state=prior_state,
        prior_submit_ok=prior_submit_ok,
        submit_attempt_recorder=submit_attempt_recorder,
        load_latest_submit_attempt=load_latest_submit_attempt,
        load_run_state=load_run_state,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
        now_iso=now_iso,
    )
    _submit_attempts.record_submit_reason_knowledge(
        knowledge_paths=knowledge_paths,
        slug=slug,
        run_id=run_id,
        problem_types=problem_types,
        submission_path=artifact_path or Path(submission_ref_text),
        error_kind=error_kind,
        reason=reason,
        action_taken="abort",
        fingerprint=fingerprint,
        details=message,
        infer_iteration=_submit_stage_duplicate.infer_iteration_from_submission_path,
        normalize_detail=normalize_detail,
        record_error_fix_insight=record_error_fix_insight,
    )
    on_message(f"[red]submit aborted[/red]: {message}")


def abort_submit_for_run(
    *,
    run_dir: Path,
    run_id: str,
    slug: str,
    knowledge_paths: object,
    problem_types: list[str],
    submission_ref: str | Path,
    submission_artifact_path: Path | None,
    artifact_mode: str | None,
    code_fingerprint: str | None,
    fingerprint: str,
    error_kind: str,
    reason: str,
    message: str,
    stdout_tail: str,
    stderr_tail: str,
    exit_code: int | None,
    submit_attempt_recorder: object | None,
    save_run_state: Callable[[dict[str, object]], object],
    resolve_submit_abort_artifact_path: Callable[..., Path | None],
    persist_submit_abort_failure: Callable[..., object],
    load_run_state: Callable[[Path], dict[str, object]],
    load_latest_submit_attempt: Callable[[Path], dict[str, object]],
    has_successful_submit_attempt: Callable[[Path], bool],
    compute_submission_sha256: Callable[[Path | None], str | None],
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    now_iso: str,
    normalize_detail: Callable[..., str],
    record_error_fix_insight: Callable[..., object],
    on_message: Callable[[str], object],
    build_error: Callable[[str], BaseException],
) -> None:
    if submit_attempt_recorder is None:
        submit_attempt_recorder = _submit_attempts.SubmitAttemptRecorder(
            run_dir=run_dir,
            save_run_state=save_run_state,
        )
    record_submit_abort_for_run(
        run_dir=run_dir,
        run_id=run_id,
        slug=slug,
        knowledge_paths=knowledge_paths,
        problem_types=problem_types,
        submission_ref=submission_ref,
        submission_artifact_path=submission_artifact_path,
        artifact_mode=artifact_mode,
        code_fingerprint=code_fingerprint or "",
        fingerprint=fingerprint,
        error_kind=error_kind,
        reason=reason,
        message=message,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        exit_code=exit_code,
        submit_attempt_recorder=submit_attempt_recorder,
        resolve_submit_abort_artifact_path=resolve_submit_abort_artifact_path,
        persist_submit_abort_failure=persist_submit_abort_failure,
        load_run_state=load_run_state,
        load_latest_submit_attempt=load_latest_submit_attempt,
        has_successful_submit_attempt=has_successful_submit_attempt,
        compute_submission_sha256=compute_submission_sha256,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
        now_iso=now_iso,
        normalize_detail=normalize_detail,
        record_error_fix_insight=record_error_fix_insight,
        on_message=on_message,
    )
    raise build_error(message)
