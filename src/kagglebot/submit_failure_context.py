from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.json_utils import load_json_object, write_json_object
from kagglebot.submission.guard import normalize_error_text
from kagglebot.submit_attempts import SubmitAttemptRecorder, build_submit_abort_record_payloads
from kagglebot.submit_failure_policy import (
    SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT,
    SubmitFailureRepairDecision,
    classify_submit_failure_repair,
    normalize_loaded_submit_failure_context,
    submit_error_requires_file_fix,
)

SUBMIT_FAILURE_CONTEXT_FILENAME = "submit_failure_context.json"


@dataclass(frozen=True)
class StaleSubmitAutofixDecision:
    clear_repaired_path: bool
    failure_context_updates: dict[str, object]


@dataclass(frozen=True)
class SubmitAutofixInputDecision:
    input_submission_path: Path
    message: str


@dataclass(frozen=True)
class SubmitAutofixAttemptContext:
    run_state: dict[str, object]
    failure_context: dict[str, object]
    latest_submit_attempt: dict[str, object]
    input_submission_path: Path
    message: str


@dataclass(frozen=True)
class SubmitAbortAutofixDecision:
    autofixable: bool
    message: str


def submit_failure_context_path(run_dir: Path) -> Path:
    return run_dir / SUBMIT_FAILURE_CONTEXT_FILENAME


def load_submit_failure_context(run_dir: Path) -> dict[str, object]:
    path = submit_failure_context_path(run_dir)
    payload = load_json_object(path)
    if payload is None:
        return {}
    return normalize_loaded_submit_failure_context(payload)


def save_submit_failure_context(run_dir: Path, payload: dict[str, object]) -> None:
    write_json_object(submit_failure_context_path(run_dir), payload)


def mark_submit_failure_context_resolved(
    *,
    run_dir: Path,
    resolution: str,
    submission_ref: str | None = None,
) -> None:
    payload = load_submit_failure_context(run_dir)
    if not payload:
        return
    payload["active"] = False
    payload["resolution"] = resolution
    payload["resolved_at"] = datetime.now(UTC).isoformat()
    if submission_ref is not None:
        payload["resolved_submission_ref"] = submission_ref
    save_submit_failure_context(run_dir, payload)


def mark_submit_failure_context_submitted(*, run_dir: Path, submission_ref: str) -> None:
    mark_submit_failure_context_resolved(
        run_dir=run_dir,
        resolution="submitted",
        submission_ref=submission_ref,
    )


def mark_submit_failure_context_duplicate_skipped(
    *,
    run_dir: Path,
    submission_ref: str,
    reason: str = "duplicate_submission_sha_seen",
) -> None:
    mark_submit_failure_context_resolved(
        run_dir=run_dir,
        resolution=reason,
        submission_ref=submission_ref,
    )


def build_submit_failure_context_payload(
    *,
    now_iso: str,
    submission_ref: str,
    artifact_path: Path | None,
    artifact_sha256: str | None,
    artifact_mode: str | None,
    code_fingerprint: str,
    fingerprint: str,
    error_kind: str,
    reason: str,
    message: str,
    stdout_tail: str,
    stderr_tail: str,
    exit_code: int | None,
    repair_decision: SubmitFailureRepairDecision,
    latest_submit_attempt: dict[str, object],
    run_state: dict[str, object],
    stdout_tail_chars: int,
    stderr_tail_chars: int,
) -> dict[str, object]:
    detail = "\n".join(part for part in (stdout_tail, stderr_tail) if part).strip()
    summary = normalize_error_text("\n".join(part for part in (message, detail) if part), max_chars=1200)
    return {
        "ts": now_iso,
        "active": True,
        "error_kind": error_kind,
        "reason": reason,
        "fingerprint": fingerprint,
        "code_fingerprint": code_fingerprint,
        "repair_target": repair_decision.repair_target,
        "repairable": repair_decision.repairable,
        "manual_next_step": repair_decision.manual_next_step,
        "message": message,
        "summary": summary,
        "submit_mode": "notebook" if submission_ref.startswith("kernel:") else "file",
        "artifact_mode": str(artifact_mode or "").strip().lower(),
        "submission_ref": submission_ref,
        "submission_artifact_path": str(artifact_path) if artifact_path is not None else "",
        "submission_artifact_sha256": artifact_sha256,
        "stdout_tail": stdout_tail[-stdout_tail_chars:],
        "stderr_tail": stderr_tail[-stderr_tail_chars:],
        "exit_code": exit_code,
        "latest_submit_attempt": latest_submit_attempt,
        "run_state_excerpt": {
            "submit_attempted": bool(run_state.get("submit_attempted")),
            "submit_ok": bool(run_state.get("submit_ok")),
            "last_reason": run_state.get("last_reason"),
            "last_error_kind": run_state.get("last_error_kind"),
            "last_submission_path": run_state.get("last_submission_path"),
            "submit_autofix_submission_path": run_state.get("submit_autofix_submission_path"),
        },
    }


def build_submit_failure_context_payload_from_error(
    *,
    now_iso: str,
    submission_ref: str,
    artifact_path: Path | None,
    artifact_sha256: str | None,
    artifact_mode: str | None,
    code_fingerprint: str,
    fingerprint: str,
    error_kind: str,
    reason: str,
    message: str,
    stdout_tail: str,
    stderr_tail: str,
    exit_code: int | None,
    latest_submit_attempt: dict[str, object],
    run_state: dict[str, object],
    stdout_tail_chars: int,
    stderr_tail_chars: int,
) -> dict[str, object]:
    detail = "\n".join(part for part in (stdout_tail, stderr_tail) if part).strip()
    repair_decision = classify_submit_failure_repair(
        reason=reason,
        error_kind=error_kind,
        detail=detail,
    )
    return build_submit_failure_context_payload(
        now_iso=now_iso,
        submission_ref=submission_ref,
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
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
        latest_submit_attempt=latest_submit_attempt,
        run_state=run_state,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
    )


def path_from_submit_reference(value: object) -> Path | None:
    text = str(value or "").strip()
    if not text or text.startswith("kernel:"):
        return None
    try:
        return Path(text)
    except TypeError:
        return None


def resolve_submit_abort_artifact_path(
    *,
    submission_ref: str | Path,
    submission_artifact_path: Path | None,
) -> Path | None:
    if submission_artifact_path is not None:
        return submission_artifact_path
    if isinstance(submission_ref, Path):
        return submission_ref
    return None


def persist_submit_abort_failure(
    *,
    run_dir: Path,
    run_id: str,
    submission_ref: str,
    submission_sha256: str | None,
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
    prior_state: dict[str, object],
    prior_submit_ok: bool,
    submit_attempt_recorder: SubmitAttemptRecorder,
    load_latest_submit_attempt: Callable[[Path], dict[str, object]],
    load_run_state: Callable[[Path], dict[str, object]],
    stdout_tail_chars: int,
    stderr_tail_chars: int,
    now_iso: str,
) -> None:
    abort_payloads = build_submit_abort_record_payloads(
        run_id=run_id,
        submission_ref=submission_ref,
        submission_sha256=submission_sha256,
        exit_code=exit_code,
        fingerprint=fingerprint,
        code_fingerprint=code_fingerprint,
        error_kind=error_kind,
        reason=reason,
        stdout=stdout_tail,
        stderr=stderr_tail,
        prior_state=prior_state,
        prior_submit_ok=prior_submit_ok,
        stdout_tail_chars=stdout_tail_chars,
        stderr_tail_chars=stderr_tail_chars,
    )
    submit_attempt_recorder.record_payloads(abort_payloads)
    save_submit_failure_context(
        run_dir,
        build_submit_failure_context_payload_from_error(
            now_iso=now_iso,
            submission_ref=submission_ref,
            artifact_path=artifact_path,
            artifact_sha256=submission_sha256,
            artifact_mode=artifact_mode,
            code_fingerprint=code_fingerprint,
            fingerprint=fingerprint,
            error_kind=error_kind,
            reason=reason,
            message=message,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            exit_code=exit_code,
            latest_submit_attempt=load_latest_submit_attempt(run_dir),
            run_state=load_run_state(run_dir),
            stdout_tail_chars=stdout_tail_chars,
            stderr_tail_chars=stderr_tail_chars,
        ),
    )


def decide_stale_submit_autofix_artifact(
    *,
    run_state: dict[str, object],
    failure_context: dict[str, object],
    submission_path: Path,
    now_iso: str,
) -> StaleSubmitAutofixDecision | None:
    repaired_path = path_from_submit_reference(run_state.get("submit_autofix_submission_path"))
    if repaired_path is None:
        return None
    if not failure_context:
        return None
    failed_artifact_path = path_from_submit_reference(
        failure_context.get("submission_artifact_path") or failure_context.get("submission_ref")
    )
    if failed_artifact_path is None:
        return None
    if submission_path == failed_artifact_path or submission_path == repaired_path:
        return None
    return StaleSubmitAutofixDecision(
        clear_repaired_path=True,
        failure_context_updates={
            "stale_repaired_artifact_cleared_at": now_iso,
            "superseded_by_submission_path": str(submission_path),
        },
    )


def apply_stale_submit_autofix_decision(
    *,
    decision: StaleSubmitAutofixDecision | None,
    failure_context: dict[str, object],
    save_run_state: Callable[[dict[str, object]], object],
    save_failure_context: Callable[[dict[str, object]], object],
) -> dict[str, object]:
    if decision is None:
        return failure_context

    if decision.clear_repaired_path:
        save_run_state({"submit_autofix_submission_path": ""})
    updated_context = dict(failure_context)
    updated_context.update(decision.failure_context_updates)
    save_failure_context(updated_context)
    return updated_context


def resolve_submit_autofix_context_for_attempt(
    *,
    run_dir: Path,
    submission_path: Path,
    load_run_state: Callable[[Path], dict[str, object]],
    load_latest_submit_attempt: Callable[[Path], dict[str, object]],
    save_run_state: Callable[[dict[str, object]], object],
    now_iso: str,
) -> SubmitAutofixAttemptContext:
    stale_autofix_state = load_run_state(run_dir)
    stale_autofix_context = load_submit_failure_context(run_dir)
    stale_autofix_decision = decide_stale_submit_autofix_artifact(
        run_state=stale_autofix_state,
        failure_context=stale_autofix_context,
        submission_path=submission_path,
        now_iso=now_iso,
    )
    apply_stale_submit_autofix_decision(
        decision=stale_autofix_decision,
        failure_context=stale_autofix_context,
        save_run_state=save_run_state,
        save_failure_context=lambda payload: save_submit_failure_context(run_dir, payload),
    )

    run_state = load_run_state(run_dir)
    failure_context = load_submit_failure_context(run_dir)
    latest_submit_attempt = load_latest_submit_attempt(run_dir)
    input_decision = decide_submit_autofix_input_submission(
        run_state=run_state,
        latest_submit_attempt=latest_submit_attempt,
        failure_context=failure_context,
        submission_path=submission_path,
    )
    return SubmitAutofixAttemptContext(
        run_state=run_state,
        failure_context=failure_context,
        latest_submit_attempt=latest_submit_attempt,
        input_submission_path=input_decision.input_submission_path,
        message=input_decision.message,
    )


def decide_submit_autofix_input_submission(
    *,
    run_state: dict[str, object],
    latest_submit_attempt: dict[str, object],
    failure_context: dict[str, object],
    submission_path: Path,
) -> SubmitAutofixInputDecision:
    repaired_path = path_from_submit_reference(run_state.get("submit_autofix_submission_path"))
    if repaired_path is None or not repaired_path.exists():
        return SubmitAutofixInputDecision(input_submission_path=submission_path, message="")

    repair_target = str(failure_context.get("repair_target") or "").strip().lower()
    failed_submission_artifact = path_from_submit_reference(
        failure_context.get("submission_artifact_path") or failure_context.get("submission_ref")
    )
    retry_detail = "\n".join(
        part
        for part in (
            str(latest_submit_attempt.get("stdout_tail") or ""),
            str(latest_submit_attempt.get("stderr_tail") or ""),
        )
        if part
    )
    should_use_repaired = (
        repair_target == SUBMIT_FAILURE_REPAIR_TARGET_SUBMISSION_ARTIFACT
        and (failed_submission_artifact is None or failed_submission_artifact == submission_path)
    ) or (
        not repair_target
        and submit_error_requires_file_fix(
            reason=run_state.get("last_reason") or latest_submit_attempt.get("reason"),
            error_kind=run_state.get("last_error_kind") or latest_submit_attempt.get("error_kind"),
            detail=retry_detail,
        )
    )
    if not should_use_repaired:
        return SubmitAutofixInputDecision(input_submission_path=submission_path, message="")

    return SubmitAutofixInputDecision(
        input_submission_path=repaired_path,
        message=(
            f"[yellow]submit retry[/yellow]: using repaired submission artifact from submit autofix: {repaired_path}"
        ),
    )


def decide_submit_abort_autofixability(
    *,
    failure_context: dict[str, object],
    run_state: dict[str, object],
) -> SubmitAbortAutofixDecision:
    if failure_context:
        reason = str(failure_context.get("reason") or "unknown").strip().lower()
        if reason == "ambiguous_notebook_bad_request":
            return SubmitAbortAutofixDecision(
                autofixable=False,
                message=(
                    "[yellow]autofix skipped[/yellow]: submit abort is an ambiguous notebook 400; "
                    "not treating it as a kernel-repairable failure."
                ),
            )
        if bool(failure_context.get("repairable")):
            return SubmitAbortAutofixDecision(autofixable=True, message="")
        repair_target = str(failure_context.get("repair_target") or "unknown").strip().lower()
        manual_next_step = str(failure_context.get("manual_next_step") or "").strip()
        suffix = f" next_step={manual_next_step}" if manual_next_step else ""
        return SubmitAbortAutofixDecision(
            autofixable=False,
            message=(
                "[yellow]autofix skipped[/yellow]: submit abort requires manual intervention "
                f"(target={repair_target}, reason={reason}){suffix}"
            ),
        )

    kind = str(run_state.get("last_error_kind") or "").strip().lower()
    reason = str(run_state.get("last_reason") or "").strip().lower()
    if kind in {"validation", "transient", "unknown"}:
        return SubmitAbortAutofixDecision(autofixable=True, message="")
    if reason == "same_error_fingerprint_recurred":
        return SubmitAbortAutofixDecision(autofixable=True, message="")
    return SubmitAbortAutofixDecision(
        autofixable=False,
        message=(
            "[yellow]autofix skipped[/yellow]: submit abort is not safely auto-fixable "
            f"(kind={kind or 'unknown'}, reason={reason or 'unknown'})"
        ),
    )


def resolve_submit_abort_autofixability_for_run(
    *,
    run_dir: Path,
    load_run_state: Callable[[Path], dict[str, object]],
) -> SubmitAbortAutofixDecision:
    return decide_submit_abort_autofixability(
        failure_context=load_submit_failure_context(run_dir),
        run_state=load_run_state(run_dir),
    )


def should_force_resubmit_after_submit_abort(run_state: dict[str, object]) -> bool:
    reason = str(run_state.get("last_reason") or "").strip().lower()
    if not reason:
        return False
    if reason in {"submission_polling_error", "submission_polling_timeout", "submission_polling_invalid_payload"}:
        return True
    return reason.startswith("submission_poll_status_")


def should_defer_submit_abort_to_next_iteration(
    *,
    compute: str,
    failure_context: dict[str, object],
    iteration: int,
    max_iterations: int,
) -> bool:
    if compute != "kaggle_gpu":
        return False
    if iteration >= max_iterations:
        return False
    return bool(failure_context.get("active")) and bool(failure_context.get("repairable"))


def format_submit_file_repair_contract_prompt() -> str:
    return """

## Submission File Repair Contract

Kaggle rejected the submission artifact itself. This autofix is not complete unless the submission file bytes or
artifact path change, and `run_state.json` records `submit_autofix_submission_path` pointing to the repaired file.
If you repair the file in place, ensure the file contents actually change.
If this is competition-specific, fix the authoritative source that generates the artifact rather than leaving only
an ad-hoc repaired copy behind.
"""


def format_submit_file_repair_contract_retry_feedback(
    *,
    baseline_path: Path | None,
    baseline_sha256: str | None,
) -> str:
    return (
        "Submission file repair contract not satisfied.\n"
        "Kaggle rejected the submission artifact itself, so this autofix must change the prepared "
        "submission file bytes or output path.\n"
        f"baseline_submission_path={baseline_path}\n"
        f"baseline_submission_sha256={baseline_sha256}\n"
        "Record the repaired artifact in run_state.json as submit_autofix_submission_path."
    )


def build_submit_failure_improvement_context(
    *,
    failure_context: dict[str, object],
    latest_submit_attempt: dict[str, object],
) -> tuple[list[str], str | None]:
    if not failure_context or not bool(failure_context.get("active")) or not bool(failure_context.get("repairable")):
        return [], None

    reason = str(failure_context.get("reason") or "").strip().lower()
    error_kind = str(failure_context.get("error_kind") or "").strip().lower()
    detail = "\n".join(
        part
        for part in (
            str(latest_submit_attempt.get("stdout_tail") or ""),
            str(latest_submit_attempt.get("stderr_tail") or ""),
            str(failure_context.get("summary") or ""),
        )
        if part
    ).strip()
    if not submit_error_requires_file_fix(reason=reason, error_kind=error_kind, detail=detail):
        return [], None

    repair_target = str(failure_context.get("repair_target") or "").strip().lower() or "submission_artifact"
    artifact_path = str(
        failure_context.get("submission_artifact_path") or failure_context.get("submission_ref") or ""
    ).strip()
    detail_text = normalize_error_text(detail or str(failure_context.get("message") or ""), max_chars=1200)
    notes = [
        "This run must fix the submission contract before further model changes.",
        f"expected_repair_target={repair_target}",
    ]
    if artifact_path:
        notes.append(f"failed_submission_artifact={artifact_path}")
    if detail_text:
        notes.append(f"kaggle_submission_error={detail_text}")
    artifact_mode = str(failure_context.get("artifact_mode") or "").strip().lower()
    if artifact_mode:
        notes.append(f"notebook_submit_artifact_mode={artifact_mode}")
    notes.extend(
        [
            "If the error is competition-specific, edit only authoritative `kernel.py`.",
            "Do not leave iter2 with the same Kaggle row/column/evaluation exception.",
            "Prioritize source generation fixes over one-off repaired artifact workarounds "
            "when the same format error recurs.",
        ]
    )
    if artifact_mode == "inference":
        notes.extend(
            [
                "This competition is notebook-only and scored on hidden/full test in Kaggle runtime.",
                "Do not rely on an embedded local submission artifact for submit.",
                "Kernel must write `/kaggle/working/submission.csv` during notebook execution.",
            ]
        )
    return notes, "Previous iteration failed Kaggle submission contract; repair submit format before further tuning."


def resolve_submit_autofix_submission_artifact(
    *,
    run_state: dict[str, object],
    latest_submit_attempt: dict[str, object],
    failure_context: dict[str, object],
    fallback_iteration_dirs: Iterable[Path],
    resolve_iteration_submission_artifact: Callable[[Path], Path | None],
) -> Path | None:
    candidates = (
        run_state.get("submit_autofix_submission_path"),
        failure_context.get("submission_artifact_path"),
        latest_submit_attempt.get("sub_path"),
        run_state.get("last_submission_path"),
    )
    for candidate in candidates:
        path = path_from_submit_reference(candidate)
        if path is None:
            continue
        if path.exists() and path.is_file():
            return path
    for iter_dir in fallback_iteration_dirs:
        if not iter_dir.exists():
            continue
        resolved = resolve_iteration_submission_artifact(iter_dir)
        if resolved is not None and resolved.exists():
            return resolved
    return None


def submit_file_fix_contract_satisfied(
    *,
    run_state: dict[str, object],
    baseline_path: Path | None,
    baseline_sha256: str | None,
    sha256_or_none: Callable[[Path | None], str | None],
) -> bool:
    candidate = path_from_submit_reference(run_state.get("submit_autofix_submission_path"))
    if candidate is None or not candidate.exists():
        return False
    candidate_sha256 = sha256_or_none(candidate)
    if candidate_sha256 is None:
        return False
    if baseline_path is None or baseline_sha256 is None:
        return True
    return candidate != baseline_path or candidate_sha256 != baseline_sha256


def format_submit_autofix_context(
    *,
    failure_context: dict[str, object],
    run_state: dict[str, object],
    latest_submit_attempt: dict[str, object],
) -> str:
    lines: list[str] = []
    if failure_context:
        lines.append("submit_failure_context:")
        for key in (
            "ts",
            "active",
            "repair_target",
            "repairable",
            "reason",
            "error_kind",
            "fingerprint",
            "submit_mode",
            "artifact_mode",
            "submission_ref",
            "submission_artifact_path",
            "submission_artifact_sha256",
            "manual_next_step",
            "summary",
        ):
            value = failure_context.get(key)
            if value in (None, "", []):
                continue
            lines.append(f"- {key}: {value}")
        latest_context_attempt = failure_context.get("latest_submit_attempt")
        if isinstance(latest_context_attempt, dict) and latest_context_attempt:
            lines.append("failure_context_latest_submit_attempt:")
            _append_submit_attempt_excerpt(lines, latest_context_attempt)
        run_state_excerpt = failure_context.get("run_state_excerpt")
        if isinstance(run_state_excerpt, dict) and run_state_excerpt:
            lines.append("failure_context_run_state:")
            for key in (
                "submit_attempted",
                "submit_ok",
                "last_reason",
                "last_error_kind",
                "last_submission_path",
                "submit_autofix_submission_path",
            ):
                value = run_state_excerpt.get(key)
                if value in (None, ""):
                    continue
                lines.append(f"- {key}: {value}")

    lines.append("run_state:")
    for key in (
        "submit_attempted",
        "submit_ok",
        "last_error_kind",
        "last_reason",
        "last_action",
        "last_submit_fingerprint",
        "last_submission_path",
        "submit_autofix_submission_path",
    ):
        value = run_state.get(key)
        if value in (None, ""):
            continue
        lines.append(f"- {key}: {value}")

    if latest_submit_attempt:
        lines.append("latest_submit_attempt:")
        _append_submit_attempt_excerpt(lines, latest_submit_attempt)
        stdout_tail = normalize_error_text(str(latest_submit_attempt.get("stdout_tail") or ""), max_chars=1200)
        stderr_tail = normalize_error_text(str(latest_submit_attempt.get("stderr_tail") or ""), max_chars=1200)
        if stdout_tail:
            lines.append(f"- stdout_tail: {stdout_tail}")
        if stderr_tail:
            lines.append(f"- stderr_tail: {stderr_tail}")
    return "\n".join(lines).strip()


def _append_submit_attempt_excerpt(lines: list[str], attempt: dict[str, object]) -> None:
    for key in (
        "ts",
        "ok",
        "exit_code",
        "error_kind",
        "reason",
        "action_taken",
        "fingerprint",
        "sub_path",
    ):
        value = attempt.get(key)
        if value in (None, ""):
            continue
        lines.append(f"- {key}: {value}")
