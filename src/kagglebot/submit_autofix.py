from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from kagglebot.exceptions import SubmissionValidationError
from kagglebot.submit_failure_context import resolve_submit_autofix_submission_artifact
from kagglebot.submit_failure_policy import submit_error_requires_file_fix


@dataclass(frozen=True)
class SubmitFileAutofixPreparation:
    path: Path | None
    summary: str
    file_fix_required: bool


def submit_file_fix_required_for_attempt(latest_submit_attempt: dict[str, object]) -> bool:
    detail = "\n".join(
        part
        for part in (
            str(latest_submit_attempt.get("stdout_tail") or ""),
            str(latest_submit_attempt.get("stderr_tail") or ""),
        )
        if part
    )
    return submit_error_requires_file_fix(
        reason=latest_submit_attempt.get("reason"),
        error_kind=latest_submit_attempt.get("error_kind"),
        detail=detail,
    )


def prepare_submit_file_autofix(
    *,
    latest_submit_attempt: dict[str, object],
    resolve_source: Callable[[], Path | None],
    validate_and_prepare: Callable[[Path], Path],
    save_repaired_path: Callable[[Path], None],
) -> SubmitFileAutofixPreparation:
    file_fix_required = submit_file_fix_required_for_attempt(latest_submit_attempt)
    if not file_fix_required:
        return SubmitFileAutofixPreparation(path=None, summary="", file_fix_required=False)

    source = resolve_source()
    if source is None:
        return SubmitFileAutofixPreparation(
            path=None,
            summary="submit autofix could not locate the submission artifact to repair.",
            file_fix_required=True,
        )

    try:
        fixed = validate_and_prepare(source)
    except SubmissionValidationError as exc:
        return SubmitFileAutofixPreparation(
            path=None,
            summary=f"submit autofix could not deterministically repair submission file: {exc}",
            file_fix_required=True,
        )

    if not fixed.exists():
        return SubmitFileAutofixPreparation(
            path=None,
            summary="submit autofix prepared a submission path but the fixed artifact does not exist.",
            file_fix_required=True,
        )

    save_repaired_path(fixed)
    if fixed == source:
        return SubmitFileAutofixPreparation(
            path=fixed,
            summary=(
                "submit autofix inspected the Kaggle-rejected submission artifact, "
                "but no deterministic file rewrite was available; source generation still needs a fix."
            ),
            file_fix_required=True,
        )
    return SubmitFileAutofixPreparation(
        path=fixed,
        summary=(
            "submit autofix created a repaired submission artifact from the Kaggle-rejected file.\n"
            f"- original_submission_path: {source}\n"
            f"- fixed_submission_path: {fixed}"
        ),
        file_fix_required=True,
    )


def prepare_submit_file_autofix_for_run(
    *,
    latest_submit_attempt: dict[str, object],
    run_state: dict[str, object],
    failure_context: dict[str, object],
    fallback_iteration_dirs: Callable[[], Iterable[Path]],
    resolve_iteration_submission_artifact: Callable[[Path], Path | None],
    validate_and_prepare: Callable[[Path], Path],
    save_repaired_path: Callable[[Path], None],
) -> SubmitFileAutofixPreparation:
    """Prepare deterministic submit-file repair using current run artifacts."""

    def resolve_source() -> Path | None:
        return resolve_submit_autofix_submission_artifact(
            run_state=run_state,
            latest_submit_attempt=latest_submit_attempt,
            failure_context=failure_context,
            fallback_iteration_dirs=fallback_iteration_dirs(),
            resolve_iteration_submission_artifact=resolve_iteration_submission_artifact,
        )

    return prepare_submit_file_autofix(
        latest_submit_attempt=latest_submit_attempt,
        resolve_source=resolve_source,
        validate_and_prepare=validate_and_prepare,
        save_repaired_path=save_repaired_path,
    )
