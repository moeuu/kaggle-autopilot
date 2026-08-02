from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from kagglebot.exceptions import SubmissionValidationError
from kagglebot.submission_artifact_resolution import (
    SubmissionArtifactResolutionError,
    atomic_copy_submission_autofix,
    find_current_iteration_dir,
    resolve_valid_submission_artifact,
)
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

    if fixed == source:
        return SubmitFileAutofixPreparation(
            path=fixed,
            summary=(
                "submit autofix inspected the Kaggle-rejected submission artifact, "
                "but no deterministic file rewrite was available; source generation still needs a fix."
            ),
            file_fix_required=True,
        )
    save_repaired_path(fixed)
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

    iteration_dirs = list(fallback_iteration_dirs())

    def resolve_source() -> Path | None:
        return resolve_submit_autofix_submission_artifact(
            run_state=run_state,
            latest_submit_attempt=latest_submit_attempt,
            failure_context=failure_context,
            fallback_iteration_dirs=iteration_dirs,
            resolve_iteration_submission_artifact=resolve_iteration_submission_artifact,
        )

    preparation = prepare_submit_file_autofix(
        latest_submit_attempt=latest_submit_attempt,
        resolve_source=resolve_source,
        validate_and_prepare=validate_and_prepare,
        save_repaired_path=save_repaired_path,
    )
    if not preparation.file_fix_required or preparation.path is not None:
        return preparation

    source = resolve_source()
    artifact_paths = [
        path
        for value in (
            run_state.get("submit_autofix_submission_path"),
            failure_context.get("submission_artifact_path"),
            failure_context.get("submission_ref"),
            latest_submit_attempt.get("sub_path"),
            run_state.get("last_submission_path"),
        )
        if (path := _path_from_value(value)) is not None
    ]
    if source is not None:
        artifact_paths.insert(0, source)
    iteration_dir = find_current_iteration_dir(
        artifact_paths=artifact_paths,
        fallback_dirs=iteration_dirs,
    )
    if iteration_dir is None:
        return preparation

    try:
        recovered = resolve_valid_submission_artifact(
            iteration_dir=iteration_dir,
            validate_and_prepare=validate_and_prepare,
        )
        repaired = atomic_copy_submission_autofix(
            source_path=recovered.prepared_path,
            iteration_dir=iteration_dir,
            validate_and_prepare=validate_and_prepare,
        )
    except (SubmissionArtifactResolutionError, SubmissionValidationError) as exc:
        return SubmitFileAutofixPreparation(
            path=None,
            summary=f"submit autofix could not deterministically recover a valid submission artifact: {exc}",
            file_fix_required=True,
        )

    save_repaired_path(repaired)
    return SubmitFileAutofixPreparation(
        path=repaired,
        summary=(
            "submit autofix recovered the kernel-reported submission artifact and created a validated run-owned copy.\n"
            f"- rejected_submission_path: {source}\n"
            f"- recovered_submission_path: {recovered.source_path}\n"
            f"- recovery_provenance: {recovered.provenance}\n"
            f"- fixed_submission_path: {repaired}"
        ),
        file_fix_required=True,
    )


def _path_from_value(value: object) -> Path | None:
    if not isinstance(value, (str, Path)):
        return None
    text = str(value).strip()
    if not text or text.startswith("kernel:"):
        return None
    return Path(text)
