from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NotebookSubmitReference:
    kernel_ref: str
    submission_ref: str
    output_file: str
    version: str


@dataclass(frozen=True)
class NotebookSubmitRetryDecision:
    retry: bool
    classification: dict[str, object]
    stderr: str
    wait_seconds: float
    message: str


def normalize_notebook_submit_artifact_mode(value: str | None) -> str:
    return str(value or "wrapper").strip().lower() or "wrapper"


def build_notebook_submit_reference(
    *,
    kernel_id: str,
    submission_artifact_path: Path | None,
    kernel_submission_path: Path | None,
    version_label: str | None,
) -> NotebookSubmitReference:
    output_path = submission_artifact_path or kernel_submission_path
    return NotebookSubmitReference(
        kernel_ref=kernel_id,
        submission_ref=f"kernel:{kernel_id}",
        output_file=output_path.name if output_path is not None else "submission.csv",
        version=str(version_label or "").strip() or "1",
    )


def build_kaggle_submit_kernel_kwargs(
    *,
    slug: str,
    reference: NotebookSubmitReference,
    message: str,
    dry_run: bool,
) -> dict[str, object]:
    return {
        "slug": slug,
        "kernel": reference.kernel_ref,
        "message": message,
        "output_file": reference.output_file,
        "version": reference.version,
        "dry_run": dry_run,
    }


def decide_ambiguous_notebook_submit_retry(
    *,
    stdout: str,
    stderr: str,
    output: str,
    exit_code: int | None,
    classify_submit_error: Callable[[str, str, int | None], dict[str, object]],
    should_retry_ambiguous: Callable[..., bool],
) -> NotebookSubmitRetryDecision:
    classification_stderr = stderr or ""
    classification = classify_submit_error(stdout, classification_stderr, exit_code)
    if str(classification.get("reason") or "unclassified_submit_error") == "unclassified_submit_error" and output:
        classification_stderr = "\n".join(part for part in [classification_stderr, output] if part)
        classification = classify_submit_error(stdout, classification_stderr, exit_code)
    retry = should_retry_ambiguous(
        reason=str(classification.get("reason") or ""),
        stdout=stdout,
        stderr=classification_stderr,
    )
    wait_seconds = float(classification.get("retry_after_seconds") or 3.0) if retry else 0.0
    message = (
        "[yellow]submit retry[/yellow]: notebook submit returned an ambiguous 400; "
        f"retrying same kernel submit in {wait_seconds:.1f}s."
        if retry
        else ""
    )
    return NotebookSubmitRetryDecision(
        retry=retry,
        classification=classification,
        stderr=classification_stderr,
        wait_seconds=wait_seconds,
        message=message,
    )
