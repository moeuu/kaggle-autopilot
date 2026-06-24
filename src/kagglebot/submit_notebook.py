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


@dataclass(frozen=True)
class NotebookSubmitCpuFallbackDecision:
    retry_on_cpu: bool
    reason: str
    message: str


@dataclass(frozen=True)
class NotebookSubmitArtifactModeDecision:
    mode: str
    reason: str
    message: str


def normalize_notebook_submit_artifact_mode(value: str | None) -> str:
    return str(value or "wrapper").strip().lower() or "wrapper"


def decide_notebook_submit_artifact_mode(
    *,
    requested_mode: str | None,
    notebook_submit_required: bool,
    code_competition: bool,
    sample_data_rows: int | None,
    submission_data_rows: int | None,
    tiny_row_limit: int = 10,
) -> NotebookSubmitArtifactModeDecision:
    mode = normalize_notebook_submit_artifact_mode(requested_mode)
    if not notebook_submit_required:
        return _artifact_mode_decision(mode)
    if mode == "inference":
        return _artifact_mode_decision(mode)
    if code_competition:
        return _artifact_mode_decision(
            "inference",
            reason="code_competition",
            message=("[yellow]submit mode[/yellow]: code competition detected; using inference-mode notebook submit."),
        )
    if _is_tiny_public_notebook_contract(
        sample_data_rows=sample_data_rows,
        submission_data_rows=submission_data_rows,
        tiny_row_limit=tiny_row_limit,
    ):
        return _artifact_mode_decision(
            "inference",
            reason="tiny_public_sample_notebook_contract",
            message=(
                "[yellow]submit mode[/yellow]: tiny notebook sample/submission detected; "
                "using inference-mode notebook submit to avoid hidden-test row mismatch."
            ),
        )
    return _artifact_mode_decision(mode)


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


def decide_submit_kernel_cpu_fallback(
    *,
    accelerator: str,
    strict_accelerator: bool,
    is_capacity_error: bool,
    is_push_error: bool,
) -> NotebookSubmitCpuFallbackDecision:
    if str(accelerator).strip().lower() != "gpu":
        return _no_cpu_fallback()
    if strict_accelerator:
        return _no_cpu_fallback()
    if is_capacity_error:
        reason = "Kaggle GPU capacity is unavailable"
    elif is_push_error:
        reason = "Kaggle notebook push failed under GPU metadata"
    else:
        return _no_cpu_fallback()
    return NotebookSubmitCpuFallbackDecision(
        retry_on_cpu=True,
        reason=reason,
        message=f"[yellow]submit notebook[/yellow]: {reason}; retrying submit kernel on CPU.",
    )


def is_submit_kernel_push_error_text(
    *,
    message: str = "",
    output: str = "",
    stdout: str = "",
    stderr: str = "",
) -> bool:
    text = "\n".join(part for part in (message, output, stdout, stderr) if part).lower()
    return (
        "kernel push error:" in text
        or "kaggle kernel push failed" in text
        or "kernel not found after push" in text
        or "notebook not found" in text
    )


def _artifact_mode_decision(
    mode: str,
    *,
    reason: str = "",
    message: str = "",
) -> NotebookSubmitArtifactModeDecision:
    return NotebookSubmitArtifactModeDecision(mode=mode, reason=reason, message=message)


def _is_tiny_public_notebook_contract(
    *,
    sample_data_rows: int | None,
    submission_data_rows: int | None,
    tiny_row_limit: int,
) -> bool:
    if sample_data_rows is None or submission_data_rows is None:
        return False
    return 0 < sample_data_rows <= tiny_row_limit and 0 < submission_data_rows <= tiny_row_limit


def _no_cpu_fallback() -> NotebookSubmitCpuFallbackDecision:
    return NotebookSubmitCpuFallbackDecision(retry_on_cpu=False, reason="", message="")
