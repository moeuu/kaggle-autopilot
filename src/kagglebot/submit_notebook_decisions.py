from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kagglebot.submission_output_naming import (
    all_submission_output_suffixes_ordered,
    configured_submission_filename_is_template,
    output_suffix,
)
from kagglebot.submit_error_classification import classify_submit_error_with_output_fallback
from kagglebot.writeup import normalize_submit_mode

_KERNEL_PUSH_VERSION_RE = re.compile(r"Kernel version\s+(?P<version>\d+)\s+successfully pushed", re.IGNORECASE)
_NOTEBOOK_SUBMIT_OUTPUT_SUFFIXES = all_submission_output_suffixes_ordered()
_NOTEBOOK_SUBMIT_EXCLUDED_OUTPUT_NAMES = {"metrics.json", "plan.json", "submission_manifest.json"}


@dataclass(frozen=True)
class NotebookSubmitReference:
    kernel_ref: str
    submission_ref: str
    output_file: str
    version: str


@dataclass(frozen=True)
class NotebookSubmitOutputReference:
    submission_artifact_path: Path | None
    reference: NotebookSubmitReference


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


def resolve_notebook_submit_artifact_mode(*, submit_mode: object, code_competition: bool) -> str:
    normalized_submit_mode = normalize_submit_mode(submit_mode, default="file")
    if normalized_submit_mode != "notebook":
        return "wrapper"
    return "inference" if code_competition else "wrapper"


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


def decide_notebook_submit_artifact_mode_for_paths(
    *,
    requested_mode: str | None,
    notebook_submit_required: bool,
    code_competition: bool,
    sample_submission_path: Path,
    fallback_sample_submission_path: Path,
    submission_path: Path,
    count_tabular_data_rows: Callable[[Path], int | None],
) -> NotebookSubmitArtifactModeDecision:
    sample_rows = count_tabular_data_rows(sample_submission_path)
    if sample_rows is None:
        sample_rows = count_tabular_data_rows(fallback_sample_submission_path)
    return decide_notebook_submit_artifact_mode(
        requested_mode=requested_mode,
        notebook_submit_required=notebook_submit_required,
        code_competition=code_competition if notebook_submit_required else False,
        sample_data_rows=sample_rows,
        submission_data_rows=count_tabular_data_rows(submission_path),
    )


def build_notebook_submit_reference(
    *,
    kernel_id: str,
    submission_artifact_path: Path | None,
    kernel_submission_path: Path | None,
    version_label: str | None,
    expected_output_file: str | None = None,
) -> NotebookSubmitReference:
    output_path = kernel_submission_path or submission_artifact_path
    return NotebookSubmitReference(
        kernel_ref=kernel_id,
        submission_ref=f"kernel:{kernel_id}",
        output_file=output_path.name if output_path is not None else _fallback_output_file(expected_output_file),
        version=str(version_label or "").strip() or "1",
    )


def build_notebook_submit_output_reference(
    *,
    kernel_id: str,
    kernel_submission_path: Path | None,
    version_label: str | None,
    copy_submission_artifact: Callable[[Path], Path],
    expected_output_file: str | None = None,
) -> NotebookSubmitOutputReference:
    submission_artifact_path = copy_submission_artifact(kernel_submission_path) if kernel_submission_path else None
    return NotebookSubmitOutputReference(
        submission_artifact_path=submission_artifact_path,
        reference=build_notebook_submit_reference(
            kernel_id=kernel_id,
            submission_artifact_path=submission_artifact_path,
            kernel_submission_path=kernel_submission_path,
            version_label=version_label,
            expected_output_file=expected_output_file,
        ),
    )


def _fallback_output_file(expected_output_file: str | None) -> str:
    expected = Path(str(expected_output_file or "").strip()).name
    if not expected:
        return "submission.csv"
    suffix = output_suffix(expected.lower(), allowed_suffixes=_NOTEBOOK_SUBMIT_OUTPUT_SUFFIXES)
    if expected.lower() in _NOTEBOOK_SUBMIT_EXCLUDED_OUTPUT_NAMES:
        return "submission.csv"
    if configured_submission_filename_is_template(expected):
        return f"submission{suffix}" if suffix else "submission.csv"
    return expected


def infer_kernel_submit_version_label(logs_dir: Path | None) -> str | None:
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


def build_submit_kernel_run_kwargs(
    *,
    slug: str,
    run_id: str,
    iteration: int,
    base_dir: Path,
    kaggle_username: str,
    kernel_name: str | None,
    accelerator: str,
    enable_internet: bool,
    submission_path: Path,
    artifact_mode: str | None,
    dry_run: bool,
    timeout_minutes: int | None,
) -> dict[str, object]:
    return {
        "slug": slug,
        "run_id": run_id,
        "iteration": iteration,
        "base_dir": base_dir,
        "kaggle_username": kaggle_username,
        "kernel_name": kernel_name,
        "accelerator": accelerator,
        "enable_internet": enable_internet,
        "submission_path": submission_path,
        "mode": normalize_notebook_submit_artifact_mode(artifact_mode),
        "dry_run": dry_run,
        "timeout_minutes": timeout_minutes,
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
    result = classify_submit_error_with_output_fallback(
        stdout=stdout,
        stderr=stderr,
        output=output,
        exit_code=exit_code,
        classify_submit_error=classify_submit_error,
        default_retry_after_seconds=3.0,
    )
    retry = should_retry_ambiguous(
        reason=result.normalized.reason,
        stdout=stdout,
        stderr=result.stderr,
    )
    wait_seconds = result.normalized.retry_after_seconds if retry else 0.0
    message = (
        "[yellow]submit retry[/yellow]: notebook submit returned an ambiguous 400; "
        f"retrying same kernel submit in {wait_seconds:.1f}s."
        if retry
        else ""
    )
    return NotebookSubmitRetryDecision(
        retry=retry,
        classification=result.classification,
        stderr=result.stderr,
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


def decide_submit_kernel_cpu_fallback_for_exception(
    *,
    accelerator: str,
    strict_accelerator: bool,
    exc: BaseException,
    is_capacity_error: Callable[[BaseException], bool],
    is_push_error: Callable[[BaseException], bool],
) -> NotebookSubmitCpuFallbackDecision:
    return decide_submit_kernel_cpu_fallback(
        accelerator=accelerator,
        strict_accelerator=strict_accelerator,
        is_capacity_error=is_capacity_error(exc),
        is_push_error=is_push_error(exc),
    )


def is_submit_kernel_push_error(exc: BaseException) -> bool:
    return is_submit_kernel_push_error_text(
        message=str(exc),
        output=str(getattr(exc, "output", "") or ""),
        stdout=str(getattr(exc, "stdout", "") or ""),
        stderr=str(getattr(exc, "stderr", "") or ""),
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
    return 0 < sample_data_rows <= tiny_row_limit and 0 <= submission_data_rows <= tiny_row_limit


def _no_cpu_fallback() -> NotebookSubmitCpuFallbackDecision:
    return NotebookSubmitCpuFallbackDecision(retry_on_cpu=False, reason="", message="")
