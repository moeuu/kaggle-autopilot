from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kagglebot.submit_error_classification import classify_submit_error_with_output_fallback


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


def build_notebook_submit_output_reference(
    *,
    kernel_id: str,
    kernel_submission_path: Path | None,
    version_label: str | None,
    copy_submission_artifact: Callable[[Path], Path],
) -> NotebookSubmitOutputReference:
    submission_artifact_path = copy_submission_artifact(kernel_submission_path) if kernel_submission_path else None
    return NotebookSubmitOutputReference(
        submission_artifact_path=submission_artifact_path,
        reference=build_notebook_submit_reference(
            kernel_id=kernel_id,
            submission_artifact_path=submission_artifact_path,
            kernel_submission_path=kernel_submission_path,
            version_label=version_label,
        ),
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


def run_submit_kernel_with_cpu_fallback(
    *,
    submit_kernel_kwargs: dict[str, object],
    run_submit_kernel: Callable[..., object],
    decide_cpu_fallback: Callable[[BaseException], NotebookSubmitCpuFallbackDecision],
    is_capacity_error: Callable[[BaseException], bool],
    wrap_error: Callable[[BaseException], BaseException],
    on_message: Callable[[str], None],
) -> object:
    try:
        return run_submit_kernel(**submit_kernel_kwargs)
    except Exception as exc:  # noqa: BLE001
        cpu_fallback_decision = decide_cpu_fallback(exc)
        if cpu_fallback_decision.retry_on_cpu:
            on_message(cpu_fallback_decision.message)
            try:
                return run_submit_kernel(**{**submit_kernel_kwargs, "accelerator": "cpu"})
            except Exception as retry_exc:  # noqa: BLE001
                raise wrap_error(retry_exc) from retry_exc
        if is_capacity_error(exc):
            raise
        raise wrap_error(exc) from exc


def run_kaggle_submit_kernel_with_retry(
    *,
    submit_kwargs: dict[str, object],
    run_kaggle_submit_kernel: Callable[..., object],
    submit_error_types: type[BaseException] | tuple[type[BaseException], ...],
    classify_submit_error: Callable[[str, str, int | None], dict[str, object]],
    should_retry_ambiguous: Callable[..., bool],
    sleep: Callable[[float], None],
    on_message: Callable[[str], None],
) -> object:
    try:
        return run_kaggle_submit_kernel(**submit_kwargs)
    except submit_error_types as exc:
        retry_decision = decide_ambiguous_notebook_submit_retry(
            stdout=str(getattr(exc, "stdout", "") or ""),
            stderr=str(getattr(exc, "stderr", "") or ""),
            output=str(getattr(exc, "output", "") or ""),
            exit_code=getattr(exc, "exit_code", None),
            classify_submit_error=classify_submit_error,
            should_retry_ambiguous=should_retry_ambiguous,
        )
        if retry_decision.retry:
            on_message(retry_decision.message)
            sleep(retry_decision.wait_seconds)
            return run_kaggle_submit_kernel(**submit_kwargs)
        raise


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
