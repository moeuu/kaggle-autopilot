from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from kagglebot.exceptions import KaggleCliError, KernelCapacityError, SubmissionCliError
from kagglebot.submit_notebook_decisions import (
    NotebookSubmitArtifactModeDecision,
    NotebookSubmitCpuFallbackDecision,
    NotebookSubmitOutputReference,
    NotebookSubmitReference,
    NotebookSubmitRetryDecision,
    build_kaggle_submit_kernel_kwargs,
    build_notebook_submit_output_reference,
    build_notebook_submit_reference,
    build_submit_kernel_run_kwargs,
    decide_ambiguous_notebook_submit_retry,
    decide_notebook_submit_artifact_mode,
    decide_notebook_submit_artifact_mode_for_paths,
    decide_submit_kernel_cpu_fallback,
    decide_submit_kernel_cpu_fallback_for_exception,
    infer_kernel_submit_version_label,
    is_submit_kernel_push_error,
    is_submit_kernel_push_error_text,
    normalize_notebook_submit_artifact_mode,
    resolve_notebook_submit_artifact_mode,
)

__all__ = [
    "NotebookSubmitArtifactModeDecision",
    "NotebookSubmitCpuFallbackDecision",
    "NotebookSubmitOutputReference",
    "NotebookSubmitReference",
    "NotebookSubmitRetryDecision",
    "NotebookSubmitRunner",
    "build_kaggle_submit_kernel_kwargs",
    "build_notebook_submit_output_reference",
    "build_notebook_submit_reference",
    "build_notebook_submit_runner_for_run",
    "build_submit_kernel_run_kwargs",
    "decide_ambiguous_notebook_submit_retry",
    "decide_notebook_submit_artifact_mode",
    "decide_notebook_submit_artifact_mode_for_paths",
    "decide_submit_kernel_cpu_fallback",
    "decide_submit_kernel_cpu_fallback_for_exception",
    "infer_kernel_submit_version_label",
    "is_submit_kernel_push_error",
    "is_submit_kernel_push_error_text",
    "normalize_notebook_submit_artifact_mode",
    "notebook_kernel_submission_error",
    "resolve_notebook_submit_artifact_mode",
    "run_kaggle_submit_kernel_with_retry",
    "run_notebook_kernel_submission",
    "run_notebook_kernel_submission_for_run",
    "run_submit_kernel_with_cpu_fallback",
]

if TYPE_CHECKING:
    from kagglebot.paths import CompetitionPaths


@dataclass(frozen=True)
class NotebookSubmitRunner:
    slug: str
    run_id: str
    paths: CompetitionPaths
    kaggle_username: str | None
    kernel_name: str | None
    accelerator: str
    strict_accelerator: bool
    dry_run: bool
    timeout_minutes: int | None
    infer_iteration_from_submission_path: Callable[[Path], int | None]
    resolve_kaggle_username: Callable[[str | None], str]
    run_submit_kernel: Callable[..., object]
    run_kaggle_submit_kernel: Callable[..., object]
    copy_submission_artifact_to_iteration_dir: Callable[..., Path]
    classify_submit_error: Callable[[str, str, int | None], dict[str, object]]
    should_retry_ambiguous: Callable[..., bool]
    sleep: Callable[[float], None]
    on_message: Callable[[str], None]
    is_capacity_error: Callable[[BaseException], bool]
    is_push_error: Callable[[BaseException], bool]
    expected_output_file: str | None = None

    def submit(
        self,
        *,
        submission_path: Path,
        message: str,
        artifact_mode: str | None,
    ) -> tuple[object, str, Path | None]:
        return run_notebook_kernel_submission_for_run(
            slug=self.slug,
            run_id=self.run_id,
            paths=self.paths,
            kaggle_username=self.kaggle_username,
            kernel_name=self.kernel_name,
            accelerator=self.accelerator,
            strict_accelerator=self.strict_accelerator,
            submission_path=submission_path,
            message=message,
            artifact_mode=artifact_mode,
            dry_run=self.dry_run,
            timeout_minutes=self.timeout_minutes,
            infer_iteration_from_submission_path=self.infer_iteration_from_submission_path,
            resolve_kaggle_username=self.resolve_kaggle_username,
            run_submit_kernel=self.run_submit_kernel,
            run_kaggle_submit_kernel=self.run_kaggle_submit_kernel,
            copy_submission_artifact_to_iteration_dir=self.copy_submission_artifact_to_iteration_dir,
            classify_submit_error=self.classify_submit_error,
            should_retry_ambiguous=self.should_retry_ambiguous,
            sleep=self.sleep,
            on_message=self.on_message,
            is_capacity_error=self.is_capacity_error,
            is_push_error=self.is_push_error,
            expected_output_file=self.expected_output_file or submission_path.name,
        )


def build_notebook_submit_runner_for_run(
    *,
    slug: str,
    run_id: str,
    paths: CompetitionPaths,
    kaggle_username: str | None,
    kernel_name: str | None,
    accelerator: str,
    strict_accelerator: bool,
    dry_run: bool,
    timeout_minutes: int | None,
    infer_iteration_from_submission_path: Callable[[Path], int | None],
    resolve_kaggle_username: Callable[[str | None], str],
    run_submit_kernel: Callable[..., object],
    run_kaggle_submit_kernel: Callable[..., object],
    copy_submission_artifact_to_iteration_dir: Callable[..., Path],
    classify_submit_error: Callable[[str, str, int | None], dict[str, object]],
    should_retry_ambiguous: Callable[..., bool],
    sleep: Callable[[float], None],
    on_message: Callable[[str], None],
    expected_output_file: str | None = None,
) -> NotebookSubmitRunner:
    return NotebookSubmitRunner(
        slug=slug,
        run_id=run_id,
        paths=paths,
        kaggle_username=kaggle_username,
        kernel_name=kernel_name,
        accelerator=accelerator,
        strict_accelerator=strict_accelerator,
        dry_run=dry_run,
        timeout_minutes=timeout_minutes,
        infer_iteration_from_submission_path=infer_iteration_from_submission_path,
        resolve_kaggle_username=resolve_kaggle_username,
        run_submit_kernel=run_submit_kernel,
        run_kaggle_submit_kernel=run_kaggle_submit_kernel,
        copy_submission_artifact_to_iteration_dir=copy_submission_artifact_to_iteration_dir,
        classify_submit_error=classify_submit_error,
        should_retry_ambiguous=should_retry_ambiguous,
        sleep=sleep,
        on_message=on_message,
        is_capacity_error=lambda exc: isinstance(exc, KernelCapacityError),
        is_push_error=lambda exc: isinstance(exc, KaggleCliError) and is_submit_kernel_push_error(exc),
        expected_output_file=expected_output_file,
    )


def run_notebook_kernel_submission(
    *,
    slug: str,
    run_id: str,
    iteration: int,
    iter_logs_dir: Path,
    base_dir: Path,
    kaggle_username: str,
    kernel_name: str | None,
    accelerator: str,
    strict_accelerator: bool,
    submission_path: Path,
    message: str,
    artifact_mode: str | None,
    dry_run: bool,
    timeout_minutes: int | None,
    run_submit_kernel: Callable[..., object],
    run_kaggle_submit_kernel: Callable[..., object],
    copy_submission_artifact: Callable[[Path], Path],
    classify_submit_error: Callable[[str, str, int | None], dict[str, object]],
    should_retry_ambiguous: Callable[..., bool],
    sleep: Callable[[float], None],
    on_message: Callable[[str], None],
    is_capacity_error: Callable[[BaseException], bool],
    is_push_error: Callable[[BaseException], bool],
    expected_output_file: str | None = None,
) -> tuple[object, str, Path | None]:
    """Run the submit notebook and submit its Kaggle output reference."""
    submit_kernel_kwargs = build_submit_kernel_run_kwargs(
        slug=slug,
        run_id=run_id,
        iteration=iteration,
        base_dir=base_dir,
        kaggle_username=kaggle_username,
        kernel_name=kernel_name,
        accelerator=accelerator,
        enable_internet=False,
        submission_path=submission_path,
        artifact_mode=artifact_mode,
        dry_run=dry_run,
        timeout_minutes=timeout_minutes,
    )
    kernel_result = run_submit_kernel_with_cpu_fallback(
        submit_kernel_kwargs=submit_kernel_kwargs,
        run_submit_kernel=run_submit_kernel,
        decide_cpu_fallback=lambda exc: decide_submit_kernel_cpu_fallback_for_exception(
            accelerator=accelerator,
            strict_accelerator=strict_accelerator,
            exc=exc,
            is_capacity_error=is_capacity_error,
            is_push_error=is_push_error,
        ),
        is_capacity_error=is_capacity_error,
        wrap_error=notebook_kernel_submission_error,
        on_message=on_message,
    )

    output_reference = build_notebook_submit_output_reference(
        kernel_id=str(getattr(kernel_result, "kernel_id")),
        kernel_submission_path=getattr(kernel_result, "submission_path", None),
        version_label=infer_kernel_submit_version_label(iter_logs_dir),
        copy_submission_artifact=copy_submission_artifact,
        expected_output_file=_expected_submit_kernel_output_file(
            submission_path=submission_path,
            artifact_mode=artifact_mode,
            expected_output_file=expected_output_file,
        ),
    )
    submit_reference = output_reference.reference
    on_message(f"[cyan]submit notebook[/cyan]: {submit_reference.kernel_ref}")
    submit_kwargs = build_kaggle_submit_kernel_kwargs(
        slug=slug,
        reference=submit_reference,
        message=message,
        dry_run=dry_run,
    )
    submit_result = run_kaggle_submit_kernel_with_retry(
        submit_kwargs=submit_kwargs,
        run_kaggle_submit_kernel=run_kaggle_submit_kernel,
        submit_error_types=SubmissionCliError,
        classify_submit_error=classify_submit_error,
        should_retry_ambiguous=should_retry_ambiguous,
        sleep=sleep,
        on_message=on_message,
    )
    return submit_result, submit_reference.submission_ref, output_reference.submission_artifact_path


def _expected_submit_kernel_output_file(
    *,
    submission_path: Path,
    artifact_mode: str | None,
    expected_output_file: str | None,
) -> str | None:
    if submission_path.is_dir() and normalize_notebook_submit_artifact_mode(artifact_mode) == "wrapper":
        return f"{submission_path.name}.zip"
    return expected_output_file


def run_notebook_kernel_submission_for_run(
    *,
    slug: str,
    run_id: str,
    paths: CompetitionPaths,
    kaggle_username: str | None,
    kernel_name: str | None,
    accelerator: str,
    strict_accelerator: bool,
    submission_path: Path,
    message: str,
    artifact_mode: str | None,
    dry_run: bool,
    timeout_minutes: int | None,
    infer_iteration_from_submission_path: Callable[[Path], int | None],
    resolve_kaggle_username: Callable[[str | None], str],
    run_submit_kernel: Callable[..., object],
    run_kaggle_submit_kernel: Callable[..., object],
    copy_submission_artifact_to_iteration_dir: Callable[..., Path],
    classify_submit_error: Callable[[str, str, int | None], dict[str, object]],
    should_retry_ambiguous: Callable[..., bool],
    sleep: Callable[[float], None],
    on_message: Callable[[str], None],
    is_capacity_error: Callable[[BaseException], bool],
    is_push_error: Callable[[BaseException], bool],
    expected_output_file: str | None = None,
) -> tuple[object, str, Path | None]:
    iteration = infer_iteration_from_submission_path(submission_path) or 1
    iter_dir = paths.iter_dir(run_id, iteration)
    return run_notebook_kernel_submission(
        slug=slug,
        run_id=run_id,
        iteration=iteration,
        base_dir=paths.base_dir.parent,
        kaggle_username=resolve_kaggle_username(kaggle_username),
        kernel_name=kernel_name,
        accelerator=accelerator,
        strict_accelerator=strict_accelerator,
        submission_path=submission_path,
        message=message,
        artifact_mode=artifact_mode,
        dry_run=dry_run,
        timeout_minutes=timeout_minutes,
        run_submit_kernel=run_submit_kernel,
        run_kaggle_submit_kernel=run_kaggle_submit_kernel,
        copy_submission_artifact=lambda source: copy_submission_artifact_to_iteration_dir(
            source=source,
            iter_dir=iter_dir,
        ),
        classify_submit_error=classify_submit_error,
        should_retry_ambiguous=should_retry_ambiguous,
        sleep=sleep,
        on_message=on_message,
        is_capacity_error=is_capacity_error,
        is_push_error=is_push_error,
        iter_logs_dir=iter_dir / "logs",
        expected_output_file=expected_output_file,
    )


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


def notebook_kernel_submission_error(exc: BaseException) -> SubmissionCliError:
    """Convert submit-kernel execution failures into the submit error type."""
    if isinstance(exc, KaggleCliError):
        output = exc.output or str(exc)
        return SubmissionCliError(
            "Notebook submission fallback failed while running Kaggle kernel.",
            command=list(exc.command or []),
            exit_code=exc.exit_code,
            output=output,
            stdout=exc.stdout,
            stderr=exc.stderr or output,
        )
    return SubmissionCliError(
        "Notebook submission fallback failed while running Kaggle kernel.",
        command=[],
        output=str(exc),
        stdout="",
        stderr=str(exc),
    )


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
