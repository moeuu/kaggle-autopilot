from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from kagglebot import kernel_outputs as _kernel_outputs
from kagglebot import submit_kernel_fidelity as _submit_kernel_fidelity
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
    review_code_submission: Callable[..., object] | None = None
    recheck_code_submission_guard: Callable[..., object] | None = None
    record_code_submission_execution: Callable[..., object] | None = None

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
            review_code_submission=self.review_code_submission,
            recheck_code_submission_guard=self.recheck_code_submission_guard,
            record_code_submission_execution=self.record_code_submission_execution,
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
    review_code_submission: Callable[..., object] | None = None,
    recheck_code_submission_guard: Callable[..., object] | None = None,
    record_code_submission_execution: Callable[..., object] | None = None,
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
        review_code_submission=review_code_submission,
        recheck_code_submission_guard=recheck_code_submission_guard,
        record_code_submission_execution=record_code_submission_execution,
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
    expected_metrics_payload: dict[str, object] | None = None,
    review_code_submission: Callable[..., object] | None = None,
    recheck_code_submission_guard: Callable[..., object] | None = None,
    record_code_submission_execution: Callable[..., object] | None = None,
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
        expected_output_file=expected_output_file,
    )
    if _is_code_output_artifact_mode(artifact_mode):
        submit_kernel_kwargs["requested_accelerator"] = accelerator
        submit_kernel_kwargs["capacity_fallback_used"] = False
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

    kernel_id = str(getattr(kernel_result, "kernel_id", "") or "").strip()
    version_label = infer_kernel_submit_version_label(iter_logs_dir)
    code_output_file_name = _expected_submit_kernel_output_file(
        submission_path=submission_path,
        artifact_mode=artifact_mode,
        expected_output_file=expected_output_file,
    )
    local_artifact_path: Path | None = None
    fidelity_report_path: Path | None = None
    try:
        local_artifact_path = _resolve_submit_kernel_local_artifact_path(
            kernel_result=kernel_result,
            artifact_mode=artifact_mode,
            code_output_file_name=code_output_file_name,
            allow_missing=dry_run,
        )
        if not dry_run and version_label is None:
            raise _invalid_code_output_error(
                "completed notebook push did not report a positive kernel version",
                kernel_id=kernel_id,
                code_output_file_name=code_output_file_name,
            )
        if not dry_run:
            metrics_path = getattr(kernel_result, "metrics_path", None)
            package_dir = base_dir / slug / "kernels" / run_id / f"submit-iter-{iteration}"
            result_expected_path = getattr(kernel_result, "fidelity_expected_path", None)
            packaged_expected_path = package_dir / _submit_kernel_fidelity.EXPECTED_FILE_NAME
            expected_contract_path = (
                result_expected_path
                if isinstance(result_expected_path, Path)
                else packaged_expected_path
                if packaged_expected_path.is_file()
                else None
            )
            output_dir = getattr(kernel_result, "output_dir", None)
            result_runtime_path = getattr(kernel_result, "fidelity_runtime_path", None)
            discovered_runtime_path = (
                _kernel_outputs.find_output_file(output_dir, _submit_kernel_fidelity.RUNTIME_FILE_NAME)
                if isinstance(output_dir, Path)
                else None
            )
            runtime_fidelity_path = (
                result_runtime_path if isinstance(result_runtime_path, Path) else discovered_runtime_path
            )
            if expected_contract_path is not None:
                fidelity_report_path = iter_logs_dir / f"submission_fidelity_report-v{version_label}.json"
                _submit_kernel_fidelity.validate_submit_kernel_runtime_fidelity(
                    artifact_mode=artifact_mode,
                    expected_metrics=expected_metrics_payload,
                    actual_metrics_path=metrics_path if isinstance(metrics_path, Path) else None,
                    expected_contract_path=expected_contract_path,
                    runtime_fidelity_path=runtime_fidelity_path,
                    submission_path=local_artifact_path,
                    package_dir=package_dir,
                    report_path=fidelity_report_path,
                    kernel_id=kernel_id,
                    kernel_version=version_label,
                    run_id=run_id,
                    iteration=iteration,
                    previous_report_paths=iter_logs_dir.glob("submission_fidelity_report-v*.json"),
                )
            else:
                _submit_kernel_fidelity.validate_submit_kernel_runtime_fidelity(
                    artifact_mode=artifact_mode,
                    expected_metrics=expected_metrics_payload,
                    actual_metrics_path=metrics_path if isinstance(metrics_path, Path) else None,
                )
    except SubmissionCliError as exc:
        _annotate_notebook_submit_error(
            exc,
            kernel_ref=kernel_id,
            kernel_version=version_label,
            code_output_file_name=code_output_file_name,
            local_artifact_path=local_artifact_path,
        )
        raise
    output_reference = build_notebook_submit_output_reference(
        kernel_id=kernel_id,
        kernel_submission_path=local_artifact_path,
        version_label=version_label,
        copy_submission_artifact=copy_submission_artifact,
        expected_output_file=code_output_file_name,
    )
    submit_reference = output_reference.reference
    review_approval: object | None = None
    execution_permit: object | None = None
    if not dry_run and _is_code_output_artifact_mode(artifact_mode) and review_code_submission is not None:
        reviewed_artifact = output_reference.submission_artifact_path
        output_dir = getattr(kernel_result, "output_dir", None)
        metrics_path = getattr(kernel_result, "metrics_path", None)
        if reviewed_artifact is None or not isinstance(output_dir, Path) or not code_output_file_name:
            raise SubmissionCliError(
                "Completed code submission is missing reviewable output evidence.",
                command=[],
                exit_code=6,
                output=f"kernel={kernel_id}; expected_output={code_output_file_name or '<missing>'}",
            )
        review_kwargs: dict[str, object] = dict(
            slug=slug,
            run_id=run_id,
            iteration=iteration,
            kernel_id=kernel_id,
            kernel_version=str(version_label),
            package_dir=base_dir / slug / "kernels" / run_id / f"submit-iter-{iteration}",
            output_dir=output_dir,
            runtime_logs_dir=iter_logs_dir,
            submission_path=reviewed_artifact,
            metrics_path=metrics_path if isinstance(metrics_path, Path) else None,
            expected_output_file=code_output_file_name,
            message=message,
            review_dir=iter_logs_dir / "submit-codex-review" / f"v{version_label}",
        )
        if fidelity_report_path is not None:
            review_kwargs["fidelity_report_path"] = fidelity_report_path
        review_approval = review_code_submission(**review_kwargs)
        if recheck_code_submission_guard is None:
            raise SubmissionCliError(
                "Code-submission reviewer is configured without its deterministic execution guard.",
                command=[],
                exit_code=6,
            )
        execution_permit = recheck_code_submission_guard(
            approval=review_approval,
            slug=slug,
            kernel_id=kernel_id,
            kernel_version=str(version_label),
            expected_output_file=code_output_file_name,
            submission_path=reviewed_artifact,
            message=message,
        )
    on_message(f"[cyan]submit notebook[/cyan]: {submit_reference.kernel_ref}")
    submit_kwargs = build_kaggle_submit_kernel_kwargs(
        slug=slug,
        reference=submit_reference,
        message=message,
        dry_run=dry_run,
        expected_output_file=code_output_file_name,
    )
    try:
        submit_result = run_kaggle_submit_kernel_with_retry(
            submit_kwargs=submit_kwargs,
            run_kaggle_submit_kernel=run_kaggle_submit_kernel,
            submit_error_types=SubmissionCliError,
            classify_submit_error=classify_submit_error,
            should_retry_ambiguous=should_retry_ambiguous,
            sleep=sleep,
            on_message=on_message,
        )
        if execution_permit is not None:
            if record_code_submission_execution is None:
                raise SubmissionCliError(
                    "Guarded code submission completed without a ledger recorder.",
                    command=[],
                    exit_code=6,
                )
            record_code_submission_execution(
                permit=execution_permit,
                slug=slug,
                message=message,
                submission_path=output_reference.submission_artifact_path,
                run_id=run_id,
                iteration=iteration,
                submission_ref=submit_reference.submission_ref,
            )
    except SubmissionCliError as exc:
        # Preserve the notebook/output identity for submit-abort diagnostics.
        # Otherwise the outer attempt loop can only report its original local
        # inference input, which is not Kaggle's code-submission output.
        _annotate_notebook_submit_error(
            exc,
            kernel_ref=submit_reference.kernel_ref,
            kernel_version=submit_reference.version,
            code_output_file_name=submit_reference.output_file,
            local_artifact_path=output_reference.submission_artifact_path,
        )
        raise
    return submit_result, submit_reference.submission_ref, output_reference.submission_artifact_path


def _is_code_output_artifact_mode(artifact_mode: str | None) -> bool:
    return normalize_notebook_submit_artifact_mode(artifact_mode) in {"gateway", "inference"}


def _resolve_submit_kernel_local_artifact_path(
    *,
    kernel_result: object,
    artifact_mode: str | None,
    code_output_file_name: str | None,
    allow_missing: bool = False,
) -> Path | None:
    discovered = getattr(kernel_result, "submission_path", None)
    discovered_path = discovered if isinstance(discovered, Path) else None
    if not _is_code_output_artifact_mode(artifact_mode) or not code_output_file_name:
        return discovered_path
    if discovered_path is not None and discovered_path.name == code_output_file_name:
        return discovered_path

    output_dir = getattr(kernel_result, "output_dir", None)
    if isinstance(output_dir, Path):
        expected_path = _kernel_outputs.find_output_file(output_dir, code_output_file_name)
        if expected_path is not None:
            return expected_path
    if allow_missing:
        return None

    raise _invalid_code_output_error(
        "completed notebook output does not contain the expected code-submission file",
        kernel_id=str(getattr(kernel_result, "kernel_id", "") or ""),
        code_output_file_name=code_output_file_name,
        discovered_file_name=discovered_path.name if discovered_path is not None else None,
    )


def _invalid_code_output_error(
    detail: str,
    *,
    kernel_id: str,
    code_output_file_name: str | None,
    discovered_file_name: str | None = None,
) -> SubmissionCliError:
    diagnostic = (
        "Invalid code submission output contract: "
        f"{detail}; kernel={kernel_id or '<missing>'}; "
        f"expected_output={code_output_file_name or '<missing>'}"
    )
    if discovered_file_name:
        diagnostic += f"; discovered_output={discovered_file_name}"
    return SubmissionCliError(
        "Notebook code-submission output contract is invalid.",
        command=[],
        exit_code=6,
        output=diagnostic,
        stdout="",
        stderr=diagnostic,
    )


def _annotate_notebook_submit_error(
    exc: SubmissionCliError,
    *,
    kernel_ref: str,
    kernel_version: str | None,
    code_output_file_name: str | None,
    local_artifact_path: Path | None,
) -> None:
    exc.submission_ref = f"kernel:{kernel_ref}" if kernel_ref else ""
    exc.submission_artifact_path = local_artifact_path
    exc.code_output_file_name = str(code_output_file_name or "")
    exc.kernel_ref = kernel_ref
    exc.kernel_version = str(kernel_version or "")


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
    review_code_submission: Callable[..., object] | None = None,
    recheck_code_submission_guard: Callable[..., object] | None = None,
    record_code_submission_execution: Callable[..., object] | None = None,
) -> tuple[object, str, Path | None]:
    iteration = infer_iteration_from_submission_path(submission_path) or 1
    iter_dir = paths.iter_dir(run_id, iteration)
    expected_metrics_payload = _submit_kernel_fidelity.load_expected_submit_metrics_snapshot(
        [
            iter_dir / "metrics.json",
            paths.base_dir / "kernels" / run_id / f"local-iter-{iteration}" / "outputs" / "metrics.json",
            iter_dir / "output" / "metrics.json",
        ]
    )
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
        expected_metrics_payload=expected_metrics_payload,
        review_code_submission=review_code_submission,
        recheck_code_submission_guard=recheck_code_submission_guard,
        record_code_submission_execution=record_code_submission_execution,
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
                return run_submit_kernel(
                    **{
                        **submit_kernel_kwargs,
                        "accelerator": "cpu",
                        "requested_accelerator": submit_kernel_kwargs.get(
                            "requested_accelerator",
                            submit_kernel_kwargs.get("accelerator", ""),
                        ),
                        "capacity_fallback_used": True,
                    }
                )
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
