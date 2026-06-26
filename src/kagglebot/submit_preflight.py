from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kagglebot import submit_abort_specs as _submit_abort_specs
from kagglebot.submit_abort_specs import SubmitAbortSpec


@dataclass(frozen=True)
class SubmitPreparedSubmissionResolution:
    prepared_submission_path: Path | None
    abort_spec: SubmitAbortSpec | None = None


@dataclass(frozen=True)
class SubmitPreparedRunContext:
    prepared_submission_path: Path
    prepared_submission_sha: str


@dataclass(frozen=True)
class SubmitRulesAcceptanceResolution:
    rules_accepted: bool
    abort_spec: SubmitAbortSpec | None = None


def resolve_prepared_submission_for_submit(
    *,
    input_submission_path: Path,
    validate_and_prepare: Callable[[Path], Path],
    validation_error_types: tuple[type[BaseException], ...],
    validation_exit_code: int | None,
    compute_error_fingerprint: Callable[[str, str], str],
) -> SubmitPreparedSubmissionResolution:
    try:
        return SubmitPreparedSubmissionResolution(prepared_submission_path=validate_and_prepare(input_submission_path))
    except validation_error_types as exc:
        return SubmitPreparedSubmissionResolution(
            prepared_submission_path=None,
            abort_spec=_submit_abort_specs.build_local_submission_validation_abort_spec(
                error=exc,
                exit_code=validation_exit_code,
                compute_error_fingerprint=compute_error_fingerprint,
            ),
        )


def require_prepared_submission_path(
    resolution: SubmitPreparedSubmissionResolution,
    *,
    build_error: Callable[[str], BaseException],
) -> Path:
    if resolution.prepared_submission_path is None:
        raise build_error("Submit validation did not produce a prepared submission path.")
    return resolution.prepared_submission_path


def prepare_submission_for_run_or_abort(
    *,
    input_submission_path: Path,
    validate_and_prepare: Callable[[Path], Path],
    validation_error_types: tuple[type[BaseException], ...],
    validation_exit_code: int | None,
    compute_error_fingerprint: Callable[[str, str], str],
    submit_aborter: object,
    submit_attempt_recorder: object,
    code_fingerprint: str,
    compute_submission_sha256: Callable[[Path | None], str | None],
    build_error: Callable[[str], BaseException],
) -> SubmitPreparedRunContext:
    prepared_resolution = resolve_prepared_submission_for_submit(
        input_submission_path=input_submission_path,
        validate_and_prepare=validate_and_prepare,
        validation_error_types=validation_error_types,
        validation_exit_code=validation_exit_code,
        compute_error_fingerprint=compute_error_fingerprint,
    )
    if prepared_resolution.abort_spec is not None:
        return submit_aborter.abort(
            submission_ref=input_submission_path,
            code_fingerprint=code_fingerprint,
            **_submit_abort_specs.build_submit_abort_spec_kwargs(prepared_resolution.abort_spec),
            submit_attempt_recorder=submit_attempt_recorder,
        )
    prepared_submission_path = require_prepared_submission_path(
        prepared_resolution,
        build_error=build_error,
    )
    return SubmitPreparedRunContext(
        prepared_submission_path=prepared_submission_path,
        prepared_submission_sha=str(compute_submission_sha256(prepared_submission_path) or "").strip(),
    )


def resolve_rules_acceptance_for_submit(
    *,
    check_rules_accepted: Callable[[], bool],
    cli_error_types: tuple[type[BaseException], ...],
    is_missing_credentials_error: Callable[[BaseException], bool],
    rules_not_accepted_exit_code: int | None,
    compute_error_fingerprint: Callable[[str, str], str],
) -> SubmitRulesAcceptanceResolution:
    try:
        rules_accepted = check_rules_accepted()
    except cli_error_types as exc:
        if not is_missing_credentials_error(exc):
            raise
        return SubmitRulesAcceptanceResolution(
            rules_accepted=False,
            abort_spec=_submit_abort_specs.build_kaggle_credentials_missing_abort_spec(
                stdout=str(getattr(exc, "stdout", "") or ""),
                stderr=str(getattr(exc, "stderr", "") or ""),
                output=str(getattr(exc, "output", "") or ""),
                exit_code=getattr(exc, "exit_code", None),
                compute_error_fingerprint=compute_error_fingerprint,
            ),
        )

    if not rules_accepted:
        return SubmitRulesAcceptanceResolution(
            rules_accepted=False,
            abort_spec=_submit_abort_specs.build_rules_not_accepted_abort_spec(
                exit_code=rules_not_accepted_exit_code,
                compute_error_fingerprint=compute_error_fingerprint,
            ),
        )
    return SubmitRulesAcceptanceResolution(rules_accepted=True)
