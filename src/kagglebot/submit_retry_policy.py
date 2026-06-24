from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SameSubmissionPathDecision:
    action: str
    reason: str
    message: str
    fingerprint: str


def compute_submit_code_fingerprint(
    *,
    src_root: Path,
    kernel_source_dir: Path,
    sha256_or_none: Callable[[Path | None], str | None],
) -> str:
    """Compute a stable fingerprint of submit-relevant local code."""
    hasher = hashlib.sha256()
    root_specs = (
        ("src", src_root),
        ("kernel", kernel_source_dir),
    )
    for label, root in root_specs:
        if not root.exists() or not root.is_dir():
            hasher.update(f"{label}:<missing>\n".encode())
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts:
                continue
            if path.suffix.lower() in {".pyc", ".pyo"}:
                continue
            try:
                rel = path.relative_to(root).as_posix()
            except ValueError:
                rel = path.name
            hasher.update(f"{label}:{rel}\n".encode())
            hasher.update((sha256_or_none(path) or "missing").encode())
            hasher.update(b"\n")
    return hasher.hexdigest()


def decide_same_submission_path_action(
    *,
    run_state: dict[str, object],
    latest_submit_attempt: dict[str, object],
    prepared_submission_path: Path,
    current_submission_sha: str,
    submit_code_fingerprint: str,
    allow_force: bool,
    notebook_submit_required: bool,
) -> SameSubmissionPathDecision:
    if allow_force or notebook_submit_required:
        return _same_path_not_applicable()

    last_submission_path = str(run_state.get("last_submission_path") or "").strip()
    if not last_submission_path or Path(last_submission_path) != prepared_submission_path:
        return _same_path_not_applicable()

    last_submission_sha = str(latest_submit_attempt.get("sub_sha256") or "").strip()
    last_reason = str(run_state.get("last_reason") or "").strip().lower()
    last_code_fingerprint = str(run_state.get("last_submit_code_fingerprint") or "").strip()
    allow_same_path_retry_reasons = {
        "bad_request",
        "notebook_only_submission_required",
        "unclassified_submit_error",
    }
    if last_reason in allow_same_path_retry_reasons:
        return SameSubmissionPathDecision(
            action="retry",
            reason="previous_submit_failure_allows_notebook_fallback",
            message=(
                "[yellow]submit retry[/yellow]: previous submit failed with "
                f"reason={last_reason}; retrying same artifact to allow notebook fallback."
            ),
            fingerprint="",
        )
    if last_submission_sha and current_submission_sha and last_submission_sha != current_submission_sha:
        return SameSubmissionPathDecision(
            action="retry",
            reason="submission_contents_changed",
            message=(
                "[yellow]submit retry[/yellow]: same artifact path but submission file contents changed; "
                "retrying repaired artifact."
            ),
            fingerprint="",
        )
    if last_code_fingerprint and last_code_fingerprint != submit_code_fingerprint:
        return SameSubmissionPathDecision(
            action="retry",
            reason="submit_code_changed",
            message="[yellow]submit retry[/yellow]: same artifact path but submit code changed; retrying in this run.",
            fingerprint="",
        )

    return SameSubmissionPathDecision(
        action="skip",
        reason="same_submission_path_reused_in_run",
        message="[yellow]submit skipped[/yellow]: same submission file already attempted in this run",
        fingerprint=str(run_state.get("last_submit_fingerprint") or run_state.get("last_fingerprint") or ""),
    )


def consume_same_submit_fingerprint_retry_allowance(
    *,
    run_state: dict[str, object],
    fingerprint: str,
    code_fingerprint: str,
    save_run_state: Callable[[dict[str, object]], None],
) -> bool:
    """Allow one repeated-fingerprint retry after submit-relevant code changes."""
    last_code_fingerprint = str(run_state.get("last_submit_code_fingerprint") or "").strip()
    prior_error_fingerprint = str(
        run_state.get("last_submit_fingerprint") or run_state.get("last_fingerprint") or ""
    ).strip()
    if not code_fingerprint:
        return False

    consumed_code_fingerprint = str(run_state.get("same_fp_allowance_code_fingerprint") or "").strip()
    consumed_error_fingerprint = str(run_state.get("same_fp_allowance_error_fingerprint") or "").strip()
    if consumed_code_fingerprint == code_fingerprint and consumed_error_fingerprint == fingerprint:
        return False

    # Backward compatibility for runs recorded before code_fingerprint tracking existed.
    # In that case we cannot compare "before vs after" code reliably, so allow exactly once.
    if not last_code_fingerprint:
        if not prior_error_fingerprint or prior_error_fingerprint != fingerprint:
            return False
        _record_same_fingerprint_allowance(
            run_state=run_state,
            fingerprint=fingerprint,
            code_fingerprint=code_fingerprint,
            save_run_state=save_run_state,
        )
        return True

    if code_fingerprint == last_code_fingerprint:
        return False

    _record_same_fingerprint_allowance(
        run_state=run_state,
        fingerprint=fingerprint,
        code_fingerprint=code_fingerprint,
        save_run_state=save_run_state,
    )
    return True


def _record_same_fingerprint_allowance(
    *,
    run_state: dict[str, object],
    fingerprint: str,
    code_fingerprint: str,
    save_run_state: Callable[[dict[str, object]], None],
) -> None:
    updates = {
        "same_fp_allowance_code_fingerprint": code_fingerprint,
        "same_fp_allowance_error_fingerprint": fingerprint,
    }
    run_state.update(updates)
    save_run_state(updates)


def _same_path_not_applicable() -> SameSubmissionPathDecision:
    return SameSubmissionPathDecision(action="proceed", reason="", message="", fingerprint="")
