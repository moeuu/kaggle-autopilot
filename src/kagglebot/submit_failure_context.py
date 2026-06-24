from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kagglebot.submission.guard import normalize_error_text
from kagglebot.submit_failure_policy import normalize_loaded_submit_failure_context

SUBMIT_FAILURE_CONTEXT_FILENAME = "submit_failure_context.json"


@dataclass(frozen=True)
class StaleSubmitAutofixDecision:
    clear_repaired_path: bool
    failure_context_updates: dict[str, object]


def submit_failure_context_path(run_dir: Path) -> Path:
    return run_dir / SUBMIT_FAILURE_CONTEXT_FILENAME


def load_submit_failure_context(run_dir: Path) -> dict[str, object]:
    path = submit_failure_context_path(run_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return normalize_loaded_submit_failure_context(payload)


def save_submit_failure_context(run_dir: Path, payload: dict[str, object]) -> None:
    path = submit_failure_context_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def mark_submit_failure_context_resolved(
    *,
    run_dir: Path,
    resolution: str,
    submission_ref: str | None = None,
) -> None:
    payload = load_submit_failure_context(run_dir)
    if not payload:
        return
    payload["active"] = False
    payload["resolution"] = resolution
    payload["resolved_at"] = datetime.now(UTC).isoformat()
    if submission_ref is not None:
        payload["resolved_submission_ref"] = submission_ref
    save_submit_failure_context(run_dir, payload)


def path_from_submit_reference(value: object) -> Path | None:
    text = str(value or "").strip()
    if not text or text.startswith("kernel:"):
        return None
    try:
        return Path(text)
    except TypeError:
        return None


def decide_stale_submit_autofix_artifact(
    *,
    run_state: dict[str, object],
    failure_context: dict[str, object],
    submission_path: Path,
    now_iso: str,
) -> StaleSubmitAutofixDecision | None:
    repaired_path = path_from_submit_reference(run_state.get("submit_autofix_submission_path"))
    if repaired_path is None:
        return None
    if not failure_context:
        return None
    failed_artifact_path = path_from_submit_reference(
        failure_context.get("submission_artifact_path") or failure_context.get("submission_ref")
    )
    if failed_artifact_path is None:
        return None
    if submission_path == failed_artifact_path or submission_path == repaired_path:
        return None
    return StaleSubmitAutofixDecision(
        clear_repaired_path=True,
        failure_context_updates={
            "stale_repaired_artifact_cleared_at": now_iso,
            "superseded_by_submission_path": str(submission_path),
        },
    )


def resolve_submit_autofix_submission_artifact(
    *,
    run_state: dict[str, object],
    latest_submit_attempt: dict[str, object],
    failure_context: dict[str, object],
    fallback_iteration_dirs: Iterable[Path],
    resolve_iteration_submission_artifact: Callable[[Path], Path | None],
) -> Path | None:
    candidates = (
        run_state.get("submit_autofix_submission_path"),
        failure_context.get("submission_artifact_path"),
        latest_submit_attempt.get("sub_path"),
        run_state.get("last_submission_path"),
    )
    for candidate in candidates:
        path = path_from_submit_reference(candidate)
        if path is None:
            continue
        if path.exists() and path.is_file():
            return path
    for iter_dir in fallback_iteration_dirs:
        if not iter_dir.exists():
            continue
        resolved = resolve_iteration_submission_artifact(iter_dir)
        if resolved is not None and resolved.exists():
            return resolved
    return None


def submit_file_fix_contract_satisfied(
    *,
    run_state: dict[str, object],
    baseline_path: Path | None,
    baseline_sha256: str | None,
    sha256_or_none: Callable[[Path | None], str | None],
) -> bool:
    candidate = path_from_submit_reference(run_state.get("submit_autofix_submission_path"))
    if candidate is None or not candidate.exists():
        return False
    candidate_sha256 = sha256_or_none(candidate)
    if candidate_sha256 is None:
        return False
    if baseline_path is None or baseline_sha256 is None:
        return True
    return candidate != baseline_path or candidate_sha256 != baseline_sha256


def format_submit_autofix_context(
    *,
    failure_context: dict[str, object],
    run_state: dict[str, object],
    latest_submit_attempt: dict[str, object],
) -> str:
    lines: list[str] = []
    if failure_context:
        lines.append("submit_failure_context:")
        for key in (
            "ts",
            "active",
            "repair_target",
            "repairable",
            "reason",
            "error_kind",
            "fingerprint",
            "submit_mode",
            "artifact_mode",
            "submission_ref",
            "submission_artifact_path",
            "submission_artifact_sha256",
            "manual_next_step",
            "summary",
        ):
            value = failure_context.get(key)
            if value in (None, "", []):
                continue
            lines.append(f"- {key}: {value}")
        latest_context_attempt = failure_context.get("latest_submit_attempt")
        if isinstance(latest_context_attempt, dict) and latest_context_attempt:
            lines.append("failure_context_latest_submit_attempt:")
            _append_submit_attempt_excerpt(lines, latest_context_attempt)
        run_state_excerpt = failure_context.get("run_state_excerpt")
        if isinstance(run_state_excerpt, dict) and run_state_excerpt:
            lines.append("failure_context_run_state:")
            for key in (
                "submit_attempted",
                "submit_ok",
                "last_reason",
                "last_error_kind",
                "last_submission_path",
                "submit_autofix_submission_path",
            ):
                value = run_state_excerpt.get(key)
                if value in (None, ""):
                    continue
                lines.append(f"- {key}: {value}")

    lines.append("run_state:")
    for key in (
        "submit_attempted",
        "submit_ok",
        "last_error_kind",
        "last_reason",
        "last_action",
        "last_submit_fingerprint",
        "last_submission_path",
        "submit_autofix_submission_path",
    ):
        value = run_state.get(key)
        if value in (None, ""):
            continue
        lines.append(f"- {key}: {value}")

    if latest_submit_attempt:
        lines.append("latest_submit_attempt:")
        _append_submit_attempt_excerpt(lines, latest_submit_attempt)
        stdout_tail = normalize_error_text(str(latest_submit_attempt.get("stdout_tail") or ""), max_chars=1200)
        stderr_tail = normalize_error_text(str(latest_submit_attempt.get("stderr_tail") or ""), max_chars=1200)
        if stdout_tail:
            lines.append(f"- stdout_tail: {stdout_tail}")
        if stderr_tail:
            lines.append(f"- stderr_tail: {stderr_tail}")
    return "\n".join(lines).strip()


def _append_submit_attempt_excerpt(lines: list[str], attempt: dict[str, object]) -> None:
    for key in (
        "ts",
        "ok",
        "exit_code",
        "error_kind",
        "reason",
        "action_taken",
        "fingerprint",
        "sub_path",
    ):
        value = attempt.get(key)
        if value in (None, ""):
            continue
        lines.append(f"- {key}: {value}")
