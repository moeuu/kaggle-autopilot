from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from kagglebot.exceptions import SubmissionCliError
from kagglebot.exec_utils import CommandResult, run_command

_PERMANENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "rules_not_accepted",
        (
            "accept the rules",
            "you must accept",
            "rules have not been accepted",
            "competition rules not accepted",
        ),
    ),
    (
        "authentication",
        (
            "kaggle.json",
            "unauthorized",
            "401",
            "api token",
            "api credentials",
        ),
    ),
    (
        "competition_unavailable",
        (
            "competition is not accepting submissions",
            "not allowed",
            "not found",
            "closed",
        ),
    ),
    (
        "submission_limit",
        (
            "maximum number of submissions",
            "submission limit",
            "max submissions",
        ),
    ),
)

_TRANSIENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "network_or_timeout",
        (
            "timed out",
            "timeout",
            "connectionerror",
            "temporarily unavailable",
            "temporary failure",
            "502",
            "503",
            "504",
        ),
    ),
)

_NORMALIZE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"/home/[^\s]+", flags=re.IGNORECASE), "<PATH>"),
    (re.compile(r"artifacts/[^\s]+", flags=re.IGNORECASE), "<ARTIFACT_PATH>"),
    (re.compile(r"runs/[^\s]+", flags=re.IGNORECASE), "<RUN_PATH>"),
    (
        re.compile(
            r"\b\d{4}-\d{2}-\d{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|[+-][0-9:]+)?\b",
            flags=re.IGNORECASE,
        ),
        "<DATETIME>",
    ),
    (re.compile(r"\b\d{8}T\d{6}Z-[a-f0-9]{8}\b", flags=re.IGNORECASE), "<RUN_ID>"),
)


@dataclass(frozen=True)
class SubmitErrorClassification:
    kind: str
    reason: str
    retry_after_seconds: float | None


def run_kaggle_submit(
    *,
    slug: str,
    submission_file: Path,
    message: str,
    dry_run: bool = False,
) -> CommandResult:
    args = [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        slug,
        "-f",
        str(submission_file),
        "-m",
        message,
    ]
    result = run_command(args, dry_run=dry_run)
    if result.returncode != 0:
        raise SubmissionCliError(
            "Kaggle CLI submit failed.",
            command=args,
            exit_code=result.returncode,
            output=result.output,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    return result


def normalize_error_text(text: str, *, max_chars: int = 6000) -> str:
    normalized = text or ""
    normalized = normalized.replace("\r", "\n")
    for pattern, replacement in _NORMALIZE_PATTERNS:
        normalized = pattern.sub(replacement, normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if len(normalized) > max_chars:
        normalized = normalized[-max_chars:]
    return normalized


def compute_error_fingerprint(stdout: str, stderr: str) -> str:
    merged = f"{stdout}\n{stderr}".strip()
    normalized = normalize_error_text(merged)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def classify_submit_error(stdout: str, stderr: str, returncode: int) -> SubmitErrorClassification:
    merged = f"{stdout}\n{stderr}".lower()
    for reason, needles in _PERMANENT_RULES:
        if any(needle in merged for needle in needles):
            return SubmitErrorClassification(kind="permanent", reason=reason, retry_after_seconds=None)
    for reason, needles in _TRANSIENT_RULES:
        if any(needle in merged for needle in needles):
            return SubmitErrorClassification(kind="transient", reason=reason, retry_after_seconds=1.0)
    if returncode == 0:
        return SubmitErrorClassification(kind="unknown", reason="no_error", retry_after_seconds=None)
    return SubmitErrorClassification(kind="unknown", reason="unclassified_submit_error", retry_after_seconds=None)
