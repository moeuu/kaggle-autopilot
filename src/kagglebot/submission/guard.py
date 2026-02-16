from __future__ import annotations

import hashlib
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from re import Pattern

from kagglebot.exceptions import SubmissionCliError

_PERMANENT_RULES: tuple[tuple[str, tuple[Pattern[str], ...]], ...] = (
    (
        "rules_not_accepted",
        (
            re.compile(r"accept the rules", flags=re.IGNORECASE),
            re.compile(r"you must accept", flags=re.IGNORECASE),
            re.compile(r"rules have not been accepted", flags=re.IGNORECASE),
            re.compile(r"competition rules not accepted", flags=re.IGNORECASE),
        ),
    ),
    (
        "authentication",
        (
            re.compile(r"kaggle\.json", flags=re.IGNORECASE),
            re.compile(r"unauthorized", flags=re.IGNORECASE),
            re.compile(r"\b401\b", flags=re.IGNORECASE),
            re.compile(r"api token", flags=re.IGNORECASE),
            re.compile(r"api credentials", flags=re.IGNORECASE),
            re.compile(r"no kaggle api credentials", flags=re.IGNORECASE),
        ),
    ),
    (
        "competition_unavailable",
        (
            re.compile(r"competition is not accepting submissions", flags=re.IGNORECASE),
            re.compile(r"not allowed", flags=re.IGNORECASE),
            re.compile(r"not found", flags=re.IGNORECASE),
            re.compile(r"closed", flags=re.IGNORECASE),
        ),
    ),
    (
        "submission_limit",
        (
            re.compile(r"maximum number of submissions", flags=re.IGNORECASE),
            re.compile(r"submission limit", flags=re.IGNORECASE),
            re.compile(r"max submissions", flags=re.IGNORECASE),
        ),
    ),
)

_TRANSIENT_RULES: tuple[tuple[str, tuple[Pattern[str], ...]], ...] = (
    (
        "network_or_timeout",
        (
            re.compile(r"timed out", flags=re.IGNORECASE),
            re.compile(r"timeout", flags=re.IGNORECASE),
            re.compile(r"connectionerror", flags=re.IGNORECASE),
            re.compile(r"temporary failure", flags=re.IGNORECASE),
            re.compile(r"temporarily unavailable", flags=re.IGNORECASE),
            re.compile(r"\b502\b", flags=re.IGNORECASE),
            re.compile(r"\b503\b", flags=re.IGNORECASE),
            re.compile(r"\b504\b", flags=re.IGNORECASE),
            re.compile(r"bad gateway", flags=re.IGNORECASE),
            re.compile(r"service unavailable", flags=re.IGNORECASE),
            re.compile(r"gateway timeout", flags=re.IGNORECASE),
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
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b", flags=re.IGNORECASE), "<DATE>"),
    (re.compile(r"\b\d{8}T\d{6}Z-[a-f0-9]{8}\b", flags=re.IGNORECASE), "<RUN_ID>"),
)


@dataclass(frozen=True)
class SubmitResult:
    returncode: int
    stdout: str
    stderr: str
    command: list[str]
    duration_sec: float

    @property
    def output(self) -> str:
        return (self.stdout + self.stderr).strip()


def run_kaggle_submit(
    *,
    slug: str,
    submission_file: Path,
    message: str,
    dry_run: bool = False,
) -> SubmitResult:
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
    if dry_run:
        return SubmitResult(
            returncode=0,
            stdout="",
            stderr="",
            command=args,
            duration_sec=0.0,
        )

    start = time.monotonic()
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SubmissionCliError(
            "Kaggle CLI submit failed: kaggle executable not found on PATH.",
            command=args,
            exit_code=127,
            output="",
            stdout="",
            stderr=str(exc),
        ) from exc
    duration = time.monotonic() - start

    stdout_text = completed.stdout or ""
    stderr_text = completed.stderr or ""
    if completed.returncode != 0:
        stdout_tail = _tail_text(stdout_text)
        stderr_tail = _tail_text(stderr_text)
        raise SubmissionCliError(
            "Kaggle CLI submit failed.",
            command=args,
            exit_code=completed.returncode,
            output=_tail_text(f"{stdout_text}\n{stderr_text}"),
            stdout=stdout_tail,
            stderr=stderr_tail,
        )
    return SubmitResult(
        returncode=completed.returncode,
        stdout=stdout_text,
        stderr=stderr_text,
        command=args,
        duration_sec=duration,
    )


def _tail_text(text: str, *, max_lines: int = 200, max_chars: int = 6000) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) > max_lines:
        text = "\n".join(lines[-max_lines:])
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


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


def classify_submit_error(stdout: str, stderr: str, returncode: int) -> dict[str, object]:
    merged = f"{stdout}\n{stderr}"
    for reason, patterns in _PERMANENT_RULES:
        if any(pattern.search(merged) for pattern in patterns):
            return {"kind": "permanent", "reason": reason, "retry_after_seconds": None}
    for reason, patterns in _TRANSIENT_RULES:
        if any(pattern.search(merged) for pattern in patterns):
            return {"kind": "transient", "reason": reason, "retry_after_seconds": 2}
    if returncode == 0:
        return {"kind": "unknown", "reason": "no_error", "retry_after_seconds": None}
    return {"kind": "unknown", "reason": "unclassified_submit_error", "retry_after_seconds": None}
