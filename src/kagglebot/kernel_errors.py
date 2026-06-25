from __future__ import annotations

import hashlib
import traceback
from pathlib import Path

from kagglebot.exceptions import KaggleCliError, KernelFailedError
from kagglebot.kernel_logs import collect_log_tail

DEFAULT_SAME_KERNEL_ERROR_REPEATS = 2


def format_kernel_error(exc: Exception) -> str:
    trace = traceback.format_exc()
    header = f"{exc.__class__.__name__}: {exc}".strip()
    if isinstance(exc, KaggleCliError) and getattr(exc, "output", ""):
        header = f"{header}\nKaggle CLI output:\n{exc.output}".strip()
    if trace and trace != "NoneType: None\n":
        return f"{header}\n{trace}".strip()
    return header


def fingerprint_error(message: str) -> str:
    normalized = " ".join(message.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def is_kernel_registration_error(exc: Exception) -> bool:
    if isinstance(exc, KernelFailedError) and "kernel not found after push" in str(exc).lower():
        return True
    if isinstance(exc, KaggleCliError) and "kernels/status" in str(getattr(exc, "output", "")).lower():
        return True
    return False


def record_kernel_error(
    *,
    logs_dir: Path,
    attempt: int,
    error_text: str,
    error_fingerprints: dict[str, int],
    max_repeats: int | None = None,
    output_dir: Path | None = None,
) -> None:
    enriched_error = error_text
    if output_dir is not None and output_dir.exists():
        log_tail = collect_log_tail(output_dir, max_lines=200)
        if log_tail and log_tail not in enriched_error:
            enriched_error = f"{enriched_error}\n\n--- kernel log tail ---\n{log_tail}"
    fingerprint = fingerprint_error(enriched_error)
    error_fingerprints[fingerprint] = error_fingerprints.get(fingerprint, 0) + 1
    repeat_limit = DEFAULT_SAME_KERNEL_ERROR_REPEATS if max_repeats is None else max_repeats
    if repeat_limit is not None and error_fingerprints[fingerprint] > repeat_limit:
        raise KernelFailedError(
            "Kernel failure repeated with the same error; aborting auto-fix loop to avoid an infinite retry."
        )
    attempt_tag = f"{attempt:02d}"
    header = (
        f"kernel_attempt: {attempt}\n"
        f"error_fingerprint: {fingerprint}\n"
        f"error_repeat: {error_fingerprints[fingerprint]}\n"
    )
    numbered_path = logs_dir / f"kernel_error-{attempt_tag}.txt"
    numbered_path.write_text(header + enriched_error + "\n", encoding="utf-8")
    (logs_dir / "kernel_error.txt").write_text(header + enriched_error + "\n", encoding="utf-8")
