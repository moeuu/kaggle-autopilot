from __future__ import annotations

import hashlib
import traceback

from kagglebot.exceptions import KaggleCliError, KernelFailedError


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
