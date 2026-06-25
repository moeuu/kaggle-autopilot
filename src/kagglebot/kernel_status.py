from __future__ import annotations


def parse_kernel_status(output: str) -> str:
    """Normalize Kaggle CLI kernel status output to a small stable vocabulary."""
    text = output.lower()
    if (
        "failure message" in text
        or "your notebook failed" in text
        or "kernelworkerstatus.error" in text
        or "kernelworkerstatus.failed" in text
        or 'status "error"' in text
        or 'status "failed"' in text
        or " failed" in text
    ):
        return "failed"
    if "complete" in text or "success" in text:
        return "complete"
    if "queued" in text or "pending" in text:
        return "queued"
    if "running" in text:
        return "running"
    return "unknown"


def is_kernel_status_running(status: str) -> bool:
    return status.lower() in {"running", "queued"}


def is_kernel_status_queued(status: str) -> bool:
    return status.lower() == "queued"


def is_kernel_status_complete(status: str) -> bool:
    return status.lower() == "complete"


def is_kernel_status_failed(status: str) -> bool:
    return status.lower() == "failed"
