from __future__ import annotations

from kagglebot.exceptions import KaggleCliError, KernelFailedError
from kagglebot.kernel_errors import (
    fingerprint_error,
    format_kernel_error,
    is_kernel_registration_error,
    record_kernel_error,
)


def test_fingerprint_error_normalizes_whitespace() -> None:
    assert fingerprint_error("same   error\nmessage") == fingerprint_error("same error message")


def test_format_kernel_error_includes_cli_output() -> None:
    error = KaggleCliError("status failed", output="GET /api/v1/kernels/status returned 404")

    formatted = format_kernel_error(error)

    assert "KaggleCliError: status failed" in formatted
    assert "Kaggle CLI output:" in formatted
    assert "kernels/status returned 404" in formatted


def test_format_kernel_error_includes_active_traceback() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        formatted = format_kernel_error(exc)

    assert "RuntimeError: boom" in formatted
    assert "Traceback (most recent call last):" in formatted


def test_is_kernel_registration_error_detects_missing_kernel_after_push() -> None:
    error = KernelFailedError("Kernel not found after push: owner/demo")

    assert is_kernel_registration_error(error)


def test_is_kernel_registration_error_detects_status_endpoint_failure() -> None:
    error = KaggleCliError("not found", output="404 while calling /api/v1/kernels/status/owner/demo")

    assert is_kernel_registration_error(error)


def test_is_kernel_registration_error_rejects_unrelated_errors() -> None:
    assert not is_kernel_registration_error(KernelFailedError("kernel failed during execution"))
    assert not is_kernel_registration_error(KaggleCliError("bad request", output="competitions/list failed"))


def test_record_kernel_error_writes_numbered_and_latest_logs(tmp_path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "local_kernel_stdout.log").write_text("line 1\nline 2\n", encoding="utf-8")
    fingerprints: dict[str, int] = {}

    record_kernel_error(
        logs_dir=logs_dir,
        attempt=3,
        error_text="RuntimeError: boom",
        error_fingerprints=fingerprints,
        output_dir=output_dir,
    )

    latest = (logs_dir / "kernel_error.txt").read_text(encoding="utf-8")
    numbered = (logs_dir / "kernel_error-03.txt").read_text(encoding="utf-8")
    assert latest == numbered
    assert "kernel_attempt: 3" in latest
    assert "error_repeat: 1" in latest
    assert "--- kernel log tail ---" in latest
    assert "line 2" in latest


def test_record_kernel_error_aborts_repeated_fingerprint(tmp_path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    fingerprints: dict[str, int] = {}

    record_kernel_error(
        logs_dir=logs_dir,
        attempt=1,
        error_text="RuntimeError: repeat",
        error_fingerprints=fingerprints,
        max_repeats=1,
    )
    try:
        record_kernel_error(
            logs_dir=logs_dir,
            attempt=2,
            error_text="RuntimeError: repeat",
            error_fingerprints=fingerprints,
            max_repeats=1,
        )
    except KernelFailedError as exc:
        assert "same error" in str(exc)
    else:  # pragma: no cover - explicit assertion path
        raise AssertionError("Expected KernelFailedError")
