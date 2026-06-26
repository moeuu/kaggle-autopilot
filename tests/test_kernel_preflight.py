from __future__ import annotations

import pytest

from kagglebot.exceptions import KernelFailedError
from kagglebot.kernel_preflight import run_kernel_source_preflight_fixes


def test_run_kernel_source_preflight_fixes_runs_callback_until_valid(tmp_path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "kernel.py").write_text("OUT = '/tmp/submission.csv'\n", encoding="utf-8")
    calls: list[tuple[str, int]] = []
    messages: list[str] = []

    def run_kernel_fix(error: str, attempt: int) -> None:
        calls.append((error, attempt))
        (kernel_dir / "kernel.py").write_text(
            "\n".join(
                [
                    "SUBMISSION = '/kaggle/working/submission.csv'",
                    "METRICS = '/kaggle/working/metrics.json'",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    run_kernel_source_preflight_fixes(
        kernel_source_dir=kernel_dir,
        dry_run=False,
        max_attempts=2,
        format_error=lambda exc: str(exc),
        run_kernel_fix=run_kernel_fix,
        on_message=messages.append,
        implementation_agent_alias="codex",
    )

    assert len(calls) == 1
    assert calls[0][1] == 1
    assert messages


def test_run_kernel_source_preflight_fixes_raises_missing_kernel_without_fix(tmp_path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()

    with pytest.raises(RuntimeError, match="requires kernel.py"):
        run_kernel_source_preflight_fixes(
            kernel_source_dir=kernel_dir,
            dry_run=False,
            max_attempts=2,
            format_error=lambda exc: f"{type(exc).__name__}: {exc}",
            run_kernel_fix=lambda error, attempt: None,
        )


def test_run_kernel_source_preflight_fixes_raises_on_dry_run(tmp_path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "kernel.py").write_text("OUT = '/tmp/submission.csv'\n", encoding="utf-8")

    with pytest.raises(KernelFailedError):
        run_kernel_source_preflight_fixes(
            kernel_source_dir=kernel_dir,
            dry_run=True,
            max_attempts=2,
            format_error=lambda exc: str(exc),
            run_kernel_fix=lambda error, attempt: None,
        )


def test_run_kernel_source_preflight_fixes_enforces_attempt_limit(tmp_path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "kernel.py").write_text("OUT = '/tmp/submission.csv'\n", encoding="utf-8")

    with pytest.raises(KernelFailedError, match="after automatic fixes"):
        run_kernel_source_preflight_fixes(
            kernel_source_dir=kernel_dir,
            dry_run=False,
            max_attempts=1,
            format_error=lambda exc: str(exc),
            run_kernel_fix=lambda error, attempt: None,
        )
