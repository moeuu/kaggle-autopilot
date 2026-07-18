from __future__ import annotations

import pytest

from kagglebot.exceptions import KernelFailedError
from kagglebot.kernel_preflight import (
    KernelFixResult,
    KernelPreflightFailure,
    run_kernel_source_preflight_fixes,
)


def test_run_kernel_source_preflight_fixes_runs_callback_until_valid(tmp_path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "kernel.py").write_text("OUT = '/tmp/submission.csv'\n", encoding="utf-8")
    calls: list[tuple[str, int]] = []
    messages: list[str] = []

    def run_kernel_fix(error: str, attempt: int) -> KernelFixResult:
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
        return KernelFixResult(agent_exit_code=0, repo_changed=False)

    run_kernel_source_preflight_fixes(
        kernel_source_dir=kernel_dir,
        dry_run=False,
        max_attempts=2,
        format_error=lambda exc: str(exc),
        run_kernel_fix=run_kernel_fix,
        diagnostics_dir=tmp_path / "autofix",
        on_message=messages.append,
        implementation_agent_alias="codex",
    )

    assert len(calls) == 1
    assert calls[0][1] == 1
    assert "check_name: kernel_source_contract" in calls[0][0]
    assert "stderr:" in calls[0][0]
    assert "Kernel sources do not reference metrics.json output" in calls[0][0]
    diagnostic_path = tmp_path / "autofix" / "attempt-1" / "kernel-preflight-error.txt"
    assert diagnostic_path.is_file()
    assert "command_or_rule:" in diagnostic_path.read_text(encoding="utf-8")
    assert messages


def test_run_kernel_source_preflight_fixes_accepts_no_diff_when_post_check_passes(monkeypatch, tmp_path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text("# unchanged\n", encoding="utf-8")
    failure = KernelPreflightFailure(
        check_name="static_rule",
        kernel_path=kernel_path,
        command_or_rule="demo rule",
        returncode=None,
        stdout="",
        stderr="original stderr",
        source_excerpt="kernel.py:1",
        kernel_sha256="a" * 64,
    )
    checks = iter((failure, None))
    monkeypatch.setattr("kagglebot.kernel_preflight.check_kernel_source_preflight", lambda **kwargs: next(checks))
    calls: list[tuple[str, int]] = []

    run_kernel_source_preflight_fixes(
        kernel_source_dir=kernel_dir,
        dry_run=False,
        max_attempts=1,
        format_error=lambda exc: str(exc),
        run_kernel_fix=lambda error, attempt: calls.append((error, attempt)) or KernelFixResult(),
    )

    assert len(calls) == 1
    assert calls[0][1] == 1
    assert "original stderr" in calls[0][0]


def test_run_kernel_source_preflight_fixes_retries_when_generated_kernel_changes(tmp_path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text("OUT = '/tmp/submission.csv'\n", encoding="utf-8")
    calls: list[int] = []

    def run_kernel_fix(error: str, attempt: int) -> KernelFixResult:  # noqa: ARG001
        calls.append(attempt)
        if attempt == 1:
            kernel_path.write_text("OUT = '/tmp/other.csv'\n", encoding="utf-8")
        else:
            kernel_path.write_text(
                "SUBMISSION = '/kaggle/working/submission.csv'\nMETRICS = '/kaggle/working/metrics.json'\n",
                encoding="utf-8",
            )
        return KernelFixResult(agent_exit_code=0, repo_changed=False)

    run_kernel_source_preflight_fixes(
        kernel_source_dir=kernel_dir,
        dry_run=False,
        max_attempts=2,
        format_error=lambda exc: str(exc),
        run_kernel_fix=run_kernel_fix,
    )

    assert calls == [1, 2]


def test_run_kernel_source_preflight_fixes_reports_unchanged_original_finding(tmp_path) -> None:
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "kernel.py").write_text("OUT = '/tmp/submission.csv'\n", encoding="utf-8")
    calls: list[int] = []

    def run_kernel_fix(error: str, attempt: int) -> KernelFixResult:  # noqa: ARG001
        calls.append(attempt)
        return KernelFixResult(
            agent_exit_code=0,
            repo_changed=False,
            regeneration_already_used=True,
        )

    with pytest.raises(KernelFailedError) as exc_info:
        run_kernel_source_preflight_fixes(
            kernel_source_dir=kernel_dir,
            dry_run=False,
            max_attempts=2,
            format_error=lambda exc: str(exc),
            run_kernel_fix=run_kernel_fix,
        )

    assert calls == [1]
    message = str(exc_info.value)
    assert "Kernel sources do not reference metrics.json output" in message
    assert "original_finding:" in message
    assert "remaining_finding:" in message
    assert message.count("check_name: kernel_source_contract") == 2
    assert "kernel_sha_before:" in message
    assert "kernel_sha_after:" in message
    assert "regeneration_already_used: True" in message
    assert isinstance(exc_info.value.__cause__, KernelFailedError)
    assert "Kernel sources do not reference metrics.json output" in str(exc_info.value.__cause__)


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
