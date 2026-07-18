from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kagglebot.exceptions import KernelFailedError
from kagglebot.validators import kernel_source_preflight_error

_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(?P<name>(?:kaggle[_-]?(?:key|username)|api[_-]?key|password|secret|"
    r"(?:access|refresh|auth|bearer)?_?token))\b\s*[:=]\s*[^\s,;]+"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[^\s,;]+")


@dataclass(frozen=True)
class KernelFixResult:
    """Observable effects of one kernel-fix agent invocation."""

    agent_exit_code: int | None = None
    repo_changed: bool = False
    changed_paths: tuple[str, ...] = ()
    kernel_sha_before: str | None = None
    kernel_sha_after: str | None = None
    regeneration_attempted: bool = False
    regeneration_used: bool = False
    regeneration_already_used: bool = False


@dataclass(frozen=True)
class KernelPreflightFailure:
    """Structured evidence from one side-effect-free kernel source check."""

    check_name: str
    kernel_path: Path
    command_or_rule: str
    returncode: int | None
    stdout: str
    stderr: str
    source_excerpt: str
    kernel_sha256: str | None


def check_kernel_source_preflight(
    *,
    kernel_source_dir: Path,
    format_error: Callable[[BaseException], str],
    deliverable_mode: str = "leaderboard",
    required_output_names: tuple[str, ...] = (),
) -> KernelPreflightFailure | None:
    """Return structured preflight evidence without mutating kernel sources."""
    error_text = kernel_source_preflight_error(
        kernel_source_dir,
        require_kaggle_input=False,
        deliverable_mode=deliverable_mode,
        required_output_names=required_output_names,
        format_error=format_error,
    )
    if error_text is None:
        return None
    output_contract = ", ".join(required_output_names) or "(default submission artifact)"
    return KernelPreflightFailure(
        check_name="kernel_source_contract",
        kernel_path=kernel_source_dir / "kernel.py",
        command_or_rule=(
            "kagglebot.validators.ensure_kernel_sources_valid("
            f"require_kaggle_input=False, deliverable_mode={deliverable_mode!r}, "
            f"required_output_names={output_contract})"
        ),
        returncode=None,
        stdout="",
        stderr=error_text,
        source_excerpt="(validator did not report a source location)",
        kernel_sha256=kernel_source_sha256(kernel_source_dir),
    )


def format_kernel_preflight_failure(failure: KernelPreflightFailure) -> str:
    """Format all actionable preflight evidence for agents and error reports."""
    return "\n".join(
        [
            f"check_name: {failure.check_name}",
            f"kernel_path: {failure.kernel_path}",
            f"kernel_sha256: {failure.kernel_sha256 or '(missing)'}",
            f"command_or_rule: {failure.command_or_rule}",
            f"returncode: {failure.returncode if failure.returncode is not None else '(not applicable)'}",
            "stdout:",
            failure.stdout or "(empty)",
            "stderr:",
            failure.stderr or "(empty)",
            "source_excerpt:",
            failure.source_excerpt or "(not available)",
        ]
    )


def persist_kernel_preflight_failure(
    *,
    failure: KernelPreflightFailure,
    diagnostics_dir: Path,
    attempt: int,
) -> Path:
    """Persist a redacted diagnostic before an automatic source-fix attempt."""
    attempt_dir = diagnostics_dir / f"attempt-{attempt}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    path = attempt_dir / "kernel-preflight-error.txt"
    path.write_text(_redact_preflight_diagnostic(format_kernel_preflight_failure(failure)) + "\n", encoding="utf-8")
    return path


def run_kernel_source_preflight_fixes(
    *,
    kernel_source_dir: Path,
    dry_run: bool,
    max_attempts: int,
    format_error: Callable[[BaseException], str],
    run_kernel_fix: Callable[[str, int], KernelFixResult | None],
    deliverable_mode: str = "leaderboard",
    required_output_names: tuple[str, ...] = (),
    diagnostics_dir: Path | None = None,
    on_message: Callable[[str], None] = print,
    implementation_agent_alias: str = "implementation agent",
) -> None:
    """Fix deterministic kernel source issues before launching a kernel run."""
    attempt = 0
    while True:
        failure = check_kernel_source_preflight(
            kernel_source_dir=kernel_source_dir,
            deliverable_mode=deliverable_mode,
            required_output_names=required_output_names,
            format_error=format_error,
        )
        if failure is None:
            return
        diagnostic = format_kernel_preflight_failure(failure)
        lowered = failure.stderr.lower()
        if "requires kernel.py" in lowered:
            message = failure.stderr
            if message.startswith("RuntimeError:"):
                message = message.split(":", 1)[1].strip()
            raise RuntimeError(message)
        attempt += 1
        if dry_run:
            raise KernelFailedError(diagnostic)
        if attempt > max_attempts:
            raise KernelFailedError(f"Kernel source preflight failed after automatic fixes.\n{diagnostic}")
        if diagnostics_dir is not None:
            persist_kernel_preflight_failure(
                failure=failure,
                diagnostics_dir=diagnostics_dir,
                attempt=attempt,
            )
        on_message(
            "[yellow]kernel preflight[/yellow]: source contract check failed; "
            f"invoking {implementation_agent_alias} fix (attempt {attempt}/{max_attempts})"
        )
        before_sha = failure.kernel_sha256
        fix_result = run_kernel_fix(diagnostic, attempt)
        remaining_failure = check_kernel_source_preflight(
            kernel_source_dir=kernel_source_dir,
            deliverable_mode=deliverable_mode,
            required_output_names=required_output_names,
            format_error=format_error,
        )
        if remaining_failure is None:
            return
        remaining_diagnostic = format_kernel_preflight_failure(remaining_failure)
        if (
            isinstance(fix_result, KernelFixResult)
            and before_sha == remaining_failure.kernel_sha256
            and _preflight_fingerprint(failure.stderr) == _preflight_fingerprint(remaining_failure.stderr)
        ):
            original_error = KernelFailedError(diagnostic)
            raise KernelFailedError(
                _format_unchanged_preflight_error(
                    kernel_path=failure.kernel_path,
                    original_finding=diagnostic,
                    remaining_finding=remaining_diagnostic,
                    before_sha=before_sha,
                    after_sha=remaining_failure.kernel_sha256,
                    fix_result=fix_result,
                )
            ) from original_error


def kernel_source_sha256(kernel_source_dir: Path) -> str | None:
    """Return the generated kernel SHA-256 without mutating its directory."""
    kernel_path = kernel_source_dir / "kernel.py"
    if not kernel_path.is_file():
        return None
    digest = hashlib.sha256()
    with kernel_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _redact_preflight_diagnostic(text: str) -> str:
    redacted = text
    for name in (
        "KAGGLE_API_TOKEN",
        "KAGGLE_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GITHUB_TOKEN",
    ):
        secret = os.environ.get(name)
        if secret:
            redacted = redacted.replace(secret, "<redacted>")
    redacted = _SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group('name')}=<redacted>", redacted)
    return _BEARER_RE.sub("Bearer <redacted>", redacted)


def _preflight_fingerprint(finding: str) -> str:
    normalized = " ".join(finding.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _format_unchanged_preflight_error(
    *,
    kernel_path: Path,
    original_finding: str,
    remaining_finding: str,
    before_sha: str | None,
    after_sha: str | None,
    fix_result: KernelFixResult,
) -> str:
    changed_paths = ", ".join(fix_result.changed_paths) or "(none)"
    fingerprint = _preflight_fingerprint(original_finding)
    return "\n".join(
        [
            "Kernel source preflight was unchanged after an automatic fix; refusing a blind retry.",
            f"check_fingerprint: {fingerprint}",
            f"kernel_path: {kernel_path}",
            f"kernel_sha_before: {before_sha or '(missing)'}",
            f"kernel_sha_after: {after_sha or '(missing)'}",
            f"fix_agent_exit_code: {fix_result.agent_exit_code}",
            f"repo_changed: {fix_result.repo_changed}",
            f"changed_paths: {changed_paths}",
            f"regeneration_attempted: {fix_result.regeneration_attempted}",
            f"regeneration_used: {fix_result.regeneration_used}",
            f"regeneration_already_used: {fix_result.regeneration_already_used}",
            "original_finding:",
            original_finding,
            "remaining_finding:",
            remaining_finding,
        ]
    )
