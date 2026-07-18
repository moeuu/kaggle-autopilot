from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from kagglebot.exceptions import KernelFailedError
from kagglebot.validators import kernel_source_preflight_error


def run_kernel_source_preflight_fixes(
    *,
    kernel_source_dir: Path,
    dry_run: bool,
    max_attempts: int,
    format_error: Callable[[BaseException], str],
    run_kernel_fix: Callable[[str, int], None],
    deliverable_mode: str = "leaderboard",
    required_output_names: tuple[str, ...] = (),
    on_message: Callable[[str], None] = print,
    implementation_agent_alias: str = "implementation agent",
) -> None:
    """Fix deterministic kernel source issues before launching a kernel run."""
    attempt = 0
    while True:
        preflight_error = kernel_source_preflight_error(
            kernel_source_dir,
            require_kaggle_input=False,
            deliverable_mode=deliverable_mode,
            required_output_names=required_output_names,
            format_error=format_error,
        )
        if preflight_error is None:
            return
        lowered = preflight_error.lower()
        if "requires kernel.py" in lowered:
            message = preflight_error
            if message.startswith("RuntimeError:"):
                message = message.split(":", 1)[1].strip()
            raise RuntimeError(message)
        attempt += 1
        if dry_run:
            raise KernelFailedError(preflight_error)
        if attempt > max_attempts:
            raise KernelFailedError(f"Kernel source preflight failed after automatic fixes.\n{preflight_error}")
        on_message(
            "[yellow]kernel preflight[/yellow]: source contract check failed; "
            f"invoking {implementation_agent_alias} fix (attempt {attempt}/{max_attempts})"
        )
        run_kernel_fix(preflight_error, attempt)
