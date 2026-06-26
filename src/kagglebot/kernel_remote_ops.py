from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich import print

from kagglebot.exceptions import KaggleCliError

KERNEL_REGISTER_RETRIES = 24
KERNEL_REGISTER_SLEEP = 5.0


@dataclass(frozen=True)
class KernelRegistrationDependencies:
    kernels_status: Callable[..., str]
    kernel_exists: Callable[[str], bool]
    kernel_id_by_title: Callable[[str], str | None]
    sleep: Callable[[float], object]


def wait_for_kernel_registration(
    kernel_id: str,
    kernel_slug: str,
    *,
    deps: KernelRegistrationDependencies,
    retries: int = KERNEL_REGISTER_RETRIES,
    sleep_interval: float = KERNEL_REGISTER_SLEEP,
) -> str | None:
    for attempt in range(1, retries + 1):
        try:
            deps.kernels_status(kernel_id, dry_run=False)
            return kernel_id
        except KaggleCliError as exc:
            detail = (exc.output or str(exc)).strip().replace("\n", " ")
            if detail:
                print(f"[yellow]kernel status unavailable[/yellow]: {detail} (attempt {attempt})")
        try:
            if deps.kernel_exists(kernel_id):
                return kernel_id
            resolved = deps.kernel_id_by_title(kernel_slug)
            if resolved:
                return resolved
        except KaggleCliError as exc:
            detail = (exc.output or str(exc)).strip().replace("\n", " ")
            if detail:
                print(f"[yellow]kernel list failed[/yellow]: {detail} (attempt {attempt})")
        deps.sleep(sleep_interval)
    return None


def resolve_kernel_id(
    kernel_id: str,
    kernel_slug: str,
    *,
    kernel_id_by_title_func: Callable[[str], str | None],
) -> str:
    try:
        resolved = kernel_id_by_title_func(kernel_slug)
    except KaggleCliError:
        return kernel_id
    if resolved and resolved != kernel_id:
        print(f"[cyan]kernel id[/cyan]: {resolved}")
        return resolved
    return kernel_id


def write_push_log(logs_dir: Path, attempt: int, output: str) -> None:
    path = logs_dir / f"kernel_push-{attempt:02d}.txt"
    path.write_text(output.strip() + "\n", encoding="utf-8")


def clear_stale_kernel_output(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for path in output_dir.iterdir():
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
                continue
            path.unlink()
        except OSError:
            continue


def try_fetch_kernel_output(
    kernel_id: str,
    *,
    output_dir: Path,
    slug: str,
    kernels_output_func: Callable[..., object],
) -> None:
    try:
        kernels_output_func(kernel_id, output_dir, slug=slug, dry_run=False, force=True, quiet=True)
    except KaggleCliError:
        return
