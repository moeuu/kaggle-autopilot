from __future__ import annotations

from pathlib import Path

from kagglebot.kaggle_api import (
    check_rules_accepted,
    competitions_files,
    kernels_status,
)
from kagglebot.kaggle_api import kernels_push as _kernels_push
from kagglebot.kaggle_api import (
    download_competition as _download_competition,
)
from kagglebot.kaggle_api import (
    kernels_output as _kernels_output,
)
from kagglebot.kaggle_api import (
    submit_competition as _submit_competition,
)

__all__ = [
    "download_competition",
    "submit_competition",
    "kernels_push",
    "kernels_status",
    "kernels_output",
    "competitions_files",
    "check_rules_accepted",
]


def download_competition(
    slug: str,
    dest_dir: Path,
    *,
    overwrite: bool = False,
    stream_output: bool = False,
    dry_run: bool = False,
) -> str:
    return _download_competition(
        slug,
        dest_dir,
        force=overwrite,
        quiet=not stream_output,
        dry_run=dry_run,
    )


def submit_competition(
    slug: str,
    submission_file: Path,
    message: str,
    *,
    stream_output: bool = False,
    dry_run: bool = False,
) -> str:
    _ = stream_output
    return _submit_competition(slug, submission_file, message, dry_run=dry_run)


def kernels_push(
    kernel_dir: Path,
    *,
    slug: str | None = None,
    stream_output: bool = False,
    dry_run: bool = False,
) -> str:
    _ = stream_output
    return _kernels_push(kernel_dir, slug=slug, dry_run=dry_run)


def kernels_output(
    kernel_id: str,
    output_dir: Path,
    *,
    slug: str | None = None,
    stream_output: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> str:
    return _kernels_output(
        kernel_id,
        output_dir,
        slug=slug,
        dry_run=dry_run,
        force=force,
        quiet=not stream_output,
    )
