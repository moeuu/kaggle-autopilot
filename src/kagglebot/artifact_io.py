from __future__ import annotations

import os
import shutil
import stat
import tempfile
from pathlib import Path

from kagglebot.submission_sample_discovery import (
    TABULAR_SUBMISSION_SUFFIXES,
    TABULAR_SUBMISSION_SUFFIXES_LENGTH_ORDERED,
    tabular_suffix,
)


def copy_artifact_if_needed(*, source: Path, destination: Path) -> Path:
    source_resolved = source.resolve()
    destination_resolved = destination.resolve()
    if source_resolved == destination_resolved:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        shutil.copytree(source, destination)
        return destination

    temporary: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
        os.close(fd)
        temporary = Path(temporary_name)
        shutil.copy2(source, temporary)

        copied_mode = stat.S_IMODE(temporary.stat().st_mode)
        temporary.chmod(copied_mode | stat.S_IWUSR)
        os.replace(temporary, destination)
    except PermissionError as exc:
        try:
            parent_stat = destination.parent.stat()
            parent_details = (
                f"mode={stat.filemode(parent_stat.st_mode)}, uid={parent_stat.st_uid}, gid={parent_stat.st_gid}"
            )
        except OSError as stat_exc:
            parent_details = f"unavailable ({stat_exc})"
        raise PermissionError(
            f"Could not copy artifact from {source} to {destination} "
            f"(resolved destination: {destination_resolved}; "
            f"destination parent {destination.parent}: {parent_details}): {exc}"
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def same_stem_tabular_artifact_filenames(filename: str) -> tuple[str, ...]:
    """Return filename plus same-stem tabular suffix alternates for artifact lookup."""

    path = Path(filename)
    suffix = tabular_suffix(path)
    if suffix not in TABULAR_SUBMISSION_SUFFIXES:
        return (filename,)
    stem = tabular_artifact_stem(path)
    candidates = [filename]
    candidates.extend(
        f"{stem}{candidate_suffix}"
        for candidate_suffix in TABULAR_SUBMISSION_SUFFIXES_LENGTH_ORDERED
        if candidate_suffix != suffix
    )
    return tuple(dict.fromkeys(candidates))


def tabular_artifact_stem(path: Path) -> str:
    suffix = tabular_suffix(path)
    name = path.name
    if suffix and name.lower().endswith(suffix):
        return name[: -len(suffix)]
    return path.stem
