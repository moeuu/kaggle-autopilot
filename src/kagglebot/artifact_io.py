from __future__ import annotations

import shutil
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
    shutil.copy2(source, destination)
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
