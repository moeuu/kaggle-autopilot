from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from kagglebot.json_utils import load_json_object_or_empty


class CompetitionDataPaths(Protocol):
    data_dir: Path
    dataset_profile_path: Path


@dataclass(frozen=True)
class LocalTrainingDataReadiness:
    ready: bool
    reason: str
    training_sources: tuple[str, ...] = ()


def assess_local_training_data(paths: CompetitionDataPaths) -> LocalTrainingDataReadiness:
    """Require structured profile readiness and an on-disk labeled-training source."""
    profile = load_json_object_or_empty(paths.dataset_profile_path)
    if not profile:
        return LocalTrainingDataReadiness(False, "dataset_profile_missing")

    sources = _find_training_sources(paths.data_dir, profile)
    if profile.get("status") == "missing_required_files" and not sources:
        return LocalTrainingDataReadiness(False, "dataset_profile_missing_required_files")
    if not sources:
        return LocalTrainingDataReadiness(False, "labeled_training_source_missing")
    return LocalTrainingDataReadiness(
        True,
        "ready",
        tuple(str(path) for path in sources),
    )


def _find_training_sources(data_dir: Path, profile: dict[str, object]) -> list[Path]:
    if not data_dir.is_dir():
        return []

    candidates: list[Path] = []
    for key in ("train_path", "train_file"):
        value = profile.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value)
        candidate = path if path.is_absolute() else data_dir / path
        candidates.append(candidate)
        if not path.is_absolute() and not candidate.exists():
            candidates.extend(match for match in data_dir.rglob(path.name) if match.is_file())

    candidates.extend(
        path
        for path in (
            data_dir / "train",
            data_dir / "training",
            data_dir / "Training",
            data_dir / "HAR" / "data",
            data_dir / "data" / "HAR" / "data",
        )
        if path.exists()
    )
    for pattern in ("train.*", "training.*", "train_labels.*", "labels.*"):
        candidates.extend(path for path in data_dir.rglob(pattern) if path.is_file())

    candidates.extend(_complete_training_archives(data_dir))
    return _dedupe_existing_nonempty_sources(candidates)


def _complete_training_archives(data_dir: Path) -> list[Path]:
    archives: list[Path] = []
    for archive in data_dir.iterdir():
        if not archive.is_file() or archive.suffix.lower() != ".zip":
            continue
        stem = archive.stem.lower()
        if not any(marker in stem for marker in ("train", "training", "har")):
            continue
        split_parts = sorted(data_dir.glob(f"{archive.stem}.z[0-9][0-9]"))
        if split_parts:
            indices = [int(part.suffix[2:]) for part in split_parts]
            expected_indices = list(range(1, 9)) if stem == "har" else list(range(1, max(indices) + 1))
            if indices != expected_indices:
                continue
        archives.append(archive)
    return archives


def _dedupe_existing_nonempty_sources(candidates: list[Path]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            normalized = candidate.resolve()
        except OSError:
            continue
        if normalized in seen or not _is_nonempty_source(normalized):
            continue
        seen.add(normalized)
        resolved.append(normalized)
    return resolved


def _is_nonempty_source(path: Path) -> bool:
    if path.is_file():
        try:
            return path.stat().st_size > 0
        except OSError:
            return False
    if not path.is_dir():
        return False
    try:
        return any(child.is_file() for child in path.rglob("*"))
    except OSError:
        return False
