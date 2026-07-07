from __future__ import annotations

from pathlib import Path

from kagglebot.artifact_io import copy_artifact_if_needed

SHAPEFILE_PRIMARY_SUFFIX = ".shp"
SHAPEFILE_COMPONENT_SUFFIXES = {".shp", ".shx", ".dbf"}
SHAPEFILE_REQUIRED_SUFFIXES = {".shp", ".dbf"}
SHAPEFILE_BUNDLE_SUFFIXES = (
    ".shp",
    ".shx",
    ".dbf",
    ".prj",
    ".cpg",
    ".qpj",
    ".sbn",
    ".sbx",
    ".qix",
    ".fbn",
    ".fbx",
    ".fix",
    ".ain",
    ".aih",
    ".ixs",
    ".mxs",
    ".atx",
    ".shp.aux.xml",
    ".shp.xml",
)
SHAPEFILE_SIDECAR_SUFFIXES = set(SHAPEFILE_BUNDLE_SUFFIXES) - {SHAPEFILE_PRIMARY_SUFFIX}


def shapefile_bundle_members(path: Path) -> list[Path]:
    stem = shapefile_stem(path)
    wanted_names = {f"{stem}{suffix}".lower() for suffix in SHAPEFILE_BUNDLE_SUFFIXES}
    try:
        members = [
            candidate
            for candidate in path.parent.iterdir()
            if candidate.is_file() and candidate.name.lower() in wanted_names
        ]
    except OSError:
        return []
    order = {suffix: index for index, suffix in enumerate(SHAPEFILE_BUNDLE_SUFFIXES)}
    return sorted(
        members,
        key=lambda member: (
            order.get(shapefile_component_suffix(member), len(order)),
            member.name.lower(),
        ),
    )


def shapefile_stem(path: Path) -> str:
    lowered = path.name.lower()
    for suffix in sorted(SHAPEFILE_BUNDLE_SUFFIXES, key=len, reverse=True):
        if lowered.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def shapefile_component_suffix(path: Path) -> str:
    lowered = path.name.lower()
    for suffix in sorted(SHAPEFILE_BUNDLE_SUFFIXES, key=len, reverse=True):
        if lowered.endswith(suffix):
            return suffix
    return path.suffix.lower()


def same_stem_shapefile_exists(path: Path) -> bool:
    stem = shapefile_stem(path)
    if not stem:
        return False
    exact = path.with_name(f"{stem}{SHAPEFILE_PRIMARY_SUFFIX}")
    if exact.exists():
        return True
    lowered = f"{stem}{SHAPEFILE_PRIMARY_SUFFIX}".lower()
    try:
        return any(candidate.is_file() and candidate.name.lower() == lowered for candidate in path.parent.iterdir())
    except OSError:
        return False


def copy_shapefile_sidecars_if_needed(*, source: Path, destination: Path) -> list[Path]:
    if shapefile_component_suffix(source) != SHAPEFILE_PRIMARY_SUFFIX:
        return []
    copied: list[Path] = []
    destination_stem = shapefile_stem(destination)
    try:
        source_resolved = source.resolve()
    except OSError:
        source_resolved = source
    for member in shapefile_bundle_members(source):
        try:
            if member.resolve() == source_resolved:
                continue
        except OSError:
            pass
        suffix = shapefile_component_suffix(member)
        copied.append(
            copy_artifact_if_needed(
                source=member,
                destination=destination.with_name(f"{destination_stem}{suffix}"),
            )
        )
    return copied
