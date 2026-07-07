from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from kagglebot.artifact_io import copy_artifact_if_needed
from kagglebot.asset_modality import asset_suffix
from kagglebot.compression_suffixes import compression_suffix_for, open_compressed_text, strip_compression_suffix
from kagglebot.geospatial_artifacts import is_envi_header_artifact

ANALYZE_PAIR_SUFFIXES = frozenset({".hdr", ".img"})
_ANALYZE_PAIR_COUNTERPART_SUFFIX = {
    ".hdr": ".img",
    ".img": ".hdr",
}
METAIMAGE_HEADER_SUFFIX = ".mhd"
NRRD_HEADER_SUFFIXES = frozenset({".nhdr", ".nrrd"})
_METAIMAGE_INLINE_DATA_TOKENS = {"local"}
_METAIMAGE_NON_FILE_DATA_TOKENS = {"list"}
_NRRD_INLINE_DATA_TOKENS = {"local"}
_NRRD_LIST_DATA_TOKEN = "list"


def is_analyze_pair_artifact(path: Path) -> bool:
    return strip_compression_suffix(asset_suffix(path)) in ANALYZE_PAIR_SUFFIXES and not is_envi_header_artifact(path)


def analyze_pair_sidecar_names(path: Path) -> list[str]:
    suffix = asset_suffix(path)
    base_suffix = strip_compression_suffix(suffix)
    counterpart_suffix = _ANALYZE_PAIR_COUNTERPART_SUFFIX.get(base_suffix)
    if counterpart_suffix is None:
        return []
    compression_suffix = compression_suffix_for(suffix) or ""
    stem = _artifact_name_without_suffix(path, suffix)
    return [f"{stem}{counterpart_suffix}{compression_suffix}"]


def analyze_pair_sidecar_specs(path: Path) -> list[tuple[Path, str]]:
    specs: list[tuple[Path, str]] = []
    for name in analyze_pair_sidecar_names(path):
        sidecar = path.parent / name
        if sidecar.is_file():
            specs.append((sidecar, name))
    return specs


def missing_analyze_pair_sidecars(path: Path) -> list[str]:
    missing: list[str] = []
    for name in analyze_pair_sidecar_names(path):
        if not (path.parent / name).is_file():
            missing.append(name)
    return missing


def copy_analyze_pair_sidecars_if_needed(*, source: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    for sidecar_path, sidecar_name in analyze_pair_sidecar_specs(source):
        copied.append(
            copy_artifact_if_needed(
                source=sidecar_path,
                destination=destination.parent / analyze_pair_destination_sidecar_name(destination, sidecar_name),
            )
        )
    return copied


def analyze_pair_destination_sidecar_name(primary_destination: Path, sidecar_name: str) -> str:
    sidecar_suffix = asset_suffix(Path(sidecar_name))
    destination_suffix = asset_suffix(primary_destination)
    destination_stem = _artifact_name_without_suffix(primary_destination, destination_suffix)
    return f"{destination_stem}{sidecar_suffix}"


def is_metaimage_header_artifact(path: Path) -> bool:
    return strip_compression_suffix(asset_suffix(path)) == METAIMAGE_HEADER_SUFFIX


def metaimage_sidecar_names(path: Path) -> list[str]:
    raw_name = _metaimage_element_data_file(path)
    if raw_name is None:
        return []
    if raw_name.strip().lower() in (_METAIMAGE_INLINE_DATA_TOKENS | _METAIMAGE_NON_FILE_DATA_TOKENS):
        return []
    name = _safe_metaimage_sidecar_name(raw_name)
    return [name] if name else []


def invalid_metaimage_sidecar_names(path: Path) -> list[str]:
    raw_name = _metaimage_element_data_file(path)
    if raw_name is None:
        return []
    if raw_name.strip().lower() in (_METAIMAGE_INLINE_DATA_TOKENS | _METAIMAGE_NON_FILE_DATA_TOKENS):
        return []
    return [] if _safe_metaimage_sidecar_name(raw_name) else [raw_name]


def metaimage_sidecar_specs(path: Path) -> list[tuple[Path, str]]:
    specs: list[tuple[Path, str]] = []
    for name in metaimage_sidecar_names(path):
        sidecar = path.parent / Path(name)
        if sidecar.is_file():
            specs.append((sidecar, name))
    return specs


def missing_metaimage_sidecars(path: Path) -> list[str]:
    missing: list[str] = []
    for name in metaimage_sidecar_names(path):
        if not (path.parent / Path(name)).is_file():
            missing.append(name)
    return missing


def copy_metaimage_sidecars_if_needed(*, source: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    for sidecar_path, sidecar_name in metaimage_sidecar_specs(source):
        copied.append(
            copy_artifact_if_needed(
                source=sidecar_path,
                destination=destination.parent / Path(sidecar_name),
            )
        )
    return copied


def is_detached_nrrd_header_artifact(path: Path) -> bool:
    return strip_compression_suffix(asset_suffix(path)) in NRRD_HEADER_SUFFIXES


def detached_nrrd_sidecar_names(path: Path) -> list[str]:
    names: list[str] = []
    for raw_name in _nrrd_data_file_values(path):
        name = _safe_medical_sidecar_name(raw_name)
        if name and name not in names:
            names.append(name)
    return names


def invalid_detached_nrrd_sidecar_names(path: Path) -> list[str]:
    invalid: list[str] = []
    for raw_name in _nrrd_data_file_values(path):
        if _safe_medical_sidecar_name(raw_name) is None and raw_name not in invalid:
            invalid.append(raw_name)
    return invalid


def detached_nrrd_sidecar_specs(path: Path) -> list[tuple[Path, str]]:
    specs: list[tuple[Path, str]] = []
    for name in detached_nrrd_sidecar_names(path):
        sidecar = path.parent / Path(name)
        if sidecar.is_file():
            specs.append((sidecar, name))
    return specs


def missing_detached_nrrd_sidecars(path: Path) -> list[str]:
    missing: list[str] = []
    for name in detached_nrrd_sidecar_names(path):
        if not (path.parent / Path(name)).is_file():
            missing.append(name)
    return missing


def copy_detached_nrrd_sidecars_if_needed(*, source: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    for sidecar_path, sidecar_name in detached_nrrd_sidecar_specs(source):
        copied.append(
            copy_artifact_if_needed(
                source=sidecar_path,
                destination=destination.parent / Path(sidecar_name),
            )
        )
    return copied


def _metaimage_element_data_file(path: Path) -> str | None:
    if not is_metaimage_header_artifact(path):
        return None
    try:
        with open_compressed_text(path) as handle:
            for index, raw_line in enumerate(handle):
                if index >= 256:
                    return None
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                if key.strip().lower() == "elementdatafile":
                    return value.strip()
    except OSError:
        return None
    return None


def _safe_metaimage_sidecar_name(value: str) -> str | None:
    return _safe_medical_sidecar_name(value)


def _artifact_name_without_suffix(path: Path, suffix: str) -> str:
    name = path.name
    if suffix and name.lower().endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def _nrrd_data_file_values(path: Path) -> list[str]:
    if not is_detached_nrrd_header_artifact(path):
        return []
    values: list[str] = []
    try:
        with open_compressed_text(path) as handle:
            for index, raw_line in enumerate(handle):
                if index >= 512:
                    return values
                line = raw_line.strip()
                if not line:
                    break
                if line.startswith("#") or line.startswith("NRRD") or ":" not in line:
                    continue
                key, _, value = line.partition(":")
                normalized_key = key.strip().lower().replace(" ", "")
                if normalized_key != "datafile":
                    continue
                raw_value = value.strip()
                lowered = raw_value.lower()
                if lowered in _NRRD_INLINE_DATA_TOKENS:
                    return []
                if lowered.startswith(_NRRD_LIST_DATA_TOKEN):
                    values.extend(_nrrd_data_file_list_values(handle, max_lines=max(0, 511 - index)))
                    return values
                if raw_value:
                    values.append(raw_value)
                    return values
    except OSError:
        return []
    return values


def _nrrd_data_file_list_values(lines: Iterable[str], *, max_lines: int) -> list[str]:
    values: list[str] = []
    for index, raw_line in enumerate(lines):
        if index >= max_lines:
            break
        line = raw_line.strip()
        if not line:
            break
        if line.startswith("#"):
            continue
        values.append(line)
    return values


def _safe_medical_sidecar_name(value: str) -> str | None:
    normalized = value.replace("\\", "/").strip()
    if not normalized:
        return None
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path.as_posix()
