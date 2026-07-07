from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from kagglebot.artifact_io import copy_artifact_if_needed
from kagglebot.asset_modality import asset_suffix
from kagglebot.compression_suffixes import open_compressed_text, strip_compression_suffix

KML_ARTIFACT_SUFFIX = ".kml"
VRT_ARTIFACT_SUFFIX = ".vrt"
ENVI_HEADER_SUFFIX = ".hdr"
ENVI_IMPLICIT_DATA_SUFFIXES = (".dat", ".bil", ".bsq", ".bip")
GEOREFERENCED_RASTER_PRIMARY_SUFFIXES = frozenset(
    {".bmp", ".gif", ".jp2", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
)
GEOREFERENCED_RASTER_GENERIC_SIDECAR_SUFFIXES = (".wld", ".prj")
GEOREFERENCED_RASTER_NAME_SIDECAR_SUFFIXES = (".aux.xml", ".ovr")
GEOREFERENCED_RASTER_WORLD_SUFFIXES_BY_PRIMARY = {
    ".bmp": (".bpw", ".bmpw"),
    ".gif": (".gfw", ".gifw"),
    ".jp2": (".j2w", ".jpw", ".jp2w"),
    ".jpg": (".jgw", ".jpgw"),
    ".jpeg": (".jgw", ".jpegw"),
    ".png": (".pgw", ".pngw"),
    ".tif": (".tfw", ".tifw"),
    ".tiff": (".tfw", ".tifw"),
    ".webp": (".webpw",),
}
MAPINFO_PRIMARY_SUFFIX = ".tab"
MAPINFO_REQUIRED_SUFFIXES = frozenset({".dat", ".id", ".map"})
MAPINFO_OPTIONAL_SUFFIXES = frozenset({".ind"})
MAPINFO_COMPONENT_SUFFIXES = frozenset(
    {
        MAPINFO_PRIMARY_SUFFIX,
        *MAPINFO_REQUIRED_SUFFIXES,
        *MAPINFO_OPTIONAL_SUFFIXES,
    }
)
MAPINFO_INTERCHANGE_PRIMARY_SUFFIX = ".mif"
MAPINFO_INTERCHANGE_SIDECAR_SUFFIX = ".mid"
_KML_BUNDLE_SUBDIR_NAMES = {"geo", "geodata", "kml", "layer", "layers", "map", "maps"}
_KML_HREF_RE = re.compile(
    r"<(?:[A-Za-z0-9_.-]+:)?href\b[^>]*>(.*?)</(?:[A-Za-z0-9_.-]+:)?href>", re.IGNORECASE | re.DOTALL
)


def is_kml_artifact(path: Path) -> bool:
    return strip_compression_suffix(asset_suffix(path)) == KML_ARTIFACT_SUFFIX


def is_vrt_artifact(path: Path) -> bool:
    return path.suffix.lower() == VRT_ARTIFACT_SUFFIX


def is_envi_header_artifact(path: Path) -> bool:
    if path.suffix.lower() != ENVI_HEADER_SUFFIX:
        return False
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for index, raw_line in enumerate(handle):
                if index >= 16:
                    return False
                line = raw_line.strip()
                if not line:
                    continue
                return line.lower() == "envi"
    except OSError:
        return False
    return False


def is_georeferenced_raster_artifact(path: Path) -> bool:
    return path.suffix.lower() in GEOREFERENCED_RASTER_PRIMARY_SUFFIXES


def georeferenced_raster_sidecar_specs(path: Path) -> list[tuple[Path, str]]:
    if not is_georeferenced_raster_artifact(path):
        return []
    specs: list[tuple[Path, str]] = []
    for name in _georeferenced_raster_candidate_sidecar_names(path):
        sidecar = path.parent / name
        if sidecar.is_file():
            specs.append((sidecar, name))
    return specs


def copy_georeferenced_raster_sidecars_if_needed(*, source: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    for sidecar_path, sidecar_name in georeferenced_raster_sidecar_specs(source):
        copied_name = georeferenced_raster_destination_sidecar_name(
            primary_source=source,
            primary_destination=destination,
            sidecar_name=sidecar_name,
        )
        copied.append(copy_artifact_if_needed(source=sidecar_path, destination=destination.parent / copied_name))
    return copied


def georeferenced_raster_destination_sidecar_name(
    *,
    primary_source: Path,
    primary_destination: Path,
    sidecar_name: str,
) -> str:
    if sidecar_name.startswith(f"{primary_source.name}."):
        return f"{primary_destination.name}{sidecar_name[len(primary_source.name) :]}"
    if sidecar_name.startswith(primary_source.stem):
        sidecar_tail = sidecar_name[len(primary_source.stem) :]
        if sidecar_tail.startswith("."):
            return f"{primary_destination.stem}{sidecar_tail}"
    return sidecar_name


def envi_sidecar_names(path: Path) -> list[str]:
    if not is_envi_header_artifact(path):
        return []
    raw_data_file = _envi_header_value(path, "data file")
    if raw_data_file:
        name = _safe_envi_sidecar_name(raw_data_file)
        return [name] if name else []
    names = _envi_implicit_sidecar_names(path)
    existing = [name for name in names if (path.parent / Path(name)).is_file()]
    return existing or names


def invalid_envi_sidecar_names(path: Path) -> list[str]:
    if not is_envi_header_artifact(path):
        return []
    raw_data_file = _envi_header_value(path, "data file")
    if not raw_data_file:
        return []
    return [] if _safe_envi_sidecar_name(raw_data_file) else [raw_data_file]


def envi_sidecar_specs(path: Path) -> list[tuple[Path, str]]:
    specs: list[tuple[Path, str]] = []
    for name in envi_sidecar_names(path):
        sidecar = path.parent / Path(name)
        if sidecar.is_file():
            specs.append((sidecar, name))
    return specs


def missing_envi_sidecars(path: Path) -> list[str]:
    missing: list[str] = []
    for name in envi_sidecar_names(path):
        if not (path.parent / Path(name)).is_file():
            missing.append(name)
    return missing


def copy_envi_sidecars_if_needed(*, source: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    for sidecar_path, sidecar_name in envi_sidecar_specs(source):
        copied.append(
            copy_artifact_if_needed(
                source=sidecar_path,
                destination=destination.parent / Path(envi_destination_sidecar_name(destination, sidecar_name)),
            )
        )
    return copied


def envi_destination_sidecar_name(primary_destination: Path, sidecar_name: str) -> str:
    sidecar_path = Path(sidecar_name)
    if sidecar_path.parent != Path("."):
        return sidecar_path.as_posix()
    suffix = sidecar_path.suffix.lower()
    return f"{primary_destination.stem}{suffix}"


def is_mapinfo_tab_artifact(path: Path) -> bool:
    return mapinfo_component_suffix(path) == MAPINFO_PRIMARY_SUFFIX


def mapinfo_component_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    return suffix if suffix in MAPINFO_COMPONENT_SUFFIXES else ""


def mapinfo_bundle_members(path: Path) -> list[Path]:
    if not is_mapinfo_tab_artifact(path):
        return []
    members = [path]
    for suffix in sorted(MAPINFO_REQUIRED_SUFFIXES | MAPINFO_OPTIONAL_SUFFIXES):
        member = path.with_suffix(suffix)
        if member.is_file():
            members.append(member)
    return members


def missing_mapinfo_sidecars(path: Path) -> list[str]:
    if not is_mapinfo_tab_artifact(path):
        return []
    missing: list[str] = []
    for suffix in sorted(MAPINFO_REQUIRED_SUFFIXES):
        sidecar = path.with_suffix(suffix)
        if not sidecar.is_file():
            missing.append(sidecar.name)
    return missing


def mapinfo_sidecar_specs(path: Path) -> list[tuple[Path, str]]:
    specs: list[tuple[Path, str]] = []
    if not is_mapinfo_tab_artifact(path):
        return specs
    for suffix in sorted(MAPINFO_REQUIRED_SUFFIXES | MAPINFO_OPTIONAL_SUFFIXES):
        sidecar = path.with_suffix(suffix)
        if sidecar.is_file():
            specs.append((sidecar, sidecar.name))
    return specs


def copy_mapinfo_sidecars_if_needed(*, source: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    for sidecar_path, sidecar_name in mapinfo_sidecar_specs(source):
        copied_name = mapinfo_destination_sidecar_name(destination, sidecar_name)
        copied.append(copy_artifact_if_needed(source=sidecar_path, destination=destination.parent / copied_name))
    return copied


def mapinfo_destination_sidecar_name(primary_destination: Path, sidecar_name: str) -> str:
    sidecar_suffix = mapinfo_component_suffix(Path(sidecar_name))
    if not sidecar_suffix:
        sidecar_suffix = Path(sidecar_name).suffix.lower()
    return f"{primary_destination.stem}{sidecar_suffix}"


def is_mapinfo_interchange_artifact(path: Path) -> bool:
    return path.suffix.lower() == MAPINFO_INTERCHANGE_PRIMARY_SUFFIX


def mapinfo_interchange_sidecar_name(path: Path) -> str | None:
    if not is_mapinfo_interchange_artifact(path):
        return None
    return f"{path.stem}{MAPINFO_INTERCHANGE_SIDECAR_SUFFIX}"


def mapinfo_interchange_sidecar_specs(path: Path) -> list[tuple[Path, str]]:
    sidecar_name = mapinfo_interchange_sidecar_name(path)
    if sidecar_name is None:
        return []
    sidecar = path.with_suffix(MAPINFO_INTERCHANGE_SIDECAR_SUFFIX)
    return [(sidecar, sidecar_name)] if sidecar.is_file() else []


def missing_mapinfo_interchange_sidecars(path: Path) -> list[str]:
    sidecar_name = mapinfo_interchange_sidecar_name(path)
    if sidecar_name is None:
        return []
    sidecar = path.with_suffix(MAPINFO_INTERCHANGE_SIDECAR_SUFFIX)
    return [] if sidecar.is_file() else [sidecar_name]


def copy_mapinfo_interchange_sidecars_if_needed(*, source: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    for sidecar_path, sidecar_name in mapinfo_interchange_sidecar_specs(source):
        copied_name = mapinfo_interchange_destination_sidecar_name(destination, sidecar_name)
        copied.append(copy_artifact_if_needed(source=sidecar_path, destination=destination.parent / copied_name))
    return copied


def mapinfo_interchange_destination_sidecar_name(primary_destination: Path, sidecar_name: str) -> str:
    sidecar_suffix = Path(sidecar_name).suffix.lower() or MAPINFO_INTERCHANGE_SIDECAR_SUFFIX
    return f"{primary_destination.stem}{sidecar_suffix}"


def vrt_artifact_root_dir(path: Path) -> Path:
    if _vrt_has_parent_relative_uri(path):
        return path.parent.parent
    return path.parent


def vrt_primary_archive_name(path: Path, *, root_dir: Path | None = None) -> str:
    root = root_dir or vrt_artifact_root_dir(path)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def vrt_sidecar_specs(path: Path, *, root_dir: Path | None = None) -> list[tuple[Path, str]]:
    return [
        (sidecar_path, archive_name)
        for sidecar_path, archive_name, _raw_source in vrt_sidecar_reference_specs(path, root_dir=root_dir)
    ]


def vrt_sidecar_reference_specs(path: Path, *, root_dir: Path | None = None) -> list[tuple[Path, str, str]]:
    specs: list[tuple[Path, str, str]] = []
    sidecar_root = root_dir or vrt_artifact_root_dir(path)
    for raw_source in vrt_raw_sidecar_names(path):
        name = _safe_vrt_sidecar_name(raw_source, source_dir=path.parent, root_dir=sidecar_root)
        if not name:
            continue
        sidecar = sidecar_root / Path(name)
        if sidecar.is_file():
            specs.append((sidecar, name, raw_source))
    return specs


def vrt_sidecar_names(path: Path, *, root_dir: Path | None = None) -> list[str]:
    names: list[str] = []
    sidecar_root = root_dir or vrt_artifact_root_dir(path)
    for raw_source in vrt_raw_sidecar_names(path):
        name = _safe_vrt_sidecar_name(raw_source, source_dir=path.parent, root_dir=sidecar_root)
        if name and name not in names:
            names.append(name)
    return names


def invalid_vrt_sidecar_names(path: Path, *, root_dir: Path | None = None) -> list[str]:
    invalid: list[str] = []
    sidecar_root = root_dir or vrt_artifact_root_dir(path)
    for raw_source in vrt_raw_sidecar_names(path):
        if (
            _safe_vrt_sidecar_name(raw_source, source_dir=path.parent, root_dir=sidecar_root) is None
            and raw_source not in invalid
        ):
            invalid.append(raw_source)
    return invalid


def missing_vrt_sidecars(path: Path, *, root_dir: Path | None = None) -> list[str]:
    missing: list[str] = []
    sidecar_root = root_dir or vrt_artifact_root_dir(path)
    for name in vrt_sidecar_names(path, root_dir=root_dir):
        if not (sidecar_root / Path(name)).is_file():
            missing.append(name)
    return missing


def copy_vrt_sidecars_if_needed(*, source: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    source_root = vrt_artifact_root_dir(source)
    destination_root = _kml_destination_root(source=source, destination=destination, root_dir=source_root)
    for sidecar_path, archive_name in vrt_sidecar_specs(source, root_dir=source_root):
        copied.append(
            copy_artifact_if_needed(
                source=sidecar_path,
                destination=destination_root / Path(archive_name),
            )
        )
    return copied


def vrt_raw_sidecar_names(path: Path) -> list[str]:
    if not is_vrt_artifact(path):
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    names: list[str] = []
    for raw_source in _vrt_source_values(text):
        source = raw_source.strip()
        if not source:
            continue
        if source not in names:
            names.append(source)
    return names


def _georeferenced_raster_candidate_sidecar_names(path: Path) -> list[str]:
    names: list[str] = []
    suffix = path.suffix.lower()
    for sidecar_suffix in GEOREFERENCED_RASTER_WORLD_SUFFIXES_BY_PRIMARY.get(suffix, ()):
        names.append(f"{path.stem}{sidecar_suffix}")
    for sidecar_suffix in GEOREFERENCED_RASTER_GENERIC_SIDECAR_SUFFIXES:
        names.append(f"{path.stem}{sidecar_suffix}")
    for sidecar_suffix in GEOREFERENCED_RASTER_NAME_SIDECAR_SUFFIXES:
        names.append(f"{path.name}{sidecar_suffix}")
        names.append(f"{path.stem}{sidecar_suffix}")
    return list(dict.fromkeys(names))


def _envi_implicit_sidecar_names(path: Path) -> list[str]:
    if path.name.lower().endswith(".dat.hdr"):
        return [path.name[: -len(ENVI_HEADER_SUFFIX)]]
    return [f"{path.stem}{suffix}" for suffix in ENVI_IMPLICIT_DATA_SUFFIXES]


def _envi_header_value(path: Path, key_name: str) -> str | None:
    normalized_target = key_name.strip().lower()
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for index, raw_line in enumerate(handle):
                if index >= 512:
                    return None
                line = raw_line.strip()
                if not line or line.startswith(";") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                if key.strip().lower() == normalized_target:
                    return value.strip().strip("{}").strip()
    except OSError:
        return None
    return None


def _safe_envi_sidecar_name(value: str) -> str | None:
    normalized = value.replace("\\", "/").strip()
    if not normalized:
        return None
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path.as_posix()


def kml_artifact_root_dir(path: Path) -> Path:
    if _kml_has_parent_relative_uri(path) and path.parent.name.lower() in _KML_BUNDLE_SUBDIR_NAMES:
        return path.parent.parent
    return path.parent


def kml_primary_archive_name(path: Path, *, root_dir: Path | None = None) -> str:
    root = root_dir or kml_artifact_root_dir(path)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def kml_sidecar_specs(path: Path, *, root_dir: Path | None = None) -> list[tuple[Path, str]]:
    return [
        (sidecar_path, archive_name)
        for sidecar_path, archive_name, _raw_href in kml_sidecar_reference_specs(path, root_dir=root_dir)
    ]


def kml_sidecar_reference_specs(path: Path, *, root_dir: Path | None = None) -> list[tuple[Path, str, str]]:
    specs: list[tuple[Path, str, str]] = []
    sidecar_root = root_dir or kml_artifact_root_dir(path)
    for raw_href in kml_raw_sidecar_names(path):
        name = _safe_kml_sidecar_name(raw_href, source_dir=path.parent, root_dir=sidecar_root)
        if not name:
            continue
        sidecar = sidecar_root / Path(name)
        if sidecar.is_file():
            specs.append((sidecar, name, raw_href))
    return specs


def kml_sidecar_names(path: Path, *, root_dir: Path | None = None) -> list[str]:
    names: list[str] = []
    sidecar_root = root_dir or kml_artifact_root_dir(path)
    for raw_href in kml_raw_sidecar_names(path):
        name = _safe_kml_sidecar_name(raw_href, source_dir=path.parent, root_dir=sidecar_root)
        if name and name not in names:
            names.append(name)
    return names


def invalid_kml_sidecar_names(path: Path, *, root_dir: Path | None = None) -> list[str]:
    invalid: list[str] = []
    sidecar_root = root_dir or kml_artifact_root_dir(path)
    for raw_href in kml_raw_sidecar_names(path):
        if (
            _safe_kml_sidecar_name(raw_href, source_dir=path.parent, root_dir=sidecar_root) is None
            and raw_href not in invalid
        ):
            invalid.append(raw_href)
    return invalid


def missing_kml_sidecars(path: Path, *, root_dir: Path | None = None) -> list[str]:
    missing: list[str] = []
    sidecar_root = root_dir or kml_artifact_root_dir(path)
    for name in kml_sidecar_names(path, root_dir=root_dir):
        if not (sidecar_root / Path(name)).is_file():
            missing.append(name)
    return missing


def copy_kml_sidecars_if_needed(*, source: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    source_root = kml_artifact_root_dir(source)
    destination_root = _kml_destination_root(source=source, destination=destination, root_dir=source_root)
    for sidecar_path, archive_name in kml_sidecar_specs(source, root_dir=source_root):
        copied.append(
            copy_artifact_if_needed(
                source=sidecar_path,
                destination=destination_root / Path(archive_name),
            )
        )
    return copied


def kml_raw_sidecar_names(path: Path) -> list[str]:
    if not is_kml_artifact(path):
        return []
    try:
        with open_compressed_text(path) as handle:
            text = handle.read()
    except (OSError, UnicodeDecodeError):
        return []
    names: list[str] = []
    for raw_href in _kml_href_values(text):
        href = raw_href.strip()
        if not href or _kml_href_is_inline(href):
            continue
        if href not in names:
            names.append(href)
    return names


def _vrt_source_values(text: str) -> list[str]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return [
            _unescape_xml_text(match.group(1)).strip()
            for match in re.finditer(
                r"<(?:[A-Za-z0-9_.-]+:)?(?:SourceFilename|SourceDataset)\b[^>]*>"
                r"(.*?)</(?:[A-Za-z0-9_.-]+:)?(?:SourceFilename|SourceDataset)>",
                text,
                re.IGNORECASE | re.DOTALL,
            )
        ]
    values: list[str] = []
    for element in root.iter():
        if _xml_local_name(element.tag) not in {"sourcefilename", "sourcedataset"}:
            continue
        if element.text and element.text.strip():
            values.append(element.text.strip())
    return values


def _vrt_has_parent_relative_uri(path: Path) -> bool:
    for raw_source in vrt_raw_sidecar_names(path):
        parsed = urlsplit(raw_source.replace("\\", "/").strip())
        if parsed.scheme or parsed.netloc:
            continue
        decoded = unquote(parsed.path)
        if ".." in PurePosixPath(decoded).parts:
            return True
    return False


def _safe_vrt_sidecar_name(value: str, *, source_dir: Path, root_dir: Path | None = None) -> str | None:
    normalized = value.replace("\\", "/").strip()
    if not normalized:
        return None
    parsed = urlsplit(normalized)
    if parsed.scheme or parsed.netloc:
        return None
    if normalized.startswith("/vsi"):
        return None
    return _safe_relative_sidecar_path_for_base(
        unquote(parsed.path),
        source_dir=source_dir,
        root_dir=root_dir or source_dir,
    )


def _kml_href_values(text: str) -> list[str]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return [_unescape_xml_text(match.group(1)).strip() for match in _KML_HREF_RE.finditer(text)]
    values: list[str] = []
    for element in root.iter():
        if _xml_local_name(element.tag) != "href":
            continue
        if element.text and element.text.strip():
            values.append(element.text.strip())
    return values


def _kml_has_parent_relative_uri(path: Path) -> bool:
    for raw_href in kml_raw_sidecar_names(path):
        parsed = urlsplit(raw_href.replace("\\", "/").strip())
        if parsed.scheme or parsed.netloc:
            continue
        decoded = unquote(parsed.path)
        if ".." in PurePosixPath(decoded).parts:
            return True
    return False


def _kml_destination_root(*, source: Path, destination: Path, root_dir: Path) -> Path:
    try:
        relative_source = source.relative_to(root_dir)
    except ValueError:
        return destination.parent
    root = destination
    for _part in relative_source.parts:
        root = root.parent
    return root


def _safe_kml_sidecar_name(value: str, *, source_dir: Path, root_dir: Path | None = None) -> str | None:
    normalized = value.replace("\\", "/").strip()
    if not normalized:
        return None
    parsed = urlsplit(normalized)
    if parsed.scheme or parsed.netloc:
        return None
    return _safe_relative_sidecar_path_for_base(
        unquote(parsed.path),
        source_dir=source_dir,
        root_dir=root_dir or source_dir,
    )


def _safe_relative_sidecar_path_for_base(value: str, *, source_dir: Path, root_dir: Path) -> str | None:
    normalized = value.replace("\\", "/").strip()
    if not normalized:
        return None
    path = PurePosixPath(normalized)
    if path.is_absolute():
        return None
    try:
        source_root = root_dir.resolve(strict=False)
        candidate = (source_dir / Path(path.as_posix())).resolve(strict=False)
        candidate.relative_to(source_root)
    except (OSError, ValueError):
        return None
    try:
        return candidate.relative_to(source_root).as_posix()
    except ValueError:
        return None


def _kml_href_is_inline(value: str) -> bool:
    normalized = value.strip()
    parsed = urlsplit(normalized)
    return normalized.lower().startswith("data:") or (not parsed.path and bool(parsed.fragment))


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _unescape_xml_text(value: str) -> str:
    return (
        value.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
        .replace("&amp;", "&")
    )
