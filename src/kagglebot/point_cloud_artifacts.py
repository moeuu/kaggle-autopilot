from __future__ import annotations

import json
import re
import shlex
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from kagglebot.artifact_io import copy_artifact_if_needed
from kagglebot.asset_modality import asset_suffix
from kagglebot.compression_suffixes import open_compressed_text, strip_compression_suffix

DAE_ARTIFACT_SUFFIX = ".dae"
GLTF_ARTIFACT_SUFFIX = ".gltf"
LAS_ARTIFACT_SUFFIXES = (".las", ".laz")
LAS_SIDECAR_SUFFIXES = (".prj", ".wkt", ".lax", ".lasx", ".aux.xml")
OBJ_ARTIFACT_SUFFIX = ".obj"
PLY_ARTIFACT_SUFFIX = ".ply"
X3D_ARTIFACT_SUFFIX = ".x3d"
USD_ARTIFACT_SUFFIXES = (".usd", ".usda", ".usdc", ".usdz")
USD_TEXT_ARTIFACT_SUFFIXES = (".usd", ".usda")
_DAE_BUNDLE_SUBDIR_NAMES = {"collada", "dae", "mesh", "meshes", "model", "models", "scene", "scenes"}
_GLTF_BUNDLE_SUBDIR_NAMES = {"gltf", "mesh", "meshes", "model", "models", "scene", "scenes"}
_USD_BUNDLE_SUBDIR_NAMES = {"mesh", "meshes", "model", "models", "scene", "scenes", "usd"}
_X3D_BUNDLE_SUBDIR_NAMES = {"mesh", "meshes", "model", "models", "scene", "scenes", "x3d"}
_DAE_INIT_FROM_RE = re.compile(
    r"<(?:[A-Za-z0-9_.-]+:)?init_from\b[^>]*>(.*?)</(?:[A-Za-z0-9_.-]+:)?init_from>",
    re.IGNORECASE | re.DOTALL,
)
_X3D_URL_ATTR_RE = re.compile(r"\b(?:[A-Za-z0-9_.-]+:)?[A-Za-z0-9_.-]*url\s*=\s*(\"[^\"]*\"|'[^']*')", re.IGNORECASE)
_USD_ASSET_REF_RE = re.compile(r"@([^@\n\r]+)@")
_MTL_TEXTURE_COMMANDS = {
    "bump",
    "decal",
    "disp",
    "map_bump",
    "map_d",
    "map_ka",
    "map_kd",
    "map_ke",
    "map_ks",
    "map_ns",
    "map_pm",
    "map_pr",
    "map_ps",
    "map_refl",
}


def is_dae_artifact(path: Path) -> bool:
    return strip_compression_suffix(asset_suffix(path)) == DAE_ARTIFACT_SUFFIX


def is_obj_artifact(path: Path) -> bool:
    return strip_compression_suffix(asset_suffix(path)) == OBJ_ARTIFACT_SUFFIX


def is_ply_artifact(path: Path) -> bool:
    return strip_compression_suffix(asset_suffix(path)) == PLY_ARTIFACT_SUFFIX


def is_x3d_artifact(path: Path) -> bool:
    return strip_compression_suffix(asset_suffix(path)) == X3D_ARTIFACT_SUFFIX


def is_gltf_artifact(path: Path) -> bool:
    return strip_compression_suffix(asset_suffix(path)) == GLTF_ARTIFACT_SUFFIX


def is_las_artifact(path: Path) -> bool:
    return strip_compression_suffix(asset_suffix(path)) in LAS_ARTIFACT_SUFFIXES


def is_usd_artifact(path: Path) -> bool:
    return strip_compression_suffix(asset_suffix(path)) in USD_ARTIFACT_SUFFIXES


def is_usd_text_artifact(path: Path) -> bool:
    return strip_compression_suffix(asset_suffix(path)) in USD_TEXT_ARTIFACT_SUFFIXES


def dae_artifact_root_dir(path: Path) -> Path:
    if _dae_has_parent_relative_uri(path) and path.parent.name.lower() in _DAE_BUNDLE_SUBDIR_NAMES:
        return path.parent.parent
    return path.parent


def gltf_artifact_root_dir(path: Path) -> Path:
    if _gltf_has_parent_relative_uri(path) and path.parent.name.lower() in _GLTF_BUNDLE_SUBDIR_NAMES:
        return path.parent.parent
    return path.parent


def x3d_artifact_root_dir(path: Path) -> Path:
    if _x3d_has_parent_relative_uri(path) and path.parent.name.lower() in _X3D_BUNDLE_SUBDIR_NAMES:
        return path.parent.parent
    return path.parent


def usd_artifact_root_dir(path: Path) -> Path:
    if _usd_has_parent_relative_uri(path) and path.parent.name.lower() in _USD_BUNDLE_SUBDIR_NAMES:
        return path.parent.parent
    return path.parent


def dae_primary_archive_name(path: Path, *, root_dir: Path | None = None) -> str:
    root = root_dir or dae_artifact_root_dir(path)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def x3d_primary_archive_name(path: Path, *, root_dir: Path | None = None) -> str:
    root = root_dir or x3d_artifact_root_dir(path)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def gltf_primary_archive_name(path: Path, *, root_dir: Path | None = None) -> str:
    root = root_dir or gltf_artifact_root_dir(path)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def usd_primary_archive_name(path: Path, *, root_dir: Path | None = None) -> str:
    root = root_dir or usd_artifact_root_dir(path)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def dae_sidecar_specs(path: Path, *, root_dir: Path | None = None) -> list[tuple[Path, str]]:
    return [
        (sidecar_path, archive_name)
        for sidecar_path, archive_name, _raw_uri in dae_sidecar_reference_specs(path, root_dir=root_dir)
    ]


def dae_sidecar_reference_specs(path: Path, *, root_dir: Path | None = None) -> list[tuple[Path, str, str]]:
    specs: list[tuple[Path, str, str]] = []
    sidecar_root = root_dir or dae_artifact_root_dir(path)
    for raw_uri in dae_raw_sidecar_names(path):
        name = _safe_dae_sidecar_name(raw_uri, source_dir=path.parent, root_dir=sidecar_root)
        if not name:
            continue
        sidecar = sidecar_root / Path(name)
        if sidecar.is_file():
            specs.append((sidecar, name, raw_uri))
    return specs


def gltf_sidecar_specs(path: Path, *, root_dir: Path | None = None) -> list[tuple[Path, str]]:
    return [
        (sidecar_path, archive_name)
        for sidecar_path, archive_name, _raw_uri in gltf_sidecar_reference_specs(path, root_dir=root_dir)
    ]


def usd_sidecar_specs(path: Path, *, root_dir: Path | None = None) -> list[tuple[Path, str]]:
    return [
        (sidecar_path, archive_name)
        for sidecar_path, archive_name, _raw_uri in usd_sidecar_reference_specs(path, root_dir=root_dir)
    ]


def x3d_sidecar_specs(path: Path, *, root_dir: Path | None = None) -> list[tuple[Path, str]]:
    return [
        (sidecar_path, archive_name)
        for sidecar_path, archive_name, _raw_uri in x3d_sidecar_reference_specs(path, root_dir=root_dir)
    ]


def ply_sidecar_specs(path: Path) -> list[tuple[Path, str]]:
    specs: list[tuple[Path, str]] = []
    for name in ply_sidecar_names(path):
        sidecar = path.parent / Path(name)
        if sidecar.is_file():
            specs.append((sidecar, name))
    return specs


def x3d_sidecar_reference_specs(path: Path, *, root_dir: Path | None = None) -> list[tuple[Path, str, str]]:
    specs: list[tuple[Path, str, str]] = []
    sidecar_root = root_dir or x3d_artifact_root_dir(path)
    for raw_uri in x3d_raw_sidecar_names(path):
        name = _safe_x3d_sidecar_name(raw_uri, source_dir=path.parent, root_dir=sidecar_root)
        if not name:
            continue
        sidecar = sidecar_root / Path(name)
        if sidecar.is_file():
            specs.append((sidecar, name, raw_uri))
    return specs


def gltf_sidecar_reference_specs(path: Path, *, root_dir: Path | None = None) -> list[tuple[Path, str, str]]:
    specs: list[tuple[Path, str, str]] = []
    sidecar_root = root_dir or gltf_artifact_root_dir(path)
    for raw_uri in gltf_raw_sidecar_names(path):
        name = _safe_gltf_sidecar_name(raw_uri, source_dir=path.parent, root_dir=sidecar_root)
        if not name:
            continue
        sidecar = sidecar_root / Path(name)
        if sidecar.is_file():
            specs.append((sidecar, name, raw_uri))
    return specs


def usd_sidecar_reference_specs(path: Path, *, root_dir: Path | None = None) -> list[tuple[Path, str, str]]:
    specs: list[tuple[Path, str, str]] = []
    sidecar_root = root_dir or usd_artifact_root_dir(path)
    for raw_uri in usd_raw_sidecar_names(path):
        name = _safe_usd_sidecar_name(raw_uri, source_dir=path.parent, root_dir=sidecar_root)
        if not name:
            continue
        sidecar = sidecar_root / Path(name)
        if sidecar.is_file():
            specs.append((sidecar, name, raw_uri))
    return specs


def las_sidecar_specs(path: Path) -> list[tuple[Path, str]]:
    if not is_las_artifact(path):
        return []
    specs: list[tuple[Path, str]] = []
    for name in las_sidecar_names(path):
        sidecar = path.parent / name
        if sidecar.is_file():
            specs.append((sidecar, name))
    return specs


def dae_sidecar_names(path: Path, *, root_dir: Path | None = None) -> list[str]:
    names: list[str] = []
    sidecar_root = root_dir or dae_artifact_root_dir(path)
    for raw_name in dae_raw_sidecar_names(path):
        name = _safe_dae_sidecar_name(raw_name, source_dir=path.parent, root_dir=sidecar_root)
        if name and name not in names:
            names.append(name)
    return names


def x3d_sidecar_names(path: Path, *, root_dir: Path | None = None) -> list[str]:
    names: list[str] = []
    sidecar_root = root_dir or x3d_artifact_root_dir(path)
    for raw_name in x3d_raw_sidecar_names(path):
        name = _safe_x3d_sidecar_name(raw_name, source_dir=path.parent, root_dir=sidecar_root)
        if name and name not in names:
            names.append(name)
    return names


def ply_sidecar_names(path: Path) -> list[str]:
    names: list[str] = []
    for raw_name in ply_raw_sidecar_names(path):
        name = _safe_ply_sidecar_name(raw_name, source_dir=path.parent, root_dir=path.parent)
        if name and name not in names:
            names.append(name)
    return names


def gltf_sidecar_names(path: Path, *, root_dir: Path | None = None) -> list[str]:
    names: list[str] = []
    sidecar_root = root_dir or gltf_artifact_root_dir(path)
    for raw_name in gltf_raw_sidecar_names(path):
        name = _safe_gltf_sidecar_name(raw_name, source_dir=path.parent, root_dir=sidecar_root)
        if name and name not in names:
            names.append(name)
    return names


def usd_sidecar_names(path: Path, *, root_dir: Path | None = None) -> list[str]:
    names: list[str] = []
    sidecar_root = root_dir or usd_artifact_root_dir(path)
    for raw_name in usd_raw_sidecar_names(path):
        name = _safe_usd_sidecar_name(raw_name, source_dir=path.parent, root_dir=sidecar_root)
        if name and name not in names:
            names.append(name)
    return names


def las_sidecar_names(path: Path) -> list[str]:
    if not is_las_artifact(path):
        return []
    return [name for name in _las_candidate_sidecar_names(path) if (path.parent / name).is_file()]


def invalid_dae_sidecar_names(path: Path, *, root_dir: Path | None = None) -> list[str]:
    invalid: list[str] = []
    sidecar_root = root_dir or dae_artifact_root_dir(path)
    for raw_name in dae_raw_sidecar_names(path):
        if (
            _safe_dae_sidecar_name(raw_name, source_dir=path.parent, root_dir=sidecar_root) is None
            and raw_name not in invalid
        ):
            invalid.append(raw_name)
    return invalid


def invalid_x3d_sidecar_names(path: Path, *, root_dir: Path | None = None) -> list[str]:
    invalid: list[str] = []
    sidecar_root = root_dir or x3d_artifact_root_dir(path)
    for raw_name in x3d_raw_sidecar_names(path):
        if (
            _safe_x3d_sidecar_name(raw_name, source_dir=path.parent, root_dir=sidecar_root) is None
            and raw_name not in invalid
        ):
            invalid.append(raw_name)
    return invalid


def invalid_ply_sidecar_names(path: Path) -> list[str]:
    invalid: list[str] = []
    for raw_name in ply_raw_sidecar_names(path):
        if (
            _safe_ply_sidecar_name(raw_name, source_dir=path.parent, root_dir=path.parent) is None
            and raw_name not in invalid
        ):
            invalid.append(raw_name)
    return invalid


def invalid_gltf_sidecar_names(path: Path, *, root_dir: Path | None = None) -> list[str]:
    invalid: list[str] = []
    sidecar_root = root_dir or gltf_artifact_root_dir(path)
    for raw_name in gltf_raw_sidecar_names(path):
        if (
            _safe_gltf_sidecar_name(raw_name, source_dir=path.parent, root_dir=sidecar_root) is None
            and raw_name not in invalid
        ):
            invalid.append(raw_name)
    return invalid


def invalid_usd_sidecar_names(path: Path, *, root_dir: Path | None = None) -> list[str]:
    invalid: list[str] = []
    sidecar_root = root_dir or usd_artifact_root_dir(path)
    for raw_name in usd_raw_sidecar_names(path):
        if (
            _safe_usd_sidecar_name(raw_name, source_dir=path.parent, root_dir=sidecar_root) is None
            and raw_name not in invalid
        ):
            invalid.append(raw_name)
    return invalid


def missing_dae_sidecars(path: Path, *, root_dir: Path | None = None) -> list[str]:
    missing: list[str] = []
    sidecar_root = root_dir or dae_artifact_root_dir(path)
    for name in dae_sidecar_names(path, root_dir=root_dir):
        if not (sidecar_root / Path(name)).is_file():
            missing.append(name)
    return missing


def missing_x3d_sidecars(path: Path, *, root_dir: Path | None = None) -> list[str]:
    missing: list[str] = []
    sidecar_root = root_dir or x3d_artifact_root_dir(path)
    for name in x3d_sidecar_names(path, root_dir=root_dir):
        if not (sidecar_root / Path(name)).is_file():
            missing.append(name)
    return missing


def missing_ply_sidecars(path: Path) -> list[str]:
    missing: list[str] = []
    for name in ply_sidecar_names(path):
        if not (path.parent / Path(name)).is_file():
            missing.append(name)
    return missing


def missing_gltf_sidecars(path: Path, *, root_dir: Path | None = None) -> list[str]:
    missing: list[str] = []
    sidecar_root = root_dir or gltf_artifact_root_dir(path)
    for name in gltf_sidecar_names(path, root_dir=root_dir):
        if not (sidecar_root / Path(name)).is_file():
            missing.append(name)
    return missing


def missing_usd_sidecars(path: Path, *, root_dir: Path | None = None) -> list[str]:
    missing: list[str] = []
    sidecar_root = root_dir or usd_artifact_root_dir(path)
    for name in usd_sidecar_names(path, root_dir=root_dir):
        if not (sidecar_root / Path(name)).is_file():
            missing.append(name)
    return missing


def copy_dae_sidecars_if_needed(*, source: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    source_root = dae_artifact_root_dir(source)
    destination_root = _uri_destination_root(source=source, destination=destination, root_dir=source_root)
    for sidecar_path, archive_name in dae_sidecar_specs(source, root_dir=source_root):
        copied.append(
            copy_artifact_if_needed(
                source=sidecar_path,
                destination=destination_root / Path(archive_name),
            )
        )
    return copied


def copy_x3d_sidecars_if_needed(*, source: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    source_root = x3d_artifact_root_dir(source)
    destination_root = _uri_destination_root(source=source, destination=destination, root_dir=source_root)
    for sidecar_path, archive_name in x3d_sidecar_specs(source, root_dir=source_root):
        copied.append(
            copy_artifact_if_needed(
                source=sidecar_path,
                destination=destination_root / Path(archive_name),
            )
        )
    return copied


def copy_ply_sidecars_if_needed(*, source: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    for sidecar_path, archive_name in ply_sidecar_specs(source):
        copied.append(
            copy_artifact_if_needed(
                source=sidecar_path,
                destination=destination.parent / Path(archive_name),
            )
        )
    return copied


def copy_gltf_sidecars_if_needed(*, source: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    source_root = gltf_artifact_root_dir(source)
    destination_root = _uri_destination_root(source=source, destination=destination, root_dir=source_root)
    for sidecar_path, archive_name in gltf_sidecar_specs(source):
        copied.append(
            copy_artifact_if_needed(
                source=sidecar_path,
                destination=destination_root / Path(archive_name),
            )
        )
    return copied


def copy_usd_sidecars_if_needed(*, source: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    source_root = usd_artifact_root_dir(source)
    destination_root = _uri_destination_root(source=source, destination=destination, root_dir=source_root)
    for sidecar_path, archive_name in usd_sidecar_specs(source, root_dir=source_root):
        copied.append(
            copy_artifact_if_needed(
                source=sidecar_path,
                destination=destination_root / Path(archive_name),
            )
        )
    return copied


def copy_las_sidecars_if_needed(*, source: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    for sidecar_path, sidecar_name in las_sidecar_specs(source):
        copied.append(
            copy_artifact_if_needed(
                source=sidecar_path,
                destination=destination.parent
                / las_destination_sidecar_name(
                    primary_source=source, primary_destination=destination, sidecar_name=sidecar_name
                ),
            )
        )
    return copied


def las_destination_sidecar_name(*, primary_source: Path, primary_destination: Path, sidecar_name: str) -> str:
    if sidecar_name.startswith(f"{primary_source.name}."):
        return f"{primary_destination.name}{sidecar_name[len(primary_source.name) :]}"
    if sidecar_name.startswith(primary_source.stem):
        sidecar_tail = sidecar_name[len(primary_source.stem) :]
        if sidecar_tail.startswith("."):
            return f"{primary_destination.stem}{sidecar_tail}"
    return sidecar_name


def dae_raw_sidecar_names(path: Path) -> list[str]:
    if not is_dae_artifact(path):
        return []
    try:
        with open_compressed_text(path) as handle:
            text = handle.read()
    except (OSError, UnicodeDecodeError):
        return []
    names: list[str] = []
    for raw_uri in _dae_init_from_values(text):
        uri = raw_uri.strip()
        if not uri or _uri_is_inline_or_fragment(uri):
            continue
        if uri not in names:
            names.append(uri)
    return names


def x3d_raw_sidecar_names(path: Path) -> list[str]:
    if not is_x3d_artifact(path):
        return []
    try:
        with open_compressed_text(path) as handle:
            text = handle.read()
    except (OSError, UnicodeDecodeError):
        return []
    names: list[str] = []
    for raw_value in _x3d_url_values(text):
        for raw_uri in _x3d_url_tokens(raw_value):
            uri = raw_uri.strip()
            if not uri or _uri_is_inline_or_fragment(uri):
                continue
            if uri not in names:
                names.append(uri)
    return names


def ply_raw_sidecar_names(path: Path) -> list[str]:
    if not is_ply_artifact(path):
        return []
    names: list[str] = []
    try:
        with open_compressed_text(path, encoding="ascii") as handle:
            for index, raw_line in enumerate(handle):
                if index >= 512:
                    break
                line = raw_line.strip()
                if not line:
                    continue
                if line.lower() == "end_header":
                    break
                raw_name = _ply_texture_file_value(line)
                if raw_name and raw_name not in names:
                    names.append(raw_name)
    except OSError:
        return []
    return names


def gltf_raw_sidecar_names(path: Path) -> list[str]:
    payload = _load_gltf_payload(path)
    if payload is None:
        return []
    names: list[str] = []
    for section in ("buffers", "images"):
        values = payload.get(section)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            raw_uri = item.get("uri")
            if not isinstance(raw_uri, str) or not raw_uri.strip():
                continue
            if _gltf_uri_is_inline(raw_uri):
                continue
            if raw_uri not in names:
                names.append(raw_uri)
    return names


def usd_raw_sidecar_names(path: Path) -> list[str]:
    if not is_usd_text_artifact(path):
        return []
    try:
        with open_compressed_text(path) as handle:
            text = handle.read()
    except (OSError, UnicodeDecodeError):
        return []
    names: list[str] = []
    for match in _USD_ASSET_REF_RE.finditer(text):
        uri = match.group(1).strip()
        if not uri or _uri_is_inline_or_fragment(uri) or uri.startswith("anon:"):
            continue
        if uri not in names:
            names.append(uri)
    return names


def _dae_init_from_values(text: str) -> list[str]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return [_unescape_xml_text(match.group(1)).strip() for match in _DAE_INIT_FROM_RE.finditer(text)]
    values: list[str] = []
    for element in root.iter():
        if _xml_local_name(element.tag) != "init_from":
            continue
        if element.text and element.text.strip():
            values.append(element.text.strip())
    return values


def _las_candidate_sidecar_names(path: Path) -> list[str]:
    names: list[str] = []
    for suffix in LAS_SIDECAR_SUFFIXES:
        names.append(f"{path.stem}{suffix}")
    for suffix in (".aux.xml", ".lasx"):
        names.append(f"{path.name}{suffix}")
    return list(dict.fromkeys(names))


def _dae_has_parent_relative_uri(path: Path) -> bool:
    for raw_uri in dae_raw_sidecar_names(path):
        parsed = urlsplit(raw_uri.replace("\\", "/").strip())
        if parsed.scheme or parsed.netloc:
            continue
        decoded = unquote(parsed.path)
        if ".." in PurePosixPath(decoded).parts:
            return True
    return False


def _x3d_has_parent_relative_uri(path: Path) -> bool:
    for raw_uri in x3d_raw_sidecar_names(path):
        parsed = urlsplit(raw_uri.replace("\\", "/").strip())
        if parsed.scheme or parsed.netloc:
            continue
        decoded = unquote(parsed.path)
        if ".." in PurePosixPath(decoded).parts:
            return True
    return False


def _gltf_has_parent_relative_uri(path: Path) -> bool:
    for raw_uri in gltf_raw_sidecar_names(path):
        parsed = urlsplit(raw_uri.replace("\\", "/").strip())
        if parsed.scheme or parsed.netloc:
            continue
        decoded = unquote(parsed.path)
        if ".." in PurePosixPath(decoded).parts:
            return True
    return False


def _usd_has_parent_relative_uri(path: Path) -> bool:
    for raw_uri in usd_raw_sidecar_names(path):
        parsed = urlsplit(raw_uri.replace("\\", "/").strip())
        if parsed.scheme or parsed.netloc:
            continue
        decoded = unquote(parsed.path)
        if ".." in PurePosixPath(decoded).parts:
            return True
    return False


def _uri_destination_root(*, source: Path, destination: Path, root_dir: Path) -> Path:
    try:
        relative_source = source.relative_to(root_dir)
    except ValueError:
        return destination.parent
    root = destination
    for _part in relative_source.parts:
        root = root.parent
    return root


def obj_sidecar_specs(path: Path) -> list[tuple[Path, str]]:
    specs: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for sidecar_path, archive_name in [*obj_material_specs(path), *obj_texture_specs(path)]:
        if archive_name in seen:
            continue
        seen.add(archive_name)
        specs.append((sidecar_path, archive_name))
    return specs


def invalid_obj_sidecar_names(path: Path) -> list[str]:
    invalid: list[str] = []
    for raw_name in [*obj_raw_material_names(path), *obj_raw_texture_names(path)]:
        if _safe_obj_sidecar_name(raw_name) is None and raw_name not in invalid:
            invalid.append(raw_name)
    return invalid


def missing_obj_sidecars(path: Path) -> list[str]:
    missing: list[str] = []
    for name in [*obj_material_names(path), *obj_texture_names(path)]:
        if not (path.parent / Path(name)).is_file():
            missing.append(name)
    return missing


def copy_obj_sidecars_if_needed(*, source: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    for sidecar_path, archive_name in obj_sidecar_specs(source):
        copied.append(
            copy_artifact_if_needed(
                source=sidecar_path,
                destination=destination.parent / Path(archive_name),
            )
        )
    return copied


def obj_material_names(path: Path) -> list[str]:
    return _safe_unique_names(obj_raw_material_names(path))


def obj_material_specs(path: Path) -> list[tuple[Path, str]]:
    specs: list[tuple[Path, str]] = []
    for name in obj_material_names(path):
        material = path.parent / Path(name)
        if material.is_file():
            specs.append((material, name))
    return specs


def obj_texture_names(path: Path) -> list[str]:
    return _safe_unique_names(obj_raw_texture_names(path))


def obj_texture_specs(path: Path) -> list[tuple[Path, str]]:
    specs: list[tuple[Path, str]] = []
    for name in obj_texture_names(path):
        texture = path.parent / Path(name)
        if texture.is_file():
            specs.append((texture, name))
    return specs


def obj_raw_material_names(path: Path) -> list[str]:
    if not is_obj_artifact(path):
        return []
    names: list[str] = []
    try:
        with open_compressed_text(path) as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                command, _, value = line.partition(" ")
                if command.lower() != "mtllib":
                    continue
                for token in value.split():
                    if token and token not in names:
                        names.append(token)
    except OSError:
        return []
    return names


def obj_raw_texture_names(path: Path) -> list[str]:
    names: list[str] = []
    invalid_material_names = set(
        invalid for invalid in obj_raw_material_names(path) if _safe_obj_sidecar_name(invalid) is None
    )
    if invalid_material_names:
        return []
    for material_path, material_name in obj_material_specs(path):
        for raw_name in _mtl_raw_texture_names(material_path):
            texture_name = _material_relative_texture_name(material_name=material_name, raw_texture_name=raw_name)
            if texture_name not in names:
                names.append(texture_name)
    return names


def _mtl_raw_texture_names(path: Path) -> list[str]:
    names: list[str] = []
    try:
        with open_compressed_text(path) as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                command, _, value = line.partition(" ")
                if command.lower() not in _MTL_TEXTURE_COMMANDS:
                    continue
                token = _last_non_option_token(value)
                if token and token not in names:
                    names.append(token)
    except OSError:
        return []
    return names


def _last_non_option_token(value: str) -> str | None:
    tokens = [token for token in value.split() if token]
    if not tokens:
        return None
    return tokens[-1]


def _material_relative_texture_name(*, material_name: str, raw_texture_name: str) -> str:
    texture_path = PurePosixPath(raw_texture_name.replace("\\", "/").strip())
    if texture_path.is_absolute():
        return texture_path.as_posix()
    material_parent = PurePosixPath(material_name).parent
    if str(material_parent) == ".":
        return texture_path.as_posix()
    return (material_parent / texture_path).as_posix()


def _safe_unique_names(raw_names: list[str], *, safe_name: Callable[[str], str | None] | None = None) -> list[str]:
    safe_name = safe_name or _safe_point_cloud_sidecar_name
    names: list[str] = []
    for raw_name in raw_names:
        name = safe_name(raw_name)
        if name and name not in names:
            names.append(name)
    return names


def _safe_obj_sidecar_name(value: str) -> str | None:
    return _safe_point_cloud_sidecar_name(value)


def _safe_ply_sidecar_name(value: str, *, source_dir: Path, root_dir: Path | None = None) -> str | None:
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


def _safe_x3d_sidecar_name(value: str, *, source_dir: Path, root_dir: Path | None = None) -> str | None:
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


def _safe_dae_sidecar_name(value: str, *, source_dir: Path, root_dir: Path | None = None) -> str | None:
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


def _safe_gltf_sidecar_name(value: str, *, source_dir: Path, root_dir: Path | None = None) -> str | None:
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


def _safe_usd_sidecar_name(value: str, *, source_dir: Path, root_dir: Path | None = None) -> str | None:
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


def _safe_point_cloud_sidecar_name(value: str) -> str | None:
    normalized = value.replace("\\", "/").strip()
    if not normalized:
        return None
    parsed = urlsplit(normalized)
    if parsed.scheme or parsed.netloc:
        return None
    return _safe_relative_sidecar_path(parsed.path)


def _safe_relative_sidecar_path(value: str) -> str | None:
    normalized = value.replace("\\", "/").strip()
    if not normalized:
        return None
    path = PurePosixPath(normalized)
    if path.is_absolute():
        return None
    parts: list[str] = []
    for part in path.parts:
        if part in {"", "."}:
            return None
        if part == "..":
            if not parts:
                return None
            parts.pop()
            continue
        parts.append(part)
    if not parts:
        return None
    return PurePosixPath(*parts).as_posix()


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


def _load_gltf_payload(path: Path) -> dict[str, object] | None:
    if not is_gltf_artifact(path):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _ply_texture_file_value(line: str) -> str | None:
    tokens = line.split()
    if len(tokens) < 3:
        return None
    if tokens[0].lower() not in {"comment", "obj_info"}:
        return None
    if tokens[1].lower() != "texturefile":
        return None
    return " ".join(tokens[2:]).strip()


def _x3d_url_values(text: str) -> list[str]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return [_unquote_xml_attribute(match.group(1)) for match in _X3D_URL_ATTR_RE.finditer(text)]
    values: list[str] = []
    for element in root.iter():
        for key, value in element.attrib.items():
            if key.rsplit("}", 1)[-1].lower().endswith("url") and value.strip():
                values.append(value.strip())
    return values


def _x3d_url_tokens(value: str) -> list[str]:
    try:
        tokens = shlex.split(value)
    except ValueError:
        tokens = value.split()
    return [token.strip().strip("\"'") for token in tokens if token.strip().strip("\"'")]


def _gltf_uri_is_inline(value: str) -> bool:
    return value.strip().lower().startswith("data:")


def _uri_is_inline_or_fragment(value: str) -> bool:
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


def _unquote_xml_attribute(value: str) -> str:
    stripped = value.strip().strip("\"'")
    return _unescape_xml_text(stripped)
