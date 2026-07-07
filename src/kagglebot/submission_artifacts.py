from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from kagglebot.artifact_io import copy_artifact_if_needed
from kagglebot.asset_modality import ARCHIVE_SUFFIXES, artifact_suffix
from kagglebot.geospatial_artifacts import (
    copy_envi_sidecars_if_needed,
    copy_georeferenced_raster_sidecars_if_needed,
    copy_kml_sidecars_if_needed,
    copy_mapinfo_interchange_sidecars_if_needed,
    copy_mapinfo_sidecars_if_needed,
    copy_vrt_sidecars_if_needed,
    envi_destination_sidecar_name,
    envi_sidecar_specs,
    georeferenced_raster_destination_sidecar_name,
    georeferenced_raster_sidecar_specs,
    is_kml_artifact,
    is_vrt_artifact,
    kml_artifact_root_dir,
    kml_primary_archive_name,
    kml_sidecar_reference_specs,
    mapinfo_destination_sidecar_name,
    mapinfo_interchange_destination_sidecar_name,
    mapinfo_interchange_sidecar_specs,
    mapinfo_sidecar_specs,
    vrt_artifact_root_dir,
    vrt_primary_archive_name,
    vrt_sidecar_reference_specs,
)
from kagglebot.json_utils import load_json_object, write_json_object
from kagglebot.medical_artifacts import (
    analyze_pair_destination_sidecar_name,
    analyze_pair_sidecar_specs,
    copy_analyze_pair_sidecars_if_needed,
    copy_detached_nrrd_sidecars_if_needed,
    copy_metaimage_sidecars_if_needed,
    detached_nrrd_sidecar_specs,
    metaimage_sidecar_specs,
)
from kagglebot.model_artifacts import (
    copy_model_artifact_sidecars_if_needed,
    copy_model_index_shards_if_needed,
    copy_tensorflow_checkpoint_sidecars_if_needed,
    model_artifact_sidecar_specs,
    model_index_shard_specs,
    tensorflow_checkpoint_sidecar_specs,
)
from kagglebot.point_cloud_artifacts import (
    copy_dae_sidecars_if_needed,
    copy_gltf_sidecars_if_needed,
    copy_las_sidecars_if_needed,
    copy_obj_sidecars_if_needed,
    copy_ply_sidecars_if_needed,
    copy_x3d_sidecars_if_needed,
    dae_artifact_root_dir,
    dae_primary_archive_name,
    dae_sidecar_reference_specs,
    gltf_artifact_root_dir,
    gltf_primary_archive_name,
    gltf_sidecar_reference_specs,
    is_dae_artifact,
    is_gltf_artifact,
    is_x3d_artifact,
    las_destination_sidecar_name,
    las_sidecar_specs,
    obj_sidecar_specs,
    ply_sidecar_specs,
    x3d_artifact_root_dir,
    x3d_primary_archive_name,
    x3d_sidecar_reference_specs,
)
from kagglebot.shapefile_artifacts import (
    copy_shapefile_sidecars_if_needed,
    shapefile_bundle_members,
    shapefile_component_suffix,
)

SUBMISSION_MANIFEST_FILENAME = "submission_manifest.json"
ARTIFACT_CLASS_TABULAR = "tabular"
ARTIFACT_CLASS_SINGLE_FILE = "single_file"
ARTIFACT_CLASS_BUNDLE = "bundle"
ARTIFACT_CLASS_MULTI_FILE_ZIP = "multi_file_zip"
ARTIFACT_CLASS_NOTEBOOK_OUTPUT = "notebook_output"
ARTIFACT_CLASS_WRITEUP = "writeup"
ARTIFACT_CLASS_UNKNOWN = "unknown"

_KNOWN_ARTIFACT_CLASSES = {
    ARTIFACT_CLASS_TABULAR,
    ARTIFACT_CLASS_SINGLE_FILE,
    ARTIFACT_CLASS_BUNDLE,
    ARTIFACT_CLASS_MULTI_FILE_ZIP,
    ARTIFACT_CLASS_NOTEBOOK_OUTPUT,
    ARTIFACT_CLASS_WRITEUP,
    ARTIFACT_CLASS_UNKNOWN,
}

_ARTIFACT_CLASS_BY_NORMALIZED = {
    "".join(char for char in value.lower() if char.isalnum()): value for value in _KNOWN_ARTIFACT_CLASSES
}

_ARTIFACT_CLASS_KEYS = ("artifact_class", "artifact_type", "type", "class")
_SUBMISSION_PATH_KEYS = (
    "submission_path",
    "submission_file",
    "submission_filename",
    "submission",
    "artifact_path",
    "artifact_file",
    "source_path",
    "local_path",
    "relative_path",
    "archive_path",
    "output_path",
    "output_file",
    "output",
    "result_path",
    "result_file",
    "result",
    "prediction_path",
    "prediction_file",
    "prediction",
    "path",
    "file",
    "filename",
    "location",
    "uri",
)
_STAGING_DIR_KEYS = (
    "staging_dir",
    "staging_path",
    "bundle_dir",
    "bundle_path",
    "folder_path",
    "directory",
    "folder",
    "dir",
)
_REQUESTED_OUTPUT_PATH_KEYS = (
    "requested_output_path",
    "requested_output_file",
    "requested_submission_path",
    "requested_submission_file",
    "requested_artifact_path",
    "expected_output_path",
    "expected_output_file",
    "expected_submission_path",
    "expected_submission_file",
    "required_output_path",
    "required_output_file",
)
_MEMBERS_KEYS = ("members", "files", "file_paths", "paths", "artifacts", "archive_members", "entries", "items")
_MEMBER_PATH_KEYS = (
    "path",
    "file",
    "filename",
    "source",
    "source_path",
    "source_file",
    "local_path",
    "local_file",
    "relative_path",
    "relative_file",
    "artifact_path",
    "input_path",
    "input_file",
)
_MEMBER_ARCHIVE_PATH_KEYS = (
    "archive_path",
    "archive_file",
    "target_path",
    "target_file",
    "destination_path",
    "destination_file",
    "destination",
    "output_path",
    "output_file",
    "arcname",
    "archive_name",
    "output_name",
    "name",
)
_NESTED_MANIFEST_KEYS = (
    "submission",
    "submission_artifact",
    "artifact",
    "output",
    "result",
    "bundle",
    "archive",
    "staging",
)
_GLOB_META_CHARS = "*?["
_ARCHIVE_ARTIFACT_SUFFIXES = set(ARCHIVE_SUFFIXES)
_DIRECT_PATH_VALUE_KEYS = (
    "path",
    "file",
    "filename",
    "source",
    "source_path",
    "local_path",
    "relative_path",
    "artifact_path",
    "artifact_file",
    "submission_path",
    "submission_file",
    "output_path",
    "output_file",
    "result_path",
    "result_file",
    "prediction_path",
    "prediction_file",
    *_REQUESTED_OUTPUT_PATH_KEYS,
    "location",
    "uri",
)


@dataclass(frozen=True)
class SubmissionManifestMember:
    source_path: Path
    archive_path: str | None = None


@dataclass(frozen=True)
class SubmissionManifestReferences:
    artifact_class: str
    submission_path: Path | None
    requested_output_path: Path | None
    staging_dir: Path | None
    members: list[Path]
    member_specs: list[SubmissionManifestMember]


def normalize_artifact_class(value: object, *, default: str = ARTIFACT_CLASS_UNKNOWN) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in _KNOWN_ARTIFACT_CLASSES:
        return normalized
    compact = _normalize_manifest_key(str(value or ""))
    if compact in _ARTIFACT_CLASS_BY_NORMALIZED:
        return _ARTIFACT_CLASS_BY_NORMALIZED[compact]
    return default


def load_submission_manifest(path: Path) -> dict[str, object] | None:
    if not path.exists() or not path.is_file():
        return None
    return load_json_object(path)


def find_submission_manifest(root: Path) -> Path | None:
    candidates = [
        root / SUBMISSION_MANIFEST_FILENAME,
        root / "output" / SUBMISSION_MANIFEST_FILENAME,
    ]
    for candidate in candidates:
        if _is_valid_submission_manifest(candidate):
            return candidate
    try:
        matches = sorted(
            path for path in root.rglob(SUBMISSION_MANIFEST_FILENAME) if _is_valid_submission_manifest(path)
        )
    except OSError:
        return None
    if not matches:
        return None
    return max(matches, key=lambda path: (path.stat().st_mtime, str(path)))


def resolve_manifest_references(
    manifest_path: Path,
) -> tuple[str, Path | None, Path | None, list[Path]]:
    details = resolve_manifest_reference_details(manifest_path)
    return details.artifact_class, details.submission_path, details.staging_dir, details.members


def resolve_manifest_reference_details(manifest_path: Path) -> SubmissionManifestReferences:
    payload = load_submission_manifest(manifest_path) or {}
    base_dir = manifest_path.parent
    payloads = _manifest_payload_candidates(payload)
    artifact_class = normalize_artifact_class(_first_manifest_value_from_payloads(payloads, _ARTIFACT_CLASS_KEYS))
    submission_path = _resolve_manifest_path(
        base_dir, _first_manifest_path_value_from_payloads(payloads, _SUBMISSION_PATH_KEYS)
    )
    requested_output_path = _resolve_manifest_path(
        base_dir, _first_manifest_path_value_from_payloads(payloads, _REQUESTED_OUTPUT_PATH_KEYS)
    )
    staging_dir = _resolve_manifest_staging_path(
        base_dir, _first_manifest_path_value_from_payloads(payloads, _STAGING_DIR_KEYS)
    )
    if staging_dir is not None and not staging_dir.is_dir():
        staging_dir = None
    member_specs = _resolve_manifest_member_specs(
        base_dir, _first_manifest_value_from_payloads(payloads, _MEMBERS_KEYS)
    )
    members = [member.source_path for member in member_specs]
    artifact_class = _infer_manifest_artifact_class(
        artifact_class=artifact_class,
        submission_path=submission_path,
        staging_dir=staging_dir,
        members=members,
    )
    return SubmissionManifestReferences(
        artifact_class=artifact_class,
        submission_path=submission_path,
        requested_output_path=requested_output_path,
        staging_dir=staging_dir,
        members=members,
        member_specs=member_specs,
    )


def _is_valid_submission_manifest(path: Path) -> bool:
    return load_submission_manifest(path) is not None


def is_submission_manifest_artifact(path: Path) -> bool:
    name = path.name.lower()
    return name == SUBMISSION_MANIFEST_FILENAME or name.endswith(f"_{SUBMISSION_MANIFEST_FILENAME}")


def _copy_submission_artifact_sidecars_if_needed(*, source: Path, destination: Path) -> None:
    copy_shapefile_sidecars_if_needed(source=source, destination=destination)
    copy_envi_sidecars_if_needed(source=source, destination=destination)
    copy_georeferenced_raster_sidecars_if_needed(source=source, destination=destination)
    copy_vrt_sidecars_if_needed(source=source, destination=destination)
    copy_kml_sidecars_if_needed(source=source, destination=destination)
    copy_mapinfo_sidecars_if_needed(source=source, destination=destination)
    copy_mapinfo_interchange_sidecars_if_needed(source=source, destination=destination)
    copy_analyze_pair_sidecars_if_needed(source=source, destination=destination)
    copy_metaimage_sidecars_if_needed(source=source, destination=destination)
    copy_detached_nrrd_sidecars_if_needed(source=source, destination=destination)
    copy_dae_sidecars_if_needed(source=source, destination=destination)
    copy_gltf_sidecars_if_needed(source=source, destination=destination)
    copy_las_sidecars_if_needed(source=source, destination=destination)
    copy_obj_sidecars_if_needed(source=source, destination=destination)
    copy_ply_sidecars_if_needed(source=source, destination=destination)
    copy_x3d_sidecars_if_needed(source=source, destination=destination)
    copy_model_index_shards_if_needed(source=source, destination=destination)
    copy_tensorflow_checkpoint_sidecars_if_needed(source=source, destination=destination)
    copy_model_artifact_sidecars_if_needed(source=source, destination=destination)


def store_submission_artifact(*, source: Path, destination_dir: Path, run_id: str) -> Path:
    if source.is_file() and is_submission_manifest_artifact(source):
        return _store_submission_manifest_artifact(source=source, destination_dir=destination_dir, run_id=run_id)
    suffix = "".join(source.suffixes)
    destination = destination_dir / f"{run_id}_submission{suffix}"
    if is_gltf_artifact(source):
        gltf_root = gltf_artifact_root_dir(source)
        gltf_archive_name = gltf_primary_archive_name(source, root_dir=gltf_root)
        if Path(gltf_archive_name).parent != Path("."):
            destination = destination_dir / f"{run_id}_submission_bundle" / gltf_archive_name
    if is_dae_artifact(source):
        dae_root = dae_artifact_root_dir(source)
        dae_archive_name = dae_primary_archive_name(source, root_dir=dae_root)
        if Path(dae_archive_name).parent != Path("."):
            destination = destination_dir / f"{run_id}_submission_bundle" / dae_archive_name
    if is_x3d_artifact(source):
        x3d_root = x3d_artifact_root_dir(source)
        x3d_archive_name = x3d_primary_archive_name(source, root_dir=x3d_root)
        if Path(x3d_archive_name).parent != Path("."):
            destination = destination_dir / f"{run_id}_submission_bundle" / x3d_archive_name
    if is_kml_artifact(source):
        kml_root = kml_artifact_root_dir(source)
        kml_archive_name = kml_primary_archive_name(source, root_dir=kml_root)
        if Path(kml_archive_name).parent != Path("."):
            destination = destination_dir / f"{run_id}_submission_bundle" / kml_archive_name
    if is_vrt_artifact(source):
        vrt_root = vrt_artifact_root_dir(source)
        vrt_archive_name = vrt_primary_archive_name(source, root_dir=vrt_root)
        if Path(vrt_archive_name).parent != Path("."):
            destination = destination_dir / f"{run_id}_submission_bundle" / vrt_archive_name
    stored = copy_artifact_if_needed(source=source, destination=destination)
    _copy_submission_artifact_sidecars_if_needed(source=source, destination=stored)
    _copy_submission_manifest_for_stored_artifact(source=source, destination=stored)
    return stored


def copy_submission_artifact_to_directory(*, source: Path, destination_dir: Path) -> Path:
    if source.is_file() and is_submission_manifest_artifact(source):
        return _copy_submission_manifest_artifact_to_directory(source=source, destination_dir=destination_dir)
    destination = destination_dir / source.name
    copied = copy_artifact_if_needed(source=source, destination=destination)
    _copy_submission_artifact_sidecars_if_needed(source=source, destination=copied)
    _copy_submission_manifest_for_stored_artifact(source=source, destination=copied)
    return copied


def submission_specific_manifest_path(submission_path: Path) -> Path:
    suffix = "".join(submission_path.suffixes)
    if suffix and submission_path.name.endswith(suffix):
        stem = submission_path.name[: -len(suffix)]
    else:
        stem = submission_path.stem
    return submission_path.with_name(f"{stem}_manifest.json")


def _store_submission_manifest_artifact(*, source: Path, destination_dir: Path, run_id: str) -> Path:
    destination = destination_dir / f"{run_id}_{SUBMISSION_MANIFEST_FILENAME}"
    payload = load_submission_manifest(source)
    if payload is None:
        return copy_artifact_if_needed(source=source, destination=destination)
    try:
        details = resolve_manifest_reference_details(source)
    except ValueError:
        return copy_artifact_if_needed(source=source, destination=destination)

    copied_payload = dict(payload)
    bundle_dir = destination_dir / f"{run_id}_submission_bundle"
    if details.staging_dir is not None and details.staging_dir.exists():
        copy_artifact_if_needed(source=details.staging_dir, destination=bundle_dir)
        copied_payload["staging_dir"] = bundle_dir.name
    elif details.member_specs:
        copied_members: list[object] = []
        for member in details.member_specs:
            if not member.source_path.exists():
                continue
            copied_specs = _copy_manifest_member_to_bundle(
                source_path=member.source_path,
                bundle_dir=bundle_dir,
                base_dir=source.parent,
                archive_path=member.archive_path,
            )
            for copied_member, archive_path in copied_specs:
                try:
                    relative_source = copied_member.relative_to(destination_dir).as_posix()
                except ValueError:
                    relative_source = copied_member.name
                if archive_path:
                    copied_members.append({"source_path": relative_source, "archive_path": archive_path})
                else:
                    copied_members.append(relative_source)
        if copied_members:
            copied_payload["members"] = copied_members
            copied_payload["staging_dir"] = bundle_dir.name
    elif details.submission_path is not None and details.submission_path.exists():
        copied_submission = _copy_manifest_submission_path_artifact(
            source_path=details.submission_path,
            destination_dir=destination_dir,
            run_id=run_id,
        )
        copied_payload["submission_path"] = copied_submission.name

    write_json_object(destination, copied_payload)
    return destination


def _copy_submission_manifest_artifact_to_directory(*, source: Path, destination_dir: Path) -> Path:
    destination = destination_dir / source.name
    payload = load_submission_manifest(source)
    if payload is None:
        return copy_artifact_if_needed(source=source, destination=destination)
    try:
        details = resolve_manifest_reference_details(source)
    except ValueError:
        return copy_artifact_if_needed(source=source, destination=destination)

    copied_payload = dict(payload)
    bundle_dir = destination_dir / "submission_bundle"
    if details.staging_dir is not None and details.staging_dir.exists():
        bundle_dir = destination_dir / details.staging_dir.name
        copy_artifact_if_needed(source=details.staging_dir, destination=bundle_dir)
        copied_payload["staging_dir"] = bundle_dir.name
    elif details.member_specs:
        copied_members: list[object] = []
        for member in details.member_specs:
            if not member.source_path.exists():
                continue
            copied_specs = _copy_manifest_member_to_bundle(
                source_path=member.source_path,
                bundle_dir=bundle_dir,
                base_dir=source.parent,
                archive_path=member.archive_path,
            )
            for copied_member, archive_path in copied_specs:
                try:
                    relative_source = copied_member.relative_to(destination_dir).as_posix()
                except ValueError:
                    relative_source = copied_member.name
                if archive_path:
                    copied_members.append({"source_path": relative_source, "archive_path": archive_path})
                else:
                    copied_members.append(relative_source)
        if copied_members:
            copied_payload["members"] = copied_members
            copied_payload["staging_dir"] = bundle_dir.name
    elif details.submission_path is not None and details.submission_path.exists():
        copied_submission = _copy_manifest_submission_path_to_directory(
            source_path=details.submission_path,
            destination_dir=destination_dir,
        )
        copied_payload["submission_path"] = copied_submission.name

    write_json_object(destination, copied_payload)
    return destination


def _copy_manifest_submission_path_to_directory(*, source_path: Path, destination_dir: Path) -> Path:
    destination = destination_dir / source_path.name
    copied = copy_artifact_if_needed(source=source_path, destination=destination)
    _copy_submission_artifact_sidecars_if_needed(source=source_path, destination=copied)
    return copied


def _copy_manifest_submission_path_artifact(*, source_path: Path, destination_dir: Path, run_id: str) -> Path:
    suffix = "".join(source_path.suffixes)
    destination = destination_dir / f"{run_id}_submission{suffix}"
    copied = copy_artifact_if_needed(source=source_path, destination=destination)
    _copy_submission_artifact_sidecars_if_needed(source=source_path, destination=copied)
    return copied


def _copy_manifest_member_to_bundle(
    *,
    source_path: Path,
    bundle_dir: Path,
    base_dir: Path,
    archive_path: str | None,
) -> list[tuple[Path, str | None]]:
    try:
        relative_path = source_path.relative_to(base_dir)
    except ValueError:
        relative_path = Path(source_path.name)
    destination = bundle_dir / relative_path
    copied = copy_artifact_if_needed(source=source_path, destination=destination)
    copied_specs: list[tuple[Path, str | None]] = [(copied, archive_path)]
    copied_specs.extend(
        _copy_manifest_shapefile_sidecars(
            source_path=source_path,
            copied_path=copied,
            archive_path=archive_path,
        )
    )
    copied_specs.extend(
        _copy_manifest_envi_sidecars(
            source_path=source_path,
            copied_path=copied,
            archive_path=archive_path,
        )
    )
    copied_specs.extend(
        _copy_manifest_georeferenced_raster_sidecars(
            source_path=source_path,
            copied_path=copied,
            archive_path=archive_path,
        )
    )
    copied_specs.extend(
        _copy_manifest_vrt_sidecars(
            source_path=source_path,
            copied_path=copied,
            archive_path=archive_path,
            base_dir=base_dir,
        )
    )
    copied_specs.extend(
        _copy_manifest_model_index_shards(
            source_path=source_path,
            copied_path=copied,
            archive_path=archive_path,
        )
    )
    copied_specs.extend(
        _copy_manifest_tensorflow_checkpoint_sidecars(
            source_path=source_path,
            copied_path=copied,
            archive_path=archive_path,
        )
    )
    copied_specs.extend(
        _copy_manifest_model_artifact_sidecars(
            source_path=source_path,
            copied_path=copied,
            archive_path=archive_path,
        )
    )
    copied_specs.extend(
        _copy_manifest_mapinfo_sidecars(
            source_path=source_path,
            copied_path=copied,
            archive_path=archive_path,
        )
    )
    copied_specs.extend(
        _copy_manifest_mapinfo_interchange_sidecars(
            source_path=source_path,
            copied_path=copied,
            archive_path=archive_path,
        )
    )
    copied_specs.extend(
        _copy_manifest_analyze_pair_sidecars(
            source_path=source_path,
            copied_path=copied,
            archive_path=archive_path,
        )
    )
    copied_specs.extend(
        _copy_manifest_metaimage_sidecars(
            source_path=source_path,
            copied_path=copied,
            archive_path=archive_path,
        )
    )
    copied_specs.extend(
        _copy_manifest_detached_nrrd_sidecars(
            source_path=source_path,
            copied_path=copied,
            archive_path=archive_path,
        )
    )
    copied_specs.extend(
        _copy_manifest_kml_sidecars(
            source_path=source_path,
            copied_path=copied,
            archive_path=archive_path,
            base_dir=base_dir,
        )
    )
    copied_specs.extend(
        _copy_manifest_dae_sidecars(
            source_path=source_path,
            copied_path=copied,
            archive_path=archive_path,
            base_dir=base_dir,
        )
    )
    copied_specs.extend(
        _copy_manifest_x3d_sidecars(
            source_path=source_path,
            copied_path=copied,
            archive_path=archive_path,
            base_dir=base_dir,
        )
    )
    copied_specs.extend(
        _copy_manifest_gltf_sidecars(
            source_path=source_path,
            copied_path=copied,
            archive_path=archive_path,
            base_dir=base_dir,
        )
    )
    copied_specs.extend(
        _copy_manifest_obj_sidecars(
            source_path=source_path,
            copied_path=copied,
            archive_path=archive_path,
        )
    )
    copied_specs.extend(
        _copy_manifest_las_sidecars(
            source_path=source_path,
            copied_path=copied,
            archive_path=archive_path,
        )
    )
    copied_specs.extend(
        _copy_manifest_ply_sidecars(
            source_path=source_path,
            copied_path=copied,
            archive_path=archive_path,
        )
    )
    return copied_specs


def _copy_manifest_shapefile_sidecars(
    *,
    source_path: Path,
    copied_path: Path,
    archive_path: str | None,
) -> list[tuple[Path, str | None]]:
    copied_specs: list[tuple[Path, str | None]] = []
    if shapefile_component_suffix(source_path) != ".shp":
        return copied_specs
    try:
        source_resolved = source_path.resolve()
    except OSError:
        source_resolved = source_path
    for member in shapefile_bundle_members(source_path):
        try:
            if member.resolve() == source_resolved:
                continue
        except OSError:
            pass
        suffix = shapefile_component_suffix(member)
        copied_member = copy_artifact_if_needed(
            source=member,
            destination=copied_path.with_name(f"{copied_path.stem}{suffix}"),
        )
        copied_specs.append((copied_member, _sidecar_archive_path(archive_path, member.name)))
    return copied_specs


def _copy_manifest_envi_sidecars(
    *,
    source_path: Path,
    copied_path: Path,
    archive_path: str | None,
) -> list[tuple[Path, str | None]]:
    copied_specs: list[tuple[Path, str | None]] = []
    for sidecar_path, sidecar_name in envi_sidecar_specs(source_path):
        copied_name = envi_destination_sidecar_name(copied_path, sidecar_name)
        copied_member = copy_artifact_if_needed(source=sidecar_path, destination=copied_path.parent / Path(copied_name))
        copied_specs.append((copied_member, _sidecar_archive_path(archive_path, copied_name)))
    return copied_specs


def _copy_manifest_georeferenced_raster_sidecars(
    *,
    source_path: Path,
    copied_path: Path,
    archive_path: str | None,
) -> list[tuple[Path, str | None]]:
    copied_specs: list[tuple[Path, str | None]] = []
    for sidecar_path, sidecar_name in georeferenced_raster_sidecar_specs(source_path):
        copied_name = georeferenced_raster_destination_sidecar_name(
            primary_source=source_path,
            primary_destination=copied_path,
            sidecar_name=sidecar_name,
        )
        copied_member = copy_artifact_if_needed(source=sidecar_path, destination=copied_path.parent / copied_name)
        copied_specs.append((copied_member, _sidecar_archive_path(archive_path, copied_name)))
    return copied_specs


def _copy_manifest_model_index_shards(
    *,
    source_path: Path,
    copied_path: Path,
    archive_path: str | None,
) -> list[tuple[Path, str | None]]:
    copied_specs: list[tuple[Path, str | None]] = []
    for shard_path, shard_name in model_index_shard_specs(source_path):
        copied_member = copy_artifact_if_needed(source=shard_path, destination=copied_path.parent / Path(shard_name))
        copied_specs.append((copied_member, _sidecar_archive_path(archive_path, shard_name)))
    return copied_specs


def _copy_manifest_tensorflow_checkpoint_sidecars(
    *,
    source_path: Path,
    copied_path: Path,
    archive_path: str | None,
) -> list[tuple[Path, str | None]]:
    copied_specs: list[tuple[Path, str | None]] = []
    for sidecar_path, sidecar_name in tensorflow_checkpoint_sidecar_specs(source_path):
        copied_member = copy_artifact_if_needed(source=sidecar_path, destination=copied_path.parent / sidecar_name)
        copied_specs.append((copied_member, _sidecar_archive_path(archive_path, sidecar_name)))
    return copied_specs


def _copy_manifest_model_artifact_sidecars(
    *,
    source_path: Path,
    copied_path: Path,
    archive_path: str | None,
) -> list[tuple[Path, str | None]]:
    copied_specs: list[tuple[Path, str | None]] = []
    for sidecar_path, sidecar_name in model_artifact_sidecar_specs(source_path):
        copied_member = copy_artifact_if_needed(source=sidecar_path, destination=copied_path.parent / sidecar_name)
        copied_specs.append((copied_member, _sidecar_archive_path(archive_path, sidecar_name)))
    return copied_specs


def _copy_manifest_mapinfo_sidecars(
    *,
    source_path: Path,
    copied_path: Path,
    archive_path: str | None,
) -> list[tuple[Path, str | None]]:
    copied_specs: list[tuple[Path, str | None]] = []
    for sidecar_path, sidecar_name in mapinfo_sidecar_specs(source_path):
        copied_name = mapinfo_destination_sidecar_name(copied_path, sidecar_name)
        copied_member = copy_artifact_if_needed(source=sidecar_path, destination=copied_path.parent / copied_name)
        copied_specs.append((copied_member, _sidecar_archive_path(archive_path, copied_name)))
    return copied_specs


def _copy_manifest_mapinfo_interchange_sidecars(
    *,
    source_path: Path,
    copied_path: Path,
    archive_path: str | None,
) -> list[tuple[Path, str | None]]:
    copied_specs: list[tuple[Path, str | None]] = []
    for sidecar_path, sidecar_name in mapinfo_interchange_sidecar_specs(source_path):
        copied_name = mapinfo_interchange_destination_sidecar_name(copied_path, sidecar_name)
        copied_member = copy_artifact_if_needed(source=sidecar_path, destination=copied_path.parent / copied_name)
        copied_specs.append((copied_member, _sidecar_archive_path(archive_path, copied_name)))
    return copied_specs


def _copy_manifest_analyze_pair_sidecars(
    *,
    source_path: Path,
    copied_path: Path,
    archive_path: str | None,
) -> list[tuple[Path, str | None]]:
    copied_specs: list[tuple[Path, str | None]] = []
    for sidecar_path, sidecar_name in analyze_pair_sidecar_specs(source_path):
        copied_name = analyze_pair_destination_sidecar_name(copied_path, sidecar_name)
        copied_member = copy_artifact_if_needed(source=sidecar_path, destination=copied_path.parent / copied_name)
        copied_specs.append((copied_member, _sidecar_archive_path(archive_path, copied_name)))
    return copied_specs


def _copy_manifest_metaimage_sidecars(
    *,
    source_path: Path,
    copied_path: Path,
    archive_path: str | None,
) -> list[tuple[Path, str | None]]:
    copied_specs: list[tuple[Path, str | None]] = []
    for sidecar_path, sidecar_name in metaimage_sidecar_specs(source_path):
        copied_member = copy_artifact_if_needed(
            source=sidecar_path, destination=copied_path.parent / Path(sidecar_name)
        )
        copied_specs.append((copied_member, _sidecar_archive_path(archive_path, sidecar_name)))
    return copied_specs


def _copy_manifest_detached_nrrd_sidecars(
    *,
    source_path: Path,
    copied_path: Path,
    archive_path: str | None,
) -> list[tuple[Path, str | None]]:
    copied_specs: list[tuple[Path, str | None]] = []
    for sidecar_path, sidecar_name in detached_nrrd_sidecar_specs(source_path):
        copied_member = copy_artifact_if_needed(
            source=sidecar_path, destination=copied_path.parent / Path(sidecar_name)
        )
        copied_specs.append((copied_member, _sidecar_archive_path(archive_path, sidecar_name)))
    return copied_specs


def _copy_manifest_obj_sidecars(
    *,
    source_path: Path,
    copied_path: Path,
    archive_path: str | None,
) -> list[tuple[Path, str | None]]:
    copied_specs: list[tuple[Path, str | None]] = []
    for sidecar_path, sidecar_name in obj_sidecar_specs(source_path):
        copied_member = copy_artifact_if_needed(
            source=sidecar_path, destination=copied_path.parent / Path(sidecar_name)
        )
        copied_specs.append((copied_member, _sidecar_archive_path(archive_path, sidecar_name)))
    return copied_specs


def _copy_manifest_las_sidecars(
    *,
    source_path: Path,
    copied_path: Path,
    archive_path: str | None,
) -> list[tuple[Path, str | None]]:
    copied_specs: list[tuple[Path, str | None]] = []
    for sidecar_path, sidecar_name in las_sidecar_specs(source_path):
        copied_name = las_destination_sidecar_name(
            primary_source=source_path,
            primary_destination=copied_path,
            sidecar_name=sidecar_name,
        )
        copied_member = copy_artifact_if_needed(source=sidecar_path, destination=copied_path.parent / copied_name)
        copied_specs.append((copied_member, _sidecar_archive_path(archive_path, copied_name)))
    return copied_specs


def _copy_manifest_ply_sidecars(
    *,
    source_path: Path,
    copied_path: Path,
    archive_path: str | None,
) -> list[tuple[Path, str | None]]:
    copied_specs: list[tuple[Path, str | None]] = []
    for sidecar_path, sidecar_name in ply_sidecar_specs(source_path):
        copied_member = copy_artifact_if_needed(
            source=sidecar_path, destination=copied_path.parent / Path(sidecar_name)
        )
        copied_specs.append((copied_member, _sidecar_archive_path(archive_path, sidecar_name)))
    return copied_specs


def _copy_manifest_kml_sidecars(
    *,
    source_path: Path,
    copied_path: Path,
    archive_path: str | None,
    base_dir: Path,
) -> list[tuple[Path, str | None]]:
    copied_specs: list[tuple[Path, str | None]] = []
    copied_root = _copied_manifest_root(source_path=source_path, copied_path=copied_path, base_dir=base_dir)
    for sidecar_path, sidecar_name, raw_href in kml_sidecar_reference_specs(source_path, root_dir=base_dir):
        copied_member = copy_artifact_if_needed(source=sidecar_path, destination=copied_root / Path(sidecar_name))
        copied_specs.append((copied_member, _uri_sidecar_archive_path(archive_path, raw_href)))
    return copied_specs


def _copy_manifest_vrt_sidecars(
    *,
    source_path: Path,
    copied_path: Path,
    archive_path: str | None,
    base_dir: Path,
) -> list[tuple[Path, str | None]]:
    copied_specs: list[tuple[Path, str | None]] = []
    copied_root = _copied_manifest_root(source_path=source_path, copied_path=copied_path, base_dir=base_dir)
    for sidecar_path, sidecar_name, raw_source in vrt_sidecar_reference_specs(source_path, root_dir=base_dir):
        copied_member = copy_artifact_if_needed(source=sidecar_path, destination=copied_root / Path(sidecar_name))
        copied_specs.append((copied_member, _uri_sidecar_archive_path(archive_path, raw_source)))
    return copied_specs


def _copy_manifest_dae_sidecars(
    *,
    source_path: Path,
    copied_path: Path,
    archive_path: str | None,
    base_dir: Path,
) -> list[tuple[Path, str | None]]:
    copied_specs: list[tuple[Path, str | None]] = []
    copied_root = _copied_manifest_root(source_path=source_path, copied_path=copied_path, base_dir=base_dir)
    for sidecar_path, sidecar_name, raw_uri in dae_sidecar_reference_specs(source_path, root_dir=base_dir):
        copied_member = copy_artifact_if_needed(source=sidecar_path, destination=copied_root / Path(sidecar_name))
        copied_specs.append((copied_member, _uri_sidecar_archive_path(archive_path, raw_uri)))
    return copied_specs


def _copy_manifest_x3d_sidecars(
    *,
    source_path: Path,
    copied_path: Path,
    archive_path: str | None,
    base_dir: Path,
) -> list[tuple[Path, str | None]]:
    copied_specs: list[tuple[Path, str | None]] = []
    copied_root = _copied_manifest_root(source_path=source_path, copied_path=copied_path, base_dir=base_dir)
    for sidecar_path, sidecar_name, raw_uri in x3d_sidecar_reference_specs(source_path, root_dir=base_dir):
        copied_member = copy_artifact_if_needed(source=sidecar_path, destination=copied_root / Path(sidecar_name))
        copied_specs.append((copied_member, _uri_sidecar_archive_path(archive_path, raw_uri)))
    return copied_specs


def _copy_manifest_gltf_sidecars(
    *,
    source_path: Path,
    copied_path: Path,
    archive_path: str | None,
    base_dir: Path,
) -> list[tuple[Path, str | None]]:
    copied_specs: list[tuple[Path, str | None]] = []
    copied_root = _copied_manifest_root(source_path=source_path, copied_path=copied_path, base_dir=base_dir)
    for sidecar_path, sidecar_name, raw_uri in gltf_sidecar_reference_specs(source_path, root_dir=base_dir):
        copied_member = copy_artifact_if_needed(source=sidecar_path, destination=copied_root / Path(sidecar_name))
        copied_specs.append((copied_member, _uri_sidecar_archive_path(archive_path, raw_uri)))
    return copied_specs


def _copied_manifest_root(*, source_path: Path, copied_path: Path, base_dir: Path) -> Path:
    try:
        relative_path = source_path.relative_to(base_dir)
    except ValueError:
        return copied_path.parent
    root = copied_path
    for _part in relative_path.parts:
        root = root.parent
    return root


def _uri_sidecar_archive_path(primary_archive_path: str | None, raw_uri: str) -> str | None:
    if not primary_archive_path:
        return None
    parsed = urlsplit(raw_uri.replace("\\", "/"))
    parsed_path = unquote(parsed.path)
    parent = PurePosixPath(primary_archive_path).parent
    parts: list[str] = []
    for part in (parent / parsed_path).parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                return None
            parts.pop()
            continue
        parts.append(part)
    if not parts:
        return None
    return PurePosixPath(*parts).as_posix()


def _sidecar_archive_path(primary_archive_path: str | None, member_name: str) -> str | None:
    if not primary_archive_path:
        return None
    parent = PurePosixPath(primary_archive_path).parent
    if str(parent) == ".":
        return Path(member_name).as_posix()
    return (parent / Path(member_name).as_posix()).as_posix()


def _copy_submission_manifest_for_stored_artifact(*, source: Path, destination: Path) -> None:
    manifest_path = source.parent / SUBMISSION_MANIFEST_FILENAME
    payload = load_submission_manifest(manifest_path)
    if payload is None:
        return
    try:
        details = resolve_manifest_reference_details(manifest_path)
    except ValueError:
        return
    if details.submission_path is None:
        return
    try:
        same_source = details.submission_path.resolve(strict=False) == source.resolve(strict=False)
    except OSError:
        same_source = False
    if not same_source:
        return
    copied_payload = dict(payload)
    copied_payload["submission_path"] = destination.name
    write_json_object(submission_specific_manifest_path(destination), copied_payload)


def _resolve_manifest_path(base_dir: Path, value: object) -> Path | None:
    text = _coerce_manifest_path_text(value)
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _resolve_manifest_staging_path(base_dir: Path, value: object) -> Path | None:
    text = _coerce_manifest_path_text(value)
    if not text:
        return None
    path = Path(text)
    _validate_manifest_source_path(path=path, kind="staging")
    if not path.is_absolute():
        path = base_dir / path
    return path


def _coerce_manifest_path_text(value: object) -> str:
    if isinstance(value, Path):
        return str(value).strip()
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        nested = _first_manifest_value(value, _DIRECT_PATH_VALUE_KEYS)
        if nested is None:
            return ""
        return _coerce_manifest_path_text(nested)
    if isinstance(value, list):
        for item in value:
            text = _coerce_manifest_path_text(item)
            if text:
                return text
        return ""
    return ""


def _first_manifest_value(payload: dict[str, object], keys: tuple[str, ...]) -> object:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    normalized_payload = {
        _normalize_manifest_key(key): value
        for key, value in payload.items()
        if isinstance(key, str) and value is not None
    }
    for key in keys:
        value = normalized_payload.get(_normalize_manifest_key(key))
        if value is not None:
            return value
    return None


def _first_manifest_value_from_payloads(payloads: list[dict[str, object]], keys: tuple[str, ...]) -> object:
    for payload in payloads:
        value = _first_manifest_value(payload, keys)
        if value is not None:
            return value
    return None


def _first_manifest_path_value_from_payloads(payloads: list[dict[str, object]], keys: tuple[str, ...]) -> object:
    for payload in payloads:
        for candidate in _manifest_values(payload, keys):
            if _coerce_manifest_path_text(candidate):
                return candidate
    return None


def _manifest_values(payload: dict[str, object], keys: tuple[str, ...]) -> list[object]:
    values: list[object] = []
    seen_ids: set[int] = set()

    def add(value: object) -> None:
        if value is None or id(value) in seen_ids:
            return
        seen_ids.add(id(value))
        values.append(value)

    for key in keys:
        add(payload.get(key))
    normalized_payload = {
        _normalize_manifest_key(key): value
        for key, value in payload.items()
        if isinstance(key, str) and value is not None
    }
    for key in keys:
        add(normalized_payload.get(_normalize_manifest_key(key)))
    return values


def _manifest_payload_candidates(payload: dict[str, object]) -> list[dict[str, object]]:
    candidates = [payload]
    seen = {id(payload)}
    for key in _NESTED_MANIFEST_KEYS:
        value = _first_manifest_value(payload, (key,))
        if isinstance(value, dict) and id(value) not in seen:
            candidates.append(value)
            seen.add(id(value))
    return candidates


def _infer_manifest_artifact_class(
    *,
    artifact_class: str,
    submission_path: Path | None,
    staging_dir: Path | None,
    members: list[Path],
) -> str:
    if artifact_class != ARTIFACT_CLASS_UNKNOWN:
        return artifact_class
    if staging_dir is not None or members:
        return ARTIFACT_CLASS_BUNDLE
    if submission_path is not None and submission_path.is_dir():
        return ARTIFACT_CLASS_BUNDLE
    if submission_path is not None and _is_archive_artifact_path(submission_path):
        return ARTIFACT_CLASS_MULTI_FILE_ZIP
    return artifact_class


def _is_archive_artifact_path(path: Path) -> bool:
    return artifact_suffix(path) in _ARCHIVE_ARTIFACT_SUFFIXES


def _normalize_manifest_key(value: str) -> str:
    return "".join(char for char in str(value).lower() if char.isalnum())


def _resolve_manifest_members(base_dir: Path, value: object) -> list[Path]:
    return [member.source_path for member in _resolve_manifest_member_specs(base_dir, value)]


def _resolve_manifest_member_specs(base_dir: Path, value: object) -> list[SubmissionManifestMember]:
    if isinstance(value, (str, Path)):
        items = [(value, None)]
    elif isinstance(value, list):
        items = [(item, None) for item in value]
    elif isinstance(value, dict):
        if _manifest_dict_is_single_member(value):
            items = [(value, None)]
        else:
            items = _manifest_member_mapping_items(base_dir, value)
    else:
        return []
    members: list[SubmissionManifestMember] = []
    seen: set[str] = set()
    for item, archive_hint in items:
        if isinstance(item, (str, Path)):
            path = Path(item)
        elif isinstance(item, dict):
            member_path = _first_manifest_value(item, _MEMBER_PATH_KEYS)
            member_path_text = _coerce_manifest_path_text(member_path)
            if not member_path_text:
                continue
            path = Path(member_path_text)
            explicit_archive_path = _first_manifest_value(item, _MEMBER_ARCHIVE_PATH_KEYS)
            explicit_archive_path_text = _coerce_manifest_member_archive_path_text(explicit_archive_path)
            if explicit_archive_path_text:
                archive_hint = explicit_archive_path_text
        else:
            continue
        _validate_manifest_source_path(path=path, kind="source")
        if not path.is_absolute():
            path = base_dir / path
        expanded = _expand_manifest_member_path(path)
        for member in expanded:
            archive_path = _manifest_member_archive_path(archive_hint, source_path=path, expanded_path=member)
            key = _manifest_member_identity(member, archive_path=archive_path)
            if key in seen:
                continue
            seen.add(key)
            members.append(
                SubmissionManifestMember(
                    source_path=member,
                    archive_path=archive_path,
                )
            )
    return members


def _validate_manifest_source_path(*, path: Path, kind: str) -> None:
    text = str(path).strip().replace("\\", "/")
    if path.is_absolute() or re.match(r"^[A-Za-z]:/", text):
        raise ValueError(f"unsafe absolute manifest {kind} path: {path}")
    if ".." in Path(text).parts:
        raise ValueError(f"unsafe path traversal in manifest {kind} path: {path}")


def _manifest_member_mapping_items(base_dir: Path, value: dict[str, object]) -> list[tuple[object, str | None]]:
    items: list[tuple[object, str | None]] = []
    for key, item in value.items():
        key_text = str(key)
        if _manifest_mapping_entry_is_source_to_archive(base_dir=base_dir, key_text=key_text, item=item):
            archive_hint = _coerce_manifest_member_archive_path_text(item)
            items.append((key_text, archive_hint))
        else:
            items.append((item, key_text))
    return items


def _manifest_mapping_entry_is_source_to_archive(*, base_dir: Path, key_text: str, item: object) -> bool:
    archive_hint = _coerce_manifest_member_archive_path_text(item)
    if not archive_hint:
        return False
    return _manifest_source_value_exists(base_dir, key_text) and not _manifest_source_value_exists(base_dir, item)


def _manifest_source_value_exists(base_dir: Path, value: object) -> bool:
    text = _coerce_manifest_path_text(value)
    if not text:
        return False
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    if any(char in str(path) for char in _GLOB_META_CHARS):
        return any(Path(match).exists() for match in glob.glob(str(path), recursive=True))
    return path.exists()


def _manifest_dict_is_single_member(value: dict[str, object]) -> bool:
    return _first_manifest_value(value, _MEMBER_PATH_KEYS) is not None


def _coerce_manifest_member_archive_path_text(value: object) -> str:
    if isinstance(value, dict):
        nested = _first_manifest_value(value, _MEMBER_ARCHIVE_PATH_KEYS + _DIRECT_PATH_VALUE_KEYS)
        if nested is None:
            return ""
        return _coerce_manifest_member_archive_path_text(nested)
    return _coerce_manifest_path_text(value)


def _expand_manifest_member_path(path: Path) -> list[Path]:
    if not any(char in str(path) for char in _GLOB_META_CHARS):
        if path.is_dir():
            members = sorted(child for child in path.rglob("*") if child.is_file() or child.is_dir())
            return members or [path]
        return [path]
    matches = sorted(Path(match) for match in glob.glob(str(path), recursive=True))
    members = [match for match in matches if match.is_file() or match.is_dir()]
    return members or [path]


def _manifest_member_identity(path: Path, *, archive_path: str | None = None) -> str:
    suffix = f"\0{archive_path}" if archive_path else ""
    return f"{path.resolve(strict=False)}{suffix}"


def _manifest_member_archive_path(value: str | None, *, source_path: Path, expanded_path: Path) -> str | None:
    cleaned = _clean_manifest_archive_path(value)
    if cleaned is None:
        return None
    if source_path.is_dir():
        try:
            relative = expanded_path.relative_to(source_path).as_posix()
        except ValueError:
            return cleaned
        return f"{cleaned.rstrip('/')}/{relative}" if relative else cleaned.rstrip("/")
    if any(char in str(source_path) for char in _GLOB_META_CHARS):
        if _archive_path_looks_like_directory(cleaned):
            relative = _manifest_glob_relative_path(source_path, expanded_path)
            return f"{cleaned.rstrip('/')}/{relative}"
        return None
    return cleaned


def _manifest_glob_relative_path(source_path: Path, expanded_path: Path) -> str:
    static_root = _manifest_glob_static_root(source_path)
    try:
        relative = expanded_path.relative_to(static_root)
    except ValueError:
        return expanded_path.name
    text = relative.as_posix()
    return text if text else expanded_path.name


def _manifest_glob_static_root(path: Path) -> Path:
    static_root = Path(path.anchor) if path.anchor else Path()
    start_index = 1 if path.anchor else 0
    for part in path.parts[start_index:]:
        if any(char in part for char in _GLOB_META_CHARS):
            break
        static_root = static_root / part
    return static_root if str(static_root) else path.parent


def _clean_manifest_archive_path(value: str | None) -> str | None:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return None
    path = Path(text)
    if path.is_absolute() or re.match(r"^[A-Za-z]:/", text):
        raise ValueError(f"unsafe manifest archive path: {value}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe path traversal in manifest archive path: {value}")
    return path.as_posix()


def _archive_path_looks_like_directory(value: str) -> bool:
    return value.endswith("/") or Path(value).suffix == ""
