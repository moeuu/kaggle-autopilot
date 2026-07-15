from __future__ import annotations

import os
import re
import sqlite3
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

import py7zr
import rarfile
import zstandard as zstd

from kagglebot.artifact_io import (
    copy_artifact_if_needed,
    same_stem_tabular_artifact_filenames,
    tabular_artifact_stem,
)
from kagglebot.asset_modality import (
    DIRECTORY_ARRAY_SUFFIXES,
    MODEL_ARTIFACT_COMPOUND_SUFFIXES,
    MODEL_ARTIFACT_NAME_TOKENS,
    MODEL_ARTIFACT_SUFFIXES,
    artifact_suffix,
    asset_suffix,
    is_data_asset_path,
)
from kagglebot.compression_suffixes import open_zstd_tar
from kagglebot.geospatial_artifacts import (
    copy_envi_sidecars_if_needed,
    copy_georeferenced_raster_sidecars_if_needed,
    copy_kml_sidecars_if_needed,
    copy_mapinfo_interchange_sidecars_if_needed,
    copy_mapinfo_sidecars_if_needed,
    copy_vrt_sidecars_if_needed,
    is_kml_artifact,
    is_vrt_artifact,
    kml_artifact_root_dir,
    kml_primary_archive_name,
    vrt_artifact_root_dir,
    vrt_primary_archive_name,
)
from kagglebot.medical_artifacts import (
    copy_analyze_pair_sidecars_if_needed,
    copy_detached_nrrd_sidecars_if_needed,
    copy_metaimage_sidecars_if_needed,
)
from kagglebot.model_artifacts import (
    MODEL_DIRECTORY_ARTIFACT_SUFFIXES,
    TENSORFLOW_CHECKPOINT_INDEX_SUFFIX,
    copy_model_artifact_sidecars_if_needed,
    copy_model_index_shards_if_needed,
    copy_tensorflow_checkpoint_sidecars_if_needed,
    is_model_directory_artifact,
    is_saved_model_marker_file,
    is_tensorflow_checkpoint_index_artifact,
    model_directory_artifact_suffix,
)
from kagglebot.output_discovery import iter_named_output_paths
from kagglebot.point_cloud_artifacts import (
    copy_dae_sidecars_if_needed,
    copy_gltf_sidecars_if_needed,
    copy_las_sidecars_if_needed,
    copy_obj_sidecars_if_needed,
    copy_ply_sidecars_if_needed,
    copy_usd_sidecars_if_needed,
    copy_x3d_sidecars_if_needed,
    dae_artifact_root_dir,
    dae_primary_archive_name,
    gltf_artifact_root_dir,
    gltf_primary_archive_name,
    is_dae_artifact,
    is_gltf_artifact,
    is_usd_artifact,
    is_x3d_artifact,
    usd_artifact_root_dir,
    usd_primary_archive_name,
    x3d_artifact_root_dir,
    x3d_primary_archive_name,
)
from kagglebot.sample_name_aliases import SAMPLE_OUTPUT_NAME_TOKENS
from kagglebot.shapefile_artifacts import (
    SHAPEFILE_PRIMARY_SUFFIX,
    SHAPEFILE_SIDECAR_SUFFIXES,
    copy_shapefile_sidecars_if_needed,
    same_stem_shapefile_exists,
    shapefile_component_suffix,
)
from kagglebot.submission_artifacts import (
    ARTIFACT_CLASS_SINGLE_FILE,
    find_submission_manifest,
    resolve_manifest_reference_details,
)
from kagglebot.submission_extension_hints import (
    ARCHIVE_SUBMISSION_SUFFIXES_ORDERED,
    NON_DIRECTORY_NON_TABULAR_SUBMISSION_SUFFIXES,
    ZSTD_TAR_ARCHIVE_SUBMISSION_SUFFIXES,
)
from kagglebot.submission_output_naming import (
    all_submission_output_suffixes,
    configured_submission_filename_is_template,
)
from kagglebot.submission_sample_discovery import (
    SQLITE_TABULAR_SUFFIXES,
    TABULAR_SUBMISSION_SUFFIXES,
    is_json_lines_tabular_suffix,
    tabular_data_row_count_capped,
    tabular_suffix,
)

_TABULAR_SUBMISSION_SUFFIXES = set(TABULAR_SUBMISSION_SUFFIXES)
_NON_TABULAR_SINGLE_FILE_SUFFIXES = set(NON_DIRECTORY_NON_TABULAR_SUBMISSION_SUFFIXES) | set(SQLITE_TABULAR_SUFFIXES)
_MODEL_SINGLE_FILE_SUFFIXES = (
    MODEL_ARTIFACT_SUFFIXES
    | MODEL_ARTIFACT_COMPOUND_SUFFIXES
    | MODEL_DIRECTORY_ARTIFACT_SUFFIXES
    | {TENSORFLOW_CHECKPOINT_INDEX_SUFFIX}
)
_DIRECTORY_OUTPUT_SUFFIXES = DIRECTORY_ARRAY_SUFFIXES | MODEL_DIRECTORY_ARTIFACT_SUFFIXES
_ZSTD_TAR_ARCHIVE_SUFFIXES = set(ZSTD_TAR_ARCHIVE_SUBMISSION_SUFFIXES)
_INTERMEDIATE_SUBMISSION_RE = re.compile(
    r"^submission(?:_[A-Za-z0-9_.-]+)?_fold(?P<fold>\d+)(?P<suffix>(?:\.[A-Za-z0-9]+)+)$",
    re.IGNORECASE,
)
_ARCHIVE_SUBMISSION_SUFFIXES = ARCHIVE_SUBMISSION_SUFFIXES_ORDERED
_COMPOUND_SUBMISSION_ARCHIVE_NAMES = tuple(f"submission{suffix}" for suffix in _ARCHIVE_SUBMISSION_SUFFIXES)
_DIRECT_SUBMISSION_SUFFIXES = all_submission_output_suffixes()
_CONFIGURED_SUBMISSION_EXCLUDED_NAMES = {"metrics.json", "plan.json", "submission_manifest.json"}
_PLAIN_SUBMISSION_DIRECTORY_NAMES = {"sub", "submission", "submissions", "submit"}
_GENERIC_SUBMISSION_NAME_TOKENS = SAMPLE_OUTPUT_NAME_TOKENS | {
    "forecast",
    "forecasts",
    "mask",
    "masks",
    "pred",
    "preds",
    "result",
    "results",
    "segmentation",
    "segmentations",
    "sub",
    "submit",
}
_GENERIC_SUBMISSION_EXCLUDE_TOKENS = {
    "cv",
    "diagnostic",
    "diagnostics",
    "feature",
    "features",
    "fold",
    "format",
    "metric",
    "metrics",
    "oof",
    "sample",
    "schema",
    "split",
    "template",
    "train",
    "valid",
    "validation",
}
LOCAL_KERNEL_OPTIONAL_ARTIFACTS: tuple[str, ...] = (
    "oof_predictions.csv",
    "split_diagnostics.json",
    "feature_suspects.csv",
    "submission_manifest.json",
    "metrics_summary.json",
    "cv_results.json",
    "cv_summary.json",
    "pipeline_diagnostics.json",
)
_PLAIN_SUBMISSION_DIRECTORY_EXCLUDED_MEMBER_NAMES = {
    name.lower() for name in LOCAL_KERNEL_OPTIONAL_ARTIFACTS
} | _CONFIGURED_SUBMISSION_EXCLUDED_NAMES


def find_submission_file(output_dir: Path) -> Path | None:
    manifest_path = find_submission_manifest(output_dir)
    if manifest_path is not None:
        manifest_details = resolve_manifest_reference_details(manifest_path)
        submission_path = manifest_details.submission_path
        if submission_path is not None and submission_path.exists() and submission_path.is_file():
            return submission_path
        if (
            submission_path is not None
            and submission_path.exists()
            and submission_path.is_dir()
            and manifest_details.artifact_class == ARTIFACT_CLASS_SINGLE_FILE
            and _is_nonempty_artifact(submission_path)
        ):
            return submission_path
        if (
            submission_path is not None
            and submission_path.exists()
            and submission_path.is_dir()
            and manifest_details.artifact_class != ARTIFACT_CLASS_SINGLE_FILE
        ):
            return manifest_path
        if manifest_details.staging_dir is not None or manifest_details.members:
            return manifest_path
    preferred = _find_preferred_submission_candidate(output_dir, require_tabular_data_rows=True)
    if preferred is not None:
        return preferred
    archive = find_submission_archive_file(output_dir)
    if archive:
        return archive
    candidate = _find_submission_by_extension(output_dir, require_tabular_data_rows=True)
    if candidate:
        return candidate
    candidate = _find_generic_submission_by_extension(output_dir, require_tabular_data_rows=True)
    if candidate:
        return candidate
    candidate = find_intermediate_submission_file(output_dir)
    if candidate:
        return candidate
    return _find_submission_by_extension(output_dir, require_tabular_data_rows=False)


def find_output_file(output_dir: Path, filename: str) -> Path | None:
    """Find the newest matching artifact within an output tree.

    Local kernels can be executed repeatedly for the same run/iteration while
    iterating on fixes. In that scenario, stale artifacts may exist alongside
    fresh ones (or nested under additional run directories). Prefer the most
    recently modified match to avoid accidentally reusing stale outputs.
    """

    candidates: list[Path] = []
    candidate_filenames = same_stem_tabular_artifact_filenames(filename)
    for candidate_filename in candidate_filenames:
        direct = output_dir / candidate_filename
        if direct.exists():
            candidates.append(direct)
    try:
        for candidate_filename in candidate_filenames:
            candidates.extend(path for path in iter_named_output_paths(output_dir, candidate_filename) if path.exists())
    except OSError:
        # Best-effort discovery; callers handle missing artifacts.
        pass
    directory_suffix = _expected_directory_output_suffix(filename)
    if directory_suffix in MODEL_DIRECTORY_ARTIFACT_SUFFIXES:
        candidates.extend(_model_directory_output_candidates(output_dir, directory_suffix))
    usable = [
        path for path in candidates if _expected_output_candidate_is_usable(path, directory_suffix=directory_suffix)
    ]
    if not usable:
        return None
    # Deterministic tie-breaker: path string.
    return max(usable, key=lambda path: (path.stat().st_mtime, str(path)))


def _expected_directory_output_suffix(filename: str) -> str:
    name = Path(str(filename or "")).name.lower()
    for suffix in sorted(_DIRECTORY_OUTPUT_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix):
            return suffix
    return ""


def _expected_output_candidate_is_usable(path: Path, *, directory_suffix: str) -> bool:
    if path.is_file():
        return True
    if not directory_suffix or not path.is_dir():
        return False
    if directory_suffix in MODEL_DIRECTORY_ARTIFACT_SUFFIXES:
        return _non_tabular_single_file_suffix(path) == directory_suffix and _is_nonempty_artifact(path)
    return asset_suffix(path) == directory_suffix and _is_nonempty_artifact(path)


def _model_directory_output_candidates(output_dir: Path, suffix: str) -> list[Path]:
    candidates: list[Path] = []
    try:
        paths = output_dir.rglob("*")
    except OSError:
        return candidates
    for path in paths:
        if not path.is_dir():
            continue
        if _non_tabular_single_file_suffix(path) != suffix:
            continue
        if not _is_nonempty_artifact(path):
            continue
        candidates.append(path)
    return candidates


def find_intermediate_submission_file(output_dir: Path) -> Path | None:
    """Find the newest fold-level submission emitted by an interrupted run."""

    candidates: list[tuple[int, Path]] = []
    try:
        paths = output_dir.rglob("submission*_fold*.*")
    except OSError:
        return None
    for path in paths:
        if not path.is_file():
            continue
        match = _INTERMEDIATE_SUBMISSION_RE.match(path.name)
        if match is None:
            continue
        if tabular_suffix(path) not in _TABULAR_SUBMISSION_SUFFIXES:
            continue
        try:
            fold = int(match.group("fold"))
        except ValueError:
            continue
        candidates.append((fold, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[1].stat().st_mtime, item[0], str(item[1])))[1]


def find_submission_archive_file(output_dir: Path) -> Path | None:
    """Find the newest archive-style code submission in an output tree."""

    names = {name.lower() for name in _COMPOUND_SUBMISSION_ARCHIVE_NAMES}
    candidates: list[Path] = []
    for name in _COMPOUND_SUBMISSION_ARCHIVE_NAMES:
        candidate = output_dir / name
        if _archive_candidate_is_usable(candidate):
            candidates.append(candidate)
    try:
        for path in output_dir.rglob("*"):
            if path.name.lower() in names and _archive_candidate_is_usable(path):
                candidates.append(path)
    except OSError:
        pass
    if not candidates:
        return None
    preferred_name = _preferred_submission_filename()
    return max(
        candidates,
        key=lambda path: (
            1 if preferred_name is not None and path.name == preferred_name else 0,
            _shapefile_submission_candidate_priority(path),
            1 if path.is_dir() else 0,
            path.stat().st_mtime,
            str(path),
        ),
    )


def resolve_local_kernel_artifacts(
    *,
    kernel_dir: Path,
    output_dir: Path,
    started_at: float,
) -> tuple[Path | None, Path | None]:
    candidates = local_kernel_artifact_roots(kernel_dir=kernel_dir, output_dir=output_dir)
    submission_candidates: list[Path] = []
    metrics_candidates: list[Path] = []
    for root in candidates:
        if not root.exists():
            continue
        sub = find_submission_file(root)
        if sub is not None and sub.exists():
            submission_candidates.append(sub)
        metric_path = find_output_file(root, "metrics.json")
        if metric_path is not None and metric_path.exists():
            metrics_candidates.append(metric_path)

    min_mtime = started_at - 1.0
    submission_path = pick_latest_artifact(submission_candidates, min_mtime=min_mtime)
    metrics_path = pick_latest_artifact(metrics_candidates, min_mtime=min_mtime)
    return submission_path, metrics_path


def resolve_local_kernel_artifact_file(
    *,
    kernel_dir: Path,
    output_dir: Path,
    started_at: float,
    filename: str,
) -> Path | None:
    file_candidates: list[Path] = []
    for root in local_kernel_artifact_roots(kernel_dir=kernel_dir, output_dir=output_dir):
        if not root.exists():
            continue
        match = find_output_file(root, filename)
        if match is not None and match.exists():
            file_candidates.append(match)
    min_mtime = started_at - 1.0
    return pick_latest_artifact(file_candidates, min_mtime=min_mtime)


def local_kernel_artifact_roots(*, kernel_dir: Path, output_dir: Path) -> list[Path]:
    return [
        output_dir,
        # Legacy generated kernels may write to the slug-level kernel_output
        # directory instead of the per-run output dir.
        kernel_dir.parents[2] / "kernel_output",
        # Many kernels treat the parent of the staged copy (run_dir) as the
        # "challenge dir" and write artifacts under run_dir/outputs.
        kernel_dir.parent / "outputs",
        kernel_dir.parent,
        kernel_dir / "outputs",
        Path("/kaggle/working"),
        kernel_dir,
    ]


def find_newest_existing_path(paths: list[Path], *, min_mtime: float | None = None) -> Path | None:
    candidates: list[tuple[float, int, str, Path]] = []
    for path in paths:
        try:
            if not path.exists():
                continue
            stat = path.stat()
        except OSError:
            continue
        if min_mtime is not None and stat.st_mtime < min_mtime:
            continue
        candidates.append((float(stat.st_mtime), int(stat.st_size), str(path), path))
    if not candidates:
        return None
    return max(candidates)[3]


def pick_latest_artifact(paths: list[Path], *, min_mtime: float) -> Path | None:
    return find_newest_existing_path(paths, min_mtime=min_mtime)


def copy_local_kernel_primary_artifacts(
    *,
    submission_path: Path,
    metrics_path: Path | None,
    output_dir: Path,
) -> tuple[Path, Path | None]:
    submission_destination = output_dir / submission_path.name
    if is_gltf_artifact(submission_path):
        submission_destination = output_dir / gltf_primary_archive_name(
            submission_path,
            root_dir=gltf_artifact_root_dir(submission_path),
        )
    if is_dae_artifact(submission_path):
        submission_destination = output_dir / dae_primary_archive_name(
            submission_path,
            root_dir=dae_artifact_root_dir(submission_path),
        )
    if is_x3d_artifact(submission_path):
        submission_destination = output_dir / x3d_primary_archive_name(
            submission_path,
            root_dir=x3d_artifact_root_dir(submission_path),
        )
    if is_usd_artifact(submission_path):
        submission_destination = output_dir / usd_primary_archive_name(
            submission_path,
            root_dir=usd_artifact_root_dir(submission_path),
        )
    if is_kml_artifact(submission_path):
        submission_destination = output_dir / kml_primary_archive_name(
            submission_path,
            root_dir=kml_artifact_root_dir(submission_path),
        )
    if is_vrt_artifact(submission_path):
        submission_destination = output_dir / vrt_primary_archive_name(
            submission_path,
            root_dir=vrt_artifact_root_dir(submission_path),
        )
    submission_dst = copy_artifact_if_needed(
        source=submission_path,
        destination=submission_destination,
    )
    copy_shapefile_sidecars_if_needed(source=submission_path, destination=submission_dst)
    copy_envi_sidecars_if_needed(source=submission_path, destination=submission_dst)
    copy_georeferenced_raster_sidecars_if_needed(source=submission_path, destination=submission_dst)
    copy_vrt_sidecars_if_needed(source=submission_path, destination=submission_dst)
    copy_kml_sidecars_if_needed(source=submission_path, destination=submission_dst)
    copy_mapinfo_sidecars_if_needed(source=submission_path, destination=submission_dst)
    copy_mapinfo_interchange_sidecars_if_needed(source=submission_path, destination=submission_dst)
    copy_analyze_pair_sidecars_if_needed(source=submission_path, destination=submission_dst)
    copy_metaimage_sidecars_if_needed(source=submission_path, destination=submission_dst)
    copy_detached_nrrd_sidecars_if_needed(source=submission_path, destination=submission_dst)
    copy_dae_sidecars_if_needed(source=submission_path, destination=submission_dst)
    copy_gltf_sidecars_if_needed(source=submission_path, destination=submission_dst)
    copy_las_sidecars_if_needed(source=submission_path, destination=submission_dst)
    copy_obj_sidecars_if_needed(source=submission_path, destination=submission_dst)
    copy_ply_sidecars_if_needed(source=submission_path, destination=submission_dst)
    copy_usd_sidecars_if_needed(source=submission_path, destination=submission_dst)
    copy_x3d_sidecars_if_needed(source=submission_path, destination=submission_dst)
    copy_model_index_shards_if_needed(source=submission_path, destination=submission_dst)
    copy_tensorflow_checkpoint_sidecars_if_needed(source=submission_path, destination=submission_dst)
    copy_model_artifact_sidecars_if_needed(source=submission_path, destination=submission_dst)
    metrics_dst = None
    if metrics_path is not None:
        metrics_dst = copy_artifact_if_needed(
            source=metrics_path,
            destination=output_dir / "metrics.json",
        )
    return submission_dst, metrics_dst


def copy_optional_local_kernel_artifacts(
    *,
    kernel_dir: Path,
    output_dir: Path,
    started_at: float,
    filenames: tuple[str, ...] = LOCAL_KERNEL_OPTIONAL_ARTIFACTS,
) -> list[Path]:
    copied: list[Path] = []
    for filename in filenames:
        optional_src = resolve_local_kernel_artifact_file(
            kernel_dir=kernel_dir,
            output_dir=output_dir,
            started_at=started_at,
            filename=filename,
        )
        if optional_src is None:
            continue
        destination_name = optional_src.name if optional_src.name != filename else filename
        copied.append(copy_artifact_if_needed(source=optional_src, destination=output_dir / destination_name))
    return copied


def _find_submission_by_extension(output_dir: Path, *, require_tabular_data_rows: bool) -> Path | None:
    preferred = _find_preferred_submission_candidate(
        output_dir,
        require_tabular_data_rows=require_tabular_data_rows,
    )
    if preferred is not None:
        return preferred
    archive = find_submission_archive_file(output_dir)
    if archive is not None:
        return archive
    candidates: list[Path] = []
    preferred_name = _preferred_submission_filename()
    for suffix in sorted(_DIRECT_SUBMISSION_SUFFIXES):
        candidate = output_dir / f"submission{suffix}"
        if _submission_candidate_is_usable(candidate, require_tabular_data_rows=require_tabular_data_rows):
            candidates.append(candidate)
    for path in output_dir.rglob("submission.*"):
        if not path.is_file() and not path.is_dir():
            continue
        if (
            tabular_suffix(path) not in _TABULAR_SUBMISSION_SUFFIXES
            and not _archive_submission_suffix(path)
            and not _non_tabular_single_file_suffix(path)
        ):
            continue
        if not _submission_candidate_is_usable(path, require_tabular_data_rows=require_tabular_data_rows):
            continue
        candidates.append(path)
    if not candidates:
        candidates.extend(_plain_submission_directory_candidates(output_dir))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: (
            1 if preferred_name is not None and path.name == preferred_name else 0,
            _shapefile_submission_candidate_priority(path),
            path.stat().st_mtime,
            str(path),
        ),
    )


def _find_preferred_submission_candidate(output_dir: Path, *, require_tabular_data_rows: bool) -> Path | None:
    preferred_name = _preferred_submission_filename()
    if preferred_name is None:
        return None
    candidates: list[Path] = []
    preferred = output_dir / preferred_name
    if _submission_candidate_is_usable(preferred, require_tabular_data_rows=require_tabular_data_rows):
        candidates.append(preferred)
    try:
        candidates.extend(
            path
            for path in iter_named_output_paths(output_dir, preferred_name)
            if _submission_candidate_is_usable(path, require_tabular_data_rows=require_tabular_data_rows)
        )
    except OSError:
        pass
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: (
            _shapefile_submission_candidate_priority(path),
            1 if path.is_dir() else 0,
            path.stat().st_mtime,
            str(path),
        ),
    )


def _find_generic_submission_by_extension(output_dir: Path, *, require_tabular_data_rows: bool) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    try:
        paths = output_dir.rglob("*")
    except OSError:
        return None
    for path in paths:
        if not path.is_file() and not path.is_dir():
            continue
        if (
            path.is_dir()
            and tabular_suffix(path) not in _TABULAR_SUBMISSION_SUFFIXES
            and not _archive_submission_suffix(path)
            and not _non_tabular_single_file_suffix(path)
        ):
            score = _generic_submission_name_score(path)
            if score <= 0:
                continue
            if not _plain_submission_directory_has_usable_file(path):
                continue
            candidates.append((max(score - 1, 1), path))
            continue
        if (
            tabular_suffix(path) not in _TABULAR_SUBMISSION_SUFFIXES
            and not _archive_submission_suffix(path)
            and not _non_tabular_single_file_suffix(path)
        ):
            continue
        score = _generic_submission_name_score(path)
        if score <= 0:
            continue
        if not _submission_candidate_is_usable(path, require_tabular_data_rows=require_tabular_data_rows):
            continue
        candidates.append((score, path))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item[0],
            _shapefile_submission_candidate_priority(item[1]),
            1 if item[1].is_dir() else 0,
            item[1].stat().st_mtime,
            str(item[1]),
        ),
    )[1]


def _shapefile_submission_candidate_priority(path: Path) -> int:
    suffix = shapefile_component_suffix(path)
    if suffix == SHAPEFILE_PRIMARY_SUFFIX:
        return 2
    if suffix in SHAPEFILE_SIDECAR_SUFFIXES and _same_stem_shapefile_exists(path):
        return -1
    return 0


def _same_stem_shapefile_exists(path: Path) -> bool:
    return same_stem_shapefile_exists(path)


def _generic_submission_name_score(path: Path) -> int:
    if _INTERMEDIATE_SUBMISSION_RE.match(path.name):
        return 0
    stem = tabular_artifact_stem(path).lower()
    tokens = {token for token in re.split(r"[^a-z0-9]+", stem) if token}
    if tokens & _GENERIC_SUBMISSION_EXCLUDE_TOKENS:
        return 0
    if tokens & _GENERIC_SUBMISSION_NAME_TOKENS:
        return 3
    artifact_candidate = _non_tabular_single_file_suffix(path)
    if artifact_candidate in _MODEL_SINGLE_FILE_SUFFIXES and (tokens & MODEL_ARTIFACT_NAME_TOKENS):
        return 3
    compact = re.sub(r"[^a-z0-9]+", "", stem)
    if compact in _GENERIC_SUBMISSION_NAME_TOKENS:
        return 2
    if artifact_candidate in _MODEL_SINGLE_FILE_SUFFIXES and (compact in MODEL_ARTIFACT_NAME_TOKENS):
        return 2
    return 0


def _submission_candidate_is_usable(path: Path, *, require_tabular_data_rows: bool) -> bool:
    if not path.is_file() and not path.is_dir():
        return False
    if path.name.lower() in _CONFIGURED_SUBMISSION_EXCLUDED_NAMES:
        return False
    candidate_tabular_suffix = tabular_suffix(path)
    non_tabular_suffix = _non_tabular_single_file_suffix(path)
    if (
        path.is_dir()
        and candidate_tabular_suffix not in _TABULAR_SUBMISSION_SUFFIXES
        and not _archive_submission_suffix(path)
        and not non_tabular_suffix
    ):
        return _plain_submission_directory_has_usable_file(path)
    if _archive_submission_suffix(path):
        return _archive_candidate_is_usable(path)
    if non_tabular_suffix and non_tabular_suffix not in _TABULAR_SUBMISSION_SUFFIXES:
        if non_tabular_suffix in SQLITE_TABULAR_SUFFIXES:
            return _sqlite_candidate_has_data_rows(path)
        return _is_nonempty_artifact(path)
    if candidate_tabular_suffix in _TABULAR_SUBMISSION_SUFFIXES:
        if require_tabular_data_rows:
            return _tabular_candidate_has_data_rows(path)
        return _tabular_candidate_is_readable(path)
    if non_tabular_suffix:
        return _is_nonempty_artifact(path)
    if not require_tabular_data_rows:
        return True
    if candidate_tabular_suffix not in _TABULAR_SUBMISSION_SUFFIXES:
        return _is_nonempty_file(path)
    return _tabular_candidate_has_data_rows(path)


def _tabular_candidate_has_data_rows(path: Path) -> bool:
    row_count = _tabular_candidate_data_row_count(path)
    return row_count is not None and row_count > 0


def _tabular_candidate_is_readable(path: Path) -> bool:
    return _tabular_candidate_data_row_count(path) is not None


def _tabular_candidate_data_row_count(path: Path) -> int | None:
    try:
        suffix = tabular_suffix(path)
        if is_json_lines_tabular_suffix(suffix):
            row_count = tabular_data_row_count_capped(path, cap=0)
            if row_count is None or row_count <= 0:
                return row_count
            import pandas as pd

            pd.read_json(path, lines=True, nrows=1)
            return row_count
        return tabular_data_row_count_capped(path, cap=0)
    except Exception:  # noqa: BLE001 - discovery must not crash on a corrupt candidate.
        return None


def _plain_submission_directory_candidates(output_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for name in sorted(_PLAIN_SUBMISSION_DIRECTORY_NAMES):
        candidate = output_dir / name
        if _plain_submission_directory_has_usable_file(candidate):
            candidates.append(candidate)
    try:
        for path in output_dir.rglob("*"):
            if path.name.lower() not in _PLAIN_SUBMISSION_DIRECTORY_NAMES:
                continue
            if _plain_submission_directory_has_usable_file(path):
                candidates.append(path)
    except OSError:
        pass
    return candidates


def _plain_submission_directory_has_usable_file(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        for child in path.rglob("*"):
            if not child.is_file() or child.is_symlink():
                continue
            if _plain_submission_directory_member_is_excluded(child):
                continue
            if child.stat().st_size > 0:
                return True
    except OSError:
        return False
    return False


def _plain_submission_directory_member_is_excluded(path: Path) -> bool:
    lowered_name = path.name.lower()
    if lowered_name in _PLAIN_SUBMISSION_DIRECTORY_EXCLUDED_MEMBER_NAMES:
        return True
    stem = tabular_artifact_stem(path).lower()
    tokens = {token for token in re.split(r"[^a-z0-9]+", stem) if token}
    return bool(tokens & _GENERIC_SUBMISSION_EXCLUDE_TOKENS)


def _preferred_submission_filename() -> str | None:
    raw = str(os.getenv("KAGGLEBOT_SUBMISSION_FILENAME") or "").strip()
    if not raw:
        return None
    name = Path(raw).name
    if not name:
        return None
    lowered_name = name.lower()
    if _is_local_kernel_optional_artifact_name(name) or lowered_name in _CONFIGURED_SUBMISSION_EXCLUDED_NAMES:
        return None
    if configured_submission_filename_is_template(name):
        return None
    return name


def _is_local_kernel_optional_artifact_name(name: str) -> bool:
    lowered_name = Path(name).name.lower()
    if lowered_name in LOCAL_KERNEL_OPTIONAL_ARTIFACTS:
        return True
    return any(
        lowered_name in same_stem_tabular_artifact_filenames(optional) for optional in LOCAL_KERNEL_OPTIONAL_ARTIFACTS
    )


def _archive_submission_suffix(path: Path) -> str:
    suffix = artifact_suffix(path)
    return suffix if suffix in _ARCHIVE_SUBMISSION_SUFFIXES else ""


def _non_tabular_single_file_suffix(path: Path) -> str:
    if is_saved_model_marker_file(path):
        return ""
    if is_model_directory_artifact(path):
        return model_directory_artifact_suffix(path)
    if is_tensorflow_checkpoint_index_artifact(path):
        return TENSORFLOW_CHECKPOINT_INDEX_SUFFIX
    suffix = asset_suffix(path)
    if suffix == TENSORFLOW_CHECKPOINT_INDEX_SUFFIX:
        return ""
    if suffix in _NON_TABULAR_SINGLE_FILE_SUFFIXES and (path.is_file() or is_data_asset_path(path)):
        return suffix
    suffix = path.suffix.lower()
    if suffix in _NON_TABULAR_SINGLE_FILE_SUFFIXES:
        return suffix
    return ""


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _archive_candidate_is_usable(path: Path) -> bool:
    if not _is_nonempty_file(path):
        return False
    suffix = _archive_submission_suffix(path)
    try:
        if suffix == ".zip":
            return _zip_archive_has_usable_file(path)
        if suffix in {".7z", ".rar"}:
            return _external_archive_has_usable_file(path, suffix=suffix)
        if suffix in _ARCHIVE_SUBMISSION_SUFFIXES:
            return _tar_archive_has_usable_file(path, suffix=suffix)
    except (
        OSError,
        ValueError,
        tarfile.TarError,
        zipfile.BadZipFile,
        zstd.ZstdError,
        py7zr.Bad7zFile,
        py7zr.DecompressionError,
        py7zr.UnsupportedCompressionMethodError,
        rarfile.Error,
    ):
        return False
    return False


def _sqlite_candidate_has_data_rows(path: Path) -> bool:
    if not _is_nonempty_file(path):
        return False
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            for table in _sqlite_user_tables(conn):
                row = conn.execute(f"SELECT 1 FROM {_quote_sqlite_identifier(table)} LIMIT 1").fetchone()
                if row is not None:
                    return True
    except sqlite3.DatabaseError:
        return False
    return False


def _sqlite_user_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type IN ('table', 'view')
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()
    return [str(row[0]) for row in rows]


def _quote_sqlite_identifier(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _zip_archive_has_usable_file(path: Path) -> bool:
    seen_names: set[str] = set()
    has_file = False
    with zipfile.ZipFile(path, "r") as archive:
        for member in archive.infolist():
            member_name = _safe_archive_member_name(member.filename)
            if member.flag_bits & 0x1 or _zip_member_is_symlink(member):
                return False
            if member.is_dir():
                member_name = f"{member_name.rstrip('/')}/"
            if member_name in seen_names:
                return False
            seen_names.add(member_name)
            if not member.is_dir():
                has_file = True
    return has_file


def _tar_archive_has_usable_file(path: Path, *, suffix: str) -> bool:
    seen_names: set[str] = set()
    has_file = False
    if suffix in _ZSTD_TAR_ARCHIVE_SUFFIXES:
        with open_zstd_tar(path) as archive:
            for member in archive:
                member_name = _safe_archive_member_name(member.name)
                if member_name in seen_names or (not member.isfile() and not member.isdir()):
                    return False
                seen_names.add(member_name)
                if member.isfile():
                    has_file = True
        return has_file
    with tarfile.open(path, "r:*") as archive:
        for member in archive.getmembers():
            member_name = _safe_archive_member_name(member.name)
            if member_name in seen_names or (not member.isfile() and not member.isdir()):
                return False
            seen_names.add(member_name)
            if member.isfile():
                has_file = True
    return has_file


def _external_archive_has_usable_file(path: Path, *, suffix: str) -> bool:
    seen_names: set[str] = set()
    has_file = False
    if suffix == ".7z":
        with py7zr.SevenZipFile(path, mode="r") as archive:
            if archive.needs_password():
                return False
            for member in archive.list():
                member_name = _safe_archive_member_name(getattr(member, "filename", ""))
                if getattr(member, "is_directory", False):
                    member_name = f"{member_name.rstrip('/')}/"
                if member_name in seen_names:
                    return False
                seen_names.add(member_name)
                if getattr(member, "is_directory", False):
                    continue
                if getattr(member, "is_symlink", False) or not getattr(member, "is_file", False):
                    return False
                has_file = True
        return has_file
    with rarfile.RarFile(path) as archive:
        for member in archive.infolist():
            member_name = _safe_archive_member_name(getattr(member, "filename", ""))
            if member.is_dir():
                member_name = f"{member_name.rstrip('/')}/"
            if member_name in seen_names:
                return False
            seen_names.add(member_name)
            if member.is_dir():
                continue
            if member.needs_password() or member.is_symlink() or not member.is_file():
                return False
            has_file = True
    return has_file


def _zip_member_is_symlink(member: zipfile.ZipInfo) -> bool:
    if member.create_system != 3:
        return False
    return stat.S_IFMT(member.external_attr >> 16) == stat.S_IFLNK


def _safe_archive_member_name(name: str) -> str:
    normalized = str(name or "").replace("\\", "/")
    if not normalized.strip():
        raise ValueError("archive contains an empty member name")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized):
        raise ValueError(f"archive member has unsafe absolute path: {name}")
    parts = PurePosixPath(normalized).parts
    if ".." in parts:
        raise ValueError(f"archive member has unsafe path traversal: {name}")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized or normalized == ".":
        raise ValueError("archive contains an empty member name")
    return normalized


def _is_nonempty_artifact(path: Path) -> bool:
    if path.is_file():
        return _is_nonempty_file(path)
    if not path.is_dir():
        return False
    try:
        return any(child.is_file() and not child.is_symlink() for child in path.rglob("*"))
    except OSError:
        return False
