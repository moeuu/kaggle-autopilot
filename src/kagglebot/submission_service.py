from __future__ import annotations

import bz2
import csv
import gzip
import lzma
import re
import sqlite3
import stat
import tarfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from io import StringIO
from pathlib import Path, PurePosixPath

import py7zr
import rarfile
import zstandard as zstd

from kagglebot.artifact_io import copy_artifact_if_needed
from kagglebot.asset_modality import (
    DIRECTORY_ARRAY_SUFFIXES,
    artifact_suffix,
)
from kagglebot.baseline_tokens import ID_LIKE_COLUMN_NAMES
from kagglebot.compression_suffixes import (
    ASSET_COMPRESSION_SUFFIXES,
    compression_suffix_for,
    open_zstd_tar,
    strip_compression_suffix,
    write_compressed_bytes,
)
from kagglebot.exceptions import SubmissionValidationError
from kagglebot.geospatial_artifacts import (
    envi_sidecar_specs,
    georeferenced_raster_sidecar_specs,
    invalid_envi_sidecar_names,
    invalid_kml_sidecar_names,
    invalid_vrt_sidecar_names,
    is_envi_header_artifact,
    is_kml_artifact,
    is_mapinfo_interchange_artifact,
    is_mapinfo_tab_artifact,
    kml_artifact_root_dir,
    kml_primary_archive_name,
    kml_sidecar_specs,
    mapinfo_bundle_members,
    mapinfo_component_suffix,
    mapinfo_interchange_sidecar_specs,
    missing_envi_sidecars,
    missing_kml_sidecars,
    missing_mapinfo_interchange_sidecars,
    missing_mapinfo_sidecars,
    missing_vrt_sidecars,
    vrt_artifact_root_dir,
    vrt_primary_archive_name,
    vrt_sidecar_specs,
)
from kagglebot.history import SubmissionLedger
from kagglebot.medical_artifacts import (
    analyze_pair_sidecar_specs,
    detached_nrrd_sidecar_specs,
    invalid_detached_nrrd_sidecar_names,
    invalid_metaimage_sidecar_names,
    is_analyze_pair_artifact,
    is_detached_nrrd_header_artifact,
    is_metaimage_header_artifact,
    metaimage_sidecar_specs,
    missing_analyze_pair_sidecars,
    missing_detached_nrrd_sidecars,
    missing_metaimage_sidecars,
)
from kagglebot.model_artifacts import (
    invalid_model_index_shard_names,
    is_model_directory_artifact,
    is_model_index_artifact,
    is_tensorflow_checkpoint_artifact,
    missing_model_index_shards,
    model_artifact_sidecar_specs,
    model_index_shard_specs,
    tensorflow_checkpoint_sidecar_specs,
)
from kagglebot.point_cloud_artifacts import (
    dae_artifact_root_dir,
    dae_primary_archive_name,
    dae_sidecar_specs,
    gltf_artifact_root_dir,
    gltf_primary_archive_name,
    gltf_sidecar_specs,
    invalid_dae_sidecar_names,
    invalid_gltf_sidecar_names,
    invalid_obj_sidecar_names,
    invalid_ply_sidecar_names,
    invalid_usd_sidecar_names,
    invalid_x3d_sidecar_names,
    is_dae_artifact,
    is_gltf_artifact,
    is_las_artifact,
    is_obj_artifact,
    is_ply_artifact,
    is_usd_artifact,
    is_x3d_artifact,
    las_sidecar_specs,
    missing_dae_sidecars,
    missing_gltf_sidecars,
    missing_obj_sidecars,
    missing_ply_sidecars,
    missing_usd_sidecars,
    missing_x3d_sidecars,
    obj_sidecar_specs,
    ply_sidecar_specs,
    usd_artifact_root_dir,
    usd_primary_archive_name,
    usd_sidecar_specs,
    x3d_artifact_root_dir,
    x3d_primary_archive_name,
    x3d_sidecar_specs,
)
from kagglebot.shapefile_artifacts import (
    SHAPEFILE_COMPONENT_SUFFIXES,
    SHAPEFILE_REQUIRED_SUFFIXES,
    shapefile_bundle_members,
    shapefile_component_suffix,
)
from kagglebot.solver.io import read_table, write_table
from kagglebot.submission.guard import run_kaggle_submit
from kagglebot.submission.validate import (
    discover_test_ids,
    infer_required_id_suffix,
    normalize_id_with_required_suffix,
    validate_submission,
)
from kagglebot.submission_artifacts import (
    ARTIFACT_CLASS_BUNDLE,
    ARTIFACT_CLASS_MULTI_FILE_ZIP,
    ARTIFACT_CLASS_SINGLE_FILE,
    ARTIFACT_CLASS_TABULAR,
    ARTIFACT_CLASS_UNKNOWN,
    SUBMISSION_MANIFEST_FILENAME,
    find_submission_manifest,
    is_submission_manifest_artifact,
    load_submission_manifest,
    normalize_artifact_class,
    resolve_manifest_reference_details,
    submission_specific_manifest_path,
)
from kagglebot.submission_extension_hints import (
    ARCHIVE_SUBMISSION_SUFFIXES,
    EXTERNAL_ARCHIVE_SUBMISSION_SUFFIXES,
    NON_DIRECTORY_NON_TABULAR_SUBMISSION_SUFFIXES,
    TAR_ARCHIVE_SUBMISSION_SUFFIXES,
    ZSTD_TAR_ARCHIVE_SUBMISSION_SUFFIXES,
)
from kagglebot.submission_format import (
    SubmissionFormatHint,
    extract_submission_section,
    load_submission_format_hint,
    parse_submission_format,
)
from kagglebot.submission_sample_discovery import (
    SQLITE_TABULAR_SUFFIXES,
    TABULAR_ARROW_IPC_SUFFIXES,
    TABULAR_EXCEL_SUFFIXES,
    TABULAR_HDF_SUFFIXES,
    TABULAR_HTML_SUFFIX_PREFIXES,
    TABULAR_PARQUET_SUFFIXES,
    TABULAR_PICKLE_SUFFIXES,
    TABULAR_STATA_SUFFIXES,
    TABULAR_STRUCTURED_SUFFIXES,
    TABULAR_SUBMISSION_SUFFIXES,
    TABULAR_TEXT_SUFFIXES,
    default_delimited_text_separator,
    find_usable_sample_submissions,
    is_delimited_text_tabular_suffix,
    is_json_lines_tabular_suffix,
    is_txt_like_tabular_suffix,
    open_tabular_text,
    sample_candidate_key,
    tabular_file_has_data_rows,
    tabular_suffix,
    write_xml_tabular_frame,
)
from kagglebot.table_columns import frame_with_normalized_table_columns, normalize_table_column_names
from kagglebot.validation import ensure_not_duplicate_submission, ensure_submission_rate_limit

_KAGGLE_SUBMISSION_SOFT_MAX_BYTES = 10_000_000
_KAGGLE_SUBMISSION_MESSAGE_MAX_CHARS = 100
_KAGGLE_SUBMISSION_COMPACT_FLOAT_FORMAT = "%.10g"
_KAGGLE_SUBMISSION_COMPACT_MIN_BYTES_SAVED = 32 * 1024
_KAGGLE_SUBMISSION_COMPACT_MIN_RELATIVE_SAVED = 0.001
_TABULAR_SUBMISSION_SUFFIXES = set(TABULAR_SUBMISSION_SUFFIXES)
_TABULAR_PICKLE_SUFFIXES = set(TABULAR_PICKLE_SUFFIXES)
_TABULAR_STRUCTURED_SUFFIXES = set(TABULAR_STRUCTURED_SUFFIXES)
_TABULAR_TEXT_SUFFIXES = set(TABULAR_TEXT_SUFFIXES)
_TABULAR_ARROW_IPC_SUFFIXES = set(TABULAR_ARROW_IPC_SUFFIXES)
_TABULAR_PARQUET_SUFFIXES = set(TABULAR_PARQUET_SUFFIXES)
_TABULAR_EXCEL_SUFFIXES = set(TABULAR_EXCEL_SUFFIXES)
_TABULAR_HDF_SUBMISSION_SUFFIXES = set(TABULAR_HDF_SUFFIXES) & _TABULAR_SUBMISSION_SUFFIXES
_TABULAR_HTML_SUFFIX_PREFIXES = tuple(TABULAR_HTML_SUFFIX_PREFIXES)
_TABULAR_STATA_SUFFIXES = set(TABULAR_STATA_SUFFIXES)
_NON_TABULAR_SINGLE_FILE_SUFFIXES = set(NON_DIRECTORY_NON_TABULAR_SUBMISSION_SUFFIXES)
_ARCHIVE_SUBMISSION_SUFFIXES = set(ARCHIVE_SUBMISSION_SUFFIXES)
_ZIP_SUBMISSION_SUFFIX = ".zip"
_EXTERNAL_ARCHIVE_SUBMISSION_SUFFIXES = EXTERNAL_ARCHIVE_SUBMISSION_SUFFIXES
_TAR_SUBMISSION_SUFFIXES = TAR_ARCHIVE_SUBMISSION_SUFFIXES
_TAR_SUBMISSION_WRITE_MODES = {
    ".tar": "w",
    ".tar.gz": "w:gz",
    ".tgz": "w:gz",
    ".tar.bz2": "w:bz2",
    ".tbz2": "w:bz2",
    ".tar.xz": "w:xz",
    ".txz": "w:xz",
    **{suffix: "w" for suffix in ZSTD_TAR_ARCHIVE_SUBMISSION_SUFFIXES},
}
_ZIP_MEMBER_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_ZIP_FILE_EXTERNAL_ATTR = 0o100644 << 16
_ZIP_DIR_EXTERNAL_ATTR = 0o40755 << 16
_TAR_MEMBER_MTIME = 0
_ID_LIKE_COLUMN_NAMES = ID_LIKE_COLUMN_NAMES
_PREDICTION_LIKE_COLUMN_NAMES = {
    "category",
    "class",
    "confidence",
    "label",
    "pred",
    "prediction",
    "prediction_string",
    "pred_string",
    "prob",
    "probability",
    "proba",
    "result",
    "score",
    "scores",
    "target",
    "output",
    "value",
    "y",
}


class _TarMembersSnapshot:
    def __init__(self, members: list[tarfile.TarInfo]):
        self._members = members

    def __enter__(self) -> _TarMembersSnapshot:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def getmembers(self) -> list[tarfile.TarInfo]:
        return list(self._members)


def _is_json_lines_suffix(suffix: str) -> bool:
    return is_json_lines_tabular_suffix(suffix)


def _is_txt_like_suffix(suffix: str) -> bool:
    return is_txt_like_tabular_suffix(suffix)


def _is_delimited_text_suffix(suffix: str) -> bool:
    return is_delimited_text_tabular_suffix(suffix)


@dataclass(frozen=True)
class SubmissionConfig:
    slug: str
    data_dir: Path
    sample_submission_path: Path
    submission_ledger_path: Path
    dry_run: bool = False
    force_submit: bool = False
    bypass_rate_limit: bool = False


@dataclass(frozen=True)
class SubmissionResult:
    message: str
    submission_path: Path
    exit_code: int
    stdout: str
    stderr: str


class SubmissionService:
    def __init__(self, config: SubmissionConfig):
        self._config = config

    def submit(
        self,
        *,
        submission_path: Path,
        message: str,
        run_id: str | None,
        submission_kind: str | None = None,
        out_of_band: bool = False,
        source_run_id: str | None = None,
        source_iteration: int | None = None,
    ) -> SubmissionResult:
        prepared_path = self.validate_and_prepare_submission(submission_path)
        return self.submit_prepared(
            prepared_path=prepared_path,
            message=message,
            run_id=run_id,
            submission_kind=submission_kind,
            out_of_band=out_of_band,
            source_run_id=source_run_id,
            source_iteration=source_iteration,
        )

    @staticmethod
    def _normalize_submission_message(message: str) -> str:
        normalized = " ".join(str(message or "").split()).strip()
        if not normalized:
            raise SubmissionValidationError("submission message is empty")
        if len(normalized) <= _KAGGLE_SUBMISSION_MESSAGE_MAX_CHARS:
            return normalized
        ellipsis = "..."
        keep = max(_KAGGLE_SUBMISSION_MESSAGE_MAX_CHARS - len(ellipsis), 0)
        if keep <= 0:
            return normalized[:_KAGGLE_SUBMISSION_MESSAGE_MAX_CHARS]
        return normalized[:keep].rstrip() + ellipsis

    def validate_and_prepare_submission(self, submission_path: Path) -> Path:
        self._validate_submission_input_exists(submission_path)
        original_submission_path = submission_path
        format_hint = self._resolve_submission_format_hint()
        artifact_class = self._resolve_submission_artifact_class(
            format_hint=format_hint, submission_path=submission_path
        )
        manifest_path = self._resolve_submission_manifest_path(submission_path)
        if manifest_path is not None:
            if load_submission_manifest(manifest_path) is None:
                raise SubmissionValidationError(f"invalid submission manifest: {manifest_path}")
            try:
                manifest_details = resolve_manifest_reference_details(manifest_path)
            except ValueError as exc:
                raise SubmissionValidationError(str(exc)) from exc
            artifact_class = manifest_details.artifact_class
            manifest_submission_path = manifest_details.submission_path
            requested_output_path = manifest_details.requested_output_path
            staging_dir = manifest_details.staging_dir
            members = manifest_details.members
            self._validate_specific_manifest_matches_submission_file(
                manifest_path=manifest_path,
                submission_path=original_submission_path,
                manifest_submission_path=manifest_submission_path,
            )
            if (
                manifest_submission_path is None
                and submission_path.is_file()
                and not is_submission_manifest_artifact(submission_path)
                and manifest_path == submission_specific_manifest_path(submission_path)
                and manifest_path.name != SUBMISSION_MANIFEST_FILENAME
            ):
                manifest_submission_path = submission_path
            if artifact_class == ARTIFACT_CLASS_UNKNOWN and manifest_submission_path is not None:
                artifact_class = self._resolve_submission_artifact_class(
                    format_hint=format_hint,
                    submission_path=manifest_submission_path,
                )
            if artifact_class in {ARTIFACT_CLASS_BUNDLE, ARTIFACT_CLASS_MULTI_FILE_ZIP}:
                prepared_bundle = self._prepare_bundle_submission(
                    base_path=manifest_path,
                    submission_path=manifest_submission_path,
                    staging_dir=staging_dir,
                    members=members,
                    member_archive_names=[member.archive_path for member in manifest_details.member_specs],
                    target_suffix=self._preferred_archive_suffix(format_hint),
                )
                return self._enforce_expected_submission_format(
                    submission_path=prepared_bundle,
                    format_hint=format_hint,
                )
            if manifest_submission_path is not None:
                if (
                    requested_output_path is not None
                    and (
                        artifact_class == ARTIFACT_CLASS_TABULAR
                        or self._is_tabular_submission(manifest_submission_path)
                    )
                    and self._requested_output_requires_non_tabular_artifact(requested_output_path)
                    and requested_output_path.name != manifest_submission_path.name
                ):
                    raise SubmissionValidationError(
                        "manifest describes a tabular fallback for a non-tabular requested output:\n"
                        f"  requested_output_path: {requested_output_path.name}\n"
                        f"  emitted_submission_path: {manifest_submission_path.name}\n"
                        "Refusing to submit the fallback as-is; produce the requested artifact or update the manifest."
                    )
                submission_path = manifest_submission_path
                artifact_class = self._resolve_submission_artifact_class(
                    format_hint=format_hint,
                    submission_path=submission_path,
                    manifest_artifact_class=artifact_class,
                )
            elif submission_path.is_file() and is_submission_manifest_artifact(submission_path):
                raise SubmissionValidationError(
                    "manifest submission artifact has no usable submission reference; "
                    "provide submission_path, staging_dir, or members/files"
                )
        if submission_path.is_dir():
            if self._is_directory_single_file_submission(submission_path, artifact_class):
                prepared_directory_asset = self._build_submission_zip(submission_path)
                self._validate_zip_submission(prepared_directory_asset)
                return prepared_directory_asset
            if artifact_class not in {ARTIFACT_CLASS_BUNDLE, ARTIFACT_CLASS_MULTI_FILE_ZIP}:
                raise SubmissionValidationError(
                    f"submission path is a directory but artifact class is not bundle/multi_file_zip: {submission_path}"
                )
            prepared_bundle = self._prepare_bundle_submission(
                base_path=submission_path,
                submission_path=None,
                staging_dir=submission_path,
                members=[],
                target_suffix=self._preferred_archive_suffix(format_hint),
            )
            return self._enforce_expected_submission_format(
                submission_path=prepared_bundle,
                format_hint=format_hint,
            )
        if self._is_zip_submission(submission_path):
            self._validate_zip_submission(submission_path)
            return self._enforce_expected_submission_format(
                submission_path=submission_path,
                format_hint=format_hint,
            )
        if self._is_tar_submission(submission_path):
            self._validate_tar_submission(
                submission_path,
                require_code_contract=self._tar_code_contract_required(format_hint),
            )
            return self._enforce_expected_submission_format(
                submission_path=submission_path,
                format_hint=format_hint,
            )
        if self._is_external_archive_submission(submission_path):
            self._validate_external_archive_submission(submission_path)
            return self._enforce_expected_submission_format(
                submission_path=submission_path,
                format_hint=format_hint,
            )
        if compression_suffix_for(submission_path.name) is not None and (
            artifact_class == ARTIFACT_CLASS_SINGLE_FILE or not self._is_tabular_submission(submission_path)
        ):
            self._validate_non_tabular_submission_file(submission_path)
        shapefile_bundle = self._prepare_shapefile_submission_if_needed(
            submission_path=submission_path,
            format_hint=format_hint,
        )
        if shapefile_bundle is not None:
            return shapefile_bundle
        mapinfo_bundle = self._prepare_mapinfo_tab_submission_if_needed(
            submission_path=submission_path,
            format_hint=format_hint,
        )
        if mapinfo_bundle is not None:
            return mapinfo_bundle
        mapinfo_interchange_bundle = self._prepare_mapinfo_interchange_submission_if_needed(
            submission_path=submission_path,
            format_hint=format_hint,
        )
        if mapinfo_interchange_bundle is not None:
            return mapinfo_interchange_bundle
        georeferenced_raster_bundle = self._prepare_georeferenced_raster_submission_if_needed(
            submission_path=submission_path
        )
        if georeferenced_raster_bundle is not None:
            return georeferenced_raster_bundle
        vrt_bundle = self._prepare_vrt_submission_if_needed(submission_path=submission_path)
        if vrt_bundle is not None:
            return vrt_bundle
        kml_bundle = self._prepare_kml_submission_if_needed(submission_path=submission_path)
        if kml_bundle is not None:
            return kml_bundle
        envi_bundle = self._prepare_envi_header_submission_if_needed(submission_path=submission_path)
        if envi_bundle is not None:
            return envi_bundle
        metaimage_bundle = self._prepare_metaimage_submission_if_needed(submission_path=submission_path)
        if metaimage_bundle is not None:
            return metaimage_bundle
        nrrd_bundle = self._prepare_detached_nrrd_submission_if_needed(submission_path=submission_path)
        if nrrd_bundle is not None:
            return nrrd_bundle
        analyze_pair_bundle = self._prepare_analyze_pair_submission_if_needed(submission_path=submission_path)
        if analyze_pair_bundle is not None:
            return analyze_pair_bundle
        dae_bundle = self._prepare_dae_submission_if_needed(submission_path=submission_path)
        if dae_bundle is not None:
            return dae_bundle
        x3d_bundle = self._prepare_x3d_submission_if_needed(submission_path=submission_path)
        if x3d_bundle is not None:
            return x3d_bundle
        usd_bundle = self._prepare_usd_submission_if_needed(submission_path=submission_path)
        if usd_bundle is not None:
            return usd_bundle
        gltf_bundle = self._prepare_gltf_submission_if_needed(submission_path=submission_path)
        if gltf_bundle is not None:
            return gltf_bundle
        las_bundle = self._prepare_las_submission_if_needed(submission_path=submission_path)
        if las_bundle is not None:
            return las_bundle
        obj_bundle = self._prepare_obj_submission_if_needed(submission_path=submission_path)
        if obj_bundle is not None:
            return obj_bundle
        ply_bundle = self._prepare_ply_submission_if_needed(submission_path=submission_path)
        if ply_bundle is not None:
            return ply_bundle
        model_index_bundle = self._prepare_model_index_submission_if_needed(submission_path=submission_path)
        if model_index_bundle is not None:
            return model_index_bundle
        tensorflow_checkpoint_bundle = self._prepare_tensorflow_checkpoint_submission_if_needed(
            submission_path=submission_path
        )
        if tensorflow_checkpoint_bundle is not None:
            return tensorflow_checkpoint_bundle
        model_sidecar_bundle = self._prepare_model_artifact_sidecar_submission_if_needed(
            submission_path=submission_path
        )
        if model_sidecar_bundle is not None:
            return model_sidecar_bundle
        if artifact_class == ARTIFACT_CLASS_SINGLE_FILE or not self._is_tabular_submission(submission_path):
            self._validate_non_tabular_submission_file(submission_path)
            return self._enforce_expected_submission_format(
                submission_path=submission_path,
                format_hint=format_hint,
            )

        sample_path = self._resolve_sample_submission_for_submission(submission_path=submission_path)
        try:
            validate_submission(str(submission_path), str(sample_path), data_dir=self._config.data_dir)
            prepared = self._prepare_submission_path(sample_path, submission_path)
            compacted = self._maybe_compact_delimited_text_submission(sample_path, prepared)
            if compacted != prepared:
                validate_submission(str(compacted), str(sample_path), data_dir=self._config.data_dir)
            return self._finalize_prepared_tabular_submission(
                sample_path=sample_path,
                submission_path=compacted,
                format_hint=format_hint,
            )
        except SubmissionValidationError as exc:
            original = exc
            message = str(exc)
            autofixable = any(
                marker in message
                for marker in (
                    "columns mismatch",
                    "row count mismatch",
                    "id column missing",
                    "id values appear to require",
                    "missing a header row",
                    "header does not resemble",
                )
            )
            if not autofixable:
                raise
            autofixed = self._attempt_autofix_submission(sample_path=sample_path, submission_path=submission_path)
            if autofixed == submission_path:
                raise
            try:
                validate_submission(str(autofixed), str(sample_path), data_dir=self._config.data_dir)
            except SubmissionValidationError as exc2:
                raise SubmissionValidationError(
                    f"{original}\n\nAutofix wrote: {autofixed}\nBut validation still failed:\n{exc2}"
                ) from exc2
            prepared = self._prepare_submission_path(sample_path, autofixed)
            compacted = self._maybe_compact_delimited_text_submission(sample_path, prepared)
            if compacted != prepared:
                validate_submission(str(compacted), str(sample_path), data_dir=self._config.data_dir)
            return self._finalize_prepared_tabular_submission(
                sample_path=sample_path,
                submission_path=compacted,
                format_hint=format_hint,
            )

    def _resolve_sample_submission_for_submission(self, *, submission_path: Path) -> Path:
        """Resolve a sample file, preferring one that already validates the current submission."""
        primary_sample = self._resolve_sample_submission()
        candidates = [primary_sample, *find_usable_sample_submissions(self._config.data_dir)]
        deduped: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            deduped.append(candidate)
        valid_candidates: list[Path] = []
        for candidate in deduped:
            try:
                validate_submission(str(submission_path), str(candidate), data_dir=self._config.data_dir)
            except SubmissionValidationError:
                continue
            valid_candidates.append(candidate)
        if valid_candidates:
            return max(valid_candidates, key=sample_candidate_key)
        if deduped:
            return max(deduped, key=sample_candidate_key)
        return primary_sample

    @staticmethod
    def _validate_submission_input_exists(path: Path) -> None:
        if not path.exists():
            raise SubmissionValidationError(f"submission file not found: {path}")

    @staticmethod
    def _is_zip_submission(path: Path) -> bool:
        return SubmissionService._submission_format_suffix(path) == _ZIP_SUBMISSION_SUFFIX

    @staticmethod
    def _is_tar_submission(path: Path) -> bool:
        return SubmissionService._submission_format_suffix(path) in _TAR_SUBMISSION_SUFFIXES

    @staticmethod
    def _is_external_archive_submission(path: Path) -> bool:
        return SubmissionService._submission_format_suffix(path) in _EXTERNAL_ARCHIVE_SUBMISSION_SUFFIXES

    @staticmethod
    def _is_tabular_submission(path: Path) -> bool:
        return tabular_suffix(path) in _TABULAR_SUBMISSION_SUFFIXES

    @staticmethod
    def _submission_format_suffix(path: Path) -> str:
        artifact_candidate = artifact_suffix(path)
        if artifact_candidate in _ARCHIVE_SUBMISSION_SUFFIXES:
            return artifact_candidate
        if artifact_candidate in _NON_TABULAR_SINGLE_FILE_SUFFIXES:
            return artifact_candidate
        tabular_candidate = tabular_suffix(path)
        if tabular_candidate in _TABULAR_SUBMISSION_SUFFIXES:
            return tabular_candidate
        return path.suffix.lower()

    @staticmethod
    def _validate_non_tabular_submission_file(path: Path) -> None:
        if not path.is_file():
            raise SubmissionValidationError(f"submission path is not a file: {path}")
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise SubmissionValidationError(f"unable to read submission file metadata: {path}") from exc
        if size <= 0:
            raise SubmissionValidationError(f"submission file is empty: {path}")
        if SubmissionService._submission_format_suffix(path) in SQLITE_TABULAR_SUFFIXES:
            SubmissionService._validate_sqlite_submission_file(path)
            return
        compression_suffix = compression_suffix_for(path.name)
        if compression_suffix is None:
            return
        try:
            has_payload = SubmissionService._compressed_submission_payload_has_bytes(
                path=path,
                compression_suffix=compression_suffix,
            )
        except (OSError, EOFError, lzma.LZMAError, zstd.ZstdError) as exc:
            raise SubmissionValidationError(f"unable to read compressed submission file: {path}") from exc
        if not has_payload:
            raise SubmissionValidationError(f"compressed submission payload is empty: {path}")

    @staticmethod
    def _validate_sqlite_submission_file(path: Path) -> None:
        try:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
                tables = SubmissionService._sqlite_user_tables(conn)
                if not tables:
                    raise SubmissionValidationError(f"SQLite submission has no user tables or views: {path}")
                for table in tables:
                    row = conn.execute(
                        f"SELECT 1 FROM {SubmissionService._quote_sqlite_identifier(table)} LIMIT 1"
                    ).fetchone()
                    if row is not None:
                        return
        except SubmissionValidationError:
            raise
        except sqlite3.DatabaseError as exc:
            raise SubmissionValidationError(f"unable to read SQLite submission file: {path}") from exc
        raise SubmissionValidationError(f"SQLite submission has no data rows: {path}")

    @staticmethod
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

    @staticmethod
    def _quote_sqlite_identifier(identifier: str) -> str:
        return '"' + str(identifier).replace('"', '""') + '"'

    @staticmethod
    def _compressed_submission_payload_has_bytes(*, path: Path, compression_suffix: str) -> bool:
        if compression_suffix == ".gz":
            with gzip.open(path, "rb") as handle:
                return bool(handle.read(1))
        if compression_suffix == ".bz2":
            with bz2.open(path, "rb") as handle:
                return bool(handle.read(1))
        if compression_suffix == ".xz":
            with lzma.open(path, "rb") as handle:
                return bool(handle.read(1))
        if compression_suffix == ".zst":
            with path.open("rb") as raw:
                with zstd.ZstdDecompressor().stream_reader(raw) as reader:
                    return bool(reader.read(1))
        return True

    def _prepare_shapefile_submission_if_needed(
        self,
        *,
        submission_path: Path,
        format_hint: SubmissionFormatHint | None,
    ) -> Path | None:
        suffix = self._submission_format_suffix(submission_path)
        expected_suffixes = set(self._expected_submission_suffixes(format_hint))
        if suffix not in SHAPEFILE_COMPONENT_SUFFIXES:
            return None
        if suffix != ".shp" and ".shp" not in expected_suffixes:
            return None
        members = shapefile_bundle_members(submission_path)
        present_suffixes = {shapefile_component_suffix(member) for member in members}
        missing = sorted(SHAPEFILE_REQUIRED_SUFFIXES - present_suffixes)
        if missing:
            missing_display = ", ".join(missing)
            raise SubmissionValidationError(
                "shapefile submission is missing required sidecar files:\n"
                f"  missing: {missing_display}\n"
                f"  file:    {submission_path}"
            )
        for member in members:
            self._validate_non_tabular_submission_file(member)
        prepared = self._build_submission_zip(submission_path, members=members)
        self._validate_zip_submission(prepared)
        return prepared

    def _prepare_mapinfo_tab_submission_if_needed(
        self,
        *,
        submission_path: Path,
        format_hint: SubmissionFormatHint | None,
    ) -> Path | None:
        if not is_mapinfo_tab_artifact(submission_path):
            return None
        members = mapinfo_bundle_members(submission_path)
        present_suffixes = {mapinfo_component_suffix(member) for member in members}
        has_sidecars = bool(present_suffixes - {".tab"})
        expects_mapinfo = self._context_mentions_mapinfo_tab(format_hint)
        if not has_sidecars and not expects_mapinfo:
            return None
        missing = missing_mapinfo_sidecars(submission_path)
        if missing:
            missing_display = ", ".join(missing)
            raise SubmissionValidationError(
                "MapInfo TAB submission is missing required sidecar files:\n"
                f"  missing: {missing_display}\n"
                f"  file:    {submission_path}"
            )
        for member in members:
            self._validate_non_tabular_submission_file(member)
        prepared = self._build_submission_zip(submission_path, members=members)
        self._validate_zip_submission(prepared)
        return prepared

    def _prepare_mapinfo_interchange_submission_if_needed(
        self,
        *,
        submission_path: Path,
        format_hint: SubmissionFormatHint | None,
    ) -> Path | None:
        if not is_mapinfo_interchange_artifact(submission_path):
            return None
        sidecar_specs = mapinfo_interchange_sidecar_specs(submission_path)
        missing = missing_mapinfo_interchange_sidecars(submission_path)
        if missing and self._context_mentions_mapinfo_interchange(format_hint):
            missing_display = ", ".join(missing)
            raise SubmissionValidationError(
                "MapInfo MIF/MID submission is missing required sidecar files:\n"
                f"  missing: {missing_display}\n"
                f"  file:    {submission_path}"
            )
        if not sidecar_specs:
            return None
        members = [submission_path, *(path for path, _archive_name in sidecar_specs)]
        for member in members:
            self._validate_non_tabular_submission_file(member)
        archive_names = [submission_path.name, *(archive_name for _path, archive_name in sidecar_specs)]
        prepared = self._build_submission_zip(
            submission_path,
            members=members,
            member_archive_names=archive_names,
        )
        self._validate_zip_submission(prepared)
        return prepared

    def _prepare_georeferenced_raster_submission_if_needed(self, *, submission_path: Path) -> Path | None:
        sidecar_specs = georeferenced_raster_sidecar_specs(submission_path)
        if not sidecar_specs:
            return None
        members = [submission_path, *(path for path, _archive_name in sidecar_specs)]
        for member in members:
            self._validate_non_tabular_submission_file(member)
        archive_names = [submission_path.name, *(archive_name for _path, archive_name in sidecar_specs)]
        prepared = self._build_submission_zip(
            submission_path,
            members=members,
            member_archive_names=archive_names,
        )
        self._validate_zip_submission(prepared)
        return prepared

    def _prepare_vrt_submission_if_needed(self, *, submission_path: Path) -> Path | None:
        vrt_root = vrt_artifact_root_dir(submission_path)
        invalid = invalid_vrt_sidecar_names(submission_path, root_dir=vrt_root)
        if invalid:
            invalid_display = ", ".join(invalid)
            raise SubmissionValidationError(
                "GDAL VRT submission references unsafe source paths:\n"
                f"  invalid: {invalid_display}\n"
                f"  file:    {submission_path}"
            )
        sidecar_specs = vrt_sidecar_specs(submission_path, root_dir=vrt_root)
        missing = missing_vrt_sidecars(submission_path, root_dir=vrt_root)
        if missing:
            missing_display = ", ".join(missing)
            raise SubmissionValidationError(
                "GDAL VRT submission is missing referenced source files:\n"
                f"  missing: {missing_display}\n"
                f"  file:    {submission_path}"
            )
        if not sidecar_specs:
            return None
        members = [submission_path, *(path for path, _archive_name in sidecar_specs)]
        for member in members:
            self._validate_non_tabular_submission_file(member)
        archive_names = [
            vrt_primary_archive_name(submission_path, root_dir=vrt_root),
            *(archive_name for _path, archive_name in sidecar_specs),
        ]
        prepared = self._build_submission_zip(
            submission_path,
            members=members,
            member_archive_names=archive_names,
        )
        self._validate_zip_submission(prepared)
        return prepared

    def _prepare_kml_submission_if_needed(self, *, submission_path: Path) -> Path | None:
        if not is_kml_artifact(submission_path):
            return None
        kml_root = kml_artifact_root_dir(submission_path)
        invalid = invalid_kml_sidecar_names(submission_path, root_dir=kml_root)
        if invalid:
            invalid_display = ", ".join(invalid)
            raise SubmissionValidationError(
                "KML submission references unsafe href paths:\n"
                f"  invalid: {invalid_display}\n"
                f"  file:    {submission_path}"
            )
        sidecar_specs = kml_sidecar_specs(submission_path, root_dir=kml_root)
        missing = missing_kml_sidecars(submission_path, root_dir=kml_root)
        if missing:
            missing_display = ", ".join(missing)
            raise SubmissionValidationError(
                "KML submission is missing referenced href sidecars:\n"
                f"  missing: {missing_display}\n"
                f"  file:    {submission_path}"
            )
        if not sidecar_specs:
            return None
        members = [submission_path, *(path for path, _archive_name in sidecar_specs)]
        for member in members:
            self._validate_non_tabular_submission_file(member)
        archive_names = [
            kml_primary_archive_name(submission_path, root_dir=kml_root),
            *(archive_name for _path, archive_name in sidecar_specs),
        ]
        prepared = self._build_submission_zip(
            submission_path,
            members=members,
            member_archive_names=archive_names,
        )
        self._validate_zip_submission(prepared)
        return prepared

    def _prepare_envi_header_submission_if_needed(self, *, submission_path: Path) -> Path | None:
        if not is_envi_header_artifact(submission_path):
            return None
        invalid = invalid_envi_sidecar_names(submission_path)
        if invalid:
            invalid_display = ", ".join(invalid)
            raise SubmissionValidationError(
                "ENVI header submission references unsafe data file paths:\n"
                f"  invalid: {invalid_display}\n"
                f"  file:    {submission_path}"
            )
        sidecar_specs = envi_sidecar_specs(submission_path)
        missing = missing_envi_sidecars(submission_path)
        if missing:
            missing_display = ", ".join(missing)
            raise SubmissionValidationError(
                "ENVI header submission is missing referenced data sidecars:\n"
                f"  missing: {missing_display}\n"
                f"  file:    {submission_path}"
            )
        if not sidecar_specs:
            return None
        members = [submission_path, *(path for path, _archive_name in sidecar_specs)]
        for member in members:
            self._validate_non_tabular_submission_file(member)
        archive_names = [submission_path.name, *(archive_name for _path, archive_name in sidecar_specs)]
        prepared = self._build_submission_zip(
            submission_path,
            members=members,
            member_archive_names=archive_names,
        )
        self._validate_zip_submission(prepared)
        return prepared

    def _prepare_metaimage_submission_if_needed(self, *, submission_path: Path) -> Path | None:
        if not is_metaimage_header_artifact(submission_path):
            return None
        invalid = invalid_metaimage_sidecar_names(submission_path)
        if invalid:
            invalid_display = ", ".join(invalid)
            raise SubmissionValidationError(
                "MetaImage submission references unsafe ElementDataFile paths:\n"
                f"  invalid: {invalid_display}\n"
                f"  file:    {submission_path}"
            )
        sidecar_specs = metaimage_sidecar_specs(submission_path)
        missing = missing_metaimage_sidecars(submission_path)
        if missing:
            missing_display = ", ".join(missing)
            raise SubmissionValidationError(
                "MetaImage submission is missing referenced ElementDataFile sidecars:\n"
                f"  missing: {missing_display}\n"
                f"  file:    {submission_path}"
            )
        if not sidecar_specs:
            return None
        members = [submission_path, *(path for path, _archive_name in sidecar_specs)]
        for member in members:
            self._validate_non_tabular_submission_file(member)
        archive_names = [submission_path.name, *(archive_name for _path, archive_name in sidecar_specs)]
        prepared = self._build_submission_zip(
            submission_path,
            members=members,
            member_archive_names=archive_names,
        )
        self._validate_zip_submission(prepared)
        return prepared

    def _prepare_detached_nrrd_submission_if_needed(self, *, submission_path: Path) -> Path | None:
        if not is_detached_nrrd_header_artifact(submission_path):
            return None
        invalid = invalid_detached_nrrd_sidecar_names(submission_path)
        if invalid:
            invalid_display = ", ".join(invalid)
            raise SubmissionValidationError(
                "detached NRRD submission references unsafe data file paths:\n"
                f"  invalid: {invalid_display}\n"
                f"  file:    {submission_path}"
            )
        sidecar_specs = detached_nrrd_sidecar_specs(submission_path)
        missing = missing_detached_nrrd_sidecars(submission_path)
        if missing:
            missing_display = ", ".join(missing)
            raise SubmissionValidationError(
                "detached NRRD submission is missing referenced data file sidecars:\n"
                f"  missing: {missing_display}\n"
                f"  file:    {submission_path}"
            )
        if not sidecar_specs:
            return None
        members = [submission_path, *(path for path, _archive_name in sidecar_specs)]
        for member in members:
            self._validate_non_tabular_submission_file(member)
        archive_names = [submission_path.name, *(archive_name for _path, archive_name in sidecar_specs)]
        prepared = self._build_submission_zip(
            submission_path,
            members=members,
            member_archive_names=archive_names,
        )
        self._validate_zip_submission(prepared)
        return prepared

    def _prepare_analyze_pair_submission_if_needed(self, *, submission_path: Path) -> Path | None:
        if not is_analyze_pair_artifact(submission_path):
            return None
        sidecar_specs = analyze_pair_sidecar_specs(submission_path)
        missing = missing_analyze_pair_sidecars(submission_path)
        if missing:
            missing_display = ", ".join(missing)
            raise SubmissionValidationError(
                "Analyze/NIfTI pair submission is missing required pair sidecars:\n"
                f"  missing: {missing_display}\n"
                f"  file:    {submission_path}"
            )
        if not sidecar_specs:
            return None
        members = [submission_path, *(path for path, _archive_name in sidecar_specs)]
        for member in members:
            self._validate_non_tabular_submission_file(member)
        archive_names = [submission_path.name, *(archive_name for _path, archive_name in sidecar_specs)]
        prepared = self._build_submission_zip(
            submission_path,
            members=members,
            member_archive_names=archive_names,
        )
        self._validate_zip_submission(prepared)
        return prepared

    def _prepare_dae_submission_if_needed(self, *, submission_path: Path) -> Path | None:
        if not is_dae_artifact(submission_path):
            return None
        dae_root = dae_artifact_root_dir(submission_path)
        invalid = invalid_dae_sidecar_names(submission_path, root_dir=dae_root)
        if invalid:
            invalid_display = ", ".join(invalid)
            raise SubmissionValidationError(
                "COLLADA submission references unsafe external URI paths:\n"
                f"  invalid: {invalid_display}\n"
                f"  file:    {submission_path}"
            )
        sidecar_specs = dae_sidecar_specs(submission_path, root_dir=dae_root)
        missing = missing_dae_sidecars(submission_path, root_dir=dae_root)
        if missing:
            missing_display = ", ".join(missing)
            raise SubmissionValidationError(
                "COLLADA submission is missing referenced external URI sidecars:\n"
                f"  missing: {missing_display}\n"
                f"  file:    {submission_path}"
            )
        if not sidecar_specs:
            return None
        members = [submission_path, *(path for path, _archive_name in sidecar_specs)]
        for member in members:
            self._validate_non_tabular_submission_file(member)
        archive_names = [
            dae_primary_archive_name(submission_path, root_dir=dae_root),
            *(archive_name for _path, archive_name in sidecar_specs),
        ]
        prepared = self._build_submission_zip(
            submission_path,
            members=members,
            member_archive_names=archive_names,
        )
        self._validate_zip_submission(prepared)
        return prepared

    def _prepare_x3d_submission_if_needed(self, *, submission_path: Path) -> Path | None:
        if not is_x3d_artifact(submission_path):
            return None
        x3d_root = x3d_artifact_root_dir(submission_path)
        invalid = invalid_x3d_sidecar_names(submission_path, root_dir=x3d_root)
        if invalid:
            invalid_display = ", ".join(invalid)
            raise SubmissionValidationError(
                "X3D submission references unsafe URL paths:\n"
                f"  invalid: {invalid_display}\n"
                f"  file:    {submission_path}"
            )
        sidecar_specs = x3d_sidecar_specs(submission_path, root_dir=x3d_root)
        missing = missing_x3d_sidecars(submission_path, root_dir=x3d_root)
        if missing:
            missing_display = ", ".join(missing)
            raise SubmissionValidationError(
                "X3D submission is missing referenced URL sidecars:\n"
                f"  missing: {missing_display}\n"
                f"  file:    {submission_path}"
            )
        if not sidecar_specs:
            return None
        members = [submission_path, *(path for path, _archive_name in sidecar_specs)]
        for member in members:
            self._validate_non_tabular_submission_file(member)
        archive_names = [
            x3d_primary_archive_name(submission_path, root_dir=x3d_root),
            *(archive_name for _path, archive_name in sidecar_specs),
        ]
        prepared = self._build_submission_zip(
            submission_path,
            members=members,
            member_archive_names=archive_names,
        )
        self._validate_zip_submission(prepared)
        return prepared

    def _prepare_usd_submission_if_needed(self, *, submission_path: Path) -> Path | None:
        if not is_usd_artifact(submission_path):
            return None
        usd_root = usd_artifact_root_dir(submission_path)
        invalid = invalid_usd_sidecar_names(submission_path, root_dir=usd_root)
        if invalid:
            invalid_display = ", ".join(invalid)
            raise SubmissionValidationError(
                "USD submission references unsafe asset sidecars:\n"
                f"  invalid: {invalid_display}\n"
                f"  file:    {submission_path}"
            )
        sidecar_specs = usd_sidecar_specs(submission_path, root_dir=usd_root)
        missing = missing_usd_sidecars(submission_path, root_dir=usd_root)
        if missing:
            missing_display = ", ".join(missing)
            raise SubmissionValidationError(
                "USD submission is missing referenced asset sidecars:\n"
                f"  missing: {missing_display}\n"
                f"  file:    {submission_path}"
            )
        if not sidecar_specs:
            return None
        members = [submission_path, *(path for path, _archive_name in sidecar_specs)]
        for member in members:
            self._validate_non_tabular_submission_file(member)
        archive_names = [
            usd_primary_archive_name(submission_path, root_dir=usd_root),
            *(archive_name for _path, archive_name in sidecar_specs),
        ]
        prepared = self._build_submission_zip(
            submission_path,
            members=members,
            member_archive_names=archive_names,
        )
        self._validate_zip_submission(prepared)
        return prepared

    def _prepare_gltf_submission_if_needed(self, *, submission_path: Path) -> Path | None:
        if not is_gltf_artifact(submission_path):
            return None
        gltf_root = gltf_artifact_root_dir(submission_path)
        invalid = invalid_gltf_sidecar_names(submission_path, root_dir=gltf_root)
        if invalid:
            invalid_display = ", ".join(invalid)
            raise SubmissionValidationError(
                "glTF submission references unsafe external URI paths:\n"
                f"  invalid: {invalid_display}\n"
                f"  file:    {submission_path}"
            )
        sidecar_specs = gltf_sidecar_specs(submission_path, root_dir=gltf_root)
        missing = missing_gltf_sidecars(submission_path, root_dir=gltf_root)
        if missing:
            missing_display = ", ".join(missing)
            raise SubmissionValidationError(
                "glTF submission is missing referenced external URI sidecars:\n"
                f"  missing: {missing_display}\n"
                f"  file:    {submission_path}"
            )
        if not sidecar_specs:
            return None
        members = [submission_path, *(path for path, _archive_name in sidecar_specs)]
        for member in members:
            self._validate_non_tabular_submission_file(member)
        archive_names = [
            gltf_primary_archive_name(submission_path, root_dir=gltf_root),
            *(archive_name for _path, archive_name in sidecar_specs),
        ]
        prepared = self._build_submission_zip(
            submission_path,
            members=members,
            member_archive_names=archive_names,
        )
        self._validate_zip_submission(prepared)
        return prepared

    def _prepare_obj_submission_if_needed(self, *, submission_path: Path) -> Path | None:
        if not is_obj_artifact(submission_path):
            return None
        invalid = invalid_obj_sidecar_names(submission_path)
        if invalid:
            invalid_display = ", ".join(invalid)
            raise SubmissionValidationError(
                "OBJ submission references unsafe material or texture paths:\n"
                f"  invalid: {invalid_display}\n"
                f"  file:    {submission_path}"
            )
        sidecar_specs = obj_sidecar_specs(submission_path)
        missing = missing_obj_sidecars(submission_path)
        if missing:
            missing_display = ", ".join(missing)
            raise SubmissionValidationError(
                "OBJ submission is missing referenced material or texture sidecars:\n"
                f"  missing: {missing_display}\n"
                f"  file:    {submission_path}"
            )
        if not sidecar_specs:
            return None
        members = [submission_path, *(path for path, _archive_name in sidecar_specs)]
        for member in members:
            self._validate_non_tabular_submission_file(member)
        archive_names = [submission_path.name, *(archive_name for _path, archive_name in sidecar_specs)]
        prepared = self._build_submission_zip(
            submission_path,
            members=members,
            member_archive_names=archive_names,
        )
        self._validate_zip_submission(prepared)
        return prepared

    def _prepare_las_submission_if_needed(self, *, submission_path: Path) -> Path | None:
        if not is_las_artifact(submission_path):
            return None
        sidecar_specs = las_sidecar_specs(submission_path)
        if not sidecar_specs:
            return None
        members = [submission_path, *(path for path, _archive_name in sidecar_specs)]
        for member in members:
            self._validate_non_tabular_submission_file(member)
        archive_names = [submission_path.name, *(archive_name for _path, archive_name in sidecar_specs)]
        prepared = self._build_submission_zip(
            submission_path,
            members=members,
            member_archive_names=archive_names,
        )
        self._validate_zip_submission(prepared)
        return prepared

    def _prepare_ply_submission_if_needed(self, *, submission_path: Path) -> Path | None:
        if not is_ply_artifact(submission_path):
            return None
        invalid = invalid_ply_sidecar_names(submission_path)
        if invalid:
            invalid_display = ", ".join(invalid)
            raise SubmissionValidationError(
                "PLY submission references unsafe TextureFile paths:\n"
                f"  invalid: {invalid_display}\n"
                f"  file:    {submission_path}"
            )
        sidecar_specs = ply_sidecar_specs(submission_path)
        missing = missing_ply_sidecars(submission_path)
        if missing:
            missing_display = ", ".join(missing)
            raise SubmissionValidationError(
                "PLY submission is missing referenced TextureFile sidecars:\n"
                f"  missing: {missing_display}\n"
                f"  file:    {submission_path}"
            )
        if not sidecar_specs:
            return None
        members = [submission_path, *(path for path, _archive_name in sidecar_specs)]
        for member in members:
            self._validate_non_tabular_submission_file(member)
        archive_names = [submission_path.name, *(archive_name for _path, archive_name in sidecar_specs)]
        prepared = self._build_submission_zip(
            submission_path,
            members=members,
            member_archive_names=archive_names,
        )
        self._validate_zip_submission(prepared)
        return prepared

    def _prepare_model_index_submission_if_needed(self, *, submission_path: Path) -> Path | None:
        if not is_model_index_artifact(submission_path):
            return None
        invalid = invalid_model_index_shard_names(submission_path)
        if invalid:
            invalid_display = ", ".join(invalid)
            raise SubmissionValidationError(
                "model index submission references unsafe shard paths:\n"
                f"  invalid: {invalid_display}\n"
                f"  file:    {submission_path}"
            )
        shard_specs = model_index_shard_specs(submission_path)
        missing = missing_model_index_shards(submission_path)
        if missing:
            missing_display = ", ".join(missing)
            raise SubmissionValidationError(
                "model index submission is missing referenced shard files:\n"
                f"  missing: {missing_display}\n"
                f"  file:    {submission_path}"
            )
        if not shard_specs:
            return None
        members = [submission_path, *(path for path, _archive_name in shard_specs)]
        for member in members:
            self._validate_non_tabular_submission_file(member)
        archive_names = [submission_path.name, *(archive_name for _path, archive_name in shard_specs)]
        prepared = self._build_submission_zip(
            submission_path,
            members=members,
            member_archive_names=archive_names,
        )
        self._validate_zip_submission(prepared)
        return prepared

    def _prepare_tensorflow_checkpoint_submission_if_needed(self, *, submission_path: Path) -> Path | None:
        if not is_tensorflow_checkpoint_artifact(submission_path):
            return None
        sidecar_specs = tensorflow_checkpoint_sidecar_specs(submission_path)
        if not sidecar_specs:
            return None
        members = [submission_path, *(path for path, _archive_name in sidecar_specs)]
        for member in members:
            self._validate_non_tabular_submission_file(member)
        archive_names = [submission_path.name, *(archive_name for _path, archive_name in sidecar_specs)]
        prepared = self._build_submission_zip(
            submission_path,
            members=members,
            member_archive_names=archive_names,
        )
        self._validate_zip_submission(prepared)
        return prepared

    def _prepare_model_artifact_sidecar_submission_if_needed(self, *, submission_path: Path) -> Path | None:
        sidecar_specs = model_artifact_sidecar_specs(submission_path)
        if not sidecar_specs:
            return None
        members = [submission_path, *(path for path, _archive_name in sidecar_specs)]
        for member in members:
            self._validate_non_tabular_submission_file(member)
        archive_names = [submission_path.name, *(archive_name for _path, archive_name in sidecar_specs)]
        prepared = self._build_submission_zip(
            submission_path,
            members=members,
            member_archive_names=archive_names,
        )
        self._validate_zip_submission(prepared)
        return prepared

    @staticmethod
    def _is_directory_single_file_submission(path: Path, artifact_class: str) -> bool:
        return artifact_class == ARTIFACT_CLASS_SINGLE_FILE and (
            artifact_suffix(path) in DIRECTORY_ARRAY_SUFFIXES or is_model_directory_artifact(path)
        )

    @staticmethod
    def _validate_zip_submission(path: Path) -> None:
        try:
            with zipfile.ZipFile(path, "r") as archive:
                members = [info for info in archive.infolist() if not info.is_dir()]
                has_non_empty_file = any(info.file_size > 0 for info in members)
                seen_names: set[str] = set()
                for info in archive.infolist():
                    member_name = SubmissionService._safe_archive_member_name(info.filename)
                    if info.flag_bits & 0x1:
                        raise SubmissionValidationError(
                            f"submission zip archive has unsupported encrypted member: {member_name}"
                        )
                    if SubmissionService._zip_member_is_symlink(info):
                        raise SubmissionValidationError(
                            f"submission zip archive has unsupported symlink member: {member_name}"
                        )
                    if info.is_dir():
                        member_name = f"{member_name.rstrip('/')}/"
                    if member_name in seen_names:
                        raise SubmissionValidationError(f"submission zip duplicate archive member name: {member_name}")
                    seen_names.add(member_name)
        except (OSError, zipfile.BadZipFile) as exc:
            raise SubmissionValidationError(f"submission zip is invalid: {path}") from exc
        if not members:
            raise SubmissionValidationError(f"submission zip has no files: {path}")
        if not has_non_empty_file:
            raise SubmissionValidationError(f"submission zip has no non-empty files: {path}")

    @classmethod
    def _validate_tar_submission(cls, path: Path, *, require_code_contract: bool = True) -> None:
        cls._validate_non_tabular_submission_file(path)
        try:
            with cls._open_tar_for_read(path) as archive:
                members = archive.getmembers()
        except (OSError, tarfile.TarError, zstd.ZstdError) as exc:
            raise SubmissionValidationError(f"submission tar archive is invalid: {path}") from exc
        seen_names: set[str] = set()
        for info in members:
            member_name = cls._safe_archive_member_name(info.name)
            if member_name in seen_names:
                raise SubmissionValidationError(f"submission tar duplicate archive member name: {member_name}")
            seen_names.add(member_name)
            if info.issym() or info.islnk():
                raise SubmissionValidationError(f"submission tar archive has unsupported link member: {info.name}")
            if not info.isfile() and not info.isdir():
                raise SubmissionValidationError(f"submission tar archive has unsupported member type: {info.name}")
        if not any(info.isfile() for info in members):
            raise SubmissionValidationError(f"submission tar archive has no files: {path}")
        if not any(info.isfile() and info.size > 0 for info in members):
            raise SubmissionValidationError(f"submission tar archive has no non-empty files: {path}")
        if not require_code_contract:
            return
        file_names = {
            cls._normalize_tar_member_name(info.name)
            for info in members
            if info.isfile() and cls._normalize_tar_member_name(info.name)
        }
        missing = sorted(required for required in ("deck.csv", "main.py") if required not in file_names)
        if missing:
            raise SubmissionValidationError(
                f"submission tar archive missing required top-level files: {', '.join(missing)}"
            )
        if not any(name.startswith("cg/") for name in file_names):
            raise SubmissionValidationError("submission tar archive missing required cg/ members")

    @classmethod
    def _validate_external_archive_submission(cls, path: Path) -> None:
        suffix = cls._submission_format_suffix(path)
        if suffix == ".7z":
            cls._validate_7z_submission(path)
            return
        if suffix == ".rar":
            cls._validate_rar_submission(path)
            return
        raise SubmissionValidationError(f"unsupported external submission archive suffix: {suffix}")

    @classmethod
    def _validate_7z_submission(cls, path: Path) -> None:
        cls._validate_non_tabular_submission_file(path)
        try:
            with py7zr.SevenZipFile(path, mode="r") as archive:
                if archive.needs_password():
                    raise SubmissionValidationError(f"submission 7z archive requires a password: {path}")
                members = archive.list()
        except (OSError, py7zr.Bad7zFile, py7zr.DecompressionError, py7zr.UnsupportedCompressionMethodError) as exc:
            raise SubmissionValidationError(f"submission 7z archive is invalid: {path}") from exc
        cls._validate_external_archive_members(
            path=path,
            archive_kind="7z",
            members=[
                (
                    getattr(member, "filename", ""),
                    bool(getattr(member, "is_directory", False)),
                    bool(getattr(member, "is_file", False)) and not bool(getattr(member, "is_symlink", False)),
                    cls._archive_member_uncompressed_size(member),
                )
                for member in members
            ],
        )

    @classmethod
    def _validate_rar_submission(cls, path: Path) -> None:
        cls._validate_non_tabular_submission_file(path)
        try:
            with rarfile.RarFile(path) as archive:
                members = archive.infolist()
        except (
            OSError,
            rarfile.BadRarFile,
            rarfile.NotRarFile,
            rarfile.RarCannotExec,
            rarfile.RarOpenError,
            rarfile.RarWarning,
            rarfile.Error,
        ) as exc:
            raise SubmissionValidationError(f"submission rar archive is invalid: {path}") from exc
        cls._validate_external_archive_members(
            path=path,
            archive_kind="rar",
            members=[
                (
                    getattr(member, "filename", ""),
                    bool(member.is_dir()),
                    bool(member.is_file()) and not bool(member.is_symlink()) and not bool(member.needs_password()),
                    cls._archive_member_uncompressed_size(member),
                )
                for member in members
            ],
        )

    @classmethod
    def _validate_external_archive_members(
        cls,
        *,
        path: Path,
        archive_kind: str,
        members: list[tuple[str, bool, bool, int | None]],
    ) -> None:
        seen_names: set[str] = set()
        has_file = False
        has_non_empty_file = False
        file_sizes_known = True
        for raw_name, is_dir, is_file, file_size in members:
            member_name = cls._safe_archive_member_name(raw_name)
            if is_dir:
                member_name = f"{member_name.rstrip('/')}/"
            if member_name in seen_names:
                raise SubmissionValidationError(
                    f"submission {archive_kind} duplicate archive member name: {member_name}"
                )
            seen_names.add(member_name)
            if is_file:
                has_file = True
                if file_size is None:
                    file_sizes_known = False
                elif file_size > 0:
                    has_non_empty_file = True
                continue
            if not is_dir:
                raise SubmissionValidationError(
                    f"submission {archive_kind} archive has unsupported member type: {raw_name}"
                )
        if not has_file:
            raise SubmissionValidationError(f"submission {archive_kind} archive has no files: {path}")
        if file_sizes_known and not has_non_empty_file:
            raise SubmissionValidationError(f"submission {archive_kind} archive has no non-empty files: {path}")

    @staticmethod
    def _archive_member_uncompressed_size(member: object) -> int | None:
        for attr_name in ("file_size", "uncompressed", "uncompressed_size", "size", "original_size"):
            value = getattr(member, attr_name, None)
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                return value
        return None

    @staticmethod
    @contextmanager
    def _open_tar_for_read(path: Path):
        suffix = SubmissionService._submission_format_suffix(path)
        if suffix in ZSTD_TAR_ARCHIVE_SUBMISSION_SUFFIXES:
            with open_zstd_tar(path) as archive:
                members = [member for member in archive]
            with _TarMembersSnapshot(members) as snapshot:
                yield snapshot
            return
        with tarfile.open(path, "r:*") as archive:
            yield archive

    @staticmethod
    def _normalize_tar_member_name(name: str) -> str:
        normalized = str(name or "").replace("\\", "/").lstrip("/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized

    @staticmethod
    def _zip_member_is_symlink(info: zipfile.ZipInfo) -> bool:
        if info.create_system != 3:
            return False
        return stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK

    @staticmethod
    def _safe_archive_member_name(name: str) -> str:
        normalized = str(name or "").replace("\\", "/")
        if not normalized.strip():
            raise SubmissionValidationError("submission archive contains an empty member name")
        if normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized):
            raise SubmissionValidationError(f"submission archive member has unsafe absolute path: {name}")
        parts = PurePosixPath(normalized).parts
        if ".." in parts:
            raise SubmissionValidationError(f"submission archive member has unsafe path traversal: {name}")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        if not normalized or normalized == ".":
            raise SubmissionValidationError("submission archive contains an empty member name")
        return normalized

    def _finalize_prepared_tabular_submission(
        self,
        *,
        sample_path: Path,
        submission_path: Path,
        format_hint: SubmissionFormatHint | None,
    ) -> Path:
        """Apply format constraints and re-validate the final prepared submission."""
        finalized = self._enforce_expected_submission_format(
            submission_path=submission_path,
            format_hint=format_hint,
        )
        if self._is_zip_submission(finalized):
            self._validate_zip_submission(finalized)
            return finalized
        if self._is_tar_submission(finalized):
            self._validate_tar_submission(
                finalized,
                require_code_contract=self._tar_code_contract_required(format_hint),
            )
            return finalized
        if self._is_tabular_submission(finalized):
            validate_submission(str(finalized), str(sample_path), data_dir=self._config.data_dir)
            return finalized
        self._validate_non_tabular_submission_file(finalized)
        return finalized

    def _resolve_submission_format_hint(self) -> SubmissionFormatHint | None:
        """Load submission format hints from context files."""
        for context_dir in self._candidate_context_dirs():
            format_hint = load_submission_format_hint(context_dir / "submission_format.md")
            if format_hint is not None and self._hint_has_any_signal(format_hint):
                return format_hint
            for name in ("overview.md", "data.md", "rules.md", "discussion.md"):
                path = context_dir / name
                if not path.exists():
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                section = extract_submission_section(text) or ""
                if not section.strip():
                    continue
                hint = parse_submission_format(section)
                if self._hint_has_any_signal(hint):
                    return hint
        return None

    @staticmethod
    def _hint_has_any_signal(hint: SubmissionFormatHint) -> bool:
        """Return whether a parsed hint has usable signals."""
        return bool(hint.columns or hint.delimiter or hint.expected_suffixes or hint.artifact_class)

    def _candidate_context_dirs(self) -> list[Path]:
        """Discover possible context directories that may contain submission hints."""
        candidates: list[Path] = []
        seen: set[Path] = set()

        def add(path: Path) -> None:
            resolved = path.resolve()
            if resolved in seen:
                return
            seen.add(resolved)
            candidates.append(resolved)

        sample_parent = self._config.sample_submission_path.parent
        data_dir = self._config.data_dir
        add(sample_parent)
        add(data_dir)
        for root in [sample_parent, *sample_parent.parents]:
            add(root / "context")
        for root in [data_dir, *data_dir.parents]:
            add(root / "context")
        return candidates

    def _enforce_expected_submission_format(
        self,
        *,
        submission_path: Path,
        format_hint: SubmissionFormatHint | None,
    ) -> Path:
        """Coerce submission artifact into the expected file format when inferable."""
        expected_suffixes = self._expected_submission_suffixes(format_hint)
        if not expected_suffixes:
            return submission_path
        current_suffix = self._submission_format_suffix(submission_path)
        if current_suffix in expected_suffixes:
            return submission_path
        preferred_suffix = expected_suffixes[0]
        converted = self._convert_submission_to_suffix(
            submission_path=submission_path,
            target_suffix=preferred_suffix,
            format_hint=format_hint,
        )
        if converted is not None:
            return converted
        expected_display = ", ".join(expected_suffixes)
        actual_display = current_suffix or "<no extension>"
        raise SubmissionValidationError(
            "submission file format mismatch:\n"
            f"  expected one of: {expected_display}\n"
            f"  actual:          {actual_display}\n"
            f"  file:            {submission_path}"
        )

    def _resolve_submission_artifact_class(
        self,
        *,
        format_hint: SubmissionFormatHint | None,
        submission_path: Path,
        manifest_artifact_class: str | None = None,
    ) -> str:
        manifest_class = normalize_artifact_class(manifest_artifact_class, default="")
        if manifest_class:
            return manifest_class
        if submission_path.is_dir() and is_model_directory_artifact(submission_path):
            return ARTIFACT_CLASS_SINGLE_FILE
        hint_class = normalize_artifact_class(getattr(format_hint, "artifact_class", None), default="")
        if hint_class:
            return hint_class
        if submission_path.is_dir():
            if artifact_suffix(submission_path) in DIRECTORY_ARRAY_SUFFIXES:
                return ARTIFACT_CLASS_SINGLE_FILE
            return ARTIFACT_CLASS_BUNDLE
        if self._is_zip_submission(submission_path):
            return ARTIFACT_CLASS_MULTI_FILE_ZIP
        if self._is_tar_submission(submission_path):
            return ARTIFACT_CLASS_SINGLE_FILE
        if self._is_tabular_submission(submission_path):
            return ARTIFACT_CLASS_TABULAR
        if submission_path.name == SUBMISSION_MANIFEST_FILENAME:
            return ARTIFACT_CLASS_UNKNOWN
        return ARTIFACT_CLASS_SINGLE_FILE

    @classmethod
    def _requested_output_requires_non_tabular_artifact(cls, requested_output_path: Path) -> bool:
        suffix = cls._submission_format_suffix(requested_output_path)
        return bool(suffix and suffix not in _TABULAR_SUBMISSION_SUFFIXES)

    @staticmethod
    def _resolve_submission_manifest_path(submission_path: Path) -> Path | None:
        if submission_path.is_file() and is_submission_manifest_artifact(submission_path):
            return submission_path
        if submission_path.is_file():
            specific_manifest = submission_specific_manifest_path(submission_path)
            if specific_manifest.is_file() and (
                specific_manifest.name != SUBMISSION_MANIFEST_FILENAME
                or SubmissionService._manifest_references_submission_file(
                    manifest_path=specific_manifest,
                    submission_path=submission_path,
                )
            ):
                return specific_manifest
            parent_manifest = submission_path.parent / SUBMISSION_MANIFEST_FILENAME
            if parent_manifest.is_file() and SubmissionService._manifest_references_submission_file(
                manifest_path=parent_manifest,
                submission_path=submission_path,
            ):
                return parent_manifest
            return None
        if submission_path.is_dir():
            return find_submission_manifest(submission_path)
        return None

    @staticmethod
    def _manifest_references_submission_file(*, manifest_path: Path, submission_path: Path) -> bool:
        try:
            details = resolve_manifest_reference_details(manifest_path)
        except ValueError:
            return False
        if details.submission_path is None:
            return False
        try:
            return details.submission_path.resolve(strict=False) == submission_path.resolve(strict=False)
        except OSError:
            return False

    @staticmethod
    def _validate_specific_manifest_matches_submission_file(
        *,
        manifest_path: Path,
        submission_path: Path,
        manifest_submission_path: Path | None,
    ) -> None:
        if (
            not submission_path.is_file()
            or is_submission_manifest_artifact(submission_path)
            or manifest_path != submission_specific_manifest_path(submission_path)
            or manifest_path.name == SUBMISSION_MANIFEST_FILENAME
            or manifest_submission_path is None
        ):
            return
        try:
            same_file = manifest_submission_path.resolve(strict=False) == submission_path.resolve(strict=False)
        except OSError:
            same_file = False
        if same_file:
            return
        raise SubmissionValidationError(
            "run-specific manifest does not reference submitted file:\n"
            f"  manifest:        {manifest_path}\n"
            f"  submitted_file:  {submission_path}\n"
            f"  manifest_file:   {manifest_submission_path}"
        )

    @staticmethod
    def _expected_submission_suffixes(format_hint: SubmissionFormatHint | None) -> list[str]:
        """Extract normalized expected suffixes from a parsed submission format hint."""
        if format_hint is None or not format_hint.expected_suffixes:
            return []
        suffixes: list[str] = []
        for suffix in format_hint.expected_suffixes:
            normalized = str(suffix or "").strip().lower()
            if not normalized.startswith("."):
                normalized = f".{normalized}"
            if normalized not in suffixes:
                suffixes.append(normalized)
        return suffixes

    @classmethod
    def _format_hint_expects_generic_tar(cls, format_hint: SubmissionFormatHint | None) -> bool:
        return any(suffix in _TAR_SUBMISSION_SUFFIXES for suffix in cls._expected_submission_suffixes(format_hint))

    def _tar_code_contract_required(self, format_hint: SubmissionFormatHint | None) -> bool:
        """Only enforce legacy deck/main/cg tar contracts when context explicitly names them."""
        if self._format_hint_expects_generic_tar(format_hint):
            return False
        required_terms = ("deck.csv", "main.py", "cg/")
        for context_dir in self._candidate_context_dirs():
            for name in ("submission_format.md", "overview.md", "data.md", "rules.md"):
                path = context_dir / name
                if not path.exists() or not path.is_file():
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore").lower()
                except OSError:
                    continue
                if all(term in text for term in required_terms):
                    return True
        return False

    def _context_mentions_mapinfo_tab(self, format_hint: SubmissionFormatHint | None) -> bool:
        expected_suffixes = set(self._expected_submission_suffixes(format_hint))
        if ".tab" not in expected_suffixes:
            return False
        for context_dir in self._candidate_context_dirs():
            for name in ("submission_format.md", "overview.md", "data.md", "rules.md"):
                path = context_dir / name
                if not path.exists() or not path.is_file():
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore").lower()
                except OSError:
                    continue
                if "mapinfo" in text or "map info" in text:
                    return True
                if all(term in text for term in (".tab", ".dat", ".map", ".id")):
                    return True
        return False

    def _context_mentions_mapinfo_interchange(self, format_hint: SubmissionFormatHint | None) -> bool:
        expected_suffixes = set(self._expected_submission_suffixes(format_hint))
        if ".mif" not in expected_suffixes:
            return False
        for context_dir in self._candidate_context_dirs():
            for name in ("submission_format.md", "overview.md", "data.md", "rules.md"):
                path = context_dir / name
                if not path.exists() or not path.is_file():
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore").lower()
                except OSError:
                    continue
                if "mif/mid" in text or "mif-mid" in text:
                    return True
                if "mapinfo" in text and ".mid" in text:
                    return True
                if all(term in text for term in (".mif", ".mid")):
                    return True
        return False

    @classmethod
    def _preferred_archive_suffix(cls, format_hint: SubmissionFormatHint | None) -> str:
        for suffix in cls._expected_submission_suffixes(format_hint):
            if suffix in _ARCHIVE_SUBMISSION_SUFFIXES:
                return suffix
        return _ZIP_SUBMISSION_SUFFIX

    def _convert_submission_to_suffix(
        self,
        *,
        submission_path: Path,
        target_suffix: str,
        format_hint: SubmissionFormatHint | None,
    ) -> Path | None:
        """Convert submission artifact to target suffix when safe and deterministic."""
        if target_suffix == ".zip":
            return self._build_submission_zip(submission_path)
        if target_suffix in _TAR_SUBMISSION_SUFFIXES:
            return self._build_submission_tar(submission_path, target_suffix=target_suffix)
        if target_suffix not in _TABULAR_SUBMISSION_SUFFIXES:
            return None
        if not self._is_tabular_submission(submission_path):
            return None
        try:
            frame = self._read_tabular_submission(submission_path)
        except Exception:
            return None
        destination = self._path_with_submission_suffix(submission_path, target_suffix)
        try:
            self._write_tabular_submission(
                frame=frame,
                destination=destination,
                target_suffix=target_suffix,
                format_hint=format_hint,
            )
        except Exception:
            return None
        return destination

    @staticmethod
    def _build_submission_zip(
        submission_path: Path,
        *,
        members: list[Path] | None = None,
        staging_dir: Path | None = None,
        member_archive_names: list[str | None] | None = None,
    ) -> Path:
        """Create a zip archive from a single file or a staged directory."""
        if submission_path.is_dir():
            source_dir = submission_path
            archive_members = members or SubmissionService._archive_member_candidates(source_dir)
            destination = source_dir.parent / f"{source_dir.name}.zip"
        else:
            source_dir = staging_dir
            archive_members = members or [submission_path]
            destination = SubmissionService._path_with_submission_suffix(submission_path, ".zip")
        if not archive_members:
            raise SubmissionValidationError(f"submission bundle has no members to archive: {submission_path}")
        seen_arcnames: set[str] = set()
        with zipfile.ZipFile(destination, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for index, member in enumerate(archive_members):
                SubmissionService._validate_local_archive_member(member)
                if source_dir is not None:
                    try:
                        arcname = member.relative_to(source_dir).as_posix()
                    except ValueError:
                        arcname = member.name
                else:
                    arcname = member.name
                if member_archive_names and index < len(member_archive_names) and member_archive_names[index]:
                    arcname = str(member_archive_names[index])
                arcname = SubmissionService._safe_archive_member_name(arcname)
                if member.is_dir():
                    arcname = f"{arcname.rstrip('/')}/"
                if arcname in seen_arcnames:
                    raise SubmissionValidationError(f"submission bundle duplicate archive member name: {arcname}")
                seen_arcnames.add(arcname)
                if member.is_dir():
                    info = zipfile.ZipInfo(arcname, date_time=_ZIP_MEMBER_TIMESTAMP)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = _ZIP_DIR_EXTERNAL_ATTR
                    archive.writestr(info, b"")
                else:
                    info = zipfile.ZipInfo(arcname, date_time=_ZIP_MEMBER_TIMESTAMP)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = _ZIP_FILE_EXTERNAL_ATTR
                    archive.writestr(info, member.read_bytes())
        return destination

    @staticmethod
    def _build_submission_tar(
        submission_path: Path,
        *,
        target_suffix: str,
        members: list[Path] | None = None,
        staging_dir: Path | None = None,
        member_archive_names: list[str | None] | None = None,
    ) -> Path:
        """Create a tar archive from a single file or staged directory."""
        mode = _TAR_SUBMISSION_WRITE_MODES.get(target_suffix)
        if mode is None:
            raise SubmissionValidationError(f"unsupported tar submission suffix: {target_suffix}")
        if submission_path.is_dir():
            source_dir = submission_path
            archive_members = members or SubmissionService._archive_member_candidates(source_dir)
            destination = source_dir.parent / f"{source_dir.name}{target_suffix}"
        else:
            source_dir = staging_dir
            archive_members = members or [submission_path]
            destination = SubmissionService._path_with_submission_suffix(submission_path, target_suffix)
        if not archive_members:
            raise SubmissionValidationError(f"submission bundle has no members to archive: {submission_path}")
        seen_arcnames: set[str] = set()
        with SubmissionService._open_tar_for_write(destination=destination, target_suffix=target_suffix) as archive:
            for index, member in enumerate(archive_members):
                SubmissionService._validate_local_archive_member(member)
                if source_dir is not None:
                    try:
                        arcname = member.relative_to(source_dir).as_posix()
                    except ValueError:
                        arcname = member.name
                else:
                    arcname = member.name
                if member_archive_names and index < len(member_archive_names) and member_archive_names[index]:
                    arcname = str(member_archive_names[index])
                arcname = SubmissionService._safe_archive_member_name(arcname)
                if arcname in seen_arcnames:
                    raise SubmissionValidationError(f"submission bundle duplicate archive member name: {arcname}")
                seen_arcnames.add(arcname)
                archive.add(
                    member,
                    arcname=arcname,
                    recursive=False,
                    filter=SubmissionService._canonical_tar_info,
                )
        return destination

    @staticmethod
    @contextmanager
    def _open_tar_for_write(destination: Path, *, target_suffix: str):
        if target_suffix in {".tar.gz", ".tgz"}:
            with destination.open("wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gzipped:
                    with tarfile.open(fileobj=gzipped, mode="w") as archive:
                        yield archive
            return
        if target_suffix in ZSTD_TAR_ARCHIVE_SUBMISSION_SUFFIXES:
            with destination.open("wb") as raw:
                with zstd.ZstdCompressor(level=9).stream_writer(raw) as compressed:
                    with tarfile.open(fileobj=compressed, mode="w") as archive:
                        yield archive
            return
        mode = _TAR_SUBMISSION_WRITE_MODES.get(target_suffix)
        if mode is None:
            raise SubmissionValidationError(f"unsupported tar submission suffix: {target_suffix}")
        with tarfile.open(destination, mode) as archive:
            yield archive

    @staticmethod
    def _validate_local_archive_member(member: Path) -> None:
        if member.is_symlink():
            raise SubmissionValidationError(f"submission bundle has unsupported symlink member: {member}")
        if not member.exists() or not (member.is_file() or member.is_dir()):
            raise SubmissionValidationError(f"submission bundle member missing: {member}")

    @staticmethod
    def _canonical_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mtime = _TAR_MEMBER_MTIME
        info.pax_headers = {}
        if info.isdir():
            info.mode = 0o755
        elif info.isfile():
            info.mode = 0o644
        return info

    @staticmethod
    def _archive_member_candidates(source_dir: Path) -> list[Path]:
        return sorted(source_dir.rglob("*"), key=lambda path: path.relative_to(source_dir).as_posix())

    def _prepare_bundle_submission(
        self,
        *,
        base_path: Path,
        submission_path: Path | None,
        staging_dir: Path | None,
        members: list[Path],
        member_archive_names: list[str | None] | None = None,
        target_suffix: str = _ZIP_SUBMISSION_SUFFIX,
    ) -> Path:
        if submission_path is not None and submission_path.exists() and submission_path.is_file():
            if (
                target_suffix in _EXTERNAL_ARCHIVE_SUBMISSION_SUFFIXES
                and self._submission_format_suffix(submission_path) == target_suffix
            ):
                self._validate_external_archive_submission(submission_path)
                return submission_path
            if self._is_zip_submission(submission_path):
                self._validate_zip_submission(submission_path)
                return submission_path
            if self._is_tar_submission(submission_path):
                self._validate_tar_submission(
                    submission_path,
                    require_code_contract=target_suffix not in _TAR_SUBMISSION_SUFFIXES,
                )
                return submission_path
            if not members and staging_dir is None:
                if target_suffix in _TAR_SUBMISSION_SUFFIXES:
                    return self._build_submission_tar(submission_path, target_suffix=target_suffix)
                return self._build_submission_zip(submission_path)
        source_dir = staging_dir
        if source_dir is None and submission_path is not None and submission_path.is_dir():
            source_dir = submission_path
        if source_dir is None and base_path.is_dir():
            source_dir = base_path
        if source_dir is None and members:
            source_dir = base_path.parent
        if source_dir is None:
            raise SubmissionValidationError("bundle submission requires a staging_dir or archive file")
        if target_suffix in _EXTERNAL_ARCHIVE_SUBMISSION_SUFFIXES:
            raise SubmissionValidationError(
                f"cannot build {target_suffix} submission archives locally; "
                f"provide a prebuilt submission{target_suffix} file"
            )
        archive_members = members or self._archive_member_candidates(source_dir)
        if target_suffix in _TAR_SUBMISSION_SUFFIXES:
            prepared_tar = self._build_submission_tar(
                source_dir,
                target_suffix=target_suffix,
                members=archive_members,
                member_archive_names=member_archive_names,
            )
            self._validate_tar_submission(prepared_tar, require_code_contract=False)
            return prepared_tar
        prepared = self._build_submission_zip(
            source_dir,
            members=archive_members,
            member_archive_names=member_archive_names,
        )
        self._validate_zip_submission(prepared)
        return prepared

    def _read_tabular_submission(self, path: Path):
        """Read a tabular submission using a suffix-aware parser."""
        try:
            return read_table(path)
        except Exception as exc:  # noqa: BLE001
            raise SubmissionValidationError(f"unable to read tabular submission: {path}") from exc

    def _write_tabular_submission(
        self,
        *,
        frame,
        destination: Path,
        target_suffix: str,
        format_hint: SubmissionFormatHint | None,
    ) -> None:
        """Write a tabular submission frame in the requested target format."""
        frame = frame_with_normalized_table_columns(frame)
        if target_suffix in _TABULAR_PARQUET_SUFFIXES:
            frame.to_parquet(destination, index=False)
            return
        if target_suffix == ".orc":
            frame.to_orc(destination, index=False)
            return
        if target_suffix in _TABULAR_HDF_SUBMISSION_SUFFIXES:
            frame.to_hdf(destination, key="submission", mode="w", format="table", index=False)
            return
        if target_suffix in _TABULAR_ARROW_IPC_SUFFIXES:
            frame.to_feather(destination)
            return
        if target_suffix == ".avro":
            write_table(frame, destination)
            return
        if target_suffix in _TABULAR_EXCEL_SUFFIXES:
            frame.to_excel(destination, index=False)
            return
        if target_suffix in _TABULAR_STATA_SUFFIXES:
            frame.to_stata(destination, write_index=False)
            return
        if target_suffix.startswith(".xml"):
            write_xml_tabular_frame(frame, destination)
            return
        if target_suffix.startswith(_TABULAR_HTML_SUFFIX_PREFIXES):
            self._write_text_submission(destination, frame.to_html(index=False))
            return
        if target_suffix in _TABULAR_PICKLE_SUFFIXES:
            frame.to_pickle(destination)
            return
        if target_suffix in _TABULAR_STRUCTURED_SUFFIXES:
            if target_suffix.startswith((".yaml", ".yml")):
                import yaml

                payload = yaml.safe_dump(frame.to_dict(orient="records"), sort_keys=False)
            else:
                payload = frame.to_json(orient="records", lines=_is_json_lines_suffix(target_suffix))
            self._write_text_submission(destination, payload)
            return
        if target_suffix in _TABULAR_TEXT_SUFFIXES:
            sep = default_delimited_text_separator(target_suffix)
            if _is_txt_like_suffix(target_suffix) or sep == ",":
                if format_hint is not None and format_hint.delimiter in {",", "\t", ";", "|"}:
                    sep = format_hint.delimiter
            self._write_text_submission(destination, frame.to_csv(index=False, sep=sep))
            return
        sep = ","
        if format_hint is not None and format_hint.delimiter in {",", "\t", ";", "|"}:
            sep = format_hint.delimiter
        self._write_text_submission(destination, frame.to_csv(index=False, sep=sep))

    def submit_prepared(
        self,
        *,
        prepared_path: Path,
        message: str,
        run_id: str | None,
        iteration: int | None = None,
        metrics_path: Path | None = None,
        offline_score: float | None = None,
        score_source: str | None = None,
        pipeline_name: str | None = None,
        submission_kind: str | None = None,
        out_of_band: bool = False,
        source_run_id: str | None = None,
        source_iteration: int | None = None,
    ) -> SubmissionResult:
        message = self._normalize_submission_message(message)
        ledger = SubmissionLedger(self._config.submission_ledger_path)
        if not self._config.bypass_rate_limit:
            ensure_submission_rate_limit(ledger)
        if not self._config.force_submit:
            ensure_not_duplicate_submission(
                ledger,
                slug=self._config.slug,
                message=message,
                submission_path=str(prepared_path),
            )

        command_result = run_kaggle_submit(
            slug=self._config.slug,
            submission_file=prepared_path,
            message=message,
            dry_run=self._config.dry_run,
        )
        if not self._config.dry_run:
            ledger.record(
                slug=self._config.slug,
                message=message,
                submission_path=prepared_path,
                run_id=run_id,
                iteration=iteration,
                metrics_path=metrics_path,
                offline_score=offline_score,
                score_source=score_source,
                pipeline_name=pipeline_name,
                submission_kind=submission_kind,
                out_of_band=out_of_band,
                source_run_id=source_run_id,
                source_iteration=source_iteration,
            )
        return SubmissionResult(
            message=message,
            submission_path=prepared_path,
            exit_code=command_result.returncode,
            stdout=command_result.stdout,
            stderr=command_result.stderr,
        )

    def _resolve_sample_submission(self) -> Path:
        sample_path = self._config.sample_submission_path

        from kagglebot.solver.io import ensure_sample_submission, find_competition_files

        discovered: Path | None = None
        try:
            _, _, discovered = find_competition_files(self._config.data_dir)
        except FileNotFoundError:
            pass
        ensured = ensure_sample_submission(self._config.data_dir)
        candidates: list[Path] = []
        for candidate in (sample_path, discovered, ensured, *self._find_synthesized_sample_submissions()):
            if candidate is None:
                continue
            if candidate.exists() and tabular_file_has_data_rows(candidate):
                candidates.append(candidate)
        discovered_text_sample = self._find_usable_sample_submission_in_data_dir()
        if discovered_text_sample is not None:
            candidates.append(discovered_text_sample)
        if candidates:
            deduped: list[Path] = []
            seen: set[Path] = set()
            for candidate in candidates:
                try:
                    resolved = candidate.resolve()
                except OSError:
                    continue
                if resolved in seen:
                    continue
                seen.add(resolved)
                deduped.append(candidate)
            if deduped:
                return max(deduped, key=sample_candidate_key)
        return sample_path

    def _find_synthesized_sample_submissions(self) -> list[Path]:
        cache_dir = self._config.data_dir / ".kagglebot_cache"
        if not cache_dir.exists() or not cache_dir.is_dir():
            return []
        try:
            candidates = [path for path in cache_dir.glob("sample_submission_synth.*") if path.is_file()]
        except OSError:
            return []
        return [path for path in candidates if tabular_suffix(path) in _TABULAR_SUBMISSION_SUFFIXES]

    def _prepare_submission_path(self, sample_path: Path, submission_path: Path) -> Path:
        if not sample_path.exists() or not submission_path.exists():
            return submission_path
        if tabular_suffix(sample_path) not in _TABULAR_TEXT_SUFFIXES:
            return submission_path
        if tabular_suffix(submission_path) not in _TABULAR_TEXT_SUFFIXES:
            return submission_path
        sample_delim = self._sniff_delimiter(sample_path)
        submission_delim = self._sniff_delimiter(submission_path)
        if (
            sample_delim == "\t"
            and submission_delim == "\t"
            and self._tabular_text_base_suffix(submission_path) not in {".tsv", ".tab"}
        ):
            target_suffix = self._suffix_with_existing_compression(submission_path, ".tsv")
            tsv_path = self._path_with_submission_suffix(submission_path, target_suffix)
            if tsv_path != submission_path:
                copy_artifact_if_needed(source=submission_path, destination=tsv_path)
            return tsv_path
        if (
            sample_delim == "|"
            and submission_delim == "|"
            and self._tabular_text_base_suffix(submission_path) != ".psv"
        ):
            target_suffix = self._suffix_with_existing_compression(submission_path, ".psv")
            psv_path = self._path_with_submission_suffix(submission_path, target_suffix)
            if psv_path != submission_path:
                copy_artifact_if_needed(source=submission_path, destination=psv_path)
            return psv_path
        return submission_path

    def _maybe_compact_delimited_text_submission(self, sample_path: Path, submission_path: Path) -> Path:
        if not sample_path.exists() or not submission_path.exists():
            return submission_path
        submission_suffix = self._submission_format_suffix(submission_path)
        if submission_suffix not in _TABULAR_TEXT_SUFFIXES:
            return submission_path
        base_suffix = self._tabular_text_base_suffix(submission_path)
        if not _is_delimited_text_suffix(base_suffix):
            return submission_path
        try:
            size_bytes = submission_path.stat().st_size
        except OSError:
            return submission_path
        if size_bytes <= _KAGGLE_SUBMISSION_SOFT_MAX_BYTES:
            return submission_path

        sample_delim = self._sniff_delimiter(sample_path)
        submission_delim = self._sniff_delimiter(
            submission_path,
            default=default_delimited_text_separator(submission_suffix) or sample_delim,
        )

        try:
            frame = self._read_delimited_text_submission(submission_path, sep=submission_delim)
        except Exception:
            return submission_path

        if submission_suffix and submission_path.name.lower().endswith(submission_suffix):
            compact_path = submission_path.with_name(
                f"{submission_path.name[: -len(submission_suffix)]}.compact{submission_suffix}"
            )
        else:
            compact_path = submission_path.with_name(f"{submission_path.stem}.compact{submission_path.suffix}")
        try:
            self._write_text_submission(
                compact_path,
                frame.to_csv(
                    index=False,
                    sep=submission_delim,
                    float_format=_KAGGLE_SUBMISSION_COMPACT_FLOAT_FORMAT,
                ),
            )
        except Exception:
            return submission_path

        try:
            compact_size = compact_path.stat().st_size
        except OSError:
            return submission_path
        bytes_saved = size_bytes - compact_size
        relative_saved = bytes_saved / size_bytes if size_bytes > 0 else 0.0
        should_use_compact = (
            compact_size < size_bytes
            and bytes_saved >= _KAGGLE_SUBMISSION_COMPACT_MIN_BYTES_SAVED
            and relative_saved >= _KAGGLE_SUBMISSION_COMPACT_MIN_RELATIVE_SAVED
        )
        if not should_use_compact:
            try:
                compact_path.unlink()
            except OSError:
                pass
            return submission_path
        return compact_path

    def _attempt_autofix_submission(self, *, sample_path: Path, submission_path: Path) -> Path:
        """Best-effort local fixups to make a submission match sample_submission.csv.

        This is intentionally conservative:
        - Only runs after strict validation fails.
        - Produces a new file next to the original submission.
        - Uses sample_submission rows/columns as the template when present.
        """
        if not sample_path.exists() or not submission_path.exists():
            return submission_path

        try:
            import pandas as pd
        except Exception:
            return submission_path

        sample_delim = self._sniff_delimiter(sample_path)
        expected = self._read_tabular_submission(sample_path)
        if expected.columns.empty:
            return submission_path

        expected_columns = list(expected.columns)
        autofix_suffix = self._autofixed_tabular_suffix(
            sample_path=sample_path,
            submission_path=submission_path,
            sample_delim=sample_delim,
        )
        sample_has_data_rows = tabular_file_has_data_rows(sample_path)
        if not sample_has_data_rows and self._is_synthesized_sample_submission(sample_path):
            format_hint = self._resolve_submission_format_hint()
            if (
                format_hint is not None
                and format_hint.columns
                and list(format_hint.columns) != expected_columns
                and self._sample_columns_look_placeholder(expected_columns)
            ):
                expected_columns = list(format_hint.columns)

        source_submission_path = submission_path
        submission_delim = self._sniff_delimiter(source_submission_path, default=sample_delim)
        columns_only_autofix = self._attempt_autofix_submission_columns_only(
            expected_columns=expected_columns,
            sample_delim=sample_delim,
            submission_path=source_submission_path,
            submission_delim=submission_delim,
            target_suffix=autofix_suffix,
        )
        if columns_only_autofix is not None:
            source_submission_path = columns_only_autofix
            submission_delim = sample_delim

        id_suffix_autofix = self._attempt_autofix_submission_id_suffix(
            sample_path=sample_path,
            expected_columns=expected_columns,
            sample_delim=sample_delim,
            submission_path=source_submission_path,
            submission_delim=submission_delim,
            target_suffix=autofix_suffix,
        )
        if id_suffix_autofix is not None:
            source_submission_path = id_suffix_autofix
            submission_delim = sample_delim

        key_cols, pred_cols = self._infer_autofix_key_and_prediction_columns(
            expected=expected,
            expected_columns=expected_columns,
        )

        if source_submission_path != submission_path:
            return source_submission_path
        if not sample_has_data_rows:
            return source_submission_path
        dataframe_autofix = self._attempt_autofix_submission_frame_aggregation(
            expected=expected,
            expected_columns=expected_columns,
            key_cols=key_cols,
            pred_cols=pred_cols,
            submission_path=source_submission_path,
            target_suffix=autofix_suffix,
            sample_delim=sample_delim,
        )
        if dataframe_autofix is not None:
            return dataframe_autofix
        if not _is_delimited_text_suffix(self._tabular_text_base_suffix(source_submission_path)):
            return source_submission_path
        if not key_cols or not pred_cols:
            return source_submission_path

        header, col_index = self._sniff_header_and_column_index(
            submission_path=source_submission_path, delim=submission_delim, expected_columns=expected_columns
        )
        if col_index is None:
            return submission_path

        key_positions = [col_index[c] for c in key_cols if c in col_index]
        pred_positions = [col_index[c] for c in pred_cols if c in col_index]
        if not key_positions or not pred_positions:
            return submission_path

        # Aggregate predictions by key (mean for duplicates).
        sums: dict[tuple[str, ...], list[float]] = {}
        counts: dict[tuple[str, ...], int] = {}
        try:
            with self._open_text_submission(source_submission_path, newline="") as handle:
                reader = csv.reader(handle, delimiter=submission_delim)
                first = True
                for row in reader:
                    if not row or all(not cell.strip() for cell in row):
                        continue
                    if first and header:
                        first = False
                        continue
                    first = False
                    if len(row) < max(key_positions + pred_positions) + 1:
                        continue
                    key = tuple(str(row[pos]).strip() for pos in key_positions)
                    if any(not part for part in key):
                        continue
                    values: list[float] = []
                    ok = True
                    for pos in pred_positions:
                        try:
                            values.append(float(str(row[pos]).strip()))
                        except ValueError:
                            ok = False
                            break
                    if not ok:
                        continue
                    if key not in sums:
                        sums[key] = values
                        counts[key] = 1
                    else:
                        sums[key] = [a + b for a, b in zip(sums[key], values, strict=False)]
                        counts[key] += 1
        except OSError:
            return submission_path

        if not sums:
            return submission_path

        if not sample_has_data_rows:
            return submission_path
        prepared = self._expand_autofix_template_if_placeholder(
            sample=expected,
            key_cols=key_cols,
            pred_cols=pred_cols,
        )
        if prepared is None:
            prepared = expected.copy()

        # Fill prediction columns by matching keys against the sample template.
        key_df = prepared[key_cols]
        for pred_idx, pred_col in enumerate(pred_cols):
            if pred_col not in prepared.columns:
                continue
            filled = []
            default_series = prepared[pred_col]
            for row_idx in range(len(prepared)):
                key = tuple(key_df.iloc[row_idx, k].strip() for k in range(len(key_cols)))
                if key in sums:
                    filled.append(sums[key][pred_idx] / counts[key])
                else:
                    filled.append(default_series.iloc[row_idx])
            prepared[pred_col] = pd.to_numeric(pd.Series(filled), errors="coerce").fillna(default_series)

        prepared = prepared[expected_columns]
        return self._write_autofixed_tabular_submission(
            frame=prepared,
            submission_path=submission_path,
            target_suffix=autofix_suffix,
            delimiter=sample_delim,
        )

    def _attempt_autofix_submission_frame_aggregation(
        self,
        *,
        expected,
        expected_columns: list[str],
        key_cols: list[str],
        pred_cols: list[str],
        submission_path: Path,
        target_suffix: str,
        sample_delim: str,
    ) -> Path | None:
        if not key_cols or not pred_cols:
            return None
        if _is_delimited_text_suffix(self._tabular_text_base_suffix(submission_path)):
            return None
        try:
            import pandas as pd
        except Exception:
            return None
        try:
            frame = self._read_tabular_submission(submission_path)
        except Exception:
            return None
        if frame.empty:
            return None
        if any(col not in frame.columns for col in key_cols + pred_cols):
            return None

        sums: dict[tuple[str, ...], list[float]] = {}
        counts: dict[tuple[str, ...], int] = {}
        for _, row in frame.iterrows():
            key = tuple(self._normalize_autofix_key_value(row[col]) for col in key_cols)
            if any(not part for part in key):
                continue
            values: list[float] = []
            ok = True
            for col in pred_cols:
                try:
                    values.append(float(row[col]))
                except (TypeError, ValueError):
                    ok = False
                    break
            if not ok:
                continue
            if key not in sums:
                sums[key] = values
                counts[key] = 1
            else:
                sums[key] = [a + b for a, b in zip(sums[key], values, strict=False)]
                counts[key] += 1
        if not sums:
            return None

        prepared = self._expand_autofix_template_if_placeholder(
            sample=expected,
            key_cols=key_cols,
            pred_cols=pred_cols,
        )
        if prepared is None:
            prepared = expected.copy()

        key_df = prepared[key_cols].astype(str)
        for pred_idx, pred_col in enumerate(pred_cols):
            if pred_col not in prepared.columns:
                continue
            default_series = prepared[pred_col]
            filled = []
            for row_idx in range(len(prepared)):
                key = tuple(self._normalize_autofix_key_value(key_df.iloc[row_idx, k]) for k in range(len(key_cols)))
                if key in sums:
                    filled.append(sums[key][pred_idx] / counts[key])
                else:
                    filled.append(default_series.iloc[row_idx])
            prepared[pred_col] = pd.to_numeric(pd.Series(filled), errors="coerce").fillna(default_series)

        try:
            prepared = prepared[expected_columns]
        except Exception:
            return None
        try:
            return self._write_autofixed_tabular_submission(
                frame=prepared,
                submission_path=submission_path,
                target_suffix=target_suffix,
                delimiter=sample_delim,
            )
        except Exception:
            return None

    @staticmethod
    def _normalize_autofix_key_value(value: object) -> str:
        if not isinstance(value, str):
            try:
                numeric = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                numeric = None
            if numeric is not None and numeric.is_integer():
                return str(int(numeric))
        return str(value).strip()

    @classmethod
    def _sample_columns_look_placeholder(cls, columns: list[str]) -> bool:
        if len(columns) != 2:
            return False
        normalized = [cls._normalize_column_name(c) for c in columns]
        id_like = cls._column_name_is_id_like(normalized[0])
        prediction_like = cls._column_name_is_prediction_like(normalized[1])
        return bool(id_like and prediction_like)

    @classmethod
    def _column_name_is_id_like(cls, normalized: str) -> bool:
        return normalized in {cls._normalize_column_name(v) for v in _ID_LIKE_COLUMN_NAMES}

    @classmethod
    def _column_name_is_prediction_like(cls, normalized: str) -> bool:
        return normalized in {cls._normalize_column_name(v) for v in _PREDICTION_LIKE_COLUMN_NAMES} | {
            "target",
            "label",
            "value",
            "y",
        }

    @classmethod
    def _infer_autofix_key_and_prediction_columns(
        cls, *, expected, expected_columns: list[str]
    ) -> tuple[list[str], list[str]]:
        pred_cols = [
            col
            for col in expected_columns
            if cls._column_name_is_prediction_like(cls._normalize_column_name(col))
            and col in expected.columns
            and cls._sample_column_is_numeric(expected[col])
        ]
        if not pred_cols:
            pred_cols = [
                col
                for col in expected_columns[1:]
                if col in expected.columns and cls._sample_column_is_numeric(expected[col])
            ]
        if not pred_cols and len(expected_columns) >= 2:
            pred_cols = expected_columns[1:]

        key_cols = [c for c in expected_columns if c not in pred_cols]
        all_columns_prediction_like = bool(expected_columns) and all(
            cls._column_name_is_prediction_like(cls._normalize_column_name(col)) for col in expected_columns
        )
        if len(expected_columns) == 1 and pred_cols == expected_columns:
            key_cols = []
        elif all_columns_prediction_like and key_cols == []:
            key_cols = []
        elif len(expected_columns) == 1 and cls._column_name_is_prediction_like(
            cls._normalize_column_name(expected_columns[0])
        ):
            pred_cols = list(expected_columns)
            key_cols = []
        elif not key_cols and expected_columns:
            key_cols = [expected_columns[0]]
        return key_cols, pred_cols

    @staticmethod
    def _is_synthesized_sample_submission(path: Path) -> bool:
        name = path.name.lower()
        if "sample_submission_synth" in name:
            return True
        if ".kagglebot_cache" in {part.lower() for part in path.parts}:
            return True
        return False

    def _attempt_autofix_submission_columns_only(
        self,
        *,
        expected_columns: list[str],
        sample_delim: str,
        submission_path: Path,
        submission_delim: str,
        target_suffix: str,
    ) -> Path | None:
        """Try to rewrite the submission with expected columns/ordering.

        This handles common failures where the submission is structurally correct but uses
        different column names/casing (e.g., `Id,Category` vs `id,prediction`), including
        header-only sample submissions.
        """
        if not expected_columns:
            return None

        try:
            frame = self._read_tabular_submission(submission_path)
        except Exception:
            return None

        actual_columns = list(frame.columns)
        if len(actual_columns) != len(expected_columns):
            return None

        mapping = self._resolve_column_mapping(expected_columns=expected_columns, actual_columns=actual_columns)
        if mapping is None:
            return None

        if expected_columns == actual_columns and all(mapping.get(col) == col for col in actual_columns):
            return None

        renamed = frame.rename(columns=mapping, errors="raise")
        try:
            renamed = renamed[expected_columns]
        except Exception:
            return None

        try:
            return self._write_autofixed_tabular_submission(
                frame=renamed,
                submission_path=submission_path,
                target_suffix=target_suffix,
                delimiter=sample_delim,
            )
        except Exception:
            return None

    def _attempt_autofix_submission_id_suffix(
        self,
        *,
        sample_path: Path,
        expected_columns: list[str],
        sample_delim: str,
        submission_path: Path,
        submission_delim: str,
        target_suffix: str,
    ) -> Path | None:
        """Append a required filename suffix to id values when inferred confidently."""
        if not expected_columns:
            return None

        try:
            frame = self._read_tabular_submission(submission_path)
        except Exception:
            return None
        if frame.empty:
            return None

        id_col = expected_columns[0]
        if not self._column_name_is_id_like(self._normalize_column_name(id_col)):
            return None
        if id_col not in frame.columns:
            return None

        raw_ids = [str(value).strip() for value in frame[id_col].tolist()]
        required_suffix = infer_required_id_suffix(
            sample_csv=sample_path,
            data_dir=self._config.data_dir,
            submission_ids=raw_ids,
        )
        if not required_suffix:
            return None

        rewritten = frame[id_col].astype(str).map(lambda raw: normalize_id_with_required_suffix(raw, required_suffix))
        current = frame[id_col].astype(str).map(str.strip)
        if rewritten.equals(current):
            return None

        frame[id_col] = rewritten
        if all(col in frame.columns for col in expected_columns):
            frame = frame[expected_columns]

        try:
            return self._write_autofixed_tabular_submission(
                frame=frame,
                submission_path=submission_path,
                target_suffix=target_suffix,
                delimiter=sample_delim,
            )
        except Exception:
            return None

    def _autofixed_tabular_suffix(self, *, sample_path: Path, submission_path: Path, sample_delim: str) -> str:
        submission_suffix = self._submission_format_suffix(submission_path)
        if submission_suffix in _TABULAR_SUBMISSION_SUFFIXES:
            return submission_suffix
        sample_suffix = self._submission_format_suffix(sample_path)
        if sample_suffix in _TABULAR_SUBMISSION_SUFFIXES:
            return sample_suffix
        if sample_delim == "\t":
            return ".tsv"
        if sample_delim == "|":
            return ".psv"
        return ".csv"

    def _write_autofixed_tabular_submission(
        self,
        *,
        frame,
        submission_path: Path,
        target_suffix: str,
        delimiter: str | None = None,
    ) -> Path:
        submission_suffix = self._submission_format_suffix(submission_path)
        if submission_suffix and submission_path.name.lower().endswith(submission_suffix):
            base_name = submission_path.name[: -len(submission_suffix)]
        else:
            base_name = submission_path.stem
        prepared_path = submission_path.with_name(f"{base_name}.autofixed{target_suffix}")
        format_hint = self._resolve_submission_format_hint()
        if delimiter in {",", "\t", ";", "|"}:
            format_hint = SubmissionFormatHint(
                columns=format_hint.columns if format_hint is not None else None,
                delimiter=delimiter,
                expected_suffixes=format_hint.expected_suffixes if format_hint is not None else None,
                artifact_class=format_hint.artifact_class if format_hint is not None else None,
                artifact_container=format_hint.artifact_container if format_hint is not None else None,
            )
        self._write_tabular_submission(
            frame=frame,
            destination=prepared_path,
            target_suffix=target_suffix,
            format_hint=format_hint,
        )
        return prepared_path

    @staticmethod
    def _path_with_submission_suffix(path: Path, target_suffix: str) -> Path:
        current_suffix = SubmissionService._submission_format_suffix(path)
        if current_suffix and path.name.lower().endswith(current_suffix):
            return path.with_name(f"{path.name[: -len(current_suffix)]}{target_suffix}")
        return path.with_suffix(target_suffix)

    @staticmethod
    def _suffix_with_existing_compression(path: Path, base_suffix: str) -> str:
        current_suffix = SubmissionService._submission_format_suffix(path)
        for compression_suffix in ASSET_COMPRESSION_SUFFIXES:
            if current_suffix.endswith(compression_suffix):
                return f"{base_suffix}{compression_suffix}"
        return base_suffix

    @staticmethod
    def _normalize_column_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(name or "").strip().lower())

    @classmethod
    def _semantic_group_for_column(cls, normalized: str) -> str:
        if normalized in {cls._normalize_column_name(v) for v in _ID_LIKE_COLUMN_NAMES}:
            return "id"
        if normalized in {cls._normalize_column_name(v) for v in _PREDICTION_LIKE_COLUMN_NAMES}:
            return "prediction"
        return normalized

    @classmethod
    def _resolve_column_mapping(
        cls, *, expected_columns: list[str], actual_columns: list[str]
    ) -> dict[str, str] | None:
        expected_norm = [cls._normalize_column_name(c) for c in expected_columns]
        actual_norm = [cls._normalize_column_name(c) for c in actual_columns]

        # Fast-paths: exact or case/format-only differences.
        if expected_columns == actual_columns:
            return {c: c for c in actual_columns}
        if expected_norm == actual_norm:
            return {actual: expected for actual, expected in zip(actual_columns, expected_columns, strict=False)}

        expected_groups = [cls._semantic_group_for_column(n) for n in expected_norm]
        actual_groups = [cls._semantic_group_for_column(n) for n in actual_norm]

        if len(expected_groups) != len(actual_groups):
            return None

        remaining_actual = set(range(len(actual_columns)))
        mapping: dict[str, str] = {}

        # 1) Prefer exact normalized matches.
        for exp_idx, exp_norm in enumerate(expected_norm):
            matches = [i for i in remaining_actual if actual_norm[i] == exp_norm]
            if len(matches) == 1:
                act_idx = matches[0]
                remaining_actual.remove(act_idx)
                mapping[actual_columns[act_idx]] = expected_columns[exp_idx]

        # 2) Then match by semantic group (id/prediction), but only when unambiguous.
        for exp_idx, exp_group in enumerate(expected_groups):
            expected_name = expected_columns[exp_idx]
            if expected_name in mapping.values():
                continue
            matches = [i for i in remaining_actual if actual_groups[i] == exp_group]
            if len(matches) != 1:
                return None
            act_idx = matches[0]
            remaining_actual.remove(act_idx)
            mapping[actual_columns[act_idx]] = expected_name

        if len(mapping) != len(expected_columns):
            return None
        return mapping

    def _expand_autofix_template_if_placeholder(
        self,
        *,
        sample,
        key_cols: list[str],
        pred_cols: list[str],
    ):
        """Return a test-sized template when sample_submission rows look truncated.

        Some competitions ship a tiny example `sample_submission.csv` while Kaggle expects
        one row per test id. If we can confidently detect that pattern, expand the template
        to all test ids and use safe numeric defaults for missing predictions.
        """
        try:
            import pandas as pd
        except Exception:
            return None

        if len(key_cols) != 1:
            return None
        id_col = key_cols[0]
        if id_col not in getattr(sample, "columns", ()):
            return None
        if len(sample) > 10:
            return None
        if sample[id_col].duplicated().any():
            return None

        test_ids = discover_test_ids(self._config.data_dir, id_col=id_col)
        if test_ids is None:
            return None
        sample_ids = sample[id_col].astype(str).tolist()
        if len(test_ids) < max(len(sample_ids) * 3, len(sample_ids) + 10):
            return None
        if sample_ids and test_ids[: len(sample_ids)] != sample_ids and not set(sample_ids).issubset(set(test_ids)):
            return None

        defaults: dict[str, float] = {col: 0.0 for col in pred_cols}
        train_path: Path | None = None
        try:
            from kagglebot.solver.io import find_competition_files

            train_path, _, _ = find_competition_files(self._config.data_dir)
        except Exception:  # noqa: BLE001
            train_path = None
        if train_path is not None and train_path.exists() and pred_cols:
            try:
                train = read_table(train_path)
            except Exception:  # noqa: BLE001
                train = None
            if train is not None:
                for col in pred_cols:
                    if col not in train.columns:
                        continue
                    series = pd.to_numeric(train[col], errors="coerce").dropna()
                    if not series.empty:
                        defaults[col] = float(series.mean())

        template = pd.DataFrame({id_col: test_ids})
        for col in pred_cols:
            template[col] = defaults.get(col, 0.0)
        # Match sample column order (and preserve any additional key columns if present).
        for col in list(sample.columns):
            if col not in template.columns:
                template[col] = ""
        return template[list(sample.columns)]

    @staticmethod
    def _sniff_header_and_column_index(
        *, submission_path: Path, delim: str, expected_columns: list[str]
    ) -> tuple[bool, dict[str, int] | None]:
        try:
            with SubmissionService._open_text_submission(submission_path, newline="") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    header_fields = next(csv.reader([line], delimiter=delim))
                    normalized = normalize_table_column_names([f.strip() for f in header_fields])
                    if normalized == expected_columns:
                        return True, {name: i for i, name in enumerate(normalized)}
                    if len(normalized) == len(expected_columns) and set(normalized) == set(expected_columns):
                        return True, {name: i for i, name in enumerate(normalized)}
                    # Headerless: assume the file is in expected column order.
                    return False, {name: i for i, name in enumerate(expected_columns)}
        except OSError:
            return False, None
        return False, None

    @staticmethod
    def _sample_column_is_numeric(sample_col) -> bool:  # type: ignore[no-untyped-def]
        try:
            import pandas as pd
        except Exception:
            return False
        if sample_col is None or getattr(sample_col, "empty", True):
            return False
        if pd.api.types.is_numeric_dtype(sample_col):
            return True
        coerced = pd.to_numeric(sample_col, errors="coerce")
        return coerced.notna().all()

    @staticmethod
    def _sniff_delimiter(path: Path, default: str | None = None) -> str:
        resolved_default = default or default_delimited_text_separator(tabular_suffix(path))
        candidates: list[str] = []
        for sep in (resolved_default, "\t", ",", ";", "|"):
            if sep and sep not in candidates:
                candidates.append(sep)
        counts = {sep: 0 for sep in candidates}
        lines_seen = 0
        try:
            with SubmissionService._open_text_submission(path) as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    lines_seen += 1
                    for sep in candidates:
                        counts[sep] += line.count(sep)
                    if lines_seen >= 100:
                        break
        except OSError:
            return resolved_default
        if lines_seen == 0:
            return resolved_default
        best = max(candidates, key=lambda sep: counts[sep])
        if counts[best] == 0:
            return resolved_default
        if counts.get(resolved_default, 0) >= counts[best]:
            return resolved_default
        return best

    @staticmethod
    def _open_text_submission(path: Path, *, newline: str | None = None):
        return open_tabular_text(path, newline=newline)

    @staticmethod
    def _read_delimited_text_submission(path: Path, *, sep: str, **kwargs):  # type: ignore[no-untyped-def]
        import pandas as pd

        with SubmissionService._open_text_submission(path) as handle:
            frame = pd.read_csv(StringIO(handle.read()), sep=sep, **kwargs)
        frame.columns = normalize_table_column_names(frame.columns)
        return frame

    @staticmethod
    def _write_text_submission(path: Path, text: str) -> None:
        SubmissionService._write_bytes_submission(path, text.encode("utf-8"))

    @staticmethod
    def _write_bytes_submission(path: Path, payload: bytes) -> None:
        write_compressed_bytes(path, payload, suffix=tabular_suffix(path))

    @staticmethod
    def _tabular_text_base_suffix(path: Path) -> str:
        return strip_compression_suffix(tabular_suffix(path))

    def _find_usable_sample_submission_in_data_dir(self) -> Path | None:
        candidates = find_usable_sample_submissions(self._config.data_dir)
        if not candidates:
            return None
        return candidates[0]
