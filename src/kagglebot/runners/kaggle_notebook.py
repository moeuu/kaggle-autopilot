from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from rich import print

from kagglebot import kaggle_cli
from kagglebot import kernel_outputs as _kernel_outputs
from kagglebot.asset_modality import (
    ASSET_COLLECTION_DIR_NAMES,
    ASSET_COMPRESSION_SUFFIXES,
    AUDIO_SUFFIXES,
    BIO_FASTQ_BASE_SUFFIXES,
    BIO_MOL_STRUCTURE_BASE_SUFFIXES,
    BIO_PDB_STRUCTURE_BASE_SUFFIXES,
    BIO_SEQUENCE_BASE_SUFFIXES,
    DATA_ASSET_SUFFIXES,
    DICOM_IMAGE_BASE_SUFFIXES,
    DICOM_IMAGE_SUFFIXES,
    DIRECTORY_ARRAY_SUFFIXES,
    DOCUMENT_HTML_BASE_SUFFIXES,
    DOCUMENT_TEXT_METADATA_SUFFIXES,
    GRAPH_EDGE_LIST_BASE_SUFFIXES,
    GRAPH_XML_BASE_SUFFIXES,
    IMAGE_SUFFIXES,
    MEDICAL_HEADER_IMAGE_BASE_SUFFIXES,
    MEDICAL_HEADER_IMAGE_SUFFIXES,
    MODEL_ARTIFACT_COMPOUND_SUFFIXES,
    MODEL_ARTIFACT_FILENAMES,
    MODEL_ARTIFACT_JSON_SIDECAR_FILENAMES,
    MODEL_ARTIFACT_TEXT_SIDECAR_FILENAMES,
    NIFTI_IMAGE_BASE_SUFFIXES,
    NIFTI_IMAGE_SUFFIXES,
    POINT_CLOUD_TEXT_METADATA_SUFFIXES,
    VIDEO_SUFFIXES,
)
from kagglebot.baseline_tokens import (
    ASSET_LABEL_TABLE_TOKENS,
    FILE_REFERENCE_NAME_TOKENS,
    ID_LIKE_COLUMN_NAMES,
    RLE_SEGMENTATION_COLUMN_TOKENS,
    TEXT_PREDICTION_NAME_TOKENS,
)
from kagglebot.env_utils import env_flag
from kagglebot.json_utils import load_json_object, write_json_object
from kagglebot.kaggle_credentials import resolve_kaggle_username
from kagglebot.kernel_sources import KernelSourceConfig, load_kernel_source_config
from kagglebot.kernel_status import parse_kernel_status
from kagglebot.role_tokens import ROLE_TRAILING_PREFIXES, TEST_DIRECT_ROLE_ALIASES
from kagglebot.runners.base import RunContext, RunResult
from kagglebot.sample_name_aliases import SAMPLE_COMPACT_NAME_ALIASES, SAMPLE_OUTPUT_NAME_TOKENS
from kagglebot.submission_artifacts import store_submission_artifact
from kagglebot.submission_extension_hints import (
    ARCHIVE_SUBMISSION_SUFFIXES,
    CODE_FENCE_LANG_TO_SUFFIX,
    COMPRESSIBLE_SUBMISSION_KEYWORD_SUFFIXES,
    COMPRESSION_TOKEN_PATTERN_SPECS,
    NON_TABULAR_SUBMISSION_SUFFIXES,
    SUBMISSION_TOKEN_PATTERN_SPECS,
    ZSTD_TAR_ARCHIVE_SUBMISSION_SUFFIXES,
)
from kagglebot.submission_format import load_submission_format_hint
from kagglebot.submission_output_naming import (
    CONFIGURED_TEMPLATE_STEMS,
    all_submission_output_suffixes_ordered,
    configured_submission_filename_is_template,
    output_filename_from_format_text,
)
from kagglebot.submission_sample_discovery import (
    DUCKDB_TABULAR_SUFFIXES,
    ROLE_ALIASES,
    ROLE_SUFFIXES,
    SQLITE_TABULAR_SUFFIXES,
    TABULAR_ANNDATA_SUFFIXES,
    TABULAR_ARFF_SUFFIXES,
    TABULAR_ARROW_IPC_SUFFIXES,
    TABULAR_EXCEL_INPUT_ONLY_SUFFIXES,
    TABULAR_EXCEL_SUFFIXES,
    TABULAR_FITS_SUFFIXES,
    TABULAR_FIXED_WIDTH_SUFFIX_PREFIXES,
    TABULAR_GEOJSON_SUFFIXES,
    TABULAR_GEOPACKAGE_SUFFIXES,
    TABULAR_HDF_SUFFIXES,
    TABULAR_HTML_SUFFIX_PREFIXES,
    TABULAR_INPUT_SUFFIXES,
    TABULAR_INPUT_SUFFIXES_ORDERED,
    TABULAR_JSON_LINES_SUFFIX_PREFIXES,
    TABULAR_KML_SUFFIXES,
    TABULAR_LOOM_SUFFIXES,
    TABULAR_MATLAB_SUFFIXES,
    TABULAR_NETCDF_SUFFIXES,
    TABULAR_NUMPY_SUFFIXES,
    TABULAR_PARQUET_SUFFIXES,
    TABULAR_PICKLE_SUFFIXES,
    TABULAR_RDATA_SUFFIXES,
    TABULAR_SAS_SUFFIXES,
    TABULAR_SHAPEFILE_SUFFIXES,
    TABULAR_SPSS_SUFFIXES,
    TABULAR_STATA_SUFFIXES,
    TABULAR_STRUCTURED_SUFFIXES,
    TABULAR_SUBMISSION_SUFFIXES,
    TABULAR_SVMLIGHT_SUFFIX_PREFIXES,
    TABULAR_TAB_DELIMITED_SUFFIX_PREFIXES,
    TABULAR_TEXT_SUFFIXES,
)
from kagglebot.writeup import infer_code_competition_from_paths

_KAGGLE_NOTEBOOK_OUTPUT_SUFFIXES = all_submission_output_suffixes_ordered()

KERNEL_TEMPLATE = r"""
import bz2
import csv
import gzip
import itertools
import json
import lzma
import os
import re
import shutil
import sqlite3
import stat
import tarfile
import warnings
import wave
import zipfile
from contextlib import contextmanager
from datetime import UTC, datetime
from io import BytesIO, StringIO, TextIOWrapper
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # pragma: no cover - compatibility for older Kaggle images
    def root_mean_squared_error(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

COMPETITION_SLUG = "__COMPETITION_SLUG__"
ACCELERATOR = "__ACCELERATOR__"
INPUT_ROOT = Path("/kaggle/input") / COMPETITION_SLUG
WORKING_DIR = Path("/kaggle/working")
EXTRACTED_INPUT_ROOT = WORKING_DIR / "extracted_input"
SYNTHETIC_TABLE_DIR = WORKING_DIR / "synthetic_tables"
SQLITE_TABULAR_SUFFIXES = set(__SQLITE_TABULAR_SUFFIXES_JSON__)
DUCKDB_TABULAR_SUFFIXES = set(__DUCKDB_TABULAR_SUFFIXES_JSON__)
TABULAR_ANNDATA_SUFFIXES = set(__TABULAR_ANNDATA_SUFFIXES_JSON__)
ASSET_COMPRESSION_SUFFIXES = tuple(__ASSET_COMPRESSION_SUFFIXES_JSON__)
COMPRESSED_INPUT_TABULAR_SUFFIXES = tuple(
    suffix
    for suffix in __TABULAR_INPUT_SUFFIXES_ORDERED_JSON__
    if suffix.endswith((*ASSET_COMPRESSION_SUFFIXES, ".zip"))
)
TABULAR_OUTPUT_SUFFIXES = set(__TABULAR_SUBMISSION_SUFFIXES_JSON__)
FALLBACK_TABULAR_SUBMISSION_SUFFIX = ".csv"
INPUT_TABULAR_SUFFIXES = set(__TABULAR_INPUT_SUFFIXES_JSON__)
TABULAR_PICKLE_SUFFIXES = set(__TABULAR_PICKLE_SUFFIXES_JSON__)
TABULAR_STRUCTURED_SUFFIXES = set(__TABULAR_STRUCTURED_SUFFIXES_JSON__)
TABULAR_TEXT_SUFFIXES = set(__TABULAR_TEXT_SUFFIXES_JSON__)
TABULAR_ARFF_SUFFIXES = set(__TABULAR_ARFF_SUFFIXES_JSON__)
TABULAR_ARROW_IPC_SUFFIXES = set(__TABULAR_ARROW_IPC_SUFFIXES_JSON__)
TABULAR_PARQUET_SUFFIXES = set(__TABULAR_PARQUET_SUFFIXES_JSON__)
TABULAR_EXCEL_INPUT_ONLY_SUFFIXES = set(__TABULAR_EXCEL_INPUT_ONLY_SUFFIXES_JSON__)
TABULAR_EXCEL_SUFFIXES = set(__TABULAR_EXCEL_SUFFIXES_JSON__)
TABULAR_FITS_SUFFIXES = set(__TABULAR_FITS_SUFFIXES_JSON__)
TABULAR_FIXED_WIDTH_SUFFIX_PREFIXES = tuple(__TABULAR_FIXED_WIDTH_SUFFIX_PREFIXES_JSON__)
TABULAR_GEOPACKAGE_SUFFIXES = set(__TABULAR_GEOPACKAGE_SUFFIXES_JSON__)
GEOJSON_FILE_SUFFIXES = set(__TABULAR_GEOJSON_SUFFIXES_JSON__)
TABULAR_HDF_SUFFIXES = set(__TABULAR_HDF_SUFFIXES_JSON__)
TABULAR_HTML_SUFFIX_PREFIXES = tuple(__TABULAR_HTML_SUFFIX_PREFIXES_JSON__)
TABULAR_JSON_LINES_SUFFIX_PREFIXES = tuple(__TABULAR_JSON_LINES_SUFFIX_PREFIXES_JSON__)
TABULAR_KML_SUFFIXES = set(__TABULAR_KML_SUFFIXES_JSON__)
TABULAR_LOOM_SUFFIXES = set(__TABULAR_LOOM_SUFFIXES_JSON__)
TABULAR_MATLAB_SUFFIXES = set(__TABULAR_MATLAB_SUFFIXES_JSON__)
TABULAR_NETCDF_SUFFIXES = set(__TABULAR_NETCDF_SUFFIXES_JSON__)
TABULAR_NUMPY_SUFFIXES = set(__TABULAR_NUMPY_SUFFIXES_JSON__)
TABULAR_RDATA_SUFFIXES = set(__TABULAR_RDATA_SUFFIXES_JSON__)
NUMPY_COLUMN_TEXT_SIDECAR_SUFFIXES = (
    ".columns.txt",
    "_columns.txt",
    ".cols.txt",
    "_cols.txt",
    ".features.txt",
    "_features.txt",
    ".feature_names.txt",
    "_feature_names.txt",
)
NUMPY_COLUMN_JSON_SIDECAR_SUFFIXES = (
    ".columns.json",
    "_columns.json",
    ".schema.json",
    "_schema.json",
    ".features.json",
    "_features.json",
    ".feature_names.json",
    "_feature_names.json",
)
NUMPY_GENERIC_COLUMN_SIDECAR_NAMES = (
    "columns.txt",
    "feature_names.txt",
    "features.txt",
    "columns.json",
    "schema.json",
    "feature_names.json",
)
TABULAR_SAS_SUFFIXES = set(__TABULAR_SAS_SUFFIXES_JSON__)
TABULAR_SHAPEFILE_SUFFIXES = set(__TABULAR_SHAPEFILE_SUFFIXES_JSON__)
TABULAR_SPSS_SUFFIXES = set(__TABULAR_SPSS_SUFFIXES_JSON__)
TABULAR_STATA_SUFFIXES = set(__TABULAR_STATA_SUFFIXES_JSON__)
TABULAR_SVMLIGHT_SUFFIX_PREFIXES = tuple(__TABULAR_SVMLIGHT_SUFFIX_PREFIXES_JSON__)
TABULAR_TAB_DELIMITED_SUFFIX_PREFIXES = tuple(__TABULAR_TAB_DELIMITED_SUFFIX_PREFIXES_JSON__)
FILE_ASSET_SUFFIXES = set(__DATA_ASSET_SUFFIXES_JSON__)
SUBMISSION_FILE_ASSET_SUFFIXES = set(__NON_TABULAR_SUBMISSION_SUFFIXES_JSON__)
MODEL_ARTIFACT_COMPOUND_SUFFIXES = set(__MODEL_ARTIFACT_COMPOUND_SUFFIXES_JSON__)
MODEL_ARTIFACT_FILENAMES = set(__MODEL_ARTIFACT_FILENAMES_JSON__)
MODEL_ARTIFACT_JSON_SIDECAR_FILENAMES = set(__MODEL_ARTIFACT_JSON_SIDECAR_FILENAMES_JSON__)
MODEL_ARTIFACT_TEXT_SIDECAR_FILENAMES = set(__MODEL_ARTIFACT_TEXT_SIDECAR_FILENAMES_JSON__)
FILE_ASSET_SUFFIXES |= MODEL_ARTIFACT_COMPOUND_SUFFIXES
ASSET_COLLECTION_DIR_NAMES = set(__ASSET_COLLECTION_DIR_NAMES_JSON__)
ARCHIVE_OUTPUT_SUFFIXES = set(__ARCHIVE_SUBMISSION_SUFFIXES_JSON__)
ZSTD_TAR_ARCHIVE_SUFFIXES = set(__ZSTD_TAR_ARCHIVE_SUFFIXES_JSON__)
CODE_FENCE_LANG_TO_SUFFIX = __CODE_FENCE_LANG_TO_SUFFIX_JSON__
SUBMISSION_TOKEN_PATTERNS = __SUBMISSION_TOKEN_PATTERN_SPECS_JSON__
COMPRESSION_TOKEN_PATTERNS = __COMPRESSION_TOKEN_PATTERN_SPECS_JSON__
COMPRESSIBLE_SUBMISSION_KEYWORD_SUFFIXES = set(__COMPRESSIBLE_SUBMISSION_KEYWORD_SUFFIXES_JSON__)
SAMPLE_OUTPUT_NAME_TOKENS = set(__SAMPLE_OUTPUT_NAME_TOKENS_JSON__)
SAMPLE_COMPACT_NAME_ALIASES = set(__SAMPLE_COMPACT_NAME_ALIASES_JSON__)
SUBMISSION_OUTPUT_SUFFIXES = TABULAR_OUTPUT_SUFFIXES | SUBMISSION_FILE_ASSET_SUFFIXES | ARCHIVE_OUTPUT_SUFFIXES
CONFIGURED_SUBMISSION_EXCLUDED_NAMES = {"metrics.json", "plan.json"}
CONFIGURED_TEMPLATE_STEMS = set(__CONFIGURED_TEMPLATE_STEMS_JSON__)
DIRECTORY_ASSET_SUFFIXES = set(__DIRECTORY_ASSET_SUFFIXES_JSON__)
IMAGE_FILE_SUFFIXES = set(__IMAGE_SUFFIXES_JSON__)
AUDIO_FILE_SUFFIXES = set(__AUDIO_SUFFIXES_JSON__)
VIDEO_FILE_SUFFIXES = set(__VIDEO_SUFFIXES_JSON__)
DICOM_FILE_BASE_SUFFIXES = set(__DICOM_BASE_SUFFIXES_JSON__)
DICOM_FILE_SUFFIXES = set(__DICOM_SUFFIXES_JSON__)
NIFTI_FILE_BASE_SUFFIXES = set(__NIFTI_BASE_SUFFIXES_JSON__)
NIFTI_FILE_SUFFIXES = set(__NIFTI_SUFFIXES_JSON__)
MEDICAL_HEADER_FILE_BASE_SUFFIXES = set(__MEDICAL_HEADER_BASE_SUFFIXES_JSON__)
MEDICAL_HEADER_FILE_SUFFIXES = set(__MEDICAL_HEADER_SUFFIXES_JSON__)
POINT_CLOUD_TEXT_METADATA_SUFFIXES = set(__POINT_CLOUD_TEXT_METADATA_SUFFIXES_JSON__)
GRAPH_XML_BASE_SUFFIXES = set(__GRAPH_XML_BASE_SUFFIXES_JSON__)
GRAPH_EDGE_LIST_BASE_SUFFIXES = set(__GRAPH_EDGE_LIST_BASE_SUFFIXES_JSON__)
DOCUMENT_HTML_BASE_SUFFIXES = set(__DOCUMENT_HTML_BASE_SUFFIXES_JSON__)
DOCUMENT_TEXT_METADATA_SUFFIXES = set(__DOCUMENT_TEXT_METADATA_SUFFIXES_JSON__)
BIO_SEQUENCE_BASE_SUFFIXES = set(__BIO_SEQUENCE_BASE_SUFFIXES_JSON__)
BIO_FASTQ_BASE_SUFFIXES = set(__BIO_FASTQ_BASE_SUFFIXES_JSON__)
BIO_PDB_STRUCTURE_BASE_SUFFIXES = set(__BIO_PDB_STRUCTURE_BASE_SUFFIXES_JSON__)
BIO_MOL_STRUCTURE_BASE_SUFFIXES = set(__BIO_MOL_STRUCTURE_BASE_SUFFIXES_JSON__)
ROLE_SUFFIXES = set(__ROLE_SUFFIXES_JSON__)
ROLE_ALIASES = {key: set(value) for key, value in __ROLE_ALIASES_JSON__.items()}
TEST_DIRECT_ROLE_ALIASES = set(__TEST_DIRECT_ROLE_ALIASES_JSON__)
ROLE_TRAILING_PREFIXES = set(__ROLE_TRAILING_PREFIXES_JSON__)
FILE_REFERENCE_NAME_TOKENS = tuple(__FILE_REFERENCE_NAME_TOKENS_JSON__)
METRICS_PATH = WORKING_DIR / "metrics.json"
SAMPLE_STAGE_RE = re.compile(r"(?:stage|phase|round)[_-]?(\d+)", re.IGNORECASE)
TEXT_PREDICTION_NAME_TOKENS = tuple(__TEXT_PREDICTION_NAME_TOKENS_JSON__)
RLE_SEGMENTATION_COLUMN_TOKENS = set(__RLE_SEGMENTATION_COLUMN_TOKENS_JSON__)
ID_LIKE_COLUMN_NAMES = set(__ID_LIKE_COLUMN_NAMES_JSON__)
ASSET_LABEL_TABLE_TOKENS = tuple(__ASSET_LABEL_TABLE_TOKENS_JSON__)


def compression_suffix_for(suffix: str) -> str | None:
    normalized = str(suffix or "").strip().lower()
    for compression_suffix in ASSET_COMPRESSION_SUFFIXES:
        if normalized.endswith(compression_suffix):
            return compression_suffix
    return None


def read_compressed_bytes(path: Path, *, suffix: str | None = None, limit: int | None = None) -> bytes:
    compression_suffix = compression_suffix_for(suffix or path.name)
    if compression_suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            return handle.read(limit) if limit is not None else handle.read()
    if compression_suffix == ".bz2":
        with bz2.open(path, "rb") as handle:
            return handle.read(limit) if limit is not None else handle.read()
    if compression_suffix == ".xz":
        with lzma.open(path, "rb") as handle:
            return handle.read(limit) if limit is not None else handle.read()
    if compression_suffix == ".zst":
        import zstandard as zstd

        with path.open("rb") as raw:
            with zstd.ZstdDecompressor().stream_reader(raw) as reader:
                return reader.read(limit) if limit is not None else reader.read()
    if limit is not None:
        with path.open("rb") as handle:
            return handle.read(limit)
    return path.read_bytes()


def open_compressed_text(path: Path, *, suffix: str | None = None, errors: str = "ignore"):
    compression_suffix = compression_suffix_for(suffix or path.name)
    if compression_suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors=errors)
    if compression_suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8", errors=errors)
    if compression_suffix == ".xz":
        return lzma.open(path, "rt", encoding="utf-8", errors=errors)
    if compression_suffix == ".zst":
        import zstandard as zstd

        return zstd.open(path, "rt", encoding="utf-8", errors=errors)
    return path.open("rt", encoding="utf-8", errors=errors)


@contextmanager
def open_compressed_binary(path: Path, *, suffix: str | None = None):
    compression_suffix = compression_suffix_for(suffix or path.name)
    if compression_suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            yield handle
        return
    if compression_suffix == ".bz2":
        with bz2.open(path, "rb") as handle:
            yield handle
        return
    if compression_suffix == ".xz":
        with lzma.open(path, "rb") as handle:
            yield handle
        return
    if compression_suffix == ".zst":
        yield BytesIO(read_compressed_bytes(path, suffix=suffix))
        return
    with path.open("rb") as handle:
        yield handle


def filename_tabular_suffix(name: str) -> str:
    lowered = str(name).lower()
    for suffix in sorted(COMPRESSED_INPUT_TABULAR_SUFFIXES, key=len, reverse=True):
        if lowered.endswith(suffix):
            return suffix
    return Path(name).suffix.lower()


def configured_submission_filename_is_template(name: str) -> bool:
    path = Path(str(name or "").strip())
    lowered = path.name.lower()
    suffix = next(
        (
            candidate
            for candidate in sorted(SUBMISSION_OUTPUT_SUFFIXES, key=len, reverse=True)
            if lowered.endswith(candidate)
        ),
        Path(lowered).suffix.lower(),
    )
    stem = lowered[: -len(suffix)] if suffix and lowered.endswith(suffix) else Path(lowered).stem
    stem = stem.strip(".")
    if stem in CONFIGURED_TEMPLATE_STEMS:
        return True
    compact = stem.replace("-", "").replace("_", "")
    if compact in SAMPLE_COMPACT_NAME_ALIASES:
        return True
    tokens = {token for token in stem.replace("-", "_").split("_") if token}
    template_tokens = {"example", "sample", "template"}
    return bool(tokens & template_tokens and tokens & SAMPLE_OUTPUT_NAME_TOKENS)


def resolve_submission_path(sample_path: Path | None = None) -> Path:
    global FALLBACK_TABULAR_SUBMISSION_SUFFIX
    default_name = "submission.csv"
    if sample_path is not None:
        sample_suffix = tabular_suffix(sample_path)
        if sample_suffix in TABULAR_OUTPUT_SUFFIXES:
            default_name = f"submission{sample_suffix}"
    default_suffix = filename_tabular_suffix(default_name)
    if default_suffix in TABULAR_OUTPUT_SUFFIXES:
        FALLBACK_TABULAR_SUBMISSION_SUFFIX = default_suffix
    raw = str(os.getenv("KAGGLEBOT_SUBMISSION_FILENAME") or default_name).strip()
    name = Path(raw).name
    if (
        not name
        or name.lower() in CONFIGURED_SUBMISSION_EXCLUDED_NAMES
        or configured_submission_filename_is_template(name)
    ):
        name = default_name
    suffix = filename_tabular_suffix(name)
    if suffix not in SUBMISSION_OUTPUT_SUFFIXES:
        lowered = name.lower()
        suffix = next(
            (
                candidate
                for candidate in sorted(SUBMISSION_FILE_ASSET_SUFFIXES | ARCHIVE_OUTPUT_SUFFIXES, key=len, reverse=True)
                if lowered.endswith(candidate)
            ),
            Path(name).suffix.lower(),
        )
    if not name or suffix not in SUBMISSION_OUTPUT_SUFFIXES:
        name = default_name
        suffix = filename_tabular_suffix(name)
    if suffix in TABULAR_OUTPUT_SUFFIXES:
        FALLBACK_TABULAR_SUBMISSION_SUFFIX = suffix
    return WORKING_DIR / name


SUBMISSION_PATH = resolve_submission_path()


def data_roots() -> list[Path]:
    roots = [INPUT_ROOT]
    if EXTRACTED_INPUT_ROOT.exists():
        roots.append(EXTRACTED_INPUT_ROOT)
    return roots


def safe_extract_path(dest_dir: Path, member_name: str) -> Path:
    dest_dir = dest_dir.resolve()
    candidate = (dest_dir / member_name).resolve()
    try:
        candidate.relative_to(dest_dir)
    except ValueError as exc:
        raise ValueError(f"unsafe archive path: {member_name}") from exc
    return candidate


def remember_archive_target(seen_targets: set[Path], target: Path, member_name: str) -> None:
    if target in seen_targets:
        raise ValueError(f"duplicate archive member target: {member_name}")
    seen_targets.add(target)


def zip_member_is_symlink(member: zipfile.ZipInfo) -> bool:
    if member.create_system != 3:
        return False
    return stat.S_IFMT(member.external_attr >> 16) == stat.S_IFLNK


def safe_extract_zip(zip_path: Path, dest_dir: Path, *, overwrite: bool = False) -> list[Path]:
    extracted = []
    with zipfile.ZipFile(zip_path) as archive:
        seen_targets = set()
        for member in archive.infolist():
            target = safe_extract_path(dest_dir, member.filename)
            remember_archive_target(seen_targets, target, member.filename)
            if member.flag_bits & 0x1:
                raise ValueError(f"unsupported encrypted zip member: {member.filename}")
            if zip_member_is_symlink(member):
                raise ValueError(f"unsupported zip symlink member: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists() and not overwrite:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            extracted.append(target)
    return extracted


def safe_extract_tar(tar_path: Path, dest_dir: Path, *, overwrite: bool = False) -> list[Path]:
    with tarfile.open(tar_path, "r:*") as archive:
        return safe_extract_tar_members(archive, dest_dir, overwrite=overwrite)


def safe_extract_tar_zst(tar_path: Path, dest_dir: Path, *, overwrite: bool = False) -> list[Path]:
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot extract {tar_path.name}: zstandard is not available in this Kaggle runtime."
        ) from exc
    with tar_path.open("rb") as raw:
        with zstd.ZstdDecompressor().stream_reader(raw) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as archive:
                return safe_extract_tar_members(archive, dest_dir, overwrite=overwrite)


def safe_extract_tar_members(archive, dest_dir: Path, *, overwrite: bool = False) -> list[Path]:
    extracted = []
    seen_targets = set()
    for member in archive:
        target = safe_extract_path(dest_dir, member.name)
        remember_archive_target(seen_targets, target, member.name)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not member.isfile():
            raise ValueError(f"unsupported tar member type: {member.name}")
        if target.exists() and not overwrite:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            continue
        with source, target.open("wb") as dst:
            shutil.copyfileobj(source, dst, length=1024 * 1024)
        extracted.append(target)
    return extracted


def safe_extract_7z(archive_path: Path, dest_dir: Path, *, overwrite: bool = False) -> list[Path]:
    try:
        import py7zr
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot extract {archive_path.name}: py7zr is not available in this Kaggle runtime."
        ) from exc
    extracted = []
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        targets = []
        seen_targets = set()
        for member in archive.list():
            name = getattr(member, "filename", "")
            target = safe_extract_path(dest_dir, name)
            remember_archive_target(seen_targets, target, name)
            if getattr(member, "is_directory", False):
                continue
            if getattr(member, "is_symlink", False) or not getattr(member, "is_file", False):
                raise ValueError(f"unsupported 7z member type: {name}")
            if target.exists() and not overwrite:
                continue
            targets.append(name)
            extracted.append(target)
        if targets:
            archive.extract(path=dest_dir, targets=targets)
    return extracted


def safe_extract_rar(archive_path: Path, dest_dir: Path, *, overwrite: bool = False) -> list[Path]:
    try:
        import rarfile
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot extract {archive_path.name}: rarfile is not available in this Kaggle runtime."
        ) from exc
    extracted = []
    with rarfile.RarFile(archive_path) as archive:
        seen_targets = set()
        for member in archive.infolist():
            member_name = getattr(member, "filename", "")
            target = safe_extract_path(dest_dir, member_name)
            remember_archive_target(seen_targets, target, member_name)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if member.needs_password():
                raise ValueError(f"unsupported password-protected rar member: {member_name}")
            if member.is_symlink() or not member.is_file():
                raise ValueError(f"unsupported rar member type: {member_name}")
            if target.exists() and not overwrite:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            extracted.append(target)
    return extracted


def is_supported_archive(path: Path) -> bool:
    return archive_output_suffix(path) in ARCHIVE_OUTPUT_SUFFIXES


def archive_output_suffix(path: Path) -> str:
    name = path.name.lower()
    for suffix in sorted(ARCHIVE_OUTPUT_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix):
            return suffix
    return ""


def is_zstd_tar_archive(path: Path) -> bool:
    return archive_output_suffix(path) in ZSTD_TAR_ARCHIVE_SUFFIXES


def extract_data_archives(max_depth: int = 2) -> list[Path]:
    EXTRACTED_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    extracted = []
    processed = set()
    for _depth in range(max(max_depth, 0) + 1):
        roots = data_roots()
        archives = [
            path
            for root in roots
            for path in sorted(root.rglob("*"))
            if path.is_file() and is_supported_archive(path) and path.resolve() not in processed
        ]
        if not archives:
            break
        for archive_path in archives:
            processed.add(archive_path.resolve())
            suffix = archive_output_suffix(archive_path)
            if suffix == ".zip":
                extracted.extend(safe_extract_zip(archive_path, EXTRACTED_INPUT_ROOT, overwrite=False))
            elif suffix == ".7z":
                extracted.extend(safe_extract_7z(archive_path, EXTRACTED_INPUT_ROOT, overwrite=False))
            elif suffix == ".rar":
                extracted.extend(safe_extract_rar(archive_path, EXTRACTED_INPUT_ROOT, overwrite=False))
            elif suffix in ZSTD_TAR_ARCHIVE_SUFFIXES:
                extracted.extend(safe_extract_tar_zst(archive_path, EXTRACTED_INPUT_ROOT, overwrite=False))
            else:
                extracted.extend(safe_extract_tar(archive_path, EXTRACTED_INPUT_ROOT, overwrite=False))
    return extracted


def find_tabular_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and tabular_suffix(p) in INPUT_TABULAR_SUFFIXES]


def tabular_suffix(path: Path) -> str:
    name = path.name.lower()
    for suffix in sorted(COMPRESSED_INPUT_TABULAR_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix):
            return suffix
    return path.suffix.lower()


def materialize_sqlite_tables(root: Path) -> list[Path]:
    if not root.exists():
        return []
    cache_dir = WORKING_DIR / "sqlite_tables" / safe_name(root.name)
    materialized = []
    for sqlite_path in root.rglob("*"):
        if not sqlite_path.is_file() or sqlite_path.suffix.lower() not in SQLITE_TABULAR_SUFFIXES:
            continue
        if "sqlite_tables" in {part.lower() for part in sqlite_path.parts}:
            continue
        try:
            tables = sqlite_user_tables(sqlite_path)
        except Exception:
            continue
        selected = select_sqlite_tables_for_materialization(tables)
        for table in selected:
            cache_dir.mkdir(parents=True, exist_ok=True)
            destination = cache_dir / f"{safe_name(sqlite_path.stem)}__{safe_name(table)}.csv"
            try:
                frame = read_sqlite_table(sqlite_path, table=table)
            except Exception:
                continue
            if frame.empty or len(frame.columns) < 2:
                continue
            frame.to_csv(destination, index=False)
            materialized.append(destination)
    return materialized


def materialize_duckdb_tables(root: Path) -> list[Path]:
    if not root.exists():
        return []
    cache_dir = WORKING_DIR / "duckdb_tables" / safe_name(root.name)
    materialized = []
    for duckdb_path in root.rglob("*"):
        if not duckdb_path.is_file() or duckdb_path.suffix.lower() not in DUCKDB_TABULAR_SUFFIXES:
            continue
        if "duckdb_tables" in {part.lower() for part in duckdb_path.parts}:
            continue
        try:
            tables = duckdb_user_tables(duckdb_path)
        except Exception:
            continue
        selected = select_duckdb_tables_for_materialization(tables)
        for table in selected:
            cache_dir.mkdir(parents=True, exist_ok=True)
            destination = cache_dir / f"{safe_name(duckdb_path.stem)}__{safe_duckdb_table_name(table)}.csv"
            try:
                frame = read_duckdb_table(duckdb_path, table=table)
            except Exception:
                continue
            if frame.empty or len(frame.columns) < 2:
                continue
            frame.to_csv(destination, index=False)
            materialized.append(destination)
    return materialized


def sqlite_user_tables(path: Path) -> list[str]:
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            "SELECT name "
            "FROM sqlite_master "
            "WHERE type IN ('table', 'view') "
            "AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
    return [str(row[0]) for row in rows]


def select_sqlite_tables_for_materialization(tables: list[str]) -> list[str]:
    role_tables = [table for table in tables if sqlite_table_has_role_name(table)]
    if role_tables:
        return role_tables
    if len(tables) <= 3:
        return list(tables)
    return []


def sqlite_table_has_role_name(table: str) -> bool:
    lowered = table.lower().replace("-", "_")
    compact = lowered.replace("_", "")
    tokens = (
        "sample_submission", "samplesubmission", "submission",
        "train", "training", "test", "labels", "label",
        "target", "features", "feature", "data",
    )
    return any(token in lowered or token in compact for token in tokens)


def read_sqlite_table(path: Path, table: str | None = None, nrows: int | None = None) -> pd.DataFrame:
    with sqlite3.connect(path) as conn:
        table_name = table or select_sqlite_table_for_path(path, sqlite_user_tables_from_connection(conn))
        if table_name is None:
            raise ValueError(f"No user tables found in SQLite database: {path}")
        query = f"SELECT * FROM {quote_sqlite_identifier(table_name)}"
        if nrows is not None:
            query += f" LIMIT {max(int(nrows), 0)}"
        return pd.read_sql_query(query, conn)


def duckdb_user_tables(path: Path) -> list[tuple[str, str]]:
    import duckdb

    conn = duckdb.connect(str(path), read_only=True)
    try:
        return duckdb_user_tables_from_connection(conn)
    finally:
        conn.close()


def duckdb_user_tables_from_connection(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        '''
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
          AND table_type IN ('BASE TABLE', 'VIEW')
        ORDER BY table_schema, table_name
        '''
    ).fetchall()
    return [(str(schema), str(table)) for schema, table in rows if str(table).strip()]


def select_duckdb_tables_for_materialization(tables: list[tuple[str, str]]) -> list[tuple[str, str]]:
    role_tables = [table for table in tables if sqlite_table_has_role_name(duckdb_table_label(table))]
    if role_tables:
        return role_tables
    if len(tables) <= 3:
        return list(tables)
    return []


def read_duckdb_table(
    path: Path,
    table: tuple[str, str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    import duckdb

    conn = duckdb.connect(str(path), read_only=True)
    try:
        table_name = table or select_duckdb_table_for_path(path, duckdb_user_tables_from_connection(conn))
        if table_name is None:
            raise ValueError(f"No user tables found in DuckDB database: {path}")
        query = f"SELECT * FROM {quote_duckdb_qualified_identifier(table_name)}"
        if nrows is not None:
            query += f" LIMIT {max(int(nrows), 0)}"
        return conn.execute(query).df()
    finally:
        conn.close()


def read_geopackage_tabular_frame(path: Path, nrows: int | None = None) -> pd.DataFrame:
    with sqlite3.connect(path) as conn:
        table_name = first_geopackage_table(conn)
        if table_name is None:
            raise ValueError(f"No feature or attribute tables found in GeoPackage: {path}")
        query = f"SELECT * FROM {quote_sqlite_identifier(table_name)}"
        if nrows is not None:
            query += f" LIMIT {max(int(nrows), 0)}"
        frame = pd.read_sql_query(query, conn)
    return decode_geopackage_blob_columns(frame)


def first_geopackage_table(conn) -> str | None:
    preferred = []
    try:
        rows = conn.execute(
            "SELECT table_name, data_type FROM gpkg_contents ORDER BY table_name"
        ).fetchall()
        for table_name, data_type in rows:
            table = str(table_name)
            if str(data_type).lower() in {"features", "attributes"} and sqlite_table_exists(conn, table):
                preferred.append(table)
    except sqlite3.DatabaseError:
        pass
    if preferred:
        return preferred[0]
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name"
    ).fetchall()
    for row in rows:
        table = str(row[0])
        if not is_geopackage_metadata_table(table):
            return table
    return None


def sqlite_table_exists(conn, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def is_geopackage_metadata_table(table: str) -> bool:
    lowered = table.lower()
    return lowered.startswith(("gpkg_", "sqlite_", "rtree_")) or lowered.endswith("_rtree")


def decode_geopackage_blob_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in frame.columns:
        if frame[column].map(lambda value: isinstance(value, (bytes, bytearray, memoryview))).any():
            frame[column] = frame[column].map(decode_geopackage_blob_value)
    return frame


def decode_geopackage_blob_value(value):
    if isinstance(value, memoryview):
        return value.tobytes().hex()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    return value


def read_shapefile_tabular_frame(path: Path, nrows: int | None = None) -> pd.DataFrame:
    dbf_path = dbf_path_for_shapefile(path)
    fields, records = read_dbf_records(dbf_path, nrows=nrows)
    columns = dedupe_column_names([field["name"] for field in fields])
    return pd.DataFrame(records, columns=columns)


def dbf_path_for_shapefile(path: Path) -> Path:
    if path.suffix.lower() == ".dbf":
        return path
    exact = path.with_suffix(".dbf")
    if exact.exists():
        return exact
    lowered_stem = path.stem.lower()
    for candidate in path.parent.iterdir():
        if candidate.is_file() and candidate.suffix.lower() == ".dbf" and candidate.stem.lower() == lowered_stem:
            return candidate
    raise ValueError(f"No .dbf sidecar found for Shapefile: {path}")


def read_dbf_records(path: Path, nrows: int | None = None):
    encoding = dbf_encoding_for_path(path)
    with path.open("rb") as handle:
        header = handle.read(32)
        if len(header) < 32:
            raise ValueError(f"Invalid DBF header: {path}")
        record_count = int.from_bytes(header[4:8], "little")
        header_length = int.from_bytes(header[8:10], "little")
        record_length = int.from_bytes(header[10:12], "little")
        fields = read_dbf_fields(handle, encoding=encoding)
        handle.seek(header_length)
        records = []
        limit = None if nrows is None else max(int(nrows), 0)
        for _ in range(record_count):
            raw_record = handle.read(record_length)
            if len(raw_record) < record_length:
                break
            if raw_record[:1] == b"*":
                continue
            offset = 1
            values = []
            for field in fields:
                length = int(field["length"])
                values.append(
                    parse_dbf_value(
                        raw_record[offset:offset + length],
                        field_type=str(field["type"]),
                        decimal_count=int(field["decimal_count"]),
                        encoding=encoding,
                    )
                )
                offset += length
            records.append(values)
            if limit is not None and len(records) >= limit:
                break
    return fields, records


def read_dbf_fields(handle, encoding: str) -> list[dict]:
    fields = []
    while True:
        marker = handle.read(1)
        if not marker or marker == b"\r":
            break
        descriptor = marker + handle.read(31)
        if len(descriptor) < 32:
            break
        name = descriptor[:11].split(b"\x00", 1)[0].decode(encoding, errors="replace").strip()
        if not name:
            continue
        fields.append(
            {
                "name": name,
                "type": chr(descriptor[11]),
                "length": int(descriptor[16]),
                "decimal_count": int(descriptor[17]),
            }
        )
    return fields


def dbf_encoding_for_path(path: Path) -> str:
    cpg_path = path.with_suffix(".cpg")
    if cpg_path.exists():
        raw_encoding = cpg_path.read_text(encoding="ascii", errors="ignore").strip()
        if raw_encoding:
            return "utf-8" if raw_encoding == "65001" else raw_encoding
    return "latin1"


def parse_dbf_value(raw_value: bytes, field_type: str, decimal_count: int, encoding: str):
    text = raw_value.decode(encoding, errors="replace").strip()
    if not text:
        return None
    normalized_type = field_type.upper()
    if normalized_type in {"N", "F", "B"}:
        try:
            return float(text) if decimal_count or any(marker in text for marker in (".", "e", "E")) else int(text)
        except ValueError:
            return text
    if normalized_type == "L":
        marker = text[:1].upper()
        if marker in {"Y", "T"}:
            return True
        if marker in {"N", "F"}:
            return False
        return None
    if normalized_type == "D" and len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def read_kml_tabular_frame(path: Path) -> pd.DataFrame:
    from xml.etree import ElementTree as ET

    root = ET.fromstring(read_kml_payload(path))
    placemarks = [node for node in root.iter() if xml_local_name(node.tag) == "Placemark"]
    records = [kml_placemark_to_record(placemark, index=idx, ET=ET) for idx, placemark in enumerate(placemarks)]
    return pd.DataFrame(records)


def read_kml_payload(path: Path) -> bytes:
    suffix = file_asset_suffix(path)
    if suffix == ".kmz":
        with zipfile.ZipFile(path, "r") as archive:
            members = [
                member
                for member in archive.infolist()
                if (
                    not member.is_dir()
                    and not member.filename.startswith("__MACOSX/")
                    and member.filename.lower().endswith(".kml")
                )
            ]
            if not members:
                raise ValueError(f"KMZ archive does not contain a KML document: {path}")
            members.sort(key=lambda member: (Path(member.filename).name.lower() != "doc.kml", member.filename.lower()))
            return archive.read(members[0])
    return read_compressed_bytes(path, suffix=suffix)


def kml_placemark_to_record(placemark, index: int, ET) -> dict:
    record = {"placemark_index": index}
    for child in list(placemark):
        local_name = xml_local_name(child.tag)
        if local_name in {"name", "description"} and child.text and child.text.strip():
            record[local_name] = child.text.strip()
    for data in placemark.iter():
        local_name = xml_local_name(data.tag)
        if local_name == "Data":
            key = str(data.attrib.get("name", "")).strip()
            value = first_kml_child_text(data, "value")
            if key and value is not None:
                record[key] = value
        elif local_name == "SimpleData":
            key = str(data.attrib.get("name", "")).strip()
            if key and data.text is not None:
                record[key] = data.text.strip()
    geometry = first_kml_geometry_node(placemark)
    if geometry is not None:
        record["geometry_type"] = xml_local_name(geometry.tag)
        coordinates = [
            text
            for node in geometry.iter()
            if xml_local_name(node.tag) == "coordinates" and (text := (node.text or "").strip())
        ]
        if coordinates:
            record["coordinates"] = " ".join(coordinates)
        record["geometry"] = ET.tostring(geometry, encoding="unicode")
    return record


def first_kml_child_text(node, child_name: str) -> str | None:
    for child in list(node):
        if xml_local_name(child.tag) == child_name and child.text is not None:
            return child.text.strip()
    return None


def first_kml_geometry_node(placemark):
    geometry_tags = {
        "Point",
        "LineString",
        "LinearRing",
        "Polygon",
        "MultiGeometry",
        "Model",
        "Track",
        "MultiTrack",
    }
    for node in placemark.iter():
        if xml_local_name(node.tag) in geometry_tags:
            return node
    return None


def xml_local_name(tag) -> str:
    text = str(tag)
    return text.rsplit("}", 1)[-1] if "}" in text else text


def sqlite_user_tables_from_connection(conn) -> list[str]:
    rows = conn.execute(
        "SELECT name "
        "FROM sqlite_master "
        "WHERE type IN ('table', 'view') "
        "AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    ).fetchall()
    return [str(row[0]) for row in rows]


def select_sqlite_table_for_path(path: Path, tables: list[str]) -> str | None:
    if not tables:
        return None
    if len(tables) == 1:
        return str(tables[0])
    path_tokens = sqlite_name_tokens(path.stem)
    ranked = []
    for table in tables:
        table_tokens = sqlite_name_tokens(table)
        score = len(path_tokens & table_tokens) * 20
        if sqlite_table_has_role_name(table):
            score += 5
        ranked.append((score, table))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked[0][1]


def select_duckdb_table_for_path(path: Path, tables: list[tuple[str, str]]) -> tuple[str, str] | None:
    if not tables:
        return None
    if len(tables) == 1:
        return tables[0]
    path_tokens = sqlite_name_tokens(path.stem)
    ranked = []
    for table in tables:
        label = duckdb_table_label(table)
        table_tokens = sqlite_name_tokens(label)
        score = len(path_tokens & table_tokens) * 20
        if sqlite_table_has_role_name(label):
            score += 5
        ranked.append((score, label, table))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked[0][2]


def sqlite_name_tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", value.lower()) if token}


def quote_sqlite_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def duckdb_table_label(table: tuple[str, str]) -> str:
    schema, name = table
    if schema and schema != "main":
        return f"{schema}.{name}"
    return name


def quote_duckdb_qualified_identifier(table: tuple[str, str]) -> str:
    schema, name = table
    if schema and schema != "main":
        return f"{quote_sqlite_identifier(schema)}.{quote_sqlite_identifier(name)}"
    return quote_sqlite_identifier(name)


def safe_duckdb_table_name(table: tuple[str, str]) -> str:
    schema, name = table
    if schema and schema != "main":
        return f"{safe_name(schema)}__{safe_name(name)}"
    return safe_name(name)


def safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return safe or "table"


def build_file_asset_index(roots: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in roots:
        for path in root.rglob("*"):
            if not is_file_asset_path(path):
                continue
            try:
                rel = path.relative_to(root).as_posix().lower()
            except ValueError:
                rel = path.name.lower()
            suffix = file_asset_suffix(path)
            keys = [rel, remove_asset_suffix(rel, suffix), path.name.lower(), file_asset_stem(path).lower()]
            parts = Path(rel).parts
            for start in range(1, len(parts)):
                suffix_path = Path(*parts[start:]).as_posix()
                keys.append(suffix_path)
                keys.append(remove_asset_suffix(suffix_path, suffix))
            seen_keys = set()
            for key in keys:
                if not key or key in seen_keys:
                    continue
                seen_keys.add(key)
                index.setdefault(key, path)
    return index


def is_file_asset_path(path: Path) -> bool:
    if path.name.lower() in MODEL_ARTIFACT_FILENAMES:
        return path.is_file()
    suffix = file_asset_suffix(path)
    if suffix not in FILE_ASSET_SUFFIXES:
        return False
    return path.is_file() or (path.is_dir() and suffix in DIRECTORY_ASSET_SUFFIXES)


def looks_like_file_reference_column(frame: pd.DataFrame, col: str) -> bool:
    lowered = str(col).lower()
    if any(token in lowered for token in FILE_REFERENCE_NAME_TOKENS):
        return True
    if col not in frame.columns:
        return False
    values = frame[col].dropna().astype(str).str.strip().head(200)
    if values.empty:
        return False
    suffix_hits = values.map(lambda value: file_asset_suffix(Path(value)) in FILE_ASSET_SUFFIXES)
    return float(suffix_hits.mean()) >= 0.5


def file_asset_suffix(path: Path) -> str:
    name = path.name.lower()
    for suffix in sorted(MODEL_ARTIFACT_COMPOUND_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix):
            return suffix
    for suffix in sorted(FILE_ASSET_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix):
            return suffix
    return path.suffix.lower()


def file_asset_stem(path: Path) -> str:
    suffix = file_asset_suffix(path)
    return remove_asset_suffix(path.name, suffix)


def remove_asset_suffix(value: str, suffix: str) -> str:
    if suffix and value.lower().endswith(suffix):
        return value[: -len(suffix)]
    return str(Path(value).with_suffix(""))


def asset_split_aliases(split: str) -> list[str]:
    aliases = [str(split or "").strip().lower()]
    aliases.extend(sorted(ROLE_ALIASES.get(aliases[0], set())))
    seen = set()
    ordered = []
    for alias in aliases:
        if alias and alias not in seen:
            seen.add(alias)
            ordered.append(alias)
    return ordered


def resolve_asset_path(value, asset_index: dict[str, Path], *, split: str) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute() and path.exists():
        return path
    basename = path.name.lower()
    normalized = raw.replace("\\", "/").lstrip("./").lower()
    split_aliases = asset_split_aliases(split)
    collection_candidates = []
    for collection in sorted(ASSET_COLLECTION_DIR_NAMES):
        for split_alias in split_aliases:
            collection_candidates.extend(
                [
                    f"{collection}/{split_alias}/{normalized}",
                    f"{collection}/{split_alias}/{basename}",
                ]
            )
    candidates = [
        *[f"{split_alias}/{normalized}" for split_alias in split_aliases],
        *[f"{split_alias}/{basename}" for split_alias in split_aliases],
        *collection_candidates,
        normalized,
        basename,
    ]
    for key in candidates:
        match = asset_index.get(key)
        if match is not None:
            return match
    direct_candidates = []
    for root in data_roots():
        direct_candidates.append(root / raw)
        direct_candidates.extend(root / split_alias / raw for split_alias in split_aliases)
        for collection in sorted(ASSET_COLLECTION_DIR_NAMES):
            for split_alias in split_aliases:
                direct_candidates.extend(
                    [
                        root / collection / split_alias / raw,
                        root / collection / split_alias / basename,
                    ]
                )
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate
    return None


def image_size(path: Path | None) -> tuple[float, float]:
    if path is None or file_asset_suffix(path) not in IMAGE_FILE_SUFFIXES:
        return np.nan, np.nan
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
        return float(width), float(height)
    except Exception:
        return np.nan, np.nan


def image_metadata(path: Path | None) -> tuple[float, float, float, float, float]:
    if path is None or file_asset_suffix(path) not in IMAGE_FILE_SUFFIXES:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
            channels = len(image.getbands())
        pixels = float(width * height)
        aspect = float(width / height) if height else np.nan
        return float(width), float(height), float(channels), pixels, aspect
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan


def mask_image_metadata(path: Path | None) -> tuple[float, float, float]:
    if path is None or file_asset_suffix(path) not in IMAGE_FILE_SUFFIXES:
        return np.nan, np.nan, np.nan
    try:
        from PIL import Image

        with Image.open(path) as image:
            array = np.asarray(image)
        if array.size == 0:
            return 0.0, np.nan, np.nan
        if array.ndim >= 3:
            foreground = np.any(array != 0, axis=-1)
            labels = np.nan
        else:
            foreground = array != 0
            labels = float(len(np.unique(array[foreground]))) if np.any(foreground) else 0.0
        nonzero = float(np.count_nonzero(foreground))
        coverage = float(nonzero / foreground.size) if foreground.size else np.nan
        return nonzero, coverage, labels
    except Exception:
        return np.nan, np.nan, np.nan


def image_intensity_metadata(path: Path | None) -> tuple[float, float, float]:
    if path is None or file_asset_suffix(path) not in IMAGE_FILE_SUFFIXES:
        return np.nan, np.nan, np.nan
    try:
        from PIL import Image

        with Image.open(path) as image:
            gray = image.convert("L")
            gray.thumbnail((128, 128))
            array = np.asarray(gray, dtype=np.float32)
        if array.size == 0:
            return np.nan, np.nan, np.nan
        return float(array.mean()), float(array.std()), float(np.count_nonzero(array) / array.size)
    except Exception:
        return np.nan, np.nan, np.nan


def image_frame_count(path: Path | None) -> float:
    if path is None or file_asset_suffix(path) not in IMAGE_FILE_SUFFIXES:
        return np.nan
    try:
        from PIL import Image

        with Image.open(path) as image:
            return float(getattr(image, "n_frames", 1))
    except Exception:
        return np.nan


def wav_duration_seconds(path: Path | None) -> float:
    if path is None or path.suffix.lower() != ".wav":
        return np.nan
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return float(handle.getnframes() / rate) if rate else np.nan
    except Exception:
        return np.nan


def audio_metadata(path: Path | None) -> tuple[float, float, float, float]:
    if path is None or file_asset_suffix(path) not in AUDIO_FILE_SUFFIXES:
        return np.nan, np.nan, np.nan, np.nan
    try:
        import soundfile as sf

        info = sf.info(str(path))
        frames = float(getattr(info, "frames", np.nan))
        samplerate = float(getattr(info, "samplerate", np.nan))
        channels = float(getattr(info, "channels", np.nan))
        seconds = float(frames / samplerate) if samplerate and not np.isnan(frames) else np.nan
        return seconds, samplerate, channels, frames
    except Exception:
        if path.suffix.lower() == ".wav":
            seconds = wav_duration_seconds(path)
            try:
                with wave.open(str(path), "rb") as handle:
                    return (
                        seconds,
                        float(handle.getframerate()),
                        float(handle.getnchannels()),
                        float(handle.getnframes()),
                    )
            except Exception:
                return seconds, np.nan, np.nan, np.nan
        return np.nan, np.nan, np.nan, np.nan


def video_metadata(path: Path | None) -> tuple[float, float, float, float, float]:
    if path is None or file_asset_suffix(path) not in VIDEO_FILE_SUFFIXES:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                return np.nan, np.nan, np.nan, np.nan, np.nan
            width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            seconds = float(frames / fps) if fps and not np.isnan(frames) else np.nan
            return width, height, fps, frames, seconds
        finally:
            capture.release()
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan


def compressed_asset_base_suffix(suffix: str) -> str:
    normalized = str(suffix or "").lower()
    for compression in ASSET_COMPRESSION_SUFFIXES:
        if normalized.endswith(compression):
            return normalized[: -len(compression)]
    return normalized


def read_binary_asset(path: Path) -> bytes:
    return read_compressed_bytes(path, suffix=file_asset_suffix(path))


def read_binary_asset_head(path: Path, *, limit: int = 65536) -> bytes:
    return read_compressed_bytes(path, suffix=file_asset_suffix(path), limit=limit)


def dicom_metadata(path: Path | None) -> tuple[float, float, float, float, float]:
    if path is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    suffix = file_asset_suffix(path)
    if suffix not in DICOM_FILE_SUFFIXES:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    try:
        import pydicom

        source = str(path) if suffix in DICOM_FILE_BASE_SUFFIXES else BytesIO(read_binary_asset(path))
        dataset = pydicom.dcmread(source, stop_before_pixels=True, force=True)
        rows = float(getattr(dataset, "Rows", np.nan))
        columns = float(getattr(dataset, "Columns", np.nan))
        instance = float(getattr(dataset, "InstanceNumber", np.nan))
        spacing = getattr(dataset, "PixelSpacing", None)
        spacing_y = float(spacing[0]) if spacing is not None and len(spacing) >= 1 else np.nan
        spacing_x = float(spacing[1]) if spacing is not None and len(spacing) >= 2 else np.nan
        return rows, columns, spacing_y, spacing_x, instance
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan


def nifti_metadata(path: Path | None) -> tuple[float, float, float, float, float, float]:
    if path is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    suffix = file_asset_suffix(path)
    if suffix not in NIFTI_FILE_SUFFIXES:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    try:
        import nibabel as nib

        if suffix in NIFTI_FILE_BASE_SUFFIXES or (
            suffix.endswith(".gz") and compressed_asset_base_suffix(suffix) in NIFTI_FILE_BASE_SUFFIXES
        ):
            image = nib.load(str(path))
        else:
            image = nib.Nifti1Image.from_bytes(read_binary_asset(path))
        shape = tuple(image.shape)
        zooms = tuple(image.header.get_zooms())
        dim_0 = float(shape[0]) if len(shape) >= 1 else np.nan
        dim_1 = float(shape[1]) if len(shape) >= 2 else np.nan
        dim_2 = float(shape[2]) if len(shape) >= 3 else np.nan
        spacing_0 = float(zooms[0]) if len(zooms) >= 1 else np.nan
        spacing_1 = float(zooms[1]) if len(zooms) >= 2 else np.nan
        spacing_2 = float(zooms[2]) if len(zooms) >= 3 else np.nan
        return dim_0, dim_1, dim_2, spacing_0, spacing_1, spacing_2
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def medical_header_metadata(path: Path | None) -> tuple[float, float, float, float, float, float]:
    if path is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    suffix = file_asset_suffix(path)
    if suffix not in MEDICAL_HEADER_FILE_SUFFIXES:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    base_suffix = compressed_asset_base_suffix(suffix)
    if base_suffix in {".nhdr", ".nrrd"}:
        return nrrd_header_metadata(path)
    if base_suffix in MEDICAL_HEADER_FILE_BASE_SUFFIXES:
        return metaimage_header_metadata(path)
    return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def nrrd_header_metadata(path: Path) -> tuple[float, float, float, float, float, float]:
    try:
        fields = parse_medical_key_value_header(read_binary_asset_head(path))
        dims = parse_float_sequence(fields.get("sizes", ""))
        spacings = parse_float_sequence(fields.get("spacings", ""))
        if not spacings:
            spacings = parse_nrrd_space_directions(fields.get("space directions", ""))
        return medical_dims_spacing_tuple(dims, spacings)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def metaimage_header_metadata(path: Path) -> tuple[float, float, float, float, float, float]:
    try:
        fields = parse_medical_key_value_header(read_binary_asset_head(path))
        dims = parse_float_sequence(fields.get("dimsize", ""))
        spacings = parse_float_sequence(fields.get("elementspacing", ""))
        if not spacings:
            spacings = parse_float_sequence(fields.get("elementsize", ""))
        return medical_dims_spacing_tuple(dims, spacings)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def parse_medical_key_value_header(payload: bytes) -> dict[str, str]:
    text = payload.decode("utf-8", errors="ignore")
    fields: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if fields:
                break
            continue
        if line.startswith("#") or line.startswith("NRRD"):
            continue
        if ":=" in line:
            key, value = line.split(":=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        elif "=" in line:
            key, value = line.split("=", 1)
        else:
            continue
        fields[key.strip().lower()] = value.strip()
    return fields


def parse_float_sequence(value: str) -> list[float]:
    values: list[float] = []
    for token in re.split(r"[\s,]+", str(value or "").strip()):
        if not token or token.lower() in {"none", "nan"}:
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def parse_nrrd_space_directions(value: str) -> list[float]:
    spacings: list[float] = []
    for group in re.findall(r"\(([^)]*)\)", str(value or "")):
        components = parse_float_sequence(group)
        if not components:
            continue
        spacings.append(float(np.sqrt(np.sum(np.asarray(components, dtype=float) ** 2))))
    return spacings


def medical_dims_spacing_tuple(
    dims: list[float],
    spacings: list[float],
) -> tuple[float, float, float, float, float, float]:
    return (
        float(dims[0]) if len(dims) >= 1 else np.nan,
        float(dims[1]) if len(dims) >= 2 else np.nan,
        float(dims[2]) if len(dims) >= 3 else np.nan,
        float(spacings[0]) if len(spacings) >= 1 else np.nan,
        float(spacings[1]) if len(spacings) >= 2 else np.nan,
        float(spacings[2]) if len(spacings) >= 3 else np.nan,
    )


def point_cloud_metadata(path: Path | None) -> tuple[float, float]:
    if path is None:
        return np.nan, np.nan
    suffix = file_asset_suffix(path)
    base_suffix = document_text_suffix(suffix)
    if base_suffix == ".ply":
        return ply_point_cloud_metadata(path)
    if base_suffix == ".obj":
        return obj_point_cloud_metadata(path)
    if base_suffix in POINT_CLOUD_TEXT_METADATA_SUFFIXES:
        return text_point_cloud_metadata(path)
    return np.nan, np.nan


def ply_point_cloud_metadata(path: Path) -> tuple[float, float]:
    try:
        point_count = np.nan
        face_count = np.nan
        with open_text_asset(path) as handle:
            for _ in range(200):
                line = handle.readline()
                if not line:
                    break
                line = line.strip().lower()
                if line.startswith("element vertex "):
                    point_count = float(int(line.split()[-1]))
                elif line.startswith("element face "):
                    face_count = float(int(line.split()[-1]))
                elif line == "end_header":
                    break
        return point_count, face_count
    except Exception:
        return np.nan, np.nan


def obj_point_cloud_metadata(path: Path) -> tuple[float, float]:
    try:
        point_count = 0
        face_count = 0
        with open_text_asset(path) as handle:
            for line in handle:
                if line.startswith("v "):
                    point_count += 1
                elif line.startswith("f "):
                    face_count += 1
        return float(point_count), float(face_count)
    except Exception:
        return np.nan, np.nan


def text_point_cloud_metadata(path: Path) -> tuple[float, float]:
    try:
        with open_text_asset(path) as handle:
            lines = [line.strip() for line in handle if line.strip()]
        if not lines:
            return 0.0, np.nan
        first = lines[0].split()
        if len(first) == 1 and first[0].isdigit():
            return float(int(first[0])), np.nan
        return float(len(lines)), np.nan
    except Exception:
        return np.nan, np.nan


def graph_metadata(path: Path | None) -> tuple[float, float]:
    if path is None:
        return np.nan, np.nan
    suffix = file_asset_suffix(path)
    base_suffix = document_text_suffix(suffix)
    if base_suffix in GRAPH_XML_BASE_SUFFIXES:
        return xml_graph_metadata(path)
    if base_suffix == ".gml":
        return gml_graph_metadata(path)
    if base_suffix == ".mtx":
        return matrix_market_graph_metadata(path)
    if base_suffix in GRAPH_EDGE_LIST_BASE_SUFFIXES:
        return edge_list_graph_metadata(path)
    return np.nan, np.nan


def xml_graph_metadata(path: Path) -> tuple[float, float]:
    try:
        from xml.etree import ElementTree as ET

        node_count = 0
        edge_count = 0
        with open_text_asset(path) as handle:
            for _event, element in ET.iterparse(handle, events=("start",)):
                local_name = str(element.tag).rsplit("}", 1)[-1].lower()
                if local_name == "node":
                    node_count += 1
                elif local_name == "edge":
                    edge_count += 1
        return float(node_count), float(edge_count)
    except Exception:
        return np.nan, np.nan


def gml_graph_metadata(path: Path) -> tuple[float, float]:
    try:
        node_count = 0
        edge_count = 0
        with open_text_asset(path) as handle:
            for line in handle:
                stripped = line.strip().lower()
                if stripped.startswith("node"):
                    node_count += 1
                elif stripped.startswith("edge"):
                    edge_count += 1
        return float(node_count), float(edge_count)
    except Exception:
        return np.nan, np.nan


def matrix_market_graph_metadata(path: Path) -> tuple[float, float]:
    try:
        with open_text_asset(path) as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("%"):
                    continue
                parts = stripped.split()
                if len(parts) >= 3 and all(part.lstrip("+-").isdigit() for part in parts[:3]):
                    rows, cols, edges = (int(parts[0]), int(parts[1]), int(parts[2]))
                    return float(max(rows, cols)), float(edges)
                break
        return np.nan, np.nan
    except Exception:
        return np.nan, np.nan


def edge_list_graph_metadata(path: Path) -> tuple[float, float]:
    try:
        nodes = set()
        edge_count = 0
        with open_text_asset(path) as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or stripped.startswith("%"):
                    continue
                parts = stripped.replace(",", " ").split()
                if len(parts) < 2:
                    continue
                nodes.add(parts[0])
                nodes.add(parts[1])
                edge_count += 1
        return float(len(nodes)), float(edge_count)
    except Exception:
        return np.nan, np.nan


def geospatial_metadata(path: Path | None) -> tuple[float, float, float, float, float]:
    if path is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    suffix = file_asset_suffix(path)
    base_suffix = document_text_suffix(suffix)
    if suffix in GEOJSON_FILE_SUFFIXES:
        return geojson_metadata(path)
    if base_suffix == ".kml" or suffix == ".kmz":
        return kml_metadata(path)
    return np.nan, np.nan, np.nan, np.nan, np.nan


def geojson_metadata(path: Path) -> tuple[float, float, float, float, float]:
    try:
        payload = load_json_payload(path)
        if not isinstance(payload, dict) or str(payload.get("type", "")).lower() != "featurecollection":
            return np.nan, np.nan, np.nan, np.nan, np.nan
        features = [feature for feature in payload.get("features", []) if isinstance(feature, dict)]
        points = []
        for feature in features:
            points.extend(geojson_geometry_points(feature.get("geometry")))
        return geospatial_bounds_tuple(float(len(features)), points)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan


def geojson_geometry_points(geometry) -> list[tuple[float, float]]:
    if not isinstance(geometry, dict):
        return []
    geometry_type = str(geometry.get("type", "")).lower()
    if geometry_type == "geometrycollection":
        points = []
        for item in geometry.get("geometries", []):
            points.extend(geojson_geometry_points(item))
        return points
    return coordinate_points(geometry.get("coordinates"))


def coordinate_points(value) -> list[tuple[float, float]]:
    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and isinstance(value[0], (int, float)) and isinstance(value[1], (int, float)):
            return [(float(value[0]), float(value[1]))]
        points = []
        for item in value:
            points.extend(coordinate_points(item))
        return points
    return []


def kml_metadata(path: Path) -> tuple[float, float, float, float, float]:
    try:
        from xml.etree import ElementTree as ET

        root = ET.fromstring(read_kml_payload(path))
        placemarks = [node for node in root.iter() if xml_local_name(node.tag) == "Placemark"]
        points = []
        for node in root.iter():
            if xml_local_name(node.tag) != "coordinates":
                continue
            points.extend(kml_coordinate_points(node.text or ""))
        return geospatial_bounds_tuple(float(len(placemarks)), points)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan


def kml_coordinate_points(text: str) -> list[tuple[float, float]]:
    points = []
    for token in text.replace("\n", " ").replace("\t", " ").split():
        parts = token.split(",")
        if len(parts) < 2:
            continue
        try:
            points.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return points


def geospatial_bounds_tuple(
    count: float,
    points: list[tuple[float, float]],
) -> tuple[float, float, float, float, float]:
    if not points:
        return count, np.nan, np.nan, np.nan, np.nan
    xs = np.asarray([point[0] for point in points], dtype=float)
    ys = np.asarray([point[1] for point in points], dtype=float)
    return count, float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def document_metadata(path: Path | None) -> tuple[float, float, float, float]:
    if path is None:
        return np.nan, np.nan, np.nan, np.nan
    suffix = file_asset_suffix(path)
    text_suffix = document_text_suffix(suffix)
    if suffix == ".pdf":
        return pdf_document_metadata(path)
    if suffix == ".docx":
        return docx_document_metadata(path)
    if text_suffix in DOCUMENT_TEXT_METADATA_SUFFIXES:
        return text_document_metadata(path, suffix=suffix)
    return np.nan, np.nan, np.nan, np.nan


def document_text_suffix(suffix: str) -> str:
    normalized = str(suffix or "").lower()
    for compression in ASSET_COMPRESSION_SUFFIXES:
        if normalized.endswith(compression):
            return normalized[: -len(compression)]
    return normalized


def pdf_document_metadata(path: Path) -> tuple[float, float, float, float]:
    try:
        payload = path.read_bytes()
        page_count = len(re.findall(rb"/Type\s*/Page\b", payload))
        return float(page_count) if page_count else np.nan, np.nan, np.nan, np.nan
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def docx_document_metadata(path: Path) -> tuple[float, float, float, float]:
    try:
        from xml.etree import ElementTree as ET

        with zipfile.ZipFile(path, "r") as archive:
            xml_payload = archive.read("word/document.xml")
        root = ET.fromstring(xml_payload)
        paragraphs = []
        for paragraph in root.iter():
            if xml_local_name(paragraph.tag) != "p":
                continue
            text = "".join(node.text or "" for node in paragraph.iter() if xml_local_name(node.tag) == "t").strip()
            if text:
                paragraphs.append(text)
        text = "\n".join(paragraphs)
        chars, words = text_counts(text)
        return np.nan, chars, words, float(len(paragraphs))
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def text_document_metadata(path: Path, *, suffix: str) -> tuple[float, float, float, float]:
    try:
        with open_html_text(path) as handle:
            text = handle.read()
        text_suffix = document_text_suffix(suffix)
        if text_suffix in DOCUMENT_HTML_BASE_SUFFIXES:
            text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
            text = re.sub(r"(?s)<[^>]+>", " ", text)
        elif text_suffix == ".rtf":
            text = re.sub(r"\\[a-zA-Z]+-?\d* ?", " ", text)
            text = re.sub(r"[{}]", " ", text)
        paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
        chars, words = text_counts(text)
        return np.nan, chars, words, float(len(paragraphs))
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def text_counts(text: str) -> tuple[float, float]:
    normalized = str(text or "")
    return float(len(normalized)), float(len(re.findall(r"\b\w+\b", normalized)))


def array_metadata(path: Path | None) -> tuple[float, float, float, float, float, float]:
    if path is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    suffix = file_asset_suffix(path)
    if suffix == ".npy":
        return numpy_file_metadata(path)
    if suffix == ".npz":
        return numpy_archive_metadata(path)
    if suffix in DIRECTORY_ASSET_SUFFIXES:
        return zarr_store_metadata(path)
    if suffix in TABULAR_NETCDF_SUFFIXES:
        return netcdf_file_metadata(path)
    if suffix in TABULAR_FITS_SUFFIXES:
        return fits_file_metadata(path)
    if suffix in TABULAR_MATLAB_SUFFIXES:
        return mat_file_metadata(path)
    if suffix in TABULAR_HDF_SUFFIXES or suffix in {".h5ad", ".loom"}:
        return hdf_file_metadata(path)
    return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def numpy_file_metadata(path: Path) -> tuple[float, float, float, float, float, float]:
    try:
        with path.open("rb") as handle:
            shape = numpy_header_shape(handle)
        return array_shape_tuple(1.0, shape)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def numpy_archive_metadata(path: Path) -> tuple[float, float, float, float, float, float]:
    try:
        shapes = []
        with zipfile.ZipFile(path, "r") as archive:
            for member in archive.infolist():
                if member.is_dir() or not member.filename.lower().endswith(".npy"):
                    continue
                with archive.open(member, "r") as handle:
                    shapes.append(numpy_header_shape(handle))
        if not shapes:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        shape = max(shapes, key=lambda item: array_shape_size(item))
        return array_shape_tuple(float(len(shapes)), shape)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def numpy_header_shape(handle) -> tuple[int, ...]:
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        shape, _fortran_order, _dtype = np.lib.format.read_array_header_1_0(handle)
    elif version == (2, 0):
        shape, _fortran_order, _dtype = np.lib.format.read_array_header_2_0(handle)
    else:
        shape, _fortran_order, _dtype = np.lib.format.read_array_header_2_0(handle)
    return tuple(int(dim) for dim in shape)


def zarr_store_metadata(path: Path) -> tuple[float, float, float, float, float, float]:
    try:
        shape = zarr_store_shape(path)
        if shape is None:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        return array_shape_tuple(1.0, shape)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def zarr_store_shape(path: Path) -> tuple[int, ...] | None:
    candidates = [path / ".zarray", path / "zarr.json", path / "attributes.json"]
    for candidate in candidates:
        shape = store_metadata_shape(candidate)
        if shape is not None:
            return shape
    for metadata_name in (".zarray", "zarr.json", "attributes.json"):
        for candidate in sorted(path.rglob(metadata_name))[:128]:
            if candidate in candidates:
                continue
            shape = store_metadata_shape(candidate)
            if shape is not None:
                return shape
    return None


def store_metadata_shape(path: Path) -> tuple[int, ...] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return shape_from_metadata_payload(payload)


def shape_from_metadata_payload(payload: dict) -> tuple[int, ...] | None:
    for key in ("shape", "dimensions", "dims"):
        shape = shape_tuple_from_value(payload.get(key))
        if shape is not None:
            return shape
    attributes = payload.get("attributes")
    if isinstance(attributes, dict):
        for key in ("shape", "dimensions", "dims"):
            shape = shape_tuple_from_value(attributes.get(key))
            if shape is not None:
                return shape
    return None


def shape_tuple_from_value(value) -> tuple[int, ...] | None:
    if not isinstance(value, list) or not value:
        return None
    shape: list[int] = []
    for dim in value:
        if isinstance(dim, bool):
            return None
        if isinstance(dim, int):
            shape.append(int(dim))
            continue
        if isinstance(dim, float) and dim.is_integer():
            shape.append(int(dim))
            continue
        return None
    return tuple(shape)


def netcdf_file_metadata(path: Path) -> tuple[float, float, float, float, float, float]:
    try:
        from scipy.io import netcdf_file

        shapes = []
        with netcdf_file(path, mode="r", mmap=False) as dataset:
            for variable in dataset.variables.values():
                shape = tuple(int(dim) for dim in getattr(variable, "shape", ()) or ())
                if shape:
                    shapes.append(shape)
        return largest_array_shape_tuple(shapes)
    except Exception:
        return hdf_file_metadata(path)


def fits_file_metadata(path: Path) -> tuple[float, float, float, float, float, float]:
    try:
        from astropy.io import fits

        shapes = []
        with fits.open(path, memmap=False) as hdus:
            for hdu in hdus:
                shape = getattr(hdu, "shape", None)
                if shape is None and getattr(hdu, "data", None) is not None:
                    shape = np.asarray(hdu.data).shape
                shape_tuple = tuple(int(dim) for dim in (shape or ()))
                if shape_tuple:
                    shapes.append(shape_tuple)
        return largest_array_shape_tuple(shapes)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def mat_file_metadata(path: Path) -> tuple[float, float, float, float, float, float]:
    try:
        from scipy.io import whosmat

        shapes = [tuple(int(dim) for dim in shape) for _name, shape, _class_name in whosmat(path) if shape]
        return largest_array_shape_tuple(shapes)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def hdf_file_metadata(path: Path) -> tuple[float, float, float, float, float, float]:
    try:
        import h5py

        shapes = []
        with h5py.File(path, "r") as handle:
            def collect_shape(_name, node):
                shape = hdf_node_shape(node, h5py=h5py)
                if shape:
                    shapes.append(shape)

            handle.visititems(collect_shape)
        return largest_array_shape_tuple(shapes)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def hdf_node_shape(node, *, h5py) -> tuple[int, ...] | None:
    if isinstance(node, h5py.Dataset):
        return tuple(int(dim) for dim in (node.shape or ()))
    if isinstance(node, h5py.Group) and "shape" in node:
        try:
            raw_shape = np.asarray(node["shape"][()]).tolist()
            if isinstance(raw_shape, (list, tuple)):
                return tuple(int(dim) for dim in raw_shape)
        except Exception:
            return None
    return None


def largest_array_shape_tuple(shapes: list[tuple[int, ...]]) -> tuple[float, float, float, float, float, float]:
    valid_shapes = [shape for shape in shapes if shape]
    if not valid_shapes:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    shape = max(valid_shapes, key=array_shape_size)
    return array_shape_tuple(float(len(valid_shapes)), shape)


def array_shape_tuple(count: float, shape: tuple[int, ...]) -> tuple[float, float, float, float, float, float]:
    dims = [float(dim) for dim in shape[:3]]
    while len(dims) < 3:
        dims.append(np.nan)
    return count, float(len(shape)), dims[0], dims[1], dims[2], float(array_shape_size(shape))


def array_shape_size(shape: tuple[int, ...]) -> int:
    size = 1
    for dim in shape:
        size *= int(dim)
    return size


def model_artifact_metadata(path: Path | None) -> tuple[float, float, float, float, float]:
    if path is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    suffix = file_asset_suffix(path)
    if suffix == ".onnx":
        return onnx_artifact_metadata(path)
    if suffix == ".safetensors":
        return safetensors_artifact_metadata(path)
    if suffix in MODEL_ARTIFACT_COMPOUND_SUFFIXES:
        return model_index_artifact_metadata(path)
    return np.nan, np.nan, np.nan, np.nan, np.nan


def onnx_artifact_metadata(path: Path) -> tuple[float, float, float, float, float]:
    try:
        import onnx

        model = onnx.load_model(str(path), load_external_data=False)
        graph = model.graph
        parameter_count = 0
        for initializer in graph.initializer:
            parameter_count += array_shape_size(tuple(int(dim) for dim in initializer.dims))
        return (
            float(len(graph.node)),
            float(len(graph.input)),
            float(len(graph.output)),
            float(len(graph.initializer)),
            float(parameter_count),
        )
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan


def safetensors_artifact_metadata(path: Path) -> tuple[float, float, float, float, float]:
    try:
        from safetensors import safe_open

        tensor_count = 0
        parameter_count = 0
        with safe_open(str(path), framework="numpy") as handle:
            for key in handle.keys():
                tensor_count += 1
                parameter_count += array_shape_size(tuple(int(dim) for dim in handle.get_tensor(key).shape))
        return np.nan, np.nan, np.nan, float(tensor_count), float(parameter_count)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan


def model_index_artifact_metadata(path: Path) -> tuple[float, float, float, float, float]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        tensor_count = len(weight_map) if isinstance(weight_map, dict) else np.nan
        total_size = np.nan
        if isinstance(metadata, dict):
            raw_size = metadata.get("total_size") or metadata.get("total_size_bytes")
            if raw_size is not None:
                total_size = float(raw_size)
        return np.nan, np.nan, np.nan, float(tensor_count), float(total_size)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan


def model_sidecar_metadata(path: Path | None) -> tuple[float, float, float]:
    if path is None:
        return np.nan, np.nan, np.nan
    name = path.name.lower()
    if name in MODEL_ARTIFACT_JSON_SIDECAR_FILENAMES:
        return json_model_sidecar_metadata(path, name=name)
    if name in MODEL_ARTIFACT_TEXT_SIDECAR_FILENAMES:
        line_count = bounded_nonempty_line_count(path)
        return np.nan, float(line_count), np.nan
    return np.nan, np.nan, np.nan


def json_model_sidecar_metadata(path: Path, *, name: str) -> tuple[float, float, float]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return np.nan, np.nan, np.nan
    top_keys = float(len(payload)) if isinstance(payload, dict) else np.nan
    vocab_size = np.nan
    added_tokens = np.nan
    if name == "tokenizer.json" and isinstance(payload, dict):
        model_payload = payload.get("model")
        if isinstance(model_payload, dict):
            vocab = model_payload.get("vocab")
            if isinstance(vocab, (dict, list)):
                vocab_size = float(len(vocab))
        raw_added_tokens = payload.get("added_tokens")
        if isinstance(raw_added_tokens, list):
            added_tokens = float(len(raw_added_tokens))
    elif name == "vocab.json" and isinstance(payload, dict):
        vocab_size = float(len(payload))
    return top_keys, vocab_size, added_tokens


def bounded_nonempty_line_count(path: Path, *, limit: int = 1_000_000) -> int:
    try:
        count = 0
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.strip():
                    count += 1
                if count >= limit:
                    break
        return count
    except Exception:
        return 0


def open_text_asset(path: Path):
    return open_compressed_text(path)


def bio_sequence_metadata(path: Path | None) -> tuple[float, float, float, float, float]:
    if path is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    suffix = file_asset_suffix(path)
    base_suffix = document_text_suffix(suffix)
    if base_suffix not in BIO_SEQUENCE_BASE_SUFFIXES:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    try:
        lengths = []
        if base_suffix in BIO_FASTQ_BASE_SUFFIXES:
            with open_text_asset(path) as handle:
                for index, line in enumerate(handle):
                    if index % 4 == 1:
                        lengths.append(len(line.strip()))
                    if len(lengths) >= 10000:
                        break
        else:
            current = 0
            with open_text_asset(path) as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if stripped.startswith(">"):
                        if current:
                            lengths.append(current)
                        current = 0
                    else:
                        current += len(stripped)
                    if len(lengths) >= 10000:
                        break
            if current and len(lengths) < 10000:
                lengths.append(current)
        if not lengths:
            return 0.0, 0.0, np.nan, np.nan, np.nan
        values = np.asarray(lengths, dtype=float)
        return float(len(lengths)), float(values.sum()), float(values.min()), float(values.mean()), float(values.max())
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan


def bio_structure_metadata(path: Path | None) -> tuple[float, float, float, float]:
    if path is None:
        return np.nan, np.nan, np.nan, np.nan
    suffix = file_asset_suffix(path)
    base_suffix = document_text_suffix(suffix)
    if base_suffix in BIO_PDB_STRUCTURE_BASE_SUFFIXES:
        return pdb_like_structure_metadata(path)
    if base_suffix in BIO_MOL_STRUCTURE_BASE_SUFFIXES:
        return mol_structure_metadata(path)
    if base_suffix == ".mol2":
        return mol2_structure_metadata(path)
    return np.nan, np.nan, np.nan, np.nan


def pdb_like_structure_metadata(path: Path) -> tuple[float, float, float, float]:
    try:
        atoms = 0
        residues = set()
        with open_text_asset(path) as handle:
            for line in handle:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                atoms += 1
                if len(line) >= 27:
                    residues.add(line[21:27].strip())
                else:
                    parts = line.split()
                    if len(parts) >= 6:
                        residues.add(parts[5])
        return float(atoms), float(len(residues)) if residues else np.nan, np.nan, 1.0 if atoms else np.nan
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def mol_structure_metadata(path: Path) -> tuple[float, float, float, float]:
    try:
        atoms = 0
        bonds = 0
        molecules = 0
        block_lines = []
        with open_text_asset(path) as handle:
            for line in handle:
                if line.strip() == "$$$$":
                    block_atoms, block_bonds = mol_block_counts(block_lines)
                    if not np.isnan(block_atoms):
                        atoms += int(block_atoms)
                        bonds += int(block_bonds) if not np.isnan(block_bonds) else 0
                        molecules += 1
                    block_lines = []
                else:
                    block_lines.append(line.rstrip("\n"))
            if block_lines:
                block_atoms, block_bonds = mol_block_counts(block_lines)
                if not np.isnan(block_atoms):
                    atoms += int(block_atoms)
                    bonds += int(block_bonds) if not np.isnan(block_bonds) else 0
                    molecules += 1
        if molecules == 0:
            return np.nan, np.nan, np.nan, np.nan
        return float(atoms), np.nan, float(bonds), float(molecules)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def mol_block_counts(lines: list[str]) -> tuple[float, float]:
    if len(lines) < 4:
        return np.nan, np.nan
    counts = lines[3]
    try:
        atoms = int(counts[:3])
        bonds = int(counts[3:6])
        return float(atoms), float(bonds)
    except Exception:
        parts = counts.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            return float(int(parts[0])), float(int(parts[1]))
    return np.nan, np.nan


def mol2_structure_metadata(path: Path) -> tuple[float, float, float, float]:
    try:
        atoms = 0
        bonds = 0
        section = ""
        with open_text_asset(path) as handle:
            for line in handle:
                stripped = line.strip()
                upper = stripped.upper()
                if upper.startswith("@<TRIPOS>"):
                    section = upper
                    continue
                if not stripped:
                    continue
                if section == "@<TRIPOS>ATOM":
                    atoms += 1
                elif section == "@<TRIPOS>BOND":
                    bonds += 1
        return float(atoms), np.nan, float(bonds), 1.0 if atoms else np.nan
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def annotation_metadata(path: Path | None) -> tuple[float, float, float, float, float]:
    if path is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    suffix = file_asset_suffix(path)
    text_suffix = document_text_suffix(suffix)
    if text_suffix == ".txt":
        return yolo_text_annotation_metadata(path)
    if text_suffix != ".json":
        return np.nan, np.nan, np.nan, np.nan, np.nan
    try:
        return annotation_payload_metadata(load_json_payload(path))
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan


def yolo_text_annotation_metadata(path: Path) -> tuple[float, float, float, float, float]:
    try:
        classes = set()
        annotations = 0
        bboxes = 0
        segmentations = 0
        saw_nonempty_line = False
        with open_text_asset(path) as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                saw_nonempty_line = True
                parts = stripped.split()
                if len(parts) < 5 or not all(is_float_token(part) for part in parts):
                    return np.nan, np.nan, np.nan, np.nan, np.nan
                classes.add(parts[0])
                annotations += 1
                if len(parts) == 5:
                    bboxes += 1
                elif len(parts) > 5:
                    segmentations += 1
        if annotations == 0 and saw_nonempty_line:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        return (
            float(annotations),
            np.nan,
            float(len(classes)) if classes else np.nan,
            float(bboxes),
            float(segmentations),
        )
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan


def is_float_token(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def annotation_payload_metadata(payload) -> tuple[float, float, float, float, float]:
    annotations = []
    images = np.nan
    categories = np.nan
    if isinstance(payload, dict):
        annotations = first_list_value(payload, ("annotations", "shapes", "objects", "instances", "labels"))
        images_value = payload.get("images")
        if isinstance(images_value, list):
            images = float(len(images_value))
        elif any(key in payload for key in ("image", "imagePath", "image_path", "imageHeight", "imageWidth")):
            images = 1.0
        categories_value = payload.get("categories")
        if isinstance(categories_value, list):
            categories = float(len(categories_value))
    elif isinstance(payload, list):
        annotations = payload

    if not isinstance(annotations, list):
        annotations = []
    if np.isnan(categories):
        labels = annotation_label_values(annotations)
        categories = float(len(labels)) if labels else np.nan

    bbox_count = 0
    segmentation_count = 0
    for item in annotations:
        if not isinstance(item, dict):
            continue
        if annotation_has_bbox(item):
            bbox_count += 1
        if annotation_has_segmentation(item):
            segmentation_count += 1
    return (
        float(len(annotations)),
        images,
        categories,
        float(bbox_count),
        float(segmentation_count),
    )


def first_list_value(payload: dict, keys: tuple[str, ...]):
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def annotation_label_values(annotations: list) -> set[str]:
    labels = set()
    for item in annotations:
        if not isinstance(item, dict):
            continue
        for key in ("category_id", "category", "class_id", "class", "label", "name"):
            value = item.get(key)
            if value is not None and str(value).strip():
                labels.add(str(value))
                break
    return labels


def annotation_has_bbox(item: dict) -> bool:
    for key in ("bbox", "bounding_box", "box", "rect"):
        value = item.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            return True
        if isinstance(value, dict) and len(value) >= 4:
            return True
    return all(key in item for key in ("x", "y", "width", "height")) or all(
        key in item for key in ("xmin", "ymin", "xmax", "ymax")
    )


def annotation_has_segmentation(item: dict) -> bool:
    for key in ("segmentation", "mask", "rle", "polygon", "points"):
        value = item.get(key)
        if isinstance(value, (list, tuple, dict)) and len(value) > 0:
            return True
        if isinstance(value, str) and value.strip():
            return True
    return False


def file_metadata_frame(values: pd.Series, asset_index: dict[str, Path], *, split: str, prefix: str) -> pd.DataFrame:
    paths = values.map(lambda value: resolve_asset_path(value, asset_index, split=split))
    sizes = paths.map(lambda path: float(path.stat().st_size) if path is not None and path.exists() else np.nan)
    suffixes = paths.map(lambda path: file_asset_suffix(path) if path is not None else "")
    dims = paths.map(image_size)
    image = paths.map(image_metadata)
    mask_image = paths.map(mask_image_metadata)
    image_intensity = paths.map(image_intensity_metadata)
    image_frames = paths.map(image_frame_count)
    durations = paths.map(wav_duration_seconds)
    audio = paths.map(audio_metadata)
    video = paths.map(video_metadata)
    dicom = paths.map(dicom_metadata)
    nifti = paths.map(nifti_metadata)
    medical_header = paths.map(medical_header_metadata)
    point_cloud = paths.map(point_cloud_metadata)
    graph = paths.map(graph_metadata)
    geospatial = paths.map(geospatial_metadata)
    document = paths.map(document_metadata)
    array = paths.map(array_metadata)
    model_artifact = paths.map(model_artifact_metadata)
    model_sidecar = paths.map(model_sidecar_metadata)
    bio_sequence = paths.map(bio_sequence_metadata)
    bio_structure = paths.map(bio_structure_metadata)
    annotation = paths.map(annotation_metadata)
    return pd.DataFrame(
        {
            f"{prefix}_exists": paths.map(lambda path: int(path is not None and path.exists())),
            f"{prefix}_bytes": sizes,
            f"{prefix}_suffix": suffixes,
            f"{prefix}_image_width": dims.map(lambda item: item[0]),
            f"{prefix}_image_height": dims.map(lambda item: item[1]),
            f"{prefix}_image_channels": image.map(lambda item: item[2]),
            f"{prefix}_image_pixels": image.map(lambda item: item[3]),
            f"{prefix}_image_aspect_ratio": image.map(lambda item: item[4]),
            f"{prefix}_image_mean_intensity": image_intensity.map(lambda item: item[0]),
            f"{prefix}_image_std_intensity": image_intensity.map(lambda item: item[1]),
            f"{prefix}_image_nonzero_fraction": image_intensity.map(lambda item: item[2]),
            f"{prefix}_image_frames": image_frames,
            f"{prefix}_mask_nonzero_pixels": mask_image.map(lambda item: item[0]),
            f"{prefix}_mask_coverage": mask_image.map(lambda item: item[1]),
            f"{prefix}_mask_labels": mask_image.map(lambda item: item[2]),
            f"{prefix}_wav_seconds": durations,
            f"{prefix}_audio_seconds": audio.map(lambda item: item[0]),
            f"{prefix}_audio_samplerate": audio.map(lambda item: item[1]),
            f"{prefix}_audio_channels": audio.map(lambda item: item[2]),
            f"{prefix}_audio_frames": audio.map(lambda item: item[3]),
            f"{prefix}_video_width": video.map(lambda item: item[0]),
            f"{prefix}_video_height": video.map(lambda item: item[1]),
            f"{prefix}_video_fps": video.map(lambda item: item[2]),
            f"{prefix}_video_frames": video.map(lambda item: item[3]),
            f"{prefix}_video_seconds": video.map(lambda item: item[4]),
            f"{prefix}_dicom_rows": dicom.map(lambda item: item[0]),
            f"{prefix}_dicom_columns": dicom.map(lambda item: item[1]),
            f"{prefix}_dicom_spacing_y": dicom.map(lambda item: item[2]),
            f"{prefix}_dicom_spacing_x": dicom.map(lambda item: item[3]),
            f"{prefix}_dicom_instance": dicom.map(lambda item: item[4]),
            f"{prefix}_nifti_dim_0": nifti.map(lambda item: item[0]),
            f"{prefix}_nifti_dim_1": nifti.map(lambda item: item[1]),
            f"{prefix}_nifti_dim_2": nifti.map(lambda item: item[2]),
            f"{prefix}_nifti_spacing_0": nifti.map(lambda item: item[3]),
            f"{prefix}_nifti_spacing_1": nifti.map(lambda item: item[4]),
            f"{prefix}_nifti_spacing_2": nifti.map(lambda item: item[5]),
            f"{prefix}_medical_dim_0": medical_header.map(lambda item: item[0]),
            f"{prefix}_medical_dim_1": medical_header.map(lambda item: item[1]),
            f"{prefix}_medical_dim_2": medical_header.map(lambda item: item[2]),
            f"{prefix}_medical_spacing_0": medical_header.map(lambda item: item[3]),
            f"{prefix}_medical_spacing_1": medical_header.map(lambda item: item[4]),
            f"{prefix}_medical_spacing_2": medical_header.map(lambda item: item[5]),
            f"{prefix}_point_count": point_cloud.map(lambda item: item[0]),
            f"{prefix}_point_faces": point_cloud.map(lambda item: item[1]),
            f"{prefix}_graph_nodes": graph.map(lambda item: item[0]),
            f"{prefix}_graph_edges": graph.map(lambda item: item[1]),
            f"{prefix}_geospatial_features": geospatial.map(lambda item: item[0]),
            f"{prefix}_geospatial_min_x": geospatial.map(lambda item: item[1]),
            f"{prefix}_geospatial_min_y": geospatial.map(lambda item: item[2]),
            f"{prefix}_geospatial_max_x": geospatial.map(lambda item: item[3]),
            f"{prefix}_geospatial_max_y": geospatial.map(lambda item: item[4]),
            f"{prefix}_document_pages": document.map(lambda item: item[0]),
            f"{prefix}_document_chars": document.map(lambda item: item[1]),
            f"{prefix}_document_words": document.map(lambda item: item[2]),
            f"{prefix}_document_paragraphs": document.map(lambda item: item[3]),
            f"{prefix}_array_count": array.map(lambda item: item[0]),
            f"{prefix}_array_ndim": array.map(lambda item: item[1]),
            f"{prefix}_array_dim_0": array.map(lambda item: item[2]),
            f"{prefix}_array_dim_1": array.map(lambda item: item[3]),
            f"{prefix}_array_dim_2": array.map(lambda item: item[4]),
            f"{prefix}_array_elements": array.map(lambda item: item[5]),
            f"{prefix}_model_nodes": model_artifact.map(lambda item: item[0]),
            f"{prefix}_model_inputs": model_artifact.map(lambda item: item[1]),
            f"{prefix}_model_outputs": model_artifact.map(lambda item: item[2]),
            f"{prefix}_model_tensors": model_artifact.map(lambda item: item[3]),
            f"{prefix}_model_parameters": model_artifact.map(lambda item: item[4]),
            f"{prefix}_model_sidecar_keys": model_sidecar.map(lambda item: item[0]),
            f"{prefix}_model_sidecar_vocab": model_sidecar.map(lambda item: item[1]),
            f"{prefix}_model_sidecar_added_tokens": model_sidecar.map(lambda item: item[2]),
            f"{prefix}_bio_sequence_records": bio_sequence.map(lambda item: item[0]),
            f"{prefix}_bio_sequence_total_length": bio_sequence.map(lambda item: item[1]),
            f"{prefix}_bio_sequence_min_length": bio_sequence.map(lambda item: item[2]),
            f"{prefix}_bio_sequence_mean_length": bio_sequence.map(lambda item: item[3]),
            f"{prefix}_bio_sequence_max_length": bio_sequence.map(lambda item: item[4]),
            f"{prefix}_bio_structure_atoms": bio_structure.map(lambda item: item[0]),
            f"{prefix}_bio_structure_residues": bio_structure.map(lambda item: item[1]),
            f"{prefix}_bio_structure_bonds": bio_structure.map(lambda item: item[2]),
            f"{prefix}_bio_structure_molecules": bio_structure.map(lambda item: item[3]),
            f"{prefix}_annotation_count": annotation.map(lambda item: item[0]),
            f"{prefix}_annotation_images": annotation.map(lambda item: item[1]),
            f"{prefix}_annotation_categories": annotation.map(lambda item: item[2]),
            f"{prefix}_annotation_bboxes": annotation.map(lambda item: item[3]),
            f"{prefix}_annotation_segmentations": annotation.map(lambda item: item[4]),
        },
        index=values.index,
    )


def add_file_reference_features(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    candidate_cols = [
        col
        for col in feature_cols
        if col in train.columns
        and col in test.columns
        and (
            looks_like_file_reference_column(train, col)
            or looks_like_file_reference_column(test, col)
        )
    ]
    if not candidate_cols:
        return []
    asset_index = build_file_asset_index(data_roots())
    added: list[str] = []
    for col in candidate_cols:
        prefix = "__file_" + "".join(ch if ch.isalnum() else "_" for ch in str(col).lower()).strip("_")
        train_meta = file_metadata_frame(train[col], asset_index, split="train", prefix=prefix)
        test_meta = file_metadata_frame(test[col], asset_index, split="test", prefix=prefix)
        if not int(train_meta[f"{prefix}_exists"].sum()) and not int(test_meta[f"{prefix}_exists"].sum()):
            continue
        for meta_col in train_meta.columns:
            combined = pd.concat([train_meta[meta_col], test_meta[meta_col]], ignore_index=True)
            if pd.api.types.is_numeric_dtype(combined):
                if combined.dropna().empty:
                    continue
            elif not combined.astype(str).str.strip().ne("").any():
                continue
            train[meta_col] = train_meta[meta_col].to_numpy()
            test[meta_col] = test_meta[meta_col].to_numpy()
            added.append(meta_col)
    return added


def summarize_file_reference_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    file_reference_feature_cols: list[str],
) -> dict[str, dict[str, dict[str, int]]]:
    summary: dict[str, dict[str, dict[str, int]]] = {}
    for col in file_reference_feature_cols:
        if not str(col).endswith("_suffix"):
            continue
        col_summary: dict[str, dict[str, int]] = {}
        for split, frame in (("train", train), ("test", test)):
            if col not in frame.columns:
                continue
            counts = frame[col].fillna("").astype(str).value_counts().to_dict()
            col_summary[split] = {str(key): int(value) for key, value in counts.items() if str(key)}
        summary[col] = col_summary
    return summary


def read_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return finalize_table_frame(read_raw_table(path, nrows=nrows))


def read_raw_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    suffix = tabular_suffix(path)
    base_suffix = tabular_base_suffix(suffix)
    if base_suffix in TABULAR_PARQUET_SUFFIXES:
        with open_tabular_binary(path) as handle:
            frame = pd.read_parquet(handle)
        return frame.head(nrows) if nrows is not None else frame
    if base_suffix == ".orc":
        with open_tabular_binary(path) as handle:
            frame = pd.read_orc(handle)
        return frame.head(nrows) if nrows is not None else frame
    if suffix in TABULAR_HDF_SUFFIXES:
        frame = read_hdf_table(path)
        return frame.head(nrows) if nrows is not None else frame
    if suffix in TABULAR_ANNDATA_SUFFIXES:
        frame = read_h5ad_tabular_frame(path)
        return frame.head(nrows) if nrows is not None else frame
    if suffix in TABULAR_LOOM_SUFFIXES:
        frame = read_loom_tabular_frame(path)
        return frame.head(nrows) if nrows is not None else frame
    if suffix in TABULAR_GEOPACKAGE_SUFFIXES:
        return read_geopackage_tabular_frame(path, nrows=nrows)
    if suffix in TABULAR_SHAPEFILE_SUFFIXES:
        return read_shapefile_tabular_frame(path, nrows=nrows)
    if suffix in TABULAR_KML_SUFFIXES:
        frame = read_kml_tabular_frame(path)
        return frame.head(nrows) if nrows is not None else frame
    if base_suffix in TABULAR_ARROW_IPC_SUFFIXES:
        with open_tabular_binary(path) as handle:
            frame = pd.read_feather(handle)
        return frame.head(nrows) if nrows is not None else frame
    if base_suffix == ".avro":
        frame = read_avro_table(path)
        return frame.head(nrows) if nrows is not None else frame
    if suffix in SQLITE_TABULAR_SUFFIXES:
        return read_sqlite_table(path, nrows=nrows)
    if suffix in DUCKDB_TABULAR_SUFFIXES:
        return read_duckdb_table(path, nrows=nrows)
    if base_suffix in TABULAR_EXCEL_SUFFIXES:
        with open_tabular_binary(path) as handle:
            frame = pd.read_excel(handle)
        return frame.head(nrows) if nrows is not None else frame
    if base_suffix in TABULAR_EXCEL_INPUT_ONLY_SUFFIXES:
        with open_tabular_binary(path) as handle:
            frame = pd.read_excel(handle, engine="pyxlsb")
        return frame.head(nrows) if nrows is not None else frame
    if suffix.startswith(TABULAR_SVMLIGHT_SUFFIX_PREFIXES):
        frame = read_svmlight_tabular_frame(path)
        return frame.head(nrows) if nrows is not None else frame
    if suffix.startswith(TABULAR_FIXED_WIDTH_SUFFIX_PREFIXES):
        frame = read_fixed_width_tabular_frame(path)
        return frame.head(nrows) if nrows is not None else frame
    if base_suffix in TABULAR_STATA_SUFFIXES:
        if suffix.endswith(".zip"):
            with open_tabular_binary(path) as handle:
                frame = pd.read_stata(handle)
        else:
            frame = pd.read_stata(path)
        return frame.head(nrows) if nrows is not None else frame
    if base_suffix in TABULAR_SAS_SUFFIXES:
        if suffix.endswith(".zip"):
            with open_tabular_binary(path) as handle:
                frame = pd.read_sas(handle, format=sas_format_for_suffix(suffix))
        else:
            frame = pd.read_sas(path, format=sas_format_for_suffix(suffix))
        return frame.head(nrows) if nrows is not None else frame
    if base_suffix in TABULAR_SPSS_SUFFIXES:
        if suffix.endswith(".zip"):
            with open_tabular_binary(path) as handle:
                frame = pd.read_spss(handle)
        else:
            frame = pd.read_spss(path)
        return frame.head(nrows) if nrows is not None else frame
    if suffix in TABULAR_MATLAB_SUFFIXES:
        frame = read_mat_tabular_frame(path)
        return frame.head(nrows) if nrows is not None else frame
    if suffix in TABULAR_RDATA_SUFFIXES:
        frame = read_rdata_tabular_frame(path)
        return frame.head(nrows) if nrows is not None else frame
    if suffix in TABULAR_NETCDF_SUFFIXES:
        frame = read_netcdf_tabular_frame(path)
        return frame.head(nrows) if nrows is not None else frame
    if suffix in TABULAR_NUMPY_SUFFIXES:
        frame = read_numpy_tabular_frame(path)
        return frame.head(nrows) if nrows is not None else frame
    if suffix in TABULAR_FITS_SUFFIXES:
        frame = read_fits_tabular_frame(path)
        return frame.head(nrows) if nrows is not None else frame
    if suffix in TABULAR_ARFF_SUFFIXES:
        frame = read_arff_tabular_frame(path)
        return frame.head(nrows) if nrows is not None else frame
    if suffix.startswith(TABULAR_HTML_SUFFIX_PREFIXES):
        frame = read_html_tabular_frame(path)
        return frame.head(nrows) if nrows is not None else frame
    if suffix.startswith(".xml"):
        frame = read_xml_tabular_frame(path)
        return frame.head(nrows) if nrows is not None else frame
    if base_suffix in TABULAR_PICKLE_SUFFIXES:
        with open_tabular_binary(path) as handle:
            frame = pd.read_pickle(handle)
        return frame.head(nrows) if nrows is not None else frame
    if suffix in TABULAR_STRUCTURED_SUFFIXES:
        return read_json_table(path, nrows=nrows)
    if suffix in TABULAR_TEXT_SUFFIXES:
        return read_text_tabular_frame(path, nrows=nrows)
    return pd.read_csv(path, nrows=nrows)


def finalize_table_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = dedupe_column_names(
        [stable_table_column_name(column, position) for position, column in enumerate(frame.columns)]
    )
    return frame


def stable_table_column_name(column: object, position: int) -> str:
    fallback = f"column_{position + 1}"
    if isinstance(column, tuple):
        parts = [
            str(part).strip()
            for part in column
            if not table_column_part_is_missing(part) and not table_column_name_is_generated_missing(str(part).strip())
        ]
        return "_".join(part for part in parts if part) or fallback
    if table_column_part_is_missing(column):
        return fallback
    name = str(column)
    stripped = name.strip()
    if not stripped or table_column_name_is_generated_missing(stripped):
        return fallback
    return name


def table_column_name_is_generated_missing(name: str) -> bool:
    return bool(re.fullmatch(r"Unnamed:\s*\d+(?:_level_\d+)?", str(name).strip(), flags=re.IGNORECASE))


def table_column_part_is_missing(value: object) -> bool:
    if value is None:
        return True
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def read_mat_tabular_frame(path: Path) -> pd.DataFrame:
    from scipy.io import loadmat

    payload = {
        key: value
        for key, value in loadmat(path, squeeze_me=True, struct_as_record=False).items()
        if not key.startswith("__")
    }
    frame = mat_variables_to_column_frame(payload)
    if frame is not None:
        return frame
    for variable_name, value in payload.items():
        frame = mat_value_to_frame(variable_name, value)
        if frame is not None:
            return frame
    raise ValueError(f"MATLAB file does not contain table-like arrays: {path}")


def mat_variables_to_column_frame(payload: dict) -> pd.DataFrame | None:
    columns_by_length: dict[int, dict[str, object]] = {}
    for name, value in payload.items():
        values = mat_value_to_series(value)
        if values is None:
            continue
        columns_by_length.setdefault(len(values), {})[str(name)] = values
    if not columns_by_length:
        return None
    columns = max(columns_by_length.values(), key=len)
    return pd.DataFrame(columns)


def mat_value_to_frame(variable_name: str, value) -> pd.DataFrame | None:
    if hasattr(value, "dtype") and getattr(value.dtype, "names", None):
        columns = {}
        for field_name in value.dtype.names:
            values = mat_value_to_series(value[field_name])
            if values is not None:
                columns[str(field_name)] = values
        return pd.DataFrame(columns) if columns else None
    values = mat_value_to_series(value)
    if values is not None:
        return pd.DataFrame({variable_name: values})
    array = np.asarray(value)
    if array.dtype.kind == "O":
        return None
    array = np.squeeze(array)
    if array.ndim != 2 or 0 in array.shape:
        return None
    columns = [f"{variable_name}_{idx}" for idx in range(array.shape[1])]
    if array.shape[1] == 1:
        columns = [variable_name]
    return pd.DataFrame(array, columns=columns)


def mat_value_to_series(value):
    array = np.asarray(value)
    if array.dtype.kind == "O":
        return None
    array = np.squeeze(array)
    if array.ndim != 1 or array.size == 0:
        return None
    return array.tolist()


def read_rdata_tabular_frame(path: Path) -> pd.DataFrame:
    try:
        import pyreadr
    except Exception as exc:
        raise ValueError("pyreadr is required to read RDS/RData tabular files") from exc

    result = pyreadr.read_r(path)
    for value in result.values():
        if hasattr(value, "columns") and hasattr(value, "head"):
            return value
    raise ValueError(f"R data file does not contain a tabular object: {path}")


def read_netcdf_tabular_frame(path: Path) -> pd.DataFrame:
    from scipy.io import netcdf_file

    variables = {}
    try:
        with netcdf_file(path, mode="r", mmap=False) as dataset:
            for name, variable in dataset.variables.items():
                variables[str(name)] = np.array(variable.data).copy()
    except Exception as exc:
        fallback = read_native_hdf_table(path)
        if fallback is not None:
            return fallback
        raise exc
    frame = netcdf_variables_to_column_frame(variables)
    if frame is None:
        fallback = read_native_hdf_table(path)
        if fallback is not None:
            return fallback
        raise ValueError(f"NetCDF file does not contain table-like variables: {path}")
    return frame


def netcdf_variables_to_column_frame(payload: dict) -> pd.DataFrame | None:
    columns_by_length: dict[int, dict[str, object]] = {}
    for name, value in payload.items():
        columns = netcdf_value_to_columns(str(name), value)
        if not columns:
            continue
        row_count = len(next(iter(columns.values())))
        columns_by_length.setdefault(row_count, {}).update(columns)
    if not columns_by_length:
        return None
    columns = max(columns_by_length.values(), key=len)
    return pd.DataFrame(columns)


def netcdf_value_to_columns(name: str, value) -> dict[str, object] | None:
    array = np.asarray(value)
    if array.shape == () or 0 in array.shape or array.dtype.kind == "O":
        return None
    if array.dtype.kind == "S":
        decoded = netcdf_decode_bytes_array(array)
        return {safe_netcdf_column_name(name): decoded} if decoded is not None else None
    if array.ndim > 1 and array.shape[-1] == 1:
        array = np.squeeze(array, axis=-1)
    if array.ndim == 1:
        return {safe_netcdf_column_name(name): array.tolist()}
    if array.ndim == 2:
        base = safe_netcdf_column_name(name)
        return {f"{base}_{idx}": array[:, idx].tolist() for idx in range(array.shape[1])}
    return None


def netcdf_decode_bytes_array(array) -> list[str] | None:
    if array.ndim == 1:
        return [netcdf_decode_bytes(value) for value in array.tolist()]
    if array.ndim == 2:
        values = []
        for row in array:
            flattened = np.asarray(row).ravel().tolist()
            values.append(netcdf_decode_bytes(b"".join(flattened)))
        return values
    return None


def netcdf_decode_bytes(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00 ")
    return str(value).rstrip("\x00 ")


def safe_netcdf_column_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_") or "value"


def read_fits_tabular_frame(path: Path) -> pd.DataFrame:
    from astropy.io import fits

    candidates = []
    with fits.open(path, memmap=False) as hdus:
        for index, hdu in enumerate(hdus):
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            name = str(getattr(hdu, "name", "") or f"hdu_{index}").lower()
            frame = fits_data_to_frame(name, data)
            if frame is not None:
                candidates.append(frame)
    if not candidates:
        raise ValueError(f"FITS file does not contain table-like HDUs: {path}")
    return max(candidates, key=lambda frame: (len(frame) * max(len(frame.columns), 1), len(frame.columns)))


def fits_data_to_frame(name: str, data) -> pd.DataFrame | None:
    array = np.asarray(data)
    if array.shape == () or 0 in array.shape:
        return None
    if getattr(array.dtype, "names", None):
        columns = {}
        for field_name in array.dtype.names or ():
            field_columns = fits_array_to_columns(str(field_name), array[field_name])
            if field_columns:
                columns.update(field_columns)
        return pd.DataFrame(columns) if columns else None
    columns = fits_array_to_columns(name, array)
    return pd.DataFrame(columns) if columns else None


def fits_array_to_columns(name: str, values) -> dict[str, object] | None:
    array = np.asarray(values)
    if array.shape == () or 0 in array.shape or array.dtype.kind == "O":
        return None
    if array.dtype.kind == "S":
        decoded = fits_decode_bytes_array(array)
        return {safe_fits_column_name(name): decoded} if decoded is not None else None
    if array.ndim > 1 and array.shape[-1] == 1:
        array = np.squeeze(array, axis=-1)
    if array.ndim == 1:
        return {safe_fits_column_name(name): array.tolist()}
    if array.ndim == 2:
        base = safe_fits_column_name(name)
        return {f"{base}_{idx}": array[:, idx].tolist() for idx in range(array.shape[1])}
    return None


def fits_decode_bytes_array(array) -> list[str] | None:
    if array.ndim == 1:
        return [fits_decode_value(value) for value in array.tolist()]
    if array.ndim == 2:
        values = []
        for row in array:
            flattened = np.asarray(row).ravel().tolist()
            values.append(fits_decode_value(b"".join(flattened)))
        return values
    return None


def fits_decode_value(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00 ")
    return str(value).rstrip("\x00 ")


def safe_fits_column_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_") or "value"


def read_numpy_tabular_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".npy":
        return numpy_array_to_frame(path.stem, np.load(path, allow_pickle=False), source_path=path)
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.files)
        if not keys:
            raise ValueError(f"NumPy archive does not contain arrays: {path}")
        if len(keys) == 1:
            return numpy_array_to_frame(keys[0], archive[keys[0]], source_path=path)
        column_frame = numpy_archive_columns_to_frame({key: archive[key] for key in keys})
        if column_frame is not None:
            return column_frame
        candidates = []
        for key in keys:
            try:
                candidates.append(numpy_array_to_frame(key, archive[key], source_path=path))
            except ValueError:
                continue
        if candidates:
            return max(candidates, key=lambda frame: (len(frame) * max(len(frame.columns), 1), len(frame.columns)))
    raise ValueError(f"NumPy file does not contain a 1D, 2D, or structured tabular array: {path}")


def numpy_archive_columns_to_frame(arrays: dict[str, object]) -> pd.DataFrame | None:
    columns = {}
    row_count: int | None = None
    for key, raw_array in arrays.items():
        array = np.asarray(raw_array)
        if array.ndim != 1 or array.size == 0 or getattr(array.dtype, "names", None):
            continue
        array = decode_numpy_array(array)
        if row_count is None:
            row_count = len(array)
        if len(array) != row_count:
            continue
        columns[str(key)] = array.tolist()
    if not columns:
        return None
    return pd.DataFrame(columns)


def numpy_array_to_frame(name: str, raw_array, *, source_path: Path | None = None) -> pd.DataFrame:
    array = np.asarray(raw_array)
    if array.shape == () or 0 in array.shape:
        raise ValueError("NumPy array is scalar or empty.")
    if getattr(array.dtype, "names", None):
        return pd.DataFrame.from_records(decode_structured_numpy_array(array))
    array = decode_numpy_array(array)
    if array.ndim == 1:
        sidecar_columns = numpy_column_names_for_path(source_path, 1) if source_path is not None else None
        column = sidecar_columns[0] if sidecar_columns else safe_numpy_column_name(name)
        return pd.DataFrame({column: array.tolist()})
    if array.ndim != 2:
        raise ValueError(f"NumPy array must be 1D or 2D for tabular loading, got shape {array.shape}.")
    columns = numpy_column_names_for_path(source_path, array.shape[1]) if source_path is not None else None
    if columns is None:
        base = safe_numpy_column_name(name)
        columns = [base] if array.shape[1] == 1 else [f"{base}_{idx}" for idx in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)


def numpy_column_names_for_path(path: Path | None, width: int) -> list[str] | None:
    if path is None or width <= 0:
        return None
    for candidate in numpy_column_sidecar_candidates(path):
        try:
            raw_columns = parse_numpy_column_sidecar(candidate)
        except (OSError, ValueError, json.JSONDecodeError, UnicodeError):
            continue
        columns = normalize_numpy_column_names(raw_columns, width)
        if columns is not None:
            return columns
    return None


def numpy_column_sidecar_candidates(path: Path) -> list[Path]:
    base = remove_asset_suffix(path.name, file_asset_suffix(path))
    candidates = [
        path.with_name(f"{base}{suffix}")
        for suffix in NUMPY_COLUMN_TEXT_SIDECAR_SUFFIXES + NUMPY_COLUMN_JSON_SIDECAR_SUFFIXES
    ]
    candidates.extend(path.with_name(name) for name in NUMPY_GENERIC_COLUMN_SIDECAR_NAMES)
    deduped = []
    seen = set()
    for candidate in candidates:
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def parse_numpy_column_sidecar(path: Path) -> list[object]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return extract_numpy_column_names(payload)
    return parse_numpy_column_text(path.read_text(encoding="utf-8"))


def parse_numpy_column_text(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) == 1:
        line = lines[0]
        for delimiter in ("\t", ",", ";"):
            if delimiter in line:
                return [token.strip().strip("\"'") for token in next(csv.reader([line], delimiter=delimiter))]
    return [line.strip().strip("\"'") for line in lines]


def extract_numpy_column_names(payload: object) -> list[object]:
    if isinstance(payload, list):
        return extract_numpy_column_list(payload)
    if not isinstance(payload, dict):
        raise ValueError("NumPy column sidecar JSON must be a list or object.")
    for key in ("columns", "feature_names", "features"):
        value = payload.get(key)
        if isinstance(value, list):
            return extract_numpy_column_list(value)
    schema = payload.get("schema")
    if isinstance(schema, dict) and isinstance(schema.get("fields"), list):
        return extract_numpy_column_list(schema["fields"])
    fields = payload.get("fields")
    if isinstance(fields, list):
        return extract_numpy_column_list(fields)
    raise ValueError("NumPy column sidecar JSON does not contain column names.")


def extract_numpy_column_list(values: list[object]) -> list[object]:
    columns = []
    for value in values:
        if isinstance(value, dict):
            if "name" not in value:
                raise ValueError("NumPy column field object is missing a name.")
            columns.append(value["name"])
        else:
            columns.append(value)
    return columns


def normalize_numpy_column_names(columns: list[object], width: int) -> list[str] | None:
    names = [str(column).strip().strip("\"'") for column in columns]
    if len(names) != width or any(not name for name in names):
        return None
    if len(set(names)) != len(names):
        return None
    return names


def decode_structured_numpy_array(array):
    records = []
    for row in array:
        record = {}
        for field_name in array.dtype.names or ():
            record[str(field_name)] = decode_numpy_scalar(row[field_name])
        records.append(record)
    return records


def decode_numpy_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    try:
        return value.item()
    except AttributeError:
        return value


def decode_numpy_array(array):
    decoded = np.asarray(array)
    if decoded.dtype.kind == "S":
        return decoded.astype(str)
    return decoded


def safe_numpy_column_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_") or "value"


def read_arff_tabular_frame(path: Path) -> pd.DataFrame:
    from scipy.io import arff

    with open_html_text(path) as handle:
        data, _metadata = arff.loadarff(handle)
    frame = pd.DataFrame(data)
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(decode_arff_value)
    return frame


def decode_arff_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def read_html_tabular_frame(path: Path) -> pd.DataFrame:
    from io import StringIO

    with open_html_text(path) as handle:
        tables = pd.read_html(StringIO(handle.read()))
    candidates = [table for table in tables if table.shape[1] > 0]
    if not candidates:
        raise ValueError(f"HTML file does not contain table-like data: {path}")
    return max(candidates, key=lambda table: (table.shape[0] * table.shape[1], table.shape[1]))


def read_xml_tabular_frame(path: Path) -> pd.DataFrame:
    from io import StringIO

    with open_html_text(path) as handle:
        return pd.read_xml(StringIO(handle.read()), parser="etree")


def read_svmlight_tabular_frame(path: Path) -> pd.DataFrame:
    from sklearn.datasets import load_svmlight_file

    with open_tabular_binary(path) as handle:
        matrix, target = load_svmlight_file(handle)
    columns = [f"feature_{idx + 1}" for idx in range(matrix.shape[1])]
    frame = pd.DataFrame.sparse.from_spmatrix(matrix, columns=columns)
    if path_mentions_role(path, "train") or np.asarray(target).any():
        frame.insert(0, "target", target)
    return frame


def read_fixed_width_tabular_frame(path: Path) -> pd.DataFrame:
    from io import StringIO

    with open_html_text(path) as handle:
        return pd.read_fwf(StringIO(handle.read()))


@contextmanager
def open_tabular_binary(path: Path):
    suffix = tabular_suffix(path)
    if suffix.endswith(".zip"):
        with zipfile.ZipFile(path, "r") as archive:
            member = select_zip_tabular_member(archive, suffix=suffix)
            yield BytesIO(archive.read(member))
        return
    with open_compressed_binary(path, suffix=suffix) as handle:
        yield handle


def open_html_text(path: Path):
    suffix = tabular_suffix(path)
    if suffix.endswith(".zip"):
        return open_zip_tabular_text(path, suffix=suffix)
    return open_compressed_text(path, suffix=suffix)


@contextmanager
def open_zip_tabular_text(path: Path, suffix: str):
    with zipfile.ZipFile(path, "r") as archive:
        member = select_zip_tabular_member(archive, suffix=suffix)
        with archive.open(member, "r") as binary:
            with TextIOWrapper(binary, encoding="utf-8", errors="ignore") as handle:
                yield handle


def select_zip_tabular_member(archive: zipfile.ZipFile, suffix: str) -> str:
    base_suffix = suffix[: -len(".zip")] if suffix.endswith(".zip") else suffix
    members = [
        info.filename
        for info in archive.infolist()
        if not info.is_dir()
        and not info.filename.startswith("__MACOSX/")
        and Path(info.filename).name
        and not Path(info.filename).name.startswith(".")
    ]
    if not members:
        raise ValueError("zip archive does not contain a tabular member")
    exact = [member for member in members if Path(member).name.lower().endswith(base_suffix)]
    if len(exact) == 1:
        return exact[0]
    if len(members) == 1:
        return members[0]
    if exact:
        return sorted(exact, key=lambda member: (len(Path(member).parts), member.lower()))[0]
    raise ValueError(f"zip archive contains multiple files and no {base_suffix} member")


def read_json_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    suffix = tabular_suffix(path)
    if suffix.startswith(TABULAR_JSON_LINES_SUFFIX_PREFIXES):
        with open_html_text(path) as handle:
            return pd.read_json(StringIO(handle.read()), lines=True, nrows=nrows)
    payload = load_yaml_payload(path) if is_yaml_suffix(suffix) else load_json_payload(path)
    frame = json_payload_to_frame(payload)
    return frame.head(nrows) if nrows is not None else frame


def is_yaml_suffix(suffix: str) -> bool:
    return suffix.startswith((".yaml", ".yml"))


def load_json_payload(path: Path):
    with open_html_text(path) as handle:
        return json.load(handle)


def load_yaml_payload(path: Path):
    import yaml

    with open_html_text(path) as handle:
        return yaml.safe_load(handle) or []


def json_payload_to_frame(payload) -> pd.DataFrame:
    if isinstance(payload, list):
        return json_records_to_frame(payload)
    if not isinstance(payload, dict):
        raise ValueError("JSON table must be an object or list of records.")
    split_frame = frame_from_split_orient(payload)
    if split_frame is not None:
        return split_frame
    geojson_frame = frame_from_geojson_feature_collection(payload)
    if geojson_frame is not None:
        return geojson_frame
    table_data = payload.get("data") if isinstance(payload.get("schema"), dict) else None
    if isinstance(table_data, list):
        return json_records_to_frame(table_data)
    for key in ("data", "records", "rows", "items", "predictions", "submission", "samples", "values"):
        value = payload.get(key)
        if isinstance(value, list):
            return json_records_to_frame(value)
    list_items = [(key, value) for key, value in payload.items() if isinstance(value, list)]
    record_lists = [(key, value) for key, value in list_items if looks_like_json_record_list(value)]
    if len(record_lists) == 1:
        return json_records_to_frame(record_lists[0][1])
    if list_items and all_lists_same_length([value for _, value in list_items]):
        return pd.DataFrame({key: value for key, value in list_items})
    if payload and all(not isinstance(value, (dict, list)) for value in payload.values()):
        return pd.DataFrame([payload])
    return flatten_single_mapping_column(pd.DataFrame(payload))


def frame_from_geojson_feature_collection(payload: dict) -> pd.DataFrame | None:
    if str(payload.get("type", "")).lower() != "featurecollection":
        return None
    features = payload.get("features")
    if not isinstance(features, list):
        return None
    records = []
    for idx, feature in enumerate(features):
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties")
        record = dict(properties) if isinstance(properties, dict) else {}
        if "id" not in record and feature.get("id") is not None:
            record["id"] = feature.get("id")
        geometry = feature.get("geometry")
        if geometry is not None:
            record["geometry"] = json.dumps(geometry, sort_keys=True)
        records.append(record or {"feature_index": idx})
    return pd.DataFrame(records)


def frame_from_split_orient(payload: dict) -> pd.DataFrame | None:
    columns = payload.get("columns")
    data = payload.get("data")
    if not isinstance(columns, list) or not isinstance(data, list):
        return None
    if not all(not isinstance(column, (dict, list)) for column in columns):
        return None
    return pd.DataFrame(data, columns=columns)


def json_records_to_frame(records: list) -> pd.DataFrame:
    if looks_like_json_record_list(records):
        return pd.json_normalize(records)
    return pd.DataFrame(records)


def looks_like_json_record_list(records: list) -> bool:
    return bool(records) and all(isinstance(item, dict) for item in records)


def all_lists_same_length(values: list[list]) -> bool:
    if not values:
        return False
    return len({len(value) for value in values}) == 1


def flatten_single_mapping_column(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame.columns) != 1:
        return frame
    series = frame.iloc[:, 0].dropna()
    if series.empty or not series.map(lambda value: isinstance(value, dict)).all():
        return frame
    return pd.json_normalize(series.tolist())


def read_hdf_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_hdf(path)
    except (KeyError, ValueError) as exc:
        close_pytables_open_files()
        try:
            with pd.HDFStore(path, mode="r") as store:
                keys = store.keys()
            if len(keys) == 1:
                return pd.read_hdf(path, key=keys[0])
        except Exception:
            pass
        fallback = read_native_hdf_table(path)
        if fallback is not None:
            return fallback
        raise exc
    except Exception as exc:
        close_pytables_open_files()
        fallback = read_native_hdf_table(path)
        if fallback is not None:
            return fallback
        raise exc


def close_pytables_open_files() -> None:
    try:
        import tables

        for handle in list(tables.file._open_files._handlers):
            handle.close()
    except Exception:
        pass


def read_native_hdf_table(path: Path) -> pd.DataFrame | None:
    import h5py

    candidates = []
    with h5py.File(path, "r") as handle:
        root_frame = hdf_group_to_column_frame(handle)
        if root_frame is not None:
            candidates.append(root_frame)
        for name, node in iter_hdf_nodes(handle):
            frame = hdf_node_to_frame(name, node)
            if frame is not None:
                candidates.append(frame)
        for _name, node in iter_hdf_nodes(handle):
            if not isinstance(node, h5py.Group):
                continue
            frame = hdf_group_to_column_frame(node)
            if frame is not None:
                candidates.append(frame)
    if not candidates:
        return None
    return max(candidates, key=lambda frame: (len(frame) * max(len(frame.columns), 1), len(frame.columns)))


def iter_hdf_nodes(group):
    for key, node in group.items():
        name = str(node.name or key).strip("/") or str(key)
        yield name, node
        if hasattr(node, "items"):
            yield from iter_hdf_nodes(node)


def hdf_node_to_frame(name: str, node) -> pd.DataFrame | None:
    import h5py

    if not isinstance(node, h5py.Dataset) or node.shape is None or 0 in node.shape:
        return None
    data = node[()]
    if getattr(data.dtype, "names", None):
        return pd.DataFrame.from_records(data)
    data = decode_hdf_array(data)
    if data.ndim == 1:
        return pd.DataFrame({safe_hdf_column_name(name): data.tolist()})
    if data.ndim != 2:
        return None
    columns = hdf_dataset_column_names(node, name, data.shape[1])
    return pd.DataFrame(np.asarray(data), columns=columns)


def hdf_group_to_column_frame(group) -> pd.DataFrame | None:
    import h5py

    columns = {}
    row_count: int | None = None
    for key, node in group.items():
        values = hdf_column_values(node, h5py=h5py)
        if values is None:
            continue
        if row_count is None:
            row_count = len(values)
        if len(values) != row_count:
            continue
        columns[str(key)] = values.tolist()
    if not columns:
        return None
    return pd.DataFrame(columns)


def hdf_column_values(node, *, h5py):
    if isinstance(node, h5py.Dataset):
        if node.shape is None or len(node.shape) != 1 or 0 in node.shape:
            return None
        return decode_hdf_array(node[()])
    if not isinstance(node, h5py.Group):
        return None
    if "codes" not in node or "categories" not in node:
        return None
    codes = decode_hdf_array(node["codes"][()])
    categories = decode_hdf_array(node["categories"][()])
    if codes.ndim != 1 or categories.ndim != 1:
        return None
    values = []
    for code in codes.tolist():
        index = int(code)
        values.append(categories[index] if 0 <= index < len(categories) else None)
    return np.asarray(values, dtype=object)


def decode_hdf_array(data):
    array = np.asarray(data)
    if array.dtype.kind == "S":
        return array.astype(str)
    if array.dtype.kind == "O":
        return np.vectorize(decode_hdf_value, otypes=[object])(array)
    return array


def decode_hdf_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def hdf_dataset_column_names(dataset, name: str, width: int) -> list[str]:
    for attr_name in ("columns", "column_names", "feature_names"):
        raw_columns = dataset.attrs.get(attr_name)
        columns = decode_hdf_columns(raw_columns, width)
        if columns is not None:
            return columns
    safe_name = safe_hdf_column_name(name)
    if width == 1:
        return [safe_name]
    return [f"{safe_name}_{idx}" for idx in range(width)]


def decode_hdf_columns(raw_columns, width: int) -> list[str] | None:
    if raw_columns is None:
        return None
    values = list(raw_columns) if not isinstance(raw_columns, (str, bytes)) else [raw_columns]
    columns = [str(decode_hdf_value(value)) for value in values]
    return columns if len(columns) == width else None


def safe_hdf_column_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name).strip("/")).strip("_") or "value"


def read_h5ad_tabular_frame(path: Path) -> pd.DataFrame:
    import h5py

    with h5py.File(path, "r") as handle:
        obs_frame = hdf_group_to_column_frame(handle["obs"]) if "obs" in handle else None
        x_frame = h5ad_x_to_frame(handle)
    frame = combine_same_row_frames(obs_frame, x_frame)
    if frame is not None:
        return frame
    fallback = read_native_hdf_table(path)
    if fallback is not None:
        return fallback
    raise ValueError(f"AnnData file does not contain table-like obs/X data: {path}")


def h5ad_x_to_frame(handle) -> pd.DataFrame | None:
    if "X" not in handle:
        return None
    raw_matrix = h5ad_x_to_array(handle["X"])
    if raw_matrix is None:
        return None
    matrix = np.asarray(raw_matrix)
    if matrix.ndim != 2 or 0 in matrix.shape:
        return None
    columns = h5ad_var_names(handle, matrix.shape[1]) or [f"x_{idx}" for idx in range(matrix.shape[1])]
    return pd.DataFrame(matrix, columns=columns)


def h5ad_x_to_array(node):
    import h5py
    from scipy import sparse

    if isinstance(node, h5py.Dataset):
        return decode_hdf_array(node[()])
    if not isinstance(node, h5py.Group) or not {"data", "indices", "indptr", "shape"} <= set(node.keys()):
        return None
    data = node["data"][()]
    indices = node["indices"][()]
    indptr = node["indptr"][()]
    shape = tuple(int(value) for value in np.asarray(node["shape"][()]).tolist())
    encoding = str(decode_hdf_value(node.attrs.get("encoding-type", b"csr_matrix"))).lower()
    matrix = sparse.csc_matrix((data, indices, indptr), shape=shape) if "csc" in encoding else sparse.csr_matrix(
        (data, indices, indptr),
        shape=shape,
    )
    return matrix.toarray()


def h5ad_var_names(handle, width: int) -> list[str] | None:
    import h5py

    var = handle.get("var")
    if var is None:
        return None
    for key in ("_index", "index", "gene_symbols", "gene_ids", "feature_name", "features"):
        if key not in var:
            continue
        values = hdf_column_values(var[key], h5py=h5py)
        columns = normalize_hdf_feature_names(values, width)
        if columns is not None:
            return columns
    return None


def read_loom_tabular_frame(path: Path) -> pd.DataFrame:
    import h5py

    with h5py.File(path, "r") as handle:
        attrs_frame = hdf_group_to_column_frame(handle["col_attrs"]) if "col_attrs" in handle else None
        matrix = decode_hdf_array(handle["matrix"][()]) if "matrix" in handle else None
        if matrix is None or np.asarray(matrix).ndim != 2 or 0 in np.asarray(matrix).shape:
            matrix_frame = None
        else:
            matrix = np.asarray(matrix).T
            columns = loom_row_feature_names(handle, matrix.shape[1]) or [f"x_{idx}" for idx in range(matrix.shape[1])]
            matrix_frame = pd.DataFrame(matrix, columns=columns)
    frame = combine_same_row_frames(attrs_frame, matrix_frame)
    if frame is not None:
        return frame
    fallback = read_native_hdf_table(path)
    if fallback is not None:
        return fallback
    raise ValueError(f"Loom file does not contain table-like col_attrs/matrix data: {path}")


def loom_row_feature_names(handle, width: int) -> list[str] | None:
    import h5py

    row_attrs = handle.get("row_attrs")
    if row_attrs is None:
        return None
    for key in ("Gene", "gene", "GeneName", "gene_name", "Accession", "feature_name"):
        if key not in row_attrs:
            continue
        values = hdf_column_values(row_attrs[key], h5py=h5py)
        columns = normalize_hdf_feature_names(values, width)
        if columns is not None:
            return columns
    return None


def normalize_hdf_feature_names(values, width: int) -> list[str] | None:
    if values is None:
        return None
    names = [safe_hdf_column_name(str(value)) for value in values.tolist()]
    if len(names) != width or any(not name for name in names):
        return None
    if len(set(names)) != len(names):
        return None
    return names


def combine_same_row_frames(left, right) -> pd.DataFrame | None:
    frames = [frame for frame in (left, right) if frame is not None and not frame.empty]
    if not frames:
        return None
    row_count = len(frames[0])
    if any(len(frame) != row_count for frame in frames):
        return max(frames, key=lambda frame: (len(frame) * max(len(frame.columns), 1), len(frame.columns)))
    combined = pd.concat([frame.reset_index(drop=True) for frame in frames], axis=1)
    combined.columns = dedupe_column_names([str(column) for column in combined.columns])
    return combined


def dedupe_column_names(columns: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    deduped = []
    for column in columns:
        base = str(column)
        count = counts.get(base, 0)
        deduped.append(base if count == 0 else f"{base}_{count}")
        counts[base] = count + 1
    return deduped


def read_avro_table(path: Path) -> pd.DataFrame:
    from fastavro import reader

    with open_tabular_binary(path) as handle:
        avro_reader = reader(handle)
        schema = getattr(avro_reader, "writer_schema", {}) or {}
        records = list(avro_reader)
    columns = [str(field.get("name")) for field in schema.get("fields", []) if field.get("name") is not None]
    return pd.DataFrame(records, columns=columns or None)


def write_avro_table(frame: pd.DataFrame, path: Path) -> None:
    from fastavro import writer

    fields = [{"name": str(column), "type": ["null", avro_field_type(frame[column])]} for column in frame.columns]
    schema = {"type": "record", "name": "SubmissionRecord", "fields": fields}
    with path.open("wb") as handle:
        writer(handle, schema, frame_to_avro_records(frame, fields))


def avro_field_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "long"
    if pd.api.types.is_float_dtype(series):
        return "double"
    return "string"


def frame_to_avro_records(frame: pd.DataFrame, fields: list[dict[str, object]]) -> list[dict[str, object]]:
    field_types = {}
    for field in fields:
        field_type = field["type"]
        candidates = field_type if isinstance(field_type, list) else [field_type]
        field_types[str(field["name"])] = next(candidate for candidate in candidates if candidate != "null")
    records = []
    for row in frame.to_dict(orient="records"):
        record = {}
        for key, value in row.items():
            name = str(key)
            if is_missing_avro_value(value):
                record[name] = None
                continue
            value_type = field_types[name]
            if value_type == "boolean":
                record[name] = bool(value)
            elif value_type == "long":
                record[name] = int(value)
            elif value_type == "double":
                record[name] = float(value)
            else:
                record[name] = str(value)
        records.append(record)
    return records


def is_missing_avro_value(value: object) -> bool:
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def read_table_with_string_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    string_columns = [str(col) for col in columns if str(col).strip()]
    if not string_columns:
        return read_table(path)
    suffix = tabular_suffix(path)
    if suffix in TABULAR_TEXT_SUFFIXES:
        frame = read_text_tabular_frame(path, dtype={col: str for col in string_columns})
        if frame_has_leading_zero_id_values(frame, string_columns):
            return frame
    return read_table(path)


def frame_has_leading_zero_id_values(frame: pd.DataFrame, columns: list[str]) -> bool:
    for col in columns:
        if col not in frame.columns:
            continue
        values = frame[col].dropna().astype(str).str.strip()
        if values.str.fullmatch(r"0\d+").any():
            return True
    return False


def preserve_id_column_values(
    test_path: Path,
    sample_path: Path,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    id_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not id_col:
        return test, sample
    if id_col in test.columns:
        test = read_table_with_string_columns(test_path, [id_col])
    if id_col in sample.columns:
        sample = read_table_with_string_columns(sample_path, [id_col])
    return test, sample


def submission_id_alignment_key(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def assign_prediction_column(
    submission: pd.DataFrame,
    test: pd.DataFrame,
    id_col: str | None,
    target_col: str,
    preds,
) -> None:
    values = np.asarray(preds)
    if id_col and id_col in submission.columns:
        if id_col in test.columns:
            mapping = pd.Series(values, index=test[id_col].map(submission_id_alignment_key))
            submission[target_col] = submission[id_col].map(submission_id_alignment_key).map(mapping)
            return
        composite_ids = infer_test_composite_submission_ids(submission[id_col], test)
        if composite_ids is not None:
            mapping = pd.Series(values, index=composite_ids.map(submission_id_alignment_key))
            submission[target_col] = submission[id_col].map(submission_id_alignment_key).map(mapping)
            return
    submission[target_col] = values


def infer_test_composite_submission_ids(sample_ids: pd.Series, test: pd.DataFrame) -> pd.Series | None:
    sample_keys = sample_ids.map(submission_id_alignment_key)
    if sample_keys.empty or sample_keys.duplicated().any():
        return None
    target_key_set = set(sample_keys)
    candidate_cols = composite_id_candidate_columns(test)
    for width in (2, 3):
        if len(candidate_cols) < width:
            continue
        for cols in itertools.combinations(candidate_cols, width):
            for sep in COMPOSITE_ID_SEPARATORS:
                composite = join_composite_id_columns(test, cols, sep)
                composite_keys = composite.map(submission_id_alignment_key)
                if composite_keys.duplicated().any():
                    continue
                if set(composite_keys) == target_key_set:
                    return composite
    return None


def composite_id_candidate_columns(test: pd.DataFrame) -> list[str]:
    id_like = []
    other = []
    for col in test.columns:
        series = test[col]
        if pd.api.types.is_float_dtype(series):
            continue
        compact = re.sub(r"[^a-z0-9]+", "", str(col).lower())
        tokens = set(re.findall(r"[a-z0-9]+", str(col).lower()))
        target = id_like if compact.endswith("id") or "id" in tokens or compact in {"user", "item", "movie"} else other
        target.append(str(col))
    return [*id_like, *other][:8]


def join_composite_id_columns(test: pd.DataFrame, columns, sep: str) -> pd.Series:
    parts = [test[col].map(submission_id_alignment_key) for col in columns]
    output = parts[0].astype(str)
    for part in parts[1:]:
        output = output + sep + part.astype(str)
    return output


COMPOSITE_ID_SEPARATORS = ("_", "-", "/", ":", ".", "|", "")


def sniff_delimiter(path: Path, default: str = ",") -> str:
    try:
        with open_html_text(path) as handle:
            line = next((candidate for candidate in handle if candidate.strip()), "")
    except OSError:
        return default
    if not line:
        return default
    counts = {sep: line.count(sep) for sep in (default, ",", chr(9), ";", "|")}
    best = max(counts, key=lambda sep: counts[sep])
    if counts[best] > 0:
        return best
    if looks_space_delimited(line):
        return r"\s+"
    return default


def read_text_tabular_frame(
    path: Path,
    dtype: dict[str, object] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    suffix = tabular_suffix(path)
    default_sep = default_delimited_text_separator(suffix)
    with open_html_text(path) as handle:
        return pd.read_csv(
            StringIO(handle.read()),
            sep=sniff_delimiter(path, default=default_sep),
            dtype=dtype,
            nrows=nrows,
        )


def looks_space_delimited(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if any(separator in stripped for separator in (",", chr(9), ";", "|")):
        return False
    parts = stripped.split()
    return len(parts) >= 2 and all(part.strip() for part in parts)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = finalize_table_frame(frame.copy())
    suffix = tabular_suffix(path)
    base_suffix = tabular_base_suffix(suffix)
    if suffix in TABULAR_PARQUET_SUFFIXES:
        frame.to_parquet(path, index=False)
        return
    if suffix == ".orc":
        frame.to_orc(path, index=False)
        return
    if suffix in TABULAR_HDF_SUFFIXES:
        frame.to_hdf(path, key="submission", mode="w", format="table", index=False)
        return
    if suffix in TABULAR_ARROW_IPC_SUFFIXES:
        frame.to_feather(path)
        return
    if suffix == ".avro":
        write_avro_table(frame, path)
        return
    if suffix in TABULAR_EXCEL_SUFFIXES:
        frame.to_excel(path, index=False)
        return
    if suffix in TABULAR_STATA_SUFFIXES:
        frame.to_stata(path, write_index=False)
        return
    if suffix.startswith(".xml"):
        write_xml_table(frame, path)
        return
    if suffix.startswith(TABULAR_HTML_SUFFIX_PREFIXES):
        write_compressed_text(path, frame.to_html(index=False))
        return
    if suffix in TABULAR_PICKLE_SUFFIXES:
        frame.to_pickle(path)
        return
    if suffix in TABULAR_STRUCTURED_SUFFIXES:
        if is_yaml_suffix(base_suffix):
            import yaml

            payload = yaml.safe_dump(frame.to_dict(orient="records"), sort_keys=False)
        else:
            payload = frame.to_json(orient="records", lines=base_suffix in TABULAR_JSON_LINES_SUFFIX_PREFIXES)
        write_compressed_text(path, payload)
        return
    if suffix in TABULAR_TEXT_SUFFIXES:
        sep = default_delimited_text_separator(suffix)
        write_compressed_text(path, frame.to_csv(index=False, sep=sep))
        return
    fallback = path.with_name(f"{requested_output_stem(path)}.tabular{FALLBACK_TABULAR_SUBMISSION_SUFFIX}")
    write_table(frame, fallback)
    manifest = path.parent / "submission_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_class": "tabular",
                "submission_path": fallback.name,
                "requested_output_path": path.name,
                "note": "Generated tabular baseline could not produce the requested non-tabular artifact directly.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_xml_table(frame: pd.DataFrame, path: Path) -> None:
    payload = frame.to_xml(index=False, parser="etree").encode("utf-8")
    write_compressed_bytes(path, payload)


def tabular_base_suffix(suffix: str) -> str:
    for compression_suffix in (*ASSET_COMPRESSION_SUFFIXES, ".zip"):
        if suffix.endswith(compression_suffix):
            return suffix[: -len(compression_suffix)]
    return suffix


def default_delimited_text_separator(suffix: str) -> str:
    base_suffix = tabular_base_suffix(suffix)
    if base_suffix == ".psv":
        return "|"
    if base_suffix in TABULAR_TAB_DELIMITED_SUFFIX_PREFIXES or base_suffix == ".txt":
        return chr(9)
    return ","


def sas_format_for_suffix(suffix: str) -> str | None:
    base_suffix = tabular_base_suffix(suffix)
    if base_suffix in {".xpt", ".xport"}:
        return "xport"
    if base_suffix == ".sas7bdat":
        return "sas7bdat"
    return None


def requested_output_stem(path: Path) -> str:
    name = path.name
    lowered = name.lower()
    suffix = next(
        (
            candidate
            for candidate in sorted(SUBMISSION_FILE_ASSET_SUFFIXES | ARCHIVE_OUTPUT_SUFFIXES, key=len, reverse=True)
            if lowered.endswith(candidate)
        ),
        path.suffix.lower(),
    )
    stem = name[: -len(suffix)] if suffix and lowered.endswith(suffix) else path.stem
    return stem.strip(".") or "submission"


def write_compressed_text(path: Path, text: str) -> None:
    write_compressed_bytes(path, text.encode("utf-8"))


def write_compressed_bytes(path: Path, payload: bytes) -> None:
    suffix = tabular_suffix(path)
    compression_suffix = compression_suffix_for(suffix)
    if compression_suffix == ".gz":
        with gzip.open(path, "wb", compresslevel=9) as handle:
            handle.write(payload)
        return
    if compression_suffix == ".bz2":
        with bz2.open(path, "wb") as handle:
            handle.write(payload)
        return
    if compression_suffix == ".xz":
        with lzma.open(path, "wb") as handle:
            handle.write(payload)
        return
    if compression_suffix == ".zst":
        import zstandard as zstd

        path.write_bytes(zstd.ZstdCompressor(level=9).compress(payload))
        return
    path.write_bytes(payload)


def tabular_file_has_data_rows(path: Path) -> bool:
    suffix = tabular_suffix(path)
    if suffix in TABULAR_TEXT_SUFFIXES:
        non_empty = 0
        try:
            with open_html_text(path) as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    non_empty += 1
                    if non_empty >= 2:
                        return True
        except OSError:
            return False
        return False
    try:
        return len(read_table(path)) > 0
    except Exception:
        return False


def tabular_file_has_two_or_more_columns(path: Path) -> bool:
    try:
        return len(read_table(path).columns) >= 2
    except Exception:
        return False


def sample_submission_name_score(path: Path) -> int:
    stem = tabular_stem(path).lower()
    compact = re.sub(r"[^a-z0-9]+", "", stem)
    tokens = name_tokens(stem)
    output_tokens = SAMPLE_OUTPUT_NAME_TOKENS
    if "sample_submission" in stem or "samplesubmission" in compact:
        return 3
    if "sample" in tokens and tokens & output_tokens:
        return 2
    if "example" in tokens and tokens & output_tokens:
        return 2
    if tokens & {"template", "templates"} and tokens & output_tokens:
        return 2
    if compact in SAMPLE_COMPACT_NAME_ALIASES:
        return 2
    if "submission" in stem:
        return 1
    return 0


def has_usable_sample_submission(files: list[Path]) -> bool:
    for path in files:
        if sample_submission_name_score(path) <= 0:
            continue
        if tabular_file_has_data_rows(path) and tabular_file_has_two_or_more_columns(path):
            return True
    return False


def find_submission_format_documents() -> list[Path]:
    document_names = ("submission_format.md", "overview.md", "data.md", "rules.md", "discussion.md")
    documents = []
    seen = set()
    for root in data_roots():
        candidates = []
        for name in document_names:
            candidates.extend([root / "context" / name, root.parent / "context" / name])
        try:
            for name in document_names:
                candidates.extend(sorted(root.rglob(name)))
        except OSError:
            pass
        for candidate in candidates:
            key = candidate.as_posix()
            if key in seen or not candidate.is_file():
                continue
            seen.add(key)
            documents.append(candidate)
    return documents


def submission_format_candidate_text(document: Path, text: str) -> str:
    if document.name.lower() == "submission_format.md":
        return text
    section = extract_submission_context_section(text)
    return section or ""


def extract_submission_context_section(text: str) -> str | None:
    lines = text.splitlines()
    heading_ranges = []
    for idx, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if match is None:
            continue
        heading = match.group(2).strip().lower()
        if not re.search(r"\b(submission|submissions|submit|format)\b", heading):
            continue
        level = len(match.group(1))
        end = len(lines)
        for next_idx in range(idx + 1, len(lines)):
            next_match = re.match(r"^(#{1,6})\s+(.+)$", lines[next_idx].strip())
            if next_match is not None and len(next_match.group(1)) <= level:
                end = next_idx
                break
        heading_ranges.append((idx, end))
    if heading_ranges:
        snippets = ["\n".join(lines[start:end]) for start, end in heading_ranges]
        return "\n\n".join(snippets)

    snippet_ranges = []
    for idx, line in enumerate(lines):
        lowered = line.lower()
        if not re.search(r"\b(submission|submissions|submit|prediction file|csv header)\b", lowered):
            continue
        start = max(0, idx - 5)
        end = min(len(lines), idx + 25)
        snippet_ranges.append((start, end))
    if not snippet_ranges:
        return None

    merged = []
    for start, end in snippet_ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return "\n\n".join("\n".join(lines[start:end]) for start, end in merged)


def clean_submission_column_name(value: str) -> str | None:
    cleaned = value.strip().strip("|").strip().strip("`'\"")
    cleaned = cleaned.strip().rstrip(".,;:")
    cleaned = re.sub(r"^\*\*(.*)\*\*$", r"\1", cleaned).strip()
    if not cleaned or len(cleaned) > 120:
        return None
    if re.search(r"[\[\]{}()<>/\\]", cleaned):
        return None
    return cleaned


def split_submission_columns(value: str) -> list[str]:
    if "\t" in value:
        raw_parts = value.split("\t")
    elif "," in value:
        raw_parts = value.split(",")
    else:
        raw_parts = re.split(r"\s+", value.strip())
    columns = []
    for part in raw_parts:
        column = clean_submission_column_name(part)
        if column is not None:
            columns.append(column)
    return columns


def parse_json_submission_columns_from_line(line: str) -> list[str] | None:
    candidates = [line.strip()]
    object_match = re.search(r"(\{.*\}|\[.*\])", line, flags=re.S)
    if object_match is not None:
        candidates.append(object_match.group(1))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            payload = payload[0]
        if not isinstance(payload, dict):
            continue
        columns = []
        for key in payload.keys():
            column = clean_submission_column_name(str(key))
            if column is not None and column not in columns:
                columns.append(column)
        if len(columns) >= 2:
            return columns
    return None


def parse_markdown_submission_columns(lines: list[str]) -> list[str] | None:
    generic_first_column_headers = {"column", "columns", "field", "fields", "name", "names"}
    for idx, line in enumerate(lines[:-1]):
        next_line = lines[idx + 1]
        if "|" not in line or not re.search(r"\|\s*:?-{3,}:?\s*(\||$)", next_line):
            continue
        cells = [
            clean_submission_column_name(cell)
            for cell in line.strip().strip("|").split("|")
        ]
        columns = [cell for cell in cells if cell is not None]
        if len(columns) < 2:
            continue
        if columns[0].lower() not in generic_first_column_headers:
            return columns

        described_columns = []
        for body_line in lines[idx + 2 :]:
            if "|" not in body_line:
                break
            body_cells = [
                clean_submission_column_name(cell)
                for cell in body_line.strip().strip("|").split("|")
            ]
            if body_cells and body_cells[0] is not None:
                described_columns.append(body_cells[0])
        if len(described_columns) >= 2:
            return described_columns
    return None


def parse_submission_format_columns(text: str) -> list[str] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    markdown_columns = parse_markdown_submission_columns(lines)
    if markdown_columns is not None:
        return markdown_columns

    in_code_block = False
    code_lines = []
    for line in lines:
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            code_lines.append(line)
    code_json_columns = parse_json_submission_columns_from_line("\n".join(code_lines))
    if code_json_columns is not None:
        return code_json_columns
    for line in code_lines:
        json_columns = parse_json_submission_columns_from_line(line)
        if json_columns is not None:
            return json_columns
        columns = split_submission_columns(line)
        if len(columns) >= 2:
            return columns

    for line in lines:
        json_columns = parse_json_submission_columns_from_line(line)
        if json_columns is not None:
            return json_columns
        if "column" in line.lower():
            quoted = [
                clean_submission_column_name(value)
                for value in re.findall(r"`([^`\n]+)`", line)
            ]
            quoted = [value for value in quoted if value is not None]
            if len(quoted) >= 2:
                return quoted

            match = re.search(
                r"\bcolumns?\b(?:\s+(?:are|is|include|includes|named))?\s*[:=]?\s+(.+)",
                line,
                flags=re.IGNORECASE,
            )
            if match:
                fragment = match.group(1).split(".", 1)[0]
                columns = split_submission_columns(fragment)
                if len(columns) >= 2:
                    return columns
        if "," in line or "\t" in line:
            header_match = re.search(r"\b(?:csv\s+)?header\b\s*[:=]\s*(.+)", line, flags=re.IGNORECASE)
            fragment = header_match.group(1) if header_match else line
            columns = split_submission_columns(fragment)
            if len(columns) >= 2:
                return columns
    return None


def asset_relative_path(path: Path) -> str:
    for root in data_roots():
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            continue
    return path.name


def asset_role_relative_path(path: Path) -> str:
    parts = Path(asset_relative_path(path)).parts
    role_tokens = role_aliases("test")
    for idx, part in enumerate(parts):
        if str(part).lower() in role_tokens:
            return "/".join(parts[idx:])
    return asset_relative_path(path)


def infer_asset_submission_id_style(format_text: str) -> str:
    suffix_pattern = "|".join(
        re.escape(suffix)
        for suffix in sorted(FILE_ASSET_SUFFIXES, key=len, reverse=True)
        if suffix
    )
    if not suffix_pattern:
        return "stem"
    pattern = rf"[\w./-]+(?:{suffix_pattern})"
    examples = re.findall(pattern, format_text, flags=re.IGNORECASE)
    for example in examples:
        normalized = example.replace("\\", "/").strip("./").lower()
        parts = [part for part in normalized.split("/") if part]
        if len(parts) >= 2 and parts[0] in role_aliases("test"):
            return "role_relative"
        if "/" in normalized:
            return "relative"
        return "name"
    return "stem"


def format_asset_submission_id(path: Path, style: str) -> str:
    if style == "relative":
        return asset_relative_path(path)
    if style == "role_relative":
        return asset_role_relative_path(path)
    if style == "name":
        return path.name
    return file_asset_stem(path)


def find_test_asset_files() -> list[Path]:
    assets = []
    seen = set()
    for root in data_roots():
        for path in sorted(root.rglob("*")):
            if not is_file_asset_path(path):
                continue
            if not path_mentions_role(path, "test") or path_mentions_role(path, "train"):
                continue
            key = path.as_posix()
            if key in seen:
                continue
            seen.add(key)
            assets.append(path)
    return assets


def compressed_tabular_suffix_from_keywords(text: str) -> str | None:
    compression_suffix = None
    for pattern, suffix in COMPRESSION_TOKEN_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            compression_suffix = suffix
            break
    if compression_suffix is None:
        return None
    for pattern, suffix in SUBMISSION_TOKEN_PATTERNS:
        if suffix not in COMPRESSIBLE_SUBMISSION_KEYWORD_SUFFIXES:
            continue
        if not re.search(pattern, text, flags=re.IGNORECASE):
            continue
        combined = f"{suffix}{compression_suffix}"
        if combined in TABULAR_OUTPUT_SUFFIXES:
            return combined
    return None


def ranked_compressed_tabular_suffix(format_text: str) -> str | None:
    suffix_mentions = []
    for idx, raw_line in enumerate(format_text.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        suffix = compressed_tabular_suffix_from_keywords(line)
        if suffix is None:
            continue
        line_lower = line.lower()
        score = 0
        if re.search(r"\b(submit|submission|upload|output|prediction file|required|format)\b", line_lower):
            score += 4
        if re.search(r"\b(sample[_ -]?submission|submission[._ -])", line_lower):
            score += 2
        if re.search(r"\b(input|train|training|test|feature|dataset|data files?)\b", line_lower):
            score -= 2
        suffix_mentions.append((score, -idx, len(suffix), suffix))
    ranked_mentions = [item for item in suffix_mentions if item[0] > 0]
    if ranked_mentions:
        return max(ranked_mentions)[3]
    return None


def ranked_tabular_suffix_from_keywords(format_text: str) -> str | None:
    suffix_mentions = []
    for idx, raw_line in enumerate(format_text.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        line_lower = line.lower()
        score = 0
        if re.search(r"\b(submit|submission|upload|output|prediction file|required|format)\b", line_lower):
            score += 4
        if re.search(r"\b(sample[_ -]?submission|submission[._ -])", line_lower):
            score += 2
        if re.search(r"\b(input|train|training|test|feature|dataset|data files?)\b", line_lower):
            score -= 2
        for pattern, suffix in SUBMISSION_TOKEN_PATTERNS:
            if suffix not in TABULAR_OUTPUT_SUFFIXES:
                continue
            if re.search(pattern, line, flags=re.IGNORECASE):
                suffix_mentions.append((score, -idx, len(suffix), suffix))
    ranked_mentions = [item for item in suffix_mentions if item[0] > 0]
    if ranked_mentions:
        return max(ranked_mentions)[3]
    return None


def synthesized_sample_suffix(format_text: str) -> str:
    lowered = format_text.lower()
    suffix_mentions = []
    for idx, raw_line in enumerate(format_text.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        line_lower = line.lower()
        mentioned_suffixes = [
            suffix
            for suffix in sorted(TABULAR_OUTPUT_SUFFIXES, key=len, reverse=True)
            if suffix and suffix in line_lower
        ]
        if not mentioned_suffixes:
            continue
        score = 0
        if re.search(r"\b(submit|submission|upload|output|prediction file|required|format)\b", line_lower):
            score += 4
        if re.search(r"\b(sample[_ -]?submission|submission[._ -])", line_lower):
            score += 2
        if re.search(r"\b(input|train|training|test|feature|dataset|data files?)\b", line_lower):
            score -= 2
        for suffix in mentioned_suffixes:
            suffix_mentions.append((score, -idx, len(suffix), suffix))
    ranked_mentions = [item for item in suffix_mentions if item[0] > 0]
    if ranked_mentions:
        return max(ranked_mentions)[3]

    compressed_suffix = ranked_compressed_tabular_suffix(format_text)
    if compressed_suffix is not None:
        return compressed_suffix

    fenced_languages = {
        match.group(1).lower()
        for match in re.finditer(r"```([A-Za-z0-9_.+-]+)", format_text)
    }
    for language, suffix in CODE_FENCE_LANG_TO_SUFFIX.items():
        if language in fenced_languages and suffix in TABULAR_OUTPUT_SUFFIXES:
            return suffix

    token_suffix = ranked_tabular_suffix_from_keywords(format_text)
    if token_suffix is not None:
        return token_suffix
    return ".csv"


def write_synthesized_sample_submission(
    submission: pd.DataFrame,
    source_doc: Path | None,
    format_text: str,
) -> Path:
    SYNTHETIC_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SYNTHETIC_TABLE_DIR / f"sample_submission_synth{synthesized_sample_suffix(format_text)}"
    write_table(submission, output_path)
    print(f"synthesized sample submission from {source_doc}: {output_path}")
    return output_path


def synthesized_submission_default_value(column: str, format_text: str) -> object:
    lowered = str(column).lower()
    normalized = normalize_submission_column_token(str(column))
    if normalized in RLE_SEGMENTATION_COLUMN_TOKENS:
        return "-"
    if "encodedpixels" in normalized or "runlength" in normalized or normalized.endswith("rle"):
        return "-"
    if any(token in lowered for token in TEXT_PREDICTION_NAME_TOKENS):
        return "-"
    column_mentions = [
        match.group(0).lower()
        for match in re.finditer(rf".{{0,40}}{re.escape(str(column))}.{{0,40}}", format_text, flags=re.IGNORECASE)
    ]
    text_type_pattern = r"\b(string|str|text|sentence|caption|answer|description)\b"
    if any(re.search(text_type_pattern, mention) for mention in column_mentions):
        return "-"
    return 0


def synthesize_tabular_sample_submission(
    columns: list[str],
    files: list[Path],
    source_doc: Path | None,
    format_text: str,
) -> Path | None:
    _, test_path = select_train_test_paths(files, sample_path=None)
    if test_path is None:
        return None
    try:
        test = read_table(test_path)
    except Exception:
        return None
    if test.empty or not len(test.columns):
        return None

    sample_id = columns[0]
    test_by_lower = {str(column).lower(): column for column in test.columns}
    id_column = test_by_lower.get(sample_id.lower())
    if id_column is None and is_id_like_column(sample_id):
        for column in test.columns:
            if is_id_like_column(column):
                id_column = column
                break
    if id_column is None:
        return None

    submission = pd.DataFrame({sample_id: test[id_column].tolist()})
    for column in columns[1:]:
        if column not in submission.columns:
            submission[column] = synthesized_submission_default_value(column, format_text)
    return write_synthesized_sample_submission(submission, source_doc, format_text)


def synthesize_asset_sample_submission(
    columns: list[str],
    format_text: str,
    source_doc: Path | None,
) -> Path | None:
    assets = find_test_asset_files()
    if not assets:
        return None
    style = infer_asset_submission_id_style(format_text)
    sample_id = columns[0]
    submission = pd.DataFrame(
        {sample_id: [format_asset_submission_id(path, style) for path in assets]}
    )
    for column in columns[1:]:
        if column not in submission.columns:
            submission[column] = synthesized_submission_default_value(column, format_text)
    return write_synthesized_sample_submission(submission, source_doc, format_text)


def synthesize_sample_submission_from_format(files: list[Path]) -> Path | None:
    if has_usable_sample_submission(files):
        return None

    columns = None
    source_doc = None
    format_text = ""
    for document in find_submission_format_documents():
        try:
            raw_text = document.read_text(encoding="utf-8", errors="ignore")
            format_text = submission_format_candidate_text(document, raw_text)
            columns = parse_submission_format_columns(format_text)
        except OSError:
            continue
        if columns is not None:
            source_doc = document
            break
    if columns is None:
        return None

    synthesized = synthesize_tabular_sample_submission(columns, files, source_doc, format_text)
    if synthesized is not None:
        return synthesized
    return synthesize_asset_sample_submission(columns, format_text, source_doc)


def pick_files(files: list[Path]) -> tuple[Path, Path, Path]:
    if not files:
        raise FileNotFoundError(f"No tabular files found under {INPUT_ROOT}.")

    def score_sample(path: Path) -> int:
        return sample_submission_name_score(path)

    def sample_stage_score(path: Path) -> int:
        matches = SAMPLE_STAGE_RE.findall(path.name.lower())
        if not matches:
            return 0
        return max(int(value) for value in matches)

    def desired_submission_stage() -> int | None:
        raw = (
            os.getenv("KAGGLEBOT_SUBMISSION_STAGE")
            or os.getenv("KAGGLEBOT_SAMPLE_SUBMISSION_STAGE")
            or ""
        ).strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError:
            return None
        return value if value > 0 else None

    def sample_candidate_key(path: Path):
        name_score = score_sample(path)
        stage_score = sample_stage_score(path)
        desired_stage = desired_submission_stage()
        stage_match = 1 if desired_stage is not None and stage_score == desired_stage else 0
        explicit_stage = 1 if stage_score > 0 else 0
        desired_distance = 0
        if desired_stage is not None:
            desired_distance = -abs(stage_score - desired_stage) if stage_score else -10000
        try:
            row_count = len(read_table(path))
        except Exception:
            row_count = 0
        return (
            name_score,
            stage_match,
            explicit_stage,
            desired_distance,
            stage_score if stage_score > 0 else 0,
            row_count,
            path.name.lower(),
        )

    sample_candidates = [path for path in files if score_sample(path) > 0]
    usable_sample_candidates = [
        path
        for path in sample_candidates
        if tabular_file_has_data_rows(path) and tabular_file_has_two_or_more_columns(path)
    ]
    ranked_sample_candidates = usable_sample_candidates or sample_candidates
    sample_path = (
        max(ranked_sample_candidates, key=sample_candidate_key)
        if ranked_sample_candidates
        else None
    )

    train_path, test_path = select_train_test_paths(files, sample_path)

    if sample_path is not None and (train_path is None or test_path is None):
        synthesized = synthesize_train_test_from_assets(files, train_path, sample_path)
        if synthesized is not None:
            return synthesized

    if train_path is None or test_path is None:
        raise FileNotFoundError("Unable to locate train/test files in competition data.")
    if sample_path is None:
        raise FileNotFoundError("Unable to locate sample submission file in competition data.")

    return train_path, test_path, sample_path


def is_asset_label_table_path(path: Path) -> bool:
    name = path.name.lower()
    if "sample" in name or "submission" in name:
        return False
    return any(token in name for token in ASSET_LABEL_TABLE_TOKENS)


def is_label_table_path(path: Path) -> bool:
    name = path.name.lower()
    stem = path.stem.lower()
    if "sample" in name or "submission" in name:
        return False
    return any(token in stem for token in ASSET_LABEL_TABLE_TOKENS)


def synthesize_train_test_from_assets(
    files: list[Path],
    train_path: Path | None,
    sample_path: Path,
) -> tuple[Path, Path, Path] | None:
    label_paths = []
    if train_path is not None:
        label_paths.append(train_path)
    label_paths.extend(
        path
        for path in files
        if path != sample_path
        and path != train_path
        and ".kagglebot_cache" not in path.parts
        and is_asset_label_table_path(path)
    )
    sample = read_table(sample_path)
    if sample.empty or not len(sample.columns):
        return None
    sample_id = str(sample.columns[0])
    if not is_id_like_column(sample_id):
        return None
    asset_index = build_file_asset_index(data_roots())
    if not asset_index:
        return None

    for candidate in label_paths:
        try:
            labels = read_table(candidate)
        except Exception:
            continue
        if labels.empty or not len(labels.columns):
            continue
        label_id = resolve_asset_label_id_column(sample_id, labels)
        if label_id not in labels.columns:
            continue

        train_ids = labels[label_id].astype(str).str.strip()
        test_ids = sample[sample_id].astype(str).str.strip()
        train_asset_paths = train_ids.map(lambda value: resolve_asset_path(value, asset_index, split="train"))
        test_asset_paths = test_ids.map(lambda value: resolve_asset_path(value, asset_index, split="test"))
        train = labels.copy()
        train[label_id] = train_ids
        train["asset_path"] = train_asset_paths.map(lambda path: str(path) if path is not None else np.nan)
        test = pd.DataFrame({sample_id: test_ids})
        test["asset_path"] = test_asset_paths.map(lambda path: str(path) if path is not None else np.nan)
        train = train[train["asset_path"].notna()].reset_index(drop=True)
        test = test[test["asset_path"].notna()].reset_index(drop=True)
        if train.empty or test.empty:
            continue

        SYNTHETIC_TABLE_DIR.mkdir(parents=True, exist_ok=True)
        train_out = SYNTHETIC_TABLE_DIR / "train_synth.csv"
        test_out = SYNTHETIC_TABLE_DIR / "test_synth.csv"
        train.to_csv(train_out, index=False)
        test.to_csv(test_out, index=False)
        print(f"synthesized train/test tables from asset files and labels: {candidate}")
        return train_out, test_out, sample_path
    return None


def resolve_asset_label_id_column(sample_id, labels: pd.DataFrame) -> str:
    sample_key = str(sample_id).lower()
    label_by_lower = {str(col).lower(): str(col) for col in labels.columns}
    if sample_key in label_by_lower:
        return label_by_lower[sample_key]
    if is_id_like_column(sample_id):
        alias = resolve_id_like_label_column(labels)
        if alias is not None:
            return alias
    return str(labels.columns[0])


def select_train_test_paths(files: list[Path], sample_path: Path | None) -> tuple[Path | None, Path | None]:
    candidates = [path for path in files if path != sample_path and ".kagglebot_cache" not in path.parts]
    train_candidates = [path for path in candidates if path_mentions_role(path, "train")]
    test_candidates = [path for path in candidates if path_mentions_role(path, "test")]
    if not train_candidates or not test_candidates:
        return train_candidates[0] if train_candidates else None, test_candidates[0] if test_candidates else None

    best = None
    for train_path in train_candidates:
        for test_path in test_candidates:
            score = train_test_pair_score(train_path, test_path, sample_path)
            tie_break = f"{train_path.as_posix()}\0{test_path.as_posix()}"
            candidate = (score, tie_break, train_path, test_path)
            if best is None or candidate > best:
                best = candidate
    return (best[2], best[3]) if best is not None else (train_candidates[0], test_candidates[0])


def path_mentions_role(path: Path, role: str) -> bool:
    stem = tabular_stem(path)
    if component_mentions_role(stem, role):
        return True
    if role == "test" and component_mentions_role(stem, "train"):
        return False
    aliases = role_aliases(role)
    return any(str(part).lower() in aliases for part in path.parts)


def component_mentions_role(value: str, role: str) -> bool:
    lowered = value.lower()
    tokens = sqlite_name_tokens(lowered)
    aliases = role_aliases(role)
    train_aliases = role_aliases("train")
    if role == "test":
        direct_test_aliases = set(TEST_DIRECT_ROLE_ALIASES)
        inference_aliases = aliases - direct_test_aliases
        if tokens & direct_test_aliases:
            return True
        if tokens & inference_aliases and not tokens & train_aliases:
            return True
    elif tokens & aliases:
        return True

    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    if compact in aliases:
        return True
    for alias in aliases:
        if compact.startswith(alias) and compact[len(alias) :] in ROLE_SUFFIXES:
            return True
    if role == "train" and compact.startswith("train"):
        return compact[len("train") :] in ROLE_SUFFIXES | {"ing"}
    if role == "test" and compact.startswith("test"):
        return compact[len("test") :] in ROLE_SUFFIXES | {"ing"}
    if compact.endswith(role):
        return compact[: -len(role)] in ROLE_TRAILING_PREFIXES
    return False


def role_aliases(role: str) -> set[str]:
    return set(ROLE_ALIASES.get(role, {role}))


def train_test_pair_score(train_path: Path, test_path: Path, sample_path: Path | None) -> int:
    try:
        train_head = read_table(train_path, nrows=5)
        test_head = read_table(test_path, nrows=5)
        sample_head = read_table(sample_path, nrows=5) if sample_path is not None else pd.DataFrame()
    except Exception:
        return -10000

    train_cols = set(train_head.columns)
    test_cols = set(test_head.columns)
    sample_cols = list(sample_head.columns)
    common_cols = train_cols & test_cols
    score = len(common_cols) * 20
    if sample_cols and sample_cols[0] in common_cols:
        score += 60
    train_stem = tabular_stem(train_path).lower()
    test_stem = tabular_stem(test_path).lower()
    if train_stem == "train":
        score += 40
    if test_stem == "test":
        score += 40
    if roleless_stem(train_path, "train") == roleless_stem(test_path, "test"):
        score += 35
    if "feature" in train_stem or "data" in train_stem:
        score += 15
    if "label" in train_stem or "target" in train_stem:
        score -= 50
    return score


def roleless_stem(path: Path, role: str) -> str:
    normalized = tabular_stem(path).lower().replace(role, "")
    return normalized.replace("_", "").replace("-", "").replace(".", "")


def tabular_stem(path: Path) -> str:
    suffix = tabular_suffix(path)
    name = path.name
    if suffix and name.lower().endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def name_tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", value.lower()) if token}


def align_train_test_column_case_to_sample(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_by_lower = {str(col).lower(): str(col) for col in sample.columns}
    train = rename_columns_case_insensitive(train, sample_by_lower)
    test = rename_columns_case_insensitive(test, sample_by_lower)
    test = align_test_column_case_to_train(train, test)
    return train, test


def align_test_column_case_to_train(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    train_by_lower = {str(col).lower(): str(col) for col in train.columns}
    return rename_columns_case_insensitive(test, train_by_lower)


def rename_columns_case_insensitive(frame: pd.DataFrame, desired_by_lower: dict[str, str]) -> pd.DataFrame:
    existing = {str(col) for col in frame.columns}
    rename = {}
    for col in frame.columns:
        source = str(col)
        desired = desired_by_lower.get(source.lower())
        if desired is None or desired == source:
            continue
        if desired in existing:
            continue
        rename[source] = desired
    return frame.rename(columns=rename) if rename else frame


def infer_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
) -> tuple[str | None, str, list[str], list[str]]:
    sample_cols = list(sample.columns)
    train_minus_test = [c for c in train.columns if c not in test.columns]
    target_cols = [c for c in sample_cols if c in train_minus_test and c in train.columns]
    if not target_cols:
        target_cols = [c for c in sample_cols if c in train.columns and c not in test.columns]
    if not target_cols:
        target_cols = [c for c in sample_cols[1:] if c in train.columns]
    if not target_cols and train_minus_test:
        target_cols = train_minus_test
    if not target_cols:
        raise ValueError("Unable to infer target columns from train/test/sample files.")

    non_targets = [c for c in sample_cols if c not in target_cols]
    id_col = next((c for c in non_targets if c in test.columns), None)
    if id_col is None and non_targets:
        candidate = non_targets[0]
        id_col = candidate if is_id_like_column(candidate) else None

    target_col = target_cols[0]
    feature_cols = [c for c in train.columns if c not in target_cols and c != id_col]
    return id_col, target_col, feature_cols, target_cols


def infer_unlabeled_numeric_score_columns(sample: pd.DataFrame, id_col: str | None) -> list[str]:
    prediction_cols = [str(col) for col in sample.columns if col != id_col]
    if len(prediction_cols) != 1:
        return []
    column = prediction_cols[0]
    if not looks_like_unlabeled_score_column(column):
        return []
    if column not in sample.columns or not pd.api.types.is_numeric_dtype(sample[column]):
        return []
    return [column]


def looks_like_unlabeled_score_column(name: str) -> bool:
    compact = re.sub(r"[^a-z0-9]+", "", name.lower())
    tokens = set(re.findall(r"[a-z0-9]+", name.lower()))
    if compact in {
        "anomaly",
        "anomalyscore",
        "outlier",
        "outlierscore",
        "fraudscore",
        "riskscore",
    }:
        return True
    return bool(tokens & {"anomaly", "outlier", "fraud", "risk"} and tokens & {"score", "prediction", "target"})


def infer_unlabeled_submission_layout(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
) -> tuple[str | None, str, list[str], list[str]] | None:
    sample_cols = list(sample.columns)
    id_col = next((col for col in sample_cols if col in test.columns), None)
    if id_col is None and sample_cols and is_id_like_column(sample_cols[0]):
        id_col = sample_cols[0]
    target_cols = infer_unlabeled_numeric_score_columns(sample, id_col=id_col)
    if not target_cols:
        return None
    feature_cols = [col for col in train.columns if col in test.columns and col != id_col]
    return id_col, target_cols[0], feature_cols, target_cols


def maybe_merge_train_labels(
    files: list[Path],
    train_path: Path,
    test_path: Path,
    sample_path: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
) -> pd.DataFrame | None:
    label_paths = [
        path
        for path in files
        if path not in {train_path, test_path, sample_path}
        and ".kagglebot_cache" not in path.parts
        and is_label_table_path(path)
    ]
    for label_path in sorted(label_paths, key=lambda path: path.as_posix()):
        try:
            labels = read_table(label_path)
        except Exception:
            continue
        if labels.empty:
            continue
        join_cols = resolve_label_join_columns(train, test, sample, labels)
        if join_cols is None:
            continue
        train_join_col, label_join_col = join_cols
        label_by_lower = {str(col).lower(): str(col) for col in labels.columns}
        target_cols = [
            (label_by_lower[str(col).lower()], str(col))
            for col in sample.columns
            if str(col).lower() != str(train_join_col).lower() and str(col).lower() in label_by_lower
        ]
        if not target_cols:
            fallback_targets = [str(col) for col in labels.columns if str(col).lower() != str(label_join_col).lower()]
            target_cols = [(fallback_targets[0], fallback_targets[0])] if len(fallback_targets) == 1 else []
        if not target_cols:
            continue
        label_subset = labels[[label_join_col, *[source for source, _ in target_cols]]].rename(
            columns={label_join_col: train_join_col, **dict(target_cols)}
        )
        merged = merge_label_subset(train, label_subset, train_join_col)
        if not merged.empty:
            print(f"merged labels from: {label_path}")
            return merged
    return None


def merge_label_subset(train: pd.DataFrame, label_subset: pd.DataFrame, join_col: str) -> pd.DataFrame:
    try:
        merged = train.merge(label_subset, on=join_col, how="inner")
    except ValueError:
        merged = pd.DataFrame()
    if not merged.empty:
        return merged

    key = "__kagglebot_label_join_key__"
    train_keyed = train.copy()
    label_keyed = label_subset.copy()
    train_keyed[key] = train_keyed[join_col].map(submission_id_alignment_key)
    label_keyed[key] = label_keyed[join_col].map(submission_id_alignment_key)
    label_keyed = label_keyed.drop(columns=[join_col])
    return train_keyed.merge(label_keyed, on=key, how="inner").drop(columns=[key])


def resolve_label_join_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[str, str] | None:
    preferred = [str(sample.columns[0])] if len(sample.columns) and is_id_like_column(sample.columns[0]) else []
    preferred.extend(["id", "ID", "row_id", "filename", "image_id", "file"])
    preferred.extend(str(col) for col in train.columns if col in test.columns)
    train_by_lower = {str(col).lower(): str(col) for col in train.columns}
    test_by_lower = {str(col).lower(): str(col) for col in test.columns}
    label_by_lower = {str(col).lower(): str(col) for col in labels.columns}
    seen = set()
    for column in preferred:
        lowered = str(column).lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        train_col = train_by_lower.get(lowered)
        test_col = test_by_lower.get(lowered)
        label_col = label_by_lower.get(lowered)
        if train_col is not None and test_col is not None and label_col is not None:
            return train_col, label_col
        if train_col is not None and test_col is not None and is_id_like_column(train_col):
            label_alias = resolve_id_like_label_column(labels)
            if label_alias is not None:
                return train_col, label_alias
    return None


def is_id_like_column(column) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
    compact = normalized.replace("_", "")
    return normalized in ID_LIKE_COLUMN_NAMES or compact in ID_LIKE_COLUMN_NAMES


def resolve_id_like_label_column(labels: pd.DataFrame) -> str | None:
    for column in labels.columns:
        if is_id_like_column(column):
            return str(column)
    return None


def infer_task(y: pd.Series) -> str:
    if y.dtype == "object":
        return "classification"
    nunique = y.nunique(dropna=True)
    if nunique <= 20:
        return "classification"
    if nunique / max(len(y), 1) <= 0.05:
        return "classification"
    return "regression"


def sample_column_default(sample: pd.DataFrame, column: str):
    if column not in sample.columns or sample.empty:
        return 0.0
    non_null = sample[column].dropna()
    if non_null.empty:
        return 0.0
    numeric = pd.to_numeric(non_null, errors="coerce").dropna()
    if not numeric.empty:
        return float(numeric.mean())
    return non_null.iloc[0]


def build_submission_template(sample: pd.DataFrame, test: pd.DataFrame, id_col: str | None) -> pd.DataFrame:
    if id_col is None and is_tiny_public_sample_without_id(sample, test):
        expanded = pd.DataFrame(index=range(len(test)))
        for col in sample.columns:
            expanded[col] = sample_column_default(sample, col)
        return expanded[list(sample.columns)]
    if (
        id_col
        and id_col in sample.columns
        and id_col in test.columns
        and 0 < len(sample) <= 10
        and len(test) > len(sample)
        and not sample[id_col].duplicated().any()
        and not test[id_col].duplicated().any()
    ):
        expanded = pd.DataFrame({id_col: test[id_col].to_numpy()})
        for col in sample.columns:
            if col != id_col:
                expanded[col] = sample_column_default(sample, col)
        return expanded[list(sample.columns)]
    return sample.copy()


def is_tiny_public_sample_without_id(sample: pd.DataFrame, test: pd.DataFrame) -> bool:
    if len(sample) <= 0 or len(sample) > 10:
        return False
    return len(test) > len(sample)


def build_preprocessor(feature_cols: list[str], train: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in feature_cols if train[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", ohe),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


def add_temporal_calendar_features(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    added: list[str] = []
    for col in feature_cols:
        if col not in train.columns or col not in test.columns:
            continue
        train_dates = parse_calendar_feature(train[col])
        test_dates = parse_calendar_feature(test[col])
        if train_dates is None or test_dates is None:
            continue
        prefix = f"__time_{safe_feature_name(col)}"
        train_derived = calendar_feature_values(train_dates, prefix=prefix)
        test_derived = calendar_feature_values(test_dates, prefix=prefix)
        for new_col in train_derived.columns:
            train[new_col] = train_derived[new_col].to_numpy()
            test[new_col] = test_derived[new_col].to_numpy()
            if new_col not in added:
                added.append(str(new_col))
    return added


def parse_calendar_feature(series: pd.Series) -> pd.Series | None:
    if not (
        pd.api.types.is_datetime64_any_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        return None
    if series.dropna().empty:
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if float(pd.Series(parsed).notna().mean()) < 0.8:
        return None
    return pd.Series(parsed)


def calendar_feature_values(parsed: pd.Series, prefix: str) -> pd.DataFrame:
    dt = pd.Series(parsed).dt
    return pd.DataFrame(
        {
            f"{prefix}_year": dt.year.astype("float64"),
            f"{prefix}_month": dt.month.astype("float64"),
            f"{prefix}_day": dt.day.astype("float64"),
            f"{prefix}_dayofweek": dt.dayofweek.astype("float64"),
            f"{prefix}_dayofyear": dt.dayofyear.astype("float64"),
            f"{prefix}_is_month_start": dt.is_month_start.astype("float64"),
            f"{prefix}_is_month_end": dt.is_month_end.astype("float64"),
        }
    )


def safe_feature_name(value) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return normalized or "feature"


def build_sklearn_model(task: str, preprocessor: ColumnTransformer) -> Pipeline:
    estimator = LogisticRegression(max_iter=2000) if task == "classification" else Ridge()
    return Pipeline([("pre", preprocessor), ("model", estimator)])


def sample_prediction_columns(sample: pd.DataFrame, id_col: str | None) -> list[str]:
    return [str(col) for col in sample.columns if col != id_col]


def normalize_submission_column_token(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def rle_segmentation_columns(sample: pd.DataFrame) -> list[str]:
    columns = []
    for col in sample.columns:
        normalized = normalize_submission_column_token(str(col))
        if normalized in RLE_SEGMENTATION_COLUMN_TOKENS:
            columns.append(str(col))
            continue
        if "encodedpixels" in normalized or "runlength" in normalized or normalized.endswith("rle"):
            columns.append(str(col))
    return columns


def infer_prediction_kind(sample: pd.DataFrame, target_col: str) -> str:
    if target_col not in sample.columns:
        return "class"
    if looks_like_text_prediction(sample[target_col], target_col):
        return "text"
    if pd.api.types.is_float_dtype(sample[target_col]):
        return "probability"
    return "class"


def looks_like_learning_to_rank_target(
    target_col: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> bool:
    compact_target = re.sub(r"[^a-z0-9]+", "", str(target_col).lower())
    if compact_target not in {"relevance", "relevancescore", "rank", "ranking", "score", "rankscore"}:
        return False
    if target_col not in train.columns or not pd.api.types.is_numeric_dtype(train[target_col]):
        return False
    feature_compacts = {re.sub(r"[^a-z0-9]+", "", str(col).lower()) for col in feature_cols}
    has_query = bool(feature_compacts & {"queryid", "qid", "searchid", "requestid", "sessionid"})
    has_item = bool(feature_compacts & {"documentid", "docid", "candidateid", "itemid", "productid", "passageid"})
    if has_query and has_item:
        return True
    train_cols = {re.sub(r"[^a-z0-9]+", "", str(col).lower()) for col in train.columns}
    test_cols = {re.sub(r"[^a-z0-9]+", "", str(col).lower()) for col in test.columns}
    return bool(
        train_cols
        & test_cols
        & {"queryid", "qid", "searchid", "requestid"}
        and train_cols
        & test_cols
        & {"documentid", "docid", "candidateid", "itemid", "passageid"}
    )


def looks_like_ordinal_target(target: pd.Series, column_name: str, feature_cols: list[str]) -> bool:
    if looks_like_user_item_interaction(feature_cols):
        return False
    compact = re.sub(r"[^a-z0-9]+", "", str(column_name).lower())
    tokens = set(re.findall(r"[a-z0-9]+", str(column_name).lower()))
    ordinal_name = bool(
        tokens
        & {
            "severity",
            "grade",
            "stage",
            "level",
            "rating",
            "risk",
            "quality",
            "ordinal",
            "class",
            "label",
        }
    ) or compact in {"risklevel", "severitygrade", "qualitygrade", "ordinaltarget"}
    if not ordinal_name:
        return False
    values = target.dropna()
    if values.empty:
        return False
    unique = int(values.nunique(dropna=True))
    if unique < 3 or unique > 20:
        return False
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty or len(numeric) != len(values):
            return False
        return bool(((numeric % 1).abs() < 1e-9).all()) and int(numeric.nunique(dropna=True)) == unique
    return False


def looks_like_user_item_interaction(feature_cols: list[str]) -> bool:
    compact_cols = {re.sub(r"[^a-z0-9]+", "", str(col).lower()) for col in feature_cols}
    has_user = bool(compact_cols & {"userid", "user", "customerid", "accountid"})
    has_item = bool(compact_cols & {"itemid", "item", "adid", "productid", "movieid"})
    return has_user and has_item


def looks_like_named_continuous_numeric_target(target: pd.Series, column_name: str, feature_cols: list[str]) -> bool:
    if looks_like_user_item_interaction(feature_cols):
        return False
    if not pd.api.types.is_numeric_dtype(target):
        return False
    numeric = pd.to_numeric(target, errors="coerce").dropna()
    if numeric.empty or int(numeric.nunique(dropna=True)) <= 1:
        return False
    compact = re.sub(r"[^a-z0-9]+", "", str(column_name).lower())
    tokens = set(re.findall(r"[a-z0-9]+", str(column_name).lower()))
    continuous_names = {
        "amount",
        "cost",
        "count",
        "demand",
        "fare",
        "income",
        "loss",
        "price",
        "profit",
        "quantity",
        "revenue",
        "sale",
        "sales",
        "spend",
        "value",
        "yield",
    }
    continuous_compacts = {
        "saleprice",
        "salesprice",
        "transactionamount",
        "purchaseamount",
        "itemcount",
        "unitcount",
        "targetvalue",
    }
    return bool(tokens & continuous_names or compact in continuous_compacts)


def ordinal_predictions_from_continuous(predictions: np.ndarray, labels: pd.Series) -> np.ndarray:
    numeric_labels = pd.to_numeric(labels, errors="coerce").dropna()
    if numeric_labels.empty:
        return np.rint(np.asarray(predictions, dtype=float)).astype(int)
    lower = float(numeric_labels.min())
    upper = float(numeric_labels.max())
    rounded = np.rint(np.asarray(predictions, dtype=float))
    return np.clip(rounded, lower, upper).astype(int)


def should_log1p_regression_target(labels: pd.Series, column_name: str) -> bool:
    return looks_like_count_regression_target(labels, column_name) or looks_like_positive_skew_regression_target(
        labels, column_name
    )


def log1p_target_values(values) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return np.log1p(np.clip(numeric, 0.0, None))


def expm1_predictions(predictions) -> np.ndarray:
    return np.expm1(np.asarray(predictions, dtype=float))


def root_mean_squared_log_error(y_true, y_pred) -> float:
    y_true_clip = np.clip(np.asarray(y_true, dtype=float), 0.0, None)
    y_pred_clip = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    return float(root_mean_squared_error(np.log1p(y_true_clip), np.log1p(y_pred_clip)))


def finalize_regression_predictions(
    predictions,
    labels: pd.Series,
    column_name: str,
    *,
    use_log1p_target: bool,
) -> np.ndarray:
    output = expm1_predictions(predictions) if use_log1p_target else np.asarray(predictions, dtype=float)
    return clip_structured_regression_predictions(output, labels, column_name)


def clip_structured_regression_predictions(predictions, labels: pd.Series, column_name: str) -> np.ndarray:
    if looks_like_count_regression_target(labels, column_name):
        return np.clip(np.asarray(predictions, dtype=float), 0.0, None)
    if looks_like_positive_skew_regression_target(labels, column_name):
        return np.clip(np.asarray(predictions, dtype=float), 0.0, None)
    bounds = bounded_regression_bounds(labels, column_name)
    if bounds is None:
        return np.asarray(predictions)
    lower, upper = bounds
    return np.clip(np.asarray(predictions, dtype=float), lower, upper)


def looks_like_count_regression_target(target: pd.Series, column_name: str) -> bool:
    if not pd.api.types.is_numeric_dtype(target):
        return False
    tokens = set(re.findall(r"[a-z0-9]+", str(column_name).lower()))
    compact = re.sub(r"[^a-z0-9]+", "", str(column_name).lower())
    if not (
        tokens
        & {
            "count",
            "counts",
            "demand",
            "quantity",
            "qty",
            "unit",
            "units",
            "trip",
            "trips",
            "ride",
            "rides",
            "rental",
            "rentals",
            "order",
            "orders",
            "booking",
            "bookings",
            "visitor",
            "visitors",
            "passenger",
            "passengers",
        }
        or compact in {"itemcount", "unitcount", "numorders", "numberoforders", "tripcount", "ridecount"}
        or compact.startswith("num")
    ):
        return False
    values = pd.to_numeric(target.dropna(), errors="coerce").dropna()
    if values.empty or bool((values < 0).any()):
        return False
    integer_like = ((values % 1).abs() < 1e-9).mean()
    return bool(float(integer_like) >= 0.95 and int(values.nunique(dropna=True)) >= 3)


def bounded_regression_bounds(labels: pd.Series, column_name: str) -> tuple[float, float] | None:
    if not looks_like_bounded_regression_target(labels, column_name):
        return None
    values = pd.to_numeric(labels.dropna(), errors="coerce").dropna()
    if values.empty:
        return None
    if float(values.max()) <= 1.0:
        return 0.0, 1.0
    return 0.0, 100.0


def looks_like_bounded_regression_target(target: pd.Series, column_name: str) -> bool:
    if not pd.api.types.is_numeric_dtype(target):
        return False
    tokens = set(re.findall(r"[a-z0-9]+", str(column_name).lower()))
    compact = re.sub(r"[^a-z0-9]+", "", str(column_name).lower())
    bounded_names = {
        "rate",
        "ratio",
        "percent",
        "percentage",
        "pct",
        "share",
        "fraction",
        "proportion",
        "probability",
        "prob",
    }
    bounded_compacts = {
        "conversionrate",
        "clickthroughrate",
        "defaultprobability",
        "winprobability",
        "targetrate",
        "targetratio",
    }
    if not (tokens & bounded_names or compact in bounded_compacts):
        return False
    values = pd.to_numeric(target.dropna(), errors="coerce").dropna()
    if values.empty or int(values.nunique(dropna=True)) < 3:
        return False
    if float(values.min()) < 0.0:
        return False
    max_value = float(values.max())
    if max_value <= 1.0:
        return True
    percent_names = {"percent", "percentage", "pct"}
    return bool((tokens & percent_names or "percent" in compact or "pct" in compact) and max_value <= 100.0)


def looks_like_positive_skew_regression_target(target: pd.Series, column_name: str) -> bool:
    if not pd.api.types.is_numeric_dtype(target):
        return False
    tokens = set(re.findall(r"[a-z0-9]+", str(column_name).lower()))
    compact = re.sub(r"[^a-z0-9]+", "", str(column_name).lower())
    skew_names = {
        "amount",
        "cost",
        "fare",
        "income",
        "price",
        "profit",
        "revenue",
        "sale",
        "sales",
        "spend",
        "value",
    }
    skew_compacts = {
        "saleprice",
        "salesprice",
        "transactionamount",
        "purchaseamount",
        "targetvalue",
    }
    if not (tokens & skew_names or compact in skew_compacts):
        return False
    values = pd.to_numeric(target.dropna(), errors="coerce").dropna()
    if len(values) < 8 or bool((values < 0).any()) or int(values.nunique(dropna=True)) < 5:
        return False
    median = float(values.median())
    if median <= 0.0:
        return False
    skew = float(values.skew())
    if pd.isna(skew):
        return False
    return bool(skew >= 1.0 and float(values.max()) / median >= 5.0)


def looks_like_text_prediction(sample_target: pd.Series, column_name: str) -> bool:
    lowered_name = str(column_name or sample_target.name or "").strip().lower()
    if any(token in lowered_name for token in TEXT_PREDICTION_NAME_TOKENS):
        return True
    values = sample_target.dropna().astype(str).str.strip()
    if values.empty:
        return False
    if float((values == "").mean()) >= 0.8:
        return True
    non_empty = values[values != ""]
    if non_empty.empty:
        return True
    return float(non_empty.str.len().mean()) >= 20.0


def looks_like_natural_language_text_target(target: pd.Series) -> bool:
    if not (
        pd.api.types.is_object_dtype(target)
        or pd.api.types.is_string_dtype(target)
        or isinstance(target.dtype, pd.CategoricalDtype)
    ):
        return False
    values = target.dropna().astype(str).str.strip().head(1000)
    values = values[values != ""]
    if len(values) < 3:
        return False
    unique_ratio = float(values.nunique(dropna=True) / max(len(values), 1))
    if unique_ratio < 0.25 and values.nunique(dropna=True) < 10:
        return False
    lengths = values.str.len()
    word_counts = values.str.count(r"\s+") + 1
    long_ratio = float((lengths >= 24).mean())
    multi_word_ratio = float((word_counts >= 4).mean())
    return bool(
        float(lengths.mean()) >= 24.0
        or float(word_counts.mean()) >= 4.0
        or long_ratio >= 0.4
        or multi_word_ratio >= 0.4
    )


def looks_like_multi_label_target(target: pd.Series, column_name: str) -> bool:
    if not (pd.api.types.is_object_dtype(target) or pd.api.types.is_string_dtype(target)):
        return False
    column_tokens = re.findall(r"[a-z0-9]+", str(column_name).lower())
    tokens = set(column_tokens)
    compact = "".join(column_tokens)
    strong_name = bool(tokens & {"labels", "tags", "classes", "categories"}) or "multilabel" in compact
    generic_name = strong_name or bool(tokens & {"label", "target", "class", "category"})
    if not generic_name:
        return False
    values = target.dropna().astype(str).str.strip().head(500)
    values = values[values != ""]
    if values.empty:
        return False

    multi_count = 0
    atomic_labels: set[str] = set()
    for value in values:
        labels = split_multi_label_value(value, allow_whitespace=strong_name)
        if len(labels) < 2:
            continue
        multi_count += 1
        atomic_labels.update(labels)
    if float(multi_count / len(values)) < 0.6:
        return False
    return len(atomic_labels) >= 2


def looks_like_probability_score_column(name: str) -> bool:
    compact = re.sub(r"[^a-z0-9]+", "", str(name).lower())
    tokens = set(re.findall(r"[a-z0-9]+", str(name).lower()))
    if compact in {
        "probability",
        "prob",
        "proba",
        "predictionprobability",
        "targetprobability",
        "risk",
        "riskscore",
        "score",
        "prediction",
        "isfraud",
        "fraudprobability",
        "fraudscore",
        "isdefault",
        "defaultprobability",
        "defaultscore",
    }:
        return True
    if tokens & {"probability", "prob", "proba"}:
        return True
    return bool(tokens & {"fraud", "default", "risk"} and tokens & {"score", "prediction", "target", "probability"})


def split_multi_label_value(value: str, *, allow_whitespace: bool) -> list[str]:
    raw = value.strip()
    if not raw:
        return []
    if any(sep in raw for sep in ("|", ";", ",")):
        parts = re.split(r"[|;,]+", raw)
    elif allow_whitespace:
        parts = re.split(r"\s+", raw)
    else:
        return []
    labels = [part.strip() for part in parts if part.strip()]
    if len(labels) < 2:
        return []
    if any(len(label) > 48 for label in labels):
        return []
    if any(not re.fullmatch(r"[A-Za-z0-9_.:+-]+", label) for label in labels):
        return []
    return labels


def text_baseline_value(y: pd.Series, sample: pd.DataFrame, target_col: str) -> str:
    values = y.dropna().astype(str).str.strip()
    values = values[values != ""]
    if not values.empty:
        return str(values.mode().iloc[0])
    if target_col in sample.columns and not sample.empty:
        sample_values = sample[target_col].dropna().astype(str).str.strip()
        sample_values = sample_values[sample_values != ""]
        if not sample_values.empty:
            return str(sample_values.iloc[0])
    return ""


def text_feature_columns(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    usable = [col for col in feature_cols if col in train.columns and col in test.columns]
    text_cols = [
        col
        for col in usable
        if (
            pd.api.types.is_object_dtype(train[col])
            or pd.api.types.is_string_dtype(train[col])
            or isinstance(train[col].dtype, pd.CategoricalDtype)
        )
    ]
    return text_cols or usable


def combine_text_features(frame: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    if not feature_cols:
        return pd.Series([""] * len(frame), index=frame.index)
    parts = []
    for col in feature_cols:
        values = frame[col].fillna("").astype(str).str.strip()
        parts.append(str(col) + "=" + values)
    return pd.concat(parts, axis=1).agg(" ".join, axis=1).str.strip()


def predict_tfidf_nearest_text(
    train_text: pd.Series,
    train_values: pd.Series,
    query_text: pd.Series,
    fallback_value: str,
    *,
    exclude_self: bool = False,
) -> np.ndarray:
    values = train_values.dropna().astype(str).str.strip()
    valid_mask = values != ""
    values = values[valid_mask]
    index_text = train_text.loc[values.index].fillna("").astype(str).str.strip()
    usable_mask = index_text != ""
    values = values[usable_mask]
    index_text = index_text[usable_mask]
    if values.empty or index_text.empty:
        return np.repeat(fallback_value, len(query_text))

    max_index_rows = 50_000
    if len(index_text) > max_index_rows:
        take = np.linspace(0, len(index_text) - 1, max_index_rows, dtype=int)
        index_text = index_text.iloc[take]
        values = values.iloc[take]

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=100_000,
            token_pattern=r"(?u)\b\w+\b",
        )
        train_matrix = vectorizer.fit_transform(index_text)
        query_matrix = vectorizer.transform(query_text.fillna("").astype(str).str.strip())
    except ValueError:
        return np.repeat(fallback_value, len(query_text))

    preds: list[str] = []
    batch_size = 256
    value_array = values.to_numpy(dtype=str)
    for start in range(0, query_matrix.shape[0], batch_size):
        stop = min(start + batch_size, query_matrix.shape[0])
        sims = (query_matrix[start:stop] @ train_matrix.T).toarray()
        if exclude_self:
            for row_idx, source_idx in enumerate(range(start, stop)):
                if source_idx < sims.shape[1]:
                    sims[row_idx, source_idx] = -np.inf
        best_idx = sims.argmax(axis=1) if sims.size else np.array([], dtype=int)
        best_score = sims.max(axis=1) if sims.size else np.array([], dtype=float)
        for idx, score in zip(best_idx, best_score, strict=False):
            preds.append(fallback_value if not np.isfinite(score) or score <= 0 else str(value_array[int(idx)]))
    return np.asarray(preds, dtype=object)


def train_predict_text_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
):
    y = train[target_col]
    fallback_value = text_baseline_value(y, sample, target_col)
    selected_cols = text_feature_columns(train, test, feature_cols)
    train_text = combine_text_features(train, selected_cols)
    test_text = combine_text_features(test, selected_cols)
    preds = predict_tfidf_nearest_text(train_text, y, test_text, fallback_value)

    y_text = y.fillna("").astype(str).str.strip()
    valid_positions = np.flatnonzero((y_text != "").to_numpy())
    if len(train_text) > 1 and len(valid_positions) > 0:
        fitted = predict_tfidf_nearest_text(
            train_text,
            y,
            train_text,
            fallback_value,
            exclude_self=True,
        )
        score = float((y_text.to_numpy(dtype=str)[valid_positions] == fitted[valid_positions]).mean())
    else:
        score = 0.0 if len(valid_positions) == 0 else float((y_text.iloc[valid_positions] == fallback_value).mean())

    return preds, {
        "task": "text",
        "metric": "leave_one_out_exact_match",
        "score": float(score),
        "prediction_kind": "text",
        "model_kind": "tfidf_nearest_neighbor" if selected_cols else "constant_text",
        "text_feature_columns": selected_cols,
    }


def train_predict_unsupervised_score(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], target_col: str):
    selected_cols = [col for col in feature_cols if col in train.columns and col in test.columns]
    preds = unsupervised_anomaly_scores(train, test, selected_cols)
    numeric_cols = [col for col in selected_cols if pd.api.types.is_numeric_dtype(train[col])]
    return preds, {
        "task": "unsupervised",
        "metric": "unsupervised_anomaly_score",
        "score": 0.0,
        "prediction_kind": "continuous",
        "model_kind": "robust_unsupervised_anomaly_score",
        "target_column": target_col,
        "feature_columns": selected_cols,
        "numeric_features": len(numeric_cols),
        "categorical_features": len(selected_cols) - len(numeric_cols),
    }


def unsupervised_anomaly_scores(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    if not feature_cols or test.empty:
        return np.zeros(len(test), dtype=float)
    components: list[np.ndarray] = []
    for col in feature_cols:
        if col not in train.columns or col not in test.columns:
            continue
        if pd.api.types.is_numeric_dtype(train[col]):
            train_numeric = pd.to_numeric(train[col], errors="coerce")
            test_numeric = pd.to_numeric(test[col], errors="coerce")
            observed = train_numeric.dropna()
            if observed.empty:
                continue
            median = float(observed.median())
            q1 = float(observed.quantile(0.25))
            q3 = float(observed.quantile(0.75))
            scale = max(q3 - q1, float(observed.std(ddof=0)), 1.0)
            values = (test_numeric.fillna(median).to_numpy(dtype=float) - median) / scale
            components.append(np.minimum(np.abs(values), 10.0) / 10.0)
        else:
            train_values = train[col].fillna("__missing__").astype(str)
            frequencies = train_values.value_counts(normalize=True)
            test_values = test[col].fillna("__missing__").astype(str)
            rarity = 1.0 - test_values.map(frequencies).fillna(0.0).to_numpy(dtype=float)
            components.append(np.clip(rarity, 0.0, 1.0))
    if not components:
        return np.zeros(len(test), dtype=float)
    raw = np.nanmean(np.vstack(components), axis=0)
    return np.clip(np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def is_class_probability_submission(
    train: pd.DataFrame,
    sample: pd.DataFrame,
    id_col: str | None,
    target_col: str,
    target_cols: list[str],
) -> bool:
    if len(target_cols) != 1 or target_col not in train.columns or target_col in sample.columns:
        return False
    prediction_cols = sample_prediction_columns(sample, id_col)
    if len(prediction_cols) < 2:
        return False
    if infer_task(train[target_col]) != "classification":
        return False
    return all(pd.api.types.is_numeric_dtype(sample[col]) for col in prediction_cols if col in sample.columns)


def is_multi_label_column_submission(
    train: pd.DataFrame,
    sample: pd.DataFrame,
    id_col: str | None,
    target_col: str,
    target_cols: list[str],
) -> bool:
    if not is_class_probability_submission(train, sample, id_col, target_col, target_cols):
        return False
    return looks_like_multi_label_target(train[target_col], target_col)


def is_numeric_multi_column_regression_submission(
    train: pd.DataFrame,
    sample: pd.DataFrame,
    id_col: str | None,
    target_col: str,
    target_cols: list[str],
) -> bool:
    if len(target_cols) != 1 or target_col not in train.columns or target_col in sample.columns:
        return False
    prediction_cols = sample_prediction_columns(sample, id_col)
    if len(prediction_cols) < 2:
        return False
    if infer_task(train[target_col]) != "regression":
        return False
    return all(pd.api.types.is_numeric_dtype(sample[col]) for col in prediction_cols if col in sample.columns)


def is_survival_single_score_submission(sample: pd.DataFrame, id_col: str | None, target_cols: list[str]) -> bool:
    event_col, time_col = survival_event_time_columns(target_cols)
    if event_col is None or time_col is None:
        return False
    prediction_cols = sample_prediction_columns(sample, id_col)
    if len(prediction_cols) != 1:
        return False
    score_col = prediction_cols[0]
    return (
        score_col not in target_cols
        and score_col in sample.columns
        and pd.api.types.is_numeric_dtype(sample[score_col])
    )


def survival_score_submission_column(sample: pd.DataFrame, id_col: str | None) -> str:
    return sample_prediction_columns(sample, id_col)[0]


def survival_event_time_columns(target_cols: list[str]) -> tuple[str | None, str | None]:
    event_col = next((col for col in target_cols if is_survival_event_column(col)), None)
    time_col = next((col for col in target_cols if is_survival_time_column(col)), None)
    return event_col, time_col


def is_survival_event_column(name: str) -> bool:
    compact = "".join(ch for ch in str(name).lower() if ch.isalnum())
    return compact in {"event", "eventobserved", "observed", "status", "efs", "censor", "censored", "death", "dead"}


def is_survival_time_column(name: str) -> bool:
    compact = "".join(ch for ch in str(name).lower() if ch.isalnum())
    return compact in {
        "time",
        "duration",
        "survivaltime",
        "timeevent",
        "timetoevent",
        "eventtime",
        "efstime",
        "os",
        "ostime",
        "dfs",
        "dfstime",
    }


def survival_risk_scores(event_predictions: np.ndarray, time_predictions: np.ndarray) -> np.ndarray:
    event_score = normalize_score_component(event_predictions)
    time_score = 1.0 - normalize_score_component(time_predictions)
    return np.clip(0.7 * event_score + 0.3 * time_score, 0.0, 1.0)


def normalize_score_component(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    fallback = 0.0 if np.isnan(values).all() else float(np.nanmedian(values))
    values = np.nan_to_num(values, nan=fallback)
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.full(values.shape, 0.5, dtype=float)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def infer_numeric_multi_column_prediction_kind(sample: pd.DataFrame, id_col: str | None) -> str:
    prediction_cols = sample_prediction_columns(sample, id_col)
    if looks_like_prediction_interval_columns(prediction_cols):
        return "prediction_interval_columns"
    if looks_like_quantile_prediction_columns(prediction_cols):
        return "quantile_columns"
    return "continuous_columns"


def infer_time_column(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> str | None:
    scored = []
    for idx, col in enumerate(feature_cols):
        if col not in train.columns or col not in test.columns:
            continue
        name_score = time_column_name_score(str(col))
        if name_score <= 0:
            continue
        holdout_score = future_temporal_holdout_score(train[col], test[col])
        if holdout_score <= 0:
            continue
        scored.append((name_score + holdout_score, -idx, col))
    if not scored:
        return None
    return max(scored)[2]


def time_column_name_score(name: str) -> float:
    tokens = set(re.findall(r"[A-Za-z0-9]+", name.lower()))
    compact = "".join(tokens)
    if compact in {"dateblocknum", "daynum", "weekofyear"}:
        return 1.0
    if tokens & {"date", "datetime", "timestamp"}:
        return 0.95
    if tokens & {"day", "daynum", "week", "month", "year"}:
        return 0.8
    if "time" in tokens and compact not in {"survivaltime", "eventtime", "timetoevent"}:
        return 0.7
    return 0.0


def future_temporal_holdout_score(train_series: pd.Series, test_series: pd.Series) -> float:
    if has_future_ordinal_holdout(train_series, test_series):
        return 1.0
    train_values = temporal_order_values(train_series.dropna().head(500))
    test_values = temporal_order_values(test_series.dropna().head(500))
    if train_values is None or test_values is None:
        return 0.0
    return 1.0 if float(np.nanmin(test_values)) > float(np.nanmax(train_values)) else 0.0


def has_future_ordinal_holdout(train_series: pd.Series, test_series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(train_series):
        return False
    train_values = pd.to_numeric(train_series, errors="coerce").dropna()
    test_values = pd.to_numeric(test_series, errors="coerce").dropna()
    if train_values.empty or test_values.empty:
        return False
    if int(train_values.nunique(dropna=True)) < 3:
        return False
    return float(test_values.min()) > float(train_values.max())


def looks_like_prediction_interval_columns(prediction_cols: list[str]) -> bool:
    roles = {prediction_interval_column_role(col) for col in prediction_cols}
    return "lower" in roles and "upper" in roles


def looks_like_quantile_prediction_columns(prediction_cols: list[str]) -> bool:
    return sum(1 for col in prediction_cols if quantile_from_prediction_column(col) is not None) >= 2


def safe_train_valid_split(x: pd.DataFrame, y, task: str, time_values: pd.Series | None = None):
    y_arr = np.asarray(y)
    if len(y_arr) < 2:
        return x.copy(), x.copy(), y_arr.copy(), y_arr.copy()
    if time_values is not None:
        split = chronological_train_valid_indices(time_values, len(y_arr))
        if split is not None:
            train_idx, valid_idx = split
            return x.iloc[train_idx], x.iloc[valid_idx], y_arr[train_idx], y_arr[valid_idx]
    if task != "classification":
        return train_test_split(x, y_arr, test_size=0.2, random_state=42)

    classes, counts = np.unique(y_arr, return_counts=True)
    if len(classes) < 2:
        return x.copy(), x.copy(), y_arr.copy(), y_arr.copy()
    if counts.min() >= 2 and max(1, int(round(len(y_arr) * 0.2))) >= len(classes):
        try:
            return train_test_split(x, y_arr, test_size=0.2, random_state=42, stratify=y_arr)
        except ValueError:
            pass

    rng = np.random.default_rng(42)
    all_indices = np.arange(len(y_arr))
    protected_train = []
    for cls in classes:
        cls_indices = all_indices[y_arr == cls]
        protected_train.append(int(cls_indices[0]))
    protected = set(protected_train)
    valid_pool = np.array([idx for idx in all_indices if idx not in protected], dtype=int)
    valid_size = max(1, int(round(len(y_arr) * 0.2)))
    valid_size = min(valid_size, len(valid_pool))
    valid_idx = rng.choice(valid_pool, size=valid_size, replace=False) if valid_size > 0 else np.array([], dtype=int)
    valid_set = set(int(idx) for idx in np.asarray(valid_idx).tolist())
    train_idx = np.array([idx for idx in all_indices if idx not in valid_set], dtype=int)
    if valid_idx.size == 0:
        valid_idx = train_idx.copy()
    return x.iloc[train_idx], x.iloc[valid_idx], y_arr[train_idx], y_arr[valid_idx]


def chronological_train_valid_indices(time_values: pd.Series, row_count: int):
    order_values = temporal_order_values(time_values)
    if order_values is None or len(order_values) != row_count:
        return None
    ordered = np.arange(row_count)[np.argsort(order_values, kind="mergesort")]
    valid_size = max(1, int(round(row_count * 0.2)))
    valid_size = min(valid_size, row_count - 1)
    if valid_size <= 0:
        return None
    return ordered[:-valid_size], ordered[-valid_size:]


def temporal_order_values(series: pd.Series):
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    else:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        values = (
            pd.Series(parsed)
            .map(lambda value: value.timestamp() if pd.notna(value) else np.nan)
            .to_numpy(dtype=float)
        )
    if values.size == 0 or float(np.isfinite(values).mean()) < 0.8:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0 or float(np.nanmax(finite)) <= float(np.nanmin(finite)):
        return None
    fill_value = float(np.nanmedian(finite))
    return np.where(np.isfinite(values), values, fill_value)


def predict_sklearn(model, x, task: str, prediction_kind: str):
    if task == "classification" and prediction_kind == "probability" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.shape[1] == 1:
            return proba[:, 0]
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba.max(axis=1)
    return model.predict(x)


def train_torch_mlp(x, y, task: str, num_classes: int, device: str):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(np.asarray(y))
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    input_dim = x.shape[1]
    output_dim = num_classes if task == "classification" and num_classes > 2 else 1

    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, output_dim),
    ).to(device)

    if task == "classification":
        loss_fn = nn.CrossEntropyLoss() if num_classes > 2 else nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(10):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            if task == "classification" and num_classes == 2:
                yb = yb.float().view(-1, 1)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
    return model


def predict_torch(model, x, task: str, num_classes: int, prediction_kind: str, device: str):
    import torch

    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        outputs = model(x_tensor).cpu().numpy()
    if task == "classification":
        if num_classes > 2:
            probs = np.exp(outputs) / np.exp(outputs).sum(axis=1, keepdims=True)
            if prediction_kind == "probability":
                return probs.max(axis=1)
            return probs.argmax(axis=1)
        probs = 1 / (1 + np.exp(-outputs.ravel()))
        if prediction_kind == "probability":
            return probs
        return (probs >= 0.5).astype(int)
    return outputs.ravel()


def train_tpu_model(x, y, task: str, num_classes: int):
    import tensorflow as tf

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)

    with strategy.scope():
        output_units = num_classes if task == "classification" and num_classes > 2 else 1
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(x.shape[1],)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(output_units),
            ]
        )

        if task == "classification":
            if num_classes > 2:
                model.add(tf.keras.layers.Activation("softmax"))
                model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            else:
                model.add(tf.keras.layers.Activation("sigmoid"))
                model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        else:
            model.compile(optimizer="adam", loss="mse")

    model.fit(x, y, batch_size=256, epochs=10, verbose=0)
    return model


def predict_tpu(model, x, task: str, num_classes: int, prediction_kind: str):
    outputs = model.predict(x, batch_size=256, verbose=0)
    if task == "classification":
        if num_classes > 2:
            probs = outputs
            if prediction_kind == "probability":
                return probs.max(axis=1)
            return probs.argmax(axis=1)
        probs = outputs.ravel()
        if prediction_kind == "probability":
            return probs
        return (probs >= 0.5).astype(int)
    return outputs.ravel()


def train_predict_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    time_col: str | None = None,
):
    if target_col not in train.columns:
        return train_predict_unsupervised_score(train, test, feature_cols, target_col)
    x = train[feature_cols]
    y = train[target_col]
    task = infer_task(y)
    prediction_kind = infer_prediction_kind(sample, target_col)
    if looks_like_learning_to_rank_target(target_col, train, test, feature_cols):
        task = "regression"
        prediction_kind = "continuous"
    if looks_like_ordinal_target(y, target_col, feature_cols):
        task = "regression"
        prediction_kind = "ordinal"
    elif looks_like_named_continuous_numeric_target(y, target_col, feature_cols):
        task = "regression"
        prediction_kind = "continuous"
    if task == "classification" and prediction_kind == "class" and looks_like_probability_score_column(target_col):
        prediction_kind = "probability"
    if (
        prediction_kind == "class"
        and target_col in sample.columns
        and looks_like_natural_language_text_target(y)
        and not looks_like_multi_label_target(y, target_col)
    ):
        prediction_kind = "text"
    if prediction_kind == "class" and looks_like_multi_label_target(y, target_col):
        prediction_kind = "text"

    if prediction_kind == "text":
        return train_predict_text_target(train, test, sample, feature_cols, target_col)

    use_log1p_target = (
        task == "regression"
        and prediction_kind != "ordinal"
        and should_log1p_regression_target(y, target_col)
    )

    label_encoder = None
    num_classes = 0
    y_encoded = y
    if task == "classification":
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        num_classes = len(label_encoder.classes_)

    constant_baseline = task == "classification" and num_classes < 2
    time_values = train[time_col] if time_col and time_col in train.columns else None
    x_train_raw, x_valid_raw, y_train, y_valid = safe_train_valid_split(x, y_encoded, task, time_values=time_values)
    y_train_fit = log1p_target_values(y_train) if use_log1p_target else y_train

    preprocessor = build_preprocessor(feature_cols, train)
    if not constant_baseline:
        x_train = preprocessor.fit_transform(x_train_raw)
        x_valid = preprocessor.transform(x_valid_raw)
        x_test_processed = preprocessor.transform(test[feature_cols])
        if hasattr(x_train, "toarray"):
            x_train = x_train.toarray()
            x_valid = x_valid.toarray()
            x_test_processed = x_test_processed.toarray()
    else:
        x_train = x_valid = x_test_processed = None

    model_kind = "sklearn"
    torch_device = "cpu"

    if constant_baseline:
        model = None
        constant_value = y_train[0] if len(y_train) else 0
        preds_valid = np.repeat(constant_value, len(y_valid))
        model_kind = "constant"
    elif ACCELERATOR == "tpu":
        try:
            model = train_tpu_model(x_train, y_train_fit, task, num_classes)
            preds_valid = predict_tpu(model, x_valid, task, num_classes, "class")
            model_kind = "tpu"
        except Exception as exc:
            raise RuntimeError(f"TPU initialization failed: {exc}") from exc
    elif ACCELERATOR == "gpu":
        try:
            import torch

            torch_device = "cuda" if torch.cuda.is_available() else "cpu"
            if torch_device == "cpu":
                print("GPU requested but CUDA not available; falling back to CPU.")
            model = train_torch_mlp(x_train, y_train_fit, task, num_classes, torch_device)
            preds_valid = predict_torch(model, x_valid, task, num_classes, "class", torch_device)
            model_kind = "torch"
        except Exception as exc:
            print(f"GPU training failed for {target_col}, falling back to sklearn: {exc}")
            model = build_sklearn_model(task, preprocessor)
            model.fit(x_train_raw, y_train_fit)
            preds_valid = predict_sklearn(model, x_valid_raw, task, "class")
            model_kind = "sklearn"
    else:
        model = build_sklearn_model(task, preprocessor)
        model.fit(x_train_raw, y_train_fit)
        preds_valid = predict_sklearn(model, x_valid_raw, task, "class")
        model_kind = "sklearn"

    if task == "regression" and prediction_kind != "ordinal":
        preds_valid = finalize_regression_predictions(
            preds_valid,
            y,
            target_col,
            use_log1p_target=use_log1p_target,
        )

    if task == "classification":
        metric = "accuracy"
        preds_eval = np.asarray(preds_valid)
        if preds_eval.ndim > 1:
            preds_eval = preds_eval.argmax(axis=1)
        score = accuracy_score(y_valid, preds_eval)
    else:
        metric = "rmsle" if use_log1p_target else "rmse"
        score = root_mean_squared_log_error(y_valid, preds_valid) if use_log1p_target else root_mean_squared_error(
            y_valid,
            preds_valid,
        )

    if constant_baseline:
        constant_value = y_train[0] if len(y_train) else 0
        preds = np.repeat(constant_value, len(test))
    elif ACCELERATOR == "tpu":
        preds = predict_tpu(model, x_test_processed, task, num_classes, prediction_kind)
    elif ACCELERATOR == "gpu":
        if model_kind == "torch":
            preds = predict_torch(model, x_test_processed, task, num_classes, prediction_kind, torch_device)
        else:
            preds = predict_sklearn(model, test[feature_cols], task, prediction_kind)
    else:
        preds = predict_sklearn(model, test[feature_cols], task, prediction_kind)
    if task == "regression" and prediction_kind != "ordinal":
        preds = finalize_regression_predictions(
            preds,
            y,
            target_col,
            use_log1p_target=use_log1p_target,
        )
    elif prediction_kind == "ordinal":
        preds = ordinal_predictions_from_continuous(np.asarray(preds, dtype=float), y)

    if (
        label_encoder is not None
        and prediction_kind == "class"
        and target_col in sample.columns
        and not pd.api.types.is_numeric_dtype(sample[target_col])
        and not pd.api.types.is_bool_dtype(sample[target_col])
    ):
        preds = label_encoder.inverse_transform(np.asarray(preds, dtype=int))

    return preds, {
        "task": task,
        "metric": metric,
        "score": float(score),
        "prediction_kind": prediction_kind,
        "model_kind": model_kind,
        "features": len(feature_cols),
        "split_strategy": "timeseries_holdout" if time_values is not None else "holdout",
        "time_column": time_col,
        "target_transform": "log1p" if use_log1p_target else None,
        "inverse_transform": "expm1" if use_log1p_target else None,
    }


def train_predict_class_probability_submission(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    id_col: str | None,
    feature_cols: list[str],
    target_col: str,
    time_col: str | None = None,
):
    prediction_cols = sample_prediction_columns(sample, id_col)
    x = train[feature_cols]
    y = train[target_col]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    time_values = train[time_col] if time_col and time_col in train.columns else None
    x_train_raw, x_valid_raw, y_train, y_valid = safe_train_valid_split(
        x,
        y_encoded,
        "classification",
        time_values=time_values,
    )

    if num_classes < 2:
        constant_value = y_train[0] if len(y_train) else 0
        valid_labels = np.repeat(constant_value, len(y_valid))
        test_proba = np.zeros((len(test), max(1, num_classes)), dtype=float)
        if num_classes:
            test_proba[:, int(constant_value)] = 1.0
        model_kind = "constant"
    else:
        preprocessor = build_preprocessor(feature_cols, train)
        model = build_sklearn_model("classification", preprocessor)
        model.fit(x_train_raw, y_train)
        valid_labels = model.predict(x_valid_raw)
        raw_proba = model.predict_proba(test[feature_cols])
        estimator_classes = np.asarray(model.named_steps["model"].classes_, dtype=int)
        test_proba = np.zeros((len(test), num_classes), dtype=float)
        for proba_idx, class_idx in enumerate(estimator_classes):
            test_proba[:, int(class_idx)] = raw_proba[:, proba_idx]
        model_kind = "sklearn"

    score = accuracy_score(y_valid, valid_labels)
    output = {}
    column_to_class = map_probability_columns_to_classes(prediction_cols, label_encoder.classes_)
    for col in prediction_cols:
        class_idx = column_to_class[col]
        if 0 <= class_idx < test_proba.shape[1]:
            output[col] = test_proba[:, class_idx]
        else:
            output[col] = np.zeros(len(test), dtype=float)
    row_sums = np.sum(np.column_stack([output[col] for col in prediction_cols]), axis=1)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    for col in prediction_cols:
        output[col] = output[col] / row_sums
    return output, {
        "task": "classification",
        "metric": "accuracy",
        "score": float(score),
        "prediction_kind": "probability_columns",
        "model_kind": model_kind,
        "target_column": target_col,
        "probability_columns": prediction_cols,
        "split_strategy": "timeseries_holdout" if time_values is not None else "holdout",
        "time_column": time_col,
    }


def map_probability_columns_to_classes(prediction_cols: list[str], classes) -> dict[str, int]:
    class_labels = [str(value) for value in classes]
    normalized_class_to_idx = {normalize_label(value): idx for idx, value in enumerate(class_labels)}
    mapping: dict[str, int] = {}
    for idx, col in enumerate(prediction_cols):
        normalized_col = normalize_label(col)
        class_idx = None
        for candidate in probability_column_label_candidates(normalized_col):
            if candidate in normalized_class_to_idx:
                class_idx = normalized_class_to_idx[candidate]
                break
        mapping[col] = class_idx if class_idx is not None else idx
    return mapping


def probability_column_label_candidates(value: str) -> list[str]:
    prefixes = ("class", "target", "label", "probability", "prob", "prediction", "pred")
    suffixes = ("probability", "proba", "prob", "prediction", "pred", "score")
    candidates = [value]
    for prefix in prefixes:
        stripped = strip_label_prefix(value, prefix)
        if stripped != value:
            candidates.append(stripped)
    for candidate in list(candidates):
        for suffix in suffixes:
            stripped = strip_label_suffix(candidate, suffix)
            if stripped != candidate:
                candidates.append(stripped)
    return candidates


def train_predict_multi_label_column_submission(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    id_col: str | None,
    feature_cols: list[str],
    target_col: str,
    time_col: str | None = None,
):
    prediction_cols = sample_prediction_columns(sample, id_col)
    observed_labels = observed_multi_label_values(train[target_col], target_col)
    column_to_label = map_multi_label_columns_to_labels(prediction_cols, observed_labels)
    label_sets = multi_label_sets(train[target_col], target_col)
    indices = np.arange(len(train))
    time_values = train[time_col] if time_col and time_col in train.columns else None
    time_split = chronological_train_valid_indices(time_values, len(train)) if time_values is not None else None
    if time_split is not None:
        train_idx, valid_idx = time_split
    elif len(indices) < 2:
        train_idx = indices
        valid_idx = indices[:0]
    else:
        train_idx, valid_idx = train_test_split(indices, test_size=0.2, random_state=42)

    output = {}
    scores = []
    model_kinds = {}
    for col in prediction_cols:
        label = column_to_label[col]
        y = np.asarray([label in labels for labels in label_sets], dtype=int)
        y_train = y[train_idx]
        y_valid = y[valid_idx]
        prior = float(y.mean()) if y.size else 0.0
        if not feature_cols or len(np.unique(y_train)) < 2 or len(np.unique(y)) < 2:
            output[col] = np.full(len(test), prior, dtype=float)
            if y_valid.size:
                scores.append(float(accuracy_score(y_valid, np.full(y_valid.shape, prior >= 0.5))))
            model_kinds[col] = "constant_prior"
            continue
        try:
            preprocessor = build_preprocessor(feature_cols, train)
            model = Pipeline(
                [
                    ("pre", preprocessor),
                    ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
                ]
            )
            model.fit(train.iloc[train_idx][feature_cols], y_train)
            if y_valid.size:
                valid_pred = model.predict(train.iloc[valid_idx][feature_cols])
                scores.append(float(accuracy_score(y_valid, valid_pred)))
            model.fit(train[feature_cols], y)
            probabilities = np.asarray(model.predict_proba(test[feature_cols]), dtype=float)
            classes = np.asarray(model.named_steps["model"].classes_, dtype=int)
            positive_idx = int(np.where(classes == 1)[0][0]) if 1 in classes else probabilities.shape[1] - 1
            output[col] = np.clip(probabilities[:, positive_idx], 0.0, 1.0)
            model_kinds[col] = "logistic_regression_ovr"
        except Exception:
            output[col] = np.full(len(test), prior, dtype=float)
            model_kinds[col] = "constant_prior_fallback"
    return output, {
        "task": "classification",
        "metric": "binary_label_accuracy_mean",
        "score": float(np.mean(scores)) if scores else 0.0,
        "prediction_kind": "multi_label_columns",
        "model_kind": "multi_label_one_vs_rest",
        "target_column": target_col,
        "prediction_columns": prediction_cols,
        "column_to_label": column_to_label,
        "model_kinds": model_kinds,
        "split_strategy": "timeseries_holdout" if time_split is not None else "holdout",
        "time_column": time_col,
    }


def observed_multi_label_values(values: pd.Series, column_name: str) -> list[str]:
    observed = []
    seen = set()
    for labels in multi_label_sets(values, column_name):
        for label in labels:
            if label not in seen:
                seen.add(label)
                observed.append(label)
    return observed


def multi_label_sets(values: pd.Series, column_name: str) -> list[set[str]]:
    allow_whitespace = multi_label_allows_whitespace(column_name)
    return [
        {normalize_multi_label_name(label) for label in split_multi_label_prediction_value(value, allow_whitespace)}
        for value in values
    ]


def split_multi_label_prediction_value(value, allow_whitespace: bool) -> list[str]:
    if pd.isna(value):
        return []
    raw = str(value).strip()
    if not raw:
        return []
    if any(sep in raw for sep in ("|", ";", ",")):
        parts = re.split(r"[|;,]+", raw)
    elif allow_whitespace:
        parts = re.split(r"\s+", raw)
    else:
        parts = [raw]
    labels = [part.strip() for part in parts if part.strip()]
    return [
        label
        for label in labels
        if len(label) <= 48 and re.fullmatch(r"[A-Za-z0-9_.:+-]+", label)
    ]


def multi_label_allows_whitespace(column_name: str) -> bool:
    column_tokens = re.findall(r"[a-z0-9]+", str(column_name).lower())
    tokens = set(column_tokens)
    compact = "".join(column_tokens)
    return bool(tokens & {"labels", "tags", "classes", "categories"}) or "multilabel" in compact


def map_multi_label_columns_to_labels(prediction_cols: list[str], observed_labels: list[str]) -> dict[str, str]:
    observed_by_normalized = {normalize_multi_label_name(label): label for label in observed_labels}
    mapping = {}
    for col in prediction_cols:
        normalized = normalize_multi_label_name(col)
        mapping[col] = observed_by_normalized.get(normalized, normalized)
    return mapping


def normalize_multi_label_name(value) -> str:
    normalized = normalize_label(value)
    for prefix in ("class", "target", "label", "probability", "prob", "prediction", "pred"):
        stripped = strip_label_prefix(normalized, prefix)
        if stripped != normalized:
            normalized = stripped
            break
    for suffix in ("probability", "proba", "prob", "prediction", "pred", "score"):
        stripped = strip_label_suffix(normalized, suffix)
        if stripped != normalized:
            return stripped
    return normalized


def normalize_label(value) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def strip_label_prefix(value: str, prefix: str) -> str:
    if value == prefix:
        return value
    if value.startswith(prefix):
        return value[len(prefix):]
    return value


def strip_label_suffix(value: str, suffix: str) -> str:
    if value == suffix:
        return value
    if value.endswith(suffix):
        return value[: -len(suffix)]
    return value


def train_predict_numeric_multi_column_submission(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    id_col: str | None,
    feature_cols: list[str],
    target_col: str,
    time_col: str | None = None,
):
    prediction_cols = sample_prediction_columns(sample, id_col)
    base_preds, target_summary = train_predict_target(train, test, sample, feature_cols, target_col, time_col)
    prediction_kind = infer_numeric_multi_column_prediction_kind(sample, id_col)
    output = expand_numeric_multi_column_predictions(
        y=train[target_col],
        base_predictions=np.asarray(base_preds, dtype=float),
        prediction_columns=prediction_cols,
        prediction_kind=prediction_kind,
    )
    target_summary = {
        **target_summary,
        "prediction_kind": prediction_kind,
        "expanded_prediction_columns": prediction_cols,
    }
    return output, target_summary


def expand_numeric_multi_column_predictions(
    *,
    y: pd.Series,
    base_predictions: np.ndarray,
    prediction_columns: list[str],
    prediction_kind: str,
) -> dict[str, np.ndarray]:
    if prediction_kind == "quantile_columns":
        return quantile_column_predictions(
            y=y,
            base_predictions=base_predictions,
            prediction_columns=prediction_columns,
        )
    if prediction_kind == "prediction_interval_columns":
        return prediction_interval_column_predictions(
            y=y,
            base_predictions=base_predictions,
            prediction_columns=prediction_columns,
        )
    return {col: base_predictions.copy() for col in prediction_columns}


def quantile_column_predictions(
    *,
    y: pd.Series,
    base_predictions: np.ndarray,
    prediction_columns: list[str],
) -> dict[str, np.ndarray]:
    numeric = pd.to_numeric(y, errors="coerce").dropna()
    if numeric.empty:
        return {col: base_predictions.copy() for col in prediction_columns}
    median = float(numeric.quantile(0.5))
    raw: dict[str, np.ndarray] = {}
    ordered: list[tuple[float, str]] = []
    for col in prediction_columns:
        quantile = quantile_from_prediction_column(col)
        if quantile is None:
            quantile = 0.5
        raw[col] = base_predictions + float(numeric.quantile(quantile)) - median
        ordered.append((quantile, col))
    if len(ordered) >= 2:
        sorted_columns = [col for _, col in sorted(ordered)]
        stacked = np.column_stack([raw[col] for col in sorted_columns])
        stacked = np.maximum.accumulate(stacked, axis=1)
        for idx, col in enumerate(sorted_columns):
            raw[col] = stacked[:, idx]
    return raw


def prediction_interval_column_predictions(
    *,
    y: pd.Series,
    base_predictions: np.ndarray,
    prediction_columns: list[str],
) -> dict[str, np.ndarray]:
    numeric = pd.to_numeric(y, errors="coerce").dropna()
    if numeric.empty:
        return {col: base_predictions.copy() for col in prediction_columns}
    median = float(numeric.quantile(0.5))
    lower_offset = float(numeric.quantile(0.1)) - median
    upper_offset = float(numeric.quantile(0.9)) - median
    output: dict[str, np.ndarray] = {}
    lower_columns: list[str] = []
    upper_columns: list[str] = []
    for col in prediction_columns:
        role = prediction_interval_column_role(col)
        if role == "lower":
            output[col] = base_predictions + lower_offset
            lower_columns.append(col)
        elif role == "upper":
            output[col] = base_predictions + upper_offset
            upper_columns.append(col)
        else:
            output[col] = base_predictions.copy()
    for lower_col in lower_columns:
        for upper_col in upper_columns:
            lower = np.minimum(output[lower_col], output[upper_col])
            upper = np.maximum(output[lower_col], output[upper_col])
            output[lower_col] = lower
            output[upper_col] = upper
    return output


def prediction_interval_column_role(name) -> str | None:
    compact = "".join(ch for ch in str(name).lower() if ch.isalnum())
    if compact in {"lower", "lo", "low", "lwr", "lowerbound", "lowerci", "lowerlimit"}:
        return "lower"
    if compact in {"upper", "hi", "high", "upr", "upperbound", "upperci", "upperlimit"}:
        return "upper"
    return None


def quantile_from_prediction_column(name) -> float | None:
    lower = str(name).lower().strip()
    compact = re.sub(r"[^a-z0-9.]+", "", lower)
    aliases = {"median": 0.5, "p50": 0.5, "q50": 0.5, "quantile50": 0.5}
    if compact in aliases:
        return aliases[compact]
    match = re.search(r"(?:^|[_\-.])(?:p|q)(0?\.\d+|0?[1-9]|[1-9][0-9])(?:$|[_\-.])", lower)
    if not match:
        match = re.search(r"(?:quantile|percentile)[_\-.]?(0?\.\d+|0?[1-9]|[1-9][0-9])", lower)
    if not match:
        match = re.search(r"(?:^|[_\-.])(0?\.\d+)(?:$|[_\-.])", lower)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    if value > 1.0:
        value /= 100.0
    if 0.0 < value < 1.0:
        return value
    return None


def main() -> None:
    print(f"competition slug: {COMPETITION_SLUG}")
    extracted = extract_data_archives()
    if extracted:
        print(f"extracted input archives: {len(extracted)} files")
    files = []
    for root in data_roots():
        files.extend(materialize_sqlite_tables(root))
        files.extend(materialize_duckdb_tables(root))
        files.extend(find_tabular_files(root))
    synthesized_sample = synthesize_sample_submission_from_format(files)
    if synthesized_sample is not None:
        files.append(synthesized_sample)
    train_path, test_path, sample_path = pick_files(files)
    print(f"train: {train_path}")
    print(f"test: {test_path}")
    print(f"sample: {sample_path}")

    train = read_table(train_path)
    test = read_table(test_path)
    sample = read_table(sample_path)
    train, test = align_train_test_column_case_to_sample(train, test, sample)

    try:
        id_col, target_col, feature_cols, target_cols = infer_target(train, test, sample)
    except ValueError:
        merged_train = maybe_merge_train_labels(files, train_path, test_path, sample_path, train, test, sample)
        if merged_train is None:
            unlabeled_layout = infer_unlabeled_submission_layout(train, test, sample)
            if unlabeled_layout is None:
                raise
            id_col, target_col, feature_cols, target_cols = unlabeled_layout
        else:
            train = merged_train
            id_col, target_col, feature_cols, target_cols = infer_target(train, test, sample)
    test, sample = preserve_id_column_values(test_path, sample_path, test, sample, id_col)
    print(f"id column: {id_col}")
    print(f"target column: {target_col}")
    if len(target_cols) > 1:
        print(f"multi-target detected; fitting one baseline per target: {target_cols}")
    temporal_calendar_feature_cols = add_temporal_calendar_features(train, test, feature_cols)
    if temporal_calendar_feature_cols:
        feature_cols = [*feature_cols, *temporal_calendar_feature_cols]
        print(f"temporal calendar features added: {temporal_calendar_feature_cols}")
    file_reference_feature_cols = add_file_reference_features(train, test, feature_cols)
    file_reference_feature_summary = summarize_file_reference_features(train, test, file_reference_feature_cols)
    if file_reference_feature_cols:
        feature_cols = [*feature_cols, *file_reference_feature_cols]
        print(f"file reference features added: {file_reference_feature_cols}")
    split_time_col = infer_time_column(train, test, feature_cols)
    if split_time_col:
        print(f"time column: {split_time_col}; using chronological holdout")
    print(f"feature count: {len(feature_cols)}")

    submission = build_submission_template(sample, test, id_col)
    target_metrics = {}
    rle_cols = rle_segmentation_columns(sample)
    if rle_cols:
        print(f"RLE segmentation submission detected; writing empty masks for: {rle_cols}")
        for current_target in rle_cols:
            submission[current_target] = "-"
            target_metrics[current_target] = {
                "task": "segmentation",
                "metric": "segmentation_rle_placeholder",
                "score": 0.0,
                "prediction_kind": "rle",
                "model_kind": "rle_empty_mask_baseline",
                "mask_columns": rle_cols,
            }
        submission_path = resolve_submission_path(sample_path)
        write_table(submission, submission_path)
        print(f"wrote submission: {submission_path}")
        primary_summary = target_metrics[rle_cols[0]]
        payload = {
            "task": primary_summary["task"],
            "metric": primary_summary["metric"],
            "score": float(primary_summary["score"]),
            "target_metrics": target_metrics,
            "temporal_calendar_feature_columns": temporal_calendar_feature_cols,
            "file_reference_feature_columns": file_reference_feature_cols,
            "file_reference_feature_summary": file_reference_feature_summary,
            "generated_at": datetime.now(UTC).isoformat(),
        }
        try:
            METRICS_PATH.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            print(f"failed to write metrics.json: {exc}")
        return
    if is_multi_label_column_submission(train, sample, id_col, target_col, target_cols):
        print(f"multi-label column submission detected; expanding {target_col} into sample columns")
        multi_label_preds, target_summary = train_predict_multi_label_column_submission(
            train,
            test,
            sample,
            id_col,
            feature_cols,
            target_col,
            split_time_col,
        )
        target_metrics[target_col] = target_summary
        print(f"validation {target_col} {target_summary['metric']}: {target_summary['score']:.4f}")
        for current_target, preds in multi_label_preds.items():
            assign_prediction_column(submission, test, id_col, current_target, preds)
    elif is_class_probability_submission(train, sample, id_col, target_col, target_cols):
        print(f"class-probability submission detected; expanding {target_col} into sample columns")
        probability_preds, target_summary = train_predict_class_probability_submission(
            train,
            test,
            sample,
            id_col,
            feature_cols,
            target_col,
            split_time_col,
        )
        target_metrics[target_col] = target_summary
        print(f"validation {target_col} {target_summary['metric']}: {target_summary['score']:.4f}")
        for current_target, preds in probability_preds.items():
            assign_prediction_column(submission, test, id_col, current_target, preds)
    elif is_survival_single_score_submission(sample, id_col, target_cols):
        print("survival event/time submission detected; emitting single risk score column")
        event_col, survival_time_col = survival_event_time_columns(target_cols)
        assert event_col is not None and survival_time_col is not None
        event_preds, event_summary = train_predict_target(
            train,
            test,
            sample,
            feature_cols,
            event_col,
            split_time_col,
        )
        time_preds, time_summary = train_predict_target(
            train,
            test,
            sample,
            feature_cols,
            survival_time_col,
            split_time_col,
        )
        score_col = survival_score_submission_column(sample, id_col)
        risk_preds = survival_risk_scores(np.asarray(event_preds, dtype=float), np.asarray(time_preds, dtype=float))
        target_metrics[event_col] = event_summary
        target_metrics[survival_time_col] = time_summary
        target_metrics[score_col] = {
            "task": "survival",
            "metric": "survival_risk_score",
            "score": 0.0,
            "prediction_kind": "continuous",
            "model_kind": "survival_risk_score",
            "event_column": event_col,
            "time_column": survival_time_col,
        }
        assign_prediction_column(submission, test, id_col, score_col, risk_preds)
    elif is_numeric_multi_column_regression_submission(train, sample, id_col, target_col, target_cols):
        print(f"numeric multi-column regression submission detected; expanding {target_col} into sample columns")
        expanded_preds, target_summary = train_predict_numeric_multi_column_submission(
            train,
            test,
            sample,
            id_col,
            feature_cols,
            target_col,
            split_time_col,
        )
        target_metrics[target_col] = target_summary
        print(f"validation {target_col} {target_summary['metric']}: {target_summary['score']:.4f}")
        for current_target, preds in expanded_preds.items():
            assign_prediction_column(submission, test, id_col, current_target, preds)
    else:
        for current_target in target_cols:
            preds, target_summary = train_predict_target(
                train,
                test,
                sample,
                feature_cols,
                current_target,
                split_time_col,
            )
            target_metrics[current_target] = target_summary
            print(f"validation {current_target} {target_summary['metric']}: {target_summary['score']:.4f}")
            assign_prediction_column(submission, test, id_col, current_target, preds)
    submission_path = resolve_submission_path(sample_path)
    write_table(submission, submission_path)
    print(f"wrote submission: {submission_path}")

    primary_summary = target_metrics[target_col]
    payload = {
        "task": primary_summary["task"],
        "metric": primary_summary["metric"],
        "score": float(primary_summary["score"]),
        "target_metrics": target_metrics,
        "temporal_calendar_feature_columns": temporal_calendar_feature_cols,
        "file_reference_feature_columns": file_reference_feature_cols,
        "file_reference_feature_summary": file_reference_feature_summary,
        "time_column": split_time_col,
        "split_strategy": "timeseries_holdout" if split_time_col else "holdout",
        "generated_at": datetime.now(UTC).isoformat(),
    }
    try:
        METRICS_PATH.write_text(json.dumps(payload, indent=2))
    except Exception as exc:
        print(f"failed to write metrics.json: {exc}")


if __name__ == "__main__":
    main()
""".strip()


def _inject_shared_suffix_literals(template: str) -> str:
    return (
        template.replace("__SQLITE_TABULAR_SUFFIXES_JSON__", json.dumps(sorted(SQLITE_TABULAR_SUFFIXES)))
        .replace("__DUCKDB_TABULAR_SUFFIXES_JSON__", json.dumps(sorted(DUCKDB_TABULAR_SUFFIXES)))
        .replace("__TABULAR_ANNDATA_SUFFIXES_JSON__", json.dumps(sorted(TABULAR_ANNDATA_SUFFIXES)))
        .replace("__ASSET_COMPRESSION_SUFFIXES_JSON__", json.dumps(list(ASSET_COMPRESSION_SUFFIXES)))
        .replace(
            "__TABULAR_INPUT_SUFFIXES_ORDERED_JSON__",
            json.dumps(list(TABULAR_INPUT_SUFFIXES_ORDERED)),
        )
        .replace("__TABULAR_SUBMISSION_SUFFIXES_JSON__", json.dumps(sorted(TABULAR_SUBMISSION_SUFFIXES)))
        .replace("__TABULAR_INPUT_SUFFIXES_JSON__", json.dumps(sorted(TABULAR_INPUT_SUFFIXES)))
        .replace("__TABULAR_PICKLE_SUFFIXES_JSON__", json.dumps(sorted(TABULAR_PICKLE_SUFFIXES)))
        .replace("__TABULAR_STRUCTURED_SUFFIXES_JSON__", json.dumps(sorted(TABULAR_STRUCTURED_SUFFIXES)))
        .replace("__TABULAR_TEXT_SUFFIXES_JSON__", json.dumps(sorted(TABULAR_TEXT_SUFFIXES)))
        .replace("__TABULAR_ARFF_SUFFIXES_JSON__", json.dumps(sorted(TABULAR_ARFF_SUFFIXES)))
        .replace("__TABULAR_ARROW_IPC_SUFFIXES_JSON__", json.dumps(sorted(TABULAR_ARROW_IPC_SUFFIXES)))
        .replace("__TABULAR_PARQUET_SUFFIXES_JSON__", json.dumps(sorted(TABULAR_PARQUET_SUFFIXES)))
        .replace("__TABULAR_EXCEL_INPUT_ONLY_SUFFIXES_JSON__", json.dumps(sorted(TABULAR_EXCEL_INPUT_ONLY_SUFFIXES)))
        .replace("__TABULAR_EXCEL_SUFFIXES_JSON__", json.dumps(sorted(TABULAR_EXCEL_SUFFIXES)))
        .replace("__TABULAR_FITS_SUFFIXES_JSON__", json.dumps(sorted(TABULAR_FITS_SUFFIXES)))
        .replace("__TABULAR_FIXED_WIDTH_SUFFIX_PREFIXES_JSON__", json.dumps(list(TABULAR_FIXED_WIDTH_SUFFIX_PREFIXES)))
        .replace("__TABULAR_GEOPACKAGE_SUFFIXES_JSON__", json.dumps(sorted(TABULAR_GEOPACKAGE_SUFFIXES)))
        .replace("__TABULAR_GEOJSON_SUFFIXES_JSON__", json.dumps(sorted(TABULAR_GEOJSON_SUFFIXES)))
        .replace("__TABULAR_HDF_SUFFIXES_JSON__", json.dumps(sorted(TABULAR_HDF_SUFFIXES)))
        .replace("__TABULAR_HTML_SUFFIX_PREFIXES_JSON__", json.dumps(list(TABULAR_HTML_SUFFIX_PREFIXES)))
        .replace("__TABULAR_JSON_LINES_SUFFIX_PREFIXES_JSON__", json.dumps(list(TABULAR_JSON_LINES_SUFFIX_PREFIXES)))
        .replace("__TABULAR_KML_SUFFIXES_JSON__", json.dumps(sorted(TABULAR_KML_SUFFIXES)))
        .replace("__TABULAR_LOOM_SUFFIXES_JSON__", json.dumps(sorted(TABULAR_LOOM_SUFFIXES)))
        .replace("__TABULAR_MATLAB_SUFFIXES_JSON__", json.dumps(sorted(TABULAR_MATLAB_SUFFIXES)))
        .replace("__TABULAR_NETCDF_SUFFIXES_JSON__", json.dumps(sorted(TABULAR_NETCDF_SUFFIXES)))
        .replace("__TABULAR_NUMPY_SUFFIXES_JSON__", json.dumps(sorted(TABULAR_NUMPY_SUFFIXES)))
        .replace("__TABULAR_RDATA_SUFFIXES_JSON__", json.dumps(sorted(TABULAR_RDATA_SUFFIXES)))
        .replace("__TABULAR_SAS_SUFFIXES_JSON__", json.dumps(sorted(TABULAR_SAS_SUFFIXES)))
        .replace("__TABULAR_SHAPEFILE_SUFFIXES_JSON__", json.dumps(sorted(TABULAR_SHAPEFILE_SUFFIXES)))
        .replace("__TABULAR_SPSS_SUFFIXES_JSON__", json.dumps(sorted(TABULAR_SPSS_SUFFIXES)))
        .replace("__TABULAR_STATA_SUFFIXES_JSON__", json.dumps(sorted(TABULAR_STATA_SUFFIXES)))
        .replace("__TABULAR_SVMLIGHT_SUFFIX_PREFIXES_JSON__", json.dumps(list(TABULAR_SVMLIGHT_SUFFIX_PREFIXES)))
        .replace(
            "__TABULAR_TAB_DELIMITED_SUFFIX_PREFIXES_JSON__",
            json.dumps(list(TABULAR_TAB_DELIMITED_SUFFIX_PREFIXES)),
        )
        .replace("__DATA_ASSET_SUFFIXES_JSON__", json.dumps(sorted(DATA_ASSET_SUFFIXES)))
        .replace("__NON_TABULAR_SUBMISSION_SUFFIXES_JSON__", json.dumps(sorted(NON_TABULAR_SUBMISSION_SUFFIXES)))
        .replace("__MODEL_ARTIFACT_COMPOUND_SUFFIXES_JSON__", json.dumps(sorted(MODEL_ARTIFACT_COMPOUND_SUFFIXES)))
        .replace("__MODEL_ARTIFACT_FILENAMES_JSON__", json.dumps(sorted(MODEL_ARTIFACT_FILENAMES)))
        .replace(
            "__MODEL_ARTIFACT_JSON_SIDECAR_FILENAMES_JSON__",
            json.dumps(sorted(MODEL_ARTIFACT_JSON_SIDECAR_FILENAMES)),
        )
        .replace(
            "__MODEL_ARTIFACT_TEXT_SIDECAR_FILENAMES_JSON__",
            json.dumps(sorted(MODEL_ARTIFACT_TEXT_SIDECAR_FILENAMES)),
        )
        .replace("__ASSET_COLLECTION_DIR_NAMES_JSON__", json.dumps(sorted(ASSET_COLLECTION_DIR_NAMES)))
        .replace("__ARCHIVE_SUBMISSION_SUFFIXES_JSON__", json.dumps(sorted(ARCHIVE_SUBMISSION_SUFFIXES)))
        .replace("__ZSTD_TAR_ARCHIVE_SUFFIXES_JSON__", json.dumps(sorted(ZSTD_TAR_ARCHIVE_SUBMISSION_SUFFIXES)))
        .replace("__CODE_FENCE_LANG_TO_SUFFIX_JSON__", json.dumps(CODE_FENCE_LANG_TO_SUFFIX, sort_keys=True))
        .replace("__SUBMISSION_TOKEN_PATTERN_SPECS_JSON__", json.dumps(SUBMISSION_TOKEN_PATTERN_SPECS))
        .replace("__COMPRESSION_TOKEN_PATTERN_SPECS_JSON__", json.dumps(COMPRESSION_TOKEN_PATTERN_SPECS))
        .replace(
            "__COMPRESSIBLE_SUBMISSION_KEYWORD_SUFFIXES_JSON__",
            json.dumps(sorted(COMPRESSIBLE_SUBMISSION_KEYWORD_SUFFIXES)),
        )
        .replace("__SAMPLE_OUTPUT_NAME_TOKENS_JSON__", json.dumps(sorted(SAMPLE_OUTPUT_NAME_TOKENS)))
        .replace("__SAMPLE_COMPACT_NAME_ALIASES_JSON__", json.dumps(sorted(SAMPLE_COMPACT_NAME_ALIASES)))
        .replace("__CONFIGURED_TEMPLATE_STEMS_JSON__", json.dumps(sorted(CONFIGURED_TEMPLATE_STEMS)))
        .replace("__DIRECTORY_ASSET_SUFFIXES_JSON__", json.dumps(sorted(DIRECTORY_ARRAY_SUFFIXES)))
        .replace("__IMAGE_SUFFIXES_JSON__", json.dumps(sorted(IMAGE_SUFFIXES)))
        .replace("__AUDIO_SUFFIXES_JSON__", json.dumps(sorted(AUDIO_SUFFIXES)))
        .replace("__VIDEO_SUFFIXES_JSON__", json.dumps(sorted(VIDEO_SUFFIXES)))
        .replace("__DICOM_BASE_SUFFIXES_JSON__", json.dumps(sorted(DICOM_IMAGE_BASE_SUFFIXES)))
        .replace("__DICOM_SUFFIXES_JSON__", json.dumps(sorted(DICOM_IMAGE_SUFFIXES)))
        .replace("__NIFTI_BASE_SUFFIXES_JSON__", json.dumps(sorted(NIFTI_IMAGE_BASE_SUFFIXES)))
        .replace("__NIFTI_SUFFIXES_JSON__", json.dumps(sorted(NIFTI_IMAGE_SUFFIXES)))
        .replace("__MEDICAL_HEADER_BASE_SUFFIXES_JSON__", json.dumps(sorted(MEDICAL_HEADER_IMAGE_BASE_SUFFIXES)))
        .replace("__MEDICAL_HEADER_SUFFIXES_JSON__", json.dumps(sorted(MEDICAL_HEADER_IMAGE_SUFFIXES)))
        .replace("__POINT_CLOUD_TEXT_METADATA_SUFFIXES_JSON__", json.dumps(sorted(POINT_CLOUD_TEXT_METADATA_SUFFIXES)))
        .replace("__GRAPH_XML_BASE_SUFFIXES_JSON__", json.dumps(sorted(GRAPH_XML_BASE_SUFFIXES)))
        .replace("__GRAPH_EDGE_LIST_BASE_SUFFIXES_JSON__", json.dumps(sorted(GRAPH_EDGE_LIST_BASE_SUFFIXES)))
        .replace("__DOCUMENT_HTML_BASE_SUFFIXES_JSON__", json.dumps(sorted(DOCUMENT_HTML_BASE_SUFFIXES)))
        .replace("__DOCUMENT_TEXT_METADATA_SUFFIXES_JSON__", json.dumps(sorted(DOCUMENT_TEXT_METADATA_SUFFIXES)))
        .replace("__BIO_SEQUENCE_BASE_SUFFIXES_JSON__", json.dumps(sorted(BIO_SEQUENCE_BASE_SUFFIXES)))
        .replace("__BIO_FASTQ_BASE_SUFFIXES_JSON__", json.dumps(sorted(BIO_FASTQ_BASE_SUFFIXES)))
        .replace("__BIO_PDB_STRUCTURE_BASE_SUFFIXES_JSON__", json.dumps(sorted(BIO_PDB_STRUCTURE_BASE_SUFFIXES)))
        .replace("__BIO_MOL_STRUCTURE_BASE_SUFFIXES_JSON__", json.dumps(sorted(BIO_MOL_STRUCTURE_BASE_SUFFIXES)))
        .replace("__ROLE_SUFFIXES_JSON__", json.dumps(sorted(ROLE_SUFFIXES)))
        .replace("__TEST_DIRECT_ROLE_ALIASES_JSON__", json.dumps(sorted(TEST_DIRECT_ROLE_ALIASES)))
        .replace("__ROLE_TRAILING_PREFIXES_JSON__", json.dumps(sorted(ROLE_TRAILING_PREFIXES)))
        .replace(
            "__ROLE_ALIASES_JSON__",
            json.dumps({role: sorted(aliases) for role, aliases in sorted(ROLE_ALIASES.items())}),
        )
        .replace("__FILE_REFERENCE_NAME_TOKENS_JSON__", json.dumps(list(FILE_REFERENCE_NAME_TOKENS)))
        .replace("__TEXT_PREDICTION_NAME_TOKENS_JSON__", json.dumps(list(TEXT_PREDICTION_NAME_TOKENS)))
        .replace("__RLE_SEGMENTATION_COLUMN_TOKENS_JSON__", json.dumps(sorted(RLE_SEGMENTATION_COLUMN_TOKENS)))
        .replace("__ID_LIKE_COLUMN_NAMES_JSON__", json.dumps(sorted(ID_LIKE_COLUMN_NAMES)))
        .replace("__ASSET_LABEL_TABLE_TOKENS_JSON__", json.dumps(list(ASSET_LABEL_TABLE_TOKENS)))
    )


KERNEL_TEMPLATE = _inject_shared_suffix_literals(KERNEL_TEMPLATE)


class KaggleNotebookRunner:
    name = "kaggle_notebook"

    def run(self, context: RunContext) -> RunResult:
        slug = context.slug
        run_id = context.run_id
        paths = context.paths

        run_dir = paths.run_dir(run_id)
        kernel_dir = run_dir / "kernel"
        output_dir = run_dir / "output"
        logs_dir = run_dir / "logs"
        summary_path = run_dir / "summary.json"

        kernel_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        kaggle_username = resolve_kaggle_username(context.kaggle_username)
        kernel_slug = build_kernel_slug(slug, run_id)
        kernel_id = f"{kaggle_username}/{kernel_slug}"

        accelerator = context.accelerator
        enable_internet = bool(context.enable_internet)
        if infer_code_competition_from_paths(paths):
            enable_internet = False

        metadata = build_kernel_metadata(
            kaggle_username=kaggle_username,
            kernel_slug=kernel_slug,
            title=kernel_slug.replace("-", " "),
            competition_slug=slug,
            accelerator=accelerator,
            enable_internet=enable_internet,
            source_config=load_kernel_source_config(paths.plan_path),
        )
        kernel_metadata_path = kernel_dir / "kernel-metadata.json"
        write_json_object(kernel_metadata_path, metadata)

        kernel_main_path = kernel_dir / "main.py"
        kernel_main_path.write_text(
            render_kernel_main(
                slug,
                accelerator,
                default_submission_filename=_default_submission_filename_from_format(paths.submission_format_md_path),
            ),
            encoding="utf-8",
        )

        commands = [
            f"kaggle kernels push -p {kernel_dir}",
            f"kaggle kernels status {kernel_id}",
            f"kaggle kernels output {kernel_id} -p {output_dir}",
        ]

        summary = {
            "schema_version": 1,
            "slug": slug,
            "run_id": run_id,
            "runner": self.name,
            "kernel_slug": kernel_slug,
            "kernel_id": kernel_id,
            "accelerator": accelerator,
            "enable_internet": enable_internet,
            "dry_run": context.dry_run,
            "generated_at": datetime.now(UTC).isoformat(),
            "commands": commands,
        }

        if context.dry_run:
            print("[yellow]DRY RUN[/yellow]: Kaggle CLI commands will not be executed.")
            for command in commands:
                print(f"[cyan]planned[/cyan]: {command}")
            write_json_object(summary_path, summary)
            return RunResult(
                run_id=run_id,
                runner=self.name,
                submission_path=None,
                summary_path=summary_path,
                analysis_path=None,
                kernel_slug=kernel_slug,
            )

        print(f"[cyan]checking competition access[/cyan]: {slug}")
        kaggle_cli.competitions_files(slug)
        print(f"[cyan]pushing kernel[/cyan]: {kernel_dir}")
        kaggle_cli.kernels_push(kernel_dir, slug=slug, stream_output=True)
        print(f"[cyan]waiting for kernel[/cyan]: {kernel_id}")
        _wait_for_kernel(kernel_id, logs_dir=logs_dir, slug=slug, kernel_dir=kernel_dir)
        print(f"[cyan]downloading kernel output[/cyan]: {output_dir}")
        kaggle_cli.kernels_output(kernel_id, output_dir, slug=slug, stream_output=True, force=True)

        submission_path = find_submission_file(output_dir)
        paths.submissions_dir.mkdir(parents=True, exist_ok=True)
        local_submission = store_submission_artifact(
            source=submission_path,
            destination_dir=paths.submissions_dir,
            run_id=run_id,
        )

        summary["submission_path"] = str(local_submission)
        write_json_object(summary_path, summary)

        return RunResult(
            run_id=run_id,
            runner=self.name,
            submission_path=local_submission,
            summary_path=summary_path,
            analysis_path=None,
            kernel_slug=kernel_slug,
        )


def sanitize_kernel_slug(value: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", value.lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    if not slug:
        raise ValueError("Kernel slug is empty after sanitization.")
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("-")
    return slug


def build_kernel_slug(competition_slug: str, run_id: str, *, max_length: int = 50) -> str:
    short_id = run_id.rsplit("-", 1)[-1]
    base_slug = sanitize_kernel_slug(competition_slug, max_length=200)
    base = f"kb-{base_slug}-{short_id}"
    if len(base) <= max_length:
        return base
    reserved = len("kb--") + len(short_id)
    room = max_length - reserved
    if room < 1:
        raise ValueError("Kernel slug length budget is too small.")
    trimmed_slug = base_slug[:room].rstrip("-")
    if not trimmed_slug:
        raise ValueError("Kernel slug is empty after trimming.")
    return f"kb-{trimmed_slug}-{short_id}"


def build_kernel_metadata(
    *,
    kaggle_username: str,
    kernel_slug: str,
    title: str,
    competition_slug: str,
    accelerator: str,
    enable_internet: bool,
    source_config: KernelSourceConfig | None = None,
) -> dict[str, object]:
    enable_gpu = accelerator == "gpu"
    enable_tpu = accelerator == "tpu"
    if enable_gpu and enable_tpu:
        raise ValueError("enable_gpu and enable_tpu cannot both be true.")
    source_config = source_config or KernelSourceConfig()
    return {
        "id": f"{kaggle_username}/{kernel_slug}",
        "title": title,
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_tpu": enable_tpu,
        "enable_internet": enable_internet,
        "competition_sources": [competition_slug],
        "dataset_sources": list(source_config.dataset_sources),
        "kernel_sources": list(source_config.kernel_sources),
        "model_sources": list(dict.fromkeys((*source_config.model_sources, *source_config.required_model_sources))),
        "keywords": [],
    }


def render_kernel_main(
    competition_slug: str,
    accelerator: str,
    *,
    default_submission_filename: str | None = None,
) -> str:
    default_name = _safe_default_submission_filename(default_submission_filename)
    return (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", competition_slug)
        .replace("__ACCELERATOR__", accelerator)
        .replace('default_name = "submission.csv"', f"default_name = {json.dumps(default_name)}")
        .strip()
    )


def _default_submission_filename_from_format(submission_format_path: Path) -> str | None:
    hint = load_submission_format_hint(submission_format_path)
    if hint is None or not hint.expected_suffixes:
        return None
    try:
        text = submission_format_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        text = ""
    return output_filename_from_format_text(
        text,
        expected_suffixes=hint.expected_suffixes,
        allowed_suffixes=_KAGGLE_NOTEBOOK_OUTPUT_SUFFIXES,
    )


def _safe_default_submission_filename(value: str | None) -> str:
    name = Path(str(value or "").strip()).name
    if not name:
        return "submission.csv"
    suffix = _template_output_suffix(name)
    if configured_submission_filename_is_template(name):
        return f"submission{suffix}" if suffix in _KAGGLE_NOTEBOOK_OUTPUT_SUFFIXES else "submission.csv"
    if suffix not in _KAGGLE_NOTEBOOK_OUTPUT_SUFFIXES:
        return "submission.csv"
    return name


def _template_output_suffix(name: str) -> str:
    lowered = str(name).lower()
    for suffix in _KAGGLE_NOTEBOOK_OUTPUT_SUFFIXES:
        if lowered.endswith(suffix):
            return suffix
    return Path(name).suffix.lower()


def find_submission_file(output_dir: Path) -> Path:
    submission_path = _kernel_outputs.find_submission_file(output_dir)
    if submission_path is None:
        raise FileNotFoundError(f"No submission artifact found under {output_dir}")
    return submission_path


def _wait_for_kernel(kernel_id: str, *, logs_dir: Path, slug: str, kernel_dir: Path) -> None:
    timeout_seconds = 60 * 60
    poll_seconds = 15
    start = time.monotonic()
    status_log = logs_dir / "kernel_status.log"
    last_status = None

    while True:
        output = kaggle_cli.kernels_status(kernel_id, slug=slug)
        status = parse_kernel_status(output)
        if status != last_status:
            print(f"[cyan]kernel status[/cyan]: {status}")
            last_status = status
        status_log.write_text(output, encoding="utf-8")
        if status == "complete":
            return
        if status == "failed":
            _stop_failed_kernel_run(
                kernel_id, kernel_dir=kernel_dir, logs_dir=logs_dir, slug=slug, status_output=output
            )
            raise RuntimeError(f"Kernel run failed: {output}")
        if time.monotonic() - start > timeout_seconds:
            raise TimeoutError("Timed out waiting for kernel completion.")
        time.sleep(poll_seconds)


def _stop_failed_kernel_run(
    kernel_id: str,
    *,
    kernel_dir: Path,
    logs_dir: Path,
    slug: str,
    status_output: str,
) -> None:
    if not env_flag("KAGGLEBOT_STOP_FAILED_KERNEL", default=True):
        return

    stop_dir = logs_dir.parent / "kernel-stop"
    stop_log = logs_dir / "kernel_stop.log"
    try:
        metadata_path = kernel_dir / "kernel-metadata.json"
        metadata = load_json_object(metadata_path)
        if metadata is None:
            raise ValueError(f"Kernel metadata is not a JSON object: {metadata_path}")
        metadata["id"] = kernel_id
        metadata["enable_gpu"] = False
        metadata["enable_tpu"] = False
        metadata["enable_internet"] = False
        metadata["code_file"] = "main.py"
        stop_dir.mkdir(parents=True, exist_ok=True)
        write_json_object(stop_dir / "kernel-metadata.json", metadata)
        (stop_dir / "main.py").write_text(
            "\n".join(
                [
                    "from pathlib import Path",
                    "Path('/kaggle/working/kagglebot_stopped_failed_gpu_kernel.txt').write_text(",
                    "    'KaggleBot replaced a failed GPU run with a CPU stop marker.\\n',",
                    "    encoding='utf-8',",
                    ")",
                    "print('KaggleBot stop marker completed.')",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"[yellow]kernel failed[/yellow]: pushing CPU stop marker for {kernel_id}")
        output = kaggle_cli.kernels_push(stop_dir, slug=slug, stream_output=True)
        stop_log.write_text(
            f"status_output:\n{status_output}\n\nstop_marker_output:\n{output}\n",
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        stop_log.write_text(
            f"status_output:\n{status_output}\n\nstop_marker_failed:\n{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        print(f"[yellow]kernel stop marker failed[/yellow]: {type(exc).__name__}: {exc}")
