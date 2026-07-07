from __future__ import annotations

import base64
import gzip
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

from kagglebot.asset_modality import artifact_suffix
from kagglebot.baseline_tokens import ID_LIKE_COLUMN_NAMES
from kagglebot.compression_suffixes import ASSET_COMPRESSION_SUFFIXES
from kagglebot.exceptions import KernelFailedError, SubmissionValidationError
from kagglebot.paths import CompetitionPaths
from kagglebot.sample_name_aliases import SAMPLE_COMPACT_NAME_ALIASES, SAMPLE_OUTPUT_NAME_TOKENS
from kagglebot.submission_extension_hints import (
    ARCHIVE_SUBMISSION_SUFFIXES_ORDERED,
    EXTERNAL_ARCHIVE_SUBMISSION_SUFFIXES,
    NON_TABULAR_SUBMISSION_SUFFIXES,
    ZSTD_TAR_ARCHIVE_SUBMISSION_SUFFIXES,
)
from kagglebot.submission_sample_discovery import (
    DUCKDB_TABULAR_SUFFIXES,
    SQLITE_TABULAR_SUFFIXES,
    TABULAR_ANNDATA_SUFFIXES,
    TABULAR_ARFF_SUFFIXES,
    TABULAR_ARROW_IPC_SUFFIXES,
    TABULAR_EXCEL_INPUT_ONLY_SUFFIXES,
    TABULAR_EXCEL_SUFFIXES,
    TABULAR_FITS_SUFFIXES,
    TABULAR_FIXED_WIDTH_SUFFIX_PREFIXES,
    TABULAR_GEOPACKAGE_SUFFIXES,
    TABULAR_HDF_SUFFIXES,
    TABULAR_HTML_SUFFIX_PREFIXES,
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
    TABULAR_SUBMISSION_SUFFIXES_ORDERED,
    TABULAR_SVMLIGHT_SUFFIX_PREFIXES,
    TABULAR_TAB_DELIMITED_SUFFIX_PREFIXES,
    TABULAR_TEXT_SUFFIXES,
    tabular_data_row_count_capped,
)
from kagglebot.submission_service import SubmissionService
from kagglebot.test_table_aliases import (
    STRONG_TEST_TABLE_TOKENS,
    TEST_TABLE_COMPACT_ALIASES,
    TEST_TABLE_EXCLUDE_TOKENS,
    TEST_TABLE_STEMS,
    WEAK_TEST_TABLE_TOKENS,
)
from kagglebot.writeup import infer_code_competition_from_paths

SUBMISSION_KERNEL_TEMPLATE = """\
from __future__ import annotations

import base64
import bz2
import csv
import gzip
import io
import json
import lzma
import os
import re
import shutil
import sqlite3
import stat
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

# This kernel exists to satisfy notebook-only competitions: it emits a prepared
# submission artifact that is already validated locally by Kagglebot.
# Training metrics.json is preserved by the runner; this submit-only wrapper
# must not overwrite it with an unscored placeholder.
#
# NOTE: We still reference `/kaggle/input` to satisfy source validators and to
# make debugging easier in the Kaggle runtime.
KAGGLE_INPUT_ROOT = "/kaggle/input"
SUBMISSION_GZIP_B64 = __SUBMISSION_GZIP_B64__
SUBMISSION_INPUT_SUFFIX = __SUBMISSION_INPUT_SUFFIX__
SUBMISSION_OUTPUT_NAME = __SUBMISSION_OUTPUT_NAME__
EXTRACTED_INPUT_DIRNAME = "extracted_input"
ASSET_COMPRESSION_SUFFIXES = __ASSET_COMPRESSION_SUFFIXES__
TABULAR_OUTPUT_SUFFIXES = __TABULAR_OUTPUT_SUFFIXES__
TABULAR_INPUT_SUFFIXES = __TABULAR_INPUT_SUFFIXES__
TABULAR_TEXT_SUFFIXES = __TABULAR_TEXT_SUFFIXES__
TABULAR_STRUCTURED_SUFFIXES = __TABULAR_STRUCTURED_SUFFIXES__
TABULAR_PICKLE_SUFFIXES = __TABULAR_PICKLE_SUFFIXES__
SQLITE_TABULAR_SUFFIXES = __SQLITE_TABULAR_SUFFIXES__
DUCKDB_TABULAR_SUFFIXES = __DUCKDB_TABULAR_SUFFIXES__
TABULAR_ANNDATA_SUFFIXES = __TABULAR_ANNDATA_SUFFIXES__
TABULAR_ARROW_IPC_SUFFIXES = __TABULAR_ARROW_IPC_SUFFIXES__
TABULAR_PARQUET_SUFFIXES = __TABULAR_PARQUET_SUFFIXES__
TABULAR_EXCEL_INPUT_ONLY_SUFFIXES = __TABULAR_EXCEL_INPUT_ONLY_SUFFIXES__
TABULAR_EXCEL_SUFFIXES = __TABULAR_EXCEL_SUFFIXES__
TABULAR_FITS_SUFFIXES = __TABULAR_FITS_SUFFIXES__
TABULAR_GEOPACKAGE_SUFFIXES = __TABULAR_GEOPACKAGE_SUFFIXES__
TABULAR_HDF_SUFFIXES = __TABULAR_HDF_SUFFIXES__
TABULAR_JSON_LINES_SUFFIX_PREFIXES = __TABULAR_JSON_LINES_SUFFIX_PREFIXES__
TABULAR_KML_SUFFIXES = __TABULAR_KML_SUFFIXES__
TABULAR_LOOM_SUFFIXES = __TABULAR_LOOM_SUFFIXES__
TABULAR_STATA_SUFFIXES = __TABULAR_STATA_SUFFIXES__
TABULAR_SAS_SUFFIXES = __TABULAR_SAS_SUFFIXES__
TABULAR_SHAPEFILE_SUFFIXES = __TABULAR_SHAPEFILE_SUFFIXES__
TABULAR_SPSS_SUFFIXES = __TABULAR_SPSS_SUFFIXES__
TABULAR_MATLAB_SUFFIXES = __TABULAR_MATLAB_SUFFIXES__
TABULAR_RDATA_SUFFIXES = __TABULAR_RDATA_SUFFIXES__
TABULAR_NETCDF_SUFFIXES = __TABULAR_NETCDF_SUFFIXES__
TABULAR_NUMPY_SUFFIXES = __TABULAR_NUMPY_SUFFIXES__
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
TABULAR_ARFF_SUFFIXES = __TABULAR_ARFF_SUFFIXES__
TABULAR_HTML_SUFFIX_PREFIXES = __TABULAR_HTML_SUFFIX_PREFIXES__
TABULAR_SVMLIGHT_SUFFIX_PREFIXES = __TABULAR_SVMLIGHT_SUFFIX_PREFIXES__
TABULAR_FIXED_WIDTH_SUFFIX_PREFIXES = __TABULAR_FIXED_WIDTH_SUFFIX_PREFIXES__
TABULAR_TAB_DELIMITED_SUFFIX_PREFIXES = __TABULAR_TAB_DELIMITED_SUFFIX_PREFIXES__
ARCHIVE_OUTPUT_SUFFIXES = __ARCHIVE_OUTPUT_SUFFIXES__
NON_TABULAR_OUTPUT_SUFFIXES = __NON_TABULAR_OUTPUT_SUFFIXES__
EXTERNAL_ARCHIVE_OUTPUT_SUFFIXES = __EXTERNAL_ARCHIVE_OUTPUT_SUFFIXES__
ZSTD_TAR_ARCHIVE_SUFFIXES = __ZSTD_TAR_ARCHIVE_SUFFIXES__
SAMPLE_OUTPUT_NAME_TOKENS = set(__SAMPLE_OUTPUT_NAME_TOKENS__)
SAMPLE_COMPACT_NAME_ALIASES = set(__SAMPLE_COMPACT_NAME_ALIASES__)
TEST_TABLE_STEMS = __TEST_TABLE_STEMS__
TEST_TABLE_COMPACT_ALIASES = set(__TEST_TABLE_COMPACT_ALIASES__)
TEST_TABLE_EXCLUDE_TOKENS = set(__TEST_TABLE_EXCLUDE_TOKENS__)
STRONG_TEST_TABLE_TOKENS = set(__STRONG_TEST_TABLE_TOKENS__)
WEAK_TEST_TABLE_TOKENS = set(__WEAK_TEST_TABLE_TOKENS__)
ID_LIKE_COLUMN_NAMES = set(__ID_LIKE_COLUMN_NAMES__)


def _compression_suffix_for(suffix: str) -> str | None:
    normalized = str(suffix or "").strip().lower()
    for compression_suffix in ASSET_COMPRESSION_SUFFIXES:
        if normalized.endswith(compression_suffix):
            return compression_suffix
    return None


def _decompress_compressed_payload(payload: bytes, suffix: str) -> bytes:
    compression_suffix = _compression_suffix_for(suffix)
    if compression_suffix == ".gz":
        return gzip.decompress(payload)
    if compression_suffix == ".bz2":
        return bz2.decompress(payload)
    if compression_suffix == ".xz":
        return lzma.decompress(payload)
    if compression_suffix == ".zst":
        import zstandard as zstd

        with zstd.ZstdDecompressor().stream_reader(io.BytesIO(payload)) as reader:
            return reader.read()
    return payload


def _compress_payload_for_suffix(payload: bytes, suffix: str) -> bytes:
    compression_suffix = _compression_suffix_for(suffix)
    if compression_suffix == ".gz":
        return gzip.compress(payload, compresslevel=9)
    if compression_suffix == ".bz2":
        return bz2.compress(payload)
    if compression_suffix == ".xz":
        return lzma.compress(payload)
    if compression_suffix == ".zst":
        import zstandard as zstd

        return zstd.ZstdCompressor(level=9).compress(payload)
    return payload


def _open_compressed_text(path: Path, suffix: str):
    compression_suffix = _compression_suffix_for(suffix)
    if compression_suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    if compression_suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8", errors="ignore")
    if compression_suffix == ".xz":
        return lzma.open(path, "rt", encoding="utf-8", errors="ignore")
    if compression_suffix == ".zst":
        import zstandard as zstd

        return zstd.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("rt", encoding="utf-8", errors="ignore")


def _normalize_table_column_names(columns) -> list[str]:
    return _dedupe_table_column_names(
        [_stable_table_column_name(column, position) for position, column in enumerate(columns)]
    )


def _frame_with_normalized_table_columns(frame):
    normalized = _normalize_table_column_names(frame.columns)
    if list(frame.columns) == normalized:
        return frame
    copied = frame.copy()
    copied.columns = normalized
    return copied


def _stable_table_column_name(column, position: int) -> str:
    fallback = f"column_{position + 1}"
    if isinstance(column, tuple):
        parts = [
            str(part).strip()
            for part in column
            if not _table_column_part_is_missing(part)
            and not _table_column_name_is_generated_missing(str(part).strip())
        ]
        return "_".join(part for part in parts if part) or fallback
    if _table_column_part_is_missing(column):
        return fallback
    name = str(column)
    stripped = name.strip()
    if not stripped or _table_column_name_is_generated_missing(stripped):
        return fallback
    return name


def _table_column_name_is_generated_missing(name: str) -> bool:
    return bool(re.fullmatch(r"Unnamed:\\s*\\d+(?:_level_\\d+)?", str(name).strip(), flags=re.IGNORECASE))


def _table_column_part_is_missing(value) -> bool:
    if value is None:
        return True
    try:
        import pandas as pd

        missing = pd.isna(value)
    except Exception:
        return False
    if isinstance(missing, bool) or type(missing).__name__ == "bool_":
        return bool(missing)
    return False


def _dedupe_table_column_names(columns) -> list[str]:
    counts: dict[str, int] = {}
    deduped: list[str] = []
    for column in columns:
        base = str(column)
        count = counts.get(base, 0)
        deduped.append(base if count == 0 else f"{base}_{count}")
        counts[base] = count + 1
    return deduped


def _tabular_suffix(path: Path) -> str:
    name = path.name.lower()
    for suffix in sorted(TABULAR_INPUT_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix):
            return suffix
    return path.suffix.lower()


def _artifact_suffix(path: Path) -> str:
    name = path.name.lower()
    for suffix in sorted(
        ARCHIVE_OUTPUT_SUFFIXES + NON_TABULAR_OUTPUT_SUFFIXES + TABULAR_INPUT_SUFFIXES,
        key=len,
        reverse=True,
    ):
        if name.endswith(suffix):
            return suffix
    return path.suffix.lower()


def _remove_artifact_suffix(value: str, suffix: str) -> str:
    if suffix and value.lower().endswith(suffix):
        return value[: -len(suffix)]
    return str(Path(value).with_suffix(""))


def _is_archive_output() -> bool:
    name = str(SUBMISSION_OUTPUT_NAME or "").lower()
    suffix = str(SUBMISSION_INPUT_SUFFIX or "").lower()
    return suffix in ARCHIVE_OUTPUT_SUFFIXES or any(name.endswith(item) for item in ARCHIVE_OUTPUT_SUFFIXES)


def _is_tabular_output() -> bool:
    suffix = str(SUBMISSION_INPUT_SUFFIX or "").lower()
    return suffix in TABULAR_OUTPUT_SUFFIXES


def _working_root() -> Path:
    return Path(os.environ.get("KAGGLEBOT_WORKING_DIR", "/kaggle/working"))


def _input_root() -> Path:
    return Path(os.environ.get("KAGGLEBOT_INPUT_ROOT", KAGGLE_INPUT_ROOT))


def _extracted_input_root() -> Path:
    return _working_root() / EXTRACTED_INPUT_DIRNAME


def _data_roots() -> list[Path]:
    roots = [_input_root()]
    extracted = _extracted_input_root()
    if extracted.exists():
        roots.append(extracted)
    return roots


def _tabular_stem(path: Path) -> str:
    suffix = _tabular_suffix(path)
    if suffix and path.name.lower().endswith(suffix):
        return path.name[: -len(suffix)]
    return path.stem


def _normalized_stem(path: Path) -> str:
    return _tabular_stem(path).lower().replace("-", "_").replace(" ", "_")


def _test_table_name_score(path: Path) -> int:
    stem = _normalized_stem(path)
    compact = stem.replace("_", "")
    if stem in TEST_TABLE_STEMS or compact in TEST_TABLE_COMPACT_ALIASES:
        return 3
    tokens = {token for token in stem.replace(".", "_").split("_") if token}
    if TEST_TABLE_EXCLUDE_TOKENS & tokens:
        return 0
    if tokens & STRONG_TEST_TABLE_TOKENS:
        return 2
    if tokens & WEAK_TEST_TABLE_TOKENS:
        return 1
    return 0


def _sample_name_score(path: Path) -> int:
    stem = _normalized_stem(path)
    compact = stem.replace("_", "")
    tokens = {token for token in stem.replace(".", "_").split("_") if token}
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


def _resolve_kernel_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path(os.getcwd()).resolve()


def _candidate_sample_paths() -> list[Path]:
    slug = os.environ.get("KAGGLEBOT_COMPETITION_SLUG") or os.environ.get("KAGGLEBOT_SLUG") or ""
    slug_variants = [slug, slug.replace("-", "_")] if slug else []
    candidates: list[Path] = []
    for root in _data_roots():
        for item in slug_variants:
            if not item:
                continue
            for suffix in TABULAR_INPUT_SUFFIXES:
                candidates.extend(
                    [
                        root / item / f"sample_submission{suffix}",
                        root / "competitions" / item / f"sample_submission{suffix}",
                    ]
                )
        for suffix in TABULAR_INPUT_SUFFIXES:
            candidates.append(root / f"sample_submission{suffix}")
    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    for root in _data_roots():
        if root.exists():
            for candidate in sorted(root.rglob("*")):
                if (
                    candidate.is_file()
                    and _tabular_suffix(candidate) in TABULAR_INPUT_SUFFIXES
                    and _sample_name_score(candidate) >= 2
                    and candidate not in seen
                ):
                    ordered.append(candidate)
                    seen.add(candidate)
    return ordered


def _candidate_test_paths() -> list[Path]:
    slug = os.environ.get("KAGGLEBOT_COMPETITION_SLUG") or os.environ.get("KAGGLEBOT_SLUG") or ""
    slug_variants = [slug, slug.replace("-", "_")] if slug else []
    candidates: list[Path] = []
    for root in _data_roots():
        for item in slug_variants:
            if not item:
                continue
            for stem in TEST_TABLE_STEMS:
                for suffix in TABULAR_INPUT_SUFFIXES:
                    candidates.extend(
                        [
                            root / item / f"{stem}{suffix}",
                            root / "competitions" / item / f"{stem}{suffix}",
                        ]
                    )
        for stem in TEST_TABLE_STEMS:
            for suffix in TABULAR_INPUT_SUFFIXES:
                candidates.append(root / f"{stem}{suffix}")
    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    discovered: list[tuple[int, Path]] = []
    for root in _data_roots():
        if root.exists():
            for candidate in sorted(root.rglob("*")):
                if (
                    candidate.is_file()
                    and _tabular_suffix(candidate) in TABULAR_INPUT_SUFFIXES
                    and candidate not in seen
                ):
                    score = _test_table_name_score(candidate)
                    if score <= 0:
                        continue
                    discovered.append((score, candidate))
                    seen.add(candidate)
    ordered.extend(path for _, path in sorted(discovered, key=lambda item: (-item[0], str(item[1]))))
    return ordered


def _safe_extract_path(dest_dir: Path, member_name: str) -> Path:
    dest_dir = dest_dir.resolve()
    candidate = (dest_dir / member_name).resolve()
    try:
        candidate.relative_to(dest_dir)
    except ValueError as exc:
        raise ValueError(f"unsafe archive path: {member_name}") from exc
    return candidate


def _remember_archive_target(seen_targets: set[Path], target: Path, member_name: str) -> None:
    if target in seen_targets:
        raise ValueError(f"duplicate archive member target: {member_name}")
    seen_targets.add(target)


def _zip_member_is_symlink(member: zipfile.ZipInfo) -> bool:
    if member.create_system != 3:
        return False
    return stat.S_IFMT(member.external_attr >> 16) == stat.S_IFLNK


def _safe_extract_zip(zip_path: Path, dest_dir: Path, *, overwrite: bool = False) -> list[Path]:
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        seen_targets: set[Path] = set()
        for member in archive.infolist():
            target = _safe_extract_path(dest_dir, member.filename)
            _remember_archive_target(seen_targets, target, member.filename)
            if member.flag_bits & 0x1:
                raise ValueError(f"unsupported encrypted zip member: {member.filename}")
            if _zip_member_is_symlink(member):
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


def _safe_extract_tar(tar_path: Path, dest_dir: Path, *, overwrite: bool = False) -> list[Path]:
    with tarfile.open(tar_path, "r:*") as archive:
        return _safe_extract_tar_members(archive, dest_dir, overwrite=overwrite)


def _safe_extract_tar_zst(tar_path: Path, dest_dir: Path, *, overwrite: bool = False) -> list[Path]:
    import zstandard as zstd

    with tar_path.open("rb") as raw:
        with zstd.ZstdDecompressor().stream_reader(raw) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as archive:
                return _safe_extract_tar_members(archive, dest_dir, overwrite=overwrite)


def _safe_extract_tar_members(archive, dest_dir: Path, *, overwrite: bool = False) -> list[Path]:
    extracted: list[Path] = []
    seen_targets: set[Path] = set()
    for member in archive:
        target = _safe_extract_path(dest_dir, member.name)
        _remember_archive_target(seen_targets, target, member.name)
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


def _safe_extract_7z(archive_path: Path, dest_dir: Path, *, overwrite: bool = False) -> list[Path]:
    try:
        import py7zr
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot extract {archive_path.name}: py7zr is not available in this Kaggle runtime."
        ) from exc
    extracted: list[Path] = []
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        targets: list[str] = []
        seen_targets: set[Path] = set()
        for member in archive.list():
            name = getattr(member, "filename", "")
            target = _safe_extract_path(dest_dir, name)
            _remember_archive_target(seen_targets, target, name)
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


def _safe_extract_rar(archive_path: Path, dest_dir: Path, *, overwrite: bool = False) -> list[Path]:
    try:
        import rarfile
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot extract {archive_path.name}: rarfile is not available in this Kaggle runtime."
        ) from exc
    extracted: list[Path] = []
    with rarfile.RarFile(archive_path) as archive:
        seen_targets: set[Path] = set()
        for member in archive.infolist():
            member_name = getattr(member, "filename", "")
            target = _safe_extract_path(dest_dir, member_name)
            _remember_archive_target(seen_targets, target, member_name)
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


def _is_supported_input_archive(path: Path) -> bool:
    return _archive_output_suffix(path) in ARCHIVE_OUTPUT_SUFFIXES


def _archive_output_suffix(path: Path) -> str:
    name = path.name.lower()
    for suffix in sorted(ARCHIVE_OUTPUT_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix):
            return suffix
    return ""


def _is_zstd_tar_archive(path: Path) -> bool:
    return _archive_output_suffix(path) in ZSTD_TAR_ARCHIVE_SUFFIXES


def _extract_input_archives(max_depth: int = 2) -> None:
    dest = _extracted_input_root()
    dest.mkdir(parents=True, exist_ok=True)
    processed: set[Path] = set()
    extracted_count = 0
    for _depth in range(max(max_depth, 0) + 1):
        archives = [
            path
            for root in _data_roots()
            for path in sorted(root.rglob("*"))
            if path.is_file() and _is_supported_input_archive(path) and path.resolve() not in processed
        ]
        if not archives:
            break
        for archive_path in archives:
            processed.add(archive_path.resolve())
            suffix = _archive_output_suffix(archive_path)
            if suffix == ".zip":
                extracted = _safe_extract_zip(archive_path, dest, overwrite=False)
            elif suffix == ".7z":
                extracted = _safe_extract_7z(archive_path, dest, overwrite=False)
            elif suffix == ".rar":
                extracted = _safe_extract_rar(archive_path, dest, overwrite=False)
            elif suffix in ZSTD_TAR_ARCHIVE_SUFFIXES:
                extracted = _safe_extract_tar_zst(archive_path, dest, overwrite=False)
            else:
                extracted = _safe_extract_tar(archive_path, dest, overwrite=False)
            extracted_count += len(extracted)
    if extracted_count:
        print(f"Runtime extracted input archives: {extracted_count} files")


def _find_sample_submission() -> Path | None:
    for candidate in _candidate_sample_paths():
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _find_test_file() -> Path | None:
    for candidate in _candidate_test_paths():
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _sniff_delimiter(path: Path, *, default: str = ",") -> str:
    try:
        suffix = _tabular_suffix(path)
        if suffix.endswith(".zip"):
            return _sniff_payload_delimiter(_decompress_submission_payload(path.read_bytes(), suffix), default=default)
        with _open_compressed_text(path, suffix) as handle:
            line = next((candidate for candidate in handle if candidate.strip()), "")
    except OSError:
        return default
    if not line:
        return default
    counts = {sep: line.count(sep) for sep in (default, ",", "\t", ";", "|")}
    best = max(counts, key=lambda sep: counts[sep])
    if counts[best] > 0:
        return best
    if _looks_space_delimited(line):
        return r"\\s+"
    return default


def _sniff_payload_delimiter(payload: bytes, *, default: str = ",") -> str:
    try:
        text = payload[:4096].decode("utf-8", errors="ignore")
    except Exception:
        return default
    line = next((candidate for candidate in text.splitlines() if candidate.strip()), "")
    if not line:
        return default
    counts = {sep: line.count(sep) for sep in (default, ",", "\t", ";", "|")}
    best = max(counts, key=lambda sep: counts[sep])
    if counts[best] > 0:
        return best
    if _looks_space_delimited(line):
        return r"\\s+"
    return default


def _looks_space_delimited(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if any(separator in stripped for separator in (",", "\t", ";", "|")):
        return False
    parts = stripped.split()
    return len(parts) >= 2 and all(part.strip() for part in parts)


def _read_tabular_path(path: Path):
    return _frame_with_normalized_table_columns(_read_tabular_path_raw(path))


def _read_tabular_path_raw(path: Path):
    import pandas as pd

    suffix = _tabular_suffix(path)
    base_suffix = _tabular_payload_base_suffix(suffix)
    if suffix in SQLITE_TABULAR_SUFFIXES:
        return _read_sqlite_table(path)
    if suffix in DUCKDB_TABULAR_SUFFIXES:
        return _read_duckdb_table(path)
    if base_suffix in TABULAR_PARQUET_SUFFIXES:
        return pd.read_parquet(_binary_tabular_path_source(path, suffix))
    if base_suffix == ".orc":
        return pd.read_orc(_binary_tabular_path_source(path, suffix))
    if suffix in TABULAR_HDF_SUFFIXES:
        return _read_hdf_table(path)
    if suffix in TABULAR_ANNDATA_SUFFIXES:
        return _read_h5ad_tabular_frame(path)
    if suffix in TABULAR_LOOM_SUFFIXES:
        return _read_loom_tabular_frame(path)
    if suffix in TABULAR_GEOPACKAGE_SUFFIXES:
        return _read_geopackage_tabular_frame(path)
    if suffix in TABULAR_SHAPEFILE_SUFFIXES:
        return _read_shapefile_tabular_frame(path)
    if suffix in TABULAR_KML_SUFFIXES:
        return _read_kml_tabular_frame(path)
    if base_suffix in TABULAR_ARROW_IPC_SUFFIXES:
        return pd.read_feather(_binary_tabular_path_source(path, suffix))
    if base_suffix == ".avro":
        if suffix.endswith(".zip"):
            return _read_avro_payload(_decompress_submission_payload(path.read_bytes(), suffix))
        return _read_avro_table(path)
    if suffix in TABULAR_EXCEL_SUFFIXES:
        return pd.read_excel(path)
    if suffix in TABULAR_EXCEL_INPUT_ONLY_SUFFIXES:
        return pd.read_excel(path, engine="pyxlsb")
    if suffix in TABULAR_STATA_SUFFIXES:
        return pd.read_stata(path)
    if base_suffix in TABULAR_SAS_SUFFIXES:
        return pd.read_sas(path, format=_sas_format_for_suffix(suffix))
    if base_suffix in TABULAR_SPSS_SUFFIXES:
        return pd.read_spss(path)
    if base_suffix in TABULAR_MATLAB_SUFFIXES:
        return _read_mat_tabular_frame(path)
    if base_suffix in TABULAR_RDATA_SUFFIXES:
        return _read_rdata_tabular_frame(path)
    if base_suffix in TABULAR_NETCDF_SUFFIXES:
        return _read_netcdf_tabular_frame(path)
    if base_suffix in TABULAR_NUMPY_SUFFIXES:
        return _read_numpy_tabular_frame(path)
    if base_suffix in TABULAR_FITS_SUFFIXES:
        return _read_fits_tabular_frame(path)
    if base_suffix in TABULAR_ARFF_SUFFIXES:
        return _read_arff_tabular_frame(path)
    if suffix.startswith(TABULAR_HTML_SUFFIX_PREFIXES):
        return _read_html_tabular_frame(path)
    if base_suffix.startswith(TABULAR_SVMLIGHT_SUFFIX_PREFIXES):
        return _read_svmlight_tabular_frame(path)
    if base_suffix.startswith(TABULAR_FIXED_WIDTH_SUFFIX_PREFIXES):
        return _read_fixed_width_tabular_frame(path)
    if suffix.startswith(".xml"):
        payload = _decompress_submission_payload(path.read_bytes(), suffix)
        return pd.read_xml(io.BytesIO(payload), parser="etree")
    if base_suffix in TABULAR_PICKLE_SUFFIXES:
        return pd.read_pickle(_binary_tabular_path_source(path, suffix))
    if _is_json_lines_suffix(suffix):
        payload = _decompress_submission_payload(path.read_bytes(), suffix)
        return pd.read_json(io.BytesIO(payload), lines=True)
    if suffix in TABULAR_STRUCTURED_SUFFIXES:
        if _is_yaml_suffix(suffix):
            payload = _decompress_submission_payload(path.read_bytes(), suffix)
            return _yaml_payload_to_frame(payload)
        return _read_json_table_path(path)
    if suffix in TABULAR_TEXT_SUFFIXES:
        default = _default_delimited_text_separator(suffix)
        payload = _decompress_submission_payload(path.read_bytes(), suffix)
        return pd.read_csv(io.BytesIO(payload), sep=_sniff_payload_delimiter(payload, default=default))
    return pd.read_csv(path)


def _read_hdf_table(path: Path):
    import pandas as pd

    try:
        return pd.read_hdf(path)
    except (KeyError, ValueError) as exc:
        _close_pytables_open_files()
        try:
            with pd.HDFStore(path, mode="r") as store:
                keys = store.keys()
            if len(keys) == 1:
                return pd.read_hdf(path, key=keys[0])
        except Exception:
            pass
        fallback = _read_native_hdf_table(path)
        if fallback is not None:
            return fallback
        raise exc
    except Exception as exc:
        _close_pytables_open_files()
        fallback = _read_native_hdf_table(path)
        if fallback is not None:
            return fallback
        raise exc


def _close_pytables_open_files() -> None:
    try:
        import tables

        for handle in list(tables.file._open_files._handlers):
            handle.close()
    except Exception:
        pass


def _read_native_hdf_table(path: Path):
    import h5py

    candidates = []
    with h5py.File(path, "r") as handle:
        root_frame = _hdf_group_to_column_frame(handle)
        if root_frame is not None:
            candidates.append(root_frame)
        for name, node in _iter_hdf_nodes(handle):
            frame = _hdf_node_to_frame(name, node)
            if frame is not None:
                candidates.append(frame)
        for _name, node in _iter_hdf_nodes(handle):
            if not isinstance(node, h5py.Group):
                continue
            frame = _hdf_group_to_column_frame(node)
            if frame is not None:
                candidates.append(frame)
    if not candidates:
        return None
    return max(candidates, key=lambda frame: (len(frame) * max(len(frame.columns), 1), len(frame.columns)))


def _iter_hdf_nodes(group):
    for key, node in group.items():
        name = str(node.name or key).strip("/") or str(key)
        yield name, node
        if hasattr(node, "items"):
            yield from _iter_hdf_nodes(node)


def _hdf_node_to_frame(name: str, node):
    import h5py
    import numpy as np
    import pandas as pd

    if not isinstance(node, h5py.Dataset) or node.shape is None or 0 in node.shape:
        return None
    data = node[()]
    if getattr(data.dtype, "names", None):
        return pd.DataFrame.from_records(data)
    data = _decode_hdf_array(data)
    if data.ndim == 1:
        return pd.DataFrame({_safe_hdf_column_name(name): data.tolist()})
    if data.ndim != 2:
        return None
    columns = _hdf_dataset_column_names(node, name, data.shape[1])
    return pd.DataFrame(np.asarray(data), columns=columns)


def _hdf_group_to_column_frame(group):
    import h5py
    import pandas as pd

    columns = {}
    row_count = None
    for key, node in group.items():
        values = _hdf_column_values(node, h5py=h5py)
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


def _hdf_column_values(node, *, h5py):
    import numpy as np

    if isinstance(node, h5py.Dataset):
        if node.shape is None or len(node.shape) != 1 or 0 in node.shape:
            return None
        return _decode_hdf_array(node[()])
    if not isinstance(node, h5py.Group):
        return None
    if "codes" not in node or "categories" not in node:
        return None
    codes = _decode_hdf_array(node["codes"][()])
    categories = _decode_hdf_array(node["categories"][()])
    if codes.ndim != 1 or categories.ndim != 1:
        return None
    values = []
    for code in codes.tolist():
        index = int(code)
        values.append(categories[index] if 0 <= index < len(categories) else None)
    return np.asarray(values, dtype=object)


def _decode_hdf_array(data):
    import numpy as np

    array = np.asarray(data)
    if array.dtype.kind == "S":
        return array.astype(str)
    if array.dtype.kind == "O":
        return np.vectorize(_decode_hdf_value, otypes=[object])(array)
    return array


def _decode_hdf_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _hdf_dataset_column_names(dataset, name: str, width: int) -> list[str]:
    for attr_name in ("columns", "column_names", "feature_names"):
        raw_columns = dataset.attrs.get(attr_name)
        columns = _decode_hdf_columns(raw_columns, width)
        if columns is not None:
            return columns
    safe_name = _safe_hdf_column_name(name)
    if width == 1:
        return [safe_name]
    return [f"{safe_name}_{idx}" for idx in range(width)]


def _decode_hdf_columns(raw_columns, width: int) -> list[str] | None:
    if raw_columns is None:
        return None
    values = list(raw_columns) if not isinstance(raw_columns, (str, bytes)) else [raw_columns]
    columns = [str(_decode_hdf_value(value)) for value in values]
    return columns if len(columns) == width else None


def _safe_hdf_column_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name).strip("/")).strip("_") or "value"


def _read_h5ad_tabular_frame(path: Path):
    import h5py
    import pandas as pd

    with h5py.File(path, "r") as handle:
        obs_frame = _hdf_group_to_column_frame(handle["obs"]) if "obs" in handle else None
        x_frame = _h5ad_x_to_frame(handle)
    frame = _combine_same_row_frames(obs_frame, x_frame, pd=pd)
    if frame is not None:
        return frame
    fallback = _read_native_hdf_table(path)
    if fallback is not None:
        return fallback
    raise ValueError(f"AnnData file does not contain table-like obs/X data: {path}")


def _h5ad_x_to_frame(handle):
    import numpy as np
    import pandas as pd

    if "X" not in handle:
        return None
    raw_matrix = _h5ad_x_to_array(handle["X"])
    if raw_matrix is None:
        return None
    matrix = np.asarray(raw_matrix)
    if matrix.ndim != 2 or 0 in matrix.shape:
        return None
    columns = _h5ad_var_names(handle, matrix.shape[1]) or [f"x_{idx}" for idx in range(matrix.shape[1])]
    return pd.DataFrame(matrix, columns=columns)


def _h5ad_x_to_array(node):
    import h5py
    import numpy as np
    from scipy import sparse

    if isinstance(node, h5py.Dataset):
        return _decode_hdf_array(node[()])
    if not isinstance(node, h5py.Group) or not {"data", "indices", "indptr", "shape"} <= set(node.keys()):
        return None
    data = node["data"][()]
    indices = node["indices"][()]
    indptr = node["indptr"][()]
    shape = tuple(int(value) for value in np.asarray(node["shape"][()]).tolist())
    encoding = str(_decode_hdf_value(node.attrs.get("encoding-type", b"csr_matrix"))).lower()
    matrix = sparse.csc_matrix((data, indices, indptr), shape=shape) if "csc" in encoding else sparse.csr_matrix(
        (data, indices, indptr),
        shape=shape,
    )
    return matrix.toarray()


def _h5ad_var_names(handle, width: int) -> list[str] | None:
    import h5py

    var = handle.get("var")
    if var is None:
        return None
    for key in ("_index", "index", "gene_symbols", "gene_ids", "feature_name", "features"):
        if key not in var:
            continue
        values = _hdf_column_values(var[key], h5py=h5py)
        columns = _normalize_hdf_feature_names(values, width)
        if columns is not None:
            return columns
    return None


def _read_loom_tabular_frame(path: Path):
    import h5py
    import numpy as np
    import pandas as pd

    with h5py.File(path, "r") as handle:
        attrs_frame = _hdf_group_to_column_frame(handle["col_attrs"]) if "col_attrs" in handle else None
        matrix = _decode_hdf_array(handle["matrix"][()]) if "matrix" in handle else None
        if matrix is None or np.asarray(matrix).ndim != 2 or 0 in np.asarray(matrix).shape:
            matrix_frame = None
        else:
            matrix = np.asarray(matrix).T
            columns = _loom_row_feature_names(handle, matrix.shape[1]) or [f"x_{idx}" for idx in range(matrix.shape[1])]
            matrix_frame = pd.DataFrame(matrix, columns=columns)
    frame = _combine_same_row_frames(attrs_frame, matrix_frame, pd=pd)
    if frame is not None:
        return frame
    fallback = _read_native_hdf_table(path)
    if fallback is not None:
        return fallback
    raise ValueError(f"Loom file does not contain table-like col_attrs/matrix data: {path}")


def _loom_row_feature_names(handle, width: int) -> list[str] | None:
    import h5py

    row_attrs = handle.get("row_attrs")
    if row_attrs is None:
        return None
    for key in ("Gene", "gene", "GeneName", "gene_name", "Accession", "feature_name"):
        if key not in row_attrs:
            continue
        values = _hdf_column_values(row_attrs[key], h5py=h5py)
        columns = _normalize_hdf_feature_names(values, width)
        if columns is not None:
            return columns
    return None


def _normalize_hdf_feature_names(values, width: int) -> list[str] | None:
    if values is None:
        return None
    names = [_safe_hdf_column_name(str(value)) for value in values.tolist()]
    if len(names) != width or any(not name for name in names):
        return None
    if len(set(names)) != len(names):
        return None
    return names


def _combine_same_row_frames(left, right, *, pd):
    frames = [frame for frame in (left, right) if frame is not None and not frame.empty]
    if not frames:
        return None
    row_count = len(frames[0])
    if any(len(frame) != row_count for frame in frames):
        return max(frames, key=lambda frame: (len(frame) * max(len(frame.columns), 1), len(frame.columns)))
    combined = pd.concat([frame.reset_index(drop=True) for frame in frames], axis=1)
    combined.columns = _dedupe_column_names([str(column) for column in combined.columns])
    return combined


def _dedupe_column_names(columns: list[str]) -> list[str]:
    counts = {}
    deduped = []
    for column in columns:
        base = str(column)
        count = counts.get(base, 0)
        deduped.append(base if count == 0 else f"{base}_{count}")
        counts[base] = count + 1
    return deduped


def _read_avro_table(path: Path):
    import pandas as pd
    from fastavro import reader

    with path.open("rb") as handle:
        avro_reader = reader(handle)
        schema = getattr(avro_reader, "writer_schema", {}) or {}
        records = list(avro_reader)
    columns = [str(field.get("name")) for field in schema.get("fields", []) if field.get("name") is not None]
    return pd.DataFrame(records, columns=columns or None)


def _binary_tabular_path_source(path: Path, suffix: str):
    if suffix.endswith(".zip"):
        return io.BytesIO(_decompress_submission_payload(path.read_bytes(), suffix))
    return path


def _read_avro_payload(payload: bytes):
    import pandas as pd
    from fastavro import reader

    avro_reader = reader(io.BytesIO(payload))
    schema = getattr(avro_reader, "writer_schema", {}) or {}
    records = list(avro_reader)
    columns = [str(field.get("name")) for field in schema.get("fields", []) if field.get("name") is not None]
    return pd.DataFrame(records, columns=columns or None)


def _read_json_table_path(path: Path):
    import pandas as pd

    suffix = _tabular_suffix(path)
    base_suffix = _uncompressed_submission_suffix(suffix)
    payload = _decompress_submission_payload(path.read_bytes(), suffix)
    if _is_json_lines_suffix(base_suffix):
        return pd.read_json(io.BytesIO(payload), lines=True)
    try:
        return _json_payload_to_frame(payload)
    except ValueError:
        return pd.read_json(io.BytesIO(payload))


def _json_payload_to_frame(payload: bytes):
    text = payload.decode("utf-8-sig")
    return _json_object_to_frame(json.loads(text))


def _yaml_payload_to_frame(payload: bytes):
    import yaml

    return _json_object_to_frame(yaml.safe_load(payload.decode("utf-8-sig")) or [])


def _json_object_to_frame(value):
    import pandas as pd

    if isinstance(value, list):
        return pd.DataFrame(value)
    if isinstance(value, dict):
        if {"columns", "data"} <= set(value) and isinstance(value.get("columns"), list):
            return pd.DataFrame(value.get("data", []), columns=value["columns"])
        geojson_frame = _geojson_feature_collection_to_frame(value)
        if geojson_frame is not None:
            return geojson_frame
        for key in (
            "data",
            "records",
            "rows",
            "items",
            "results",
            "predictions",
            "submission",
        ):
            nested = value.get(key)
            if isinstance(nested, (list, dict)):
                return _json_object_to_frame(nested)
        try:
            return pd.DataFrame(value)
        except ValueError:
            return pd.DataFrame([value])
    raise ValueError("JSON payload does not contain tabular records.")


def _geojson_feature_collection_to_frame(value):
    import pandas as pd

    if str(value.get("type", "")).lower() != "featurecollection":
        return None
    features = value.get("features")
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


def _read_mat_tabular_frame(path: Path):
    import numpy as np
    import pandas as pd
    from scipy.io import loadmat

    payload = {
        key: value
        for key, value in loadmat(path, squeeze_me=True, struct_as_record=False).items()
        if not key.startswith("__")
    }
    frame = _mat_variables_to_column_frame(payload, np=np, pd=pd)
    if frame is not None:
        return frame
    for variable_name, value in payload.items():
        frame = _mat_value_to_frame(variable_name=variable_name, value=value, np=np, pd=pd)
        if frame is not None:
            return frame
    raise ValueError(f"MATLAB file does not contain table-like arrays: {path}")


def _mat_variables_to_column_frame(payload, *, np, pd):
    columns_by_length = {}
    for name, value in payload.items():
        values = _mat_value_to_series(value, np=np)
        if values is None:
            continue
        columns_by_length.setdefault(len(values), {})[str(name)] = values
    if not columns_by_length:
        return None
    columns = max(columns_by_length.values(), key=len)
    return pd.DataFrame(columns)


def _mat_value_to_frame(*, variable_name: str, value, np, pd):
    if hasattr(value, "dtype") and getattr(value.dtype, "names", None):
        columns = {}
        for field_name in value.dtype.names:
            values = _mat_value_to_series(value[field_name], np=np)
            if values is not None:
                columns[str(field_name)] = values
        return pd.DataFrame(columns) if columns else None
    values = _mat_value_to_series(value, np=np)
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


def _mat_value_to_series(value, *, np):
    array = np.asarray(value)
    if array.dtype.kind == "O":
        return None
    array = np.squeeze(array)
    if array.ndim != 1 or array.size == 0:
        return None
    return array.tolist()


def _read_rdata_tabular_frame(path: Path):
    import pyreadr

    result = pyreadr.read_r(path)
    for value in result.values():
        if hasattr(value, "columns") and hasattr(value, "head"):
            return value
    raise ValueError(f"R data file does not contain a tabular object: {path}")


def _read_netcdf_tabular_frame(path: Path):
    import numpy as np
    import pandas as pd
    from scipy.io import netcdf_file

    variables = {}
    try:
        with netcdf_file(path, mode="r", mmap=False) as dataset:
            for name, variable in dataset.variables.items():
                variables[str(name)] = np.array(variable.data).copy()
    except Exception as exc:
        fallback = _read_native_hdf_table(path)
        if fallback is not None:
            return fallback
        raise exc
    frame = _netcdf_variables_to_column_frame(variables, np=np, pd=pd)
    if frame is None:
        fallback = _read_native_hdf_table(path)
        if fallback is not None:
            return fallback
        raise ValueError(f"NetCDF file does not contain table-like variables: {path}")
    return frame


def _netcdf_variables_to_column_frame(payload, *, np, pd):
    columns_by_length = {}
    for name, value in payload.items():
        columns = _netcdf_value_to_columns(str(name), value, np=np)
        if not columns:
            continue
        row_count = len(next(iter(columns.values())))
        columns_by_length.setdefault(row_count, {}).update(columns)
    if not columns_by_length:
        return None
    columns = max(columns_by_length.values(), key=len)
    return pd.DataFrame(columns)


def _netcdf_value_to_columns(name: str, value, *, np):
    array = np.asarray(value)
    if array.shape == () or 0 in array.shape or array.dtype.kind == "O":
        return None
    if array.dtype.kind == "S":
        decoded = _netcdf_decode_bytes_array(array, np=np)
        return {_safe_netcdf_column_name(name): decoded} if decoded is not None else None
    if array.ndim > 1 and array.shape[-1] == 1:
        array = np.squeeze(array, axis=-1)
    if array.ndim == 1:
        return {_safe_netcdf_column_name(name): array.tolist()}
    if array.ndim == 2:
        base = _safe_netcdf_column_name(name)
        return {f"{base}_{idx}": array[:, idx].tolist() for idx in range(array.shape[1])}
    return None


def _netcdf_decode_bytes_array(array, *, np):
    if array.ndim == 1:
        return [_netcdf_decode_bytes(value) for value in array.tolist()]
    if array.ndim == 2:
        values = []
        for row in array:
            flattened = np.asarray(row).ravel().tolist()
            values.append(_netcdf_decode_bytes(b"".join(flattened)))
        return values
    return None


def _netcdf_decode_bytes(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\\x00 ")
    return str(value).rstrip("\\x00 ")


def _safe_netcdf_column_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_") or "value"


def _read_fits_tabular_frame(path: Path):
    import numpy as np
    import pandas as pd
    from astropy.io import fits

    candidates = []
    with fits.open(path, memmap=False) as hdus:
        for index, hdu in enumerate(hdus):
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            name = str(getattr(hdu, "name", "") or f"hdu_{index}").lower()
            frame = _fits_data_to_frame(name, data, np=np, pd=pd)
            if frame is not None:
                candidates.append(frame)
    if not candidates:
        raise ValueError(f"FITS file does not contain table-like HDUs: {path}")
    return max(candidates, key=lambda frame: (len(frame) * max(len(frame.columns), 1), len(frame.columns)))


def _fits_data_to_frame(name: str, data, *, np, pd):
    array = np.asarray(data)
    if array.shape == () or 0 in array.shape:
        return None
    if getattr(array.dtype, "names", None):
        columns = {}
        for field_name in array.dtype.names or ():
            field_columns = _fits_array_to_columns(str(field_name), array[field_name], np=np)
            if field_columns:
                columns.update(field_columns)
        return pd.DataFrame(columns) if columns else None
    columns = _fits_array_to_columns(name, array, np=np)
    return pd.DataFrame(columns) if columns else None


def _fits_array_to_columns(name: str, values, *, np):
    array = np.asarray(values)
    if array.shape == () or 0 in array.shape or array.dtype.kind == "O":
        return None
    if array.dtype.kind == "S":
        decoded = _fits_decode_bytes_array(array, np=np)
        return {_safe_fits_column_name(name): decoded} if decoded is not None else None
    if array.ndim > 1 and array.shape[-1] == 1:
        array = np.squeeze(array, axis=-1)
    if array.ndim == 1:
        return {_safe_fits_column_name(name): array.tolist()}
    if array.ndim == 2:
        base = _safe_fits_column_name(name)
        return {f"{base}_{idx}": array[:, idx].tolist() for idx in range(array.shape[1])}
    return None


def _fits_decode_bytes_array(array, *, np):
    if array.ndim == 1:
        return [_fits_decode_value(value) for value in array.tolist()]
    if array.ndim == 2:
        values = []
        for row in array:
            flattened = np.asarray(row).ravel().tolist()
            values.append(_fits_decode_value(b"".join(flattened)))
        return values
    return None


def _fits_decode_value(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\\x00 ")
    return str(value).rstrip("\\x00 ")


def _safe_fits_column_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_") or "value"


def _read_numpy_tabular_frame(path: Path):
    import numpy as np

    if path.suffix.lower() == ".npy":
        return _numpy_array_to_frame(path.stem, np.load(path, allow_pickle=False), source_path=path)
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.files)
        if not keys:
            raise ValueError(f"NumPy archive does not contain arrays: {path}")
        if len(keys) == 1:
            return _numpy_array_to_frame(keys[0], archive[keys[0]], source_path=path)
        column_frame = _numpy_archive_columns_to_frame({key: archive[key] for key in keys})
        if column_frame is not None:
            return column_frame
        candidates = []
        for key in keys:
            try:
                candidates.append(_numpy_array_to_frame(key, archive[key], source_path=path))
            except ValueError:
                continue
        if candidates:
            return max(candidates, key=lambda frame: (len(frame) * max(len(frame.columns), 1), len(frame.columns)))
    raise ValueError(f"NumPy file does not contain a 1D, 2D, or structured tabular array: {path}")


def _numpy_archive_columns_to_frame(arrays):
    import numpy as np
    import pandas as pd

    columns = {}
    row_count = None
    for key, raw_array in arrays.items():
        array = np.asarray(raw_array)
        if array.ndim != 1 or array.size == 0 or getattr(array.dtype, "names", None):
            continue
        array = _decode_numpy_array(array)
        if row_count is None:
            row_count = len(array)
        if len(array) != row_count:
            continue
        columns[str(key)] = array.tolist()
    if not columns:
        return None
    return pd.DataFrame(columns)


def _numpy_array_to_frame(name: str, raw_array, *, source_path: Path | None = None):
    import numpy as np
    import pandas as pd

    array = np.asarray(raw_array)
    if array.shape == () or 0 in array.shape:
        raise ValueError("NumPy array is scalar or empty.")
    if getattr(array.dtype, "names", None):
        return pd.DataFrame.from_records(_decode_structured_numpy_array(array))
    array = _decode_numpy_array(array)
    if array.ndim == 1:
        sidecar_columns = _numpy_column_names_for_path(source_path, 1) if source_path is not None else None
        column = sidecar_columns[0] if sidecar_columns else _safe_numpy_column_name(name)
        return pd.DataFrame({column: array.tolist()})
    if array.ndim != 2:
        raise ValueError(f"NumPy array must be 1D or 2D for tabular loading, got shape {array.shape}.")
    columns = _numpy_column_names_for_path(source_path, array.shape[1]) if source_path is not None else None
    if columns is None:
        base = _safe_numpy_column_name(name)
        columns = [base] if array.shape[1] == 1 else [f"{base}_{idx}" for idx in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)


def _numpy_column_names_for_path(path: Path | None, width: int) -> list[str] | None:
    if path is None or width <= 0:
        return None
    for candidate in _numpy_column_sidecar_candidates(path):
        try:
            raw_columns = _parse_numpy_column_sidecar(candidate)
        except (OSError, ValueError, json.JSONDecodeError, UnicodeError):
            continue
        columns = _normalize_numpy_column_names(raw_columns, width)
        if columns is not None:
            return columns
    return None


def _numpy_column_sidecar_candidates(path: Path) -> list[Path]:
    base = _remove_artifact_suffix(path.name, _artifact_suffix(path))
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


def _parse_numpy_column_sidecar(path: Path) -> list[object]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _extract_numpy_column_names(payload)
    return _parse_numpy_column_text(path.read_text(encoding="utf-8"))


def _parse_numpy_column_text(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) == 1:
        line = lines[0]
        for delimiter in ("\t", ",", ";"):
            if delimiter in line:
                return [token.strip().strip("\\\"'") for token in next(csv.reader([line], delimiter=delimiter))]
    return [line.strip().strip("\\\"'") for line in lines]


def _extract_numpy_column_names(payload: object) -> list[object]:
    if isinstance(payload, list):
        return _extract_numpy_column_list(payload)
    if not isinstance(payload, dict):
        raise ValueError("NumPy column sidecar JSON must be a list or object.")
    for key in ("columns", "feature_names", "features"):
        value = payload.get(key)
        if isinstance(value, list):
            return _extract_numpy_column_list(value)
    schema = payload.get("schema")
    if isinstance(schema, dict) and isinstance(schema.get("fields"), list):
        return _extract_numpy_column_list(schema["fields"])
    fields = payload.get("fields")
    if isinstance(fields, list):
        return _extract_numpy_column_list(fields)
    raise ValueError("NumPy column sidecar JSON does not contain column names.")


def _extract_numpy_column_list(values: list[object]) -> list[object]:
    columns = []
    for value in values:
        if isinstance(value, dict):
            if "name" not in value:
                raise ValueError("NumPy column field object is missing a name.")
            columns.append(value["name"])
        else:
            columns.append(value)
    return columns


def _normalize_numpy_column_names(columns: list[object], width: int) -> list[str] | None:
    names = [str(column).strip().strip("\\\"'") for column in columns]
    if len(names) != width or any(not name for name in names):
        return None
    if len(set(names)) != len(names):
        return None
    return names


def _decode_structured_numpy_array(array):
    records = []
    for row in array:
        record = {}
        for field_name in array.dtype.names or ():
            record[str(field_name)] = _decode_numpy_scalar(row[field_name])
        records.append(record)
    return records


def _decode_numpy_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    try:
        return value.item()
    except AttributeError:
        return value


def _decode_numpy_array(array):
    import numpy as np

    decoded = np.asarray(array)
    if decoded.dtype.kind == "S":
        return decoded.astype(str)
    return decoded


def _safe_numpy_column_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_") or "value"


def _read_arff_tabular_frame(path: Path):
    import pandas as pd
    from scipy.io import arff

    payload = _decompress_submission_payload(path.read_bytes(), _tabular_suffix(path))
    data, _metadata = arff.loadarff(io.StringIO(payload.decode("utf-8", errors="ignore")))
    frame = pd.DataFrame(data)
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(_decode_arff_value)
    return frame


def _decode_arff_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _read_html_tabular_frame(path: Path):
    import pandas as pd

    suffix = _tabular_suffix(path)
    payload = _decompress_submission_payload(path.read_bytes(), suffix)
    tables = pd.read_html(io.StringIO(payload.decode("utf-8", errors="ignore")))
    candidates = [table for table in tables if table.shape[1] > 0]
    if not candidates:
        raise ValueError(f"HTML file does not contain table-like data: {path}")
    return max(candidates, key=lambda table: (table.shape[0] * table.shape[1], table.shape[1]))


def _path_mentions_role(path: Path, role: str) -> bool:
    text = path.stem.lower()
    tokens = [token for token in text.replace("-", "_").split("_") if token]
    return role.lower() in tokens or text == role.lower()


def _read_svmlight_tabular_frame(path: Path):
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_svmlight_file

    suffix = _tabular_suffix(path)
    payload = _decompress_submission_payload(path.read_bytes(), suffix)
    matrix, target = load_svmlight_file(io.BytesIO(payload))
    columns = [f"feature_{idx + 1}" for idx in range(matrix.shape[1])]
    frame = pd.DataFrame.sparse.from_spmatrix(matrix, columns=columns)
    if _path_mentions_role(path, "train") or np.asarray(target).any():
        frame.insert(0, "target", target)
    return frame


def _read_fixed_width_tabular_frame(path: Path):
    import pandas as pd

    suffix = _tabular_suffix(path)
    payload = _decompress_submission_payload(path.read_bytes(), suffix)
    return pd.read_fwf(io.StringIO(payload.decode("utf-8", errors="ignore")))


def _read_sqlite_table(path: Path):
    import pandas as pd

    with sqlite3.connect(path) as conn:
        table = _first_sqlite_table(conn)
        if table is None:
            raise ValueError(f"No readable table found in {path}")
        return pd.read_sql_query(f"SELECT * FROM {_quote_sqlite_identifier(table)}", conn)


def _read_duckdb_table(path: Path):
    import duckdb

    conn = duckdb.connect(str(path), read_only=True)
    try:
        table = _first_duckdb_table(conn)
        if table is None:
            raise ValueError(f"No readable table found in {path}")
        return conn.execute(f"SELECT * FROM {_quote_duckdb_qualified_identifier(table)}").df()
    finally:
        conn.close()


def _read_geopackage_tabular_frame(path: Path):
    import pandas as pd

    with sqlite3.connect(path) as conn:
        table = _first_geopackage_table(conn)
        if table is None:
            raise ValueError(f"No feature or attribute tables found in GeoPackage: {path}")
        frame = pd.read_sql_query(f"SELECT * FROM {_quote_sqlite_identifier(table)}", conn)
    return _decode_geopackage_blob_columns(frame)


def _first_geopackage_table(conn: sqlite3.Connection) -> str | None:
    preferred = []
    try:
        rows = conn.execute("SELECT table_name, data_type FROM gpkg_contents ORDER BY table_name").fetchall()
        for table_name, data_type in rows:
            table = str(table_name)
            if str(data_type).lower() in {"features", "attributes"} and _sqlite_table_exists(conn, table):
                preferred.append(table)
    except sqlite3.DatabaseError:
        pass
    if preferred:
        return preferred[0]
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name").fetchall()
    for row in rows:
        table = str(row[0])
        if not _is_geopackage_metadata_table(table):
            return table
    return None


def _sqlite_table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _is_geopackage_metadata_table(table: str) -> bool:
    lowered = table.lower()
    return lowered.startswith(("gpkg_", "sqlite_", "rtree_")) or lowered.endswith("_rtree")


def _decode_geopackage_blob_columns(frame):
    for column in frame.columns:
        if frame[column].map(lambda value: isinstance(value, (bytes, bytearray, memoryview))).any():
            frame[column] = frame[column].map(_decode_geopackage_blob_value)
    return frame


def _decode_geopackage_blob_value(value):
    if isinstance(value, memoryview):
        return value.tobytes().hex()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    return value


def _read_shapefile_tabular_frame(path: Path):
    import pandas as pd

    dbf_path = _dbf_path_for_shapefile(path)
    fields, records = _read_dbf_records(dbf_path)
    columns = _dedupe_column_names([field["name"] for field in fields])
    return pd.DataFrame(records, columns=columns)


def _dbf_path_for_shapefile(path: Path) -> Path:
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


def _read_dbf_records(path: Path):
    encoding = _dbf_encoding_for_path(path)
    with path.open("rb") as handle:
        header = handle.read(32)
        if len(header) < 32:
            raise ValueError(f"Invalid DBF header: {path}")
        record_count = int.from_bytes(header[4:8], "little")
        header_length = int.from_bytes(header[8:10], "little")
        record_length = int.from_bytes(header[10:12], "little")
        fields = _read_dbf_fields(handle, encoding=encoding)
        handle.seek(header_length)
        records = []
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
                    _parse_dbf_value(
                        raw_record[offset : offset + length],
                        field_type=str(field["type"]),
                        decimal_count=int(field["decimal_count"]),
                        encoding=encoding,
                    )
                )
                offset += length
            records.append(values)
    return fields, records


def _read_dbf_fields(handle, *, encoding: str) -> list[dict[str, object]]:
    fields = []
    while True:
        marker = handle.read(1)
        if not marker or marker == b"\\r":
            break
        descriptor = marker + handle.read(31)
        if len(descriptor) < 32:
            break
        name = descriptor[:11].split(b"\\x00", 1)[0].decode(encoding, errors="replace").strip()
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


def _dbf_encoding_for_path(path: Path) -> str:
    cpg_path = path.with_suffix(".cpg")
    if cpg_path.exists():
        raw_encoding = cpg_path.read_text(encoding="ascii", errors="ignore").strip()
        if raw_encoding:
            return "utf-8" if raw_encoding == "65001" else raw_encoding
    return "latin1"


def _parse_dbf_value(raw_value: bytes, *, field_type: str, decimal_count: int, encoding: str):
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


def _read_kml_tabular_frame(path: Path):
    import pandas as pd
    from xml.etree import ElementTree as ET

    root = ET.fromstring(_read_kml_payload(path))
    placemarks = [node for node in root.iter() if _xml_local_name(node.tag) == "Placemark"]
    records = [_kml_placemark_to_record(placemark, index=idx, ET=ET) for idx, placemark in enumerate(placemarks)]
    return pd.DataFrame(records)


def _read_kml_payload(path: Path) -> bytes:
    suffix = _tabular_suffix(path)
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
    return _decompress_compressed_payload(path.read_bytes(), suffix)


def _kml_placemark_to_record(placemark, *, index: int, ET):
    record = {"placemark_index": index}
    for child in list(placemark):
        local_name = _xml_local_name(child.tag)
        if local_name in {"name", "description"} and child.text and child.text.strip():
            record[local_name] = child.text.strip()
    for data in placemark.iter():
        local_name = _xml_local_name(data.tag)
        if local_name == "Data":
            key = str(data.attrib.get("name", "")).strip()
            value = _first_kml_child_text(data, "value")
            if key and value is not None:
                record[key] = value
        elif local_name == "SimpleData":
            key = str(data.attrib.get("name", "")).strip()
            if key and data.text is not None:
                record[key] = data.text.strip()
    geometry = _first_kml_geometry_node(placemark)
    if geometry is not None:
        record["geometry_type"] = _xml_local_name(geometry.tag)
        coordinates = [
            text
            for node in geometry.iter()
            if _xml_local_name(node.tag) == "coordinates" and (text := (node.text or "").strip())
        ]
        if coordinates:
            record["coordinates"] = " ".join(coordinates)
        record["geometry"] = ET.tostring(geometry, encoding="unicode")
    return record


def _first_kml_child_text(node, child_name: str) -> str | None:
    for child in list(node):
        if _xml_local_name(child.tag) == child_name and child.text is not None:
            return child.text.strip()
    return None


def _first_kml_geometry_node(placemark):
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
        if _xml_local_name(node.tag) in geometry_tags:
            return node
    return None


def _xml_local_name(tag) -> str:
    text = str(tag)
    return text.rsplit("}", 1)[-1] if "}" in text else text


def _first_sqlite_table(conn: sqlite3.Connection) -> str | None:
    rows = conn.execute(
        '''
        SELECT name
        FROM sqlite_master
        WHERE type IN ('table', 'view')
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        '''
    ).fetchall()
    return str(rows[0][0]) if rows else None


def _quote_sqlite_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _first_duckdb_table(conn) -> tuple[str, str] | None:
    rows = conn.execute(
        '''
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
          AND table_type IN ('BASE TABLE', 'VIEW')
        ORDER BY table_schema, table_name
        '''
    ).fetchall()
    if not rows:
        return None
    schema, table = rows[0]
    return str(schema), str(table)


def _quote_duckdb_qualified_identifier(table: tuple[str, str]) -> str:
    schema, name = table
    if schema and schema != "main":
        return f"{_quote_sqlite_identifier(schema)}.{_quote_sqlite_identifier(name)}"
    return _quote_sqlite_identifier(name)


def _uncompressed_submission_suffix(suffix: str) -> str:
    for compression_suffix in ASSET_COMPRESSION_SUFFIXES:
        if suffix.endswith(compression_suffix):
            return suffix[: -len(compression_suffix)]
    return suffix


def _tabular_payload_base_suffix(suffix: str) -> str:
    base_suffix = _uncompressed_submission_suffix(suffix)
    if base_suffix.endswith(".zip") and base_suffix != ".zip":
        return base_suffix[: -len(".zip")]
    return base_suffix


def _sas_format_for_suffix(suffix: str) -> str | None:
    base_suffix = _tabular_payload_base_suffix(suffix)
    if base_suffix in {".xpt", ".xport"}:
        return "xport"
    if base_suffix == ".sas7bdat":
        return "sas7bdat"
    return None


def _is_json_lines_suffix(suffix: str) -> bool:
    return suffix.startswith(TABULAR_JSON_LINES_SUFFIX_PREFIXES)


def _is_yaml_suffix(suffix: str) -> bool:
    return suffix.startswith((".yaml", ".yml"))


def _is_tsv_like_suffix(suffix: str) -> bool:
    return suffix.startswith(TABULAR_TAB_DELIMITED_SUFFIX_PREFIXES)


def _is_psv_like_suffix(suffix: str) -> bool:
    return suffix.startswith(".psv")


def _default_delimited_text_separator(suffix: str) -> str:
    base_suffix = _uncompressed_submission_suffix(suffix)
    if _is_psv_like_suffix(base_suffix):
        return "|"
    if _is_tsv_like_suffix(base_suffix) or base_suffix == ".txt":
        return "\t"
    return ","


def _decompress_submission_payload(payload: bytes, suffix: str) -> bytes:
    if suffix.endswith(".zip"):
        return _zip_tabular_payload(payload, suffix=suffix)
    return _decompress_compressed_payload(payload, suffix)


def _zip_tabular_payload(payload: bytes, *, suffix: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
        member = _select_zip_tabular_member(archive, suffix=suffix)
        return archive.read(member)


def _select_zip_tabular_member(archive: zipfile.ZipFile, *, suffix: str) -> str:
    base_suffix = suffix[: -len(".zip")] if suffix.endswith(".zip") else suffix
    members = [
        info.filename
        for info in archive.infolist()
        if not info.is_dir()
        and not info.filename.startswith("__MACOSX/")
        and PurePosixPath(info.filename).name
        and not PurePosixPath(info.filename).name.startswith(".")
    ]
    if not members:
        raise ValueError("zip archive does not contain a tabular member")
    exact = [member for member in members if PurePosixPath(member).name.lower().endswith(base_suffix)]
    if len(exact) == 1:
        return exact[0]
    if len(members) == 1:
        return members[0]
    if exact:
        return sorted(exact, key=lambda member: (len(PurePosixPath(member).parts), member.lower()))[0]
    raise ValueError(f"zip archive contains multiple files and no {base_suffix} member")


def _compress_submission_payload(payload: bytes, suffix: str) -> bytes:
    return _compress_payload_for_suffix(payload, suffix)


def _read_embedded_submission(payload: bytes):
    return _frame_with_normalized_table_columns(_read_embedded_submission_raw(payload))


def _read_embedded_submission_raw(payload: bytes):
    import pandas as pd

    suffix = str(SUBMISSION_INPUT_SUFFIX or ".csv").lower()
    payload = _decompress_submission_payload(payload, suffix)
    base_suffix = _uncompressed_submission_suffix(suffix)
    if base_suffix in TABULAR_PARQUET_SUFFIXES:
        return pd.read_parquet(io.BytesIO(payload))
    if base_suffix == ".orc":
        return pd.read_orc(io.BytesIO(payload))
    if base_suffix in TABULAR_HDF_SUFFIXES:
        with tempfile.NamedTemporaryFile(suffix=base_suffix) as handle:
            handle.write(payload)
            handle.flush()
            return _read_hdf_table(Path(handle.name))
    if base_suffix in TABULAR_ARROW_IPC_SUFFIXES:
        return pd.read_feather(io.BytesIO(payload))
    if base_suffix == ".avro":
        return _read_avro_payload(payload)
    if base_suffix == ".ods":
        return pd.read_excel(io.BytesIO(payload), engine="odf")
    if base_suffix in TABULAR_EXCEL_SUFFIXES:
        return pd.read_excel(io.BytesIO(payload))
    if base_suffix in TABULAR_EXCEL_INPUT_ONLY_SUFFIXES:
        return pd.read_excel(io.BytesIO(payload), engine="pyxlsb")
    if base_suffix in TABULAR_STATA_SUFFIXES:
        return pd.read_stata(io.BytesIO(payload))
    if base_suffix in TABULAR_SAS_SUFFIXES:
        return pd.read_sas(io.BytesIO(payload), format=_sas_format_for_suffix(suffix))
    if base_suffix in TABULAR_SPSS_SUFFIXES:
        return pd.read_spss(io.BytesIO(payload))
    if base_suffix in TABULAR_MATLAB_SUFFIXES:
        with tempfile.NamedTemporaryFile(suffix=base_suffix) as handle:
            handle.write(payload)
            handle.flush()
            return _read_mat_tabular_frame(Path(handle.name))
    if base_suffix in TABULAR_ARFF_SUFFIXES:
        with tempfile.NamedTemporaryFile(suffix=base_suffix) as handle:
            handle.write(payload)
            handle.flush()
            return _read_arff_tabular_frame(Path(handle.name))
    if base_suffix.startswith(TABULAR_HTML_SUFFIX_PREFIXES):
        tables = pd.read_html(io.StringIO(payload.decode("utf-8", errors="ignore")))
        candidates = [table for table in tables if table.shape[1] > 0]
        if not candidates:
            raise ValueError("HTML submission payload does not contain table-like data.")
        return max(candidates, key=lambda table: (table.shape[0] * table.shape[1], table.shape[1]))
    if base_suffix == ".xml":
        return pd.read_xml(io.BytesIO(payload), parser="etree")
    if base_suffix in TABULAR_PICKLE_SUFFIXES:
        return pd.read_pickle(io.BytesIO(payload))
    if base_suffix in TABULAR_STRUCTURED_SUFFIXES:
        if _is_json_lines_suffix(base_suffix):
            return pd.read_json(io.BytesIO(payload), lines=True)
        if _is_yaml_suffix(base_suffix):
            return _yaml_payload_to_frame(payload)
        try:
            return _json_payload_to_frame(payload)
        except ValueError:
            return pd.read_json(io.BytesIO(payload))
    if base_suffix in TABULAR_TEXT_SUFFIXES:
        default = _default_delimited_text_separator(base_suffix)
        return pd.read_csv(io.BytesIO(payload), sep=_sniff_payload_delimiter(payload, default=default))
    return pd.read_csv(io.BytesIO(payload))


def _frame_to_submission_bytes(frame) -> bytes:
    frame = _frame_with_normalized_table_columns(frame)
    suffix = str(SUBMISSION_INPUT_SUFFIX or ".csv").lower()
    base_suffix = _uncompressed_submission_suffix(suffix)
    if base_suffix in TABULAR_PARQUET_SUFFIXES:
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        payload = buffer.getvalue()
    elif base_suffix == ".orc":
        buffer = io.BytesIO()
        frame.to_orc(buffer, index=False)
        payload = buffer.getvalue()
    elif base_suffix in TABULAR_HDF_SUFFIXES:
        with tempfile.NamedTemporaryFile(suffix=base_suffix) as handle:
            frame.to_hdf(handle.name, key="submission", mode="w", format="table", index=False)
            handle.seek(0)
            payload = handle.read()
    elif base_suffix in TABULAR_ARROW_IPC_SUFFIXES:
        buffer = io.BytesIO()
        frame.to_feather(buffer)
        payload = buffer.getvalue()
    elif base_suffix == ".avro":
        buffer = io.BytesIO()
        _write_avro_payload(frame, buffer)
        payload = buffer.getvalue()
    elif base_suffix == ".ods":
        buffer = io.BytesIO()
        frame.to_excel(buffer, index=False, engine="odf")
        payload = buffer.getvalue()
    elif base_suffix in TABULAR_EXCEL_SUFFIXES:
        buffer = io.BytesIO()
        frame.to_excel(buffer, index=False)
        payload = buffer.getvalue()
    elif base_suffix in TABULAR_STATA_SUFFIXES:
        buffer = io.BytesIO()
        frame.to_stata(buffer, write_index=False)
        payload = buffer.getvalue()
    elif base_suffix == ".xml":
        payload = frame.to_xml(index=False, parser="etree").encode("utf-8")
    elif base_suffix.startswith(TABULAR_HTML_SUFFIX_PREFIXES):
        payload = frame.to_html(index=False).encode("utf-8")
    elif base_suffix in TABULAR_PICKLE_SUFFIXES:
        buffer = io.BytesIO()
        frame.to_pickle(buffer)
        payload = buffer.getvalue()
    elif base_suffix in TABULAR_STRUCTURED_SUFFIXES:
        if _is_yaml_suffix(base_suffix):
            import yaml

            payload = yaml.safe_dump(frame.to_dict(orient="records"), sort_keys=False).encode("utf-8")
        else:
            payload = frame.to_json(orient="records", lines=_is_json_lines_suffix(base_suffix)).encode("utf-8")
    else:
        sep = _default_delimited_text_separator(base_suffix)
        payload = frame.to_csv(index=False, sep=sep).encode("utf-8")
    return _compress_submission_payload(payload, suffix)


def _write_avro_payload(frame, buffer) -> None:
    from fastavro import writer

    fields = [{"name": str(column), "type": ["null", _avro_field_type(frame[column])]} for column in frame.columns]
    schema = {"type": "record", "name": "SubmissionRecord", "fields": fields}
    writer(buffer, schema, _frame_to_avro_records(frame, fields))


def _avro_field_type(series) -> str:
    import pandas as pd

    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "long"
    if pd.api.types.is_float_dtype(series):
        return "double"
    return "string"


def _frame_to_avro_records(frame, fields: list[dict[str, object]]) -> list[dict[str, object]]:
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
            if _is_missing_avro_value(value):
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


def _is_missing_avro_value(value: object) -> bool:
    import numpy as np
    import pandas as pd

    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _numeric_frame(frame):
    import pandas as pd

    converted = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        converted[column] = pd.to_numeric(frame[column], errors="coerce")
    return converted


def _looks_like_probability_matrix(values) -> bool:
    if values.empty:
        return False
    row_sums = values.sum(axis=1)
    finite = row_sums.notna() & (row_sums > 0)
    if not finite.any():
        return False
    return bool((row_sums[finite] - 1.0).abs().median() <= 1e-4)


def _normalize_probability_rows(values):
    clipped = values.clip(lower=1e-12)
    row_sums = clipped.sum(axis=1).replace(0, 1.0)
    return clipped.div(row_sums, axis=0)


def _alignment_id_key(value) -> str:
    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except Exception:
        pass
    try:
        value = value.item()
    except Exception:
        pass
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _looks_like_runtime_id_column(column) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
    if not normalized:
        return False
    compact = normalized.replace("_", "")
    if normalized in ID_LIKE_COLUMN_NAMES or compact in ID_LIKE_COLUMN_NAMES:
        return True
    if compact in {
        "idx",
        "index",
        "name",
    }:
        return True
    return compact.endswith("id") or compact.endswith("identifier")


def _read_runtime_test_frame():
    test_path = _find_test_file()
    if test_path is None:
        return None, None
    try:
        test = _read_tabular_path(test_path)
        test.columns = [str(col) for col in test.columns]
        return test, test_path
    except Exception as exc:
        print(f"Runtime test frame read skipped: {exc}")
        return None, test_path


def _resolve_runtime_id_column(sample, test=None):
    sample_cols = [str(col) for col in sample.columns]
    if len(sample_cols) < 2:
        return None
    candidate = sample_cols[0]
    if test is not None and candidate in [str(col) for col in test.columns]:
        return candidate
    if _looks_like_runtime_id_column(candidate):
        return candidate
    return None


def _runtime_expected_row_count(sample, test=None) -> int:
    if test is not None and len(test) > 0 and (len(sample) <= 0 or len(sample) <= 10 or len(test) > len(sample)):
        return len(test)
    return len(sample)


def _alignment_template_without_id(sample, target_cols: list[str], expected_len: int):
    import pandas as pd

    rows = []
    for _idx in range(max(expected_len, 0)):
        row = {}
        for col in target_cols:
            non_null = sample[col].dropna() if col in sample.columns else []
            row[col] = non_null.iloc[0] if len(non_null) else ""
        rows.append(row)
    return pd.DataFrame(rows, columns=[str(col) for col in sample.columns])


def _alignment_template(sample, id_col: str):
    import pandas as pd

    sample_ids = sample[id_col].map(_alignment_id_key)
    test, _test_path = _read_runtime_test_frame()
    if test is None:
        return sample.copy(), sample_ids
    if (
        id_col not in test.columns
        or len(sample) <= 0
        or len(sample) > 10
        or len(test) <= len(sample)
        or sample_ids.duplicated().any()
        or test[id_col].map(_alignment_id_key).duplicated().any()
    ):
        return sample.copy(), sample_ids
    out = pd.DataFrame({id_col: test[id_col].to_numpy()})
    for col in sample.columns:
        if col == id_col:
            continue
        non_null = sample[col].dropna()
        out[col] = non_null.iloc[0] if len(non_null) else ""
    return out[[str(col) for col in sample.columns]], out[id_col].map(_alignment_id_key)


def _aligned_submission_bytes(payload: bytes) -> bytes:
    sample_path = _find_sample_submission()
    if sample_path is None:
        return payload
    try:
        import pandas as pd

        sample = _read_tabular_path(sample_path)
        submission = _read_embedded_submission(payload)
    except Exception as exc:
        print(f"Runtime sample alignment skipped: {exc}")
        return payload

    if sample.empty:
        return payload

    sample_cols = [str(col) for col in sample.columns]
    submission.columns = [str(col) for col in submission.columns]
    test, _test_path = _read_runtime_test_frame()
    id_col = _resolve_runtime_id_column(sample, test)
    target_cols = [col for col in sample_cols if col != id_col] if id_col is not None else sample_cols
    if not target_cols:
        return payload
    if id_col is None:
        common_targets = [col for col in target_cols if col in submission.columns]
        if not common_targets:
            return payload
        expected_len = _runtime_expected_row_count(sample, test)
        out = _alignment_template_without_id(sample, target_cols, expected_len)
        sub_targets = submission[common_targets].copy()
        numeric_targets = _numeric_frame(sub_targets)
        all_numeric = bool(numeric_targets.notna().any().all())
        if all_numeric:
            fallback = numeric_targets.mean(axis=0).fillna(0.0)
            aligned_rows = []
            for idx in range(expected_len):
                if idx < len(numeric_targets) and numeric_targets.iloc[idx].notna().all():
                    aligned_rows.append(numeric_targets.iloc[idx])
                else:
                    aligned_rows.append(fallback)
            aligned = pd.DataFrame(aligned_rows, columns=common_targets).reset_index(drop=True)
            for col in common_targets:
                out[col] = aligned[col].astype(float).to_numpy()
        else:
            fallback_values = {}
            for col in common_targets:
                non_null = submission[col].dropna()
                if len(non_null):
                    fallback_values[col] = non_null.iloc[0]
                else:
                    sample_non_null = sample[col].dropna() if col in sample.columns else []
                    fallback_values[col] = sample_non_null.iloc[0] if len(sample_non_null) else ""
            aligned_rows = []
            for idx in range(expected_len):
                if idx < len(submission):
                    aligned_rows.append(submission.loc[idx, common_targets])
                else:
                    aligned_rows.append(fallback_values)
            aligned = pd.DataFrame(aligned_rows, columns=common_targets).reset_index(drop=True)
            for col in common_targets:
                out[col] = aligned[col].to_numpy()
        missing_targets = [col for col in target_cols if col not in common_targets]
        if missing_targets:
            print(f"Runtime sample alignment kept sample defaults for missing target columns: {missing_targets}")
        return _frame_to_submission_bytes(out[sample_cols])

    if id_col not in submission.columns:
        if len(submission) == len(sample) and all(col in submission.columns for col in target_cols):
            out = sample.copy()
            out[target_cols] = submission[target_cols].to_numpy()
            return _frame_to_submission_bytes(out)
        return payload

    out, output_ids = _alignment_template(sample, id_col)
    common_targets = [col for col in target_cols if col in submission.columns]
    if not common_targets:
        return payload

    submission_ids = submission[id_col].map(_alignment_id_key)
    sub_targets = submission[common_targets].copy()
    numeric_targets = _numeric_frame(sub_targets)
    all_numeric = bool(numeric_targets.notna().any().all())
    probability_matrix = all_numeric and len(common_targets) > 1 and _looks_like_probability_matrix(numeric_targets)

    if all_numeric:
        fallback = numeric_targets.mean(axis=0).fillna(0.0)
        if probability_matrix:
            total = float(fallback.sum())
            fallback = (fallback.clip(lower=1e-12) / total) if total > 0 else fallback + (1.0 / len(fallback))
        lookup_values = _normalize_probability_rows(numeric_targets) if probability_matrix else numeric_targets
        lookup = {
            key: lookup_values.iloc[idx]
            for idx, key in enumerate(submission_ids)
            if idx < len(lookup_values) and lookup_values.iloc[idx].notna().all()
        }
        aligned_rows = [lookup.get(key, fallback) for key in output_ids]
        aligned = pd.DataFrame(aligned_rows, columns=common_targets).reset_index(drop=True)
        for col in common_targets:
            out[col] = aligned[col].astype(float).to_numpy()
    else:
        fallback_values = {}
        for col in common_targets:
            non_null = submission[col].dropna()
            if len(non_null):
                fallback_values[col] = non_null.iloc[0]
            else:
                sample_non_null = sample[col].dropna() if col in sample.columns else []
                fallback_values[col] = sample_non_null.iloc[0] if len(sample_non_null) else ""
        lookup = {
            key: submission.loc[idx, common_targets]
            for idx, key in enumerate(submission_ids)
            if idx < len(submission)
        }
        aligned_rows = [lookup.get(key, fallback_values) for key in output_ids]
        aligned = pd.DataFrame(aligned_rows, columns=common_targets).reset_index(drop=True)
        for col in common_targets:
            out[col] = aligned[col].to_numpy()

    missing_targets = [col for col in target_cols if col not in common_targets]
    if missing_targets:
        print(f"Runtime sample alignment kept sample defaults for missing target columns: {missing_targets}")
    return _frame_to_submission_bytes(out[sample_cols])


def _validate_runtime_submission_bytes(payload: bytes) -> None:
    if _is_archive_output():
        _validate_archive_submission_bytes(payload)
        return
    if not _is_tabular_output():
        if not payload:
            raise RuntimeError(f"Runtime submission validation failed: {SUBMISSION_OUTPUT_NAME} is empty.")
        return
    try:
        import pandas as pd

        submission = _read_embedded_submission(payload)
    except Exception as exc:
        raise RuntimeError(f"Runtime submission validation failed: unable to read {SUBMISSION_OUTPUT_NAME}.") from exc

    if submission.empty:
        raise RuntimeError(f"Runtime submission validation failed: {SUBMISSION_OUTPUT_NAME} has no data rows.")

    submission.columns = [str(col) for col in submission.columns]
    sample_path = _find_sample_submission()
    if sample_path is None:
        if submission.isna().any(axis=None):
            raise RuntimeError(
                f"Runtime submission validation failed: {SUBMISSION_OUTPUT_NAME} contains empty values."
            )
        return

    try:
        import pandas as pd

        sample = _read_tabular_path(sample_path)
    except Exception as exc:
        raise RuntimeError(f"Runtime submission validation failed: unable to read {sample_path}.") from exc

    sample.columns = [str(col) for col in sample.columns]
    expected_columns = list(sample.columns)
    if list(submission.columns) != expected_columns:
        raise RuntimeError(
            "Runtime submission validation failed: columns mismatch. "
            f"expected={expected_columns} actual={list(submission.columns)}"
        )
    if not expected_columns:
        raise RuntimeError(f"Runtime submission validation failed: {sample_path.name} has no columns.")

    test = None
    test_path = None
    if _find_test_file() is not None:
        test, test_path = _read_runtime_test_frame()
    id_col = _resolve_runtime_id_column(sample, test)
    if id_col is None:
        expected_len = _runtime_expected_row_count(sample, test)
        if len(submission) != expected_len:
            expected_source = test_path.name if test_path is not None and test is not None else sample_path.name
            raise RuntimeError(
                "Runtime submission validation failed: row count mismatch. "
                f"expected {expected_len} from {expected_source}, actual {len(submission)}."
            )
        if submission.isna().any(axis=None):
            raise RuntimeError(f"Runtime submission validation failed: {SUBMISSION_OUTPUT_NAME} contains empty values.")
        return

    if id_col not in submission.columns:
        raise RuntimeError(f"Runtime submission validation failed: id column missing: {id_col}")

    expected_ids = None
    expected_source = sample_path.name
    if test is not None and id_col in test.columns and len(test) > 0:
        expected_ids = test[id_col].astype(str).tolist()
        expected_source = test_path.name

    if expected_ids is None and id_col in sample.columns and len(sample) > 0:
        expected_ids = sample[id_col].astype(str).tolist()

    if expected_ids is not None:
        actual_ids = submission[id_col].astype(str).tolist()
        if len(actual_ids) != len(expected_ids):
            raise RuntimeError(
                "Runtime submission validation failed: row count mismatch. "
                f"expected {len(expected_ids)} from {expected_source}, actual {len(actual_ids)}."
            )
        if actual_ids != expected_ids:
            raise RuntimeError(
                "Runtime submission validation failed: id order/value mismatch. "
                f"expected first ids from {expected_source}: {expected_ids[:5]}, actual: {actual_ids[:5]}."
            )

    if submission.isna().any(axis=None):
        raise RuntimeError(f"Runtime submission validation failed: {SUBMISSION_OUTPUT_NAME} contains empty values.")


def _validate_archive_submission_bytes(payload: bytes) -> None:
    if not payload:
        raise RuntimeError(f"Runtime submission validation failed: {SUBMISSION_OUTPUT_NAME} is empty.")
    suffix = str(SUBMISSION_INPUT_SUFFIX or "").lower()
    if suffix in EXTERNAL_ARCHIVE_OUTPUT_SUFFIXES:
        return
    try:
        if suffix == ".zip":
            with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
                seen_names: set[str] = set()
                has_file = False
                for info in archive.infolist():
                    member_name = _safe_archive_member_name(info.filename)
                    if info.flag_bits & 0x1:
                        raise RuntimeError(
                            f"Runtime submission validation failed: {SUBMISSION_OUTPUT_NAME} "
                            f"has unsupported encrypted member: {member_name}"
                        )
                    if _zip_member_is_symlink(info):
                        raise RuntimeError(
                            f"Runtime submission validation failed: {SUBMISSION_OUTPUT_NAME} "
                            f"has unsupported symlink member: {member_name}"
                        )
                    if info.is_dir():
                        member_name = f"{member_name.rstrip('/')}/"
                    if member_name in seen_names:
                        raise RuntimeError(
                            f"Runtime submission validation failed: {SUBMISSION_OUTPUT_NAME} "
                            f"has duplicate archive member: {member_name}"
                        )
                    seen_names.add(member_name)
                    if not info.is_dir():
                        has_file = True
                if not has_file:
                    raise RuntimeError(
                        f"Runtime submission validation failed: {SUBMISSION_OUTPUT_NAME} archive has no files."
                    )
            return
        if suffix in ZSTD_TAR_ARCHIVE_SUFFIXES:
            import zstandard as zstd

            with zstd.ZstdDecompressor().stream_reader(io.BytesIO(payload)) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as archive:
                    if not _validate_tar_submission_members(archive):
                        raise RuntimeError(
                            f"Runtime submission validation failed: {SUBMISSION_OUTPUT_NAME} archive has no files."
                        )
            return
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
            if not _validate_tar_submission_members(archive.getmembers()):
                raise RuntimeError(
                    f"Runtime submission validation failed: {SUBMISSION_OUTPUT_NAME} archive has no files."
                )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Runtime submission validation failed: {SUBMISSION_OUTPUT_NAME} is invalid.") from exc


def _validate_tar_submission_members(members) -> bool:
    seen_names: set[str] = set()
    has_file = False
    for info in members:
        member_name = _safe_archive_member_name(info.name)
        if member_name in seen_names:
            raise RuntimeError(
                f"Runtime submission validation failed: {SUBMISSION_OUTPUT_NAME} "
                f"has duplicate archive member: {member_name}"
            )
        seen_names.add(member_name)
        if not info.isfile() and not info.isdir():
            raise RuntimeError(
                f"Runtime submission validation failed: {SUBMISSION_OUTPUT_NAME} "
                f"has unsupported tar member type: {member_name}"
            )
        if info.isfile():
            has_file = True
    return has_file


def _zip_member_is_symlink(info: zipfile.ZipInfo) -> bool:
    if info.create_system != 3:
        return False
    return stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK


def _safe_archive_member_name(name: str) -> str:
    normalized = str(name or "").replace("\\\\", "/")
    if not normalized.strip():
        raise RuntimeError(f"Runtime submission validation failed: {SUBMISSION_OUTPUT_NAME} has empty archive member.")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized):
        raise RuntimeError(
            f"Runtime submission validation failed: {SUBMISSION_OUTPUT_NAME} has unsafe archive path: {name}"
        )
    if ".." in PurePosixPath(normalized).parts:
        raise RuntimeError(
            f"Runtime submission validation failed: {SUBMISSION_OUTPUT_NAME} has unsafe archive path: {name}"
        )
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized or normalized == ".":
        raise RuntimeError(f"Runtime submission validation failed: {SUBMISSION_OUTPUT_NAME} has empty archive member.")
    return normalized


def main() -> None:
    _extract_input_archives()
    dst = _working_root() / SUBMISSION_OUTPUT_NAME
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = gzip.decompress(base64.b64decode(SUBMISSION_GZIP_B64.encode("ascii")))
    except Exception as exc:
        raise RuntimeError("Failed to decode embedded submission payload.") from exc
    if _is_tabular_output():
        payload = _aligned_submission_bytes(payload)
    _validate_runtime_submission_bytes(payload)
    dst.write_bytes(payload)
    print(f\"Wrote {dst} (bytes={dst.stat().st_size})\")


if __name__ == \"__main__\":
    main()
"""


_ARCHIVE_SUBMISSION_SUFFIXES = ARCHIVE_SUBMISSION_SUFFIXES_ORDERED
_TABULAR_SUBMISSION_SUFFIXES = set(TABULAR_SUBMISSION_SUFFIXES)
_NON_TABULAR_SINGLE_FILE_SUFFIXES = set(NON_TABULAR_SUBMISSION_SUFFIXES)
_EXTERNAL_ARCHIVE_OUTPUT_SUFFIXES = EXTERNAL_ARCHIVE_SUBMISSION_SUFFIXES
_ZSTD_TAR_ARCHIVE_SUFFIXES = ZSTD_TAR_ARCHIVE_SUBMISSION_SUFFIXES


def _submission_artifact_suffix(path: Path) -> str:
    suffix = artifact_suffix(path)
    if suffix in _ARCHIVE_SUBMISSION_SUFFIXES or suffix in _TABULAR_SUBMISSION_SUFFIXES:
        return suffix
    if suffix in _NON_TABULAR_SINGLE_FILE_SUFFIXES:
        return suffix
    return path.suffix.lower()


def _safe_submission_archive_member_name(name: str) -> str:
    normalized = str(name).replace("\\", "/").strip()
    if not normalized:
        raise KernelFailedError("Submission directory contains an empty archive member name.")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise KernelFailedError(f"Submission directory contains an unsafe archive member name: {name}")
    return path.as_posix()


def _directory_submission_payload(submission_path: Path) -> tuple[bytes, str]:
    members = sorted(submission_path.rglob("*"), key=lambda path: path.relative_to(submission_path).as_posix())
    if not members:
        raise KernelFailedError(f"Submission directory has no members: {submission_path}")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member in members:
            arcname = _safe_submission_archive_member_name(member.relative_to(submission_path).as_posix())
            info = zipfile.ZipInfo(arcname)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            if member.is_dir():
                info.filename = f"{info.filename.rstrip('/')}/"
                archive.writestr(info, b"")
            elif member.is_file():
                archive.writestr(info, member.read_bytes())
    return buffer.getvalue(), f"{submission_path.name}.zip"


def _submission_payload(submission_path: Path) -> tuple[bytes, str]:
    if submission_path.is_dir():
        return _directory_submission_payload(submission_path)
    return submission_path.read_bytes(), submission_path.name


def render_submission_kernel_script(submission_path: Path) -> str:
    """Render a self-contained submit-only kernel script with embedded submission bytes."""
    _validate_external_archive_before_embedding(submission_path)
    submission_bytes, output_name = _submission_payload(submission_path)
    compressed = gzip.compress(submission_bytes, compresslevel=9)
    payload_b64 = base64.b64encode(compressed).decode("ascii")
    submission_suffix = _submission_artifact_suffix(Path(output_name))
    return (
        SUBMISSION_KERNEL_TEMPLATE.replace("__SUBMISSION_GZIP_B64__", json.dumps(payload_b64))
        .replace("__SUBMISSION_INPUT_SUFFIX__", json.dumps(submission_suffix))
        .replace("__SUBMISSION_OUTPUT_NAME__", json.dumps(output_name))
        .replace("__ASSET_COMPRESSION_SUFFIXES__", repr(ASSET_COMPRESSION_SUFFIXES))
        .replace("__TABULAR_OUTPUT_SUFFIXES__", repr(TABULAR_SUBMISSION_SUFFIXES_ORDERED))
        .replace("__TABULAR_INPUT_SUFFIXES__", repr(TABULAR_INPUT_SUFFIXES_ORDERED))
        .replace("__TABULAR_TEXT_SUFFIXES__", repr(tuple(sorted(TABULAR_TEXT_SUFFIXES))))
        .replace("__TABULAR_STRUCTURED_SUFFIXES__", repr(tuple(sorted(TABULAR_STRUCTURED_SUFFIXES))))
        .replace("__TABULAR_PICKLE_SUFFIXES__", repr(tuple(sorted(TABULAR_PICKLE_SUFFIXES))))
        .replace("__SQLITE_TABULAR_SUFFIXES__", repr(tuple(sorted(SQLITE_TABULAR_SUFFIXES))))
        .replace("__DUCKDB_TABULAR_SUFFIXES__", repr(tuple(sorted(DUCKDB_TABULAR_SUFFIXES))))
        .replace("__SAMPLE_OUTPUT_NAME_TOKENS__", repr(tuple(sorted(SAMPLE_OUTPUT_NAME_TOKENS))))
        .replace("__SAMPLE_COMPACT_NAME_ALIASES__", repr(tuple(sorted(SAMPLE_COMPACT_NAME_ALIASES))))
        .replace("__TEST_TABLE_STEMS__", repr(TEST_TABLE_STEMS))
        .replace("__TEST_TABLE_COMPACT_ALIASES__", repr(tuple(sorted(TEST_TABLE_COMPACT_ALIASES))))
        .replace("__TEST_TABLE_EXCLUDE_TOKENS__", repr(tuple(sorted(TEST_TABLE_EXCLUDE_TOKENS))))
        .replace("__STRONG_TEST_TABLE_TOKENS__", repr(tuple(sorted(STRONG_TEST_TABLE_TOKENS))))
        .replace("__WEAK_TEST_TABLE_TOKENS__", repr(tuple(sorted(WEAK_TEST_TABLE_TOKENS))))
        .replace("__ID_LIKE_COLUMN_NAMES__", repr(tuple(sorted(ID_LIKE_COLUMN_NAMES))))
        .replace("__TABULAR_ANNDATA_SUFFIXES__", repr(tuple(sorted(TABULAR_ANNDATA_SUFFIXES))))
        .replace("__TABULAR_ARROW_IPC_SUFFIXES__", repr(tuple(sorted(TABULAR_ARROW_IPC_SUFFIXES))))
        .replace("__TABULAR_PARQUET_SUFFIXES__", repr(tuple(sorted(TABULAR_PARQUET_SUFFIXES))))
        .replace("__TABULAR_EXCEL_INPUT_ONLY_SUFFIXES__", repr(tuple(sorted(TABULAR_EXCEL_INPUT_ONLY_SUFFIXES))))
        .replace("__TABULAR_EXCEL_SUFFIXES__", repr(tuple(sorted(TABULAR_EXCEL_SUFFIXES))))
        .replace("__TABULAR_FITS_SUFFIXES__", repr(tuple(sorted(TABULAR_FITS_SUFFIXES))))
        .replace("__TABULAR_GEOPACKAGE_SUFFIXES__", repr(tuple(sorted(TABULAR_GEOPACKAGE_SUFFIXES))))
        .replace("__TABULAR_HDF_SUFFIXES__", repr(tuple(sorted(TABULAR_HDF_SUFFIXES))))
        .replace("__TABULAR_JSON_LINES_SUFFIX_PREFIXES__", repr(tuple(TABULAR_JSON_LINES_SUFFIX_PREFIXES)))
        .replace("__TABULAR_KML_SUFFIXES__", repr(tuple(sorted(TABULAR_KML_SUFFIXES))))
        .replace("__TABULAR_LOOM_SUFFIXES__", repr(tuple(sorted(TABULAR_LOOM_SUFFIXES))))
        .replace("__TABULAR_STATA_SUFFIXES__", repr(tuple(sorted(TABULAR_STATA_SUFFIXES))))
        .replace("__TABULAR_SAS_SUFFIXES__", repr(tuple(sorted(TABULAR_SAS_SUFFIXES))))
        .replace("__TABULAR_SHAPEFILE_SUFFIXES__", repr(tuple(sorted(TABULAR_SHAPEFILE_SUFFIXES))))
        .replace("__TABULAR_SPSS_SUFFIXES__", repr(tuple(sorted(TABULAR_SPSS_SUFFIXES))))
        .replace("__TABULAR_MATLAB_SUFFIXES__", repr(tuple(sorted(TABULAR_MATLAB_SUFFIXES))))
        .replace("__TABULAR_RDATA_SUFFIXES__", repr(tuple(sorted(TABULAR_RDATA_SUFFIXES))))
        .replace("__TABULAR_NETCDF_SUFFIXES__", repr(tuple(sorted(TABULAR_NETCDF_SUFFIXES))))
        .replace("__TABULAR_NUMPY_SUFFIXES__", repr(tuple(sorted(TABULAR_NUMPY_SUFFIXES))))
        .replace("__TABULAR_ARFF_SUFFIXES__", repr(tuple(sorted(TABULAR_ARFF_SUFFIXES))))
        .replace("__TABULAR_HTML_SUFFIX_PREFIXES__", repr(tuple(TABULAR_HTML_SUFFIX_PREFIXES)))
        .replace("__TABULAR_SVMLIGHT_SUFFIX_PREFIXES__", repr(tuple(TABULAR_SVMLIGHT_SUFFIX_PREFIXES)))
        .replace("__TABULAR_FIXED_WIDTH_SUFFIX_PREFIXES__", repr(tuple(TABULAR_FIXED_WIDTH_SUFFIX_PREFIXES)))
        .replace("__TABULAR_TAB_DELIMITED_SUFFIX_PREFIXES__", repr(tuple(TABULAR_TAB_DELIMITED_SUFFIX_PREFIXES)))
        .replace("__ARCHIVE_OUTPUT_SUFFIXES__", repr(_ARCHIVE_SUBMISSION_SUFFIXES))
        .replace("__NON_TABULAR_OUTPUT_SUFFIXES__", repr(tuple(sorted(_NON_TABULAR_SINGLE_FILE_SUFFIXES))))
        .replace("__EXTERNAL_ARCHIVE_OUTPUT_SUFFIXES__", repr(_EXTERNAL_ARCHIVE_OUTPUT_SUFFIXES))
        .replace("__ZSTD_TAR_ARCHIVE_SUFFIXES__", repr(_ZSTD_TAR_ARCHIVE_SUFFIXES))
    )


def _validate_external_archive_before_embedding(submission_path: Path) -> None:
    if submission_path.is_dir():
        return
    suffix = _submission_artifact_suffix(submission_path)
    if suffix not in _EXTERNAL_ARCHIVE_OUTPUT_SUFFIXES:
        return
    try:
        SubmissionService._validate_external_archive_submission(submission_path)
    except SubmissionValidationError as exc:
        raise KernelFailedError(f"Refusing to embed invalid external archive submission: {submission_path}") from exc


def reject_static_tiny_code_competition_submission(
    *,
    slug: str,
    base_dir: Path,
    submission_path: Path,
    tiny_row_limit: int = 10,
) -> None:
    """Fail fast before embedding tiny public-test submissions for code competitions."""
    if count_tabular_data_rows_at_most(submission_path, limit=tiny_row_limit) is not True:
        return
    paths = CompetitionPaths(slug=slug, artifacts_dir=base_dir)
    if not infer_code_competition_from_paths(paths):
        return
    raise KernelFailedError(
        "Refusing to build a static wrapper submit kernel for a code/notebook competition "
        f"with only {tiny_row_limit} or fewer submission rows. "
        "Use notebook submit artifact mode 'inference' so Kaggle reruns the authoritative kernel "
        "against the hidden/full test set."
    )


def count_tabular_data_rows_at_most(path: Path, *, limit: int) -> bool | None:
    row_count = tabular_data_row_count_capped(path, cap=limit)
    if row_count is None:
        return None
    return row_count <= limit


def count_csv_data_rows_at_most(path: Path, *, limit: int) -> bool | None:
    """Backward-compatible alias for count_tabular_data_rows_at_most."""
    return count_tabular_data_rows_at_most(path, limit=limit)
