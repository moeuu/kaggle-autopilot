from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
import zipfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from io import BytesIO, StringIO, TextIOWrapper
from pathlib import Path

from kagglebot.asset_collections import ASSET_COLLECTION_DIR_NAMES
from kagglebot.compression_suffixes import (
    ASSET_COMPRESSION_SUFFIXES,
    open_compressed_binary,
    open_compressed_text,
    write_compressed_bytes,
)
from kagglebot.role_tokens import ROLE_ALIASES, ROLE_TRAILING_PREFIXES, TEST_DIRECT_ROLE_ALIASES
from kagglebot.sample_name_aliases import SAMPLE_COMPACT_NAME_ALIASES, SAMPLE_OUTPUT_NAME_TOKENS
from kagglebot.table_columns import normalize_table_column_names

_SAMPLE_STAGE_PATTERN = re.compile(r"(?:stage|phase|round)[_-]?(\d+)", re.IGNORECASE)
_TABULAR_TEXT_BASE_SUFFIXES = (".csv", ".tsv", ".tab", ".psv", ".txt")
_TABULAR_STRUCTURED_BASE_SUFFIXES = (".json", ".jsonl", ".jsonlines", ".ndjson", ".geojson", ".yaml", ".yml")
_TABULAR_PICKLE_BASE_SUFFIXES = (".pkl", ".pickle")
_TABULAR_PARQUET_BASE_SUFFIXES = (".parquet", ".parq", ".pq")
_TABULAR_ARROW_IPC_BASE_SUFFIXES = (".feather", ".ftr", ".arrow", ".ipc")
_TABULAR_BINARY_BASE_SUFFIXES = (
    *_TABULAR_PARQUET_BASE_SUFFIXES,
    ".orc",
    *_TABULAR_ARROW_IPC_BASE_SUFFIXES,
    ".avro",
    ".hdf",
    ".hdf5",
    ".dta",
    ".xml",
    ".xls",
    ".xlsm",
    ".xlsx",
    ".ods",
)
_TABULAR_STATA_BASE_SUFFIXES = (".dta",)
_TABULAR_SAS_BASE_SUFFIXES = (".sas7bdat", ".xpt", ".xport")
_TABULAR_SPSS_BASE_SUFFIXES = (".sav", ".zsav")
_TABULAR_MATLAB_BASE_SUFFIXES = (".mat",)
_TABULAR_RDATA_BASE_SUFFIXES = (".rds", ".rda", ".rdata")
_TABULAR_NETCDF_BASE_SUFFIXES = (".nc", ".netcdf", ".cdf", ".nc4")
_TABULAR_NUMPY_BASE_SUFFIXES = (".npy", ".npz")
_TABULAR_FITS_BASE_SUFFIXES = (".fits", ".fit", ".fts")
_TABULAR_ANNDATA_BASE_SUFFIXES = (".h5ad",)
_TABULAR_LOOM_BASE_SUFFIXES = (".loom",)
_TABULAR_GEOPACKAGE_BASE_SUFFIXES = (".gpkg", ".geopackage")
_TABULAR_SHAPEFILE_BASE_SUFFIXES = (".shp", ".dbf")
_TABULAR_KML_BASE_SUFFIXES = (".kml", ".kmz")
_TABULAR_COMPRESSIBLE_KML_BASE_SUFFIXES = (".kml",)
_NUMPY_COLUMN_TEXT_SIDECAR_SUFFIXES = (
    ".columns.txt",
    "_columns.txt",
    ".cols.txt",
    "_cols.txt",
    ".features.txt",
    "_features.txt",
    ".feature_names.txt",
    "_feature_names.txt",
)
_NUMPY_COLUMN_JSON_SIDECAR_SUFFIXES = (
    ".columns.json",
    "_columns.json",
    ".schema.json",
    "_schema.json",
    ".features.json",
    "_features.json",
    ".feature_names.json",
    "_feature_names.json",
)
_NUMPY_GENERIC_COLUMN_SIDECAR_NAMES = (
    "columns.txt",
    "feature_names.txt",
    "features.txt",
    "columns.json",
    "schema.json",
    "feature_names.json",
)
_TABULAR_ARFF_BASE_SUFFIXES = (".arff",)
_TABULAR_HTML_BASE_SUFFIXES = (".html", ".htm")
_DUCKDB_TABULAR_BASE_SUFFIXES = (".duckdb", ".ddb")
_TABULAR_DELIMITED_INPUT_ONLY_BASE_SUFFIXES = (".dat",)
_TABULAR_FIXED_WIDTH_BASE_SUFFIXES = (".fwf", ".fixed", ".fixedwidth")
_TABULAR_EXCEL_BASE_SUFFIXES = (".ods", ".xls", ".xlsm", ".xlsx")
_TABULAR_EXCEL_INPUT_ONLY_BASE_SUFFIXES = (".xlsb",)
_TABULAR_SVMLIGHT_BASE_SUFFIXES = (".svm", ".svmlight", ".libsvm")
_ZIP_WRAPPED_TABULAR_BASE_SUFFIXES = (
    *_TABULAR_TEXT_BASE_SUFFIXES,
    *_TABULAR_STRUCTURED_BASE_SUFFIXES,
    *_TABULAR_PARQUET_BASE_SUFFIXES,
    ".orc",
    *_TABULAR_ARROW_IPC_BASE_SUFFIXES,
    ".avro",
    *_TABULAR_PICKLE_BASE_SUFFIXES,
    ".xml",
    *_TABULAR_HTML_BASE_SUFFIXES,
    *_TABULAR_ARFF_BASE_SUFFIXES,
    *_TABULAR_EXCEL_BASE_SUFFIXES,
    *_TABULAR_STATA_BASE_SUFFIXES,
    *_TABULAR_SAS_BASE_SUFFIXES,
    *_TABULAR_SPSS_BASE_SUFFIXES,
    *_TABULAR_DELIMITED_INPUT_ONLY_BASE_SUFFIXES,
    *_TABULAR_FIXED_WIDTH_BASE_SUFFIXES,
    *_TABULAR_EXCEL_INPUT_ONLY_BASE_SUFFIXES,
    *_TABULAR_SVMLIGHT_BASE_SUFFIXES,
)
_ZIP_WRAPPED_TABULAR_INPUT_SUFFIXES = tuple(f"{base}.zip" for base in _ZIP_WRAPPED_TABULAR_BASE_SUFFIXES)
_TABULAR_INPUT_ONLY_BASE_SUFFIXES = (
    ".h5",
    ".geojson",
    *_TABULAR_DELIMITED_INPUT_ONLY_BASE_SUFFIXES,
    *_TABULAR_FIXED_WIDTH_BASE_SUFFIXES,
    *_TABULAR_EXCEL_INPUT_ONLY_BASE_SUFFIXES,
    *_TABULAR_SVMLIGHT_BASE_SUFFIXES,
    *_TABULAR_SAS_BASE_SUFFIXES,
    *_TABULAR_SPSS_BASE_SUFFIXES,
    *_TABULAR_MATLAB_BASE_SUFFIXES,
    *_TABULAR_RDATA_BASE_SUFFIXES,
    *_TABULAR_NETCDF_BASE_SUFFIXES,
    *_TABULAR_NUMPY_BASE_SUFFIXES,
    *_TABULAR_FITS_BASE_SUFFIXES,
    *_TABULAR_ANNDATA_BASE_SUFFIXES,
    *_TABULAR_LOOM_BASE_SUFFIXES,
    *_TABULAR_GEOPACKAGE_BASE_SUFFIXES,
    *_TABULAR_SHAPEFILE_BASE_SUFFIXES,
    *_TABULAR_KML_BASE_SUFFIXES,
    *_TABULAR_ARFF_BASE_SUFFIXES,
    *_DUCKDB_TABULAR_BASE_SUFFIXES,
)
_COMPRESSIBLE_TABULAR_INPUT_ONLY_BASE_SUFFIXES = (
    ".geojson",
    *_TABULAR_DELIMITED_INPUT_ONLY_BASE_SUFFIXES,
    *_TABULAR_FIXED_WIDTH_BASE_SUFFIXES,
    *_TABULAR_SVMLIGHT_BASE_SUFFIXES,
    *_TABULAR_FITS_BASE_SUFFIXES,
    *_TABULAR_COMPRESSIBLE_KML_BASE_SUFFIXES,
    *_TABULAR_ARFF_BASE_SUFFIXES,
)
_ROLE_SUFFIXES = {
    "",
    "data",
    "feature",
    "features",
    "label",
    "labels",
    "metadata",
    "meta",
    "set",
    "table",
    *ASSET_COLLECTION_DIR_NAMES,
}
ROLE_SUFFIXES = frozenset(_ROLE_SUFFIXES)
_TABULAR_TEXT_SUFFIXES = set(_TABULAR_TEXT_BASE_SUFFIXES)
_TABULAR_STRUCTURED_SUFFIXES = set(_TABULAR_STRUCTURED_BASE_SUFFIXES)
_TABULAR_PICKLE_SUFFIXES = set(_TABULAR_PICKLE_BASE_SUFFIXES)
_SQLITE_TABULAR_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
_TABULAR_SAMPLE_SUFFIXES = (
    _TABULAR_TEXT_SUFFIXES
    | _TABULAR_STRUCTURED_SUFFIXES
    | _TABULAR_PICKLE_SUFFIXES
    | _SQLITE_TABULAR_SUFFIXES
    | set(_TABULAR_BINARY_BASE_SUFFIXES)
    | set(_TABULAR_INPUT_ONLY_BASE_SUFFIXES)
    | set(_TABULAR_HTML_BASE_SUFFIXES)
)
_TABULAR_COMPRESSION_SUFFIXES = ASSET_COMPRESSION_SUFFIXES


def _compressed_tabular_suffixes(base_suffixes: Sequence[str]) -> tuple[str, ...]:
    return tuple(f"{base}{compression}" for base in base_suffixes for compression in _TABULAR_COMPRESSION_SUFFIXES)


_COMPRESSED_TABULAR_TEXT_SUFFIXES = _compressed_tabular_suffixes(_TABULAR_TEXT_BASE_SUFFIXES)
_COMPRESSED_TABULAR_STRUCTURED_SUFFIXES = _compressed_tabular_suffixes(_TABULAR_STRUCTURED_BASE_SUFFIXES)
_COMPRESSED_TABULAR_PICKLE_SUFFIXES = _compressed_tabular_suffixes(_TABULAR_PICKLE_BASE_SUFFIXES)
_COMPRESSED_TABULAR_BINARY_SUFFIXES = _compressed_tabular_suffixes((".xml",))
_COMPRESSED_TABULAR_INPUT_ONLY_SUFFIXES = _compressed_tabular_suffixes(_COMPRESSIBLE_TABULAR_INPUT_ONLY_BASE_SUFFIXES)
_COMPRESSED_TABULAR_HTML_SUFFIXES = _compressed_tabular_suffixes(_TABULAR_HTML_BASE_SUFFIXES)
_GEOJSON_TABLE_BASE_SUFFIXES = (".geojson",)
_JSON_OBJECT_TABLE_BASE_SUFFIXES = (".json", *_GEOJSON_TABLE_BASE_SUFFIXES)
_YAML_TABLE_BASE_SUFFIXES = (".yaml", ".yml")
_JSON_TABLE_SUFFIXES = frozenset(
    _JSON_OBJECT_TABLE_BASE_SUFFIXES + _compressed_tabular_suffixes(_JSON_OBJECT_TABLE_BASE_SUFFIXES)
)
TABULAR_GEOJSON_SUFFIXES = frozenset(
    _GEOJSON_TABLE_BASE_SUFFIXES + _compressed_tabular_suffixes(_GEOJSON_TABLE_BASE_SUFFIXES)
)
_YAML_TABLE_SUFFIXES = frozenset(_YAML_TABLE_BASE_SUFFIXES + _compressed_tabular_suffixes(_YAML_TABLE_BASE_SUFFIXES))
_COMPRESSED_TABULAR_SUFFIXES = tuple(
    sorted(
        _COMPRESSED_TABULAR_TEXT_SUFFIXES
        + _COMPRESSED_TABULAR_STRUCTURED_SUFFIXES
        + _COMPRESSED_TABULAR_PICKLE_SUFFIXES
        + _COMPRESSED_TABULAR_BINARY_SUFFIXES
        + _COMPRESSED_TABULAR_HTML_SUFFIXES
        + _COMPRESSED_TABULAR_INPUT_ONLY_SUFFIXES
        + _ZIP_WRAPPED_TABULAR_INPUT_SUFFIXES,
        key=len,
        reverse=True,
    )
)
SQLITE_TABULAR_SUFFIXES = frozenset(_SQLITE_TABULAR_SUFFIXES)
DUCKDB_TABULAR_SUFFIXES = frozenset(_DUCKDB_TABULAR_BASE_SUFFIXES)
TABULAR_INPUT_SUFFIXES = frozenset(_TABULAR_SAMPLE_SUFFIXES | set(_COMPRESSED_TABULAR_SUFFIXES))
TABULAR_SUBMISSION_SUFFIXES = frozenset(
    TABULAR_INPUT_SUFFIXES
    - SQLITE_TABULAR_SUFFIXES
    - set(_TABULAR_INPUT_ONLY_BASE_SUFFIXES)
    - set(_COMPRESSED_TABULAR_INPUT_ONLY_SUFFIXES)
    - set(_ZIP_WRAPPED_TABULAR_INPUT_SUFFIXES)
)
TABULAR_TEXT_SUFFIXES = frozenset(
    _TABULAR_TEXT_SUFFIXES
    | set(_COMPRESSED_TABULAR_TEXT_SUFFIXES)
    | {suffix for suffix in _ZIP_WRAPPED_TABULAR_INPUT_SUFFIXES if suffix.startswith(_TABULAR_TEXT_BASE_SUFFIXES)}
    | set(_TABULAR_DELIMITED_INPUT_ONLY_BASE_SUFFIXES)
    | {
        suffix
        for suffix in _COMPRESSED_TABULAR_INPUT_ONLY_SUFFIXES
        if suffix.startswith(_TABULAR_DELIMITED_INPUT_ONLY_BASE_SUFFIXES)
    }
    | {
        suffix
        for suffix in _ZIP_WRAPPED_TABULAR_INPUT_SUFFIXES
        if suffix.startswith(_TABULAR_DELIMITED_INPUT_ONLY_BASE_SUFFIXES)
    }
)
TABULAR_STRUCTURED_SUFFIXES = frozenset(
    _TABULAR_STRUCTURED_SUFFIXES
    | set(_COMPRESSED_TABULAR_STRUCTURED_SUFFIXES)
    | {suffix for suffix in _ZIP_WRAPPED_TABULAR_INPUT_SUFFIXES if suffix.startswith(_TABULAR_STRUCTURED_BASE_SUFFIXES)}
)
TABULAR_PICKLE_SUFFIXES = frozenset(_TABULAR_PICKLE_SUFFIXES | set(_COMPRESSED_TABULAR_PICKLE_SUFFIXES))
TABULAR_ARFF_SUFFIXES = frozenset(
    _TABULAR_ARFF_BASE_SUFFIXES
    + tuple(
        suffix for suffix in _COMPRESSED_TABULAR_INPUT_ONLY_SUFFIXES if suffix.startswith(_TABULAR_ARFF_BASE_SUFFIXES)
    )
    + tuple(suffix for suffix in _ZIP_WRAPPED_TABULAR_INPUT_SUFFIXES if suffix.startswith(_TABULAR_ARFF_BASE_SUFFIXES))
)
TABULAR_ARROW_IPC_SUFFIXES = frozenset(_TABULAR_ARROW_IPC_BASE_SUFFIXES)
TABULAR_PARQUET_SUFFIXES = frozenset(_TABULAR_PARQUET_BASE_SUFFIXES)
TABULAR_EXCEL_INPUT_ONLY_SUFFIXES = frozenset(_TABULAR_EXCEL_INPUT_ONLY_BASE_SUFFIXES)
TABULAR_EXCEL_SUFFIXES = frozenset(_TABULAR_EXCEL_BASE_SUFFIXES)
TABULAR_FIXED_WIDTH_SUFFIX_PREFIXES = _TABULAR_FIXED_WIDTH_BASE_SUFFIXES
TABULAR_HDF_SUFFIXES = frozenset({".h5", ".hdf", ".hdf5"})
TABULAR_HTML_SUFFIX_PREFIXES = _TABULAR_HTML_BASE_SUFFIXES
TABULAR_JSON_LINES_SUFFIX_PREFIXES = (".jsonl", ".jsonlines", ".ndjson")
TABULAR_MATLAB_SUFFIXES = frozenset(_TABULAR_MATLAB_BASE_SUFFIXES)
TABULAR_RDATA_SUFFIXES = frozenset(_TABULAR_RDATA_BASE_SUFFIXES)
TABULAR_NETCDF_SUFFIXES = frozenset(_TABULAR_NETCDF_BASE_SUFFIXES)
TABULAR_NUMPY_SUFFIXES = frozenset(_TABULAR_NUMPY_BASE_SUFFIXES)
TABULAR_FITS_SUFFIXES = frozenset(
    _TABULAR_FITS_BASE_SUFFIXES
    + tuple(
        suffix for suffix in _COMPRESSED_TABULAR_INPUT_ONLY_SUFFIXES if suffix.startswith(_TABULAR_FITS_BASE_SUFFIXES)
    )
)
TABULAR_ANNDATA_SUFFIXES = frozenset(_TABULAR_ANNDATA_BASE_SUFFIXES)
TABULAR_DUCKDB_SUFFIXES = DUCKDB_TABULAR_SUFFIXES
TABULAR_LOOM_SUFFIXES = frozenset(_TABULAR_LOOM_BASE_SUFFIXES)
TABULAR_GEOPACKAGE_SUFFIXES = frozenset(_TABULAR_GEOPACKAGE_BASE_SUFFIXES)
TABULAR_SHAPEFILE_SUFFIXES = frozenset(_TABULAR_SHAPEFILE_BASE_SUFFIXES)
TABULAR_KML_SUFFIXES = frozenset(
    _TABULAR_KML_BASE_SUFFIXES
    + tuple(
        suffix
        for suffix in _COMPRESSED_TABULAR_INPUT_ONLY_SUFFIXES
        if suffix.startswith(_TABULAR_COMPRESSIBLE_KML_BASE_SUFFIXES)
    )
)
TABULAR_SAS_SUFFIXES = frozenset(_TABULAR_SAS_BASE_SUFFIXES)
TABULAR_SPSS_SUFFIXES = frozenset(_TABULAR_SPSS_BASE_SUFFIXES)
TABULAR_STATA_SUFFIXES = frozenset(_TABULAR_STATA_BASE_SUFFIXES)
TABULAR_TAB_DELIMITED_SUFFIX_PREFIXES = (".tsv", ".tab")
TABULAR_SVMLIGHT_SUFFIX_PREFIXES = _TABULAR_SVMLIGHT_BASE_SUFFIXES
TABULAR_TSV_LIKE_SUFFIX_PREFIXES = (".tsv", ".tab", ".txt")
TABULAR_TXT_LIKE_SUFFIX_PREFIXES = (".txt",)
TABULAR_PSV_LIKE_SUFFIX_PREFIXES = (".psv",)
_TABULAR_BASE_SUFFIXES_ORDERED = (
    _TABULAR_TEXT_BASE_SUFFIXES
    + _TABULAR_BINARY_BASE_SUFFIXES
    + _TABULAR_HTML_BASE_SUFFIXES
    + _TABULAR_INPUT_ONLY_BASE_SUFFIXES
    + _TABULAR_STRUCTURED_BASE_SUFFIXES
    + _TABULAR_PICKLE_BASE_SUFFIXES
)
_COMPRESSED_TABULAR_SUFFIXES_ORDERED = (
    tuple(
        f"{base}{compression}"
        for compression in _TABULAR_COMPRESSION_SUFFIXES
        for base in (
            _TABULAR_TEXT_BASE_SUFFIXES
            + _TABULAR_DELIMITED_INPUT_ONLY_BASE_SUFFIXES
            + _TABULAR_FIXED_WIDTH_BASE_SUFFIXES
            + (".xml",)
            + _TABULAR_STRUCTURED_BASE_SUFFIXES
            + _TABULAR_PICKLE_BASE_SUFFIXES
            + _TABULAR_ARFF_BASE_SUFFIXES
            + _TABULAR_FITS_BASE_SUFFIXES
            + _TABULAR_COMPRESSIBLE_KML_BASE_SUFFIXES
            + _TABULAR_HTML_BASE_SUFFIXES
            + _TABULAR_SVMLIGHT_BASE_SUFFIXES
        )
    )
    + _ZIP_WRAPPED_TABULAR_INPUT_SUFFIXES
)
TABULAR_SUBMISSION_SUFFIXES_ORDERED = tuple(
    suffix
    for suffix in _TABULAR_BASE_SUFFIXES_ORDERED + _COMPRESSED_TABULAR_SUFFIXES_ORDERED
    if suffix in TABULAR_SUBMISSION_SUFFIXES
)
TABULAR_SUBMISSION_SUFFIXES_LENGTH_ORDERED = tuple(sorted(TABULAR_SUBMISSION_SUFFIXES, key=len, reverse=True))
TABULAR_INPUT_SUFFIXES_ORDERED = tuple(
    suffix
    for suffix in _TABULAR_BASE_SUFFIXES_ORDERED + _COMPRESSED_TABULAR_SUFFIXES_ORDERED
    if suffix in TABULAR_INPUT_SUFFIXES
) + tuple(sorted(SQLITE_TABULAR_SUFFIXES))


def preferred_tabular_submission_suffix(
    expected_suffixes: Sequence[str] | None,
    *,
    default: str = ".csv",
) -> str:
    """Return the first expected suffix this runtime can write as a tabular submission."""
    for raw_suffix in expected_suffixes or ():
        suffix = str(raw_suffix or "").strip().lower()
        if not suffix:
            continue
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        if suffix in TABULAR_SUBMISSION_SUFFIXES:
            return suffix
    return default


def preferred_rowless_tabular_sample_suffix(
    expected_suffixes: Sequence[str] | None,
    *,
    default: str = ".csv",
) -> str:
    """Return a writable suffix that can preserve a header-only sample schema."""
    for raw_suffix in expected_suffixes or ():
        suffix = str(raw_suffix or "").strip().lower()
        if not suffix:
            continue
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        if suffix not in TABULAR_SUBMISSION_SUFFIXES:
            continue
        if suffix in TABULAR_STRUCTURED_SUFFIXES:
            continue
        if suffix in TABULAR_HDF_SUFFIXES or suffix == ".orc" or suffix.startswith(".xml"):
            continue
        return suffix
    return default


def is_json_lines_tabular_suffix(suffix: str) -> bool:
    """Return whether a normalized tabular suffix names JSON Lines data."""
    return suffix.startswith(TABULAR_JSON_LINES_SUFFIX_PREFIXES)


def is_tsv_like_tabular_suffix(suffix: str) -> bool:
    """Return whether a normalized tabular suffix should use tab delimiters."""
    return suffix.startswith(TABULAR_TSV_LIKE_SUFFIX_PREFIXES)


def is_tab_delimited_tabular_suffix(suffix: str) -> bool:
    """Return whether a normalized tabular suffix explicitly names tab-delimited data."""
    return suffix.startswith(TABULAR_TAB_DELIMITED_SUFFIX_PREFIXES)


def is_psv_like_tabular_suffix(suffix: str) -> bool:
    """Return whether a normalized tabular suffix should use pipe delimiters."""
    return suffix.startswith(TABULAR_PSV_LIKE_SUFFIX_PREFIXES)


def is_txt_like_tabular_suffix(suffix: str) -> bool:
    """Return whether a normalized tabular suffix names generic text-delimited data."""
    return suffix.startswith(TABULAR_TXT_LIKE_SUFFIX_PREFIXES)


def default_delimited_text_separator(suffix: str) -> str:
    """Return the default delimiter for a normalized tabular text suffix."""
    base_suffix = zip_wrapped_tabular_base_suffix(suffix)
    if is_psv_like_tabular_suffix(base_suffix):
        return "|"
    if is_tsv_like_tabular_suffix(base_suffix):
        return "\t"
    return ","


def is_delimited_text_tabular_suffix(suffix: str) -> bool:
    """Return whether a normalized suffix is a direct delimited text table suffix."""
    return suffix in _TABULAR_TEXT_SUFFIXES


def tabular_suffix(path: Path) -> str:
    name = path.name.lower()
    for suffix in _COMPRESSED_TABULAR_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return path.suffix.lower()


def tabular_stem(path: Path) -> str:
    suffix = tabular_suffix(path)
    name = path.name
    if suffix and name.lower().endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def path_mentions_role(path: Path, role: str) -> bool:
    """Return whether a tabular path component clearly identifies a train/test role."""
    stem = tabular_stem(path)
    if component_mentions_role(stem, role):
        return True
    if role == "test" and component_mentions_role(stem, "train"):
        return False
    aliases = _role_aliases(role)
    return any(str(part).lower() in aliases for part in path.parts)


def component_mentions_role(value: str, role: str) -> bool:
    lowered = str(value).lower()
    tokens = _name_tokens(lowered)
    aliases = _role_aliases(role)
    train_aliases = _role_aliases("train")
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
        if compact.startswith(alias) and compact[len(alias) :] in _ROLE_SUFFIXES:
            return True
    if role == "train" and compact.startswith("train"):
        return compact[len("train") :] in _ROLE_SUFFIXES | {"ing", "set"}
    if role == "test" and compact.startswith("test"):
        return compact[len("test") :] in _ROLE_SUFFIXES | {"ing", "set"}
    if compact.endswith(role):
        return compact[: -len(role)] in ROLE_TRAILING_PREFIXES
    return False


def roleless_stem(path: Path, role: str) -> str:
    normalized = tabular_stem(path).lower().replace(role, "")
    return normalized.replace("_", "").replace("-", "").replace(".", "")


def _role_aliases(role: str) -> set[str]:
    return set(ROLE_ALIASES.get(role, frozenset({role})))


def _name_tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", value.lower()) if token}


def is_tabular_sample_path(path: Path) -> bool:
    suffix = tabular_suffix(path)
    return suffix in _TABULAR_SAMPLE_SUFFIXES or suffix in _COMPRESSED_TABULAR_SUFFIXES


def is_tabular_data_path(path: Path) -> bool:
    """Return whether a path has a supported tabular input/data suffix."""

    return is_tabular_sample_path(path)


def _is_text_tabular_suffix(path: Path) -> bool:
    suffix = tabular_suffix(path)
    return suffix in _TABULAR_TEXT_SUFFIXES or suffix in _COMPRESSED_TABULAR_TEXT_SUFFIXES


def select_sample_submission_path(files: Sequence[Path]) -> Path | None:
    """Pick the most plausible sample-submission file from discovered tabular files."""
    candidates = [path for path in files if sample_name_score(path) > 0]
    if not candidates:
        return None
    usable = [path for path in candidates if tabular_file_has_data_rows(path)]
    ranked = usable or candidates
    return max(ranked, key=sample_candidate_key)


def find_usable_sample_submissions(data_dir: Path) -> list[Path]:
    """List usable sample-submission tabular candidates in ranked preference order."""
    if not data_dir.exists():
        return []
    candidates: list[Path] = []
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        if not is_tabular_sample_path(path):
            continue
        if sample_name_score(path) < 2:
            continue
        if not tabular_file_has_data_rows(path):
            continue
        if not tabular_file_has_two_or_more_columns(path):
            continue
        candidates.append(path)
    return sorted(candidates, key=sample_candidate_key, reverse=True)


def sample_candidate_key(path: Path) -> tuple[int, int, int, int, int, int, str]:
    """Return ranking key for sample-submission candidates."""
    name_score = sample_name_score(path)
    stage_score = sample_stage_score(path)
    desired_stage = resolve_desired_submission_stage()
    stage_match = 1 if (desired_stage is not None and stage_score == desired_stage) else 0
    explicit_stage = 1 if stage_score > 0 else 0
    stage_preference = stage_score if stage_score > 0 else 0
    row_count = tabular_data_row_count(path)
    desired_distance = 0
    if desired_stage is not None:
        desired_distance = -abs(stage_score - desired_stage) if stage_score else -10_000
    return (name_score, stage_match, explicit_stage, desired_distance, stage_preference, row_count, path.name.lower())


def sample_name_score(path: Path) -> int:
    """Score how clearly a filename indicates a sample-submission file."""
    stem = tabular_stem(path).lower()
    compact = re.sub(r"[^a-z0-9]+", "", stem)
    tokens = _name_tokens(stem)
    output_tokens = set(SAMPLE_OUTPUT_NAME_TOKENS)
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


def sample_stage_score(path: Path) -> int:
    """Extract stage/phase/round number from filename for ranking."""
    matches = _SAMPLE_STAGE_PATTERN.findall(path.name.lower())
    if not matches:
        return 0
    return max(int(value) for value in matches)


def resolve_desired_submission_stage() -> int | None:
    raw = (
        os.environ.get("KAGGLEBOT_SUBMISSION_STAGE") or os.environ.get("KAGGLEBOT_SAMPLE_SUBMISSION_STAGE") or ""
    ).strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def tabular_file_has_data_rows(path: Path) -> bool:
    """Return whether a delimited tabular file includes at least one data row."""
    if not _is_text_tabular_suffix(path):
        return _structured_tabular_data_row_count(path) > 0
    non_empty = 0
    try:
        with _open_text_tabular(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                non_empty += 1
                if non_empty >= 2:
                    return True
    except OSError:
        return True
    return False


def tabular_data_row_count(path: Path) -> int:
    """Return the number of non-empty data rows in a tabular file."""
    if not _is_text_tabular_suffix(path):
        return _structured_tabular_data_row_count(path)
    non_empty = 0
    try:
        with _open_text_tabular(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                non_empty += 1
    except OSError:
        return 0
    return max(0, non_empty - 1)


def tabular_data_row_count_capped(path: Path, *, cap: int = 10) -> int | None:
    """Return supported tabular data rows, capped at cap + 1 for tiny-contract checks."""
    if not path.is_file():
        return None
    if _is_text_tabular_suffix(path):
        return _text_tabular_data_row_count_capped(path, cap=cap)
    if not is_tabular_sample_path(path):
        return None
    row_count = _structured_tabular_row_count_capped(path, cap=cap)
    if row_count is None:
        return None
    return cap + 1 if row_count > cap else row_count


def _text_tabular_data_row_count_capped(path: Path, *, cap: int) -> int | None:
    non_empty = 0
    try:
        with _open_text_tabular(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                non_empty += 1
                data_rows = max(0, non_empty - 1)
                if data_rows > cap:
                    return cap + 1
    except OSError:
        return None
    return max(0, non_empty - 1)


def _json_lines_data_row_count_capped(path: Path, *, cap: int) -> int | None:
    non_empty = 0
    try:
        with _open_text_tabular(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                non_empty += 1
                if non_empty > cap:
                    return cap + 1
    except OSError:
        return None
    return non_empty


def csv_has_two_or_more_columns(path: Path) -> bool:
    return tabular_file_has_two_or_more_columns(path)


def tabular_file_has_two_or_more_columns(path: Path) -> bool:
    if not _is_text_tabular_suffix(path):
        return _structured_tabular_file_has_two_or_more_columns(path)
    return tabular_text_has_two_or_more_columns(path)


def tabular_text_has_two_or_more_columns(path: Path) -> bool:
    try:
        with _open_text_tabular(path, newline="") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = next(csv.reader([line], delimiter=sniff_tabular_text_delimiter(path, sample_line=line)))
                return len(row) >= 2
    except OSError:
        return False
    return False


def open_tabular_text(path: Path, *, newline: str | None = None):
    """Open a text-like tabular file, including supported compression suffixes."""

    suffix = tabular_suffix(path)
    if suffix.endswith(".zip"):
        return _open_zip_tabular_text(path, suffix=suffix, newline=newline)
    return open_compressed_text(path, suffix=suffix, encoding="utf-8", errors="ignore", newline=newline)


@contextmanager
def _open_zip_tabular_text(path: Path, *, suffix: str, newline: str | None = None):
    with zipfile.ZipFile(path, "r") as archive:
        member = _select_zip_tabular_member(archive, suffix=suffix)
        with archive.open(member, "r") as binary:
            with TextIOWrapper(binary, encoding="utf-8", errors="ignore", newline=newline) as handle:
                yield handle


def _select_zip_tabular_member(archive: zipfile.ZipFile, *, suffix: str) -> str:
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


def zip_wrapped_tabular_base_suffix(suffix: str) -> str:
    return suffix[: -len(".zip")] if suffix.endswith(".zip") else suffix


def read_zip_tabular_member_bytes(path: Path, *, suffix: str | None = None) -> bytes:
    resolved_suffix = suffix or tabular_suffix(path)
    with zipfile.ZipFile(path, "r") as archive:
        member = _select_zip_tabular_member(archive, suffix=resolved_suffix)
        return archive.read(member)


def _open_text_tabular(path: Path, *, newline: str | None = None):
    return open_tabular_text(path, newline=newline)


def _structured_tabular_file_has_two_or_more_columns(path: Path) -> bool:
    frame = _read_structured_tabular_sample(path)
    if frame is None:
        return False
    return len(frame.columns) >= 2 and not frame.empty


def _structured_tabular_data_row_count(path: Path) -> int:
    frame = _read_structured_tabular_sample(path)
    if frame is None:
        return 0
    return len(frame)


def _structured_tabular_full_row_count(path: Path) -> int | None:
    suffix = tabular_suffix(path)
    if suffix in _SQLITE_TABULAR_SUFFIXES:
        return _sqlite_table_row_count(path)
    if suffix in DUCKDB_TABULAR_SUFFIXES:
        return duckdb_table_row_count(path)
    frame = _read_structured_tabular_frame(path)
    if frame is None:
        return None
    return int(len(frame))


def _structured_tabular_row_count_capped(path: Path, *, cap: int) -> int | None:
    suffix = tabular_suffix(path)
    base_suffix = zip_wrapped_tabular_base_suffix(suffix)
    if suffix in _SQLITE_TABULAR_SUFFIXES:
        return _sqlite_table_row_count(path)
    if suffix in DUCKDB_TABULAR_SUFFIXES:
        return duckdb_table_row_count(path)
    if is_json_lines_tabular_suffix(suffix):
        return _json_lines_data_row_count_capped(path, cap=cap)
    if base_suffix in TABULAR_PARQUET_SUFFIXES:
        row_count = _parquet_row_count(path, suffix=suffix)
        if row_count is not None:
            return row_count
    return _structured_tabular_full_row_count(path)


def _read_structured_tabular_sample(path: Path):
    return _finalize_tabular_frame_or_none(_read_raw_structured_tabular_sample(path))


def _read_raw_structured_tabular_sample(path: Path):
    try:
        import pandas as pd
    except Exception:  # noqa: BLE001
        return None
    suffix = tabular_suffix(path)
    base_suffix = zip_wrapped_tabular_base_suffix(suffix)
    try:
        if base_suffix in TABULAR_PARQUET_SUFFIXES:
            return pd.read_parquet(_binary_tabular_source(path, suffix=suffix)).head(1)
        elif base_suffix == ".orc":
            return pd.read_orc(_binary_tabular_source(path, suffix=suffix)).head(1)
        elif base_suffix in TABULAR_ARROW_IPC_SUFFIXES:
            return pd.read_feather(_binary_tabular_source(path, suffix=suffix)).head(1)
        elif base_suffix == ".avro":
            return read_avro_tabular_frame(path).head(1)
        elif suffix in TABULAR_HDF_SUFFIXES:
            return _read_hdf_sample(path)
        elif is_json_lines_tabular_suffix(suffix):
            return _read_json_lines_frame(path, nrows=1)
        elif _is_json_table_suffix(suffix):
            try:
                return _read_json_tabular_frame(path).head(1)
            except ValueError:
                return pd.read_json(path).head(1)
        elif _is_yaml_table_suffix(suffix):
            return _read_yaml_tabular_frame(path).head(1)
        elif base_suffix in TABULAR_EXCEL_SUFFIXES:
            return pd.read_excel(_binary_tabular_source(path, suffix=suffix)).head(1)
        elif base_suffix in TABULAR_EXCEL_INPUT_ONLY_SUFFIXES:
            return pd.read_excel(_binary_tabular_source(path, suffix=suffix), engine="pyxlsb").head(1)
        elif suffix.startswith(TABULAR_SVMLIGHT_SUFFIX_PREFIXES):
            return read_svmlight_tabular_frame(path).head(1)
        elif suffix.startswith(TABULAR_FIXED_WIDTH_SUFFIX_PREFIXES):
            return read_fixed_width_tabular_frame(path).head(1)
        elif base_suffix in TABULAR_STATA_SUFFIXES:
            return pd.read_stata(_binary_tabular_source(path, suffix=suffix)).head(1)
        elif base_suffix in TABULAR_SAS_SUFFIXES:
            return pd.read_sas(_binary_tabular_source(path, suffix=suffix), format=_sas_format_for_suffix(suffix)).head(
                1
            )
        elif base_suffix in TABULAR_SPSS_SUFFIXES:
            return pd.read_spss(_binary_tabular_source(path, suffix=suffix)).head(1)
        elif suffix in TABULAR_MATLAB_SUFFIXES:
            return read_mat_tabular_frame(path).head(1)
        elif suffix in TABULAR_RDATA_SUFFIXES:
            return read_rdata_tabular_frame(path).head(1)
        elif suffix in TABULAR_NETCDF_SUFFIXES:
            return read_netcdf_tabular_frame(path).head(1)
        elif suffix in TABULAR_NUMPY_SUFFIXES:
            return read_numpy_tabular_frame(path).head(1)
        elif suffix in TABULAR_FITS_SUFFIXES:
            return read_fits_tabular_frame(path).head(1)
        elif suffix in TABULAR_ANNDATA_SUFFIXES:
            return read_h5ad_tabular_frame(path).head(1)
        elif suffix in TABULAR_LOOM_SUFFIXES:
            return read_loom_tabular_frame(path).head(1)
        elif suffix in TABULAR_GEOPACKAGE_SUFFIXES:
            return read_geopackage_tabular_frame(path, nrows=1)
        elif suffix in TABULAR_SHAPEFILE_SUFFIXES:
            return read_shapefile_tabular_frame(path, nrows=1)
        elif suffix in TABULAR_KML_SUFFIXES:
            return read_kml_tabular_frame(path).head(1)
        elif suffix in TABULAR_ARFF_SUFFIXES:
            return read_arff_tabular_frame(path).head(1)
        elif suffix.startswith(TABULAR_HTML_SUFFIX_PREFIXES):
            return read_html_tabular_frame(path).head(1)
        elif suffix.startswith(".xml"):
            return read_xml_tabular_frame(path).head(1)
        elif base_suffix in _TABULAR_PICKLE_SUFFIXES or suffix in _COMPRESSED_TABULAR_PICKLE_SUFFIXES:
            return pd.read_pickle(_binary_tabular_source(path, suffix=suffix)).head(1)
        elif suffix in _SQLITE_TABULAR_SUFFIXES:
            return _read_sqlite_sample(path)
        elif suffix in DUCKDB_TABULAR_SUFFIXES:
            return read_duckdb_tabular_frame(path, nrows=1)
        else:
            return None
    except Exception:  # noqa: BLE001
        return None


def _read_structured_tabular_frame(path: Path):
    return _finalize_tabular_frame_or_none(_read_raw_structured_tabular_frame(path))


def _read_raw_structured_tabular_frame(path: Path):
    try:
        import pandas as pd
    except Exception:  # noqa: BLE001
        return None
    suffix = tabular_suffix(path)
    base_suffix = zip_wrapped_tabular_base_suffix(suffix)
    try:
        if base_suffix in TABULAR_PARQUET_SUFFIXES:
            return pd.read_parquet(_binary_tabular_source(path, suffix=suffix))
        if base_suffix == ".orc":
            return pd.read_orc(_binary_tabular_source(path, suffix=suffix))
        if base_suffix in TABULAR_ARROW_IPC_SUFFIXES:
            return pd.read_feather(_binary_tabular_source(path, suffix=suffix))
        if base_suffix == ".avro":
            return read_avro_tabular_frame(path)
        if suffix in TABULAR_HDF_SUFFIXES:
            return _read_hdf_frame(path)
        if is_json_lines_tabular_suffix(suffix):
            return _read_json_lines_frame(path)
        if _is_json_table_suffix(suffix):
            try:
                return _read_json_tabular_frame(path)
            except ValueError:
                return pd.read_json(path)
        if _is_yaml_table_suffix(suffix):
            return _read_yaml_tabular_frame(path)
        if base_suffix in TABULAR_EXCEL_SUFFIXES:
            return pd.read_excel(_binary_tabular_source(path, suffix=suffix))
        if base_suffix in TABULAR_EXCEL_INPUT_ONLY_SUFFIXES:
            return pd.read_excel(_binary_tabular_source(path, suffix=suffix), engine="pyxlsb")
        if suffix.startswith(TABULAR_SVMLIGHT_SUFFIX_PREFIXES):
            return read_svmlight_tabular_frame(path)
        if suffix.startswith(TABULAR_FIXED_WIDTH_SUFFIX_PREFIXES):
            return read_fixed_width_tabular_frame(path)
        if base_suffix in TABULAR_STATA_SUFFIXES:
            return pd.read_stata(_binary_tabular_source(path, suffix=suffix))
        if base_suffix in TABULAR_SAS_SUFFIXES:
            return pd.read_sas(_binary_tabular_source(path, suffix=suffix), format=_sas_format_for_suffix(suffix))
        if base_suffix in TABULAR_SPSS_SUFFIXES:
            return pd.read_spss(_binary_tabular_source(path, suffix=suffix))
        if suffix in TABULAR_MATLAB_SUFFIXES:
            return read_mat_tabular_frame(path)
        if suffix in TABULAR_RDATA_SUFFIXES:
            return read_rdata_tabular_frame(path)
        if suffix in TABULAR_NETCDF_SUFFIXES:
            return read_netcdf_tabular_frame(path)
        if suffix in TABULAR_NUMPY_SUFFIXES:
            return read_numpy_tabular_frame(path)
        if suffix in TABULAR_FITS_SUFFIXES:
            return read_fits_tabular_frame(path)
        if suffix in TABULAR_ANNDATA_SUFFIXES:
            return read_h5ad_tabular_frame(path)
        if suffix in TABULAR_LOOM_SUFFIXES:
            return read_loom_tabular_frame(path)
        if suffix in TABULAR_GEOPACKAGE_SUFFIXES:
            return read_geopackage_tabular_frame(path)
        if suffix in TABULAR_SHAPEFILE_SUFFIXES:
            return read_shapefile_tabular_frame(path)
        if suffix in TABULAR_KML_SUFFIXES:
            return read_kml_tabular_frame(path)
        if suffix in TABULAR_ARFF_SUFFIXES:
            return read_arff_tabular_frame(path)
        if suffix.startswith(TABULAR_HTML_SUFFIX_PREFIXES):
            return read_html_tabular_frame(path)
        if suffix.startswith(".xml"):
            return read_xml_tabular_frame(path)
        if base_suffix in _TABULAR_PICKLE_SUFFIXES or suffix in _COMPRESSED_TABULAR_PICKLE_SUFFIXES:
            return pd.read_pickle(_binary_tabular_source(path, suffix=suffix))
        if suffix in _SQLITE_TABULAR_SUFFIXES:
            return _read_sqlite_sample(path)
        if suffix in DUCKDB_TABULAR_SUFFIXES:
            return read_duckdb_tabular_frame(path)
        return None
    except Exception:  # noqa: BLE001
        return None


def _read_json_tabular_frame(path: Path):
    with open_tabular_text(path) as handle:
        return _json_object_to_frame(json.load(handle))


def _read_yaml_tabular_frame(path: Path):
    import yaml

    with open_tabular_text(path) as handle:
        return _json_object_to_frame(yaml.safe_load(handle) or [])


def read_avro_tabular_frame(path: Path):
    from fastavro import reader

    handle = _binary_tabular_source(path, suffix=tabular_suffix(path))
    if isinstance(handle, Path):
        with handle.open("rb") as raw:
            return _avro_reader_to_frame(reader(raw))
    return _avro_reader_to_frame(reader(handle))


def _avro_reader_to_frame(avro_reader):
    import pandas as pd

    schema = getattr(avro_reader, "writer_schema", {}) or {}
    records = list(avro_reader)
    columns = [str(field.get("name")) for field in schema.get("fields", []) if field.get("name") is not None]
    return pd.DataFrame(records, columns=columns or None)


def _binary_tabular_source(path: Path, *, suffix: str):
    if suffix.endswith(".zip"):
        return BytesIO(read_zip_tabular_member_bytes(path, suffix=suffix))
    return path


def _sas_format_for_suffix(suffix: str) -> str | None:
    base_suffix = zip_wrapped_tabular_base_suffix(suffix)
    if base_suffix in {".xpt", ".xport"}:
        return "xport"
    if base_suffix == ".sas7bdat":
        return "sas7bdat"
    return None


def _read_json_lines_frame(path: Path, *, nrows: int | None = None):
    import pandas as pd

    with open_tabular_text(path) as handle:
        return pd.read_json(StringIO(handle.read()), lines=True, nrows=nrows)


def read_mat_tabular_frame(path: Path):
    """Read a MATLAB .mat file as a DataFrame when it contains table-like arrays."""
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
    columns_by_length: dict[int, dict[str, object]] = {}
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


def read_rdata_tabular_frame(path: Path):
    """Read an RDS/RData object as a DataFrame when it contains a table."""
    try:
        import pyreadr
    except Exception as exc:  # noqa: BLE001
        raise ValueError("pyreadr is required to read RDS/RData tabular files") from exc

    result = pyreadr.read_r(path)
    for value in result.values():
        if hasattr(value, "columns") and hasattr(value, "head"):
            return value
    raise ValueError(f"R data file does not contain a tabular object: {path}")


def read_netcdf_tabular_frame(path: Path):
    """Read a NetCDF file as a DataFrame when it contains table-like variables."""
    import numpy as np
    import pandas as pd
    from scipy.io import netcdf_file

    variables = {}
    try:
        with netcdf_file(path, mode="r", mmap=False) as dataset:
            for name, variable in dataset.variables.items():
                variables[str(name)] = np.array(variable.data).copy()
    except Exception as exc:
        fallback = _read_native_hdf_tabular_frame(path)
        if fallback is not None:
            return fallback
        raise exc
    frame = netcdf_variables_to_column_frame(variables, np=np, pd=pd)
    if frame is None:
        fallback = _read_native_hdf_tabular_frame(path)
        if fallback is not None:
            return fallback
        raise ValueError(f"NetCDF file does not contain table-like variables: {path}")
    return frame


def netcdf_variables_to_column_frame(payload, *, np, pd):
    columns_by_length: dict[int, dict[str, object]] = {}
    for name, value in payload.items():
        columns = netcdf_value_to_columns(str(name), value, np=np)
        if not columns:
            continue
        row_count = len(next(iter(columns.values())))
        columns_by_length.setdefault(row_count, {}).update(columns)
    if not columns_by_length:
        return None
    columns = max(columns_by_length.values(), key=len)
    return pd.DataFrame(columns)


def netcdf_value_to_columns(name: str, value, *, np) -> dict[str, object] | None:
    array = np.asarray(value)
    if array.shape == () or 0 in array.shape or array.dtype.kind == "O":
        return None
    if array.dtype.kind == "S":
        decoded = netcdf_decode_bytes_array(array, np=np)
        return {safe_netcdf_column_name(name): decoded} if decoded is not None else None
    if array.ndim > 1 and array.shape[-1] == 1:
        array = np.squeeze(array, axis=-1)
    if array.ndim == 1:
        return {safe_netcdf_column_name(name): array.tolist()}
    if array.ndim == 2:
        base = safe_netcdf_column_name(name)
        return {f"{base}_{idx}": array[:, idx].tolist() for idx in range(array.shape[1])}
    return None


def netcdf_decode_bytes_array(array, *, np) -> list[str] | None:
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


def read_fits_tabular_frame(path: Path):
    """Read a FITS binary/table HDU or simple 1D/2D image array as a DataFrame."""
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


def fits_data_to_frame(name: str, data):
    import numpy as np
    import pandas as pd

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
    import numpy as np

    array = np.asarray(values)
    if array.shape == () or 0 in array.shape or array.dtype.kind == "O":
        return None
    if array.dtype.kind == "S":
        decoded = fits_decode_bytes_array(array, np=np)
        return {safe_fits_column_name(name): decoded} if decoded is not None else None
    if array.ndim > 1 and array.shape[-1] == 1:
        array = np.squeeze(array, axis=-1)
    if array.ndim == 1:
        return {safe_fits_column_name(name): array.tolist()}
    if array.ndim == 2:
        base = safe_fits_column_name(name)
        return {f"{base}_{idx}": array[:, idx].tolist() for idx in range(array.shape[1])}
    return None


def fits_decode_bytes_array(array, *, np) -> list[str] | None:
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


def read_numpy_tabular_frame(path: Path):
    """Read a 1D, 2D, or structured NumPy array/archive as a DataFrame."""
    import numpy as np

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


def numpy_archive_columns_to_frame(arrays: dict[str, object]):
    import numpy as np
    import pandas as pd

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


def numpy_array_to_frame(name: str, raw_array, *, source_path: Path | None = None):
    import numpy as np
    import pandas as pd

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
    base = path.name[: -len(path.suffix)] if path.suffix else path.stem
    candidates = [
        path.with_name(f"{base}{suffix}")
        for suffix in _NUMPY_COLUMN_TEXT_SIDECAR_SUFFIXES + _NUMPY_COLUMN_JSON_SIDECAR_SUFFIXES
    ]
    candidates.extend(path.with_name(name) for name in _NUMPY_GENERIC_COLUMN_SIDECAR_NAMES)
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


def normalize_numpy_column_names(columns: Sequence[object], width: int) -> list[str] | None:
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
    import numpy as np

    decoded = np.asarray(array)
    if decoded.dtype.kind == "S":
        return decoded.astype(str)
    return decoded


def safe_numpy_column_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_") or "value"


def read_arff_tabular_frame(path: Path):
    """Read a Weka/OpenML ARFF file as a DataFrame."""
    import pandas as pd
    from scipy.io import arff

    with open_tabular_text(path) as handle:
        data, _metadata = arff.loadarff(handle)
    frame = pd.DataFrame(data)
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(_decode_arff_value)
    return frame


def _decode_arff_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def read_html_tabular_frame(path: Path):
    """Read the most table-like HTML table from an HTML file."""
    import pandas as pd

    with open_tabular_text(path) as handle:
        tables = pd.read_html(StringIO(handle.read()))
    candidates = [table for table in tables if table.shape[1] > 0]
    if not candidates:
        raise ValueError(f"HTML file does not contain table-like data: {path}")
    return max(candidates, key=lambda table: (table.shape[0] * table.shape[1], table.shape[1]))


def read_xml_tabular_frame(path: Path):
    """Read an XML table, including compressed XML variants."""
    import pandas as pd

    with open_tabular_text(path) as handle:
        return pd.read_xml(StringIO(handle.read()), parser="etree")


def write_xml_tabular_frame(frame, path: Path) -> None:
    """Write an XML table, including compressed XML variants."""
    payload = frame.to_xml(index=False, parser="etree").encode("utf-8")
    write_compressed_bytes(path, payload, suffix=tabular_suffix(path))


def read_svmlight_tabular_frame(path: Path):
    """Read a LibSVM/SVMLight sparse feature file as a DataFrame."""
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_svmlight_file

    with open_tabular_binary(path) as handle:
        matrix, target = load_svmlight_file(handle)
    columns = [f"feature_{idx + 1}" for idx in range(matrix.shape[1])]
    frame = pd.DataFrame.sparse.from_spmatrix(matrix, columns=columns)
    if path_mentions_role(path, "train") or np.asarray(target).any():
        frame.insert(0, "target", target)
    return frame


def read_fixed_width_tabular_frame(path: Path):
    """Read a fixed-width text table as a DataFrame, including compressed variants."""
    import pandas as pd

    with open_tabular_text(path) as handle:
        return pd.read_fwf(StringIO(handle.read()))


@contextmanager
def open_tabular_binary(path: Path) -> Iterator[object]:
    suffix = tabular_suffix(path)
    if suffix.endswith(".zip"):
        with zipfile.ZipFile(path, "r") as archive:
            member = _select_zip_tabular_member(archive, suffix=suffix)
            yield BytesIO(archive.read(member))
        return
    with open_compressed_binary(path, suffix=suffix) as handle:
        yield handle


def _is_json_table_suffix(suffix: str) -> bool:
    return suffix in _JSON_TABLE_SUFFIXES


def _is_yaml_table_suffix(suffix: str) -> bool:
    return suffix in _YAML_TABLE_SUFFIXES


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


def _geojson_feature_collection_to_frame(value: dict[str, object]):
    import pandas as pd

    if str(value.get("type", "")).lower() != "featurecollection":
        return None
    features = value.get("features")
    if not isinstance(features, list):
        return None
    records: list[dict[str, object]] = []
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


def _read_sqlite_sample(path: Path):
    try:
        import pandas as pd
    except Exception:  # noqa: BLE001
        return None
    try:
        with sqlite3.connect(path) as conn:
            table = _first_sqlite_table(conn)
            if table is None:
                return None
            return pd.read_sql_query(f"SELECT * FROM {_quote_sqlite_identifier(table)} LIMIT 1", conn)
    except Exception:  # noqa: BLE001
        return None


def _read_hdf_sample(path: Path):
    frame = _read_hdf_frame(path)
    if frame is None:
        return None
    return frame.head(1)


def _read_hdf_frame(path: Path):
    try:
        return read_hdf_tabular_frame(path)
    except Exception:  # noqa: BLE001
        return None


def read_hdf_tabular_frame(path: Path):
    try:
        import pandas as pd
    except Exception as exc:  # noqa: BLE001
        raise ValueError("pandas is required to read HDF tabular files") from exc
    try:
        return pd.read_hdf(path)
    except (KeyError, ValueError) as exc:
        _close_pytables_open_files()
        try:
            with pd.HDFStore(path, mode="r") as store:
                keys = store.keys()
            if len(keys) == 1:
                return pd.read_hdf(path, key=keys[0])
        except Exception:  # noqa: BLE001
            pass
        fallback = _read_native_hdf_tabular_frame(path)
        if fallback is not None:
            return fallback
        raise exc
    except Exception as exc:  # noqa: BLE001
        _close_pytables_open_files()
        fallback = _read_native_hdf_tabular_frame(path)
        if fallback is not None:
            return fallback
        raise exc


def _close_pytables_open_files() -> None:
    try:
        import tables

        for handle in list(tables.file._open_files._handlers):
            handle.close()
    except Exception:  # noqa: BLE001
        pass


def _read_native_hdf_tabular_frame(path: Path):
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
    row_count: int | None = None
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


def read_h5ad_tabular_frame(path: Path):
    """Read an AnnData .h5ad file as obs metadata plus X feature columns."""
    import h5py
    import pandas as pd

    with h5py.File(path, "r") as handle:
        obs_frame = _hdf_group_to_column_frame(handle["obs"]) if "obs" in handle else None
        x_frame = _h5ad_x_to_frame(handle)
    frame = _combine_same_row_frames(obs_frame, x_frame, pd=pd)
    if frame is not None:
        return frame
    fallback = _read_native_hdf_tabular_frame(path)
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
    columns = _h5ad_var_names(handle, matrix.shape[1])
    if columns is None:
        columns = [f"x_{idx}" for idx in range(matrix.shape[1])]
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
    matrix = (
        sparse.csc_matrix((data, indices, indptr), shape=shape)
        if "csc" in encoding
        else sparse.csr_matrix(
            (data, indices, indptr),
            shape=shape,
        )
    )
    return matrix.toarray()


def _h5ad_var_names(handle, width: int) -> list[str] | None:
    var = handle.get("var")
    if var is None:
        return None
    for key in ("_index", "index", "gene_symbols", "gene_ids", "feature_name", "features"):
        if key not in var:
            continue
        values = _hdf_column_values(var[key], h5py=__import__("h5py"))
        columns = _normalize_hdf_feature_names(values, width)
        if columns is not None:
            return columns
    return None


def read_loom_tabular_frame(path: Path):
    """Read a Loom file as column attributes plus transposed matrix feature columns."""
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
    fallback = _read_native_hdf_tabular_frame(path)
    if fallback is not None:
        return fallback
    raise ValueError(f"Loom file does not contain table-like col_attrs/matrix data: {path}")


def _loom_row_feature_names(handle, width: int) -> list[str] | None:
    row_attrs = handle.get("row_attrs")
    if row_attrs is None:
        return None
    for key in ("Gene", "gene", "GeneName", "gene_name", "Accession", "feature_name"):
        if key not in row_attrs:
            continue
        values = _hdf_column_values(row_attrs[key], h5py=__import__("h5py"))
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


def _finalize_tabular_frame_or_none(frame):
    if frame is None:
        return None
    return _finalize_tabular_frame(frame)


def _finalize_tabular_frame(frame):
    frame.columns = normalize_table_column_names(frame.columns)
    return frame


def _dedupe_column_names(columns: Sequence[str]) -> list[str]:
    counts: dict[str, int] = {}
    deduped = []
    for column in columns:
        base = str(column)
        count = counts.get(base, 0)
        deduped.append(base if count == 0 else f"{base}_{count}")
        counts[base] = count + 1
    return deduped


def _sqlite_table_row_count(path: Path) -> int | None:
    try:
        with sqlite3.connect(path) as conn:
            table = _first_sqlite_table(conn)
            if table is None:
                return None
            row = conn.execute(f"SELECT COUNT(*) FROM {_quote_sqlite_identifier(table)}").fetchone()
    except Exception:  # noqa: BLE001
        return None
    if row is None:
        return None
    return int(row[0])


def duckdb_user_tables(path: Path) -> list[tuple[str, str]]:
    """Return user-visible DuckDB tables/views as (schema, table) pairs."""
    try:
        import duckdb
    except Exception as exc:  # noqa: BLE001
        raise ValueError("duckdb is required to read DuckDB tabular files") from exc

    conn = duckdb.connect(str(path), read_only=True)
    try:
        rows = conn.execute(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
              AND table_type IN ('BASE TABLE', 'VIEW')
            ORDER BY table_schema, table_name
            """
        ).fetchall()
    finally:
        conn.close()
    return [(str(schema), str(table)) for schema, table in rows if str(table).strip()]


def read_duckdb_tabular_frame(
    path: Path,
    *,
    table: tuple[str, str] | None = None,
    nrows: int | None = None,
):
    """Read one user table from a DuckDB database as a pandas DataFrame."""
    try:
        import duckdb
    except Exception as exc:  # noqa: BLE001
        raise ValueError("duckdb is required to read DuckDB tabular files") from exc

    conn = duckdb.connect(str(path), read_only=True)
    try:
        table_name = table or _select_duckdb_table_for_path(path, duckdb_user_tables_from_connection(conn))
        if table_name is None:
            raise ValueError(f"No user tables found in DuckDB database: {path}")
        query = f"SELECT * FROM {_quote_duckdb_qualified_identifier(table_name)}"
        if nrows is not None:
            query += f" LIMIT {max(int(nrows), 0)}"
        return conn.execute(query).df()
    finally:
        conn.close()


def duckdb_table_row_count(path: Path) -> int | None:
    try:
        import duckdb
    except Exception:  # noqa: BLE001
        return None

    conn = duckdb.connect(str(path), read_only=True)
    try:
        table = _select_duckdb_table_for_path(path, duckdb_user_tables_from_connection(conn))
        if table is None:
            return None
        row = conn.execute(f"SELECT COUNT(*) FROM {_quote_duckdb_qualified_identifier(table)}").fetchone()
    except Exception:  # noqa: BLE001
        return None
    finally:
        conn.close()
    if row is None:
        return None
    return int(row[0])


def read_geopackage_tabular_frame(path: Path, *, nrows: int | None = None):
    """Read the first table-like GeoPackage layer as a pandas DataFrame."""
    try:
        import pandas as pd
    except Exception as exc:  # noqa: BLE001
        raise ValueError("pandas is required to read GeoPackage tabular files") from exc

    with sqlite3.connect(path) as conn:
        table = _first_geopackage_table(conn)
        if table is None:
            raise ValueError(f"No feature or attribute tables found in GeoPackage: {path}")
        sql = f"SELECT * FROM {_quote_sqlite_identifier(table)}"
        if nrows is not None:
            sql += f" LIMIT {max(int(nrows), 0)}"
        frame = pd.read_sql_query(sql, conn)
    return _decode_geopackage_blob_columns(frame)


def _first_geopackage_table(conn: sqlite3.Connection) -> str | None:
    preferred = []
    try:
        rows = conn.execute(
            """
            SELECT table_name, data_type
            FROM gpkg_contents
            ORDER BY table_name
            """
        ).fetchall()
        for table_name, data_type in rows:
            table = str(table_name)
            if str(data_type).lower() in {"features", "attributes"} and _sqlite_table_exists(conn, table):
                preferred.append(table)
    except sqlite3.DatabaseError:
        pass
    if preferred:
        return preferred[0]
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type IN ('table', 'view')
        ORDER BY name
        """
    ).fetchall()
    for row in rows:
        table = str(row[0])
        if not _is_geopackage_metadata_table(table):
            return table
    return None


def _sqlite_table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type IN ('table', 'view')
          AND name = ?
        LIMIT 1
        """,
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


def read_shapefile_tabular_frame(path: Path, *, nrows: int | None = None):
    """Read a Shapefile/DBF attribute table as a pandas DataFrame."""
    try:
        import pandas as pd
    except Exception as exc:  # noqa: BLE001
        raise ValueError("pandas is required to read Shapefile attribute tables") from exc

    dbf_path = _dbf_path_for_shapefile(path)
    fields, records = _read_dbf_records(dbf_path, nrows=nrows)
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


def _read_dbf_records(path: Path, *, nrows: int | None = None):
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
                    _parse_dbf_value(
                        raw_record[offset : offset + length],
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


def _read_dbf_fields(handle, *, encoding: str) -> list[dict[str, object]]:
    fields = []
    while True:
        marker = handle.read(1)
        if not marker:
            break
        if marker == b"\r":
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


def _dbf_encoding_for_path(path: Path) -> str:
    cpg_path = path.with_suffix(".cpg")
    if cpg_path.exists():
        raw_encoding = cpg_path.read_text(encoding="ascii", errors="ignore").strip()
        if raw_encoding:
            if raw_encoding == "65001":
                return "utf-8"
            return raw_encoding
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


def read_kml_tabular_frame(path: Path):
    """Read KML/KMZ Placemark records as a pandas DataFrame."""
    from xml.etree import ElementTree as ET

    import pandas as pd

    root = ET.fromstring(_read_kml_payload(path))
    placemarks = [node for node in root.iter() if _xml_local_name(node.tag) == "Placemark"]
    records = [
        _kml_placemark_to_record(placemark, index=idx, element_tree=ET) for idx, placemark in enumerate(placemarks)
    ]
    return pd.DataFrame(records)


def _read_kml_payload(path: Path) -> bytes:
    suffix = tabular_suffix(path)
    if suffix == ".kmz":
        with zipfile.ZipFile(path, "r") as archive:
            members = [
                member
                for member in archive.infolist()
                if not member.is_dir()
                and not member.filename.startswith("__MACOSX/")
                and member.filename.lower().endswith(".kml")
            ]
            if not members:
                raise ValueError(f"KMZ archive does not contain a KML document: {path}")
            members.sort(key=lambda member: (Path(member.filename).name.lower() != "doc.kml", member.filename.lower()))
            return archive.read(members[0])
    with open_compressed_binary(path, suffix=suffix) as handle:
        return handle.read()


def _kml_placemark_to_record(placemark, *, index: int, element_tree):
    record: dict[str, object] = {"placemark_index": index}
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
        record["geometry"] = element_tree.tostring(geometry, encoding="unicode")
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


def _xml_local_name(tag: object) -> str:
    text = str(tag)
    return text.rsplit("}", 1)[-1] if "}" in text else text


def _parquet_row_count(path: Path, *, suffix: str | None = None) -> int | None:
    try:
        import pyarrow.parquet as pq

        resolved_suffix = suffix or tabular_suffix(path)
        metadata = pq.ParquetFile(_binary_tabular_source(path, suffix=resolved_suffix)).metadata
    except Exception:  # noqa: BLE001
        return None
    if metadata is None:
        return None
    return int(metadata.num_rows)


def _first_sqlite_table(conn: sqlite3.Connection) -> str | None:
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type IN ('table', 'view')
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()
    return str(rows[0][0]) if rows else None


def _quote_sqlite_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def duckdb_user_tables_from_connection(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
          AND table_type IN ('BASE TABLE', 'VIEW')
        ORDER BY table_schema, table_name
        """
    ).fetchall()
    return [(str(schema), str(table)) for schema, table in rows if str(table).strip()]


def select_duckdb_tables_for_materialization(tables: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    role_tables = [table for table in tables if _database_table_has_role_name(_duckdb_table_label(table))]
    if role_tables:
        return role_tables
    if len(tables) <= 3:
        return list(tables)
    return []


def _select_duckdb_table_for_path(path: Path, tables: Sequence[tuple[str, str]]) -> tuple[str, str] | None:
    if not tables:
        return None
    if len(tables) == 1:
        return tables[0]
    path_tokens = _database_name_tokens(path.stem)
    ranked: list[tuple[int, str, tuple[str, str]]] = []
    for table in tables:
        label = _duckdb_table_label(table)
        table_tokens = _database_name_tokens(label)
        score = len(path_tokens & table_tokens) * 20
        if _database_table_has_role_name(label):
            score += 5
        ranked.append((score, label, table))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked[0][2]


def _database_table_has_role_name(table: str) -> bool:
    lowered = table.lower().replace("-", "_")
    compact = lowered.replace("_", "")
    tokens = (
        "sample_submission",
        "samplesubmission",
        "submission",
        "train",
        "training",
        "test",
        "labels",
        "label",
        "target",
        "features",
        "feature",
        "data",
    )
    return any(token in lowered or token in compact for token in tokens)


def _database_name_tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", value.lower()) if token}


def _duckdb_table_label(table: tuple[str, str]) -> str:
    schema, name = table
    if schema and schema != "main":
        return f"{schema}.{name}"
    return name


def _quote_duckdb_qualified_identifier(table: tuple[str, str]) -> str:
    schema, name = table
    if schema and schema != "main":
        return f"{_quote_duckdb_identifier(schema)}.{_quote_duckdb_identifier(name)}"
    return _quote_duckdb_identifier(name)


def _quote_duckdb_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def sniff_tabular_text_delimiter(path: Path, *, sample_line: str | None = None) -> str:
    suffix = tabular_suffix(path)
    default = default_delimited_text_separator(suffix)
    line = sample_line
    if line is None:
        try:
            with _open_text_tabular(path) as handle:
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
        return r"\s+"
    return default


def _looks_space_delimited(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if any(separator in stripped for separator in (",", "\t", ";", "|")):
        return False
    parts = stripped.split()
    if len(parts) < 2:
        return False
    return all(part.strip() for part in parts)
