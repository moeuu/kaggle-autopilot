from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from io import BytesIO, StringIO
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from kagglebot.asset_modality import (
    ASSET_COLLECTION_DIR_NAMES,
    DATA_ASSET_SUFFIXES,
    artifact_stem,
    artifact_suffix,
    is_data_asset_path,
)
from kagglebot.baseline_tokens import ASSET_LABEL_TABLE_TOKENS, ID_LIKE_COLUMN_NAMES, TEXT_PREDICTION_NAME_TOKENS
from kagglebot.compression_suffixes import write_compressed_bytes as _write_compressed_payload
from kagglebot.role_tokens import ROLE_ALIASES, TEST_INFERENCE_ROLE_TOKENS
from kagglebot.submission_extension_hints import ARCHIVE_SUBMISSION_SUFFIXES, NON_TABULAR_SUBMISSION_SUFFIXES
from kagglebot.submission_format import load_submission_format_hint
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
    TABULAR_TEXT_SUFFIXES,
    default_delimited_text_separator,
    duckdb_user_tables,
    is_json_lines_tabular_suffix,
    preferred_tabular_submission_suffix,
    read_duckdb_tabular_frame,
    read_rdata_tabular_frame,
    read_zip_tabular_member_bytes,
    select_duckdb_tables_for_materialization,
    zip_wrapped_tabular_base_suffix,
)
from kagglebot.submission_sample_discovery import (
    component_mentions_role as _component_mentions_role,
)
from kagglebot.submission_sample_discovery import (
    is_tabular_data_path as _is_tabular_path,
)
from kagglebot.submission_sample_discovery import (
    open_tabular_text as _open_tabular_text,
)
from kagglebot.submission_sample_discovery import (
    path_mentions_role as _shared_path_mentions_role,
)
from kagglebot.submission_sample_discovery import (
    read_arff_tabular_frame as _read_arff_tabular_frame,
)
from kagglebot.submission_sample_discovery import (
    read_fits_tabular_frame as _read_fits_tabular_frame,
)
from kagglebot.submission_sample_discovery import (
    read_fixed_width_tabular_frame as _read_fixed_width_tabular_frame,
)
from kagglebot.submission_sample_discovery import (
    read_geopackage_tabular_frame as _read_geopackage_tabular_frame,
)
from kagglebot.submission_sample_discovery import (
    read_h5ad_tabular_frame as _read_h5ad_tabular_frame,
)
from kagglebot.submission_sample_discovery import (
    read_hdf_tabular_frame as _read_hdf_tabular_frame,
)
from kagglebot.submission_sample_discovery import (
    read_html_tabular_frame as _read_html_tabular_frame,
)
from kagglebot.submission_sample_discovery import (
    read_kml_tabular_frame as _read_kml_tabular_frame,
)
from kagglebot.submission_sample_discovery import (
    read_loom_tabular_frame as _read_loom_tabular_frame,
)
from kagglebot.submission_sample_discovery import (
    read_mat_tabular_frame as _read_mat_tabular_frame,
)
from kagglebot.submission_sample_discovery import (
    read_netcdf_tabular_frame as _read_netcdf_tabular_frame,
)
from kagglebot.submission_sample_discovery import (
    read_numpy_tabular_frame as _read_numpy_tabular_frame,
)
from kagglebot.submission_sample_discovery import (
    read_shapefile_tabular_frame as _read_shapefile_tabular_frame,
)
from kagglebot.submission_sample_discovery import (
    read_svmlight_tabular_frame as _read_svmlight_tabular_frame,
)
from kagglebot.submission_sample_discovery import (
    read_xml_tabular_frame as _read_xml_tabular_frame,
)
from kagglebot.submission_sample_discovery import (
    roleless_stem as _shared_roleless_stem,
)
from kagglebot.submission_sample_discovery import (
    select_sample_submission_path as _select_sample_submission_path,
)
from kagglebot.submission_sample_discovery import (
    sniff_tabular_text_delimiter as _sniff_tabular_text_delimiter,
)
from kagglebot.submission_sample_discovery import (
    tabular_data_row_count as _tabular_data_row_count,
)
from kagglebot.submission_sample_discovery import (
    tabular_file_has_data_rows as _tabular_file_has_data_rows,
)
from kagglebot.submission_sample_discovery import (
    tabular_stem as _shared_tabular_stem,
)
from kagglebot.submission_sample_discovery import (
    tabular_suffix as _tabular_suffix,
)
from kagglebot.submission_sample_discovery import (
    write_xml_tabular_frame as _write_xml_tabular_frame,
)
from kagglebot.submission_templates import build_submission_template_for_test
from kagglebot.table_columns import frame_with_normalized_table_columns, normalize_table_column_names
from kagglebot.validators import extract_data_archives


@dataclass(frozen=True)
class CompetitionData:
    train: pd.DataFrame
    test: pd.DataFrame
    sample: pd.DataFrame
    id_column: str | None
    target_column: str
    feature_columns: list[str]
    task: str
    prediction_kind: str
    target_columns: list[str] = field(default_factory=list)
    task_by_target: dict[str, str] = field(default_factory=dict)
    prediction_kind_by_target: dict[str, str] = field(default_factory=dict)
    data_dir: Path | None = None
    sample_weight_column: str | None = None
    group_column: str | None = None
    time_column: str | None = None


def find_competition_files(data_dir: Path) -> tuple[Path, Path, Path]:
    extract_data_archives(data_dir, overwrite=False)
    materialize_sqlite_tables(data_dir)
    materialize_duckdb_tables(data_dir)
    files = _find_tabular_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No tabular files found under {data_dir}.")

    sample_path = _select_sample_submission_path(files)

    if sample_path is None:
        synthesized_sample = _maybe_synthesize_sample_submission(data_dir)
        if synthesized_sample is not None:
            sample_path = synthesized_sample
        else:
            raise FileNotFoundError("Unable to locate sample submission file in competition data.")

    train_path, test_path = _select_train_test_paths(files=files, sample_path=sample_path)
    if train_path is None or test_path is None:
        synthesized = _synthesize_train_test_from_assets(data_dir=data_dir, sample_path=sample_path)
        if synthesized is not None:
            return synthesized
        raise FileNotFoundError("Unable to locate train/test files in competition data.")

    return train_path, test_path, sample_path


def ensure_sample_submission(data_dir: Path) -> Path | None:
    """
    Ensure a usable sample submission exists for this competition.

    Preference order:
    1) Use an existing sample-submission file shipped with the competition data
       (including multi-stage files like `SampleSubmissionStage1.csv`).
    2) If no usable sample file exists, try to synthesize one from
       `context/submission_format.md` plus discovered test IDs (e.g., filenames under
       `images/test`).
    """
    if not data_dir.exists():
        return None
    extract_data_archives(data_dir, overwrite=False)
    materialize_sqlite_tables(data_dir)
    materialize_duckdb_tables(data_dir)
    try:
        files = _find_tabular_files(data_dir)
    except OSError:
        files = []
    candidate = _select_sample_submission_path(files)
    if candidate is not None and _tabular_file_has_data_rows(candidate):
        return candidate
    return _maybe_synthesize_sample_submission(data_dir)


def infer_submission_layout(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
) -> tuple[str | None, list[str], list[str]]:
    sample_cols = list(sample.columns)
    train_cols = list(train.columns)
    test_cols = list(test.columns)
    train_minus_test = [col for col in train_cols if col not in test_cols]
    target_cols = _infer_target_columns(train=train, test=test, sample=sample, train_minus_test=train_minus_test)
    id_col = _pick_id_column(sample_cols=sample_cols, target_cols=target_cols, test_cols=test_cols)

    # Feature columns must be present in BOTH train and test; otherwise they cannot be
    # used for inference and will break downstream schema/validation logic.
    common_non_target = [col for col in train_cols if col in test_cols and col not in target_cols]
    feature_cols = list(common_non_target)

    if id_col and id_col in feature_cols:
        feature_cols.remove(id_col)
    if not feature_cols:
        # If removing the id column would leave no features, keep the common columns
        # (including id) as a last-ditch fallback.
        feature_cols = list(common_non_target)

    return id_col, target_cols, feature_cols


def infer_target(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> tuple[str, str, list[str]]:
    id_col, target_cols, feature_cols = infer_submission_layout(train=train, test=test, sample=sample)
    if not target_cols:
        raise ValueError("Unable to infer target columns from train/test/sample files.")
    resolved_id = id_col or ""
    return resolved_id, target_cols[0], feature_cols


def infer_task(y: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(y):
        return "classification"
    if pd.api.types.is_object_dtype(y) or isinstance(y.dtype, pd.CategoricalDtype):
        return "classification"
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    nunique = y.nunique(dropna=True)
    if nunique <= 20:
        return "classification"
    if nunique / max(len(y), 1) <= 0.05:
        return "classification"
    return "regression"


def task_for_prediction_kind(task: str, prediction_kind: str) -> str:
    if prediction_kind == "text":
        return "text"
    return task


def infer_prediction_kind(sample_target: pd.Series, *, column_name: str | None = None) -> str:
    if _looks_like_text_prediction(sample_target, column_name=column_name):
        return "text"
    if pd.api.types.is_float_dtype(sample_target) or pd.api.types.is_complex_dtype(sample_target):
        return "probability"
    if pd.api.types.is_numeric_dtype(sample_target):
        values = pd.to_numeric(sample_target, errors="coerce").dropna().to_numpy()
        if values.size and np.isin(np.unique(values), np.array([0.0, 1.0])).all():
            return "class"
        return "continuous"
    return "class"


def _looks_like_text_prediction(sample_target: pd.Series, *, column_name: str | None) -> bool:
    lowered_name = str(column_name or sample_target.name or "").strip().lower()
    if any(token in lowered_name for token in TEXT_PREDICTION_NAME_TOKENS):
        return True
    values = sample_target.dropna().astype(str).str.strip()
    if values.empty:
        return False
    empty_ratio = float((values == "").mean())
    if empty_ratio >= 0.8:
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


def _looks_like_multi_label_target(target: pd.Series, *, column_name: str) -> bool:
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
        labels = _split_multi_label_value(value, allow_whitespace=strong_name)
        if len(labels) < 2:
            continue
        multi_count += 1
        atomic_labels.update(labels)
    if float(multi_count / len(values)) < 0.6:
        return False
    return len(atomic_labels) >= 2


def _split_multi_label_value(value: str, *, allow_whitespace: bool) -> list[str]:
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


def _infer_prediction_kind_for_target(
    *,
    sample: pd.DataFrame,
    id_col: str | None,
    target_col: str,
    target_cols: list[str],
) -> str:
    if target_col in sample.columns:
        return infer_prediction_kind(sample[target_col], column_name=target_col)
    prediction_cols = [col for col in sample.columns if col != id_col]
    if (
        len(target_cols) == 1
        and len(prediction_cols) >= 2
        and all(pd.api.types.is_numeric_dtype(sample[col]) for col in prediction_cols)
    ):
        return "probability_columns"
    return "continuous"


def _refine_numeric_multi_column_prediction_kind(
    *,
    sample: pd.DataFrame,
    id_col: str | None,
    task: str,
    prediction_kind: str,
) -> str:
    if prediction_kind != "probability_columns" or task == "classification":
        return prediction_kind
    prediction_cols = [str(col) for col in sample.columns if col != id_col]
    if _looks_like_prediction_interval_columns(prediction_cols):
        return "prediction_interval_columns"
    if _looks_like_quantile_prediction_columns(prediction_cols):
        return "quantile_columns"
    return "continuous_columns"


def _looks_like_prediction_interval_columns(prediction_cols: Sequence[str]) -> bool:
    compact_cols = {re.sub(r"[^a-z0-9]+", "", str(col).lower()) for col in prediction_cols}
    lower_tokens = {"lower", "lo", "low", "lwr", "lowerbound", "lowerci", "lowerlimit"}
    upper_tokens = {"upper", "hi", "high", "upr", "upperbound", "upperci", "upperlimit"}
    return bool(compact_cols & lower_tokens) and bool(compact_cols & upper_tokens)


def _looks_like_quantile_prediction_columns(prediction_cols: Sequence[str]) -> bool:
    return sum(1 for col in prediction_cols if _quantile_from_prediction_column(col) is not None) >= 2


def _quantile_from_prediction_column(name: object) -> float | None:
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
    raw = match.group(1)
    try:
        value = float(raw)
    except ValueError:
        return None
    if value > 1.0:
        value /= 100.0
    if 0.0 < value < 1.0:
        return value
    return None


def load_competition_data(data_dir: Path, *, target_column_override: str | None = None) -> CompetitionData:
    train_path, test_path, sample_path = find_competition_files(data_dir)
    train = _read_table(train_path)
    test = _read_table(test_path)
    sample = _read_table(sample_path)
    train, test = _align_train_test_column_case_to_sample(train=train, test=test, sample=sample)

    if target_column_override and target_column_override not in train.columns:
        merged = _maybe_merge_train_labels(
            data_dir=data_dir,
            train_path=train_path,
            test_path=test_path,
            sample_path=sample_path,
            train=train,
            test=test,
            sample=sample,
            target_column_override=target_column_override,
        )
        if merged is not None:
            train = merged

    if target_column_override and target_column_override in train.columns:
        id_col, inferred_targets, feature_cols = infer_submission_layout(train=train, test=test, sample=sample)
        target_cols = [target_column_override]
        if id_col and id_col in feature_cols:
            feature_cols = [c for c in feature_cols if c != id_col]
        if target_column_override not in inferred_targets:
            feature_cols = [c for c in train.columns if c != target_column_override and c != id_col]
    else:
        id_col, target_cols, feature_cols = infer_submission_layout(train=train, test=test, sample=sample)
        if not target_cols:
            merged = _maybe_merge_train_labels(
                data_dir=data_dir,
                train_path=train_path,
                test_path=test_path,
                sample_path=sample_path,
                train=train,
                test=test,
                sample=sample,
                target_column_override=None,
            )
            if merged is not None:
                train = merged
                id_col, target_cols, feature_cols = infer_submission_layout(train=train, test=test, sample=sample)
        if not target_cols:
            id_col = _pick_id_column(sample_cols=list(sample.columns), target_cols=[], test_cols=list(test.columns))
            target_cols = _infer_unlabeled_numeric_score_columns(sample=sample, id_col=id_col)
            if target_cols:
                feature_cols = [col for col in train.columns if col in test.columns and col != id_col]
    if not target_cols:
        raise ValueError("Unable to infer target columns from train/test/sample files.")
    if id_col:
        train, test, sample = _preserve_id_column_values(
            train=train,
            test=test,
            sample=sample,
            train_path=train_path,
            test_path=test_path,
            sample_path=sample_path,
            id_col=id_col,
        )
        train, test = _align_train_test_column_case_to_sample(train=train, test=test, sample=sample)

    target_col = target_cols[0]
    sample_weight_column = infer_sample_weight_column(
        train=train,
        test=test,
        feature_cols=feature_cols,
        target_cols=target_cols,
        id_col=id_col,
    )
    if sample_weight_column is not None:
        feature_cols = [col for col in feature_cols if col != sample_weight_column]
    group_column = infer_group_column(train=train, feature_cols=feature_cols)
    time_column = infer_time_column(train=train, test=test, feature_cols=feature_cols)
    task_by_target = {
        col: (
            "regression"
            if col in train.columns and _target_should_use_continuous_model(col, train, test, feature_cols)
            else infer_task(train[col])
            if col in train.columns
            else "unsupervised"
        )
        for col in target_cols
    }
    prediction_kind_by_target = {
        col: _infer_prediction_kind_for_target(sample=sample, id_col=id_col, target_col=col, target_cols=target_cols)
        for col in target_cols
    }
    prediction_kind_by_target = {
        col: (
            "ordinal"
            if col in train.columns
            and _looks_like_ordinal_target(
                train[col],
                column_name=col,
                feature_cols=feature_cols,
            )
            else "continuous"
            if col in train.columns
            and (
                _looks_like_learning_to_rank_target(
                    target_col=col,
                    train=train,
                    test=test,
                    feature_cols=feature_cols,
                )
                or _looks_like_named_continuous_numeric_target(
                    train[col],
                    column_name=col,
                    feature_cols=feature_cols,
                )
                or _looks_like_bounded_regression_target(train[col], column_name=col)
            )
            else prediction_kind_by_target[col]
        )
        for col in target_cols
    }
    prediction_kind_by_target = {
        col: (
            "probability"
            if col in train.columns
            and task_by_target[col] == "classification"
            and prediction_kind_by_target[col] == "class"
            and _looks_like_probability_score_column(col)
            else prediction_kind_by_target[col]
        )
        for col in target_cols
    }
    prediction_kind_by_target = {
        col: (
            "text"
            if col in train.columns
            and col in sample.columns
            and prediction_kind_by_target[col] == "class"
            and looks_like_natural_language_text_target(train[col])
            and not _looks_like_multi_label_target(train[col], column_name=col)
            else prediction_kind_by_target[col]
        )
        for col in target_cols
    }
    prediction_kind_by_target = {
        col: (
            "text"
            if col in train.columns
            and col in sample.columns
            and prediction_kind_by_target[col] == "class"
            and _looks_like_multi_label_target(train[col], column_name=col)
            else prediction_kind_by_target[col]
        )
        for col in target_cols
    }
    prediction_kind_by_target = {
        col: (
            "multi_label_columns"
            if col in train.columns
            and col not in sample.columns
            and prediction_kind_by_target[col] == "probability_columns"
            and _looks_like_multi_label_target(train[col], column_name=col)
            else prediction_kind_by_target[col]
        )
        for col in target_cols
    }
    prediction_kind_by_target = {
        col: _refine_numeric_multi_column_prediction_kind(
            sample=sample,
            id_col=id_col,
            task=task_by_target[col],
            prediction_kind=prediction_kind_by_target[col],
        )
        for col in target_cols
    }
    task_by_target = {
        col: task_for_prediction_kind(task_by_target[col], prediction_kind_by_target[col]) for col in target_cols
    }
    task = task_by_target[target_col]
    prediction_kind = prediction_kind_by_target[target_col]

    return CompetitionData(
        train=train,
        test=test,
        sample=sample,
        id_column=id_col,
        target_column=target_col,
        target_columns=target_cols,
        feature_columns=feature_cols,
        task=task,
        prediction_kind=prediction_kind,
        task_by_target=task_by_target,
        prediction_kind_by_target=prediction_kind_by_target,
        data_dir=data_dir,
        sample_weight_column=sample_weight_column,
        group_column=group_column,
        time_column=time_column,
    )


def infer_time_column(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> str | None:
    scored: list[tuple[float, int, str]] = []
    for index, col in enumerate(feature_cols):
        if col not in train.columns or col not in test.columns:
            continue
        name_score = _time_column_name_score(str(col))
        if name_score <= 0:
            continue
        holdout_score = _future_temporal_holdout_score(train[col], test[col])
        if holdout_score <= 0:
            continue
        scored.append((name_score + holdout_score, -index, str(col)))
    if not scored:
        return None
    return max(scored)[2]


def _time_column_name_score(name: str) -> float:
    tokens = set(_column_tokens(name))
    compact = "".join(_column_tokens(name))
    if compact in {"dateblocknum", "daynum", "weekofyear"}:
        return 1.0
    if tokens & {"date", "datetime", "timestamp"}:
        return 0.95
    if tokens & {"day", "daynum", "week", "month", "year"}:
        return 0.8
    if "time" in tokens and compact not in {"survivaltime", "eventtime", "timetoevent"}:
        return 0.7
    return 0.0


def _future_temporal_holdout_score(train_series: pd.Series, test_series: pd.Series) -> float:
    if _has_future_ordinal_holdout(train_series, test_series):
        return 1.0
    train_dates = _parse_temporal_series(train_series)
    test_dates = _parse_temporal_series(test_series)
    if train_dates.empty or test_dates.empty:
        return 0.0
    return 1.0 if bool(test_dates.min() > train_dates.max()) else 0.0


def _parse_temporal_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series.dropna().head(500), errors="coerce", utc=True)
    else:
        sample = series.dropna().astype(str).head(500)
        if sample.empty:
            return pd.Series(dtype="datetime64[ns, UTC]")
        parsed = pd.to_datetime(sample, errors="coerce", utc=True)
    if len(parsed) == 0 or float(pd.Series(parsed).notna().mean()) < 0.8:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return pd.Series(parsed).dropna()


def _has_future_ordinal_holdout(train_series: pd.Series, test_series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(train_series):
        return False
    train_values = pd.to_numeric(train_series, errors="coerce").dropna()
    test_values = pd.to_numeric(test_series, errors="coerce").dropna()
    if train_values.empty or test_values.empty:
        return False
    if int(train_values.nunique(dropna=True)) < 3:
        return False
    return float(test_values.min()) > float(train_values.max())


def infer_sample_weight_column(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    id_col: str | None,
) -> str | None:
    excluded = {str(col) for col in target_cols}
    if id_col is not None:
        excluded.add(str(id_col))
    scored: list[tuple[float, int, str]] = []
    for index, col in enumerate(str(column) for column in train.columns):
        if col in excluded or col not in train.columns:
            continue
        name_score = _sample_weight_column_name_score(col, train_only=col not in test.columns)
        if name_score <= 0:
            continue
        values_score = _sample_weight_values_score(train[col])
        if values_score <= 0:
            continue
        feature_penalty = 0.15 if col in feature_cols and col in test.columns else 0.0
        scored.append((name_score + values_score - feature_penalty, -index, col))
    if not scored:
        return None
    return max(scored)[2]


def _sample_weight_column_name_score(name: str, *, train_only: bool) -> float:
    tokens = set(_column_tokens(name))
    compact = "".join(_column_tokens(name))
    explicit_names = {
        "sampleweight",
        "rowweight",
        "observationweight",
        "instanceweight",
        "exampleweight",
        "evalweight",
        "evaluationweight",
        "metricweight",
        "targetweight",
    }
    if compact in explicit_names:
        return 1.0
    if "weight" in tokens and tokens & {
        "sample",
        "row",
        "observation",
        "instance",
        "example",
        "eval",
        "evaluation",
        "metric",
    }:
        return 0.95
    if compact in {"weight", "weights"} and train_only:
        return 0.72
    return 0.0


def _sample_weight_values_score(series: pd.Series) -> float:
    numeric = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    if len(numeric) < 2:
        return 0.0
    if float((numeric < 0).mean()) > 0.0:
        return 0.0
    if float((numeric > 0).mean()) < 0.8:
        return 0.0
    return 0.35 if int(numeric.nunique(dropna=True)) > 1 else 0.2


def infer_group_column(*, train: pd.DataFrame, feature_cols: list[str]) -> str | None:
    scored: list[tuple[float, int, str]] = []
    for index, col in enumerate(feature_cols):
        if col not in train.columns:
            continue
        name_score = _group_column_name_score(str(col))
        if name_score <= 0:
            continue
        values_score = _group_column_values_score(train[col])
        if values_score <= 0:
            continue
        scored.append((name_score + values_score, -index, str(col)))
    if not scored:
        return None
    return max(scored)[2]


def _group_column_name_score(name: str) -> float:
    tokens = set(_column_tokens(name))
    compact = "".join(_column_tokens(name))
    if compact in {"id", "rowid", "index", "targetid", "predictionid"}:
        return 0.0
    entity_tokens = {
        "account",
        "author",
        "case",
        "center",
        "customer",
        "device",
        "document",
        "donor",
        "entity",
        "group",
        "household",
        "installation",
        "patient",
        "participant",
        "session",
        "site",
        "source",
        "study",
        "subject",
        "user",
        "visit",
    }
    if tokens & entity_tokens:
        score = 0.8
        if "id" in tokens or compact.endswith("id"):
            score += 0.2
        return score
    if compact in {
        "patientid",
        "subjectid",
        "participantid",
        "sessionid",
        "visitid",
        "caseid",
        "studyid",
        "siteid",
        "userid",
        "customerid",
        "accountid",
        "householdid",
        "donorid",
        "authorid",
        "deviceid",
        "sourceid",
        "groupid",
    }:
        return 1.0
    return 0.0


def _group_column_values_score(series: pd.Series) -> float:
    sample = series.dropna().head(1000)
    if len(sample) < 4:
        return 0.0
    unique_count = int(sample.nunique(dropna=True))
    if unique_count < 2 or unique_count >= len(sample):
        return 0.0
    value_counts = sample.astype(str).value_counts(dropna=True)
    if value_counts.empty or int(value_counts.max()) < 2:
        return 0.0
    repeat_ratio = 1.0 - (float(unique_count) / float(len(sample)))
    if repeat_ratio < 0.05:
        return 0.0
    return min(0.5, repeat_ratio)


def _column_tokens(name: str) -> list[str]:
    return [token for token in re.split(r"[^A-Za-z0-9]+", name.lower()) if token]


def _align_train_test_column_case_to_sample(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_by_lower = {str(col).lower(): str(col) for col in sample.columns}
    train = _rename_columns_case_insensitive(frame=train, desired_by_lower=sample_by_lower)
    test = _rename_columns_case_insensitive(frame=test, desired_by_lower=sample_by_lower)
    test = _align_test_column_case_to_train(train=train, test=test)
    return train, test


def _align_test_column_case_to_train(*, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    train_by_lower = {str(col).lower(): str(col) for col in train.columns}
    return _rename_columns_case_insensitive(frame=test, desired_by_lower=train_by_lower)


def _rename_columns_case_insensitive(
    *,
    frame: pd.DataFrame,
    desired_by_lower: Mapping[str, str],
) -> pd.DataFrame:
    existing = {str(col) for col in frame.columns}
    rename: dict[str, str] = {}
    for col in frame.columns:
        source = str(col)
        desired = desired_by_lower.get(source.lower())
        if desired is None or desired == source:
            continue
        if desired in existing:
            continue
        rename[source] = desired
    return frame.rename(columns=rename) if rename else frame


def write_submission(
    sample: pd.DataFrame,
    test: pd.DataFrame,
    preds,
    *,
    id_column: str | None,
    target_column: str | None = None,
    target_columns: Sequence[str] | None = None,
    output_path: Path,
) -> Path:
    resolved_targets = _resolve_target_columns(
        sample=sample,
        id_column=id_column,
        target_column=target_column,
        target_columns=target_columns,
        preds=preds,
    )
    submission = build_submission_template_for_test(
        sample_submission=sample,
        test_df=test,
        id_col=id_column,
        target_cols=resolved_targets,
    )
    pred_table = _normalize_prediction_table(preds=preds, target_columns=resolved_targets, row_count=len(test))

    for col in resolved_targets:
        submission[col] = _align_prediction_column(
            sample=submission,
            test=test,
            values=pred_table[col],
            id_column=id_column,
            target_column=col,
        )
        submission[col] = _coerce_prediction_dtype(
            sample[col],
            submission[col],
        )

    return write_table(submission, output_path)


def _find_tabular_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and _is_tabular_path(p)]


def _select_train_test_paths(*, files: list[Path], sample_path: Path) -> tuple[Path | None, Path | None]:
    candidates = [
        path
        for path in files
        if path != sample_path
        and (".kagglebot_cache" not in {part.lower() for part in path.parts} or _is_database_materialized_path(path))
    ]
    train_candidates = [path for path in candidates if _path_mentions_role(path, "train")]
    test_candidates = [path for path in candidates if _path_mentions_role(path, "test")]
    if not train_candidates or not test_candidates:
        schema_pair = _select_train_test_paths_by_schema(candidates=candidates, sample_path=sample_path)
        if schema_pair is not None:
            return schema_pair
        return train_candidates[0] if train_candidates else None, test_candidates[0] if test_candidates else None

    best: tuple[int, str, Path, Path] | None = None
    for train_path in train_candidates:
        for test_path in test_candidates:
            score = _train_test_pair_score(train_path=train_path, test_path=test_path, sample_path=sample_path)
            tie_break = f"{train_path.as_posix()}\0{test_path.as_posix()}"
            candidate = (score, tie_break, train_path, test_path)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        return train_candidates[0], test_candidates[0]
    return best[2], best[3]


def _select_train_test_paths_by_schema(*, candidates: list[Path], sample_path: Path) -> tuple[Path, Path] | None:
    try:
        sample_head = _read_table_schema_head(sample_path)
    except Exception:  # noqa: BLE001
        return None
    sample_cols = [str(col) for col in sample_head.columns]
    if len(sample_cols) < 2:
        return None
    sample_id = sample_cols[0]
    sample_targets = sample_cols[1:]

    profiles = [_schema_file_profile(path, sample_id=sample_id, sample_targets=sample_targets) for path in candidates]
    profiles = [profile for profile in profiles if profile is not None]
    if len(profiles) < 2:
        return None

    best: tuple[int, str, Path, Path] | None = None
    for train_profile in profiles:
        for test_profile in profiles:
            if train_profile["path"] == test_profile["path"]:
                continue
            score = _schema_train_test_pair_score(
                train_profile=train_profile,
                test_profile=test_profile,
                sample_path=sample_path,
                sample_id=sample_id,
                sample_targets=sample_targets,
            )
            if score <= 0:
                continue
            train_path = train_profile["path"]
            test_path = test_profile["path"]
            tie_break = f"{train_path.as_posix()}\0{test_path.as_posix()}"
            candidate = (score, tie_break, train_path, test_path)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        return None
    return best[2], best[3]


def _schema_file_profile(
    path: Path,
    *,
    sample_id: str,
    sample_targets: Sequence[str],
) -> dict[str, object] | None:
    try:
        head = _read_table_schema_head(path)
    except Exception:  # noqa: BLE001
        return None
    columns = [str(col) for col in head.columns]
    if len(columns) < 2:
        return None
    column_set = set(columns)
    return {
        "path": path,
        "columns": column_set,
        "has_sample_id": sample_id in column_set,
        "sample_target_count": sum(1 for col in sample_targets if col in column_set),
        "row_count": _safe_tabular_row_count(path),
    }


def _schema_train_test_pair_score(
    *,
    train_profile: dict[str, object],
    test_profile: dict[str, object],
    sample_path: Path,
    sample_id: str,
    sample_targets: Sequence[str],
) -> int:
    train_cols = train_profile["columns"]
    test_cols = test_profile["columns"]
    if not isinstance(train_cols, set) or not isinstance(test_cols, set):
        return 0

    common_cols = train_cols & test_cols
    if not common_cols:
        return 0

    train_target_count = sum(1 for col in sample_targets if col in train_cols and col not in test_cols)
    test_target_count = sum(1 for col in sample_targets if col in test_cols)
    if train_target_count <= 0:
        return 0

    score = len(common_cols) * 30
    score += train_target_count * 120
    if sample_id in train_cols:
        score += 20
    if sample_id in test_cols:
        score += 50
    if test_target_count:
        score -= test_target_count * 150

    sample_rows = _safe_tabular_row_count(sample_path)
    test_rows = test_profile.get("row_count")
    train_rows = train_profile.get("row_count")
    if isinstance(sample_rows, int) and isinstance(test_rows, int) and sample_rows > 0:
        if test_rows == sample_rows:
            score += 80
        elif abs(test_rows - sample_rows) <= max(2, int(sample_rows * 0.05)):
            score += 30
    if isinstance(train_rows, int) and isinstance(test_rows, int) and train_rows > test_rows:
        score += 5

    train_path = train_profile["path"]
    test_path = test_profile["path"]
    if isinstance(train_path, Path) and _path_mentions_role(train_path, "train"):
        score += 40
    if isinstance(test_path, Path) and _path_mentions_role(test_path, "test"):
        score += 40
    return score


def _safe_tabular_row_count(path: Path) -> int | None:
    try:
        return _tabular_data_row_count(path)
    except Exception:  # noqa: BLE001
        return None


def _is_sqlite_materialized_path(path: Path) -> bool:
    parts = [part.lower() for part in path.parts]
    return ".kagglebot_cache" in parts and "sqlite" in parts


def _is_database_materialized_path(path: Path) -> bool:
    parts = [part.lower() for part in path.parts]
    return ".kagglebot_cache" in parts and bool({"sqlite", "duckdb"} & set(parts))


def _path_mentions_role(path: Path, role: str) -> bool:
    return _shared_path_mentions_role(path, role)


def _train_test_pair_score(*, train_path: Path, test_path: Path, sample_path: Path) -> int:
    try:
        train_head = _read_table_schema_head(train_path)
        test_head = _read_table_schema_head(test_path)
        sample_head = _read_table_schema_head(sample_path)
    except Exception:  # noqa: BLE001
        return -10_000

    train_cols = set(train_head.columns)
    test_cols = set(test_head.columns)
    sample_cols = list(sample_head.columns)
    common_cols = train_cols & test_cols

    score = len(common_cols) * 20
    if sample_cols and sample_cols[0] in common_cols:
        score += 60
    train_stem = _tabular_stem(train_path).lower()
    test_stem = _tabular_stem(test_path).lower()
    if train_stem == "train":
        score += 40
    if test_stem == "test":
        score += 40
    if _roleless_stem(train_path, "train") == _roleless_stem(test_path, "test"):
        score += 35
    if "feature" in train_stem or "data" in train_stem:
        score += 15
    if "label" in train_stem or "target" in train_stem:
        score -= 50
    return score


def _roleless_stem(path: Path, role: str) -> str:
    return _shared_roleless_stem(path, role)


def _tabular_stem(path: Path) -> str:
    return _shared_tabular_stem(path)


def _read_table_head(path: Path, *, nrows: int = 5) -> pd.DataFrame:
    return _finalize_table_frame(_read_raw_table_head(path, nrows=nrows))


def _read_table_schema_head(path: Path, *, nrows: int = 5) -> pd.DataFrame:
    """Read only enough rows to compare schemas without inferring text value types."""
    if _tabular_suffix(path) in TABULAR_TEXT_SUFFIXES:
        return _finalize_table_frame(_read_text_tabular_frame(path, dtype=str, nrows=nrows))
    return _read_table_head(path, nrows=nrows)


def _read_raw_table_head(path: Path, *, nrows: int = 5) -> pd.DataFrame:
    suffix = _tabular_suffix(path)
    base_suffix = zip_wrapped_tabular_base_suffix(suffix)
    if base_suffix in TABULAR_PARQUET_SUFFIXES:
        return pd.read_parquet(_binary_tabular_source(path, suffix=suffix)).head(nrows)
    if base_suffix == ".orc":
        return pd.read_orc(_binary_tabular_source(path, suffix=suffix)).head(nrows)
    if suffix in TABULAR_HDF_SUFFIXES:
        return _read_hdf_table(path).head(nrows)
    if suffix in TABULAR_ANNDATA_SUFFIXES:
        return _read_h5ad_tabular_frame(path).head(nrows)
    if suffix in TABULAR_LOOM_SUFFIXES:
        return _read_loom_tabular_frame(path).head(nrows)
    if suffix in TABULAR_GEOPACKAGE_SUFFIXES:
        return _read_geopackage_tabular_frame(path, nrows=nrows)
    if suffix in TABULAR_SHAPEFILE_SUFFIXES:
        return _read_shapefile_tabular_frame(path, nrows=nrows)
    if suffix in TABULAR_KML_SUFFIXES:
        return _read_kml_tabular_frame(path).head(nrows)
    if base_suffix in TABULAR_ARROW_IPC_SUFFIXES:
        return pd.read_feather(_binary_tabular_source(path, suffix=suffix)).head(nrows)
    if base_suffix == ".avro":
        return _read_avro_table(path).head(nrows)
    if suffix in _SQLITE_SUFFIXES:
        return _read_sqlite_table(path, nrows=nrows)
    if suffix in _DUCKDB_SUFFIXES:
        return read_duckdb_tabular_frame(path, nrows=nrows)
    if base_suffix in TABULAR_EXCEL_SUFFIXES:
        return pd.read_excel(_binary_tabular_source(path, suffix=suffix)).head(nrows)
    if base_suffix in TABULAR_EXCEL_INPUT_ONLY_SUFFIXES:
        return pd.read_excel(_binary_tabular_source(path, suffix=suffix), engine="pyxlsb").head(nrows)
    if suffix.startswith(TABULAR_SVMLIGHT_SUFFIX_PREFIXES):
        return _read_svmlight_tabular_frame(path).head(nrows)
    if suffix.startswith(TABULAR_FIXED_WIDTH_SUFFIX_PREFIXES):
        return _read_fixed_width_tabular_frame(path).head(nrows)
    if base_suffix in TABULAR_STATA_SUFFIXES:
        return pd.read_stata(_binary_tabular_source(path, suffix=suffix)).head(nrows)
    if base_suffix in TABULAR_SAS_SUFFIXES:
        return pd.read_sas(_binary_tabular_source(path, suffix=suffix), format=_sas_format_for_suffix(suffix)).head(
            nrows
        )
    if base_suffix in TABULAR_SPSS_SUFFIXES:
        return pd.read_spss(_binary_tabular_source(path, suffix=suffix)).head(nrows)
    if suffix in TABULAR_MATLAB_SUFFIXES:
        return _read_mat_tabular_frame(path).head(nrows)
    if suffix in TABULAR_RDATA_SUFFIXES:
        return read_rdata_tabular_frame(path).head(nrows)
    if suffix in TABULAR_NETCDF_SUFFIXES:
        return _read_netcdf_tabular_frame(path).head(nrows)
    if suffix in TABULAR_NUMPY_SUFFIXES:
        return _read_numpy_tabular_frame(path).head(nrows)
    if suffix in TABULAR_FITS_SUFFIXES:
        return _read_fits_tabular_frame(path).head(nrows)
    if suffix in TABULAR_ARFF_SUFFIXES:
        return _read_arff_tabular_frame(path).head(nrows)
    if suffix.startswith(TABULAR_HTML_SUFFIX_PREFIXES):
        return _read_html_tabular_frame(path).head(nrows)
    if suffix.startswith(".xml"):
        return _read_xml_tabular_frame(path).head(nrows)
    if base_suffix in TABULAR_PICKLE_SUFFIXES:
        return pd.read_pickle(_binary_tabular_source(path, suffix=suffix)).head(nrows)
    if _is_json_lines_suffix(suffix):
        return pd.read_json(StringIO(_read_compressed_text(path)), lines=True, nrows=nrows)
    if suffix in TABULAR_STRUCTURED_SUFFIXES:
        return _read_json_table(path, nrows=nrows)
    if suffix in TABULAR_TEXT_SUFFIXES:
        return _read_text_tabular_frame(path, nrows=nrows)
    return pd.read_csv(path, nrows=nrows)


def read_table(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    """Read a tabular artifact using the format implied by the path suffix."""
    if nrows is not None:
        return _read_table_head(path, nrows=nrows)
    return _finalize_table_frame(_read_raw_table(path))


def _read_raw_table(path: Path) -> pd.DataFrame:
    suffix = _tabular_suffix(path)
    base_suffix = zip_wrapped_tabular_base_suffix(suffix)
    if base_suffix in TABULAR_PARQUET_SUFFIXES:
        return pd.read_parquet(_binary_tabular_source(path, suffix=suffix))
    if base_suffix == ".orc":
        return pd.read_orc(_binary_tabular_source(path, suffix=suffix))
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
        return pd.read_feather(_binary_tabular_source(path, suffix=suffix))
    if base_suffix == ".avro":
        return _read_avro_table(path)
    if suffix in _SQLITE_SUFFIXES:
        return _read_sqlite_table(path)
    if suffix in _DUCKDB_SUFFIXES:
        return read_duckdb_tabular_frame(path)
    if base_suffix in TABULAR_EXCEL_SUFFIXES:
        return pd.read_excel(_binary_tabular_source(path, suffix=suffix))
    if base_suffix in TABULAR_EXCEL_INPUT_ONLY_SUFFIXES:
        return pd.read_excel(_binary_tabular_source(path, suffix=suffix), engine="pyxlsb")
    if suffix.startswith(TABULAR_SVMLIGHT_SUFFIX_PREFIXES):
        return _read_svmlight_tabular_frame(path)
    if suffix.startswith(TABULAR_FIXED_WIDTH_SUFFIX_PREFIXES):
        return _read_fixed_width_tabular_frame(path)
    if base_suffix in TABULAR_STATA_SUFFIXES:
        return pd.read_stata(_binary_tabular_source(path, suffix=suffix))
    if base_suffix in TABULAR_SAS_SUFFIXES:
        return pd.read_sas(_binary_tabular_source(path, suffix=suffix), format=_sas_format_for_suffix(suffix))
    if base_suffix in TABULAR_SPSS_SUFFIXES:
        return pd.read_spss(_binary_tabular_source(path, suffix=suffix))
    if suffix in TABULAR_MATLAB_SUFFIXES:
        return _read_mat_tabular_frame(path)
    if suffix in TABULAR_RDATA_SUFFIXES:
        return read_rdata_tabular_frame(path)
    if suffix in TABULAR_NETCDF_SUFFIXES:
        return _read_netcdf_tabular_frame(path)
    if suffix in TABULAR_NUMPY_SUFFIXES:
        return _read_numpy_tabular_frame(path)
    if suffix in TABULAR_FITS_SUFFIXES:
        return _read_fits_tabular_frame(path)
    if suffix in TABULAR_ARFF_SUFFIXES:
        return _read_arff_tabular_frame(path)
    if suffix.startswith(TABULAR_HTML_SUFFIX_PREFIXES):
        return _read_html_tabular_frame(path)
    if suffix.startswith(".xml"):
        return _read_xml_tabular_frame(path)
    if base_suffix in TABULAR_PICKLE_SUFFIXES:
        return pd.read_pickle(_binary_tabular_source(path, suffix=suffix))
    if suffix in TABULAR_STRUCTURED_SUFFIXES:
        return _read_json_table(path)
    if suffix in TABULAR_TEXT_SUFFIXES:
        return _read_text_tabular_frame(path)
    return pd.read_csv(path)


def _finalize_table_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = normalize_table_column_names(frame.columns)
    return frame


def _read_table(path: Path) -> pd.DataFrame:
    return read_table(path)


def _read_hdf_table(path: Path) -> pd.DataFrame:
    return _read_hdf_tabular_frame(path)


def _read_avro_table(path: Path) -> pd.DataFrame:
    from fastavro import reader

    handle = _binary_tabular_source(path, suffix=_tabular_suffix(path))
    if isinstance(handle, Path):
        with handle.open("rb") as raw:
            return _avro_reader_to_frame(reader(raw))
    return _avro_reader_to_frame(reader(handle))


def _avro_reader_to_frame(avro_reader) -> pd.DataFrame:
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


def _write_avro_table(frame: pd.DataFrame, path: Path) -> None:
    from fastavro import writer

    fields = [{"name": str(column), "type": ["null", _avro_field_type(frame[column])]} for column in frame.columns]
    schema = {"type": "record", "name": "SubmissionRecord", "fields": fields}
    records = _frame_to_avro_records(frame, fields)
    with path.open("wb") as handle:
        writer(handle, schema, records)


def _avro_field_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "long"
    if pd.api.types.is_float_dtype(series):
        return "double"
    return "string"


def _frame_to_avro_records(frame: pd.DataFrame, fields: list[dict[str, object]]) -> list[dict[str, object]]:
    field_types = {}
    for schema_field in fields:
        field_type = schema_field["type"]
        candidates = field_type if isinstance(field_type, list) else [field_type]
        field_types[str(schema_field["name"])] = next(candidate for candidate in candidates if candidate != "null")
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
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _preserve_id_column_values(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    train_path: Path,
    test_path: Path,
    sample_path: Path,
    id_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    del train_path
    test = _read_table_with_string_columns(test_path, columns=[id_col]) if id_col in test.columns else test
    sample = _read_table_with_string_columns(sample_path, columns=[id_col]) if id_col in sample.columns else sample
    return train, test, sample


def _read_table_with_string_columns(path: Path, *, columns: Sequence[str]) -> pd.DataFrame:
    suffix = _tabular_suffix(path)
    string_columns = [str(col) for col in columns if str(col).strip()]
    if not string_columns:
        return _read_table(path)
    if suffix in TABULAR_TEXT_SUFFIXES:
        sep = _sniff_tabular_text_delimiter(path)
        string_frame = _finalize_table_frame(
            _read_text_tabular_frame(path, sep=sep, dtype={col: str for col in string_columns})
        )
        if _frame_has_leading_zero_id_values(string_frame, columns=string_columns):
            return string_frame
    return _read_table(path)


def _frame_has_leading_zero_id_values(frame: pd.DataFrame, *, columns: Sequence[str]) -> bool:
    for col in columns:
        if col not in frame.columns:
            continue
        values = frame[col].dropna().astype(str).str.strip()
        if values.str.fullmatch(r"0\d+").any():
            return True
    return False


def _is_json_lines_suffix(suffix: str) -> bool:
    return is_json_lines_tabular_suffix(suffix)


def _read_json_table(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    suffix = _tabular_suffix(path)
    if _is_json_lines_suffix(suffix):
        return pd.read_json(StringIO(_read_compressed_text(path)), lines=True, nrows=nrows)
    payload = _load_yaml_payload(path) if _is_yaml_suffix(suffix) else _load_json_payload(path)
    frame = _json_payload_to_frame(payload)
    return frame.head(nrows) if nrows is not None else frame


def _is_yaml_suffix(suffix: str) -> bool:
    return suffix.startswith((".yaml", ".yml"))


def _read_text_tabular_frame(
    path: Path,
    *,
    sep: str | None = None,
    dtype: dict[str, object] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    delimiter = sep if sep is not None else _sniff_tabular_text_delimiter(path)
    return pd.read_csv(StringIO(_read_compressed_text(path)), sep=delimiter, dtype=dtype, nrows=nrows)


def _read_compressed_text(path: Path) -> str:
    with _open_tabular_text(path) as handle:
        return handle.read()


def _load_json_payload(path: Path) -> object:
    return json.loads(_read_compressed_text(path))


def _load_yaml_payload(path: Path) -> object:
    import yaml

    return yaml.safe_load(_read_compressed_text(path)) or []


def _json_payload_to_frame(payload: object) -> pd.DataFrame:
    if isinstance(payload, list):
        return _json_records_to_frame(payload)
    if not isinstance(payload, dict):
        raise ValueError("JSON table must be an object or list of records.")

    split_frame = _frame_from_split_orient(payload)
    if split_frame is not None:
        return split_frame

    geojson_frame = _frame_from_geojson_feature_collection(payload)
    if geojson_frame is not None:
        return geojson_frame

    table_data = payload.get("data") if isinstance(payload.get("schema"), dict) else None
    if isinstance(table_data, list):
        return _json_records_to_frame(table_data)

    for key in ("data", "records", "rows", "items", "predictions", "submission", "samples", "values"):
        value = payload.get(key)
        if isinstance(value, list):
            return _json_records_to_frame(value)

    list_items = [(key, value) for key, value in payload.items() if isinstance(value, list)]
    record_lists = [(key, value) for key, value in list_items if _looks_like_json_record_list(value)]
    if len(record_lists) == 1:
        return _json_records_to_frame(record_lists[0][1])
    if list_items and _all_lists_same_length([value for _, value in list_items]):
        return pd.DataFrame({key: value for key, value in list_items})
    if payload and all(not isinstance(value, (dict, list)) for value in payload.values()):
        return pd.DataFrame([payload])

    frame = pd.DataFrame(payload)
    return _flatten_single_mapping_column(frame)


def _frame_from_geojson_feature_collection(payload: dict[str, object]) -> pd.DataFrame | None:
    if str(payload.get("type", "")).lower() != "featurecollection":
        return None
    features = payload.get("features")
    if not isinstance(features, list):
        return None
    records: list[dict[str, object]] = []
    for idx, feature in enumerate(features):
        if not isinstance(feature, Mapping):
            continue
        properties = feature.get("properties")
        record = dict(properties) if isinstance(properties, Mapping) else {}
        if "id" not in record and feature.get("id") is not None:
            record["id"] = feature.get("id")
        geometry = feature.get("geometry")
        if geometry is not None:
            record["geometry"] = json.dumps(geometry, sort_keys=True)
        if record:
            records.append(record)
        else:
            records.append({"feature_index": idx})
    return pd.DataFrame(records)


def _frame_from_split_orient(payload: dict[str, object]) -> pd.DataFrame | None:
    columns = payload.get("columns")
    data = payload.get("data")
    if not isinstance(columns, list) or not isinstance(data, list):
        return None
    if not all(not isinstance(column, (dict, list)) for column in columns):
        return None
    return pd.DataFrame(data, columns=columns)


def _json_records_to_frame(records: list[object]) -> pd.DataFrame:
    if _looks_like_json_record_list(records):
        return pd.json_normalize(records)
    return pd.DataFrame(records)


def _looks_like_json_record_list(records: list[object]) -> bool:
    return bool(records) and all(isinstance(item, Mapping) for item in records)


def _all_lists_same_length(values: Sequence[list[object]]) -> bool:
    if not values:
        return False
    lengths = {len(value) for value in values}
    return len(lengths) == 1


def _flatten_single_mapping_column(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame.columns) != 1:
        return frame
    series = frame.iloc[:, 0].dropna()
    if series.empty or not series.map(lambda value: isinstance(value, Mapping)).all():
        return frame
    return pd.json_normalize(series.tolist())


_SQLITE_SUFFIXES = SQLITE_TABULAR_SUFFIXES
_DUCKDB_SUFFIXES = DUCKDB_TABULAR_SUFFIXES
_SQLITE_ROLE_TOKENS = (
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


def materialize_sqlite_tables(root: Path) -> list[Path]:
    """Export likely competition tables from SQLite files into cache CSVs."""
    if not root.exists():
        return []
    materialized: list[Path] = []
    sqlite_paths = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in _SQLITE_SUFFIXES
        and ".kagglebot_cache" not in {part.lower() for part in path.parts}
    ]
    cache_dir = root / ".kagglebot_cache" / "sqlite"
    for sqlite_path in sqlite_paths:
        try:
            tables = _sqlite_user_tables(sqlite_path)
        except Exception:  # noqa: BLE001
            continue
        selected = _select_sqlite_tables_for_materialization(tables)
        if not selected:
            continue
        cache_dir.mkdir(parents=True, exist_ok=True)
        for table in selected:
            destination = cache_dir / f"{_safe_sqlite_name(sqlite_path.stem)}__{_safe_sqlite_name(table)}.csv"
            try:
                if destination.exists() and destination.stat().st_mtime >= sqlite_path.stat().st_mtime:
                    materialized.append(destination)
                    continue
            except OSError:
                pass
            try:
                frame = _read_sqlite_table(sqlite_path, table=table)
            except Exception:  # noqa: BLE001
                continue
            if frame.empty or len(frame.columns) < 2:
                continue
            frame.to_csv(destination, index=False)
            materialized.append(destination)
    return materialized


def materialize_duckdb_tables(root: Path) -> list[Path]:
    """Export likely competition tables from DuckDB files into cache CSVs."""
    if not root.exists():
        return []
    materialized: list[Path] = []
    duckdb_paths = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in _DUCKDB_SUFFIXES
        and ".kagglebot_cache" not in {part.lower() for part in path.parts}
    ]
    cache_dir = root / ".kagglebot_cache" / "duckdb"
    for duckdb_path in duckdb_paths:
        try:
            tables = duckdb_user_tables(duckdb_path)
        except Exception:  # noqa: BLE001
            continue
        selected = select_duckdb_tables_for_materialization(tables)
        if not selected:
            continue
        cache_dir.mkdir(parents=True, exist_ok=True)
        for table in selected:
            destination = cache_dir / f"{_safe_sqlite_name(duckdb_path.stem)}__{_safe_duckdb_table_name(table)}.csv"
            try:
                if destination.exists() and destination.stat().st_mtime >= duckdb_path.stat().st_mtime:
                    materialized.append(destination)
                    continue
            except OSError:
                pass
            try:
                frame = read_duckdb_tabular_frame(duckdb_path, table=table)
            except Exception:  # noqa: BLE001
                continue
            if frame.empty or len(frame.columns) < 2:
                continue
            frame.to_csv(destination, index=False)
            materialized.append(destination)
    return materialized


def _sqlite_user_tables(path: Path) -> list[str]:
    with sqlite3.connect(path) as conn:
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


def _select_sqlite_tables_for_materialization(tables: Sequence[str]) -> list[str]:
    role_tables = [table for table in tables if _sqlite_table_has_role_name(table)]
    if role_tables:
        return role_tables
    if len(tables) <= 3:
        return list(tables)
    return []


def _sqlite_table_has_role_name(table: str) -> bool:
    lowered = table.lower().replace("-", "_")
    compact = lowered.replace("_", "")
    return any(token in lowered or token in compact for token in _SQLITE_ROLE_TOKENS)


def _read_sqlite_table(path: Path, *, table: str | None = None, nrows: int | None = None) -> pd.DataFrame:
    with sqlite3.connect(path) as conn:
        table_name = table or _select_sqlite_table_for_path(path, _sqlite_user_tables_from_connection(conn))
        if table_name is None:
            raise ValueError(f"No user tables found in SQLite database: {path}")
        sql = f"SELECT * FROM {_quote_sqlite_identifier(table_name)}"
        if nrows is not None:
            sql += f" LIMIT {max(int(nrows), 0)}"
        return pd.read_sql_query(sql, conn)


def _sqlite_user_tables_from_connection(conn: sqlite3.Connection) -> list[str]:
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


def _select_sqlite_table_for_path(path: Path, tables: Sequence[str]) -> str | None:
    if not tables:
        return None
    if len(tables) == 1:
        return str(tables[0])
    path_tokens = _sqlite_name_tokens(path.stem)
    ranked: list[tuple[int, str]] = []
    for table in tables:
        table_tokens = _sqlite_name_tokens(table)
        score = len(path_tokens & table_tokens) * 20
        if _sqlite_table_has_role_name(table):
            score += 5
        ranked.append((score, table))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked[0][1]


def _sqlite_name_tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", value.lower()) if token}


def _quote_sqlite_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _safe_sqlite_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "table"


def _safe_duckdb_table_name(table: tuple[str, str]) -> str:
    schema, name = table
    if schema and schema != "main":
        return f"{_safe_sqlite_name(schema)}__{_safe_sqlite_name(name)}"
    return _safe_sqlite_name(name)


def _maybe_merge_train_labels(
    *,
    data_dir: Path,
    train_path: Path,
    test_path: Path,
    sample_path: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    target_column_override: str | None,
) -> pd.DataFrame | None:
    label_paths = [
        path
        for path in _find_tabular_files(data_dir)
        if path not in {train_path, test_path, sample_path}
        and ".kagglebot_cache" not in path.parts
        and _is_label_table_name(path)
    ]
    for label_path in sorted(label_paths, key=lambda path: path.as_posix()):
        try:
            labels = _read_table(label_path)
        except Exception:  # noqa: BLE001
            continue
        if labels.empty:
            continue
        join_cols = _resolve_label_join_columns(train=train, test=test, sample=sample, labels=labels)
        if join_cols is None:
            continue
        train_join_col, label_join_col = join_cols
        target_cols = _resolve_label_target_columns(
            labels=labels,
            sample=sample,
            train_join_col=train_join_col,
            label_join_col=label_join_col,
            target_column_override=target_column_override,
        )
        if not target_cols:
            continue
        label_subset = labels[[label_join_col, *[source for source, _ in target_cols]]].rename(
            columns={label_join_col: train_join_col, **dict(target_cols)}
        )
        merged = _merge_label_subset(train=train, label_subset=label_subset, join_col=train_join_col)
        if merged.empty:
            continue
        return merged
    return None


def _merge_label_subset(*, train: pd.DataFrame, label_subset: pd.DataFrame, join_col: str) -> pd.DataFrame:
    try:
        merged = train.merge(label_subset, on=join_col, how="inner")
    except ValueError:
        merged = pd.DataFrame()
    if not merged.empty:
        return merged

    key = "__kagglebot_label_join_key__"
    train_keyed = train.copy()
    label_keyed = label_subset.copy()
    train_keyed[key] = train_keyed[join_col].map(_submission_id_alignment_key)
    label_keyed[key] = label_keyed[join_col].map(_submission_id_alignment_key)
    label_keyed = label_keyed.drop(columns=[join_col])
    merged = train_keyed.merge(label_keyed, on=key, how="inner").drop(columns=[key])
    return merged


def _is_label_table_name(path: Path) -> bool:
    stem = path.stem.lower()
    name = path.name.lower()
    if "sample" in name or "submission" in name:
        return False
    return any(token in stem for token in _LABEL_TABLE_TOKENS)


def _resolve_label_join_columns(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[str, str] | None:
    preferred = [str(sample.columns[0])] if len(sample.columns) and _is_id_like_column(sample.columns[0]) else []
    preferred.extend(["id", "ID", "row_id", "filename", "image_id", "file"])
    preferred.extend(str(col) for col in train.columns if col in test.columns)
    train_by_lower = {str(col).lower(): str(col) for col in train.columns}
    test_by_lower = {str(col).lower(): str(col) for col in test.columns}
    label_by_lower = {str(col).lower(): str(col) for col in labels.columns}
    seen: set[str] = set()
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
        if train_col is not None and test_col is not None and _is_id_like_column(train_col):
            label_alias = _resolve_id_like_label_column(labels)
            if label_alias is not None:
                return train_col, label_alias
    return None


_ID_LIKE_COLUMN_NAMES = ID_LIKE_COLUMN_NAMES


def _is_id_like_column(column: object) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
    compact = normalized.replace("_", "")
    return normalized in _ID_LIKE_COLUMN_NAMES or compact in _ID_LIKE_COLUMN_NAMES


def _resolve_id_like_label_column(labels: pd.DataFrame) -> str | None:
    for column in labels.columns:
        if _is_id_like_column(column):
            return str(column)
    return None


def _resolve_label_target_columns(
    *,
    labels: pd.DataFrame,
    sample: pd.DataFrame,
    train_join_col: str,
    label_join_col: str,
    target_column_override: str | None,
) -> list[tuple[str, str]]:
    label_by_lower = {str(col).lower(): str(col) for col in labels.columns}
    if target_column_override:
        label_col = label_by_lower.get(str(target_column_override).lower())
        if label_col is not None:
            return [(label_col, target_column_override)]
    sample_targets = [
        (label_by_lower[str(col).lower()], str(col))
        for col in sample.columns
        if str(col).lower() != str(train_join_col).lower() and str(col).lower() in label_by_lower
    ]
    if sample_targets:
        return sample_targets
    label_targets = [str(col) for col in labels.columns if str(col).lower() != str(label_join_col).lower()]
    return [(label_targets[0], label_targets[0])] if len(label_targets) == 1 else []


def write_table(frame: pd.DataFrame, path: Path) -> Path:
    """Write a tabular artifact using the format implied by the path suffix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = frame_with_normalized_table_columns(frame)
    suffix = _tabular_suffix(path)
    if suffix in TABULAR_PARQUET_SUFFIXES:
        frame.to_parquet(path, index=False)
        return path
    if suffix == ".orc":
        frame.to_orc(path, index=False)
        return path
    if suffix in TABULAR_HDF_SUFFIXES:
        frame.to_hdf(path, key="submission", mode="w", format="table", index=False)
        return path
    if suffix in TABULAR_ARROW_IPC_SUFFIXES:
        frame.to_feather(path)
        return path
    if suffix == ".avro":
        _write_avro_table(frame, path)
        return path
    if suffix in TABULAR_EXCEL_SUFFIXES:
        frame.to_excel(path, index=False)
        return path
    if suffix in TABULAR_STATA_SUFFIXES:
        frame.to_stata(path, write_index=False)
        return path
    if suffix.startswith(".xml"):
        _write_xml_tabular_frame(frame, path)
        return path
    if suffix.startswith(TABULAR_HTML_SUFFIX_PREFIXES):
        _write_compressed_text(path, frame.to_html(index=False))
        return path
    if suffix in TABULAR_PICKLE_SUFFIXES:
        frame.to_pickle(path)
        return path
    if _is_json_lines_suffix(suffix):
        _write_compressed_text(path, frame.to_json(orient="records", lines=True))
        return path
    if suffix in TABULAR_STRUCTURED_SUFFIXES:
        if _is_yaml_suffix(suffix):
            import yaml

            _write_compressed_text(path, yaml.safe_dump(frame.to_dict(orient="records"), sort_keys=False))
        else:
            _write_compressed_text(path, frame.to_json(orient="records"))
        return path
    if suffix in TABULAR_TEXT_SUFFIXES:
        sep = default_delimited_text_separator(suffix)
        _write_compressed_text(path, frame.to_csv(index=False, sep=sep))
        return path
    if _path_requires_non_tabular_artifact(path):
        return _write_tabular_fallback_submission(frame, requested_path=path)
    frame.to_csv(path, index=False)
    return path


def _path_requires_non_tabular_artifact(path: Path) -> bool:
    suffix = artifact_suffix(path)
    if suffix in NON_TABULAR_SUBMISSION_SUFFIXES or suffix in ARCHIVE_SUBMISSION_SUFFIXES:
        return suffix not in TABULAR_SUBMISSION_SUFFIXES
    return False


def _write_tabular_fallback_submission(frame: pd.DataFrame, *, requested_path: Path) -> Path:
    frame = frame_with_normalized_table_columns(frame)
    fallback = requested_path.with_name(
        f"{_requested_output_stem(requested_path)}.tabular{_configured_tabular_fallback_suffix()}"
    )
    write_table(frame, fallback)
    manifest = requested_path.parent / "submission_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_class": "tabular",
                "submission_path": fallback.name,
                "requested_output_path": requested_path.name,
                "note": "Local tabular baseline could not produce the requested non-tabular artifact directly.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return fallback


def _configured_tabular_fallback_suffix() -> str:
    submission_filename = os.environ.get("KAGGLEBOT_SUBMISSION_FILENAME", "")
    if submission_filename:
        suffix = _tabular_suffix(Path(submission_filename))
        if suffix in TABULAR_SUBMISSION_SUFFIXES:
            return suffix
    for env_name in ("KAGGLEBOT_SAMPLE_SUBMISSION_PATH", "KAGGLEBOT_SAMPLE_SUBMISSION_FILENAME"):
        sample_filename = os.environ.get(env_name, "")
        if not sample_filename:
            continue
        suffix = _tabular_suffix(Path(Path(sample_filename).name))
        if suffix in TABULAR_SUBMISSION_SUFFIXES:
            return suffix
    return ".csv"


def _requested_output_stem(path: Path) -> str:
    return artifact_stem(path)


def _write_compressed_text(path: Path, text: str) -> None:
    _write_compressed_bytes(path, text.encode("utf-8"))


def _write_compressed_bytes(path: Path, payload: bytes) -> None:
    _write_compressed_payload(path, payload, suffix=_tabular_suffix(path))


def _coerce_prediction_dtype(sample_series: pd.Series, pred_series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(sample_series.dtype):
        if pd.api.types.is_bool_dtype(pred_series.dtype):
            return pred_series
        if pd.api.types.is_numeric_dtype(pred_series.dtype):
            values = pred_series.dropna().to_numpy()
            if values.size == 0:
                return pred_series
            binary_mask = np.isclose(values, 0.0) | np.isclose(values, 1.0)
            if binary_mask.all():
                return pred_series.astype(bool)
            return pred_series
        lowered = pred_series.astype(str).str.lower()
        if set(lowered.dropna().unique()).issubset({"true", "false"}):
            return lowered == "true"
    return pred_series


def _infer_target_columns(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    train_minus_test: list[str],
) -> list[str]:
    sample_cols = list(sample.columns)
    # 1) Most reliable: in sample, in train, not in test.
    candidates = [col for col in sample_cols if col in train_minus_test and col in train.columns]
    if candidates:
        return candidates

    # 2) Any sample cols present in train and not obvious ID-like.
    aligned = [col for col in sample_cols if col in train.columns]
    filtered = [col for col in aligned if col not in test.columns]
    if filtered:
        return filtered
    if len(aligned) > 1:
        return aligned[1:]
    if aligned and sample_cols and aligned[0] == sample_cols[0] and len(sample_cols) > 1:
        if len(train_minus_test) == 1 and _sample_has_numeric_prediction_columns(sample=sample, id_col=aligned[0]):
            return train_minus_test
        if len(train_minus_test) >= 2:
            survival_targets = _infer_survival_target_columns(train=train, train_minus_test=train_minus_test)
            if survival_targets:
                return survival_targets
        return []
    if aligned:
        return aligned

    # 3) Fallback to train-test diff.
    if train_minus_test:
        return train_minus_test
    return []


def _infer_unlabeled_numeric_score_columns(*, sample: pd.DataFrame, id_col: str | None) -> list[str]:
    prediction_cols = [str(col) for col in sample.columns if col != id_col]
    if len(prediction_cols) != 1:
        return []
    column = prediction_cols[0]
    if not _looks_like_unlabeled_score_column(column):
        return []
    if column not in sample.columns or not pd.api.types.is_numeric_dtype(sample[column]):
        return []
    return [column]


def _looks_like_unlabeled_score_column(name: str) -> bool:
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


def _looks_like_probability_score_column(name: str) -> bool:
    compact = re.sub(r"[^a-z0-9]+", "", name.lower())
    tokens = set(re.findall(r"[a-z0-9]+", name.lower()))
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


def _looks_like_learning_to_rank_target(
    *,
    target_col: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: Sequence[str],
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
        train_cols & test_cols & {"queryid", "qid", "searchid", "requestid"}
        and train_cols & test_cols & {"documentid", "docid", "candidateid", "itemid", "passageid"}
    )


def _target_should_use_continuous_model(
    target_col: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: Sequence[str],
) -> bool:
    return (
        _looks_like_learning_to_rank_target(
            target_col=target_col,
            train=train,
            test=test,
            feature_cols=feature_cols,
        )
        or _looks_like_ordinal_target(
            train[target_col],
            column_name=target_col,
            feature_cols=feature_cols,
        )
        or _looks_like_named_continuous_numeric_target(
            train[target_col],
            column_name=target_col,
            feature_cols=feature_cols,
        )
        or _looks_like_bounded_regression_target(train[target_col], column_name=target_col)
    )


def _looks_like_named_continuous_numeric_target(
    target: pd.Series,
    *,
    column_name: str,
    feature_cols: Sequence[str],
) -> bool:
    if _looks_like_user_item_interaction(feature_cols):
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


def _looks_like_bounded_regression_target(target: pd.Series, *, column_name: str) -> bool:
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


def _looks_like_ordinal_target(target: pd.Series, *, column_name: str, feature_cols: Sequence[str]) -> bool:
    if _looks_like_user_item_interaction(feature_cols):
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


def _looks_like_user_item_interaction(feature_cols: Sequence[str]) -> bool:
    compact_cols = {re.sub(r"[^a-z0-9]+", "", str(col).lower()) for col in feature_cols}
    has_user = bool(compact_cols & {"userid", "user", "customerid", "accountid"})
    has_item = bool(compact_cols & {"itemid", "item", "adid", "productid", "movieid"})
    return has_user and has_item


def _infer_survival_target_columns(*, train: pd.DataFrame, train_minus_test: list[str]) -> list[str]:
    event_cols = [col for col in train_minus_test if _is_survival_event_column(str(col))]
    time_cols = [col for col in train_minus_test if _is_survival_time_column(str(col))]
    if not event_cols or not time_cols:
        return []
    for event_col in event_cols:
        if not _looks_like_binary_event(train[event_col]):
            continue
        for time_col in time_cols:
            if pd.api.types.is_numeric_dtype(train[time_col]):
                return [event_col, time_col]
    return []


def _is_survival_event_column(name: str) -> bool:
    compact = re.sub(r"[^a-z0-9]+", "", name.lower())
    return compact in {"event", "eventobserved", "observed", "status", "efs", "censor", "censored", "death", "dead"}


def _is_survival_time_column(name: str) -> bool:
    compact = re.sub(r"[^a-z0-9]+", "", name.lower())
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


def _looks_like_binary_event(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    values = series.dropna()
    if values.empty:
        return False
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        return not numeric.empty and set(numeric.unique()).issubset({0, 1, 0.0, 1.0})
    lowered = values.astype(str).str.strip().str.lower()
    return set(lowered.unique()).issubset({"0", "1", "true", "false", "yes", "no", "event", "censored"})


def _sample_has_numeric_prediction_columns(*, sample: pd.DataFrame, id_col: str | None) -> bool:
    prediction_cols = [col for col in sample.columns if col != id_col]
    return len(prediction_cols) >= 2 and all(pd.api.types.is_numeric_dtype(sample[col]) for col in prediction_cols)


def _pick_id_column(*, sample_cols: list[str], target_cols: list[str], test_cols: list[str]) -> str | None:
    non_targets = [col for col in sample_cols if col not in target_cols]
    if not non_targets:
        return None
    test_overlap = [col for col in non_targets if col in test_cols]
    if test_overlap:
        return test_overlap[0]
    candidate = non_targets[0]
    return candidate if _is_id_like_column(candidate) else None


_ASSET_LABEL_TABLE_TOKENS = ASSET_LABEL_TABLE_TOKENS

_LABEL_TABLE_TOKENS = _ASSET_LABEL_TABLE_TOKENS


def _synthesize_train_test_from_assets(data_dir: Path, sample_path: Path) -> tuple[Path, Path, Path] | None:
    label_files = sorted(
        [p for p in data_dir.rglob("*") if _is_asset_label_table_path(p)],
        key=_asset_label_table_key,
        reverse=True,
    )
    if not label_files:
        return None

    sample = _read_table(sample_path)
    if sample.empty:
        return None
    sample_id = sample.columns[0]
    if not _is_id_like_column(sample_id):
        return None

    id_to_path = _discover_asset_paths(data_dir)
    if not id_to_path:
        return None

    test_ids = sample[sample_id].astype(str)
    for label_path in label_files:
        try:
            labels = _read_table(label_path)
        except Exception:  # noqa: BLE001
            continue
        if labels.empty or not len(labels.columns):
            continue
        label_id = _resolve_asset_label_id_column(sample_id=sample_id, labels=labels)
        if label_id not in labels.columns:
            continue

        train_ids = labels[label_id].astype(str)
        train = labels.copy()
        train[label_id] = train_ids
        train["asset_path"] = train_ids.map(lambda value: _lookup_asset_path(id_to_path, value, split="train"))
        test = pd.DataFrame({sample_id: test_ids})
        test["asset_path"] = test_ids.map(lambda value: _lookup_asset_path(id_to_path, value, split="test"))
        train = train[train["asset_path"].notna()].reset_index(drop=True)
        test = test[test["asset_path"].notna()].reset_index(drop=True)
        if train.empty or test.empty:
            continue

        cache_dir = data_dir / ".kagglebot_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        train_out = cache_dir / "train_synth.csv"
        test_out = cache_dir / "test_synth.csv"
        train.to_csv(train_out, index=False)
        test.to_csv(test_out, index=False)
        return train_out, test_out, sample_path
    return None


def _resolve_asset_label_id_column(*, sample_id: object, labels: pd.DataFrame) -> str:
    sample_key = str(sample_id).lower()
    label_by_lower = {str(col).lower(): str(col) for col in labels.columns}
    if sample_key in label_by_lower:
        return label_by_lower[sample_key]
    if _is_id_like_column(sample_id):
        alias = _resolve_id_like_label_column(labels)
        if alias is not None:
            return alias
    return str(labels.columns[0])


def _is_asset_label_table_path(path: Path) -> bool:
    if not path.is_file():
        return False
    if _tabular_suffix(path) not in TABULAR_SUBMISSION_SUFFIXES:
        return False
    name = path.name.lower()
    if "sample" in name or "submission" in name:
        return False
    return any(token in name for token in _ASSET_LABEL_TABLE_TOKENS)


def _asset_label_table_key(path: Path) -> tuple[int, int, str]:
    name = path.name.lower()
    has_train = 1 if "train" in name or "training" in name else 0
    best_token_score = 0
    for idx, token in enumerate(reversed(_ASSET_LABEL_TABLE_TOKENS), start=1):
        if token in name:
            best_token_score = max(best_token_score, idx)
    return has_train, best_token_score, path.as_posix()


def _discover_asset_paths(data_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    candidates = [path for path in data_dir.rglob("*") if is_data_asset_path(path)]
    for path in sorted(candidates, key=lambda item: _asset_priority_key(data_dir=data_dir, path=item)):
        for key in _asset_lookup_keys(data_dir=data_dir, path=path):
            mapping.setdefault(_normalize_asset_lookup_key(key), str(path))
    return mapping


def _lookup_asset_path(mapping: Mapping[str, str], value: object, *, split: str | None = None) -> str | None:
    for key in _asset_split_lookup_candidates(value, split=split):
        match = mapping.get(_normalize_asset_lookup_key(key))
        if match is not None:
            return match
    return None


def _asset_split_lookup_candidates(value: object, *, split: str | None) -> list[str]:
    normalized = _normalize_asset_lookup_key(value)
    if not normalized:
        return []
    basename = Path(normalized).name
    candidates: list[str] = []
    if split:
        for split_alias in _asset_split_aliases(split):
            candidates.extend(
                [
                    f"{split_alias}/{normalized}",
                    f"{split_alias}/{basename}",
                ]
            )
            for collection in sorted(_ASSET_COLLECTION_DIR_NAMES):
                candidates.extend(
                    [
                        f"{collection}/{split_alias}/{normalized}",
                        f"{collection}/{split_alias}/{basename}",
                    ]
                )
    candidates.extend([normalized, basename])

    deduped: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _asset_split_aliases(split: str) -> list[str]:
    aliases = [str(split or "").strip().lower()]
    aliases.extend(sorted(ROLE_ALIASES.get(aliases[0], frozenset())))
    seen: set[str] = set()
    ordered: list[str] = []
    for alias in aliases:
        if alias and alias not in seen:
            seen.add(alias)
            ordered.append(alias)
    return ordered


def _asset_lookup_keys(*, data_dir: Path, path: Path) -> list[str]:
    keys = [artifact_stem(path), path.name]
    try:
        relative = path.relative_to(data_dir)
    except ValueError:
        relative = path
    relative_posix = relative.as_posix()
    keys.append(relative_posix)
    suffix = artifact_suffix(path)
    if suffix and relative_posix.lower().endswith(suffix):
        keys.append(relative_posix[: -len(suffix)])

    parts = relative.parts
    for start in range(1, len(parts)):
        suffix_path = Path(*parts[start:]).as_posix()
        keys.append(suffix_path)
        if suffix and suffix_path.lower().endswith(suffix):
            keys.append(suffix_path[: -len(suffix)])

    deduped: list[str] = []
    for key in keys:
        if key and key not in deduped:
            deduped.append(key)
    return deduped


def _normalize_asset_lookup_key(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.strip("/").lower()


def _asset_priority_key(*, data_dir: Path, path: Path) -> tuple[int, str]:
    rel_parts = [part.lower() for part in path.relative_to(data_dir).parts]
    is_preferred_test = _path_parts_start_with_asset_role(rel_parts, "test")
    is_preferred_train = _path_parts_start_with_asset_role(rel_parts, "train")
    contains_test = "test" in rel_parts
    contains_train = "train" in rel_parts

    if is_preferred_test:
        rank = 0
    elif contains_test:
        rank = 1
    elif is_preferred_train:
        rank = 2
    elif contains_train:
        rank = 3
    else:
        rank = 4
    return rank, str(path)


def _maybe_synthesize_sample_submission(data_dir: Path) -> Path | None:
    for context_dir in _candidate_context_dirs(data_dir):
        usable = _find_context_sample_submission(context_dir)
        if usable is not None:
            return usable
        format_path = context_dir / "submission_format.md"
        synthesized = _synthesize_sample_from_submission_format(data_dir=data_dir, submission_format_path=format_path)
        if synthesized is not None:
            return synthesized
    return None


def _candidate_context_dirs(data_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    direct = data_dir.parent / "context"
    if direct not in seen:
        candidates.append(direct)
        seen.add(direct)

    for parent in [data_dir, *data_dir.parents]:
        candidate = parent / "context"
        if candidate in seen:
            continue
        candidates.append(candidate)
        seen.add(candidate)
    return candidates


def _find_context_sample_submission(context_dir: Path) -> Path | None:
    if not context_dir.exists():
        return None
    try:
        files = _find_tabular_files(context_dir)
    except OSError:
        return None
    candidate = _select_sample_submission_path(files)
    if candidate is None:
        return None
    return _is_usable_sample_submission(candidate)


def _is_usable_sample_submission(path: Path) -> Path | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        frame = _read_table(path)
    except Exception:  # noqa: BLE001
        return None
    if frame.empty or len(frame.columns) < 2:
        return None
    return path


def _synthesize_sample_from_submission_format(*, data_dir: Path, submission_format_path: Path) -> Path | None:
    if not submission_format_path.exists() or not submission_format_path.is_file():
        return None
    hint = load_submission_format_hint(submission_format_path)
    header = hint.columns if hint and hint.columns else _extract_submission_header(submission_format_path)
    if not header:
        return None
    id_column = header[0]
    target_columns = header[1:]
    if not target_columns:
        return None

    asset_id_style = _infer_asset_submission_id_style(
        submission_format_path=submission_format_path, id_column=id_column
    )
    test_ids = _discover_test_ids(
        data_dir,
        id_column=id_column,
        target_columns=target_columns,
        asset_id_style=asset_id_style,
    )
    if not test_ids:
        return None

    payload: dict[str, list[object]] = {id_column: test_ids}
    for col in target_columns:
        lowered = col.lower()
        if "prediction" in lowered and "string" in lowered:
            payload[col] = ["-"] * len(test_ids)
        else:
            payload[col] = [0] * len(test_ids)
    frame = pd.DataFrame(payload)

    cache_dir = data_dir / ".kagglebot_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = _synthesized_sample_suffix(hint.expected_suffixes if hint else None)
    out = cache_dir / f"sample_submission_synth{suffix}"
    write_table(frame, out)
    return out


def _synthesized_sample_suffix(expected_suffixes: Sequence[str] | None) -> str:
    return preferred_tabular_submission_suffix(expected_suffixes)


def _extract_submission_header(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    in_block = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_block = not in_block
            continue
        if not in_block:
            continue
        if "," not in line:
            continue
        cols = [part.strip() for part in line.split(",") if part.strip()]
        if len(cols) >= 2:
            return cols
    return []


def _discover_test_ids(
    data_dir: Path,
    *,
    id_column: str,
    target_columns: Sequence[str] | None = None,
    asset_id_style: str | None = None,
) -> list[str]:
    tabular_ids = _discover_tabular_test_ids(data_dir=data_dir, id_column=id_column, target_columns=target_columns)
    if tabular_ids:
        return tabular_ids
    asset_ids = _discover_asset_test_ids(data_dir, id_style=asset_id_style)
    if asset_ids:
        return asset_ids
    return []


def _discover_asset_test_ids(data_dir: Path, *, id_style: str | None = None) -> list[str]:
    preferred_assets: list[Path] = []
    for preferred in _preferred_asset_role_dirs(data_dir, "test"):
        try:
            preferred_assets.extend(path for path in preferred.rglob("*") if is_data_asset_path(path))
        except OSError:
            continue
    if preferred_assets:
        assets = sorted(set(preferred_assets), key=lambda path: _asset_test_candidate_key(data_dir=data_dir, path=path))
        return [_format_asset_submission_id(path, data_dir=data_dir, id_style=id_style) for path in assets]

    assets = []
    for path in data_dir.rglob("*"):
        if not is_data_asset_path(path):
            continue
        if _asset_path_mentions_inference_role(data_dir=data_dir, path=path):
            assets.append(path)
    assets = sorted(assets, key=lambda path: _asset_test_candidate_key(data_dir=data_dir, path=path))
    return [_format_asset_submission_id(path, data_dir=data_dir, id_style=id_style) for path in assets]


def _format_asset_submission_id(path: Path, *, data_dir: Path, id_style: str | None) -> str:
    if id_style == "stem":
        return artifact_stem(path)
    if id_style == "relative":
        return _asset_relative_id(path=path, data_dir=data_dir)
    if id_style == "role_relative":
        return _asset_role_relative_id(path=path, data_dir=data_dir)
    return path.name


def _infer_asset_submission_id_style(*, submission_format_path: Path, id_column: str) -> str | None:
    try:
        text = submission_format_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    examples = _extract_submission_id_examples(text=text, id_column=id_column)
    if not examples:
        return None
    path_examples = [_submission_id_example_parts(value) for value in examples]
    path_examples = [parts for parts in path_examples if len(parts) >= 2]
    if path_examples:
        if any(parts[0].lower() in _ASSET_COLLECTION_DIR_NAMES for parts in path_examples):
            return "relative"
        if any(_asset_component_mentions_inference_role(parts[0]) for parts in path_examples):
            return "role_relative"
        return "relative"
    with_suffix = [
        value
        for value in examples
        if artifact_suffix(Path(value)) in DATA_ASSET_SUFFIXES or Path(value).suffix.lower() in DATA_ASSET_SUFFIXES
    ]
    if with_suffix:
        return "name"
    if all(artifact_suffix(Path(value)) == "" and Path(value).suffix == "" for value in examples):
        return "stem"
    return None


def _asset_relative_id(*, path: Path, data_dir: Path) -> str:
    try:
        return path.relative_to(data_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _asset_role_relative_id(*, path: Path, data_dir: Path) -> str:
    relative = _asset_relative_id(path=path, data_dir=data_dir)
    parts = [part for part in relative.split("/") if part]
    for idx, part in enumerate(parts):
        if _asset_component_mentions_inference_role(part):
            return "/".join(parts[idx:])
    return path.name


def _submission_id_example_parts(value: str) -> list[str]:
    normalized = str(value or "").strip().strip("`").replace("\\", "/").strip("/")
    return [part for part in normalized.split("/") if part]


def _asset_component_mentions_inference_role(value: str) -> bool:
    lowered = str(value or "").lower()
    if _component_mentions_role(lowered, "test"):
        return True
    if _component_mentions_role(lowered, "train"):
        return False
    tokens = {token for token in re.split(r"[^a-z0-9]+", lowered) if token}
    return bool(tokens & _ASSET_INFERENCE_ROLE_TOKENS)


def _extract_submission_id_examples(*, text: str, id_column: str) -> list[str]:
    examples: list[str] = []
    for block in re.findall(r"```(?:[A-Za-z0-9_+-]+)?\s*\n(.*?)```", text, flags=re.S):
        examples.extend(_extract_submission_id_examples_from_lines(block.splitlines(), id_column=id_column))
    examples.extend(_extract_submission_id_examples_from_lines(text.splitlines(), id_column=id_column))
    deduped: list[str] = []
    for value in examples:
        cleaned = str(value).strip().strip("`").strip()
        if not cleaned or cleaned == id_column or cleaned in deduped:
            continue
        deduped.append(cleaned)
    return deduped[:5]


def _extract_submission_id_examples_from_lines(lines: Sequence[str], *, id_column: str) -> list[str]:
    stripped = [line.strip() for line in lines if line.strip()]
    for idx, line in enumerate(stripped[:-1]):
        if "|" in line and idx + 1 < len(stripped) and "|" in stripped[idx + 1] and "-" in stripped[idx + 1]:
            header = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
            if id_column not in header:
                continue
            id_idx = header.index(id_column)
            examples: list[str] = []
            for row in stripped[idx + 2 :]:
                if "|" not in row:
                    break
                cells = [cell.strip().strip("`") for cell in row.strip("|").split("|")]
                if id_idx < len(cells) and cells[id_idx]:
                    examples.append(cells[id_idx])
            if examples:
                return examples

    for line in stripped:
        parsed = _parse_delimited_submission_line(line)
        if parsed is None or id_column not in parsed:
            continue
        id_idx = parsed.index(id_column)
        examples: list[str] = []
        for row in stripped[stripped.index(line) + 1 :]:
            row_values = _parse_delimited_submission_line(row)
            if row_values is None or id_idx >= len(row_values):
                continue
            examples.append(row_values[id_idx])
        if examples:
            return examples
    return []


def _parse_delimited_submission_line(line: str) -> list[str] | None:
    if "\t" in line:
        delimiter = "\t"
    elif "," in line:
        delimiter = ","
    elif "|" in line and line.count("|") >= 2:
        return [cell.strip().strip("`") for cell in line.strip().strip("|").split("|") if cell.strip()]
    else:
        return None
    try:
        return [cell.strip().strip("`") for cell in next(csv.reader([line], delimiter=delimiter))]
    except csv.Error:
        return None


_ASSET_INFERENCE_ROLE_TOKENS = TEST_INFERENCE_ROLE_TOKENS
_ASSET_COLLECTION_DIR_NAMES = ASSET_COLLECTION_DIR_NAMES


def _asset_path_mentions_inference_role(*, data_dir: Path, path: Path) -> bool:
    try:
        parts = path.relative_to(data_dir).parts
    except ValueError:
        parts = path.parts
    lowered_parts = [str(part).lower() for part in parts]
    if any(_component_mentions_role(part, "test") for part in lowered_parts):
        return True
    if any(_component_mentions_role(part, "train") for part in lowered_parts):
        return False
    for part in lowered_parts:
        tokens = {token for token in re.split(r"[^a-z0-9]+", part) if token}
        if tokens & _ASSET_INFERENCE_ROLE_TOKENS:
            return True
    return False


def _asset_test_candidate_key(*, data_dir: Path, path: Path) -> tuple[int, str]:
    try:
        parts = [str(part).lower() for part in path.relative_to(data_dir).parts]
    except ValueError:
        parts = [str(part).lower() for part in path.parts]
    if _path_parts_start_with_asset_role(parts, "test"):
        rank = 0
    elif any(_component_mentions_role(part, "test") for part in parts):
        rank = 1
    else:
        rank = 2
    return rank, path.as_posix()


def _preferred_asset_role_dirs(data_dir: Path, role: str) -> list[Path]:
    if not data_dir.exists():
        return []
    candidates: list[Path] = []
    try:
        paths = [data_dir, *[path for path in data_dir.rglob("*") if path.is_dir()]]
    except OSError:
        paths = [data_dir]
    for path in paths:
        try:
            parts = [str(part).lower() for part in path.relative_to(data_dir).parts]
        except ValueError:
            parts = [str(part).lower() for part in path.parts]
        if _path_parts_start_with_asset_role(parts, role):
            candidates.append(path)
    return sorted(candidates, key=lambda path: (len(path.parts), path.as_posix()))


def _path_parts_start_with_asset_role(parts: Sequence[str], role: str) -> bool:
    if len(parts) < 2:
        return False
    first, second = parts[0], parts[1]
    return (first in _ASSET_COLLECTION_DIR_NAMES and _component_mentions_role(second, role)) or (
        _component_mentions_role(first, role) and second in _ASSET_COLLECTION_DIR_NAMES
    )


def _discover_tabular_test_ids(
    *,
    data_dir: Path,
    id_column: str,
    target_columns: Sequence[str] | None = None,
) -> list[str]:
    try:
        files = _find_tabular_files(data_dir)
    except OSError:
        return []
    role_candidates = [
        path
        for path in files
        if ".kagglebot_cache" not in path.parts
        and _path_mentions_role(path, "test")
        and _select_sample_submission_path([path]) is None
    ]
    ids = _read_ids_from_ranked_test_candidates(
        role_candidates,
        id_column=id_column,
        target_columns=target_columns,
    )
    if ids:
        return ids

    schema_candidates = [
        path
        for path in files
        if ".kagglebot_cache" not in path.parts
        and not _path_mentions_role(path, "train")
        and _select_sample_submission_path([path]) is None
        and _is_roleless_test_id_candidate(path, id_column=id_column, target_columns=target_columns)
    ]
    return _read_ids_from_ranked_test_candidates(
        schema_candidates,
        id_column=id_column,
        target_columns=target_columns,
    )


def _read_ids_from_ranked_test_candidates(
    candidates: Sequence[Path],
    *,
    id_column: str,
    target_columns: Sequence[str] | None,
) -> list[str]:
    for path in sorted(
        candidates,
        key=lambda item: _test_id_candidate_key(item, id_column=id_column, target_columns=target_columns),
        reverse=True,
    ):
        try:
            frame = _read_table(path)
        except Exception:  # noqa: BLE001
            continue
        if frame.empty:
            continue
        id_source = id_column if id_column in frame.columns else (str(frame.columns[0]) if len(frame.columns) else None)
        if id_source is None:
            continue
        values = [str(value) for value in frame[id_source].astype(str).tolist()]
        if values:
            return values
    return []


def _is_roleless_test_id_candidate(
    path: Path,
    *,
    id_column: str,
    target_columns: Sequence[str] | None,
) -> bool:
    try:
        columns = set(_read_table_schema_head(path, nrows=1).columns)
    except Exception:  # noqa: BLE001
        return False
    if id_column not in columns:
        return False
    target_set = {str(col) for col in target_columns or [] if str(col).strip()}
    if target_set and columns & target_set:
        return False
    stem = _tabular_stem(path).lower()
    tokens = {token for token in re.split(r"[^a-z0-9]+", stem) if token}
    if tokens & TEST_INFERENCE_ROLE_TOKENS:
        return True
    return bool(target_set)


def _test_id_candidate_key(
    path: Path,
    *,
    id_column: str,
    target_columns: Sequence[str] | None,
) -> tuple[int, int, int, int, int, str]:
    stem = _tabular_stem(path).lower()
    exact_test = 1 if stem == "test" else 0
    stem_tokens = {token for token in re.split(r"[^a-z0-9]+", stem) if token}
    likely_eval_file = 1 if stem_tokens & TEST_INFERENCE_ROLE_TOKENS else 0
    likely_feature_file = 1 if any(token in stem for token in ("feature", "data")) else 0
    try:
        columns = set(_read_table_schema_head(path, nrows=1).columns)
    except Exception:  # noqa: BLE001
        columns = set()
    has_requested_id = 1 if id_column in columns else 0
    target_set = {str(col) for col in target_columns or [] if str(col).strip()}
    avoids_target_columns = 1 if not target_set or not (columns & target_set) else 0
    return (has_requested_id, avoids_target_columns, exact_test, likely_eval_file, likely_feature_file, path.as_posix())


def _resolve_target_columns(
    *,
    sample: pd.DataFrame,
    id_column: str | None,
    target_column: str | None,
    target_columns: Sequence[str] | None,
    preds=None,
) -> list[str]:
    if target_columns is not None:
        resolved = [str(col) for col in target_columns if str(col).strip()]
    elif target_column is not None:
        resolved = [target_column]
    else:
        resolved = [col for col in sample.columns if col != id_column]
    if not resolved:
        raise ValueError("No target columns resolved for submission writing.")

    missing = [col for col in resolved if col not in sample.columns]
    if missing:
        prediction_cols = [str(col) for col in sample.columns if col != id_column]
        if (
            len(missing) == 1
            and target_columns is None
            and target_column in missing
            and _preds_can_fill_columns(preds=preds, columns=prediction_cols)
        ):
            return prediction_cols
        raise ValueError(f"Target columns not found in sample submission: {missing}")
    return resolved


def _preds_can_fill_columns(*, preds, columns: list[str]) -> bool:
    if not columns:
        return False
    if isinstance(preds, Mapping):
        return all(col in preds for col in columns)
    if preds is None:
        return False
    arr = np.asarray(preds)
    return arr.ndim == 2 and arr.shape[1] == len(columns)


def _normalize_prediction_table(
    *,
    preds,
    target_columns: list[str],
    row_count: int,
) -> dict[str, np.ndarray]:
    if isinstance(preds, Mapping):
        normalized: dict[str, np.ndarray] = {}
        for col in target_columns:
            if col not in preds:
                raise ValueError(f"Missing predictions for target column '{col}'.")
            values = np.asarray(preds[col]).ravel()
            if len(values) != row_count:
                raise ValueError(f"Prediction length mismatch for '{col}': expected {row_count}, got {len(values)}.")
            normalized[col] = values
        return normalized

    pred_array = np.asarray(preds)
    if pred_array.ndim == 1:
        if len(target_columns) != 1:
            raise ValueError("1D predictions provided for multi-target submission.")
        if len(pred_array) != row_count:
            raise ValueError(f"Prediction length mismatch: expected {row_count}, got {len(pred_array)}.")
        return {target_columns[0]: pred_array.ravel()}

    if pred_array.ndim == 2:
        if pred_array.shape[0] != row_count:
            raise ValueError(f"Prediction row count mismatch: expected {row_count}, got {pred_array.shape[0]}.")
        if pred_array.shape[1] != len(target_columns):
            raise ValueError(
                f"Prediction column count mismatch: expected {len(target_columns)}, got {pred_array.shape[1]}."
            )
        return {col: pred_array[:, idx] for idx, col in enumerate(target_columns)}

    raise ValueError("Unsupported predictions shape for submission writing.")


def _align_prediction_column(
    *,
    sample: pd.DataFrame,
    test: pd.DataFrame,
    values: np.ndarray,
    id_column: str | None,
    target_column: str,
) -> pd.Series:
    if id_column and id_column in test.columns and id_column in sample.columns:
        if not test[id_column].duplicated().any() and not sample[id_column].duplicated().any():
            pred_map = pd.Series(values, index=test[id_column].map(_submission_id_alignment_key))
            aligned = sample[id_column].map(_submission_id_alignment_key).map(pred_map)
            if aligned.isna().any():
                raise ValueError(
                    f"Missing predictions after aligning by id column '{id_column}' for target '{target_column}'."
                )
            return aligned
    if id_column and id_column in sample.columns and id_column not in test.columns:
        composite_ids = _infer_test_composite_submission_ids(sample[id_column], test)
        if composite_ids is not None:
            pred_map = pd.Series(values, index=composite_ids.map(_submission_id_alignment_key))
            aligned = sample[id_column].map(_submission_id_alignment_key).map(pred_map)
            if aligned.isna().any():
                raise ValueError(
                    f"Missing predictions after aligning by composite id column '{id_column}' "
                    f"for target '{target_column}'."
                )
            return aligned
    if len(values) != len(sample):
        raise ValueError(
            f"Prediction length does not match submission rows for target '{target_column}': "
            f"expected {len(sample)}, got {len(values)}."
        )
    return pd.Series(values, index=sample.index)


def _infer_test_composite_submission_ids(sample_ids: pd.Series, test: pd.DataFrame) -> pd.Series | None:
    sample_keys = sample_ids.map(_submission_id_alignment_key)
    if sample_keys.empty or sample_keys.duplicated().any():
        return None
    target_key_set = set(sample_keys)
    candidate_cols = _composite_id_candidate_columns(test)
    for width in (2, 3):
        for cols in combinations(candidate_cols, width):
            for sep in _COMPOSITE_ID_SEPARATORS:
                composite = _join_composite_id_columns(test, cols, sep=sep)
                composite_keys = composite.map(_submission_id_alignment_key)
                if composite_keys.duplicated().any():
                    continue
                if set(composite_keys) == target_key_set:
                    return composite
    return None


def _composite_id_candidate_columns(test: pd.DataFrame) -> list[str]:
    id_like: list[str] = []
    other: list[str] = []
    for col in test.columns:
        series = test[col]
        if pd.api.types.is_float_dtype(series):
            continue
        compact = re.sub(r"[^a-z0-9]+", "", str(col).lower())
        tokens = set(re.findall(r"[a-z0-9]+", str(col).lower()))
        target = id_like if compact.endswith("id") or "id" in tokens or compact in {"user", "item", "movie"} else other
        target.append(str(col))
    return [*id_like, *other][:8]


def _join_composite_id_columns(test: pd.DataFrame, columns: Sequence[str], *, sep: str) -> pd.Series:
    parts = [test[col].map(_submission_id_alignment_key) for col in columns]
    output = parts[0].astype(str)
    for part in parts[1:]:
        output = output + sep + part.astype(str)
    return output


_COMPOSITE_ID_SEPARATORS = ("_", "-", "/", ":", ".", "|", "")


def _submission_id_alignment_key(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()
