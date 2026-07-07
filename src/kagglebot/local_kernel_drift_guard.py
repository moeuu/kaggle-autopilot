from __future__ import annotations

import os
import time
from pathlib import Path

from kagglebot import local_kernel_context as _local_kernel_context
from kagglebot.env_utils import env_truthy
from kagglebot.json_utils import write_json_object
from kagglebot.solver.io import materialize_sqlite_tables, read_table
from kagglebot.submission_sample_discovery import is_tabular_data_path, path_mentions_role, tabular_suffix

ZERO_OVERLAP_DRIFT_GUARD_FILENAME = "zero_overlap_drift_guard.json"
ZERO_OVERLAP_DRIFT_MIN_TVD = 0.20
ZERO_OVERLAP_DRIFT_MIN_ABS_CORR = 0.08
ZERO_OVERLAP_DRIFT_MIN_ZERO_OVERLAP_RATIO = 0.50
ZERO_OVERLAP_DRIFT_MAX_CAT_UNIQUE_RATIO = 0.98


def infer_target_column_from_frames(*, train_columns: list[str], test_columns: list[str]) -> str | None:
    test_set = set(test_columns)
    candidates = [col for col in train_columns if col not in test_set]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return candidates[-1]
    for name in ("target", "label", "y"):
        if name in train_columns and name not in test_set:
            return name
    return None


def is_categorical_like_series(*, series, n_rows: int) -> bool:
    dtype_name = str(getattr(series, "dtype", "")).lower()
    if any(token in dtype_name for token in ("object", "category", "string", "bool")):
        return True
    try:
        nunique = int(series.nunique(dropna=True))
    except Exception:
        return False
    if nunique <= 0:
        return False
    unique_ratio = nunique / max(1, n_rows)
    return unique_ratio <= ZERO_OVERLAP_DRIFT_MAX_CAT_UNIQUE_RATIO


def categorical_tvd(*, train_series, test_series) -> float:
    train_values = train_series.fillna("__nan__").astype(str).value_counts(normalize=True)
    test_values = test_series.fillna("__nan__").astype(str).value_counts(normalize=True)
    keys = set(train_values.index) | set(test_values.index)
    if not keys:
        return 0.0
    total_variation = 0.0
    for key in keys:
        total_variation += abs(float(train_values.get(key, 0.0)) - float(test_values.get(key, 0.0)))
    return 0.5 * total_variation


def abs_corr_with_target(*, feature_series, target_series) -> float:
    try:
        target_numeric = target_series.astype(float)
    except Exception:
        return 0.0
    if target_numeric.nunique(dropna=True) <= 1:
        return 0.0
    dtype_name = str(getattr(feature_series, "dtype", "")).lower()
    try:
        if any(token in dtype_name for token in ("object", "string", "category", "bool")):
            encoded = feature_series.fillna("__nan__").astype(str).factorize()[0]
            encoded_series = target_numeric.__class__(encoded, index=target_numeric.index)
            corr = target_numeric.corr(encoded_series)
        else:
            corr = target_numeric.corr(feature_series.astype(float))
    except Exception:
        return 0.0
    if corr is None:
        return 0.0
    try:
        value = abs(float(corr))
    except Exception:
        return 0.0
    if value != value:
        return 0.0
    return value


def build_zero_overlap_drift_guard_payload(
    *,
    train_df,
    test_df,
    target_col: str | None,
    id_col: str | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "enabled": False,
        "drop_columns": [],
        "reason": "guard_not_triggered",
        "thresholds": {
            "min_tvd": ZERO_OVERLAP_DRIFT_MIN_TVD,
            "min_abs_corr": ZERO_OVERLAP_DRIFT_MIN_ABS_CORR,
            "min_zero_overlap_ratio": ZERO_OVERLAP_DRIFT_MIN_ZERO_OVERLAP_RATIO,
        },
        "suspects": [],
    }
    if target_col is None or target_col not in train_df.columns:
        payload["reason"] = "missing_target_column"
        return payload

    feature_cols = [col for col in train_df.columns if col != target_col and col in test_df.columns]
    if not feature_cols:
        payload["reason"] = "no_common_feature_columns"
        return payload

    n_rows = int(len(train_df))
    categorical_checked = 0
    zero_overlap_checked = 0
    suspects: list[dict[str, object]] = []
    drop_columns: list[str] = []
    target_series = train_df[target_col]
    for column in feature_cols:
        if id_col is not None and column == id_col:
            continue
        train_series = train_df[column]
        test_series = test_df[column]
        if not is_categorical_like_series(series=train_series, n_rows=n_rows):
            continue
        categorical_checked += 1
        train_keys = set(train_series.dropna().astype(str).unique().tolist())
        test_keys = set(test_series.dropna().astype(str).unique().tolist())
        if not train_keys or not test_keys:
            continue
        overlap = len(train_keys & test_keys)
        if overlap != 0:
            continue
        zero_overlap_checked += 1
        drift = categorical_tvd(train_series=train_series, test_series=test_series)
        corr = abs_corr_with_target(feature_series=train_series, target_series=target_series)
        candidate = {
            "column": column,
            "overlap_unique_count": overlap,
            "train_unique": len(train_keys),
            "test_unique": len(test_keys),
            "drift_tvd": drift,
            "abs_corr_target": corr,
        }
        suspects.append(candidate)
        if drift >= ZERO_OVERLAP_DRIFT_MIN_TVD and corr >= ZERO_OVERLAP_DRIFT_MIN_ABS_CORR:
            drop_columns.append(column)

    zero_overlap_ratio = zero_overlap_checked / categorical_checked if categorical_checked > 0 else 0.0
    payload["suspects"] = sorted(
        suspects,
        key=lambda item: float(item.get("drift_tvd", 0.0)) * float(item.get("abs_corr_target", 0.0)),
        reverse=True,
    )
    payload["stats"] = {
        "categorical_checked": categorical_checked,
        "zero_overlap_checked": zero_overlap_checked,
        "zero_overlap_ratio": zero_overlap_ratio,
    }
    payload["drop_columns"] = sorted(set(drop_columns))
    if payload["drop_columns"] and zero_overlap_ratio >= ZERO_OVERLAP_DRIFT_MIN_ZERO_OVERLAP_RATIO:
        payload["enabled"] = True
        payload["reason"] = "zero_overlap_high_drift_detected"
    return payload


def prepare_zero_overlap_drift_guard(*, base_dir: Path, slug: str, context_dir: Path) -> Path | None:
    if os.getenv("KAGGLEBOT_ENABLE_ZERO_OVERLAP_DRIFT_GUARD") is not None and not env_truthy(
        "KAGGLEBOT_ENABLE_ZERO_OVERLAP_DRIFT_GUARD"
    ):
        return None
    data_dir = base_dir / slug / "data"
    materialize_sqlite_tables(data_dir)
    train_path = _find_named_tabular_file(data_dir, "train")
    test_path = _find_named_tabular_file(data_dir, "test")
    if not train_path.exists() or not test_path.exists():
        return None
    try:
        train_df = read_table(train_path)
        test_df = read_table(test_path)
    except Exception:
        return None
    if train_df.empty or test_df.empty:
        return None

    target_col, id_col = _local_kernel_context.load_dataset_profile_identity(context_dir=context_dir)
    if target_col is None:
        target_col = infer_target_column_from_frames(
            train_columns=[str(col) for col in train_df.columns],
            test_columns=[str(col) for col in test_df.columns],
        )
    payload = build_zero_overlap_drift_guard_payload(
        train_df=train_df,
        test_df=test_df,
        target_col=target_col,
        id_col=id_col,
    )
    payload["target_column"] = target_col
    payload["id_column"] = id_col
    payload["generated_at_epoch"] = int(time.time())

    guard_path = context_dir / ZERO_OVERLAP_DRIFT_GUARD_FILENAME
    write_json_object(guard_path, payload)
    return guard_path


def _find_named_tabular_file(data_dir: Path, stem: str) -> Path:
    if not data_dir.exists():
        return data_dir / f"{stem}.csv"
    direct_matches: list[Path] = []
    try:
        for path in data_dir.iterdir():
            if path.is_file() and is_tabular_data_path(path) and _tabular_stem(path).lower() == stem:
                direct_matches.append(path)
    except OSError:
        direct_matches = []
    if direct_matches:
        return sorted(direct_matches, key=lambda path: (len(path.parts), str(path)))[0]
    matches = _named_tabular_file_matches(data_dir=data_dir, stem=stem, include_cache=False)
    if not matches:
        matches = _named_tabular_file_matches(data_dir=data_dir, stem=stem, include_cache=True)
    if not matches:
        return data_dir / f"{stem}.csv"
    return max(
        matches,
        key=lambda item: (
            item[0],
            -len(item[1].relative_to(data_dir).parts),
            str(item[1]).lower(),
        ),
    )[1]


def _named_tabular_file_matches(*, data_dir: Path, stem: str, include_cache: bool) -> list[tuple[int, Path]]:
    matches: list[tuple[int, Path]] = []
    try:
        paths = data_dir.rglob("*")
        for path in paths:
            if not path.is_file() or not is_tabular_data_path(path):
                continue
            if not include_cache and ".kagglebot_cache" in {part.lower() for part in path.parts}:
                continue
            score = _named_tabular_file_score(path, stem=stem)
            if score > 0:
                matches.append((score, path))
    except OSError:
        return []
    return matches


def _named_tabular_file_score(path: Path, *, stem: str) -> int:
    normalized_stem = stem.lower()
    path_stem = _tabular_stem(path).lower()
    if path_stem == normalized_stem:
        return 4
    if normalized_stem == "train" and path_mentions_role(path, "test"):
        return 0
    if normalized_stem == "test" and path_mentions_role(path, "train"):
        return 0
    if normalized_stem in {"train", "test"} and path_mentions_role(path, normalized_stem):
        return 3
    return 0


def _tabular_stem(path: Path) -> str:
    suffix = tabular_suffix(path)
    name = path.name
    if suffix and name.lower().endswith(suffix):
        return name[: -len(suffix)]
    return path.stem
