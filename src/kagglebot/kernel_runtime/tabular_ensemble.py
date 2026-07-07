from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from kagglebot.asset_modality import (
    artifact_stem,
    artifact_suffix,
)
from kagglebot.compression_suffixes import write_compressed_bytes as _write_compressed_payload
from kagglebot.submission_output_naming import (
    configured_submission_filename_is_template,
    non_tabular_submission_output_suffixes,
)
from kagglebot.submission_templates import build_submission_template_for_test
from kagglebot.table_columns import frame_with_normalized_table_columns

try:
    from catboost import CatBoostClassifier
except ImportError as exc:  # pragma: no cover - dependency is expected in kernel runtimes
    raise RuntimeError("catboost is required for shared tabular ensemble helpers.") from exc

try:
    from lightgbm import LGBMClassifier
except ImportError as exc:  # pragma: no cover - dependency is expected in kernel runtimes
    raise RuntimeError("lightgbm is required for shared tabular ensemble helpers.") from exc

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - dependency is expected in kernel runtimes
    raise RuntimeError("xgboost is required for shared tabular ensemble helpers.") from exc

from kagglebot.submission_sample_discovery import (
    TABULAR_ARROW_IPC_SUFFIXES,
    TABULAR_EXCEL_SUFFIXES,
    TABULAR_HDF_SUFFIXES,
    TABULAR_HTML_SUFFIX_PREFIXES,
    TABULAR_JSON_LINES_SUFFIX_PREFIXES,
    TABULAR_PARQUET_SUFFIXES,
    TABULAR_PICKLE_SUFFIXES,
    TABULAR_STATA_SUFFIXES,
    TABULAR_STRUCTURED_SUFFIXES,
    TABULAR_SUBMISSION_SUFFIXES,
    TABULAR_TEXT_SUFFIXES,
    default_delimited_text_separator,
    tabular_suffix,
    write_xml_tabular_frame,
)

_REQUESTED_NON_TABULAR_OUTPUT_SUFFIXES = non_tabular_submission_output_suffixes()


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


FAST_DEV = env_flag("KAGGLEBOT_FAST_DEV", False)
OUTER_FOLDS = 2 if FAST_DEV else 5
DEFAULT_PSEUDO_THRESHOLD = float(os.getenv("KAGGLEBOT_PSEUDO_THRESHOLD", "0.995"))
PREFER_CUDA = not env_flag("KAGGLEBOT_DISABLE_CUDA", False)
DEFAULT_XGB_N_ESTIMATORS = int(os.getenv("KAGGLEBOT_XGB_ESTIMATORS", "50000"))
CPU_FALLBACK_N_ESTIMATORS = int(os.getenv("KAGGLEBOT_XGB_CPU_FALLBACK_ESTIMATORS", "300" if FAST_DEV else "8000"))
FAST_DEV_N_ESTIMATORS = int(os.getenv("KAGGLEBOT_FAST_DEV_XGB_ESTIMATORS", "150"))
DEFAULT_CB_ITERATIONS = int(os.getenv("KAGGLEBOT_CB_ITERATIONS", "8000"))
DEFAULT_LGBM_ESTIMATORS = int(os.getenv("KAGGLEBOT_LGBM_ESTIMATORS", "12000"))
_FOLD_NUMBER_RE = re.compile(r"(\d+)")
_ARTIFACT_STEM_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_TABULAR_OUTPUT_SUFFIXES = TABULAR_SUBMISSION_SUFFIXES


@dataclass
class PipelineResult:
    name: str
    oof_preds: np.ndarray
    test_preds: np.ndarray
    cv_score: float
    fold_scores: list[dict[str, Any]]
    feature_manifest: dict[str, Any]
    metadata: dict[str, Any]
    test_predictions_by_fold: dict[str, np.ndarray]
    oof_predictions_by_fold: dict[str, np.ndarray]
    valid_indices_by_fold: dict[str, np.ndarray]


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    model_family: str
    model_seeds: list[int]
    params_override: dict[str, Any]
    enable_pseudo: bool = False
    pseudo_threshold: float = DEFAULT_PSEUDO_THRESHOLD


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true_arr = np.asarray(y_true)
    if np.unique(y_true_arr).size < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def clip_predictions(preds: np.ndarray) -> np.ndarray:
    arr = np.asarray(preds, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(arr, 1e-6, 1 - 1e-6)


def normalize_continuous_predictions(preds: np.ndarray, *, fill_value: float = 0.0) -> np.ndarray:
    arr = np.asarray(preds, dtype=np.float64)
    return np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)


def normalize_structured_regression_predictions(
    preds: np.ndarray,
    *,
    prediction_kind: str | None = None,
    target_labels: pd.Series | None = None,
    target_col: str | None = None,
) -> np.ndarray:
    output = normalize_continuous_predictions(preds)
    kind = normalize_prediction_kind(prediction_kind)
    column_name = str(target_col or "")
    if kind == "count_regression" or (
        target_labels is not None and looks_like_count_regression_target(target_labels, column_name)
    ):
        return np.clip(output, 0.0, None)
    if kind == "positive_skew_regression" or (
        target_labels is not None and looks_like_positive_skew_regression_target(target_labels, column_name)
    ):
        return np.clip(output, 0.0, None)
    bounds = None
    if kind == "bounded_regression" and target_labels is None:
        bounds = (0.0, 1.0)
    elif target_labels is not None:
        bounds = bounded_regression_bounds(target_labels, column_name)
    if bounds is None:
        return output
    lower, upper = bounds
    return np.clip(output, lower, upper)


def normalize_prediction_kind(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "probability").strip().lower()).strip("_")


def prediction_kind_is_probability(value: str | None) -> bool:
    kind = normalize_prediction_kind(value)
    return kind in {
        "probability",
        "probabilities",
        "probability_columns",
        "binary_probability",
        "classification_probability",
        "classification",
        "binary_classification",
    }


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


def build_prediction_range(
    preds: np.ndarray,
    *,
    prediction_kind: str | None = None,
    target_labels: pd.Series | None = None,
    target_col: str | None = None,
) -> list[float]:
    values = (
        clip_predictions(preds)
        if prediction_kind_is_probability(prediction_kind)
        else normalize_structured_regression_predictions(
            preds,
            prediction_kind=prediction_kind,
            target_labels=target_labels,
            target_col=target_col,
        )
    )
    return [float(values.min()), float(values.max())]


def resolve_component_models(result: PipelineResult) -> list[str]:
    components = result.metadata.get("blend_components")
    if isinstance(components, list):
        values = [str(item).strip() for item in components if str(item).strip()]
        if values:
            return values
    return [result.name]


def build_prediction_correlation_summary(results: list[PipelineResult]) -> dict[str, float | None]:
    single_model_results = [result for result in results if result.metadata.get("kind", "single") == "single"]
    corr_values: list[float] = []
    for idx, first in enumerate(single_model_results[:-1]):
        first_preds = np.asarray(first.oof_preds, dtype=np.float64)
        if first_preds.size <= 1:
            continue
        for second in single_model_results[idx + 1 :]:
            second_preds = np.asarray(second.oof_preds, dtype=np.float64)
            if second_preds.shape != first_preds.shape:
                continue
            corr = np.corrcoef(first_preds, second_preds)[0, 1]
            if np.isfinite(corr):
                corr_values.append(float(abs(corr)))
    if not corr_values:
        return {
            "pair_count": 0,
            "mean_abs_corr": None,
            "max_abs_corr": None,
            "min_abs_corr": None,
        }
    return {
        "pair_count": len(corr_values),
        "mean_abs_corr": float(np.mean(corr_values)),
        "max_abs_corr": float(np.max(corr_values)),
        "min_abs_corr": float(np.min(corr_values)),
    }


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


def mirror_json(filename: str, payload: Any, output_dirs: list[Path]) -> None:
    for output_dir in output_dirs:
        write_json(output_dir / filename, payload)


def mirror_df(filename: str, df: pd.DataFrame, output_dirs: list[Path]) -> None:
    for output_dir in output_dirs:
        write_table(df, output_dir / filename)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = frame_with_normalized_table_columns(df)
    suffix = tabular_suffix(path)
    if suffix in TABULAR_PARQUET_SUFFIXES:
        df.to_parquet(path, index=False)
        return
    if suffix == ".orc":
        df.to_orc(path, index=False)
        return
    if suffix in TABULAR_HDF_SUFFIXES:
        df.to_hdf(path, key="submission", mode="w", format="table", index=False)
        return
    if suffix in TABULAR_ARROW_IPC_SUFFIXES:
        df.to_feather(path)
        return
    if suffix == ".avro":
        write_avro_table(df, path)
        return
    if suffix in TABULAR_EXCEL_SUFFIXES:
        df.to_excel(path, index=False)
        return
    if suffix in TABULAR_STATA_SUFFIXES:
        df.to_stata(path, write_index=False)
        return
    if suffix.startswith(".xml"):
        write_xml_tabular_frame(df, path)
        return
    if suffix.startswith(TABULAR_HTML_SUFFIX_PREFIXES):
        write_compressed_text(path, df.to_html(index=False))
        return
    if suffix in TABULAR_PICKLE_SUFFIXES:
        df.to_pickle(path)
        return
    if suffix in TABULAR_STRUCTURED_SUFFIXES:
        if suffix.startswith((".yaml", ".yml")):
            import yaml

            payload = yaml.safe_dump(df.to_dict(orient="records"), sort_keys=False)
        else:
            payload = df.to_json(orient="records", lines=suffix.startswith(TABULAR_JSON_LINES_SUFFIX_PREFIXES))
        write_compressed_text(path, payload)
        return
    if suffix in TABULAR_TEXT_SUFFIXES:
        write_compressed_text(path, df.to_csv(index=False, sep=default_delimited_text_separator(suffix)))
        return
    df.to_csv(path, index=False)


def write_avro_table(df: pd.DataFrame, path: Path) -> None:
    from fastavro import writer

    fields = [{"name": str(column), "type": ["null", avro_field_type(df[column])]} for column in df.columns]
    schema = {"type": "record", "name": "SubmissionRecord", "fields": fields}
    with path.open("wb") as handle:
        writer(handle, schema, frame_to_avro_records(df, fields))


def avro_field_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "long"
    if pd.api.types.is_float_dtype(series):
        return "double"
    return "string"


def frame_to_avro_records(df: pd.DataFrame, fields: list[dict[str, object]]) -> list[dict[str, object]]:
    field_types = {}
    for field in fields:
        field_type = field["type"]
        candidates = field_type if isinstance(field_type, list) else [field_type]
        field_types[str(field["name"])] = next(candidate for candidate in candidates if candidate != "null")
    records = []
    for row in df.to_dict(orient="records"):
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


def write_compressed_text(path: Path, text: str) -> None:
    write_compressed_bytes(path, text.encode("utf-8"))


def write_compressed_bytes(path: Path, payload: bytes) -> None:
    _write_compressed_payload(path, payload, suffix=tabular_suffix(path))


def resolve_submission_filename() -> str:
    default_name = _default_submission_filename_from_sample_env()
    raw = str(os.getenv("KAGGLEBOT_SUBMISSION_FILENAME") or default_name).strip()
    name = Path(raw).name
    suffix = tabular_suffix(Path(name))
    if not name or configured_submission_filename_is_template(name) or name.lower() in {"metrics.json", "plan.json"}:
        return default_name
    if suffix not in _TABULAR_OUTPUT_SUFFIXES:
        requested = requested_non_tabular_submission_filename()
        if requested is not None:
            return f"{_requested_output_stem(Path(requested))}.tabular{_configured_tabular_fallback_suffix()}"
        return default_name
    return name


def _configured_tabular_fallback_suffix() -> str:
    submission_filename = str(os.getenv("KAGGLEBOT_SUBMISSION_FILENAME") or "").strip()
    if submission_filename:
        suffix = tabular_suffix(Path(Path(submission_filename).name))
        if suffix in _TABULAR_OUTPUT_SUFFIXES:
            return suffix
    default_suffix = tabular_suffix(Path(_default_submission_filename_from_sample_env()))
    if default_suffix in _TABULAR_OUTPUT_SUFFIXES:
        return default_suffix
    return ".csv"


def _default_submission_filename_from_sample_env() -> str:
    raw = str(
        os.getenv("KAGGLEBOT_SAMPLE_SUBMISSION_PATH") or os.getenv("KAGGLEBOT_SAMPLE_SUBMISSION_FILENAME") or ""
    ).strip()
    sample_suffix = tabular_suffix(Path(Path(raw).name)) if raw else ""
    if sample_suffix in _TABULAR_OUTPUT_SUFFIXES:
        return f"submission{sample_suffix}"
    return "submission.csv"


def requested_non_tabular_submission_filename() -> str | None:
    raw = str(os.getenv("KAGGLEBOT_SUBMISSION_FILENAME") or "").strip()
    name = Path(raw).name
    if not name:
        return None
    if configured_submission_filename_is_template(name) or name.lower() in {"metrics.json", "plan.json"}:
        return None
    tabular_candidate_suffix = tabular_suffix(Path(name))
    if tabular_candidate_suffix in _TABULAR_OUTPUT_SUFFIXES:
        return None
    suffix = artifact_suffix(Path(name))
    if suffix in _REQUESTED_NON_TABULAR_OUTPUT_SUFFIXES:
        return name
    return None


def _requested_output_stem(path: Path) -> str:
    return artifact_stem(path)


def submission_artifact_filename(*, stem: str | None = None) -> str:
    base = resolve_submission_filename()
    path = Path(base)
    suffix = tabular_suffix(path)
    if not stem:
        return base
    return f"submission_{stem}{suffix}"


def mirror_npy(filename: str, arr: np.ndarray, output_dirs: list[Path]) -> None:
    for output_dir in output_dirs:
        np.save(output_dir / filename, arr)


def train_xgb_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    model_seed: int,
    params_override: dict[str, Any],
    sample_weight: np.ndarray | None = None,
) -> tuple[XGBClassifier, dict[str, Any]]:
    params = {
        "n_estimators": FAST_DEV_N_ESTIMATORS if FAST_DEV else DEFAULT_XGB_N_ESTIMATORS,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "gamma": 0.05,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "enable_categorical": True,
        "tree_method": "hist",
        "device": "cuda" if PREFER_CUDA else "cpu",
        "early_stopping_rounds": 50 if FAST_DEV else 500,
        "random_state": model_seed,
        "verbosity": 0,
    }
    params.update(params_override)
    params["random_state"] = model_seed
    if FAST_DEV:
        params["early_stopping_rounds"] = min(int(params.get("early_stopping_rounds", 50)), 50)
    try:
        model = XGBClassifier(**params)
        fit_kwargs: dict[str, Any] = {
            "eval_set": [(x_valid, y_valid)],
            "verbose": False,
        }
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(x_train, y_train, **fit_kwargs)
        return model, {
            "device": params["device"],
            "best_iteration": getattr(model, "best_iteration", None),
            "params": dict(params),
        }
    except Exception as exc:
        if params["device"] == "cpu":
            raise
        params["device"] = "cpu"
        params["n_estimators"] = min(int(params["n_estimators"]), CPU_FALLBACK_N_ESTIMATORS)
        model = XGBClassifier(**params)
        fit_kwargs = {
            "eval_set": [(x_valid, y_valid)],
            "verbose": False,
        }
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(x_train, y_train, **fit_kwargs)
        return model, {
            "device": "cpu",
            "best_iteration": getattr(model, "best_iteration", None),
            "fallback_reason": str(exc),
            "params": dict(params),
        }


def train_catboost_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    model_seed: int,
    params_override: dict[str, Any],
    cat_features: list[str],
    sample_weight: np.ndarray | None = None,
) -> tuple[CatBoostClassifier, dict[str, Any]]:
    params = {
        "iterations": 150 if FAST_DEV else DEFAULT_CB_ITERATIONS,
        "learning_rate": 0.01 if not FAST_DEV else 0.05,
        "depth": 8,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "task_type": "GPU" if PREFER_CUDA else "CPU",
        "random_seed": model_seed,
        "verbose": False,
        "allow_writing_files": False,
        "od_type": "Iter",
        "od_wait": 100 if FAST_DEV else 300,
    }
    params.update(params_override)
    params["random_seed"] = model_seed
    try:
        model = CatBoostClassifier(**params)
        fit_kwargs: dict[str, Any] = {
            "eval_set": (x_valid, y_valid),
            "cat_features": cat_features,
            "verbose": False,
        }
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(x_train, y_train, **fit_kwargs)
        return model, {
            "device": str(params["task_type"]).lower(),
            "best_iteration": getattr(model, "get_best_iteration", lambda: None)(),
            "params": dict(params),
        }
    except Exception as exc:
        if str(params["task_type"]).upper() == "CPU":
            raise
        print(f"[kernel] CatBoost GPU failed; retrying on CPU: {type(exc).__name__}: {exc}", flush=True)
        params["task_type"] = "CPU"
        model = CatBoostClassifier(**params)
        fit_kwargs = {
            "eval_set": (x_valid, y_valid),
            "cat_features": cat_features,
            "verbose": False,
        }
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(x_train, y_train, **fit_kwargs)
        return model, {
            "device": "cpu",
            "best_iteration": getattr(model, "get_best_iteration", lambda: None)(),
            "fallback_reason": str(exc),
            "params": dict(params),
        }


def train_lgbm_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    model_seed: int,
    params_override: dict[str, Any],
    sample_weight: np.ndarray | None = None,
) -> tuple[LGBMClassifier, dict[str, Any]]:
    params = {
        "n_estimators": 200 if FAST_DEV else DEFAULT_LGBM_ESTIMATORS,
        "learning_rate": 0.02,
        "num_leaves": 63,
        "min_child_samples": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_lambda": 1.0,
        "objective": "binary",
        "metric": "auc",
        "random_state": model_seed,
        "verbosity": -1,
    }
    params.update(params_override)
    params["random_state"] = model_seed
    model = LGBMClassifier(**params)
    fit_kwargs: dict[str, Any] = {
        "eval_set": [(x_valid, y_valid)],
        "eval_metric": "auc",
        "callbacks": [],
    }
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(x_train, y_train, **fit_kwargs)
    return model, {
        "device": "cpu",
        "best_iteration": getattr(model, "best_iteration_", None),
        "params": dict(params),
    }


def maybe_apply_pseudo_labels(
    model: XGBClassifier,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    x_test: pd.DataFrame,
    model_seed: int,
    threshold: float,
    enabled: bool,
    params_override: dict[str, Any],
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, float]]:
    base_valid = clip_predictions(model.predict_proba(x_valid)[:, 1])
    base_test = clip_predictions(model.predict_proba(x_test)[:, 1])
    base_auc = safe_auc(y_valid, base_valid)
    if not enabled:
        return (
            base_valid,
            base_test,
            {
                "status": "skipped_disabled",
                "threshold": threshold,
                "candidate_count": 0,
                "base_auc": base_auc,
                "pl_auc": base_auc,
            },
            {},
        )
    mask = (base_test > threshold) | (base_test < (1.0 - threshold))
    pl_log: dict[str, Any] = {
        "status": "rejected",
        "threshold": threshold,
        "candidate_count": int(mask.sum()),
        "base_auc": base_auc,
        "pl_auc": base_auc,
    }
    if mask.sum() == 0:
        pl_log["status"] = "skipped_no_candidates"
        return base_valid, base_test, pl_log, {}

    x_train_pl = pd.concat([x_train, x_test.loc[mask].reset_index(drop=True)], axis=0, ignore_index=True)
    y_train_pl = np.concatenate([y_train, (base_test[mask] > 0.5).astype(np.int8)])
    sample_weight_pl = None
    if sample_weight is not None:
        pseudo_weight = np.ones(int(mask.sum()), dtype=np.float32)
        sample_weight_pl = np.concatenate([sample_weight, pseudo_weight])
    pl_model, pl_meta = train_xgb_model(
        x_train_pl,
        y_train_pl,
        x_valid,
        y_valid,
        model_seed,
        params_override=params_override,
        sample_weight=sample_weight_pl,
    )
    pl_valid = clip_predictions(pl_model.predict_proba(x_valid)[:, 1])
    pl_test = clip_predictions(pl_model.predict_proba(x_test)[:, 1])
    pl_auc = safe_auc(y_valid, pl_valid)
    pl_log["pl_auc"] = pl_auc
    if pl_auc > base_auc:
        pl_log["status"] = "accepted"
        return (
            pl_valid,
            pl_test,
            pl_log,
            {"pseudo_model_device": pl_meta.get("device"), "pseudo_best_iteration": pl_meta.get("best_iteration")},
        )
    return (
        base_valid,
        base_test,
        pl_log,
        {"pseudo_model_device": pl_meta.get("device"), "pseudo_best_iteration": pl_meta.get("best_iteration")},
    )


def validate_submission(
    sample_submission: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str,
    target_col: str,
    preds: Any,
    *,
    prediction_kind: str | None = None,
    target_labels: pd.Series | None = None,
) -> pd.DataFrame:
    target_cols = _resolve_submission_target_columns(
        sample_submission=sample_submission,
        id_col=id_col,
        target_col=target_col,
        preds=preds,
    )
    submission = build_submission_template_for_test(
        sample_submission=sample_submission,
        test_df=test_df,
        id_col=id_col,
        target_cols=target_cols,
    )
    assert id_col in submission.columns, f"Sample submission missing id column: {id_col}"
    for column in target_cols:
        assert column in submission.columns, f"Sample submission missing target column: {column}"
    assert list(submission.columns) == list(sample_submission.columns), "Submission columns changed unexpectedly."
    assert len(submission) == len(test_df), "Submission row count must match test row count."
    test_ids = test_df[id_col].reset_index(drop=True)
    sample_ids = submission[id_col].reset_index(drop=True)
    assert sample_ids.equals(test_ids), "Sample submission ids must align exactly with test ids."
    pred_table = _normalize_submission_predictions(
        preds=preds,
        target_cols=target_cols,
        row_count=len(test_df),
        prediction_kind=prediction_kind,
        target_labels=target_labels,
    )
    for column in target_cols:
        submission[column] = pred_table[column]
    return submission


def _resolve_submission_target_columns(
    *,
    sample_submission: pd.DataFrame,
    id_col: str,
    target_col: str,
    preds: Any,
) -> list[str]:
    if target_col in sample_submission.columns:
        return [target_col]
    prediction_cols = [str(col) for col in sample_submission.columns if col != id_col]
    if not prediction_cols:
        return [target_col]
    if isinstance(preds, dict):
        common = [col for col in prediction_cols if col in preds]
        if common:
            return common
    arr = np.asarray(preds)
    if arr.ndim == 2 and arr.shape[1] == len(prediction_cols):
        return prediction_cols
    return [target_col]


def _normalize_submission_predictions(
    *,
    preds: Any,
    target_cols: list[str],
    row_count: int,
    prediction_kind: str | None = None,
    target_labels: pd.Series | None = None,
) -> dict[str, np.ndarray]:
    if isinstance(preds, dict):
        normalized: dict[str, np.ndarray] = {}
        for column in target_cols:
            assert column in preds, f"Missing predictions for target column: {column}"
            values = np.asarray(preds[column], dtype=np.float64).ravel()
            assert values.shape[0] == row_count, (
                f"Prediction row count for {column} must match test row count: {values.shape[0]} != {row_count}"
            )
            normalized[column] = values
        if prediction_kind_is_probability(prediction_kind):
            _renormalize_probability_columns(normalized)
        if len(target_cols) == 1 and prediction_kind_is_probability(prediction_kind):
            normalized[target_cols[0]] = clip_predictions(normalized[target_cols[0]])
        elif len(target_cols) == 1:
            normalized[target_cols[0]] = normalize_structured_regression_predictions(
                normalized[target_cols[0]],
                prediction_kind=prediction_kind,
                target_labels=target_labels,
                target_col=target_cols[0],
            )
        return normalized

    arr = np.asarray(preds, dtype=np.float64)
    if arr.ndim == 1:
        assert len(target_cols) == 1, "1D predictions cannot fill multiple submission target columns."
        values = (
            clip_predictions(arr.ravel())
            if prediction_kind_is_probability(prediction_kind)
            else normalize_structured_regression_predictions(
                arr.ravel(),
                prediction_kind=prediction_kind,
                target_labels=target_labels,
                target_col=target_cols[0],
            )
        )
        assert values.shape[0] == row_count, (
            f"Prediction row count must match test row count: {values.shape[0]} != {row_count}"
        )
        return {target_cols[0]: values}
    if arr.ndim == 2:
        assert arr.shape[0] == row_count, (
            f"Prediction row count must match test row count: {arr.shape[0]} != {row_count}"
        )
        assert arr.shape[1] == len(target_cols), (
            f"Prediction column count must match target columns: {arr.shape[1]} != {len(target_cols)}"
        )
        normalized = {column: arr[:, idx] for idx, column in enumerate(target_cols)}
        if prediction_kind_is_probability(prediction_kind):
            _renormalize_probability_columns(normalized)
        return normalized
    raise AssertionError(f"Unsupported prediction shape for submission: {arr.shape}")


def _renormalize_probability_columns(predictions: dict[str, np.ndarray]) -> None:
    if len(predictions) <= 1:
        return
    matrix = np.column_stack([predictions[column] for column in predictions]).astype(np.float64)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=0.0)
    matrix = np.clip(matrix, 0.0, None)
    row_sums = matrix.sum(axis=1)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    normalized = matrix / row_sums[:, None]
    for idx, column in enumerate(predictions):
        predictions[column] = normalized[:, idx]


def safe_artifact_stem(value: str) -> str:
    stem = _ARTIFACT_STEM_RE.sub("_", str(value).strip()).strip("._-")
    return stem or "candidate"


def _fold_sort_key(item: tuple[str, np.ndarray]) -> tuple[int, str]:
    match = _FOLD_NUMBER_RE.search(item[0])
    if match is None:
        return (10**9, item[0])
    return (int(match.group(1)), item[0])


def _fold_number_from_key(fold_key: str, fallback: int) -> int:
    match = _FOLD_NUMBER_RE.search(fold_key)
    if match is None:
        return fallback
    return int(match.group(1))


def write_fold_intermediate_artifacts(
    *,
    output_dirs: list[Path],
    result: PipelineResult,
    sample_submission: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str,
    target_col: str,
) -> list[dict[str, Any]]:
    """Persist fold-level predictions plus a valid fold submission when possible."""

    records: list[dict[str, Any]] = []
    candidate_stem = safe_artifact_stem(result.name)
    for fallback_idx, (fold_key, test_preds) in enumerate(
        sorted(result.test_predictions_by_fold.items(), key=_fold_sort_key),
        start=1,
    ):
        fold_number = _fold_number_from_key(fold_key, fallback_idx)
        artifact_stem = f"{candidate_stem}_fold{fold_number}"
        test_arr = np.asarray(test_preds, dtype=np.float64)
        oof_arr = result.oof_predictions_by_fold.get(fold_key)
        valid_idx = result.valid_indices_by_fold.get(fold_key)
        record: dict[str, Any] = {
            "candidate": result.name,
            "fold": fold_number,
            "fold_key": fold_key,
            "test_predictions_path": f"test_preds_{artifact_stem}.npy",
            "oof_predictions_path": f"oof_preds_{artifact_stem}.npy" if oof_arr is not None else None,
            "metadata_path": f"preds_{artifact_stem}_metadata.json",
            "candidate_path": f"candidate_{artifact_stem}.json",
            "submission_path": submission_artifact_filename(stem=artifact_stem),
            "status": "pending",
        }
        metadata = {
            "candidate": result.name,
            "fold": fold_number,
            "fold_key": fold_key,
            "cv_score": float(result.cv_score),
            "test_prediction_count": int(test_arr.shape[0]),
            "expected_test_rows": int(len(test_df)),
            "valid_indices": np.asarray(valid_idx).tolist() if valid_idx is not None else None,
            "metadata": result.metadata,
        }
        if test_arr.shape[0] == len(test_df):
            try:
                submission = validate_submission(
                    sample_submission=sample_submission,
                    test_df=test_df,
                    id_col=id_col,
                    target_col=target_col,
                    preds=test_arr,
                    prediction_kind=result.metadata.get("prediction_kind") or result.metadata.get("target_semantics"),
                )
            except AssertionError as exc:
                record["status"] = "invalid_submission"
                record["reason"] = str(exc)
            else:
                mirror_df(str(record["submission_path"]), submission, output_dirs)
                record["status"] = "available"
        else:
            record["status"] = "skipped_submission"
            record["reason"] = f"fold test predictions have {test_arr.shape[0]} rows; expected {len(test_df)}"

        mirror_npy(str(record["test_predictions_path"]), test_arr, output_dirs)
        if oof_arr is not None:
            mirror_npy(str(record["oof_predictions_path"]), np.asarray(oof_arr, dtype=np.float64), output_dirs)
        mirror_json(str(record["metadata_path"]), metadata, output_dirs)
        mirror_json(str(record["candidate_path"]), {**record, "metadata": metadata}, output_dirs)
        records.append(record)
    return records


def write_submission_manifest(
    *,
    output_dirs: list[Path],
    final_result: PipelineResult,
    summary: dict[str, Any],
    fold_artifacts: list[dict[str, Any]] | None = None,
) -> None:
    for output_dir in output_dirs:
        submission_filename = submission_artifact_filename()
        submission_path = output_dir / submission_filename
        submission_sha = None
        if submission_path.exists():
            submission_sha = sha256(submission_path.read_bytes()).hexdigest()
        payload = {
            "artifact_class": "tabular",
            "submission_path": submission_filename,
            "staging_dir": None,
            "members": [],
            "selected_pipeline": final_result.name,
            "selected_kind": final_result.metadata.get("kind", "single"),
            "component_models": resolve_component_models(final_result),
            "metrics_summary_path": "metrics_summary.json",
            "cv_results_path": "cv_results.json",
            "sha256": submission_sha,
            "prediction_range": summary.get("prediction_range"),
            "fold_artifacts": fold_artifacts or summary.get("fold_artifacts", []),
        }
        requested_output_path = requested_non_tabular_submission_filename()
        if requested_output_path is not None and requested_output_path != submission_filename:
            payload["requested_output_path"] = requested_output_path
            payload["note"] = (
                "Tabular ensemble runtime emitted a tabular fallback because the requested output filename "
                "uses a non-tabular artifact suffix."
            )
        write_json(output_dir / "submission_manifest.json", payload)
