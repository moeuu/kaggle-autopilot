from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kagglebot.baseline_tokens import RLE_SEGMENTATION_COLUMN_TOKENS
from kagglebot.compute import Compute, detect_local_gpu
from kagglebot.exceptions import GPUNotAvailableError
from kagglebot.rna_structure import (
    RnaStructureTask,
    build_coordinate_baseline_predictions,
    detect_rna_structure_task,
    evaluate_coordinate_predictions,
    extract_target_id,
    load_rna_structure_task,
    write_rna_structure_submission,
)
from kagglebot.solver.evaluate import EvaluationResult
from kagglebot.solver.io import CompetitionData, load_competition_data, read_table, write_submission, write_table
from kagglebot.solver.metrics import compute_metric, infer_direction, metric_requires_proba
from kagglebot.submission_sample_discovery import (
    TABULAR_INPUT_SUFFIXES_ORDERED,
    is_tabular_data_path,
    tabular_suffix,
)
from kagglebot.training import (
    OFFICIAL_COMBINED_METRIC,
    compute_map50_95,
    derive_right_place,
    detect_yolo_submission_task,
    evaluate_combined_metric,
    find_yolo_data_layout,
    format_prediction_string,
    infer_detection_submission_schema,
    infer_pairwise_object_classes,
    predict_detector,
    prepare_ultralytics_dataset,
    resolve_yolo_image_reference,
    train_detector,
    train_val_split,
    tune_right_place_params,
)


@dataclass(frozen=True)
class TrainingOutcome:
    submission_path: Path
    evaluation: EvaluationResult
    model_name: str
    model_summary: dict[str, object]
    accelerator: str


def train_evaluate_and_predict(
    *,
    data_dir: Path,
    output_path: Path,
    compute: Compute,
    strict_accelerator: bool,
    seed: int,
    score_source: str,
    metric: str,
    direction: str,
    holdout_frac: float,
    cv_folds: int,
    plan_score_source: str | None,
    target_override: str | None,
) -> TrainingOutcome:
    del cv_folds, plan_score_source

    if detect_rna_structure_task(data_dir):
        return _train_rna_structure_submission(
            data_dir=data_dir,
            output_path=output_path,
            seed=seed,
        )

    # Route detector-style sample submissions before tabular target overrides can misclassify them.
    base_data = load_competition_data(data_dir)
    if detect_yolo_submission_task(data_dir=data_dir, sample_df=base_data.sample):
        return _train_vision_yolo_submission(
            data_dir=data_dir,
            sample_df=base_data.sample,
            output_path=output_path,
            compute=compute,
            strict_accelerator=strict_accelerator,
            seed=seed,
        )
    if detect_rle_segmentation_submission(base_data.sample):
        return _train_rle_segmentation_submission(
            data=base_data,
            output_path=output_path,
        )

    data = (
        load_competition_data(data_dir, target_column_override=target_override)
        if target_override is not None
        else base_data
    )
    return _train_tabular_baseline_submission(
        data=data,
        output_path=output_path,
        seed=seed,
        score_source=score_source,
        metric=metric,
        direction=direction,
        holdout_frac=holdout_frac,
    )


def detect_rle_segmentation_submission(sample_df: pd.DataFrame) -> bool:
    return bool(_rle_segmentation_columns(sample_df))


def _rle_segmentation_columns(sample_df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for col in sample_df.columns:
        normalized = _normalize_submission_column_token(str(col))
        if normalized in RLE_SEGMENTATION_COLUMN_TOKENS:
            columns.append(str(col))
            continue
        if "encodedpixels" in normalized or "runlength" in normalized or normalized.endswith("rle"):
            columns.append(str(col))
    return columns


def _normalize_submission_column_token(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _train_rle_segmentation_submission(*, data: CompetitionData, output_path: Path) -> TrainingOutcome:
    mask_columns = _rle_segmentation_columns(data.sample)
    if not mask_columns:
        raise RuntimeError("RLE segmentation baseline requires at least one mask/RLE submission column.")

    submission = data.sample.copy()
    for col in mask_columns:
        submission[col] = "-"
    submission_path = write_table(submission, output_path)

    model_summary = {
        "model": "rle_empty_mask_baseline",
        "rows": int(len(submission)),
        "mask_columns": mask_columns,
        "id_column": data.id_column,
    }
    print(
        f"[local train] model=rle_empty_mask_baseline rows={len(submission)} mask_columns={','.join(mask_columns)}",
        flush=True,
    )
    return TrainingOutcome(
        submission_path=submission_path,
        evaluation=EvaluationResult(
            score_source="holdout",
            metric="segmentation_rle_placeholder",
            direction="maximize",
            value=0.0,
            std=None,
            train_score=None,
            val_score=None,
            fold_scores=None,
        ),
        model_name="rle_empty_mask_baseline",
        model_summary=model_summary,
        accelerator="cpu",
    )


def _train_tabular_baseline_submission(
    *,
    data: CompetitionData,
    output_path: Path,
    seed: int,
    score_source: str,
    metric: str,
    direction: str,
    holdout_frac: float,
) -> TrainingOutcome:
    if not data.target_columns:
        raise RuntimeError("Tabular baseline requires at least one target column.")

    train_index, valid_index = _tabular_holdout_indices(
        train=data.train,
        primary_target=data.target_column,
        task=data.task,
        seed=seed,
        holdout_frac=holdout_frac,
        group_column=data.group_column,
        time_column=data.time_column,
    )
    predictions: dict[str, np.ndarray] | np.ndarray
    predictions_by_target: dict[str, np.ndarray] = {}
    validation_scores: list[float] = []
    target_summaries: dict[str, dict[str, object]] = {}

    for target_col in data.target_columns:
        task = data.task_by_target.get(target_col, data.task)
        prediction_kind = data.prediction_kind_by_target.get(target_col, data.prediction_kind)
        target_output, target_score, target_summary = _fit_predict_tabular_target(
            data=data,
            target_col=target_col,
            task=task,
            prediction_kind=prediction_kind,
            train_index=train_index,
            valid_index=valid_index,
            seed=seed,
            metric=metric,
        )
        if isinstance(target_output, dict):
            predictions_by_target.update(target_output)
        else:
            predictions_by_target[target_col] = target_output
        if target_score is not None and np.isfinite(target_score):
            validation_scores.append(float(target_score))
        target_summaries[target_col] = target_summary

    survival_score_col = _survival_score_submission_column(data)
    if survival_score_col is not None:
        event_col, time_col = _survival_event_time_columns(data.target_columns)
        predictions = {
            survival_score_col: _survival_risk_scores(
                event_predictions=np.asarray(predictions_by_target[event_col], dtype=float),
                time_predictions=np.asarray(predictions_by_target[time_col], dtype=float),
            )
        }
        target_summaries[survival_score_col] = {
            "model": "survival_risk_score",
            "event_column": event_col,
            "time_column": time_col,
            "prediction_column": survival_score_col,
        }
        submission_path = write_submission(
            sample=data.sample,
            test=data.test,
            preds=predictions,
            id_column=data.id_column,
            target_columns=[survival_score_col],
            output_path=output_path,
        )
    elif data.prediction_kind in {"probability_columns", "multi_label_columns"} and len(data.target_columns) == 1:
        prediction_columns = [str(col) for col in data.sample.columns if col != data.id_column]
        predictions = {col: predictions_by_target[col] for col in prediction_columns}
        submission_path = write_submission(
            sample=data.sample,
            test=data.test,
            preds=predictions,
            id_column=data.id_column,
            target_columns=prediction_columns,
            output_path=output_path,
        )
    elif (
        data.prediction_kind in {"quantile_columns", "prediction_interval_columns", "continuous_columns"}
        and len(data.target_columns) == 1
    ):
        target_col = data.target_columns[0]
        prediction_columns = [str(col) for col in data.sample.columns if col != data.id_column]
        predictions = _expand_numeric_multi_column_predictions(
            data=data,
            target_col=target_col,
            base_predictions=np.asarray(predictions_by_target[target_col], dtype=float),
            prediction_columns=prediction_columns,
            prediction_kind=data.prediction_kind,
        )
        if target_col in target_summaries:
            target_summaries[target_col]["expanded_prediction_kind"] = data.prediction_kind
            target_summaries[target_col]["expanded_prediction_columns"] = prediction_columns
        submission_path = write_submission(
            sample=data.sample,
            test=data.test,
            preds=predictions,
            id_column=data.id_column,
            target_columns=prediction_columns,
            output_path=output_path,
        )
    else:
        predictions = {col: predictions_by_target[col] for col in data.target_columns}
        submission_path = write_submission(
            sample=data.sample,
            test=data.test,
            preds=predictions,
            id_column=data.id_column,
            target_columns=data.target_columns,
            output_path=output_path,
        )

    value = float(np.mean(validation_scores)) if validation_scores else 0.0
    resolved_direction = infer_direction(metric, direction)
    temporal_calendar_feature_columns: list[str] = []
    for summary in target_summaries.values():
        for feature in summary.get("temporal_calendar_features", []):
            feature_name = str(feature)
            if feature_name not in temporal_calendar_feature_columns:
                temporal_calendar_feature_columns.append(feature_name)
    model_summary = {
        "model": "local_tabular_baseline",
        "targets": list(data.target_columns),
        "task_by_target": dict(data.task_by_target),
        "prediction_kind_by_target": dict(data.prediction_kind_by_target),
        "feature_columns": list(data.feature_columns),
        "temporal_calendar_feature_columns": temporal_calendar_feature_columns,
        "group_column": data.group_column,
        "time_column": data.time_column,
        "split_strategy": "group_shuffle_split"
        if data.group_column
        else "timeseries_holdout"
        if data.time_column
        else "holdout",
        "target_summaries": target_summaries,
    }
    print(
        "[local train] model=local_tabular_baseline "
        f"targets={len(data.target_columns)} "
        f"metric={metric} "
        f"val={value:.6f}",
        flush=True,
    )

    return TrainingOutcome(
        submission_path=submission_path,
        evaluation=EvaluationResult(
            score_source=score_source,
            metric=str(metric),
            direction=resolved_direction,
            value=value,
            std=0.0 if validation_scores else None,
            train_score=None,
            val_score=value,
            fold_scores=validation_scores or None,
        ),
        model_name="local_tabular_baseline",
        model_summary=model_summary,
        accelerator="cpu",
    )


def _fit_predict_tabular_target(
    *,
    data: CompetitionData,
    target_col: str,
    task: str,
    prediction_kind: str,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    seed: int,
    metric: str,
) -> tuple[np.ndarray | dict[str, np.ndarray], float | None, dict[str, object]]:
    if target_col not in data.train.columns:
        if task == "unsupervised":
            return _fit_predict_unsupervised_score(data=data, target_col=target_col)
        return _constant_missing_target_predictions(data=data, target_col=target_col, prediction_kind=prediction_kind)

    train = data.train.iloc[train_index].copy()
    valid = data.train.iloc[valid_index].copy()
    feature_cols = [col for col in data.feature_columns if col in data.train.columns and col in data.test.columns]
    train_labeled = train[train[target_col].notna()].copy()
    valid_labeled = valid[valid[target_col].notna()].copy() if not valid.empty else valid
    full_labeled = data.train[data.train[target_col].notna()].copy()
    y_train = train_labeled[target_col]
    y_valid = valid_labeled[target_col] if not valid_labeled.empty else pd.Series(dtype=y_train.dtype)
    output_prediction_kind = _metric_aware_prediction_kind(
        task=task,
        prediction_kind=prediction_kind,
        metric=metric,
        y=y_train,
    )
    sample_weight_col = data.sample_weight_column
    train_weight = _sample_weights_for_frame(train_labeled, sample_weight_col)
    valid_weight = _sample_weights_for_frame(valid_labeled, sample_weight_col)
    full_weight = _sample_weights_for_frame(full_labeled, sample_weight_col)

    if task == "text" or prediction_kind == "text":
        return _fit_predict_text_target(
            data=data,
            target_col=target_col,
            train_labeled=train_labeled,
            valid_labeled=valid_labeled,
            full_labeled=full_labeled,
            feature_cols=feature_cols,
        )

    if prediction_kind == "multi_label_columns":
        return _fit_predict_multi_label_columns(
            data=data,
            target_col=target_col,
            train_labeled=train_labeled,
            valid_labeled=valid_labeled,
            full_labeled=full_labeled,
            feature_cols=feature_cols,
            seed=seed,
        )

    if not feature_cols or y_train.dropna().empty:
        return _constant_target_predictions(
            data=data,
            target_col=target_col,
            prediction_kind=output_prediction_kind,
            valid_index=valid_index,
            metric=metric,
        )

    test_features = data.test.copy()
    temporal_calendar_features = _add_temporal_calendar_features(
        frames=[train_labeled, valid_labeled, full_labeled],
        test=test_features,
        feature_cols=feature_cols,
    )
    feature_cols = [*feature_cols, *temporal_calendar_features]

    x_train = train_labeled[feature_cols]
    x_valid = valid_labeled[feature_cols] if not valid_labeled.empty else pd.DataFrame(columns=feature_cols)
    x_full = full_labeled[feature_cols]
    x_test = test_features[feature_cols]

    if task == "classification":
        if y_train.nunique(dropna=True) < 2:
            return _constant_target_predictions(
                data=data,
                target_col=target_col,
                prediction_kind=output_prediction_kind,
                valid_index=valid_index,
                metric=metric,
            )
        estimator = Pipeline(
            steps=[
                ("preprocess", _build_tabular_preprocessor(x_train)),
                (
                    "model",
                    LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced"),
                ),
            ]
        )
        try:
            estimator.fit(x_train, y_train, **_pipeline_sample_weight_kwargs(train_weight))
            valid_score = _score_tabular_predictions(
                estimator=estimator,
                x_valid=x_valid,
                y_valid=y_valid,
                task=task,
                metric=metric,
                sample_weight=valid_weight,
            )
            estimator.fit(x_full, full_labeled[target_col], **_pipeline_sample_weight_kwargs(full_weight))
            test_output = _classification_test_output(
                estimator=estimator,
                x_test=x_test,
                sample=data.sample,
                id_column=data.id_column,
                target_col=target_col,
                prediction_kind=output_prediction_kind,
            )
            return (
                test_output,
                valid_score,
                {
                    "model": "logistic_regression",
                    "features": len(feature_cols),
                    "sample_weight_column": sample_weight_col,
                    "prediction_kind": output_prediction_kind,
                    "temporal_calendar_features": temporal_calendar_features,
                },
            )
        except Exception as exc:  # noqa: BLE001
            constant, score, summary = _constant_target_predictions(
                data=data,
                target_col=target_col,
                prediction_kind=output_prediction_kind,
                valid_index=valid_index,
                metric=metric,
            )
            summary["fallback_reason"] = str(exc)
            return constant, score, summary

    estimator = Pipeline(
        steps=[
            ("preprocess", _build_tabular_preprocessor(x_train)),
            ("model", Ridge()),
        ]
    )
    try:
        use_log1p_target = prediction_kind != "ordinal" and _should_log1p_regression_target(
            data.train[target_col], column_name=target_col
        )
        fit_y_train = _log1p_target_values(y_train) if use_log1p_target else y_train
        fit_y_full = _log1p_target_values(full_labeled[target_col]) if use_log1p_target else full_labeled[target_col]
        estimator.fit(x_train, fit_y_train, **_pipeline_sample_weight_kwargs(train_weight))
        valid_pred = estimator.predict(x_valid) if len(x_valid) else np.array([])
        if use_log1p_target:
            valid_pred = _expm1_predictions(valid_pred)
            valid_pred = _clip_structured_regression_predictions(
                valid_pred, data.train[target_col], column_name=target_col
            )
        valid_score = _safe_metric(metric=metric, y_true=y_valid, y_pred=valid_pred, sample_weight=valid_weight)
        estimator.fit(x_full, fit_y_full, **_pipeline_sample_weight_kwargs(full_weight))
        test_pred = np.asarray(estimator.predict(x_test))
        if use_log1p_target:
            test_pred = _expm1_predictions(test_pred)
        if prediction_kind == "ordinal":
            test_pred = _ordinal_predictions_from_continuous(test_pred, data.train[target_col])
        else:
            test_pred = _clip_structured_regression_predictions(
                test_pred, data.train[target_col], column_name=target_col
            )
        target_summary = {
            "model": "ridge_log1p" if use_log1p_target else "ridge",
            "features": len(feature_cols),
            "sample_weight_column": sample_weight_col,
            "temporal_calendar_features": temporal_calendar_features,
        }
        if use_log1p_target:
            target_summary["target_transform"] = "log1p"
            target_summary["inverse_transform"] = "expm1"
        return (
            test_pred,
            valid_score,
            target_summary,
        )
    except Exception as exc:  # noqa: BLE001
        constant, score, summary = _constant_target_predictions(
            data=data,
            target_col=target_col,
            prediction_kind=prediction_kind,
            valid_index=valid_index,
            metric=metric,
        )
        summary["fallback_reason"] = str(exc)
        return constant, score, summary


def _tabular_holdout_indices(
    *,
    train: pd.DataFrame,
    primary_target: str,
    task: str,
    seed: int,
    holdout_frac: float,
    group_column: str | None = None,
    time_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(train))
    if len(indices) < 3:
        return indices, indices[:0]
    test_size = min(max(float(holdout_frac or 0.2), 0.1), 0.5)
    if group_column and group_column in train.columns:
        grouped_split = _group_holdout_indices(
            train=train,
            group_column=group_column,
            indices=indices,
            test_size=test_size,
            seed=seed,
        )
        if grouped_split is not None:
            return grouped_split
    if time_column and time_column in train.columns:
        time_split = _chronological_holdout_indices(
            train=train,
            time_column=time_column,
            indices=indices,
            test_size=test_size,
        )
        if time_split is not None:
            return time_split
    stratify = None
    if task == "classification" and primary_target in train.columns:
        counts = train[primary_target].value_counts(dropna=False)
        if len(counts) > 1 and int(counts.min()) >= 2:
            stratify = train[primary_target]
    try:
        train_idx, valid_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        train_idx, valid_idx = train_test_split(indices, test_size=test_size, random_state=seed)
    return np.asarray(train_idx), np.asarray(valid_idx)


def _chronological_holdout_indices(
    *,
    train: pd.DataFrame,
    time_column: str,
    indices: np.ndarray,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    order_values = _temporal_order_values(train[time_column])
    if order_values is None:
        return None
    ordered = indices[np.argsort(order_values, kind="mergesort")]
    valid_size = max(1, int(round(len(ordered) * test_size)))
    valid_size = min(valid_size, len(ordered) - 1)
    if valid_size <= 0:
        return None
    train_idx = ordered[:-valid_size]
    valid_idx = ordered[-valid_size:]
    if len(train_idx) == 0 or len(valid_idx) == 0:
        return None
    return np.asarray(train_idx), np.asarray(valid_idx)


def _temporal_order_values(series: pd.Series) -> np.ndarray | None:
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    else:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        values = (
            pd.Series(parsed).map(lambda value: value.timestamp() if pd.notna(value) else np.nan).to_numpy(dtype=float)
        )
    if values.size == 0 or float(np.isfinite(values).mean()) < 0.8:
        return None
    if np.nanmax(values) <= np.nanmin(values):
        return None
    fill_value = float(np.nanmedian(values[np.isfinite(values)]))
    return np.where(np.isfinite(values), values, fill_value)


def _group_holdout_indices(
    *,
    train: pd.DataFrame,
    group_column: str,
    indices: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    groups = train[group_column].astype(str).fillna("__missing_group__").to_numpy()
    if len(np.unique(groups)) < 2:
        return None
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    try:
        train_idx, valid_idx = next(splitter.split(indices, groups=groups))
    except ValueError:
        return None
    if len(train_idx) == 0 or len(valid_idx) == 0:
        return None
    return np.asarray(train_idx), np.asarray(valid_idx)


def _build_tabular_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])]
    categorical_cols = [col for col in frame.columns if col not in numeric_cols]
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=64)),
                    ]
                ),
                categorical_cols,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _add_temporal_calendar_features(
    *,
    frames: list[pd.DataFrame],
    test: pd.DataFrame,
    feature_cols: list[str],
) -> list[str]:
    reference = next((frame for frame in frames if not frame.empty), None)
    if reference is None:
        reference = test
    added: list[str] = []
    for col in feature_cols:
        if col not in reference.columns or col not in test.columns:
            continue
        train_dates = _parse_calendar_feature(reference[col])
        test_dates = _parse_calendar_feature(test[col])
        if train_dates is None or test_dates is None:
            continue
        prefix = f"__time_{_safe_feature_name(col)}"
        derived = _calendar_feature_values(train_dates, prefix=prefix)
        derived_test = _calendar_feature_values(test_dates, prefix=prefix)
        new_cols = list(derived.columns)
        for frame in frames:
            if col not in frame.columns:
                continue
            frame_dates = _parse_calendar_feature(frame[col])
            if frame_dates is None:
                for new_col in new_cols:
                    frame[new_col] = np.nan
                continue
            frame_derived = _calendar_feature_values(frame_dates, prefix=prefix)
            for new_col in new_cols:
                frame[new_col] = frame_derived[new_col].to_numpy()
        for new_col in new_cols:
            test[new_col] = derived_test[new_col].to_numpy()
            if new_col not in added:
                added.append(new_col)
    return added


def _parse_calendar_feature(series: pd.Series) -> pd.Series | None:
    if not (
        pd.api.types.is_datetime64_any_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        return None
    values = series.dropna()
    if values.empty:
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if float(pd.Series(parsed).notna().mean()) < 0.8:
        return None
    return pd.Series(parsed)


def _calendar_feature_values(parsed: pd.Series, *, prefix: str) -> pd.DataFrame:
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


def _safe_feature_name(value: object) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return normalized or "feature"


def _sample_weights_for_frame(frame: pd.DataFrame, column: str | None) -> np.ndarray | None:
    if not column or column not in frame.columns or frame.empty:
        return None
    weights = pd.to_numeric(frame[column], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    weights = np.where(np.isfinite(weights), weights, 1.0)
    weights = np.clip(weights, 0.0, None)
    if weights.size == 0 or float(weights.sum()) <= 0.0:
        return None
    return weights.astype(np.float64, copy=False)


def _pipeline_sample_weight_kwargs(sample_weight: np.ndarray | None) -> dict[str, np.ndarray]:
    if sample_weight is None:
        return {}
    return {"model__sample_weight": sample_weight}


def _classification_test_output(
    *,
    estimator: Pipeline,
    x_test: pd.DataFrame,
    sample: pd.DataFrame,
    id_column: str | None,
    target_col: str,
    prediction_kind: str,
) -> np.ndarray | dict[str, np.ndarray]:
    if prediction_kind == "probability_columns":
        prediction_cols = [str(col) for col in sample.columns if col != id_column]
        probabilities = _predict_probability_columns(
            estimator=estimator,
            x_test=x_test,
            prediction_cols=prediction_cols,
        )
        return {col: probabilities[:, idx] for idx, col in enumerate(prediction_cols)}
    if prediction_kind in {"probability", "continuous"} and hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(x_test)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return np.asarray(probabilities[:, -1])
    return np.asarray(estimator.predict(x_test))


def _metric_aware_prediction_kind(
    *,
    task: str,
    prediction_kind: str,
    metric: str,
    y: pd.Series,
) -> str:
    if task != "classification" or prediction_kind != "class" or not metric_requires_proba(metric):
        return prediction_kind
    if y.nunique(dropna=True) != 2:
        return prediction_kind
    return "probability"


def _predict_probability_columns(
    *, estimator: Pipeline, x_test: pd.DataFrame, prediction_cols: list[str]
) -> np.ndarray:
    if not hasattr(estimator, "predict_proba"):
        return np.full((len(x_test), len(prediction_cols)), 1.0 / max(len(prediction_cols), 1))
    probabilities = np.asarray(estimator.predict_proba(x_test), dtype=float)
    classes = [str(value) for value in getattr(estimator[-1], "classes_", [])]
    output = np.zeros((len(x_test), len(prediction_cols)), dtype=float)
    for col_idx, col in enumerate(prediction_cols):
        normalized_col = _normalize_probability_column_name(col)
        class_idx = next(
            (idx for idx, value in enumerate(classes) if _normalize_probability_column_name(value) == normalized_col),
            None,
        )
        if class_idx is not None and class_idx < probabilities.shape[1]:
            output[:, col_idx] = probabilities[:, class_idx]
    row_sums = output.sum(axis=1)
    missing = row_sums <= 0
    if missing.any():
        output[missing, :] = 1.0 / max(len(prediction_cols), 1)
        row_sums = output.sum(axis=1)
    return output / row_sums[:, None]


def _normalize_probability_column_name(value: object) -> str:
    normalized = "".join(ch for ch in str(value).lower() if ch.isalnum())
    for prefix in ("class", "target", "label", "prob", "probability", "pred"):
        if normalized.startswith(prefix) and len(normalized) > len(prefix):
            normalized = normalized[len(prefix) :]
            break
    for suffix in ("probability", "proba", "prob", "prediction", "pred", "score"):
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def _score_tabular_predictions(
    *,
    estimator: Pipeline,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    task: str,
    metric: str,
    sample_weight: np.ndarray | None = None,
) -> float | None:
    if len(x_valid) == 0 or y_valid.empty:
        return None
    if task == "classification" and metric_requires_proba(metric) and hasattr(estimator, "predict_proba"):
        pred = estimator.predict_proba(x_valid)
        if pred.ndim == 2 and pred.shape[1] == 2:
            pred = pred[:, 1]
    else:
        pred = estimator.predict(x_valid)
    return _safe_metric(metric=metric, y_true=y_valid, y_pred=pred, sample_weight=sample_weight)


def _safe_metric(*, metric: str, y_true, y_pred, sample_weight: np.ndarray | None = None) -> float | None:
    if len(y_true) == 0:
        return None
    try:
        return float(compute_metric(metric, y_true, y_pred, sample_weight=sample_weight))
    except Exception:  # noqa: BLE001
        try:
            if pd.api.types.is_numeric_dtype(pd.Series(y_true)):
                return float(np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight)))
            return float(accuracy_score(y_true, y_pred, sample_weight=sample_weight))
        except Exception:  # noqa: BLE001
            return None


def _ordinal_predictions_from_continuous(predictions: np.ndarray, labels: pd.Series) -> np.ndarray:
    numeric_labels = pd.to_numeric(labels, errors="coerce").dropna()
    if numeric_labels.empty:
        return np.rint(np.asarray(predictions, dtype=float)).astype(int)
    lower = float(numeric_labels.min())
    upper = float(numeric_labels.max())
    rounded = np.rint(np.asarray(predictions, dtype=float))
    return np.clip(rounded, lower, upper).astype(int)


def _clip_structured_regression_predictions(
    predictions: np.ndarray, labels: pd.Series, *, column_name: str
) -> np.ndarray:
    if _looks_like_count_regression_target(labels, column_name=column_name):
        return np.clip(np.asarray(predictions, dtype=float), 0.0, None)
    if _looks_like_positive_skew_regression_target(labels, column_name=column_name):
        return np.clip(np.asarray(predictions, dtype=float), 0.0, None)
    bounds = _bounded_regression_bounds(labels, column_name=column_name)
    if bounds is None:
        return predictions
    lower, upper = bounds
    return np.clip(np.asarray(predictions, dtype=float), lower, upper)


def _should_log1p_regression_target(labels: pd.Series, *, column_name: str) -> bool:
    return _looks_like_count_regression_target(
        labels, column_name=column_name
    ) or _looks_like_positive_skew_regression_target(labels, column_name=column_name)


def _log1p_target_values(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return np.log1p(np.clip(numeric, 0.0, None))


def _expm1_predictions(predictions: np.ndarray) -> np.ndarray:
    return np.expm1(np.asarray(predictions, dtype=float))


def _looks_like_count_regression_target(target: pd.Series, *, column_name: str) -> bool:
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


def _bounded_regression_bounds(labels: pd.Series, *, column_name: str) -> tuple[float, float] | None:
    if not _looks_like_bounded_regression_target(labels, column_name=column_name):
        return None
    values = pd.to_numeric(labels.dropna(), errors="coerce").dropna()
    if values.empty:
        return None
    if float(values.max()) <= 1.0:
        return 0.0, 1.0
    return 0.0, 100.0


def _looks_like_positive_skew_regression_target(target: pd.Series, *, column_name: str) -> bool:
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


def _fit_predict_multi_label_columns(
    *,
    data: CompetitionData,
    target_col: str,
    train_labeled: pd.DataFrame,
    valid_labeled: pd.DataFrame,
    full_labeled: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
) -> tuple[dict[str, np.ndarray], float | None, dict[str, object]]:
    prediction_cols = [str(col) for col in data.sample.columns if col != data.id_column]
    observed_labels = _observed_multi_label_values(full_labeled[target_col], column_name=target_col)
    column_to_label = _map_multi_label_columns(prediction_cols, observed_labels)
    output: dict[str, np.ndarray] = {}
    scores: list[float] = []
    model_kinds: dict[str, str] = {}

    train_sets = _multi_label_sets(train_labeled[target_col], column_name=target_col)
    valid_sets = _multi_label_sets(valid_labeled[target_col], column_name=target_col)
    full_sets = _multi_label_sets(full_labeled[target_col], column_name=target_col)

    for column in prediction_cols:
        label = column_to_label[column]
        train_binary = np.asarray([label in labels for labels in train_sets], dtype=int)
        valid_binary = np.asarray([label in labels for labels in valid_sets], dtype=int)
        full_binary = np.asarray([label in labels for labels in full_sets], dtype=int)
        prior = float(full_binary.mean()) if full_binary.size else 0.0

        if not feature_cols or len(np.unique(train_binary)) < 2 or len(np.unique(full_binary)) < 2:
            output[column] = np.full(len(data.test), prior, dtype=float)
            if valid_binary.size:
                scores.append(float(accuracy_score(valid_binary, np.full(valid_binary.shape, prior >= 0.5))))
            model_kinds[column] = "constant_prior"
            continue

        estimator = Pipeline(
            steps=[
                ("preprocess", _build_tabular_preprocessor(train_labeled[feature_cols])),
                (
                    "model",
                    LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced"),
                ),
            ]
        )
        try:
            estimator.fit(train_labeled[feature_cols], train_binary)
            if valid_binary.size:
                valid_pred = estimator.predict(valid_labeled[feature_cols])
                scores.append(float(accuracy_score(valid_binary, valid_pred)))
            estimator.fit(full_labeled[feature_cols], full_binary)
            probabilities = np.asarray(estimator.predict_proba(data.test[feature_cols]), dtype=float)
            classes = np.asarray(estimator.named_steps["model"].classes_, dtype=int)
            positive_idx = int(np.where(classes == 1)[0][0]) if 1 in classes else probabilities.shape[1] - 1
            output[column] = np.clip(probabilities[:, positive_idx], 0.0, 1.0)
            model_kinds[column] = "logistic_regression_ovr"
        except Exception:  # noqa: BLE001
            output[column] = np.full(len(data.test), prior, dtype=float)
            model_kinds[column] = "constant_prior_fallback"

    score = float(np.mean(scores)) if scores else None
    return (
        output,
        score,
        {
            "model": "multi_label_one_vs_rest",
            "features": len(feature_cols),
            "prediction_kind": "multi_label_columns",
            "prediction_columns": prediction_cols,
            "column_to_label": column_to_label,
            "model_kinds": model_kinds,
        },
    )


def _observed_multi_label_values(values: pd.Series, *, column_name: str) -> list[str]:
    observed: list[str] = []
    seen: set[str] = set()
    for labels in _multi_label_sets(values, column_name=column_name):
        for label in labels:
            if label not in seen:
                seen.add(label)
                observed.append(label)
    return observed


def _multi_label_sets(values: pd.Series, *, column_name: str) -> list[set[str]]:
    allow_whitespace = _multi_label_allows_whitespace(column_name)
    return [
        {
            _normalize_probability_column_name(label)
            for label in _split_multi_label_prediction_value(value, allow_whitespace)
        }
        for value in values
    ]


def _split_multi_label_prediction_value(value: object, allow_whitespace: bool) -> list[str]:
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
    return [label for label in labels if len(label) <= 48 and re.fullmatch(r"[A-Za-z0-9_.:+-]+", label)]


def _multi_label_allows_whitespace(column_name: str) -> bool:
    column_tokens = re.findall(r"[a-z0-9]+", str(column_name).lower())
    tokens = set(column_tokens)
    compact = "".join(column_tokens)
    return bool(tokens & {"labels", "tags", "classes", "categories"}) or "multilabel" in compact


def _map_multi_label_columns(prediction_cols: list[str], observed_labels: list[str]) -> dict[str, str]:
    observed_by_normalized = {_normalize_probability_column_name(label): label for label in observed_labels}
    mapping: dict[str, str] = {}
    for column in prediction_cols:
        normalized = _normalize_probability_column_name(column)
        mapping[column] = observed_by_normalized.get(normalized, normalized)
    return mapping


def _safe_text_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float | None:
    if y_true.empty:
        return None
    try:
        return float(accuracy_score(y_true.astype(str), pd.Series(y_pred).astype(str)))
    except Exception:  # noqa: BLE001
        return None


def _fit_predict_text_target(
    *,
    data: CompetitionData,
    target_col: str,
    train_labeled: pd.DataFrame,
    valid_labeled: pd.DataFrame,
    full_labeled: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, float | None, dict[str, object]]:
    default_value = _text_baseline_value(data.train[target_col], data.sample, target_col)
    selected_cols = _text_feature_columns(train=data.train, test=data.test, feature_cols=feature_cols)
    valid_y = valid_labeled[target_col] if target_col in valid_labeled else pd.Series(dtype=object)
    if not selected_cols or full_labeled.empty:
        pred = np.array([default_value] * len(data.test), dtype=object)
        valid_pred = np.array([default_value] * len(valid_labeled), dtype=object)
        return (
            pred,
            _safe_text_accuracy(valid_y, valid_pred),
            {
                "model": "constant_text",
                "default": str(default_value),
                "text_feature_columns": selected_cols,
            },
        )

    train_text = _combine_text_features(train_labeled, selected_cols)
    valid_text = _combine_text_features(valid_labeled, selected_cols)
    full_text = _combine_text_features(full_labeled, selected_cols)
    test_text = _combine_text_features(data.test, selected_cols)

    valid_pred = _predict_tfidf_nearest_text(
        train_text=train_text,
        train_values=train_labeled[target_col],
        query_text=valid_text,
        fallback_value=default_value,
    )
    pred = _predict_tfidf_nearest_text(
        train_text=full_text,
        train_values=full_labeled[target_col],
        query_text=test_text,
        fallback_value=default_value,
    )
    return (
        pred,
        _safe_text_accuracy(valid_y, valid_pred),
        {
            "model": "tfidf_nearest_neighbor",
            "default": str(default_value),
            "text_feature_columns": selected_cols,
        },
    )


def _text_baseline_value(y: pd.Series, sample: pd.DataFrame, target_col: str) -> str:
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


def _text_feature_columns(*, train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> list[str]:
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


def _combine_text_features(frame: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    if not feature_cols:
        return pd.Series([""] * len(frame), index=frame.index)
    parts = []
    for col in feature_cols:
        if col not in frame.columns:
            values = pd.Series([""] * len(frame), index=frame.index)
        else:
            values = frame[col].fillna("").astype(str).str.strip()
        parts.append(str(col) + "=" + values)
    return pd.concat(parts, axis=1).agg(" ".join, axis=1).str.strip()


def _predict_tfidf_nearest_text(
    *,
    train_text: pd.Series,
    train_values: pd.Series,
    query_text: pd.Series,
    fallback_value: str,
) -> np.ndarray:
    values = train_values.dropna().astype(str).str.strip()
    valid_mask = values != ""
    values = values[valid_mask]
    index_text = train_text.loc[values.index].fillna("").astype(str).str.strip()
    usable_mask = index_text != ""
    values = values[usable_mask]
    index_text = index_text[usable_mask]
    if values.empty or index_text.empty:
        return np.repeat(fallback_value, len(query_text)).astype(object)

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
        return np.repeat(fallback_value, len(query_text)).astype(object)

    preds: list[str] = []
    value_array = values.to_numpy(dtype=str)
    batch_size = 256
    for start in range(0, query_matrix.shape[0], batch_size):
        stop = min(start + batch_size, query_matrix.shape[0])
        sims = (query_matrix[start:stop] @ train_matrix.T).toarray()
        best_idx = sims.argmax(axis=1) if sims.size else np.array([], dtype=int)
        best_score = sims.max(axis=1) if sims.size else np.array([], dtype=float)
        for idx, score in zip(best_idx, best_score, strict=False):
            preds.append(fallback_value if not np.isfinite(score) or score <= 0 else str(value_array[int(idx)]))
    return np.asarray(preds, dtype=object)


def _fit_predict_unsupervised_score(
    *,
    data: CompetitionData,
    target_col: str,
) -> tuple[np.ndarray, float | None, dict[str, object]]:
    feature_cols = [col for col in data.feature_columns if col in data.train.columns and col in data.test.columns]
    scores = _unsupervised_anomaly_scores(train=data.train, test=data.test, feature_cols=feature_cols)
    numeric_features = [col for col in feature_cols if pd.api.types.is_numeric_dtype(data.train[col])]
    categorical_features = [col for col in feature_cols if col not in numeric_features]
    return (
        scores,
        None,
        {
            "model": "robust_unsupervised_anomaly_score",
            "target": target_col,
            "features": len(feature_cols),
            "numeric_features": len(numeric_features),
            "categorical_features": len(categorical_features),
        },
    )


def _unsupervised_anomaly_scores(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
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


def _constant_missing_target_predictions(
    *,
    data: CompetitionData,
    target_col: str,
    prediction_kind: str,
) -> tuple[np.ndarray | dict[str, np.ndarray], float | None, dict[str, object]]:
    if prediction_kind == "probability_columns":
        prediction_cols = [str(col) for col in data.sample.columns if col != data.id_column]
        values = np.full(len(data.test), 1.0 / max(len(prediction_cols), 1))
        return {col: values.copy() for col in prediction_cols}, None, {"model": "constant_missing_target"}
    values = np.zeros(len(data.test), dtype=float)
    return values, None, {"model": "constant_missing_target", "target": target_col}


def _constant_target_predictions(
    *,
    data: CompetitionData,
    target_col: str,
    prediction_kind: str,
    valid_index: np.ndarray,
    metric: str,
) -> tuple[np.ndarray | dict[str, np.ndarray], float | None, dict[str, object]]:
    y = data.train[target_col] if target_col in data.train.columns else pd.Series(dtype=float)
    if prediction_kind == "probability_columns":
        prediction_cols = [str(col) for col in data.sample.columns if col != data.id_column]
        values = _constant_probability_columns(y, prediction_cols=prediction_cols, row_count=len(data.test))
        valid_score = None
        return values, valid_score, {"model": "dummy_probability", "columns": prediction_cols}
    if pd.api.types.is_numeric_dtype(y):
        default = float(pd.to_numeric(y, errors="coerce").dropna().mean()) if not y.dropna().empty else 0.0
        pred = np.full(len(data.test), default)
        valid_pred = np.full(len(valid_index), default)
        pred = _clip_structured_regression_predictions(pred, y, column_name=target_col)
        valid_pred = _clip_structured_regression_predictions(valid_pred, y, column_name=target_col)
    else:
        default = _most_common_value(y, fallback="")
        pred = np.array([default] * len(data.test), dtype=object)
        valid_pred = np.array([default] * len(valid_index), dtype=object)
    valid_y = data.train.iloc[valid_index][target_col] if target_col in data.train.columns else pd.Series(dtype=object)
    valid_weight = _sample_weights_for_frame(data.train.iloc[valid_index], data.sample_weight_column)
    return (
        pred,
        _safe_metric(metric=metric, y_true=valid_y, y_pred=valid_pred, sample_weight=valid_weight),
        {
            "model": "constant",
            "default": str(default),
            "sample_weight_column": data.sample_weight_column,
        },
    )


def _constant_probability_columns(
    y: pd.Series,
    *,
    prediction_cols: list[str],
    row_count: int,
) -> dict[str, np.ndarray]:
    if not prediction_cols:
        return {}
    counts = y.astype(str).value_counts(normalize=True)
    values = np.zeros((row_count, len(prediction_cols)), dtype=float)
    for idx, col in enumerate(prediction_cols):
        normalized_col = _normalize_probability_column_name(col)
        match = next(
            (label for label in counts.index if _normalize_probability_column_name(label) == normalized_col),
            None,
        )
        values[:, idx] = float(counts.get(match, 0.0)) if match is not None else 0.0
    row_sums = values.sum(axis=1)
    if not np.all(row_sums > 0):
        values[:, :] = 1.0 / len(prediction_cols)
    else:
        values = values / row_sums[:, None]
    return {col: values[:, idx] for idx, col in enumerate(prediction_cols)}


def _survival_score_submission_column(data: CompetitionData) -> str | None:
    event_col, time_col = _survival_event_time_columns(data.target_columns)
    if not event_col or not time_col:
        return None
    prediction_cols = [str(col) for col in data.sample.columns if col != data.id_column]
    if len(prediction_cols) != 1:
        return None
    score_col = prediction_cols[0]
    if score_col in data.target_columns:
        return None
    if not pd.api.types.is_numeric_dtype(data.sample[score_col]):
        return None
    return score_col


def _survival_event_time_columns(target_columns: list[str]) -> tuple[str | None, str | None]:
    event_col = next((col for col in target_columns if _is_survival_event_column(col)), None)
    time_col = next((col for col in target_columns if _is_survival_time_column(col)), None)
    return event_col, time_col


def _is_survival_event_column(name: str) -> bool:
    compact = "".join(ch for ch in str(name).lower() if ch.isalnum())
    return compact in {"event", "eventobserved", "observed", "status", "efs", "censor", "censored", "death", "dead"}


def _is_survival_time_column(name: str) -> bool:
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


def _survival_risk_scores(*, event_predictions: np.ndarray, time_predictions: np.ndarray) -> np.ndarray:
    event_score = _normalize_score_component(event_predictions)
    time_score = 1.0 - _normalize_score_component(time_predictions)
    return np.clip(0.7 * event_score + 0.3 * time_score, 0.0, 1.0)


def _normalize_score_component(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    values = np.nan_to_num(values, nan=float(np.nanmedian(values)) if not np.isnan(values).all() else 0.0)
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.full(values.shape, 0.5, dtype=float)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _expand_numeric_multi_column_predictions(
    *,
    data: CompetitionData,
    target_col: str,
    base_predictions: np.ndarray,
    prediction_columns: list[str],
    prediction_kind: str,
) -> dict[str, np.ndarray]:
    if prediction_kind == "quantile_columns":
        return _quantile_column_predictions(
            y=data.train[target_col],
            base_predictions=base_predictions,
            prediction_columns=prediction_columns,
        )
    if prediction_kind == "prediction_interval_columns":
        return _prediction_interval_column_predictions(
            y=data.train[target_col],
            base_predictions=base_predictions,
            prediction_columns=prediction_columns,
        )
    return {col: base_predictions.copy() for col in prediction_columns}


def _quantile_column_predictions(
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
        quantile = _quantile_from_prediction_column(col)
        if quantile is None:
            quantile = 0.5
        offset = float(numeric.quantile(quantile)) - median
        raw[col] = base_predictions + offset
        ordered.append((quantile, col))
    if len(ordered) >= 2:
        stacked = np.column_stack([raw[col] for _, col in sorted(ordered)])
        stacked = np.maximum.accumulate(stacked, axis=1)
        for idx, (_, col) in enumerate(sorted(ordered)):
            raw[col] = stacked[:, idx]
    return raw


def _prediction_interval_column_predictions(
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
        role = _prediction_interval_column_role(col)
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


def _prediction_interval_column_role(name: object) -> str | None:
    compact = "".join(ch for ch in str(name).lower() if ch.isalnum())
    if compact in {"lower", "lo", "low", "lwr", "lowerbound", "lowerci", "lowerlimit"}:
        return "lower"
    if compact in {"upper", "hi", "high", "upr", "upperbound", "upperci", "upperlimit"}:
        return "upper"
    return None


def _quantile_from_prediction_column(name: object) -> float | None:
    import re

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


def _most_common_value(series: pd.Series, *, fallback):
    values = series.dropna()
    if values.empty:
        return fallback
    return values.mode(dropna=True).iloc[0]


def _train_vision_yolo_submission(
    *,
    data_dir: Path,
    sample_df: pd.DataFrame,
    output_path: Path,
    compute: Compute,
    strict_accelerator: bool,
    seed: int,
) -> TrainingOutcome:
    schema = infer_detection_submission_schema(sample_df)
    if schema is None:
        raise RuntimeError("Vision detector baseline requires a detection-style sample submission schema.")
    layout = find_yolo_data_layout(data_dir)
    if layout is None:
        raise RuntimeError(f"Unable to resolve YOLO image/label directories under {data_dir}.")

    train_labels: pd.DataFrame | None = None
    labels_path: Path | None = None
    if schema.right_place_column is not None:
        labels_path, train_labels = load_vision_train_labels(data_dir)
        train_files, val_files = train_val_split(train_labels_df=train_labels, seed=seed)
    else:
        train_files, val_files = train_val_split_files(
            filenames=discover_yolo_labeled_train_files(data_dir),
            seed=seed,
        )
    if not train_files or not val_files:
        raise RuntimeError("Unable to build non-empty train/validation split for vision detector.")

    dataset_yaml = prepare_ultralytics_dataset(
        root_out=output_path.parent,
        data_dir=data_dir,
        train_files=train_files,
        val_files=val_files,
    )
    detector = train_detector(
        dataset_yaml=dataset_yaml,
        seed=seed,
        device=_select_device(compute=compute, strict_accelerator=strict_accelerator),
        time_budget_min=None,
    )

    val_paths = [
        resolve_yolo_image_reference(name, images_dir=layout.train_images_dir, data_dir=data_dir) for name in val_files
    ]
    val_dets = _detection_results_with_reference_aliases(predict_detector(detector, val_paths), val_files, val_paths)

    test_filenames = sample_df[schema.image_column].astype(str).tolist()
    test_paths = [
        resolve_yolo_image_reference(name, images_dir=layout.test_images_dir, data_dir=data_dir)
        for name in test_filenames
    ]
    test_dets = _detection_results_with_reference_aliases(
        predict_detector(detector, test_paths),
        test_filenames,
        test_paths,
    )

    class_inference = infer_pairwise_object_classes(layout.train_labels_dir)
    primary_cls = class_inference.primary_cls
    secondary_cls = class_inference.secondary_cls
    class_details = class_inference.details
    val_truth = (
        train_labels[train_labels["filename"].isin(set(val_files))].copy()
        if train_labels is not None
        else pd.DataFrame({"filename": val_files})
    )

    map50_95 = compute_map50_95(
        data_dir=data_dir,
        filenames=val_truth["filename"].astype(str).tolist(),
        dets_by_file=val_dets,
    )
    if schema.right_place_column is not None:
        tuned = tune_right_place_params(
            val_truth_df=val_truth,
            val_dets_by_file=val_dets,
            head_cls=primary_cls,
            shemagh_cls=secondary_cls,
        )
        eval_payload = evaluate_combined_metric(
            val_truth_df=val_truth,
            val_dets_by_file=val_dets,
            detector_map50_95=map50_95,
            head_cls=primary_cls,
            shemagh_cls=secondary_cls,
            tuned_params=tuned,
        )
    else:
        tuned = {"prediction_score_thr": 0.1}
        eval_payload = {
            "map50_95": float(map50_95),
            "f1": None,
            "combined": float(map50_95),
            "thresholds": tuned,
        }

    prediction_score_thr = float(tuned.get("prediction_score_thr", 0.1))
    rows: list[dict[str, object]] = []
    for row_idx, filename in enumerate(test_filenames):
        dets = _detection_result_for_reference(test_dets, filename)
        output_row = {col: sample_df.iloc[row_idx][col] for col in schema.columns}
        output_row[schema.image_column] = filename
        output_row[schema.prediction_column] = format_prediction_string(dets, score_thr=prediction_score_thr)
        if schema.right_place_column is not None:
            output_row[schema.right_place_column] = int(
                derive_right_place(
                    dets=dets,
                    head_cls=primary_cls,
                    shemagh_cls=secondary_cls,
                    params=tuned,
                )
            )
        rows.append(output_row)

    submission = pd.DataFrame(rows, columns=list(schema.columns))
    submission[schema.prediction_column] = submission[schema.prediction_column].fillna("-").replace("", "-")
    submission_path = write_table(submission, output_path)

    f1_value = eval_payload["f1"]
    evaluation_metric = OFFICIAL_COMBINED_METRIC if schema.right_place_column is not None else "map50_95"
    model_summary = {
        "model": "vision_yolo",
        "backend": str(getattr(detector, "backend", "unknown")),
        "device": str(getattr(detector, "device", "cpu")),
        "submission_schema": {
            "image_column": schema.image_column,
            "prediction_column": schema.prediction_column,
            "right_place_column": schema.right_place_column,
        },
        "primary_cls": int(primary_cls),
        "secondary_cls": int(secondary_cls),
        "head_cls": int(primary_cls),
        "shemagh_cls": int(secondary_cls),
        "class_inference": class_details,
        "map50_95": float(eval_payload["map50_95"]),
        "f1": float(f1_value) if isinstance(f1_value, (int, float)) else None,
        "combined": float(eval_payload["combined"]),
        "thresholds": eval_payload["thresholds"],
        "labels_path": str(labels_path) if labels_path is not None else None,
    }
    print(
        "[local train] model=vision_yolo "
        f"backend={model_summary['backend']} "
        f"map50_95={model_summary['map50_95']:.6f} "
        f"f1={model_summary['f1'] if model_summary['f1'] is not None else 'n/a'} "
        f"combined={model_summary['combined']:.6f}",
        flush=True,
    )

    return TrainingOutcome(
        submission_path=submission_path,
        evaluation=EvaluationResult(
            score_source="holdout",
            metric=evaluation_metric,
            direction="maximize",
            value=float(eval_payload["combined"]),
            std=0.0,
            train_score=None,
            val_score=float(eval_payload["combined"]),
            fold_scores=[float(eval_payload["combined"])],
        ),
        model_name="vision_yolo",
        model_summary=model_summary,
        accelerator=_accelerator_label(str(getattr(detector, "device", "cpu"))),
    )


def _detection_result_for_reference(dets_by_file: dict[str, np.ndarray], reference: object) -> np.ndarray:
    text = str(reference)
    for key in (text, Path(text).name):
        if key in dets_by_file:
            return dets_by_file[key]
    return np.empty((0, 6), dtype=float)


def _detection_results_with_reference_aliases(
    dets_by_file: dict[str, np.ndarray],
    references: list[str],
    resolved_paths: list[Path] | None = None,
) -> dict[str, np.ndarray]:
    aliased = dict(dets_by_file)
    paths = resolved_paths or [None] * len(references)
    for reference, resolved_path in zip(references, paths, strict=False):
        text = str(reference)
        basename = Path(text).name
        value = None
        path_keys = (str(resolved_path), resolved_path.name) if resolved_path is not None else ()
        for key in (text, *path_keys, basename):
            if key in aliased:
                value = aliased[key]
                break
        if value is None:
            continue
        aliased.setdefault(text, value)
        aliased.setdefault(basename, value)
    return aliased


_TABULAR_LABEL_SUFFIXES = TABULAR_INPUT_SUFFIXES_ORDERED
_VISION_LABEL_STEM_KEYS = {"trainlabels", "trainlabel", "traininglabels", "labels"}
_VISION_LABEL_IMAGE_COLUMN_ALIASES = (
    "filename",
    "imageid",
    "imagepath",
    "imagefile",
    "imagename",
    "image",
    "imgid",
    "imgpath",
    "imgfile",
    "imgname",
    "img",
    "photoid",
    "photopath",
    "photofile",
    "photoname",
    "photo",
    "pictureid",
    "picturepath",
    "picturefile",
    "picturename",
    "picture",
    "file",
    "filepath",
    "path",
    "id",
)
_VISION_LABEL_TARGET_COLUMN_ALIASES = (
    "rightplace",
    "placement",
    "isrightplace",
    "validplacement",
    "target",
    "label",
    "class",
    "y",
)


def load_vision_train_labels(data_dir: Path) -> tuple[Path, pd.DataFrame]:
    labels_path = find_vision_train_labels_path(data_dir)
    if labels_path is None:
        expected = ", ".join(f"train_labels{suffix}" for suffix in _TABULAR_LABEL_SUFFIXES)
        raise FileNotFoundError(
            f"Missing training labels for vision task under {data_dir}; expected one of {expected}."
        )

    train_labels = normalize_vision_train_labels(read_table(labels_path), source_name=labels_path.name)
    train_labels["filename"] = train_labels["filename"].astype(str)
    train_labels["right_place"] = train_labels["right_place"].map(_coerce_binary_vision_label).astype(int)
    return labels_path, train_labels


def normalize_vision_train_labels(frame: pd.DataFrame, *, source_name: str = "train labels") -> pd.DataFrame:
    image_column, target_column = infer_vision_train_label_columns(frame)
    missing = []
    if image_column is None:
        missing.append("filename/image id")
    if target_column is None:
        missing.append("right_place/target")
    if missing:
        raise ValueError(f"{source_name} missing required columns: {missing}")
    normalized = frame[[image_column, target_column]].copy()
    normalized.columns = ["filename", "right_place"]
    return normalized


def infer_vision_train_label_columns(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    columns = [str(col) for col in frame.columns]
    normalized = {_normalize_vision_label_column(col): col for col in columns}
    image_column = next(
        (normalized[key] for key in _VISION_LABEL_IMAGE_COLUMN_ALIASES if key in normalized),
        None,
    )
    target_column = next(
        (
            normalized[key]
            for key in _VISION_LABEL_TARGET_COLUMN_ALIASES
            if key in normalized and normalized[key] != image_column
        ),
        None,
    )
    return image_column, target_column


def _normalize_vision_label_column(value: object) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _coerce_binary_vision_label(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return int(round(float(value)))
    normalized = str(value).strip().lower()
    if normalized in {"true", "t", "yes", "y", "positive", "pos"}:
        return 1
    if normalized in {"false", "f", "no", "n", "negative", "neg", ""}:
        return 0
    try:
        return int(round(float(normalized)))
    except ValueError:
        return 1 if normalized else 0


def train_val_split_files(
    *,
    filenames: list[str],
    seed: int,
    val_frac: float = 0.2,
) -> tuple[list[str], list[str]]:
    unique = sorted({str(item) for item in filenames if str(item)})
    if len(unique) < 2:
        return unique, []
    val_count = max(1, int(round(len(unique) * val_frac)))
    if val_count >= len(unique):
        val_count = len(unique) - 1
    train_part, val_part = train_test_split(
        unique,
        test_size=val_count,
        random_state=seed,
        shuffle=True,
    )
    return list(train_part), list(val_part)


def discover_yolo_labeled_train_files(data_dir: Path) -> list[str]:
    layout = find_yolo_data_layout(data_dir)
    if layout is None:
        return []

    images_by_relative_stem: dict[str, str] = {}
    images_by_basename_stem: dict[str, str] = {}
    for path in sorted(layout.train_images_dir.rglob("*")):
        if path.is_file():
            try:
                rel = path.relative_to(layout.train_images_dir)
            except ValueError:
                continue
            images_by_relative_stem.setdefault(rel.with_suffix("").as_posix(), rel.as_posix())
            images_by_basename_stem.setdefault(path.stem, rel.as_posix())

    filenames: list[str] = []
    for label_path in sorted(layout.train_labels_dir.rglob("*.txt")):
        rel_stem = label_path.relative_to(layout.train_labels_dir).with_suffix("").as_posix()
        filename = images_by_relative_stem.get(rel_stem)
        if filename is None:
            filename = images_by_basename_stem.get(Path(rel_stem).name)
        if filename is not None:
            filenames.append(filename)
    return sorted(dict.fromkeys(filenames))


def find_vision_train_labels_path(data_dir: Path) -> Path | None:
    for suffix in _TABULAR_LABEL_SUFFIXES:
        direct = data_dir / f"train_labels{suffix}"
        if direct.is_file():
            return direct

    try:
        candidates = [
            path
            for path in data_dir.rglob("*")
            if path.is_file()
            and is_tabular_data_path(path)
            and _normalized_label_stem(path) in _VISION_LABEL_STEM_KEYS
            and ".kagglebot_cache" not in {part.lower() for part in path.parts}
        ]
    except OSError:
        return None
    if not candidates:
        return None
    return min(candidates, key=lambda path: (len(path.relative_to(data_dir).parts), str(path).lower()))


def _normalized_label_stem(path: Path) -> str:
    suffix = tabular_suffix(path)
    name = path.name
    stem = name[: -len(suffix)] if suffix and name.lower().endswith(suffix) else path.stem
    return "".join(ch for ch in stem.lower() if ch.isalnum())


def _select_device(*, compute: Compute, strict_accelerator: bool) -> str:
    if compute != Compute.local_gpu:
        return "cpu"
    availability = detect_local_gpu()
    if availability.cuda:
        return "cuda"
    if availability.mps:
        return "mps"
    if strict_accelerator:
        raise GPUNotAvailableError(
            "No local GPU detected for --compute local_gpu. Disable --strict-accelerator to fall back to CPU."
        )
    return "cpu"


def _train_rna_structure_submission(
    *,
    data_dir: Path,
    output_path: Path,
    seed: int,
) -> TrainingOutcome:
    task = load_rna_structure_task(data_dir)
    label_id_column = task.label_id_column
    sample_id_column = task.sample_id_column

    target_ids = sorted({str(value).strip() for value in task.train_sequences[task.sequence_id_column].astype(str)})
    if len(target_ids) < 2:
        raise RuntimeError("RNA structure baseline requires at least two labeled targets for holdout evaluation.")

    rng = np.random.default_rng(seed)
    shuffled = list(target_ids)
    rng.shuffle(shuffled)
    split_index = max(1, int(round(len(shuffled) * 0.8)))
    if split_index >= len(shuffled):
        split_index = len(shuffled) - 1
    train_ids = set(shuffled[:split_index])
    valid_ids = set(shuffled[split_index:])
    if not train_ids or not valid_ids:
        raise RuntimeError("Unable to create a non-empty RNA structure train/validation split.")

    train_labels = task.train_labels[
        task.train_labels[label_id_column].astype(str).map(extract_target_id).isin(train_ids)
    ].copy()
    valid_labels = task.train_labels[
        task.train_labels[label_id_column].astype(str).map(extract_target_id).isin(valid_ids)
    ].copy()
    if train_labels.empty or valid_labels.empty:
        raise RuntimeError("RNA structure baseline split produced empty train or validation labels.")

    valid_sample = _rna_validation_sample_from_labels(task=task, valid_labels=valid_labels)
    valid_predictions = build_coordinate_baseline_predictions(
        train_labels=train_labels,
        sample_submission=valid_sample,
        label_id_column=label_id_column,
    )
    valid_submission_path = _sibling_tabular_artifact_path(output_path, stem="validation_submission")
    write_rna_structure_submission(
        sample_submission=valid_sample,
        predictions_by_target=valid_predictions,
        output_path=valid_submission_path,
    )
    valid_submission = read_table(valid_submission_path)
    if sample_id_column != label_id_column and sample_id_column in valid_submission.columns:
        valid_submission = valid_submission.copy()
        valid_submission[label_id_column] = valid_submission[sample_id_column].astype(str)
    val_rmse = evaluate_coordinate_predictions(
        truth=valid_labels,
        predictions=valid_submission,
        id_column=label_id_column,
    )

    submission_predictions = build_coordinate_baseline_predictions(
        train_labels=task.train_labels,
        sample_submission=task.sample_submission,
        label_id_column=label_id_column,
    )
    submission_path = write_rna_structure_submission(
        sample_submission=task.sample_submission,
        predictions_by_target=submission_predictions,
        output_path=output_path,
    )

    model_summary = {
        "model": "rna_coordinate_mean_baseline",
        "task": task.target_kind,
        "sequence_id_column": task.sequence_id_column,
        "sample_id_column": task.sample_id_column,
        "label_id_column": task.label_id_column,
        "coordinate_triplets": len(task.sample_coordinate_triplets),
        "train_targets": len(train_ids),
        "valid_targets": len(valid_ids),
        "val_rmse": float(val_rmse),
    }
    print(
        "[local train] model=rna_coordinate_mean_baseline "
        f"val_rmse={float(val_rmse):.6f} "
        f"targets={len(train_ids) + len(valid_ids)}",
        flush=True,
    )

    return TrainingOutcome(
        submission_path=submission_path,
        evaluation=EvaluationResult(
            score_source="holdout",
            metric="rmse",
            direction="minimize",
            value=float(val_rmse),
            std=0.0,
            train_score=None,
            val_score=float(val_rmse),
            fold_scores=[float(val_rmse)],
        ),
        model_name="rna_coordinate_mean_baseline",
        model_summary=model_summary,
        accelerator="cpu",
    )


def _rna_validation_sample_from_labels(*, task: RnaStructureTask, valid_labels: pd.DataFrame) -> pd.DataFrame:
    sample_columns = [str(column) for column in task.sample_submission.columns]
    coordinate_columns = set(task.sample_coordinate_columns)
    payload: dict[str, object] = {}
    for column in sample_columns:
        if column == task.sample_id_column:
            payload[column] = valid_labels[task.label_id_column].astype(str).to_numpy()
        elif column in coordinate_columns:
            payload[column] = np.zeros(len(valid_labels), dtype=float)
        elif column in valid_labels.columns:
            payload[column] = valid_labels[column].to_numpy()
        else:
            payload[column] = np.full(len(valid_labels), "", dtype=object)
    return pd.DataFrame(payload, columns=sample_columns)


def _accelerator_label(device: str) -> str:
    if device in {"0", "cuda"}:
        return "cuda"
    if device == "mps":
        return "mps"
    return "cpu"


def _sibling_tabular_artifact_path(path: Path, *, stem: str) -> Path:
    suffix = tabular_suffix(path) if is_tabular_data_path(path) else ""
    if not suffix:
        suffix = ".csv"
    return path.parent / f"{stem}{suffix}"
