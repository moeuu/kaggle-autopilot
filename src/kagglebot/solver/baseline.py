from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

from kagglebot.compute import Compute, detect_local_gpu
from kagglebot.exceptions import GPUNotAvailableError
from kagglebot.solver.evaluate import EvaluationResult, ScoreSelection, select_score_source
from kagglebot.solver.io import CompetitionData, load_competition_data, write_submission
from kagglebot.solver.metrics import compute_metric, infer_direction, metric_requires_proba


@dataclass(frozen=True)
class TrainingOutcome:
    submission_path: Path
    evaluation: EvaluationResult
    model_name: str
    model_summary: dict[str, object]
    accelerator: str


@dataclass(frozen=True)
class TargetTransform:
    name: str
    forward: Callable[[np.ndarray], np.ndarray]
    inverse: Callable[[np.ndarray], np.ndarray]


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _torch_device_for_accelerator(accelerator: str) -> str:
    if accelerator in {"cuda", "mps"}:
        return accelerator
    return "cpu"


def _resolve_torch_device(device: str):
    import torch

    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    data = load_competition_data(data_dir, target_column_override=target_override)
    data, feature_report = _augment_features(data)
    label_encoder = None
    if data.task == "classification":
        label_encoder = LabelEncoder()
        label_encoder.fit(data.train[data.target_column])
    target_transform = _resolve_target_transform(metric, data)

    selection = _select_accelerator(compute, strict_accelerator)
    selection_score = select_score_source(
        score_source=score_source,
        plan_score_source=plan_score_source,
        data_dir=data_dir,
        train=data.train,
        test=data.test,
        target_col=data.target_column,
        id_col=data.id_column,
    )

    metric_direction = infer_direction(metric, direction)
    candidates = _build_candidates(data, selection, seed=seed)
    evaluation, best_candidate, best_model, best_preprocessor = _evaluate_candidates(
        data=data,
        candidates=candidates,
        selection=selection_score,
        metric=metric,
        direction=metric_direction,
        seed=seed,
        holdout_frac=holdout_frac,
        cv_folds=cv_folds,
        label_encoder=label_encoder,
        target_transform=target_transform,
    )
    torch_device = None
    if best_candidate.get("framework") == "torch":
        torch_device = str(best_candidate.get("device", "cpu"))

    preds = _predict_with_model(
        data,
        best_model,
        best_preprocessor,
        metric,
        selection=selection_score,
        prediction_kind=data.prediction_kind,
        label_encoder=label_encoder,
        target_transform=target_transform,
        seed=seed,
        torch_device=torch_device,
    )

    submission_path = write_submission(
        data.sample,
        data.test,
        preds,
        id_column=data.id_column,
        target_column=data.target_column,
        output_path=output_path,
    )

    return TrainingOutcome(
        submission_path=submission_path,
        evaluation=evaluation,
        model_name=best_candidate["name"],
        model_summary={
            "model": best_candidate["name"],
            "params": best_candidate.get("params", {}),
            "preprocessing": best_candidate.get("preprocessing", {}),
            "target_transform": target_transform.name if target_transform else None,
            "feature_engineering": feature_report,
            "device": torch_device,
        },
        accelerator=selection,
    )


def _select_accelerator(compute: Compute, strict: bool) -> str:
    if compute != Compute.local_gpu:
        return "cpu"
    availability = detect_local_gpu()
    if availability.cuda:
        return "cuda"
    if availability.mps:
        return "mps"
    if strict:
        raise GPUNotAvailableError(
            "No local GPU detected for --compute local_gpu. Disable --strict-accelerator to fall back to CPU."
        )
    return "cpu"


def _resolve_target_transform(metric: str, data: CompetitionData) -> TargetTransform | None:
    if data.task != "regression":
        return None
    metric_lower = metric.lower()
    if "rmsle" in metric_lower:
        return TargetTransform(
            name="log1p",
            forward=lambda y: np.log1p(np.clip(np.asarray(y, dtype=float), 0, None)),
            inverse=lambda y: np.clip(np.expm1(np.asarray(y, dtype=float)), 0, None),
        )
    return None


def _build_candidates(data: CompetitionData, accelerator: str, *, seed: int) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    if data.task == "classification":
        if _torch_available():
            torch_device = _torch_device_for_accelerator(accelerator)
            candidates.append(
                {
                    "name": f"torch_mlp_{torch_device}",
                    "framework": "torch",
                    "device": torch_device,
                    "model": None,
                    "params": {"epochs": 5, "hidden_dim": 128, "dropout": 0.2},
                    "preprocessing": _build_linear_preprocessor(data),
                }
            )
        candidates.append(
            {
                "name": "logreg",
                "model": LogisticRegression(max_iter=2000, class_weight="balanced"),
                "preprocessing": _build_linear_preprocessor(data),
            }
        )
        candidates.append(
            {
                "name": "hist_gb",
                "model": HistGradientBoostingClassifier(max_depth=7, learning_rate=0.05, max_iter=300),
                "preprocessing": _build_tree_preprocessor(data),
            }
        )
        if accelerator == "cuda":
            candidates.append(
                {
                    "name": "catboost_gpu",
                    "model": CatBoostClassifier(
                        iterations=600,
                        depth=8,
                        learning_rate=0.05,
                        l2_leaf_reg=3.0,
                        loss_function="Logloss",
                        eval_metric="Accuracy",
                        auto_class_weights="Balanced",
                        task_type="GPU",
                        random_seed=seed,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                    "preprocessing": None,
                }
            )
        else:
            candidates.append(
                {
                    "name": "catboost_cpu",
                    "model": CatBoostClassifier(
                        iterations=600,
                        depth=8,
                        learning_rate=0.05,
                        l2_leaf_reg=3.0,
                        loss_function="Logloss",
                        eval_metric="Accuracy",
                        auto_class_weights="Balanced",
                        task_type="CPU",
                        random_seed=seed,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                    "preprocessing": None,
                }
            )
    else:
        if _torch_available():
            torch_device = _torch_device_for_accelerator(accelerator)
            candidates.append(
                {
                    "name": f"torch_mlp_{torch_device}",
                    "framework": "torch",
                    "device": torch_device,
                    "model": None,
                    "params": {"epochs": 5, "hidden_dim": 128, "dropout": 0.2},
                    "preprocessing": _build_linear_preprocessor(data),
                }
            )
        candidates.append({"name": "ridge", "model": Ridge(), "preprocessing": _build_linear_preprocessor(data)})
        candidates.append(
            {
                "name": "hist_gb",
                "model": HistGradientBoostingRegressor(),
                "preprocessing": _build_tree_preprocessor(data),
            }
        )
        if accelerator == "cuda":
            candidates.append(
                {
                    "name": "catboost_gpu",
                    "model": CatBoostRegressor(
                        iterations=600,
                        depth=8,
                        learning_rate=0.05,
                        l2_leaf_reg=3.0,
                        loss_function="RMSE",
                        eval_metric="RMSE",
                        task_type="GPU",
                        random_seed=seed,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                    "preprocessing": None,
                }
            )
        else:
            candidates.append(
                {
                    "name": "catboost_cpu",
                    "model": CatBoostRegressor(
                        iterations=600,
                        depth=8,
                        learning_rate=0.05,
                        l2_leaf_reg=3.0,
                        loss_function="RMSE",
                        eval_metric="RMSE",
                        task_type="CPU",
                        random_seed=seed,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                    "preprocessing": None,
                }
            )
    return candidates


def _augment_features(data: CompetitionData) -> tuple[CompetitionData, dict[str, object]]:
    train = data.train.copy()
    test = data.test.copy()
    feature_cols = list(data.feature_columns)
    added = []

    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
    agg_features = []
    if num_cols:
        train_num = train[num_cols]
        test_num = test[num_cols]
        agg_map = {
            "agg__num_sum": (train_num.sum(axis=1, skipna=True), test_num.sum(axis=1, skipna=True)),
            "agg__num_mean": (train_num.mean(axis=1, skipna=True), test_num.mean(axis=1, skipna=True)),
            "agg__num_std": (
                train_num.std(axis=1, skipna=True).fillna(0.0),
                test_num.std(axis=1, skipna=True).fillna(0.0),
            ),
            "agg__num_min": (train_num.min(axis=1, skipna=True), test_num.min(axis=1, skipna=True)),
            "agg__num_max": (train_num.max(axis=1, skipna=True), test_num.max(axis=1, skipna=True)),
            "agg__num_missing": (train_num.isna().sum(axis=1), test_num.isna().sum(axis=1)),
            "agg__num_zero": ((train_num == 0).sum(axis=1), (test_num == 0).sum(axis=1)),
        }
        for name, (train_series, test_series) in agg_map.items():
            if name in train.columns:
                continue
            train[name] = train_series
            test[name] = test_series
            feature_cols.append(name)
            added.append(name)
            agg_features.append(name)

    pattern = re.compile(r"([-+]?\d*\.?\d+)")
    extracted_cols = []
    cat_cols = [c for c in feature_cols if train[c].dtype == "object"]
    for col in cat_cols:
        train_series = train[col].astype("string")
        test_series = test[col].astype("string")
        train_extracted = train_series.str.extract(pattern, expand=False)
        non_null = train_series.notna().sum()
        if non_null == 0:
            continue
        ratio = float(train_extracted.notna().sum()) / float(non_null)
        if ratio < 0.2:
            continue
        safe_name = _safe_feature_name(col)
        new_col = f"num__{safe_name}"
        if new_col in train.columns:
            continue
        train[new_col] = pd.to_numeric(train_extracted, errors="coerce")
        test[new_col] = pd.to_numeric(test_series.str.extract(pattern, expand=False), errors="coerce")
        feature_cols.append(new_col)
        added.append(new_col)
        extracted_cols.append(new_col)

    report = {
        "numeric_aggregates": agg_features,
        "numeric_extraction": extracted_cols,
        "total_added": len(added),
    }
    return (
        CompetitionData(
            train=train,
            test=test,
            sample=data.sample,
            id_column=data.id_column,
            target_column=data.target_column,
            feature_columns=feature_cols,
            task=data.task,
            prediction_kind=data.prediction_kind,
        ),
        report,
    )


def _safe_feature_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    return cleaned.strip("_") or "feature"


def _build_linear_preprocessor(data: CompetitionData) -> ColumnTransformer:
    cat_cols = [c for c in data.feature_columns if data.train[c].dtype == "object"]
    num_cols = [c for c in data.feature_columns if c not in cat_cols]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


def _build_tree_preprocessor(data: CompetitionData) -> ColumnTransformer:
    cat_cols = [c for c in data.feature_columns if data.train[c].dtype == "object"]
    num_cols = [c for c in data.feature_columns if c not in cat_cols]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


def _evaluate_candidates(
    *,
    data: CompetitionData,
    candidates: list[dict[str, object]],
    selection: ScoreSelection,
    metric: str,
    direction: str,
    seed: int,
    holdout_frac: float,
    cv_folds: int,
    label_encoder: LabelEncoder | None,
    target_transform: TargetTransform | None,
) -> tuple[EvaluationResult, dict[str, object], object, ColumnTransformer | None]:
    results: list[dict[str, object]] = []
    failures: list[tuple[str, Exception]] = []

    for candidate in candidates:
        name = str(candidate["name"])
        preprocessor = candidate.get("preprocessing")
        model = candidate.get("model")
        try:
            if candidate.get("framework") == "torch":
                scores, train_score, val_score = _evaluate_torch(
                    data,
                    selection=selection,
                    metric=metric,
                    seed=seed,
                    holdout_frac=holdout_frac,
                    cv_folds=cv_folds,
                    direction=direction,
                    label_encoder=label_encoder,
                    target_transform=target_transform,
                    preprocessor=preprocessor,
                    device=str(candidate.get("device", "cpu")),
                )
                model = None
            else:
                scores, train_score, val_score = _evaluate_sklearn_or_catboost(
                    data,
                    model=model,
                    preprocessor=preprocessor,  # type: ignore[arg-type]
                    selection=selection,
                    metric=metric,
                    seed=seed,
                    holdout_frac=holdout_frac,
                    cv_folds=cv_folds,
                    label_encoder=label_encoder,
                    target_transform=target_transform,
                )
            mean_score = float(np.mean(scores)) if scores else float("nan")
            std_score = float(np.std(scores)) if scores else None
            results.append(
                {
                    "candidate": candidate,
                    "model": model,
                    "preprocessor": preprocessor,
                    "scores": scores,
                    "mean": mean_score,
                    "std": std_score,
                    "train_score": train_score,
                    "val_score": val_score,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            continue

    if not results:
        detail = ", ".join(name for name, _ in failures) if failures else "unknown"
        raise RuntimeError(f"No candidate models were evaluated. Failed: {detail}")

    best_record = _select_best_candidate(results, direction)
    best_candidate = best_record["candidate"]
    best_model = best_record["model"]
    best_preprocessor = best_record["preprocessor"]
    best_score = float(best_record["mean"])
    best_std = best_record["std"]
    best_train_score = best_record["train_score"]
    best_val_score = best_record["val_score"]
    best_fold_scores = best_record["scores"]

    evaluation = EvaluationResult(
        score_source=selection.source,
        metric=metric,
        direction=direction,  # type: ignore[arg-type]
        value=best_score,
        std=best_std if isinstance(best_std, float) else None,
        train_score=best_train_score if isinstance(best_train_score, float) else None,
        val_score=best_val_score if isinstance(best_val_score, float) else None,
        fold_scores=best_fold_scores if isinstance(best_fold_scores, list) else None,
    )
    return evaluation, best_candidate, best_model, best_preprocessor


def _candidate_family(candidate: dict[str, object]) -> str:
    if candidate.get("framework") == "torch":
        return "torch"
    if candidate.get("name") in {"logreg", "ridge"}:
        return "linear"
    return "other"


def _best_by_metric(records: list[dict[str, object]], direction: str) -> dict[str, object]:
    best: dict[str, object] | None = None
    for record in records:
        score = float(record["mean"])
        if best is None or _is_better(score, float(best["mean"]), direction):
            best = record
    if best is None:
        raise RuntimeError("No candidates to select.")
    return best


def _linear_clearly_better(linear: dict[str, object], torch: dict[str, object], direction: str) -> bool:
    linear_score = float(linear["mean"])
    torch_score = float(torch["mean"])
    if direction == "minimize":
        delta = torch_score - linear_score
    else:
        delta = linear_score - torch_score
    linear_std = float(linear["std"]) if isinstance(linear.get("std"), (int, float)) else 0.0
    torch_std = float(torch["std"]) if isinstance(torch.get("std"), (int, float)) else 0.0
    margin = max(linear_std, torch_std)
    return delta > margin


def _select_best_candidate(records: list[dict[str, object]], direction: str) -> dict[str, object]:
    primary = [r for r in records if _candidate_family(r["candidate"]) in {"torch", "linear"}]
    if primary:
        best_primary = _best_by_metric(primary, direction)
        if _candidate_family(best_primary["candidate"]) == "linear":
            torch_records = [r for r in primary if _candidate_family(r["candidate"]) == "torch"]
            if torch_records:
                best_torch = _best_by_metric(torch_records, direction)
                if not _linear_clearly_better(best_primary, best_torch, direction):
                    return best_torch
        return best_primary
    return _best_by_metric(records, direction)


def _evaluate_sklearn_or_catboost(
    data: CompetitionData,
    *,
    model,
    preprocessor: ColumnTransformer | None,
    selection: ScoreSelection,
    metric: str,
    seed: int,
    holdout_frac: float,
    cv_folds: int,
    label_encoder: LabelEncoder | None,
    target_transform: TargetTransform | None,
) -> tuple[list[float], float | None, float | None]:
    if selection.source == "test":
        x_train, y_train, x_eval, y_eval = _prepare_test_split(data, selection, preprocessor)
        fitted, train_score = _fit_and_score(
            model,
            x_train,
            y_train,
            x_train,
            y_train,
            metric,
            data,
            preprocessor,
            label_encoder,
            target_transform,
        )
        _, eval_score = _fit_and_score(
            model,
            x_train,
            y_train,
            x_eval,
            y_eval,
            metric,
            data,
            preprocessor,
            label_encoder,
            target_transform,
        )
        return [eval_score], train_score, eval_score

    if selection.source == "holdout":
        x_train, x_val, y_train, y_val = _holdout_split(data, seed, holdout_frac)
        fitted, train_score = _fit_and_score(
            model,
            x_train,
            y_train,
            x_train,
            y_train,
            metric,
            data,
            preprocessor,
            label_encoder,
            target_transform,
        )
        _, val_score = _fit_and_score(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            metric,
            data,
            preprocessor,
            label_encoder,
            target_transform,
        )
        return [val_score], train_score, val_score

    scores = []
    splitter = _splitter(data, seed, cv_folds)
    for train_idx, val_idx in splitter.split(data.train[data.feature_columns], data.train[data.target_column]):
        x_tr = data.train.iloc[train_idx][data.feature_columns]
        x_val = data.train.iloc[val_idx][data.feature_columns]
        y_tr = data.train.iloc[train_idx][data.target_column]
        y_val = data.train.iloc[val_idx][data.target_column]
        _, fold_score = _fit_and_score(
            model,
            x_tr,
            y_tr,
            x_val,
            y_val,
            metric,
            data,
            preprocessor,
            label_encoder,
            target_transform,
        )
        scores.append(fold_score)
    return scores, None, float(np.mean(scores)) if scores else None


def _evaluate_torch(
    data: CompetitionData,
    *,
    selection: ScoreSelection,
    metric: str,
    seed: int,
    holdout_frac: float,
    cv_folds: int,
    direction: str,
    label_encoder: LabelEncoder | None,
    target_transform: TargetTransform | None,
    preprocessor: ColumnTransformer | None,
    device: str,
) -> tuple[list[float], float | None, float | None]:
    if selection.source == "test":
        x_train, y_train, x_eval, y_eval = _prepare_test_split(data, selection, preprocessor)
        x_train_proc, x_eval_proc = _apply_preprocessor(preprocessor, x_train, x_eval)
        _, train_score = _fit_torch_and_score(
            x_train_proc,
            y_train,
            x_train_proc,
            y_train,
            metric,
            data,
            label_encoder,
            target_transform,
            seed=seed,
            device=device,
        )
        _, eval_score = _fit_torch_and_score(
            x_train_proc,
            y_train,
            x_eval_proc,
            y_eval,
            metric,
            data,
            label_encoder,
            target_transform,
            seed=seed,
            device=device,
        )
        return [eval_score], train_score, eval_score

    if selection.source == "holdout":
        x_train, x_val, y_train, y_val = _holdout_split(data, seed, holdout_frac)
        x_train_proc, x_val_proc = _apply_preprocessor(preprocessor, x_train, x_val)
        _, train_score = _fit_torch_and_score(
            x_train_proc,
            y_train,
            x_train_proc,
            y_train,
            metric,
            data,
            label_encoder,
            target_transform,
            seed=seed,
            device=device,
        )
        _, val_score = _fit_torch_and_score(
            x_train_proc,
            y_train,
            x_val_proc,
            y_val,
            metric,
            data,
            label_encoder,
            target_transform,
            seed=seed,
            device=device,
        )
        return [val_score], train_score, val_score

    scores = []
    splitter = _splitter(data, seed, cv_folds)
    for train_idx, val_idx in splitter.split(data.train[data.feature_columns], data.train[data.target_column]):
        x_tr = data.train.iloc[train_idx][data.feature_columns]
        x_val = data.train.iloc[val_idx][data.feature_columns]
        y_tr = data.train.iloc[train_idx][data.target_column]
        y_val = data.train.iloc[val_idx][data.target_column]
        x_tr_proc, x_val_proc = _apply_preprocessor(preprocessor, x_tr, x_val)
        _, fold_score = _fit_torch_and_score(
            x_tr_proc,
            y_tr,
            x_val_proc,
            y_val,
            metric,
            data,
            label_encoder,
            target_transform,
            seed=seed,
            device=device,
        )
        scores.append(fold_score)
    return scores, None, float(np.mean(scores)) if scores else None


def _fit_and_score(
    model,
    x_train,
    y_train,
    x_eval,
    y_eval,
    metric,
    data,
    preprocessor,
    label_encoder,
    target_transform: TargetTransform | None,
):
    x_train_proc, x_eval_proc = _apply_preprocessor(preprocessor, x_train, x_eval)
    x_train_proc = _prepare_catboost_frame(model, x_train_proc)
    x_eval_proc = _prepare_catboost_frame(model, x_eval_proc)
    y_train_enc = _encode_target(data, y_train, label_encoder, target_transform=target_transform, apply_transform=True)
    fit_kwargs = _catboost_fit_kwargs(model, x_train_proc)
    model.fit(x_train_proc, y_train_enc, **fit_kwargs)
    preds = _predict_for_metric(model, x_eval_proc, data, metric, target_transform)
    y_eval_enc = _encode_target(data, y_eval, label_encoder, target_transform=None, apply_transform=False)
    score = compute_metric(metric, y_eval_enc, preds)
    return model, score


def _catboost_fit_kwargs(model, x):
    if not model.__class__.__name__.startswith("CatBoost"):
        return {}
    if not hasattr(x, "columns"):
        return {}
    cat_features = [
        idx
        for idx, col in enumerate(x.columns)
        if pd.api.types.is_object_dtype(x[col])
        or pd.api.types.is_categorical_dtype(x[col])
        or pd.api.types.is_string_dtype(x[col])
    ]
    if not cat_features:
        return {}
    return {"cat_features": cat_features}


def _prepare_catboost_frame(model, frame):
    if model is None:
        return frame
    if not model.__class__.__name__.startswith("CatBoost"):
        return frame
    if not hasattr(frame, "columns"):
        return frame
    frame = frame.copy()
    for col in frame.columns:
        if pd.api.types.is_object_dtype(frame[col]) or pd.api.types.is_categorical_dtype(frame[col]):
            frame[col] = frame[col].astype("string").fillna("missing")
    return frame


def _fit_torch_and_score(
    x_train,
    y_train,
    x_eval,
    y_eval,
    metric,
    data,
    label_encoder,
    target_transform: TargetTransform | None,
    *,
    seed: int,
    device: str,
):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    resolved_device = _resolve_torch_device(device)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    x_train_proc = np.asarray(_dense_array(x_train), dtype=np.float32)
    x_eval_proc = np.asarray(_dense_array(x_eval), dtype=np.float32)
    y_train_enc = _encode_target(data, y_train, label_encoder, target_transform=target_transform, apply_transform=True)
    y_eval_enc = _encode_target(data, y_eval, label_encoder, target_transform=None, apply_transform=False)

    x_tensor = torch.tensor(x_train_proc, dtype=torch.float32).to(resolved_device)
    y_tensor = torch.tensor(y_train_enc).to(resolved_device)
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    input_dim = x_tensor.shape[1]
    num_classes = int(np.unique(y_train_enc).size) if data.task == "classification" else 0
    output_dim = 1 if data.task == "regression" or num_classes <= 2 else num_classes
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, output_dim),
    ).to(resolved_device)

    if data.task == "classification":
        loss_fn = nn.CrossEntropyLoss() if num_classes > 2 else nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(5):
        for xb, yb in loader:
            optimizer.zero_grad()
            outputs = model(xb)
            if data.task == "classification" and output_dim == 1:
                yb = yb.float().view(-1, 1)
            loss = loss_fn(outputs, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        eval_tensor = torch.tensor(x_eval_proc, dtype=torch.float32).to(resolved_device)
        outputs = model(eval_tensor).cpu().numpy()

    preds = _torch_outputs_to_preds(outputs, data, metric, prediction_kind=None)
    if data.task == "regression" and target_transform is not None:
        preds = target_transform.inverse(preds)
    score = compute_metric(metric, y_eval_enc, preds)
    return preds, score


def _apply_preprocessor(preprocessor: ColumnTransformer | None, x_train, x_eval):
    if preprocessor is None:
        return x_train, x_eval
    x_train_proc = preprocessor.fit_transform(x_train)
    x_eval_proc = preprocessor.transform(x_eval)
    return _dense_array(x_train_proc), _dense_array(x_eval_proc)


def _dense_array(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return matrix


def _encode_target(
    data: CompetitionData,
    y,
    label_encoder: LabelEncoder | None,
    *,
    target_transform: TargetTransform | None,
    apply_transform: bool,
):
    if data.task != "classification":
        arr = np.asarray(y, dtype=float)
        if apply_transform and target_transform is not None:
            return target_transform.forward(arr)
        return arr
    if label_encoder is None:
        encoder = LabelEncoder()
        return encoder.fit_transform(y)
    return label_encoder.transform(y)


def _predict_for_metric(
    model,
    x,
    data: CompetitionData,
    metric: str,
    target_transform: TargetTransform | None,
):
    if data.task == "classification" and metric_requires_proba(metric):
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x)
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            if proba.ndim == 2:
                return proba
        if hasattr(model, "decision_function"):
            scores = model.decision_function(x)
            if scores.ndim == 1:
                return 1 / (1 + np.exp(-scores))
            return scores
    preds = model.predict(x)
    if data.task == "regression" and target_transform is not None:
        return target_transform.inverse(preds)
    return preds


def _predict_for_submission(
    model,
    x,
    data: CompetitionData,
    metric: str,
    prediction_kind: str,
    target_transform: TargetTransform | None,
):
    if data.task == "classification" and (metric_requires_proba(metric) or prediction_kind == "probability"):
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x)
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            return proba
        if hasattr(model, "decision_function"):
            scores = model.decision_function(x)
            if scores.ndim == 1:
                return 1 / (1 + np.exp(-scores))
            return scores
    preds = model.predict(x)
    if data.task == "regression" and target_transform is not None:
        return target_transform.inverse(preds)
    return preds


def _torch_outputs_to_preds(outputs, data: CompetitionData, metric: str, prediction_kind: str | None):
    if data.task == "classification":
        if outputs.ndim == 2 and outputs.shape[1] > 1:
            if metric_requires_proba(metric) or prediction_kind == "probability":
                exp = np.exp(outputs)
                return exp / exp.sum(axis=1, keepdims=True)
            return outputs.argmax(axis=1)
        logits = outputs.ravel()
        probs = 1 / (1 + np.exp(-logits))
        if metric_requires_proba(metric) or prediction_kind == "probability":
            return probs
        return (probs >= 0.5).astype(int)
    return outputs.ravel()


def _holdout_split(data: CompetitionData, seed: int, holdout_frac: float):
    stratify = data.train[data.target_column] if data.task == "classification" else None
    return train_test_split(
        data.train[data.feature_columns],
        data.train[data.target_column],
        test_size=holdout_frac,
        random_state=seed,
        stratify=stratify,
    )


def _splitter(data: CompetitionData, seed: int, folds: int):
    if data.task == "classification":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return KFold(n_splits=folds, shuffle=True, random_state=seed)


def _prepare_test_split(data: CompetitionData, selection: ScoreSelection, preprocessor: ColumnTransformer | None):
    labeled = selection.labeled_test
    if labeled is None:
        raise ValueError("Labeled test data required for score_source=test.")
    x_train = data.train[data.feature_columns]
    y_train = data.train[data.target_column]
    x_eval = labeled.frame[data.feature_columns]
    y_eval = labeled.target
    return x_train, y_train, x_eval, y_eval


def _predict_with_model(
    data: CompetitionData,
    model,
    preprocessor: ColumnTransformer | None,
    metric: str,
    *,
    selection: ScoreSelection,
    prediction_kind: str,
    label_encoder: LabelEncoder | None,
    target_transform: TargetTransform | None,
    seed: int,
    torch_device: str | None,
):
    x_train = data.train[data.feature_columns]
    x_test = data.test[data.feature_columns]
    if preprocessor is not None:
        x_train = preprocessor.fit_transform(x_train)
        x_test = preprocessor.transform(x_test)
        x_train = _dense_array(x_train)
        x_test = _dense_array(x_test)
    x_train = _prepare_catboost_frame(model, x_train)
    x_test = _prepare_catboost_frame(model, x_test)
    if model is None:
        return _predict_torch_full(
            data,
            metric,
            x_train,
            x_test,
            label_encoder,
            target_transform,
            seed=seed,
            device=torch_device or "cpu",
        )
    y_train_enc = _encode_target(
        data,
        data.train[data.target_column],
        label_encoder,
        target_transform=target_transform,
        apply_transform=True,
    )
    fit_kwargs = _catboost_fit_kwargs(model, x_train)
    model.fit(x_train, y_train_enc, **fit_kwargs)
    preds = _predict_for_submission(model, x_test, data, metric, prediction_kind, target_transform)
    if data.task == "classification" and prediction_kind == "class":
        if preds.ndim > 1:
            preds = preds.argmax(axis=1)
        if data.train[data.target_column].dtype == "object" and label_encoder is not None:
            preds = label_encoder.inverse_transform(np.asarray(preds, dtype=int))
    return preds


def _predict_torch_full(
    data: CompetitionData,
    metric: str,
    x_train,
    x_test,
    label_encoder: LabelEncoder | None,
    target_transform: TargetTransform | None,
    *,
    seed: int,
    device: str,
):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    resolved_device = _resolve_torch_device(device)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    x_train = np.asarray(_dense_array(x_train), dtype=np.float32)
    x_test = np.asarray(_dense_array(x_test), dtype=np.float32)
    y_train_enc = _encode_target(
        data,
        data.train[data.target_column],
        label_encoder,
        target_transform=target_transform,
        apply_transform=True,
    )

    x_tensor = torch.tensor(x_train, dtype=torch.float32).to(resolved_device)
    y_tensor = torch.tensor(y_train_enc).to(resolved_device)
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    input_dim = x_tensor.shape[1]
    num_classes = int(np.unique(y_train_enc).size) if data.task == "classification" else 0
    output_dim = 1 if data.task == "regression" or num_classes <= 2 else num_classes
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, output_dim),
    ).to(resolved_device)

    if data.task == "classification":
        loss_fn = nn.CrossEntropyLoss() if num_classes > 2 else nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(5):
        for xb, yb in loader:
            optimizer.zero_grad()
            outputs = model(xb)
            if data.task == "classification" and output_dim == 1:
                yb = yb.float().view(-1, 1)
            loss = loss_fn(outputs, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(resolved_device)
        outputs = model(x_test_tensor).cpu().numpy()

    preds = _torch_outputs_to_preds(outputs, data, metric, prediction_kind=data.prediction_kind)
    if data.task == "regression" and target_transform is not None:
        preds = target_transform.inverse(preds)
    if data.task == "classification" and data.prediction_kind == "class" and label_encoder is not None:
        if preds.ndim > 1:
            preds = preds.argmax(axis=1)
        preds = label_encoder.inverse_transform(np.asarray(preds, dtype=int))
    return preds


def _is_better(candidate: float, best: float, direction: str) -> bool:
    if direction == "minimize":
        return candidate < best
    return candidate > best
