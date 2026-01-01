from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from rich import print
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from kagglebot.analyzer.types import CompetitionMetadata
from kagglebot.paths import CompetitionPaths


@dataclass(frozen=True)
class TrainingResult:
    best_model_name: str
    model_path: Path
    model_info_path: Path
    report_path: Path
    submission_path: Path
    sample_submission_path: Path


def train_tabular(
    metadata: CompetitionMetadata,
    *,
    paths: CompetitionPaths,
    time_budget_minutes: int,
    model_names: list[str] | None,
    cv_folds: int | None,
    accelerator: str = "none",
    strict_accelerator: bool = False,
    random_seed: int = 42,
) -> TrainingResult:
    schema = metadata.schema
    if len(schema.target_columns) != 1:
        raise ValueError("Multi-target competitions are not supported yet.")

    if metadata.strategy.use_stacking:
        print("[yellow]stacking not implemented[/yellow]: training best single model only")

    train_df = pd.read_csv(schema.train_path)

    target_col = schema.target_columns[0]
    feature_cols = schema.feature_columns

    x = train_df[feature_cols]
    y = train_df[target_col]

    base_models = model_names if model_names is not None else metadata.strategy.models
    raw_models = [name.strip().lower() for name in base_models if name.strip()]
    if model_names is None and "hist_gb" not in raw_models:
        raw_models.append("hist_gb")
    seen: set[str] = set()
    model_list = [name for name in raw_models if not (name in seen or seen.add(name))]
    if not model_list:
        raise ValueError("No models configured for training.")

    folds = _sanitize_folds(metadata.task, y, cv_folds or metadata.strategy.cv_folds)
    time_budget_seconds = max(time_budget_minutes, 1) * 60
    start = time.monotonic()

    results: dict[str, dict[str, Any]] = {}
    best_model_name = ""
    best_score: float | None = None
    best_direction = metadata.metric_direction

    splitter = _make_splitter(metadata.task, folds, random_seed)
    class_count = y.nunique(dropna=True) if metadata.task == "classification" else None

    for model_name in model_list:
        if time.monotonic() - start > time_budget_seconds:
            print("[yellow]time budget reached[/yellow]: skipping remaining models")
            break

        if model_name not in {"logreg", "ridge", "catboost", "catboost_gpu", "hist_gb"}:
            print(f"[yellow]skipping unknown model[/yellow]: {model_name}")
            continue
        if metadata.task == "classification" and model_name == "ridge":
            print(f"[yellow]skipping regression model for classification[/yellow]: {model_name}")
            continue
        if metadata.task == "regression" and model_name == "logreg":
            print(f"[yellow]skipping classification model for regression[/yellow]: {model_name}")
            continue

        fold_scores: list[float] = []
        fold_start = time.monotonic()
        model_failed = False
        for train_idx, val_idx in splitter.split(x, y):
            x_tr, x_va = x.iloc[train_idx], x.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

            model = _build_model(model_name, metadata, feature_cols, class_count=class_count)
            if model_name in {"catboost", "catboost_gpu"}:
                try:
                    _fit_catboost(model, x_tr, y_tr, x_va, y_va, schema, random_seed)
                except Exception as exc:  # noqa: BLE001
                    if strict_accelerator and accelerator == "gpu":
                        raise RuntimeError("CatBoost GPU training failed.") from exc
                    print(f"[yellow]catboost training failed[/yellow]: {exc}")
                    model_failed = True
                    break
                preds = model.predict(x_va)
            else:
                model.fit(x_tr, y_tr)
                preds = model.predict(x_va)

            score = _score_predictions(metadata.task, y_va, preds)
            fold_scores.append(score)

        if model_failed:
            continue

        duration = time.monotonic() - fold_start
        results[model_name] = {
            "fold_scores": fold_scores,
            "mean": float(np.mean(fold_scores)),
            "std": float(np.std(fold_scores)),
            "training_time_seconds": duration,
        }

        candidate = results[model_name]["mean"]
        if _is_better_score(best_score, candidate, best_direction):
            best_score = candidate
            best_model_name = model_name

    if not best_model_name:
        raise RuntimeError("No models were trained within the time budget.")

    best_model = _build_model(best_model_name, metadata, feature_cols, class_count=class_count)
    if best_model_name in {"catboost", "catboost_gpu"}:
        _fit_catboost(best_model, x, y, None, None, schema, random_seed)
    else:
        best_model.fit(x, y)

    paths.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = _save_model(best_model_name, best_model, paths.models_dir)
    model_info_path = paths.model_info_path
    _write_model_info(
        model_info_path,
        metadata=metadata,
        model_name=best_model_name,
        model_type="catboost" if best_model_name in {"catboost", "catboost_gpu"} else "sklearn",
        model_path=model_path,
        feature_columns=feature_cols,
    )

    report_path = paths.training_report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "schema_version": 1,
        "slug": metadata.slug,
        "task": metadata.task,
        "metric": metadata.metric,
        "metric_direction": metadata.metric_direction,
        "cv_folds": folds,
        "models": results,
        "best_model": best_model_name,
        "best_score": results[best_model_name]["mean"],
        "time_budget_minutes": time_budget_minutes,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    submission_path = paths.submission_csv
    return TrainingResult(
        best_model_name=best_model_name,
        model_path=model_path,
        model_info_path=model_info_path,
        report_path=report_path,
        submission_path=submission_path,
        sample_submission_path=schema.sample_submission_path,
    )


def train_torch_tabular(
    metadata: CompetitionMetadata,
    *,
    paths: CompetitionPaths,
    time_budget_minutes: int,
    device: str,
    random_seed: int = 42,
) -> TrainingResult:
    schema = metadata.schema
    if len(schema.target_columns) != 1:
        raise ValueError("Multi-target competitions are not supported yet.")

    train_df = pd.read_csv(schema.train_path)
    target_col = schema.target_columns[0]
    feature_cols = schema.feature_columns

    x = train_df[feature_cols]
    y = train_df[target_col]

    preprocessor = _build_preprocessor(schema, feature_cols)
    x_processed = preprocessor.fit_transform(x)
    if hasattr(x_processed, "toarray"):
        x_processed = x_processed.toarray()

    label_encoder = None
    num_classes = 0
    y_encoded = y
    if metadata.task == "classification":
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        num_classes = len(label_encoder.classes_)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_processed, y_encoded, test_size=0.2, random_state=random_seed
    )

    start = time.monotonic()
    model = _train_torch_model(
        x_train,
        y_train,
        task=metadata.task,
        num_classes=num_classes,
        device=device,
        seed=random_seed,
    )
    duration = time.monotonic() - start

    preds_valid = _predict_torch_array(
        model,
        x_valid,
        task=metadata.task,
        num_classes=num_classes,
        prediction_kind=metadata.prediction_kind,
        device=device,
    )
    if metadata.task == "classification":
        if metadata.prediction_kind == "probability":
            preds_eval = (np.asarray(preds_valid) >= 0.5).astype(int)
        else:
            preds_eval = np.asarray(preds_valid)
        score = float(accuracy_score(y_valid, preds_eval))
    else:
        score = float(mean_squared_error(y_valid, preds_valid, squared=False))

    paths.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = paths.models_dir / "torch_mlp.pt"
    _save_torch_model(model, model_path)

    preprocessor_path = paths.models_dir / "torch_preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    label_encoder_path = None
    if label_encoder is not None:
        label_encoder_path = paths.models_dir / "torch_label_encoder.joblib"
        joblib.dump(label_encoder, label_encoder_path)

    model_info_path = paths.model_info_path
    extra = {
        "preprocessor_path": str(preprocessor_path),
        "label_encoder_path": str(label_encoder_path) if label_encoder_path else None,
        "input_dim": int(x_processed.shape[1]),
        "num_classes": int(num_classes),
        "device": device,
    }
    _write_model_info(
        model_info_path,
        metadata=metadata,
        model_name="torch_mlp",
        model_type="torch",
        model_path=model_path,
        feature_columns=feature_cols,
        extra=extra,
    )

    report_path = paths.training_report_path
    report_payload = {
        "schema_version": 1,
        "slug": metadata.slug,
        "task": metadata.task,
        "metric": metadata.metric,
        "metric_direction": metadata.metric_direction,
        "cv_folds": 1,
        "models": {
            "torch_mlp": {
                "fold_scores": [score],
                "mean": score,
                "std": 0.0,
                "training_time_seconds": duration,
            }
        },
        "best_model": "torch_mlp",
        "best_score": score,
        "time_budget_minutes": time_budget_minutes,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    submission_path = paths.submission_csv
    return TrainingResult(
        best_model_name="torch_mlp",
        model_path=model_path,
        model_info_path=model_info_path,
        report_path=report_path,
        submission_path=submission_path,
        sample_submission_path=schema.sample_submission_path,
    )


def predict_tabular(
    metadata: CompetitionMetadata,
    *,
    paths: CompetitionPaths,
    model_info_path: Path | None = None,
) -> Path:
    schema = metadata.schema
    info_path = model_info_path or paths.model_info_path
    if not info_path.exists():
        raise FileNotFoundError(f"Model info not found: {info_path}. Run training first.")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    model_type = info["model_type"]
    feature_columns = info["feature_columns"]
    id_column = info["id_column"]
    target_column = info["target_column"]
    prediction_kind = info.get("prediction_kind", metadata.prediction_kind)

    test_df = pd.read_csv(schema.test_path)
    sample_df = pd.read_csv(schema.sample_submission_path)

    x_test = test_df[feature_columns]

    if model_type == "catboost":
        model = _load_catboost(metadata.task, Path(info["model_path"]))
        preds = _predict_catboost(model, x_test, prediction_kind)
    elif model_type == "torch":
        preprocessor = joblib.load(info["preprocessor_path"])
        label_encoder = None
        if info.get("label_encoder_path"):
            label_encoder = joblib.load(info["label_encoder_path"])
        preds = _predict_torch_model(
            info["model_path"],
            preprocessor=preprocessor,
            label_encoder=label_encoder,
            x_test=x_test,
            task=info["task"],
            prediction_kind=prediction_kind,
        )
    else:
        model = joblib.load(info["model_path"])
        preds = _predict_sklearn(model, x_test, prediction_kind)

    submission = _build_submission(
        sample_df,
        test_df,
        preds,
        id_column=id_column,
        target_column=target_column,
    )

    submission_path = paths.submission_csv
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    return submission_path


def _build_model(
    model_name: str,
    metadata: CompetitionMetadata,
    feature_columns: list[str],
    *,
    class_count: int | None,
):
    schema = metadata.schema
    if model_name in {"catboost", "catboost_gpu"}:
        task_type = "GPU" if model_name == "catboost_gpu" else "CPU"
        if metadata.task == "classification":
            if class_count and class_count > 2:
                loss_function = "MultiClass"
                eval_metric = "Accuracy"
            else:
                loss_function = "Logloss"
                eval_metric = "Accuracy"
            return CatBoostClassifier(
                depth=6,
                learning_rate=0.1,
                iterations=500,
                loss_function=loss_function,
                eval_metric=eval_metric,
                random_seed=42,
                verbose=False,
                allow_writing_files=False,
                task_type=task_type,
            )
        return CatBoostRegressor(
            depth=6,
            learning_rate=0.1,
            iterations=500,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            task_type=task_type,
        )

    pre = _build_preprocessor(schema, feature_columns)

    if model_name == "hist_gb":
        if metadata.task == "classification":
            estimator = HistGradientBoostingClassifier(random_state=42)
        else:
            estimator = HistGradientBoostingRegressor(random_state=42)
    else:
        if metadata.task == "classification":
            estimator = LogisticRegression(max_iter=2000)
        else:
            estimator = Ridge()

    return Pipeline([("pre", pre), ("model", estimator)])


def _build_preprocessor(schema, feature_columns: list[str]) -> ColumnTransformer:
    cat_cols = schema.categorical_columns + schema.datetime_columns
    num_cols = [col for col in feature_columns if col not in cat_cols]
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


def _make_splitter(task: str, folds: int, seed: int):
    if task == "classification":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return KFold(n_splits=folds, shuffle=True, random_state=seed)


def _sanitize_folds(task: str, y: pd.Series, folds: int) -> int:
    if len(y) < 2:
        raise ValueError("Not enough samples for cross-validation.")
    min_folds = 2
    if folds < min_folds:
        return min_folds
    if task == "classification":
        counts = y.value_counts(dropna=True)
        if counts.empty:
            return min_folds
        if counts.min() < min_folds:
            raise ValueError("Not enough samples per class for cross-validation.")
        return int(max(min_folds, min(folds, counts.min())))
    return int(max(min_folds, min(folds, len(y))))


def _fit_catboost(
    model,
    x_tr: pd.DataFrame,
    y_tr: pd.Series,
    x_va: pd.DataFrame | None,
    y_va: pd.Series | None,
    schema,
    seed: int,
) -> None:
    cat_cols = schema.categorical_columns + schema.datetime_columns
    cat_features = [x_tr.columns.get_loc(col) for col in cat_cols if col in x_tr.columns]
    fit_kwargs = {"cat_features": cat_features}
    if x_va is not None and y_va is not None:
        fit_kwargs["eval_set"] = (x_va, y_va)
        fit_kwargs["early_stopping_rounds"] = 50
    model.set_params(random_seed=seed)
    model.fit(x_tr, y_tr, **fit_kwargs)


def _score_predictions(task: str, y_true, y_pred) -> float:
    y_pred = np.asarray(y_pred).ravel()
    if task == "classification":
        return float(accuracy_score(y_true, y_pred))
    return float(mean_squared_error(y_true, y_pred, squared=False))


def _is_better_score(best_score: float | None, candidate: float, direction: str) -> bool:
    if best_score is None:
        return True
    if direction == "maximize":
        return candidate > best_score
    return candidate < best_score


def _save_model(model_name: str, model, models_dir: Path) -> Path:
    if model_name in {"catboost", "catboost_gpu"}:
        suffix = "catboost_gpu" if model_name == "catboost_gpu" else "catboost"
        path = models_dir / f"{suffix}.cbm"
        model.save_model(path)
        return path
    path = models_dir / f"{model_name}.joblib"
    joblib.dump(model, path)
    return path


def _write_model_info(
    path: Path,
    *,
    metadata: CompetitionMetadata,
    model_name: str,
    model_type: str,
    model_path: Path,
    feature_columns: list[str],
    extra: dict[str, object] | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "model_name": model_name,
        "model_type": model_type,
        "model_path": str(model_path),
        "id_column": metadata.schema.id_column,
        "target_column": metadata.schema.target_columns[0],
        "feature_columns": feature_columns,
        "categorical_columns": metadata.schema.categorical_columns,
        "datetime_columns": metadata.schema.datetime_columns,
        "numeric_columns": metadata.schema.numeric_columns,
        "task": metadata.task,
        "prediction_kind": metadata.prediction_kind,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_catboost(task: str, path: Path):
    if task == "classification":
        model = CatBoostClassifier()
    else:
        model = CatBoostRegressor()
    model.load_model(path)
    return model


def _predict_sklearn(model, x: pd.DataFrame, prediction_kind: str) -> np.ndarray:
    if prediction_kind == "probability":
        proba = model.predict_proba(x)
        if proba.shape[1] == 1:
            return proba[:, 0]
        return proba[:, 1]
    return np.asarray(model.predict(x)).ravel()


def _predict_catboost(model, x: pd.DataFrame, prediction_kind: str) -> np.ndarray:
    if prediction_kind == "probability":
        proba = model.predict_proba(x)
        if proba.shape[1] == 1:
            return proba[:, 0]
        return proba[:, 1]
    return np.asarray(model.predict(x)).ravel()


def _train_torch_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    task: str,
    num_classes: int,
    device: str,
    seed: int,
):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y)
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


def _save_torch_model(model, path: Path) -> None:
    import torch

    torch.save(model.state_dict(), path)


def _load_torch_model(path: Path, *, input_dim: int, task: str, num_classes: int):
    import torch
    import torch.nn as nn

    output_dim = num_classes if task == "classification" and num_classes > 2 else 1
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, output_dim),
    )
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def _predict_torch_array(
    model,
    x: np.ndarray,
    *,
    task: str,
    num_classes: int,
    prediction_kind: str,
    device: str,
) -> np.ndarray:
    import torch

    model.to(device)
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


def _predict_torch_model(
    model_path: str,
    *,
    preprocessor,
    label_encoder,
    x_test: pd.DataFrame,
    task: str,
    prediction_kind: str,
) -> np.ndarray:
    x_processed = preprocessor.transform(x_test)
    if hasattr(x_processed, "toarray"):
        x_processed = x_processed.toarray()

    input_dim = x_processed.shape[1]
    num_classes = len(label_encoder.classes_) if label_encoder is not None else 0
    model = _load_torch_model(Path(model_path), input_dim=input_dim, task=task, num_classes=num_classes)
    preds = _predict_torch_array(
        model,
        x_processed,
        task=task,
        num_classes=num_classes,
        prediction_kind=prediction_kind,
        device="cpu",
    )
    if label_encoder is not None and prediction_kind == "class":
        preds = label_encoder.inverse_transform(np.asarray(preds, dtype=int))
    return preds


def _build_submission(
    sample_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preds: np.ndarray,
    *,
    id_column: str,
    target_column: str,
) -> pd.DataFrame:
    submission = sample_df.copy()
    if id_column in test_df.columns:
        if test_df[id_column].duplicated().any():
            raise ValueError(f"Duplicate ids detected in test column '{id_column}'.")
        pred_map = pd.Series(preds, index=test_df[id_column])
        submission[target_column] = submission[id_column].map(pred_map)
        if submission[target_column].isna().any():
            raise ValueError("Missing predictions after aligning by id column.")
    else:
        if len(preds) != len(submission):
            raise ValueError("Prediction length does not match sample_submission rows.")
        submission[target_column] = preds
    return submission
