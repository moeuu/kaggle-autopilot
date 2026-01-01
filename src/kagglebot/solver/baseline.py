from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from kagglebot.compute import Compute, detect_local_gpu
from kagglebot.exceptions import GPUNotAvailableError
from kagglebot.solver.io import CompetitionData, load_competition_data, write_submission


@dataclass(frozen=True)
class BaselineResult:
    submission_path: Path
    metrics_path: Path
    model_type: str


def train_and_predict(
    *,
    data_dir: Path,
    output_path: Path,
    compute: Compute,
    strict_accelerator: bool,
    seed: int,
    metrics_path: Path,
) -> BaselineResult:
    data = load_competition_data(data_dir)
    result = _train_and_predict(data, compute=compute, strict_accelerator=strict_accelerator, seed=seed)
    preds = result.predictions
    if data.task == "classification" and data.prediction_kind == "class":
        if data.train[data.target_column].dtype == "object":
            encoder = LabelEncoder()
            encoder.fit(data.train[data.target_column])
            preds = encoder.inverse_transform(np.asarray(preds, dtype=int))

    submission_path = write_submission(
        data.sample,
        data.test,
        preds,
        id_column=data.id_column,
        target_column=data.target_column,
        output_path=output_path,
    )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "schema_version": 1,
        "task": data.task,
        "metric": result.metric_name,
        "score": float(result.score),
        "model_type": result.model_type,
        "accelerator": result.accelerator,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return BaselineResult(submission_path=submission_path, metrics_path=metrics_path, model_type=result.model_type)


@dataclass(frozen=True)
class _FitResult:
    predictions: np.ndarray
    score: float
    metric_name: str
    model_type: str
    accelerator: str


def _train_and_predict(
    data: CompetitionData,
    *,
    compute: Compute,
    strict_accelerator: bool,
    seed: int,
) -> _FitResult:
    x = data.train[data.feature_columns]
    y = data.train[data.target_column]

    label_encoder = None
    num_classes = 0
    y_encoded = y
    if data.task == "classification":
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        num_classes = len(label_encoder.classes_)

    stratify = y_encoded if data.task == "classification" else None
    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y_encoded,
        test_size=0.2,
        random_state=seed,
        stratify=stratify,
    )

    if compute == Compute.local_gpu:
        availability = detect_local_gpu()
        if availability.cuda:
            return _train_catboost_gpu(
                data,
                x_train=x_train,
                x_valid=x_valid,
                y_train=y_train,
                y_valid=y_valid,
                num_classes=num_classes,
            )
        if availability.mps:
            return _train_torch_mps(
                data,
                x_train=x_train,
                x_valid=x_valid,
                y_train=y_train,
                y_valid=y_valid,
                num_classes=num_classes,
                seed=seed,
            )
        if strict_accelerator:
            raise GPUNotAvailableError(
                "No local GPU detected for --compute local_gpu. Disable --strict-accelerator to fall back to CPU."
            )

    return _train_cpu(
        data,
        x_train=x_train,
        x_valid=x_valid,
        y_train=y_train,
        y_valid=y_valid,
    )


def _build_preprocessor(feature_cols: list[str], train_df: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in feature_cols if train_df[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]
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


def _train_cpu(
    data: CompetitionData,
    *,
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_train,
    y_valid,
) -> _FitResult:
    preprocessor = _build_preprocessor(data.feature_columns, data.train)
    if data.task == "classification":
        estimator = LogisticRegression(max_iter=2000)
        metric_name = "accuracy"
        model_type = "logreg"
    else:
        estimator = Ridge()
        metric_name = "rmse"
        model_type = "ridge"

    model = Pipeline([("pre", preprocessor), ("model", estimator)])
    model.fit(x_train, y_train)

    preds_valid = _predict_sklearn(model, x_valid, data.task, "class")
    score = _score_predictions(data.task, y_valid, preds_valid)

    preds = _predict_sklearn(model, data.test[data.feature_columns], data.task, data.prediction_kind)
    return _FitResult(
        predictions=np.asarray(preds),
        score=score,
        metric_name=metric_name,
        model_type=model_type,
        accelerator="cpu",
    )


def _train_catboost_gpu(
    data: CompetitionData,
    *,
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_train,
    y_valid,
    num_classes: int,
) -> _FitResult:
    cat_cols = [c for c in data.feature_columns if data.train[c].dtype == "object"]
    cat_features = [x_train.columns.get_loc(c) for c in cat_cols if c in x_train.columns]

    if data.task == "classification":
        loss_function = "Logloss" if num_classes <= 2 else "MultiClass"
        model = CatBoostClassifier(
            depth=6,
            learning_rate=0.1,
            iterations=300,
            loss_function=loss_function,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            task_type="GPU",
        )
        metric_name = "accuracy"
    else:
        model = CatBoostRegressor(
            depth=6,
            learning_rate=0.1,
            iterations=300,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            task_type="GPU",
        )
        metric_name = "rmse"

    start = time.monotonic()
    model.fit(x_train, y_train, cat_features=cat_features, eval_set=(x_valid, y_valid))
    _ = time.monotonic() - start

    preds_valid = model.predict(x_valid)
    score = _score_predictions(data.task, y_valid, preds_valid)

    preds = _predict_catboost(model, data.test[data.feature_columns], data.prediction_kind)
    return _FitResult(
        predictions=np.asarray(preds),
        score=score,
        metric_name=metric_name,
        model_type="catboost_gpu",
        accelerator="gpu",
    )


def _train_torch_mps(
    data: CompetitionData,
    *,
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_train,
    y_valid,
    num_classes: int,
    seed: int,
) -> _FitResult:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    preprocessor = _build_preprocessor(data.feature_columns, data.train)
    x_train_enc = preprocessor.fit_transform(x_train)
    x_valid_enc = preprocessor.transform(x_valid)
    x_test_enc = preprocessor.transform(data.test[data.feature_columns])
    if hasattr(x_train_enc, "toarray"):
        x_train_enc = x_train_enc.toarray()
        x_valid_enc = x_valid_enc.toarray()
        x_test_enc = x_test_enc.toarray()

    torch.manual_seed(seed)
    device = "mps"

    x_tensor = torch.tensor(x_train_enc, dtype=torch.float32)
    y_tensor = torch.tensor(y_train)
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    input_dim = x_train_enc.shape[1]
    output_dim = num_classes if data.task == "classification" and num_classes > 2 else 1
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, output_dim),
    ).to(device)

    if data.task == "classification":
        loss_fn = nn.CrossEntropyLoss() if num_classes > 2 else nn.BCEWithLogitsLoss()
        metric_name = "accuracy"
    else:
        loss_fn = nn.MSELoss()
        metric_name = "rmse"
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(10):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            if data.task == "classification" and num_classes == 2:
                yb = yb.float().view(-1, 1)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()

    preds_valid = _predict_torch(model, x_valid_enc, data.task, num_classes, "class", device)
    score = _score_predictions(data.task, y_valid, preds_valid)
    preds = _predict_torch(model, x_test_enc, data.task, num_classes, data.prediction_kind, device)

    return _FitResult(
        predictions=np.asarray(preds),
        score=score,
        metric_name=metric_name,
        model_type="torch_mps",
        accelerator="gpu",
    )


def _score_predictions(task: str, y_true, y_pred) -> float:
    y_pred = np.asarray(y_pred).ravel()
    if task == "classification":
        return float(accuracy_score(y_true, y_pred))
    return float(mean_squared_error(y_true, y_pred, squared=False))


def _predict_sklearn(model, x: pd.DataFrame, task: str, prediction_kind: str) -> np.ndarray:
    if task == "classification" and prediction_kind == "probability" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.shape[1] == 1:
            return proba[:, 0]
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba.max(axis=1)
    return np.asarray(model.predict(x)).ravel()


def _predict_catboost(model, x: pd.DataFrame, prediction_kind: str) -> np.ndarray:
    if prediction_kind == "probability":
        proba = model.predict_proba(x)
        if proba.shape[1] == 1:
            return proba[:, 0]
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba.max(axis=1)
    return np.asarray(model.predict(x)).ravel()


def _predict_torch(model, x: np.ndarray, task: str, num_classes: int, prediction_kind: str, device: str) -> np.ndarray:
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
