from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
    label_encoder = None
    if data.task == "classification":
        label_encoder = LabelEncoder()
        label_encoder.fit(data.train[data.target_column])

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
    candidates = _build_candidates(data, selection)
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
    )

    preds = _predict_with_model(
        data,
        best_model,
        best_preprocessor,
        metric,
        selection=selection_score,
        prediction_kind=data.prediction_kind,
        label_encoder=label_encoder,
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


def _build_candidates(data: CompetitionData, accelerator: str) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    if data.task == "classification":
        candidates.append(
            {
                "name": "logreg",
                "model": LogisticRegression(max_iter=2000),
                "preprocessing": _build_linear_preprocessor(data),
            }
        )
        candidates.append(
            {
                "name": "hist_gb",
                "model": HistGradientBoostingClassifier(),
                "preprocessing": _build_tree_preprocessor(data),
            }
        )
        if accelerator == "cuda":
            candidates.append(
                {
                    "name": "catboost_gpu",
                    "model": CatBoostClassifier(
                        iterations=300,
                        depth=6,
                        learning_rate=0.1,
                        loss_function="Logloss",
                        task_type="GPU",
                        verbose=False,
                        allow_writing_files=False,
                    ),
                    "preprocessing": None,
                }
            )
        elif accelerator == "mps":
            candidates.append(
                {"name": "torch_mlp_mps", "model": None, "preprocessing": _build_linear_preprocessor(data)}
            )
        else:
            candidates.append(
                {
                    "name": "catboost_cpu",
                    "model": CatBoostClassifier(
                        iterations=300,
                        depth=6,
                        learning_rate=0.1,
                        loss_function="Logloss",
                        task_type="CPU",
                        verbose=False,
                        allow_writing_files=False,
                    ),
                    "preprocessing": None,
                }
            )
    else:
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
                        iterations=300,
                        depth=6,
                        learning_rate=0.1,
                        loss_function="RMSE",
                        task_type="GPU",
                        verbose=False,
                        allow_writing_files=False,
                    ),
                    "preprocessing": None,
                }
            )
        elif accelerator == "mps":
            candidates.append(
                {"name": "torch_mlp_mps", "model": None, "preprocessing": _build_linear_preprocessor(data)}
            )
        else:
            candidates.append(
                {
                    "name": "catboost_cpu",
                    "model": CatBoostRegressor(
                        iterations=300,
                        depth=6,
                        learning_rate=0.1,
                        loss_function="RMSE",
                        task_type="CPU",
                        verbose=False,
                        allow_writing_files=False,
                    ),
                    "preprocessing": None,
                }
            )
    return candidates


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
) -> tuple[EvaluationResult, dict[str, object], object, ColumnTransformer | None]:
    best_score: float | None = None
    best_candidate: dict[str, object] | None = None
    best_model = None
    best_preprocessor = None
    best_train_score: float | None = None
    best_val_score: float | None = None
    best_std: float | None = None
    best_fold_scores: list[float] | None = None

    for candidate in candidates:
        name = candidate["name"]
        if name == "torch_mlp_mps":
            scores, train_score, val_score = _evaluate_torch(
                data,
                selection=selection,
                metric=metric,
                seed=seed,
                holdout_frac=holdout_frac,
                cv_folds=cv_folds,
                direction=direction,
                label_encoder=label_encoder,
            )
            model = None
            preprocessor = candidate["preprocessing"]
        else:
            model = candidate["model"]
            preprocessor = candidate["preprocessing"]
            scores, train_score, val_score = _evaluate_sklearn_or_catboost(
                data,
                model=model,
                preprocessor=preprocessor,
                selection=selection,
                metric=metric,
                seed=seed,
                holdout_frac=holdout_frac,
                cv_folds=cv_folds,
                label_encoder=label_encoder,
            )

        mean_score = float(np.mean(scores)) if scores else float("nan")
        std_score = float(np.std(scores)) if scores else None

        if best_score is None or _is_better(mean_score, best_score, direction):
            best_score = mean_score
            best_std = std_score
            best_candidate = candidate
            best_model = model
            best_preprocessor = preprocessor
            best_train_score = train_score
            best_val_score = val_score
            best_fold_scores = scores

    if best_candidate is None:
        raise RuntimeError("No candidate models were evaluated.")

    evaluation = EvaluationResult(
        score_source=selection.source,
        metric=metric,
        direction=direction,  # type: ignore[arg-type]
        value=float(best_score),
        std=best_std,
        train_score=best_train_score,
        val_score=best_val_score,
        fold_scores=best_fold_scores,
    )
    if best_candidate["name"] == "torch_mlp_mps":
        return evaluation, best_candidate, None, best_preprocessor
    return evaluation, best_candidate, best_model, best_preprocessor


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
) -> tuple[list[float], float | None, float | None]:
    if selection.source == "test":
        x_train, y_train, x_eval, y_eval = _prepare_test_split(data, selection, preprocessor)
        fitted, train_score = _fit_and_score(
            model, x_train, y_train, x_train, y_train, metric, data, preprocessor, label_encoder
        )
        _, eval_score = _fit_and_score(
            model, x_train, y_train, x_eval, y_eval, metric, data, preprocessor, label_encoder
        )
        return [eval_score], train_score, eval_score

    if selection.source == "holdout":
        x_train, x_val, y_train, y_val = _holdout_split(data, seed, holdout_frac)
        fitted, train_score = _fit_and_score(
            model, x_train, y_train, x_train, y_train, metric, data, preprocessor, label_encoder
        )
        _, val_score = _fit_and_score(model, x_train, y_train, x_val, y_val, metric, data, preprocessor, label_encoder)
        return [val_score], train_score, val_score

    scores = []
    splitter = _splitter(data, seed, cv_folds)
    for train_idx, val_idx in splitter.split(data.train[data.feature_columns], data.train[data.target_column]):
        x_tr = data.train.iloc[train_idx][data.feature_columns]
        x_val = data.train.iloc[val_idx][data.feature_columns]
        y_tr = data.train.iloc[train_idx][data.target_column]
        y_val = data.train.iloc[val_idx][data.target_column]
        _, fold_score = _fit_and_score(model, x_tr, y_tr, x_val, y_val, metric, data, preprocessor, label_encoder)
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
) -> tuple[list[float], float | None, float | None]:
    if selection.source == "test":
        x_train, y_train, x_eval, y_eval = _prepare_test_split(data, selection, None)
        preds_eval, train_score = _fit_torch_and_score(x_train, y_train, x_train, y_train, metric, data, label_encoder)
        preds_eval, eval_score = _fit_torch_and_score(x_train, y_train, x_eval, y_eval, metric, data, label_encoder)
        return [eval_score], train_score, eval_score

    if selection.source == "holdout":
        x_train, x_val, y_train, y_val = _holdout_split(data, seed, holdout_frac)
        preds_train, train_score = _fit_torch_and_score(x_train, y_train, x_train, y_train, metric, data, label_encoder)
        preds_val, val_score = _fit_torch_and_score(x_train, y_train, x_val, y_val, metric, data, label_encoder)
        return [val_score], train_score, val_score

    scores = []
    splitter = _splitter(data, seed, cv_folds)
    for train_idx, val_idx in splitter.split(data.train[data.feature_columns], data.train[data.target_column]):
        x_tr = data.train.iloc[train_idx][data.feature_columns]
        x_val = data.train.iloc[val_idx][data.feature_columns]
        y_tr = data.train.iloc[train_idx][data.target_column]
        y_val = data.train.iloc[val_idx][data.target_column]
        _, fold_score = _fit_torch_and_score(x_tr, y_tr, x_val, y_val, metric, data, label_encoder)
        scores.append(fold_score)
    return scores, None, float(np.mean(scores)) if scores else None


def _fit_and_score(model, x_train, y_train, x_eval, y_eval, metric, data, preprocessor, label_encoder):
    x_train_proc, x_eval_proc = _apply_preprocessor(preprocessor, x_train, x_eval)
    model.fit(x_train_proc, _encode_target(data, y_train, label_encoder))
    preds = _predict_for_metric(model, x_eval_proc, data, metric)
    score = compute_metric(metric, _encode_target(data, y_eval, label_encoder), preds)
    return model, score


def _fit_torch_and_score(x_train, y_train, x_eval, y_eval, metric, data, label_encoder):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    x_train_proc = _dense_array(x_train)
    x_eval_proc = _dense_array(x_eval)
    y_train_enc = _encode_target(data, y_train, label_encoder)
    y_eval_enc = _encode_target(data, y_eval, label_encoder)

    device = torch.device("mps")
    x_tensor = torch.tensor(x_train_proc, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train_enc).to(device)
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    input_dim = x_tensor.shape[1]
    output_dim = 1 if data.task == "regression" else int(np.unique(y_train_enc).size)
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, output_dim),
    ).to(device)

    if data.task == "classification":
        loss_fn = nn.CrossEntropyLoss() if output_dim > 2 else nn.BCEWithLogitsLoss()
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
        eval_tensor = torch.tensor(x_eval_proc, dtype=torch.float32).to(device)
        outputs = model(eval_tensor).cpu().numpy()

    preds = _torch_outputs_to_preds(outputs, data, metric, prediction_kind=None)
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


def _encode_target(data: CompetitionData, y, label_encoder: LabelEncoder | None):
    if data.task != "classification":
        return np.asarray(y)
    if label_encoder is None:
        encoder = LabelEncoder()
        return encoder.fit_transform(y)
    return label_encoder.transform(y)


def _predict_for_metric(model, x, data: CompetitionData, metric: str):
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
    return model.predict(x)


def _predict_for_submission(model, x, data: CompetitionData, metric: str, prediction_kind: str):
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
    return model.predict(x)


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
):
    x_train = data.train[data.feature_columns]
    x_test = data.test[data.feature_columns]
    if preprocessor is not None:
        x_train = preprocessor.fit_transform(x_train)
        x_test = preprocessor.transform(x_test)
        x_train = _dense_array(x_train)
        x_test = _dense_array(x_test)
    if model is None:
        return _predict_torch_full(data, metric, x_train, x_test, label_encoder)
    model.fit(x_train, _encode_target(data, data.train[data.target_column], label_encoder))
    preds = _predict_for_submission(model, x_test, data, metric, prediction_kind)
    if data.task == "classification" and prediction_kind == "class":
        if preds.ndim > 1:
            preds = preds.argmax(axis=1)
        if data.train[data.target_column].dtype == "object" and label_encoder is not None:
            preds = label_encoder.inverse_transform(np.asarray(preds, dtype=int))
    return preds


def _predict_torch_full(data: CompetitionData, metric: str, x_train, x_test, label_encoder: LabelEncoder | None):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("mps")
    x_train = _dense_array(x_train)
    x_test = _dense_array(x_test)
    y_train_enc = _encode_target(data, data.train[data.target_column], label_encoder)

    x_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train_enc).to(device)
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    input_dim = x_tensor.shape[1]
    output_dim = 1 if data.task == "regression" else int(np.unique(y_train_enc).size)
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, output_dim),
    ).to(device)

    if data.task == "classification":
        loss_fn = nn.CrossEntropyLoss() if output_dim > 2 else nn.BCEWithLogitsLoss()
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
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
        outputs = model(x_test_tensor).cpu().numpy()

    preds = _torch_outputs_to_preds(outputs, data, metric, prediction_kind=data.prediction_kind)
    if data.task == "classification" and data.prediction_kind == "class" and label_encoder is not None:
        if preds.ndim > 1:
            preds = preds.argmax(axis=1)
        preds = label_encoder.inverse_transform(np.asarray(preds, dtype=int))
    return preds


def _is_better(candidate: float, best: float, direction: str) -> bool:
    if direction == "minimize":
        return candidate < best
    return candidate > best
