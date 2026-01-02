from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from rich import print

from kagglebot.exceptions import KernelFailedError, KernelTimeoutError, RulesNotAcceptedError
from kagglebot.kaggle_api import check_rules_accepted, kernels_init, kernels_output, kernels_push, kernels_status
from kagglebot.validators import validate_kernel_package


@dataclass(frozen=True)
class KernelRunResult:
    kernel_id: str
    output_dir: Path
    submission_path: Path | None
    metrics_path: Path | None


def sanitize_kernel_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    return cleaned[:50]


def find_submission_file(output_dir: Path) -> Path | None:
    return _find_output_file(output_dir, "submission.csv")


def resolve_kaggle_username(explicit: str | None) -> str:
    if explicit:
        return explicit
    env_user = os.getenv("KAGGLE_USERNAME")
    if env_user:
        return env_user
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        data = json.loads(kaggle_json.read_text(encoding="utf-8"))
        if "username" in data:
            return str(data["username"])
    raise ValueError("Kaggle username not found. Provide --kaggle-username or set KAGGLE_USERNAME.")


def run_kernel(
    *,
    slug: str,
    run_id: str,
    iteration: int,
    base_dir: Path,
    kaggle_username: str,
    kernel_name: str | None,
    accelerator: str,
    enable_internet: bool,
    score_source: str,
    metric: str,
    direction: str,
    holdout_frac: float,
    cv_folds: int,
    seed: int,
    dry_run: bool,
    timeout_minutes: int | None,
) -> KernelRunResult:
    kernel_dir = base_dir / slug / "kernels" / run_id
    output_dir = base_dir / slug / "runs" / run_id / f"iter-{iteration}" / "output"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dry_run and not check_rules_accepted(slug, dry_run=False):
        raise RulesNotAcceptedError("Competition rules not accepted.")

    if not dry_run:
        print(f"[cyan]kernel init[/cyan]: {kernel_dir}")
        kernels_init(kernel_dir, dry_run=False)

    kernel_slug = _resolve_kernel_slug(kernel_name, slug, run_id, iteration)
    kernel_id = f"{kaggle_username}/{kernel_slug}"
    _write_kernel_metadata(
        kernel_dir=kernel_dir,
        kernel_id=kernel_id,
        title=kernel_slug,
        code_file="kernel.py",
        accelerator=accelerator,
        enable_internet=enable_internet,
        competition_slug=slug,
    )
    _write_kernel_script(
        kernel_dir=kernel_dir,
        slug=slug,
        accelerator=accelerator,
        score_source=score_source,
        metric=metric,
        direction=direction,
        holdout_frac=holdout_frac,
        cv_folds=cv_folds,
        seed=seed,
        run_id=run_id,
        iteration=iteration,
    )

    validate_kernel_package(kernel_dir)

    if dry_run:
        return KernelRunResult(kernel_id=kernel_id, output_dir=output_dir, submission_path=None, metrics_path=None)

    print(f"[cyan]kernel push[/cyan]: {kernel_dir}")
    kernels_push(kernel_dir, slug=slug, dry_run=False)
    print(f"[cyan]kernel status[/cyan]: {kernel_id}")
    _wait_for_kernel(kernel_id, slug, timeout_minutes)
    print(f"[cyan]kernel output[/cyan]: {output_dir}")
    kernels_output(kernel_id, output_dir, slug=slug, dry_run=False)

    submission_path = _find_output_file(output_dir, "submission.csv")
    metrics_path = _find_output_file(output_dir, "metrics.json")
    return KernelRunResult(
        kernel_id=kernel_id, output_dir=output_dir, submission_path=submission_path, metrics_path=metrics_path
    )


def _resolve_kernel_slug(kernel_name: str | None, slug: str, run_id: str, iteration: int) -> str:
    if kernel_name:
        return sanitize_kernel_slug(kernel_name)
    suffix = f"{run_id[-6:]}-i{iteration}"
    prefix = f"kagglebot-{slug}"
    max_len = 50
    allowed_prefix_len = max_len - len(suffix) - 1
    if allowed_prefix_len < 1:
        prefix = "kagglebot"
    else:
        prefix = prefix[:allowed_prefix_len].rstrip("-")
    base = f"{prefix}-{suffix}"
    return sanitize_kernel_slug(base)


def _write_kernel_metadata(
    *,
    kernel_dir: Path,
    kernel_id: str,
    title: str,
    code_file: str,
    accelerator: str,
    enable_internet: bool,
    competition_slug: str,
) -> None:
    meta_path = kernel_dir / "kernel-metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {}
    meta.update(
        {
            "id": kernel_id,
            "title": title,
            "code_file": code_file,
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": accelerator == "gpu",
            "enable_tpu": accelerator == "tpu",
            "enable_internet": bool(enable_internet),
            "competition_sources": [competition_slug],
            "dataset_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
    )
    if meta["enable_gpu"] and meta["enable_tpu"]:
        raise ValueError("kernel-metadata.json cannot enable both GPU and TPU.")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _write_kernel_script(
    *,
    kernel_dir: Path,
    slug: str,
    accelerator: str,
    score_source: str,
    metric: str,
    direction: str,
    holdout_frac: float,
    cv_folds: int,
    seed: int,
    run_id: str,
    iteration: int,
) -> None:
    script = _render_kernel_main(
        slug=slug,
        accelerator=accelerator,
        score_source=score_source,
        metric=metric,
        direction=direction,
        holdout_frac=holdout_frac,
        cv_folds=cv_folds,
        seed=seed,
        run_id=run_id,
        iteration=iteration,
    )
    (kernel_dir / "kernel.py").write_text(script, encoding="utf-8")


def _wait_for_kernel(kernel_id: str, slug: str, timeout_minutes: int | None) -> None:
    deadline = None
    if timeout_minutes is not None:
        deadline = time.monotonic() + max(timeout_minutes, 1) * 60
    last_status = None
    while True:
        output = kernels_status(kernel_id, slug=slug, dry_run=False)
        status = _parse_kernel_status(output).lower()
        if status != last_status:
            print(f"[cyan]kernel status[/cyan]: {status}")
            last_status = status
        if "complete" in status:
            return
        if "error" in status or "fail" in status:
            raise KernelFailedError(f"Kaggle kernel failed: {output}")
        time.sleep(10)
        if deadline is not None and time.monotonic() > deadline:
            raise KernelTimeoutError("Kaggle kernel did not complete within timeout.")


def _parse_kernel_status(output: str) -> str:
    match = re.search(r"status\\s+\\\"?([A-Za-z0-9_.-]+)\\\"?", output)
    if match:
        return match.group(1)
    return output.strip() or "unknown"


def _find_output_file(output_dir: Path, filename: str) -> Path | None:
    candidate = output_dir / filename
    if candidate.exists():
        return candidate
    matches = list(output_dir.rglob(filename))
    if matches:
        return matches[0]
    return None


def _render_kernel_main(
    *,
    slug: str,
    accelerator: str,
    score_source: str,
    metric: str,
    direction: str,
    holdout_frac: float,
    cv_folds: int,
    seed: int,
    run_id: str,
    iteration: int,
) -> str:
    return f'''import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

CONFIG = {{
    "slug": "{slug}",
    "accelerator": "{accelerator}",
    "score_source": "{score_source}",
    "metric": "{metric}",
    "direction": "{direction}",
    "holdout_frac": {holdout_frac},
    "cv_folds": {cv_folds},
    "seed": {seed},
    "run_id": "{run_id}",
    "iteration": {iteration},
}}

INPUT_ROOT = Path("/kaggle/input") / CONFIG["slug"]
WORKING = Path("/kaggle/working")
SUBMISSION_PATH = WORKING / "submission.csv"
METRICS_PATH = WORKING / "metrics.json"


def find_csvs(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.csv") if p.is_file()]


def pick_files(csvs: list[Path]) -> tuple[Path, Path, Path]:
    if not csvs:
        raise FileNotFoundError("No CSV files found.")
    def score_sample(path: Path) -> int:
        name = path.name.lower()
        if "sample_submission" in name:
            return 3
        if "submission" in name:
            return 1
        return 0
    sample = sorted(csvs, key=score_sample, reverse=True)[0]
    train = next((p for p in csvs if "train" in p.name.lower()), None)
    test = next((p for p in csvs if "test" in p.name.lower()), None)
    if train is None or test is None:
        raise FileNotFoundError("train.csv or test.csv not found.")
    return train, test, sample


def infer_target(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> tuple[str, str, list[str]]:
    id_col = sample.columns[0]
    candidates = [c for c in train.columns if c not in test.columns and c in sample.columns]
    target_cols = candidates or list(sample.columns[1:])
    if len(target_cols) != 1:
        raise ValueError("Only single-target competitions supported.")
    target = target_cols[0]
    features = [c for c in train.columns if c != target]
    if id_col in features:
        features.remove(id_col)
    return id_col, target, features


def infer_task(y: pd.Series) -> str:
    if y.dtype == "object":
        return "classification"
    nunique = y.nunique(dropna=True)
    if nunique <= 20 or nunique / max(len(y), 1) <= 0.05:
        return "classification"
    return "regression"


def metric_requires_proba(metric: str) -> bool:
    metric = metric.lower()
    return "logloss" in metric or "auc" in metric


def compute_metric(metric: str, y_true, y_pred) -> float:
    metric = metric.lower()
    if metric == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if metric == "rmsle":
        y_true = np.clip(np.asarray(y_true, dtype=float), 0, None)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
        return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))))
    if metric in ("logloss", "log_loss"):
        return float(log_loss(y_true, y_pred))
    if metric == "auc":
        return float(roc_auc_score(y_true, y_pred))
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_preprocessor(features: list[str], train: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in features if train[c].dtype == "object"]
    num_cols = [c for c in features if c not in cat_cols]
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


def find_label_file(root: Path) -> Path | None:
    for name in ["test_labels.csv", "labels.csv", "y_test.csv"]:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def select_score_source(test: pd.DataFrame, target_col: str, id_col: str) -> tuple[str, pd.Series | None]:
    source = CONFIG["score_source"]
    if source in ("auto", "test"):
        if target_col in test.columns:
            return "test", test[target_col]
        label_path = find_label_file(INPUT_ROOT)
        if label_path is not None:
            labels = pd.read_csv(label_path)
            if target_col in labels.columns and id_col in labels.columns:
                merged = test.merge(labels[[id_col, target_col]], on=id_col, how="inner")
                if not merged.empty:
                    return "test", merged[target_col]
        if source == "test":
            raise RuntimeError("score_source=test requested but no labeled test found.")
        return "holdout", None
    return source, None


def predict_for_metric(model, x, task: str, metric: str):
    if task == "classification" and metric_requires_proba(metric):
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x)
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            return proba
    return model.predict(x)


def predict_for_submission(model, x, task: str, metric: str, prediction_kind: str):
    if task == "classification" and (metric_requires_proba(metric) or prediction_kind == "probability"):
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x)
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            return proba
    return model.predict(x)


def evaluate_holdout(model, pre, x, y, task: str, metric: str, prediction_kind: str):
    stratify = y if task == "classification" else None
    x_tr, x_val, y_tr, y_val = train_test_split(
        x, y, test_size=CONFIG["holdout_frac"], random_state=CONFIG["seed"], stratify=stratify
    )
    x_tr_p = pre.fit_transform(x_tr)
    x_val_p = pre.transform(x_val)
    model.fit(x_tr_p, y_tr)
    preds = predict_for_metric(model, x_val_p, task, metric)
    return compute_metric(metric, y_val, preds), None


def evaluate_cv(model, pre, x, y, task: str, metric: str, prediction_kind: str):
    splitter = (
        StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=CONFIG["seed"])
        if task == "classification"
        else KFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=CONFIG["seed"])
    )
    scores = []
    for train_idx, val_idx in splitter.split(x, y):
        x_tr, x_val = x.iloc[train_idx], x.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        x_tr_p = pre.fit_transform(x_tr)
        x_val_p = pre.transform(x_val)
        model.fit(x_tr_p, y_tr)
        preds = predict_for_metric(model, x_val_p, task, metric)
        scores.append(compute_metric(metric, y_val, preds))
    return float(np.mean(scores)), float(np.std(scores))


def build_model(task: str):
    if CONFIG["accelerator"] == "gpu":
        try:
            import xgboost as xgb
            if task == "classification":
                return xgb.XGBClassifier(tree_method="gpu_hist", max_depth=6, n_estimators=200, learning_rate=0.1)
            return xgb.XGBRegressor(tree_method="gpu_hist", max_depth=6, n_estimators=200, learning_rate=0.1)
        except Exception:
            pass
    if task == "classification":
        return LogisticRegression(max_iter=2000)
    return Ridge()


def train_tpu(x_train, y_train, x_eval, task: str):
    import tensorflow as tf
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)

    with strategy.scope():
        output_units = 1 if task == "regression" else int(np.unique(y_train).size)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(x_train.shape[1],)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(output_units),
            ]
        )
        if task == "classification":
            if output_units > 2:
                model.add(tf.keras.layers.Activation("softmax"))
                model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            else:
                model.add(tf.keras.layers.Activation("sigmoid"))
                model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        else:
            model.compile(optimizer="adam", loss="mse")

    model.fit(x_train, y_train, epochs=5, batch_size=256, verbose=0)
    outputs = model.predict(x_eval, batch_size=256, verbose=0)
    return outputs


def main() -> None:
    train_path, test_path, sample_path = pick_files(find_csvs(INPUT_ROOT))
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)

    id_col, target_col, features = infer_target(train, test, sample)
    task = infer_task(train[target_col])
    prediction_kind = "probability" if sample[target_col].dtype.kind in {{"f", "c"}} else "class"

    label_encoder = None
    y = train[target_col]
    if task == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    x = train[features]
    pre = build_preprocessor(features, train)
    score_source, test_labels = select_score_source(test, target_col, id_col)

    std = None
    if CONFIG["accelerator"] == "tpu":
        x_full = pre.fit_transform(x)
        if hasattr(x_full, "toarray"):
            x_full = x_full.toarray()
        if score_source == "cv":
            scores = []
            splitter = (
                StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=CONFIG["seed"])
                if task == "classification"
                else KFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=CONFIG["seed"])
            )
            for train_idx, val_idx in splitter.split(x_full, y):
                preds = train_tpu(x_full[train_idx], y[train_idx], x_full[val_idx], task)
                scores.append(compute_metric(CONFIG["metric"], y[val_idx], preds))
            score = float(np.mean(scores))
            std = float(np.std(scores))
        else:
            preds = train_tpu(x_full, y, x_full, task)
            score = compute_metric(CONFIG["metric"], y, preds)
    else:
        model = build_model(task)
        if score_source == "cv":
            score, std = evaluate_cv(model, pre, x, y, task, CONFIG["metric"], prediction_kind)
        elif score_source == "test" and test_labels is not None:
            x_train_p = pre.fit_transform(x)
            x_test_p = pre.transform(test[features])
            model.fit(x_train_p, y)
            preds = predict_for_metric(model, x_test_p, task, CONFIG["metric"])
            score = compute_metric(CONFIG["metric"], test_labels, preds)
        else:
            score, std = evaluate_holdout(model, pre, x, y, task, CONFIG["metric"], prediction_kind)

    x_full = pre.fit_transform(x)
    if hasattr(x_full, "toarray"):
        x_full = x_full.toarray()
    if CONFIG["accelerator"] == "tpu":
        test_features = pre.transform(test[features])
        if hasattr(test_features, "toarray"):
            test_features = test_features.toarray()
        preds = train_tpu(x_full, y, test_features, task)
    else:
        model = build_model(task)
        model.fit(x_full, y)
        test_x = pre.transform(test[features])
        preds = predict_for_submission(model, test_x, task, CONFIG["metric"], prediction_kind)
    if task == "classification" and prediction_kind == "class" and label_encoder is not None:
        if preds.ndim > 1:
            preds = preds.argmax(axis=1)
        preds = label_encoder.inverse_transform(np.asarray(preds, dtype=int))

    submission = sample.copy()
    submission[target_col] = preds
    submission.to_csv(SUBMISSION_PATH, index=False)

    metrics = {{
        "run_id": CONFIG["run_id"],
        "iter": CONFIG["iteration"],
        "score_source": score_source,
        "metric": CONFIG["metric"],
        "direction": CONFIG["direction"],
        "offline_value": float(score),
        "offline_std": float(std) if std is not None else None,
        "folds": CONFIG["cv_folds"] if score_source == "cv" else None,
        "holdout_frac": CONFIG["holdout_frac"] if score_source == "holdout" else None,
        "seed": CONFIG["seed"],
        "target_score": None,
        "met_target": False,
        "top1_public_score": None,
        "top1_public_timestamp": None,
        "compare_to_top1_note": "heuristic; not directly comparable",
        "compute": "kaggle_{accelerator}",
        "accelerator": "{accelerator}",
        "git_commit": None,
        "timestamp": int(datetime.utcnow().timestamp()),
    }}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
'''
