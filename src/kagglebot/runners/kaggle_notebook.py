from __future__ import annotations

import json
import os
import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

from rich import print

from kagglebot import kaggle_cli
from kagglebot.kernel_sources import KernelSourceConfig, load_kernel_source_config
from kagglebot.runners.base import RunContext, RunResult
from kagglebot.submission_artifacts import find_submission_manifest, resolve_manifest_references

KERNEL_TEMPLATE = r"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

COMPETITION_SLUG = "__COMPETITION_SLUG__"
ACCELERATOR = "__ACCELERATOR__"
INPUT_ROOT = Path("/kaggle/input") / COMPETITION_SLUG
WORKING_DIR = Path("/kaggle/working")
SUBMISSION_PATH = WORKING_DIR / "submission.csv"
METRICS_PATH = WORKING_DIR / "metrics.json"


def find_tabular_files(root: Path) -> list[Path]:
    suffixes = {".csv", ".tsv", ".txt", ".parquet", ".json", ".jsonl"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\\t")
    return pd.read_csv(path)


def pick_files(files: list[Path]) -> tuple[Path, Path, Path]:
    if not files:
        raise FileNotFoundError(f"No tabular files found under {INPUT_ROOT}.")

    def score_sample(path: Path) -> int:
        name = path.name.lower()
        if "sample_submission" in name:
            return 3
        if "sample" in name and "submission" in name:
            return 2
        if "submission" in name:
            return 1
        return 0

    sample_candidates = sorted(files, key=score_sample, reverse=True)
    sample_path = sample_candidates[0] if score_sample(sample_candidates[0]) > 0 else None

    train_path = None
    test_path = None
    for path in files:
        name = path.name.lower()
        if "train" in name and train_path is None:
            train_path = path
        if "test" in name and test_path is None:
            test_path = path

    if train_path is None or test_path is None:
        raise FileNotFoundError("Unable to locate train/test files in competition data.")
    if sample_path is None:
        raise FileNotFoundError("Unable to locate sample submission file in competition data.")

    return train_path, test_path, sample_path


def infer_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
) -> tuple[str | None, str, list[str], list[str]]:
    sample_cols = list(sample.columns)
    train_minus_test = [c for c in train.columns if c not in test.columns]
    target_cols = [c for c in sample_cols if c in train_minus_test and c in train.columns]
    if not target_cols:
        target_cols = [c for c in sample_cols if c in train.columns and c not in test.columns]
    if not target_cols:
        target_cols = [c for c in sample_cols[1:] if c in train.columns]
    if not target_cols and train_minus_test:
        target_cols = train_minus_test
    if not target_cols:
        raise ValueError("Unable to infer target columns from train/test/sample files.")

    non_targets = [c for c in sample_cols if c not in target_cols]
    id_col = next((c for c in non_targets if c in test.columns), None)
    if id_col is None and non_targets:
        id_col = non_targets[0]

    target_col = target_cols[0]
    feature_cols = [c for c in train.columns if c not in target_cols and c != id_col]
    return id_col, target_col, feature_cols, target_cols


def infer_task(y: pd.Series) -> str:
    if y.dtype == "object":
        return "classification"
    nunique = y.nunique(dropna=True)
    if nunique <= 20:
        return "classification"
    if nunique / max(len(y), 1) <= 0.05:
        return "classification"
    return "regression"


def build_preprocessor(feature_cols: list[str], train: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in feature_cols if train[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", ohe),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


def build_sklearn_model(task: str, preprocessor: ColumnTransformer) -> Pipeline:
    estimator = LogisticRegression(max_iter=2000) if task == "classification" else Ridge()
    return Pipeline([("pre", preprocessor), ("model", estimator)])


def predict_sklearn(model, x, task: str, prediction_kind: str):
    if task == "classification" and prediction_kind == "probability" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.shape[1] == 1:
            return proba[:, 0]
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba.max(axis=1)
    return model.predict(x)


def train_torch_mlp(x, y, task: str, num_classes: int, device: str):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(np.asarray(y))
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


def predict_torch(model, x, task: str, num_classes: int, prediction_kind: str, device: str):
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


def train_tpu_model(x, y, task: str, num_classes: int):
    import tensorflow as tf

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)

    with strategy.scope():
        output_units = num_classes if task == "classification" and num_classes > 2 else 1
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(x.shape[1],)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(output_units),
            ]
        )

        if task == "classification":
            if num_classes > 2:
                model.add(tf.keras.layers.Activation("softmax"))
                model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            else:
                model.add(tf.keras.layers.Activation("sigmoid"))
                model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        else:
            model.compile(optimizer="adam", loss="mse")

    model.fit(x, y, batch_size=256, epochs=10, verbose=0)
    return model


def predict_tpu(model, x, task: str, num_classes: int, prediction_kind: str):
    outputs = model.predict(x, batch_size=256, verbose=0)
    if task == "classification":
        if num_classes > 2:
            probs = outputs
            if prediction_kind == "probability":
                return probs.max(axis=1)
            return probs.argmax(axis=1)
        probs = outputs.ravel()
        if prediction_kind == "probability":
            return probs
        return (probs >= 0.5).astype(int)
    return outputs.ravel()


def main() -> None:
    print(f"competition slug: {COMPETITION_SLUG}")
    files = find_tabular_files(INPUT_ROOT)
    train_path, test_path, sample_path = pick_files(files)
    print(f"train: {train_path}")
    print(f"test: {test_path}")
    print(f"sample: {sample_path}")

    train = read_table(train_path)
    test = read_table(test_path)
    sample = read_table(sample_path)

    id_col, target_col, feature_cols, target_cols = infer_target(train, test, sample)
    print(f"id column: {id_col}")
    print(f"target column: {target_col}")
    if len(target_cols) > 1:
        print(f"multi-target detected; using primary target for baseline: {target_col} ({target_cols})")
    print(f"feature count: {len(feature_cols)}")

    x = train[feature_cols]
    y = train[target_col]
    task = infer_task(y)
    prediction_kind = "probability" if pd.api.types.is_float_dtype(sample[target_col]) else "class"

    label_encoder = None
    num_classes = 0
    y_encoded = y
    if task == "classification":
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        num_classes = len(label_encoder.classes_)

    stratify = y_encoded if task == "classification" else None
    x_train_raw, x_valid_raw, y_train, y_valid = train_test_split(
        x,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    preprocessor = build_preprocessor(feature_cols, train)
    x_train = preprocessor.fit_transform(x_train_raw)
    x_valid = preprocessor.transform(x_valid_raw)
    x_test_processed = preprocessor.transform(test[feature_cols])
    if hasattr(x_train, "toarray"):
        x_train = x_train.toarray()
        x_valid = x_valid.toarray()
        x_test_processed = x_test_processed.toarray()

    model_kind = "sklearn"
    torch_device = "cpu"

    if ACCELERATOR == "tpu":
        try:
            model = train_tpu_model(x_train, y_train, task, num_classes)
            preds_valid = predict_tpu(model, x_valid, task, num_classes, "class")
            model_kind = "tpu"
        except Exception as exc:
            raise RuntimeError(f"TPU initialization failed: {exc}") from exc
    elif ACCELERATOR == "gpu":
        try:
            import torch

            torch_device = "cuda" if torch.cuda.is_available() else "cpu"
            if torch_device == "cpu":
                print("GPU requested but CUDA not available; falling back to CPU.")
            model = train_torch_mlp(x_train, y_train, task, num_classes, torch_device)
            preds_valid = predict_torch(model, x_valid, task, num_classes, "class", torch_device)
            model_kind = "torch"
        except Exception as exc:
            print(f"GPU training failed, falling back to sklearn: {exc}")
            model = build_sklearn_model(task, preprocessor)
            model.fit(x_train_raw, y_train)
            preds_valid = predict_sklearn(model, x_valid_raw, task, "class")
            model_kind = "sklearn"
    else:
        model = build_sklearn_model(task, preprocessor)
        model.fit(x_train_raw, y_train)
        preds_valid = predict_sklearn(model, x_valid_raw, task, "class")
        model_kind = "sklearn"

    if task == "classification":
        metric = "accuracy"
        preds_eval = np.asarray(preds_valid)
        if preds_eval.ndim > 1:
            preds_eval = preds_eval.argmax(axis=1)
        score = accuracy_score(y_valid, preds_eval)
    else:
        metric = "rmse"
        score = mean_squared_error(y_valid, preds_valid, squared=False)
    print(f"validation {metric}: {score:.4f}")

    if ACCELERATOR == "tpu":
        preds = predict_tpu(model, x_test_processed, task, num_classes, prediction_kind)
    elif ACCELERATOR == "gpu":
        if model_kind == "torch":
            preds = predict_torch(model, x_test_processed, task, num_classes, prediction_kind, torch_device)
        else:
            preds = predict_sklearn(model, test[feature_cols], task, prediction_kind)
    else:
        preds = predict_sklearn(model, test[feature_cols], task, prediction_kind)

    if (
        label_encoder is not None
        and prediction_kind == "class"
        and not pd.api.types.is_numeric_dtype(sample[target_col])
        and not pd.api.types.is_bool_dtype(sample[target_col])
    ):
        preds = label_encoder.inverse_transform(np.asarray(preds, dtype=int))

    submission = sample.copy()
    if id_col and id_col in test.columns and id_col in submission.columns:
        mapping = pd.Series(preds, index=test[id_col])
        submission[target_col] = submission[id_col].map(mapping)
    else:
        submission[target_col] = preds
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"wrote submission: {SUBMISSION_PATH}")

    payload = {
        "task": task,
        "metric": metric,
        "score": float(score),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    try:
        METRICS_PATH.write_text(json.dumps(payload, indent=2))
    except Exception as exc:
        print(f"failed to write metrics.json: {exc}")


if __name__ == "__main__":
    main()
""".strip()


class KaggleNotebookRunner:
    name = "kaggle_notebook"

    def run(self, context: RunContext) -> RunResult:
        slug = context.slug
        run_id = context.run_id
        paths = context.paths

        run_dir = paths.run_dir(run_id)
        kernel_dir = run_dir / "kernel"
        output_dir = run_dir / "output"
        logs_dir = run_dir / "logs"
        summary_path = run_dir / "summary.json"

        kernel_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        kaggle_username = resolve_kaggle_username(context.kaggle_username)
        kernel_slug = build_kernel_slug(slug, run_id)
        kernel_id = f"{kaggle_username}/{kernel_slug}"

        accelerator = context.accelerator

        metadata = build_kernel_metadata(
            kaggle_username=kaggle_username,
            kernel_slug=kernel_slug,
            title=kernel_slug.replace("-", " "),
            competition_slug=slug,
            accelerator=accelerator,
            enable_internet=context.enable_internet,
            source_config=load_kernel_source_config(paths.plan_path),
        )
        kernel_metadata_path = kernel_dir / "kernel-metadata.json"
        kernel_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        kernel_main_path = kernel_dir / "main.py"
        kernel_main_path.write_text(render_kernel_main(slug, accelerator), encoding="utf-8")

        commands = [
            f"kaggle kernels push -p {kernel_dir}",
            f"kaggle kernels status {kernel_id}",
            f"kaggle kernels output {kernel_id} -p {output_dir}",
        ]

        summary = {
            "schema_version": 1,
            "slug": slug,
            "run_id": run_id,
            "runner": self.name,
            "kernel_slug": kernel_slug,
            "kernel_id": kernel_id,
            "accelerator": accelerator,
            "enable_internet": context.enable_internet,
            "dry_run": context.dry_run,
            "generated_at": datetime.now(UTC).isoformat(),
            "commands": commands,
        }

        if context.dry_run:
            print("[yellow]DRY RUN[/yellow]: Kaggle CLI commands will not be executed.")
            for command in commands:
                print(f"[cyan]planned[/cyan]: {command}")
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return RunResult(
                run_id=run_id,
                runner=self.name,
                submission_path=None,
                summary_path=summary_path,
                analysis_path=None,
                kernel_slug=kernel_slug,
            )

        print(f"[cyan]checking competition access[/cyan]: {slug}")
        kaggle_cli.competitions_files(slug)
        print(f"[cyan]pushing kernel[/cyan]: {kernel_dir}")
        kaggle_cli.kernels_push(kernel_dir, slug=slug, stream_output=True)
        print(f"[cyan]waiting for kernel[/cyan]: {kernel_id}")
        _wait_for_kernel(kernel_id, logs_dir=logs_dir, slug=slug, kernel_dir=kernel_dir)
        print(f"[cyan]downloading kernel output[/cyan]: {output_dir}")
        kaggle_cli.kernels_output(kernel_id, output_dir, slug=slug, stream_output=True, force=True)

        submission_path = find_submission_file(output_dir)
        paths.submissions_dir.mkdir(parents=True, exist_ok=True)
        local_submission = paths.submissions_dir / f"{run_id}_{submission_path.name}"
        shutil.copy2(submission_path, local_submission)

        summary["submission_path"] = str(local_submission)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return RunResult(
            run_id=run_id,
            runner=self.name,
            submission_path=local_submission,
            summary_path=summary_path,
            analysis_path=None,
            kernel_slug=kernel_slug,
        )


def sanitize_kernel_slug(value: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", value.lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    if not slug:
        raise ValueError("Kernel slug is empty after sanitization.")
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("-")
    return slug


def build_kernel_slug(competition_slug: str, run_id: str, *, max_length: int = 50) -> str:
    short_id = run_id.rsplit("-", 1)[-1]
    base_slug = sanitize_kernel_slug(competition_slug, max_length=200)
    base = f"kb-{base_slug}-{short_id}"
    if len(base) <= max_length:
        return base
    reserved = len("kb--") + len(short_id)
    room = max_length - reserved
    if room < 1:
        raise ValueError("Kernel slug length budget is too small.")
    trimmed_slug = base_slug[:room].rstrip("-")
    if not trimmed_slug:
        raise ValueError("Kernel slug is empty after trimming.")
    return f"kb-{trimmed_slug}-{short_id}"


def build_kernel_metadata(
    *,
    kaggle_username: str,
    kernel_slug: str,
    title: str,
    competition_slug: str,
    accelerator: str,
    enable_internet: bool,
    source_config: KernelSourceConfig | None = None,
) -> dict[str, object]:
    enable_gpu = accelerator == "gpu"
    enable_tpu = accelerator == "tpu"
    if enable_gpu and enable_tpu:
        raise ValueError("enable_gpu and enable_tpu cannot both be true.")
    source_config = source_config or KernelSourceConfig()
    return {
        "id": f"{kaggle_username}/{kernel_slug}",
        "title": title,
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_tpu": enable_tpu,
        "enable_internet": enable_internet,
        "competition_sources": [competition_slug],
        "dataset_sources": list(source_config.dataset_sources),
        "kernel_sources": list(source_config.kernel_sources),
        "model_sources": list(source_config.model_sources),
        "keywords": [],
    }


def render_kernel_main(competition_slug: str, accelerator: str) -> str:
    return (
        KERNEL_TEMPLATE.replace("__COMPETITION_SLUG__", competition_slug)
        .replace("__ACCELERATOR__", accelerator)
        .strip()
    )


def resolve_kaggle_username(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.getenv("KAGGLE_USERNAME")
    if env:
        return env
    config_dir = Path(os.getenv("KAGGLE_CONFIG_DIR", "~/.kaggle")).expanduser()
    config_path = config_dir / "kaggle.json"
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        username = payload.get("username")
        if username:
            return str(username)
    raise ValueError(
        "Kaggle username is required for kaggle_* compute modes. "
        "Set --kaggle-username, KAGGLE_USERNAME, or ~/.kaggle/kaggle.json."
    )


def find_submission_file(output_dir: Path) -> Path:
    manifest_path = find_submission_manifest(output_dir)
    if manifest_path is not None:
        _, submission_path, staging_dir, members = resolve_manifest_references(manifest_path)
        if submission_path is not None and submission_path.exists() and submission_path.is_file():
            return submission_path
        if staging_dir is not None or members:
            return manifest_path
    matches = [path for path in output_dir.rglob("submission.csv") if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"No submission.csv found under {output_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple submission.csv files found under {output_dir}")
    return matches[0]


def _wait_for_kernel(kernel_id: str, *, logs_dir: Path, slug: str, kernel_dir: Path) -> None:
    timeout_seconds = 60 * 60
    poll_seconds = 15
    start = time.monotonic()
    status_log = logs_dir / "kernel_status.log"
    last_status = None

    while True:
        output = kaggle_cli.kernels_status(kernel_id, slug=slug)
        status = _parse_kernel_status(output)
        if status != last_status:
            print(f"[cyan]kernel status[/cyan]: {status}")
            last_status = status
        status_log.write_text(output, encoding="utf-8")
        if status == "complete":
            return
        if status == "failed":
            _stop_failed_kernel_run(
                kernel_id, kernel_dir=kernel_dir, logs_dir=logs_dir, slug=slug, status_output=output
            )
            raise RuntimeError(f"Kernel run failed: {output}")
        if time.monotonic() - start > timeout_seconds:
            raise TimeoutError("Timed out waiting for kernel completion.")
        time.sleep(poll_seconds)


def _parse_kernel_status(output: str) -> str:
    text = output.lower()
    if (
        "failure message" in text
        or "your notebook failed" in text
        or "kernelworkerstatus.error" in text
        or "kernelworkerstatus.failed" in text
        or 'status "error"' in text
        or 'status "failed"' in text
        or " failed" in text
    ):
        return "failed"
    if "complete" in text or "success" in text:
        return "complete"
    if "running" in text or "queued" in text or "pending" in text:
        return "running"
    return "unknown"


def _stop_failed_kernel_run(
    kernel_id: str,
    *,
    kernel_dir: Path,
    logs_dir: Path,
    slug: str,
    status_output: str,
) -> None:
    if os.getenv("KAGGLEBOT_STOP_FAILED_KERNEL", "1").strip().lower() in {"0", "false", "no", "off"}:
        return

    stop_dir = logs_dir.parent / "kernel-stop"
    stop_log = logs_dir / "kernel_stop.log"
    try:
        metadata_path = kernel_dir / "kernel-metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["id"] = kernel_id
        metadata["enable_gpu"] = False
        metadata["enable_tpu"] = False
        metadata["enable_internet"] = False
        metadata["code_file"] = "main.py"
        stop_dir.mkdir(parents=True, exist_ok=True)
        (stop_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (stop_dir / "main.py").write_text(
            "\n".join(
                [
                    "from pathlib import Path",
                    "Path('/kaggle/working/kagglebot_stopped_failed_gpu_kernel.txt').write_text(",
                    "    'KaggleBot replaced a failed GPU run with a CPU stop marker.\\n',",
                    "    encoding='utf-8',",
                    ")",
                    "print('KaggleBot stop marker completed.')",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"[yellow]kernel failed[/yellow]: pushing CPU stop marker for {kernel_id}")
        output = kaggle_cli.kernels_push(stop_dir, slug=slug, stream_output=True)
        stop_log.write_text(
            f"status_output:\n{status_output}\n\nstop_marker_output:\n{output}\n",
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        stop_log.write_text(
            f"status_output:\n{status_output}\n\nstop_marker_failed:\n{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        print(f"[yellow]kernel stop marker failed[/yellow]: {type(exc).__name__}: {exc}")
