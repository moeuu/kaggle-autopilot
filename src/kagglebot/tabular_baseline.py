from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from rich import print
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from kagglebot.paths import CompetitionPaths, repo_root


@dataclass(frozen=True)
class RunOutputs:
    sample_submission: str
    submission: str
    model_path: str


def _find_required(paths: CompetitionPaths) -> tuple[Path, Path, Path]:
    raw = paths.data_raw
    sample = raw / "sample_submission.csv"
    train = raw / "train.csv"
    test = raw / "test.csv"
    if not sample.exists():
        raise FileNotFoundError(f"Missing {sample}")
    if not train.exists():
        raise FileNotFoundError(f"Missing {train}")
    if not test.exists():
        raise FileNotFoundError(f"Missing {test}")
    return train, test, sample


def _choose_model(y: pd.Series):
    # Rough heuristic (MVP).
    # - Few unique numeric values -> classification, otherwise regression.
    nunique = y.nunique(dropna=True)
    if y.dtype == "object":
        # String labels -> classification.
        return LogisticRegression(max_iter=2000)
    if nunique <= 20:
        return LogisticRegression(max_iter=2000)
    return Ridge()


def train_and_make_submission(slug: str) -> RunOutputs:
    paths = CompetitionPaths(slug=slug, repo_root=repo_root())
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.submissions_dir.mkdir(parents=True, exist_ok=True)

    train_path, test_path, sample_path = _find_required(paths)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)

    # Infer target column from sample_submission (non-id).
    if sample.shape[1] < 2:
        raise ValueError("sample_submission.csv must have at least 2 columns (id + target).")

    id_col = sample.columns[0]
    target_cols = list(sample.columns[1:])
    if len(target_cols) != 1:
        raise NotImplementedError("MVP supports only single-target competitions for now.")

    target = target_cols[0]

    if target not in train_df.columns:
        raise ValueError(f"Target column '{target}' not found in train.csv columns.")

    # Features.
    x = train_df.drop(columns=[target])
    y = train_df[target]

    # Align columns so the same preprocessing can be applied to test.
    # Keep ID column if needed, but avoid using it for training.
    if id_col in x.columns:
        x = x.drop(columns=[id_col])
    x_test = test_df.copy()
    if id_col in x_test.columns:
        x_test_noid = x_test.drop(columns=[id_col])
    else:
        x_test_noid = x_test

    cat_cols = [c for c in x.columns if x[c].dtype == "object"]
    num_cols = [c for c in x.columns if c not in cat_cols]

    pre = ColumnTransformer(
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

    model = _choose_model(y)

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    # MVP: small holdout to sanity-check training.
    x_tr, x_va, y_tr, y_va = train_test_split(x, y, test_size=0.2, random_state=42)
    pipe.fit(x_tr, y_tr)
    score = pipe.score(x_va, y_va)
    print(f"[cyan]quick validation score[/cyan]: {score:.4f}")

    # Predict.
    preds = pipe.predict(x_test_noid)

    # Build submission file (match sample).
    submission = sample.copy()
    if id_col in test_df.columns:
        # Align by id column in case row order differs.
        submission[id_col] = test_df[id_col].values
    submission[target] = preds

    submission_path = paths.submission_csv
    submission.to_csv(submission_path, index=False)

    model_path = paths.models_dir / "baseline.joblib"
    joblib.dump(pipe, model_path)

    return RunOutputs(
        sample_submission=str(sample_path),
        submission=str(submission_path),
        model_path=str(model_path),
    )
