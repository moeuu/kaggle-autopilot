from __future__ import annotations

import shutil
from pathlib import Path


def ensure_local_sample_submission_file(*, base_dir: Path, slug: str) -> Path | None:
    """Ensure data/sample_submission.csv exists and expand tiny placeholder templates."""
    competition_dir = base_dir / slug
    data_dir = competition_dir / "data"
    canonical_path = data_dir / "sample_submission.csv"
    if canonical_path.exists():
        expand_placeholder_sample_submission(canonical_path=canonical_path, data_dir=data_dir)
        return canonical_path
    source_path = resolve_sample_submission_source(
        context_dir=competition_dir / "context",
        data_dir=data_dir,
    )
    if source_path is None:
        return None
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, canonical_path)
    expand_placeholder_sample_submission(canonical_path=canonical_path, data_dir=data_dir)
    return canonical_path


def resolve_sample_submission_source(*, context_dir: Path, data_dir: Path) -> Path | None:
    context_sample = context_dir / "sample_submission.csv"
    if context_sample.exists():
        return context_sample
    if not data_dir.exists():
        return None
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        if "sample_submission" not in name:
            continue
        if path.suffix.lower() != ".csv":
            continue
        return path
    return None


def expand_placeholder_sample_submission(*, canonical_path: Path, data_dir: Path) -> None:
    """Expand tiny sample_submission templates to full test ids when confidently detected."""
    try:
        import pandas as pd
    except Exception:
        return

    test_path = data_dir / "test.csv"
    if not canonical_path.exists() or not test_path.exists():
        return
    try:
        sample = pd.read_csv(canonical_path)
    except Exception:
        return
    if sample.empty or len(sample.columns) < 2:
        return

    id_col = str(sample.columns[0])
    pred_cols = [str(col) for col in sample.columns if str(col) != id_col]
    if not pred_cols:
        return
    if len(sample) > 10 or sample[id_col].duplicated().any():
        return

    try:
        test = pd.read_csv(test_path, usecols=[id_col], dtype={id_col: str})
    except Exception:
        return
    if id_col not in test.columns:
        return

    test_ids = test[id_col].astype(str).tolist()
    sample_ids = sample[id_col].astype(str).tolist()
    if len(test_ids) <= max(len(sample_ids) * 3, len(sample_ids) + 10):
        return
    if sample_ids and test_ids[: len(sample_ids)] != sample_ids:
        return

    defaults = placeholder_prediction_defaults(
        sample=sample,
        data_dir=data_dir,
        id_col=id_col,
        prediction_columns=pred_cols,
    )
    expanded = pd.DataFrame({id_col: test_ids})
    for col in pred_cols:
        expanded[col] = defaults.get(col, 0.0)
    canonical_columns = [str(col) for col in sample.columns]
    for col in canonical_columns:
        if col not in expanded.columns:
            expanded[col] = ""
    expanded = expanded[canonical_columns]
    expanded.to_csv(canonical_path, index=False)


def placeholder_prediction_defaults(
    *,
    sample,
    data_dir: Path,
    id_col: str,
    prediction_columns: list[str],
) -> dict[str, float]:
    """Estimate stable default values for expanded placeholder prediction columns."""
    try:
        import pandas as pd
    except Exception:
        return {col: 0.0 for col in prediction_columns}

    defaults: dict[str, float] = {}
    for col in prediction_columns:
        sample_series = pd.to_numeric(sample[col], errors="coerce").dropna()
        defaults[col] = float(sample_series.mean()) if not sample_series.empty else 0.0

    train_path = data_dir / "train.csv"
    if not train_path.exists():
        return defaults
    train_cols = [col for col in prediction_columns if col != id_col]
    if not train_cols:
        return defaults
    try:
        train = pd.read_csv(train_path, usecols=train_cols)
    except Exception:
        return defaults
    for col in train_cols:
        if col not in train.columns:
            continue
        train_series = pd.to_numeric(train[col], errors="coerce").dropna()
        if not train_series.empty:
            defaults[col] = float(train_series.mean())
    return defaults
