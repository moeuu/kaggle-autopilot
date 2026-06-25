from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from kagglebot.submission_templates import build_submission_template_for_test

_SAMPLE_STAGE_PATTERN = re.compile(r"(?:stage|phase|round)[_-]?(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class CompetitionData:
    train: pd.DataFrame
    test: pd.DataFrame
    sample: pd.DataFrame
    id_column: str | None
    target_column: str
    feature_columns: list[str]
    task: str
    prediction_kind: str
    target_columns: list[str] = field(default_factory=list)
    task_by_target: dict[str, str] = field(default_factory=dict)
    prediction_kind_by_target: dict[str, str] = field(default_factory=dict)
    data_dir: Path | None = None


def find_competition_files(data_dir: Path) -> tuple[Path, Path, Path]:
    files = _find_tabular_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No tabular files found under {data_dir}.")

    sample_path = _select_sample_submission_path(files)

    train_path = None
    test_path = None
    for path in files:
        name = path.name.lower()
        if "train" in name and train_path is None:
            train_path = path
        if "test" in name and test_path is None:
            test_path = path

    if sample_path is None:
        synthesized_sample = _maybe_synthesize_sample_submission(data_dir)
        if synthesized_sample is not None:
            sample_path = synthesized_sample
        else:
            raise FileNotFoundError("Unable to locate sample submission file in competition data.")
    if train_path is None or test_path is None:
        synthesized = _synthesize_train_test_from_assets(data_dir=data_dir, sample_path=sample_path)
        if synthesized is not None:
            return synthesized
        raise FileNotFoundError("Unable to locate train/test files in competition data.")

    return train_path, test_path, sample_path


def _select_sample_submission_path(files: Sequence[Path]) -> Path | None:
    """Pick the most plausible sample-submission file from discovered tabular files."""
    candidates = [path for path in files if _sample_name_score(path) > 0]
    if not candidates:
        return None
    usable = [path for path in candidates if _tabular_file_has_data_rows(path)]
    ranked = usable or candidates
    return max(ranked, key=_sample_candidate_key)


def _sample_candidate_key(path: Path) -> tuple[int, int, int, int, int, int, str]:
    """Return ranking key for sample-submission candidates."""
    name_score = _sample_name_score(path)
    stage_score = _sample_stage_score(path)
    desired_stage = _resolve_desired_submission_stage()
    stage_match = 1 if (desired_stage is not None and stage_score == desired_stage) else 0
    explicit_stage = 1 if stage_score > 0 else 0
    stage_preference = stage_score if stage_score > 0 else 0
    row_count = _tabular_data_row_count(path)
    desired_distance = 0
    if desired_stage is not None:
        desired_distance = -abs(stage_score - desired_stage) if stage_score else -10_000
    return (name_score, stage_match, explicit_stage, desired_distance, stage_preference, row_count, path.name.lower())


def _sample_name_score(path: Path) -> int:
    """Score how clearly a filename indicates a sample-submission file."""
    name = path.name.lower()
    compact = name.replace("_", "")
    if "sample_submission" in name or "samplesubmission" in compact:
        return 3
    if "sample" in name and "submission" in name:
        return 2
    if "submission" in name:
        return 1
    return 0


def _sample_stage_score(path: Path) -> int:
    """Extract stage/phase/round number from filename for ranking."""
    matches = _SAMPLE_STAGE_PATTERN.findall(path.name.lower())
    if not matches:
        return 0
    return max(int(value) for value in matches)


def _resolve_desired_submission_stage() -> int | None:
    raw = (
        os.environ.get("KAGGLEBOT_SUBMISSION_STAGE") or os.environ.get("KAGGLEBOT_SAMPLE_SUBMISSION_STAGE") or ""
    ).strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _tabular_file_has_data_rows(path: Path) -> bool:
    """Return whether a delimited tabular file includes at least one data row."""
    if path.suffix.lower() not in {".csv", ".tsv", ".txt"}:
        return True
    non_empty = 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                non_empty += 1
                if non_empty >= 2:
                    return True
    except OSError:
        return True
    return False


def _tabular_data_row_count(path: Path) -> int:
    """Return the number of non-empty data rows in a tabular file."""
    if path.suffix.lower() not in {".csv", ".tsv", ".txt"}:
        return 0
    non_empty = 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                non_empty += 1
    except OSError:
        return 0
    return max(0, non_empty - 1)


def ensure_sample_submission(data_dir: Path) -> Path | None:
    """
    Ensure a usable sample submission exists for this competition.

    Preference order:
    1) Use an existing sample-submission file shipped with the competition data
       (including multi-stage files like `SampleSubmissionStage1.csv`).
    2) If no usable sample file exists, try to synthesize one from
       `context/submission_format.md` plus discovered test IDs (e.g., filenames under
       `images/test`).
    """
    if not data_dir.exists():
        return None
    try:
        files = _find_tabular_files(data_dir)
    except OSError:
        files = []
    candidate = _select_sample_submission_path(files)
    if candidate is not None and _tabular_file_has_data_rows(candidate):
        return candidate
    return _maybe_synthesize_sample_submission(data_dir)


def infer_submission_layout(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
) -> tuple[str | None, list[str], list[str]]:
    sample_cols = list(sample.columns)
    train_cols = list(train.columns)
    test_cols = list(test.columns)
    train_minus_test = [col for col in train_cols if col not in test_cols]
    target_cols = _infer_target_columns(train=train, test=test, sample=sample, train_minus_test=train_minus_test)
    id_col = _pick_id_column(sample_cols=sample_cols, target_cols=target_cols, test_cols=test_cols)

    # Feature columns must be present in BOTH train and test; otherwise they cannot be
    # used for inference and will break downstream schema/validation logic.
    common_non_target = [col for col in train_cols if col in test_cols and col not in target_cols]
    feature_cols = list(common_non_target)

    if id_col and id_col in feature_cols:
        feature_cols.remove(id_col)
    if not feature_cols:
        # If removing the id column would leave no features, keep the common columns
        # (including id) as a last-ditch fallback.
        feature_cols = list(common_non_target)

    return id_col, target_cols, feature_cols


def infer_target(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> tuple[str, str, list[str]]:
    id_col, target_cols, feature_cols = infer_submission_layout(train=train, test=test, sample=sample)
    if not target_cols:
        raise ValueError("Unable to infer target columns from train/test/sample files.")
    resolved_id = id_col or ""
    return resolved_id, target_cols[0], feature_cols


def infer_task(y: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(y):
        return "classification"
    if pd.api.types.is_object_dtype(y) or isinstance(y.dtype, pd.CategoricalDtype):
        return "classification"
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    nunique = y.nunique(dropna=True)
    if nunique <= 20:
        return "classification"
    if nunique / max(len(y), 1) <= 0.05:
        return "classification"
    return "regression"


def infer_prediction_kind(sample_target: pd.Series) -> str:
    if pd.api.types.is_float_dtype(sample_target) or pd.api.types.is_complex_dtype(sample_target):
        return "probability"
    if pd.api.types.is_numeric_dtype(sample_target):
        values = pd.to_numeric(sample_target, errors="coerce").dropna().to_numpy()
        if values.size and np.isin(np.unique(values), np.array([0.0, 1.0])).all():
            return "class"
        return "continuous"
    return "class"


def load_competition_data(data_dir: Path, *, target_column_override: str | None = None) -> CompetitionData:
    train_path, test_path, sample_path = find_competition_files(data_dir)
    train = _read_table(train_path)
    test = _read_table(test_path)
    sample = _read_table(sample_path)

    if target_column_override and target_column_override in train.columns:
        id_col, inferred_targets, feature_cols = infer_submission_layout(train=train, test=test, sample=sample)
        target_cols = [target_column_override]
        if id_col and id_col in feature_cols:
            feature_cols = [c for c in feature_cols if c != id_col]
        if target_column_override not in inferred_targets:
            feature_cols = [c for c in train.columns if c != target_column_override and c != id_col]
    else:
        id_col, target_cols, feature_cols = infer_submission_layout(train=train, test=test, sample=sample)
    if not target_cols:
        raise ValueError("Unable to infer target columns from train/test/sample files.")

    target_col = target_cols[0]
    task_by_target = {col: infer_task(train[col]) for col in target_cols}
    prediction_kind_by_target = {
        col: infer_prediction_kind(sample[col]) if col in sample.columns else "continuous" for col in target_cols
    }
    task = task_by_target[target_col]
    prediction_kind = prediction_kind_by_target[target_col]

    return CompetitionData(
        train=train,
        test=test,
        sample=sample,
        id_column=id_col,
        target_column=target_col,
        target_columns=target_cols,
        feature_columns=feature_cols,
        task=task,
        prediction_kind=prediction_kind,
        task_by_target=task_by_target,
        prediction_kind_by_target=prediction_kind_by_target,
        data_dir=data_dir,
    )


def write_submission(
    sample: pd.DataFrame,
    test: pd.DataFrame,
    preds,
    *,
    id_column: str | None,
    target_column: str | None = None,
    target_columns: Sequence[str] | None = None,
    output_path: Path,
) -> Path:
    resolved_targets = _resolve_target_columns(
        sample=sample,
        id_column=id_column,
        target_column=target_column,
        target_columns=target_columns,
    )
    submission = build_submission_template_for_test(
        sample_submission=sample,
        test_df=test,
        id_col=id_column,
        target_cols=resolved_targets,
    )
    pred_table = _normalize_prediction_table(preds=preds, target_columns=resolved_targets, row_count=len(test))

    for col in resolved_targets:
        submission[col] = _align_prediction_column(
            sample=submission,
            test=test,
            values=pred_table[col],
            id_column=id_column,
            target_column=col,
        )
        submission[col] = _coerce_prediction_dtype(
            sample[col],
            submission[col],
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return output_path


def _find_tabular_files(root: Path) -> list[Path]:
    suffixes = {".csv", ".tsv", ".txt", ".parquet", ".json", ".jsonl"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _coerce_prediction_dtype(sample_series: pd.Series, pred_series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(sample_series.dtype):
        if pd.api.types.is_bool_dtype(pred_series.dtype):
            return pred_series
        if pd.api.types.is_numeric_dtype(pred_series.dtype):
            values = pred_series.dropna().to_numpy()
            if values.size == 0:
                return pred_series
            binary_mask = np.isclose(values, 0.0) | np.isclose(values, 1.0)
            if binary_mask.all():
                return pred_series.astype(bool)
            return pred_series
        lowered = pred_series.astype(str).str.lower()
        if set(lowered.dropna().unique()).issubset({"true", "false"}):
            return lowered == "true"
    return pred_series


def _infer_target_columns(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    train_minus_test: list[str],
) -> list[str]:
    sample_cols = list(sample.columns)
    # 1) Most reliable: in sample, in train, not in test.
    candidates = [col for col in sample_cols if col in train_minus_test and col in train.columns]
    if candidates:
        return candidates

    # 2) Any sample cols present in train and not obvious ID-like.
    aligned = [col for col in sample_cols if col in train.columns]
    filtered = [col for col in aligned if col not in test.columns]
    if filtered:
        return filtered
    if len(aligned) > 1:
        return aligned[1:]
    if aligned:
        return aligned

    # 3) Fallback to train-test diff.
    if train_minus_test:
        return train_minus_test
    return []


def _pick_id_column(*, sample_cols: list[str], target_cols: list[str], test_cols: list[str]) -> str | None:
    non_targets = [col for col in sample_cols if col not in target_cols]
    if not non_targets:
        return None
    test_overlap = [col for col in non_targets if col in test_cols]
    if test_overlap:
        return test_overlap[0]
    return non_targets[0]


def _synthesize_train_test_from_assets(data_dir: Path, sample_path: Path) -> tuple[Path, Path, Path] | None:
    label_files = [
        p
        for p in data_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".csv", ".tsv", ".txt", ".parquet", ".json", ".jsonl"}
        and any(token in p.name.lower() for token in ("label", "labels", "train_labels", "training_labels"))
    ]
    if not label_files:
        return None
    labels = _read_table(label_files[0])
    if labels.empty:
        return None

    sample = _read_table(sample_path)
    if sample.empty:
        return None
    sample_id = sample.columns[0]
    label_id = sample_id if sample_id in labels.columns else labels.columns[0]
    if label_id not in labels.columns:
        return None

    id_to_path = _discover_asset_paths(data_dir)
    if not id_to_path:
        return None

    train_ids = labels[label_id].astype(str)
    test_ids = sample[sample_id].astype(str)
    train = labels.copy()
    train[label_id] = train_ids
    train["asset_path"] = train_ids.map(id_to_path)
    test = pd.DataFrame({sample_id: test_ids})
    test["asset_path"] = test_ids.map(id_to_path)
    train = train[train["asset_path"].notna()].reset_index(drop=True)
    test = test[test["asset_path"].notna()].reset_index(drop=True)
    if train.empty or test.empty:
        return None

    cache_dir = data_dir / ".kagglebot_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_out = cache_dir / "train_synth.csv"
    test_out = cache_dir / "test_synth.csv"
    train.to_csv(train_out, index=False)
    test.to_csv(test_out, index=False)
    return train_out, test_out, sample_path


def _discover_asset_paths(data_dir: Path) -> dict[str, str]:
    suffixes = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".gif",
        ".tif",
        ".tiff",
        ".webp",
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
        ".aac",
    }
    mapping: dict[str, str] = {}
    candidates = [path for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() in suffixes]
    for path in sorted(candidates, key=lambda item: _asset_priority_key(data_dir=data_dir, path=item)):
        stem = path.stem
        mapping.setdefault(stem, str(path))
        mapping.setdefault(path.name, str(path))
    return mapping


def _asset_priority_key(*, data_dir: Path, path: Path) -> tuple[int, str]:
    rel_parts = [part.lower() for part in path.relative_to(data_dir).parts]
    is_images_test = len(rel_parts) >= 3 and rel_parts[0] == "images" and rel_parts[1] == "test"
    is_images_train = len(rel_parts) >= 3 and rel_parts[0] == "images" and rel_parts[1] == "train"
    contains_test = "test" in rel_parts
    contains_train = "train" in rel_parts

    if is_images_test:
        rank = 0
    elif contains_test:
        rank = 1
    elif is_images_train:
        rank = 2
    elif contains_train:
        rank = 3
    else:
        rank = 4
    return rank, str(path)


def _maybe_synthesize_sample_submission(data_dir: Path) -> Path | None:
    for context_dir in _candidate_context_dirs(data_dir):
        candidate = context_dir / "sample_submission.csv"
        usable = _is_usable_sample_submission(candidate)
        if usable is not None:
            return usable
        format_path = context_dir / "submission_format.md"
        synthesized = _synthesize_sample_from_submission_format(data_dir=data_dir, submission_format_path=format_path)
        if synthesized is not None:
            return synthesized
    return None


def _candidate_context_dirs(data_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    direct = data_dir.parent / "context"
    if direct not in seen:
        candidates.append(direct)
        seen.add(direct)

    for parent in [data_dir, *data_dir.parents]:
        candidate = parent / "context"
        if candidate in seen:
            continue
        candidates.append(candidate)
        seen.add(candidate)
    return candidates


def _is_usable_sample_submission(path: Path) -> Path | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        frame = pd.read_csv(path, nrows=1)
    except Exception:  # noqa: BLE001
        return None
    if frame.empty or len(frame.columns) < 2:
        return None
    return path


def _synthesize_sample_from_submission_format(*, data_dir: Path, submission_format_path: Path) -> Path | None:
    if not submission_format_path.exists() or not submission_format_path.is_file():
        return None
    header = _extract_submission_header(submission_format_path)
    if not header:
        return None
    id_column = header[0]
    target_columns = header[1:]
    if not target_columns:
        return None

    test_ids = _discover_test_ids(data_dir, id_column=id_column)
    if not test_ids:
        return None

    payload: dict[str, list[object]] = {id_column: test_ids}
    for col in target_columns:
        lowered = col.lower()
        if "prediction" in lowered and "string" in lowered:
            payload[col] = ["-"] * len(test_ids)
        else:
            payload[col] = [0] * len(test_ids)
    frame = pd.DataFrame(payload)

    cache_dir = data_dir / ".kagglebot_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / "sample_submission_synth.csv"
    frame.to_csv(out, index=False)
    return out


def _extract_submission_header(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    in_block = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_block = not in_block
            continue
        if not in_block:
            continue
        if "," not in line:
            continue
        cols = [part.strip() for part in line.split(",") if part.strip()]
        if len(cols) >= 2:
            return cols
    return []


def _discover_test_ids(data_dir: Path, *, id_column: str) -> list[str]:
    _ = id_column
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
    preferred = data_dir / "images" / "test"
    if preferred.exists():
        images = sorted([p for p in preferred.iterdir() if p.is_file() and p.suffix.lower() in image_suffixes])
    else:
        images = []
        for path in data_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in image_suffixes:
                continue
            if any(part.lower() == "test" for part in path.parts):
                images.append(path)
        images = sorted(images)
    return [path.name for path in images]


def _resolve_target_columns(
    *,
    sample: pd.DataFrame,
    id_column: str | None,
    target_column: str | None,
    target_columns: Sequence[str] | None,
) -> list[str]:
    if target_columns is not None:
        resolved = [str(col) for col in target_columns if str(col).strip()]
    elif target_column is not None:
        resolved = [target_column]
    else:
        resolved = [col for col in sample.columns if col != id_column]
    if not resolved:
        raise ValueError("No target columns resolved for submission writing.")

    missing = [col for col in resolved if col not in sample.columns]
    if missing:
        raise ValueError(f"Target columns not found in sample submission: {missing}")
    return resolved


def _normalize_prediction_table(
    *,
    preds,
    target_columns: list[str],
    row_count: int,
) -> dict[str, np.ndarray]:
    if isinstance(preds, Mapping):
        normalized: dict[str, np.ndarray] = {}
        for col in target_columns:
            if col not in preds:
                raise ValueError(f"Missing predictions for target column '{col}'.")
            values = np.asarray(preds[col]).ravel()
            if len(values) != row_count:
                raise ValueError(f"Prediction length mismatch for '{col}': expected {row_count}, got {len(values)}.")
            normalized[col] = values
        return normalized

    pred_array = np.asarray(preds)
    if pred_array.ndim == 1:
        if len(target_columns) != 1:
            raise ValueError("1D predictions provided for multi-target submission.")
        if len(pred_array) != row_count:
            raise ValueError(f"Prediction length mismatch: expected {row_count}, got {len(pred_array)}.")
        return {target_columns[0]: pred_array.ravel()}

    if pred_array.ndim == 2:
        if pred_array.shape[0] != row_count:
            raise ValueError(f"Prediction row count mismatch: expected {row_count}, got {pred_array.shape[0]}.")
        if pred_array.shape[1] != len(target_columns):
            raise ValueError(
                f"Prediction column count mismatch: expected {len(target_columns)}, got {pred_array.shape[1]}."
            )
        return {col: pred_array[:, idx] for idx, col in enumerate(target_columns)}

    raise ValueError("Unsupported predictions shape for submission writing.")


def _align_prediction_column(
    *,
    sample: pd.DataFrame,
    test: pd.DataFrame,
    values: np.ndarray,
    id_column: str | None,
    target_column: str,
) -> pd.Series:
    if id_column and id_column in test.columns and id_column in sample.columns:
        if not test[id_column].duplicated().any() and not sample[id_column].duplicated().any():
            pred_map = pd.Series(values, index=test[id_column])
            aligned = sample[id_column].map(pred_map)
            if aligned.isna().any():
                raise ValueError(
                    f"Missing predictions after aligning by id column '{id_column}' for target '{target_column}'."
                )
            return aligned
    if len(values) != len(sample):
        raise ValueError(
            f"Prediction length does not match submission rows for target '{target_column}': "
            f"expected {len(sample)}, got {len(values)}."
        )
    return pd.Series(values, index=sample.index)
