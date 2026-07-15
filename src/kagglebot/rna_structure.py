from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from kagglebot.solver.io import read_table, write_table
from kagglebot.submission_sample_discovery import is_tabular_data_path, sample_candidate_key, sample_name_score

_COORD_COL_RE = re.compile(r"^(?P<axis>[xyz])_(?P<copy>\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class CoordinateTriplet:
    copy_index: int
    x_col: str
    y_col: str
    z_col: str


@dataclass(frozen=True)
class RnaStructureFiles:
    train_sequences_path: Path
    test_sequences_path: Path
    train_labels_path: Path
    sample_submission_path: Path


@dataclass(frozen=True)
class RnaStructureTask:
    files: RnaStructureFiles
    train_sequences: pd.DataFrame
    test_sequences: pd.DataFrame
    train_labels: pd.DataFrame
    sample_submission: pd.DataFrame
    sequence_id_column: str
    sequence_column: str
    sample_id_column: str
    label_id_column: str
    sample_anchor_columns: list[str]
    label_anchor_columns: list[str]
    sample_coordinate_triplets: list[CoordinateTriplet]
    label_coordinate_triplets: list[CoordinateTriplet]

    @property
    def sample_coordinate_columns(self) -> list[str]:
        return [
            col for triplet in self.sample_coordinate_triplets for col in (triplet.x_col, triplet.y_col, triplet.z_col)
        ]

    @property
    def label_coordinate_columns(self) -> list[str]:
        return [
            col for triplet in self.label_coordinate_triplets for col in (triplet.x_col, triplet.y_col, triplet.z_col)
        ]

    @property
    def target_kind(self) -> str:
        return "residue_coordinates"


def detect_rna_structure_task(data_dir: Path) -> bool:
    return find_rna_structure_files(data_dir) is not None


def find_rna_structure_files(data_dir: Path) -> RnaStructureFiles | None:
    if not data_dir.exists():
        return None

    tabular_files = [path for path in data_dir.rglob("*") if path.is_file() and is_tabular_data_path(path)]
    if not tabular_files:
        return None

    train_sequences_path = _pick_best_path(tabular_files, include_tokens=("train", "sequence"))
    test_sequences_path = _pick_best_path(tabular_files, include_tokens=("test", "sequence"))
    train_labels_path = _pick_best_path(tabular_files, include_tokens=("train", "label"))
    sample_submission_path = _pick_best_sample_submission(tabular_files)

    if not all((train_sequences_path, test_sequences_path, train_labels_path, sample_submission_path)):
        return None

    # RNA detection is a best-effort classifier that runs for every dataset.
    # Non-tabular competitions (for example ARC) can use JSON objects whose
    # values have different lengths. Those are valid competition artifacts but
    # cannot be represented by a DataFrame, so treat them as "not RNA" instead
    # of aborting bootstrap before the competition-specific path can run.
    train_head = _try_read_table_head(train_sequences_path)
    test_head = _try_read_table_head(test_sequences_path)
    label_head = _try_read_table_head(train_labels_path)
    sample_head = _try_read_table_head(sample_submission_path)
    if any(frame is None for frame in (train_head, test_head, label_head, sample_head)):
        return None

    if not _looks_like_sequence_table(train_head):
        return None
    if not _looks_like_sequence_table(test_head):
        return None
    if not infer_coordinate_triplets(sample_head.columns):
        return None
    if not infer_coordinate_triplets(label_head.columns):
        return None

    sample_anchor_columns = infer_anchor_columns(sample_head.columns)
    label_anchor_columns = infer_anchor_columns(label_head.columns)
    if not sample_anchor_columns or not label_anchor_columns:
        return None

    return RnaStructureFiles(
        train_sequences_path=train_sequences_path,
        test_sequences_path=test_sequences_path,
        train_labels_path=train_labels_path,
        sample_submission_path=sample_submission_path,
    )


def load_rna_structure_task(data_dir: Path) -> RnaStructureTask:
    files = find_rna_structure_files(data_dir)
    if files is None:
        raise FileNotFoundError(f"No RNA sequence/structure task layout detected under {data_dir}.")

    train_sequences = _read_table(files.train_sequences_path)
    test_sequences = _read_table(files.test_sequences_path)
    train_labels = _read_table(files.train_labels_path)
    sample_submission = _read_table(files.sample_submission_path)

    sequence_column = _resolve_sequence_column(train_sequences=train_sequences, test_sequences=test_sequences)
    sequence_id_column = _resolve_sequence_id_column(train_sequences=train_sequences, test_sequences=test_sequences)
    sample_id_column = str(sample_submission.columns[0])
    label_id_column = _resolve_label_id_column(train_labels=train_labels)
    sample_triplets = infer_coordinate_triplets(sample_submission.columns)
    label_triplets = infer_coordinate_triplets(train_labels.columns)
    sample_anchor_columns = infer_anchor_columns(sample_submission.columns)
    label_anchor_columns = infer_anchor_columns(train_labels.columns)

    if not sample_triplets:
        raise ValueError("RNA sample submission is missing coordinate triplets.")
    if not label_triplets:
        raise ValueError("RNA training labels are missing coordinate triplets.")

    _validate_sequence_ids(
        frame=train_sequences,
        id_column=sequence_id_column,
        expected_ids={extract_target_id(value) for value in train_labels[label_id_column].astype(str)},
        frame_name="train_sequences",
    )
    _validate_sequence_ids(
        frame=test_sequences,
        id_column=sequence_id_column,
        expected_ids={extract_target_id(value) for value in sample_submission[sample_id_column].astype(str)},
        frame_name="test_sequences",
    )

    return RnaStructureTask(
        files=files,
        train_sequences=train_sequences,
        test_sequences=test_sequences,
        train_labels=train_labels,
        sample_submission=sample_submission,
        sequence_id_column=sequence_id_column,
        sequence_column=sequence_column,
        sample_id_column=sample_id_column,
        label_id_column=label_id_column,
        sample_anchor_columns=sample_anchor_columns,
        label_anchor_columns=label_anchor_columns,
        sample_coordinate_triplets=sample_triplets,
        label_coordinate_triplets=label_triplets,
    )


def infer_coordinate_triplets(columns: pd.Index | list[str]) -> list[CoordinateTriplet]:
    grouped: dict[int, dict[str, str]] = {}
    for raw_column in columns:
        column = str(raw_column)
        match = _COORD_COL_RE.fullmatch(column)
        if match is None:
            continue
        copy_index = int(match.group("copy"))
        axis = match.group("axis").lower()
        grouped.setdefault(copy_index, {})[axis] = column

    triplets: list[CoordinateTriplet] = []
    for copy_index in sorted(grouped):
        axes = grouped[copy_index]
        if {"x", "y", "z"}.issubset(axes):
            triplets.append(
                CoordinateTriplet(
                    copy_index=copy_index,
                    x_col=axes["x"],
                    y_col=axes["y"],
                    z_col=axes["z"],
                )
            )
    return triplets


def infer_anchor_columns(columns: pd.Index | list[str]) -> list[str]:
    ordered = [str(column) for column in columns]
    triplets = infer_coordinate_triplets(ordered)
    if not triplets:
        return [column for column in ordered if _COORD_COL_RE.fullmatch(column) is None]
    first_coord = min(ordered.index(triplet.x_col) for triplet in triplets)
    return ordered[:first_coord]


def extract_target_id(value: object) -> str:
    text = str(value).strip()
    head, sep, tail = text.rpartition("_")
    if sep and tail.isdigit():
        return head
    return text


def extract_residue_index(value: object) -> int | None:
    text = str(value).strip()
    _, sep, tail = text.rpartition("_")
    if not sep or not tail.isdigit():
        return None
    return int(tail)


def write_rna_structure_submission(
    *,
    sample_submission: pd.DataFrame,
    predictions_by_target: dict[str, np.ndarray],
    output_path: Path,
) -> Path:
    sample_id_column = str(sample_submission.columns[0])
    triplets = infer_coordinate_triplets(sample_submission.columns)
    if not triplets:
        raise ValueError("Sample submission does not contain RNA coordinate columns.")

    coordinate_columns = [col for triplet in triplets for col in (triplet.x_col, triplet.y_col, triplet.z_col)]
    submission = sample_submission.copy()
    for column in coordinate_columns:
        submission[column] = pd.to_numeric(submission[column], errors="coerce").astype(float)
    default_prediction = _default_coordinate_prediction(coordinate_columns)

    for index, row in submission.iterrows():
        target_id = extract_target_id(row[sample_id_column])
        residue_index = extract_residue_index(row[sample_id_column])
        values = predictions_by_target.get(target_id)
        if values is None or residue_index is None or residue_index <= 0 or residue_index > len(values):
            coord_row = default_prediction
        else:
            coord_row = _normalize_coordinate_row(values[residue_index - 1], coordinate_columns=coordinate_columns)
        for column, value in coord_row.items():
            submission.at[index, column] = value

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return write_table(submission, output_path)


def evaluate_coordinate_predictions(
    *,
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    id_column: str,
) -> float:
    truth_ids = truth[id_column].astype(str)
    pred_ids = predictions[id_column].astype(str)
    merged = truth.assign(__id__=truth_ids).merge(
        predictions.assign(__id__=pred_ids),
        on="__id__",
        suffixes=("_truth", "_pred"),
        how="inner",
    )
    triplets = infer_coordinate_triplets(truth.columns)
    if merged.empty or not triplets:
        raise ValueError("Unable to evaluate RNA coordinate predictions without overlapping ids and coordinates.")

    squared_errors: list[np.ndarray] = []
    for triplet in triplets:
        axes_truth = merged[[f"{triplet.x_col}_truth", f"{triplet.y_col}_truth", f"{triplet.z_col}_truth"]].to_numpy(
            dtype=float
        )
        axes_pred = merged[[f"{triplet.x_col}_pred", f"{triplet.y_col}_pred", f"{triplet.z_col}_pred"]].to_numpy(
            dtype=float
        )
        squared_errors.append((axes_truth - axes_pred) ** 2)
    stacked = np.concatenate(squared_errors, axis=1)
    return float(np.sqrt(np.mean(stacked)))


def build_coordinate_baseline_predictions(
    *,
    train_labels: pd.DataFrame,
    sample_submission: pd.DataFrame,
    label_id_column: str,
) -> dict[str, np.ndarray]:
    sample_id_column = str(sample_submission.columns[0])
    label_triplets = infer_coordinate_triplets(train_labels.columns)
    sample_triplets = infer_coordinate_triplets(sample_submission.columns)
    if not label_triplets or not sample_triplets:
        raise ValueError("RNA coordinate baseline requires coordinate columns in labels and sample submission.")

    primary_triplet = label_triplets[0]
    label_frame = train_labels.copy()
    label_frame["__target_id__"] = label_frame[label_id_column].astype(str).map(extract_target_id)
    label_frame["__residue_index__"] = label_frame[label_id_column].astype(str).map(extract_residue_index)
    label_frame = label_frame[label_frame["__residue_index__"].notna()].copy()
    label_frame["__residue_index__"] = label_frame["__residue_index__"].astype(int)

    per_resname = (
        label_frame.groupby("resname")[[primary_triplet.x_col, primary_triplet.y_col, primary_triplet.z_col]].mean()
        if "resname" in label_frame.columns
        else pd.DataFrame()
    )
    per_residue = label_frame.groupby("__residue_index__")[
        [primary_triplet.x_col, primary_triplet.y_col, primary_triplet.z_col]
    ].mean()
    overall = label_frame[[primary_triplet.x_col, primary_triplet.y_col, primary_triplet.z_col]].mean()

    predictions_by_target: dict[str, np.ndarray] = {}
    grouped_rows = sample_submission.assign(
        __target_id__=sample_submission[sample_id_column].astype(str).map(extract_target_id),
        __residue_index__=sample_submission[sample_id_column].astype(str).map(extract_residue_index),
    ).groupby("__target_id__", sort=False)

    for target_id, group in grouped_rows:
        rows: list[list[float]] = []
        for _, row in group.iterrows():
            residue_index = int(row["__residue_index__"]) if pd.notna(row["__residue_index__"]) else 0
            if "resname" in row and row["resname"] in per_resname.index:
                coords = per_resname.loc[row["resname"]]
            elif residue_index in per_residue.index:
                coords = per_residue.loc[residue_index]
            else:
                coords = overall
            rows.append(
                [
                    float(coords[primary_triplet.x_col]),
                    float(coords[primary_triplet.y_col]),
                    float(coords[primary_triplet.z_col]),
                ]
            )
        predictions_by_target[str(target_id)] = np.asarray(rows, dtype=float)
    return predictions_by_target


def _default_coordinate_prediction(columns: list[str]) -> dict[str, float]:
    return {column: 0.0 for column in columns}


def _normalize_coordinate_row(values: np.ndarray | list[float], *, coordinate_columns: list[str]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size < 3:
        raise ValueError(f"RNA coordinate row must contain at least 3 values, got {arr.size}.")
    x, y, z = float(arr[0]), float(arr[1]), float(arr[2])
    normalized: dict[str, float] = {}
    for offset in range(0, len(coordinate_columns), 3):
        normalized[coordinate_columns[offset]] = x
        normalized[coordinate_columns[offset + 1]] = y
        normalized[coordinate_columns[offset + 2]] = z
    return normalized


def _read_table_head(path: Path) -> pd.DataFrame:
    return read_table(path, nrows=5)


def _try_read_table_head(path: Path) -> pd.DataFrame | None:
    try:
        return _read_table_head(path)
    except (ImportError, OSError, TypeError, ValueError):
        return None


def _read_table(path: Path) -> pd.DataFrame:
    return read_table(path)


def _pick_best_sample_submission(paths: list[Path]) -> Path | None:
    candidates = [path for path in paths if sample_name_score(path) > 0]
    if not candidates:
        return None
    return max(candidates, key=sample_candidate_key)


def _pick_best_path(paths: list[Path], *, include_tokens: tuple[str, ...]) -> Path | None:
    candidates: list[tuple[tuple[int, int, str], Path]] = []
    for path in paths:
        lowered = path.name.lower()
        score = sum(1 for token in include_tokens if token in lowered)
        if score <= 0:
            continue
        exact = 1 if all(token in lowered for token in include_tokens) else 0
        candidates.append(((-exact, -score, lowered), path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _looks_like_sequence_table(frame: pd.DataFrame) -> bool:
    lowered = {str(column).strip().lower() for column in frame.columns}
    return "sequence" in lowered and any("id" in column for column in lowered)


def _resolve_sequence_column(*, train_sequences: pd.DataFrame, test_sequences: pd.DataFrame) -> str:
    common = [str(column) for column in train_sequences.columns if column in test_sequences.columns]
    for column in common:
        if str(column).strip().lower() == "sequence":
            return column
    raise ValueError("RNA sequence tables are missing a shared 'sequence' column.")


def _resolve_sequence_id_column(*, train_sequences: pd.DataFrame, test_sequences: pd.DataFrame) -> str:
    common = [str(column) for column in train_sequences.columns if column in test_sequences.columns]
    for column in common:
        lowered = column.strip().lower()
        if lowered == "target_id" or lowered == "sequence_id":
            return column
    for column in common:
        lowered = column.strip().lower()
        if lowered == "id" or lowered.endswith("_id") or "id" in lowered:
            return column
    if common:
        return common[0]
    raise ValueError("RNA sequence tables do not share an identifier column.")


def _resolve_label_id_column(*, train_labels: pd.DataFrame) -> str:
    for column in train_labels.columns:
        lowered = str(column).strip().lower()
        if lowered == "id" or lowered.endswith("_id") or "id" in lowered:
            return str(column)
    raise ValueError("RNA label table does not expose an identifier column.")


def _validate_sequence_ids(
    *,
    frame: pd.DataFrame,
    id_column: str,
    expected_ids: set[str],
    frame_name: str,
) -> None:
    actual_ids = {str(value).strip() for value in frame[id_column].astype(str).tolist()}
    missing = sorted(expected_ids - actual_ids)
    if missing:
        preview = missing[:5]
        raise ValueError(f"{frame_name} is missing RNA target ids referenced by labels/sample: {preview}")
