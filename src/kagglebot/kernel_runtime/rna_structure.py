from __future__ import annotations

from pathlib import Path

import pandas as pd

from kagglebot.rna_structure import (
    build_coordinate_baseline_predictions,
    evaluate_coordinate_predictions,
    load_rna_structure_task,
    write_rna_structure_submission,
)

__all__ = [
    "build_coordinate_baseline_predictions",
    "evaluate_coordinate_predictions",
    "load_rna_structure_task",
    "write_rna_structure_submission",
]


def write_baseline_submission(*, data_dir: Path, output_path: Path) -> Path:
    task = load_rna_structure_task(data_dir)
    predictions = build_coordinate_baseline_predictions(
        train_labels=task.train_labels,
        sample_submission=task.sample_submission,
        label_id_column=task.label_id_column,
    )
    return write_rna_structure_submission(
        sample_submission=task.sample_submission,
        predictions_by_target=predictions,
        output_path=output_path,
    )


def score_submission_against_labels(*, labels: pd.DataFrame, submission: pd.DataFrame, id_column: str = "ID") -> float:
    return evaluate_coordinate_predictions(truth=labels, predictions=submission, id_column=id_column)
