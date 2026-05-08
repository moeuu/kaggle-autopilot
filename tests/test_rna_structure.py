from __future__ import annotations

import numpy as np
import pandas as pd

from kagglebot.knowledge import build_dataset_profile, derive_problem_types
from kagglebot.rna_structure import (
    build_coordinate_baseline_predictions,
    detect_rna_structure_task,
    evaluate_coordinate_predictions,
    load_rna_structure_task,
    write_rna_structure_submission,
)


def _write_rna_structure_fixture(tmp_path):
    data_dir = tmp_path / "rna-structure-demo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "target_id": ["RNA1", "RNA2"],
            "sequence": ["ACG", "GU"],
            "description": ["demo one", "demo two"],
        }
    ).to_csv(data_dir / "train_sequences.csv", index=False)
    pd.DataFrame(
        {
            "target_id": ["RNA3"],
            "sequence": ["AC"],
            "description": ["demo test"],
        }
    ).to_csv(data_dir / "test_sequences.csv", index=False)
    pd.DataFrame(
        {
            "ID": ["RNA1_1", "RNA1_2", "RNA1_3", "RNA2_1", "RNA2_2"],
            "resname": ["A", "C", "G", "G", "U"],
            "resid": [1, 2, 3, 1, 2],
            "x_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y_1": [1.5, 2.5, 3.5, 4.5, 5.5],
            "z_1": [2.0, 3.0, 4.0, 5.0, 6.0],
        }
    ).to_csv(data_dir / "train_labels.csv", index=False)
    pd.DataFrame(
        {
            "ID": ["RNA3_1", "RNA3_2"],
            "resname": ["A", "C"],
            "resid": [1, 2],
            "x_1": [0.0, 0.0],
            "y_1": [0.0, 0.0],
            "z_1": [0.0, 0.0],
            "x_2": [0.0, 0.0],
            "y_2": [0.0, 0.0],
            "z_2": [0.0, 0.0],
        }
    ).to_csv(data_dir / "sample_submission.csv", index=False)
    return data_dir


def test_detect_and_load_rna_structure_task(tmp_path) -> None:
    data_dir = _write_rna_structure_fixture(tmp_path)

    assert detect_rna_structure_task(data_dir) is True

    task = load_rna_structure_task(data_dir)
    assert task.sequence_id_column == "target_id"
    assert task.sequence_column == "sequence"
    assert task.sample_id_column == "ID"
    assert task.label_id_column == "ID"
    assert task.sample_anchor_columns == ["ID", "resname", "resid"]
    assert [triplet.copy_index for triplet in task.sample_coordinate_triplets] == [1, 2]
    assert [triplet.copy_index for triplet in task.label_coordinate_triplets] == [1]


def test_build_dataset_profile_identifies_rna_structure(tmp_path) -> None:
    data_dir = _write_rna_structure_fixture(tmp_path)

    profile = build_dataset_profile(data_dir)

    assert profile["status"] == "ok"
    assert profile["modality"] == "rna_structure"
    assert profile["target_kind"] == "residue_coordinates"
    assert "rna_structure" in profile["tags"]
    assert "coordinate_regression" in profile["tags"]
    problem_types = derive_problem_types(profile)
    assert "rna_structure:regression" in problem_types
    assert "rna_structure" in problem_types
    assert "rna" in problem_types


def test_write_rna_structure_submission_replicates_coordinates_across_triplets(tmp_path) -> None:
    data_dir = _write_rna_structure_fixture(tmp_path)
    task = load_rna_structure_task(data_dir)

    predictions = {"RNA3": np.asarray([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=float)}
    output_path = tmp_path / "submission.csv"
    write_rna_structure_submission(
        sample_submission=task.sample_submission,
        predictions_by_target=predictions,
        output_path=output_path,
    )
    frame = pd.read_csv(output_path)
    assert frame["x_1"].tolist() == [10.0, 20.0]
    assert frame["y_1"].tolist() == [11.0, 21.0]
    assert frame["z_1"].tolist() == [12.0, 22.0]
    assert frame["x_2"].tolist() == [10.0, 20.0]
    assert frame["y_2"].tolist() == [11.0, 21.0]
    assert frame["z_2"].tolist() == [12.0, 22.0]


def test_build_coordinate_baseline_predictions_and_evaluate(tmp_path) -> None:
    data_dir = _write_rna_structure_fixture(tmp_path)
    task = load_rna_structure_task(data_dir)

    valid_sample = pd.DataFrame(
        {
            "ID": ["RNA2_1", "RNA2_2"],
            "resname": ["G", "U"],
            "resid": [1, 2],
            "x_1": [0.0, 0.0],
            "y_1": [0.0, 0.0],
            "z_1": [0.0, 0.0],
        }
    )
    predictions = build_coordinate_baseline_predictions(
        train_labels=task.train_labels[task.train_labels["ID"].str.startswith("RNA1_")],
        sample_submission=valid_sample,
        label_id_column="ID",
    )
    output_path = tmp_path / "valid_submission.csv"
    write_rna_structure_submission(
        sample_submission=valid_sample,
        predictions_by_target=predictions,
        output_path=output_path,
    )
    predicted = pd.read_csv(output_path)
    truth = task.train_labels[task.train_labels["ID"].str.startswith("RNA2_")].copy()
    score = evaluate_coordinate_predictions(truth=truth, predictions=predicted, id_column="ID")
    assert score >= 0.0
