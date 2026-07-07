from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd

from kagglebot.compute import Compute
from kagglebot.knowledge import build_dataset_profile, derive_problem_types
from kagglebot.rna_structure import (
    _read_table_head,
    build_coordinate_baseline_predictions,
    detect_rna_structure_task,
    evaluate_coordinate_predictions,
    load_rna_structure_task,
    write_rna_structure_submission,
)
from kagglebot.solver.initial_model import train_evaluate_and_predict
from kagglebot.solver.io import read_table


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


def test_rna_structure_read_table_head_uses_limited_reader(tmp_path, monkeypatch) -> None:
    path = tmp_path / "table.csv"
    path.write_text("id,target\n1,0.1\n2,0.2\n3,0.3\n4,0.4\n5,0.5\n6,0.6\n", encoding="utf-8")
    calls: list[int | None] = []

    import kagglebot.rna_structure as rna_structure

    real_read_table = rna_structure.read_table

    def spy_read_table(path, *, nrows=None):
        calls.append(nrows)
        return real_read_table(path, nrows=nrows)

    monkeypatch.setattr(rna_structure, "read_table", spy_read_table)

    head = _read_table_head(path)

    assert calls == [5]
    assert head["id"].tolist() == [1, 2, 3, 4, 5]


def test_detect_and_load_rna_structure_task_supports_mixed_tabular_formats(tmp_path) -> None:
    data_dir = tmp_path / "rna-structure-demo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train_sequences.tsv").write_text(
        "target_id\tsequence\tdescription\nRNA1\tACG\tdemo one\nRNA2\tGU\tdemo two\n",
        encoding="utf-8",
    )
    (data_dir / "test_sequences.jsonl").write_text(
        '{"target_id": "RNA3", "sequence": "AC", "description": "demo test"}\n',
        encoding="utf-8",
    )
    (data_dir / "train_labels.tsv").write_text(
        "ID\tresname\tresid\tx_1\ty_1\tz_1\n"
        "RNA1_1\tA\t1\t1.0\t1.5\t2.0\n"
        "RNA1_2\tC\t2\t2.0\t2.5\t3.0\n"
        "RNA2_1\tG\t1\t4.0\t4.5\t5.0\n",
        encoding="utf-8",
    )
    (data_dir / "prediction_template.jsonl").write_text(
        '{"ID": "RNA3_1", "resname": "A", "resid": 1, "x_1": 0.0, "y_1": 0.0, "z_1": 0.0}\n'
        '{"ID": "RNA3_2", "resname": "C", "resid": 2, "x_1": 0.0, "y_1": 0.0, "z_1": 0.0}\n',
        encoding="utf-8",
    )

    assert detect_rna_structure_task(data_dir) is True
    task = load_rna_structure_task(data_dir)

    assert task.files.train_sequences_path.name == "train_sequences.tsv"
    assert task.files.test_sequences_path.name == "test_sequences.jsonl"
    assert task.files.train_labels_path.name == "train_labels.tsv"
    assert task.files.sample_submission_path.name == "prediction_template.jsonl"
    assert task.test_sequences["target_id"].tolist() == ["RNA3"]


def test_detect_and_load_rna_structure_task_supports_compressed_and_excel_tables(tmp_path) -> None:
    data_dir = tmp_path / "rna-structure-demo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(data_dir / "train_sequences.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("target_id,sequence,description\nRNA1,ACG,demo one\nRNA2,GU,demo two\n")
    pd.DataFrame(
        {
            "target_id": ["RNA3"],
            "sequence": ["AC"],
            "description": ["demo test"],
        }
    ).to_excel(data_dir / "test_sequences.xlsx", index=False)
    pd.DataFrame(
        {
            "ID": ["RNA1_1", "RNA1_2", "RNA2_1"],
            "resname": ["A", "C", "G"],
            "resid": [1, 2, 1],
            "x_1": [1.0, 2.0, 4.0],
            "y_1": [1.5, 2.5, 4.5],
            "z_1": [2.0, 3.0, 5.0],
        }
    ).to_excel(data_dir / "train_labels.xlsx", index=False)
    with gzip.open(data_dir / "sample_submission.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(
            '{"ID": "RNA3_1", "resname": "A", "resid": 1, "x_1": 0.0, "y_1": 0.0, "z_1": 0.0}\n'
            '{"ID": "RNA3_2", "resname": "C", "resid": 2, "x_1": 0.0, "y_1": 0.0, "z_1": 0.0}\n'
        )

    assert detect_rna_structure_task(data_dir) is True
    task = load_rna_structure_task(data_dir)

    assert task.files.train_sequences_path.name == "train_sequences.csv.gz"
    assert task.files.test_sequences_path.name == "test_sequences.xlsx"
    assert task.files.train_labels_path.name == "train_labels.xlsx"
    assert task.files.sample_submission_path.name == "sample_submission.jsonl.gz"
    assert task.test_sequences["target_id"].tolist() == ["RNA3"]


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


def test_write_rna_structure_submission_respects_jsonl_suffix(tmp_path) -> None:
    data_dir = _write_rna_structure_fixture(tmp_path)
    task = load_rna_structure_task(data_dir)

    output_path = tmp_path / "submission.jsonl"
    write_rna_structure_submission(
        sample_submission=task.sample_submission,
        predictions_by_target={"RNA3": np.asarray([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=float)},
        output_path=output_path,
    )

    frame = pd.read_json(output_path, lines=True)
    assert list(frame.columns) == list(task.sample_submission.columns)
    assert frame["x_1"].tolist() == [10.0, 20.0]


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


def test_initial_model_rna_structure_respects_jsonl_output_suffix(tmp_path) -> None:
    data_dir = tmp_path / "rna-structure-demo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train_sequences.tsv").write_text(
        "target_id\tsequence\tdescription\nRNA1\tACG\tdemo one\nRNA2\tGU\tdemo two\n",
        encoding="utf-8",
    )
    (data_dir / "test_sequences.jsonl").write_text(
        '{"target_id": "RNA1", "sequence": "ACG", "description": "demo validation one"}\n'
        '{"target_id": "RNA2", "sequence": "GU", "description": "demo validation two"}\n'
        '{"target_id": "RNA3", "sequence": "AC", "description": "demo test"}\n',
        encoding="utf-8",
    )
    (data_dir / "train_labels.tsv").write_text(
        "ID\tresname\tresid\tx_1\ty_1\tz_1\n"
        "RNA1_1\tA\t1\t1.0\t1.5\t2.0\n"
        "RNA1_2\tC\t2\t2.0\t2.5\t3.0\n"
        "RNA2_1\tG\t1\t4.0\t4.5\t5.0\n"
        "RNA2_2\tU\t2\t5.0\t5.5\t6.0\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.jsonl").write_text(
        '{"ID": "RNA1_1", "resname": "A", "resid": 1, "x_1": 0.0, "y_1": 0.0, "z_1": 0.0}\n'
        '{"ID": "RNA1_2", "resname": "C", "resid": 2, "x_1": 0.0, "y_1": 0.0, "z_1": 0.0}\n'
        '{"ID": "RNA2_1", "resname": "G", "resid": 1, "x_1": 0.0, "y_1": 0.0, "z_1": 0.0}\n'
        '{"ID": "RNA2_2", "resname": "U", "resid": 2, "x_1": 0.0, "y_1": 0.0, "z_1": 0.0}\n'
        '{"ID": "RNA3_1", "resname": "A", "resid": 1, "x_1": 0.0, "y_1": 0.0, "z_1": 0.0}\n'
        '{"ID": "RNA3_2", "resname": "C", "resid": 2, "x_1": 0.0, "y_1": 0.0, "z_1": 0.0}\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "submission.jsonl"

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=output_path,
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=42,
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    assert outcome.submission_path == output_path
    assert output_path.exists()
    assert (tmp_path / "validation_submission.jsonl").exists()
    assert read_table(output_path)["ID"].tolist() == ["RNA1_1", "RNA1_2", "RNA2_1", "RNA2_2", "RNA3_1", "RNA3_2"]
    assert outcome.evaluation.metric == "rmse"


def test_initial_model_rna_structure_uses_manifest_fallback_for_non_tabular_output_suffix(tmp_path) -> None:
    data_dir = _write_rna_structure_fixture(tmp_path)
    output_path = tmp_path / "answers.nii.gz"

    outcome = train_evaluate_and_predict(
        data_dir=data_dir,
        output_path=output_path,
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=42,
        score_source="holdout",
        metric="rmse",
        direction="minimize",
        holdout_frac=0.2,
        cv_folds=3,
        plan_score_source=None,
        target_override=None,
    )

    assert outcome.submission_path == tmp_path / "answers.tabular.csv"
    assert not output_path.exists()
    assert (tmp_path / "validation_submission.csv").exists()
    assert read_table(outcome.submission_path)["ID"].tolist() == ["RNA3_1", "RNA3_2"]
    manifest = json.loads((tmp_path / "submission_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_class"] == "tabular"
    assert manifest["submission_path"] == "answers.tabular.csv"
    assert manifest["requested_output_path"] == "answers.nii.gz"
