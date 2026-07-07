from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kagglebot.compute import Compute
from kagglebot.solver import initial_model
from kagglebot.solver.initial_model import (
    _train_vision_yolo_submission,
    discover_yolo_labeled_train_files,
    find_vision_train_labels_path,
    infer_vision_train_label_columns,
    load_vision_train_labels,
    normalize_vision_train_labels,
)


def test_load_vision_train_labels_supports_jsonl(tmp_path: Path) -> None:
    labels_path = tmp_path / "train_labels.jsonl"
    labels_path.write_text(
        '{"filename": "a.jpg", "right_place": 1}\n{"filename": "b.jpg", "right_place": 0}\n',
        encoding="utf-8",
    )

    resolved, labels = load_vision_train_labels(tmp_path)

    assert resolved == labels_path
    assert labels["filename"].tolist() == ["a.jpg", "b.jpg"]
    assert labels["right_place"].tolist() == [1, 0]


def test_load_vision_train_labels_supports_csv_gz(tmp_path: Path) -> None:
    labels_path = tmp_path / "train_labels.csv.gz"
    with gzip.open(labels_path, "wt", encoding="utf-8") as handle:
        handle.write("filename,right_place\na.jpg,1\nb.jpg,0\n")

    resolved, labels = load_vision_train_labels(tmp_path)

    assert resolved == labels_path
    assert labels["filename"].tolist() == ["a.jpg", "b.jpg"]
    assert labels["right_place"].tolist() == [1, 0]


def test_find_vision_train_labels_path_discovers_excel_label_file(tmp_path: Path) -> None:
    nested = tmp_path / "metadata"
    nested.mkdir()
    labels_path = nested / "TrainLabels.xlsx"
    pd.DataFrame({"filename": ["a.jpg"], "right_place": [1]}).to_excel(labels_path, index=False)

    assert find_vision_train_labels_path(tmp_path) == labels_path


def test_find_vision_train_labels_path_discovers_common_non_csv_label_name(tmp_path: Path) -> None:
    nested = tmp_path / "metadata"
    nested.mkdir()
    labels_path = nested / "TrainLabels.tsv"
    labels_path.write_text("filename\tright_place\na.jpg\t1\n", encoding="utf-8")

    assert find_vision_train_labels_path(tmp_path) == labels_path


def test_load_vision_train_labels_accepts_image_id_and_target_aliases(tmp_path: Path) -> None:
    labels_path = tmp_path / "train_labels.csv"
    labels_path.write_text("image_id,target\na.jpg,TRUE\nb.jpg,false\n", encoding="utf-8")

    resolved, labels = load_vision_train_labels(tmp_path)

    assert resolved == labels_path
    assert labels["filename"].tolist() == ["a.jpg", "b.jpg"]
    assert labels["right_place"].tolist() == [1, 0]


@pytest.mark.parametrize(
    ("image_column", "target_column"),
    [
        ("image_path", "target"),
        ("image_file", "label"),
        ("img_path", "class"),
        ("photo_file", "valid_placement"),
    ],
)
def test_load_vision_train_labels_accepts_common_image_reference_aliases(
    tmp_path: Path,
    image_column: str,
    target_column: str,
) -> None:
    labels_path = tmp_path / "train_labels.csv"
    labels_path.write_text(
        f"{image_column},{target_column}\ntrain/a.jpg,yes\ntrain/b.jpg,no\n",
        encoding="utf-8",
    )

    resolved, labels = load_vision_train_labels(tmp_path)

    assert resolved == labels_path
    assert labels["filename"].tolist() == ["train/a.jpg", "train/b.jpg"]
    assert labels["right_place"].tolist() == [1, 0]


def test_normalize_vision_train_labels_infers_common_aliases() -> None:
    frame = pd.DataFrame({"image": ["a.jpg"], "label": ["yes"]})

    assert infer_vision_train_label_columns(frame) == ("image", "label")
    normalized = normalize_vision_train_labels(frame)

    assert normalized.columns.tolist() == ["filename", "right_place"]
    assert normalized["filename"].tolist() == ["a.jpg"]
    assert normalized["right_place"].tolist() == ["yes"]


def test_load_vision_train_labels_reports_actual_file_for_missing_columns(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.tsv"
    labels_path.write_text("image\tfold\na.jpg\t1\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"labels\.tsv missing required columns"):
        load_vision_train_labels(tmp_path)


def test_discover_yolo_labeled_train_files_matches_image_stems(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "a.jpg").write_bytes(b"a")
    (image_dir / "b.png").write_bytes(b"b")
    (label_dir / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (label_dir / "b.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    assert discover_yolo_labeled_train_files(tmp_path) == ["a.jpg", "b.png"]


def test_discover_yolo_labeled_train_files_supports_train_images_layout(tmp_path: Path) -> None:
    image_dir = tmp_path / "train" / "images"
    test_dir = tmp_path / "test" / "images"
    label_dir = tmp_path / "train" / "labels"
    image_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "a.jpg").write_bytes(b"a")
    (label_dir / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    assert discover_yolo_labeled_train_files(tmp_path) == ["a.jpg"]


def test_discover_yolo_labeled_train_files_prefers_nested_relative_label_matches(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"
    (image_dir / "fold_a").mkdir(parents=True)
    (image_dir / "fold_b").mkdir(parents=True)
    (label_dir / "fold_b").mkdir(parents=True)
    (image_dir / "fold_a" / "same.jpg").write_bytes(b"a")
    (image_dir / "fold_b" / "same.jpg").write_bytes(b"b")
    (label_dir / "fold_b" / "same.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    assert discover_yolo_labeled_train_files(tmp_path) == ["fold_b/same.jpg"]


def test_train_vision_yolo_submission_writes_generic_prediction_string_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_train = tmp_path / "images" / "train"
    image_test = tmp_path / "images" / "test"
    label_train = tmp_path / "labels" / "train"
    image_train.mkdir(parents=True)
    image_test.mkdir(parents=True)
    label_train.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        (image_train / name).write_bytes(b"image")
        (label_train / f"{Path(name).stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (image_test / "t1.jpg").write_bytes(b"image")

    monkeypatch.setattr(initial_model, "train_detector", lambda **_: object())
    monkeypatch.setattr(
        initial_model,
        "predict_detector",
        lambda _detector, paths: {
            path.name: np.asarray([[0.0, 0.9, 0.5, 0.5, 0.2, 0.2]], dtype=float) for path in paths
        },
    )
    monkeypatch.setattr(initial_model, "compute_map50_95", lambda **_: 0.5)

    output_path = tmp_path / "submission.csv"
    outcome = _train_vision_yolo_submission(
        data_dir=tmp_path,
        sample_df=pd.DataFrame({"id": [10], "image_id": ["t1.jpg"], "prediction_string": [""]}),
        output_path=output_path,
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=42,
    )

    submission = pd.read_csv(output_path)
    assert submission.columns.tolist() == ["id", "image_id", "prediction_string"]
    assert submission["image_id"].tolist() == ["t1.jpg"]
    assert submission["prediction_string"].iloc[0].startswith("0 0.900000")
    assert outcome.evaluation.metric == "map50_95"
    assert outcome.model_summary["submission_schema"] == {
        "image_column": "image_id",
        "prediction_column": "prediction_string",
        "right_place_column": None,
    }


def test_train_vision_yolo_submission_uses_manifest_fallback_for_non_tabular_output_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_train = tmp_path / "images" / "train"
    image_test = tmp_path / "images" / "test"
    label_train = tmp_path / "labels" / "train"
    image_train.mkdir(parents=True)
    image_test.mkdir(parents=True)
    label_train.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        (image_train / name).write_bytes(b"image")
        (label_train / f"{Path(name).stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (image_test / "t1.jpg").write_bytes(b"image")

    monkeypatch.setattr(initial_model, "train_detector", lambda **_: object())
    monkeypatch.setattr(
        initial_model,
        "predict_detector",
        lambda _detector, paths: {
            path.name: np.asarray([[0.0, 0.9, 0.5, 0.5, 0.2, 0.2]], dtype=float) for path in paths
        },
    )
    monkeypatch.setattr(initial_model, "compute_map50_95", lambda **_: 0.5)

    output_path = tmp_path / "answers.nii.gz"
    outcome = _train_vision_yolo_submission(
        data_dir=tmp_path,
        sample_df=pd.DataFrame({"id": [10], "image_id": ["t1.jpg"], "prediction_string": [""]}),
        output_path=output_path,
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=42,
    )

    assert outcome.submission_path == tmp_path / "answers.tabular.csv"
    assert not output_path.exists()
    submission = pd.read_csv(outcome.submission_path)
    assert submission["image_id"].tolist() == ["t1.jpg"]
    assert submission["prediction_string"].iloc[0].startswith("0 0.900000")
    manifest = json.loads((tmp_path / "submission_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_class"] == "tabular"
    assert manifest["submission_path"] == "answers.tabular.csv"
    assert manifest["requested_output_path"] == "answers.nii.gz"


def test_train_vision_yolo_submission_resolves_dataset_relative_image_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_train = tmp_path / "images" / "train"
    image_test = tmp_path / "images" / "test"
    label_train = tmp_path / "labels" / "train"
    image_train.mkdir(parents=True)
    image_test.mkdir(parents=True)
    label_train.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        (image_train / name).write_bytes(b"image")
        (label_train / f"{Path(name).stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (image_test / "t1.jpg").write_bytes(b"image")

    captured_paths: list[list[Path]] = []
    monkeypatch.setattr(initial_model, "train_detector", lambda **_: object())

    def fake_predict(_detector, paths):  # noqa: ANN001
        captured_paths.append(list(paths))
        return {path.name: np.asarray([[0.0, 0.9, 0.5, 0.5, 0.2, 0.2]], dtype=float) for path in paths}

    monkeypatch.setattr(initial_model, "predict_detector", fake_predict)
    monkeypatch.setattr(initial_model, "compute_map50_95", lambda **_: 0.5)

    output_path = tmp_path / "submission.csv"
    _train_vision_yolo_submission(
        data_dir=tmp_path,
        sample_df=pd.DataFrame({"id": [10], "image_path": ["images/test/t1.jpg"], "prediction_string": [""]}),
        output_path=output_path,
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=42,
    )

    submission = pd.read_csv(output_path)
    assert captured_paths[1] == [image_test / "t1.jpg"]
    assert submission.columns.tolist() == ["id", "image_path", "prediction_string"]
    assert submission["image_path"].tolist() == ["images/test/t1.jpg"]
    assert submission["prediction_string"].iloc[0].startswith("0 0.900000")


def test_train_vision_yolo_submission_keeps_duplicate_basename_predictions_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_train = tmp_path / "images" / "train"
    image_test = tmp_path / "images" / "test"
    label_train = tmp_path / "labels" / "train"
    image_train.mkdir(parents=True)
    (image_test / "fold_a").mkdir(parents=True)
    (image_test / "fold_b").mkdir(parents=True)
    label_train.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        (image_train / name).write_bytes(b"image")
        (label_train / f"{Path(name).stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (image_test / "fold_a" / "same.jpg").write_bytes(b"a")
    (image_test / "fold_b" / "same.jpg").write_bytes(b"b")

    monkeypatch.setattr(initial_model, "train_detector", lambda **_: object())

    def fake_predict(_detector, paths):  # noqa: ANN001
        return {
            str(path): np.asarray([[0.0, 0.91 if "fold_a" in path.as_posix() else 0.73, 0.5, 0.5, 0.2, 0.2]])
            for path in paths
        }

    monkeypatch.setattr(initial_model, "predict_detector", fake_predict)
    monkeypatch.setattr(initial_model, "compute_map50_95", lambda **_: 0.5)

    output_path = tmp_path / "submission.csv"
    _train_vision_yolo_submission(
        data_dir=tmp_path,
        sample_df=pd.DataFrame(
            {
                "image_path": ["images/test/fold_a/same.jpg", "images/test/fold_b/same.jpg"],
                "prediction_string": ["", ""],
            }
        ),
        output_path=output_path,
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=42,
    )

    submission = pd.read_csv(output_path)
    assert submission["image_path"].tolist() == ["images/test/fold_a/same.jpg", "images/test/fold_b/same.jpg"]
    assert "0.910000" in submission["prediction_string"].iloc[0]
    assert "0.730000" in submission["prediction_string"].iloc[1]


def test_train_vision_yolo_submission_resolves_dataset_relative_train_label_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_train = tmp_path / "images" / "train"
    image_test = tmp_path / "images" / "test"
    label_train = tmp_path / "labels" / "train"
    image_train.mkdir(parents=True)
    image_test.mkdir(parents=True)
    label_train.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        (image_train / name).write_bytes(b"image")
        (label_train / f"{Path(name).stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (image_test / "t1.jpg").write_bytes(b"image")
    (tmp_path / "train_labels.csv").write_text(
        "image_path,right_place\nimages/train/a.jpg,1\nimages/train/b.jpg,0\n",
        encoding="utf-8",
    )

    captured_paths: list[list[Path]] = []
    monkeypatch.setattr(initial_model, "train_detector", lambda **_: object())
    monkeypatch.setattr(initial_model, "tune_right_place_params", lambda **_: {"prediction_score_thr": 0.1})
    monkeypatch.setattr(
        initial_model,
        "evaluate_combined_metric",
        lambda **_: {
            "map50_95": 0.5,
            "f1": 1.0,
            "combined": 0.75,
            "thresholds": {"prediction_score_thr": 0.1},
        },
    )

    def fake_predict(_detector, paths):  # noqa: ANN001
        captured_paths.append(list(paths))
        return {path.name: np.asarray([[0.0, 0.9, 0.5, 0.5, 0.2, 0.2]], dtype=float) for path in paths}

    monkeypatch.setattr(initial_model, "predict_detector", fake_predict)
    monkeypatch.setattr(initial_model, "compute_map50_95", lambda **_: 0.5)

    output_path = tmp_path / "submission.csv"
    _train_vision_yolo_submission(
        data_dir=tmp_path,
        sample_df=pd.DataFrame(
            {
                "id": [10],
                "image_path": ["images/test/t1.jpg"],
                "prediction_string": [""],
                "right_place": [0],
            }
        ),
        output_path=output_path,
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=42,
    )

    submission = pd.read_csv(output_path)
    assert all(path.parent == image_train for path in captured_paths[0])
    assert captured_paths[1] == [image_test / "t1.jpg"]
    assert submission["image_path"].tolist() == ["images/test/t1.jpg"]
    assert submission["prediction_string"].iloc[0].startswith("0 0.900000")
    assert "right_place" in submission.columns


def test_train_vision_yolo_submission_uses_train_images_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_train = tmp_path / "train" / "images"
    image_test = tmp_path / "test" / "images"
    label_train = tmp_path / "train" / "labels"
    image_train.mkdir(parents=True)
    image_test.mkdir(parents=True)
    label_train.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        (image_train / name).write_bytes(b"image")
        (label_train / f"{Path(name).stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (image_test / "t1.jpg").write_bytes(b"image")

    captured_paths: list[list[Path]] = []
    monkeypatch.setattr(initial_model, "train_detector", lambda **_: object())

    def fake_predict(_detector, paths):  # noqa: ANN001
        captured_paths.append(list(paths))
        return {path.name: np.asarray([[0.0, 0.9, 0.5, 0.5, 0.2, 0.2]], dtype=float) for path in paths}

    monkeypatch.setattr(initial_model, "predict_detector", fake_predict)
    monkeypatch.setattr(initial_model, "compute_map50_95", lambda **_: 0.5)

    _train_vision_yolo_submission(
        data_dir=tmp_path,
        sample_df=pd.DataFrame({"id": [10], "image_id": ["t1.jpg"], "prediction_string": [""]}),
        output_path=tmp_path / "submission.csv",
        compute=Compute.local_gpu,
        strict_accelerator=False,
        seed=42,
    )

    assert captured_paths[0][0].parent == image_train
    assert captured_paths[1][0].parent == image_test
