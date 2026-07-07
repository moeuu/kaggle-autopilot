from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kagglebot.training.vision_yolo import (
    _store_detection_result,
    _torchvision_training_image_label_paths,
    compute_map50_95,
    detect_yolo_submission_task,
    find_yolo_data_layout,
    infer_detection_submission_schema,
    prepare_ultralytics_dataset,
    resolve_yolo_image_reference,
    train_val_split,
)


def test_train_val_split_handles_single_row() -> None:
    train_files, val_files = train_val_split(
        pd.DataFrame({"filename": ["one.jpg"], "right_place": [1]}),
        seed=42,
    )

    assert train_files == ["one.jpg"]
    assert val_files == []


def test_train_val_split_uses_adjusted_integer_validation_count() -> None:
    frame = pd.DataFrame(
        {
            "filename": [f"{idx}.jpg" for idx in range(3)],
            "right_place": [0, 1, 1],
        }
    )

    train_files, val_files = train_val_split(frame, seed=42, val_frac=0.8)

    assert len(train_files) == 1
    assert len(val_files) == 2
    assert sorted(train_files + val_files) == ["0.jpg", "1.jpg", "2.jpg"]


def test_prepare_ultralytics_dataset_copies_when_symlink_unavailable(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    image_dir = data_dir / "images" / "train"
    label_dir = data_dir / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "one.jpg").write_bytes(b"image")
    (label_dir / "one.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    def fail_symlink(self: Path, target: Path, target_is_directory: bool = False) -> None:  # noqa: ARG001
        raise OSError("symlink unavailable")

    monkeypatch.setattr(Path, "symlink_to", fail_symlink)

    dataset_yaml = prepare_ultralytics_dataset(tmp_path / "work", data_dir, ["one.jpg"], [])

    staged_image = dataset_yaml.parent / "images" / "train" / "one.jpg"
    staged_label = dataset_yaml.parent / "labels" / "train" / "one.txt"
    assert staged_image.read_bytes() == b"image"
    assert staged_label.read_text(encoding="utf-8") == "0 0.5 0.5 0.1 0.1\n"
    assert "names: ['class_0', 'class_1']" in dataset_yaml.read_text(encoding="utf-8")
    assert not staged_image.is_symlink()


def test_infer_detection_submission_schema_accepts_generic_prediction_string() -> None:
    sample = pd.DataFrame({"id": [1], "image_id": ["a.jpg"], "prediction_string": [""]})

    schema = infer_detection_submission_schema(sample)

    assert schema is not None
    assert schema.image_column == "image_id"
    assert schema.prediction_column == "prediction_string"
    assert schema.right_place_column is None


def test_infer_detection_submission_schema_accepts_image_path_alias() -> None:
    sample = pd.DataFrame({"image_path": ["images/test/a.jpg"], "prediction_string": [""]})

    schema = infer_detection_submission_schema(sample)

    assert schema is not None
    assert schema.image_column == "image_path"
    assert schema.prediction_column == "prediction_string"


def test_detect_yolo_submission_task_accepts_generic_detection_schema(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    for rel in ("images/train", "images/test", "labels/train"):
        (data_dir / rel).mkdir(parents=True)
    sample = pd.DataFrame({"id": [1], "image_id": ["a.jpg"], "prediction_string": [""]})

    assert detect_yolo_submission_task(data_dir=data_dir, sample_df=sample) is True


def test_detect_yolo_submission_task_accepts_top_level_images_layout(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    for rel in ("train_images", "test_images", "train_labels"):
        (data_dir / rel).mkdir(parents=True)
    sample = pd.DataFrame({"id": [1], "image_id": ["a.jpg"], "prediction_string": [""]})

    assert detect_yolo_submission_task(data_dir=data_dir, sample_df=sample) is True


def test_prepare_ultralytics_dataset_accepts_train_images_layout(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    image_dir = data_dir / "train" / "images"
    test_dir = data_dir / "test" / "images"
    label_dir = data_dir / "train" / "labels"
    image_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "one.jpg").write_bytes(b"image")
    (label_dir / "one.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    layout = find_yolo_data_layout(data_dir)
    dataset_yaml = prepare_ultralytics_dataset(tmp_path / "work", data_dir, ["one.jpg"], [])

    assert layout is not None
    assert layout.train_images_dir == image_dir
    assert (dataset_yaml.parent / "images" / "train" / "one.jpg").exists()
    assert (dataset_yaml.parent / "labels" / "train" / "one.txt").exists()


def test_prepare_ultralytics_dataset_accepts_top_level_images_layout(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    image_dir = data_dir / "train_images"
    test_dir = data_dir / "test_images"
    label_dir = data_dir / "train_labels"
    image_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "one.jpg").write_bytes(b"image")
    (label_dir / "one.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    layout = find_yolo_data_layout(data_dir)
    dataset_yaml = prepare_ultralytics_dataset(tmp_path / "work", data_dir, ["one.jpg"], [])

    assert layout is not None
    assert layout.train_images_dir == image_dir
    assert layout.test_images_dir == test_dir
    assert layout.train_labels_dir == label_dir
    assert (dataset_yaml.parent / "images" / "train" / "one.jpg").exists()
    assert (dataset_yaml.parent / "labels" / "train" / "one.txt").exists()


def test_prepare_ultralytics_dataset_accepts_dataset_relative_train_paths(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    image_dir = data_dir / "images" / "train"
    label_dir = data_dir / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "one.jpg").write_bytes(b"one")
    (image_dir / "two.jpg").write_bytes(b"two")
    (label_dir / "one.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (label_dir / "two.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    dataset_yaml = prepare_ultralytics_dataset(
        tmp_path / "work",
        data_dir,
        ["images/train/one.jpg"],
        ["images/train/two.jpg"],
    )

    assert (dataset_yaml.parent / "images" / "train" / "one.jpg").read_bytes() == b"one"
    assert (dataset_yaml.parent / "images" / "val" / "two.jpg").read_bytes() == b"two"
    assert not (dataset_yaml.parent / "images" / "train" / "images" / "train" / "one.jpg").exists()
    assert (dataset_yaml.parent / "labels" / "train" / "one.txt").read_text(encoding="utf-8").strip()
    assert (dataset_yaml.parent / "labels" / "val" / "two.txt").read_text(encoding="utf-8").strip()


def test_prepare_ultralytics_dataset_discovers_nested_label_class_ids(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    image_dir = data_dir / "images" / "train" / "fold_a"
    label_dir = data_dir / "labels" / "train" / "fold_a"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "one.jpg").write_bytes(b"one")
    (label_dir / "one.txt").write_text("3 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    dataset_yaml = prepare_ultralytics_dataset(tmp_path / "work", data_dir, ["fold_a/one.jpg"], [])

    assert "class_3" in dataset_yaml.read_text(encoding="utf-8")


def test_resolve_yolo_image_reference_accepts_dataset_relative_path(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    image_dir = data_dir / "images" / "test"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "one.jpg"
    image_path.write_bytes(b"image")

    resolved = resolve_yolo_image_reference("images/test/one.jpg", images_dir=image_dir, data_dir=data_dir)

    assert resolved == image_path


def test_compute_map50_95_resolves_nested_dataset_relative_labels(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    image_dir = data_dir / "images" / "train" / "fold_a"
    label_dir = data_dir / "labels" / "train" / "fold_a"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "one.jpg").write_bytes(b"image")
    (label_dir / "one.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    score = compute_map50_95(
        data_dir=data_dir,
        filenames=["images/train/fold_a/one.jpg"],
        dets_by_file={"images/train/fold_a/one.jpg": np.asarray([[0, 0.9, 0.5, 0.5, 0.2, 0.2]], dtype=float)},
    )

    assert score > 0.99


def test_compute_map50_95_ignores_detection_aliases_without_ground_truth(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    image_dir = data_dir / "images" / "train" / "fold_a"
    label_dir = data_dir / "labels" / "train" / "fold_a"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "one.jpg").write_bytes(b"image")
    (label_dir / "one.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    dets = np.asarray([[0, 0.9, 0.5, 0.5, 0.2, 0.2]], dtype=float)

    score = compute_map50_95(
        data_dir=data_dir,
        filenames=["images/train/fold_a/one.jpg"],
        dets_by_file={
            "images/train/fold_a/one.jpg": dets,
            "one.jpg": dets,
        },
    )

    assert score > 0.99


def test_torchvision_training_paths_preserve_nested_image_label_pairs(tmp_path: Path) -> None:
    images_dir = tmp_path / "yolo_ds" / "images" / "train"
    labels_dir = tmp_path / "yolo_ds" / "labels" / "train"
    (images_dir / "fold_a").mkdir(parents=True)
    (images_dir / "fold_b").mkdir(parents=True)
    (labels_dir / "fold_a").mkdir(parents=True)
    (labels_dir / "fold_b").mkdir(parents=True)
    (images_dir / "fold_a" / "same.jpg").write_bytes(b"a")
    (images_dir / "fold_b" / "same.jpg").write_bytes(b"b")
    (labels_dir / "fold_a" / "same.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (labels_dir / "fold_b" / "same.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    image_paths, label_paths = _torchvision_training_image_label_paths(images_dir=images_dir, labels_dir=labels_dir)

    assert [path.relative_to(images_dir).as_posix() for path in image_paths] == [
        "fold_a/same.jpg",
        "fold_b/same.jpg",
    ]
    assert [path.relative_to(labels_dir).as_posix() for path in label_paths] == [
        "fold_a/same.txt",
        "fold_b/same.txt",
    ]


def test_store_detection_result_keeps_full_path_for_duplicate_basenames(tmp_path: Path) -> None:
    out: dict[str, np.ndarray] = {}
    first = tmp_path / "fold_a" / "same.jpg"
    second = tmp_path / "fold_b" / "same.jpg"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)

    _store_detection_result(out, first, np.asarray([[0, 0.91, 0.5, 0.5, 0.2, 0.2]], dtype=float))
    _store_detection_result(out, second, np.asarray([[0, 0.73, 0.5, 0.5, 0.2, 0.2]], dtype=float))

    assert float(out[str(first)][0, 1]) == 0.91
    assert float(out[str(second)][0, 1]) == 0.73
    assert float(out["same.jpg"][0, 1]) == 0.91
