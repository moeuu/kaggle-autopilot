from __future__ import annotations

from pathlib import Path

import pandas as pd

from kagglebot.training.vision_yolo import prepare_ultralytics_dataset, train_val_split


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
    assert not staged_image.is_symlink()
