from __future__ import annotations

import pandas as pd

from kagglebot.training.vision_yolo import train_val_split


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
