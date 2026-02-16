from __future__ import annotations

from pathlib import Path

import pandas as pd

from kagglebot.solver.io import find_competition_files


def test_find_competition_files_synthesizes_from_assets(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.DataFrame({"id": ["a1", "a2"], "target": [0.0, 0.0]})
    sample.to_csv(data_dir / "sample_submission.csv", index=False)
    labels = pd.DataFrame({"id": ["a1", "a2"], "target": [1, 0]})
    labels.to_csv(data_dir / "train_labels.csv", index=False)

    assets = data_dir / "images"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "a1.jpg").write_bytes(b"fake")
    (assets / "a2.jpg").write_bytes(b"fake")

    train_path, test_path, sample_path = find_competition_files(data_dir)

    assert sample_path.name == "sample_submission.csv"
    assert train_path.name == "train_synth.csv"
    assert test_path.name == "test_synth.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    assert "asset_path" in train.columns
    assert "asset_path" in test.columns


def test_find_competition_files_synthesizes_sample_submission_from_context(tmp_path: Path) -> None:
    base_dir = tmp_path / "comp"
    data_dir = base_dir / "data"
    context_dir = base_dir / "context"
    data_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)

    (context_dir / "submission_format.md").write_text(
        "\n".join(
            [
                "## Submission File",
                "```",
                "filename,right_place,prediction_string",
                "0.jpg,1,-",
                "```",
            ]
        ),
        encoding="utf-8",
    )

    labels = pd.DataFrame({"filename": ["0.jpg", "1.jpg"], "right_place": ["FALSE", "TRUE"]})
    labels.to_csv(data_dir / "train_labels.csv", index=False)

    images_train = data_dir / "images" / "train"
    images_test = data_dir / "images" / "test"
    images_train.mkdir(parents=True, exist_ok=True)
    images_test.mkdir(parents=True, exist_ok=True)
    (images_train / "0.jpg").write_bytes(b"fake")
    (images_train / "1.jpg").write_bytes(b"fake")
    (images_test / "2.jpg").write_bytes(b"fake")

    train_path, test_path, sample_path = find_competition_files(data_dir)

    assert sample_path.name == "sample_submission_synth.csv"
    sample = pd.read_csv(sample_path)
    assert sample.columns.tolist() == ["filename", "right_place", "prediction_string"]
    assert sample["filename"].tolist() == ["2.jpg"]
    assert sample["right_place"].tolist() == [0]
    assert sample["prediction_string"].tolist() == ["-"]

    assert train_path.name == "train_synth.csv"
    assert test_path.name == "test_synth.csv"


def test_find_competition_files_uses_ancestor_context_for_nested_data_dir(tmp_path: Path) -> None:
    base_dir = tmp_path / "comp"
    data_dir = base_dir / "data" / "raw"
    context_dir = base_dir / "context"
    data_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)

    (context_dir / "submission_format.md").write_text(
        "\n".join(
            [
                "## Submission File",
                "```",
                "filename,right_place,prediction_string",
                "```",
            ]
        ),
        encoding="utf-8",
    )

    labels = pd.DataFrame({"filename": ["0.jpg", "1.jpg"], "right_place": ["FALSE", "TRUE"]})
    labels.to_csv(data_dir / "train_labels.csv", index=False)

    images_train = data_dir / "images" / "train"
    images_test = data_dir / "images" / "test"
    images_train.mkdir(parents=True, exist_ok=True)
    images_test.mkdir(parents=True, exist_ok=True)
    (images_train / "0.jpg").write_bytes(b"fake")
    (images_train / "1.jpg").write_bytes(b"fake")
    (images_test / "2.jpg").write_bytes(b"fake")

    train_path, test_path, sample_path = find_competition_files(data_dir)

    assert sample_path.name == "sample_submission_synth.csv"
    assert train_path.name == "train_synth.csv"
    assert test_path.name == "test_synth.csv"
