from __future__ import annotations

import json
from pathlib import Path

from kagglebot.data_readiness import assess_local_training_data
from kagglebot.paths import CompetitionPaths


def _paths(tmp_path: Path) -> CompetitionPaths:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True)
    paths.data_dir.mkdir(parents=True)
    return paths


def test_nested_profile_training_file_is_ready(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    nested_train = paths.data_dir / "data" / "train_01" / "train.csv"
    nested_train.parent.mkdir(parents=True)
    nested_train.write_text("feature,target\n1,0\n", encoding="utf-8")
    paths.dataset_profile_path.write_text(
        json.dumps({"status": "ok", "train_file": "train.csv"}),
        encoding="utf-8",
    )

    readiness = assess_local_training_data(paths)

    assert readiness.ready is True
    assert readiness.reason == "ready"
    assert readiness.training_sources == (str(nested_train.resolve()),)


def test_nested_training_file_is_found_without_profile_path_hint(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    nested_train = paths.data_dir / "folds" / "train.parquet"
    nested_train.parent.mkdir(parents=True)
    nested_train.write_bytes(b"non-empty")
    paths.dataset_profile_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    readiness = assess_local_training_data(paths)

    assert readiness.ready is True
    assert readiness.training_sources == (str(nested_train.resolve()),)


def test_missing_required_files_without_training_source_is_not_ready(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    (paths.data_dir / "test.csv").write_text("id\n1\n", encoding="utf-8")
    paths.dataset_profile_path.write_text(
        json.dumps({"status": "missing_required_files"}),
        encoding="utf-8",
    )

    readiness = assess_local_training_data(paths)

    assert readiness.ready is False
    assert readiness.reason == "dataset_profile_missing_required_files"
