from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from kagglebot.local_kernel_metrics_normalization import (
    URBAN_FLOOD_FLAT_FULL_REQUIRED_FILES,
    detect_full_dataset_layout,
    normalize_local_kernel_metrics,
)


def _write_urban_flood_flat_full_files(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in URBAN_FLOOD_FLAT_FULL_REQUIRED_FILES:
        (data_dir / name).write_text("stub\n", encoding="utf-8")


def _write_urban_flood_flat_full_files_with_sample(data_dir: Path, *, sample_name: str) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in URBAN_FLOOD_FLAT_FULL_REQUIRED_FILES - {"sample_submission.csv"}:
        (data_dir / name).write_text("stub\n", encoding="utf-8")
    (data_dir / sample_name).write_text(
        '{"id":1,"target":0}\n' if sample_name.endswith(".jsonl") else "id,target\n1,0\n", encoding="utf-8"
    )


def _write_urban_flood_flat_full_files_with_required_replacements(
    data_dir: Path,
    *,
    replacements: dict[str, str],
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in URBAN_FLOOD_FLAT_FULL_REQUIRED_FILES:
        output_name = replacements.get(name, name)
        (data_dir / output_name).write_text("stub\n", encoding="utf-8")


def test_detect_full_dataset_layout_recognizes_flat_full_data(tmp_path: Path) -> None:
    data_dir = tmp_path / "demo" / "data"
    _write_urban_flood_flat_full_files(data_dir)

    assert detect_full_dataset_layout(data_dir) == "flat_full"


def test_detect_full_dataset_layout_accepts_non_csv_sample_submission(tmp_path: Path) -> None:
    data_dir = tmp_path / "demo" / "data"
    _write_urban_flood_flat_full_files_with_sample(data_dir, sample_name="sample_submission.jsonl")

    assert detect_full_dataset_layout(data_dir) == "flat_full"


def test_detect_full_dataset_layout_accepts_compressed_sample_submission(tmp_path: Path) -> None:
    data_dir = tmp_path / "demo" / "data"
    _write_urban_flood_flat_full_files_with_sample(data_dir, sample_name="sample_submission.csv.gz")

    assert detect_full_dataset_layout(data_dir) == "flat_full"


@pytest.mark.parametrize("suffix", [".orc", ".hdf", ".hdf5"])
def test_detect_full_dataset_layout_accepts_binary_sample_submission(tmp_path: Path, suffix: str) -> None:
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in URBAN_FLOOD_FLAT_FULL_REQUIRED_FILES - {"sample_submission.csv"}:
        (data_dir / name).write_text("stub\n", encoding="utf-8")
    sample = data_dir / f"sample_submission{suffix}"
    frame = pd.DataFrame({"id": [1], "target": [0]})
    if suffix == ".orc":
        frame.to_orc(sample, index=False)
    else:
        frame.to_hdf(sample, key="submission", mode="w", format="table", index=False)

    assert detect_full_dataset_layout(data_dir) == "flat_full"


def test_detect_full_dataset_layout_accepts_non_csv_required_tables(tmp_path: Path) -> None:
    data_dir = tmp_path / "demo" / "data"
    _write_urban_flood_flat_full_files_with_required_replacements(
        data_dir,
        replacements={
            "timesteps.csv": "timesteps.csv.gz",
            "test_2d_nodes_dynamic_all.csv": "test_2d_nodes_dynamic_all.parquet",
        },
    )

    assert detect_full_dataset_layout(data_dir) == "flat_full"


def test_normalize_local_kernel_metrics_promotes_detected_flat_full_data(tmp_path: Path) -> None:
    data_dir = tmp_path / "demo" / "data"
    _write_urban_flood_flat_full_files(data_dir)

    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "metric": "rmse",
                "offline_value": 0.1,
                "score_source": "sample_diagnostic",
            }
        ),
        encoding="utf-8",
    )

    normalized = normalize_local_kernel_metrics(
        slug="demo",
        data_dir=data_dir,
        metrics_path=metrics_path,
        score_source="cv",
    )

    assert normalized == metrics_path
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["score_source"] == "cv"
    assert payload["dataset_kind"] == "full"
    assert payload["dataset_mode"] == "full"
    assert payload["full_dataset_resolved"] is True
    assert payload["data_root_layout"] == "flat_full"
    assert payload["metrics_normalized_by"] == "kernel_runner.local_full_data_guard"


def test_normalize_local_kernel_metrics_keeps_unknown_layout_unchanged(tmp_path: Path) -> None:
    data_dir = tmp_path / "demo" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("id\n2\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n2,0\n", encoding="utf-8")

    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "metric": "rmse",
                "offline_value": 0.1,
                "score_source": "sample_diagnostic",
            }
        ),
        encoding="utf-8",
    )

    normalize_local_kernel_metrics(
        slug="demo",
        data_dir=data_dir,
        metrics_path=metrics_path,
        score_source="cv",
    )

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["score_source"] == "sample_diagnostic"
    assert "full_dataset_resolved" not in payload


def test_normalize_local_kernel_metrics_ignores_invalid_or_non_object_metrics(tmp_path: Path) -> None:
    data_dir = tmp_path / "demo" / "data"
    _write_urban_flood_flat_full_files(data_dir)

    invalid = tmp_path / "invalid_metrics.json"
    invalid.write_text("{", encoding="utf-8")
    assert (
        normalize_local_kernel_metrics(
            slug="demo",
            data_dir=data_dir,
            metrics_path=invalid,
            score_source="cv",
        )
        == invalid
    )
    assert invalid.read_text(encoding="utf-8") == "{"

    array_payload = tmp_path / "array_metrics.json"
    array_payload.write_text("[]", encoding="utf-8")
    assert (
        normalize_local_kernel_metrics(
            slug="demo",
            data_dir=data_dir,
            metrics_path=array_payload,
            score_source="cv",
        )
        == array_payload
    )
    assert array_payload.read_text(encoding="utf-8") == "[]"
