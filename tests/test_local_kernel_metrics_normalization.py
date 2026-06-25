from __future__ import annotations

import json
from pathlib import Path

from kagglebot.local_kernel_metrics_normalization import (
    URBAN_FLOOD_FLAT_FULL_REQUIRED_FILES,
    normalize_local_kernel_metrics,
)


def _write_urban_flood_flat_full_files(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in URBAN_FLOOD_FLAT_FULL_REQUIRED_FILES:
        (data_dir / name).write_text("stub\n", encoding="utf-8")


def test_normalize_local_kernel_metrics_promotes_urban_flood_flat_full_data(tmp_path: Path) -> None:
    data_dir = tmp_path / "urban-flood-modelling" / "data"
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
        slug="urban-flood-modelling",
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
    assert payload["metrics_normalized_by"] == "kernel_runner.local_full_data_guard"


def test_normalize_local_kernel_metrics_keeps_other_slugs_unchanged(tmp_path: Path) -> None:
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
    data_dir = tmp_path / "urban-flood-modelling" / "data"
    _write_urban_flood_flat_full_files(data_dir)

    invalid = tmp_path / "invalid_metrics.json"
    invalid.write_text("{", encoding="utf-8")
    assert (
        normalize_local_kernel_metrics(
            slug="urban-flood-modelling",
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
            slug="urban-flood-modelling",
            data_dir=data_dir,
            metrics_path=array_payload,
            score_source="cv",
        )
        == array_payload
    )
    assert array_payload.read_text(encoding="utf-8") == "[]"
