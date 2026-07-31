from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagglebot.exceptions import KernelFailedError
from kagglebot.kernel_plan_validation import (
    find_runtime_hyperparameter_sequence_paths,
    validate_local_kernel_plan_runtime_hyperparameters,
)


def _write_pipeline_plan(
    tmp_path: Path,
    *,
    name: str,
    key_hyperparameters: dict[str, object],
) -> Path:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "pipelines": [
                    {
                        "name": name,
                        "key_hyperparameters": key_hyperparameters,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return plan_path


def test_find_runtime_hyperparameter_sequence_paths_reports_nested_lists() -> None:
    paths = find_runtime_hyperparameter_sequence_paths(
        {
            "depth": 8,
            "regularization": {
                "dropout": [0.05, 0.1],
                "weight_decay": 0.01,
            },
        }
    )

    assert paths == ["key_hyperparameters.regularization.dropout"]


def test_find_runtime_hyperparameter_sequence_paths_allows_literal_aggregations() -> None:
    paths = find_runtime_hyperparameter_sequence_paths(
        {
            "allowed_aggregations": ["count", "distinct_count"],
            "learning_rate": [0.01, 0.1],
        }
    )

    assert paths == ["key_hyperparameters.learning_rate"]


def test_validate_local_kernel_plan_runtime_hyperparameters_allows_missing_plan(tmp_path: Path) -> None:
    validate_local_kernel_plan_runtime_hyperparameters(tmp_path / "missing-plan.json")


def test_validate_local_kernel_plan_runtime_hyperparameters_rejects_non_object_payload(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("[]", encoding="utf-8")

    with pytest.raises(KernelFailedError, match="must be a JSON object"):
        validate_local_kernel_plan_runtime_hyperparameters(plan_path)


def test_validate_local_kernel_plan_runtime_hyperparameters_rejects_sequences(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "pipelines": [
                    {
                        "name": "pipe_a",
                        "key_hyperparameters": {"dropout": [0.05, 0.1]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(KernelFailedError, match="unresolved hyperparameter sequences"):
        validate_local_kernel_plan_runtime_hyperparameters(plan_path)


def test_validate_local_kernel_plan_runtime_hyperparameters_allows_ranker_runtime_grids(
    tmp_path: Path,
) -> None:
    plan_path = _write_pipeline_plan(
        tmp_path,
        name="mapping_conditioned_catboost_ranker",
        key_hyperparameters={
            "iterations": 700,
            "blend_weight_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
            "temperature_grid": [0.5, 0.75, 1.0, 1.5, 2.0],
        },
    )

    validate_local_kernel_plan_runtime_hyperparameters(plan_path)


def test_validate_local_kernel_plan_runtime_hyperparameters_rejects_other_ranker_sequences(
    tmp_path: Path,
) -> None:
    plan_path = _write_pipeline_plan(
        tmp_path,
        name="mapping_conditioned_catboost_ranker",
        key_hyperparameters={
            "iterations": [500, 700],
            "blend_weight_grid": [0.0, 0.5, 1.0],
            "temperature_grid": [0.5, 1.0, 2.0],
        },
    )

    with pytest.raises(KernelFailedError, match="key_hyperparameters.iterations"):
        validate_local_kernel_plan_runtime_hyperparameters(plan_path)


def test_validate_local_kernel_plan_runtime_hyperparameters_rejects_runtime_grid_on_other_pipeline(
    tmp_path: Path,
) -> None:
    plan_path = _write_pipeline_plan(
        tmp_path,
        name="other_ranker",
        key_hyperparameters={"blend_weight_grid": [0.0, 0.5, 1.0]},
    )

    with pytest.raises(KernelFailedError, match="key_hyperparameters.blend_weight_grid"):
        validate_local_kernel_plan_runtime_hyperparameters(plan_path)


@pytest.mark.parametrize(
    ("grid_name", "invalid_grid"),
    [
        ("blend_weight_grid", [-0.1, 0.5]),
        ("blend_weight_grid", [0.0, 1.1]),
        ("blend_weight_grid", []),
        ("blend_weight_grid", [0.0, "0.5"]),
        ("blend_weight_grid", [0.0, float("nan")]),
        ("blend_weight_grid", [0.0, 0.5, 0.5]),
        ("temperature_grid", [0.0, 1.0]),
        ("temperature_grid", [-0.5, 1.0]),
        ("temperature_grid", []),
        ("temperature_grid", [0.5, "1.0"]),
        ("temperature_grid", [0.5, float("inf")]),
        ("temperature_grid", [0.5, 1.0, 1.0]),
    ],
)
def test_validate_local_kernel_plan_runtime_hyperparameters_rejects_invalid_ranker_runtime_grids(
    tmp_path: Path,
    grid_name: str,
    invalid_grid: list[object],
) -> None:
    plan_path = _write_pipeline_plan(
        tmp_path,
        name="mapping_conditioned_catboost_ranker",
        key_hyperparameters={grid_name: invalid_grid},
    )

    with pytest.raises(KernelFailedError, match=rf"key_hyperparameters\.{grid_name}"):
        validate_local_kernel_plan_runtime_hyperparameters(plan_path)


def test_validate_local_kernel_plan_runtime_hyperparameters_allows_literal_aggregations(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "pipelines": [
                    {
                        "name": "linked_event_motif_graph_50",
                        "key_hyperparameters": {
                            "allowed_aggregations": ["count", "distinct_count"],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    validate_local_kernel_plan_runtime_hyperparameters(plan_path)


@pytest.mark.parametrize(
    "invalid_aggregations",
    [
        [],
        ["count", 1],
        [["count"], "distinct_count"],
    ],
)
def test_validate_local_kernel_plan_runtime_hyperparameters_rejects_invalid_literal_aggregations(
    tmp_path: Path, invalid_aggregations: list[object]
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "pipelines": [
                    {
                        "name": "linked_event_motif_graph_50",
                        "key_hyperparameters": {"allowed_aggregations": invalid_aggregations},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(KernelFailedError, match="key_hyperparameters.allowed_aggregations"):
        validate_local_kernel_plan_runtime_hyperparameters(plan_path)
