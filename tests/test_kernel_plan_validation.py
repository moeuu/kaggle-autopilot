from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagglebot.exceptions import KernelFailedError
from kagglebot.kernel_plan_validation import (
    find_runtime_hyperparameter_sequence_paths,
    validate_local_kernel_plan_runtime_hyperparameters,
)


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
