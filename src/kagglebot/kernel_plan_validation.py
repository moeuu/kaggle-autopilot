from __future__ import annotations

import json
import math
from pathlib import Path

from kagglebot.exceptions import KernelFailedError
from kagglebot.json_utils import read_json_object

_LITERAL_SEQUENCE_PATH_SUFFIXES = {
    "key_hyperparameters.allowed_aggregations",
}
_ALLOWED_RUNTIME_SEQUENCE_PATHS = {
    (
        "mapping_conditioned_catboost_ranker",
        "key_hyperparameters.blend_weight_grid",
    ),
    (
        "mapping_conditioned_catboost_ranker",
        "key_hyperparameters.temperature_grid",
    ),
}


def _is_allowed_runtime_sequence(*, pipeline_name: str | None, path: str, value: object) -> bool:
    if (pipeline_name, path) not in _ALLOWED_RUNTIME_SEQUENCE_PATHS:
        return False
    if not isinstance(value, list) or not value:
        return False

    numeric_values: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return False
        try:
            numeric_item = float(item)
        except (OverflowError, ValueError):
            return False
        if not math.isfinite(numeric_item):
            return False
        numeric_values.append(numeric_item)

    if len(set(numeric_values)) != len(numeric_values):
        return False
    if path == "key_hyperparameters.blend_weight_grid":
        return all(0.0 <= item <= 1.0 for item in numeric_values)
    return all(item > 0.0 for item in numeric_values)


def find_runtime_hyperparameter_sequence_paths(
    value: object,
    *,
    prefix: str = "key_hyperparameters",
    pipeline_name: str | None = None,
) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            paths.extend(
                find_runtime_hyperparameter_sequence_paths(
                    item,
                    prefix=f"{prefix}.{key}",
                    pipeline_name=pipeline_name,
                )
            )
        return paths
    if isinstance(value, (list, tuple)):
        is_literal_string_sequence = (
            any(prefix.endswith(suffix) for suffix in _LITERAL_SEQUENCE_PATH_SUFFIXES)
            and bool(value)
            and all(isinstance(item, str) and item.strip() for item in value)
        )
        if is_literal_string_sequence or _is_allowed_runtime_sequence(
            pipeline_name=pipeline_name,
            path=prefix,
            value=value,
        ):
            return paths
        paths.append(prefix)
    return paths


def validate_local_kernel_plan_runtime_hyperparameters(plan_path: Path) -> None:
    if not plan_path.exists():
        return
    try:
        payload = read_json_object(plan_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KernelFailedError(f"Local kernel staged plan is unreadable: {plan_path} ({exc})") from exc
    except ValueError as exc:
        raise KernelFailedError(f"Local kernel staged plan must be a JSON object: {plan_path}") from exc

    pipelines = payload.get("pipelines")
    if not isinstance(pipelines, list):
        return

    for index, item in enumerate(pipelines):
        if not isinstance(item, dict) or "key_hyperparameters" not in item:
            continue
        name = str(item.get("name") or f"pipeline_{index + 1}")
        key_hyperparameters = item.get("key_hyperparameters")
        if not isinstance(key_hyperparameters, dict):
            raise KernelFailedError(
                f"Local kernel staged plan has non-object key_hyperparameters for pipeline '{name}'."
            )
        sequence_paths = find_runtime_hyperparameter_sequence_paths(
            key_hyperparameters,
            pipeline_name=name,
        )
        if sequence_paths:
            raise KernelFailedError(
                "Local kernel staged plan contains unresolved hyperparameter sequences for pipeline "
                f"'{name}': {', '.join(sequence_paths)}"
            )
