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


def _runtime_sequence_binding_matches(sequence_value: object, toggle_value: object) -> bool:
    if not isinstance(sequence_value, (list, tuple)) or not sequence_value:
        return False
    if not isinstance(toggle_value, str) or not toggle_value.strip():
        return False

    parsed_values = [item.strip() for item in toggle_value.split(",")]
    if not parsed_values or any(not item for item in parsed_values):
        return False

    if all(isinstance(item, str) for item in sequence_value):
        return list(sequence_value) == parsed_values

    if not all(not isinstance(item, bool) and isinstance(item, (int, float)) for item in sequence_value):
        return False
    try:
        numeric_sequence = [float(item) for item in sequence_value]
        numeric_toggle = [float(item) for item in parsed_values]
    except (OverflowError, ValueError):
        return False
    return (
        all(math.isfinite(item) for item in numeric_sequence)
        and all(math.isfinite(item) for item in numeric_toggle)
        and numeric_sequence == numeric_toggle
    )


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
    runtime_toggles: dict[str, object] | None = None,
) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            paths.extend(
                find_runtime_hyperparameter_sequence_paths(
                    item,
                    prefix=f"{prefix}.{key}",
                    pipeline_name=pipeline_name,
                    runtime_toggles=runtime_toggles,
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
        toggle_key = prefix.rsplit(".", 1)[-1].upper()
        toggle_value = runtime_toggles.get(toggle_key) if runtime_toggles is not None else None
        if _runtime_sequence_binding_matches(value, toggle_value):
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
    toggles = payload.get("toggles")
    runtime_toggles = toggles if isinstance(toggles, dict) else None

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
            runtime_toggles=runtime_toggles,
        )
        if sequence_paths:
            raise KernelFailedError(
                "Local kernel staged plan contains unresolved hyperparameter sequences for pipeline "
                f"'{name}': {', '.join(sequence_paths)}"
            )
