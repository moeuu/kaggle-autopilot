from __future__ import annotations

import json
from pathlib import Path

from kagglebot.exceptions import KernelFailedError
from kagglebot.json_utils import read_json_object


def find_runtime_hyperparameter_sequence_paths(value: object, *, prefix: str = "key_hyperparameters") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            paths.extend(find_runtime_hyperparameter_sequence_paths(item, prefix=f"{prefix}.{key}"))
        return paths
    if isinstance(value, (list, tuple)):
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
        sequence_paths = find_runtime_hyperparameter_sequence_paths(key_hyperparameters)
        if sequence_paths:
            raise KernelFailedError(
                "Local kernel staged plan contains unresolved hyperparameter sequences for pipeline "
                f"'{name}': {', '.join(sequence_paths)}"
            )
