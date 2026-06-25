from __future__ import annotations

import json
from pathlib import Path


def read_json_object(path: Path, *, errors: str = "strict") -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8", errors=errors))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def load_json_object(path: Path, *, errors: str = "strict") -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return read_json_object(path, errors=errors)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None


def load_json_object_or_empty(path: Path, *, errors: str = "strict") -> dict[str, object]:
    return load_json_object(path, errors=errors) or {}


def write_json_object(
    path: Path,
    payload: dict[str, object],
    *,
    ensure_ascii: bool = True,
    sort_keys: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=ensure_ascii, sort_keys=sort_keys),
        encoding="utf-8",
    )
