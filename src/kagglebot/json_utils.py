from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
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


def load_json_array(path: Path, *, errors: str = "strict") -> list[object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors=errors))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if isinstance(payload, list):
        return payload
    return None


def load_jsonl_records(path: Path, *, errors: str = "strict", limit: int | None = None) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    max_records = max(0, int(limit)) if limit is not None else None
    if max_records == 0:
        return records
    try:
        lines = path.read_text(encoding="utf-8", errors=errors).splitlines()
    except (OSError, UnicodeDecodeError):
        return records
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
            if max_records is not None and len(records) >= max_records:
                break
    return records


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


def write_json_array(
    path: Path,
    payload: list[object],
    *,
    ensure_ascii: bool = True,
    sort_keys: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=ensure_ascii, sort_keys=sort_keys),
        encoding="utf-8",
    )


def jsonl_record_text(
    payload: Mapping[str, object],
    *,
    ensure_ascii: bool = True,
    sort_keys: bool = False,
) -> str:
    return json.dumps(payload, ensure_ascii=ensure_ascii, sort_keys=sort_keys) + "\n"


def write_jsonl_records(
    path: Path,
    records: Iterable[Mapping[str, object]],
    *,
    ensure_ascii: bool = True,
    sort_keys: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(jsonl_record_text(record, ensure_ascii=ensure_ascii, sort_keys=sort_keys) for record in records),
        encoding="utf-8",
    )


def append_jsonl_record(
    path: Path,
    payload: Mapping[str, object],
    *,
    ensure_ascii: bool = True,
    sort_keys: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(jsonl_record_text(payload, ensure_ascii=ensure_ascii, sort_keys=sort_keys))
