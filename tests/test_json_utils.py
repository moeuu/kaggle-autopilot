from __future__ import annotations

import json

import pytest

from kagglebot.json_utils import (
    load_json_array,
    load_json_object,
    load_json_object_or_empty,
    read_json_object,
    write_json_array,
    write_json_object,
)


def test_load_json_object_returns_dict_payload(tmp_path) -> None:
    path = tmp_path / "payload.json"
    path.write_text('{"ok": true}', encoding="utf-8")

    assert load_json_object(path) == {"ok": True}


def test_read_json_object_returns_dict_payload(tmp_path) -> None:
    path = tmp_path / "payload.json"
    path.write_text('{"ok": true}', encoding="utf-8")

    assert read_json_object(path) == {"ok": True}


def test_read_json_object_rejects_non_object_payload(tmp_path) -> None:
    path = tmp_path / "payload.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        read_json_object(path)


def test_load_json_object_rejects_missing_invalid_or_non_object_payload(tmp_path) -> None:
    assert load_json_object(tmp_path / "missing.json") is None

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert load_json_object(invalid) is None

    array_payload = tmp_path / "array.json"
    array_payload.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_json_object(array_payload) is None


def test_load_json_object_supports_read_errors_policy(tmp_path) -> None:
    payload = tmp_path / "payload.json"
    payload.write_bytes(b'{"ok": true}\xff')

    assert load_json_object(payload) is None
    assert load_json_object(payload, errors="ignore") == {"ok": True}


def test_load_json_object_or_empty_returns_empty_dict_for_missing_or_invalid_payload(tmp_path) -> None:
    assert load_json_object_or_empty(tmp_path / "missing.json") == {}

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert load_json_object_or_empty(invalid) == {}


def test_load_json_array_returns_list_payload(tmp_path) -> None:
    path = tmp_path / "payload.json"
    path.write_text('[1, "x"]', encoding="utf-8")

    assert load_json_array(path) == [1, "x"]


def test_load_json_array_rejects_missing_invalid_or_non_array_payload(tmp_path) -> None:
    assert load_json_array(tmp_path / "missing.json") is None

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert load_json_array(invalid) is None

    object_payload = tmp_path / "object.json"
    object_payload.write_text('{"ok": true}', encoding="utf-8")
    assert load_json_array(object_payload) is None


def test_write_json_object_creates_parent_and_writes_indented_json(tmp_path) -> None:
    path = tmp_path / "nested" / "payload.json"

    write_json_object(path, {"z": "あ", "a": 1}, ensure_ascii=True, sort_keys=True)

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "z": "あ"}
    assert "\\u3042" in path.read_text(encoding="utf-8")


def test_write_json_array_creates_parent_and_writes_indented_json(tmp_path) -> None:
    path = tmp_path / "nested" / "payload.json"

    write_json_array(path, ["z", "あ"], ensure_ascii=True)

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == ["z", "あ"]
    assert "\\u3042" in path.read_text(encoding="utf-8")
