from __future__ import annotations

from kagglebot.json_utils import load_json_object


def test_load_json_object_returns_dict_payload(tmp_path) -> None:
    path = tmp_path / "payload.json"
    path.write_text('{"ok": true}', encoding="utf-8")

    assert load_json_object(path) == {"ok": True}


def test_load_json_object_rejects_missing_invalid_or_non_object_payload(tmp_path) -> None:
    assert load_json_object(tmp_path / "missing.json") is None

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert load_json_object(invalid) is None

    array_payload = tmp_path / "array.json"
    array_payload.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_json_object(array_payload) is None
