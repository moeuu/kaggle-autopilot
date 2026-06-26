from __future__ import annotations

import json

import pytest

from kagglebot.json_utils import (
    append_jsonl_record,
    jsonl_record_text,
    load_json_array,
    load_json_object,
    load_json_object_or_empty,
    load_jsonl_records,
    parse_json_object_text,
    read_json_object,
    write_json_array,
    write_json_object,
    write_jsonl_records,
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


def test_parse_json_object_text_returns_only_object_payloads() -> None:
    assert parse_json_object_text('{"ok": true}') == {"ok": True}
    assert parse_json_object_text("[1, 2, 3]") is None
    assert parse_json_object_text("{") is None


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


def test_load_jsonl_records_returns_dict_rows_and_skips_invalid_rows(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"ok": true}',
                "not-json",
                "[1, 2]",
                "",
                '{"value": 3}',
            ]
        ),
        encoding="utf-8",
    )

    assert load_jsonl_records(path) == [{"ok": True}, {"value": 3}]


def test_load_jsonl_records_returns_empty_for_missing_or_unreadable_payload(tmp_path) -> None:
    missing = tmp_path / "missing.jsonl"
    invalid_encoding = tmp_path / "invalid.jsonl"
    invalid_encoding.write_bytes(b'{"ok": true}\xff\n')

    assert load_jsonl_records(missing) == []
    assert load_jsonl_records(invalid_encoding) == []
    assert load_jsonl_records(invalid_encoding, errors="ignore") == [{"ok": True}]


def test_load_jsonl_records_limit_counts_valid_dict_rows(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n".join(
            [
                "not-json",
                '{"first": true}',
                "[1, 2]",
                '{"second": true}',
                '{"third": true}',
            ]
        ),
        encoding="utf-8",
    )

    assert load_jsonl_records(path, limit=2) == [{"first": True}, {"second": True}]
    assert load_jsonl_records(path, limit=0) == []


def test_load_jsonl_records_reverse_returns_recent_valid_rows(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"first": true}',
                "not-json",
                "[1, 2]",
                '{"second": true}',
                '{"third": true}',
            ]
        ),
        encoding="utf-8",
    )

    assert load_jsonl_records(path, reverse=True, limit=2) == [{"third": True}, {"second": True}]


def test_load_jsonl_records_limit_stops_before_later_decode_error(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_bytes(b'{"first": true}\n\xff{"bad": true}\n')

    assert load_jsonl_records(path, limit=1) == [{"first": True}]
    assert load_jsonl_records(path) == [{"first": True}]


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


def test_jsonl_record_text_writes_single_line_with_newline() -> None:
    text = jsonl_record_text({"z": "あ", "a": 1}, ensure_ascii=True, sort_keys=True)

    assert text.endswith("\n")
    assert json.loads(text) == {"a": 1, "z": "あ"}
    assert "\\u3042" in text


def test_write_jsonl_records_creates_parent_and_overwrites(tmp_path) -> None:
    path = tmp_path / "nested" / "records.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"old": true}\n', encoding="utf-8")

    write_jsonl_records(path, [{"b": 2}, {"a": 1}], sort_keys=True)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"b": 2}, {"a": 1}]


def test_write_jsonl_records_accepts_generator(tmp_path) -> None:
    path = tmp_path / "records.jsonl"

    write_jsonl_records(path, ({"row": idx} for idx in range(3)))

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"row": 0}, {"row": 1}, {"row": 2}]


def test_append_jsonl_record_creates_parent_and_appends(tmp_path) -> None:
    path = tmp_path / "nested" / "records.jsonl"

    append_jsonl_record(path, {"b": 2}, sort_keys=True)
    append_jsonl_record(path, {"a": 1}, sort_keys=True)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"b": 2}, {"a": 1}]
