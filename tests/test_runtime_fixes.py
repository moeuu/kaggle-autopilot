from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from kagglebot.paths import CompetitionPaths
from kagglebot.runtime_fixes import (
    extract_candidate_groups,
    extract_missing_module,
    infer_column_mapping,
    maybe_write_column_fill,
    maybe_write_column_map,
    maybe_write_device_coerce,
    maybe_write_object_coerce,
    record_blocked_module,
    save_blocked_modules,
    scan_tabular_headers,
)


@dataclass(frozen=True)
class DummyConfig:
    paths: CompetitionPaths


def _config(tmp_path: Path) -> DummyConfig:
    return DummyConfig(paths=CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"))


def test_maybe_write_column_fill_from_missing_columns_error(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert maybe_write_column_fill(config, "ValueError: test.csv missing columns: ['a', 'b']") is True

    payload = json.loads((config.paths.context_dir / "column_fill.json").read_text(encoding="utf-8"))
    assert payload["files"]["test.csv"] == ["a", "b"]
    assert payload["missing_columns"] == []


def test_maybe_write_column_fill_merges_keyerror_payload(tmp_path: Path) -> None:
    config = _config(tmp_path)
    fill_path = config.paths.context_dir / "column_fill.json"
    fill_path.parent.mkdir(parents=True, exist_ok=True)
    fill_path.write_text(
        json.dumps({"source": "autofix", "created_at": "2026-01-01T00:00:00+00:00", "missing_columns": ["id"]}),
        encoding="utf-8",
    )

    assert maybe_write_column_fill(config, "KeyError: \"['session_id'] not in index\"") is True

    payload = json.loads(fill_path.read_text(encoding="utf-8"))
    assert payload["missing_columns"] == ["id", "session_id"]
    assert "updated_at" in payload


def test_maybe_write_runtime_coerce_artifacts(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert maybe_write_object_coerce(config, "TypeError: numpy.object_ cannot be converted") is True
    assert maybe_write_device_coerce(config, "Expected all tensors to be on the same device") is True

    assert (config.paths.context_dir / "object_coerce.json").exists()
    assert (config.paths.context_dir / "device_coerce.json").exists()


def test_column_map_scans_headers_and_infers_aliases(tmp_path: Path) -> None:
    config = _config(tmp_path)
    data_dir = config.paths.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.csv").write_text("session_identifier,target\n1,0\n", encoding="utf-8")

    assert scan_tabular_headers(data_dir) == {"train.csv": ["session_identifier", "target"]}
    error_text = "could not resolve column session/visit candidates: ['session_id', 'visit_id']"
    assert extract_candidate_groups(error_text)
    assert maybe_write_column_map(config, error_text)

    payload = json.loads((config.paths.context_dir / "column_map.json").read_text(encoding="utf-8"))
    assert payload["mapping"]["session_identifier"] == "session_id"


def test_column_mapping_handles_non_string_group_tokens() -> None:
    columns_by_file = {"train.csv": ["session_id", "target"]}
    groups = [["session_id", None, 123], ["target", object()]]

    mapping = infer_column_mapping(columns_by_file, groups)

    assert mapping["session_id"] == "session_id"
    assert mapping["target"] == "target"


def test_missing_module_and_blocked_modules_round_trip(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"

    assert extract_missing_module("ModuleNotFoundError: No module named 'polars'") == "polars"
    assert record_blocked_module(context_dir, "polars") == ["polars"]
    assert record_blocked_module(context_dir, "polars") == ["polars"]
    save_blocked_modules(context_dir, [])
    assert not (context_dir / "blocked_modules.json").exists()
