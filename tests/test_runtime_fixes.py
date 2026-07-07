from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from kagglebot.paths import CompetitionPaths
from kagglebot.runtime_fixes import (
    apply_lightweight_runtime_fix,
    error_strategy_skip_reason,
    extract_candidate_groups,
    extract_missing_module,
    infer_column_mapping,
    is_non_autofixable_runtime_error,
    maybe_write_column_fill,
    maybe_write_column_map,
    maybe_write_device_coerce,
    maybe_write_object_coerce,
    record_blocked_module,
    save_blocked_modules,
    scan_tabular_headers,
    write_lightweight_autofix_note,
)


@dataclass(frozen=True)
class DummyConfig:
    paths: CompetitionPaths


def _config(tmp_path: Path) -> DummyConfig:
    return DummyConfig(paths=CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts"))


def test_error_strategy_skip_reason_detects_deterministic_failures() -> None:
    reason = error_strategy_skip_reason(
        stage="kernel_fix",
        error_text=(
            "ValueError: Kernel source validation failed:\n- Kernel sources do not reference metrics.json output."
        ),
    )
    assert reason is not None
    assert "deterministic" in reason
    reason_data = error_strategy_skip_reason(
        stage="kernel_fix",
        error_text=("FileNotFoundError: Data directory not found: /tmp/artifacts/demo/artifacts/demo/data"),
    )
    assert reason_data is not None
    assert "path resolution" in reason_data
    reason_submission = error_strategy_skip_reason(
        stage="kernel_fix",
        error_text=(
            "ValueError: Kernel source validation failed:\n"
            "- Kernel sources do not reference a supported submission output artifact."
        ),
    )
    assert reason_submission is not None
    assert "submission output" in reason_submission
    reason_submission_generic = error_strategy_skip_reason(
        stage="kernel_fix",
        error_text=(
            "ValueError: Kernel source validation failed:\n"
            "- Kernel sources do not reference submission output artifact."
        ),
    )
    assert reason_submission_generic is not None
    assert "submission output" in reason_submission_generic
    reason_metric = error_strategy_skip_reason(
        stage="autofix",
        error_text=(
            "RuntimeError: Competition metric mismatch persisted after metric-only repairs "
            "(attempts=2, target=auc/maximize, kernel=accuracy/maximize)."
        ),
    )
    assert reason_metric is not None
    assert "metric mismatch" in reason_metric


def test_error_strategy_skip_reason_detects_metric_mismatch_for_submit_autofix() -> None:
    reason = error_strategy_skip_reason(
        stage="submit_autofix",
        error_text=(
            "RuntimeError: Competition metric mismatch persisted after metric-only repairs "
            "(attempts=3, target=auc/maximize, kernel=accuracy/maximize)."
        ),
    )
    assert reason is not None
    assert "metric mismatch" in reason


def test_is_non_autofixable_runtime_error_detects_kernel_first_failures() -> None:
    assert is_non_autofixable_runtime_error(RuntimeError("This mode requires kernel.py")) is True
    assert is_non_autofixable_runtime_error(RuntimeError("Kernel-first training is required")) is True
    assert is_non_autofixable_runtime_error(RuntimeError("transient")) is False


def test_write_lightweight_autofix_note_creates_parent_and_records_reason(tmp_path: Path) -> None:
    note_path = tmp_path / "agent" / "kernel_fix_note.txt"

    note = write_lightweight_autofix_note(
        note_path=note_path,
        artifact_name="column_fill.json",
        reason="missing column error",
    )

    assert note_path.read_text(encoding="utf-8") == note
    assert "column_fill.json created for missing column error" in note
    assert "retry without modifying kernel sources" in note


def test_apply_lightweight_runtime_fix_uses_first_changed_action(tmp_path: Path) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    def broken_action(config: object, error_text: str) -> bool:  # noqa: ARG001
        calls.append("broken")
        raise RuntimeError("ignored")

    def skipped_action(config: object, error_text: str) -> bool:  # noqa: ARG001
        calls.append("skipped")
        return False

    def changed_action(config: object, error_text: str) -> bool:  # noqa: ARG001
        calls.append("changed")
        return True

    note_path = tmp_path / "agent" / "note.txt"
    result = apply_lightweight_runtime_fix(
        config=config,
        error_text="runtime error",
        note_path=note_path,
        actions=(
            ("broken.json", "broken reason", broken_action),
            ("skipped.json", "skipped reason", skipped_action),
            ("changed.json", "changed reason", changed_action),
        ),
    )

    assert result is not None
    assert result.artifact_name == "changed.json"
    assert result.reason == "changed reason"
    assert calls == ["broken", "skipped", "changed"]
    assert "changed.json created for changed reason" in note_path.read_text(encoding="utf-8")


def test_maybe_write_column_fill_from_missing_columns_error(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert maybe_write_column_fill(config, "ValueError: test.csv missing columns: ['a', 'b']") is True

    payload = json.loads((config.paths.context_dir / "column_fill.json").read_text(encoding="utf-8"))
    assert payload["files"]["test.csv"] == ["a", "b"]
    assert payload["missing_columns"] == []


def test_maybe_write_column_fill_from_compressed_file_missing_columns_error(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert maybe_write_column_fill(config, "ValueError: train.csv.gz missing columns: ['a', 'b']") is True

    payload = json.loads((config.paths.context_dir / "column_fill.json").read_text(encoding="utf-8"))
    assert payload["files"]["train.csv.gz"] == ["a", "b"]


def test_maybe_write_column_fill_from_zstd_compressed_file_missing_columns_error(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert maybe_write_column_fill(config, "ValueError: train.csv.zst missing columns: ['a', 'b']") is True

    payload = json.loads((config.paths.context_dir / "column_fill.json").read_text(encoding="utf-8"))
    assert payload["files"]["train.csv.zst"] == ["a", "b"]


def test_maybe_write_column_fill_from_zstd_pickle_missing_columns_error(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert maybe_write_column_fill(config, "ValueError: train.pkl.zst missing columns: ['a', 'b']") is True

    payload = json.loads((config.paths.context_dir / "column_fill.json").read_text(encoding="utf-8"))
    assert payload["files"]["train.pkl.zst"] == ["a", "b"]


def test_maybe_write_column_fill_from_excel_file_missing_columns_error(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert maybe_write_column_fill(config, "ValueError: test.xlsx missing columns: ['session_id']") is True

    payload = json.loads((config.paths.context_dir / "column_fill.json").read_text(encoding="utf-8"))
    assert payload["files"]["test.xlsx"] == ["session_id"]


def test_maybe_write_column_fill_from_sqlite_file_missing_columns_error(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert maybe_write_column_fill(config, "ValueError: competition.sqlite missing columns: ['target']") is True

    payload = json.loads((config.paths.context_dir / "column_fill.json").read_text(encoding="utf-8"))
    assert payload["files"]["competition.sqlite"] == ["target"]


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


def test_scan_tabular_headers_reads_compressed_csv(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with gzip.open(data_dir / "train.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("session_identifier,target\n1,0\n")

    assert scan_tabular_headers(data_dir) == {"train.csv.gz": ["session_identifier", "target"]}


def test_scan_tabular_headers_reads_compressed_txt_with_detected_delimiter(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with gzip.open(data_dir / "train.txt.gz", "wt", encoding="utf-8") as handle:
        handle.write("session_identifier;target\n1;0\n")

    assert scan_tabular_headers(data_dir) == {"train.txt.gz": ["session_identifier", "target"]}


def test_scan_tabular_headers_uses_suffix_default_when_delimiter_sniff_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.psv").write_text("session_identifier|target\n1|0\n", encoding="utf-8")

    def fail_sniff(*args: object, **kwargs: object) -> str:  # noqa: ARG001
        raise OSError("sniff failed")

    monkeypatch.setattr("kagglebot.runtime_fixes.sniff_tabular_text_delimiter", fail_sniff)

    assert scan_tabular_headers(data_dir) == {"train.psv": ["session_identifier", "target"]}


def test_scan_tabular_headers_reads_excel(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"session_identifier": [1], "target": [0]}).to_excel(data_dir / "train.xlsx", index=False)

    assert scan_tabular_headers(data_dir) == {"train.xlsx": ["session_identifier", "target"]}


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
