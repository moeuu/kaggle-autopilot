from __future__ import annotations

import stat
from pathlib import Path

from kagglebot.artifact_io import copy_artifact_if_needed, same_stem_tabular_artifact_filenames


def test_same_stem_tabular_artifact_filenames_expands_submission_suffixes() -> None:
    candidates = same_stem_tabular_artifact_filenames("oof_predictions.csv")

    assert candidates[0] == "oof_predictions.csv"
    assert "oof_predictions.jsonl" in candidates
    assert "oof_predictions.parquet" in candidates
    assert "oof_predictions.csv.gz" in candidates


def test_same_stem_tabular_artifact_filenames_handles_compound_suffix() -> None:
    candidates = same_stem_tabular_artifact_filenames("oof_predictions.jsonl.gz")

    assert candidates[0] == "oof_predictions.jsonl.gz"
    assert "oof_predictions.csv" in candidates
    assert "oof_predictions.parquet" in candidates
    assert candidates.index("oof_predictions.csv.gz") < candidates.index("oof_predictions.csv")
    assert candidates.index("oof_predictions.jsonl.zst") < candidates.index("oof_predictions.jsonl")


def test_same_stem_tabular_artifact_filenames_preserves_json_lines_alias_suffixes() -> None:
    jsonlines_candidates = same_stem_tabular_artifact_filenames("oof_predictions.jsonlines.zst")
    ndjson_candidates = same_stem_tabular_artifact_filenames("oof_predictions.ndjson.xz")

    assert jsonlines_candidates[0] == "oof_predictions.jsonlines.zst"
    assert "oof_predictions.ndjson.zst" in jsonlines_candidates
    assert "oof_predictions.jsonl.zst" in jsonlines_candidates
    assert ndjson_candidates[0] == "oof_predictions.ndjson.xz"
    assert "oof_predictions.jsonlines.xz" in ndjson_candidates
    assert "oof_predictions.jsonl.xz" in ndjson_candidates


def test_same_stem_tabular_artifact_filenames_leaves_non_tabular_name_unchanged() -> None:
    assert same_stem_tabular_artifact_filenames("model.onnx") == ("model.onnx",)


def test_copy_artifact_replaces_read_only_destination(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    destination = tmp_path / "mirror" / "kernel.py"
    source.write_text("new content\n", encoding="utf-8")
    source.chmod(0o444)
    destination.parent.mkdir()
    destination.write_text("old content\n", encoding="utf-8")
    destination.chmod(0o444)

    assert copy_artifact_if_needed(source=source, destination=destination) == destination
    assert destination.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert stat.S_IMODE(destination.stat().st_mode) & stat.S_IWUSR

    assert copy_artifact_if_needed(source=source, destination=destination) == destination
    assert destination.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert stat.S_IMODE(destination.stat().st_mode) & stat.S_IWUSR
