from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagglebot.exceptions import KernelFailedError
from kagglebot.local_kernel_aux_inputs import (
    relative_aux_stage_path,
    resolve_required_aux_input,
    stage_local_kernel_aux_inputs,
    stage_local_path_alias,
)


def test_resolve_required_aux_input_prefers_data_context_then_root(tmp_path: Path) -> None:
    competition_dir = tmp_path / "demo"
    (competition_dir / "data").mkdir(parents=True)
    (competition_dir / "context").mkdir()
    data_file = competition_dir / "data" / "lexicon.csv"
    context_file = competition_dir / "context" / "metadata.csv"
    root_file = competition_dir / "labels.csv"
    data_file.write_text("token,norm\n", encoding="utf-8")
    context_file.write_text("id,value\n", encoding="utf-8")
    root_file.write_text("id,label\n", encoding="utf-8")

    assert resolve_required_aux_input(competition_dir=competition_dir, spec="lexicon.csv") == data_file.resolve()
    assert resolve_required_aux_input(competition_dir=competition_dir, spec="metadata.csv") == context_file.resolve()
    assert resolve_required_aux_input(competition_dir=competition_dir, spec="labels.csv") == root_file.resolve()
    assert (
        resolve_required_aux_input(competition_dir=competition_dir, spec="context/metadata.csv")
        == context_file.resolve()
    )
    assert resolve_required_aux_input(competition_dir=competition_dir, spec="missing.csv") is None


def test_relative_aux_stage_path_keeps_competition_relative_paths(tmp_path: Path) -> None:
    competition_dir = tmp_path / "demo"
    source_path = competition_dir / "context" / "metadata.csv"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("id,value\n", encoding="utf-8")

    assert relative_aux_stage_path(
        competition_dir=competition_dir, source_path=source_path, spec="context/metadata.csv"
    ) == Path("context/metadata.csv")


def test_stage_local_path_alias_replaces_stale_file(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    target = tmp_path / "stage" / "source.csv"
    source.write_text("fresh\n", encoding="utf-8")
    target.parent.mkdir(parents=True)
    target.write_text("stale\n", encoding="utf-8")

    stage_local_path_alias(source_path=source, target_path=target)

    assert target.read_text(encoding="utf-8") == "fresh\n"


def test_stage_local_kernel_aux_inputs_sets_env_and_stages_files(tmp_path: Path) -> None:
    base_dir = tmp_path / "artifacts"
    slug = "demo"
    competition_dir = base_dir / slug
    (competition_dir / "data").mkdir(parents=True)
    (competition_dir / "context").mkdir()
    (competition_dir / "data" / "lexicon.csv").write_text("token,norm\n", encoding="utf-8")
    (competition_dir / "context" / "metadata.csv").write_text("id,value\n", encoding="utf-8")
    (competition_dir / "plan.json").write_text(
        json.dumps(
            {
                "domain_adaptation": {"allow_kernel_finetune": True},
                "text_runtime": {
                    "required_aux_inputs": ["data/lexicon.csv", "context/metadata.csv"],
                    "metadata_supervision": "high_precision",
                    "constraint_rewrite_mode": "soft",
                    "group_key_columns": ["document_id", "scribe_id"],
                },
            }
        ),
        encoding="utf-8",
    )
    kernel_stage_dir = tmp_path / "stage"

    env, notes = stage_local_kernel_aux_inputs(base_dir=base_dir, slug=slug, kernel_stage_dir=kernel_stage_dir)

    assert env == {
        "KAGGLEBOT_ALLOW_KERNEL_FINETUNE": "1",
        "KAGGLEBOT_TEXT_METADATA_SUPERVISION": "high_precision",
        "KAGGLEBOT_TEXT_CONSTRAINT_REWRITE_MODE": "soft",
        "KAGGLEBOT_TEXT_GROUP_KEYS": "document_id,scribe_id",
        "KAGGLEBOT_AUX_INPUT_ROOT": str(kernel_stage_dir / "aux_inputs"),
        "KAGGLEBOT_REQUIRED_AUX_INPUTS": "data/lexicon.csv,context/metadata.csv",
    }
    assert notes == ["staged 2 text aux input(s)"]
    assert (kernel_stage_dir / "aux_inputs" / "data" / "lexicon.csv").exists()
    assert (kernel_stage_dir / "aux_inputs" / "context" / "metadata.csv").exists()


def test_stage_local_kernel_aux_inputs_reports_missing_required_files(tmp_path: Path) -> None:
    base_dir = tmp_path / "artifacts"
    slug = "demo"
    competition_dir = base_dir / slug
    competition_dir.mkdir(parents=True)
    (competition_dir / "plan.json").write_text(
        json.dumps({"text_runtime": {"required_aux_inputs": ["missing.csv"]}}),
        encoding="utf-8",
    )

    with pytest.raises(KernelFailedError, match="Required text runtime aux inputs could not be resolved"):
        stage_local_kernel_aux_inputs(base_dir=base_dir, slug=slug, kernel_stage_dir=tmp_path / "stage")
