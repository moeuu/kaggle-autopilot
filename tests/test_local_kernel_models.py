from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagglebot.exceptions import KernelFailedError
from kagglebot.local_kernel_models import (
    discover_local_model_dirs,
    model_ref_aliases,
    resolve_local_model_dir_for_hint,
    sanitize_local_model_stage_name,
    stage_local_kernel_models,
)


def _write_model_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_text("weights", encoding="utf-8")
    return path


def test_discover_local_model_dirs_finds_loadable_kernel_models(tmp_path: Path) -> None:
    base_dir = tmp_path / "artifacts"
    slug = "demo"
    model_dir = _write_model_dir(base_dir / slug / "kernel" / "models" / "demo-model")
    (base_dir / slug / "kernel" / "models" / "not-loadable").mkdir(parents=True)

    discovered = discover_local_model_dirs(base_dir=base_dir, slug=slug)

    assert model_dir in discovered
    assert all(path.name != "not-loadable" for path in discovered)


def test_resolve_local_model_dir_for_hint_prefers_specific_alias(tmp_path: Path) -> None:
    weak = _write_model_dir(tmp_path / "models" / "google" / "demo")
    strong = _write_model_dir(tmp_path / "models" / "google" / "final-byt5-demo")

    resolved = resolve_local_model_dir_for_hint(
        hint="google/final-byt5-demo",
        candidate_dirs=[weak, strong],
    )

    assert resolved == strong


def test_stage_local_kernel_models_sets_generic_and_pipeline_env(tmp_path: Path) -> None:
    base_dir = tmp_path / "artifacts"
    slug = "demo"
    plan_path = base_dir / slug / "plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(
            {
                "kaggle_kernel_sources": {
                    "model_sources": ["generic-demo-model"],
                    "pipeline_model_hints": {"seq2seq strong": ["pipeline-demo-model"]},
                }
            }
        ),
        encoding="utf-8",
    )
    _write_model_dir(base_dir / slug / "kernel" / "models" / "generic-demo-model")
    _write_model_dir(base_dir / slug / "context" / "reference_inputs" / "pipeline-demo-model")
    kernel_stage_dir = tmp_path / "stage"

    env, notes = stage_local_kernel_models(base_dir=base_dir, slug=slug, kernel_stage_dir=kernel_stage_dir)

    assert env["KAGGLEBOT_MODEL_PATHS"] == str(kernel_stage_dir / "models" / "generic_demo_model")
    assert env["KAGGLEBOT_MODEL_PATHS_SEQ2SEQ_STRONG"] == str(kernel_stage_dir / "models" / "pipeline_demo_model")
    assert (kernel_stage_dir / "models" / "generic_demo_model" / "config.json").exists()
    assert (kernel_stage_dir / "models" / "pipeline_demo_model" / "model.safetensors").exists()
    assert notes == [
        "staged 1 generic local model source(s)",
        "staged 1 local model source(s) for pipeline=seq2seq strong",
    ]


def test_stage_local_kernel_models_requires_declared_seq2seq_sources(tmp_path: Path) -> None:
    base_dir = tmp_path / "artifacts"
    slug = "demo"
    plan_path = base_dir / slug / "plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(
            {
                "kaggle_kernel_sources": {
                    "pipeline_model_hints": {"seq2seq": ["missing-model"]},
                    "required_local_seq2seq_pipelines": ["seq2seq"],
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(KernelFailedError, match="Required local seq2seq model sources"):
        stage_local_kernel_models(base_dir=base_dir, slug=slug, kernel_stage_dir=tmp_path / "stage")


def test_model_hint_aliases_and_stage_name_are_stable() -> None:
    assert model_ref_aliases("owner/final-byt5-demo")[:4] == (
        "owner/final-byt5-demo",
        "owner--final-byt5-demo",
        "owner-final-byt5-demo",
        "final-byt5-demo",
    )
    assert sanitize_local_model_stage_name("owner/final-byt5-demo") == "owner_final_byt5_demo"
    assert sanitize_local_model_stage_name("///") == "model"
