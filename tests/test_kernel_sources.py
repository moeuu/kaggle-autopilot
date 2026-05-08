from __future__ import annotations

import json
from pathlib import Path

from kagglebot.kernel_sources import load_kernel_source_config


def test_load_kernel_source_config_parses_text_runtime_fields(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "kaggle_kernel_sources": {
                    "dataset_sources": ["alice/demo-dataset"],
                    "required_local_seq2seq_pipelines": ["seq2seq_main"],
                },
                "domain_adaptation": {
                    "adapted_checkpoint_hints": ["alice/adapted-model"],
                    "allow_kernel_finetune": True,
                },
                "text_runtime": {
                    "required_aux_inputs": ["data/lexicon.csv", "context/metadata.csv"],
                    "metadata_supervision": "high_precision",
                    "constraint_rewrite_mode": "soft",
                    "group_key_columns": ["document_id", "text_id"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    config = load_kernel_source_config(plan_path)

    assert config.dataset_sources == ("alice/demo-dataset",)
    assert config.required_local_seq2seq_pipelines == ("seq2seq_main",)
    assert config.domain_adaptation.adapted_checkpoint_hints == ("alice/adapted-model",)
    assert config.domain_adaptation.allow_kernel_finetune is True
    assert config.text_runtime.required_aux_inputs == ("data/lexicon.csv", "context/metadata.csv")
    assert config.text_runtime.metadata_supervision == "high_precision"
    assert config.text_runtime.constraint_rewrite_mode == "soft"
    assert config.text_runtime.group_key_columns == ("document_id", "text_id")
    assert config.has_text_runtime_features() is True


def test_load_kernel_source_config_defaults_when_optional_text_fields_missing(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"kaggle_kernel_sources": {"dataset_sources": ["alice/demo"]}}, indent=2),
        encoding="utf-8",
    )

    config = load_kernel_source_config(plan_path)

    assert config.dataset_sources == ("alice/demo",)
    assert config.domain_adaptation.adapted_checkpoint_hints == ()
    assert config.domain_adaptation.allow_kernel_finetune is False
    assert config.text_runtime.required_aux_inputs == ()
    assert config.text_runtime.metadata_supervision == ""
    assert config.has_text_runtime_features() is False
