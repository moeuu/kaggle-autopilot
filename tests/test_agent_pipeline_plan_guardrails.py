from __future__ import annotations

import json
from pathlib import Path

from kagglebot.orchestrator.agent_pipeline import _write_plan_payload
from kagglebot.paths import CompetitionPaths


def _base_payload() -> dict[str, object]:
    return {
        "target_metric": "f1",
        "target_direction": "maximize",
        "target_score": 0.5,
        "score_source": "holdout",
        "holdout_frac": 0.2,
        "cv_folds": 5,
        "seed": 42,
        "max_iterations": 3,
        "patience": 2,
        "min_improvement": 0.0,
        "pipelines": [{"name": "pipe1"}, {"name": "pipe2"}],
        "toggles": {
            "USE_GEOMETRY_CLASSIFIER": False,
            "USE_CROP_CLASSIFIER": False,
            "ALLOW_PRETRAINED_WEIGHTS": False,
        },
        "evaluation_protocol": {
            "cv_type": "stratified",
            "n_folds": 5,
            "seeds": [42],
            "primary_metric": "f1",
        },
        "stop_policy": {"max_iterations": 3, "error_fingerprint_abort": True},
    }


def test_write_plan_payload_applies_imbalance_guardrails(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(
        json.dumps(
            {
                "task": "classification",
                "modality": "image",
                "train_rows": 651,
                "target_stats": {"top_class_ratio": 0.9923},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths.rules_md_path.write_text("External data is strictly prohibited.", encoding="utf-8")

    _write_plan_payload(paths, _base_payload())
    persisted = json.loads(paths.plan_path.read_text(encoding="utf-8"))

    assert persisted["score_source"] == "cv"
    assert persisted["cv_folds"] == 2
    assert persisted["evaluation_protocol"]["n_folds"] == 2
    assert persisted["evaluation_protocol"]["seeds"] == [42, 2024, 777]
    assert persisted["eval_seeds"] == [42, 2024, 777]
    classifier_flags = {key: persisted["toggles"][key] for key in ("USE_GEOMETRY_CLASSIFIER", "USE_CROP_CLASSIFIER")}
    assert any(bool(value) for value in classifier_flags.values())
    assert persisted["toggles"]["ALLOW_PRETRAINED_WEIGHTS"] is False


def test_write_plan_payload_enables_pretrained_when_rules_allow(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(
        json.dumps(
            {
                "task": "classification",
                "modality": "image",
                "train_rows": 2000,
                "target_stats": {"top_class_ratio": 0.7},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths.rules_md_path.write_text("External data is allowed for this challenge.", encoding="utf-8")

    payload = _base_payload()
    payload["score_source"] = "cv"
    _write_plan_payload(paths, payload)
    persisted = json.loads(paths.plan_path.read_text(encoding="utf-8"))
    assert persisted["toggles"]["ALLOW_PRETRAINED_WEIGHTS"] is True
