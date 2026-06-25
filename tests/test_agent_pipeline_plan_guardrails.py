from __future__ import annotations

import json
from pathlib import Path

from kagglebot.paths import CompetitionPaths
from kagglebot.plan_policy import validate_plan_payload, write_plan_payload


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

    write_plan_payload(paths, _base_payload())
    persisted = json.loads(paths.plan_path.read_text(encoding="utf-8"))

    assert persisted["score_source"] == "cv"
    assert persisted["cv_folds"] == 2
    assert persisted["evaluation_protocol"]["n_folds"] == 2
    assert persisted["evaluation_protocol"]["seeds"] == [42, 2024, 777]
    assert persisted["eval_seeds"] == [42, 2024, 777]
    classifier_flags = {key: persisted["toggles"][key] for key in ("USE_GEOMETRY_CLASSIFIER", "USE_CROP_CLASSIFIER")}
    assert any(bool(value) for value in classifier_flags.values())
    assert persisted["toggles"]["ALLOW_PRETRAINED_WEIGHTS"] is False


def test_write_plan_payload_ignores_invalid_existing_plan(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.plan_path.parent.mkdir(parents=True, exist_ok=True)
    paths.plan_path.write_text("{", encoding="utf-8")

    write_plan_payload(paths, _base_payload())
    persisted = json.loads(paths.plan_path.read_text(encoding="utf-8"))

    assert persisted["target_metric"] == "f1"
    assert persisted["max_iterations"] == 5


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
    write_plan_payload(paths, payload)
    persisted = json.loads(paths.plan_path.read_text(encoding="utf-8"))
    assert persisted["toggles"]["ALLOW_PRETRAINED_WEIGHTS"] is True


def test_write_plan_payload_disables_fast_dev_toggle(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(
        json.dumps(
            {
                "task": "classification",
                "modality": "tabular",
                "train_rows": 5000,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths.rules_md_path.write_text("External data is allowed for this challenge.", encoding="utf-8")

    payload = _base_payload()
    payload["toggles"]["FAST_DEV"] = True
    write_plan_payload(paths, payload)
    persisted = json.loads(paths.plan_path.read_text(encoding="utf-8"))
    assert persisted["toggles"]["FAST_DEV"] is False
    assert persisted["score_source"] == "cv"
    assert persisted["max_iterations"] == 5
    assert persisted["patience"] == 4
    assert persisted["eval_seeds"] == [42, 2024, 777]
    assert persisted["evaluation_protocol"]["seeds"] == [42, 2024, 777]
    assert persisted["stop_policy"]["max_iterations"] == 5


def test_write_plan_payload_caps_long_heavy_runs_to_three_iterations(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(
        json.dumps({"task": "ocr", "modality": "image", "train_rows": 5000}, indent=2),
        encoding="utf-8",
    )
    paths.rules_md_path.write_text("External data is allowed for this challenge.", encoding="utf-8")

    payload = _base_payload()
    payload["time_budget_min"] = None
    payload["max_iterations"] = 5
    payload["stop_policy"] = {"max_iterations": 5, "error_fingerprint_abort": True}
    write_plan_payload(paths, payload)
    persisted = json.loads(paths.plan_path.read_text(encoding="utf-8"))

    assert persisted["max_iterations"] == 3
    assert persisted["stop_policy"]["max_iterations"] == 3


def test_write_plan_payload_forces_training_and_validation(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(
        json.dumps({"task": "classification", "modality": "text", "train_rows": 5000}, indent=2),
        encoding="utf-8",
    )
    paths.rules_md_path.write_text("External data is allowed for this challenge.", encoding="utf-8")

    payload = _base_payload()
    payload["toggles"].update(
        {
            "ENABLE_TRAINING": False,
            "ENABLE_REFERENCE_TRAINING": False,
            "RUN_VALIDATION_GENERATION": False,
            "DISABLE_VALIDATION": True,
            "PACKAGING_ONLY": True,
            "ALLOW_IDENTITY_ADAPTER": True,
        }
    )
    payload["runtime_budget"] = {
        "enable_reference_training": False,
        "run_validation_generation": False,
        "validation_generation_max_samples_rtx3060": 0,
        "max_val_samples": 0,
        "adapter_packaging_only": True,
        "allow_unscored_submission": True,
    }

    write_plan_payload(paths, payload)
    persisted = json.loads(paths.plan_path.read_text(encoding="utf-8"))

    toggles = persisted["toggles"]
    assert toggles["ENABLE_TRAINING"] is True
    assert toggles["ENABLE_REFERENCE_TRAINING"] is True
    assert toggles["RUN_VALIDATION_GENERATION"] is True
    assert toggles["DISABLE_VALIDATION"] is False
    assert toggles["PACKAGING_ONLY"] is False
    assert toggles["ALLOW_IDENTITY_ADAPTER"] is False

    runtime = persisted["runtime_budget"]
    assert runtime["enable_reference_training"] is True
    assert runtime["run_validation_generation"] is True
    assert runtime["validation_generation_max_samples_rtx3060"] >= 64
    assert runtime["max_val_samples"] >= 128
    assert runtime["adapter_packaging_only"] is False
    assert runtime["allow_unscored_submission"] is False


def test_write_plan_payload_forces_cv_for_non_generalizable_score_source(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(
        json.dumps(
            {
                "task": "classification",
                "modality": "tabular",
                "train_rows": 5000,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths.rules_md_path.write_text("External data is allowed for this challenge.", encoding="utf-8")

    payload = _base_payload()
    payload["score_source"] = "auto"
    write_plan_payload(paths, payload)
    persisted = json.loads(paths.plan_path.read_text(encoding="utf-8"))

    assert persisted["score_source"] == "cv"


def test_write_plan_payload_scalarizes_pipeline_key_hyperparameters(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(
        json.dumps(
            {
                "task": "classification",
                "modality": "tabular",
                "train_rows": 5000,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths.rules_md_path.write_text("External data is allowed for this challenge.", encoding="utf-8")

    payload = _base_payload()
    payload["score_source"] = "cv"
    payload["pipelines"] = [
        {
            "name": "pipe1",
            "features": ["basic"],
            "models": ["XGBoost"],
            "key_hyperparameters": {
                "dropout": [0.05, 0.1],
                "optimizer": {"lr": [0.01, 0.005]},
                "empty": [],
            },
            "runtime_memory": "low",
            "failure_modes": ["overfit"],
            "fallbacks": ["smaller"],
        },
        {
            "name": "pipe2",
            "features": ["basic"],
            "models": ["LightGBM"],
            "key_hyperparameters": {"depth": [6, 8]},
            "runtime_memory": "low",
            "failure_modes": ["overfit"],
            "fallbacks": ["shallower"],
        },
    ]

    write_plan_payload(paths, payload)
    persisted = json.loads(paths.plan_path.read_text(encoding="utf-8"))

    assert persisted["pipelines"][0]["key_hyperparameters"] == {
        "dropout": 0.05,
        "optimizer": {"lr": 0.01},
    }
    assert persisted["pipelines"][1]["key_hyperparameters"] == {"depth": 6}


def test_write_plan_payload_sets_default_winner_target_for_leaderboard(tmp_path: Path) -> None:
    paths = CompetitionPaths(slug="demo", artifacts_dir=tmp_path / "artifacts")
    paths.context_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_profile_path.write_text(
        json.dumps(
            {
                "task": "classification",
                "modality": "tabular",
                "train_rows": 5000,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    write_plan_payload(paths, _base_payload())
    persisted = json.loads(paths.plan_path.read_text(encoding="utf-8"))

    assert persisted["target_medal"] == "winner"
    assert persisted["target_rank_percentile"] == 0.001


def test_validate_plan_payload_requires_multi_family_blend_for_high_accuracy_tabular() -> None:
    payload = _base_payload()
    payload["suites"] = [
        {
            "name": "competition_only",
            "train_mode": "competition_only",
            "feature_recipe": "full",
            "lightweight": False,
            "promotion_stage": "full_eval",
        },
        {
            "name": "competition_plus_original",
            "train_mode": "competition_plus_original",
            "feature_recipe": "full",
            "lightweight": False,
            "promotion_stage": "ablation_fast",
        },
        {
            "name": "orig_signal_only",
            "train_mode": "competition_only",
            "feature_recipe": "orig_signal_only",
            "lightweight": True,
            "promotion_stage": "ablation_fast",
        },
    ]
    payload["pipelines"] = [
        {
            "name": "xgb_only",
            "features": ["basic"],
            "models": ["XGBoost"],
            "key_hyperparameters": {"depth": 6},
            "runtime_memory": "medium",
            "failure_modes": ["overfit"],
            "fallbacks": ["smaller xgb"],
        },
        {
            "name": "xgb_only_v2",
            "features": ["basic"],
            "models": ["XGBoost"],
            "key_hyperparameters": {"depth": 8},
            "runtime_memory": "medium",
            "failure_modes": ["overfit"],
            "fallbacks": ["smaller xgb"],
        },
    ]
    profile = {
        "task": "classification",
        "modality": "tabular",
        "tags": ["tabular", "binary"],
        "train_rows": 10000,
        "categorical_columns": ["a", "b", "c"],
        "high_cardinality_columns": ["c"],
    }

    issues = validate_plan_payload(payload, profile=profile)

    assert any("CatBoost" in issue for issue in issues)
    assert any("LightGBM or a second CatBoost/XGBoost variant" in issue for issue in issues)
    assert any("OOF blend" in issue for issue in issues)


def test_validate_plan_payload_requires_suite_aware_ablation_for_high_accuracy_tabular() -> None:
    payload = _base_payload()
    payload["pipelines"] = [
        {
            "name": "catboost_raw",
            "features": ["raw_cat"],
            "models": ["CatBoost"],
            "key_hyperparameters": {"depth": 8},
            "runtime_memory": "medium",
            "failure_modes": ["overfit"],
            "fallbacks": ["shallower"],
        },
        {
            "name": "xgb_encoded",
            "features": ["target_encoding"],
            "models": ["XGBoost"],
            "key_hyperparameters": {"depth": 6},
            "runtime_memory": "medium",
            "failure_modes": ["overfit"],
            "fallbacks": ["smaller"],
        },
        {
            "name": "blend_rank",
            "features": ["oof_predictions"],
            "models": ["weighted rank blend"],
            "key_hyperparameters": {"weights": "search"},
            "runtime_memory": "low",
            "failure_modes": ["no gain"],
            "fallbacks": ["single best"],
        },
    ]
    profile = {
        "task": "classification",
        "modality": "tabular",
        "tags": ["tabular", "binary"],
        "train_rows": 10000,
        "categorical_columns": ["a", "b", "c"],
        "high_cardinality_columns": ["c"],
    }

    issues = validate_plan_payload(payload, profile=profile)

    assert any("suite-aware ablations" in issue for issue in issues)
