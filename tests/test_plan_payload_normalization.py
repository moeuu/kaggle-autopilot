from __future__ import annotations

import kagglebot.orchestrator.agent_pipeline as agent_pipeline


def test_normalize_plan_payload_injects_pipeline_names() -> None:
    payload = {
        "pipelines": [
            {
                "features": ["a"],
                "models": ["LightGBM Classifier"],
                "key_hyperparameters": {},
                "runtime_memory": "low",
                "failure_modes": [],
                "fallbacks": [],
            },
            {
                "name": "already_set",
                "features": [],
                "models": [],
                "key_hyperparameters": {},
                "runtime_memory": "low",
                "failure_modes": [],
                "fallbacks": [],
            },
            {
                "features": [],
                "models": [],
                "key_hyperparameters": {},
                "runtime_memory": "low",
                "failure_modes": [],
                "fallbacks": [],
            },
        ]
    }

    normalized = agent_pipeline._normalize_plan_payload(payload)  # noqa: SLF001
    pipelines = normalized["pipelines"]
    assert isinstance(pipelines, list)
    assert pipelines[0]["name"] == "lightgbm_classifier"
    assert pipelines[1]["name"] == "already_set"
    assert pipelines[2]["name"] == "pipeline_3"


def test_validate_plan_payload_tolerates_missing_pipeline_names() -> None:
    payload = {
        "target_metric": "roc_auc",
        "target_direction": "maximize",
        "target_score": 0.9,
        "score_source": "cv",
        "holdout_frac": 0.2,
        "cv_folds": 5,
        "seed": 42,
        "max_iterations": 3,
        "patience": 2,
        "min_improvement": 0.0,
        "pipelines": [
            {
                "features": ["a"],
                "models": ["XGBoost"],
                "key_hyperparameters": {},
                "runtime_memory": "low",
                "failure_modes": [],
                "fallbacks": [],
            },
            {
                "features": ["b"],
                "models": ["LightGBM"],
                "key_hyperparameters": {},
                "runtime_memory": "low",
                "failure_modes": [],
                "fallbacks": [],
            },
        ],
        "toggles": {"FAST_DEV": False},
        "evaluation_protocol": {
            "cv_type": "StratifiedKFold",
            "n_folds": 5,
            "seeds": [42, 2024, 3407],
            "primary_metric": "roc_auc",
        },
        "stop_policy": {"max_iterations": 3, "error_fingerprint_abort": True},
    }

    issues = agent_pipeline._validate_plan_payload(payload)  # noqa: SLF001
    assert issues == []
