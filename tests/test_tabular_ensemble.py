from __future__ import annotations

import numpy as np
import pandas as pd

import kagglebot.kernel_runtime.tabular_ensemble as tabular_ensemble
from kagglebot.kernel_runtime.tabular_ensemble import (
    PipelineResult,
    build_prediction_correlation_summary,
    maybe_apply_pseudo_labels,
    resolve_component_models,
    train_catboost_model,
    train_xgb_model,
)


def _pipeline_result(
    name: str,
    preds: list[float],
    *,
    kind: str = "single",
    blend_components: list[str] | None = None,
) -> PipelineResult:
    arr = np.asarray(preds, dtype=np.float64)
    metadata = {"kind": kind}
    if blend_components is not None:
        metadata["blend_components"] = blend_components
    return PipelineResult(
        name=name,
        oof_preds=arr,
        test_preds=arr,
        cv_score=0.9,
        fold_scores=[],
        feature_manifest={},
        metadata=metadata,
        test_predictions_by_fold={},
        oof_predictions_by_fold={},
        valid_indices_by_fold={},
    )


def test_resolve_component_models_prefers_blend_components() -> None:
    result = _pipeline_result("blend_top", [0.1, 0.9], kind="weighted_blend", blend_components=["xgb_a", "cb_b"])

    assert resolve_component_models(result) == ["xgb_a", "cb_b"]


def test_build_prediction_correlation_summary_uses_single_models_only() -> None:
    result_a = _pipeline_result("xgb_a", [0.1, 0.2, 0.8, 0.9])
    result_b = _pipeline_result("cb_b", [0.12, 0.22, 0.82, 0.88])
    blend = _pipeline_result(
        "blend_ab", [0.11, 0.21, 0.81, 0.89], kind="rank_blend", blend_components=["xgb_a", "cb_b"]
    )

    summary = build_prediction_correlation_summary([result_a, result_b, blend])

    assert summary["pair_count"] == 1
    assert summary["mean_abs_corr"] is not None
    assert summary["max_abs_corr"] is not None
    assert summary["min_abs_corr"] is not None


def test_train_xgb_model_forwards_sample_weight(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeXGBClassifier:
        best_iteration = 12

        def __init__(self, **params):
            captured["params"] = params

        def fit(self, x_train, y_train, **kwargs):
            captured["x_train"] = x_train
            captured["y_train"] = y_train
            captured["fit_kwargs"] = kwargs
            return self

    monkeypatch.setattr(tabular_ensemble, "XGBClassifier", FakeXGBClassifier)

    x_train = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    y_train = np.array([0, 1, 0], dtype=np.int8)
    x_valid = pd.DataFrame({"feature": [4.0, 5.0]})
    y_valid = np.array([1, 0], dtype=np.int8)
    sample_weight = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    _, meta = train_xgb_model(
        x_train,
        y_train,
        x_valid,
        y_valid,
        model_seed=42,
        params_override={},
        sample_weight=sample_weight,
    )

    fit_kwargs = captured["fit_kwargs"]
    assert isinstance(fit_kwargs, dict)
    assert np.array_equal(fit_kwargs["sample_weight"], sample_weight)
    assert fit_kwargs["eval_set"] == [(x_valid, y_valid)]
    assert meta["best_iteration"] == 12


def test_maybe_apply_pseudo_labels_extends_sample_weight(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class BaseModel:
        def predict_proba(self, frame):
            if frame["feature"].iloc[0] < 5.0:
                probs = np.array([0.2, 0.8], dtype=np.float64)
            else:
                probs = np.array([0.995, 0.005], dtype=np.float64)
            return np.column_stack([1.0 - probs, probs])

    class PseudoModel:
        def predict_proba(self, frame):
            if frame["feature"].iloc[0] < 5.0:
                probs = np.array([0.2, 0.8], dtype=np.float64)
            else:
                probs = np.array([0.99, 0.01], dtype=np.float64)
            return np.column_stack([1.0 - probs, probs])

    def fake_train_xgb_model(
        x_train,
        y_train,
        x_valid,
        y_valid,
        model_seed,
        params_override,
        sample_weight=None,
    ):
        captured["x_train"] = x_train
        captured["y_train"] = y_train
        captured["sample_weight"] = sample_weight
        return PseudoModel(), {"device": "cpu", "best_iteration": 3}

    monkeypatch.setattr(tabular_ensemble, "train_xgb_model", fake_train_xgb_model)

    x_train = pd.DataFrame({"feature": [1.0, 2.0]})
    y_train = np.array([0, 1], dtype=np.int8)
    x_valid = pd.DataFrame({"feature": [3.0, 4.0]})
    y_valid = np.array([0, 1], dtype=np.int8)
    x_test = pd.DataFrame({"feature": [5.0, 6.0]})
    sample_weight = np.array([1.5, 2.5], dtype=np.float32)

    _, _, pl_log, pl_meta = maybe_apply_pseudo_labels(
        model=BaseModel(),
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        x_test=x_test,
        model_seed=42,
        threshold=0.99,
        enabled=True,
        params_override={},
        sample_weight=sample_weight,
    )

    assert np.array_equal(captured["sample_weight"], np.array([1.5, 2.5, 1.0, 1.0], dtype=np.float32))
    assert captured["y_train"].tolist() == [0, 1, 1, 0]
    assert pl_log["candidate_count"] == 2
    assert pl_meta["pseudo_model_device"] == "cpu"


def test_train_catboost_model_logs_gpu_fallback(monkeypatch, capsys) -> None:
    init_task_types: list[str] = []

    class FakeCatBoostClassifier:
        def __init__(self, **params):
            self.params = params
            init_task_types.append(str(params["task_type"]))

        def fit(self, x_train, y_train, **kwargs):
            if str(self.params["task_type"]).upper() == "GPU":
                raise RuntimeError("CUDA init failed")
            return self

        def get_best_iteration(self):
            return 17

    monkeypatch.setattr(tabular_ensemble, "CatBoostClassifier", FakeCatBoostClassifier)
    monkeypatch.setattr(tabular_ensemble, "PREFER_CUDA", True)

    x_train = pd.DataFrame({"cat": ["a", "b", "c"], "num": [1.0, 2.0, 3.0]})
    y_train = np.array([0, 1, 0], dtype=np.int8)
    x_valid = pd.DataFrame({"cat": ["a", "b"], "num": [4.0, 5.0]})
    y_valid = np.array([1, 0], dtype=np.int8)

    _, meta = train_catboost_model(
        x_train,
        y_train,
        x_valid,
        y_valid,
        model_seed=42,
        params_override={},
        cat_features=["cat"],
        sample_weight=None,
    )

    captured = capsys.readouterr()
    assert init_task_types == ["GPU", "CPU"]
    assert meta["device"] == "cpu"
    assert meta["fallback_reason"] == "CUDA init failed"
    assert "CatBoost GPU failed; retrying on CPU: RuntimeError: CUDA init failed" in captured.out
