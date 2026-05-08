from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.competition_artifact


def _load_s6e3_runtime():
    repo_root = Path(__file__).resolve().parents[1]
    runtime_path = repo_root / "artifacts" / "playground-series-s6e3" / "kernel" / "runtime_s6e3_impl.py"
    kernel_runtime_dir = repo_root / "src" / "kagglebot" / "kernel_runtime"
    if str(kernel_runtime_dir) not in sys.path:
        sys.path.insert(0, str(kernel_runtime_dir))
    spec = importlib.util.spec_from_file_location("s6e3_runtime_test", runtime_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_s6e3_runtime_builds_required_ablation_suites() -> None:
    runtime = _load_s6e3_runtime()

    suites = runtime.build_suite_specs()

    assert [suite.name for suite in suites] == [
        "comp_only",
        "orig_only",
        "comp_plus_orig",
        "orig_signal_only",
    ]
    assert suites[1].train_mode == "original_only"
    assert suites[2].train_mode == "competition_plus_original"
    assert suites[3].feature_recipe == "orig_signal_only"
    assert runtime.MULTISEED_MODEL_SEEDS == [42, 2024, 777]

    pipeline_names = [spec.name for spec in runtime.build_pipeline_specs(suites[0], name_suffix="_check")]
    assert any(name.startswith("catboost_rawcat_multiseed") for name in pipeline_names)
    assert any(name.startswith("catboost_origstats_multiseed") for name in pipeline_names)
    assert any(name.startswith("lgbm_te_multiseed") for name in pipeline_names)
    assert any(name.startswith("xgb_tuned_multiseed") for name in pipeline_names)
    assert any(name.startswith("xgb_origpair_multiseed") for name in pipeline_names)


def test_s6e3_runtime_orig_only_ablation_is_default(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runtime_path = repo_root / "artifacts" / "playground-series-s6e3" / "kernel" / "runtime_s6e3_impl.py"
    kernel_runtime_dir = repo_root / "src" / "kagglebot" / "kernel_runtime"
    monkeypatch.delenv("KAGGLEBOT_ENABLE_ORIG_ONLY_ABLATION", raising=False)
    if str(kernel_runtime_dir) not in sys.path:
        sys.path.insert(0, str(kernel_runtime_dir))
    spec = importlib.util.spec_from_file_location("s6e3_runtime_orig_only_test", runtime_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    suites = module.build_suite_specs()

    assert [suite.name for suite in suites] == [
        "comp_only",
        "orig_only",
        "comp_plus_orig",
        "orig_signal_only",
    ]


def test_s6e3_runtime_build_training_source_uses_original_rows() -> None:
    runtime = _load_s6e3_runtime()

    orig_df = pd.DataFrame(
        {
            "tenure": [3.0, 7.0],
            "MonthlyCharges": [25.0, 70.0],
            runtime.TARGET_NAME: [1, 0],
        }
    )
    artifacts = runtime.ReferenceArtifacts(
        train_df=pd.DataFrame(),
        test_df=pd.DataFrame(),
        orig_df=orig_df,
        orig_source_path="orig.csv",
        base_numeric_cols=["tenure", "MonthlyCharges"],
        base_categorical_cols=[],
        new_numeric_cols=[],
        new_categorical_cols=[],
        num_as_cat_cols=[],
        non_te_cats=[],
        te_columns=[],
        model_base_cols=["tenure", "MonthlyCharges"],
        feature_cols=["tenure", "MonthlyCharges"],
        suite_name="comp_plus_orig",
        train_mode="competition_plus_original",
        feature_recipe="full",
        original_row_weight=runtime.ORIGINAL_ROW_WEIGHT,
        orig_feature_status={},
    )
    fold_train = pd.DataFrame({"tenure": [1.0, 2.0], "MonthlyCharges": [10.0, 20.0]})
    y_train = np.array([0, 1], dtype=np.int8)

    training_source = runtime.build_training_source(
        fold_train=fold_train,
        y_train=y_train,
        artifacts=artifacts,
    )

    assert len(training_source.frame) == 4
    assert training_source.target.tolist() == [0, 1, 1, 0]
    assert np.allclose(
        training_source.sample_weight,
        [1.0, 1.0, runtime.ORIGINAL_ROW_WEIGHT, runtime.ORIGINAL_ROW_WEIGHT],
    )


def test_s6e3_runtime_adds_ngram_tokens_and_frequency_features() -> None:
    runtime = _load_s6e3_runtime()

    train = pd.DataFrame(
        {
            "SeniorCitizen": [0, 1],
            "tenure": [1.0, 2.0],
            "MonthlyCharges": [10.0, 20.0],
            "TotalCharges": [10.0, 40.0],
            "Contract": ["Month-to-month", "One year"],
            "InternetService": ["DSL", "Fiber optic"],
            "PaymentMethod": ["Bank transfer", "Credit card"],
            "OnlineSecurity": ["No", "Yes"],
            "TechSupport": ["No", "Yes"],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "Yes"],
            "OnlineBackup": ["No", "Yes"],
            "DeviceProtection": ["No", "Yes"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "Yes"],
        }
    )
    test = train.copy()
    orig = train.copy()
    orig[runtime.TARGET_NAME] = [0, 1]

    new_numeric_cols, new_categorical_cols, _, _, orig_feature_status = runtime.add_reference_features(
        frames=[train, test, orig],
        base_numeric_cols=list(runtime.BASE_NUMERIC_COLS),
        base_categorical_cols=[col for col in train.columns if col not in runtime.BASE_NUMERIC_COLS],
        orig_df=orig,
        include_interactions=True,
        include_pair_tokens=True,
        include_trigram_tokens=True,
        include_orig_signal=True,
        feature_recipe="full",
        original_row_weight=runtime.ORIGINAL_ROW_WEIGHT,
    )

    assert "Contract__InternetService__PaymentMethod" in new_categorical_cols
    assert "tenure_bin" in new_categorical_cols
    assert "tenure_bin__Contract" in new_categorical_cols
    assert "MonthlyCharges_bin" in new_categorical_cols
    assert "TotalCharges_bin" in new_categorical_cols
    assert "ORIG_proba_Contract__InternetService" in new_numeric_cols
    assert train["tenure_bin"].tolist() == ["0_6", "0_6"]
    assert "FREQCAT_Contract__InternetService" in new_numeric_cols
    assert orig_feature_status["signal_status"] == "informative"


def test_s6e3_runtime_prefers_distinct_families_for_blend_components() -> None:
    runtime = _load_s6e3_runtime()
    from kagglebot.kernel_runtime.tabular_blend import select_top_blend_components

    def _result(name: str, cv_score: float, family: str) -> object:
        return runtime.PipelineResult(
            name=name,
            oof_preds=np.array([0.1, 0.9], dtype=np.float64),
            test_preds=np.array([0.2, 0.8], dtype=np.float64),
            cv_score=cv_score,
            fold_scores=[],
            feature_manifest={},
            metadata={"kind": "single", "model_family": family, "model_seeds": [42]},
            test_predictions_by_fold={},
            oof_predictions_by_fold={},
            valid_indices_by_fold={},
        )

    selected = select_top_blend_components(
        [
            _result("xgb_best", 0.9200, "xgb"),
            _result("xgb_second", 0.9198, "xgb"),
            _result("cat_best", 0.9195, "catboost"),
        ]
    )

    assert [result.name for result in selected] == ["xgb_best", "cat_best"]


def test_s6e3_runtime_builds_logit_blend_result() -> None:
    runtime = _load_s6e3_runtime()
    runtime.OUTER_FOLDS = 2
    bundle = runtime.DatasetBundle(
        train_df=pd.DataFrame({"id": [0, 1]}),
        test_df=pd.DataFrame({"id": [0, 1]}),
        sample_submission=pd.DataFrame({"id": [0, 1], runtime.TARGET_NAME: [0.0, 0.0]}),
        id_col="id",
        target_col=runtime.TARGET_NAME,
        feature_cols=[],
        target_values=np.array([0, 1], dtype=np.int8),
        data_dir=Path("."),
    )
    artifacts = runtime.ReferenceArtifacts(
        train_df=pd.DataFrame(),
        test_df=pd.DataFrame(),
        orig_df=None,
        orig_source_path=None,
        base_numeric_cols=[],
        base_categorical_cols=[],
        new_numeric_cols=[],
        new_categorical_cols=[],
        num_as_cat_cols=[],
        non_te_cats=[],
        te_columns=[],
        model_base_cols=[],
        feature_cols=[],
        suite_name="comp_only",
        train_mode="competition_only",
        feature_recipe="full",
        original_row_weight=runtime.ORIGINAL_ROW_WEIGHT,
        orig_feature_status={},
    )
    first = runtime.PipelineResult(
        name="xgb_a",
        oof_preds=np.array([0.2, 0.8], dtype=np.float64),
        test_preds=np.array([0.25, 0.75], dtype=np.float64),
        cv_score=0.91,
        fold_scores=[],
        feature_manifest={"final_feature_count": 3},
        metadata={"kind": "single", "model_family": "xgb", "model_seeds": [42]},
        test_predictions_by_fold={"fold_1": np.array([0.25]), "fold_2": np.array([0.75])},
        oof_predictions_by_fold={"fold_1": np.array([0.2]), "fold_2": np.array([0.8])},
        valid_indices_by_fold={"fold_1": np.array([0]), "fold_2": np.array([1])},
    )
    second = runtime.PipelineResult(
        name="cat_b",
        oof_preds=np.array([0.3, 0.7], dtype=np.float64),
        test_preds=np.array([0.35, 0.65], dtype=np.float64),
        cv_score=0.909,
        fold_scores=[],
        feature_manifest={"final_feature_count": 3},
        metadata={"kind": "single", "model_family": "catboost", "model_seeds": [42]},
        test_predictions_by_fold={"fold_1": np.array([0.35]), "fold_2": np.array([0.65])},
        oof_predictions_by_fold={"fold_1": np.array([0.3]), "fold_2": np.array([0.7])},
        valid_indices_by_fold={"fold_1": np.array([0]), "fold_2": np.array([1])},
    )

    result = runtime.make_logit_blend_result(
        bundle=bundle,
        artifacts=artifacts,
        results_by_name={"xgb_a": first, "cat_b": second},
        first_name="xgb_a",
        second_name="cat_b",
        first_weight=0.5,
    )

    assert result.name.startswith("logit_blend_xgb_a_cat_b_w50")
    assert result.metadata["method"] == "logit"
    assert result.metadata["kind"] == "logit_blend"
    assert np.all(result.oof_preds > 0.0)
    assert np.all(result.oof_preds < 1.0)
