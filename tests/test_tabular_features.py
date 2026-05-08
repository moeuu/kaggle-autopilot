from __future__ import annotations

import numpy as np
import pandas as pd

from kagglebot.kernel_runtime.tabular_features import (
    TabularFeatureArtifacts,
    add_tabular_reference_features,
    build_joined_token,
    build_training_source,
)


def test_add_tabular_reference_features_builds_custom_tokens_and_orig_signals() -> None:
    train = pd.DataFrame(
        {
            "num": [1.0, 2.0],
            "cat": ["a", "b"],
            "cat2": ["x", "y"],
            "services": ["Yes", "No"],
        }
    )
    test = train.copy()
    orig = train.copy()
    orig["target"] = [0, 1]

    new_numeric_cols, new_categorical_cols, _, _, orig_feature_status = add_tabular_reference_features(
        frames=[train, test, orig],
        base_numeric_cols=["num"],
        base_categorical_cols=["cat", "cat2"],
        orig_df=orig,
        include_interactions=True,
        include_pair_tokens=True,
        include_trigram_tokens=False,
        include_orig_signal=True,
        feature_recipe="full",
        interaction_categoricals=[("cat", "cat2")],
        pair_token_categoricals=[("cat", "cat2")],
        target_name="target",
        original_row_weight=0.5,
        numeric_feature_builders={"num_x2": lambda frame: frame["num"] * 2.0},
        categorical_feature_builders={"cat_joined": lambda frame: build_joined_token(frame, ("cat", "cat2"))},
    )

    assert "num_x2" in new_numeric_cols
    assert "cat_joined" in new_categorical_cols
    assert "cat__cat2" in new_categorical_cols
    assert "FREQ_num" in new_numeric_cols
    assert "FREQCAT_cat_joined" in new_numeric_cols
    assert "ORIG_proba_cat" in new_numeric_cols
    assert orig_feature_status["signal_status"] == "informative"


def test_build_training_source_combines_competition_and_original_rows() -> None:
    artifacts = TabularFeatureArtifacts(
        train_df=pd.DataFrame(),
        test_df=pd.DataFrame(),
        orig_df=pd.DataFrame({"num": [3.0, 4.0], "target": [1, 0]}),
        orig_source_path="orig.csv",
        base_numeric_cols=["num"],
        base_categorical_cols=[],
        new_numeric_cols=[],
        new_categorical_cols=[],
        num_as_cat_cols=[],
        non_te_cats=[],
        te_columns=[],
        model_base_cols=["num"],
        feature_cols=["num"],
        suite_name="comp_plus_orig",
        train_mode="competition_plus_original",
        feature_recipe="full",
        orig_feature_status={},
    )
    fold_train = pd.DataFrame({"num": [1.0, 2.0]})
    y_train = np.asarray([0, 1], dtype=np.int8)

    training_source = build_training_source(
        fold_train=fold_train,
        y_train=y_train,
        artifacts=artifacts,
        target_name="target",
        original_row_weight=0.425,
    )

    assert len(training_source.frame) == 4
    assert training_source.target.tolist() == [0, 1, 1, 0]
    assert np.allclose(training_source.sample_weight, [1.0, 1.0, 0.425, 0.425])
