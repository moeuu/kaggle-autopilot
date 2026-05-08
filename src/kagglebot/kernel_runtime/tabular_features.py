from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

FeatureBuilder = Callable[[pd.DataFrame], pd.Series]


@dataclass
class TabularFeatureArtifacts:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    orig_df: pd.DataFrame | None
    orig_source_path: str | None
    base_numeric_cols: list[str]
    base_categorical_cols: list[str]
    new_numeric_cols: list[str]
    new_categorical_cols: list[str]
    num_as_cat_cols: list[str]
    non_te_cats: list[str]
    te_columns: list[str]
    model_base_cols: list[str]
    feature_cols: list[str]
    suite_name: str
    train_mode: str
    feature_recipe: str
    orig_feature_status: dict[str, Any]


@dataclass(frozen=True)
class TrainingSource:
    frame: pd.DataFrame
    target: np.ndarray
    sample_weight: np.ndarray


def safe_fill_categorical(series: pd.Series) -> pd.Series:
    return series.fillna("__MISSING__").astype(str)


def build_joined_token(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    token = safe_fill_categorical(frame[columns[0]])
    for col in columns[1:]:
        token = token + "__" + safe_fill_categorical(frame[col])
    return token


def add_tabular_reference_features(
    *,
    frames: list[pd.DataFrame],
    base_numeric_cols: list[str],
    base_categorical_cols: list[str],
    orig_df: pd.DataFrame | None,
    include_interactions: bool,
    include_pair_tokens: bool,
    include_trigram_tokens: bool,
    include_orig_signal: bool,
    feature_recipe: str,
    service_cols: Sequence[str] = (),
    interaction_categoricals: Sequence[tuple[str, str]] = (),
    pair_token_categoricals: Sequence[Sequence[str]] = (),
    trigram_token_categoricals: Sequence[Sequence[str]] = (),
    target_name: str,
    original_row_weight: float,
    numeric_feature_builders: Mapping[str, FeatureBuilder] | None = None,
    categorical_feature_builders: Mapping[str, FeatureBuilder] | None = None,
) -> tuple[list[str], list[str], list[str], list[str], dict[str, Any]]:
    new_numeric_cols: list[str] = []
    new_categorical_cols: list[str] = []
    num_as_cat_cols: list[str] = []
    non_te_cats: list[str] = []
    numeric_feature_builders = dict(numeric_feature_builders or {})
    categorical_feature_builders = dict(categorical_feature_builders or {})

    for frame in frames:
        is_original = bool(orig_df is not None and frame is orig_df)
        frame["is_original_row"] = np.float32(1.0 if is_original else 0.0)
        frame["source_weight"] = np.float32(original_row_weight if is_original else 1.0)
    new_numeric_cols.extend(["is_original_row", "source_weight"])

    for col in base_numeric_cols:
        source_series = [frame[col] for frame in frames if col in frame.columns]
        freq_map = pd.concat(source_series, ignore_index=True).value_counts(normalize=True)
        freq_col = f"FREQ_{col}"
        for frame in frames:
            frame[freq_col] = frame[col].map(freq_map).fillna(0.0).astype(np.float32)
        new_numeric_cols.append(freq_col)

    for feature_name, builder in numeric_feature_builders.items():
        for frame in frames:
            frame[feature_name] = pd.to_numeric(builder(frame), errors="coerce").fillna(0.0).astype(np.float32)
        new_numeric_cols.append(feature_name)

    for feature_name, builder in categorical_feature_builders.items():
        for frame in frames:
            frame[feature_name] = safe_fill_categorical(builder(frame))
        new_categorical_cols.append(feature_name)

    ngram_categorical_cols: list[str] = []
    if include_pair_tokens:
        for columns in pair_token_categoricals:
            if any(any(col not in frame.columns for col in columns) for frame in frames):
                continue
            feature_name = "__".join(columns)
            for frame in frames:
                frame[feature_name] = build_joined_token(frame, columns)
            ngram_categorical_cols.append(feature_name)
    if include_trigram_tokens:
        for columns in trigram_token_categoricals:
            if any(any(col not in frame.columns for col in columns) for frame in frames):
                continue
            feature_name = "__".join(columns)
            for frame in frames:
                frame[feature_name] = build_joined_token(frame, columns)
            ngram_categorical_cols.append(feature_name)
    new_categorical_cols.extend(ngram_categorical_cols)

    if include_interactions:
        for left_col, right_col in interaction_categoricals:
            if any(left_col not in frame.columns or right_col not in frame.columns for frame in frames):
                continue
            feature_name = f"{left_col}__{right_col}"
            if feature_name in new_categorical_cols:
                continue
            for frame in frames:
                frame[feature_name] = build_joined_token(frame, (left_col, right_col))
            new_categorical_cols.append(feature_name)

    for col in base_categorical_cols + new_categorical_cols:
        source_series = [safe_fill_categorical(frame[col]) for frame in frames if col in frame.columns]
        freq_map = pd.concat(source_series, ignore_index=True).value_counts(normalize=True)
        freq_col = f"FREQCAT_{col}"
        for frame in frames:
            frame[freq_col] = safe_fill_categorical(frame[col]).map(freq_map).fillna(0.0).astype(np.float32)
        new_numeric_cols.append(freq_col)

    orig_target_mean = 0.5
    informative_lookup_cols: list[str] = []
    fallback_lookup_cols: list[str] = []
    orig_numeric_signal_cols = list(dict.fromkeys(base_numeric_cols + list(numeric_feature_builders.keys())))
    orig_categorical_signal_cols = list(dict.fromkeys(base_categorical_cols + new_categorical_cols))
    if include_orig_signal and orig_df is not None and target_name in orig_df.columns:
        orig_target_mean = float(pd.Series(orig_df[target_name]).mean())
        yes_orig = orig_df.loc[orig_df[target_name] == 1]
        no_orig = orig_df.loc[orig_df[target_name] == 0]
        eps = 1e-4
        for col in orig_numeric_signal_cols + orig_categorical_signal_cols:
            lookup_col = f"ORIG_proba_{col}"
            mapping = (
                pd.DataFrame({col: orig_df[col], target_name: orig_df[target_name]})
                .groupby(col, observed=False)[target_name]
                .mean()
            )
            for frame in frames:
                frame[lookup_col] = frame[col].map(mapping).fillna(orig_target_mean).astype(np.float32)
            new_numeric_cols.append(lookup_col)
            if int(mapping.nunique(dropna=True)) > 1:
                informative_lookup_cols.append(lookup_col)
            else:
                fallback_lookup_cols.append(lookup_col)
            if col in orig_categorical_signal_cols:
                freq_col = f"ORIG_freq_{col}"
                llr_col = f"ORIG_llr_{col}"
                freq_map = safe_fill_categorical(orig_df[col]).value_counts(normalize=True)
                yes_freq = safe_fill_categorical(yes_orig[col]).value_counts(normalize=True)
                no_freq = safe_fill_categorical(no_orig[col]).value_counts(normalize=True)
                all_keys = sorted(set(yes_freq.index) | set(no_freq.index))
                llr_map = {
                    key: float(np.log((float(yes_freq.get(key, 0.0)) + eps) / (float(no_freq.get(key, 0.0)) + eps)))
                    for key in all_keys
                }
                for frame in frames:
                    key_series = safe_fill_categorical(frame[col])
                    frame[freq_col] = key_series.map(freq_map).fillna(0.0).astype(np.float32)
                    frame[llr_col] = key_series.map(llr_map).fillna(0.0).astype(np.float32)
                new_numeric_cols.extend([freq_col, llr_col])
                if llr_map:
                    informative_lookup_cols.extend([freq_col, llr_col])
            else:
                z_col = f"ORIG_z_{col}"
                numeric = pd.to_numeric(orig_df[col], errors="coerce")
                mean = float(numeric.mean()) if numeric.notna().any() else 0.0
                std = float(numeric.std(ddof=0)) if numeric.notna().any() else 0.0
                if std <= 1e-9:
                    std = 1.0
                for frame in frames:
                    values = pd.to_numeric(frame[col], errors="coerce")
                    frame[z_col] = ((values - mean) / std).fillna(0.0).astype(np.float32)
                new_numeric_cols.append(z_col)
                informative_lookup_cols.append(z_col)
    elif include_orig_signal:
        for col in orig_numeric_signal_cols + orig_categorical_signal_cols:
            lookup_col = f"ORIG_proba_{col}"
            for frame in frames:
                frame[lookup_col] = np.float32(orig_target_mean)
            new_numeric_cols.append(lookup_col)
            fallback_lookup_cols.append(lookup_col)
            if col in orig_categorical_signal_cols:
                freq_col = f"ORIG_freq_{col}"
                llr_col = f"ORIG_llr_{col}"
                for frame in frames:
                    frame[freq_col] = np.float32(0.0)
                    frame[llr_col] = np.float32(0.0)
                new_numeric_cols.extend([freq_col, llr_col])
                fallback_lookup_cols.extend([freq_col, llr_col])
            else:
                z_col = f"ORIG_z_{col}"
                for frame in frames:
                    frame[z_col] = np.float32(0.0)
                new_numeric_cols.append(z_col)
                fallback_lookup_cols.append(z_col)

    for col in base_numeric_cols:
        cat_col = f"CAT_{col}"
        for frame in frames:
            frame[cat_col] = frame[col].astype(str)
        num_as_cat_cols.append(cat_col)

    orig_feature_status = {
        "original_data_found": orig_df is not None,
        "resolved_path": None,
        "signal_status": (
            "informative" if informative_lookup_cols else ("constant_fallback" if include_orig_signal else "disabled")
        ),
        "informative_lookup_cols": sorted(set(informative_lookup_cols)),
        "constant_lookup_cols": sorted(set(fallback_lookup_cols)),
        "default_probability": orig_target_mean,
        "feature_recipe": feature_recipe,
        "service_cols": list(service_cols),
    }
    return new_numeric_cols, new_categorical_cols, num_as_cat_cols, non_te_cats, orig_feature_status


def build_stat_feature_names(te_columns: list[str], stat_names: Sequence[str]) -> list[str]:
    return [f"TE1_{col}_{stat}" for col in te_columns for stat in stat_names]


def cross_fit_stat_features(
    train_frame: pd.DataFrame,
    y_train: np.ndarray,
    te_columns: list[str],
    *,
    target_name: str,
    stat_names: Sequence[str],
    inner_folds: int,
    random_state: int = 42,
) -> pd.DataFrame:
    if not te_columns:
        return pd.DataFrame(index=train_frame.index)
    encoded = {
        name: np.zeros(len(train_frame), dtype=np.float32) for name in build_stat_feature_names(te_columns, stat_names)
    }
    inner_splitter = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
    y_series = pd.Series(y_train)
    for inner_train_idx, inner_valid_idx in inner_splitter.split(train_frame, y_train):
        inner_train = train_frame.iloc[inner_train_idx].reset_index(drop=True)
        inner_valid = train_frame.iloc[inner_valid_idx].reset_index(drop=True)
        inner_target = y_series.iloc[inner_train_idx].reset_index(drop=True)
        for col in te_columns:
            grouped = (
                pd.DataFrame({col: inner_train[col], target_name: inner_target})
                .groupby(col, observed=False)[target_name]
                .agg(list(stat_names))
            )
            mapped_series = inner_valid[col]
            for stat_name in stat_names:
                feature_name = f"TE1_{col}_{stat_name}"
                encoded[feature_name][inner_valid_idx] = (
                    mapped_series.map(grouped[stat_name]).fillna(0.0).to_numpy(dtype=np.float32)
                )
    return pd.DataFrame(encoded, index=train_frame.index)


def apply_stat_features(
    train_frame: pd.DataFrame,
    y_train: np.ndarray,
    apply_frame: pd.DataFrame,
    te_columns: list[str],
    *,
    target_name: str,
    stat_names: Sequence[str],
) -> pd.DataFrame:
    if not te_columns:
        return pd.DataFrame(index=apply_frame.index)
    encoded: dict[str, np.ndarray] = {}
    y_series = pd.Series(y_train)
    for col in te_columns:
        grouped = (
            pd.DataFrame({col: train_frame[col], target_name: y_series})
            .groupby(col, observed=False)[target_name]
            .agg(list(stat_names))
        )
        mapped_series = apply_frame[col]
        for stat_name in stat_names:
            feature_name = f"TE1_{col}_{stat_name}"
            encoded[feature_name] = mapped_series.map(grouped[stat_name]).fillna(0.0).to_numpy(dtype=np.float32)
    return pd.DataFrame(encoded, index=apply_frame.index)


def build_encoded_matrices(
    *,
    train_source: pd.DataFrame,
    fold_valid: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
    artifacts: TabularFeatureArtifacts,
    target_name: str,
    stat_names: Sequence[str],
    inner_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    te_feature_names = [f"TE_{col}" for col in artifacts.te_columns]
    if artifacts.te_columns:
        te_train_input = train_source.loc[:, artifacts.te_columns].astype(str)
        te_valid_input = fold_valid.loc[:, artifacts.te_columns].astype(str)
        te_test_input = test_df.loc[:, artifacts.te_columns].astype(str)

        train_stats = cross_fit_stat_features(
            te_train_input,
            y_train,
            artifacts.te_columns,
            target_name=target_name,
            stat_names=stat_names,
            inner_folds=inner_folds,
        )
        valid_stats = apply_stat_features(
            te_train_input,
            y_train,
            te_valid_input,
            artifacts.te_columns,
            target_name=target_name,
            stat_names=stat_names,
        )
        test_stats = apply_stat_features(
            te_train_input,
            y_train,
            te_test_input,
            artifacts.te_columns,
            target_name=target_name,
            stat_names=stat_names,
        )

        te = TargetEncoder(
            cv=inner_folds,
            shuffle=True,
            smooth="auto",
            target_type="binary",
            random_state=42,
        )
        train_te = pd.DataFrame(
            te.fit_transform(te_train_input, y_train),
            columns=te_feature_names,
            index=train_source.index,
        ).astype(np.float32)
        valid_te = pd.DataFrame(
            te.transform(te_valid_input),
            columns=te_feature_names,
            index=fold_valid.index,
        ).astype(np.float32)
        test_te = pd.DataFrame(
            te.transform(te_test_input),
            columns=te_feature_names,
            index=test_df.index,
        ).astype(np.float32)
    else:
        train_stats = pd.DataFrame(index=train_source.index)
        valid_stats = pd.DataFrame(index=fold_valid.index)
        test_stats = pd.DataFrame(index=test_df.index)
        train_te = pd.DataFrame(index=train_source.index)
        valid_te = pd.DataFrame(index=fold_valid.index)
        test_te = pd.DataFrame(index=test_df.index)

    train_base = train_source.loc[:, artifacts.model_base_cols].copy()
    valid_base = fold_valid.loc[:, artifacts.model_base_cols].copy()
    test_base = test_df.loc[:, artifacts.model_base_cols].copy()
    for frame in [train_base, valid_base, test_base]:
        for col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0).astype(np.float32)

    x_train = pd.concat([train_base, train_stats, train_te], axis=1)
    x_valid = pd.concat([valid_base, valid_stats, valid_te], axis=1)
    x_test = pd.concat([test_base, test_stats, test_te], axis=1)
    return x_train, x_valid, x_test, list(x_train.columns)


def build_training_source(
    *,
    fold_train: pd.DataFrame,
    y_train: np.ndarray,
    artifacts: TabularFeatureArtifacts,
    target_name: str,
    original_row_weight: float,
) -> TrainingSource:
    if artifacts.train_mode == "competition_only":
        frame = fold_train.reset_index(drop=True).copy()
        if "is_original_row" in frame.columns:
            frame["is_original_row"] = np.float32(0.0)
        if "source_weight" in frame.columns:
            frame["source_weight"] = np.float32(1.0)
        return TrainingSource(
            frame=frame,
            target=np.asarray(y_train, dtype=np.int8),
            sample_weight=np.ones(len(frame), dtype=np.float32),
        )
    if artifacts.orig_df is None or target_name not in artifacts.orig_df.columns:
        raise RuntimeError(f"Suite {artifacts.suite_name} requires original dataset labels but they are unavailable.")
    orig_features = artifacts.orig_df.drop(columns=[target_name], errors="ignore").reset_index(drop=True).copy()
    orig_target = artifacts.orig_df[target_name].to_numpy(dtype=np.int8)
    if artifacts.train_mode == "original_only":
        if "is_original_row" in orig_features.columns:
            orig_features["is_original_row"] = np.float32(1.0)
        if "source_weight" in orig_features.columns:
            orig_features["source_weight"] = np.float32(1.0)
        return TrainingSource(
            frame=orig_features,
            target=orig_target,
            sample_weight=np.ones(len(orig_features), dtype=np.float32),
        )
    if artifacts.train_mode == "competition_plus_original":
        comp_frame = fold_train.reset_index(drop=True).copy()
        if "is_original_row" in comp_frame.columns:
            comp_frame["is_original_row"] = np.float32(0.0)
        if "source_weight" in comp_frame.columns:
            comp_frame["source_weight"] = np.float32(1.0)
        if "is_original_row" in orig_features.columns:
            orig_features["is_original_row"] = np.float32(1.0)
        if "source_weight" in orig_features.columns:
            orig_features["source_weight"] = np.float32(original_row_weight)
        combined_frame = pd.concat([comp_frame, orig_features], axis=0, ignore_index=True)
        combined_target = np.concatenate([np.asarray(y_train, dtype=np.int8), orig_target])
        combined_weight = np.concatenate(
            [
                np.ones(len(comp_frame), dtype=np.float32),
                np.full(len(orig_features), original_row_weight, dtype=np.float32),
            ]
        )
        return TrainingSource(frame=combined_frame, target=combined_target, sample_weight=combined_weight)
    raise AssertionError(f"Unsupported train_mode: {artifacts.train_mode}")


def build_raw_categorical_matrices(
    *,
    train_source: pd.DataFrame,
    fold_valid: pd.DataFrame,
    test_df: pd.DataFrame,
    artifacts: TabularFeatureArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    feature_cols = list(artifacts.feature_cols)
    cat_features = [
        col
        for col in list(artifacts.base_categorical_cols) + list(artifacts.new_categorical_cols)
        if col in feature_cols
    ]

    def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.loc[:, feature_cols].copy()
        for col in feature_cols:
            if col in cat_features:
                prepared[col] = safe_fill_categorical(prepared[col])
            else:
                prepared[col] = pd.to_numeric(prepared[col], errors="coerce").fillna(0.0).astype(np.float32)
        return prepared

    x_train = _prepare(train_source)
    x_valid = _prepare(fold_valid)
    x_test = _prepare(test_df)
    return x_train, x_valid, x_test, feature_cols, cat_features
