from __future__ import annotations


def classify_cause_category(text: str) -> str:
    normalized = text.lower()
    rules: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("external_signal", ("orig_proba", "original data", "external signal", "reference dataset", "reference input")),
        ("pseudo_labeling", ("pseudo-label", "pseudo label")),
        ("online_mismatch", ("public leaderboard", "online mismatch", "lb mismatch", "online regressed")),
        ("data_leakage", ("leak", "target leak", "leakage")),
        ("overfitting", ("overfit", "train/val gap", "generalization gap")),
        ("underfitting", ("underfit", "model too simple", "high bias")),
        ("feature_engineering", ("feature", "encoding", "missing value", "imputation")),
        ("hyperparameter", ("hyperparameter", "learning rate", "max_depth", "n_estimators", "regularization")),
        ("validation_strategy", ("cross-validation", "cv", "fold", "holdout", "split strategy")),
        ("class_imbalance", ("imbalance", "minority class", "class weight", "threshold")),
        ("resource_constraints", ("gpu utilization", "resource", "timeout", "batch size")),
    )
    for category, keywords in rules:
        if any(keyword in normalized for keyword in keywords):
            return category
    return "unknown"


def classify_fix_category(text: str) -> str:
    normalized = text.lower()
    rules: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "external_data_recovery",
            ("recover orig_proba", "original data", "reference dataset", "reference input", "stage reference"),
        ),
        ("pseudo_label_disable", ("disable pseudo-label", "disable pseudo label", "avoid pseudo-label")),
        ("model_diversification", ("family diversity", "model-family", "same-family", "diversification")),
        ("stronger_model", ("catboost", "xgboost", "lightgbm", "transformer", "neural network", "model upgrade")),
        ("feature_engineering", ("feature", "encoding", "imputation", "interaction", "transformation")),
        ("hyperparameter_tuning", ("hyperparameter", "learning rate", "max_depth", "n_estimators", "tuning")),
        ("regularization", ("regularization", "dropout", "early stopping", "l1", "l2")),
        ("validation_strategy", ("cross-validation", "cv", "fold", "holdout", "split")),
        ("ensembling", ("ensemble", "stacking", "blending", "average")),
        ("data_cleaning", ("outlier", "duplicate", "cleaning", "leakage fix")),
        ("training_budget", ("epoch", "iterations", "batch size", "training budget")),
    )
    for category, keywords in rules:
        if any(keyword in normalized for keyword in keywords):
            return category
    return "unknown"


def classify_error_category(text: str) -> str:
    normalized = text.lower()
    rules: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("external_signal_missing", ("orig_proba", "original_data_found", "constant_fallback", "reference input")),
        ("pseudo_label_failure", ("pseudo-label", "pseudo label", "accepted folds", "accepted candidates")),
        ("online_mismatch", ("public leaderboard regressed", "online mismatch", "lb mismatch")),
        ("dependency_missing", ("modulenotfounderror", "no module named", "importerror")),
        ("schema_mismatch", ("missing columns", "column", "schema", "keyerror")),
        ("device_mismatch", ("same device", "cuda", "cpu", "device")),
        ("oom", ("out of memory", "cuda out of memory", "oom")),
        ("network", ("connectionerror", "dns", "network", "name resolution")),
        ("kaggle_cli", ("kaggle cli", "competitions submit", "kernels push")),
        ("timeout", ("timeout", "timed out", "deadline exceeded")),
        ("validation", ("row count mismatch", "submission", "validation error")),
    )
    for category, keywords in rules:
        if any(keyword in normalized for keyword in keywords):
            return category
    return "unknown"
