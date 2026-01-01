from __future__ import annotations

from kagglebot.analyzer.types import ModelingStrategy


def build_strategy(
    task: str,
    *,
    time_budget_minutes: int,
    cv_folds: int,
    models: list[str] | None,
    use_stacking: bool,
) -> ModelingStrategy:
    if models:
        selected_models = [m.strip().lower() for m in models if m.strip()]
    else:
        if task == "classification":
            selected_models = ["logreg", "hist_gb", "catboost"]
        else:
            selected_models = ["ridge", "hist_gb", "catboost"]

    preprocessing = [
        "impute_median_numeric",
        "impute_mode_categorical",
        "onehot_encode_for_linear_models",
    ]

    return ModelingStrategy(
        preprocessing=preprocessing,
        models=selected_models,
        cv_folds=cv_folds,
        use_stacking=use_stacking,
        time_budget_minutes=time_budget_minutes,
    )
