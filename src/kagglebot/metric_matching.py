from __future__ import annotations

from kagglebot.solver.metrics import canonical_metric


def normalize_metric_name(name: str | None) -> str:
    """Normalize a metric label for loose string comparison."""
    if not name:
        return ""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def canonical_metric_name_for_match(name: str | None) -> str:
    """Return a canonical normalized metric token for mismatch checks."""
    normalized = normalize_metric_name(name)
    if not normalized:
        return ""
    canonical = normalize_metric_name(canonical_metric(str(name)))
    metric = canonical or normalized
    alias_map = {
        "balancedacc": "balancedaccuracy",
        "balancedaccuracy": "balancedaccuracy",
        "brier": "brierscore",
        "brierloss": "brierscore",
        "brierscoreloss": "brierscore",
        # UMUD uses a custom normalized MAE score, not generic MAE.
        "normalizedmeanabsoluteerroracrosspadegflmmandmtmm": "umudscore",
        "umudnormalizedmae": "umudscore",
        "umudscore": "umudscore",
        "umudscorenormalizedmeanabsoluteerroracrosspadegflmmandmtmm": "umudscore",
    }
    return alias_map.get(metric, metric)


def metrics_equivalent(left: str | None, right: str | None) -> bool:
    """Return True when two metric labels represent the same metric."""
    left_metric = canonical_metric_name_for_match(left)
    right_metric = canonical_metric_name_for_match(right)
    return bool(left_metric) and left_metric == right_metric


def infer_metric_direction_for_mismatch(metric: str, fallback_direction: str) -> tuple[str, bool]:
    metric_name = canonical_metric(metric)
    if metric_name in {
        "aurc",
        "rmse",
        "rmsle",
        "mae",
        "mape",
        "mse",
        "logloss",
        "mcrmse",
        "smape",
        "pinball_loss",
        "interval_score",
    }:
        return "minimize", True
    if metric_name in {
        "auc",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "average_precision",
        "r2",
        "r_squared",
        "ndcg",
        "concordance_index",
        "pearson",
        "spearman",
        "quadratic_weighted_kappa",
    }:
        return "maximize", True
    metric_lower = metric.lower()
    if "risk-coverage" in metric_lower or "risk coverage" in metric_lower:
        return "minimize", True
    if any(key in metric_lower for key in ["loss", "error"]):
        return "minimize", True
    if any(key in metric_lower for key in ["auc", "accuracy", "f1", "precision", "recall", "ap", "r2", "map"]):
        return "maximize", True
    return fallback_direction, False
