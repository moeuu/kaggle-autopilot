from __future__ import annotations

from dataclasses import dataclass

from kagglebot.metric_matching import metrics_equivalent
from kagglebot.scalar_utils import tolerant_int
from kagglebot.score_utils import should_update_best_score
from kagglebot.solver.metrics import canonical_metric

MAJOR_TOP1_GAP = 0.03
MODERATE_TOP1_GAP = 0.01
BEST_SCORE_OUTLIER_TOP1_ABS_MARGIN = 0.02
BEST_SCORE_OUTLIER_TOP1_REL_MARGIN = 0.01
REGRESSION_GUARD_ABS_DROP_PROB = 0.03
REGRESSION_GUARD_ABS_DROP_DEFAULT = 0.10
CONSERVATIVE_COLLAPSE_MAX_FEATURES = 5

GENERIC_FALLBACK_METRICS = frozenset(
    {
        "accuracy",
        "auc",
        "average_precision",
        "brier_score",
        "f1",
        "logloss",
        "mae",
        "mape",
        "mse",
        "precision",
        "r2",
        "recall",
        "rmse",
        "rmsle",
    }
)
BOUNDED_CLASSIFICATION_METRICS = frozenset(
    {
        "accuracy",
        "auc",
        "f1",
        "precision",
        "recall",
        "average_precision",
    }
)
MAXIMIZE_PROBABILITY_LIKE_METRICS = frozenset(
    {
        "auc",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "average_precision",
        "r2",
        "r_squared",
    }
)


@dataclass(frozen=True)
class IterationPhase:
    metric_direction: str

    def delta_from_best(self, best_score: float | None, current_score: float) -> float | None:
        if best_score is None:
            return None
        if self.metric_direction == "minimize":
            return best_score - current_score
        return current_score - best_score

    def should_update_best(self, best_score: float | None, current_score: float, min_improvement: float) -> bool:
        return should_update_best_score(best_score, current_score, self.metric_direction, min_improvement)


def resolve_explicit_official_metric_override(
    payload: dict[str, object] | None,
    *,
    target_metric: str | None,
    evaluation_metric: str | None,
) -> str | None:
    """Trust an explicitly declared official kernel metric over a stale generic fallback target."""
    if not isinstance(payload, dict):
        return None
    raw_official = payload.get("official_metric")
    if not isinstance(raw_official, str) or not raw_official.strip():
        return None
    official_metric = raw_official.strip()
    if not evaluation_metric or not metrics_equivalent(official_metric, evaluation_metric):
        return None
    if not target_metric or metrics_equivalent(official_metric, target_metric):
        return None
    if canonical_metric(target_metric) not in GENERIC_FALLBACK_METRICS:
        return None
    return official_metric


def is_confirmed_first_place(rank: int | None, source: str | None) -> bool:
    if rank != 1:
        return False
    if source is None:
        return True
    normalized = source.strip().lower()
    return normalized not in {"leaderboard_score_estimate", "score_estimate"}


def classify_improvement_mode(value: float, top1_score: float | None, direction: str) -> tuple[str, float | None]:
    if top1_score is None:
        return "major_overhaul", None
    gap = top1_score - value if direction == "maximize" else value - top1_score
    if gap >= MAJOR_TOP1_GAP:
        return "major_overhaul", gap
    if gap >= MODERATE_TOP1_GAP:
        return "moderate_update", gap
    return "minor_tuning", gap


def score_delta_vs_reference(current: float, reference: float, direction: str) -> float:
    """Return signed delta where positive means current is better than reference."""
    if direction == "minimize":
        return reference - current
    return current - reference


def normalize_code_reference_score_for_comparison(
    *,
    current: float,
    reference: float | None,
    metric: str,
) -> float | None:
    if reference is None:
        return None
    metric_name = canonical_metric(metric)
    current_value = float(current)
    reference_value = float(reference)
    bounded_metric = metric_name in BOUNDED_CLASSIFICATION_METRICS or any(
        token in str(metric).strip().lower().replace("_", " ")
        for token in ("auc", "accuracy", "f1", "precision", "recall", "average precision")
    )
    if bounded_metric and 0.0 <= current_value <= 1.0 and 1.0 < reference_value <= 10_000.0:
        # Kaggle notebook titles often encode 0.948 as either "94.8" or
        # "0948". Scale integer/percentage shorthands into the metric's
        # bounded range instead of treating them as literal scores.
        while reference_value > 1.0:
            reference_value /= 10.0
        return reference_value
    return reference_value


def score_drop_vs_best(*, best_score: float | None, current_score: float, direction: str) -> float | None:
    if best_score is None:
        return None
    if direction == "maximize":
        return float(best_score) - float(current_score)
    return float(current_score) - float(best_score)


def regression_drop_threshold(*, metric: str, direction: str) -> float:
    if direction == "maximize" and canonical_metric(metric) in MAXIMIZE_PROBABILITY_LIKE_METRICS:
        return REGRESSION_GUARD_ABS_DROP_PROB
    return REGRESSION_GUARD_ABS_DROP_DEFAULT


def is_severe_regression_vs_best(
    *, metric: str, direction: str, best_score: float | None, current_score: float
) -> bool:
    drop = score_drop_vs_best(best_score=best_score, current_score=current_score, direction=direction)
    if drop is None:
        return False
    threshold = regression_drop_threshold(metric=metric, direction=direction)
    return drop > threshold


def is_conservative_feature_collapse(payload: dict[str, object] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    robust_subset_payload = payload.get("robust_subset_report")
    if not isinstance(robust_subset_payload, dict):
        return False
    selected_feature_count_raw = payload.get("selected_feature_count")
    selected_feature_count = (
        int(selected_feature_count_raw) if isinstance(selected_feature_count_raw, int | float) else None
    )
    selected_features = robust_subset_payload.get("selected_features")
    selected_subset_size = len(selected_features) if isinstance(selected_features, list) else None
    if selected_feature_count is not None and selected_feature_count <= CONSERVATIVE_COLLAPSE_MAX_FEATURES:
        return True
    if selected_subset_size is not None and selected_subset_size <= CONSERVATIVE_COLLAPSE_MAX_FEATURES:
        return True
    return False


def effective_best_score_for_progress(
    *,
    prev_best: float | None,
    current_score: float,
    top1_score: float | None,
    direction: str,
) -> tuple[float | None, dict[str, object] | None]:
    """
    Clamp an implausible previous best into a top1-proximate band before no-improve checks.

    This avoids driving improvement-mode escalation from a stale outlier best score.
    """
    if prev_best is None or top1_score is None:
        return prev_best, None

    margin = BEST_SCORE_OUTLIER_TOP1_ABS_MARGIN + (
        BEST_SCORE_OUTLIER_TOP1_REL_MARGIN * max(abs(float(top1_score)), 1.0)
    )
    if direction == "maximize":
        cap = float(top1_score) + margin
        if float(prev_best) > cap and float(current_score) <= cap:
            return cap, {
                "applied": True,
                "reason": "clip_prev_best_above_top1_band",
                "prev_best": float(prev_best),
                "effective_best": cap,
                "top1_score": float(top1_score),
                "margin": float(margin),
            }
        return prev_best, None

    floor = float(top1_score) - margin
    if float(prev_best) < floor and float(current_score) >= floor:
        return floor, {
            "applied": True,
            "reason": "clip_prev_best_below_top1_band",
            "prev_best": float(prev_best),
            "effective_best": floor,
            "top1_score": float(top1_score),
            "margin": float(margin),
        }
    return prev_best, None


def should_update_best_accuracy_candidate(
    *,
    current_potential: dict[str, object],
    best_potential: dict[str, object] | None,
    current_score: float,
    best_score: float | None,
    direction: str,
) -> bool:
    if best_potential is None or best_score is None:
        return True
    current_priority = tolerant_int(current_potential.get("frontier_priority")) or 0
    best_priority = tolerant_int(best_potential.get("frontier_priority")) or 0
    if current_priority != best_priority:
        return current_priority > best_priority
    return should_update_best_score(best_score, current_score, direction, 0.0)
