from __future__ import annotations

from kagglebot.scalar_utils import finite_float
from kagglebot.solver.metrics import normalize_direction


def should_update_best_score(best: float | None, current: float, direction: str, min_improvement: float) -> bool:
    if best is None:
        return True
    eps = 1e-9
    if direction == "minimize":
        improvement = best - current
        return improvement >= (min_improvement - eps)
    improvement = current - best
    return improvement >= (min_improvement - eps)


def score_gap(*, current: object, reference: object, direction: str) -> float | None:
    """Return positive delta when current is better than reference."""
    current_float = finite_float(current)
    reference_float = finite_float(reference)
    if current_float is None or reference_float is None:
        return None
    if normalize_direction(direction, default="minimize") == "minimize":
        return reference_float - current_float
    return current_float - reference_float
