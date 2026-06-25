from __future__ import annotations


def should_update_best_score(best: float | None, current: float, direction: str, min_improvement: float) -> bool:
    if best is None:
        return True
    eps = 1e-9
    if direction == "minimize":
        improvement = best - current
        return improvement >= (min_improvement - eps)
    improvement = current - best
    return improvement >= (min_improvement - eps)
