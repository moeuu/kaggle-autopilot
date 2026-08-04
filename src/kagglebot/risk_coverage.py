from __future__ import annotations

from typing import Any

import numpy as np


def risk_coverage_auc(
    risks: Any,
    confidences: Any,
    *,
    sample_weight: Any | None = None,
) -> float:
    """Integrate cumulative risk over coverage, ordered by confidence.

    ``risks`` contains one non-negative loss per prediction and ``confidences``
    determines the descending selective-prediction order. The initial rectangle
    from zero coverage is included so a constant risk of one has AURC one.
    """
    risk_values = np.asarray(risks, dtype=float).reshape(-1)
    confidence_values = np.asarray(confidences, dtype=float).reshape(-1)
    if risk_values.size == 0:
        raise ValueError("AURC requires at least one risk value.")
    if risk_values.size != confidence_values.size:
        raise ValueError("AURC risks and confidences must have matching lengths.")
    if not np.all(np.isfinite(risk_values)) or np.any(risk_values < 0.0):
        raise ValueError("AURC risks must be finite and non-negative.")
    if not np.all(np.isfinite(confidence_values)):
        raise ValueError("AURC confidences must be finite.")

    if sample_weight is None:
        weights = np.ones(risk_values.size, dtype=float)
    else:
        weights = np.asarray(sample_weight, dtype=float).reshape(-1)
        if weights.size != risk_values.size:
            raise ValueError("AURC sample weights must match the risk length.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("AURC sample weights must be finite and non-negative.")
        positive = weights > 0.0
        risk_values = risk_values[positive]
        confidence_values = confidence_values[positive]
        weights = weights[positive]
        if risk_values.size == 0:
            raise ValueError("AURC requires at least one positive sample weight.")

    order = np.argsort(-confidence_values, kind="stable")
    ordered_risks = risk_values[order]
    ordered_weights = weights[order]
    cumulative_weights = np.cumsum(ordered_weights)
    cumulative_risk = np.cumsum(ordered_risks * ordered_weights) / cumulative_weights
    coverage = cumulative_weights / cumulative_weights[-1]

    area = cumulative_risk[0] * coverage[0]
    if cumulative_risk.size > 1:
        widths = np.diff(coverage)
        area += np.sum(widths * (cumulative_risk[:-1] + cumulative_risk[1:]) / 2.0)
    return float(area)
