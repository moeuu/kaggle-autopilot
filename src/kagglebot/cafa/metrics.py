from __future__ import annotations

from typing import Any

import numpy as np


def compute_ia_weighted_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ia_weights: np.ndarray,
    ontology_mask: np.ndarray | None = None,
) -> float:
    """Compute IA-weighted precision."""
    if ontology_mask is not None:
        y_true = y_true[:, ontology_mask]
        y_pred = y_pred[:, ontology_mask]
        ia_weights = ia_weights[ontology_mask]

    tp = (y_true * y_pred).sum(axis=0)
    fp = ((1 - y_true) * y_pred).sum(axis=0)

    numerator = (ia_weights * tp).sum()
    denominator = (ia_weights * (tp + fp)).sum()
    return float(numerator / (denominator + 1e-10))


def compute_ia_weighted_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ia_weights: np.ndarray,
    ontology_mask: np.ndarray | None = None,
) -> float:
    """Compute IA-weighted recall."""
    if ontology_mask is not None:
        y_true = y_true[:, ontology_mask]
        y_pred = y_pred[:, ontology_mask]
        ia_weights = ia_weights[ontology_mask]

    tp = (y_true * y_pred).sum(axis=0)
    fn = (y_true * (1 - y_pred)).sum(axis=0)

    numerator = (ia_weights * tp).sum()
    denominator = (ia_weights * (tp + fn)).sum()
    return float(numerator / (denominator + 1e-10))


def compute_ia_weighted_f1(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float,
    ia_weights: np.ndarray,
    ontology_masks: dict[str, np.ndarray],
    knowledge_types: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute IA-weighted F1 per ontology and mean."""
    y_pred = (y_pred_probs > threshold).astype(int)

    results: dict[str, Any] = {}
    for ontology in ("MF", "BP", "CC"):
        mask = ontology_masks[ontology]
        prec = compute_ia_weighted_precision(y_true, y_pred, ia_weights, mask)
        rec = compute_ia_weighted_recall(y_true, y_pred, ia_weights, mask)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)

        results[f"{ontology}_precision"] = prec
        results[f"{ontology}_recall"] = rec
        results[f"{ontology}_f1"] = f1

    results["mean_f1"] = (results["MF_f1"] + results["BP_f1"] + results["CC_f1"]) / 3.0

    if knowledge_types is not None:
        results["knowledge_type"] = _compute_by_knowledge_type(
            y_true,
            y_pred_probs,
            threshold,
            ia_weights,
            ontology_masks,
            knowledge_types,
        )

    return results


def _compute_by_knowledge_type(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float,
    ia_weights: np.ndarray,
    ontology_masks: dict[str, np.ndarray],
    knowledge_types: np.ndarray,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    for ktype in np.unique(knowledge_types):
        mask = knowledge_types == ktype
        if mask.sum() == 0:
            continue
        metrics = compute_ia_weighted_f1(
            y_true[mask],
            y_pred_probs[mask],
            threshold,
            ia_weights,
            ontology_masks,
        )
        grouped[str(ktype)] = {"mean_f1": float(metrics["mean_f1"])}
    return grouped
