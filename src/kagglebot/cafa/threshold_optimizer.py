from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm


def optimize_thresholds_per_ontology(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    ia_weights: np.ndarray,
    ontology_masks: dict[str, np.ndarray],
    n_thresholds: int = 100,
) -> dict[str, dict[int, float]]:
    """
    Optimize per-term thresholds to maximize IA-weighted F1.

    Returns:
        {ontology: {term_idx: optimal_threshold}}
    """
    optimized_thresholds: dict[str, dict[int, float]] = {}

    for ontology in ("MF", "BP", "CC"):
        mask = ontology_masks[ontology]
        ontology_thresholds: dict[int, float] = {}
        term_indices = np.where(mask)[0]

        for term_idx in tqdm(term_indices, desc=f"Optimizing {ontology}"):
            y_true_term = y_true[:, term_idx]
            y_pred_term = y_pred_probs[:, term_idx]

            if y_true_term.sum() == 0:
                ontology_thresholds[int(term_idx)] = 0.5
                continue

            precisions, recalls, thresholds = precision_recall_curve(y_true_term, y_pred_term)
            thresholds = np.append(thresholds, 1.0)
            ia = ia_weights[term_idx]

            denom = ia * precisions + recalls + 1e-10
            f1_scores = 2 * ia * precisions * recalls / denom

            if len(f1_scores) > n_thresholds:
                idxs = np.linspace(0, len(f1_scores) - 1, n_thresholds).astype(int)
                f1_scores = f1_scores[idxs]
                thresholds = thresholds[idxs]

            best_idx = int(np.argmax(f1_scores))
            ontology_thresholds[int(term_idx)] = float(thresholds[best_idx])

        optimized_thresholds[ontology] = ontology_thresholds

    return optimized_thresholds


def apply_thresholds(
    y_pred_probs: np.ndarray,
    thresholds: dict[str, dict[int, float]],
    ontology_masks: dict[str, np.ndarray],
) -> np.ndarray:
    """Apply per-term thresholds to predictions."""
    y_pred = np.zeros_like(y_pred_probs, dtype=int)

    for ontology in ("MF", "BP", "CC"):
        _ = ontology_masks.get(ontology)
        for term_idx, threshold in thresholds.get(ontology, {}).items():
            y_pred[:, term_idx] = (y_pred_probs[:, term_idx] > threshold).astype(int)

    return y_pred


def save_thresholds(thresholds: dict[str, dict[int, float]], path: str | Path) -> None:
    payload = {ontology: {str(k): v for k, v in values.items()} for ontology, values in thresholds.items()}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_thresholds(path: str | Path) -> dict[str, dict[int, float]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {ontology: {int(k): float(v) for k, v in values.items()} for ontology, values in raw.items()}
