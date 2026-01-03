from __future__ import annotations

from typing import Iterable

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from kagglebot.cafa.threshold_optimizer import apply_thresholds


def build_ancestor_index(go_graph: nx.DiGraph, term_to_idx: dict[str, int]) -> dict[int, list[int]]:
    ancestor_map: dict[int, list[int]] = {}
    for term, idx in term_to_idx.items():
        if term not in go_graph:
            continue
        ancestors = nx.ancestors(go_graph, term)
        ancestor_indices = [term_to_idx[a] for a in ancestors if a in term_to_idx]
        ancestor_map[idx] = ancestor_indices
    return ancestor_map


def propagate_go_predictions(
    predictions: np.ndarray,
    scores: np.ndarray,
    go_graph: nx.DiGraph,
    term_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate predictions up GO DAG.
    If protein has term T, all ancestors must also be predicted.
    Ancestor scores = max(current_score, child_score).
    """
    prop_predictions = predictions.copy()
    prop_scores = scores.copy()

    ancestor_map = build_ancestor_index(go_graph, term_to_idx)

    for i in tqdm(range(predictions.shape[0]), desc="Propagating GO terms"):
        positive_indices = np.where(predictions[i] == 1)[0]
        for term_idx in positive_indices:
            for anc_idx in ancestor_map.get(int(term_idx), []):
                prop_predictions[i, anc_idx] = 1
                prop_scores[i, anc_idx] = max(prop_scores[i, anc_idx], scores[i, term_idx])

    return prop_predictions, prop_scores


def _iter_batches(loader: Iterable[dict]) -> Iterable[dict]:
    for batch in loader:
        yield batch


def run_inference(
    model: torch.nn.Module,
    test_loader: Iterable[dict],
    device: torch.device,
    thresholds: dict[str, dict[int, float]],
    go_graph: nx.DiGraph,
    term_to_idx: dict[str, int],
    ontology_masks: dict[str, np.ndarray],
    max_terms_per_protein: int = 1500,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Run inference on test set with all post-processing.

    Returns:
        predictions: (num_proteins, num_terms) binary
        scores: (num_proteins, num_terms) probabilities
        protein_ids: list of protein IDs
    """
    model.eval()
    all_scores: list[torch.Tensor] = []
    protein_ids: list[str] = []

    with torch.no_grad():
        total = None
        try:
            total = len(test_loader)  # type: ignore[arg-type]
        except TypeError:
            total = None
        for batch in tqdm(_iter_batches(test_loader), desc="Inference", total=total):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            protein_ids.extend(batch["protein_id"])

            outputs = model(input_ids, attention_mask)
            scores_batch = torch.cat([outputs["MF"], outputs["BP"], outputs["CC"]], dim=1)
            all_scores.append(scores_batch.cpu())

    all_scores_np = torch.cat(all_scores).numpy()

    predictions = apply_thresholds(all_scores_np, thresholds, ontology_masks)
    predictions, all_scores_np = propagate_go_predictions(predictions, all_scores_np, go_graph, term_to_idx)

    for i in range(predictions.shape[0]):
        if predictions[i].sum() > max_terms_per_protein:
            term_scores = all_scores_np[i]
            top_indices = np.argsort(term_scores)[-max_terms_per_protein:]
            mask = np.zeros_like(predictions[i])
            mask[top_indices] = 1
            predictions[i] = predictions[i] * mask

    return predictions, all_scores_np, protein_ids
