from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset

try:  # pragma: no cover - optional dependency
    import obonet
except Exception:  # noqa: BLE001
    obonet = None


def _normalize_fasta_id(header: str) -> str:
    header = header.strip()
    if header.startswith(">"):
        header = header[1:]
    if "|" in header:
        parts = header.split("|")
        if len(parts) >= 2 and parts[1].strip():
            return parts[1].strip()
    return header.split()[0].strip()


def _iter_fasta(filepath: Path) -> Iterable[tuple[str, str]]:
    header: str | None = None
    seq_parts: list[str] = []
    with filepath.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            yield header, "".join(seq_parts)


def parse_fasta(filepath: str | Path) -> dict[str, str]:
    """Parse FASTA file, return {protein_id: sequence}."""
    path = Path(filepath)
    sequences: dict[str, str] = {}
    for header, seq in _iter_fasta(path):
        protein_id = _normalize_fasta_id(header)
        if protein_id:
            sequences[protein_id] = seq
    return sequences


def load_go_graph(obo_path: str | Path) -> nx.DiGraph:
    """Load GO graph using obonet."""
    if obonet is None:
        raise ImportError("obonet is required to load GO graph. Install with `uv add obonet`.")
    graph = obonet.read_obo(str(obo_path))
    if not isinstance(graph, nx.DiGraph):
        graph = nx.DiGraph(graph)
    return graph


def load_ia_weights(ia_path: str | Path) -> dict[str, float]:
    """Load information accretion weights {go_term: weight}."""
    weights: dict[str, float] = {}
    path = Path(ia_path)
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if len(row) < 2:
                continue
            term = row[0].strip()
            try:
                weight = float(row[1])
            except ValueError:
                continue
            weights[term] = weight
    return weights


def create_multilabel_targets(
    train_terms_path: str | Path,
    go_graph: nx.DiGraph,
) -> tuple[dict[str, list[str]], dict[str, int], dict[str, np.ndarray]]:
    """
    Returns:
    - protein_to_terms: {protein_id: [GO terms]}
    - term_to_idx: {GO term: index} (ordered by ontology blocks MF/BP/CC)
    - ontology_masks: {'MF': bool_array, 'BP': bool_array, 'CC': bool_array}
    """
    path = Path(train_terms_path)
    protein_to_terms: dict[str, list[str]] = defaultdict(list)
    term_to_aspect: dict[str, str] = {}

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            protein_id = (row.get("EntryID") or row.get("entry_id") or row.get("protein") or "").strip()
            term = (row.get("term") or row.get("go_term") or "").strip()
            aspect = (row.get("aspect") or row.get("ontology") or "").strip()
            if not protein_id or not term:
                continue
            if term not in go_graph:
                continue
            protein_to_terms[protein_id].append(term)
            if term not in term_to_aspect and aspect:
                term_to_aspect[term] = aspect

    aspect_map = {
        "F": "MF",
        "P": "BP",
        "C": "CC",
        "MF": "MF",
        "BP": "BP",
        "CC": "CC",
    }

    ontology_terms: dict[str, list[str]] = {"MF": [], "BP": [], "CC": []}
    for term, aspect in term_to_aspect.items():
        ontology = aspect_map.get(aspect.upper())
        if ontology is None:
            continue
        ontology_terms[ontology].append(term)

    for ontology in ontology_terms:
        ontology_terms[ontology] = sorted(set(ontology_terms[ontology]))

    ordered_terms = ontology_terms["MF"] + ontology_terms["BP"] + ontology_terms["CC"]
    term_to_idx = {term: idx for idx, term in enumerate(ordered_terms)}

    total_terms = len(ordered_terms)
    mf_size = len(ontology_terms["MF"])
    bp_size = len(ontology_terms["BP"])
    cc_size = len(ontology_terms["CC"])

    mask_mf = np.zeros(total_terms, dtype=bool)
    mask_bp = np.zeros(total_terms, dtype=bool)
    mask_cc = np.zeros(total_terms, dtype=bool)

    mask_mf[:mf_size] = True
    mask_bp[mf_size : mf_size + bp_size] = True
    mask_cc[mf_size + bp_size : mf_size + bp_size + cc_size] = True

    ontology_masks = {"MF": mask_mf, "BP": mask_bp, "CC": mask_cc}
    return dict(protein_to_terms), term_to_idx, ontology_masks


class CAFADataset(Dataset):
    """PyTorch dataset for CAFA proteins."""

    def __init__(
        self,
        sequences: dict[str, str],
        labels: dict[str, list[int]] | None,
        tokenizer: Any,
        max_length: int = 1024,
        num_terms: int | None = None,
    ) -> None:
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_terms = num_terms
        self.protein_ids = list(sequences.keys())
        if labels is not None and num_terms is None:
            raise ValueError("num_terms is required when labels are provided.")

    def __len__(self) -> int:
        return len(self.protein_ids)

    def _encode_sequence(self, sequence: str) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            sequence,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {"input_ids": encoded["input_ids"].squeeze(0), "attention_mask": encoded["attention_mask"].squeeze(0)}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        protein_id = self.protein_ids[idx]
        sequence = self.sequences[protein_id]
        encoded = self._encode_sequence(sequence)
        item: dict[str, Any] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "protein_id": protein_id,
        }
        if self.labels is not None:
            label_vec = torch.zeros(self.num_terms, dtype=torch.float32)
            for term_idx in self.labels.get(protein_id, []):
                if 0 <= term_idx < self.num_terms:
                    label_vec[term_idx] = 1.0
            item["labels"] = label_vec
        return item
