from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kagglebot.cafa.data_loader import create_multilabel_targets, parse_fasta
from kagglebot.cafa.inference import propagate_go_predictions, run_inference
from kagglebot.cafa.submission import create_submission, validate_submission


class DummyTokenizer:
    def __call__(self, sequence: str, max_length: int, truncation: bool, padding: str, return_tensors: str):
        length = max_length
        input_ids = torch.zeros((1, length), dtype=torch.long)
        attention_mask = torch.ones((1, length), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class DummyModel(torch.nn.Module):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        batch = input_ids.shape[0]
        mf = torch.full((batch, 2), 0.6)
        bp = torch.full((batch, 1), 0.4)
        cc = torch.full((batch, 1), 0.7)
        return {"MF": mf, "BP": bp, "CC": cc}


def test_fasta_parsing(tmp_path: Path) -> None:
    fasta_path = tmp_path / "sample.fasta"
    fasta_path.write_text(
        ">sp|P12345|Example protein\nMKTAA\n>Q9XYZ1 Some desc\nGGGCCC\n",
        encoding="utf-8",
    )
    sequences = parse_fasta(fasta_path)
    assert sequences["P12345"] == "MKTAA"
    assert sequences["Q9XYZ1"] == "GGGCCC"


def test_create_multilabel_targets(tmp_path: Path) -> None:
    graph = nx.DiGraph()
    graph.add_edge("GO:0001", "GO:0002")
    graph.add_edge("GO:0003", "GO:0004")

    tsv_path = tmp_path / "train_terms.tsv"
    tsv_path.write_text(
        "EntryID\tterm\taspect\n" "P1\tGO:0001\tF\n" "P1\tGO:0003\tP\n" "P2\tGO:0004\tC\n",
        encoding="utf-8",
    )

    protein_to_terms, term_to_idx, ontology_masks = create_multilabel_targets(tsv_path, graph)
    assert set(protein_to_terms["P1"]) == {"GO:0001", "GO:0003"}
    assert "GO:0001" in term_to_idx
    assert ontology_masks["MF"].sum() == 1
    assert ontology_masks["BP"].sum() == 1
    assert ontology_masks["CC"].sum() == 1


def test_go_propagation() -> None:
    graph = nx.DiGraph()
    graph.add_edge("GO:0001", "GO:0002")
    term_to_idx = {"GO:0001": 0, "GO:0002": 1}

    predictions = np.array([[0, 1]], dtype=int)
    scores = np.array([[0.2, 0.9]], dtype=float)

    prop_pred, prop_scores = propagate_go_predictions(predictions, scores, graph, term_to_idx)
    assert prop_pred[0, 0] == 1
    assert prop_scores[0, 0] == 0.9


def test_submission_format(tmp_path: Path) -> None:
    predictions = np.array([[1, 0], [0, 1]], dtype=int)
    scores = np.array([[0.7, 0.1], [0.2, 0.9]], dtype=float)
    protein_ids = ["P1", "P2"]
    idx_to_term = {0: "GO:0001", 1: "GO:0002"}

    out_path = tmp_path / "submission.tsv"
    create_submission(predictions, scores, protein_ids, idx_to_term, out_path)
    assert out_path.exists()
    assert validate_submission(out_path, set(idx_to_term.values()))


def test_end_to_end_small(tmp_path: Path) -> None:
    sequences = {"P1": "AAAA", "P2": "BBBB"}
    tokenizer = DummyTokenizer()

    from kagglebot.cafa.data_loader import CAFADataset

    dataset = CAFADataset(sequences, labels=None, tokenizer=tokenizer, max_length=8)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    term_to_idx = {"GO:MF1": 0, "GO:MF2": 1, "GO:BP1": 2, "GO:CC1": 3}
    ontology_masks = {
        "MF": np.array([1, 1, 0, 0], dtype=bool),
        "BP": np.array([0, 0, 1, 0], dtype=bool),
        "CC": np.array([0, 0, 0, 1], dtype=bool),
    }
    thresholds = {
        "MF": {0: 0.5, 1: 0.5},
        "BP": {2: 0.5},
        "CC": {3: 0.5},
    }

    graph = nx.DiGraph()
    graph.add_edge("GO:MF1", "GO:MF2")

    model = DummyModel()
    predictions, scores, protein_ids = run_inference(
        model,
        loader,
        device=torch.device("cpu"),
        thresholds=thresholds,
        go_graph=graph,
        term_to_idx=term_to_idx,
        ontology_masks=ontology_masks,
        max_terms_per_protein=1500,
    )

    assert predictions.shape == scores.shape
    assert len(protein_ids) == 2
