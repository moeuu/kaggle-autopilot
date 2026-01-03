from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, EsmModel

from kagglebot.cafa.data_loader import CAFADataset, create_multilabel_targets, load_go_graph, parse_fasta
from kagglebot.cafa.inference import run_inference
from kagglebot.cafa.model import MultiLabelGOPredictor
from kagglebot.cafa.submission import create_submission, validate_submission
from kagglebot.cafa.threshold_optimizer import load_thresholds
from kagglebot.paths import CompetitionPaths


def _load_term_mapping(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    term_to_idx = {k: int(v) for k, v in payload["term_to_idx"].items()}
    idx_to_term = {int(k): v for k, v in payload["idx_to_term"].items()}
    return term_to_idx, idx_to_term


def main(
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    thresholds_path: str | Path | None = None,
    term_mapping_path: str | Path | None = None,
    output_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
) -> None:
    slug = "cafa-6-protein-function-prediction"
    paths = CompetitionPaths(slug=slug, artifacts_dir=Path(artifacts_dir) if artifacts_dir else Path("artifacts"))

    if config_path is None:
        config_path = paths.plan_path
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))

    data_dir = paths.data_dir
    train_dir = data_dir / "Train"
    test_dir = data_dir / "Test"

    go_graph = load_go_graph(train_dir / "go-basic.obo")
    protein_to_terms, term_to_idx, ontology_masks = create_multilabel_targets(train_dir / "train_terms.tsv", go_graph)

    if term_mapping_path:
        term_to_idx, idx_to_term = _load_term_mapping(Path(term_mapping_path))
    else:
        idx_to_term = {idx: term for term, idx in term_to_idx.items()}

    test_sequences = parse_fasta(test_dir / "testsuperset.fasta")

    tokenizer = AutoTokenizer.from_pretrained(config["embedding_model"])
    test_dataset = CAFADataset(
        test_sequences,
        labels=None,
        tokenizer=tokenizer,
        max_length=int(config.get("max_sequence_length", 1024)),
    )

    batch_size = int(config.get("batch_size", 8))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model = EsmModel.from_pretrained(config["embedding_model"])
    num_labels = {
        "MF": int(ontology_masks["MF"].sum()),
        "BP": int(ontology_masks["BP"].sum()),
        "CC": int(ontology_masks["CC"].sum()),
    }
    model = MultiLabelGOPredictor(esm_model, num_labels=num_labels)

    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required for inference")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if thresholds_path is None:
        raise ValueError("thresholds_path is required for inference")
    thresholds = load_thresholds(thresholds_path)

    predictions, scores, protein_ids = run_inference(
        model,
        test_loader,
        device,
        thresholds,
        go_graph,
        term_to_idx,
        ontology_masks,
        max_terms_per_protein=int(config.get("max_go_terms_per_protein", 1500)),
    )

    if output_path is None:
        output_path = paths.submissions_dir / "cafa_submission.tsv"

    create_submission(predictions, scores, protein_ids, idx_to_term, output_path)
    validate_submission(output_path, set(term_to_idx.keys()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CAFA 6 inference and create submission.")
    parser.add_argument("--config-path", type=str, default=None, help="Path to plan.json config.")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--thresholds-path", type=str, required=True, help="Path to thresholds.json.")
    parser.add_argument("--term-mapping-path", type=str, default=None, help="Path to term mapping JSON.")
    parser.add_argument("--output-path", type=str, default=None, help="Output submission path.")
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Artifacts directory root.")
    args = parser.parse_args()
    main(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        thresholds_path=args.thresholds_path,
        term_mapping_path=args.term_mapping_path,
        output_path=args.output_path,
        artifacts_dir=args.artifacts_dir,
    )
