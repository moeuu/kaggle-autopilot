from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, EsmModel, get_cosine_schedule_with_warmup

from kagglebot.cafa.data_loader import (
    CAFADataset,
    create_multilabel_targets,
    load_go_graph,
    load_ia_weights,
    parse_fasta,
)
from kagglebot.cafa.loss import IAWeightedAsymmetricLoss
from kagglebot.cafa.model import MultiLabelGOPredictor, create_lora_model
from kagglebot.cafa.threshold_optimizer import optimize_thresholds_per_ontology, save_thresholds
from kagglebot.paths import CompetitionPaths


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_label_index(protein_to_terms: dict[str, list[str]], term_to_idx: dict[str, int]) -> dict[str, list[int]]:
    labels: dict[str, list[int]] = {}
    for protein_id, terms in protein_to_terms.items():
        indices = [term_to_idx[term] for term in terms if term in term_to_idx]
        labels[protein_id] = indices
    return labels


def _collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)
            scores = torch.cat([outputs["MF"], outputs["BP"], outputs["CC"]], dim=1)
            preds.append(scores.cpu())
            labels.append(batch["labels"])
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()


def _default_paths(slug: str, artifacts_dir: Path | None) -> CompetitionPaths:
    if artifacts_dir is None:
        return CompetitionPaths(slug=slug, artifacts_dir=Path("artifacts").resolve())
    return CompetitionPaths(slug=slug, artifacts_dir=artifacts_dir.resolve())


def main(config_path: str | Path | None = None, artifacts_dir: str | Path | None = None) -> dict[str, Any]:
    slug = "cafa-6-protein-function-prediction"
    paths = _default_paths(slug, Path(artifacts_dir) if artifacts_dir else None)

    if config_path is None:
        config_path = paths.plan_path
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    _set_seed(int(config.get("seed", 42)))

    data_dir = paths.data_dir
    train_dir = data_dir / "Train"

    train_sequences = parse_fasta(train_dir / "train_sequences.fasta")
    go_graph = load_go_graph(train_dir / "go-basic.obo")
    ia_weights_dict = load_ia_weights(data_dir / "IA.tsv")

    protein_to_terms, term_to_idx, ontology_masks = create_multilabel_targets(train_dir / "train_terms.tsv", go_graph)
    labels_idx = _build_label_index(protein_to_terms, term_to_idx)

    num_terms = len(term_to_idx)
    idx_to_term = {idx: term for term, idx in term_to_idx.items()}

    ia_weights = np.zeros(num_terms, dtype=np.float32)
    for term, idx in term_to_idx.items():
        ia_weights[idx] = ia_weights_dict.get(term, 0.0)

    protein_ids = sorted(train_sequences.keys())
    train_ids, val_ids = train_test_split(
        protein_ids,
        test_size=float(config.get("holdout_frac", 0.2)),
        random_state=int(config.get("seed", 42)),
    )

    train_sequences_split = {pid: train_sequences[pid] for pid in train_ids}
    val_sequences_split = {pid: train_sequences[pid] for pid in val_ids}

    tokenizer = AutoTokenizer.from_pretrained(config["embedding_model"])

    train_dataset = CAFADataset(
        train_sequences_split,
        labels_idx,
        tokenizer,
        max_length=int(config.get("max_sequence_length", 1024)),
        num_terms=num_terms,
    )
    val_dataset = CAFADataset(
        val_sequences_split,
        labels_idx,
        tokenizer,
        max_length=int(config.get("max_sequence_length", 1024)),
        num_terms=num_terms,
    )

    batch_size = int(config.get("batch_size", 8))
    num_workers = int(config.get("num_workers", 0))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    esm_model = EsmModel.from_pretrained(config["embedding_model"])
    esm_model = create_lora_model(esm_model, int(config.get("lora_rank", 16)), int(config.get("lora_alpha", 32)))

    num_labels = {
        "MF": int(ontology_masks["MF"].sum()),
        "BP": int(ontology_masks["BP"].sum()),
        "CC": int(ontology_masks["CC"].sum()),
    }

    model = MultiLabelGOPredictor(esm_model, num_labels=num_labels)
    model.to(device)

    ia_weights_tensor = torch.tensor(ia_weights, dtype=torch.float32)
    ontology_masks_tensor = {k: torch.tensor(v, dtype=torch.bool) for k, v in ontology_masks.items()}

    criterion = IAWeightedAsymmetricLoss(
        ia_weights_tensor,
        ontology_masks_tensor,
        gamma_pos=float(config.get("asymmetric_loss_gamma_pos", 0)),
        gamma_neg=float(config.get("asymmetric_loss_gamma_neg", 4)),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )

    grad_steps = int(config.get("gradient_accumulation_steps", 1))
    total_steps = max(1, (len(train_loader) + grad_steps - 1) // grad_steps)
    total_steps *= int(config.get("max_epochs", 1))
    warmup_steps = int(total_steps * float(config.get("warmup_ratio", 0.1)))

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = paths.runs_dir / f"cafa_train_{run_id}"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    from kagglebot.cafa.trainer import CAFATrainer

    trainer = CAFATrainer(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        config,
        checkpoint_dir,
        ia_weights,
        ontology_masks,
    )

    trainer.train(int(config.get("max_epochs", 1)))

    y_true, y_pred_probs = _collect_predictions(model, val_loader, device)
    thresholds = optimize_thresholds_per_ontology(
        y_true,
        y_pred_probs,
        ia_weights,
        ontology_masks,
        n_thresholds=100,
    )
    thresholds_path = run_dir / "thresholds.json"
    save_thresholds(thresholds, thresholds_path)

    mapping_path = run_dir / "term_mapping.json"
    mapping_payload = {"term_to_idx": term_to_idx, "idx_to_term": {str(k): v for k, v in idx_to_term.items()}}
    mapping_path.write_text(json.dumps(mapping_payload, indent=2), encoding="utf-8")

    return {
        "run_dir": str(run_dir),
        "thresholds_path": str(thresholds_path),
        "term_mapping_path": str(mapping_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CAFA 6 ESM-2 LoRA model.")
    parser.add_argument("--config-path", type=str, default=None, help="Path to plan.json config.")
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Artifacts directory root.")
    args = parser.parse_args()
    main(config_path=args.config_path, artifacts_dir=args.artifacts_dir)
