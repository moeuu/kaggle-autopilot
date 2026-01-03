"""Self-contained CAFA 6 training script for Kaggle kernels."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from math import floor, log10
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def _install_dependencies() -> None:
    packages = [
        "transformers",
        "peft",
        "accelerate",
        "obonet",
        "networkx",
        "biopython",
        "pandas",
        "numpy",
        "scikit-learn",
        "tqdm",
    ]
    try:
        import transformers  # noqa: F401
        import peft  # noqa: F401
        import obonet  # noqa: F401
        import networkx  # noqa: F401
    except Exception:
        import subprocess

        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _time_guard(start: float, limit_seconds: int) -> None:
    if time.time() - start > limit_seconds:
        raise TimeoutError("Time budget exceeded in Kaggle kernel.")


def _iter_fasta(path: Path):
    header = None
    seq_parts = []
    with path.open("r") as handle:
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


def _normalize_fasta_id(header: str) -> str:
    header = header.strip()
    if header.startswith(">"):
        header = header[1:]
    if "|" in header:
        parts = header.split("|")
        if len(parts) >= 2 and parts[1].strip():
            return parts[1].strip()
    return header.split()[0].strip()


def parse_fasta(path: Path) -> dict[str, str]:
    sequences = {}
    for header, seq in _iter_fasta(path):
        seq_id = _normalize_fasta_id(header)
        sequences[seq_id] = seq
    return sequences


def load_ia_weights(path: Path) -> dict[str, float]:
    weights = {}
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            try:
                weights[parts[0]] = float(parts[1])
            except ValueError:
                continue
    return weights


def create_multilabel_targets(train_terms_path: Path, go_graph) -> tuple[dict[str, list[str]], dict[str, int], dict[str, np.ndarray]]:
    import csv

    protein_to_terms = defaultdict(list)
    term_to_aspect = {}
    with train_terms_path.open("r") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            protein = (row.get("EntryID") or "").strip()
            term = (row.get("term") or "").strip()
            aspect = (row.get("aspect") or "").strip()
            if not protein or not term:
                continue
            if term not in go_graph:
                continue
            protein_to_terms[protein].append(term)
            if term not in term_to_aspect and aspect:
                term_to_aspect[term] = aspect

    aspect_map = {"F": "MF", "P": "BP", "C": "CC"}
    ontology_terms = {"MF": [], "BP": [], "CC": []}
    for term, aspect in term_to_aspect.items():
        ontology = aspect_map.get(aspect.upper())
        if ontology is None:
            continue
        ontology_terms[ontology].append(term)

    for key in ontology_terms:
        ontology_terms[key] = sorted(set(ontology_terms[key]))

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

    return dict(protein_to_terms), term_to_idx, {"MF": mask_mf, "BP": mask_bp, "CC": mask_cc}


class CAFADataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=1024, num_terms=None):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_terms = num_terms
        self.protein_ids = list(sequences.keys())

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, idx):
        protein_id = self.protein_ids[idx]
        sequence = self.sequences[protein_id]
        encoded = self.tokenizer(
            sequence,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "protein_id": protein_id,
        }
        if self.labels is not None:
            label_vec = torch.zeros(self.num_terms, dtype=torch.float32)
            for term_idx in self.labels.get(protein_id, []):
                if 0 <= term_idx < self.num_terms:
                    label_vec[term_idx] = 1.0
            item["labels"] = label_vec
        return item


class MultiLabelGOPredictor(torch.nn.Module):
    def __init__(self, esm_model, num_labels, embedding_dim=None, dropout=0.1, freeze_backbone=False):
        super().__init__()
        self.esm = esm_model
        if freeze_backbone:
            for param in self.esm.parameters():
                param.requires_grad = False
        if embedding_dim is None:
            embedding_dim = getattr(self.esm.config, "hidden_size", None) or getattr(self.esm.config, "dim", None)
        self.dropout = torch.nn.Dropout(dropout)
        self.mf_head = torch.nn.Linear(embedding_dim, num_labels["MF"])
        self.bp_head = torch.nn.Linear(embedding_dim, num_labels["BP"])
        self.cc_head = torch.nn.Linear(embedding_dim, num_labels["CC"])

    def forward(self, input_ids, attention_mask):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = self.dropout(pooled)
        return {
            "MF": torch.sigmoid(self.mf_head(pooled)),
            "BP": torch.sigmoid(self.bp_head(pooled)),
            "CC": torch.sigmoid(self.cc_head(pooled)),
        }


def create_lora_model(base_model, lora_rank=16, lora_alpha=32):
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["query", "key", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    return get_peft_model(base_model, lora_config)


class IAWeightedAsymmetricLoss(torch.nn.Module):
    def __init__(self, ia_weights, ontology_masks, gamma_pos=0, gamma_neg=4, clip=0.05, reduction="mean"):
        super().__init__()
        self.ia_weights = ia_weights
        self.ontology_masks = ontology_masks
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction

    def forward(self, predictions, targets):
        total_loss = 0.0
        eps = 1e-7
        for ontology in ("MF", "BP", "CC"):
            mask = self.ontology_masks[ontology].to(targets.device)
            preds = predictions[ontology]
            targs = targets[:, mask]
            weights = self.ia_weights[mask].to(preds.device)

            preds_pos = torch.clamp(preds, min=self.clip, max=1.0 - eps)
            preds_neg = torch.clamp(preds, min=eps, max=1.0 - self.clip)

            pos_loss = -targs * (1 - preds_pos) ** self.gamma_pos * torch.log(preds_pos)
            neg_loss = -(1 - targs) * preds_neg**self.gamma_neg * torch.log(1 - preds_neg)

            loss = (pos_loss + neg_loss) * weights.unsqueeze(0)
            if self.reduction == "mean":
                total_loss += loss.mean()
            else:
                total_loss += loss.sum()
        return total_loss / 3.0


def compute_ia_weighted_f1(y_true, y_pred_probs, threshold, ia_weights, ontology_masks):
    y_pred = (y_pred_probs > threshold).astype(int)
    results = {}
    for ontology in ("MF", "BP", "CC"):
        mask = ontology_masks[ontology]
        yt = y_true[:, mask]
        yp = y_pred[:, mask]
        weights = ia_weights[mask]
        tp = (yt * yp).sum(axis=0)
        fp = ((1 - yt) * yp).sum(axis=0)
        fn = (yt * (1 - yp)).sum(axis=0)
        prec = (weights * tp).sum() / ((weights * (tp + fp)).sum() + 1e-10)
        rec = (weights * tp).sum() / ((weights * (tp + fn)).sum() + 1e-10)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        results[f"{ontology}_f1"] = f1
    results["mean_f1"] = (results["MF_f1"] + results["BP_f1"] + results["CC_f1"]) / 3.0
    return results


def optimize_thresholds(y_true, y_pred_probs, ia_weights, ontology_masks):
    from sklearn.metrics import precision_recall_curve

    thresholds = {"MF": {}, "BP": {}, "CC": {}}
    for ontology in ("MF", "BP", "CC"):
        mask = ontology_masks[ontology]
        term_indices = np.where(mask)[0]
        for term_idx in term_indices:
            yt = y_true[:, term_idx]
            yp = y_pred_probs[:, term_idx]
            if yt.sum() == 0:
                thresholds[ontology][int(term_idx)] = 0.5
                continue
            precisions, recalls, thr = precision_recall_curve(yt, yp)
            thr = np.append(thr, 1.0)
            ia = ia_weights[term_idx]
            denom = ia * precisions + recalls + 1e-10
            f1_scores = 2 * ia * precisions * recalls / denom
            best_idx = int(np.argmax(f1_scores))
            thresholds[ontology][int(term_idx)] = float(thr[best_idx])
    return thresholds


def apply_thresholds(y_pred_probs, thresholds):
    y_pred = np.zeros_like(y_pred_probs, dtype=int)
    for ontology in ("MF", "BP", "CC"):
        for term_idx, threshold in thresholds[ontology].items():
            y_pred[:, term_idx] = (y_pred_probs[:, term_idx] > threshold).astype(int)
    return y_pred


def build_ancestor_index(go_graph, term_to_idx):
    import networkx as nx

    ancestor_map = {}
    for term, idx in term_to_idx.items():
        if term not in go_graph:
            continue
        ancestors = nx.ancestors(go_graph, term)
        ancestor_map[idx] = [term_to_idx[a] for a in ancestors if a in term_to_idx]
    return ancestor_map


def propagate_go_predictions(predictions, scores, go_graph, term_to_idx):
    prop_predictions = predictions.copy()
    prop_scores = scores.copy()
    ancestor_map = build_ancestor_index(go_graph, term_to_idx)
    for i in range(predictions.shape[0]):
        positive_indices = np.where(predictions[i] == 1)[0]
        for term_idx in positive_indices:
            for anc_idx in ancestor_map.get(int(term_idx), []):
                prop_predictions[i, anc_idx] = 1
                prop_scores[i, anc_idx] = max(prop_scores[i, anc_idx], scores[i, term_idx])
    return prop_predictions, prop_scores


def round_to_n_significant_figures(x: float, n: int = 3) -> float:
    if x == 0:
        return 0.0
    return round(x, -int(floor(log10(abs(x)))) + (n - 1))


def create_submission(predictions, scores, protein_ids, idx_to_term, output_path):
    rows = []
    for i, protein_id in enumerate(protein_ids):
        term_indices = np.where(predictions[i] == 1)[0]
        for term_idx in term_indices:
            go_term = idx_to_term[int(term_idx)]
            score = float(scores[i, term_idx])
            if score <= 0:
                continue
            score = round_to_n_significant_figures(score, 3)
            score = min(max(score, 1e-10), 1.0)
            rows.append(f"{protein_id}\t{go_term}\t{score}")
    output_path.write_text("\n".join(rows))


def main() -> None:
    start = time.time()
    max_seconds = int(9 * 60 * 60 - 300)

    _install_dependencies()
    _set_seed(42)

    import obonet
    from transformers import AutoTokenizer, EsmModel

    slug = "cafa-6-protein-function-prediction"
    input_root = Path("/kaggle/input") / slug
    train_dir = input_root / "Train"
    test_dir = input_root / "Test"
    working = Path("/kaggle/working")

    config = {
        "embedding_model": "facebook/esm2_t33_650M_UR50D",
        "batch_size": 8,
        "max_epochs": 3,
        "learning_rate": 1e-4,
        "lora_rank": 16,
        "lora_alpha": 32,
        "max_sequence_length": 1024,
        "warmup_ratio": 0.1,
        "weight_decay": 1e-5,
        "gradient_accumulation_steps": 2,
        "asymmetric_loss_gamma_neg": 4,
        "asymmetric_loss_gamma_pos": 0,
        "max_go_terms_per_protein": 1500,
    }

    train_sequences = parse_fasta(train_dir / "train_sequences.fasta")
    test_sequences = parse_fasta(test_dir / "testsuperset.fasta")
    go_graph = obonet.read_obo(str(train_dir / "go-basic.obo"))
    ia_weights_dict = load_ia_weights(input_root / "IA.tsv")

    protein_to_terms, term_to_idx, ontology_masks = create_multilabel_targets(train_dir / "train_terms.tsv", go_graph)
    labels_idx = {pid: [term_to_idx[t] for t in terms if t in term_to_idx] for pid, terms in protein_to_terms.items()}

    num_terms = len(term_to_idx)
    idx_to_term = {idx: term for term, idx in term_to_idx.items()}

    ia_weights = np.zeros(num_terms, dtype=np.float32)
    for term, idx in term_to_idx.items():
        ia_weights[idx] = ia_weights_dict.get(term, 0.0)

    tokenizer = AutoTokenizer.from_pretrained(config["embedding_model"])

    train_ids = sorted(train_sequences.keys())
    split_idx = int(len(train_ids) * 0.8)
    train_split = {pid: train_sequences[pid] for pid in train_ids[:split_idx]}
    val_split = {pid: train_sequences[pid] for pid in train_ids[split_idx:]}

    train_dataset = CAFADataset(train_split, labels_idx, tokenizer, config["max_sequence_length"], num_terms)
    val_dataset = CAFADataset(val_split, labels_idx, tokenizer, config["max_sequence_length"], num_terms)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model = EsmModel.from_pretrained(config["embedding_model"])
    esm_model = create_lora_model(esm_model, config["lora_rank"], config["lora_alpha"])

    num_labels = {
        "MF": int(ontology_masks["MF"].sum()),
        "BP": int(ontology_masks["BP"].sum()),
        "CC": int(ontology_masks["CC"].sum()),
    }

    model = MultiLabelGOPredictor(esm_model, num_labels)
    model.to(device)

    criterion = IAWeightedAsymmetricLoss(
        torch.tensor(ia_weights, dtype=torch.float32),
        {k: torch.tensor(v, dtype=torch.bool) for k, v in ontology_masks.items()},
        gamma_pos=config["asymmetric_loss_gamma_pos"],
        gamma_neg=config["asymmetric_loss_gamma_neg"],
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    grad_steps = config["gradient_accumulation_steps"]
    scaler = GradScaler()

    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar, start=1):
            _time_guard(start, max_seconds)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(enabled=device.type == "cuda"):
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss = loss / grad_steps

            scaler.scale(loss).backward()

            if step % grad_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * grad_steps
            pbar.set_postfix({"loss": loss.item() * grad_steps})

        print(f"Epoch {epoch}: loss={total_loss / max(len(train_loader), 1):.4f}")

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)
            preds = torch.cat([outputs["MF"], outputs["BP"], outputs["CC"]], dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(batch["labels"])

    y_pred_probs = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    metrics = compute_ia_weighted_f1(y_true, y_pred_probs, 0.5, ia_weights, ontology_masks)
    print(f"Validation mean F1: {metrics['mean_f1']:.4f}")

    thresholds = optimize_thresholds(y_true, y_pred_probs, ia_weights, ontology_masks)

    test_dataset = CAFADataset(test_sequences, None, tokenizer, config["max_sequence_length"], num_terms)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2)

    scores_list = []
    protein_ids = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            _time_guard(start, max_seconds)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            protein_ids.extend(batch["protein_id"])
            outputs = model(input_ids, attention_mask)
            scores = torch.cat([outputs["MF"], outputs["BP"], outputs["CC"]], dim=1)
            scores_list.append(scores.cpu())

    all_scores = torch.cat(scores_list).numpy()
    predictions = apply_thresholds(all_scores, thresholds)
    predictions, all_scores = propagate_go_predictions(predictions, all_scores, go_graph, term_to_idx)

    for i in range(predictions.shape[0]):
        if predictions[i].sum() > config["max_go_terms_per_protein"]:
            term_scores = all_scores[i]
            top_indices = np.argsort(term_scores)[-config["max_go_terms_per_protein"] :]
            mask = np.zeros_like(predictions[i])
            mask[top_indices] = 1
            predictions[i] = predictions[i] * mask

    submission_path = working / "submission.csv"
    create_submission(predictions, all_scores, protein_ids, idx_to_term, submission_path)

    metrics_path = working / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    main()
