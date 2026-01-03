"""CAFA 6 protein function prediction pipeline."""

from kagglebot.cafa.data_loader import CAFADataset, create_multilabel_targets, load_go_graph, load_ia_weights, parse_fasta
from kagglebot.cafa.embeddings import ESM2Embedder
from kagglebot.cafa.inference import run_inference
from kagglebot.cafa.loss import IAWeightedAsymmetricLoss
from kagglebot.cafa.metrics import compute_ia_weighted_f1
from kagglebot.cafa.model import MultiLabelGOPredictor, create_lora_model
from kagglebot.cafa.threshold_optimizer import apply_thresholds, optimize_thresholds_per_ontology

__all__ = [
    "CAFADataset",
    "ESM2Embedder",
    "IAWeightedAsymmetricLoss",
    "MultiLabelGOPredictor",
    "apply_thresholds",
    "compute_ia_weighted_f1",
    "create_lora_model",
    "create_multilabel_targets",
    "load_go_graph",
    "load_ia_weights",
    "optimize_thresholds_per_ontology",
    "parse_fasta",
    "run_inference",
]
