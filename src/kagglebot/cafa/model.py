from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


class MultiLabelGOPredictor(nn.Module):
    def __init__(
        self,
        esm_model: nn.Module,
        num_labels: dict[str, int],
        embedding_dim: int | None = None,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.esm = esm_model

        if freeze_backbone:
            for param in self.esm.parameters():
                param.requires_grad = False

        if embedding_dim is None:
            embedding_dim = getattr(self.esm.config, "hidden_size", None) or getattr(self.esm.config, "dim", None)
        if embedding_dim is None:
            raise ValueError("embedding_dim must be provided when model config has no hidden_size.")

        self.dropout = nn.Dropout(dropout)
        self.mf_head = nn.Linear(embedding_dim, num_labels["MF"])
        self.bp_head = nn.Linear(embedding_dim, num_labels["BP"])
        self.cc_head = nn.Linear(embedding_dim, num_labels["CC"])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        masked = hidden * mask
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = masked.sum(dim=1) / lengths
        pooled = self.dropout(pooled)
        return {
            "MF": torch.sigmoid(self.mf_head(pooled)),
            "BP": torch.sigmoid(self.bp_head(pooled)),
            "CC": torch.sigmoid(self.cc_head(pooled)),
        }


def create_lora_model(base_model: nn.Module, lora_rank: int = 16, lora_alpha: int = 32) -> nn.Module:
    """Apply LoRA to ESM-2 model."""
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["query", "key", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    return get_peft_model(base_model, lora_config)
