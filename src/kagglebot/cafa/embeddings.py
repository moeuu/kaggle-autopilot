from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel


class ESM2Embedder:
    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D", device: str | None = None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def _batch_iter(self, sequences: list[str], batch_size: int) -> Iterable[list[str]]:
        for i in range(0, len(sequences), batch_size):
            yield sequences[i : i + batch_size]

    @torch.no_grad()
    def embed_sequences(
        self,
        sequences: list[str],
        batch_size: int = 8,
        max_length: int = 1024,
    ) -> torch.Tensor:
        """Extract embeddings for list of sequences."""
        embeddings: list[torch.Tensor] = []
        total_batches = max(1, (len(sequences) + batch_size - 1) // batch_size)
        for batch in tqdm(self._batch_iter(sequences, batch_size), total=total_batches, desc="Embedding", unit="batch"):
            encoded = self.tokenizer(
                batch,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1)
            masked = hidden * mask
            lengths = mask.sum(dim=1).clamp(min=1)
            pooled = masked.sum(dim=1) / lengths
            embeddings.append(pooled.cpu())
        return torch.cat(embeddings, dim=0)

    def save_embeddings(self, embeddings: torch.Tensor, filepath: str | Path) -> None:
        """Save embeddings to disk."""
        torch.save(embeddings, str(filepath))

    def load_embeddings(self, filepath: str | Path) -> torch.Tensor:
        """Load cached embeddings."""
        return torch.load(str(filepath), map_location="cpu")
