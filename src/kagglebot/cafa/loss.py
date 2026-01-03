from __future__ import annotations

import torch
import torch.nn as nn


class IAWeightedAsymmetricLoss(nn.Module):
    def __init__(
        self,
        ia_weights: torch.Tensor,
        ontology_masks: dict[str, torch.Tensor],
        gamma_pos: float = 0,
        gamma_neg: float = 4,
        clip: float = 0.05,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.ia_weights = ia_weights
        self.ontology_masks = ontology_masks
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction

    def forward(self, predictions: dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """Compute asymmetric focal loss weighted by IA per ontology."""
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
