from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from kagglebot.cafa.metrics import compute_ia_weighted_f1


class CAFATrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device: torch.device,
        config: dict,
        checkpoint_dir: str | Path,
        ia_weights: np.ndarray,
        ontology_masks: dict[str, np.ndarray],
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.ia_weights = ia_weights
        self.ontology_masks = ontology_masks

        self.scaler = GradScaler() if config.get("use_mixed_precision") else None
        self.best_val_f1 = 0.0
        self.epochs_since_improvement = 0

    def train_epoch(self, epoch: int) -> float:
        """Train one epoch, return average loss."""
        self.model.train()
        total_loss = 0.0
        grad_steps = int(self.config.get("gradient_accumulation_steps", 1))

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with autocast(enabled=self.config.get("use_mixed_precision", False)):
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                loss = loss / grad_steps

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % grad_steps == 0 or step == len(self.train_loader):
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()

            total_loss += loss.item() * grad_steps
            pbar.set_postfix({"loss": loss.item() * grad_steps})

        return total_loss / max(len(self.train_loader), 1)

    def validate(self, threshold: float = 0.5) -> dict:
        """Validate and compute IA-weighted F1."""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]

                outputs = self.model(input_ids, attention_mask)
                preds = torch.cat([outputs["MF"], outputs["BP"], outputs["CC"]], dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels)

        all_preds_np = torch.cat(all_preds).numpy()
        all_labels_np = torch.cat(all_labels).numpy()

        return compute_ia_weighted_f1(all_labels_np, all_preds_np, threshold, self.ia_weights, self.ontology_masks)

    def save_checkpoint(self, epoch: int, val_f1: float) -> Path:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "val_f1": val_f1,
            "config": self.config,
        }
        path = self.checkpoint_dir / f"checkpoint_epoch{epoch}_f1{val_f1:.4f}.pt"
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
        return path

    def train(self, num_epochs: int) -> None:
        """Full training loop with early stopping."""
        patience = int(self.config.get("early_stopping_patience", 2))
        min_delta = float(self.config.get("early_stopping_min_delta", 0.0))

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate()
            val_f1 = float(val_metrics["mean_f1"])

            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_f1={val_f1:.4f}")
            self.save_checkpoint(epoch, val_f1)

            if val_f1 > self.best_val_f1 + min_delta:
                self.best_val_f1 = val_f1
                self.epochs_since_improvement = 0
            else:
                self.epochs_since_improvement += 1
                print("No improvement.")
                if self.epochs_since_improvement >= patience:
                    print("Early stopping triggered.")
                    break
