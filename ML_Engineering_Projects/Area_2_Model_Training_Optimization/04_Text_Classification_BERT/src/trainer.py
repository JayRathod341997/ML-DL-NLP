from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from .config import TrainConfig
from .model import BERTClassifier


class Trainer:
    """Training loop with AdamW, linear warmup, gradient clipping, and checkpointing."""

    def __init__(self, model: BERTClassifier, config: TrainConfig) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_accuracy = 0.0
        self.history: list[dict] = []

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.num_epochs + 1):
            train_loss = self._train_epoch(train_loader, scheduler, epoch)
            val_loss, val_accuracy = self._eval_epoch(val_loader)

            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_accuracy, 4),
            }
            self.history.append(record)
            print(
                f"Epoch {epoch}/{self.config.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f}"
            )

            if self.config.save_best and val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self._save_checkpoint("best_model")

        self._save_checkpoint("final_model")

    def _train_epoch(self, loader: DataLoader, scheduler, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _eval_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return total_loss / len(loader), correct / total

    def _save_checkpoint(self, name: str) -> None:
        path = self.config.output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        print(f"Checkpoint saved: {path}")
