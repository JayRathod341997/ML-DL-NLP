from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from .model import BERTClassifier


class Evaluator:
    """Runs model inference on a DataLoader and produces evaluation metrics."""

    def __init__(self, model: BERTClassifier, device: str = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def evaluate(
        self,
        loader: DataLoader,
        label_names: list[str] | None = None,
    ) -> dict:
        """Run inference and compute metrics.

        Returns:
            dict with keys: accuracy, f1_macro, classification_report, predictions, labels
        """
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"]
                logits = self.model(input_ids, attention_mask)
                preds = logits.argmax(dim=-1).cpu()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = float((all_preds == all_labels).mean())
        report = classification_report(
            all_labels, all_preds, target_names=label_names, output_dict=True
        )
        return {
            "accuracy": round(accuracy, 4),
            "f1_macro": round(report["macro avg"]["f1-score"], 4),
            "classification_report": report,
            "predictions": all_preds,
            "labels": all_labels,
        }

    def plot_confusion_matrix(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        label_names: list[str] | None = None,
        output_path: str | Path | None = None,
    ) -> None:
        cm = confusion_matrix(labels, predictions)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_names,
            yticklabels=label_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150)
        else:
            plt.show()
        plt.close()
