"""Evaluate a saved checkpoint on the test set.

Usage:
    uv run python scripts/evaluate.py --checkpoint checkpoints/best_model
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.config import TrainConfig
from src.dataset import load_datasets
from src.model import BERTClassifier
from src.evaluator import Evaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BERT text classifier")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", default="ag_news")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = TrainConfig(dataset_name=args.dataset, batch_size=args.batch_size, device=args.device)
    _, _, test_ds, label_names = load_datasets(config)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0)

    config.num_labels = len(label_names) if label_names else config.num_labels
    model = BERTClassifier(config)
    evaluator = Evaluator(model, device=args.device)
    results = evaluator.evaluate(test_loader, label_names=label_names)

    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1: {results['f1_macro']:.4f}")
    print("\nClassification Report:")
    for cls, metrics in results["classification_report"].items():
        if isinstance(metrics, dict):
            print(f"  {cls}: precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} f1={metrics['f1-score']:.3f}")

    evaluator.plot_confusion_matrix(
        results["labels"],
        results["predictions"],
        label_names=label_names,
        output_path="confusion_matrix.png",
    )
    print("\nConfusion matrix saved to confusion_matrix.png")


if __name__ == "__main__":
    main()
