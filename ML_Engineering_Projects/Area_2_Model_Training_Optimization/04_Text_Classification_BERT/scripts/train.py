"""Train a DistilBERT text classifier.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --dataset ag_news --epochs 3 --lr 2e-5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.config import TrainConfig
from src.dataset import load_datasets
from src.model import BERTClassifier
from src.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BERT text classifier")
    parser.add_argument("--dataset", default="ag_news")
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = TrainConfig(
        model_name=args.model,
        dataset_name=args.dataset,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(f"Config: {config}")
    print(f"Device: {args.device}")

    print("Loading datasets...")
    train_ds, val_ds, test_ds, label_names = load_datasets(config)
    config.num_labels = len(label_names) if label_names else config.num_labels
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"Labels: {label_names}")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, num_workers=0)

    print("Initialising model...")
    model = BERTClassifier(config)

    trainer = Trainer(model, config)
    print("Training...")
    trainer.train(train_loader, val_loader)
    print(f"\nBest val accuracy: {trainer.best_val_accuracy:.4f}")
    print(f"Training history: {trainer.history}")


if __name__ == "__main__":
    main()
