"""Train sentence embeddings with CosineSimilarityLoss on STS-B.

Usage:
    uv run python scripts/train_sts.py
    uv run python scripts/train_sts.py --epochs 4 --batch-size 32
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_stsb_pairs
from src.evaluator import STSEvaluator
from src.trainer import train_with_cosine_similarity


def main() -> None:
    parser = argparse.ArgumentParser(description="Train with CosineSimilarityLoss on STS-B")
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("models/sts_model"))
    args = parser.parse_args()

    print("Loading STS-B training data...")
    train_examples = load_stsb_pairs(split="train")
    val_evaluator = STSEvaluator(split="validation").get_st_evaluator()

    print(f"Training: {len(train_examples)} pairs, {args.epochs} epochs...")
    model = train_with_cosine_similarity(
        base_model=args.base_model,
        train_examples=train_examples,
        val_evaluator=val_evaluator,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print(f"\nModel saved to {args.output_dir}")
    print("\nFinal evaluation on STS-B test:")
    results = STSEvaluator(split="test").evaluate(model)
    print(f"  Spearman r: {results['spearman_r']:.4f}")
    print(f"  Pearson r:  {results['pearson_r']:.4f}")


if __name__ == "__main__":
    main()
