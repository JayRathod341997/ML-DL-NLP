"""Train sentence embeddings with MultipleNegativesRankingLoss on NLI data.

Usage:
    uv run python scripts/train_mnrl.py
    uv run python scripts/train_mnrl.py --epochs 1 --batch-size 64
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_nli_pairs, load_stsb_pairs
from src.evaluator import STSEvaluator
from src.trainer import train_with_mnrl


def main() -> None:
    parser = argparse.ArgumentParser(description="Train with MNRL on NLI")
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit NLI samples")
    parser.add_argument("--output-dir", type=Path, default=Path("models/mnrl_model"))
    args = parser.parse_args()

    print(f"Loading NLI data (max_samples={args.max_samples})...")
    train_examples = load_nli_pairs(max_samples=args.max_samples)

    print("Setting up STS-B evaluator...")
    evaluator = STSEvaluator(split="validation").get_st_evaluator()

    print(f"Training with MNRL: {len(train_examples)} pairs, {args.epochs} epoch(s)...")
    model = train_with_mnrl(
        base_model=args.base_model,
        train_examples=train_examples,
        val_evaluator=evaluator,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(f"\nModel saved to {args.output_dir}")

    print("\nFinal evaluation on STS-B test set:")
    sts_eval = STSEvaluator(split="test")
    results = sts_eval.evaluate(model)
    print(f"  Spearman r: {results['spearman_r']:.4f}")
    print(f"  Pearson r:  {results['pearson_r']:.4f}")


if __name__ == "__main__":
    main()
