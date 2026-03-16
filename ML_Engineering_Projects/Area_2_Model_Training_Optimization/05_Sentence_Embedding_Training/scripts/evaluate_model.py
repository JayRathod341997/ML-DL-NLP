"""Evaluate a trained sentence embedding model on STS-B.

Usage:
    uv run python scripts/evaluate_model.py --model-dir models/mnrl_model
    uv run python scripts/evaluate_model.py --model sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from src.evaluator import STSEvaluator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=None, help="Local model directory")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="HuggingFace model ID (used if --model-dir not set)")
    args = parser.parse_args()

    model_path = args.model_dir or args.model
    print(f"Loading model: {model_path}")
    model = SentenceTransformer(model_path)

    for split in ["validation", "test"]:
        evaluator = STSEvaluator(split=split)
        results = evaluator.evaluate(model)
        print(f"STS-B [{split}] — Spearman r: {results['spearman_r']:.4f} | Pearson r: {results['pearson_r']:.4f}")


if __name__ == "__main__":
    main()
