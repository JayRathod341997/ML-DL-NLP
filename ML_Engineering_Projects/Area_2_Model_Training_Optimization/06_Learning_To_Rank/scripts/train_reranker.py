"""Fine-tune a CrossEncoder on MS MARCO pairs.

Usage:
    uv run python scripts/train_reranker.py --epochs 1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm


def load_msmarco_pairs(max_samples: int = 50000) -> list[InputExample]:
    """Load MS MARCO as (query, passage, label) pairs for CrossEncoder training."""
    ds = load_dataset("ms_marco", "v1.1", split="train", streaming=True)
    examples = []
    for row in tqdm(ds, total=max_samples, desc="Loading"):
        query = row["query"]
        for text, is_selected in zip(
            row["passages"]["passage_text"], row["passages"]["is_selected"]
        ):
            examples.append(InputExample(texts=[query, text], label=float(is_selected)))
        if len(examples) >= max_samples:
            break
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--output", type=Path, default=Path("models/reranker"))
    args = parser.parse_args()

    print(f"Loading {args.max_samples} MS MARCO pairs...")
    examples = load_msmarco_pairs(args.max_samples)
    print(f"Loaded {len(examples)} pairs")

    model = CrossEncoder(args.base_model, num_labels=1, max_length=512)
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    model.fit(
        train_dataloader=loader,
        epochs=args.epochs,
        output_path=str(args.output),
        show_progress_bar=True,
    )
    print(f"Reranker saved to {args.output}")


if __name__ == "__main__":
    main()
