"""Run batch inference on a file or HuggingFace dataset.

Usage:
    uv run python scripts/run_batch.py --dataset imdb --split test --output results/predictions.parquet
    uv run python scripts/run_batch.py --input data/texts.jsonl --output results/out.parquet
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import torch

from src.config import BatchConfig
from src.pipeline import BatchPipeline


@click.command()
@click.option("--input", "input_file", type=Path, default=None, help="Input JSONL or CSV file")
@click.option("--dataset", default=None, help="HuggingFace dataset name (e.g., imdb)")
@click.option("--split", default="test", help="Dataset split")
@click.option("--output", type=Path, required=True, help="Output Parquet file path")
@click.option("--model", default="distilbert-base-uncased-finetuned-sst-2-english")
@click.option("--batch-size", default=64, type=int)
@click.option("--text-column", default="text")
def main(input_file, dataset, split, output, model, batch_size, text_column):
    """Run batch ML inference on a dataset or file."""
    if not input_file and not dataset:
        raise click.UsageError("Provide either --input or --dataset")

    config = BatchConfig(
        model_name=model,
        batch_size=batch_size,
        text_column=text_column,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"Config: model={model}, batch_size={batch_size}, device={config.device}")

    pipeline = BatchPipeline(config)

    if input_file:
        stats = pipeline.run_on_file(input_file, output)
    else:
        stats = pipeline.run_on_dataset(dataset, split, output)

    print(f"\nResults saved to {output}")
    print(f"Throughput: {stats['throughput_per_second']:.1f} samples/sec")


if __name__ == "__main__":
    main()
