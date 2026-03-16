"""Build reference profile from a dataset.

Usage:
    uv run python scripts/profile_reference.py --dataset imdb --split test
    uv run python scripts/profile_reference.py --input data/reference.parquet
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import pandas as pd
from transformers import pipeline as hf_pipeline

from src.reference_profiler import ReferenceProfiler


@click.command()
@click.option("--dataset", default=None)
@click.option("--split", default="test")
@click.option("--input", "input_file", type=Path, default=None)
@click.option("--output", type=Path, default=Path("data/reference_profile.json"))
@click.option("--model", default="distilbert-base-uncased-finetuned-sst-2-english")
@click.option("--max-samples", type=int, default=5000)
def main(dataset, split, input_file, output, model, max_samples):
    """Build reference profile from a dataset."""
    if not dataset and not input_file:
        raise click.UsageError("Provide --dataset or --input")

    if input_file:
        df = pd.read_parquet(input_file) if str(input_file).endswith(".parquet") else pd.read_csv(input_file)
    else:
        from datasets import load_dataset
        ds = load_dataset(dataset, split=split)
        df = pd.DataFrame({"text": ds["text"][:max_samples]})

    print(f"Loaded {len(df)} samples")

    # Add derived features
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()

    # Run predictions
    print("Running model predictions for reference distribution...")
    pipe = hf_pipeline("text-classification", model=model, truncation=True)
    results = pipe(df["text"].tolist()[:max_samples], batch_size=64)
    df["predicted_label"] = [r["label"] for r in results]
    df["predicted_score"] = [r["score"] for r in results]

    profiler = ReferenceProfiler()
    profile = profiler.profile(df[["text_length", "word_count", "predicted_score"]])
    profile["label_distribution"] = profiler.profile_predictions(df["predicted_label"].tolist())

    profiler.save(profile, output)
    print(f"Profile saved to {output}")
    print(f"Label distribution: {profile['label_distribution']['label_distribution']}")


if __name__ == "__main__":
    main()
