"""Run drift detection on a new batch of data.

Usage:
    uv run python scripts/monitor_batch.py --dataset yelp_review_full --split test
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import numpy as np
import pandas as pd
from transformers import pipeline as hf_pipeline

from src.reference_profiler import ReferenceProfiler
from src.drift_detector import DriftDetector
from src.prediction_monitor import PredictionMonitor
from src.alerter import Alerter


@click.command()
@click.option("--dataset", default=None)
@click.option("--split", default="test")
@click.option("--reference", type=Path, default=Path("data/reference_profile.json"))
@click.option("--output-dir", type=Path, default=Path("data/drift_results"))
@click.option("--model", default="distilbert-base-uncased-finetuned-sst-2-english")
@click.option("--max-samples", type=int, default=2000)
def main(dataset, split, reference, output_dir, model, max_samples):
    """Run drift detection on a new batch."""
    if not reference.exists():
        print(f"Reference profile not found: {reference}")
        print("Run profile_reference.py first.")
        return

    if dataset:
        from datasets import load_dataset
        ds = load_dataset(dataset, split=split)
        current_texts = ds["text"][:max_samples]
    else:
        raise click.UsageError("Provide --dataset")

    current_df = pd.DataFrame({"text": current_texts})
    current_df["text_length"] = current_df["text"].str.len()
    current_df["word_count"] = current_df["text"].str.split().str.len()

    print("Running model predictions on current batch...")
    pipe = hf_pipeline("text-classification", model=model, truncation=True)
    results = pipe(current_df["text"].tolist(), batch_size=64)
    current_df["predicted_label"] = [r["label"] for r in results]
    current_df["predicted_score"] = [r["score"] for r in results]

    # Load reference profile
    profiler = ReferenceProfiler()
    ref_profile = profiler.load(reference)

    # Build reference DataFrame from profile
    detector = DriftDetector()

    # Run drift tests on numeric features
    drift_results = []
    for col in ["text_length", "word_count", "predicted_score"]:
        if col in ref_profile.get("columns", {}):
            ref_stats = ref_profile["columns"][col]
            ref_arr = np.random.normal(ref_stats["mean"], ref_stats["std"] + 1e-6, 5000)
            cur_arr = current_df[col].dropna().values

            ks = detector.ks_test(ref_arr, cur_arr)
            ks.feature = col
            psi = detector.psi(ref_arr, cur_arr)
            psi.feature = col
            drift_results.extend([ks, psi])
            print(f"{col}: KS p={ks.p_value} (drift={ks.is_drift}), PSI={psi.score} (drift={psi.is_drift})")

    # Prediction distribution drift
    ref_label_dist = ref_profile.get("label_distribution", {}).get("label_counts", {})
    cur_label_counts = current_df["predicted_label"].value_counts().to_dict()
    chi2_result = detector.chi_squared(ref_label_dist, cur_label_counts)
    drift_results.append(chi2_result)
    print(f"Prediction distribution chi2: p={chi2_result.p_value} (drift={chi2_result.is_drift})")

    # Alerter
    alerter = Alerter()
    alerts = alerter.check(drift_results)
    print(f"\n{len(alerts)} alerts generated")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df = pd.DataFrame([
        {"feature": r.feature, "test": r.test, "score": r.score, "p_value": r.p_value, "is_drift": r.is_drift}
        for r in drift_results
    ])
    results_df.to_csv(output_dir / f"drift_{ts}.csv", index=False)
    print(f"Results saved to {output_dir}/drift_{ts}.csv")


if __name__ == "__main__":
    main()
