from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class ReferenceProfiler:
    """Computes statistical profile of reference data for drift comparison.

    Computed statistics per column:
    - Numeric: mean, std, min, max, quantiles [0.05, 0.25, 0.5, 0.75, 0.95]
    - Categorical: value counts, num_unique
    - Label distribution: class probabilities
    """

    def profile(self, df: pd.DataFrame) -> dict:
        """Compute reference profile from a DataFrame.

        Args:
            df: Reference dataset (training or known-good production data).

        Returns:
            Profile dict with per-column statistics.
        """
        profile = {"num_samples": len(df), "columns": {}}

        for col in df.columns:
            series = df[col].dropna()
            if series.dtype in (float, int) or pd.api.types.is_numeric_dtype(series):
                profile["columns"][col] = {
                    "type": "numeric",
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "quantiles": {
                        str(q): float(series.quantile(q))
                        for q in [0.05, 0.25, 0.5, 0.75, 0.95]
                    },
                }
            else:
                vc = series.value_counts(normalize=True).to_dict()
                profile["columns"][col] = {
                    "type": "categorical",
                    "value_frequencies": {str(k): float(v) for k, v in vc.items()},
                    "num_unique": int(series.nunique()),
                }
        return profile

    def profile_predictions(self, labels: list[str]) -> dict:
        """Profile a list of predicted labels."""
        from collections import Counter

        counts = Counter(labels)
        total = len(labels)
        return {
            "num_predictions": total,
            "label_distribution": {k: round(v / total, 4) for k, v in counts.items()},
            "label_counts": dict(counts),
        }

    def save(self, profile: dict, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(profile, f, indent=2)
        print(f"Reference profile saved to {path}")

    def load(self, path: str | Path) -> dict:
        with open(path) as f:
            return json.load(f)
