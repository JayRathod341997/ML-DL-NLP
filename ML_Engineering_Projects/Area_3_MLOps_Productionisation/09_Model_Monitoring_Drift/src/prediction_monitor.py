from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PredictionSnapshot:
    timestamp: str
    label_distribution: dict[str, float]
    entropy: float
    num_predictions: int


class PredictionMonitor:
    """Tracks prediction label distribution over time using a rolling window."""

    def __init__(self, window_size: int = 10) -> None:
        self.window_size = window_size
        self._history: deque[PredictionSnapshot] = deque(maxlen=window_size)

    def update(self, labels: list[str], timestamp: str | None = None) -> PredictionSnapshot:
        """Add a new batch of predictions to the monitor.

        Args:
            labels: List of predicted label strings.
            timestamp: ISO timestamp string (defaults to now).

        Returns:
            PredictionSnapshot for this batch.
        """
        from datetime import datetime

        if not timestamp:
            timestamp = datetime.now().isoformat()

        counts = defaultdict(int)
        for label in labels:
            counts[label] += 1
        total = len(labels)
        dist = {k: round(v / total, 4) for k, v in counts.items()}
        probs = np.array(list(dist.values()))
        entropy = float(-np.sum(probs * np.log2(probs + 1e-8)))

        snapshot = PredictionSnapshot(
            timestamp=timestamp,
            label_distribution=dist,
            entropy=round(entropy, 4),
            num_predictions=total,
        )
        self._history.append(snapshot)
        return snapshot

    def get_history_df(self) -> pd.DataFrame:
        """Return history as a DataFrame for visualisation."""
        if not self._history:
            return pd.DataFrame()
        rows = []
        for snap in self._history:
            row = {"timestamp": snap.timestamp, "entropy": snap.entropy}
            row.update({f"label_{k}": v for k, v in snap.label_distribution.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def detect_entropy_drop(self, threshold: float = 0.5) -> bool:
        """Return True if recent entropy is significantly lower than baseline.

        A drop in entropy means the model is becoming overconfident — possible sign of drift.
        """
        if len(self._history) < 3:
            return False
        baseline = np.mean([s.entropy for s in list(self._history)[:-1]])
        current = self._history[-1].entropy
        return bool(baseline > 0 and (baseline - current) / baseline > threshold)
