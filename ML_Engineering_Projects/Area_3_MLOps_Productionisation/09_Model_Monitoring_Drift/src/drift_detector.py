from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency


@dataclass
class DriftResult:
    feature: str
    test: str          # "ks", "psi", "chi2", "mmd"
    score: float       # test statistic or PSI/MMD value
    p_value: float | None  # None for PSI/MMD
    is_drift: bool     # True if drift detected


class DriftDetector:
    """Statistical tests for detecting distribution drift.

    Tests:
    - KS (Kolmogorov-Smirnov): two-sample test for continuous features
    - PSI (Population Stability Index): binned distribution divergence
    - Chi-squared: categorical feature distribution comparison
    - MMD (Maximum Mean Discrepancy): embedding space drift
    """

    def __init__(
        self,
        ks_threshold: float = 0.05,  # p-value below = drift
        psi_threshold: float = 0.1,   # PSI above = warning; > 0.25 = alert
    ) -> None:
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold

    def ks_test(self, reference: np.ndarray, current: np.ndarray) -> DriftResult:
        """KS test for continuous feature drift."""
        stat, p_value = ks_2samp(reference, current)
        return DriftResult(
            feature="",
            test="ks",
            score=round(float(stat), 4),
            p_value=round(float(p_value), 4),
            is_drift=p_value < self.ks_threshold,
        )

    def psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> DriftResult:
        """Population Stability Index (PSI) for binned distributions.

        PSI < 0.1: no drift
        PSI 0.1-0.25: moderate drift
        PSI > 0.25: significant drift
        """
        eps = 1e-8
        bins_edges = np.percentile(reference, np.linspace(0, 100, bins + 1))
        bins_edges = np.unique(bins_edges)

        ref_counts = np.histogram(reference, bins=bins_edges)[0] + eps
        cur_counts = np.histogram(current, bins=bins_edges)[0] + eps

        ref_pct = ref_counts / ref_counts.sum()
        cur_pct = cur_counts / cur_counts.sum()

        psi_value = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return DriftResult(
            feature="",
            test="psi",
            score=round(psi_value, 4),
            p_value=None,
            is_drift=psi_value > self.psi_threshold,
        )

    def chi_squared(
        self,
        reference_counts: dict[str, int],
        current_counts: dict[str, int],
    ) -> DriftResult:
        """Chi-squared test for categorical distribution drift."""
        all_keys = sorted(set(reference_counts) | set(current_counts))
        ref = np.array([reference_counts.get(k, 0) for k in all_keys]) + 1
        cur = np.array([current_counts.get(k, 0) for k in all_keys]) + 1

        # Scale reference to same total as current
        ref_scaled = ref / ref.sum() * cur.sum()
        _, p_value, _, _ = chi2_contingency(np.array([ref_scaled, cur]))
        stat = float(np.sum((cur - ref_scaled) ** 2 / ref_scaled))
        return DriftResult(
            feature="label_distribution",
            test="chi2",
            score=round(stat, 4),
            p_value=round(float(p_value), 4),
            is_drift=p_value < self.ks_threshold,
        )

    def mmd(
        self,
        reference_embeddings: np.ndarray,
        current_embeddings: np.ndarray,
        sample_size: int = 1000,
    ) -> DriftResult:
        """Maximum Mean Discrepancy using RBF kernel.

        Computes a distance between two sets of embeddings.
        Larger values indicate more drift in embedding space.
        """
        # Subsample for efficiency
        n_ref = min(sample_size, len(reference_embeddings))
        n_cur = min(sample_size, len(current_embeddings))
        ref = reference_embeddings[np.random.choice(len(reference_embeddings), n_ref, replace=False)]
        cur = current_embeddings[np.random.choice(len(current_embeddings), n_cur, replace=False)]

        # RBF kernel bandwidth = median heuristic
        all_dists = np.linalg.norm(ref[:, None] - ref[None, :], axis=-1).flatten()
        bandwidth = float(np.median(all_dists[all_dists > 0])) or 1.0

        def rbf(X: np.ndarray, Y: np.ndarray) -> float:
            dists = np.sum((X[:, None] - Y[None, :]) ** 2, axis=-1)
            return float(np.exp(-dists / (2 * bandwidth ** 2)).mean())

        mmd_val = rbf(ref, ref) - 2 * rbf(ref, cur) + rbf(cur, cur)
        return DriftResult(
            feature="embeddings",
            test="mmd",
            score=round(float(mmd_val), 6),
            p_value=None,
            is_drift=mmd_val > 0.01,  # heuristic threshold
        )

    def detect_all(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> list[DriftResult]:
        """Run KS + PSI tests on all numeric columns."""
        results = []
        for col in reference_df.select_dtypes(include=[np.number]).columns:
            if col in current_df.columns:
                ref = reference_df[col].dropna().values
                cur = current_df[col].dropna().values
                ks_result = self.ks_test(ref, cur)
                ks_result.feature = col
                results.append(ks_result)
                psi_result = self.psi(ref, cur)
                psi_result.feature = col
                results.append(psi_result)
        return results
