from __future__ import annotations

from scipy.stats import pearsonr, spearmanr
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from .data_loader import load_stsb_pairs


class STSEvaluator:
    """Evaluates a SentenceTransformer on STS-B with Spearman and Pearson correlation."""

    def __init__(self, split: str = "test") -> None:
        self.split = split
        self._pairs = load_stsb_pairs(split=split)

    def evaluate(self, model: SentenceTransformer) -> dict:
        """Run evaluation and return Spearman r, Pearson r, and cosine similarities.

        Returns:
            dict with keys: spearman_r, pearson_r, num_pairs
        """
        sentences1 = [p.texts[0] for p in self._pairs]
        sentences2 = [p.texts[1] for p in self._pairs]
        gold_scores = np.array([p.label for p in self._pairs])

        emb1 = model.encode(sentences1, normalize_embeddings=True)
        emb2 = model.encode(sentences2, normalize_embeddings=True)
        cosine_scores = (emb1 * emb2).sum(axis=1)

        spearman = float(spearmanr(gold_scores, cosine_scores).correlation)
        pearson = float(pearsonr(gold_scores, cosine_scores)[0])

        return {
            "spearman_r": round(spearman, 4),
            "pearson_r": round(pearson, 4),
            "num_pairs": len(self._pairs),
            "split": self.split,
        }

    def get_st_evaluator(self) -> EmbeddingSimilarityEvaluator:
        """Return a sentence-transformers EmbeddingSimilarityEvaluator for use in training."""
        return EmbeddingSimilarityEvaluator(
            sentences1=[p.texts[0] for p in self._pairs],
            sentences2=[p.texts[1] for p in self._pairs],
            scores=[p.label for p in self._pairs],
            name=f"stsb-{self.split}",
        )
