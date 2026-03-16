"""Wrappers and documentation for sentence-transformers loss functions.

This module documents the two main training losses used in this project.
The actual loss classes come from the sentence-transformers library.
"""

from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss, CosineSimilarityLoss


def get_mnrl_loss(model: SentenceTransformer) -> MultipleNegativesRankingLoss:
    """MultipleNegativesRankingLoss (MNRL).

    Best for: large-scale training with (anchor, positive) pairs.
    Data format: InputExample(texts=[anchor, positive]) — NO label needed.
    Batch strategy: All other positives in batch act as negatives.
    Efficiency: O(N^2) negatives per batch of N pairs.
    Paper: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (2019)
    """
    return MultipleNegativesRankingLoss(model)


def get_cosine_similarity_loss(model: SentenceTransformer) -> CosineSimilarityLoss:
    """CosineSimilarityLoss.

    Best for: fine-grained similarity with labelled pairs.
    Data format: InputExample(texts=[s1, s2], label=0.75) — float label [0, 1].
    Training objective: MSE between predicted cosine similarity and gold label.
    Use with: STS-B dataset (labels normalised to [0, 1]).
    """
    return CosineSimilarityLoss(model)
