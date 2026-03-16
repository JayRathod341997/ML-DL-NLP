from __future__ import annotations

from pathlib import Path

from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from torch.utils.data import DataLoader

from .losses import get_mnrl_loss, get_cosine_similarity_loss


def train_with_mnrl(
    base_model: str,
    train_examples,
    val_evaluator=None,
    output_dir: str | Path = "models/mnrl_model",
    num_epochs: int = 1,
    batch_size: int = 64,
    warmup_ratio: float = 0.1,
    learning_rate: float = 2e-5,
) -> SentenceTransformer:
    """Fine-tune a SentenceTransformer with MultipleNegativesRankingLoss.

    Args:
        base_model: HuggingFace model ID or local path.
        train_examples: List of InputExample(texts=[anchor, positive]).
        val_evaluator: Optional STS evaluator for validation.
        output_dir: Where to save the trained model.
        num_epochs: Training epochs (1 epoch on SNLI+MultiNLI ≈ good).

    Returns:
        Trained SentenceTransformer model.
    """
    model = SentenceTransformer(base_model)
    loss = get_mnrl_loss(model)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    model.fit(
        train_objectives=[(train_dataloader, loss)],
        evaluator=val_evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        output_path=str(output_dir),
        show_progress_bar=True,
    )
    return model


def train_with_cosine_similarity(
    base_model: str,
    train_examples,
    val_evaluator=None,
    output_dir: str | Path = "models/sts_model",
    num_epochs: int = 4,
    batch_size: int = 32,
) -> SentenceTransformer:
    """Fine-tune a SentenceTransformer with CosineSimilarityLoss on STS-B.

    Args:
        base_model: HuggingFace model ID or local path.
        train_examples: List of InputExample(texts=[s1, s2], label=float).

    Returns:
        Trained SentenceTransformer model.
    """
    model = SentenceTransformer(base_model)
    loss = get_cosine_similarity_loss(model)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    model.fit(
        train_objectives=[(train_dataloader, loss)],
        evaluator=val_evaluator,
        epochs=num_epochs,
        warmup_steps=100,
        output_path=str(output_dir),
        show_progress_bar=True,
    )
    return model
