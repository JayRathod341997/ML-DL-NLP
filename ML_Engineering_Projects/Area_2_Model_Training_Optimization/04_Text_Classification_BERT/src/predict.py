from __future__ import annotations

import torch
from transformers import AutoTokenizer

from .config import TrainConfig
from .model import BERTClassifier


class Predictor:
    """Load a trained checkpoint and run inference on new texts."""

    def __init__(self, checkpoint_dir: str, config: TrainConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = BERTClassifier(config)
        state = torch.load(f"{checkpoint_dir}/model.pt", map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> tuple[int, float]:
        """Predict class label and confidence for a single text."""
        enc = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(
                enc["input_ids"].to(self.device),
                enc["attention_mask"].to(self.device),
            )
        probs = torch.softmax(logits, dim=-1)
        pred = int(probs.argmax().item())
        confidence = float(probs.max().item())
        return pred, confidence

    def predict_batch(self, texts: list[str]) -> list[tuple[int, float]]:
        return [self.predict(t) for t in texts]
