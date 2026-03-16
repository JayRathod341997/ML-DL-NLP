from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .config import BatchConfig


class TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int) -> None:
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {k: v[idx] for k, v in self.encodings.items()}


class BatchModelRunner:
    """Runs batched inference using a HuggingFace model and PyTorch DataLoader.

    Uses torch.no_grad() and torch.autocast for memory-efficient inference.
    """

    def __init__(self, config: BatchConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.model_name)
        self.model.to(self.device)
        self.model.eval()
        self._labels = list(self.model.config.id2label.values())

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Run inference on a list of texts.

        Returns:
            List of dicts with 'label' and 'score' keys.
        """
        if not texts:
            return []

        dataset = TextDataset(texts, self.tokenizer, self.config.max_length)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

        all_labels, all_scores = [], []
        with torch.no_grad():
            with torch.autocast(self.config.device):
                for batch in loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    pred_ids = probs.argmax(dim=-1)
                    for pred_id, prob in zip(pred_ids, probs):
                        all_labels.append(self._labels[pred_id.item()])
                        all_scores.append(round(float(prob.max().item()), 4))

        return [{"label": l, "score": s} for l, s in zip(all_labels, all_scores)]
