from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from .config import TrainConfig


class BERTClassifier(nn.Module):
    """DistilBERT (or any HuggingFace encoder) + linear classification head."""

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(hidden_size, config.num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Shape (batch, seq_len)
            attention_mask: Shape (batch, seq_len)

        Returns:
            logits: Shape (batch, num_labels)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_output))

    def freeze_encoder_layers(self, num_layers: int) -> None:
        """Freeze the first num_layers transformer layers (useful for small datasets)."""
        for name, param in self.encoder.named_parameters():
            # Freeze embeddings and first N transformer blocks
            if "transformer.layer" in name:
                layer_num = int(name.split("transformer.layer.")[1].split(".")[0])
                if layer_num < num_layers:
                    param.requires_grad = False
            elif "embeddings" in name:
                param.requires_grad = False
