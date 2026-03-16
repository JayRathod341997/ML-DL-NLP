"""
BERT-like Model Definition
"""

import torch
import torch.nn as nn
from transformers import BertModel


class BERTClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        outputs = self.bert(x)
        pooled = outputs.pooler_output
        return self.classifier(pooled)
