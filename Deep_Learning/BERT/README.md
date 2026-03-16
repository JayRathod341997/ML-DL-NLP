# BERT (Bidirectional Encoder Representations from Transformers)

## Overview

BERT is a transformer-based model that learns bidirectional context by pre-training on masked language modeling and next sentence prediction.

### Key Characteristics

- **Bidirectional**: Reads context from both directions
- **Pre-training + Fine-tuning**: Two-stage approach
- **Masked LM**: Predicts masked tokens
- **Next Sentence Prediction**: Understands sentence relationships

## Architecture

```
Input → Transformer Encoder → [CLS] + Segment Embeddings → Output
```

## Pre-training Tasks

1. **Masked Language Modeling**: Predict masked tokens
2. **Next Sentence Prediction**: Predict if B follows A

## Fine-tuning

- Classification
- Named Entity Recognition
- Question Answering

## Quick Start

```bash
cd Deep_Learning/BERT
uv run python codes/train.py
```

## Limitations

- Memory intensive
- Slow inference
- Fixed context length
