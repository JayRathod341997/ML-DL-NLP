# Transformer Networks

## Overview

Transformers use self-attention mechanisms to process sequential data without recurrence, enabling parallelization and capturing long-range dependencies.

### Key Characteristics

- **Self-Attention**: Weighs importance of all positions
- **Parallel Processing**: No sequential dependency
- **Positional Encoding**: Adds position information
- **Multi-Head Attention**: Multiple attention patterns

## Architecture

```
Input → Embedding → Positional Encoding → Encoder/Decoder → Output
```

## Components

1. **Multi-Head Attention**: Multiple attention heads
2. **Feed-Forward Networks**: Position-wise FFN
3. **Add & Norm**: Residual connections + LayerNorm
4. **Positional Encoding**: Position information

## Quick Start

```bash
cd Deep_Learning/Transformers
uv run python codes/train.py
```

## Applications

- Machine Translation
- Text Generation
- Question Answering
- BERT, GPT models

## Limitations

- Quadratic complexity in attention
- Memory intensive
- Limited context window
