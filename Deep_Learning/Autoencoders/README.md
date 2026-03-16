# Autoencoders

## Overview

Autoencoders are neural networks that learn to compress data into a lower-dimensional latent space and then reconstruct it. They consist of an encoder (compression) and decoder (reconstruction).

### Key Characteristics

- **Unsupervised Learning**: No labels needed
- **Dimensionality Reduction**: Learn compact representations
- **Feature Learning**: Automatically discover important features
- **Reconstruction**: Output aims to match input

## Architecture

```
Input → Encoder → Latent Space → Decoder → Output
```

### Components

1. **Encoder**: Compresses input to latent representation
2. **Latent Space**: Compressed bottleneck representation
3. **Decoder**: Reconstructs input from latent space
4. **Loss**: Reconstruction loss (MSE)

## Quick Start

```bash
cd Deep_Learning/Autoencoders
uv venv
uv pip install torch numpy pandas matplotlib
uv run python codes/train.py
```

## Types

| Type | Description |
|------|-------------|
| Vanilla | Basic encoder-decoder |
| Denoising | Learns to remove noise |
| Variational | Learns probability distribution |
| Sparse | Promotes sparse representations |
| Contractive | Penalizes Jacobian |

## Applications

- Dimensionality reduction
- Anomaly detection
- Image denoising
- Recommendation systems
- Data compression

## Limitations

- May lose subtle details
- Limited to reconstruction tasks
- Hard to evaluate quality
- Can overfit to training data
