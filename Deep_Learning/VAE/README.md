# Variational Autoencoders (VAE)

## Overview

VAEs are generative models that learn a probability distribution of the latent space, allowing sampling to generate new data.

### Key Characteristics

- **Probabilistic**: Learn distribution parameters (μ, σ)
- **Generative**: Can sample new data
- **Latent Space**: Continuous, structured representation
- **Loss**: Reconstruction + KL Divergence

## Architecture

```
Input → Encoder → μ, σ → Sampling → Decoder → Output
```

## Loss Function

```
Loss = Reconstruction Loss + KL Divergence
```

## Applications

- Image generation
- Data augmentation
- Anomaly detection
- Representation learning

## Quick Start

```bash
cd Deep_Learning/VAE
uv run python codes/train.py
```

## Limitations

- Blurry outputs
- Posterior collapse
- Limited capacity
