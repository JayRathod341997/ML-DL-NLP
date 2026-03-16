# Generative Adversarial Networks (GAN)

## Overview

GANs consist of two neural networks (generator and discriminator) that compete in a game. The generator creates fake samples, the discriminator distinguishes real from fake.

### Key Characteristics

- **Adversarial Training**: Generator vs Discriminator
- **Zero-sum Game**: One wins, other loses
- **No Explicit Likelihood**: Implicit generation
- **High-quality Outputs**: Realistic images

## Architecture

```
Generator: Noise → Fake Images
Discriminator: Real/Fake → Probability
```

## Training

1. Train Discriminator: distinguish real from fake
2. Train Generator: fool the discriminator
3. Alternate until Nash equilibrium

## Types

| Type | Description |
|------|-------------|
| DCGAN | Deep Convolutional GAN |
| WGAN | Wasserstein GAN |
| Conditional | Class-conditional generation |
| StyleGAN | Style-based generation |

## Applications

- Image synthesis
- Data augmentation
- Art generation
- Image-to-image translation

## Quick Start

```bash
cd Deep_Learning/GAN
uv run python codes/train.py
```

## Limitations

- Mode collapse
- Training instability
- Requires careful tuning
- No explicit probability
