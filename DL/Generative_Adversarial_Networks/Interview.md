# GANs - Interview Questions (with Answers)

## Basic

### Q1: What are the two networks in a GAN?
**Answer:** Generator (creates fake samples) and Discriminator (classifies real vs fake).

### Q2: What is the input to the generator?
**Answer:** Random noise vector (latent variable) sampled from a simple distribution.

## Intermediate

### Q3: What is mode collapse?
**Answer:** The generator produces limited varieties of outputs, ignoring parts of the true data distribution.

### Q4: Why can GAN training be unstable?
**Answer:** It’s a minimax game; if one network becomes too strong, gradients can vanish or oscillate.

## Advanced

### Q5: Name a stabilization technique.
**Answer:** WGAN (Wasserstein GAN), gradient penalty, spectral normalization, label smoothing, balanced training schedules.

