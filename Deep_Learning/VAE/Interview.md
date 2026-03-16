# VAE Interview Questions

## Q1: What is the difference between VAE and autoencoder?

**Answer:** VAE learns a probability distribution in latent space, allowing sampling for generation. Regular autoencoder learns deterministic mapping.

## Q2: What is KL divergence loss in VAE?

**Answer:** KL divergence measures how much the learned latent distribution differs from a prior (usually standard normal), encouraging organized latent space.

## Q3: How does VAE generate new images?

**Answer:** Sample from the learned latent distribution, then pass through decoder to generate new images.

## Q4: What is posterior collapse?

**Answer:** When the decoder ignores latent code, producing similar outputs regardless of input.
