# GANs (Generative Adversarial Networks) - Explain Like I'm 5

## What is a GAN?

Imagine two characters:
- **Counterfeiter (Generator):** tries to make fake money
- **Police (Discriminator):** tries to catch the fake money

They play a game:
1) Generator makes fakes
2) Discriminator learns to spot fakes
3) Generator learns to make better fakes

Over time, the generator gets very good at making realistic samples.

## Core Idea

- Generator `G(z)` turns random noise `z` into fake data
- Discriminator `D(x)` outputs how real/fake a sample looks

They optimize opposing objectives (a minimax game).

## Where It’s Used

- Data augmentation (when real data is limited)
- Synthetic image generation
- Privacy-preserving synthetic datasets (with careful evaluation)

## Benefits

- Can generate sharp, realistic samples
- Learns data distribution without explicit likelihood in classic GANs

## Limitations

- Training can be unstable
- Mode collapse (generator produces limited variety)
- Harder to evaluate quality objectively

## Example in This Folder

We use a tiny 2D dataset (Gaussian mixture) from CSV:
- Dataset: `dataset/gaussian_mixture_2d.csv`
- Code trains a toy GAN to generate 2D points.

## Enterprise-Level Example

In an enterprise CV pipeline:
- Generate synthetic defect images for rare defect types
- Use synthetic + real data to improve defect detection coverage
- Maintain strict validation to ensure synthetic data doesn’t introduce bias

