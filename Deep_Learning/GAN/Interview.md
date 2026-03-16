# GAN Interview Questions

## Q1: What is the generator in GAN?

**Answer:** Neural network that creates fake samples from random noise. It learns to produce realistic samples that fool the discriminator.

## Q2: What is the discriminator in GAN?

**Answer:** Neural network that classifies samples as real or fake. It learns to distinguish real data from generator outputs.

## Q3: What is mode collapse?

**Answer:** When generator produces limited variety of outputs, fooling discriminator with same sample repeatedly.

## Q4: What is the training objective in GAN?

**Answer:** Minimax game - generator tries to minimize discriminator's accuracy, discriminator tries to maximize its accuracy.

## Q5: Difference between GAN and VAE?

**Answer:** GAN produces sharper images but harder to train; VAE produces blurry images but stable training. GAN has implicit density, VAE has explicit.
