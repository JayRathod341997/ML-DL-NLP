# Autoencoders - Interview Questions

## Foundational Questions (Definitions)

### Q1: What is an autoencoder and how does it work?

**Short Answer:** An autoencoder is an unsupervised neural network that learns to compress data into a lower-dimensional latent representation and then reconstruct it back. It consists of an encoder that maps input to the bottleneck, and a decoder that reconstructs the input from the bottleneck.

**Deep Dive:** Autoencoders are trained to minimize reconstruction error—learning to output something as close to the input as possible. The architecture has three main components:

1. **Encoder**: Maps input x to latent representation z: z = f(x)
2. **Bottleneck (Latent Space)**: The compressed representation where dimensions are reduced
3. **Decoder**: Reconstructs input from latent space: x' = g(z)

The training objective is: L(x, x') = ||x - x'||² (MSE loss)

Key properties:
- The bottleneck dimension is typically much smaller than input
- Learns useful features without labeled data
- Can be used for dimensionality reduction, similar to PCA but non-linear

---

### Q2: What is the difference between undercomplete and overcomplete autoencoders?

**Short Answer:** Undercomplete autoencoders have a bottleneck smaller than the input (dimensionality reduction). Overcomplete autoencoders have a bottleneck larger than input, requiring regularization to prevent trivial learning.

**Deep Dive:**

| Type | Bottleneck Size | Purpose | Challenge |
|------|-----------------|---------|------------|
| Undercomplete | < input | Compression, feature learning | May not capture enough info |
| Overcomplete | > input | Sparse coding, denoising | Can learn identity function |

**Undercomplete**: Forces learning of the most important features; effective for compression.

**Overcomplete**: Can potentially learn identity function (trivial solution), so needs:
- Denoising autoencoders (corrupt input, reconstruct clean)
- Sparse autoencoders (add sparsity penalty)
- Contractive autoencoders (penalize Jacobian norm)

---

### Q3: Explain the latent space representation in autoencoders.

**Short Answer:** The latent space is the bottleneck layer where data is compressed. A well-trained autoencoder's latent space captures the essential structure of data, enabling interpolation, generation, and anomaly detection.

**Deep Dive:** Properties of a good latent space:

1. **Continuous**: Similar inputs map to similar latent vectors
2. **Complete**: Can represent all variation in data
3. **Disentangled**: Independent factors are separated

**Types of latent spaces**:
- **Deterministic**: Single z for each x (standard AE)
- **Variational**: Distribution parameters (VAE), enables sampling

**Applications of latent space**:
- Interpolation between samples
- Anomaly detection (high reconstruction error = anomaly)
- Data generation (decode random z)
- Transfer learning (use encoder as feature extractor)

---

## Applied Questions (How to Tune/Train)

### Q4: How do you choose the bottleneck dimension for an autoencoder?

**Short Answer:** Start with 2-10x compression for low-dimensional data, or progressively increase until reconstruction loss stops improving significantly. Use validation data to find the sweet spot between compression and reconstruction quality.

**Deep Dive:** Guidelines for choosing latent dimension:

**For low-dimensional data (<1000 features)**:
- Try 10%, 25%, 50% of original dimension
- Monitor reconstruction loss on validation set

**For high-dimensional data (images)**:
- MNIST (784 dims): Try 32-128
- CIFAR-10 (3072 dims): Try 128-512
- Consider using information theory approaches

**Trade-offs**:
- Too small: High compression, poor reconstruction
- Too large: Overfitting, no dimensionality reduction
- Use early stopping to prevent overfitting

**Practical tip**: Plot reconstruction error vs. latent dimension—find the "elbow" where additional dimensions give diminishing returns.

---

### Q5: What is a denoising autoencoder (DAE) and when would you use it?

**Short Answer:** A denoising autoencoder corrupts the input with noise and learns to reconstruct the original clean input. It's used when you want robust feature learning or data denoising.

**Deep Dive:** DAE Architecture:

```
Noisy Input → Encoder → Latent z → Decoder → Clean Output
```

**Noise types**:
- Gaussian noise (add random values)
- Masking noise (randomly set inputs to zero)
- Salt-and-pepper noise (random bits flip)

**Why it works**:
- Forces network to learn structure, not memorize
- Better generalization than standard AE
- More robust latent representations

**Applications**:
- Image denoising
- Pretext tasks for representation learning
- Molecular property prediction

**Training tip**: Noise level is a hyperparameter—too much makes learning impossible, too little provides no benefit. Typical: 10-30% corruption.

---

### Q6: How do you train an autoencoder for anomaly detection?

**Short Answer:** Train on normal data only, then use reconstruction error as an anomaly score. High reconstruction error indicates anomalies since the network hasn't learned to reconstruct abnormal patterns.

**Deep Dive:** Steps for anomaly detection with autoencoders:

1. **Train only on normal data**: Learn the manifold of "good" samples
2. **Compute reconstruction errors**: For each sample, calculate ||x - x'||
3. **Set threshold**: Use validation set to find optimal error threshold
4. **Detect anomalies**: Flag samples with error > threshold

**Threshold selection methods**:
- Mean + k×std of reconstruction errors
- Percentile-based (e.g., top 5% as anomalies)
- Precision-recall curve optimization

**Advantages**:
- Unsupervised—no labels needed
- Works for high-dimensional data
- Can detect novel anomalies

**Limitations**:
- May miss anomalies similar to normal data
- Sensitive to noise in normal data

---

### Q7: What are the differences between autoencoders and variational autoencoders (VAEs)?

**Short Answer:** Standard AEs produce deterministic latent vectors; VAEs model the latent space as a probability distribution, enabling sampling and generation. VAEs add KL-divergence loss to regularize the latent space.

**Deep Dive:**

| Feature | Autoencoder | VAE |
|---------|-------------|-----|
| Latent space | Deterministic point | Probability distribution |
| Output | Fixed reconstruction | Can sample diverse outputs |
| Loss | Reconstruction only | Reconstruction + KL divergence |
| Training | Standard backprop | Reparameterization trick |
| Generation | Requires interpolation | Sample from distribution |

**VAE Loss**:
```
L = Reconstruction Loss + β × KL(q(z|x) || p(z))
```

**Reparameterization trick**: To sample z ~ N(μ, σ²), instead use:
```
z = μ + σ × ε, where ε ~ N(0, 1)
```

**VAE advantages**: Better latent space structure, enables generation
**AE advantages**: Simpler, often better reconstruction

---

## Architectural Questions (Edge Cases and Trade-offs)

### Q8: What is contractive autoencoder and what problem does it solve?

**Small Answer:** A contractive autoencoder adds a penalty on the Jacobian of the encoder, making the latent representation less sensitive to small changes in input. This creates more robust features.

**Deep Dive:** Contractive Autoencoder (CAE) Loss:

```
L = ||x - x'||² + λ ||∂f(x)/∂x||²
```

Where f(x) is the encoder output and λ controls contraction.

**Why it helps**:
- Robust to small input variations
- Learned features generalize better
- Captures local structure of data manifold

**Trade-offs**:
- Too large λ: Contract too much, lose information
- Too small λ: Behaves like standard AE

**Relation to other regularized AEs**:
- Denoising: Adds noise during training
- Sparse: Adds sparsity penalty
- Contractive: Penalizes sensitivity to input

---

### Q9: How do you handle very high-dimensional inputs in autoencoders?

**Short Answer:** Use convolutional autoencoders (CAE) for images/videos, or progressively growing architectures. For text, use sequence-to-sequence or transformer-based approaches.

**Deep Dive:** Architectures for high-dimensional data:

**Convolutional Autoencoders**:
- Encoder: Conv layers with pooling
- Decoder: Upsampling + Conv layers
- Preserves spatial structure
- Much fewer parameters than fully connected

**Fully Convolutional**:
- Uses transposed convolutions for upsampling
- Better for image generation

**Progressive/Stacked**:
- Train layer-by-layer
- Gradually increase capacity

**Practical considerations**:
- Start with small latent dimension
- Use skip connections (like U-Net)
- Consider memory-efficient implementations

---

### Q10: What are the limitations of autoencoders for generation tasks?

**Short Answer:** Standard autoencoders don't guarantee a structured latent space—sampling arbitrary points may produce meaningless outputs. VAEs address this but can suffer from posterior collapse. GANs often produce sharper results.

**Deep Dive:** Limitations:

1. **Unstructured latent space**: Points not on the data manifold produce garbage
2. **Posterior collapse (VAE)**: Latent space ignores input, decoder learns ignoring z
3. **Blurry reconstructions**: L2 loss favors average, not sharp outputs
4. **No adversarial training**: GANs often produce sharper results

**Solutions**:
- VAE with stronger KL term
- Adversarial autoencoders (combine AE + GAN)
- Diffusion-based models for higher quality
- Use perceptual losses

**When to choose what**:
- Reconstruction focus → Standard AE
- Generation focus → VAE or GAN
- Best quality → GANs or Diffusion models

---

## Answer Key

| Category | Question | Key Concept |
|----------|----------|-------------|
| Foundational | Q1 | AE architecture basics |
| Foundational | Q2 | Undercomplete vs overcomplete |
| Foundational | Q3 | Latent space properties |
| Applied | Q4 | Bottleneck dimension selection |
| Applied | Q5 | Denoising autoencoders |
| Applied | Q6 | Anomaly detection |
| Applied | Q7 | AE vs VAE differences |
| Architectural | Q8 | Contractive AE |
| Architectural | Q9 | High-dimensional handling |
| Architectural | Q10 | Generation limitations |
