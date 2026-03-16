# Variational Autoencoders - Interview Questions

## Foundational Questions (Definitions)

### Q1: What is a Variational Autoencoder (VAE)?

**Short Answer:** A VAE is a generative model that learns to encode data into a probabilistic latent space. Unlike standard autoencoders that produce deterministic outputs, VAEs model the latent space as a probability distribution, enabling sampling and generation.

**Deep Dive:** VAE Architecture:

```
Encoder: x → q(z|x) = N(μ(x), σ(x))
  - Outputs mean μ and variance σ for each latent dimension
  
Decoder: z → p(x|z)
  - Reconstructs input from latent sample

Latent space: z ~ q(z|x), sample from the distribution
```

**Key Components**:
- **Encoder**: Maps input to distribution parameters (μ, σ)
- **Latent space**: Probabilistic bottleneck (not deterministic)
- **Decoder**: Generates output from sampled latent vector
- **Reparameterization trick**: Enables gradient flow through sampling

**Training objective**: Maximize Evidence Lower Bound (ELBO)
```
L = E[log p(x|z)] - KL(q(z|x) || p(z))
  = Reconstruction - KL Divergence
```

---

### Q2: Explain the reparameterization trick in VAEs.

**Short Answer:** The reparameterization trick moves the random sampling operation out of the network by expressing the latent variable as a deterministic function of parameters and independent noise. This enables backpropagation through the sampling step.

**Deep Dive:** Problem without reparameterization:

```
z ~ q(z|x) = N(μ, σ²)
```
- Sampling is stochastic, gradient can't flow through it

Solution with reparameterization:

```
ε ~ N(0, 1)  # Independent noise
z = μ + σ × ε  # Deterministic function
```

Now gradient can flow through μ and σ:

```
∂L/∂μ and ∂L/∂σ computable via chain rule
```

**Variants**:
- **Standard**: z = μ + σ × ε
- **Log-variance**: z = μ + exp(0.5 × log_var) × ε (more stable)
- **Snake**: Use softplus instead of exp for σ

---

### Q3: What is the ELBO (Evidence Lower Bound)?

**Short Answer:** ELBO is the variational lower bound on log-likelihood that VAEs optimize. It consists of a reconstruction term and a KL divergence regularization term, balancing between reconstruction quality and latent space structure.

**Deep Dive:** ELBO Derivation:

```
log p(x) = L(q) + KL(q(z|x) || p(z|x))
         ≥ L(q)  (since KL ≥ 0)

where L(q) = E[log p(x|z)] - KL(q(z|x) || p(z))
```

**ELBO Components**:

1. **Reconstruction term**: E[log p(x|z)]
   - Measures how well decoder reconstructs input
   - Higher = better reconstruction

2. **KL divergence term**: KL(q(z|x) || p(z))
   - Regularizes latent space
   - Pushes q(z|x) toward prior p(z) = N(0, I)
   - Lower = latent space closer to standard normal

**Trade-off**:
- β-VAE: L = E[log p(x|z)] + β × KL(...)
- β > 1: More disentanglement, worse reconstruction
- β < 1: Better reconstruction, less structure

---

## Applied Questions (How to Tune/Train)

### Q4: How do you choose the latent dimension in VAE?

**Short Answer:** Start with a small dimension (10-50) and increase until reconstruction quality stops improving significantly. Use β-VAE for disentanglement if interpretable latent features are needed.

**Deep Dive:** Guidelines:

**For simple data (MNIST)**:
- Latent dim: 10-32
- MNIST digits can be represented in ~10 dimensions

**For complex data (CIFAR, faces)**:
- Latent dim: 64-256
- More complex data needs more capacity

**Trade-offs**:
- Too small: Poor reconstruction, can't capture variation
- Too large: Overfitting, less structured latent space
- With β > 1: Can use larger dim with disentanglement

**Practical approach**:
1. Train with latent dim = 10, 20, 50, 100
2. Plot reconstruction error vs latent dim
3. Choose at the "elbow" point
4. Consider downstream task performance

---

### Q5: What is posterior collapse and how do you prevent it?

**Short Answer:** Posterior collapse occurs when the latent variable z becomes uninformative—the decoder ignores it and learns to reconstruct directly from input statistics. This defeats the purpose of the latent space.

**Deep Dive:** Symptoms:
- KL loss drops to near zero
- Decoder learns to ignore z
- Generated samples lack diversity
- Latent space becomes useless

**Why it happens**:
- Decoder too powerful
- KL weight too high initially
- Training too aggressive

**Solutions**:

| Method | Description |
|--------|-------------|
| KL annealing | Gradually increase KL weight during training |
| Free bits | Prevent KL from dropping below threshold |
| β-VAE with lower β | Use β < 1 initially |
| Improved decoder | Use stronger encoder, weaker decoder |
| Skip connections | Connect encoder directly to decoder |

---

### Q6: How do VAEs handle different data types (images, text, tabular)?

**Short Answer:** The decoder architecture changes based on data type—convolutional for images, RNN/transformer for text, MLP for tabular. The VAE framework (encoder → latent → decoder) remains the same.

**Deep Dive:**

| Data Type | Encoder | Decoder | Considerations |
|-----------|---------|---------|-----------------|
| Images | CNN | Deconv/Transposed CNN | Spatial structure |
| Text | RNN/Transformer | RNN/Transformer | Sequential, discrete |
| Tabular | MLP | MLP | Mixed data types |
| Audio | WaveNet | WaveNet | Temporal structure |

**Image VAE specifics**:
- Encoder: Conv layers, pooling
- Decoder: Transposed conv, upsampling
- Loss: Perceptual loss can help

**Text VAE specifics**:
- Autoregressive decoder needs care
- Often has posterior collapse issues
- Word embeddings matter

---

### Q7: What are the differences between VAE and GAN for generation?

**Short Answer:** VAEs optimize reconstruction + KL divergence, producing blurry but complete latent spaces. GANs use adversarial training, producing sharper images but with unstable training and potential mode collapse.

**Deep Dive:** Comparison:

| Aspect | VAE | GAN |
|--------|-----|-----|
| Training | Stable, deterministic | Unstable, adversarial |
| Latent space | Complete, interpolable | May have gaps |
| Output quality | Blurry | Sharp |
| Mode collapse | None | Possible |
| Diversity | Full | Limited |
| Reconstruction | Exact | Approximate |

**VAE-GAN Hybrid**:
- Use VAE encoder, GAN generator, GAN discriminator
- Combines VAE structure with GAN quality
- More complex training but better results

---

## Architectural Questions (Edge Cases and Trade-offs)

### Q8: What is β-VAE and how does β affect learning?

**Short Answer:** β-VAE adds a weight β to the KL term in ELBO: L = Reconstruction + β × KL. Higher β forces more disentanglement but worsens reconstruction. Lower β improves reconstruction but gives less structured latent space.

**Deep Dive:** β Parameter effects:

**β = 1**: Standard VAE
- Balances reconstruction and latent structure
- Good starting point

**β > 1 (e.g., 4-10)**: Strong regularization
- More disentangled latent factors
- Can learn interpretable features
- Worse reconstruction
- May cause posterior collapse if too high

**β < 1**: Weak regularization
- Better reconstruction
- Less structured latent space
- May not learn useful representations

**Disentanglement**:
- Independent latent dimensions capture independent factors
- Higher β → more disentanglement
- Trade-off: reconstruction vs. interpretability

---

### Q9: How do you implement a VAE for conditional generation?

**Short Answer:** Conditional VAE (CVAE) adds conditioning information (class labels) to both encoder and decoder. The model learns p(x|y, z) rather than p(x|z), enabling controlled generation.

**Deep Dive:** CVAE Architecture:

```
Encoder: (x, y) → (μ, σ)  # Input + condition
Decoder: (z, y) → x'     # Latent + condition

Loss = Reconstruction + KL(q(z|x,y) || p(z|y))
     = Reconstruction + KL(q(z|x,y) || N(0,I))  # prior can depend on y
```

**Applications**:
- Class-conditional generation
- Image-to-image translation
- Controlled attribute manipulation

**Training**:
- Similar to VAE
- Include class labels in both encoder/decoder
- Can combine with supervised learning

---

### Q10: What are the limitations of VAEs and how do modern variants address them?

**Short Answer:** VAEs suffer from blurry outputs (L2 loss), posterior collapse, and limited expressiveness. Modern variants like VQ-VAE, diffusion models, and neural ODEs address these issues.

**Deep Dive:** Limitations and Solutions:

| Limitation | Solution |
|------------|----------|
| Blurry outputs | Perceptual loss, GAN loss, diffusion |
| Posterior collapse | KL annealing, free bits |
| Limited expressiveness | VQ-VAE (discrete), IAF |
| Poor samples | VAE-GAN, more capacity |

**VQ-VAE (Vector Quantized)**:
- Discrete latent space using codebook
- Prevents posterior collapse
- Enables autoregressive decoding

**Diffusion Models** (current state-of-art):
- Instead of autoencoding, denoise
- Can be viewed as hierarchical VAE
- Much better sample quality

**Neural ODEs**:
- Continuous-time latent dynamics
- More flexible posterior

---

## Answer Key

| Category | Question | Key Concept |
|----------|----------|-------------|
| Foundational | Q1 | VAE architecture basics |
| Foundational | Q2 | Reparameterization trick |
| Foundational | Q3 | ELBO objective |
| Applied | Q4 | Latent dimension selection |
| Applied | Q5 | Posterior collapse |
| Applied | Q6 | Different data types |
| Applied | Q7 | VAE vs GAN comparison |
| Architectural | Q8 | β-VAE parameter |
| Architectural | Q9 | Conditional VAE |
| Architectural | Q10 | VAE limitations |
