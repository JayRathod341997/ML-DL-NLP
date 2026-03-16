# Generative Adversarial Networks - Interview Questions

## Foundational Questions (Definitions)

### Q1: What is a Generative Adversarial Network (GAN)?

**Short Answer:** A GAN is a generative model framework where two neural networks (generator and discriminator) compete in a minimax game. The generator creates fake samples, and the discriminator tries to distinguish real from fake samples.

**Deep Dive:** GAN Architecture:

```
Generator: Random noise z → Fake samples G(z)
Discriminator: Real/Fake → Probability D(x)

Training objective: min_G max_D E[log D(x)] + E[log(1 - D(G(z)))]
```

**Key Components**:

1. **Generator (G)**: Takes random noise, produces fake data
   - Learns to fool the discriminator
   - Typically a neural network (CNN for images, MLP for tabular)

2. **Discriminator (D)**: Binary classifier
   - Outputs probability that input is real
   - Receives both real and fake samples

**Training Process**:
- Fix G, train D to distinguish real/fake
- Fix D, train G to fool D
- Alternate until Nash equilibrium (D = 0.5 everywhere)

---

### Q2: Explain the minimax game in GAN training.

**Short Answer:** The generator and discriminator play a zero-sum game where the generator tries to minimize the discriminator's ability to distinguish fakes, while the discriminator tries to maximize its accuracy. This creates adversarial competition.

**Deep Dive:** The GAN objective function:

```
min_G max_D V(D, G) = E_x~p_data[log D(x)] + E_z~p_z[log(1 - D(G(z)))]
```

**For the discriminator** (maximize):
- Wants D(x) → 1 for real samples (log D(x))
- Wants D(G(z)) → 0 for fake samples (log(1-D(G(z))))

**For the generator** (minimize):
- Wants D(G(z)) → 1 to fool discriminator (log(1-D(G(z))) → -∞)

**Nash Equilibrium**: Both networks reach a state where:
- Generator produces perfect samples
- Discriminator can't distinguish (D(x) = 0.5)

**Training challenges**:
- Mode collapse: Generator produces limited variety
- Non-convex optimization: May not reach equilibrium
- Vanishing gradients: When D is too good too fast

---

### Q3: What is mode collapse in GANs?

**Short Answer:** Mode collapse occurs when the generator learns to produce only a subset of the data distribution (a few modes), failing to capture the full diversity of real data. The generator finds one type of output that fools the discriminator.

**Deep Dive:** Example of mode collapse:
- Training on a dataset with 10 digit classes
- Generator produces only digit "7" samples
- Discriminator learns to reject "7" fakes but generator doesn't learn other digits
- Result: High quality "7" images, but no diversity

**Why it happens**:
- Generator finds a weak spot in discriminator
- When D can't distinguish one type of fake, G stuck on that type
- Generator doesn't have incentive to explore new outputs

**Solutions**:

| Method | Description |
|--------|-------------|
| Minibatch discrimination | Let D see multiple samples at once |
| Unrolled GANs | Train D with future G updates |
| WGAN-GP | Wasserstein distance with gradient penalty |
| Spectral normalization | Normalize D weights |
| Diverse learning rates | Different LR for G and D |

---

## Applied Questions (How to Tune/Train)

### Q4: How do you balance generator and discriminator training?

**Short Answer:** A common rule is training the discriminator more often (1-5 steps per generator step). Monitor losses—if discriminator loss goes to zero quickly, it's too strong. Use learning rate scheduling.

**Deep Dive:** Training balance strategies:

**Discriminator too strong**:
- Symptoms: D loss → 0, G loss plateaus
- Solutions: Reduce D training steps, increase G learning rate, use spectral normalization

**Generator too strong**:
- Symptoms: D loss spikes, G loss very low
- Solutions: Reduce G training steps, use label smoothing

**Practical guidelines**:
```
# Start with:
for _ in range(k):  # k = 1-5
    Train D (real + fake)
Train G

# Monitor:
- D accuracy should be ~50-70%
- If D = 100% accurate, G not learning
- If D = 0% accurate (even on real), D is broken
```

**Modern approaches**:
- WGAN: More stable training, no balance needed
- Self-attention GAN: Better long-range dependencies

---

### Q5: What is the difference between vanilla GAN, DCGAN, and WGAN?

**Short Answer:** Vanilla GAN uses simple MLPs; DCGAN uses deep convolutions for better image generation; WGAN uses Wasserstein distance for more stable training. Each improves on training stability and output quality.

**Deep Dive:**

| GAN Type | Architecture | Key Features |
|----------|--------------|--------------|
| Vanilla | MLP for G and D | Original formulation, unstable |
| DCGAN | CNN for G and D | Stable, produces sharp images |
| WGAN | Any architecture | Wasserstein loss, mode seeking |
| WGAN-GP | Same as WGAN | Gradient penalty, smoother |
| StyleGAN | Style-based generator | High-quality, controllable synthesis |

**DCGAN Best Practices**:
- Batch normalization in G and D
- LeakyReLU in discriminator
- Tanh output for G (normalized data)
- No pooling, use strided convolutions

**WGAN Key Changes**:
- Remove sigmoid from D output
- Use Wasserstein distance: D(x) - D(G(z))
- Weight clipping or gradient penalty

---

### Q6: How do you evaluate GAN performance?

**Short Answer:** Use Inception Score (IS) for quality/diversity trade-off and Fréchet Inception Distance (FID) for detailed comparison. Also use human evaluation and visual inspection for subjective assessment.

**Deep Dive:** Evaluation metrics:

**Inception Score (IS)**:
```
IS = exp(E_x[KL(p(y|x) || p(y))])
```
- Higher = better quality + diversity
- Doesn't compare to real data
- Can be gamed by mode collapse

**Fréchet Inception Distance (FID)**:
```
FID = ||μ_real - μ_fake||² + Tr(Σ_real + Σ_fake - 2(Σ_realΣ_fake)^0.5)
```
- Lower = better (0 = perfect)
- Compares real and generated distributions
- More reliable than IS

**Other metrics**:
- Precision/Recall (precision = quality, recall = diversity)
- Perceptual path length (smooth interpolation)
- Human evaluation (subjective)

**Common practice**:
- Use multiple metrics
- Visual inspection still important
- Report both quality and diversity

---

### Q7: What is gradient penalty in WGAN-GP and why is it used?

**Short Answer:** WGAN-GP adds a gradient penalty term to the loss to enforce the 1-Lipschitz constraint that Wasserstein GAN requires. This prevents the need for weight clipping and leads to smoother gradients.

**Deep Dive:** WGAN-GP Loss:

```
L = E[D(G(z))] - E[D(x)] + λ E[||∇_x D(x)||_2 - 1)²]
```

**Why gradient penalty**:
- WGAN requires D to be 1-Lipschitz (gradients bounded by 1)
- Original WGAN used weight clipping (problematic)
- Gradient penalty enforces the constraint directly

**Implementation**:
- Sample random points between real and fake distributions
- Compute gradient of D at those points
- Penalize deviation from gradient = 1

**Benefits**:
- More stable training
- No careful weight clipping
- Works with various architectures
- Reduces mode collapse

---

## Architectural Questions (Edge Cases and Trade-offs)

### Q8: What are the advantages and disadvantages of conditional GANs?

**Short Answer:** Conditional GANs (cGANs) add class labels as input to both G and D, enabling controlled generation. However, they require labeled data and can suffer from class imbalance.

**Deep Dive:** cGAN Architecture:

```
G: (noise z, class c) → Fake image
D: (image, class c) → Real/Fake probability
```

**Advantages**:
- Controllable generation (specify class)
- Can handle multiple domains
- Useful for image-to-image translation

**Disadvantages**:
- Need labeled data
- Training can be unbalanced across classes
- May generate class-specific artifacts

**Applications**:
- Class-conditional image generation
- Image-to-image translation (pix2pix)
- Text-to-image synthesis

---

### Q9: How do progressive growing GANs (ProGAN) achieve high-resolution generation?

**Short Answer:** ProGAN starts with low-resolution images (4×4) and progressively adds layers to increase resolution. This staged approach stabilizes training and enables very high-resolution synthesis.

**Deep Dive:** Progressive Growing Strategy:

```
Step 1: Train 4×4 generator
Step 2: Fade in 8×8 layers
Step 3: Train 8×8 generator
Step 4: Fade in 16×16 layers
...
Step N: Train 1024×1024 generator
```

**Key techniques**:
- Fade-in: Gradually add new layers (like a mask)
- Equalized learning rate: Scale weights by fan-in
- Pixel-wise feature normalization
- Mini-batch standard deviation (at high res)

**Benefits**:
- More stable training
- Faster convergence
- Can generate 1024×1024+ images
- Less mode collapse

---

### Q10: What are the trade-offs between GANs and VAEs for generation?

**Short Answer:** GANs produce sharper, more realistic images but can suffer from mode collapse and don't provide exact latent space interpolation. VAEs offer complete latent space but produce blurry images.

**Deep Dive:**

| Aspect | GAN | VAE |
|--------|-----|-----|
| Image quality | Sharp, realistic | Blurry, averaged |
| Latent space | Unstructured | Structured |
| Mode collapse | Common | None |
| Training | Unstable | Stable |
| Loss | Adversarial | Reconstruction + KL |
| Interpolation | May produce artifacts | Smooth |
| Diversity | Limited | Full |

**When to use GANs**:
- High-quality image synthesis
- Style transfer
- When realism is paramount

**When to use VAEs**:
- Latent space analysis needed
- Data compression
- When training stability matters
- Anomaly detection

**Modern trend**: Combine both (e.g., VAE-GAN) for best of both worlds.

---

## Answer Key

| Category | Question | Key Concept |
|----------|----------|-------------|
| Foundational | Q1 | GAN architecture basics |
| Foundational | Q2 | Minimax game theory |
| Foundational | Q3 | Mode collapse |
| Applied | Q4 | G/D training balance |
| Applied | Q5 | GAN variant comparisons |
| Applied | Q6 | Evaluation metrics |
| Applied | Q7 | Gradient penalty |
| Architectural | Q8 | Conditional GANs |
| Architectural | Q9 | Progressive GANs |
| Architectural | Q10 | GAN vs VAE trade-offs |
