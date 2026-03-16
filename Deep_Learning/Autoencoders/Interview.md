# Autoencoders Interview Questions

## Q1: What is the difference between encoder and decoder in autoencoders?

**Answer:** The encoder compresses input into latent representation, while the decoder reconstructs input from latent representation. Encoder reduces dimensions; decoder expands them.

---

## Q2: What is the latent space in autoencoders?

**Answer:** Latent space is the compressed bottleneck representation. It's where encoded features are stored in lower dimensions.

---

## Q3: What loss function do autoencoders use?

**Answer:** Reconstruction loss - typically Mean Squared Error (MSE) for continuous data or cross-entropy for discrete data.

---

## Q4: What is a denoising autoencoder?

**Answer:** It learns to reconstruct clean data from noisy input. Noise is added to input, and network learns to remove it.

---

## Q5: What is undercomplete vs overcomplete autoencoder?

**Answer:** Undercomplete has smaller latent dimension than input (dimensionality reduction). Overcomplete has larger latent dimension (feature learning), prone to overfitting.
