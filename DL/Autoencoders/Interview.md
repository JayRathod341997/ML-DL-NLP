# Autoencoders - Interview Questions (with Answers)

## Basic

### Q1: What is an autoencoder?
**Answer:** A neural network trained to reconstruct its input, typically via an encoder-decoder architecture with a bottleneck.

### Q2: What is the bottleneck / latent space?
**Answer:** The compressed representation produced by the encoder.

## Intermediate

### Q3: How can autoencoders be used for anomaly detection?
**Answer:** Train on normal data; anomalies have larger reconstruction error.

### Q4: What is a denoising autoencoder?
**Answer:** An autoencoder trained to reconstruct clean input from a corrupted/noisy version.

## Advanced

### Q5: What can go wrong with anomaly detection autoencoders?
**Answer:** If model capacity is high or training data includes anomalies, reconstruction error may not separate well; threshold selection and monitoring are critical.

