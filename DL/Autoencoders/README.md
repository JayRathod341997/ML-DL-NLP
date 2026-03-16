# Autoencoders - Explain Like I'm 5

## What is an Autoencoder?

Imagine you have a magic box that tries to **copy** your drawing.

But the box has a tiny hole: only a small message can pass through.  
So it must learn to **compress** the drawing into a small code and then **reconstruct** it.

That’s an **autoencoder**:
- **Encoder:** compresses input into a latent code
- **Decoder:** reconstructs the input from that code

## Why Autoencoders Are Useful

If the model learns to reconstruct “normal” data well, then:
- Normal samples reconstruct with low error
- Anomalies reconstruct with high error

This makes autoencoders useful for **anomaly detection**.

## Where It’s Used

- Anomaly detection in transactions, sensors, logs
- Dimensionality reduction / feature learning
- Denoising (denoising autoencoders)

## Benefits

- Learns compact representations without labels
- Useful as a feature-learning step in some pipelines

## Limitations

- An autoencoder can sometimes reconstruct anomalies too if capacity is too high
- Threshold selection for anomaly detection needs validation

## Example in This Folder

- Dataset: `dataset/anomaly_2d.csv` (2D points + label)
- Code shows training an autoencoder on normal points and using reconstruction error.

## Enterprise-Level Example

In a payments system:
- Train an autoencoder on “normal” transaction feature vectors
- In production, compute reconstruction error per transaction
- Route high-error transactions for additional verification or rules-based checks

