# Siamese Networks

## Overview

Siamese networks consist of two identical subnetworks that learn similarity between inputs.

### Key Characteristics

- **Shared Weights**: Both branches have same weights
- **Similarity Learning**: Learn to distinguish similar/dissimilar
- **One-shot Learning**: Can recognize new classes from few examples

## Architecture

```
Input1 → Shared Network ↘
                          → Distance → Similarity
Input2 → Shared Network ↗
```

## Applications

- Face Verification
- Signature Verification
- Product Matching

## Quick Start

```bash
cd Deep_Learning/Siamese_Networks
uv run python codes/train.py
```
