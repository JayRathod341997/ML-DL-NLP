# Recurrent Neural Networks (RNN)

## Overview

A Recurrent Neural Network (RNN) is a type of neural network designed for processing sequential data. Unlike feedforward neural networks, RNNs have connections that loop back on themselves, allowing information to persist across time steps. This makes them ideal for tasks involving sequences like time series, text, and audio.

### Key Characteristics

- **Sequential Processing**: Processes data one element at a time while maintaining memory of previous inputs
- **Hidden State**: Maintains a hidden state that captures information about the sequence
- **Variable Length**: Can handle inputs of varying lengths
- **Temporal Dependencies**: Captures relationships between elements in a sequence

## Core Concept

### Forward Propagation

The fundamental RNN operation computes the hidden state at each time step:

```
h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
```

Where:
- **x_t**: Input at time step t
- **h_{t-1}**: Hidden state from previous time step
- **W_ih**: Input-to-hidden weights
- **W_hh**: Hidden-to-hidden (recurrent) weights
- **h_t**: Current hidden state

### Types of RNN

| Type | Use Case |
|------|----------|
| One-to-One | Image classification |
| One-to-Many | Image captioning |
| Many-to-One | Sentiment classification |
| Many-to-Many | Machine translation, POS tagging |

## Quick Start

### Prerequisites

- Python 3.9+
- UV package manager installed

### Installation

1. **Create virtual environment:**
```bash
cd Deep_Learning/RNN
uv venv
```

2. **Install dependencies:**
```bash
uv pip install torch torchvision numpy pandas matplotlib scikit-learn
```

3. **Activate environment:**
```bash
# On Windows
.venv\Scripts\activate

# On Mac/Linux
source .venv/bin/activate
```

### Training

**Run the training script:**
```bash
uv run python codes/train.py
```

**Run with custom parameters:**
```bash
uv run python codes/train.py --epochs 50 --lr 0.001 --batch-size 32
```

### Testing

**Evaluate model performance:**
```bash
uv run python -c "from codes.model import RNN; import torch; model = RNN(); x = torch.randn(10, 1, 20); h = model.init_hidden(1); output, h = model(x, h); print(output.shape)"
```

## Dataset

### Fetching Data via Script

Download training datasets:

```bash
uv run python codes/fetch_data.py
```

### Included Datasets

| Dataset | Description |
|---------|-------------|
| IMDB | Sentiment analysis (25K reviews) |
| Penn Treebank | Language modeling |
| Custom Sequences | Place in `datasets/` |

## Architecture

### Basic RNN Architecture

```
Input (seq_len, batch_size, input_size)
  → RNN Layer
  → Fully Connected Layer
  → Output
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Hidden Size | 256 | Units in hidden state |
| Num Layers | 2 | Stacked RNN layers |
| Dropout | 0.2 | Regularization |
| Bidirectional | False | Process sequence both directions |

## Applications

- **Natural Language Processing**: Text classification, language modeling, machine translation
- **Time Series Prediction**: Stock prices, weather forecasting
- **Speech Recognition**: Convert audio to text
- **Video Analysis**: Frame-by-frame processing

## Limitations

- **Vanishing Gradients**: Difficulty learning long-term dependencies
- **Exploding Gradients**: Can cause numerical instability
- **Slow Training**: Sequential processing limits parallelization
- **Limited Memory**: Hard to capture very long-range dependencies

## Solutions to Limitations

- **LSTM (Long Short-Term Memory)**: Gated mechanisms for long-term memory
- **GRU (Gated Recurrent Unit)**: Simplified gating mechanism
- **Gradient Clipping**: Prevents exploding gradients
- **Bidirectional RNNs**: Capture context from both directions
