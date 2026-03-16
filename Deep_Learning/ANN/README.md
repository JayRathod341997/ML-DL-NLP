# Artificial Neural Networks (ANN)

## Overview

An Artificial Neural Network (ANN) is a computational model inspired by the biological neural networks in animal brains. It consists of interconnected nodes (neurons) organized in layers that process information through weighted connections. ANNs learn patterns from data through a process called training, where the network adjusts connection weights to minimize prediction error.

### Key Characteristics

- **Feedforward Architecture**: Information flows in one direction from input to output
- **Learning**: Automatically adjusts weights based on error signals via backpropagation
- **Universal Approximation**: Can approximate any continuous function given enough neurons
- **Non-linear Modeling**: Can capture complex non-linear relationships in data

## Core Concept

### Mathematical Foundation

The fundamental unit of an ANN is the **neuron** (also called a perceptron in its simplest form). The mathematical operation performed by a single neuron is:

```
y = σ(W · x + b)
```

Where:
- **x** = Input vector of shape `(input_size,)`
- **W** = Weight matrix of shape `(input_size, output_size)`
- **b** = Bias vector of shape `(output_size,)`
- **σ** = Activation function (non-linear)
- **y** = Output vector of shape `(output_size,)`

### Forward Propagation

For a network with L layers, the forward pass is computed as:

```
a^(0) = x                           # Input layer
a^(l) = σ(z^(l))                   # Hidden/Output layers
z^(l) = W^(l) · a^(l-1) + b^(l)    # Pre-activation
```

### Loss Functions

- **Binary Cross-Entropy** (Classification): `L = -Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]`
- **Mean Squared Error** (Regression): `L = (1/n)·Σ(y - ŷ)²`
- **Cross-Entropy** (Multi-class): `L = -ΣΣ[y_ij·log(ŷ_ij)]`

### Activation Functions

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| Sigmoid | σ(x) = 1/(1+e^(-x)) | (0, 1) | Binary output |
| ReLU | max(0, x) | [0, ∞) | Hidden layers |
| Tanh | (e^x - e^(-x))/(e^x + e^(-x)) | (-1, 1) | Hidden layers |
| Softmax | e^(x_i)/Σe^(x_j) | (0, 1) | Multi-class output |

## Quick Start

### Prerequisites

- Python 3.9+
- UV package manager installed

### Installation

1. **Create virtual environment:**
```bash
cd Deep_Learning/ANN
uv venv
```

2. **Install dependencies:**
```bash
uv pip install torch torchvision numpy pandas matplotlib scikit-learn
```

3. **Activate environment and verify:**
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
uv run python codes/train.py --epochs 100 --lr 0.001 --batch_size 32
```

### Testing

**Evaluate model performance:**
```bash
uv run python -c "from codes.model import NeuralNetwork; import torch; model = NeuralNetwork(); print(model(torch.randn(1, 10)))"
```

## Dataset

### Fetching Data via Script

We provide a script to download and prepare the datasets:

```bash
uv run python codes/fetch_data.py
```

This script downloads:
- **MNIST**: Handwritten digits (60,000 training, 10,000 test)
- **CIFAR-10**: 32x32 RGB images (50,000 training, 10,000 test)

### Fetching Data via Kaggle API

For custom datasets from Kaggle:

1. **Install Kaggle CLI:**
```bash
uv pip install kaggle
```

2. **Set up credentials:**
```bash
# Place kaggle.json in ~/.kaggle/
# Or set environment variables
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

3. **Download dataset:**
```bash
uv run python -c "import kaggle; kaggle.api.dataset_download_files('dataset-name', path='datasets/', unzip=True)"
```

### Included Datasets

The `datasets/` folder contains sample data for exercises:

| Dataset | Type | Description |
|---------|------|-------------|
| `train.csv` | Tabular | Features for training |
| `test.csv` | Tabular | Features for testing |
| `labels.csv` | Text | Class labels |

### Custom Dataset Loading

```python
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Load CSV data
data = pd.read_csv('datasets/train.csv')
X = data.drop('label', axis=1).values
y = data['label'].values

# Convert to tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)

# Create DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Architecture

### Network Structure

```
Input Layer (784) → Hidden 1 (256) → Hidden 2 (128) → Output (10)
```

- **Input Layer**: 784 neurons (28×28 flattened image)
- **Hidden Layer 1**: 256 neurons with ReLU activation
- **Hidden Layer 2**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax (for 10 classes)

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Learning Rate | 0.001 | Step size for gradient descent |
| Batch Size | 32 | Samples per training iteration |
| Epochs | 50 | Complete passes through dataset |
| Hidden Units | [256, 128] | Neurons per hidden layer |
| Dropout | 0.2 | Regularization rate |

## Applications

- **Pattern Recognition**: Image classification, facial recognition
- **Prediction**: Stock prices, weather forecasting
- **Natural Language Processing**: Text classification, sentiment analysis
- **Medical Diagnosis**: Disease detection from medical images

## Limitations

- **Black Box**: Difficult to interpret decision-making process
- **Data Hungry**: Requires large amounts of training data
- **Computationally Intensive**: Needs GPU for large-scale training
- **Overfitting**: Prone to memorizing training data without generalization
