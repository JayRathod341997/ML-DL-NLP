# Convolutional Neural Networks (CNN)

## Overview

A Convolutional Neural Network (CNN) is a deep learning architecture specifically designed for processing structured grid data, particularly images. CNNs use convolutional layers that apply learnable filters to extract spatial hierarchies of features, making them highly effective for image classification, object detection, and visual recognition tasks.

### Key Characteristics

- **Local Connectivity**: Each neuron connects only to a local region (receptive field) of the input
- **Parameter Sharing**: The same filter weights are used across the entire input
- **Spatial Invariance**: Pooling layers provide translation invariance
- **Hierarchical Feature Learning**: Early layers learn edges, later layers learn complex patterns

## Core Concept

### Convolution Operation

The convolution operation is the fundamental building block of CNNs:

```
Output[i,j] = Σ Σ Input[i+m, j+n] × Filter[m,n] + Bias
```

Where:
- **Input**: Input feature map of size (H, W, C_in)
- **Filter**: Kernel of size (K, K, C_in) with C_out output channels
- **Output**: Feature map of size (H_out, W_out, C_out)

### Output Size Calculation

```
H_out = (H - K + 2×Padding) / Stride + 1
W_out = (W - K + 2×Padding) / Stride + 1
```

### Layer Types

| Layer | Purpose | Parameters |
|-------|---------|-------------|
| Conv2d | Extract features | Kernel size, stride, padding, filters |
| MaxPool2d | Downsample | Pool size, stride |
| AvgPool2d | Downsample | Pool size, stride |
| BatchNorm2d | Normalize | Number of features |
| Dropout2d | Regularize | Dropout probability |

### Architecture Pattern

```
Input → Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → FC → Output
```

## Quick Start

### Prerequisites

- Python 3.9+
- UV package manager installed

### Installation

1. **Create virtual environment:**
```bash
cd Deep_Learning/CNN
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
uv run python codes/train.py --epochs 50 --lr 0.001 --batch-size 64
```

### Testing

**Evaluate model performance:**
```bash
uv run python -c "from codes.model import CNN; import torch; model = CNN(); x = torch.randn(1, 1, 28, 28); print(model(x).shape)"
```

## Dataset

### Fetching Data via Script

Download training datasets:

```bash
uv run python codes/fetch_data.py
```

This script downloads:
- **MNIST**: Handwritten digits (28×28 grayscale, 60,000 training)
- **CIFAR-10**: 32×32 RGB images (50,000 training, 10 classes)

### Fetching Data via Kaggle API

For custom image datasets:

1. **Install Kaggle CLI:**
```bash
uv pip install kaggle
```

2. **Download dataset:**
```bash
uv run python -c "import kaggle; kaggle.api.dataset_download_files('dataset-name', path='datasets/', unzip=True)"
```

### Included Datasets

| Dataset | Description | Classes |
|---------|-------------|---------|
| MNIST | Handwritten digits (28×28) | 10 |
| CIFAR-10 | 32×32 RGB images | 10 |
| Custom Images | Place in `datasets/images/` | Varies |

### Data Loading Example

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

## Architecture

### LeNet-5 Inspired Architecture

```
Input (1×28×28)
  → Conv (6 filters, 5×5) → ReLU → Pool (2×2)
  → Conv (16 filters, 5×5) → ReLU → Pool (2×2)
  → Flatten
  → FC (120) → ReLU → Dropout
  → FC (84) → ReLU → Dropout
  → FC (10) → Softmax
```

### Modern CNN Variants

| Architecture | Key Innovation | Use Case |
|--------------|-----------------|----------|
| AlexNet | ReLU, Dropout | ImageNet classification |
| VGG | Deep networks (16-19 layers) | Transfer learning |
| ResNet | Skip connections | Very deep networks |
| Inception | Multi-scale filters | Efficient computation |

## Applications

- **Image Classification**: Cat vs Dog, disease detection
- **Object Detection**: YOLO, SSD, Faster R-CNN
- **Semantic Segmentation**: U-Net, Mask R-CNN
- **Face Recognition**: Siamese networks
- **Medical Imaging**: Tumor detection, X-ray analysis

## Limitations

- **Computationally Intensive**: Requires GPU for large datasets
- **Data Hungry**: Needs thousands of labeled images
- **Spatial Information Loss**: Fully connected layers lose location info
- **Invariance Trade-off**: Pooling provides translation invariance but loses precise location
