# ResNet (Residual Networks)

## Overview

ResNet uses skip connections (residual connections) to enable training of very deep networks, solving the degradation problem.

### Key Characteristics

- **Skip Connections**: Identity mappings
- **Deep Networks**: 50-152 layers
- **Residual Blocks**: Learning residual functions

## Architecture

```
Input → Conv → Residual Block → ... → Output
         ↓______________↑
```

## Quick Start

```bash
cd Deep_Learning/ResNet
uv run python codes/train.py
```

## Applications

- Image Classification
- Object Detection
- Semantic Segmentation
