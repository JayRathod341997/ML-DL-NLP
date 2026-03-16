# Neural Style Transfer

## Overview

Neural Style Transfer separates and recombines content and style from images using CNNs.

### Key Characteristics

- **Content Representation**: From deep CNN layers
- **Style Representation**: From Gram matrices
- **Optimization**: Minimize content + style loss

## Architecture

```
Content Image → CNN → Content Features
Style Image → CNN → Style Features
Combined Loss → Optimize
```

## Quick Start

```bash
cd Deep_Learning/Neural_Style_Transfer
uv run python codes/train.py
```

## Applications

- Art Generation
- Photo Editing
- Image Enhancement
