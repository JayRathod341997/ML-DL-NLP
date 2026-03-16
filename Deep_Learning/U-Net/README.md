# U-Net

## Overview

U-Net is a CNN architecture for biomedical image segmentation with encoder-decoder structure and skip connections.

### Key Characteristics

- **Encoder**: Captures context
- **Decoder**: Enables precise localization
- **Skip Connections**: Preserve spatial information

## Architecture

```
     Encoder     Decoder
Input → ... → → ... → Output
       ↙↘       ↙↘
       Skip Connections
```

## Quick Start

```bash
cd Deep_Learning/U-Net
uv run python codes/train.py
```

## Applications

- Medical Image Segmentation
- Satellite Imaging
- Self-driving Cars
