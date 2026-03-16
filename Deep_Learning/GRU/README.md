# Gated Recurrent Units (GRU)

## Overview

Gated Recurrent Units (GRU) are a type of RNN introduced in 2014 as a simpler alternative to LSTM. They combine the forget and input gates into a single "update" gate and merge the cell state and hidden state.

### Key Characteristics

- **Fewer Parameters**: ~20% fewer than LSTM
- **Update Gate**: Controls how much past information to keep
- **Reset Gate**: Controls how much past information to forget
- **Simpler Architecture**: Easier to train and faster to compute

## Core Concept

### GRU Equations

```
z_t = σ(W_z · [h_{t-1}, x_t])  # Update gate
r_t = σ(W_r · [h_{t-1}, x_t])  # Reset gate
h̃_t = tanh(W · [r_t * h_{t-1}, x_t])  # New memory
h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t  # Final hidden state
```

## Quick Start

```bash
cd Deep_Learning/GRU
uv venv
uv pip install torch torchvision numpy pandas matplotlib
uv run python codes/train.py
```

## Architecture

### Standard GRU

```
Input → [GRU Cells] → FC → Output
```

## Applications

- Language modeling
- Machine translation
- Speech recognition
- Time series forecasting

## Comparison with LSTM

| Aspect | GRU | LSTM |
|--------|-----|------|
| Gates | 2 | 3 |
| Parameters | Fewer | More |
| Memory | Hidden only | Cell + Hidden |
| Speed | Faster | Slower |
| Performance | Similar | Similar |

## Limitations

- May not capture all long-term dependencies as well as LSTM
- Less flexible than LSTM due to fewer gates
- Can still suffer from vanishing gradients on very long sequences
