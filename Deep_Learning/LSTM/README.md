# Long Short-Term Memory Networks (LSTM)

## Overview

Long Short-Term Memory (LSTM) networks are a special type of Recurrent Neural Network designed to learn long-term dependencies. Introduced by Hochreiter and Schmidhuber in 1997, LSTMs address the vanishing gradient problem through gating mechanisms that allow information to flow unchanged over long sequences.

### Key Characteristics

- **Gating Mechanisms**: Input, forget, and output gates control information flow
- **Long-term Memory**: Can remember information for thousands of time steps
- **Cell State**: Maintains a "highway" for information flow
- **Gradient Flow**: Mitigates vanishing gradient through linear carousels

## Core Concept

### LSTM Architecture

The LSTM has three gates:

1. **Forget Gate**: Decides what information to discard from the cell state
   ```
   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
   ```

2. **Input Gate**: Decides what new information to store
   ```
   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
   C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
   ```

3. **Output Gate**: Decides what to output
   ```
   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
   h_t = o_t * tanh(C_t)
   ```

### Cell State Update

```
C_t = f_t * C_{t-1} + i_t * C̃_t
```

## Quick Start

### Prerequisites

- Python 3.9+
- UV package manager installed

### Installation

```bash
cd Deep_Learning/LSTM
uv venv
uv pip install torch torchvision numpy pandas matplotlib scikit-learn
```

### Training

```bash
uv run python codes/train.py
```

## Architecture

### Standard LSTM

```
Input → [LSTM Cells] → FC → Output
```

### Variants

| Variant | Description |
|---------|-------------|
| Bidirectional LSTM | Process sequence both directions |
| Stacked LSTM | Multiple LSTM layers |
| Peephole LSTM | Gates see cell state |

## Applications

- **Machine Translation**: Google Translate
- **Speech Recognition**: Voice assistants
- **Time Series Forecasting**: Stock prediction
- **Text Generation**: Language modeling

## Limitations

- Complex architecture with many parameters
- Slower training than basic RNN
- Can still suffer from short-term memory issues
