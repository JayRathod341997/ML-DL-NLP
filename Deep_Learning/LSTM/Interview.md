# Long Short-Term Memory (LSTM) - Interview Questions

## Foundational Questions (Definitions)

### Q1: What is an LSTM and why was it invented?

**Short Answer:** LSTM is a recurrent neural network architecture designed to solve the vanishing gradient problem in standard RNNs. It uses gating mechanisms to control information flow, enabling learning of long-term dependencies in sequential data.

**Deep Dive:** Standard RNNs suffer from:
- Vanishing gradients: Gradients shrink exponentially over time
- Exploding gradients: Gradients grow exponentially
- Can't learn dependencies beyond 5-10 timesteps

LSTM Solution:
- **Cell state**: Information highway across timesteps
- **Gates**: Control what to keep, forget, and output
- **Additive gradients**: Preserve gradient flow

LSTM gates:
1. **Forget gate**: What to discard from cell state
2. **Input gate**: What new information to store
3. **Output gate**: What to output

---

### Q2: Explain the LSTM architecture and its components.

**Short Answer:** LSTM has a cell state (the "memory") and three gates (forget, input, output). The cell state runs straight down the network with minor linear interactions, allowing information to flow unchanged easily.

**Deep Dive:** LSTM Equations:

```
Forget gate: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input gate: i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Candidate: C_t = tanh(W_c · [h_{t-1}, x_t] + b_c)

Cell state: C_t = f_t * C_{t-1} + i_t * C_t

Output gate: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
Hidden state: h_t = o_t * tanh(C_t)
```

**Visualization**:
```
x_t → [Forget][Input][Output] → h_t
         ↓         ↓         ↓
       C_{t-1} → C_t → C_{t+1}
```

**Key insight**: The cell state is the "highway"—gradients can flow through it without vanishing.

---

### Q3: What is the difference between LSTM and GRU?

**Short Answer:** GRU (Gated Recurrent Unit) combines the forget and input gates into an update gate, and merges the cell state with hidden state. GRU has fewer parameters (2 gates vs 3), trains faster, but may not capture dependencies as well as LSTM.

**Deep Dive:** Comparison:

| Feature | LSTM | GRU |
|---------|------|-----|
| Gates | 3 (forget, input, output) | 2 (update, reset) |
| Parameters | More | Fewer |
| Cell state | Yes | Merged with hidden |
| Speed | Slower | Faster |
| Memory | Better long-term | Good enough |
| When to use | Complex sequences | Simpler, faster needed |

**GRU Equations**:
```
Update gate: z_t = σ(W_z · [h_{t-1}, x_t])
Reset gate: r_t = σ(W_r · [h_{t-1}, x_t])
Candidate: h_t = tanh(W · [r_t * h_{t-1}, x_t])
Hidden: h_t = (1-z_t) * h_{t-1} + z_t * h_t
```

**When to choose**:
- LSTM: When training data is large, complex patterns
- GRU: When data is limited, faster training needed

---

## Applied Questions (How to Tune/Train)

### Q4: How do you handle variable-length sequences in LSTM?

**Short Answer:** Use padding with masking or pack sequences with packed sequences in PyTorch. Set appropriate sequence lengths and use the `pack_padded_sequence` and `pad_packed_sequence` functions.

**Deep Dive:** Methods for variable-length sequences:

**1. Padding + Masking**:
```python
# Pad sequences to same length
padded_sequences = pad_sequence(sequences, batch_first=True)

# Create mask (True for valid, False for padding)
mask = (sequences != PAD_TOKEN)

# In LSTM, pass mask to ignore padded values
# Or use packed sequences
```

**2. Packed Sequences** (PyTorch):
```python
# Sort by length (required)
lengths, idx = lengths.sort(descending=True)
sequences = sequences[idx]

# Pack
packed = pack_padded_sequence(sequences, lengths, batch_first=True)

# Forward pass
output, hidden = lstm(packed)

# Unpack
output, _ = pad_packed_sequence(output, batch_first=True)
```

**Best practices**:
- Sort sequences by length (descending) for efficiency
- Use masking to ignore padded values in loss
- Consider packing for memory efficiency

---

### Q5: What are bidirectional LSTMs and when should you use them?

**Short Answer:** Bidirectional LSTMs process sequences in both forward and backward directions, combining representations from both. Use when the entire sequence is available (not for real-time streaming) and context from both past and future matters.

**Deep Dive:** Bidirectional LSTM:

```
Forward: x_1, x_2, ..., x_n → h_forward_1, ..., h_forward_n
Backward: x_n, x_{n-1}, ..., x_1 → h_backward_n, ..., h_backward_1

Output: [h_forward, h_backward] for each timestep
```

**Use cases**:
- **NLP**: Sentiment analysis, POS tagging, NER
- **Time series**: When full sequence available
- **Speech recognition**: Acoustic modeling

**When NOT to use**:
- Real-time applications (need full sequence)
- Causal sequences (future shouldn't affect past)
- Memory constraints (2x computation)

---

### Q6: How do you prevent overfitting in LSTM networks?

**Short Answer:** Use dropout between layers (variational dropout), add L2 regularization, use early stopping, and monitor validation loss. Also consider reducing model complexity or using gradient clipping.

**Deep Dive:** Regularization techniques for LSTM:

**1. Dropout**:
```python
# Apply dropout to inputs (variational dropout)
lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.3)
# Dropout between layers (num_layers > 1)
```

**2. L2 Regularization**:
```python
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
```

**3. Gradient Clipping**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**4. Early Stopping**:
```python
if val_loss > best_val_loss + patience:
    break
```

**5. Other techniques**:
- Reduce hidden size
- Fewer layers
- Smaller batch size (more noise)
- Label smoothing

---

### Q7: How do you choose hyperparameters for LSTM?

**Short Answer:** Start with common defaults (128 hidden units, 2 layers), then tune based on validation performance. Key hyperparameters: hidden size, number of layers, dropout, learning rate, batch size.

**Deep Dive:** Hyperparameter guidelines:

| Hyperparameter | Typical Range | Notes |
|----------------|---------------|-------|
| Hidden size | 64-512 | Larger for complex tasks |
| Num layers | 1-4 | More layers = harder to train |
| Dropout | 0.1-0.5 | Between layers only |
| Learning rate | 1e-4 to 1e-3 | Use with scheduler |
| Batch size | 32-256 | Larger = more stable |
| Sequence length | Task-dependent | Memory constraints |

**Tuning strategy**:
1. Start with defaults
2. If overfitting: increase dropout, reduce size
3. If underfitting: increase capacity
4. Use learning rate scheduling
5. Consider bidirectional if applicable

---

## Architectural Questions (Edge Cases and Trade-offs)

### Q8: What is the difference between stateful and stateless LSTM?

**Short Answer:** Stateless LSTM resets cell state between sequences in a batch; stateful LSTM maintains cell state across sequences within a batch. Stateful requires careful batch ordering but captures longer dependencies.

**Deep Dive:**

| Mode | Cell State | Use Case |
|------|------------|----------|
| Stateless | Reset between sequences | Independent sequences |
| State | Retained across sequences | Continuous long sequences |

**Stateful LSTM**:
```python
lstm = nn.LSTM(batch_first=True, stateful=True)
# Cell state carries over
```

**Considerations**:
- Stateful: Order sequences logically, same batch = related
- Stateless: Shuffle between epochs, independent
- Stateful often needs sorted sequences

---

### Q9: How do attention mechanisms improve upon LSTMs?

**Short Answer:** Attention allows the model to focus on relevant parts of the input when producing each output, solving the bottleneck of encoding everything into a fixed-size vector. Transformers (attention-only) have largely replaced LSTMs for sequence tasks.

**Deep Dive:** LSTM limitation:
- Encode entire sequence into single vector h_n
- Information gets compressed, especially for long sequences

Attention solution:
- For each output step, attend to all input positions
- Compute weighted sum of input representations
- Can focus on relevant information

```python
# Attention mechanism
attention_weights = softmax(score(h_t, h_i))
context = sum(attention_weights * values)
output = concat(context, h_t)
```

**Evolution**:
1. Seq2Seq → Attention → Bahdanau Attention
2. Attention → Transformer (self-attention)
3. Transformers now dominate NLP

**Why Transformers beat LSTMs**:
- Parallel computation (no sequential)
- Direct long-range connections
- More expressive
- Better GPU utilization

---

### Q10: What are the computational costs of LSTMs compared to other architectures?

**Short Answer:** LSTMs have O(n × d²) complexity per timestep where n is sequence length and d is hidden size. They process sequentially (limited parallelization) and are slower than CNNs or Transformers for equivalent parameter counts.

**Deep Dive:** Complexity comparison:

| Architecture | Forward Pass | Parallelizable |
|-------------|--------------|----------------|
| RNN/LSTM | O(n × d²) | Partially |
| CNN | O(n × k × d) | Fully |
| Transformer | O(n² × d) | Fully |

**LSTM Challenges**:
- Sequential: Can't parallelize across timesteps
- Memory: Cell state needs O(d) memory per layer
- Long sequences: O(n) memory for backprop through time

**Practical implications**:
- LSTMs slower than CNNs for similar compute
- Transformers better for long sequences
- Consider truncated BPTT for very long sequences

---

## Answer Key

| Category | Question | Key Concept |
|----------|----------|-------------|
| Foundational | Q1 | LSTM purpose & invention |
| Foundational | Q2 | LSTM architecture & gates |
| Foundational | Q3 | LSTM vs GRU comparison |
| Applied | Q4 | Variable-length sequences |
| Applied | Q5 | Bidirectional LSTM |
| Applied | Q6 | Overfitting prevention |
| Applied | Q7 | Hyperparameter tuning |
| Architectural | Q8 | Stateful vs stateless |
| Architectural | Q9 | Attention vs LSTM |
| Architectural | Q10 | Computational complexity |
