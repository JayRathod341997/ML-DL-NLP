# Gated Recurrent Unit (GRU) - Interview Questions

## Foundational Questions (Definitions)

### Q1: What is a GRU and how does it differ from standard RNNs?

**Short Answer:** GRU is a gated RNN variant that uses update and reset gates to control information flow, solving the vanishing gradient problem in standard RNNs while being simpler than LSTM.

**Deep Dive:** GRU combines the cell state and hidden state into a single vector, using two gating mechanisms:

```
Update gate: z_t = σ(W_z · [h_{t-1}, x_t])
Reset gate: r_t = σ(W_r · [h_{t-1}, x_t])
Candidate: h_t = tanh(W · [r_t * h_{t-1}, x_t])
Hidden: h_t = (1 - z_t) * h_{t-1} + z_t * h_t
```

**Advantages over standard RNN**:
- Solves vanishing gradient problem
- Can learn long-term dependencies
- Similar benefits to LSTM but simpler

---

### Q2: Explain the update gate in GRU.

**Short Answer:** The update gate controls how much of the previous hidden state to carry forward. A value close to 1 means keeping most past information; a value close to 0 means focusing more on the current input.

**Deep Dive:** The update gate equation:

```
z_t = σ(W_z · [h_{t-1}, x_t] + b_z)
```

**Behavior**:
- **z_t ≈ 1**: Keep previous hidden state mostly unchanged (long-term memory)
- **z_t ≈ 0**: Reset to new candidate hidden state (short-term focus)

This allows the network to adaptively decide how much past information to preserve versus how much to update with new information.

---

### Q3: What is the reset gate and its role?

**Short Answer:** The reset gate determines how much of the previous hidden state to ignore when computing new memory. A low reset value lets the network "forget" irrelevant past information.

**Deep Dive:** The reset gate equation:

```
r_t = σ(W_r · [h_{t-1}, x_t] + b_r)
```

**Behavior**:
- **r_t ≈ 1**: Use all past information (standard recurrence)
- **r_t ≈ 0**: Ignore previous hidden state (compute from current input only)

This is useful when:
- Past information is no longer relevant
- Processing new "episode" or segment
- Breaking long dependency chains

---

## Applied Questions (How to Tune/Train)

### Q4: When should you choose GRU over LSTM?

**Short Answer:** Choose GRU when training speed matters, dataset is moderate size, or you need a simpler model. Use LSTM when dealing with very long sequences or when maximum capacity is needed.

**Deep Dive:** Decision factors:

| Factor | GRU Better | LSTM Better |
|--------|------------|-------------|
| Data size | Small to medium | Large |
| Sequence length | Short to moderate | Long |
| Training time | Critical | Can afford slower |
| Memory | Limited | Sufficient |
| Complexity | Simpler needed | Complex patterns |

**Practical guidelines**:
- Start with GRU as default
- Upgrade to LSTM if GRU underperforms
- Consider computational budget

---

### Q5: Why is GRU faster than LSTM?

**Short Answer:** GRU has fewer parameters (2 gates vs 3) and no separate cell state, resulting in fewer matrix multiplications, less memory, and faster computation.

**Deep Dive:** Parameter comparison:

| Component | GRU | LSTM |
|-----------|-----|------|
| Gates | 2 | 3 |
| Cell state | No (merged with hidden) | Yes |
| Parameters per layer | ~2/3 of LSTM | More |

**Time complexity**: GRU is approximately 20-30% faster per forward/backward pass.

**Memory**: GRU uses less GPU memory, enabling larger batch sizes.

---

### Q6: How do you implement bidirectional GRU?

**Short Answer:** Bidirectional GRU processes sequences in both forward and backward directions, concatenating the outputs. Use when the entire sequence is available and context from both directions matters.

**Deep Dive:** Bidirectional GRU:

```
Forward GRU: x_1, ..., x_n → h_fwd_1, ..., h_fwd_n
Backward GRU: x_n, ..., x_1 → h_bwd_1, ..., h_bwd_n

Output for each position: [h_fwd, h_bwd]
```

**Use cases**:
- NLP tasks (NER, sentiment analysis)
- When future context helps
- Offline processing (not real-time)

**Implementation**:
```python
nn.GRU(input_size, hidden_size, bidirectional=True)
# Output will have shape: (seq_len, batch, hidden_size * 2)
```

---

### Q7: How do GRU and LSTM compare on long-term dependencies?

**Short Answer:** LSTM's explicit cell state handles long-term dependencies slightly better in very long sequences, but GRU performs comparably for most practical sequence lengths. Both can capture dependencies over 100+ timesteps.

**Deep Dive:** Empirical observations:

- **Sequences < 50 timesteps**: No significant difference
- **Sequences 50-200 timesteps**: Similar performance
- **Sequences > 200 timesteps**: LSTM may have slight edge

**Why LSTM might be better for very long sequences**:
- Separate cell state provides dedicated "highway"
- Three gates provide more expressiveness
- More parameters = more capacity

**However, GRU advantages**:
- Fewer parameters = less overfitting
- Often generalizes better with less data

---

## Architectural Questions (Edge Cases and Trade-offs)

### Q8: What are the trade-offs between using fewer vs more GRU layers?

**Short Answer:** Fewer layers (1-2) are faster and less prone to overfitting. More layers (3+) can capture more complex patterns but require more data and computation.

**Deep Dive:** Layer considerations:

| Layers | Pros | Cons |
|--------|------|------|
| 1 | Fast, simple, less overfitting | May not capture complexity |
| 2 | Good balance | Moderate complexity |
| 3+ | Very complex patterns | Harder to train, overfitting |

**Guidelines**:
- Start with 1-2 layers
- Add layers if underfitting
- Use dropout between layers
- Consider bidirectional before deeper

---

### Q9: How does GRU handle variable-length sequences?

**Short Answer:** Use padding with masking or packed sequences. Sort sequences by length and use pack_padded_sequence to skip computation on padded tokens.

**Deep Dive:** Implementation:

```python
# Sort by length (required for packing)
lengths, perm_idx = lengths.sort(descending=True)
x = x[perm_idx]

# Pack
packed = pack_padded_sequence(x, lengths, batch_first=True)

# GRU forward
output, hidden = gru(packed)

# Unpack
output, _ = pad_packed_sequence(output, batch_first=True)
```

**Important**: Always sort sequences in descending order for packing to work correctly.

---

### Q10: Can GRU be used for sequence-to-sequence tasks?

**Short Answer:** Yes, GRU works well for seq2seq tasks with encoder-decoder architecture. For very long sequences or when attention is used, performance is comparable to LSTM.

**Deep Dive:** Seq2Seq with GRU:

**Encoder**: GRU processes input sequence, final hidden state carries context
```python
encoder_gru = nn.GRU(input_size, hidden_size)
_, hidden = encoder_gru(inputs)
```

**Decoder**: GRU generates output, conditioned on encoder hidden
```python
decoder_gru = nn.GRU(output_size, hidden_size)
output, hidden = decoder_gru(input.unsqueeze(0), hidden)
```

**With Attention**:
- Attention over encoder hidden states
- Improves performance on long sequences
- Commonly used in NMT, summarization

---

## Answer Key

| Category | Question | Key Concept |
|----------|----------|-------------|
| Foundational | Q1 | GRU vs RNN differences |
| Foundational | Q2 | Update gate function |
| Foundational | Q3 | Reset gate function |
| Applied | Q4 | GRU vs LSTM choice |
| Applied | Q5 | Computational efficiency |
| Applied | Q6 | Bidirectional GRU |
| Applied | Q7 | Long-term dependency comparison |
| Architectural | Q8 | Layer depth trade-offs |
| Architectural | Q9 | Variable-length sequences |
| Architectural | Q10 | Seq2seq tasks |
