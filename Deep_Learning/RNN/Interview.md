# Recurrent Neural Networks - Interview Questions

## Foundational Questions (Definitions)

### Q1: What is a Recurrent Neural Network (RNN) and how does it process sequential data?

**Short Answer:** An RNN is a neural network designed for sequential data that maintains a hidden state capturing information from previous time steps, allowing it to process inputs of varying lengths while considering context.

**Deep Dive:** RNNs process sequential data by maintaining a "memory" (hidden state) that gets updated at each time step. Unlike feedforward networks that treat each input independently, RNNs share weights across time steps. The forward pass at each time step t computes: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b), where the hidden state h_t contains information about the entire sequence up to that point. This shared parameter approach allows RNNs to handle variable-length sequences and capture temporal dependencies. However, this also leads to challenges like vanishing/exploding gradients when processing long sequences.

---

### Q2: Explain the difference between stateless and stateful RNNs.

**Short Answer:** Stateless RNNs reset the hidden state between sequences (each sequence is processed independently), while stateful RNNs maintain the hidden state across batches, preserving information between training sequences.

**Deep Dive:** 
- **Stateless RNN**: The hidden state is initialized to zero (or a learned initial state) at the beginning of each training sequence. This is the default in most frameworks and works well when sequences are independent.
- **Stateful RNN**: The final hidden state of one batch becomes the initial hidden state for the next batch. This is useful when there are true dependencies between sequences (e.g., consecutive frames in a video), but requires careful batch preparation.

Key considerations: Stateful RNNs need sequences to be properly ordered and may be harder to train due to longer dependency chains. They also require more memory since hidden states must be maintained.

---

### Q3: What is the vanishing gradient problem in RNNs and why is it more severe than in feedforward networks?

**Short Answer:** Vanishing gradients in RNNs occur when gradients exponentially decay through time (vanishing over many time steps), making it impossible to learn long-term dependencies. It's more severe because gradients are multiplied through each time step, compounding the decay.

**Deep Dive:** In RNNs, backpropagation through time (BPTT) multiplies gradients at each time step. If the recurrent weight matrix has eigenvalues < 1, gradients exponentially shrink. After T time steps: ∂L/∂W ≈ Σ (λ)^T where λ < 1. This means after 50-100 time steps, gradients become practically zero. Unlike feedforward networks where depth is fixed, RNNs can have effectively "infinite" depth in the time dimension. Solutions include:
- LSTM/GRU networks with gating mechanisms
- Gradient clipping to prevent exploding gradients
- Skip connections in time (e.g., residual connections)
- Using ReLU activations instead of tanh

---

## Applied Questions (How to Tune/Train)

### Q4: How do you choose between using a basic RNN, LSTM, or GRU for your task?

**Short Answer:** Start with GRU (computationally efficient, fewer parameters), upgrade to LSTM if you need the forget gate, and only use basic RNN for very short sequences with sufficient data.

**Deep Dive:**

| Model | Parameters | Training Speed | Memory | Best For |
|-------|-----------|----------------|--------|----------|
| Basic RNN | Fewest | Fastest | Lowest | Very short sequences |
| GRU | Medium | Fast | Medium | General seq-to-seq |
| LSTM | Most | Slower | Highest | Long dependencies |

**When to choose:**
- **Basic RNN**: Only for simple tasks with very short sequences (< 20 time steps) and abundant data
- **GRU**: Default choice for most applications - faster training with performance close to LSTM
- **LSTM**: When you explicitly need the "forget gate" capability, or when research shows LSTM works better for your specific domain

The classic paper "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" (2014) showed GRU matches LSTM on many tasks while being more efficient.

---

### Q5: What is backpropagation through time (BPTT) and how does it work?

**Short Answer:** BPTT is the training algorithm for RNNs that unrolls the network through time, computes gradients at each time step, and sums them to update weights.

**Deep Dive:** BPTT works as follows:
1. **Unroll**: For a sequence of length T, create T copies of the network
2. **Forward Pass**: Compute outputs and store activations at each time step
3. **Backward Pass**: For each time step t from T to 1:
   - Compute gradient of loss with respect to outputs at time t
   - Propagate gradient back through the unrolled network
   - Accumulate gradients: ∂L/∂W = Σ_t ∂L_t/∂W

**Key considerations:**
- **Truncated BPTT**: For very long sequences, only backpropagate through a limited number of time steps (e.g., 35 steps) to save memory and computation
- **Stateful vs Stateless**: Affects how far back gradients can flow
- **Numerical Stability**: Watch for NaN/Inf values due to exploding gradients

---

### Q6: How do you handle variable-length sequences in RNNs?

**Short Answer:** Use padding with packing - pad sequences to a fixed length, then use packed sequences (PackedSequence in PyTorch) to efficiently skip padding tokens during computation.

**Deep Dive:** Two approaches:
1. **Padding**: Pad all sequences to the same length using a special PAD token. During training, create a mask to ignore PAD tokens in loss computation.
2. **Packing**: Store sequences as variable-length, packing them together for efficient computation.

Best practices:
- Pad with zeros or a special token (e.g., <PAD>)
- Use pack_padded_sequence in PyTorch for efficient batching
- Create attention masks to prevent attending to padding
- Apply masking to loss functions: masked_loss = loss[valid_tokens].mean()

Example PyTorch:
```python
# Sort by length (required for pack_padded_sequence)
 lengths = [len(seq) for seq in sequences]
 _, idx_sort = torch.sort(torch.tensor(lengths), descending=True)
 _, idx_unsort = torch.sort(idx_sort)
 
 # Pack
 packed = pack_padded_sequence(sequences[idx_sort], lengths[idx_sort])
 output, hidden = self.rnn(packed)
 
 # Unpack
 output, _ = pad_packed_sequence(output)
```

---

## Architectural Questions (Edge Cases and Trade-offs)

### Q7: Compare unidirectional and bidirectional RNNs. When would you use each?

**Short Answer:** Unidirectional RNNs process sequences forward (past → future), while bidirectional processes both forward and backward, requiring the entire sequence upfront. Use bidirectional for tasks where full context is available (e.g., text classification), unidirectional for real-time streaming.

**Deep Dive:**

| Aspect | Unidirectional | Bidirectional |
|--------|----------------|---------------|
| Context | Past only | Past + Future |
| Latency | Can be real-time | Must see full sequence |
| Memory | O(n) | O(2n) |
| Use Cases | Streaming, generation | Classification, translation |

**When to use bidirectional:**
- **Text Classification**: You have the full text upfront (sentiment analysis, spam detection)
- **Named Entity Recognition**: Need context from both directions
- **Machine Translation Encoder**: Encode full source sentence
- **Any task where full sequence is available**

**When to use unidirectional:**
- **Real-time Applications**: Can't wait for future data (speech recognition)
- **Language Modeling**: Can only use past context (predicting next word)
- **Streaming Data**: Processing continues indefinitely
- **Memory Constraints**: Bidirectional doubles memory requirements

---

### Q8: What is teacher forcing and what are its advantages and disadvantages?

**Short Answer:** Teacher forcing uses actual ground truth inputs from the previous time step during training instead of the model's predictions, accelerating training but causing exposure bias during inference.

**Deep Dive:**

**Teacher Forcing:**
```
# During training (at each time step t)
actual_input = ground_truth[t-1]  # Use actual, not predicted
hidden = rnn(actual_input, hidden)
```

**Without Teacher Forcing (self-feeding):**
```
# During training
predicted_input = model.output[t-1]  # Use prediction
hidden = rnn(predicted_input, hidden)
```

**Advantages:**
- Faster convergence (no error propagation from early mistakes)
- More stable training gradients
- Enables stacking many layers

**Disadvantages:**
- **Exposure Bias**: Model never sees its own mistakes during training, so errors compound during inference
- **Mismatch**: Training differs from inference

**Solutions:**
- **Scheduled Sampling**: Gradually transition from teacher forcing to self-feeding
- **Professor Forcing**: Use adversarial training to align teacher-forced and self-feeding distributions
- **Curriculum Learning**: Start with teacher forcing, slowly introduce self-feeding

---

### Q9: How do you prevent overfitting in RNNs?

**Short Answer:** Use dropout (applied between layers, not within time steps), early stopping, and proper regularization. For RNNs specifically, use recurrent dropout and consider zoneout.

**Deep Dive:**

**Standard Regularization:**
1. **Dropout**: Apply between RNN layers (not within time steps, which damages temporal structure)
   ```python
   # In PyTorch: dropout parameter in RNN layer
   rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=0.3)
   ```
2. **L2 Regularization**: Weight decay on all parameters
3. **Early Stopping**: Monitor validation loss

**RNN-Specific Techniques:**
1. **Recurrent Dropout**: Drop connections between time steps (introduced in "A Theoretically Grounded Application of Dropout in Recurrent Neural Networks")
   ```python
   # In PyTorch
   rnn = nn.LSTM(input_size, hidden_size, dropout=0.3, recurrent_dropout=0.2)
   ```
2. **Zoneout**: Randomly keep hidden states from previous time steps (regularization for recurrent connections)
3. **Gradient Clipping**: Clip gradients to prevent exploding gradients (doesn't prevent overfitting but stabilizes training)
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

**Data Augmentation for Sequences:**
- Noise injection
- Time warping
- Random cutting (Randomly remove subsequences)

---

### Q10: Explain the differences between RNN output modes: many-to-many, many-to-one, and one-to-many.

**Short Answer:** 
- **Many-to-One**: Multiple inputs → single output (classification)
- **One-to-Many**: Single input → multiple outputs (caption generation)
- **Many-to-Many**: Multiple inputs → multiple outputs (translation, tagging)

**Deep Dive:**

| Architecture | Use Case | Example |
|--------------|-----------|---------|
| **Many-to-One** | Sequence classification | Sentiment analysis (text → positive/negative) |
| **One-to-Many** | Generation from single input | Image captioning (image → word sequence) |
| **Many-to-Many (Aligned)** | Sequence tagging | POS tagging, NER (each word → tag) |
| **Many-to-Many (Delayed)** | Sequence-to-sequence | Machine translation, chatbot |

**Architecture Diagrams:**

```
Many-to-One:          One-to-Many:          Many-to-Many:
x x x x x → ○        ○ → x x x x x        x x x x x → x x x x x
(input seq)          (single input)       (input seq)
    ↓                    ↓                    ↓
  h h h h h          h → h → h → h       h → h → h → h
    ↓                    ↓                    ↓
    ○                   x x x x x           x x x x x
 (single output)      (output seq)         (output seq)
```

**Implementation considerations:**
- Many-to-One: Take the last hidden state (or use attention)
- One-to-Many: Use the same RNN cell repeatedly with shifted outputs
- Many-to-Many: Can be "aligned" (same length) or "delayed" (encoder-decoder)

---

## Answer Key

| Category | Question | Key Concept |
|----------|----------|-------------|
| Foundational | Q1 | RNN sequential processing |
| Foundational | Q2 | Stateful vs stateless |
| Foundational | Q3 | Vanishing gradients in BPTT |
| Applied | Q4 | Choosing RNN/LSTM/GRU |
| Applied | Q5 | BPTT algorithm |
| Applied | Q6 | Variable-length sequences |
| Architectural | Q7 | Bidirectional RNNs |
| Architectural | Q8 | Teacher forcing |
| Architectural | Q9 | Overfitting prevention |
| Architectural | Q10 | Output modes |
