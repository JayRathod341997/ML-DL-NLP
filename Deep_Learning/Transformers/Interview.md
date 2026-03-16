# Transformers - Interview Questions

## Foundational Questions (Definitions)

### Q1: What is the self-attention mechanism and how does it work?

**Short Answer:** Self-attention allows each position in a sequence to attend to all positions in the previous layer, computing a weighted sum where weights represent relevance. It captures dependencies regardless of distance.

**Deep Dive:** Self-attention computation:

```
For each token:
1. Create Query (Q), Key (K), Value (V) vectors from input
2. Compute attention scores: attention(Q, K) = softmax(QK^T / √d_k)
3. Weight values: output = attention × V

Multi-head attention:
- Run attention in parallel h times (different learned projections)
- Concatenate outputs
- Linear projection to final output
```

**Why self-attention over RNNs**:
- Parallel computation (no sequential processing)
- Direct long-range connections
- Captures dependencies in constant number of steps
- Better for GPU parallelization

---

### Q2: Why do Transformers need positional encoding?

**Short Answer:** Unlike RNNs, Transformers process all tokens simultaneously and have no inherent sense of order. Positional encoding adds position information to preserve sequence order.

**Deep Dive:** Positional encoding options:

**Sinusoidal (original)**:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Learned positional embeddings**:
- Learn position embeddings like word embeddings
- More parameters but more flexible

**Why sinusoids work**:
- Each position gets unique encoding
- Encodings for nearby positions are similar
- Model can learn to attend by relative position

---

### Q3: What is multi-head attention and why use multiple heads?

**Short Answer:** Multi-head attention runs several attention mechanisms in parallel, each with different learned projections. This allows the model to capture different types of relationships simultaneously.

**Deep Dive:** Multi-head mechanism:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Benefits of multiple heads**:
- Learn different types of relationships:
  - Syntactic dependencies
  - Semantic associations
  - Local vs global context
- Each head can specialize
- More expressive than single attention

**Typical configurations**:
- BERT: 12-16 heads
- GPT: 12-96 heads
- Dimension per head: 64 (so 12 heads = 768 total)

---

## Applied Questions (How to Tune/Train)

### Q4: What is the difference between encoder-only, decoder-only, and encoder-decoder Transformers?

**Short Answer:** Encoder-only (BERT) processes input for understanding; Decoder-only (GPT) generates autoregressively; Encoder-decoder (T5, BART) for sequence-to-sequence tasks.

**Deep Dive:**

| Architecture | Use Cases | Attention |
|--------------|-----------|-----------|
| Encoder-only | Classification, NER, QA | Bidirectional |
| Decoder-only | Text generation | Causal (left-to-right) |
| Encoder-decoder | Translation, summarization | Encoder-decoder |

**Examples**:
- **Encoder-only**: BERT, RoBERTa, DistilBERT
- **Decoder-only**: GPT, GPT-2, GPT-3, GPT-4
- **Encoder-decoder**: T5, BART, MarianMT

---

### Q5: How do you handle long sequences in Transformers?

**Short Answer:** Use sliding window attention, sparse attention, or linear attention. Also consider chunking, hierarchical processing, or efficient architectures like Longformer.

**Deep Dive:** Challenges with long sequences:
- O(n²) attention complexity
- Memory quadratic in sequence length

**Solutions**:

| Method | Complexity | Description |
|--------|------------|-------------|
| Sliding window | O(n × w) | Only attend to w neighbors |
| Sparse attention | O(n × √n) | Fixed sparsity pattern |
| Linear attention | O(n) | Kernel-based attention |
| Longformer | O(n) | Sliding + global attention |
| Reformer | O(n log n) | Locality-sensitive hashing |

**Practical tips**:
- Start with standard Transformer
- Use shorter sequences if possible
- Consider pretrained models with long context

---

### Q6: What is the role of feed-forward networks in Transformers?

**Short Answer:** The feed-forward network (FFN) after each attention layer provides non-linear transformation and increases model capacity. It processes each position independently with the same weights.

**Deep Dive:** FFN in Transformer:

```python
# Usually two linear layers with ReLU
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

**Why needed**:
- Adds non-linearity (attention is linear)
- Processes each position separately
- Increases model capacity
- Often the majority of parameters

**Typical size**:
- Hidden dimension: 4 × model_dim
- BERT-base: 768 → 3072 → 768

---

### Q7: How do you fine-tune a pretrained Transformer model?

**Short Answer:** Replace the task head and train all or part of the model on your data. Use lower learning rate for pretrained layers, consider freezing embeddings for domain adaptation.

**Deep Dive:** Fine-tuning strategies:

**Full fine-tuning**:
```python
# Train all parameters
optimizer = Adam(model.parameters(), lr=2e-5)
```

**Freeze + fine-tune**:
- Freeze embeddings or early layers
- Train only classifier head initially
- Gradually unfreeze

**Layer-wise learning rate decay**:
```python
# Different LR for different layers
lr = base_lr × 0.9^(layer_index_from_top)
```

**Best practices**:
- Learning rate: 2e-5 to 5e-5 (much lower than training)
- Few epochs (2-5)
- Warmup (10% of steps)
- Weight decay for regularization

---

## Architectural Questions (Edge Cases and Trade-offs)

### Q8: What is the difference between GPT and BERT pretraining objectives?

**Short Answer:** BERT uses masked language modeling (predict masked tokens from context). GPT uses causal language modeling (predict next token from previous). BERT is bidirectional; GPT is unidirectional.

**Deep Dive:** Pretraining comparison:

| Objective | Direction | Method | Use Case |
|-----------|-----------|--------|----------|
| MLM (BERT) | Bidirectional | Mask 15% tokens, predict | Understanding |
| CLM (GPT) | Unidirectional | Predict next token | Generation |
| Permutation (XLNet) | Permutation | Random order prediction | Both |

**Training differences**:
- BERT: 15% masked, predict from both sides
- GPT: All tokens, predict from left only
- XLNet: All permutations, predict from random position

---

### Q9: What are the scaling laws for Transformers?

**Short Answer:** Model performance improves predictably with more compute, data, and parameters following power laws. However, optimal ratios between model size and data exist.

**Deep Dive:** Scaling observations:

**Key findings** (Chinchilla paper):
- More data as important as more parameters
- Optimal: 20 tokens per parameter
- Compute-optimal: 3× more data than parameters

**Practical implications**:
- Don't just scale parameters
- Scale data proportionally
- Smaller models trained longer beat larger models trained less

**Recommendations**:
- For given compute budget: balance model + data
- More parameters need more training tokens
- Don't overtrain or undertrain

---

### Q10: What are the main differences between Transformer-XL and standard Transformers?

**Short Answer:** Transformer-XL uses segment-level recurrence and relative positional encoding to handle very long sequences, enabling learning dependencies beyond the fixed context length.

**Deep Dive:** Transformer-XL innovations:

**1. Segment-level Recurrence**:
```
Cache hidden states from previous segment
At new segment: use cached + current hidden states
→ Enables infinitely long context
```

**2. Relative Positional Encoding**:
- Instead of absolute positions, use relative offsets
- More generalizable to different lengths
- Enables attention beyond seen context

**Benefits**:
- Capture 1000+ token dependencies
- No context fragmentation
- Faster inference (reuse cache)

---

## Answer Key

| Category | Question | Key Concept |
|----------|----------|-------------|
| Foundational | Q1 | Self-attention mechanism |
| Foundational | Q2 | Positional encoding |
| Foundational | Q3 | Multi-head attention |
| Applied | Q4 | Encoder/decoder differences |
| Applied | Q5 | Long sequence handling |
| Applied | Q6 | Feed-forward networks |
| Applied | Q7 | Fine-tuning strategies |
| Architectural | Q8 | GPT vs BERT objectives |
| Architectural | Q9 | Scaling laws |
| Architectural | Q10 | Transformer-XL improvements |
