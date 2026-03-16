# Long Short-Term Memory Networks - Interview Questions

## Foundational Questions

### Q1: Explain the LSTM gating mechanism and how it solves the vanishing gradient problem.

**Short Answer:** LSTMs use input, forget, and output gates to control information flow, allowing selective retention of information over long sequences and solving vanishing gradients through linear identity connections.

**Deep Dive:** The three gates work together: Forget gate (f_t) decides what to discard from previous cell state; Input gate (i_t) decides what new information to add; Output gate (o_t) decides what to output. The cell state (C_t) acts as a "conveyor belt" running down the entire chain, with only minor linear interactions. This linear carousel (C_t = f_t * C_{t-1} + i_t * C̃_t) allows gradients to flow unchanged through many time steps, solving vanishing gradients.

---

### Q2: What is the difference between LSTM and GRU?

**Short Answer:** GRU combines forget and input gates into an "update" gate, has fewer parameters than LSTM, and doesn't have a cell state - just a hidden state.

**Deep Dive:**
- LSTM has 3 gates (input, forget, output) + cell state
- GRU has 2 gates (update, reset) + hidden state only
- GRU is computationally faster (fewer gates = fewer matrix multiplications)
- LSTM is better when you need the extra flexibility of the cell state
- Empirical studies show similar performance on most tasks

---

### Q3: How does peephole connection work in LSTMs?

**Short Answer:** Peephole connections allow gates to "peek" at the cell state, providing additional context for gate decisions.

**Deep Dive:** In standard LSTM, gates only see previous hidden state and current input. Peepholes add direct connections from cell state to gates, allowing: f_t = σ(W_f · [h_{t-1}, x_t, C_{t-1}]). This helps gates make better decisions based on long-term memory.

---

## Applied Questions

### Q4: When would you use bidirectional LSTM?

**Short Answer:** Use bidirectional LSTM when you have access to the full sequence at once and need context from both past and future (e.g., POS tagging, machine translation encoder).

**Deep Dive:** Bidirectional LSTM runs two LSTMs - one forward on the sequence, one backward. The outputs are concatenated. This provides complete context. Not suitable for: real-time applications, language modeling (need only past context), streaming data.

---

### Q5: How do you prevent overfitting in LSTMs?

**Answer:** Use dropout (between layers, not recurrent connections), recurrent dropout, early stopping, and proper regularization.

**Deep Dive:** Dropout in LSTMs must be applied carefully:
- Use dropout between LSTM layers (standard dropout)
- Use recurrent_dropout within LSTM for dropout on recurrent connections
- Never apply dropout on recurrent connections through time (damages memory)
- Combine with L2 regularization for best results

---

## Architectural Questions

### Q6: What are the differences between stateful and stateless LSTM?

**Short Answer:** Stateful LSTM maintains cell state between batches; stateless resets state at each batch.

**Deep Dive:** Stateful: useful when sequences truly connect across batches (video frames). Requires careful batch ordering. More memory, harder to train. Stateless: standard approach, each batch is independent. Easier to train, less memory.

---

### Q7: How do you choose the number of LSTM layers?

**Answer:** Start with 1-2 layers. Add more only if underfitting. More layers = more capacity but harder to train.

**Deep Dive:**
- 1 layer: Most tasks
- 2 layers: Complex sequence tasks
- 3+ layers: Only with skip connections and careful regularization
- Watch for vanishing gradients in deeper stacks
- Consider using smaller hidden size with more layers vs larger hidden size with fewer

---

## Answer Key

| Question | Key Concept |
|----------|-------------|
| Q1 | LSTM gating mechanisms |
| Q2 | LSTM vs GRU |
| Q3 | Peephole connections |
| Q4 | Bidirectional LSTM |
| Q5 | Overfitting prevention |
| Q6 | Stateful vs stateless |
| Q7 | Layer selection |
