# Long Short-Term Memory Networks - Quiz

## True/False Questions

### Question 1
**Statement:** LSTMs have a separate cell state and hidden state.

**Answer:** True

**Explanation:** LSTMs maintain two types of state: cell state (C_t) for long-term memory and hidden state (h_t) for short-term output. This separation allows better control over information flow.

---

### Question 2
**Statement:** The forget gate in LSTM decides what new information to add to the cell state.

**Answer:** False

**Explanation:** The forget gate decides what information to REMOVE from the cell state. The input gate decides what new information to add.

---

### Question 3
**Statement:** GRU has more parameters than LSTM.

**Answer:** False

**Explanation:** GRU has fewer parameters than LSTM because it combines the forget and input gates into a single "update" gate and doesn't have a separate cell state.

---

### Question 4
**Statement:** Bidirectional LSTM can be used for real-time speech recognition.

**Answer:** False

**Explanation:** Bidirectional LSTM requires the entire sequence before processing, as it needs future context. For real-time applications, unidirectional LSTM is required.

---

### Question 5
**Statement:** LSTMs can theoretically learn dependencies across thousands of time steps.

**Answer:** True

**Explanation:** Due to the gating mechanism and the linear carousel in the cell state, LSTMs can propagate gradients without vanishing, allowing learning of very long-term dependencies.

---

## Multiple Choice Questions

### Question 6
How many gates does a standard LSTM have?

A) 1  
B) 2  
C) 3  
D) 4

**Answer:** C

**Explanation:** Standard LSTM has 3 gates: input gate, forget gate, and output gate.

---

### Question 7
What is the primary advantage of LSTM over basic RNN?

A) Faster training  
B) Solves vanishing gradient problem  
C) Fewer parameters  
D) Simpler architecture

**Answer:** B

**Explanation:** The gating mechanism in LSTM allows it to solve the vanishing gradient problem, enabling learning of long-term dependencies.

---

### Question 8
Which gate in LSTM decides what to output?

A) Input gate  
B) Forget gate  
C) Output gate  
D) Update gate

**Answer:** C

**Explanation:** The output gate (o_t) decides what parts of the cell state to output.

---

### Question 9
In peephole LSTM, gates can access:

A) Only current input  
B) Only previous hidden state  
C) Cell state  
D) Future inputs

**Answer:** C

**Explanation:** Peephole connections allow gates to see the cell state, providing additional context for decision-making.

---

### Question 10
What is recommended dropout usage in LSTMs?

A) Apply on all connections  
B) Apply only on recurrent connections  
C) Apply between layers, and use recurrent_dropout within  
D) Don't use dropout with LSTM

**Answer:** C

**Explanation:** Dropout should be applied between layers (standard dropout) and within LSTM layers using recurrent_dropout parameter, not on all recurrent connections through time.

---

## Answer Key

| Question | Type | Answer | Key Concept |
|----------|------|--------|--------------|
| Q1 | T/F | True | Cell and hidden states |
| Q2 | T/F | False | Gate functions |
| Q3 | T/F | False | GRU parameters |
| Q4 | T/F | False | Bidirectional limitations |
| Q5 | T/F | True | Long-term dependencies |
| Q6 | MCQ | C | 3 gates |
| Q7 | MCQ | B | Vanishing gradient |
| Q8 | MCQ | C | Output gate |
| Q9 | MCQ | C | Peephole connections |
| Q10 | MCQ | C | Dropout in LSTM |

## Brief Explanations

1. **Q1 (True):** LSTMs have both cell state (long-term) and hidden state (short-term).
2. **Q2 (False):** Forget gate removes info; input gate adds new info.
3. **Q3 (False):** GRU has fewer gates and thus fewer parameters.
4. **Q4 (False):** Bidirectional requires full sequence upfront.
5. **Q5 (True):** Linear carousel enables long-term gradient flow.
6. **Q6 (C):** Input, forget, and output gates.
7. **Q7 (B):** Gates enable gradient flow without vanishing.
8. **Q8 (C):** Output gate controls what to output.
9. **Q9 (C):** Peepholes let gates see cell state.
10. **Q10 (C):** Proper dropout placement is crucial for LSTMs.
