# Recurrent Neural Networks - Quiz

## True/False Questions

### Question 1
**Statement:** In an RNN, the same weights are shared across all time steps during forward propagation.

**Answer:** True

**Explanation:** One of the key characteristics of RNNs is parameter sharing. The same weight matrices (W_ih, W_hh) are applied at every time step, which allows the network to handle variable-length sequences and generalize across different positions in the sequence.

---

### Question 2
**Statement:** Bidirectional RNNs can be used for real-time speech recognition.

**Answer:** False

**Explanation:** Bidirectional RNNs require the entire sequence to be available before processing, as they need to see future context. For real-time applications, unidirectional RNNs are preferred because they can process input as it arrives.

---

### Question 3
**Statement:** Teacher forcing always improves final model performance during inference.

**Answer:** False

**Explanation:** Teacher forcing can cause "exposure bias" - the model becomes accustomed to receiving correct inputs at each time step during training, but during inference it must use its own predictions, which may be incorrect and compound errors.

---

### Question 4
**Statement:** LSTMs can learn long-term dependencies better than basic RNNs due to their gating mechanisms.

**Answer:** True

**Explanation:** LSTMs use input, forget, and output gates that allow them to selectively remember or forget information over long sequences. This addresses the vanishing gradient problem in basic RNNs, enabling effective learning of long-term dependencies.

---

### Question 5
**Statement:** The vanishing gradient problem is more severe in RNNs than in deep feedforward networks.

**Answer:** True

**Explanation:** In RNNs, gradients are backpropagated through time, meaning they're multiplied by the same weight matrix at each time step. This can result in gradients that exponentially decay (or explode) over many time steps, making it harder to learn long-range dependencies compared to feedforward networks.

---

## Multiple Choice Questions

### Question 6
What is the primary purpose of the hidden state in an RNN?

A) To store the output of the network  
B) To capture and maintain information from previous time steps  
C) To reduce the number of parameters  
D) To apply the activation function

**Answer:** B

**Explanation:** The hidden state serves as the "memory" of the RNN. It captures information from all previous time steps and is updated at each step, allowing the network to maintain context across the sequence.

---

### Question 7
Which of the following is NOT a solution to the vanishing gradient problem in RNNs?

A) Using LSTM or GRU  
B) Increasing the learning rate  
C) Using gradient clipping  
D) Adding skip connections

**Answer:** B

**Explanation:** Increasing the learning rate would not solve vanishing gradients and could actually make training more unstable. The solutions include using gated mechanisms (LSTM/GRU), gradient clipping (for exploding gradients), skip connections, and proper initialization.

---

### Question 8
What type of RNN architecture is best suited for sentiment analysis of a movie review?

A) One-to-Many  
B) Many-to-One  
C) Many-to-Many (aligned)  
D) One-to-One

**Answer:** B

**Explanation:** Sentiment analysis takes a sequence of words (many inputs) and produces a single classification output (positive/negative), which is a many-to-one architecture.

---

### Question 9
What is "truncated backpropagation through time"?

A) A method to speed up forward pass  
B) Only backpropagating gradients through a limited number of time steps  
C) A type of RNN architecture  
D) A method for initializing weights

**Answer:** B

**Explanation:** Truncated BPTT limits the number of time steps over which backpropagation occurs. This is useful for very long sequences to reduce computational and memory costs, at the cost of not capturing very long-range dependencies.

---

### Question 10
In PyTorch, which method is used to efficiently handle variable-length sequences in RNNs?

A) torch.randn()  
B) torch.nn.utils.rnn_pad_sequence()  
C) pack_padded_sequence()  
D) nn.Linear()

**Answer:** C

**Explanation:** pack_padded_sequence() in PyTorch allows efficient processing of variable-length sequences by packing them together and skipping computation for padded elements, which saves computation time.

---

## Answer Key

| Question | Type | Answer | Key Concept |
|----------|------|--------|--------------|
| Q1 | T/F | True | Parameter sharing in RNNs |
| Q2 | T/F | False | Bidirectional RNN limitations |
| Q3 | T/F | False | Teacher forcing drawbacks |
| Q4 | T/F | True | LSTM gating mechanisms |
| Q5 | T/F | True | Vanishing gradients severity |
| Q6 | MCQ | B | Hidden state purpose |
| Q7 | MCQ | B | Vanishing gradient solutions |
| Q8 | MCQ | B | Many-to-one architecture |
| Q9 | MCQ | B | Truncated BPTT |
| Q10 | MCQ | C | Variable-length sequences |

## Brief Explanations

1. **Q1 (True):** RNNs use the same weights at each time step, enabling parameter sharing and handling variable-length sequences.
2. **Q2 (False):** Bidirectional RNNs require the entire sequence upfront, unsuitable for real-time processing.
3. **Q3 (False):** Teacher forcing causes exposure bias where the model hasn't learned to recover from its own mistakes.
4. **Q4 (True):** LSTM gates allow selective retention of information, solving the vanishing gradient problem.
5. **Q5 (True):** BPTT multiplies gradients through time, causing more severe vanishing/exploding gradients.
6. **Q6 (B):** The hidden state maintains context from previous time steps.
7. **Q7 (B):** Increasing learning rate doesn't fix vanishing gradients and may cause instability.
8. **Q8 (B):** Sentiment analysis takes multiple words and produces one classification.
9. **Q9 (B):** Truncated BPTT limits backpropagation steps for efficiency.
10. **Q10 (C):** pack_padded_sequence handles variable-length sequences efficiently.
