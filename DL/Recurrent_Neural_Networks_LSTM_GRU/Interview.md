# RNN / LSTM / GRU - Interview Questions (with Answers)

## Basic

### Q1: What is an RNN?
**Answer:** A neural network for sequential data that updates a hidden state at each time step.

### Q2: Why are RNNs used for sequences?
**Answer:** They model order and temporal dependencies by carrying state forward.

## Intermediate

### Q3: What is vanishing gradient in RNNs?
**Answer:** Gradients shrink across many time steps, making it hard to learn long-range dependencies.

### Q4: How do LSTMs help?
**Answer:** LSTMs use gates (input/forget/output) and a cell state to better preserve information.

### Q5: How do GRUs differ from LSTMs?
**Answer:** GRUs are simpler (fewer gates) and often train faster; both address long dependency issues.

## Advanced

### Q6: What is teacher forcing?
**Answer:** In seq2seq training, feeding the true previous token as input during training rather than the model’s prediction.

### Q7: Why are Transformers often preferred now?
**Answer:** Better parallelization and stronger modeling of long-range dependencies via attention.

