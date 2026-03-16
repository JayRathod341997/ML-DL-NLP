# Transformers Interview Questions

## Q1: What is self-attention?

**Answer:** Mechanism that computes attention between all positions in a sequence, allowing the model to weigh the importance of different parts of the input.

## Q2: Why use positional encoding?

**Answer:** Transformers have no recurrence, so positional encoding adds information about token positions in the sequence.

## Q3: What is multi-head attention?

**Answer:** Multiple attention heads run in parallel, each learning different types of relationships between tokens.

## Q4: Difference between encoder and decoder?

**Answer:** Encoder processes input sequence, decoder generates output. Encoder uses self-attention, decoder uses both self-attention and encoder-decoder attention.
