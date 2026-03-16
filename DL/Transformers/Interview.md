# Transformers - Interview Questions (with Answers)

## Basic

### Q1: What is attention?
**Answer:** A mechanism that computes weighted combinations of values based on similarity between queries and keys.

### Q2: What is self-attention?
**Answer:** Attention where queries, keys, and values all come from the same sequence.

## Intermediate

### Q3: Why do Transformers need positional encoding?
**Answer:** Self-attention alone is permutation-invariant; positional information is needed to represent order.

### Q4: What is multi-head attention?
**Answer:** Multiple attention heads learn different relations in parallel and are concatenated.

## Advanced

### Q5: Why is attention O(n^2) in sequence length?
**Answer:** It computes pairwise interactions between tokens (n x n attention matrix).

### Q6: Name ways to handle long sequences.
**Answer:** Sparse/linear attention variants, chunking, retrieval augmentation, sliding window attention.

