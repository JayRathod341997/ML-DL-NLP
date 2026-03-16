# Transformers & BERT - Interview Questions

## Q1: What is a Transformer?

A Transformer is a deep learning architecture introduced in 2017 that uses self-attention mechanism to process sequences in parallel, rather than sequentially like RNNs.

## Q2: What is Self-Attention?

Self-attention allows the model to weigh the importance of different words in a sequence when processing each word. It computes attention scores between all pairs of positions.

## Q3: What does BERT stand for?

BERT = Bidirectional Encoder Representations from Transformers

It's bidirectional because it reads text left-to-right AND right-to-left.

## Q4: Difference between encoder and decoder in Transformers?

- Encoder: Processes input sequence, creates representations
- Decoder: Generates output sequence auto-regressively

## Q5: What is transfer learning in BERT?

Pre-train on large corpus (like Wikipedia), then fine-tune on specific task. This is why BERT achieves state-of-the-art on many tasks.

## Q6: What are the two training objectives for BERT?

1. Masked Language Modeling (MLM)
2. Next Sentence Prediction (NSP)
