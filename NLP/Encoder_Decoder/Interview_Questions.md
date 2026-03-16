# Encoder-Decoder - Interview Questions

## Q1: What is Seq2Seq?

Sequence-to-sequence learning transforms one sequence to another, like translation.

## Q2: What does the encoder do?

Encoder processes input sequence and creates a context vector (hidden state) representing the input.

## Q3: What does the decoder do?

Decoder takes context vector and generates output sequence token by token.

## Q4: What is attention?

Attention allows decoder to focus on relevant parts of input when generating each output token, rather than using fixed context.

## Q5: What is the difference between encoder and decoder only models?

- Encoder-only (BERT): Good for understanding tasks
- Decoder-only (GPT): Good for generation tasks
- Encoder-Decoder (T5, BART): Good for both
