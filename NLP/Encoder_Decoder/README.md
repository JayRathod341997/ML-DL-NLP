# Encoder-Decoder (Seq2Seq)

## Explain Like I'm 5

Imagine you speak only English and someone speaks only Spanish! You need a translator in the middle!

Encoder-Decoder is like having a translator:
1. Encoder listens to English and creates a "meaning" (encoding)
2. Decoder takes that meaning and speaks Spanish (decoding)

This is called Sequence-to-Sequence (Seq2Seq) learning!

## What is Encoder-Decoder?

Neural network architecture that transforms one sequence to another:
- **Encoder**: Processes input sequence → context vector
- **Decoder**: Uses context vector → output sequence

## Components:

1. **Encoder**: RNN/LSTM/Transformer Encoder
2. **Decoder**: RNN/LSTM/Transformer Decoder
3. **Attention**: Helps decoder focus on relevant input parts

## Where Used?

1. **Machine Translation**: English → French
2. **Text Summarization**: Long text → Short summary
3. **Question Answering**: Question → Answer
4. **Chatbots**: User message → Bot response

## Evolution:

1. Basic Seq2Seq (RNN)
2. With Attention
3. Transformer (now standard)

## Enterprise Example:

**Google Translate** uses encoder-decoder architecture with attention to translate between 100+ languages!
