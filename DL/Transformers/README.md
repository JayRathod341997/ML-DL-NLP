# Transformers - Explain Like I'm 5

## What is a Transformer?

Imagine a teacher reading a sentence.  
When the teacher sees the word “it”, they look back at the whole sentence to figure out what “it” refers to.

A **Transformer** does something similar using **attention**:
it learns to “pay attention” to the most relevant parts of the input.

## Key Idea: Self-Attention

Self-attention lets each token look at other tokens and decide what matters.
This helps with long-range relationships and enables parallel processing (no step-by-step recurrence like RNNs).

## Where It’s Used

- NLP: translation, chatbots, search, summarization
- Vision Transformers (ViT) for images
- Recommendations and time series (in some setups)

## Benefits

- Strong performance on many sequence tasks
- Parallelizable training (faster than RNNs on long sequences)

## Limitations

- Attention can be expensive for very long sequences (quadratic cost in classic form)
- Requires lots of data/compute for best results

## Example in This Folder

We use a tiny CSV dataset of integer sequences for classification:
- Inputs: `x1..x8`
- Label: `y=1` if the last two numbers are equal, else 0

Dataset: `dataset/int_sequences.csv`

## Enterprise-Level Example

In an enterprise customer-support system:
- A Transformer model routes tickets (billing, tech issue, refund)
- Extracts entities (order id, product)
- Suggests next-best actions to agents with audit logging and PII controls

