# RNN / LSTM / GRU - Explain Like I'm 5

## What are RNNs?

Imagine you’re reading a story word by word.
You remember what happened earlier, so the next word makes sense.

An **RNN (Recurrent Neural Network)** is a model that processes a sequence **one step at a time**, carrying a “memory” (hidden state).

## The Problem with Vanilla RNNs

For long sequences, vanilla RNNs can forget important early information due to **vanishing gradients**.

## LSTM and GRU (Better Memory)

LSTM and GRU add “gates” that decide:
- what to remember
- what to forget

This helps with longer dependencies.

## Where They’re Used

- Time series forecasting (demand, sensors)
- Sequence classification (anomaly detection on logs)
- Older NLP pipelines (now often replaced by Transformers)

## Benefits

- Handles ordered data (sequence matters)
- Works on variable-length sequences

## Limitations

- Hard to parallelize (step-by-step)
- Long-range modeling is usually better with Transformers

## Example in This Folder

We use a tiny CSV dataset of integer sequences:
- Input: `x1..x5`
- Label: `y = 1` if the sequence sum is >= 3, else `0`

Files:
- Dataset: `dataset/sequence_sum.csv`
- Code:
  - `codes/01_in_memory_example.py` (generate sequences in-memory)
  - `codes/02_csv_example.py` (train from CSV)

## Enterprise-Level Example

In predictive maintenance:
- Input: sensor readings over time for a machine
- Model: LSTM detects early failure patterns
- Output: probability of failure in the next N days
- Actions: schedule maintenance, reduce downtime, optimize spare parts inventory

