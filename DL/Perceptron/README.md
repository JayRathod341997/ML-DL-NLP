# Perceptron - Explain Like I'm 5

## What is a Perceptron?

Imagine a toy robot that must answer **YES** or **NO**.

You give it a few simple clues (numbers), and it:
1) multiplies each clue by an importance score (a **weight**),
2) adds them up,
3) if the total is big enough, it says **YES**; otherwise **NO**.

That’s a **Perceptron**: the simplest “neural network” (a single neuron).

## The Idea (Simple Math)

For inputs `x1, x2, ...` and weights `w1, w2, ...`:

```
score = w·x + b
prediction = 1 if score >= 0 else 0
```

`b` is a **bias** (a little push).

## How It Learns (Training)

If it predicts wrong, it nudges weights:

```
w = w + lr * (y - y_hat) * x
b = b + lr * (y - y_hat)
```

Where:
- `lr` = learning rate (small step size)
- `y` = true label, `y_hat` = predicted label

## Where It’s Used

- As a **building block** to understand modern deep learning
- Very simple **binary classification** problems that are **linearly separable**

## Benefit

- Easy to understand and implement
- Fast to train on small data

## Limitation (Important)

- Can only solve **linearly separable** problems.
- Example it cannot solve: **XOR** (needs at least a hidden layer / MLP).

## Visualization (Decision Boundary)

For 2 features, a perceptron learns a straight line:

```
Class 1   o o o
          o o
------------------  <- a line (hyperplane)
Class 0   x x x x
        x x x
```

## Example in This Folder

- Dataset: `dataset/and_gate.csv` (AND logic gate)
- Code:
  - `codes/01_in_memory_example.py` (AND, in-memory)
  - `codes/02_csv_example.py` (AND, from CSV)

## Enterprise-Level Example

In a real enterprise system, a perceptron by itself is usually too limited, but the same concept (a linear layer + bias) appears everywhere:
- **Ad click prediction** pipelines often start with linear models as baselines
- **Fraud screening** uses linear scoring layers as interpretable components
- Deep networks use many layers, and the perceptron is the “atom” of those layers

