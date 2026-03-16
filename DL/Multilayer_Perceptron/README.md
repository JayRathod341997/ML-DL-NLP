# Multilayer Perceptron (MLP) - Explain Like I'm 5

## What is an MLP?

Imagine you have a group of tiny robots (neurons).  
One robot asks another robot for help, and together they decide the answer.

That “teamwork” is a **hidden layer**.

An **MLP** is a neural network with:
- an **input layer**
- one or more **hidden layers**
- an **output layer**

Hidden layers use **non-linear activations** (like ReLU) so the network can learn complex patterns.

## Why It Matters

A single perceptron can only draw a straight line.  
An MLP can learn **curves** and solve problems like **XOR**.

## Visualization (Tiny Network)

```
 x1   x2
  \   /
  [hidden neurons]
      |
   output
```

## Where It’s Used

- Tabular classification / regression
- Recommendation features + embeddings (small MLP heads)
- Credit risk scoring, churn prediction, demand forecasting

## Benefits

- Learns non-linear relationships
- Flexible: more layers/neurons can increase capacity

## Limitations

- Needs careful tuning (learning rate, depth, regularization)
- Can overfit on small data
- For images/text, CNNs/Transformers are usually better than a plain MLP

## Example in This Folder

- Dataset: `dataset/xor.csv`
- Code:
  - `codes/01_in_memory_example.py` (XOR, generated in-memory)
  - `codes/02_csv_example.py` (XOR, from CSV)

## Enterprise-Level Example

In an enterprise fraud system, an MLP is often used as the “last-mile” model:
- Inputs: engineered features + learned embeddings (merchant, device, user)
- Output: fraud risk score
- Deployment: real-time scoring service with monitoring (drift, calibration, latency)

