# Multilayer Perceptron (MLP) - Interview Questions (with Answers)

## Basic

### Q1: What makes an MLP different from a perceptron?
**Answer:** An MLP has one or more hidden layers with non-linear activations, allowing it to learn non-linear decision boundaries.

### Q2: Why do we need activation functions?
**Answer:** Without non-linearity, stacked linear layers collapse into a single linear transformation, limiting the model to linear relationships.

## Intermediate

### Q3: What is backpropagation?
**Answer:** A method to compute gradients of the loss w.r.t. parameters using the chain rule, enabling gradient-based optimization.

### Q4: What are common activation functions?
**Answer:** ReLU, LeakyReLU, GELU, Tanh, Sigmoid (Sigmoid/Tanh often used less in deep hidden layers due to saturation).

### Q5: What causes vanishing gradients?
**Answer:** Gradients shrink as they propagate backward, often due to saturating activations (sigmoid/tanh) and deep networks.

## Advanced

### Q6: How do you reduce overfitting in an MLP?
**Answer:** Regularization (L2/weight decay), dropout, early stopping, more data, and proper validation.

### Q7: How do you choose the number of hidden layers/neurons?
**Answer:** Start simple, then increase capacity based on validation performance; use hyperparameter search and monitor overfitting.

