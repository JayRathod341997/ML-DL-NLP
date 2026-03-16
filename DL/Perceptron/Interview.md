# Perceptron - Interview Questions (with Answers)

## Basic

### Q1: What is a perceptron?
**Answer:** A perceptron is a single-layer binary classifier that computes a weighted sum of inputs plus bias and applies a step function to output 0/1.

### Q2: What is the perceptron learning rule?
**Answer:** Update weights based on prediction error: `w = w + lr*(y - y_hat)*x`, and similarly update bias.

### Q3: What activation does a perceptron use?
**Answer:** Traditionally a step (Heaviside) function for hard 0/1 outputs.

## Intermediate

### Q4: What does “linearly separable” mean?
**Answer:** There exists a hyperplane (a line in 2D) that separates the classes perfectly.

### Q5: Why can’t a single perceptron solve XOR?
**Answer:** XOR is not linearly separable; it requires a non-linear decision boundary, which needs at least one hidden layer with non-linear activation.

### Q6: How is a perceptron related to logistic regression?
**Answer:** Both are linear models. Logistic regression uses a sigmoid and outputs probabilities, trained via likelihood/gradient methods; perceptron uses a step function and a mistake-driven update rule.

## Advanced

### Q7: Does the perceptron always converge?
**Answer:** It converges if the data is linearly separable and the learning rate is appropriate (Perceptron Convergence Theorem). If not separable, it may not converge.

### Q8: What’s the role of bias?
**Answer:** Bias shifts the decision boundary; without it, the boundary must pass through the origin.

