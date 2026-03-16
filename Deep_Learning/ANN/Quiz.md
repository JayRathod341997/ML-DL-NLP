# Artificial Neural Networks - Quiz

## True/False Questions

### Question 1
**Statement:** A neural network with a single neuron (perceptron) can solve the XOR problem.

**Answer:** False

**Explanation:** The perceptron is a linear classifier. The XOR problem is not linearly separable—there's no straight line that can separate the two classes (0 and 1) in XOR's input space. This limitation led to the development of multi-layer perceptrons (MLPs) with hidden layers that can learn non-linear decision boundaries.

---

### Question 2
**Statement:** Increasing the number of hidden layers always improves model performance.

**Answer:** False

**Explanation:** More layers don't guarantee better performance. Very deep networks can suffer from vanishing gradients, are harder to optimize, and may overfit if the dataset is small. The optimal architecture depends on problem complexity, dataset size, and available compute. Sometimes 2-3 layers suffice, while other tasks require dozens of layers.

---

### Question 3
**Statement:** ReLU activation can cause "dying neurons" during training.

**Answer:** True

**Explanation:** When a neuron's weighted sum is negative, ReLU outputs 0, and the gradient is also 0. During backpropagation, this neuron stops learning and can permanently "die" (always output 0). This happens especially with high learning rates or poor initialization. Leaky ReLU and ELU variants address this issue.

---

### Question 4
**Statement:** Batch normalization eliminates the need for dropout in deep networks.

**Answer:** False

**Explanation:** Batch normalization (BatchNorm) and dropout serve different purposes. BatchNorm normalizes layer inputs, stabilizes training, and provides slight regularization. Dropout randomly deactivates neurons during training, forcing redundant representations. They can be used together and are complementary—BatchNorm doesn't replace dropout.

---

### Question 5
**Statement:** A neural network trained with stochastic gradient descent (SGD) will always converge to the same solution regardless of initialization.

**Answer:** False

**Explanation:** Neural network loss surfaces are non-convex with many local minima and saddle points. Different initializations can lead to different final solutions with varying performance. This is why multiple runs with different seeds or techniques like ensemble methods are used in practice.

---

## Multiple Choice Questions

### Question 6
What is the mathematical operation performed by a single neuron in a neural network?

A) Output = Activation(Input + Bias)  
B) Output = Activation(Weight × Input + Bias)  
C) Output = Weight × Input + Bias  
D) Output = Activation(Weight + Input)

**Answer:** B

**Explanation:** A single neuron computes a weighted sum of inputs (Weight × Input), adds a bias term, then applies an activation function. This is the fundamental building block of all neural networks. Mathematically: y = σ(W·x + b)

---

### Question 7
Which of the following is NOT a method to prevent overfitting in neural networks?

A) Dropout  
B) L2 Regularization  
C) Increasing batch size  
D) Data augmentation

**Answer:** C

**Explanation:** Increasing batch size helps with training stability and computation efficiency but doesn't directly prevent overfitting. In fact, very large batches can sometimes lead to worse generalization. Dropout, L2 regularization, and data augmentation are established techniques to reduce overfitting.

---

### Question 8
What does a learning rate of 0.001 represent in gradient descent?

A) The number of training epochs  
B) The step size for weight updates  
C) The number of hidden layers  
D) The batch size for training

**Answer:** B

**Explanation:** Learning rate (η = 0.001) controls the step size during gradient descent. The weight update rule is: w_new = w_old - η × ∂L/∂w. A smaller learning rate means smaller steps, more iterations needed, but potentially more stable convergence.

---

### Question 9
Which optimizer automatically adapts learning rates for each parameter?

A) SGD  
B) Mini-batch GD  
C) Adam  
D) Batch GD

**Answer:** C

**Explanation:** Adam (Adaptive Moment Estimation) maintains per-parameter learning rates based on first and second moment estimates of gradients. It adapts the learning rate during training based on how frequently each parameter is updated. SGD uses a fixed learning rate for all parameters.

---

### Question 10
What is the purpose of the softmax function in the output layer?

A) To introduce non-linearity in hidden layers  
B) To output class probabilities that sum to 1  
C) To reduce gradient vanishing  
D) To speed up computation

**Answer:** B

**Explanation:** Softmax converts raw logits into a probability distribution over classes. For each class i: softmax(x_i) = e^(x_i) / Σ e^(x_j). The output values sum to 1, making it ideal for multi-class classification. Sigmoid is used for binary classification; softmax is for multi-class.

---

## Answer Key

| Question | Type | Answer | Key Concept |
|----------|------|--------|--------------|
| Q1 | T/F | False | Perceptron limitation on XOR |
| Q2 | T/F | False | Depth vs performance trade-off |
| Q3 | T/F | True | Dying ReLU problem |
| Q4 | T/F | False | BatchNorm vs Dropout |
| Q5 | T/F | False | Non-convex loss surfaces |
| Q6 | MCQ | B | Neuron computation |
| Q7 | MCQ | C | Overfitting prevention methods |
| Q8 | MCQ | B | Learning rate definition |
| Q9 | MCQ | C | Adam optimizer |
| Q10 | MCQ | B | Softmax function |

## Brief Explanations

1. **Q1 (False):** Perceptrons are linear classifiers; XOR requires non-linear boundaries.
2. **Q2 (False):** Too many layers cause optimization difficulties and overfitting.
3. **Q3 (True):** ReLU can output zero for negative inputs, stopping gradient flow.
4. **Q4 (False):** BatchNorm and dropout are complementary regularization techniques.
5. **Q5 (False):** Different initializations can lead to different local minima.
6. **Q6 (B):** y = σ(W·x + b) is the neuron formula.
7. **Q7 (C):** Batch size affects training dynamics, not regularization directly.
8. **Q8 (B):** Learning rate determines gradient descent step size.
9. **Q9 (C):** Adam uses adaptive learning rates per parameter.
10. **Q10 (B):** Softmax produces valid probability distributions.
