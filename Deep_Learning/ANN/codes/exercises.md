# Artificial Neural Networks - Exercises

## Exercise 1: Binary Classification with Custom Data

**Objective**: Create a neural network to classify whether a student will pass or fail based on study hours and previous test scores.

**Dataset**: Create a synthetic dataset with:
- Study hours (1-10 hours)
- Previous test score (0-100)
- Pass/Fail label (1 for pass, 0 for fail)

**Requirements**:
1. Generate 1000 data points with some correlation between study hours, previous scores, and pass/fail outcome
2. Implement a neural network with 1 hidden layer (5 neurons)
3. Use sigmoid activation for hidden and output layers
4. Train for 1000 epochs
5. Achieve at least 85% accuracy on test set

**Expected Outcome**: A working ANN that can predict student performance.

---

## Exercise 2: Multi-class Classification

**Objective**: Classify iris flowers into 3 species using the famous Iris dataset.

**Dataset**: Use sklearn's load_iris() dataset

**Requirements**:
1. Load the Iris dataset
2. Implement a neural network with 2 hidden layers (10 and 5 neurons respectively)
3. Use ReLU activation for hidden layers and softmax for output
4. Handle multi-class classification (3 classes)
5. Train for 2000 epochs
6. Achieve at least 90% accuracy

**Expected Outcome**: A neural network that can classify iris species with high accuracy.

---

## Exercise 3: Regression Task

**Objective**: Predict house prices based on features like size, number of bedrooms, and location score.

**Dataset**: Create synthetic housing data:
- House size (500-5000 sq ft)
- Number of bedrooms (1-6)
- Location score (1-10)
- Price (correlated with features)

**Requirements**:
1. Generate 2000 data points
2. Modify the ANN for regression (linear output activation)
3. Use MSE loss instead of cross-entropy
4. Implement proper scaling for the output
5. Train until convergence
6. Achieve R² score > 0.85

**Expected Outcome**: A regression ANN that can predict house prices.

---

## Exercise 4: XOR Problem

**Objective**: Solve the classic XOR problem that cannot be solved with a single-layer perceptron.

**Dataset**: XOR truth table
- (0,0) → 0
- (0,1) → 1
- (1,0) → 1
- (1,1) → 0

**Requirements**:
1. Create the XOR dataset (duplicate each combination 100 times for training)
2. Implement a neural network with at least 2 hidden neurons
3. Use sigmoid or tanh activation
4. Train until the network correctly classifies all XOR inputs
5. Visualize the decision boundary

**Expected Outcome**: A neural network that successfully solves the XOR problem.

---

## Exercise 5: Handwritten Digit Recognition

**Objective**: Classify handwritten digits (0-9) from the MNIST-like dataset.

**Dataset**: Use sklearn's load_digits() dataset

**Requirements**:
1. Load the digits dataset (1797 samples, 64 features)
2. Implement a neural network with 2 hidden layers (128 and 64 neurons)
3. Use ReLU activation for hidden layers and softmax for output
4. Add dropout (0.2) to prevent overfitting
5. Train for 5000 epochs with early stopping
6. Achieve at least 95% accuracy

**Expected Outcome**: A neural network that can recognize handwritten digits.

---

## Exercise 6: Custom Activation Function

**Objective**: Implement and test a custom activation function.

**Requirements**:
1. Implement the Swish activation function: f(x) = x * sigmoid(x)
2. Implement its derivative
3. Replace ReLU with Swish in a binary classification task
4. Compare performance with ReLU and sigmoid
5. Analyze convergence speed and final accuracy

**Expected Outcome**: Understanding of how different activation functions affect training.

---

## Exercise 7: Weight Initialization Comparison

**Objective**: Compare different weight initialization methods.

**Requirements**:
1. Implement three initialization methods:
   - Random initialization (normal distribution)
   - Xavier/Glorot initialization
   - He initialization
2. Train the same network architecture with each method
3. Compare convergence speed and final performance
4. Visualize loss curves for each method

**Expected Outcome**: Understanding of how weight initialization affects training.

---

## Exercise 8: Learning Rate Scheduling

**Objective**: Implement and compare different learning rate schedules.

**Requirements**:
1. Implement three learning rate schedules:
   - Constant learning rate
   - Step decay (reduce by half every 1000 epochs)
   - Exponential decay
2. Train the same network with each schedule
3. Compare training dynamics and final performance
4. Visualize learning rate and loss over time

**Expected Outcome**: Understanding of how learning rate scheduling affects training.

---

## Exercise 9: Regularization Techniques

**Objective**: Implement and compare regularization methods.

**Requirements**:
1. Implement L1 and L2 regularization
2. Implement dropout
3. Train a network prone to overfitting with each technique
4. Compare training vs validation performance
5. Analyze which technique works best for the given problem

**Expected Outcome**: Understanding of regularization techniques and their effects.

---

## Exercise 10: Advanced Architecture

**Objective**: Build a more sophisticated neural network.

**Requirements**:
1. Implement batch normalization
2. Add momentum to gradient descent
3. Use Adam optimizer instead of basic gradient descent
4. Compare performance with the basic implementation
5. Analyze training speed and stability improvements

**Expected Outcome**: A more robust and efficient neural network implementation.

---

## Bonus Exercise: Real-world Dataset

**Objective**: Apply ANN to a real-world dataset of your choice.

**Requirements**:
1. Choose a dataset from Kaggle or UCI Machine Learning Repository
2. Perform exploratory data analysis
3. Preprocess the data appropriately
4. Implement and train an ANN
5. Evaluate performance using appropriate metrics
6. Document your findings and challenges

**Expected Outcome**: Experience applying ANN to real-world problems.

---

## Submission Guidelines

For each exercise:

1. **Code**: Well-commented, readable implementation
2. **Results**: Accuracy, loss curves, confusion matrices as appropriate
3. **Analysis**: Brief explanation of what you learned
4. **Challenges**: Any difficulties encountered and how you solved them

## Evaluation Criteria

- **Correctness**: Does the implementation work as expected?
- **Performance**: Does it achieve the required accuracy/score?
- **Code Quality**: Is the code well-structured and documented?
- **Analysis**: Are the results properly analyzed and explained?
- **Creativity**: Any innovative approaches or insights?

Good luck with your exercises! Remember to experiment, make mistakes, and learn from them.