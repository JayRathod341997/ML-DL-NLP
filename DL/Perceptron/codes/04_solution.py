"""
Solution: Perceptron on OR gate
"""

import numpy as np


def perceptron_train(X, y, lr=0.1, epochs=20):
    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0

    for _ in range(epochs):
        for xi, yi in zip(X, y):
            score = float(np.dot(w, xi) + b)
            y_hat = 1 if score >= 0 else 0
            err = yi - y_hat
            w += lr * err * xi
            b += lr * err

    return w, b


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 1, 1, 1], dtype=int)

w, b = perceptron_train(X, y, lr=0.2, epochs=25)
pred = ((X @ w + b) >= 0).astype(int)

print("Weights:", w)
print("Bias:", b)
print("Predictions:", pred.tolist())
print("Targets:", y.tolist())

