"""
Perceptron - CSV Example (AND gate)
Reads: ../dataset/and_gate.csv
"""

from pathlib import Path

import numpy as np

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise SystemExit("Missing dependency: pandas. Install with `pip install pandas`.") from e


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


here = Path(__file__).resolve().parent
csv_path = (here / ".." / "dataset" / "and_gate.csv").resolve()
df = pd.read_csv(csv_path)

X = df[["x1", "x2"]].to_numpy(dtype=float)
y = df["y"].to_numpy(dtype=int)

w, b = perceptron_train(X, y, lr=0.2, epochs=25)
pred = ((X @ w + b) >= 0).astype(int)

print("Loaded:", csv_path)
print("Weights:", w)
print("Bias:", b)
print("Predictions:", pred.tolist())
print("Targets:", y.tolist())

