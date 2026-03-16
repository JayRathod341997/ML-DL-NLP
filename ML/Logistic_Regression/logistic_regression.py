"""
Logistic Regression - Code Example with In-Memory Data
========================================================
This example demonstrates Logistic Regression using scikit-learn
with in-memory data for predicting exam pass/fail based on study hours.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# =============================================================================
# STEP 1: Create In-Memory Data
# =============================================================================
# Hours studied - Input feature (X)
# Passed exam (1) or Failed (0) - Target variable (y)

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

print("=" * 60)
print("LOGISTIC REGRESSION - EXAM PASS/FAIL PREDICTION")
print("=" * 60)
print("\n📊 Training Data:")
print("-" * 40)
for hours, passed in zip(X.flatten(), y):
    result = "PASSED" if passed == 1 else "FAILED"
    print(f"   Studied {hours} hour(s) → {result}")

# =============================================================================
# STEP 2: Create and Train the Model
# =============================================================================

# Create Logistic Regression model
model = LogisticRegression(random_state=42)

# Train the model with our data
model.fit(X, y)

# =============================================================================
# STEP 3: Get Model Information
# =============================================================================

print("\n" + "=" * 60)
print("📈 Model Information:")
print("-" * 40)
print(f"   Coefficient: {model.coef_[0][0]:.4f}")
print(f"   Intercept: {model.intercept_[0]:.4f}")

# =============================================================================
# STEP 4: Make Predictions
# =============================================================================

# New study hours to predict
new_hours = np.array([[2.5], [4.5], [5.5], [7.5]])

# Get predictions and probabilities
predictions = model.predict(new_hours)
probabilities = model.predict_proba(new_hours)

print("\n" + "=" * 60)
print("🔮 Predictions:")
print("-" * 40)
for hours, pred, prob in zip(new_hours.flatten(), predictions, probabilities):
    result = "PASS" if pred == 1 else "FAIL"
    print(f"   Studied {hours} hours:")
    print(f"      → Prediction: {result}")
    print(f"      → Probability: Fail={prob[0]*100:.1f}%, Pass={prob[1]*100:.1f}%")

# =============================================================================
# STEP 5: Model Evaluation
# =============================================================================

# Calculate accuracy
accuracy = model.score(X, y)

print("\n" + "=" * 60)
print("📊 Model Evaluation:")
print("-" * 40)
print(f"   Training Accuracy: {accuracy*100:.1f}%")

# Get predictions on training data
train_predictions = model.predict(X)
correct = np.sum(train_predictions == y)
print(f"   Correct Predictions: {correct}/{len(y)}")

# =============================================================================
# STEP 6: Visualization
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Scatter plot with decision boundary
ax1 = axes[0]

# Plot training data
colors = ["red" if label == 0 else "green" for label in y]
ax1.scatter(X, y, c=colors, s=100, zorder=5, label="Training Data")

# Plot decision boundary
X_line = np.linspace(0, 9, 100).reshape(-1, 1)
y_prob = model.predict_proba(X_line)[:, 1]
ax1.plot(X_line, y_prob, "b-", linewidth=2, label="Sigmoid Curve")
ax1.axhline(
    y=0.5, color="gray", linestyle="--", alpha=0.7, label="Decision Boundary (50%)"
)

ax1.set_xlabel("Hours Studied", fontsize=12)
ax1.set_ylabel("Probability of Passing", fontsize=12)
ax1.set_title("Logistic Regression: Sigmoid Curve", fontsize=14)
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.1, 1.1)

# Add annotations
ax1.annotate("FAIL Zone", xy=(2, 0.2), fontsize=10, color="red")
ax1.annotate("PASS Zone", xy=(6, 0.8), fontsize=10, color="green")

# Plot 2: Bar chart of predictions with probabilities
ax2 = axes[1]

hours_labels = [f"{h} hrs" for h in new_hours.flatten()]
pass_probs = probabilities[:, 1] * 100

colors = ["green" if prob > 50 else "red" for prob in pass_probs]
bars = ax2.bar(hours_labels, pass_probs, color=colors, alpha=0.7, edgecolor="black")

ax2.axhline(y=50, color="gray", linestyle="--", linewidth=2, label="Decision Threshold")
ax2.set_xlabel("Hours Studied", fontsize=12)
ax2.set_ylabel("Probability of Passing (%)", fontsize=12)
ax2.set_title("Predictions for New Students", fontsize=14)
ax2.set_ylim(0, 100)
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")

# Add value labels on bars
for bar, prob in zip(bars, pass_probs):
    height = bar.get_height()
    ax2.annotate(
        f"{prob:.1f}%",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
    )

plt.tight_layout()
plt.savefig(
    "ML/Logistic_Regression/logistic_regression_plot.png", dpi=150, bbox_inches="tight"
)
plt.close()

print("\n" + "=" * 60)
print("✅ Visualization saved to:")
print("   ML/Logistic_Regression/logistic_regression_plot.png")
print("=" * 60)

# =============================================================================
# Additional: Probability Calculation
# =============================================================================

print("\n" + "=" * 60)
print("🔢 Understanding the Sigmoid:")
print("-" * 40)
print("   The sigmoid formula: P = 1 / (1 + e^-(mx + b))")
print(
    f"   For 5.5 hours: P = 1 / (1 + e^-({model.coef_[0][0]:.2f}*5.5 + {model.intercept_[0]:.2f}))"
)
print(f"   = {probabilities[2][1]*100:.1f}% chance of passing")
print("=" * 60)
