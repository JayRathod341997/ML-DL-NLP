"""
Support Vector Machine (SVM) - Code Example with In-Memory Data
================================================================
This example demonstrates SVM using scikit-learn
with in-memory data for Iris flower classification.
"""

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# =============================================================================
# STEP 1: Create In-Memory Data (Iris Dataset - 2 features)
# =============================================================================

# Using only petal length and petal width for visualization
# Features: Petal Length, Petal Width

X = np.array(
    [
        # Setosa (class 0)
        [1.4, 0.2],
        [1.4, 0.2],
        [1.3, 0.2],
        [1.5, 0.2],
        [1.4, 0.3],
        [1.7, 0.4],
        [1.4, 0.3],
        [1.5, 0.2],
        [1.4, 0.2],
        [1.5, 0.1],
        # Versicolor (class 1)
        [4.7, 1.4],
        [4.5, 1.5],
        [4.9, 1.5],
        [4.0, 1.3],
        [4.6, 1.4],
        [4.5, 1.5],
        [4.7, 1.6],
        [3.3, 1.0],
        [4.6, 1.3],
        [3.9, 1.4],
        # Virginica (class 2)
        [6.0, 2.5],
        [5.1, 1.9],
        [5.9, 2.1],
        [5.6, 1.8],
        [5.8, 2.2],
        [6.6, 2.1],
        [4.5, 1.7],
        [6.0, 2.5],
        [5.2, 2.3],
        [5.0, 1.5],
    ]
)

# Labels
y = np.array([0] * 10 + [1] * 10 + [2] * 10)

class_names = ["Setosa", "Versicolor", "Virginica"]
feature_names = ["Petal Length", "Petal Width"]

print("=" * 60)
print("SUPPORT VECTOR MACHINE (SVM) - IRIS CLASSIFICATION")
print("=" * 60)
print("\n📊 Training Data (First 3 samples per class):")
print("-" * 50)

for class_idx in range(3):
    print(f"\n   {class_names[class_idx]}:")
    class_data = X[y == class_idx][:3]
    for i, row in enumerate(class_data):
        print(f"      {i+1}. Length: {row[0]}, Width: {row[1]}")

# =============================================================================
# STEP 2: Train SVM Models
# =============================================================================

# Linear SVM
svm_linear = SVC(kernel="linear", random_state=42)
svm_linear.fit(X, y)

# RBF (Radial Basis Function) SVM
svm_rbf = SVC(kernel="rbf", random_state=42)
svm_rbf.fit(X, y)

print("\n" + "=" * 60)
print("📈 Model Comparison:")
print("-" * 40)
print(f"   Linear SVM Training Accuracy: {svm_linear.score(X, y)*100:.1f}%")
print(f"   RBF SVM Training Accuracy: {svm_rbf.score(X, y)*100:.1f}%")

# =============================================================================
# STEP 3: Make Predictions
# =============================================================================

new_flowers = np.array(
    [
        [1.8, 0.5],  # New flower 1
        [4.5, 1.5],  # New flower 2
        [5.5, 2.0],  # New flower 3
    ]
)

pred_linear = svm_linear.predict(new_flowers)
pred_rbf = svm_rbf.predict(new_flowers)

print("\n" + "=" * 60)
print("🔮 Predictions:")
print("-" * 40)

for i, (flower, pred_l, pred_r) in enumerate(zip(new_flowers, pred_linear, pred_rbf)):
    print(f"\n   Flower {i+1}: Length={flower[0]}, Width={flower[1]}")
    print(f"      Linear SVM: {class_names[pred_l]}")
    print(f"      RBF SVM: {class_names[pred_r]}")

# =============================================================================
# STEP 4: Support Vectors
# =============================================================================

print("\n" + "=" * 60)
print("🎯 Support Vectors:")
print("-" * 40)

n_support_linear = svm_linear.n_support_
n_support_rbf = svm_rbf.n_support_

print(f"   Linear SVM: {n_support_linear} support vectors per class")
print(f"   RBF SVM: {n_support_rbf} support vectors per class")

print("\n   Support Vectors (RBF SVM):")
for class_idx in range(3):
    sv = svm_rbf.support_vectors_[np.where(svm_rbf.support_ == class_idx)[0][:3]]
    print(
        f"      {class_names[class_idx]}: {len(svm_rbf.support_vectors_[y == class_idx])} vectors"
    )

# =============================================================================
# STEP 5: Visualization
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Linear SVM
ax1 = axes[0]

colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

# Create mesh grid
h = 0.1
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 0].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict on mesh
Z = svm_linear.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
ax1.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
ax1.contour(xx, yy, Z, colors="black", linewidths=1, alpha=0.5)

# Plot data points
for class_idx in range(3):
    mask = y == class_idx
    ax1.scatter(
        X[mask, 0],
        X[mask, 1],
        c=colors[class_idx],
        s=80,
        label=class_names[class_idx],
        alpha=0.7,
        edgecolors="black",
    )

# Plot support vectors
ax1.scatter(
    svm_linear.support_vectors_[:, 0],
    svm_linear.support_vectors_[:, 1],
    s=200,
    facecolors="none",
    edgecolors="black",
    linewidths=2,
    label="Support Vectors",
)

ax1.set_xlabel("Petal Length", fontsize=12)
ax1.set_ylabel("Petal Width", fontsize=12)
ax1.set_title("Linear SVM", fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: RBF SVM
ax2 = axes[1]

# Predict on mesh
Z = svm_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
ax2.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
ax2.contour(xx, yy, Z, colors="black", linewidths=1, alpha=0.5)

# Plot data points
for class_idx in range(3):
    mask = y == class_idx
    ax2.scatter(
        X[mask, 0],
        X[mask, 1],
        c=colors[class_idx],
        s=80,
        label=class_names[class_idx],
        alpha=0.7,
        edgecolors="black",
    )

# Plot support vectors
ax2.scatter(
    svm_rbf.support_vectors_[:, 0],
    svm_rbf.support_vectors_[:, 1],
    s=200,
    facecolors="none",
    edgecolors="black",
    linewidths=2,
    label="Support Vectors",
)

ax2.set_xlabel("Petal Length", fontsize=12)
ax2.set_ylabel("Petal Width", fontsize=12)
ax2.set_title("RBF SVM (Non-linear)", fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ML/Support_Vector_Machine/svm_plot.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n" + "=" * 60)
print("✅ Visualization saved to:")
print("   ML/Support_Vector_Machine/svm_plot.png")
print("=" * 60)

# =============================================================================
# Kernel Comparison
# =============================================================================

print("\n" + "=" * 60)
print("🔄 Kernel Comparison:")
print("-" * 40)
print("   Linear: Best when data is linearly separable")
print("   RBF: Best for complex, non-linear boundaries")
print("   Poly: Uses polynomial curves")
print("=" * 60)
