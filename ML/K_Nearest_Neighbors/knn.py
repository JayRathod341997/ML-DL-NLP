"""
K-Nearest Neighbors (KNN) - Code Example with In-Memory Data
===============================================================
This example demonstrates KNN using scikit-learn
with in-memory data for Iris flower classification.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# =============================================================================
# STEP 1: Create In-Memory Data (Iris-like Dataset)
# =============================================================================

# Features: Sepal Length, Sepal Width
# Classes: 0=Setosa, 1=Versicolor, 2=Virginica

X = np.array(
    [
        [5.1, 3.5],  # Setosa
        [4.9, 3.0],  # Setosa
        [4.7, 3.2],  # Setosa
        [4.6, 3.1],  # Setosa
        [5.0, 3.6],  # Setosa
        [7.0, 3.2],  # Versicolor
        [6.4, 3.2],  # Versicolor
        [6.9, 3.1],  # Versicolor
        [5.5, 2.3],  # Versicolor
        [6.5, 2.8],  # Versicolor
        [6.3, 3.3],  # Virginica
        [6.7, 3.0],  # Virginica
        [6.0, 2.7],  # Virginica
        [6.7, 3.1],  # Virginica
        [5.8, 2.7],  # Virginica
    ]
)

# Labels
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

class_names = ["Setosa", "Versicolor", "Virginica"]
feature_names = ["Sepal Length", "Sepal Width"]

print("=" * 60)
print("K-NEAREST NEIGHBORS (KNN) - IRIS FLOWER CLASSIFICATION")
print("=" * 60)
print("\n📊 Training Data:")
print("-" * 40)

for i, (features, label) in enumerate(zip(X, y)):
    print(
        f"   {i+1}. Length: {features[0]}, Width: {features[1]} → {class_names[label]}"
    )

# =============================================================================
# STEP 2: Create and Train KNN Model
# =============================================================================

# Create KNN with K=3 neighbors
k = 3
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X, y)

print("\n" + "=" * 60)
print(f"🌿 KNN Model (K={k}):")
print("-" * 40)
print(f"   Number of Neighbors (K): {k}")
print(f"   Distance Metric: {model.metric}")
print(f"   Training Accuracy: {model.score(X, y)*100:.1f}%")

# =============================================================================
# STEP 3: Make Predictions
# =============================================================================

# Mystery flowers to classify
new_flowers = np.array(
    [
        [5.5, 2.6],  # Mystery 1
        [6.2, 2.9],  # Mystery 2
        [4.8, 3.0],  # Mystery 3
    ]
)

predictions = model.predict(new_flowers)
probabilities = model.predict_proba(new_flowers)
distances, indices = model.kneighbors(new_flowers)

print("\n" + "=" * 60)
print("🔮 Predictions for New Flowers:")
print("-" * 40)

for i, (flower, pred, prob) in enumerate(zip(new_flowers, predictions, probabilities)):
    print(f"\n   Flower {i+1}: Sepal Length={flower[0]}, Width={flower[1]}")
    print(f"      → Prediction: {class_names[pred]}")
    print(f"      → Distances to {k} neighbors: {distances[i]}")
    print(f"      → Neighbor classes: {[class_names[y[idx]] for idx in indices[i]]}")
    print(f"      → Probabilities:")
    for j, class_name in enumerate(class_names):
        print(f"         {class_name}: {prob[j]*100:.1f}%")

# =============================================================================
# STEP 4: Show Voting Process
# =============================================================================

print("\n" + "=" * 60)
print("🗳️  Voting Process (K=3):")
print("-" * 40)

for i, (flower, pred) in enumerate(zip(new_flowers, predictions)):
    print(f"\n   Flower {i+1} voting:")
    vote_counts = {0: 0, 1: 0, 2: 0}
    for idx in indices[i]:
        label = y[idx]
        vote_counts[label] += 1
        print(f"      - Neighbor: ({X[idx][0]}, {X[idx][1]}) = {class_names[label]}")

    print(
        f"   Votes: {class_names[0]}={vote_counts[0]}, {class_names[1]}={vote_counts[1]}, {class_names[2]}={vote_counts[2]}"
    )
    print(f"   Winner: {class_names[pred]}")

# =============================================================================
# STEP 5: Visualization
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Scatter plot with neighbors
ax1 = axes[0]

colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
markers = ["o", "s", "^"]

# Plot training data
for class_idx, (class_name, color, marker) in enumerate(
    zip(class_names, colors, markers)
):
    mask = y == class_idx
    ax1.scatter(
        X[mask, 0],
        X[mask, 1],
        c=color,
        marker=marker,
        s=100,
        label=class_name,
        alpha=0.7,
        edgecolors="black",
    )

# Plot new flowers
for i, flower in enumerate(new_flowers):
    ax1.scatter(
        flower[0],
        flower[1],
        c="yellow",
        marker="*",
        s=300,
        edgecolors="black",
        linewidths=2,
        zorder=5,
    )
    ax1.annotate(
        f"Mystery {i+1}",
        (flower[0], flower[1]),
        textcoords="offset points",
        xytext=(5, 5),
    )

# Draw lines to neighbors for first mystery flower
for idx in indices[0]:
    ax1.plot(
        [new_flowers[0][0], X[idx][0]],
        [new_flowers[0][1], X[idx][1]],
        "k--",
        alpha=0.3,
        linewidth=1,
    )

ax1.set_xlabel("Sepal Length", fontsize=12)
ax1.set_ylabel("Sepal Width", fontsize=12)
ax1.set_title("KNN Classification (K=3)", fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Decision boundaries visualization
ax2 = axes[1]

# Create mesh grid for decision boundary
h = 0.1  # step size
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict on mesh
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
ax2.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
ax2.contour(xx, yy, Z, colors="black", linewidths=0.5, alpha=0.5)

# Plot data points
for class_idx, (class_name, color, marker) in enumerate(
    zip(class_names, colors, markers)
):
    mask = y == class_idx
    ax2.scatter(
        X[mask, 0],
        X[mask, 1],
        c=color,
        marker=marker,
        s=100,
        label=class_name,
        edgecolors="black",
    )

# Plot new flowers
for i, flower in enumerate(new_flowers):
    ax2.scatter(
        flower[0],
        flower[1],
        c="yellow",
        marker="*",
        s=300,
        edgecolors="black",
        linewidths=2,
        zorder=5,
    )

ax2.set_xlabel("Sepal Length", fontsize=12)
ax2.set_ylabel("Sepal Width", fontsize=12)
ax2.set_title("KNN Decision Boundaries", fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ML/K_Nearest_Neighbors/knn_plot.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n" + "=" * 60)
print("✅ Visualization saved to:")
print("   ML/K_Nearest_Neighbors/knn_plot.png")
print("=" * 60)
