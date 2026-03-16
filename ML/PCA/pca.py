"""
Principal Component Analysis (PCA) - Code Example with In-Memory Data
===================================================================
This example demonstrates PCA using scikit-learn
with in-memory data for dimension reduction.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# =============================================================================
# STEP 1: Create In-Memory Data (Student Scores)
# =============================================================================

# Features: Math, Science, English, History scores

X = np.array(
    [
        [90, 85, 70, 75],  # Student 1
        [85, 80, 65, 70],  # Student 2
        [95, 90, 75, 80],  # Student 3
        [70, 65, 90, 85],  # Student 4
        [65, 60, 85, 80],  # Student 5
        [75, 70, 95, 90],  # Student 6
        [80, 75, 88, 82],  # Student 7
        [88, 82, 72, 78],  # Student 8
        [78, 73, 80, 76],  # Student 9
        [82, 77, 85, 80],  # Student 10
    ]
)

feature_names = ["Math", "Science", "English", "History"]
student_labels = [f"Student {i+1}" for i in range(10)]

print("=" * 60)
print("PRINCIPAL COMPONENT ANALYSIS (PCA) - STUDENT SCORES")
print("=" * 60)
print("\n📊 Original Data (4 Features):")
print("-" * 60)
print("   Student | Math | Science | English | History")
print("   " + "-" * 55)
for i, row in enumerate(X):
    print(
        f"      {i+1}    | {row[0]:3d}  |   {row[1]:3d}   |   {row[2]:3d}   |   {row[3]:3d}"
    )

# =============================================================================
# STEP 2: Standardize Data (Important for PCA!)
# =============================================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n" + "=" * 60)
print("📏 Standardized Data (Mean=0, Std=1):")
print("-" * 60)
print("   Student | Math    | Science | English | History")
print("   " + "-" * 55)
for i, row in enumerate(X_scaled):
    print(
        f"      {i+1}    | {row[0]:6.2f} | {row[1]:6.2f} | {row[2]:6.2f} | {row[3]:6.2f}"
    )

# =============================================================================
# STEP 3: Apply PCA
# =============================================================================

# Apply PCA to reduce from 4 to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\n" + "=" * 60)
print("🎯 PCA Results (4D → 2D):")
print("-" * 40)
print(f"   Original dimensions: {X.shape[1]}")
print(f"   Reduced dimensions: {X_pca.shape[1]}")

# Explained variance
explained_var = pca.explained_variance_ratio_
print(f"\n   Explained Variance:")
print(f"      PC1: {explained_var[0]*100:.1f}%")
print(f"      PC2: {explained_var[1]*100:.1f}%")
print(f"      Total: {sum(explained_var)*100:.1f}%")

print("\n" + "=" * 60)
print("📊 Transformed Data (2 Principal Components):")
print("-" * 40)
print("   Student |    PC1    |    PC2")
print("   " + "-" * 35)
for i, row in enumerate(X_pca):
    print(f"      {i+1}    | {row[0]:8.2f} | {row[1]:8.2f}")

# =============================================================================
# STEP 4: Component Loadings (What does each PC mean?)
# =============================================================================

print("\n" + "=" * 60)
print("🔍 Principal Component Loadings:")
print("-" * 40)
print("   Feature   |    PC1    |    PC2")
print("   " + "-" * 35)

loadings = pca.components_
for feat_idx, feat_name in enumerate(feature_names):
    print(
        f"   {feat_name:8s} | {loadings[0][feat_idx]:8.3f} | {loadings[1][feat_idx]:8.3f}"
    )

print("\n   Interpretation:")
print("      PC1 = Overall Academic Performance")
print("      PC2 = STEM vs Arts Direction")

# =============================================================================
# STEP 5: Full PCA (All Components)
# =============================================================================

pca_full = PCA()
pca_full.fit(X_scaled)

print("\n" + "=" * 60)
print("📈 Explained Variance by Component:")
print("-" * 40)

cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
for i, (var, cum_var) in enumerate(
    zip(pca_full.explained_variance_ratio_, cumulative_var)
):
    bar = "█" * int(var * 30)
    print(f"   PC{i+1}: {var*100:5.1f}% {bar} (Cumulative: {cum_var*100:.1f}%)")

# =============================================================================
# STEP 6: Visualization
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: PCA 2D scatter
ax1 = axes[0]
scatter = ax1.scatter(
    X_pca[:, 0], X_pca[:, 1], c=range(10), cmap="viridis", s=150, edgecolors="black"
)

for i, label in enumerate(student_labels):
    ax1.annotate(
        label,
        (X_pca[i, 0], X_pca[i, 1]),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=8,
    )

ax1.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% variance)", fontsize=12)
ax1.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% variance)", fontsize=12)
ax1.set_title("PCA: Students in 2D", fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

# Plot 2: Explained variance
ax2 = axes[1]
components = range(1, 5)
bars = ax2.bar(
    components,
    pca_full.explained_variance_ratio_ * 100,
    color="#4ECDC4",
    alpha=0.7,
    edgecolor="black",
)
ax2.plot(
    components,
    cumulative_var * 100,
    "ro-",
    linewidth=2,
    markersize=8,
    label="Cumulative",
)

ax2.set_xlabel("Principal Component", fontsize=12)
ax2.set_ylabel("Explained Variance (%)", fontsize=12)
ax2.set_title("Explained Variance by Component", fontsize=14)
ax2.set_xticks(components)
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")

for bar, var in zip(bars, pca_full.explained_variance_ratio_ * 100):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{var:.1f}%",
        ha="center",
        fontsize=10,
    )

# Plot 3: Component loadings heatmap
ax3 = axes[2]
loadings_matrix = loadings.T  # Transpose for better visualization
im = ax3.imshow(loadings_matrix, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)

ax3.set_xticks([0, 1])
ax3.set_xticklabels(["PC1", "PC2"])
ax3.set_yticks(range(len(feature_names)))
ax3.set_yticklabels(feature_names)
ax3.set_title("Feature Loadings", fontsize=14)

# Add colorbar
cbar = plt.colorbar(im, ax=ax3)
cbar.set_label("Loading", fontsize=10)

# Add values to heatmap
for i in range(len(feature_names)):
    for j in range(2):
        text = ax3.text(
            j,
            i,
            f"{loadings_matrix[i, j]:.2f}",
            ha="center",
            va="center",
            color="black",
            fontsize=10,
        )

plt.tight_layout()
plt.savefig("ML/PCA/pca_plot.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n" + "=" * 60)
print("✅ Visualization saved to:")
print("   ML/PCA/pca_plot.png")
print("=" * 60)

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 60)
print("📋 PCA Summary:")
print("-" * 40)
print("   ✓ Reduced 4 features to 2")
print(f"   ✓ Kept {sum(explained_var)*100:.1f}% of information")
print("   ✓ PC1 = Overall academic ability")
print("   ✓ PC2 = STEM vs Arts direction")
print("=" * 60)
