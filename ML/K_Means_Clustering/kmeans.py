"""
K-Means Clustering - Code Example with In-Memory Data
======================================================
This example demonstrates K-Means clustering using scikit-learn
with in-memory data for customer segmentation.
"""

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# =============================================================================
# STEP 1: Create In-Memory Data (Customer Data)
# =============================================================================

# Features: Visits per Month, Amount Spent ($)

X = np.array(
    [
        [1, 200],  # Customer 1
        [2, 300],  # Customer 2
        [1, 250],  # Customer 3
        [8, 800],  # Customer 4
        [10, 900],  # Customer 5
        [7, 750],  # Customer 6
        [4, 450],  # Customer 7
        [5, 500],  # Customer 8
        [3, 350],  # Customer 9
        [6, 600],  # Customer 10
    ]
)

print("=" * 60)
print("K-MEANS CLUSTERING - CUSTOMER SEGMENTATION")
print("=" * 60)
print("\n📊 Customer Data:")
print("-" * 40)
print("   Customer | Visits/Month | Spent ($)")
print("   " + "-" * 35)
for i, (visits, spent) in enumerate(X):
    print(f"      {i+1}    |      {visits}      |   ${spent}")

# =============================================================================
# STEP 2: Apply K-Means Clustering
# =============================================================================

# Number of clusters
k = 3

# Create and fit K-Means model
model = KMeans(n_clusters=k, random_state=42, n_init=10)
model.fit(X)

# Get cluster labels and centroids
labels = model.labels_
centroids = model.cluster_centers_

print("\n" + "=" * 60)
print(f"🌲 Clustering Results (K={k}):")
print("-" * 40)

# Group customers by cluster
for cluster_id in range(k):
    cluster_customers = np.where(labels == cluster_id)[0]
    print(f"\n   Cluster {cluster_id} ({len(cluster_customers)} customers):")
    for idx in cluster_customers:
        print(f"      Customer {idx+1}: {X[idx]}")

# =============================================================================
# STEP 3: Cluster Statistics
# =============================================================================

print("\n" + "=" * 60)
print("📊 Cluster Statistics:")
print("-" * 40)

for cluster_id in range(k):
    cluster_points = X[labels == cluster_id]
    avg_visits = cluster_points[:, 0].mean()
    avg_spent = cluster_points[:, 1].mean()

    print(f"\n   Cluster {cluster_id}:")
    print(f"      Average Visits: {avg_visits:.1f}/month")
    print(f"      Average Spent: ${avg_spent:.0f}")

    # Determine customer type
    if avg_visits < 3:
        customer_type = "Occasional"
    elif avg_visits < 7:
        customer_type = "Regular"
    else:
        customer_type = "VIP"
    print(f"      Type: {customer_type} Customers")

# =============================================================================
# STEP 4: Visualize Clusters
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Scatter plot with clusters
ax1 = axes[0]

colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
cluster_names = ["Occasional", "Regular", "VIP"]

for cluster_id in range(k):
    mask = labels == cluster_id
    ax1.scatter(
        X[mask, 0],
        X[mask, 1],
        c=colors[cluster_id],
        s=150,
        label=f"Cluster {cluster_id}: {cluster_names[cluster_id]}",
        alpha=0.7,
        edgecolors="black",
    )

# Plot centroids
ax1.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c="yellow",
    marker="X",
    s=300,
    edgecolors="black",
    linewidths=2,
    label="Centroids",
    zorder=5,
)

ax1.set_xlabel("Visits per Month", fontsize=12)
ax1.set_ylabel("Spent ($)", fontsize=12)
ax1.set_title("K-Means Customer Segmentation", fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add customer numbers
for i, (visits, spent) in enumerate(X):
    ax1.annotate(
        f"{i+1}", (visits, spent), textcoords="offset points", xytext=(5, 5), fontsize=9
    )

# Plot 2: Decision boundaries
ax2 = axes[1]

# Create mesh grid
h = 0.5
x_min, x_max = 0, 12
y_min, y_max = 100, 1000
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict clusters on mesh
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
ax2.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
ax2.contour(xx, yy, Z, colors="black", linewidths=1, alpha=0.5)

# Plot data points
for cluster_id in range(k):
    mask = labels == cluster_id
    ax2.scatter(
        X[mask, 0],
        X[mask, 1],
        c=colors[cluster_id],
        s=150,
        label=f"Cluster {cluster_id}",
        alpha=0.7,
        edgecolors="black",
    )

# Plot centroids
ax2.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c="yellow",
    marker="X",
    s=300,
    edgecolors="black",
    linewidths=2,
    zorder=5,
)

ax2.set_xlabel("Visits per Month", fontsize=12)
ax2.set_ylabel("Spent ($)", fontsize=12)
ax2.set_title("K-Means Decision Boundaries", fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ML/K_Means_Clustering/kmeans_plot.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n" + "=" * 60)
print(" Visualization saved to:")
print("   ML/K_Means_Clustering/kmeans_plot.png")
print("=" * 60)

# =============================================================================
# STEP 5: Elbow Method (Finding optimal K)
# =============================================================================

print("\n" + "=" * 60)
print("📈 Elbow Method (Finding Optimal K):")
print("-" * 40)

inertias = []
K_range = range(1, 6)

for k_val in K_range:
    kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

print("   K | Inertia (Sum of squared distances)")
print("   " + "-" * 40)
for k_val, inertia in zip(K_range, inertias):
    bar = "█" * int(inertia / 50)
    print(f"   {k_val} | {inertia:.1f} {bar}")

print("\n   Lower inertia = better fit")
print("   Look for 'elbow' where decrease slows down")
print("=" * 60)
