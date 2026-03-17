"""
Naive Bayes - Code Example with In-Memory Data
===============================================
This example demonstrates Naive Bayes using scikit-learn
with in-memory data for tennis play prediction.
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

# =============================================================================
# STEP 1: Create In-Memory Data (Weather Tennis Dataset)
# =============================================================================

# Using encoded data (Gaussian Naive Bayes works with continuous features)
# Outlook: 0=Overcast, 1=Rainy, 2=Sunny
# Humidity: Normalized (0-1)
# Wind: 0=Weak, 1=Strong

X = np.array(
    [
        [2, 0.9, 0],  # Sunny, High, Weak -> No
        [2, 0.9, 1],  # Sunny, High, Strong -> No
        [2, 0.4, 0],  # Sunny, Normal, Weak -> Yes
        [2, 0.4, 1],  # Sunny, Normal, Strong -> Yes
        [0, 0.9, 0],  # Overcast, High, Weak -> Yes
        [0, 0.9, 1],  # Overcast, High, Strong -> Yes
        [0, 0.4, 0],  # Overcast, Normal, Weak -> Yes
        [0, 0.4, 1],  # Overcast, Normal, Strong -> Yes
        [1, 0.9, 0],  # Rainy, High, Weak -> No
        [1, 0.9, 1],  # Rainy, High, Strong -> No
        [1, 0.4, 0],  # Rainy, Normal, Weak -> Yes
        [1, 0.4, 1],  # Rainy, Normal, Strong -> No
    ]
)

# Labels: 0=Don't Play, 1=Play Tennis
y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0])

class_names = ["Don't Play", "Play Tennis"]
feature_names = ["Outlook", "Humidity", "Wind"]

outlook_map = {0: "Overcast", 1: "Rainy", 2: "Sunny"}
humidity_map = {0.9: "High", 0.4: "Normal"}
wind_map = {0: "Weak", 1: "Strong"}

print("=" * 60)
print("NAIVE BAYES - TENNIS PLAY PREDICTION")
print("=" * 60)
print("\n📊 Training Data:")
print("-" * 40)

for i, (features, label) in enumerate(zip(X, y)):
    outlook = outlook_map[features[0]]
    humidity = humidity_map[features[1]]
    wind = wind_map[features[2]]
    result = class_names[label]
    print(f"   {i+1}. {outlook}, {humidity}, {wind} → {result}")

# =============================================================================
# STEP 2: Train Naive Bayes Model
# =============================================================================

model = GaussianNB()
model.fit(X, y)

print("\n" + "=" * 60)
print("📈 Model Information:")
print("-" * 40)
print(f"   Model Type: Gaussian Naive Bayes")
print(f"   Classes: {class_names}")
print(f"   Training Accuracy: {model.score(X, y)*100:.1f}%")

# =============================================================================
# STEP 3: Make Predictions
# =============================================================================

# New days to predict
new_days = np.array(
    [
        [2, 0.9, 0],  # Sunny, High, Weak
        [1, 0.4, 0],  # Rainy, Normal, Weak
        [0, 0.4, 1],  # Overcast, Normal, Strong
    ]
)

predictions = model.predict(new_days)
probabilities = model.predict_proba(new_days)

print("\n" + "=" * 60)
print("🔮 Predictions for New Days:")
print("-" * 40)

for i, (day, pred, prob) in enumerate(zip(new_days, predictions, probabilities)):
    outlook = outlook_map[day[0]]
    humidity = humidity_map[day[1]]
    wind = wind_map[day[2]]
    result = class_names[pred]

    print(f"\n   Day {i+1}: {outlook}, {humidity}, {wind}")
    print(f"      → Prediction: {result}")
    print(f"      → Probabilities:")
    for j, class_name in enumerate(class_names):
        bar = "█" * int(prob[j] * 20)
        print(f"         {class_name}: {prob[j]*100:5.1f}% {bar}")

# =============================================================================
# STEP 4: Show Probability Calculation
# =============================================================================

print("\n" + "=" * 60)
print("🔢 Probability Calculation (First Day):")
print("-" * 40)

# Get the probability components
print(f"   New Day: Sunny (2), High Humidity (0.9), Weak Wind (0)")
print(f"\n   Class Priors:")
print(f"      P(Play) = {model.class_prior_[1]:.3f}")
print(f"      P(Don't Play) = {model.class_prior_[0]:.3f}")

print(f"\n   Feature Probabilities (using Gaussian distribution):")

# Get mean and var for each class
for class_idx, class_name in enumerate(class_names):
    print(f"\n   {class_name}:")
    means = model.theta_[class_idx]
    vars_ = model.var_[class_idx]
    for feat_idx, feat_name in enumerate(feature_names):
        print(
            f"      {feat_name}: mean={means[feat_idx]:.2f}, var={vars_[feat_idx]:.2f}"
        )

# =============================================================================
# STEP 5: Visualization
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Feature distributions by class
ax1 = axes[0]

# Create histograms for each feature by class
colors = ["#FF6B6B", "#4ECDC4"]
for class_idx, (class_name, color) in enumerate(zip(class_names, colors)):
    mask = y == class_idx
    class_data = X[mask, 1]  # Humidity feature
    ax1.hist(class_data, bins=10, alpha=0.5, label=class_name, color=color)

ax1.set_xlabel("Humidity (normalized)", fontsize=12)
ax1.set_ylabel("Frequency", fontsize=12)
ax1.set_title("Naive Bayes: Feature Distribution by Class", fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Prediction probabilities
ax2 = axes[1]
x_pos = np.arange(len(new_days))
width = 0.35

probs_play = probabilities[:, 1]
probs_dont = probabilities[:, 0]

bars1 = ax2.bar(
    x_pos - width / 2, probs_play * 100, width, label="Play Tennis", color="#4ECDC4"
)
bars2 = ax2.bar(
    x_pos + width / 2, probs_dont * 100, width, label="Don't Play", color="#FF6B6B"
)

ax2.set_xlabel("Test Day", fontsize=12)
ax2.set_ylabel("Probability (%)", fontsize=12)
ax2.set_title("Naive Bayes: Prediction Probabilities", fontsize=14)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(["Day 1\n(Sunny)", "Day 2\n(Rainy)", "Day 3\n(Overcast)"])
ax2.legend()
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3, axis="y")

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax2.annotate(
        f"{height:.0f}%",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
    )

for bar in bars2:
    height = bar.get_height()
    ax2.annotate(
        f"{height:.0f}%",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.tight_layout()
plt.savefig("ML/Naive_Bayes/naive_bayes_plot.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n" + "=" * 60)
print(" Visualization saved to:")
print("   ML/Naive_Bayes/naive_bayes_plot.png")
print("=" * 60)
