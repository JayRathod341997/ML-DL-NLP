"""
Random Forest - Code Example with In-Memory Data
================================================
This example demonstrates Random Forest using scikit-learn
with in-memory data for classifying fruits.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# =============================================================================
# STEP 1: Create In-Memory Data (Fruit Dataset)
# =============================================================================

# Features: Weight (g), Color (encoded), Shape (encoded)
# Classes: Apple, Orange, Lemon

# Encoding:
# Color: 0=Orange, 1=Red, 2=Yellow
# Shape: 0=Oval, 1=Round

X = np.array(
    [
        [150, 1, 1],  # Apple: 150g, Red, Round
        [170, 1, 1],  # Apple: 170g, Red, Round
        [180, 1, 1],  # Apple: 180g, Red, Round
        [200, 1, 1],  # Apple: 200g, Red, Round
        [140, 0, 1],  # Orange: 140g, Orange, Round
        [160, 0, 1],  # Orange: 160g, Orange, Round
        [180, 0, 1],  # Orange: 180g, Orange, Round
        [200, 0, 1],  # Orange: 200g, Orange, Round
        [100, 2, 0],  # Lemon: 100g, Yellow, Oval
        [120, 2, 0],  # Lemon: 120g, Yellow, Oval
        [140, 2, 0],  # Lemon: 140g, Yellow, Oval
        [130, 2, 0],  # Lemon: 130g, Yellow, Oval
    ]
)

# Labels: 0=Apple, 1=Orange, 2=Lemon
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

feature_names = ["Weight", "Color", "Shape"]
class_names = ["Apple", "Orange", "Lemon"]

print("=" * 60)
print("RANDOM FOREST - FRUIT CLASSIFICATION")
print("=" * 60)
print("\n📊 Training Data:")
print("-" * 40)

color_map = {0: "Orange", 1: "Red", 2: "Yellow"}
shape_map = {0: "Oval", 1: "Round"}

for i, (features, label) in enumerate(zip(X, y)):
    weight = features[0]
    color = color_map[features[1]]
    shape = shape_map[features[2]]
    fruit = class_names[label]
    print(f"   {i+1}. {weight}g, {color}, {shape} → {fruit}")

# =============================================================================
# STEP 2: Create and Train Random Forest
# =============================================================================

# Create Random Forest with 10 trees
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

print("\n" + "=" * 60)
print("🌲 Forest Information:")
print("-" * 40)
print(f"   Number of Trees: {model.n_estimators}")
print(f"   Number of Classes: {model.n_classes_}")
print(f"   Number of Features: {model.n_features_in_}")
print(f"   Training Accuracy: {model.score(X, y)*100:.1f}%")

# =============================================================================
# STEP 3: Make Predictions
# =============================================================================

# New fruits to classify
new_fruits = np.array(
    [
        [160, 1, 1],  # 160g, Red, Round
        [150, 0, 1],  # 150g, Orange, Round
        [130, 2, 0],  # 130g, Yellow, Oval
    ]
)

predictions = model.predict(new_fruits)
probabilities = model.predict_proba(new_fruits)

print("\n" + "=" * 60)
print("🔮 Predictions for New Fruits:")
print("-" * 40)

for i, (fruit, pred, prob) in enumerate(zip(new_fruits, predictions, probabilities)):
    weight = fruit[0]
    color = color_map[fruit[1]]
    shape = shape_map[fruit[2]]
    predicted_fruit = class_names[pred]

    print(f"\n   Fruit {i+1}: {weight}g, {color}, {shape}")
    print(f"      → Prediction: {predicted_fruit}")
    print(f"      → Probabilities:")
    for j, class_name in enumerate(class_names):
        bar = "█" * int(prob[j] * 20)
        print(f"         {class_name}: {prob[j]*100:5.1f}% {bar}")

# =============================================================================
# STEP 4: Individual Tree Predictions
# =============================================================================

print("\n" + "=" * 60)
print("🗳️  Individual Tree Votes:")
print("-" * 40)

test_fruit = new_fruits[0]  # First test fruit
tree_predictions = [tree.predict([test_fruit])[0] for tree in model.estimators_]

print(
    f"   Test Fruit: {test_fruit[0]}g, {color_map[test_fruit[1]]}, {shape_map[test_fruit[2]]}"
)
print(f"   Individual Tree Predictions:")

vote_counts = {0: 0, 1: 0, 2: 0}
for i, pred in enumerate(tree_predictions):
    print(f"      Tree {i+1}: {class_names[pred]}")
    vote_counts[pred] += 1

print(f'"   Vote Summary:')
for class_idx, count in vote_counts.items():
    print(f"      {class_names[class_idx]}: {count} votes")
print(
    f"   Final Prediction: {class_names[max(vote_counts, key=vote_counts.get)]} (Majority!)"
)

# =============================================================================
# STEP 5: Feature Importance
# =============================================================================

print("\n" + "=" * 60)
print("📊 Feature Importance:")
print("-" * 40)

importances = model.feature_importances_
for name, importance in zip(feature_names, importances):
    bar = "█" * int(importance * 40)
    print(f"   {name}: {importance:.3f} {bar}")

# =============================================================================
# STEP 6: Visualization
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Feature Importance
ax1 = axes[0]
importance_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
bars = ax1.barh(feature_names, importances, color=importance_colors)
ax1.set_xlabel("Importance", fontsize=12)
ax1.set_title("Random Forest: Feature Importance", fontsize=14)
ax1.grid(True, alpha=0.3, axis="x")

for bar, importance in zip(bars, importances):
    ax1.text(
        importance + 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{importance:.3f}",
        va="center",
    )

# Plot 2: Prediction Probabilities
ax2 = axes[1]
x_pos = np.arange(len(new_fruits))
width = 0.25

for i, (class_name, color) in enumerate(zip(class_names, importance_colors)):
    probs = probabilities[:, i]
    ax2.bar(x_pos + i * width, probs, width, label=class_name, color=color, alpha=0.8)

ax2.set_xlabel("Test Fruit", fontsize=12)
ax2.set_ylabel("Probability", fontsize=12)
ax2.set_title("Random Forest: Prediction Probabilities", fontsize=14)
ax2.set_xticks(x_pos + width)
ax2.set_xticklabels(["Fruit 1", "Fruit 2", "Fruit 3"])
ax2.legend()
ax2.set_ylim(0, 1.1)
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("ML/Random_Forest/random_forest_plot.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n" + "=" * 60)
print("✅ Visualization saved to:")
print("   ML/Random_Forest/random_forest_plot.png")
print("=" * 60)
