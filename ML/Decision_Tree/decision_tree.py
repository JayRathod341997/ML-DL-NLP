"""
Decision Tree - Code Example with In-Memory Data
=================================================
This example demonstrates Decision Tree using scikit-learn
with in-memory data for predicting whether to play tennis.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# =============================================================================
# STEP 1: Create In-Memory Data (Tennis Weather Dataset)
# =============================================================================

# Features: Outlook, Humidity, Wind
# Target: Play Tennis (1=Yes, 0=No)

# Encoding:
# Outlook: 0=Overcast, 1=Rainy, 2=Sunny
# Humidity: 0=Normal, 1=High
# Wind: 0=Weak, 1=Strong

X = np.array(
    [
        [2, 1, 0],  # Sunny, High, Weak -> No
        [2, 1, 1],  # Sunny, High, Strong -> No
        [2, 0, 0],  # Sunny, Normal, Weak -> Yes
        [2, 0, 1],  # Sunny, Normal, Strong -> Yes
        [0, 1, 0],  # Overcast, High, Weak -> Yes
        [0, 1, 1],  # Overcast, High, Strong -> Yes
        [0, 0, 0],  # Overcast, Normal, Weak -> Yes
        [0, 0, 1],  # Overcast, Normal, Strong -> Yes
        [1, 1, 0],  # Rainy, High, Weak -> No
        [1, 1, 1],  # Rainy, High, Strong -> No
        [1, 0, 0],  # Rainy, Normal, Weak -> Yes
        [1, 0, 1],  # Rainy, Normal, Strong -> No
    ]
)

y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0])

# Feature names and class names
feature_names = ["Outlook", "Humidity", "Wind"]
class_names = ["Don't Play", "Play Tennis"]

print("=" * 60)
print("DECISION TREE - TENNIS PLAY PREDICTION")
print("=" * 60)
print("\n📊 Training Data:")
print("-" * 40)

outlook_map = {0: "Overcast", 1: "Rainy", 2: "Sunny"}
humidity_map = {0: "Normal", 1: "High"}
wind_map = {0: "Weak", 1: "Strong"}

for i, (features, label) in enumerate(zip(X, y)):
    outlook = outlook_map[features[0]]
    humidity = humidity_map[features[1]]
    wind = wind_map[features[2]]
    result = class_names[label]
    print(f"   {i+1}. {outlook}, {humidity}, {wind} → {result}")

# =============================================================================
# STEP 2: Create and Train the Model
# =============================================================================

# Create Decision Tree with limited depth for visualization
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# =============================================================================
# STEP 3: Tree Information
# =============================================================================

print("\n" + "=" * 60)
print("🌳 Tree Information:")
print("-" * 40)
print(f"   Tree Depth: {model.get_depth()}")
print(f"   Number of Leaves: {model.get_n_leaves()}")
print(f"   Training Accuracy: {model.score(X, y)*100:.1f}%")

# =============================================================================
# STEP 4: Make Predictions
# =============================================================================

# New days to predict
# Test case 1: Sunny, High humidity, Weak wind
# Test case 2: Rainy, Normal humidity, Strong wind
# Test case 3: Overcast, Normal humidity, Strong wind

new_days = np.array(
    [
        [2, 1, 0],  # Sunny, High, Weak
        [1, 0, 1],  # Rainy, Normal, Strong
        [0, 0, 1],  # Overcast, Normal, Strong
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
    print(f"      → Decision: {result}")
    print(f"      → Confidence: {max(prob)*100:.1f}%")

# =============================================================================
# STEP 5: Visualize Decision Tree
# =============================================================================

plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title("Decision Tree: Should I Play Tennis?", fontsize=16)
plt.tight_layout()
plt.savefig("ML/Decision_Tree/decision_tree_plot.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n" + "=" * 60)
print("✅ Decision Tree Visualization saved to:")
print("   ML/Decision_Tree/decision_tree_plot.png")
print("=" * 60)

# =============================================================================
# STEP 6: Print Tree Rules (Text Format)
# =============================================================================

print("\n" + "=" * 60)
print("📋 Decision Tree Rules:")
print("-" * 40)

tree_rules = export_text(model, feature_names=feature_names)
print(tree_rules)

# =============================================================================
# Feature Importance
# =============================================================================

print("=" * 60)
print("📊 Feature Importance:")
print("-" * 40)

importances = model.feature_importances_
for name, importance in zip(feature_names, importances):
    bar = "█" * int(importance * 20)
    print(f"   {name}: {importance:.3f} {bar}")

print("=" * 60)
