"""
Linear Regression - Code Example with In-Memory Data
======================================================
This example demonstrates Linear Regression using scikit-learn
with in-memory data for predicting ice cream sales based on temperature.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# =============================================================================
# STEP 1: Create In-Memory Data
# =============================================================================
# Temperature data (in Celsius) - Input feature (X)
# Ice creams sold - Target variable (y)

# Our training data: [Temperature, Ice Creams Sold]
X = np.array([[20], [25], [30], [35], [40]])  # Reshaped for sklearn
y = np.array([40, 55, 80, 110, 145])

print("=" * 60)
print("LINEAR REGRESSION - ICE CREAM SALES PREDICTION")
print("=" * 60)
print("\n📊 Training Data:")
print("-" * 40)
for temp, sales in zip(X.flatten(), y):
    print(f"   Temperature: {temp}°C → Ice Creams Sold: {sales}")

# =============================================================================
# STEP 2: Create and Train the Model
# =============================================================================

# Create Linear Regression model
model = LinearRegression()

# Train the model with our data
model.fit(X, y)

# =============================================================================
# STEP 3: Get Model Parameters
# =============================================================================

# Get the slope (coefficient) and intercept
slope = model.coef_[0]
intercept = model.intercept_

print("\n" + "=" * 60)
print("📈 Model Results:")
print("-" * 40)
print(f"   Slope (m): {slope:.2f}")
print(f"   Intercept (b): {intercept:.2f}")
print(f"   Formula: y = {slope:.2f}x + {intercept:.2f}")

# =============================================================================
# STEP 4: Make Predictions
# =============================================================================

# Predict for new temperatures
new_temperatures = np.array([[22], [28], [38], [32], [42]])
predictions = model.predict(new_temperatures)

print("\n" + "=" * 60)
print("🔮 Predictions:")
print("-" * 40)
for temp, pred in zip(new_temperatures.flatten(), predictions):
    print(f"   Temperature: {temp}°C → Predicted Sales: {pred:.0f} ice creams")

# =============================================================================
# STEP 5: Model Evaluation
# =============================================================================

# Calculate R-squared (how good is our model?)
r_squared = model.score(X, y)

print("\n" + "=" * 60)
print("📊 Model Evaluation:")
print("-" * 40)
print(f"   R-squared: {r_squared:.4f}")
print(f"   This means {r_squared*100:.1f}% of the variation in ice cream sales")
print(f"   can be explained by temperature!")

# =============================================================================
# STEP 6: Visualization
# =============================================================================

plt.figure(figsize=(10, 6))

# Plot training data points
plt.scatter(X, y, color="blue", s=100, label="Training Data", zorder=5)

# Plot regression line
X_line = np.linspace(15, 45, 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, color="red", linewidth=2, label="Regression Line")

# Plot predictions
plt.scatter(
    new_temperatures,
    predictions,
    color="green",
    s=100,
    marker="^",
    label="Predictions",
    zorder=5,
)

plt.xlabel("Temperature (°C)", fontsize=12)
plt.ylabel("Ice Creams Sold", fontsize=12)
plt.title("Linear Regression: Ice Cream Sales Prediction", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Add equation annotation
equation = f"y = {slope:.2f}x + {intercept:.2f}\nR² = {r_squared:.4f}"
plt.annotate(
    equation,
    xy=(0.05, 0.95),
    xycoords="axes fraction",
    fontsize=11,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.tight_layout()
plt.savefig(
    "ML/Linear_Regression/linear_regression_plot.png", dpi=150, bbox_inches="tight"
)
plt.close()

print("\n" + "=" * 60)
print("✅ Visualization saved to:")
print("   ML/Linear_Regression/linear_regression_plot.png")
print("=" * 60)

# =============================================================================
# Additional: Manual Calculation Verification
# =============================================================================

print("\n" + "=" * 60)
print("🔢 Manual Calculation Verification:")
print("-" * 40)
print("   For temperature = 38°C:")
manual_pred = slope * 38 + intercept
print(f"   y = {slope:.2f}(38) + ({intercept:.2f})")
print(f"   y = {slope * 38:.2f} + ({intercept:.2f})")
print(f"   y = {manual_pred:.2f}")
print(f"   Rounded: {round(manual_pred)} ice creams")
print("=" * 60)
