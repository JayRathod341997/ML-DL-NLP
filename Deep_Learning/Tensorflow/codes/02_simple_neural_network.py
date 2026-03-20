"""
================================================================================
CHAPTER 2: YOUR FIRST NEURAL NETWORK - THE MAGIC WAND
================================================================================

Story: Remember when you learned to ride a bicycle? At first, it seemed
impossible! But with practice, your brain learned to balance, pedal, and steer
all at once. That's exactly what a neural network does - it learns patterns
through examples!

In this chapter, we'll build our first neural network to recognize handwritten
digits (0-9) using the MNIST dataset. This is like teaching a child to
recognize numbers!

We'll cover:
- Loading the MNIST dataset
- Understanding the data structure
- Building a simple neural network
- Training and evaluating the model
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

print("=" * 70)
print("🧙‍♂️ CHAPTER 2: YOUR FIRST NEURAL NETWORK")
print("=" * 70)

# ==============================================================================
# PART 1: LOADING THE DATASET - Gathering Your Training Materials
# ==============================================================================

print("\n📚 PART 1: Loading the MNIST Dataset")
print("-" * 50)

# MNIST = Modified National Institute of Standards and Technology
# It's like a textbook of handwritten digits!
# 60,000 training images, 10,000 test images

# Load the dataset (TensorFlow has it built-in!)
mnist = tf.keras.datasets.mnist

# Split into training and testing sets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(f"Training images shape: {train_images.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Test images shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}")

# Each image is 28x28 pixels (like a small photo)
# Labels are numbers 0-9

# ==============================================================================
# PART 2: EXPLORING THE DATA - What's in Our Textbook?
# ==============================================================================

print("\n\n🔍 PART 2: Exploring the Data")
print("-" * 50)

# Let's look at some examples
print(f"\nFirst training image (as numbers):")
print(train_images[0])  # It's a matrix of pixel values (0-255)

print(f"\nFirst training label: {train_labels[0]}")
print(f"Pixel value range: {train_images[0].min()} to {train_images[0].max()}")

# Normalize the data (scale to 0-1 for better training!)
# This is like resizing all your photos to the same size
train_images = train_images / 255.0
test_images = test_images / 255.0

print(f"\nAfter normalization:")
print(f"Pixel value range: {train_images[0].min():.2f} to {train_images[0].max():.2f}")


# Let's visualize a few examples!
def plot_samples(images, labels, title):
    """Plot a grid of sample images"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap="gray")
        ax.set_title(f"Label: {labels[i]}")
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# Uncomment to see the visualization
# plot_samples(train_images, train_labels, "MNIST Sample Digits")

print("\n✅ Data loaded and normalized!")
print("   Each image is 28x28 pixels, values scaled to 0-1")

# ==============================================================================
# PART 3: BUILDING THE MODEL - Assembling Your Neural Network
# ==============================================================================

print("\n\n🏗️ PART 3: Building the Neural Network")
print("-" * 50)

"""
Our Neural Network Architecture:
--------------------------------
Input Layer:     28 x 28 = 784 neurons (flattened image)
    ↓
Hidden Layer 1: 128 neurons + ReLU activation
    ↓
Hidden Layer 2: 64 neurons + ReLU activation  
    ↓
Output Layer:    10 neurons (digits 0-9) + Softmax activation
"""

# Create a Sequential model (like stacking LEGO bricks!)
model = tf.keras.Sequential(
    [
        # Flatten the 28x28 image into a 1D array of 784 values
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        # First hidden layer - 128 neurons
        tf.keras.layers.Dense(128, activation="relu"),
        # Second hidden layer - 64 neurons
        tf.keras.layers.Dense(64, activation="relu"),
        # Output layer - 10 neurons (one for each digit 0-9)
        # Softmax converts outputs to probabilities that sum to 1
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

# Let's see our model's structure!
print("\n📋 Model Summary:")
model.summary()

# ==============================================================================
# PART 4: COMPILING THE MODEL - Setting Up the Training
# ==============================================================================

print("\n\n⚙️ PART 4: Compiling the Model")
print("-" * 50)

"""
Training Configuration:
-----------------------
- Optimizer: Adam (adaptive learning - like having a smart tutor)
- Loss: Sparse Categorical Crossentropy (for classification)
- Metrics: Accuracy (what percentage did we get right?)
"""

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

print("✅ Model compiled!")
print("   Optimizer: Adam")
print("   Loss: Sparse Categorical Crossentropy")
print("   Metrics: Accuracy")

# ==============================================================================
# PART 5: TRAINING THE MODEL - The Learning Phase!
# ==============================================================================

print("\n\n🎓 PART 5: Training the Model")
print("-" * 50)

print("Training in progress... This is like teaching a child! 🤓")
print("-" * 40)

# Train the model
history = model.fit(
    train_images,
    train_labels,
    epochs=5,  # How many times to see the entire dataset
    validation_split=0.2,  # Use 20% for validation
    verbose=1,  # Show progress
)

print("\n✅ Training complete!")

# ==============================================================================
# PART 6: EVALUATING THE MODEL - Did We Learn Anything?
# ==============================================================================

print("\n\n📊 PART 6: Evaluating the Model")
print("-" * 50)

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)

print(f"\n🎯 Test Results:")
print(f"   Test Loss: {test_loss:.4f}")
print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# ==============================================================================
# PART 7: MAKING PREDICTIONS - Using Our Model!
# ==============================================================================

print("\n\n🔮 PART 7: Making Predictions")
print("-" * 50)

# Get predictions for first 5 test images
predictions = model.predict(test_images[:5])

print("Predictions for first 5 test images:")
for i, pred in enumerate(predictions):
    predicted_digit = tf.argmax(pred).numpy()
    confidence = tf.reduce_max(pred).numpy() * 100
    actual_digit = test_labels[i]

    print(
        f"   Image {i+1}: Predicted={predicted_digit}, "
        f"Confidence={confidence:.1f}%, Actual={actual_digit} "
        f"{'✅' if predicted_digit == actual_digit else '❌'}"
    )

# ==============================================================================
# EXERCISE: Improve the Model!
# ==============================================================================

print("\n\n" + "=" * 70)
print("🎯 PRACTICE EXERCISES")
print("=" * 70)

print(
    """
Challenge 1: Add More Layers
-----------------------------
Add another hidden layer with 32 neurons to the model.
Does accuracy improve?

Challenge 2: Change the Optimizer
---------------------------------
Try using 'sgd' (Stochastic Gradient Descent) instead of 'adam'.
How does it affect training speed and accuracy?

Challenge 3: More Epochs
-------------------------
Train for 10 epochs instead of 5.
What happens to the accuracy?

Challenge 4: Regularization
----------------------------
Add dropout layers to prevent overfitting.
tf.keras.layers.Dropout(0.2)
"""
)

# Here's an example solution for Challenge 1:
print("\n📝 Solution for Challenge 1 (Adding more layers):")
print("-" * 40)

improved_model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(256, activation="relu"),  # Larger first layer
        tf.keras.layers.Dropout(0.2),  # Prevent overfitting
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),  # New layer!
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

improved_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

print("Improved model created with 5 hidden layers!")

# ==============================================================================
# KEY TAKEAWAYS
# ==============================================================================

print("\n\n" + "=" * 70)
print("📚 KEY TAKEAWAYS")
print("=" * 70)
print(
    """
1. MNIST is the "Hello World" of deep learning datasets
2. Neural networks learn by adjusting weights through backpropagation
3. Flatten layer converts 2D images to 1D vectors
4. Dense layers are fully connected layers
5. ReLU activation: f(x) = max(0, x) - introduces non-linearity
6. Softmax activation: converts logits to probabilities
7. Adam optimizer adapts learning rate during training
8. Always normalize input data to range [0, 1]

🎉 Congratulations! You've built and trained your first neural network!
   You've taken your first step into the world of deep learning!
"""
)

print("\n✨ Chapter 2 Complete! Ready for more adventures?")
