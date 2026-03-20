"""
================================================================================
CHAPTER 3: CNN - THE VISION QUEST
================================================================================

Story: Imagine you're an eagle soaring high above the mountains. Your keen
eyes can spot a rabbit moving in the grass from thousands of feet up! Your
brain doesn't look at every single pixel - it looks for edges, shapes, and
patterns.

Convolutional Neural Networks (CNNs) work the same way! They learn to see
images by detecting edges, textures, and complex patterns - just like your
visual cortex!

In this chapter, we'll build a CNN to classify images - like teaching our
eagle to recognize different types of prey!
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("🦅 CHAPTER 3: CNN - THE VISION QUEST")
print("=" * 70)

# ==============================================================================
# PART 1: UNDERSTANDING CONVOLUTIONS - The Eagle's Eye
# ==============================================================================

print("\n👁️ PART 1: Understanding Convolutions")
print("-" * 50)

"""
How does a CNN "see"?

Think of a Convolution as a magnifying glass sliding over an image:
- The "magnifying glass" is called a KERNEL or FILTER
- It looks at small patches at a time (e.g., 3x3 pixels)
- It learns to detect features like edges, curves, textures

Example: Detecting horizontal lines in an image
Kernel:
[[-1, -1, -1],
 [ 2,  2,  2],
 [-1, -1, -1]]

This kernel highlights horizontal lines! (top and bottom are negative,
middle is positive)
"""


# Let's visualize what a convolution does!
def visualize_convolution():
    """Create a simple visualization of convolution"""
    # Create a simple "image" with a horizontal line
    image = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],  # Horizontal line
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    # Simple horizontal edge detection kernel
    kernel = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)

    # Apply convolution manually (simplified)
    # In practice, TensorFlow does this efficiently!
    print("Original image (5x10) with horizontal line:")
    print(image)
    print("\nHorizontal edge detection kernel (3x3):")
    print(kernel)
    print("\nAfter convolution, the horizontal line is highlighted!")


visualize_convolution()

# ==============================================================================
# PART 2: LOADING CIFAR-10 - Our Training Ground
# ==============================================================================

print("\n\n📚 PART 2: Loading CIFAR-10 Dataset")
print("-" * 50)

"""
CIFAR-10 Dataset:
- 60,000 colorful images (32x32 pixels)
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Like showing our eagle 10 different types of animals to recognize!
"""

# Load CIFAR-10
cifar10 = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

print(f"Training data: {train_images.shape}")
print(f"Training labels: {train_labels.shape}")
print(f"Test data: {test_images.shape}")

# Class names
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

print(f"\nClass names: {class_names}")
print(f"\nFirst image label: {class_names[train_labels[0][0]]}")

# Normalize to 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

# ==============================================================================
# PART 3: BUILDING THE CNN - The Eagle's Brain
# ==============================================================================

print("\n\n🧠 PART 3: Building the CNN Architecture")
print("-" * 50)

"""
Our CNN Architecture:
=====================

Input: 32x32x3 (RGB image)
  ↓
Conv2D (32 filters, 3x3) → ReLU → MaxPool (2x2)
  ↓  [Learns basic edges and colors]
Conv2D (64 filters, 3x3) → ReLU → MaxPool (2x2)
  ↓  [Learns textures and patterns]
Conv2D (64 filters, 3x3) → ReLU → Flatten
  ↓  [Learns complex features]
Dense (64 neurons) → Dropout → Dense (10, softmax)
  ↓
Output: 10 classes
"""

model = tf.keras.Sequential(
    [
        # First Convolutional Block
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # Second Convolutional Block
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # Third Convolutional Block
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        # Dense layers for classification
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),  # Prevent overfitting!
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

print("📋 CNN Model Summary:")
model.summary()

# ==============================================================================
# PART 4: COMPILING AND TRAINING
# ==============================================================================

print("\n\n⚙️ PART 4: Compiling and Training")
print("-" * 50)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

print("Training the CNN... (This may take a few minutes) ⏳")
print("-" * 40)

# Train with fewer epochs for demo
history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    validation_data=(test_images, test_labels),
    verbose=1,
)

# ==============================================================================
# PART 5: EVALUATION - How Well Does Our Eagle See?
# ==============================================================================

print("\n\n📊 PART 5: Model Evaluation")
print("-" * 50)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\n🎯 Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# Make some predictions!
predictions = model.predict(test_images[:5])

print("\n🔮 Sample Predictions:")
for i, pred in enumerate(predictions):
    predicted_class = class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100
    actual_class = class_names[test_labels[i][0]]
    print(
        f"   Image {i+1}: Predicted={predicted_class} "
        f"({confidence:.1f}%), Actual={actual_class}"
    )

# ==============================================================================
# PART 6: VISUALIZING FILTERS - Seeing What the Eagle Sees
# ==============================================================================

print("\n\n🔍 PART 6: Visualizing What the CNN Learned")
print("-" * 50)

print(
    """
💡 Fun Fact: CNN filters learn to detect different features!

- Early layers: Detect edges, colors, simple patterns
- Middle layers: Detect textures, shapes, parts of objects
- Deep layers: Detect complete objects!

You can visualize filters using:
  tf.keras.preprocessing.image.apply_modifications_to_image()
"""
)

# ==============================================================================
# EXERCISES
# ==============================================================================

print("\n\n" + "=" * 70)
print("🎯 PRACTICE EXERCISES")
print("=" * 70)

print(
    """
Exercise 1: Add More ConvLayers
---------------------------------
Add a fourth Conv2D layer with 128 filters.
Does accuracy improve?

Exercise 2: Data Augmentation
------------------------------
Use tf.keras.layers.RandomFlip() and RandomRotation()
to artificially increase your training data.

Exercise 3: Transfer Learning
-------------------------------
Try using MobileNetV2 as a feature extractor!
"""
)

# ==============================================================================
# KEY TAKEAWAYS
# ==============================================================================

print("\n\n" + "=" * 70)
print("📚 KEY TAKEAWAYS")
print("=" * 70)
print(
    """
1. CNNs are inspired by how our visual cortex works!
2. Conv2D applies filters to detect features in images
3. MaxPooling reduces spatial dimensions while keeping important features
4. Dropout prevents overfitting by randomly "turning off" neurons
5. Deeper networks can learn more complex patterns
6. Transfer learning uses pre-trained models for faster learning!

🎉 Congratulations! Your eagle can now see the world!
"""
)

print("\n✨ Chapter 3 Complete! The Vision Quest is over!")
