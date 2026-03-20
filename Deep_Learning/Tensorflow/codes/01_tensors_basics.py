"""
================================================================================
CHAPTER 1: MEETING TENSORS - THE BUILDING BLOCKS
================================================================================

Story: Imagine you're a wizard learning to cast spells. Before you can create
magical effects, you need to understand the fundamental elements - in our case,
TENSORS! Just as every spell needs mana and incantations, every deep learning
model needs tensors to work its magic.

In this chapter, we'll learn:
- What tensors are (the building blocks of TensorFlow)
- Different types of tensors
- Basic operations with tensors

Think of tensors as containers that hold data. A scalar is 0D, a vector is 1D,
a matrix is 2D, and tensors can be n-dimensional!
"""

import tensorflow as tf
import numpy as np

print("=" * 70)
print("🧙‍♂️ CHAPTER 1: MEETING TENSORS - THE BUILDING BLOCKS")
print("=" * 70)

# ==============================================================================
# PART 1: Creating Your First Tensors
# ==============================================================================

print("\n📦 PART 1: Creating Your First Tensors")
print("-" * 50)

# 🌟 SCALAR (0D Tensor) - Like a single drop of water
# A single number, no dimensions
scalar = tf.constant(42)
print(f"Scalar (0D): {scalar.numpy()}")
print(f"  Shape: {scalar.shape}")
print(f"  Dtype: {scalar.dtype}")

# 🌟 VECTOR (1D Tensor) - Like a row of houses
# A list of numbers - has length but no width
vector = tf.constant([1, 2, 3, 4, 5])
print(f"\nVector (1D): {vector.numpy()}")
print(f"  Shape: {vector.shape}")
print(f"  Length: {vector.shape[0]}")

# 🌟 MATRIX (2D Tensor) - Like a spreadsheet or chessboard
# A table of numbers with rows and columns
matrix = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\nMatrix (2D):\n{matrix.numpy()}")
print(f"  Shape: {matrix.shape} (rows x columns)")

# 🌟 TENSOR (3D and beyond) - Like a cube of data!
# Think of a stack of matrices
tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"\n3D Tensor:\n{tensor_3d.numpy()}")
print(f"  Shape: {tensor_3d.shape}")

# ==============================================================================
# PART 2: TensorFlow vs NumPy - Best Friends!
# ==============================================================================

print("\n\n🤝 PART 2: TensorFlow Meets NumPy")
print("-" * 50)

# Create a numpy array
numpy_array = np.array([1, 2, 3, 4, 5])

# Convert to TensorFlow tensor
tf_tensor = tf.constant(numpy_array)
print(f"NumPy array: {numpy_array}")
print(f"TensorFlow tensor: {tf_tensor.numpy()}")

# Convert back to NumPy
back_to_numpy = tf_tensor.numpy()
print(f"Back to NumPy: {back_to_numpy}")

# ==============================================================================
# PART 3: Tensor Operations - The Magic Spells
# ==============================================================================

print("\n\n✨ PART 3: Tensor Operations - Magic Spells")
print("-" * 50)

# Create two tensors for operations
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

print(f"Tensor A:\n{a.numpy()}")
print(f"Tensor B:\n{b.numpy()}")

# 🔮 Addition
print("\n🔮 Addition (A + B):")
print(tf.add(a, b).numpy())

# 🔮 Multiplication (element-wise)
print("\n🔮 Element-wise Multiplication (A * B):")
print(tf.multiply(a, b).numpy())

# 🔮 Matrix Multiplication
print("\n🔮 Matrix Multiplication (A @ B):")
print(tf.matmul(a, b).numpy())

# 🔮 Finding the maximum
print(f"\n🔮 Maximum value in A: {tf.reduce_max(a).numpy()}")

# 🔮 Sum of all elements
print(f"🔮 Sum of all elements in B: {tf.reduce_sum(b).numpy()}")

# ==============================================================================
# PART 4: Changing Shapes - Like Clay!
# ==============================================================================

print("\n\n🎨 PART 4: Reshaping Tensors")
print("-" * 50)

# Create a tensor with 12 elements
original = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
print(f"Original tensor: {original.numpy()}")
print(f"Original shape: {original.shape}")

# Reshape to different shapes (must have same total elements!)
print("\nReshaped to (3, 4):")
print(tf.reshape(original, [3, 4]).numpy())

print("\nReshaped to (2, 6):")
print(tf.reshape(original, [2, 6]).numpy())

print("\nReshaped to (2, 2, 3):")
print(tf.reshape(original, [2, 2, 3]).numpy())

# ==============================================================================
# PART 5: One-hot Encoding - Categorical Magic
# ==============================================================================

print("\n\n🎯 PART 5: One-Hot Encoding")
print("-" * 50)

# Imagine you have categories: 0=Apple, 1=Banana, 2=Cherry
# One-hot encoding converts them to binary vectors
indices = tf.constant([0, 1, 2, 0, 1, 2])
depth = 3

one_hot = tf.one_hot(indices, depth)
print(f"Indices: {indices.numpy()}")
print(f"One-hot encoded:\n{one_hot.numpy()}")

# ==============================================================================
# EXERCISE: Practice Time!
# ==============================================================================

print("\n\n" + "=" * 70)
print("🎯 PRACTICE EXERCISE")
print("=" * 70)

print(
    """
Challenge: Create a simple calculator using tensors!

1. Create two tensors with values [1, 2, 3, 4, 5] and [10, 20, 30, 40, 50]
2. Find their sum, difference, and product
3. Calculate the mean of each tensor
4. Reshape the second tensor to a 5x1 matrix
"""
)

# Your turn! Try to solve this challenge.
# Solution is below, but try first!

# Solution:
tensor1 = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
tensor2 = tf.constant([10, 20, 30, 40, 50], dtype=tf.float32)

print("Solution:")
print(f"  Sum: {tf.add(tensor1, tensor2).numpy()}")
print(f"  Difference: {tf.subtract(tensor2, tensor1).numpy()}")
print(f"  Product: {tf.multiply(tensor1, tensor2).numpy()}")
print(f"  Mean of tensor1: {tf.reduce_mean(tensor1).numpy()}")
print(f"  Mean of tensor2: {tf.reduce_mean(tensor2).numpy()}")
print(f"  Reshaped tensor2 (5x1):\n{tf.reshape(tensor2, [5, 1]).numpy()}")

# ==============================================================================
# KEY TAKEAWAYS
# ==============================================================================

print("\n\n" + "=" * 70)
print("📚 KEY TAKEAWAYS")
print("=" * 70)
print(
    """
1. Tensors are the fundamental building blocks of TensorFlow
2. They come in different dimensions: 0D (scalar), 1D (vector), 2D (matrix), 3D+
3. TensorFlow tensors can be easily converted to/from NumPy
4. Tensors support various mathematical operations
5. We can reshape tensors as long as the total number of elements stays the same

Next Chapter: We'll learn about Variables and how to create trainable tensors!
"""
)

print(
    "\n✨ Chapter 1 Complete! You're one step closer to becoming a TensorFlow Wizard!"
)
