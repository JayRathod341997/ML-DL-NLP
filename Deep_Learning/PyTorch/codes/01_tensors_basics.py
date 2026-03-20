"""
================================================================================
CHAPTER 1: TENSORS - THE BUILDING BLOCKS
================================================================================

Story: Welcome to PyTorch! Think of tensors as the DNA of deep learning -
they're the fundamental building blocks that hold and transform your data.

In this chapter, we'll explore:
- What tensors are (like numpy arrays but with superpowers!)
- Different tensor operations
- GPU acceleration
- The magic of automatic differentiation

Tensors in PyTorch are similar to NumPy arrays, but with one key difference:
they can be computed on GPUs for massive speedup!
"""

import torch
import numpy as np

print("=" * 70)
print("🔥 CHAPTER 1: MEETING TENSORS")
print("=" * 70)

# ==============================================================================
# PART 1: CREATING TENSORS
# ==============================================================================

print("\n📦 PART 1: Creating Tensors")
print("-" * 50)

# 🌟 From Python list
tensor_from_list = torch.tensor([1, 2, 3, 4, 5])
print(f"From list: {tensor_from_list}")

# 🌟 Zeros tensor (useful for initializing)
zeros = torch.zeros(3, 4)
print(f"\nZeros (3x4):\n{zeros}")

# 🌟 Ones tensor
ones = torch.ones(2, 3)
print(f"\nOnes (2x3):\n{ones}")

# 🌟 Random values (uniform distribution)
rand_uniform = torch.rand(2, 2)
print(f"\nRandom uniform:\n{rand_uniform}")

# 🌟 Random values (normal distribution)
rand_normal = torch.randn(3, 3)
print(f"\nRandom normal:\n{rand_normal}")

# 🌟 From NumPy array
numpy_array = np.array([1, 2, 3, 4, 5])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(f"\nFrom NumPy: {tensor_from_numpy}")

# ==============================================================================
# PART 2: TENSOR PROPERTIES
# ==============================================================================

print("\n\n📋 PART 2: Tensor Properties")
print("-" * 50)

x = torch.tensor([[1, 2, 3], [4, 5, 6]])

print(f"Tensor:\n{x}")
print(f"  Shape: {x.shape}")
print(f"  Dimensions (ndim): {x.ndim}")
print(f"  Data type (dtype): {x.dtype}")
print(f"  Device: {x.device}")

# ==============================================================================
# PART 3: TENSOR OPERATIONS
# ==============================================================================

print("\n\n✨ PART 3: Tensor Operations")
print("-" * 50)

a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

print(f"Tensor A:\n{a}")
print(f"\nTensor B:\n{b}")

# Addition
print("\n➕ Addition (A + B):")
print(a + b)
print(torch.add(a, b))

# Multiplication (element-wise)
print("\n✖️ Element-wise Multiplication:")
print(a * b)
print(torch.mul(a, b))

# Matrix Multiplication
print("\n🔮 Matrix Multiplication (A @ B):")
print(torch.matmul(a, b))
print(a @ b)

# ==============================================================================
# PART 4: RESHAPING TENSORS
# ==============================================================================

print("\n\n🎨 PART 4: Reshaping Tensors")
print("-" * 50)

original = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
print(f"Original: {original}")
print(f"Shape: {original.shape}")

# Reshape
reshaped = original.view(3, 4)
print(f"\nReshaped to (3, 4):\n{reshaped}")

# Transpose
transposed = reshaped.t()
print(f"\nTransposed:\n{transposed}")

# Flatten
flattened = reshaped.flatten()
print(f"\nFlattened: {flattened}")

# ==============================================================================
# PART 5: GRADIENTS - THE MAGIC OF AUTOGRAD
# ==============================================================================

print("\n\n🌈 PART 5: Automatic Differentiation (Autograd)")
print("-" * 50)

"""
This is where PyTorch shines! The autograd package automatically computes
gradients - essential for training neural networks!
"""

# Create tensor with gradient tracking
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Perform operations
y = x * 2
z = y.mean()

# Backward pass - computes gradients automatically!
z.backward()

# x.grad contains the gradients (dz/dx)
print(f"x: {x}")
print(f"y = x * 2: {y}")
print(f"z = mean(y): {z}")
print(f"dz/dx (gradients): {x.grad}")

"""
Explanation:
- z = mean(x * 2) = mean([2, 4, 6]) = 4
- dz/dx = 2/n for each element (where n=3)
- So grad = [2/3, 2/3, 2/3]
"""

# ==============================================================================
# PART 6: GPU ACCELERATION
# ==============================================================================

print("\n\n🚀 PART 6: GPU Acceleration")
print("-" * 50)

# Check if GPU is available
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # Move tensor to GPU
    device = torch.device("cuda")
    tensor_gpu = torch.tensor([1, 2, 3], device=device)
    print(f"Tensor on GPU: {tensor_gpu}")
    print(f"Device: {tensor_gpu.device}")

    # Move back to CPU
    tensor_cpu = tensor_gpu.cpu()
    print(f"Back to CPU: {tensor_cpu}")
else:
    print("No GPU available, using CPU")
    print("Tip: For faster training, use a machine with a GPU!")

# ==============================================================================
# EXERCISES
# ==============================================================================

print("\n\n" + "=" * 70)
print("🎯 PRACTICE EXERCISES")
print("=" * 70)

print(
    """
Exercise 1: Create a 5x5 matrix of random numbers
         Calculate its transpose
         Find the sum of all elements

Exercise 2: Create two tensors with requires_grad=True
         Multiply them element-wise
         Compute the backward pass
         Print the gradients

Exercise 3: If CUDA is available, create tensors on both
         CPU and GPU and compare the device property
"""
)

# Solution for Exercise 1:
print("\n📝 Solution 1:")
matrix = torch.rand(5, 5)
transposed = matrix.t()
total_sum = matrix.sum()
print(f"  5x5 Random matrix created")
print(f"  Transpose shape: {transposed.shape}")
print(f"  Sum of all elements: {total_sum.item():.4f}")

# ==============================================================================
# KEY TAKEAWAYS
# ==============================================================================

print("\n\n" + "=" * 70)
print("📚 KEY TAKEAWAYS")
print("=" * 70)
print(
    """
1. Tensors are the fundamental data structure in PyTorch
2. They're similar to NumPy arrays but can run on GPUs
3. requires_grad=True enables automatic differentiation
4. PyTorch operations feel natural and Pythonic
5. Use .to('cuda') or .to('cpu') to move tensors between devices

Next: We'll explore the amazing world of autograd!
"""
)

print("\n✨ Chapter 1 Complete! Ready for more PyTorch adventures?")
