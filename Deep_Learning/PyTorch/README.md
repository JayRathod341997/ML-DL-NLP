# PyTorch Fundamentals

Welcome to the amazing world of PyTorch! 🔥 In this folder, we'll explore PyTorch, 
the powerful deep learning framework used by researchers and industry professionals 
all over the world.

## The Story of PyTorch

Imagine you're building a robot from scratch. You need to understand each component - 
the gears, wires, and circuits. PyTorch is like having a workshop full of tools that 
let you build, test, and improve your robot step by step.

PyTorch was created by Meta AI (formerly Facebook) and has become one of the most 
popular deep learning frameworks because of its:
- Dynamic computation graphs (flexible and easy to debug!)
- Pythonic style (feels natural to Python developers)
- Strong research support (used in most cutting-edge papers)
- Production deployment with TorchServe

**Our Learning Journey:**
- Chapter 1: Tensors - The Building Blocks 🧱
- Chapter 2: Autograd - The Magic of Differentiation ✨
- Chapter 3: Neural Networks - Building Your First Model 🧠
- Chapter 4: Training Loop - Teaching Your Model 🎓
- Chapter 5: CNNs - Seeing the World 👁️
- Chapter 6: RNNs - Understanding Sequences 📜

## Prerequisites

Before we begin, make sure you have:
- Python 3.8 or higher
- PyTorch
- NumPy
- Matplotlib

```bash
pip install torch torchvision numpy matplotlib pandas
```

## What's Inside This Folder?

### 📖 README Files
- **Main README** (this file) - Your journey guide

### 💻 Code Examples (`codes/`)
1. **01_tensors_basics.py** - Understanding PyTorch tensors
   - Creating tensors
   - Operations and transformations
   - GPU vs CPU tensors

2. **02_autograd_magic.py** - Automatic Differentiation
   - Understanding gradients
   - Backpropagation basics
   - Custom gradients

3. **03_neural_networks.py** - Building Networks
   - Using nn.Module
   - Creating custom layers
   - Forward and backward passes

4. **04_training_loop.py** - The Training Process
   - Complete training pipeline
   - Loss functions and optimizers
   - Model evaluation

5. **05_cnn_vision.py** - Image Classification
   - Building CNN architectures
   - Training on real datasets
   - Feature visualization

6. **06_rnn_sequences.py** - Sequence Models
   - RNN, LSTM, GRU
   - Text classification
   - Time series prediction

7. **07_transfer_learning.py** - Leveraging Pre-trained Models
   - Using torchvision models
   - Fine-tuning strategies
   - Feature extraction

8. **08_save_load_models.py** - Model Persistence
   - Saving and loading models
   - Checkpoints
   - Model serialization

9. **09_deployment_torchserve.py** - Production Ready
   - TorchServe basics
   - Model serving
   - API creation

10. **10_gpu_acceleration.py** - GPU Training
    - Moving tensors to GPU
    - CUDA fundamentals
    - Multi-GPU training

### 📊 Datasets (`datasets/`)
- Sample datasets for practice
- MNIST digit classification
- Custom generated data

### ⚠️ Production Issues (`Production_Issues.md`)
Real-world challenges:
- GPU memory management
- Model optimization for inference
- Numerical stability
- Distributed training
- Model serving

### 🎯 Exercises (`Exercises.md`)
Hands-on challenges:
- Beginner: Tensor operations
- Intermediate: Build a custom model
- Advanced: Implement attention mechanism

## Quick Start

Let's begin your PyTorch journey!

```python
import torch

# Your first tensor - like a multi-dimensional array
x = torch.tensor([1, 2, 3, 4, 5])
print(x)  # tensor([1, 2, 3, 4, 5])

# Operations feel natural!
y = x + 2
print(y)  # tensor([3, 4, 5, 6, 7])

# GPU acceleration? Just move it!
if torch.cuda.is_available():
    x_gpu = x.to('cuda')
    print(x_gpu)  # tensor([1, 2, 3, 4, 5], device='cuda:0')
```

## Learning Path

```
Week 1: Tensors, Autograd, and Basic Operations
    ↓
Week 2: Building Neural Networks and Training Loops
    ↓
Week 3: CNNs and Image Classification
    ↓
Week 4: RNNs and Sequence Models
    ↓
Week 5: Transfer Learning and Advanced Techniques
    ↓
Week 6: Production Deployment with TorchServe
```

## PyTorch vs TensorFlow

| Feature | PyTorch | TensorFlow |
|---------|---------|------------|
| Computation Graph | Dynamic (define-by-run) | Static (define-and-run) |
| Debugging | Easy (use pdb, print) | Harder (TF functions) |
| Community | Growing fast | Large, established |
| Production | TorchServe | TensorFlow Serving |
| Mobile | Limited | Better (TFLite) |

## Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| CUDA Out of Memory | Training crashes on GPU | Reduce batch size, clear cache |
| Gradient Not Computing | Weights not updating | Set `requires_grad=True` |
| Model Not Training | Loss stays same | Check learning rate, data |
| GPU Not Found | Can't use CUDA | Check GPU availability |

## Resources

- [PyTorch Official Docs](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning with PyTorch](https://pytorch.org/deep-learning-with-pytorch)

---

*Remember: In PyTorch, you have full control. Every operation is explicit, making it perfect for research and experimentation!* 🔥
