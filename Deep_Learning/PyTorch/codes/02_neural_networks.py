"""
================================================================================
CHAPTER 2: NEURAL NETWORKS - BUILDING YOUR FIRST MODEL
================================================================================

Story: Imagine you're teaching a child to recognize cats and dogs. You show
them many pictures and say "this is a cat" or "this is a dog". Over time,
they learn to distinguish between them.

Neural networks learn the same way! They adjust their internal "weights"
to minimize mistakes. In this chapter, we'll build our first neural network
in PyTorch using the nn.Module class!

Key concepts:
- nn.Module: The base class for all neural networks
- Forward pass: Making predictions
- Backward pass: Learning from mistakes
- Loss functions: Measuring how wrong we are
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

print("=" * 70)
print("🧠 CHAPTER 2: BUILDING NEURAL NETWORKS")
print("=" * 70)

# ==============================================================================
# PART 1: THE MNIST DATASET - Our Training Data
# ==============================================================================

print("\n📚 PART 1: Loading MNIST Dataset")
print("-" * 50)

"""
MNIST is the "Hello World" of deep learning!
- 60,000 training images of handwritten digits (0-9)
- 10,000 test images
- Each image is 28x28 grayscale pixels
"""

# Define transforms (preprocessing)
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert to tensor [0, 1]
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
    ]
)

# Load datasets
train_set = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

test_set = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# Data loaders (batches)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

print(f"Training set: {len(train_set)} images")
print(f"Test set: {len(test_set)} images")
print(f"Number of batches (train): {len(train_loader)}")

# ==============================================================================
# PART 2: BUILDING THE NEURAL NETWORK
# ==============================================================================

print("\n\n🏗️ PART 2: Building Your Neural Network")
print("-" * 50)

"""
Architecture:
- Input: 28x28 = 784 neurons (flattened image)
- Hidden Layer 1: 128 neurons + ReLU
- Hidden Layer 2: 64 neurons + ReLU  
- Output: 10 neurons (digits 0-9) + LogSoftmax
"""


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Flatten layer - we'll handle this in forward()
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(28 * 28, 128)  # Input -> Hidden 1
        self.relu1 = nn.ReLU()  # Activation
        self.fc2 = nn.Linear(128, 64)  # Hidden 1 -> Hidden 2
        self.relu2 = nn.ReLU()  # Activation
        self.fc3 = nn.Linear(64, 10)  # Hidden 2 -> Output

        # We use LogSoftmax for numerical stability
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Flatten the input
        x = self.flatten(x)

        # Forward through layers
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        x = self.log_softmax(x)

        return x


# Create the model
model = NeuralNetwork()
print("Model created!")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# ==============================================================================
# PART 3: LOSS FUNCTION AND OPTIMIZER
# ==============================================================================

print("\n\n⚙️ PART 3: Loss Function and Optimizer")
print("-" * 50)

"""
Loss Function: NLLLoss (Negative Log Likelihood)
- Measures how different our predictions are from actual labels

Optimizer: Stochastic Gradient Descent (SGD)
- Updates weights to minimize loss
- learning_rate: How big steps to take
- momentum: Helps escape local minima
"""

# Loss function (for classification with LogSoftmax output)
criterion = nn.NLLLoss()

# Optimizer (SGD)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

print(f"Loss function: {criterion}")
print(f"Optimizer: {optimizer}")
print(f"Learning rate: 0.01")

# ==============================================================================
# PART 4: TRAINING THE MODEL
# ==============================================================================

print("\n\n🎓 PART 4: Training the Model")
print("-" * 50)

"""
The Training Loop:
1. Clear gradients (zero_grad())
2. Forward pass (model(input))
3. Calculate loss (criterion)
4. Backward pass (loss.backward())
5. Update weights (optimizer.step())
"""

num_epochs = 5

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # 1. Zero gradients
        optimizer.zero_grad()

        # 2. Forward pass
        output = model(data)

        # 3. Calculate loss
        loss = criterion(output, target)

        # 4. Backward pass
        loss.backward()

        # 5. Update weights
        optimizer.step()

        # Track statistics
        running_loss += loss.item()

        # Get predictions
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        # Print every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(
                f"  Epoch [{epoch+1}/{num_epochs}] "
                f"Batch [{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    # Calculate epoch statistics
    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={accuracy:.2f}%")

print("\n✅ Training complete!")

# ==============================================================================
# PART 5: EVALUATING THE MODEL
# ==============================================================================

print("\n\n📊 PART 5: Evaluating the Model")
print("-" * 50)

# Test the model
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# ==============================================================================
# PART 6: MAKING PREDICTIONS
# ==============================================================================

print("\n\n🔮 PART 6: Making Predictions")
print("-" * 50)

# Get a sample image
sample_data, sample_target = test_set[0]
sample_data = sample_data.unsqueeze(0)  # Add batch dimension

# Make prediction
with torch.no_grad():
    output = model(sample_data)
    probs = torch.exp(output)  # Convert log probabilities to probabilities
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_class].item()

print(f"Predicted class: {pred_class}")
print(f"Confidence: {confidence*100:.2f}%")
print(f"Actual class: {sample_target}")

# ==============================================================================
# EXERCISES
# ==============================================================================

print("\n\n" + "=" * 70)
print("🎯 PRACTICE EXERCISES")
print("=" * 70)

print(
    """
Exercise 1: Add More Layers
----------------------------
Add another hidden layer with 256 neurons.
Does training accuracy improve?

Exercise 2: Change Activation
------------------------------
Try using different activation functions:
- torch.nn.LeakyReLU()
- torch.nn.Tanh()
- torch.nn.Sigmoid()

Exercise 3: Different Optimizer
--------------------------------
Try different optimizers:
- torch.optim.Adam()
- torch.optim.RMSprop()
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
1. nn.Module is the base class for all neural networks
2. Define __init__() to set up layers
3. Define forward() to specify how data flows
4. Training loop: zero_grad → forward → loss → backward → step
5. Use nn.LogSoftmax + NLLLoss for classification
6. Use torch.max() to get predicted class

🎉 Congratulations! You've built and trained your first PyTorch neural network!
"""
)

print("\n✨ Chapter 2 Complete! Ready for more adventures?")
