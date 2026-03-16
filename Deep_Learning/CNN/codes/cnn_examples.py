"""
Convolutional Neural Networks (CNN) Examples
This file demonstrates various CNN concepts and implementations.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=== Convolutional Neural Networks (CNN) Examples ===\n")

# Example 1: Simple CNN for Digit Recognition
print("1. Simple CNN for Digit Recognition")
print("-" * 40)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # After 2 pooling layers: 28->14->7
        self.fc2 = nn.Linear(128, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # First conv block
        x = self.pool(self.relu(self.conv1(x)))
        # Second conv block
        x = self.pool(self.relu(self.conv2(x)))
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create sample data for demonstration
def create_sample_data():
    # Create synthetic image data
    np.random.seed(42)
    X = np.random.randn(1000, 1, 28, 28)  # 1000 samples, 1 channel, 28x28
    y = np.random.randint(0, 10, 1000)     # 10 classes
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train the CNN
def train_cnn():
    X_train, X_test, y_train, y_test = create_sample_data()
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model, loss, and optimizer
    model = SimpleCNN(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(5):  # Reduced epochs for demo
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')
    
    # Test the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = accuracy_score(y_test, predicted.numpy())
        print(f'Test Accuracy: {accuracy:.4f}')
    
    return model

# Run the simple CNN example
model = train_cnn()
print()

# Example 2: CNN Visualization
print("2. CNN Feature Visualization")
print("-" * 30)

def visualize_cnn_features():
    # Create a simple filter to visualize what CNNs detect
    filter_kernel = np.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]])  # Edge detection filter
    
    # Create a sample image
    sample_image = np.random.randn(10, 10)
    sample_image[3:7, 3:7] = 2.0  # Add a bright square
    
    # Apply convolution manually to show the concept
    def apply_convolution(image, kernel):
        h, w = image.shape
        k_h, k_w = kernel.shape
        result = np.zeros((h - k_h + 1, w - k_w + 1))
        
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = np.sum(image[i:i+k_h, j:j+k_w] * kernel)
        
        return result
    
    # Apply the filter
    filtered_image = apply_convolution(sample_image, filter_kernel)
    
    print("Original Image (10x10):")
    print(np.round(sample_image, 2))
    print("\nEdge Detection Filter:")
    print(filter_kernel)
    print("\nFiltered Image (Edge Detection):")
    print(np.round(filtered_image, 2))
    
    return sample_image, filtered_image

original, filtered = visualize_cnn_features()
print()

# Example 3: Transfer Learning with Pre-trained CNN
print("3. Transfer Learning Example")
print("-" * 28)

def demonstrate_transfer_learning():
    print("Transfer Learning Concept:")
    print("- Use pre-trained models (e.g., ResNet, VGG) trained on ImageNet")
    print("- Freeze early layers (general features)")
    print("- Fine-tune later layers for specific task")
    print("- Add new classifier head")
    print()
    
    # Show model architecture concept
    print("Typical Transfer Learning Architecture:")
    print("Input -> [Pre-trained CNN Layers] -> [New Classifier Layers] -> Output")
    print()
    print("Benefits:")
    print("- Faster training")
    print("- Better performance with less data")
    print("- Leverage learned features from large datasets")

demonstrate_transfer_learning()
print()

# Example 4: Data Augmentation
print("4. Data Augmentation Techniques")
print("-" * 32)

def demonstrate_augmentation():
    print("Common Data Augmentation Techniques:")
    print("1. Geometric Transformations:")
    print("   - Rotation (±10-30 degrees)")
    print("   - Translation (shift left/right/up/down)")
    print("   - Scaling (zoom in/out)")
    print("   - Flipping (horizontal/vertical)")
    print("   - Cropping (random crops)")
    print()
    print("2. Color/Intensity Adjustments:")
    print("   - Brightness adjustment")
    print("   - Contrast adjustment")
    print("   - Saturation adjustment")
    print("   - Noise addition")
    print()
    print("3. Advanced Techniques:")
    print("   - Mixup (blend images)")
    print("   - Cutout (randomly mask regions)")
    print("   - CutMix (replace regions with patches from other images)")
    print()
    print("Example PyTorch transforms:")
    print("""
    transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    """)

demonstrate_augmentation()
print()

# Example 5: CNN Architectures Comparison
print("5. Famous CNN Architectures")
print("-" * 28)

def compare_architectures():
    architectures = {
        "LeNet-5 (1998)": {
            "Purpose": "Handwritten digit recognition",
            "Layers": "7 layers",
            "Innovation": "First successful CNN",
            "Params": "~60K"
        },
        "AlexNet (2012)": {
            "Purpose": "ImageNet classification",
            "Layers": "8 layers", 
            "Innovation": "ReLU, Dropout, GPU training",
            "Params": "~60M"
        },
        "VGG (2014)": {
            "Purpose": "ImageNet classification",
            "Layers": "16-19 layers",
            "Innovation": "Small 3x3 filters throughout",
            "Params": "~138M"
        },
        "ResNet (2015)": {
            "Purpose": "ImageNet classification", 
            "Layers": "18-152 layers",
            "Innovation": "Residual connections",
            "Params": "~25M (ResNet-50)"
        },
        "Inception (2014)": {
            "Purpose": "ImageNet classification",
            "Layers": "22 layers",
            "Innovation": "Multi-scale convolutions",
            "Params": "~23M"
        }
    }
    
    for name, details in architectures.items():
        print(f"\n{name}:")
        for key, value in details.items():
            print(f"  {key}: {value}")

compare_architectures()
print()

# Example 6: CNN Math - Output Size Calculation
print("6. CNN Output Size Calculation")
print("-" * 31)

def calculate_output_size():
    def get_output_size(input_size, filter_size, padding, stride):
        return (input_size - filter_size + 2 * padding) // stride + 1
    
    examples = [
        {"input": 32, "filter": 5, "padding": 1, "stride": 2},
        {"input": 28, "filter": 3, "padding": 0, "stride": 1},
        {"input": 224, "filter": 7, "padding": 3, "stride": 2},
    ]
    
    print("Output Size Formula: (W - F + 2P) / S + 1")
    print("Where: W=input size, F=filter size, P=padding, S=stride")
    print()
    
    for i, example in enumerate(examples, 1):
        output = get_output_size(
            example["input"], 
            example["filter"], 
            example["padding"], 
            example["stride"]
        )
        print(f"Example {i}: Input={example['input']}, Filter={example['filter']}, "
              f"Padding={example['padding']}, Stride={example['stride']}")
        print(f"  Output Size: {output}")
        print()

calculate_output_size()

# Example 7: Common CNN Problems and Solutions
print("7. Common CNN Problems and Solutions")
print("-" * 36)

def common_problems():
    problems = {
        "Overfitting": {
            "Symptoms": "High training accuracy, low validation accuracy",
            "Solutions": [
                "Data augmentation",
                "Dropout layers",
                "L2 regularization", 
                "Early stopping",
                "Transfer learning"
            ]
        },
        "Underfitting": {
            "Symptoms": "Low training and validation accuracy",
            "Solutions": [
                "Increase model complexity",
                "Train longer",
                "Reduce regularization",
                "Better architecture"
            ]
        },
        "Vanishing Gradients": {
            "Symptoms": "Very slow training, early layers don't learn",
            "Solutions": [
                "Residual connections",
                "Batch normalization",
                "Better initialization",
                "Gradient clipping"
            ]
        },
        "Memory Issues": {
            "Symptoms": "Out of memory errors during training",
            "Solutions": [
                "Reduce batch size",
                "Use mixed precision training",
                "Model pruning",
                "Gradient checkpointing"
            ]
        }
    }
    
    for problem, details in problems.items():
        print(f"\n{problem}:")
        print(f"  Symptoms: {details['Symptoms']}")
        print(f"  Solutions:")
        for solution in details['Solutions']:
            print(f"    - {solution}")

common_problems()

print("\n" + "="*50)
print("CNN Examples Complete!")
print("Key Takeaways:")
print("1. CNNs excel at image processing through convolution operations")
print("2. Transfer learning can significantly improve performance")
print("3. Data augmentation helps prevent overfitting")
print("4. Architecture choice depends on task complexity and constraints")
print("5. Proper training techniques are crucial for good performance")