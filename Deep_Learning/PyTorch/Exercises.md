# PyTorch Exercises

> 🎯 Hands-on challenges to master PyTorch!

## Exercise Overview

| Level | Topic | Difficulty | Est. Time |
|-------|-------|------------|-----------|
| 1 | Tensor Basics | ⭐ | 15 min |
| 2 | Neural Networks | ⭐⭐ | 30 min |
| 3 | CNN Image Classification | ⭐⭐⭐ | 45 min |
| 4 | RNN Text Classification | ⭐⭐⭐ | 45 min |
| 5 | Custom Dataset | ⭐⭐⭐⭐ | 60 min |
| 6 | Transfer Learning | ⭐⭐⭐⭐⭐ | 90 min |

---

## Exercise 1: Tensor Basics ⭐

**Objective:** Master PyTorch tensors

### Task:
```python
import torch

# 1. Create a tensor with values from 1 to 10
# 2. Reshape to (2, 5)
# 3. Multiply by 2
# 4. Find max value
# 5. Calculate mean

# YOUR CODE HERE:
x = ...  # Step 1
x = ...  # Step 2
x = ...  # Step 3
max_val = ...  # Step 4
mean_val = ...  # Step 5

print(f"Result: max={max_val}, mean={mean_val:.2f}")
```

### Bonus: Try creating tensors on GPU!
```python
if torch.cuda.is_available():
    x_gpu = x.to('cuda')
    print(x_gpu.device)
```

---

## Exercise 2: Neural Networks ⭐⭐

**Objective:** Build a classifier for Fashion MNIST

### Challenge: Fashion MNIST Classifier

Fashion MNIST has 10 classes:
- T-shirt/top, Trouser, Pullover, Dress, Coat
- Sandal, Shirt, Sneaker, Bag, Ankle boot

### Tasks:
1. Load Fashion MNIST dataset
2. Build a neural network with 2+ hidden layers
3. Train for 5 epochs
4. Achieve >85% test accuracy

```python
# Starter code:
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = datasets.FashionMNIST(
    './data', train=True, download=True, transform=transform
)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# Define model
class FashionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # YOUR CODE HERE: Define layers
    
    def forward(self, x):
        # YOUR CODE HERE: Define forward pass

model = FashionClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(5):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1} complete")
```

---

## Exercise 3: CNN Image Classification ⭐⭐⭐

**Objective:** Build a CNN for CIFAR-10

### Challenge: CIFAR-10 Classifier

CIFAR-10 has 10 classes:
- Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

### Tasks:
1. Build a CNN with 3+ convolutional layers
2. Use max pooling between layers
3. Add dropout for regularization
4. Achieve >70% test accuracy

```python
# Architecture hint:
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Classifier
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 32 -> 16
        x = self.pool(torch.relu(self.conv2(x)))  # 16 -> 8
        x = self.pool(torch.relu(self.conv3(x)))  # 8 -> 4
        
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
```

---

## Exercise 4: RNN Text Classification ⭐⭐⭐

**Objective:** Build an RNN for sentiment analysis

### Challenge: IMDB Sentiment Classifier

### Tasks:
1. Create vocabulary from text
2. Build an LSTM model
3. Classify movie reviews as positive/negative

```python
# Hint: Use nn.Embedding and nn.LSTM

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Use the last hidden state
        output = self.fc(hidden[-1])
        return output
```

---

## Exercise 5: Custom Dataset ⭐⭐⭐⭐

**Objective:** Create a custom dataset for your own data

### Challenge: Custom Image Dataset

```python
from torch.utils.data import Dataset
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # Add your labels
```

---

## Exercise 6: Transfer Learning ⭐⭐⭐⭐⭐

**Objective:** Use pre-trained models for new tasks

### Challenge: Flower Classifier with MobileNet

```python
import torchvision.models as models

# Load pre-trained MobileNetV2
model = models.mobilenet_v2(pretrained=True)

# Freeze base model
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier = nn.Sequential(
    nn.Linear(1280, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 5)  # 5 flower classes
)

# Train only the new classifier
# Then optionally unfreeze and fine-tune
```

---

## Advanced Challenges

### Challenge A: Implement Attention Mechanism
Build a self-attention layer from scratch!

### Challenge B: Build a GAN
Generate fake MNIST digits!

### Challenge C: Deploy with TorchServe
Serve your model with TorchServe!

---

## Tips for Success

1. **Use Debug Mode:** Add print statements to understand tensor shapes
2. **Check GPU:** Use `.to('cuda')` for faster training
3. **Monitor Loss:** Watch for NaN values
4. **Start Simple:** Begin with small models and data
5. **Read Errors:** PyTorch error messages are helpful!

---

## Common PyTorch Patterns

```python
# Training Loop Template
model.train()  # Set to training mode

for data, target in train_loader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()  # Set to evaluation mode

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        # Calculate accuracy
```

---

*Remember: Practice makes perfect! Keep coding and you'll master PyTorch in no time!* 💪
