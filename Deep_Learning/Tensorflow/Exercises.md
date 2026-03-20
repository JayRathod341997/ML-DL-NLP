# TensorFlow Exercises

> 🎯 Hands-on challenges to master TensorFlow!

## Exercise Overview

| Level | Topic | Difficulty | Est. Time |
|-------|-------|------------|-----------|
| 1 | Tensor Basics | ⭐ | 15 min |
| 2 | Simple Neural Network | ⭐⭐ | 30 min |
| 3 | CNN Image Classification | ⭐⭐⭐ | 45 min |
| 4 | RNN Text Generation | ⭐⭐⭐ | 45 min |
| 5 | Custom Training Loop | ⭐⭐⭐⭐ | 60 min |
| 6 | Transfer Learning | ⭐⭐⭐⭐⭐ | 90 min |

---

## Exercise 1: Tensor Basics ⭐

**Objective:** Master the building blocks of TensorFlow

### Task 1.1: Create and Manipulate Tensors
```python
# Create a tensor with values 1 to 20
# Reshape it to different dimensions
# Perform element-wise operations

# YOUR CODE HERE:
import tensorflow as tf

# Step 1: Create tensor with values 1-20
tensor = tf.constant(...)

# Step 2: Reshape to (4, 5)
reshaped = ...

# Step 3: Multiply by 2
multiplied = ...

# Step 4: Find max value
max_val = ...
```

### Task 1.2: Matrix Operations
```python
# Create two 3x3 matrices
# Perform matrix multiplication
# Calculate the determinant

# Expected output: Result of A @ B
```

**✅ Solution:** [View Solution](codes/solutions/exercise_1_solution.py)

---

## Exercise 2: Simple Neural Network ⭐⭐

**Objective:** Build a neural network to classify fashion items

### Challenge: Fashion MNIST Classifier

The Fashion MNIST dataset contains 10 categories of clothing:
- 0: T-shirt/top
- 1: Trouser
- 2: Pullover
- 3: Dress
- 4: Coat
- 5: Sandal
- 6: Shirt
- 7: Sneaker
- 8: Bag
- 9: Ankle boot

### Tasks:
1. Load the Fashion MNIST dataset
2. Build a neural network with at least 2 hidden layers
3. Train for 5 epochs
4. Achieve >85% accuracy on test set

```python
# Starter code:
import tensorflow as tf

# Load data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = ...

# Normalize
train_images = ...
test_images = ...

# Build model
model = tf.keras.Sequential([
    # YOUR CODE HERE
])

# Compile and train
model.compile(...)
model.fit(...)
```

### Bonus:
- Add a validation set
- Use EarlyStopping callback
- Visualize some predictions

---

## Exercise 3: CNN Image Classification ⭐⭐⭐

**Objective:** Build a CNN to distinguish between cats and dogs

### Challenge: Cats vs Dogs Classifier

### Tasks:
1. Load cats vs dogs dataset (or use tf.keras.datasets.cats_vs_dogs)
2. Build a CNN with at least 3 convolutional layers
3. Use data augmentation (rotation, flipping)
4. Train and achieve >80% accuracy

```python
# Architecture hint:
model = tf.keras.Sequential([
    # Conv Block 1
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Conv Block 2
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Conv Block 3
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Dense layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])
```

---

## Exercise 4: RNN Text Generation ⭐⭐⭐

**Objective:** Build a character-level RNN to generate text

### Challenge: Shakespeare Text Generator

### Story Context:
Imagine you're training an AI to write like Shakespeare! You'll teach it to predict the next character based on previous characters.

### Tasks:
1. Load Shakespeare text dataset
2. Create character-level encoding
3. Build an LSTM model
4. Generate new text!

```python
# Steps:
# 1. Load and preprocess text
shakespeare_url = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
shakespeare_file = tf.keras.utils.get_file(
    'shakespeare.txt', 
    shakespeare_url
)
with open(shakespeare_file, 'rb') as f:
    text = f.read().decode(encoding='utf-8')

# 2. Create vocabulary
vocab = sorted(set(text))

# 3. Create encoding
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

# 4. Create sequences
seq_length = 100
# ... create training sequences

# 5. Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(vocab), 256),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(len(vocab), activation='softmax')
])
```

---

## Exercise 5: Custom Training Loop ⭐⭐⭐⭐

**Objective:** Understand the training process deeply by implementing it from scratch

### Challenge: Custom Training for MNIST

Instead of using `model.fit()`, implement your own training loop!

### Tasks:
1. Build a simple model
2. Implement forward pass
3. Implement backpropagation manually (using GradientTape)
4. Track metrics manually
5. Compare with built-in training

```python
# Starter code:
import tensorflow as tf

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Load data
(train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255.0

# Custom training loop
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

# Run training
for epoch in range(5):
    ...
```

---

## Exercise 6: Transfer Learning ⭐⭐⭐⭐⭐

**Objective:** Use pre-trained models for new tasks

### Challenge: Flower Classifier with Transfer Learning

### Story:
You want to build a flower classifier but don't have enough data. 
Use a pre-trained model (MobileNetV2) and fine-tune it for your task!

### Tasks:
1. Load MobileNetV2 without top layers
2. Add custom classification head
3. Freeze base model initially
4. Train only the new layers
5. Unfreeze and fine-tune
6. Achieve >90% accuracy

```python
# Solution hint:
# Step 1: Load pre-trained model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Step 2: Freeze base model
base_model.trainable = False

# Step 3: Build model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(5, activation='softmax')  # 5 flower classes
])

# Step 4: Train only new layers
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=3)

# Step 5: Unfreeze and fine-tune
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), ...)
model.fit(train_dataset, epochs=5)
```

---

## Advanced Challenges

### Challenge A: Build a GAN
Generate fake MNIST digits using a Generative Adversarial Network!

### Challenge B: Object Detection
Use YOLO or SSD for real-time object detection!

### Challenge C: Neural Style Transfer
Apply artistic styles to images!

---

## Answer Key

Each exercise has a detailed solution in:
- `codes/solutions/exercise_1_solution.py`
- `codes/solutions/exercise_2_solution.py`
- And so on...

---

## Tips for Success

1. **Start Simple:** Don't try to build complex models immediately
2. **Debug:** Use `print()` and `model.summary()` to understand your model
3. **Patience:** Training takes time, especially with larger models
4. **Learn from Errors:** Read error messages carefully - they contain clues!
5. **Experiment:** Try different architectures, learning rates, batch sizes

---

*Remember: Every expert was once a beginner. Keep practicing!* 💪
