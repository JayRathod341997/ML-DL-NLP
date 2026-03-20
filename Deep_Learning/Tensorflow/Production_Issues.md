# TensorFlow Production Issues and Solutions

> 🛠️ A comprehensive guide to common issues you'll face when deploying TensorFlow models to production

## Table of Contents

1. [Memory Issues](#memory-issues)
2. [Model Performance Issues](#model-performance-issues)
3. [Training Stability Issues](#training-stability-issues)
4. [Data Pipeline Issues](#data-pipeline-issues)
5. [Model Serving Issues](#model-serving-issues)
6. [Model Versioning Issues](#model-versioning-issues)
7. [Numerical Stability Issues](#numerical-stability-issues)

---

## 1. Memory Issues

### Problem: Out of Memory (OOM) During Training

**Symptoms:**
- Training crashes with "OOM when allocating tensor"
- Process gets killed unexpectedly
- Memory usage keeps growing

**Solutions:**

```python
# Solution 1: Reduce Batch Size
# Start smaller if OOM occurs
model.fit(x_train, y_train, batch_size=16)  # Try 16 instead of 32/64

# Solution 2: Enable Mixed Precision
# Uses float16 for computations, float32 for storage
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Solution 3: Clear Session
# Clear memory between experiments
tf.keras.backend.clear_session()

# Solution 4: Use Generators
# Don't load all data into memory
def data_generator():
    for batch in large_dataset:
        yield batch
```

### Problem: Memory Leak

**Symptoms:**
- Memory usage increases over time during training
- Model performance degrades after many epochs

**Solutions:**
```python
# Solution: Properly manage callbacks and data
class MemoryMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Clear any cached data
        tf.keras.backend.clear_session()

# Always close datasets properly
dataset = tf.data.Dataset.from_tensor_slices(...)
iterator = iter(dataset)
# Make sure to properly close when done
```

---

## 2. Model Performance Issues

### Problem: Slow Training

**Symptoms:**
- Training takes much longer than expected
- GPU utilization is low

**Solutions:**
```python
# Solution 1: Enable XLA Compilation
@tf.function(jit_compile=True)
def train_step(x, y):
    ...

# Solution 2: Use tf.data Properly
# BAD: Loading all data at once
dataset = tf.data.Dataset.from_tensor_slices((X, y))

# GOOD: Optimized pipeline
dataset = (
    tf.data.Dataset.from_tensor_slices((X, y))
    .shuffle(buffer_size=10000)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)  # Prefetch next batch while training
)

# Solution 3: Use GPU
# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Use GPU
    with tf.device('/GPU:0'):
        model.fit(...)
```

### Problem: Low Model Accuracy

**Symptoms:**
- Model doesn't learn
- Accuracy stuck at baseline (random)
- Validation accuracy much lower than training

**Solutions:**
```python
# Solution 1: Check Data Preprocessing
# Make sure data is normalized properly
X = (X - mean) / std  # Not just dividing by 255!

# Solution 2: Increase Model Capacity
model = tf.keras.Sequential([
    ...
    tf.keras.layers.Dense(256, activation='relu'),  # More neurons
    tf.keras.layers.Dense(256, activation='relu'),
    ...
])

# Solution 3: Learning Rate Tuning
# Try different learning rates
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Solution 4: Check for Data Leakage
# Make sure train/test sets are truly separate
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

## 3. Training Stability Issues

### Problem: NaN Loss (Numerical Instability)

**Symptoms:**
- Loss becomes NaN during training
- Model produces NaN predictions

**Solutions:**
```python
# Solution 1: Gradient Clipping
optimizer = tf.keras.optimizers.Adam(
    clipnorm=1.0  # Clip gradients by norm
    # OR
    clipvalue=0.5  # Clip gradients by value
)

# Solution 2: Reduce Learning Rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Solution 3: Use Label Smoothing
# Reduces overconfidence
loss = tf.keras.losses.CategoricalCrossentropy(
    label_smoothing=0.1
)

# Solution 4: Check for Log(0) or Division by Zero
# Add small epsilon
eps = 1e-7
output = tf.math.log(x + eps)  # Instead of tf.math.log(x)
```

### Problem: Gradient Vanishing/Exploding

**Symptoms:**
- Training doesn't progress
- Loss stays constant or explodes

**Solutions:**
```python
# Solution 1: Use Proper Initialization
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, kernel_initializer='he_normal'),
    ...
])

# Solution 2: Use Residual Connections
# For very deep networks
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
    
    def call(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.keras.activations.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.keras.layers.Add()([x, residual])  # Skip connection!
        return tf.keras.activations.relu(x)

# Solution 3: Use LSTM/GRU Instead of Simple RNN
# They have gating mechanisms to control gradient flow
```

---

## 4. Data Pipeline Issues

### Problem: Data Loading Bottleneck

**Symptoms:**
- Training is slow despite GPU being powerful
- CPU usage is very high during training

**Solutions:**
```python
# Solution 1: Use Efficient File Format
# Convert to TFRecord for faster loading
def create_tfrecord_dataset():
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    # Write to TFRecord
    with tf.io.TFRecordWriter('data.tfrecord') as writer:
        for image, label in dataset:
            feature = {
                'image': _bytes_feature(image.numpy()),
                'label': _int64_feature(label)
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

# Solution 2: Parallel Processing
dataset = (
    dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)
```

### Problem: Data Mismatch Between Training and Inference

**Symptoms:**
- Good validation accuracy but poor production performance
- Predictions are inconsistent

**Solutions:**
```python
# Solution: Ensure Preprocessing is Consistent
# BAD: Different preprocessing for train and inference
# GOOD: Create preprocessing layer
preprocessing = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Resizing(224, 224),
])

# Use same model for training and inference
model = tf.keras.Sequential([
    preprocessing,  # Include preprocessing in model!
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1)
])
```

---

## 5. Model Serving Issues

### Problem: Slow Inference

**Symptoms:**
- Predictions take too long
- High latency in production

**Solutions:**
```python
# Solution 1: Quantization
# Reduce model size and speed up inference
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

# Solution 2: Pruning
# Remove unimportant weights
pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model)

# Solution 3: Use TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']
interpreter.set_tensor(input_index, input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_index)
```

### Problem: Version Compatibility

**Symptoms:**
- Saved model doesn't load
- Errors when upgrading TensorFlow version

**Solutions:**
```python
# Solution: Use SavedModel Format with Versioning
import os

model_version = '001'
export_path = os.path.join('models', model_version)
model.save(export_path, save_format='tf')

# Always save metadata
model.save('model.h5', include_optimizer=False)  # For compatibility

# Load with specific version
model = tf.keras.models.load_model('models/001')
```

---

## 6. Model Versioning Issues

### Problem: Model Registry and Rollback

**Symptoms:**
- Can't track which model is in production
- Need to rollback to previous version

**Solutions:**
```python
# Solution: Implement Model Registry
import json
from datetime import datetime

class ModelRegistry:
    def __init__(self, base_path='models'):
        self.base_path = base_path
    
    def save_model(self, model, metrics, description=''):
        version = self._get_next_version()
        path = f"{self.base_path}/v{version}"
        
        model.save(path)
        
        metadata = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'description': description
        }
        
        with open(f"{path}/metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        return version
    
    def load_model(self, version):
        path = f"{self.base_path}/v{version}"
        return tf.keras.models.load_model(path)
```

---

## 7. Numerical Stability Issues

### Problem: Overflow/Underflow

**Symptoms:**
- Extreme values in predictions
- Loss becomes infinity

**Solutions:**
```python
# Solution: Add Clipping and Epsilon
class StableSoftmax(tf.keras.layers.Layer):
    def call(self, x):
        # Subtract max for numerical stability
        x = x - tf.reduce_max(x, axis=-1, keepdims=True)
        exp_x = tf.exp(x)
        return exp_x / tf.reduce_sum(exp_x, axis=-1, keepdims=True)

# Solution: Use log-sum-exp trick for cross-entropy
def stable_cross_entropy(y_true, y_pred):
    # Clip predictions to avoid log(0)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    return -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
```

---

## Quick Reference: Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `OOM when allocating tensor` | Not enough GPU memory | Reduce batch size, use mixed precision |
| `NaN loss` | Numerical instability | Clip gradients, reduce LR, add epsilon |
| `InvalidArgumentError` | Shape mismatch | Check input shapes |
| `Resource exhausted` | Memory leak | Clear session, use generators |

---

## Best Practices Summary

1. ✅ Always normalize data consistently
2. ✅ Use callbacks for monitoring (EarlyStopping, ModelCheckpoint)
3. ✅ Save models in SavedModel format
4. ✅ Test inference with sample inputs
5. ✅ Monitor model metrics in production
6. ✅ Implement proper error handling
7. ✅ Use version control for models
8. ✅ Document preprocessing steps

---

*Remember: Production issues are learning opportunities! Every problem you solve makes you a better ML engineer!* 🚀
