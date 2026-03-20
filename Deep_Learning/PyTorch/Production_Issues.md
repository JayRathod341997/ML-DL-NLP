# PyTorch Production Issues and Solutions

> 🛠️ A comprehensive guide to common issues you'll face when deploying PyTorch models to production

## Table of Contents

1. [GPU and Memory Issues](#gpu-and-memory-issues)
2. [Model Training Issues](#model-training-issues)
3. [Data Pipeline Issues](#data-pipeline-issues)
4. [Model Serving Issues](#model-serving-issues)
5. [Numerical Stability Issues](#numerical-stability-issues)
6. [Distributed Training Issues](#distributed-training-issues)
7. [Model Optimization Issues](#model-optimization-issues)

---

## 1. GPU and Memory Issues

### Problem: CUDA Out of Memory

**Symptoms:**
- RuntimeError: CUDA out of memory
- Training crashes on GPU

**Solutions:**
```python
# Solution 1: Reduce Batch Size
train_loader = DataLoader(dataset, batch_size=16)  # Try smaller batch

# Solution 2: Clear CUDA Cache
import torch
torch.cuda.empty_cache()

# Solution 3: Delete Unused Variables
del variable_name
torch.cuda.empty_cache()

# Solution 4: Use Gradient Checkpointing
# Trades compute for memory
model = torch.utils.checkpoint.checkpoint_sequential(layers, 2, input)

# Solution 5: Mixed Precision Training
# Uses float16 for most operations
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

for data, target in train_loader:
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Problem: GPU Not Available

**Symptoms:**
- Can't find CUDA devices
- Model runs very slowly on CPU

**Solutions:**
```python
# Solution 1: Check GPU Availability
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Solution 2: Set Device Explicitly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)

# Solution 3: Check PyTorch Version
print(torch.__version__)
# Make sure it was installed with CUDA support
```

---

## 2. Model Training Issues

### Problem: Model Not Learning

**Symptoms:**
- Loss stays constant
- Accuracy doesn't improve

**Solutions:**
```python
# Solution 1: Check Learning Rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Try smaller LR

# Solution 2: Check Data Preprocessing
# Make sure data is normalized correctly
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Mean and std
])

# Solution 3: Check for NaN in Gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"NaN in gradient: {name}")

# Solution 4: Use Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
```

### Problem: Training is Slow

**Symptoms:**
- Epochs take too long
- GPU utilization is low

**Solutions:**
```python
# Solution 1: Use DataLoader with Multiple Workers
train_loader = DataLoader(
    dataset, 
    batch_size=64,
    num_workers=4,      # Parallel data loading
    pin_memory=True      # Faster GPU transfer
)

# Solution 2: Use torch.no_grad() for Inference
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)

# Solution 3: Use @torch.jit.script for Deployment
@torch.jit.script
def forward_function(x):
    return model(x)

# Solution 4: Enable cuDNN Benchmarking
torch.backends.cudnn.benchmark = True
```

---

## 3. Data Pipeline Issues

### Problem: Data Loading Bottleneck

**Symptoms:**
- GPU waits for data
- CPU usage is high

**Solutions:**
```python
# Solution: Optimize DataLoader
train_loader = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True,
    num_workers=4,           # Parallel workers
    pin_memory=True,         # Faster GPU transfer
    prefetch_factor=2,       # Prefetch batches
    persistent_workers=True  # Keep workers alive
)
```

### Problem: Data Augmentation Issues

**Symptoms:**
- Training improves but test doesn't
- Model overfits to augmentations

**Solutions:**
```python
# Solution: Use Proper Augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

---

## 4. Model Serving Issues

### Problem: Slow Inference in Production

**Symptoms:**
- High latency predictions
- Server timeout errors

**Solutions:**
```python
# Solution 1: TorchScript
model.eval()
model_scripted = torch.jit.script(model)
model_scripted.save('model.pt')

# Solution 2: Quantization
model_quantized = torch.quantization.quantize_dynamic(
    model, 
    {nn.Linear, nn.Conv2d},
    dtype=torch.qint8
)

# Solution 3: ONNX Export
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11
)
```

### Problem: Model Version Mismatch

**Symptoms:**
- Can't load saved model
- Different results than expected

**Solutions:**
```python
# Solution: Save Complete State
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pt')

# Load with version check
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 5. Numerical Stability Issues

### Problem: NaN Loss

**Symptoms:**
- Loss becomes NaN
- Gradients explode

**Solutions:**
```python
# Solution 1: Gradient Clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Solution 2: Use Softmax with Log
# Instead of:
output = torch.softmax(x, dim=1)
loss = -torch.log(output[target])

# Use:
output = torch.log_softmax(x, dim=1)
loss = nn.NLLLoss()(output, target)

# Solution 3: Add Epsilon
eps = 1e-7
output = torch.clamp(output, min=eps)
```

### Problem: Gradient Vanishing

**Symptoms:**
- Early layers don't update
- Model doesn't learn

**Solutions:**
```python
# Solution 1: Use Residual Connections
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    
    def forward(self, x):
        residual = x
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        x += residual  # Skip connection!
        return torch.relu(x)

# Solution 2: Use LSTM/GRU Instead of RNN
# They have gating mechanisms to control gradient flow
```

---

## 6. Distributed Training Issues

### Problem: Multi-GPU Training Errors

**Symptoms:**
- Only one GPU is used
- NCCL errors

**Solutions:**
```python
# Solution: Use DistributedDataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
model = DDP(model, device_ids=[local_rank])

# Use distributed sampler
train_sampler = DistributedSampler(train_set)
train_loader = DataLoader(train_set, sampler=train_sampler)
```

---

## 7. Model Optimization Issues

### Problem: Model Size Too Large

**Symptoms:**
- Slow inference
- High memory usage

**Solutions:**
```python
# Solution 1: Pruning
import torch.nn.utils.prune as prune

# Prune 50% of connections
prune.l1_unstructured(model.fc1, name='weight', amount=0.5)

# Solution 2: Knowledge Distillation
# Train smaller model from larger one
# See PyTorch knowledge distillation tutorials

# Solution 3: Efficient Architectures
# Use MobileNet, EfficientNet instead of ResNet
```

---

## Quick Reference: Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| CUDA OOM | Not enough GPU memory | Reduce batch, clear cache |
| RuntimeError | Shape mismatch | Check input dimensions |
| NaN in loss | Numerical instability | Clip gradients, reduce LR |
| No CUDA | GPU not available | Check CUDA installation |

---

## Best Practices for Production

1. ✅ Always use `model.eval()` for inference
2. ✅ Use `torch.no_grad()` for inference
3. ✅ Save complete model checkpoints
4. ✅ Test with production data format
5. ✅ Monitor model metrics in production
6. ✅ Implement proper error handling
7. ✅ Version your models
8. ✅ Use TorchScript for deployment

---

*Remember: Production issues are part of every ML engineer's journey. Keep learning and improving!* 🚀
