# Residual Networks (ResNet) - Interview Questions

## Foundational Questions (Definitions)

### Q1: What is a residual connection and why was it invented?

**Short Answer:** A residual connection (skip connection) adds the input to the output of a few layers, forming a "residual block." This enables training very deep networks by allowing gradients to flow directly through the connection, solving the degradation problem.

**Deep Dive:** ResNet innovation:

```
Output = F(x) + x
```
Where F(x) is the learned residual mapping.

**Why it works**:
- Degradation problem: Adding layers caused higher training error
- Identity mapping is the easiest to learn
- Gradients can flow directly through skip connection
- Network can learn identity when needed

**Key insight**: It's easier to learn identity + small perturbation than to learn identity from scratch.

---

### Q2: What is the degradation problem in deep networks?

**Short Answer:** The degradation problem is when deeper networks have higher training error than shallower ones, not due to overfitting but optimization difficulty. ResNet solves this by using skip connections.

**Deep Dive:** Observations:
- 56-layer plain network > 20-layer plain network in error
- This is NOT overfitting (training error is also worse)
- Caused by optimization difficulty, not insufficient capacity

**Why it happens**:
- Harder to optimize many stacked non-linear layers
- Gradient signal diminishes through many layers
- Network struggles to learn identity mapping

**ResNet solution**:
- Skip connections provide gradient highway
- Identity path is always learnable
- Deeper networks can be as trainable as shallow

---

### Q3: Explain the ResNet block architecture.

**Short Answer:** A ResNet block consists of two or three convolutional layers with skip connections. The block learns residual mapping F(x) while x is added directly to the output.

**Deep Dive:** Basic ResNet block:

```
x → Conv3x3 → BN → ReLU → Conv3x3 → BN → (+ x) → ReLU → out
```

**Two block types**:

| Type | Use | Structure |
|------|-----|-----------|
| Basic | ResNet-18, 34 | Two 3x3 convs |
| Bottleneck | ResNet-50, 101, 152 | 1x1 → 3x3 → 1x1 |

**Bottleneck design**:
```
1x1 conv: reduce dimension
3x3 conv: process
1x1 conv: restore dimension
```

This reduces parameters by ~3× while maintaining capacity.

---

## Applied Questions (How to Train)

### Q4: How do you implement a ResNet from scratch?

**Short Answer:** Build residual blocks with skip connections, stack them with increasing channels and decreasing spatial dimensions. Use bottleneck blocks for deeper networks.

**Deep Dive:** Implementation structure:

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip connection
        return F.relu(out)
```

---

### Q5: What is the difference between ResNet-18, 50, 101, and 152?

**Short Answer:** They differ in depth (number of layers) and block type. 18/34 use basic blocks; 50/101/152 use bottleneck blocks for efficiency. More layers = more capacity but also harder to train.

**Deep Dive:** ResNet variants:

| Model | Layers | Block Type | Parameters | Top-5 Error |
|-------|--------|------------|------------|--------------|
| ResNet-18 | 18 | Basic | 11.7M | ~30% |
| ResNet-34 | 34 | Basic | 21.8M | ~26% |
| ResNet-50 | 50 | Bottleneck | 25.6M | ~22% |
| ResNet-101 | 101 | Bottleneck | 44.5M | ~21% |
| ResNet-152 | 152 | Bottleneck | 60.2M | ~21% |

**Choosing**:
- 18: Quick experiments, edge devices
- 50: Good balance of accuracy/speed
- 101/152: Maximum accuracy, more compute

---

### Q6: How do ResNets compare to VGG networks?

**Short Answer:** ResNets are deeper (50-152 vs 16-19) but with fewer parameters due to bottleneck blocks. ResNets achieve better accuracy while being easier to train.

**Deep Dive:** Comparison:

| Aspect | ResNet | VGG |
|--------|--------|-----|
| Depth | 50-152 layers | 16-19 layers |
| Parameters | ~25M (50) | ~138M (VGG-16) |
| Error (ImageNet) | 22% vs 28% | Less efficient |
| Skip connections | Yes | No |
| Design | Modular | Simple stack |

**Why ResNet better**:
- Skip connections enable deeper networks
- Bottleneck blocks reduce parameters
- More efficient use of parameters

---

### Q7: How do you use transfer learning with ResNet?

**Short Answer:** Use pretrained ResNet as feature extractor or fine-tune. Replace final FC layer for your number of classes, optionally freeze early layers.

**Deep Dive:** Transfer learning strategies:

**1. Feature extraction** (small dataset):
```python
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False
# Replace classifier
model.fc = nn.Linear(512, num_classes)
```

**2. Fine-tuning** (moderate dataset):
```python
# Train entire model with lower LR
optimizer = Adam(model.parameters(), lr=1e-4)
```

**3. Discriminative learning rates**:
```python
# Lower LR for early layers
optimizer = Adam([
    {'params': model.conv1.parameters(), 'lr': 1e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
])
```

---

## Architectural Questions (Edge Cases and Trade-offs)

### Q8: What are the trade-offs between using more skip connections vs fewer?

**Short Answer:** More skip connections make training easier but add parameters and may reduce feature learning. Fewer skip connections increase training difficulty but potentially learn richer representations.

**Deep Dive:** Considerations:

| Factor | More Skips | Fewer Skips |
|--------|------------|--------------|
| Training | Easier, faster converge | Harder |
| Gradient flow | Better | May vanish |
| Parameters | Slightly more | Less |
| Representations | May be simpler | Richer |

**ResNet design**:
- Skip every 2-3 conv layers
- Bottleneck block has 2 convs + skip
- Works well in practice

---

### Q9: How does ResNet handle different image resolutions?

**Short Answer:** ResNet uses strided convolutions in first layer and transition blocks to handle different input sizes. Works well for various resolutions with minimal modification.

**Deep Dive:** Handling resolutions:

**Standard approach**:
- Input: 224×224
- First conv: stride 2, reduces to 112×112
- Multiple stages halve spatial dims
- Global average pool → FC

**For other sizes**:
- Smaller: Remove first stride, increase after pooling
- Larger: Works directly (computational cost increases)
- Fine-tuning may help

---

### Q10: What are the successors and improvements to ResNet?

**Short Answer:** ResNeXt uses grouped convolutions; DenseNet connects all layers; SENet adds channel attention; EfficientNet uses neural architecture search. Modern architectures often combine these ideas.

**Deep Dive:** ResNet evolution:

| Architecture | Innovation | Key Idea |
|--------------|-------------|-----------|
| ResNeXt | Grouped convolutions | More parallel paths |
| DenseNet | Dense connections | All-to-all connection |
| SENet | Squeeze-and-Excitation | Channel attention |
| EfficientNet | NAS + scaling | Compound scaling |
| EfficientNetV2 | Faster training | Smaller strides, fused |

**Current state**:
- Vision Transformers (ViT) now compete
- ConvNeXt brings modern tricks to ConvNets
- Hybrid approaches combine both

---

## Answer Key

| Category | Question | Key Concept |
|----------|----------|-------------|
| Foundational | Q1 | Skip connections |
| Foundational | Q2 | Degradation problem |
| Foundational | Q3 | ResNet block architecture |
| Applied | Q4 | Implementation from scratch |
| Applied | Q5 | ResNet variant comparison |
| Applied | Q6 | ResNet vs VGG |
| Applied | Q7 | Transfer learning |
| Architectural | Q8 | Skip connection trade-offs |
| Architectural | Q9 | Handling different resolutions |
| Architectural | Q10 | ResNet successors |
