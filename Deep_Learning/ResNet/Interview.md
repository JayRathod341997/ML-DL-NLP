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
