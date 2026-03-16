# CNN Exercises

## Exercise 1: Build a Simple CNN from Scratch

**Objective**: Create a CNN to classify handwritten digits using the MNIST dataset.

**Instructions**:
1. Load the MNIST dataset
2. Build a CNN with at least 2 convolutional layers
3. Add pooling layers and fully connected layers
4. Train the model for 10 epochs
5. Evaluate the model performance

**Expected Output**: Model accuracy > 95% on test set

**Starter Code**:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define your CNN architecture
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        # Your code here
    
    def forward(self, x):
        # Your code here
        return x

# Load data and train model
# Your code here
```

## Exercise 2: Transfer Learning with ResNet

**Objective**: Use a pre-trained ResNet model to classify cats vs dogs.

**Instructions**:
1. Download the Dogs vs Cats dataset from Kaggle
2. Load a pre-trained ResNet-18 model
3. Replace the final layer for binary classification
4. Freeze early layers and fine-tune the model
5. Achieve > 90% accuracy

**Dataset**: https://www.kaggle.com/c/dogs-vs-cats/data

**Starter Code**:
```python
import torchvision.models as models

# Load pre-trained ResNet
model = models.resnet18(pretrained=True)

# Modify final layer
model.fc = nn.Linear(model.fc.in_features, 2)

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False
```

## Exercise 3: Data Augmentation Implementation

**Objective**: Implement custom data augmentation techniques.

**Instructions**:
1. Create a custom dataset class
2. Implement at least 5 different augmentation techniques
3. Apply augmentations during training
4. Compare model performance with and without augmentation

**Augmentation Techniques to Implement**:
- Random rotation
- Random flipping
- Random cropping
- Brightness adjustment
- Gaussian noise addition

## Exercise 4: CNN Visualization

**Objective**: Visualize what features different layers of a CNN learn.

**Instructions**:
1. Load a pre-trained CNN (VGG or ResNet)
2. Extract feature maps from different layers
3. Visualize the feature maps for different input images
4. Analyze what patterns each layer detects

**Expected Output**: Grid of feature maps showing edge detection, texture detection, and object parts

## Exercise 5: Object Detection with CNN

**Objective**: Implement a simple object detection system.

**Instructions**:
1. Use a CNN to extract features from images
2. Implement a sliding window approach
3. Classify each window as containing an object or not
4. Apply non-maximum suppression to remove duplicate detections

**Dataset**: Use any object detection dataset (Pascal VOC, COCO subset)

## Exercise 6: CNN for Medical Image Classification

**Objective**: Build a CNN to classify medical images (e.g., X-rays).

**Instructions**:
1. Find a medical imaging dataset (CheXpert, COVID-19 X-ray)
2. Preprocess the images appropriately
3. Build a CNN suitable for medical image analysis
4. Implement proper validation strategy
5. Evaluate using appropriate metrics (precision, recall, F1-score)

**Considerations**:
- Class imbalance handling
- Data preprocessing for medical images
- Appropriate evaluation metrics

## Exercise 7: CNN Architecture Comparison

**Objective**: Compare performance of different CNN architectures.

**Instructions**:
1. Implement at least 3 different CNN architectures
2. Train all models on the same dataset
3. Compare performance, training time, and model size
4. Analyze trade-offs between different architectures

**Architectures to Compare**:
- Simple CNN (your own design)
- VGG-style network
- ResNet-style network

## Exercise 8: Adversarial Examples

**Objective**: Generate and defend against adversarial examples.

**Instructions**:
1. Train a CNN on any dataset
2. Generate adversarial examples using FGSM (Fast Gradient Sign Method)
3. Test model robustness against adversarial examples
4. Implement defense mechanisms (adversarial training, input preprocessing)

**Expected Output**: Analysis of model vulnerability and defense effectiveness

## Exercise 9: Real-time Object Recognition

**Objective**: Build a real-time object recognition system.

**Instructions**:
1. Use OpenCV for video capture
2. Implement a lightweight CNN for real-time inference
3. Process video frames and display predictions
4. Optimize for speed while maintaining accuracy

**Requirements**:
- Process at least 10 frames per second
- Display bounding boxes and labels
- Handle multiple objects in frame

## Exercise 10: CNN for Style Transfer

**Objective**: Implement neural style transfer using CNN features.

**Instructions**:
1. Load a pre-trained CNN (VGG19)
2. Extract content and style features
3. Optimize an image to match content and style
4. Create artistic images combining content and style

**Expected Output**: Images with content from one image and style from another

## Bonus Exercises

### Exercise B1: Custom Loss Functions
Implement custom loss functions for specific tasks:
- Focal loss for imbalanced datasets
- Triplet loss for face recognition
- Dice loss for segmentation

### Exercise B2: Model Compression
Implement techniques to compress CNN models:
- Pruning (remove unimportant connections)
- Quantization (reduce precision)
- Knowledge distillation (train small model from large model)

### Exercise B3: Attention Mechanisms
Add attention mechanisms to CNNs:
- Squeeze-and-Excitation blocks
- Spatial attention
- Self-attention layers

### Exercise B4: 3D CNNs
Implement 3D CNNs for video analysis:
- Process video as 3D volumes
- Extract spatiotemporal features
- Classify video actions or events

### Exercise B5: CNN for Time Series
Adapt CNNs for time series data:
- 1D convolutions for temporal data
- Multi-channel input for multiple sensors
- Predict future values or classify patterns

## Evaluation Criteria

For each exercise, consider:
1. **Correctness**: Does the implementation work as expected?
2. **Performance**: Is the model achieving good accuracy/speed?
3. **Code Quality**: Is the code well-structured and documented?
4. **Understanding**: Can you explain your implementation choices?
5. **Creativity**: Did you add any improvements or variations?

## Tips for Success

1. **Start Simple**: Begin with basic implementations before adding complexity
2. **Use Pre-trained Models**: Leverage existing models when possible
3. **Monitor Training**: Use validation metrics to track progress
4. **Experiment**: Try different hyperparameters and architectures
5. **Visualize**: Create plots and visualizations to understand your model
6. **Document**: Keep track of your experiments and results

## Resources

- PyTorch Documentation: https://pytorch.org/docs/
- TensorFlow Documentation: https://www.tensorflow.org/api_docs
- Papers with Code: https://paperswithcode.com/
- Kaggle Datasets: https://www.kaggle.com/datasets

Good luck with your CNN exercises! Remember to experiment, learn from failures, and build your understanding progressively.