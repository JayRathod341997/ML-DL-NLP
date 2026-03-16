# CNN Datasets

This directory contains information about popular datasets used for training and testing Convolutional Neural Networks.

## Image Classification Datasets

### 1. MNIST Handwritten Digits
- **Description**: 70,000 grayscale images of handwritten digits (0-9)
- **Image Size**: 28x28 pixels
- **Classes**: 10 (digits 0-9)
- **Usage**: Beginner CNN projects, digit recognition
- **Download**: Included in most deep learning frameworks (PyTorch, TensorFlow)

### 2. CIFAR-10
- **Description**: 60,000 32x32 color images in 10 classes
- **Image Size**: 32x32 pixels
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Usage**: Standard benchmark for image classification
- **Download**: https://www.cs.toronto.edu/~kriz/cifar.html

### 3. CIFAR-100
- **Description**: 60,000 32x32 color images in 100 classes
- **Image Size**: 32x32 pixels
- **Classes**: 100 (20 superclasses with 5 subclasses each)
- **Usage**: More challenging classification task
- **Download**: https://www.cs.toronto.edu/~kriz/cifar.html

### 4. ImageNet
- **Description**: Over 14 million labeled images
- **Image Size**: Various sizes (typically resized to 224x224)
- **Classes**: 1000+ object categories
- **Usage**: Large-scale image classification, transfer learning
- **Download**: http://www.image-net.org/

## Object Detection Datasets

### 5. Pascal VOC
- **Description**: 20 object categories with bounding box annotations
- **Images**: ~11,000 images
- **Classes**: 20 object categories
- **Usage**: Object detection, segmentation
- **Download**: http://host.robots.ox.ac.uk/pascal/VOC/

### 6. COCO (Common Objects in Context)
- **Description**: Complex everyday scenes with object detection, segmentation, and captioning
- **Images**: Over 330,000 images
- **Classes**: 80 object categories
- **Usage**: Object detection, instance segmentation, keypoint detection
- **Download**: https://cocodataset.org/

### 7. Open Images Dataset
- **Description**: ~9 million images with object annotations
- **Images**: 9+ million images
- **Classes**: 600+ object categories
- **Usage**: Large-scale object detection and classification
- **Download**: https://storage.googleapis.com/openimages/web/index.html

## Medical Imaging Datasets

### 8. CheXpert
- **Description**: Chest X-rays with pathology labels
- **Images**: 224,316 chest radiographs
- **Classes**: 14 pathologies
- **Usage**: Medical image classification
- **Download**: https://stanfordmlgroup.github.io/competitions/chexpert/

### 9. COVID-19 Image Data Collection
- **Description**: Chest X-rays and CT scans for COVID-19 detection
- **Images**: Various sizes and modalities
- **Classes**: COVID-19 positive/negative, other conditions
- **Usage**: Medical diagnosis, pandemic response
- **Download**: https://github.com/ieee8023/covid-chestxray-dataset

### 10. ISIC (Skin Lesion) Archive
- **Description**: Dermoscopic images for skin cancer detection
- **Images**: 28,000+ dermoscopic images
- **Classes**: Melanoma, nevus, seborrheic keratosis, etc.
- **Usage**: Medical diagnosis, cancer detection
- **Download**: https://challenge.isic-archive.com/

## Specialized Datasets

### 11. Fashion-MNIST
- **Description**: 70,000 grayscale images of fashion items
- **Image Size**: 28x28 pixels
- **Classes**: 10 fashion categories
- **Usage**: Alternative to MNIST for more challenging classification
- **Download**: https://github.com/zalandoresearch/fashion-mnist

### 12. SVHN (Street View House Numbers)
- **Description**: House numbers from Google Street View
- **Images**: Over 600,000 digit images
- **Classes**: 10 digits (0-9)
- **Usage**: Real-world digit recognition
- **Download**: http://ufldl.stanford.edu/housenumbers/

### 13. Caltech-101/256
- **Description**: Images of objects in 101/256 categories
- **Images**: 9,000+ images (101) / 30,000+ images (256)
- **Classes**: 101 or 256 object categories
- **Usage**: Object recognition research
- **Download**: http://www.vision.caltech.edu/Image_Datasets/

### 14. Flowers-102
- **Description**: 102 flower categories commonly occurring in the United Kingdom
- **Images**: 8,189 images
- **Classes**: 102 flower categories
- **Usage**: Fine-grained classification
- **Download**: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

## Face Recognition Datasets

### 15. LFW (Labeled Faces in the Wild)
- **Description**: Face photographs for unconstrained face recognition
- **Images**: 13,000+ face images
- **Classes**: 5,749 people
- **Usage**: Face recognition, verification
- **Download**: http://vis-www.cs.umass.edu/lfw/

### 16. CelebA
- **Description**: Large-scale face attributes dataset
- **Images**: 202,599 face images
- **Classes**: 40 binary attributes, 10,177 identities
- **Usage**: Face attribute recognition, generation
- **Download**: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

## How to Use These Datasets

### Automatic Download
Many datasets can be downloaded automatically using deep learning frameworks:

```python
# PyTorch example
from torchvision import datasets
train_data = datasets.MNIST(root='./data', train=True, download=True)

# TensorFlow example
import tensorflow_datasets as tfds
dataset = tfds.load('cifar10', split='train')
```

### Manual Download
For datasets not included in frameworks:

1. Visit the download links provided above
2. Follow the dataset's terms of use
3. Download the data to your local machine
4. Use appropriate data loaders for your framework

### Data Preprocessing
Common preprocessing steps:

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to model input size
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(           # Normalize with ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

## Dataset Selection Guidelines

### For Beginners:
- **MNIST**: Simple digit recognition
- **Fashion-MNIST**: Slightly more complex classification
- **CIFAR-10**: Standard color image classification

### For Intermediate:
- **CIFAR-100**: More classes, more challenging
- **SVHN**: Real-world digit recognition
- **Flowers-102**: Fine-grained classification

### For Advanced:
- **ImageNet**: Large-scale classification
- **COCO**: Object detection and segmentation
- **Medical datasets**: Domain-specific applications

### For Research:
- **Custom datasets**: Specific to your research question
- **Large-scale datasets**: ImageNet, Open Images
- **Specialized datasets**: Medical, satellite, etc.

## Important Notes

1. **License Compliance**: Always check and comply with dataset licenses
2. **Data Privacy**: Be mindful of privacy concerns, especially with face datasets
3. **Bias Awareness**: Datasets may contain biases that affect model performance
4. **Data Quality**: Verify data quality and consistency before training
5. **Computational Resources**: Larger datasets require more storage and compute power

## Creating Your Own Dataset

If you need a custom dataset:

1. **Data Collection**: Gather images relevant to your task
2. **Annotation**: Label images appropriately (classification, bounding boxes, etc.)
3. **Splitting**: Create train/validation/test splits
4. **Preprocessing**: Standardize image formats and sizes
5. **Documentation**: Document dataset characteristics and usage

For more information on dataset creation and management, refer to the specific dataset documentation and best practices in the machine learning community.