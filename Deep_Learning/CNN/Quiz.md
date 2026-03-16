# Convolutional Neural Networks - Quiz

## True/False Questions

### Question 1
**Statement:** In a CNN, each neuron in a convolutional layer is connected to all neurons in the previous layer.

**Answer:** False

**Explanation:** This is true for fully connected (dense) layers, not convolutional layers. In CNNs, each neuron connects only to a local region called the receptive field, which is a key feature enabling efficient feature extraction and parameter sharing.

---

### Question 2
**Statement:** Increasing the number of filters in a convolutional layer always improves model accuracy.

**Answer:** False

**Explanation:** More filters increase model capacity but also increase parameters, computational cost, and risk of overfitting. There's a trade-off between model capacity and generalization. The optimal number depends on dataset size, task complexity, and available compute.

---

### Question 3
**Statement:** Batch normalization can only be applied during training, not inference.

**Answer:** False

**Explanation:** BatchNorm uses moving average statistics (computed during training) at inference time, not mini-batch statistics. It always normalizes inputs, but the statistics source differs between training and evaluation modes.

---

### Question 4
**Statement:** Pooling layers help provide translation invariance to CNNs.

**Answer:** True

**Explanation:** Pooling (especially max pooling) reduces the sensitivity to exact feature positions. If a feature shifts slightly within the pooling window, the output remains the same. This provides some degree of translation invariance.

---

### Question 5
**Statement:** Transfer learning always requires fine-tuning the entire pre-trained network.

**Answer:** False

**Explanation:** For small datasets, you can freeze all pre-trained layers and only train a new classifier head. This is called feature extraction. Full fine-tuning is only recommended with sufficient data to avoid catastrophic forgetting.

---

## Multiple Choice Questions

### Question 6
What is the receptive field of a neuron in a CNN?

A) The number of filters in the layer  
B) The input region that affects the neuron's output  
C) The size of the output feature map  
D) The number of output classes

**Answer:** B

**Explanation:** The receptive field is the region of the input image that influences a particular neuron's output. Early layers have small receptive fields (detecting edges), deeper layers have larger receptive fields (detecting entire objects).

---

### Question 7
Which layer type is most commonly used to reduce spatial dimensions in CNNs?

A) Dropout  
B) Batch Normalization  
C) Pooling  
D) Fully Connected

**Answer:** C

**Explanation:** Pooling layers (max pooling or average pooling) reduce spatial dimensions by aggregating values in local windows. This decreases computational load and provides spatial invariance. Strided convolutions can also reduce dimensions.

---

### Question 8
What is the purpose of zero-padding in convolutional layers?

A) To increase the number of parameters  
B) To preserve spatial dimensions after convolution  
C) To speed up training  
D) To normalize the output

**Answer:** B

**Explanation:** Zero-padding adds zeros around the input border, allowing the output feature map to maintain (or control) spatial dimensions. Without padding, each convolution reduces dimensions by (kernel_size - 1). Padding="same" preserves input dimensions.

---

### Question 9
Which of the following is NOT a benefit of using pre-trained models for transfer learning?

A) Reduced training time  
B) Better generalization with small datasets  
C) Automatic feature extraction without any training  
D) Leveraging knowledge from large datasets

**Answer:** C

**Explanation:** While pre-trained models provide good initial features, they typically require some fine-tuning for the new task. Using them "without any training" would only work if the tasks are identical. Transfer learning reduces training time (A), helps with small datasets (B), and leverages ImageNet knowledge (D).

---

### Question 10
What happens during data augmentation in CNN training?

A) The test dataset is modified  
B) Training images are artificially transformed to increase variety  
C) The model architecture is changed  
D) The loss function is modified

**Answer:** B

**Explanation:** Data augmentation applies random transformations (rotation, flipping, cropping, color changes) to training images, creating modified versions. This effectively increases dataset size and helps the model learn invariance to these transformations, improving generalization.

---

## Answer Key

| Question | Type | Answer | Key Concept |
|----------|------|--------|--------------|
| Q1 | T/F | False | Local connectivity in CNNs |
| Q2 | T/F | False | Filter count trade-offs |
| Q3 | T/F | False | BatchNorm at inference |
| Q4 | T/F | True | Translation invariance via pooling |
| Q5 | T/F | False | Transfer learning strategies |
| Q6 | MCQ | B | Receptive field definition |
| Q7 | MCQ | C | Pooling for downsampling |
| Q8 | MCQ | B | Zero-padding purpose |
| Q9 | MCQ | C | Transfer learning limitations |
| Q10 | MCQ | B | Data augmentation effects |

## Brief Explanations

1. **Q1 (False):** CNNs use local connectivity, not full connectivity.
2. **Q2 (False):** More filters mean more parameters and potential overfitting.
3. **Q3 (False):** BatchNorm uses moving averages at inference time.
4. **Q4 (True):** Pooling provides translation invariance.
5. **Q5 (False):** Transfer learning can freeze base layers.
6. **Q6 (B):** Receptive field is the input region affecting output.
7. **Q7 (C):** Pooling reduces spatial dimensions.
8. **Q8 (B):** Zero-padding preserves spatial size.
9. **Q9 (C):** Pre-trained models still need fine-tuning.
10. **Q10 (B):** Augmentation transforms training images.
