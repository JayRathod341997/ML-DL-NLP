# Convolutional Neural Network (CNN) - Explain Like I'm 5

## What is a CNN?

Imagine you’re looking at a LEGO picture.
Instead of staring at the whole picture at once, you look at **small parts**:
- edges
- corners
- simple shapes

A **CNN** is a neural network that learns to detect small patterns and then combines them to understand the whole image.

## Key Idea: Convolution (Sliding Window)

CNN uses a small “scanner” (a **filter / kernel**) that slides over an image:

```
Image (4x4)      Filter (2x2)
[....]           [..]
[....]   -->     [..]
[....]
[....]
```

The filter learns useful patterns (like horizontal/vertical lines).

## Where It’s Used

- Image classification (product defects, medical imaging)
- Object detection (retail shelf monitoring, traffic cameras)
- OCR / document understanding

## Benefits

- Learns spatial patterns well
- Fewer parameters than a fully-connected network on raw pixels

## Limitations

- Needs lots of labeled data for high accuracy (without transfer learning)
- Not ideal for long text sequences (Transformers are typically better)
- Training can be compute-heavy for large images

## Example in This Folder

We use a tiny synthetic “image” dataset stored as CSV:
- Dataset: `dataset/line_images_4x4.csv`
  - Each row is a 4x4 image flattened into 16 pixel values
  - Label: 0 = horizontal line, 1 = vertical line

## Enterprise-Level Example

In manufacturing quality control:
- Cameras capture product images on the assembly line
- A CNN detects defects (scratches, missing components)
- The system triggers alerts, rejects faulty items, and logs evidence for audits

