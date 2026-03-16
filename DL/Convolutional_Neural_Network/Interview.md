# CNN - Interview Questions (with Answers)

## Basic

### Q1: What problem do CNNs solve well?
**Answer:** Learning spatial patterns in images (and other grid-like data such as spectrograms).

### Q2: What is a convolution?
**Answer:** A sliding filter operation that computes local dot-products over an input, producing feature maps.

### Q3: What is pooling?
**Answer:** Downsampling (e.g., max/avg pooling) to reduce spatial size and add translation robustness.

## Intermediate

### Q4: Why do CNNs use fewer parameters than fully-connected nets on images?
**Answer:** Parameter sharing: the same filter weights are used at every spatial location.

### Q5: What is a feature map?
**Answer:** The output of applying a convolution filter over an input.

## Advanced

### Q6: What is the receptive field?
**Answer:** The region of the input that influences a particular output activation; it grows with depth.

### Q7: What’s the difference between 1D, 2D, and 3D convolutions?
**Answer:** They convolve over 1, 2, or 3 spatial/temporal dimensions (e.g., text/time-series, images, videos).

