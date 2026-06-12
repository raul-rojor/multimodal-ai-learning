# ============================================
# PART 1: MORNING - WHY CNNs? (30 min)
# ============================================

print("\n" + "=" * 60)
print("MORNING PART 1: THE IMAGE PROBLEM")
print("=" * 60)

"""
WHY IMAGES NEED CNNs
File: why_cnns.py

THE PROBLEM WITH RAW PIXELS:
  A 224×224 color image has 224 × 224 × 3 = 150,528 numbers.
  If you connected every pixel to every neuron (like a normal neural network):
    - First layer would have 150,528 × 512 ≈ 77 million weights
    - Too many to train, too much memory, overfits easily
  
THE CNN SOLUTION:
  - Same filter slides across entire image (shares weights)
  - Dramatically reduces parameters
  - Detects patterns regardless of position (translation invariance)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

print("=" * 50)
print("IMAGE AS NUMBERS")
print("=" * 50)

# Create a tiny 5x5 grayscale image (values 0-255)
image = np.array([
    [10, 10, 10, 10, 10],
    [10, 10, 10, 10, 10],
    [10, 10, 200, 10, 10],   # bright spot in center
    [10, 10, 10, 10, 10],
    [10, 10, 10, 10, 10]
])

print("Tiny 5x5 image (grayscale, 0-255):")
print(image)

# QUESTION: How would a normal neural network process this?
# ANSWER: Flatten to 25 numbers, losing all spatial structure

flattened = image.flatten()
print(f"\nFlattened to 1D: {flattened}")
print("Spatial information (pixel positions) is now lost!")

# QUESTION: How does a CNN preserve spatial information?
# ANSWER: Keeps 2D structure, slides filters across it

print("\n" + "=" * 50)
print("WHY CNNS WORK")
print("=" * 50)

print("""
Property 1: LOCAL CONNECTIVITY
  - Each neuron only looks at a small region (e.g., 3×3 pixels)
  - Captures local patterns (edges, corners, textures)

Property 2: WEIGHT SHARING
  - Same filter slides across entire image
  - Detects same pattern anywhere in image
  - Example: Edge detector works on top-left AND bottom-right

Property 3: TRANSLATION INVARIANCE
  - Cat left of image = same as cat right of image
  - Pooling layers help achieve this

Property 4: HIERARCHICAL FEATURES
  - First layers: edges, corners
  - Middle layers: textures, patterns (eyes, wheels)
  - Final layers: complete objects (faces, cars)
""")

# Create a simple visual of convolution
plt.figure(figsize=(10, 4))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.title('Original 5×5 Image')
plt.xticks(range(5))
plt.yticks(range(5))

# Add a simple filter visualization
filter_vis = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

plt.subplot(1, 3, 2)
plt.imshow(filter_vis, cmap='RdBu', vmin=-1, vmax=1)
plt.title('Edge Detection Filter\n(3×3)')
plt.xticks(range(3))
plt.yticks(range(3))

# Convolved result
convolved = convolve2d(image, filter_vis, mode='same')
plt.subplot(1, 3, 3)
plt.imshow(convolved, cmap='gray')
plt.title('After Convolution\n(edges highlighted)')
plt.xticks(range(5))
plt.yticks(range(5))

plt.tight_layout()
plt.show()

print("\nObserve: The bright spot (200) created a strong response in the convolved image.")
print("The filter detected that this pixel is different from its neighbors.")