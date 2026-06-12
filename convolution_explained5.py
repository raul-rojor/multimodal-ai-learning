# ============================================
# PART 2: MORNING - CONVOLUTION EXPLAINED (45 min)
# ============================================

print("\n" + "=" * 60)
print("MORNING PART 2: CONVOLUTION OPERATION")
print("=" * 60)

"""
CONVOLUTION FROM SCRATCH
File: convolution_explained.py

What convolution does:
  Slide a small filter (kernel) over the image.
  At each position, multiply element-wise and sum.
  Output is a new "feature map".
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("CONVOLUTION STEP-BY-STEP")
print("=" * 50)

# Tiny 4x4 image
image = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

# Simple 2x2 filter (detects diagonal patterns)
filter_2x2 = np.array([
    [1, 0],
    [0, -1]
])

print(f"Image (4×4):\n{image}")
print(f"\nFilter (2×2):\n{filter_2x2}")

# TODO 1: Implement convolution manually
def convolve_manual(image, filter_2d):
    """Manual 2D convolution (no padding, stride=1)"""
    h, w = image.shape
    fh, fw = filter_2d.shape
    output_h = h - fh + 1
    output_w = w - fw + 1
    output = np.zeros((output_h, output_w))
    
    for i in range(output_h):
        for j in range(output_w):
            # Extract region
            region = image[i:i+fh, j:j+fw]
            # Element-wise multiply and sum
            output[i, j] = np.sum(region * filter_2d)
    
    return output

result = convolve_manual(image, filter_2x2)
print(f"\nConvolution result (3×3):\n{result}")

# Verify with scipy
from scipy.signal import convolve2d
result_scipy = convolve2d(image, filter_2x2, mode='valid')
print(f"\nVerified with scipy:\n{result_scipy}")
print("Match? ✓")

print("\n" + "=" * 50)
print("WHAT THE FILTER DETECTS")
print("=" * 50)

# Create a test image with different patterns
test_image = np.zeros((20, 20))
# Horizontal line (row 10)
test_image[10, :] = 1
# Vertical line (column 5)
test_image[:, 5] = 1

# Different filters
filters = {
    'Horizontal Edge': np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ]),
    'Vertical Edge': np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ]),
    'Blur': np.ones((3, 3)) / 9,
    'Sharpen': np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
}

plt.figure(figsize=(12, 8))

# Original image
plt.subplot(2, 3, 1)
plt.imshow(test_image, cmap='gray')
plt.title('Original Image\n(Horizontal + Vertical lines)')
plt.axis('off')

# Apply each filter
for i, (name, f) in enumerate(filters.items(), start=2):
    result = convolve2d(test_image, f, mode='same')
    plt.subplot(2, 3, i)
    plt.imshow(result, cmap='gray')
    plt.title(name)
    plt.axis('off')

plt.tight_layout()
plt.show()

print("\nOBSERVATIONS:")
print("  - Horizontal edge filter → detects horizontal line (row 10)")
print("  - Vertical edge filter → detects vertical line (column 5)")
print("  - Blur filter → makes everything fuzzy")
print("  - Sharpen filter → enhances edges")

print("\n" + "=" * 50)
print("KEY INSIGHTS")
print("=" * 50)

print("""
1. DIFFERENT FILTERS DETECT DIFFERENT PATTERNS:
   - Edge filters → detect boundaries
   - Blur filters → smooth out noise
   - Gabor filters → detect textures and orientations

2. MULTIPLE FILTERS PER LAYER:
   - A CNN layer has MANY filters (e.g., 64, 128, 256)
   - Each filter learns to detect something different
   - Output has depth = number of filters

3. FILTERS ARE LEARNED:
   - Not hand-designed (except simple ones above)
   - Model learns optimal filters during training
   - First layers: simple patterns (edges, colors)
   - Deeper layers: complex patterns (eyes, wheels, faces)
""")