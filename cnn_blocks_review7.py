# ============================================
# PART 1: MORNING - CNN REVIEW (30 min)
# ============================================

print("\n" + "=" * 60)
print("MORNING PART 1: CNN BUILDING BLOCKS REVIEW")
print("=" * 60)

"""
CNN BUILDING BLOCKS - QUICK REVIEW
File: cnn_blocks_review.py

Four operations that turn images into embeddings:
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print("=" * 50)
print("BLOCK 1: CONVOLUTION")
print("=" * 50)

# Convolution: slide filter over image, multiply and sum
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
print(f"Conv2d(3→16, 3×3): {conv}")
print("  - Multiplies filter values by image patch, sums all products, then adds bias → single number per filter")
print("  - Input: (batch, 3 (rgb), H, W)")
print("  - Output: (batch, 16 (filter layers), H, W)")
print("  - Learns 16 different filters")

print("\n" + "=" * 50)
print("BLOCK 2: ACTIVATION (ReLU)")
print("=" * 50)

relu = nn.ReLU()
print("ReLU: max(0, x)")
print("  - Removes negative values")
print("  - Adds non-linearity")
print("  - Since convolution layers are linear transformations, we need non-linearity to learn complex patterns," \
      "otherwise multiple conv layers would just be a single linear transformation")

print("\n" + "=" * 50)
print("BLOCK 3: POOLING")
print("=" * 50)

pool = nn.MaxPool2d(kernel_size=2, stride=2)
print(f"MaxPool2d(2×2, stride=2): {pool}")
print("  - Input: (batch, channels, H, W)")
print("  - Output: (batch, channels, H/2, W/2)")
print("  - Takes maximum in each 2×2 window")
print("  - No learned parameters")

print("\n" + "=" * 50)
print("BLOCK 4: FLATTEN + FULLY CONNECTED")
print("=" * 50)

# After conv layers: (batch, 128, 4, 4) = 128*4*4 = 2048 numbers per image
flatten = nn.Flatten()
linear = nn.Linear(2048, 512)

print(f"Flatten: (batch, 128, 4, 4) → (batch, 2048)")
print(f"Linear: (batch, 2048) → (batch, 512)")
print("  - Final 512 numbers are the IMAGE EMBEDDING")

print("\n" + "=" * 50)
print("TYPICAL CNN PIPELINE")
print("=" * 50)

print("""
Input (3, 224, 224)
    │
    ▼
Conv(3→64) + ReLU + Pool → (64, 112, 112)
    │
    ▼
Conv(64→128) + ReLU + Pool → (128, 56, 56)
    │
    ▼
Conv(128→256) + ReLU + Pool → (256, 28, 28)
    │
    ▼
Conv(256→512) + ReLU + Pool → (512, 14, 14)
    │
    ▼
Flatten → (512*14*14 = 100,352)
    │
    ▼
Linear(100352 → 512) → (512)
    | Every output value is a weighted sum of all 100,352 input values, plus bias
    │
    ▼
OUTPUT: Image embedding
""")

# Visualize feature map sizes
sizes = [(3, 224, 224), (64, 112, 112), (128, 56, 56), (256, 28, 28), (512, 14, 14)]
names = ['Input', 'After Conv1', 'After Conv2', 'After Conv3', 'After Conv4']
channels, heights, widths = zip(*sizes)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(names, heights, color='steelblue')
plt.ylabel('Height/Width (pixels)')
plt.title('Spatial Size Decreases')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.bar(names, channels, color='coral')
plt.ylabel('Number of Channels (filters)')
plt.title('Channel Depth Increases')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()