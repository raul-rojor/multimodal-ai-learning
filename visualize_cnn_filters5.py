"""
VISUALIZING CNN FILTERS

See what the CNN actually learns to detect.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

print("=" * 50)
print("VISUALIZING LEARNED FILTERS")
print("=" * 50)

# Create a small CNN
class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.conv1(x)

model = TinyCNN()

# Look at the first layer filters
filters = model.conv1.weight.data.numpy()
print(f"Filter shape: {filters.shape}")  # (16, 3, 3, 3)
print("16 filters, each 3×3 with 3 color channels (RGB)")

# Visualize the filters
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.flatten()

for i in range(16):
    # Take the filter and normalize for display
    filter_img = filters[i]
    # Combine RGB channels
    filter_img = filter_img.transpose(1, 2, 0)
    filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-8)
    
    axes[i].imshow(filter_img)
    axes[i].set_title(f'Filter {i+1}')
    axes[i].axis('off')

plt.suptitle('First Layer Filters (Randomly Initialized)\nThese will learn patterns during training')
plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("WHAT THESE FILTERS BECOME AFTER TRAINING")
print("=" * 50)

print("""
After training on many images, filters learn to detect:

  Early filters (Layer 1):
    - Color blobs
    - Oriented edges (horizontal, vertical, diagonal)
    - Simple textures

  Middle filters (Layer 2):
    - Combinations of edges (corners, curves)
    - Simple shapes (circles, squares)
    - Textures (dots, stripes)

  Late filters (Layer 3+):
    - Complex patterns (eyes, wheels, faces)
    - Object parts (legs, windows, handles)
    - High-level concepts

This hierarchy is LEARNED, not programmed.
""")

print("\n" + "=" * 50)
print("SUMMARY: WHAT YOU LEARNED TODAY")
print("=" * 50)

print("""
✓ Why raw pixels don't work (too many parameters)
✓ What convolution does (slide filter, multiply, sum)
✓ What filters detect (edges, textures, patterns)
✓ What pooling does (reduce size, translation invariance)
✓ How to build a CNN in PyTorch
✓ How to extract embeddings from images

TOMORROW: Transformers for text (same embedding concept, different architecture)
""")