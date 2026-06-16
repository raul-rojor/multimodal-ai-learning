# ============================================
# PART 4: AFTERNOON - VISUALIZE FEATURE MAPS (30 min)
# ============================================

print("\n" + "=" * 60)
print("AFTERNOON PART 2: VISUALIZING CNN FEATURE MAPS")
print("=" * 60)

"""
VISUALIZING WHAT THE CNN SEES
File: visualize_feature_maps.py

Run this to see how different layers detect different patterns.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Create a simple test image: horizontal and vertical lines
def create_test_image():
    img = torch.zeros(1, 3, 64, 64)
    # Horizontal line (middle)
    img[0, :, 32, :] = 1
    # Vertical line (middle)
    img[0, :, :, 32] = 1
    return img

# Simple CNN to visualize
class VisualizableCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        self.feature_maps = []
        x = self.conv1(x)
        self.feature_maps.append(x.clone())
        x = self.relu(x)
        x = self.conv2(x)
        self.feature_maps.append(x.clone())
        return x

model = VisualizableCNN()
test_image = create_test_image()

with torch.no_grad():
    output = model(test_image)

# Visualize
fig, axes = plt.subplots(2, 8, figsize=(16, 4))

# Layer 1 feature maps (8 filters)
for i in range(8):
    axes[0, i].imshow(model.feature_maps[0][0, i].numpy(), cmap='hot')
    axes[0, i].set_title(f'Layer1 Filter {i+1}')
    axes[0, i].axis('off')

# Layer 2 feature maps (first 8 of 16)
for i in range(8):
    axes[1, i].imshow(model.feature_maps[1][0, i].numpy(), cmap='hot')
    axes[1, i].set_title(f'Layer2 Filter {i+1}')
    axes[1, i].axis('off')

plt.suptitle('Feature Maps: Layer 1 (edges) → Layer 2 (patterns)')
plt.tight_layout()
plt.show()

print("\nOBSERVATION:")
print("  Layer 1: Detects edges, lines, simple patterns")
print("  Layer 2: Combines edges into more complex patterns")
print("  Deeper layers would detect shapes (eyes, wheels, faces)")