"""
POOLING AND BUILDING A CNN

Pooling: Reduces spatial size while keeping important information.
Common pooling: Max pooling (take maximum in each region)
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("MAX POOLING EXPLAINED")
print("=" * 50)

# Feature map (output of convolution)
feature_map = np.array([
    [1, 3, 2, 4],
    [5, 6, 8, 7],
    [2, 1, 9, 3],
    [4, 5, 2, 1]
])

print(f"Feature map (4×4):\n{feature_map}")

# TODO 1: Implement max pooling with 2×2 window, stride 2
def max_pooling_2x2(x):
    """2×2 max pooling with stride 2"""
    h, w = x.shape
    output_h = h // 2
    output_w = w // 2
    output = np.zeros((output_h, output_w))
    
    for i in range(output_h):
        for j in range(output_w):
            region = x[i*2:i*2+2, j*2:j*2+2]
            output[i, j] = np.max(region)
    
    return output

pooled = max_pooling_2x2(feature_map)
print(f"\nAfter 2×2 Max Pooling (4×4 → 2×2):\n{pooled}")

print("\n" + "=" * 50)
print("WHY POOLING HELPS")
print("=" * 50)

print("""
1. REDUCES SIZE:
   - 224×224 → after pooling: 112×112
   - Fewer parameters in next layers
   - Faster computation

2. TRANSLATION INVARIANCE:
   - If cat shifts by 1 pixel, max pooling might still activate
   - Model becomes less sensitive to exact position

3. PREVENTS OVERFITTING:
   - Fewer parameters = less memorization
   - Forces model to learn general patterns

TYPES OF POOLING:
  - Max pooling: Take maximum (most common)
  - Average pooling: Take average
  - Global average pooling: Average entire feature map (used at end)
""")

print("\n" + "=" * 50)
print("TYPICAL CNN ARCHITECTURE")
print("=" * 50)

print("""
┌─────────────────────────────────────────────────────────────────┐
│                    CNN ARCHITECTURE PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT IMAGE (224×224×3)                                        │
│       │                                                         │
│       ▼                                                         │
│  CONV LAYER 1: 32 filters, 3×3                                  │
│       │ → Feature maps: 224×224×32                              │
│       ▼                                                         │
│  MAX POOL: 2×2                                                  │
│       │ → Size halves: 112×112×32                               │
│       ▼                                                         │
│  CONV LAYER 2: 64 filters, 3×3                                  │
│       │ → Feature maps: 112×112×64                              │
│       ▼                                                         │
│  MAX POOL: 2×2                                                  │
│       │ → Size halves: 56×56×64                                 │
│       ▼                                                         │
│  CONV LAYER 3: 128 filters, 3×3                                 │
│       │ → Feature maps: 56×56×128                               │
│       ▼                                                         │
│  MAX POOL: 2×2                                                  │
│       │ → Size halves: 28×28×128                                │
│       ▼                                                         │
│  FLATTEN: Convert 2D to 1D                                      │
│       │ → 28×28×128 = 100,352 numbers                           │
│       ▼                                                         │
│  FULLY CONNECTED LAYER: 512 neurons                             │
│       │ → Each neuron sums all (weight x number) then adds bias │
│       ▼                                                         │
│  OUTPUT EMBEDDING: 512 numbers                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 50)
print("SIZE CALCULATIONS")
print("=" * 50)

def calc_output_size(input_size, kernel_size, padding, stride):
    """Calculate output size after convolution"""
    output = (input_size - kernel_size + 2 * padding) // stride + 1
    return output

input_size = 224
kernel = 3
padding = 1  # Same padding (output size = input size)
stride = 1

output = calc_output_size(input_size, kernel, padding, stride)
print(f"Input: {input_size}×{input_size}")
print(f"Kernel: {kernel}×{kernel}, padding={padding}, stride={stride}")
print(f"Output size: {output}×{output} (same as input with padding=1)")

print("\nAfter 2×2 pooling:")
print(f"Output: {output//2}×{output//2}")

# Visualize pooling effect
plt.figure(figsize=(10, 4))

# Original feature map
plt.subplot(1, 2, 1)
plt.imshow(feature_map, cmap='hot')
plt.title(f'Before Pooling\n{feature_map.shape[0]}×{feature_map.shape[1]}')
plt.colorbar()

# After pooling
plt.subplot(1, 2, 2)
plt.imshow(pooled, cmap='hot')
plt.title(f'After Max Pooling\n{pooled.shape[0]}×{pooled.shape[1]}')
plt.colorbar()

plt.tight_layout()
plt.show()