"""
SVD ON REAL IMAGES (MNIST digits)
File: svd_images.py

This shows how SVD compresses actual images
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Load handwritten digits (8×8 images)
digits = load_digits()
images = digits.data  # 1797 images, each 64 pixels (8×8)
print(f"Dataset: {images.shape[0]} images, each {images.shape[1]} pixels")

# TODO 1: Pick one image and look at it
image_index = 42
image = images[image_index].reshape(8, 8)

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title(f'Original: Digit {digits.target[image_index]}')
plt.axis('off')

# TODO 2: Apply SVD to the image (any image is just a matrix)
U, S, Vt = np.linalg.svd(image, full_matrices=False)

print(f"Singular values of this image: {S.round(2)}")
print(f"First singular value (most important): {S[0]:.2f}")
print(f"Last singular value (least important): {S[-1]:.2f}")

# TODO 3: Reconstruct with different numbers of dimensions
for k in [1, 2, 4, 8]:
    # Keep only top k singular values
    S_k = np.zeros_like(S)
    S_k[:k] = S[:k]
    reconstructed = U @ np.diag(S_k) @ Vt
    
    plt.subplot(1, 3, 2 if k == 2 else 3)
    plt.imshow(reconstructed, cmap='gray')
    plt.title(f'{k} dimensions')
    plt.axis('off')
    
    # Error
    error = np.mean((image - reconstructed)**2)
    print(f"  {k} dimensions → error: {error:.4f}")

plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("WHAT YOU JUST LEARNED")
print("=" * 50)
print("""
1. Images (and embeddings) can be compressed using SVD
2. First few dimensions capture most of the information
3. This is why CLIP uses ~512 dimensions (not 10,000)
4. The same math works for TEXT embeddings too!

YOUR CLIP MODEL (Week 3) will learn embeddings that behave like SVD.
The difference: CLIP learns them from data, SVD computes them directly.
""")