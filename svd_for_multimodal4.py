# ============================================
# PART 3: AFTERNOON - SVD (The Core of CLIP)
# ============================================

print("\n" + "=" * 60)
print("AFTERNOON: SVD - SINGULAR VALUE DECOMPOSITION")
print("=" * 60)

"""
SVD: THE MATH BEHIND CLIP EMBEDDINGS
File: svd_for_multimodal.py

WHAT IS SVD?
  Any matrix can be factored into 3 parts: A = U @ S @ V.T
  
  U = left singular vectors (image basis)
  S = singular values - diagonal matrix (importance of each direction)
  V = right singular vectors (text basis)

HOW CLIP USES SVD:
  CLIP learns embeddings so that:
    image_embeddings × text_embeddings.T = similarity matrix
  This is essentially learning a factored representation (like SVD)
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("SVD DECOMPOSITION EXPLAINED")
print("=" * 50)

# Create a simple matrix (imagine 3 images × 4 texts similarity)
A = np.array([
    [0.9, 0.1, 0.2, 0.1],
    [0.1, 0.8, 0.1, 0.2],
    [0.2, 0.1, 0.9, 0.1]
])

print("Original matrix A (3 images × 4 texts):")
print(A.round(2))
print("Goal: Find image embeddings and text embeddings so that A = image_emb @ text_emb.T")

# Perform SVD
U, S, Vt = np.linalg.svd(A, full_matrices=False)

print(f"\nU shape: {U.shape} (image basis)")
print(f"S shape: {S.shape} (singular values)")
print(f"Vt shape: {Vt.shape} (text basis)")

print(f"\nSingular values (importance of each dimension):")
for i, s in enumerate(S):
    print(f"  Dimension {i+1}: {s:.3f}")

# Reconstruct from SVD
A_reconstructed = U @ np.diag(S) @ Vt

print(f"\nReconstructed matrix (should match original):")
print(A_reconstructed.round(2))

# ============================================
# TODO 1: Use SVD to create embeddings
# ============================================

print("\\n" + "=" * 50)
print("USING SVD TO CREATE EMBEDDINGS")
print("=" * 50)

# Image embeddings = U * sqrt(S)
# Text embeddings = Vt.T * sqrt(S)
# This way: image_emb @ text_emb.T = A

sqrt_S = np.sqrt(S)
image_embeddings = U * sqrt_S
text_embeddings = (Vt.T * sqrt_S)

print(f"Image embeddings shape: {image_embeddings.shape}")
print(f"Text embeddings shape: {text_embeddings.shape}")

# Verify: embeddings multiply to original matrix
reconstructed = image_embeddings @ text_embeddings.T
print(f"\nReconstructed from embeddings (should match original):")
print(reconstructed.round(2))

print("\n" + "=" * 50)
print("WHAT THE EMBEDDINGS LOOK LIKE")
print("=" * 50)

print("\nImage embeddings (each image = 3 numbers):")
for i, emb in enumerate(image_embeddings):
    print(f"  Image {i+1}: {emb.round(3)}")

print("\nText embeddings (each text = 3 numbers):")
for i, emb in enumerate(text_embeddings):
    print(f"  Text {i+1}: {emb.round(3)}")

# ============================================
# TODO 2: Dimensionality reduction (keep only top k dimensions)
# ============================================

print("\n" + "=" * 50)
print("DIMENSIONALITY REDUCTION WITH SVD")
print("=" * 50)

# Keep only top 2 dimensions (out of 3)
k = 2
image_emb_k = image_embeddings[:, :k]
text_emb_k = text_embeddings[:, :k]
A_reduced = image_emb_k @ text_emb_k.T

print(f"Original matrix (3×4):")
print(A.round(2))
print(f"\nReconstructed with only {k} dimensions:")
print(A_reduced.round(2))
print(f"\nError: {np.mean((A - A_reduced)**2):.4f}")

print("\n" + "=" * 50)
print("KEY INSIGHT FOR MULTIMODAL AI")
print("=" * 50)
print("""
1. CLIP learns embeddings so that image_emb @ text_emb.T ≈ similarity

2. SVD shows that ANY similarity matrix can be factored this way

3. The singular values tell you the "intrinsic dimension"
   → If after 100 dimensions singular values become tiny, you don't need more

4. CLIP typically uses 512 dimensions (enough to capture most information)

5. Temperature scaling (from Day 1) happens BEFORE this multiplication
   → logits = similarity / temperature
   → Then softmax, then cross-entropy

YOU NOW UNDERSTAND THE MATH BEHIND CLIP!
""")

# Visualize singular values (scree plot)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.bar(range(1, len(S)+1), S, color='steelblue')
plt.xlabel('Dimension')
plt.ylabel('Singular Value')
plt.title('Importance of Each Dimension')
plt.xticks(range(1, len(S)+1))

plt.subplot(1, 2, 2)
cumulative_variance = np.cumsum(S**2) / np.sum(S**2)
plt.plot(range(1, len(S)+1), cumulative_variance, 'ro-', linewidth=2)
plt.xlabel('Number of Dimensions')
plt.ylabel('Cumulative Variance Explained')
plt.title('How Many Dimensions to Keep?')
plt.axhline(y=0.95, color='k', linestyle='--', label='95% threshold')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

print("""
SCREE PLOT INTERPRETATION:
  - Sharp drop = dimensions that matter
  - Long tail = dimensions you can drop
  - CLIP keeps ~512 dimensions because that's where the drop happens
""")