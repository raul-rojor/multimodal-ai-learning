# ============================================
# PART 2: MORNING - EIGENVALUES INTUITION (45 min)
# ============================================

print("\n" + "=" * 60)
print("MORNING PART 2: EIGENVALUES - WHAT STAYS THE SAME")
print("=" * 60)

"""
EIGENVALUES FOR MULTIMODAL AI
File: eigenvalues_intro.py

WHAT EIGENVALUES TELL YOU:
  When you multiply a matrix by a vector, most vectors change direction.
  EIGENVECTORS are special vectors that DON'T change direction.
  EIGENVALUES tell you how much they stretch or shrink.

ANALOGY:
  Imagine stretching a rubber sheet.
  Most points move to new positions.
  But points along certain lines just get pulled outward (same line).
  Those lines = eigenvectors. The stretch amount = eigenvalues.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("EIGENVALUES INTUITION")
print("=" * 50)

# Create a simple scaling matrix
# This matrix stretches vectors along the x-axis by 2x
matrix = np.array([[2, 0],
                   [0, 1]])

print(f"Matrix A = \n{matrix}")
print("This matrix STRETCHES vectors horizontally by 2x")
print("Vertical direction stays the same")

# Eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(matrix)

print(f"\nEigenvalues: {eigenvalues}")
print(f"Eigenvalue 1 = {eigenvalues[0]:.1f} → stretch factor along eigenvector 1")
print(f"Eigenvalue 2 = {eigenvalues[1]:.1f} → stretch factor along eigenvector 2")

print(f"\nEigenvectors (as columns):")
print(eigenvectors)
print("Eigenvector 1: [1, 0] (horizontal axis)")
print("Eigenvector 2: [0, 1] (vertical axis)")

# TODO 1: Apply matrix to a vector and see the effect
vector = np.array([1, 1])
transformed = matrix @ vector

print(f"\nOriginal vector: {vector}")
print(f"After matrix multiply: {transformed}")
print("Horizontal component doubled (x=1→2), vertical unchanged (y=1→1)")

# TODO 2: Visualize eigenvectors
plt.figure(figsize=(8, 8))
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)

# Plot eigenvectors as arrows
origin = np.array([0, 0])
for i in range(2):
    vec = eigenvectors[:, i] * eigenvalues[i]  # Scale by eigenvalue
    plt.arrow(origin[0], origin[1], vec[0], vec[1], 
              head_width=0.1, head_length=0.1, 
              color=['red', 'blue'][i], alpha=0.7, width=0.02)
    plt.text(vec[0]*1.1, vec[1]*1.1, f'λ={eigenvalues[i]:.1f}', 
             fontsize=12, color=['red', 'blue'][i])

# Plot random points transformed
for _ in range(20):
    point = np.random.randn(2) * 0.5
    transformed_point = matrix @ point
    plt.plot(point[0], point[1], 'go', alpha=0.5, markersize=4)
    plt.plot(transformed_point[0], transformed_point[1], 'ro', alpha=0.5, markersize=4)
    plt.plot([point[0], transformed_point[0]], 
             [point[1], transformed_point[1]], 'gray', alpha=0.3)

plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Green → Original points, Red → After matrix\\nRed arrows = eigenvectors (direction that stays same)')
plt.legend(['Eigenvector 1 (λ=2)', 'Eigenvector 2 (λ=1)'], loc='upper right')
plt.show()

print("\n" + "=" * 50)
print("WHY EIGENVALUES MATTER FOR MULTIMODAL AI")
print("=" * 50)
print("""
1. ATTENTION MECHANISMS (Transformers):
   - Attention weights are eigenvectors of certain matrices
   - Largest eigenvalue = most important information flow

2. DIMENSIONALITY REDUCTION:
   - Keep dimensions with largest eigenvalues (most variance)
   - Drop dimensions with tiny eigenvalues (noise)

3. UNDERSTANDING EMBEDDINGS:
   - Eigenvalues reveal how much information each dimension holds
   - CLIP embeddings use this to compress images/text into 512 numbers

4. STABILITY OF TRAINING:
   - If eigenvalues are too large → gradients explode
   - If eigenvalues are too small → gradients vanish
   - Modern architectures carefully control eigenvalues
""")

print("\\n" + "=" * 50)
print("SIMPLE RULE FOR MULTIMODAL AI")
print("=" * 50)
print("""
Eigenvalue > 1 → Dimension gets STRETCHED (amplified)
Eigenvalue = 1 → Dimension stays the SAME
Eigenvalue < 1 → Dimension gets SQUEEZED (attenuated)
Eigenvalue = 0 → Dimension gets DESTROYED (information lost)

Good embeddings have eigenvalues that balance all.
""")