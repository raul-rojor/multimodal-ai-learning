"""
UNDERSTANDING THE 512-DIM EMBEDDING SPACE

Your image encoder maps 224×224×3 = 150,528 numbers → 512 numbers.
Why 512? And what do those numbers mean?
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

print("=" * 50)
print("WHY 512 DIMENSIONS?")
print("=" * 50)

print("""
Trade-off:

  More dimensions (e.g., 1024):
    ✓ More expressive power (can represent more distinctions)
    ✗ More parameters to train
    ✗ Slower inference
    ✗ Risk of overfitting

  Fewer dimensions (e.g., 128):
    ✓ Faster, fewer parameters
    ✗ May lose important distinctions
    ✗ "Bottleneck" may be too tight

CLIP uses 512 as a balance. It works well for most tasks.
""")

# Simulate embeddings for different image categories
np.random.seed(42)

def make_embedding(base, noise=0.1):
    return base + np.random.randn(512) * noise

# Create base embeddings for categories
cat_base = np.random.randn(512)
dog_base = np.random.randn(512)
car_base = np.random.randn(512)

# Generate multiple embeddings per category
cat_embs = [make_embedding(cat_base, 0.15) for _ in range(10)]
dog_embs = [make_embedding(dog_base, 0.15) for _ in range(10)]
car_embs = [make_embedding(car_base, 0.15) for _ in range(10)]

all_embs = cat_embs + dog_embs + car_embs
labels = ['cat']*10 + ['dog']*10 + ['car']*10

# Reduce to 2D for visualization
pca = PCA(n_components=2)
embs_2d = pca.fit_transform(all_embs)

# Plot
colors = {'cat': 'red', 'dog': 'blue', 'car': 'green'}
plt.figure(figsize=(8, 6))
for label in ['cat', 'dog', 'car']:
    mask = [l == label for l in labels]
    plt.scatter(embs_2d[mask, 0], embs_2d[mask, 1], 
                c=colors[label], label=label, alpha=0.7)

plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.title('Embedding Space: Similar Images Cluster Together')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\nOBSERVATION:")
print("  - Cats cluster together (red)")
print("  - Dogs cluster together (blue)")
print("  - Cars cluster together (green)")
print("  - The embedding space is STRUCTURED: similar images → close vectors")
print("  - Distance in embedding space = semantic dissimilarity")

print("\n" + "=" * 50)
print("WHAT EACH OF THE 512 DIMENSIONS REPRESENTS")
print("=" * 50)

print("""
Not interpretable individually (like pixels). Instead:
  Each dimension captures a latent feature that the CNN learned:
    - Dimension 47: Might activate strongly for "round objects"
    - Dimension 203: Might activate for "texture of fur"
    - Dimension 401: Might activate for "blue color"

BUT: These are not human-interpretable. Since each dimension is a weighted sum of all previous parameters,
they're learned patterns (bits from every filter+pixel-informed parameter value)

WHAT MATTERS: The PATTERN across all 512 dimensions.
  Two images with similar patterns → close in embedding space → similar content.
""")