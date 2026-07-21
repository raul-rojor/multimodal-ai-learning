"""
UNDERSTANDING CONTRASTIVE LOSS

Visualizing how contrastive loss works.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("CONTRASTIVE LOSS STEP-BY-STEP")
print("=" * 50)

# Create dummy embeddings
batch_size = 4
dim = 8

# Random image and text embeddings
image_embs = torch.randn(batch_size, dim)
text_embs = torch.randn(batch_size, dim)

# Normalize (so dot product = cosine similarity)
image_embs = F.normalize(image_embs, p=2, dim=1)
text_embs = F.normalize(text_embs, p=2, dim=1)

# Step 1: Compute similarity matrix
similarity = image_embs @ text_embs.T
print(f"Similarity matrix (raw):")
print(similarity)

# Step 2: Apply temperature
temperature = 0.07
similarity_scaled = similarity / temperature
print(f"\nAfter temperature scaling (temp={temperature}):")
print(similarity_scaled)

# Step 3: Softmax (row-wise, image→text)
# For each image, which text is most similar?
probs_i2t = F.softmax(similarity_scaled, dim=1)
print(f"\nSoftmax (rows): image→text probabilities:")
print(probs_i2t)

# Step 4: Labels (diagonal = matching pairs)
labels = torch.arange(batch_size)
print(f"\nLabels (matching pairs): {labels.tolist()}")

# Step 5: Cross-entropy loss (image→text)
loss_i2t = F.cross_entropy(similarity_scaled, labels)
print(f"\nLoss (image→text): {loss_i2t.item():.4f}")

# Step 6: Text→image loss (columns)
loss_t2i = F.cross_entropy(similarity_scaled.T, labels)
print(f"Loss (text→image): {loss_t2i.item():.4f}")

# Step 7: Combined
loss = (loss_i2t + loss_t2i) / 2
print(f"Final loss: {loss.item():.4f}")

print("\n" + "=" * 50)
print("WHAT THE LOSS WANTS")
print("=" * 50)

print("""
The loss wants the similarity matrix to look like this:

  ┌─────────────────────────────────────┐
  │  [1.0, 0.0, 0.0, 0.0]               │
  │  [0.0, 1.0, 0.0, 0.0]               │
  │  [0.0, 0.0, 1.0, 0.0]               │
  │  [0.0, 0.0, 0.0, 1.0]               │
  └─────────────────────────────────────┘

  WHERE: Diagonal = 1 (matching pairs)
         Off-diagonal = 0 (non-matching pairs)

  During training, it pushes:
    - Matching pairs closer (higher similarity)
    - Non-matching pairs apart (lower similarity)

  Result: The embedding space becomes structured!
""")

# Visualize before and after
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Before training (random)
axes[0].imshow(similarity.numpy(), cmap='hot', vmin=0, vmax=1)
axes[0].set_title('Before Training (Random)')
axes[0].set_xlabel('Texts')
axes[0].set_ylabel('Images')
axes[0].set_xticks(range(batch_size))
axes[0].set_yticks(range(batch_size))

# After training (ideal)
ideal = torch.eye(batch_size)
axes[1].imshow(ideal.numpy(), cmap='hot', vmin=0, vmax=1)
axes[1].set_title('After Training (Ideal)')
axes[1].set_xlabel('Texts')
axes[1].set_ylabel('Images')
axes[1].set_xticks(range(batch_size))
axes[1].set_yticks(range(batch_size))

plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("CONTRASTIVE LOSS IN ONE SENTENCE")
print("=" * 50)

print("""
Contrastive loss maximizes similarity for matching pairs 
and minimizes similarity for non-matching pairs, 
using cross-entropy on the similarity matrix.

This is EXACTLY what we learned in Week 1!
    softmax(similarity / temperature) + cross-entropy
""")