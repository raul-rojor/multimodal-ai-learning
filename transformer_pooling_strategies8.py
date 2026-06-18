# ============================================
# PART 4: AFTERNOON - UNDERSTANDING POOLING (30 min)
# ============================================

print("\n" + "=" * 60)
print("AFTERNOON PART 2: POOLING STRATEGIES EXPLAINED")
print("=" * 60)

"""
UNDERSTANDING POOLING STRATEGIES
File: pooling_strategies.py

How do we go from (seq_len, 512) to a single (512) embedding (i.e., from vectors each representing a token to 
a single vector representing the entire text sequence)?
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print("=" * 50)
print("THE POOLING PROBLEM")
print("=" * 50)

# Simulate real transformer output: 10 words, each 512-dim
seq_len = 10
dim = 8  # Small for visualization
word_vectors = torch.randn(seq_len, dim)

print(f"Transformer output: {word_vectors.shape}")
print("10 word vectors, each 8-dimensional")
print("We need ONE vector for the whole sentence")

print("\n" + "=" * 50)
print("STRATEGY 1: MEAN POOLING")
print("=" * 50)

mean_pool = word_vectors.mean(dim=0)
print(f"Mean pooling: {mean_pool.round(3)}")
print("  - Add all word vectors, divide by number of words")
print("  - Simple and effective")
print("  - Ignores word order (but transformer already encoded order)")

print("\n" + "=" * 50)
print("STRATEGY 2: MAX POOLING")
print("=" * 50)

max_pool = word_vectors.max(dim=0)[0]
print(f"Max pooling: {max_pool.round(3)}")
print("  - Take maximum value in each dimension across all words")
print("  - Captures strongest activation")
print("  - Can lose information from words that are important but don't maximize")

print("\n" + "=" * 50)
print("STRATEGY 3: CLS TOKEN (BERT-style)")
print("=" * 50)

print("""
  - Add special [CLS] token at position 0
  - Take ONLY that token's output as the sentence embedding
  - [CLS] token learns to aggregate information from all words
  - Works well but adds a special token

  Example: "[CLS] the cat sat on the mat"
  Output: Use ONLY the [CLS] position
""")

print("\n" + "=" * 50)
print("WHAT WE USE (MEAN POOLING WITH MASK)")
print("=" * 50)

print("""
  - Average over only REAL tokens (exclude padding)
  - Mask ensures padding doesn't dilute the embedding
  - Best balance of simplicity and effectiveness
""")

# Visualize the difference
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.bar(range(dim), mean_pool, color='steelblue')
plt.title('Mean Pooling')
plt.xlabel('Dimension')
plt.ylabel('Value')

plt.subplot(1, 3, 2)
plt.bar(range(dim), max_pool, color='coral')
plt.title('Max Pooling')
plt.xlabel('Dimension')

plt.subplot(1, 3, 3)
plt.bar(range(dim), word_vectors[0], color='green')
plt.title('CLS Token (position 0)')
plt.xlabel('Dimension')

plt.tight_layout()
plt.show()