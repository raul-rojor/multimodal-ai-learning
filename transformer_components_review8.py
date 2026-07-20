"""
TRANSFORMER COMPONENTS - QUICK REVIEW

Four operations that turn text into embeddings:
"""

import torch
import torch.nn as nn
import math

print("=" * 50)
print("COMPONENT 1: EMBEDDING LAYER")
print("=" * 50)

# Token ID → Vector
embedding = nn.Embedding(num_embeddings=10000, embedding_dim=512)
print(f"Embedding(10000, 512): {embedding}")
print("  - Input: token ID (e.g., 42)")
print("  - Output: 512-dim vector")
print("  - Learned a vector for each token in vocabulary")

print("\n" + "=" * 50)
print("COMPONENT 2: POSITIONAL ENCODING")
print("=" * 50)

def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = math.sin(pos / (10000 ** (2*i / d_model)))
            if i+1 < d_model:
                pe[pos, i+1] = math.cos(pos / (10000 ** (2*i / d_model)))
    return pe

pe = positional_encoding(10, 512)
print(f"PositionalEncoding(10, 512): {pe.shape}")
print("  - Same shape as embeddings")
print("  - Added to embeddings (not concatenated)")
print("  - Fixed formula, no learned parameters")

print("\n" + "=" * 50)
print("COMPONENT 3: SELF-ATTENTION")
print("=" * 50)

attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
print(f"MultiheadAttention(512, 8 heads): {attention}")
print("  - Input: (batch, seq_len, 512)")
print("  - Output: (batch, seq_len, 512)")
print("  - Each word looks at every other word")

print("\n" + "=" * 50)
print("COMPONENT 4: FEED-FORWARD NETWORK")
print("=" * 50)

ffn = nn.Sequential(
    nn.Linear(512, 2048),  # Expand
    nn.ReLU(),             # Non-linearity
    nn.Linear(2048, 512)   # Compress back
)
print(f"FFN: Linear(512→2048) → ReLU → Linear(2048→512)")
print("""
Input (batch, 512)
    │
    ▼
Linear1: (batch, 512) → (batch, 2048)   ← Expand (each output is a linear combination of all 512 inputs)
    │
    ▼
ReLU: (batch, 2048) → (batch, 2048)     ← Non-linearity
    │
    ▼
Linear2: (batch, 2048) → (batch, 512)   ← Compress (each output is a linear combination of all 2048 inputs)
    │
    ▼
Output (batch, 512)                     ← Already compressed)""")
print("Refine embedded meaning locally in each position based on what it learned from other words in the attention step.")
print("  - Applied independently to EACH position")
print("  - Adds capacity to learn complex patterns")

print("\n" + "=" * 50)
print("COMPONENT 5: LAYER NORMALIZATION")
print("=" * 50)

layer_norm = nn.LayerNorm(512)
print(f"LayerNorm(512): {layer_norm}")
print("  - Normalizes each position to mean=0, std=1")
print("  - Stabilizes training")

print("\n" + "=" * 50)
print("COMPONENT 6: RESIDUAL CONNECTION (ADD)")
print("=" * 50)

print("""
  output = input + layer(input)
  (Add the original input back)

  Why:
    - Prevents vanishing gradients
    - Allows model to learn "adjustments" rather than overwriting
    - Makes deeper networks trainable
""")

print("\n" + "=" * 50)
print("COMPLETE TRANSFORMER BLOCK")
print("=" * 50)

print("""
Input (seq_len, 512)
    │
    ▼
  ┌─────────────────────────────────────┐
  │  NORM + SELF-ATTENTION + ADD        │
  │  (each word looks at all words)     │
  └─────────────────────────────────────┘
    │
    ▼
  ┌─────────────────────────────────────┐
  │  NORM + FFN + ADD                   │
  │  (per-position MLP)                 │
  └─────────────────────────────────────┘
    │
    ▼
Output (seq_len, 512)
""")