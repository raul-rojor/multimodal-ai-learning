# ============================================
# PART 5: EVENING - TEXT ENCODER VS IMAGE ENCODER (30 min)
# ============================================

print("\n" + "=" * 60)
print("EVENING: SIDE-BY-SIDE COMPARISON")
print("=" * 60)

"""
SIDE-BY-SIDE: IMAGE ENCODER VS TEXT ENCODER
File: encoder_comparison.py

See how both encoders produce 512-dim embeddings, but from different inputs.
"""

import torch

print("=" * 50)
print("IMAGE ENCODER")
print("=" * 50)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ INPUT:  (batch, 3, 224, 224)  (RGB image)                       │
│                                                                 │
│ OPERATIONS:                                                     │
│   1. Conv2d(3→64, 3×3) + ReLU + Pool                            │
│   2. Conv2d(64→128, 3×3) + ReLU + Pool                          │
│   3. Conv2d(128→256, 3×3) + ReLU + Pool                         │
│   4. Conv2d(256→512, 3×3) + ReLU + Pool                         │
│   5. AdaptiveAvgPool2d(7×7)                                     │
│   6. Flatten: 512 × 7 × 7 = 25,088                              │
│   7. Linear(25088 → 512)                                        │
│   8. L2 Normalize                                               │
│                                                                 │
│ OUTPUT: (batch, 512)  (image embedding)                         │
│                                                                 │
│ WHAT IT LEARNS: Spatial patterns (edges, textures, objects)     │
└─────────────────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 50)
print("TEXT ENCODER")
print("=" * 50)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ INPUT:  (batch, seq_len)  (token IDs)                           │
│                                                                 │
│ OPERATIONS:                                                     │
│   1. Token Embedding: (batch, seq_len) → (batch, seq_len, 512)  │
│   2. Add Positional Encoding                                    │
│   3. Transformer Layers (×4):                                   │
│      - Self-Attention (each word looks at all words)            │
│      - Feed-Forward (per position)                              │ 
│      - Layer Norm + Residual connections                        │
│   4. Mean Pooling over real tokens                              │
│   5. Linear(512 → 512)                                          │
│   6. L2 Normalize                                               │
│                                                                 │
│ OUTPUT: (batch, 512)  (text embedding)                          │
│                                                                 │
│ WHAT IT LEARNS: Sequential patterns (grammar, meaning, context) │
└─────────────────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 50)
print("BOTH ENCODERS PRODUCE 512-DIM EMBEDDINGS")
print("=" * 50)

print("""
┌─────────────────────────────────────────────────────────────────┐
│                     SAME OUTPUT SPACE                           │
│                                                                  │
│  image_embedding (512)  ←→  text_embedding (512)               │
│                                                                  │
│  similarity = image_embedding @ text_embedding.T               │
│                                                                  │
│  HIGH similarity = image matches text                          │
│  LOW similarity = image doesn't match text                     │
└─────────────────────────────────────────────────────────────────┘

TOMORROW: We train both encoders together so matching pairs have HIGH similarity.
""")