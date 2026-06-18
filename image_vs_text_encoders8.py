# ============================================
# PART 3: AFTERNOON - COMPARE IMAGE AND TEXT ENCODERS (30 min)
# ============================================

print("\n" + "=" * 60)
print("AFTERNOON: COMPARING IMAGE AND TEXT ENCODERS")
print("=" * 60)

"""
COMPARING IMAGE AND TEXT ENCODERS
File: compare_encoders.py

Both encoders output 512-dim embeddings. Now they can be compared!
"""

import torch
import torch.nn.functional as F

# Import our encoders
from image_encoder7 import ImageEncoder
from text_encoder8 import TextEncoder

print("=" * 50)
print("CREATING BOTH ENCODERS")
print("=" * 50)

image_encoder = ImageEncoder(embedding_dim=512)
text_encoder = TextEncoder(vocab_size=30000, embedding_dim=512)

print(f"Image encoder parameters: {sum(p.numel() for p in image_encoder.parameters()):,}")
print(f"Text encoder parameters: {sum(p.numel() for p in text_encoder.parameters()):,}")

print("\n" + "=" * 50)
print("BOTH OUTPUT 512-DIM EMBEDDINGS")
print("=" * 50)

batch_size = 4

# Dummy images
dummy_images = torch.randn(batch_size, 3, 224, 224)

# Dummy text tokens
dummy_tokens = torch.randint(0, 30000, (batch_size, 32))
dummy_mask = torch.ones(batch_size, 32)

with torch.no_grad():
    image_embs = image_encoder(dummy_images)
    text_embs = text_encoder(dummy_tokens, dummy_mask)

print(f"Image embeddings shape: {image_embs.shape}")
print(f"Text embeddings shape: {text_embs.shape}")
print(f"Both are ({batch_size}, 512) → can compute similarity!")

# Compute similarity matrix
similarity_matrix = image_embs @ text_embs.T
print(f"\nSimilarity matrix shape: {similarity_matrix.shape}")
print("Rows = images, Columns = texts")
print("Each entry = cosine similarity (since both are normalized)")

print("\n" + "=" * 50)
print("WHAT THIS MEANS")
print("=" * 50)

print("""
We now have TWO encoders that output in the SAME embedding space:

  1. Image Encoder: 224×224×3 pixels → 512 numbers
  2. Text Encoder:  variable length text → 512 numbers

  similarity = image_emb @ text_emb.T

  High similarity = image and text match
  Low similarity = image and text don't match

TOMORROW: Train these to make correct pairs have high similarity!
""")

# Show similarity matrix values
print("\n" + "=" * 50)
print("SIMILARITY MATRIX VALUES (Random, untrained)")
print("=" * 50)

print(similarity_matrix.cpu().numpy().round(3))
print("\n(Currently random because encoders are untrained)")
print("After training: diagonal entries (matching pairs) will be high!")