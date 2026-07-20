"""
POSITIONAL ENCODING - ADDING WORD ORDER

PROBLEM: Embedding layer treats "cat sat" same as "sat cat" (same bag of vectors)
SOLUTION: Add positional information so model knows word order
"""

from ast import If

import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("THE POSITION PROBLEM")
print("=" * 50)

# Same words, different order
sentence1 = ["the", "cat", "sat"]
sentence2 = ["sat", "the", "cat"]

print(f"Sentence 1: {sentence1}")
print(f"Sentence 2: {sentence2}")

# With just embeddings (no position), these would be identical
print("\nWithout positional encoding: Both sentences have SAME representation")
print("(just a bag of word vectors in different order)")

print("\n" + "=" * 50)
print("SINE/COSINE POSITIONAL ENCODING (from the original Transformer)")
print("=" * 50)


def get_positional_encoding(seq_len, d_model):
    """
    Create positional encodings for a sequence.

    Each position needs a unique vector. We build it by creating several sine waves
    at different frequencies (10000^(2i/d_model) in formula below). For each speed,
    we need TWO numbers to avoid collisions: sine and cosine. So frequency 1 uses
    dimensions 0 and 1 of the embedding to be outputted (2i and 2i+1 in formula; 0 and 1 in this case),
    frequency 2 uses dimensions 2 and 3, frequency 3 uses dimensions 4 and 5, and so on.
    That is why two subsequent dimensions form one frequency.
    If more than 2 dimensions were used for the same frequency, the model would have 
    trouble distinguishing positions and only sine/cosine are needed per frequency
    so useless info would be included or dim(position embedding) would be < d_model. 
    
    Formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Where:
        pos = position in sequence (0, 1, 2, ...)
        i = dimension index
        d_model = embedding dimension
    """
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000 ** (2*i / d_model)))
            if i + 1 < d_model:
                pe[pos, i+1] = np.cos(pos / (10000 ** (2*i / d_model)))
    return pe

seq_len = 50
d_model = 128
positional_encodings = get_positional_encoding(seq_len, d_model)

print(f"Positional encodings shape: {positional_encodings.shape}")
print("Rows = positions in sequence (0 to 49)")
print("Columns = embedding dimensions (0 to 127)")

# Visualize
plt.figure(figsize=(12, 6))
plt.imshow(positional_encodings.T, cmap='RdBu', aspect='auto')
plt.colorbar(label='Encoding value')
plt.xlabel('Position in Sequence')
plt.ylabel('Embedding Dimension')
plt.title('Positional Encodings\nEach dimension has a different frequency pattern')
plt.tight_layout()
plt.show()

# Show a few dimensions
plt.figure(figsize=(12, 4))
for dim in [0, 1, 2, 3]:
    plt.plot(positional_encodings[:, dim], label=f'Dimension {dim}')
plt.xlabel('Position')
plt.ylabel('Encoding Value')
plt.title('Positional Encoding Patterns\nDifferent dimensions = different frequencies')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n" + "=" * 50)
print("HOW POSITIONAL ENCODING WORKS")
print("=" * 50)

# Take a sentence
sentence = ["the", "cat", "sat", "on", "the", "mat"]
seq_len = len(sentence)
embed_dim = 8

# Random word embeddings
word_embeddings = np.random.randn(seq_len, embed_dim)

# Positional encodings
pos_encodings = get_positional_encoding(seq_len, embed_dim)

# Final input = word_embedding + positional_encoding
final_embeddings = word_embeddings + pos_encodings

print(f"Word embedding at position 0: {word_embeddings[0, :4].round(3)}")
print(f"Positional encoding at position 0: {pos_encodings[0, :4].round(3)}")
print(f"Final (sum): {final_embeddings[0, :4].round(3)}")
print("\nNow the model knows BOTH what the word IS and WHERE it is!")

print("\n" + "=" * 50)
print("WHY SINE/COSINE?")
print("=" * 50)

print("""
1. DIFFERENT FREQUENCIES FOR DIFFERENT DIMENSIONS:
   - Low dimensions in outputted position embeddings: slow-changing (distinguish far positions)
   - High dimensions in outputted position embeddings: fast-changing (distinguish nearby positions
     since numbers repeat at further positions)

2. RELATIVE POSITION INFORMATION:
   - The encoding at position pos + k can be expressed as a linear function of pos
   - Allows model to learn relationships between positions

3. NO LEARNED PARAMETERS:
   - Fixed formula, doesn't need training
   - Works for sequences longer than seen during training
""")