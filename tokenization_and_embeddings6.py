"""
TOKENIZATION AND EMBEDDING LAYERS

Step 1: Convert text to numbers (tokenization)
Step 2: Convert numbers to vectors (embedding layer)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

print("=" * 50)
print("STEP 1: TOKENIZATION")
print("=" * 50)

# Simple word-level tokenization
text = "I love cats"
tokens = text.lower().split()
print(f"Text: {text}")
print(f"Tokens: {tokens}")

# Build vocabulary
vocab = {"<PAD>": 0, "<UNK>": 1}
for token in tokens:
    if token not in vocab:
        vocab[token] = len(vocab)

print(f"\nVocabulary: {vocab}")

# Convert to token IDs
token_ids = [vocab[token] for token in tokens]
print(f"Token IDs: {token_ids}")

print("\n" + "=" * 50)
print("SUBWORD TOKENIZATION (Modern approach)")
print("=" * 50)

print("""
PROBLEM WITH WORD-LEVEL:
  - Unknown words: "catssss" not in vocab
  - Huge vocabulary: English has 500,000+ words
  - Misses patterns: "run", "running", "ran" are unrelated numbers

SUBWORD SOLUTION (Byte-Pair Encoding / BPE):
  - Split rare words into common subwords
  - "running" → "run" + "ning"
  - "catssss" → "cat" + "ssss"
  - Vocabulary size ~30,000-50,000 (manageable)

EXAMPLE (simplified):
  "cats" → ["cat", "s"]
  "running" → ["run", "ning"]
  "unhappiness" → ["un", "happiness"]
""")

print("\n" + "=" * 50)
print("STEP 2: EMBEDDING LAYER")
print("=" * 50)

# Simulate an embedding layer
vocab_size = 10  # 10 possible words
embedding_dim = 4  # Each word becomes a 4-dimensional vector

# Random embedding matrix (in practice, these are learned)
embedding_matrix = np.random.randn(vocab_size, embedding_dim)

print(f"Vocabulary size: {vocab_size} words")
print(f"Embedding dimension: {embedding_dim}")
print(f"Embedding matrix shape: {embedding_matrix.shape}")

# Look up embedding for word with ID 3
word_id = 3
embedding = embedding_matrix[word_id]
print(f"\nWord ID {word_id} → embedding: {embedding.round(3)}")

# Embed a whole sentence
sentence_ids = [2, 5, 3, 1]
sentence_embeddings = embedding_matrix[sentence_ids]
print(f"Sentence token IDs: {sentence_ids}")
print(f"Sentence embeddings shape: {sentence_embeddings.shape}")
print("Each token becomes a vector. The sentence becomes a sequence of vectors.")

print("\n" + "=" * 50)
print("VISUALIZING EMBEDDINGS")
print("=" * 50)

# Create word embeddings for similar words
words = ["cat", "kitten", "dog", "puppy", "car", "truck", "bicycle"]
np.random.seed(42)

# Manually create embeddings so similar words are close
embeddings_dict = {
    "cat": [0.8, 0.7, 0.1],
    "kitten": [0.7, 0.8, 0.1],
    "dog": [0.6, 0.6, 0.2],
    "puppy": [0.5, 0.7, 0.2],
    "car": [0.1, 0.1, 0.9],
    "truck": [0.1, 0.2, 0.8],
    "bicycle": [0.2, 0.1, 0.7]
}

# Convert to array for PCA
word_list = list(embeddings_dict.keys())
embedding_array = np.array([embeddings_dict[w] for w in word_list])

# Reduce to 2D for visualization
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embedding_array)

# Plot
plt.figure(figsize=(10, 8))
colors = ['red'] * 4 + ['blue'] * 3
for i, word in enumerate(word_list):
    plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], 
                color=colors[i], s=100)
    plt.text(embeddings_2d[i, 0] + 0.02, embeddings_2d[i, 1] + 0.02, 
             word, fontsize=12)

plt.title('Word Embeddings\nSimilar words cluster together')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.grid(True, alpha=0.3)
plt.show()

print("\nOBSERVATION:")
print("  - cat/kitten/dog/puppy cluster together (animals)")
print("  - car/truck/bicycle cluster together (vehicles)")
print("  - The embedding layer learns these relationships automatically!")
print("  - Similar meaning → similar vectors → close in space")