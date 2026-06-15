# ============================================
# PART 4: AFTERNOON - SELF-ATTENTION (1 hour)
# ============================================

print("\n" + "=" * 60)
print("AFTERNOON PART 2: SELF-ATTENTION")
print("=" * 60)

"""
SELF-ATTENTION FROM SCRATCH
File: self_attention.py

The core innovation of Transformers: Every word looks at every other word.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("SELF-ATTENTION STEP-BY-STEP")
print("=" * 50)

# Example sentence: "The animal didn't cross the street because it was too tired"
# We want to understand what "it" refers to

words = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired"]
seq_len = len(words)
d_model = 8  # embedding dimension

# Simulated word embeddings (in reality, these come from embedding layer + positional encoding)
np.random.seed(42)
embeddings = np.random.randn(seq_len, d_model)

print(f"Sentence: {' '.join(words)}")
print(f"Sequence length: {seq_len}")
print(f"Embedding dimension: {d_model}")

# Step 1: Create Query, Key, Value matrices
print("\n" + "=" * 50)
print("STEP 1: CREATE Q, K, V")
print("=" * 50)

"""
CONCEPTUAL EXPLANATION: QUERY, KEY, VALUE

BEFORE THE CODE, UNDERSTAND THIS:

ANALOGY: LIBRARY SEARCH

Imagine you walk into a library and say "I want books about cats."

- Your QUESTION ("cats") is the QUERY (Q)
- Each book's TITLE and DESCRIPTION is the KEY (K)
- The book's FULL CONTENT is the VALUE (V)

The library:
  1. Compares your QUERY against every book's KEY (how similar is "cats" to this book's title?)
  2. Gets an ATTENTION SCORE for each book
  3. Returns a WEIGHTED SUM of all books' VALUES (more weight on relevant books)

That weighted sum is your answer: information from ALL books, but focused on the relevant ones.


IN TRANSFORMERS:

The sentence: "The animal didn't cross the street because it was too tired"

When processing the word "it":

  QUERY (Q) = "What am I looking for?" (from "it")
  KEYS (K)   = "What information do I have?" (from every word)
  VALUES (V) = "What information do I pass forward?" (from every word)

Step 1: Compare Q (from "it") with every word's key
  → "it" has high similarity with "animal"
  → "it" has low similarity with "street"

Step 2: Softmax gives attention weights (e.g., 80% to "animal", 5% to "street")

Step 3: Take weighted sum of VALUES
  → Output for "it" is mostly VALUE of "animal"

RESULT: "it" now contains information from "animal" (coreference resolved!)

THE MATRICES:

Each word has its own:
  Q = W_Q × word_embedding   (what I'm looking for)
  K = W_K × word_embedding   (what I offer)
  V = W_V × word_embedding   (what I pass)

W_Q, W_K, W_V are LEARNED matrices (same for all words).

Attention = softmax( Q @ K.T / sqrt(dim) ) @ V

  Q @ K.T  = Compare every word's Q with every word's K
  softmax  = Turn similarities into probabilities
  @ V      = Weighted sum of VALUES


ONE SENTENCE:

QUERY = what I'm looking for, KEY = what others have, VALUE = what others give.
"""

# Learnable weight matrices (in practice, these are learned during training)
W_Q = np.random.randn(d_model, d_model) / np.sqrt(d_model)
W_K = np.random.randn(d_model, d_model) / np.sqrt(d_model)
W_V = np.random.randn(d_model, d_model) / np.sqrt(d_model)

# Compute Q, K, V for each word
Q = embeddings @ W_Q  # (seq_len, d_model)
K = embeddings @ W_K  # (seq_len, d_model)
V = embeddings @ W_V  # (seq_len, d_model)

print(f"Q shape: {Q.shape} (Query: 'what am I looking for?')")
print(f"K shape: {K.shape} (Key: 'what do I have in token embeddings batch?')")
print(f"V shape: {V.shape} (Value: 'what information do I pass from this batch?')")

# Step 2: Compute attention scores
print("\n" + "=" * 50)
print("STEP 2: ATTENTION SCORES (Q @ K.T)")
print("=" * 50)

attention_scores = Q @ K.T  # (seq_len, seq_len)
print(f"Attention scores shape: {attention_scores.shape}")
print("Each entry (i,j) = how much word i attends to word j")

# Step 3: Scale (divide by sqrt(d_model))
print("\n" + "=" * 50)
print("STEP 3: SCALING")
print("=" * 50)

scale = np.sqrt(d_model)
attention_scores_scaled = attention_scores / scale
print(f"Divided by sqrt({d_model}) = {scale:.1f}")
print("Scaling prevents dot products from becoming too large (which would kill gradients)")

# Step 4: Apply softmax
print("\n" + "=" * 50)
print("STEP 4: SOFTMAX (ATTENTION WEIGHTS)")
print("=" * 50)

# Softmax along rows (each word's attention to all words sums to 1)
attention_weights = np.exp(attention_scores_scaled)
attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)

print(f"Attention weights shape: {attention_weights.shape}")
print("Each row sums to 1.0")

# Show attention for the word "it" (position 7)
it_pos = words.index("it")
it_attention = attention_weights[it_pos]
top_attended = np.argsort(it_attention)[-5:][::-1]

print(f"\nWord '{words[it_pos]}' (position {it_pos}) pays most attention to:")
for pos in top_attended:
    print(f"  '{words[pos]}' (pos {pos}): weight = {it_attention[pos]:.3f}")

# Step 5: Weighted sum of Values
print("\n" + "=" * 50)
print("STEP 5: WEIGHTED SUM OF VALUES")
print("=" * 50)

output = attention_weights @ V
print(f"Output shape: {output.shape}")
print("Each word's attention output is a weighted combination of ALL words' values")
print("'it' now contains information from 'animal' because it attended to it!")

print("\n" + "=" * 50)
print("VISUALIZING ATTENTION WEIGHTS")
print("=" * 50)

# Plot attention matrix
plt.figure(figsize=(10, 8))
plt.imshow(attention_weights, cmap='Blues', aspect='auto')
plt.colorbar(label='Attention Weight')
plt.xticks(range(seq_len), words, rotation=90)
plt.yticks(range(seq_len), words)
plt.xlabel('Words being ATTENDED TO (Keys)')
plt.ylabel('Words DOING THE ATTENDING (Queries)')
plt.title('Self-Attention Weights\nRow = which words this word looks at')
plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("MULTI-HEAD ATTENTION")
print("=" * 50)

print("""
Instead of one attention computation, Transformers use MULTIPLE heads:

  Head 1: Looks at grammatical relationships (subject-verb)
  Head 2: Looks at coreference resolution ("it" → "animal")
  Head 3: Looks at positional relationships (nearby words)
  Head 4: Looks at semantic similarity (synonyms)
  ... (typically 8-16 heads)

Each head has its own W_Q, W_K, W_V matrices.
Outputs from all heads are concatenated and projected.
""")

print("\n" + "=" * 50)
print("THE COMPLETE TRANSFORMER BLOCK")
print("=" * 50)

print("""
┌─────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER BLOCK                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT (word_embeddings + positional_encoding)                  │
│       │                                                         │
│       ▼                                                         │
│  MULTI-HEAD SELF-ATTENTION                                      │
│       │ → Each word looks at all others                         │
│       ▼                                                         │
│  ADD + NORMALIZE (residual connection)                          │
│       │ → output = input + attention_output                     │
│       ▼                                                         │
│  FEED-FORWARD NETWORK (per-word MLP)                            │
│       │ → Two linear layers with ReLU in between                │
│       ▼                                                         │
│  ADD + NORMALIZE                                                │
│       │                                                         │
│       ▼                                                         │
│  OUTPUT (context-aware word representations)                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Input = token embeddings + positional encoding (at position 0)
Output = transformed representations (same shape, now context-aware)
      
1. MULTI-HEAD SELF-ATTENTION (explained above)
   - Multiple attention mechanisms in parallel
   - Each head learns different relationships (grammar, coreference, proximity)
   - Outputs concatenated and projected

2. RESIDUAL CONNECTION ("ADD")
   - output = input + attention_output
   - Prevents vanishing gradients in deep networks
   - Allows model to learn "adjustments" rather than overwriting

3. LAYER NORMALIZATION ("NORM")
   - Normalizes activations to have mean=0, std=1
   - Stabilizes training
   - Applied BEFORE attention (Pre-LN) or AFTER (Post-LN)

4. FEED-FORWARD NETWORK (FFN)
   - Two linear layers with ReLU in between
   - Same FFN applied independently to EACH position
   - Formula: FFN(x) = ReLU(x × W1 + b1) × W2 + b2
   - Adds non-linearity and transforms representations

COMPLETE BLOCK SEQUENCE:
   Input → Norm → Attention → Add → Norm → FFN → Add → Output
   * Norm can be before or after attention/FFN (Pre-LN vs Post-LN)


For text encoding, we take the final outputs (one embedding per token)and pool them
to get a single embedding for the entire text (comparing image to text, not token).
  Common pooling methods:
  - CLS token (BERT): Use the first token's output as the sentence embedding
  - Mean pooling: Average all token outputs to get a single vector
  - Max pooling: Take the max value across tokens for each dimension
""")