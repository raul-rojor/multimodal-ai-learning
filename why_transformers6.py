"""
WHY TEXT NEEDS TRANSFORMERS

THE PROBLEM WITH RAW TEXT:
  - Variable length: "cat" (1 word) vs "the fluffy cat sat on the mat" (7 words)
  - Word order matters: "dog bites man" ≠ "man bites dog"
  - Words relate to each other: "it" might refer to "cat" 5 words earlier

TRADITIONAL SOLUTIONS (that didn't work well):
  - RNNs/LSTMs: Process sequentially, slow, forget long-range dependencies
  - CNNs on text: Good but limited receptive field

TRANSFORMER SOLUTION:
  - Self-attention: Every word looks at every other word in ONE step
  - Parallel processing (not sequential like RNNs)
  - Captures long-range dependencies easily
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("TEXT AS NUMBERS")
print("=" * 50)

# Simple vocabulary
vocab = {"cat": 0, "sat": 1, "on": 2, "the": 3, "mat": 4}
sentence = "the cat sat on the mat"
words = sentence.split()

# Convert to numbers
token_ids = [vocab[word] for word in words]
print(f"Sentence: {sentence}")
print(f"Words: {words}")
print(f"Token IDs: {token_ids}")

print("\n" + "=" * 50)
print("WHY WORD ORDER MATTERS")
print("=" * 50)

sentence1 = "dog bites man"
sentence2 = "man bites dog"

print(f"Sentence 1: {sentence1}")
print(f"Sentence 2: {sentence2}")
print("Same words, different order → completely different meaning!")

print("\n" + "=" * 50)
print("WHY LONG-RANGE DEPENDENCIES MATTER")
print("=" * 50)

long_sentence = "The cat that lived in the house on the hill with the red door sat on the mat"
print(f"Sentence: {long_sentence}")
print("'sat' refers back to 'cat' (8 words earlier)")
print("Transformer attention can connect these directly")

print("\n" + "=" * 50)
print("RNN vs TRANSFORMER")
print("=" * 50)

print("""
RNN (Old approach):
  Word1 → Word2 → Word3 → ... → WordN
  (Sequential, slow, loses information at each step)

TRANSFORMER (Modern approach):
  Word1 ──┐
  Word2 ──┼── ALL PAIRS SIMULTANEOUSLY ──→ Output
  Word3 ──┤
  ...   ──┘
  (Parallel, fast, every word sees every other word)
""")

# Visualize attention matrix
plt.figure(figsize=(8, 6))

# Simulated attention weights (which words pay attention to which)
words_display = ["the", "cat", "sat", "on", "the", "mat"]
n = len(words_display)
attention = np.random.rand(n, n)
# Make it look like real attention (diagonal + some off-diagonal)
attention = attention / attention.sum(axis=1, keepdims=True)

plt.imshow(attention, cmap='Blues', aspect='auto')
plt.colorbar(label='Attention Weight')
plt.xticks(range(n), words_display, rotation=45)
plt.yticks(range(n), words_display)
plt.xlabel('Words being ATTENDED TO')
plt.ylabel('Words DOING THE ATTENDING')
plt.title('Self-Attention Matrix\nEach row shows which words a word pays attention to')
plt.tight_layout()
plt.show()

print("\nInterpretation: Row 'cat' shows which words 'cat' pays attention to")
print("High weight on diagonal = word pays attention to itself")
print("High weight off-diagonal = word relates to other words")