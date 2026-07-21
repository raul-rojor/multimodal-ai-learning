"""
TOPIC 4: CONTRASTIVE LEARNING
"""

print("""
WHAT CONTRASTIVE LEARNING DOES:
Teaches a model that:
  - Image + correct text should have HIGH similarity
  - Image + wrong text should have LOW similarity

THE TRICK: No labeled categories needed
  - Just pairs of (image, caption) from the internet
  - Model learns: pull matching pairs together, push mismatched apart

CLIP'S CONTRASTIVE LOSS (InfoNCE):
  - For each image in batch, find its matching text
  - Maximize similarity of correct pair
  - Minimize similarity of all other (image, wrong_text) pairs

THIS IS WHY YOU NEEDED PROBABILITY + EXPECTATION:
  - Contrastive loss = -E[log( P(correct_pair) )]
  - Where P(correct_pair) = exp(sim) / sum(exp(sim over all pairs))
  - That's softmax with temperature! (You learned this yesterday)
""")

# No code for this topic - just conceptual understanding
print("""
CONCEPT CHECK:
1. Contrastive learning needs PAIRS (image, text)
2. It pulls correct pairs together (high similarity)
3. It pushes incorrect pairs apart (low similarity)
4. The loss function is cross-entropy on similarity scores
5. Temperature controls how "picky" the model is

YOU ALREADY UNDERSTAND THIS IF:
- You know what softmax does
- You know what cross-entropy does
- You know why P(text) matters (from yesterday)
""")