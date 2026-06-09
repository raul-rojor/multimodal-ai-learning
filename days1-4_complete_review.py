# ============================================
# PART 1: MORNING - COMPLETE MATH REVIEW (1.5 hours)
# ============================================

print("\n" + "=" * 60)
print("MORNING: MASTER REVIEW - ALL CONCEPTS")
print("=" * 60)

"""
WEEK 1 COMPLETE MATH REVIEW
File: days1-4_complete_review.py

Run this file. Answer each QUESTION before the reveal.
"""

import numpy as np

print("=" * 50)
print("CONCEPT 1: BAYES' RULE")
print("=" * 50)

# QUESTION: Write the formula for Bayes' Rule
# ANSWER: P(A|B) = P(B|A) × P(A) / P(B)

# QUESTION: Why does multimodal AI need P(text) (the prior)?
# ANSWER: Without priors, model picks most accurate word ("zebra") 
#         instead of most natural word ("animal")

p_rare_word = 0.0001
p_common = 0.5
likelihood_image_is_rare_word = 0.95
likelihood_image_is_common_word = 0.90

posterior_image_is_rare_word = likelihood_image_is_rare_word * p_rare_word
posterior_image_is_common_word = likelihood_image_is_common_word * p_common

print(f"Rare word posterior: {posterior_image_is_rare_word:.6f}")
print(f"Common word posterior: {posterior_image_is_common_word:.3f}")
print("Common word wins despite lower accuracy because it's more natural")

print("\n" + "=" * 50)
print("CONCEPT 2: TEMPERATURE & SOFTMAX")
print("=" * 50)

# QUESTION: What does temperature do to probabilities?
# ANSWER: Low temp (<1) = more confident, picks top choice aggressively
#         High temp (>1) = less confident, more random/creative

logits = np.array([3.0, 1.0, 0.5])

for temp in [0.3, 1.0, 3.0]:
    probs = np.exp(logits/temp) / np.sum(np.exp(logits/temp))
    print(f"Temp={temp}: {[round(p,3) for p in probs]}")

print("\n" + "=" * 50)
print("CONCEPT 3: CROSS-ENTROPY LOSS")
print("=" * 50)

# QUESTION: What does cross-entropy measure?
# ANSWER: Surprise. Low when prediction correct, high when wrong.

def ce(pred, correct_class=True):
    import numpy as np
    pred = np.clip(pred, 1e-8, 1 - 1e-8)
    if correct_class:
        return -np.log(pred)
    else:
        return -np.log(1 - pred)

print(f"Predict 0.99 (correct) → loss = {ce(0.99):.3f}")
print(f"Predict 0.50 (correct) → loss = {ce(0.50):.3f}")
print(f"Predict 0.01 (correct) → loss = {ce(0.01):.3f}")
print("Very wrong predictions get HUGE loss (forces model to be confident)")

print("\n" + "=" * 50)
print("CONCEPT 4: MATRIX MULTIPLICATION AS SIMILARITY")
print("=" * 50)

# QUESTION: What does image_embeddings @ text_embeddings.T compute?
# ANSWER: All pairwise similarities between images and texts in a batch

images = np.random.randn(3, 4)
texts = np.random.randn(2, 4)
similarity = images @ texts.T
print(f"3 images × 2 texts → similarity matrix shape: {similarity.shape}")
print("This is the CORE computation of CLIP. One line compares everything.")

print("\n" + "=" * 50)
print("CONCEPT 5: SVD (SINGULAR VALUE DECOMPOSITION)")
print("=" * 50)

# QUESTION: What does SVD prove about multimodal learning?
# ANSWER: Any similarity matrix can be factored into image and text embeddings

A = np.array([[0.9, 0.2], [0.3, 0.8]])
U, S, Vt = np.linalg.svd(A)
similarity_from_embeddings = U @ Vt
similarity_from_svd = U @ np.diag(S) @ Vt
print(f"Original matrix:\n{A}")
print(f"Reconstructed from SVD:\n{similarity_from_svd.round(2)}")
print("SVD proves that learning embeddings to factor similarity is mathematically sound")
print("SVD IS NOT HOW CLIP LEARNS; it's a math proof working backwards to show that embedding matrices can build similarity matrices")
print("The embeddings that clip learns are not equal to U and Vt, they are two matrices that together absorb the matrix S")

print("\n" + "=" * 50)
print("CONCEPT 6: GRADIENT DESCENT")
print("=" * 50)

# QUESTION: How does gradient descent minimize loss?
# ANSWER: new_weight = old_weight - learning_rate × gradient
#         Move opposite to uphill direction to go downhill

weight = 3.0
target = 1.0
learning_rate = 0.1

for step in range(5):
    grad = 2 * (weight - target)
    weight = weight - learning_rate * grad
    print(f"Step {step+1}: weight = {weight:.3f}")

print("Weight moves from 3.0 toward target 1.0")

print("\n" + "=" * 50)
print("CONCEPT 7: EXPECTED VALUE")
print("=" * 50)

# QUESTION: What is expected value?
# ANSWER: Average outcome weighted by probability = sum(x × P(x))

outcomes = [100, 20, 0]
probs = [0.01, 0.05, 0.94]
expected = sum(o * p for o, p in zip(outcomes, probs))
print(f"Expected winnings: ${expected:.2f}")
print("Neural networks minimize expected loss over training data")

print("\n" + "=" * 50)
print("CONCEPT 8: CONDITIONAL PROBABILITY")
print("=" * 50)

# QUESTION: What is P(A|B)?
# ANSWER: Probability of A given B happened = P(A,B) / P(B)

p_cloud = 0.4
p_rain_given_cloud = 0.7
p_rain_and_cloud = p_rain_given_cloud * p_cloud
print(f"P(Rain,Cloud) = {p_rain_and_cloud}")
print("CLIP outputs P(text|image) = probability of text given the image")

print("\n" + "=" * 50)
print("THE COMPLETE FORMULA")
print("=" * 50)

print("""
Loss = -E[ log( softmax( (image_encoder(image) @ text_encoder(text).T) / temperature ) ) ]

You now understand every piece of this formula.
""")