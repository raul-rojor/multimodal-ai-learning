# ============================================
# PART 2: MORNING - SELF-ASSESSMENT QUIZ (45 min)
# ============================================

print("\n" + "=" * 60)
print("MORNING PART 2: SELF-ASSESSMENT QUIZ")
print("=" * 60)

"""
WEEK 1 SELF-ASSESSMENT
File: week1_self_assessment.py

Answer each question honestly. No peeking at answers until you write yours.
"""

import numpy as np

print("=" * 50)
print("QUESTION 1")
print("=" * 50)
print("You have logits [2.0, 1.0, 0.0] and temperature = 0.5.")
print("After softmax, approximately what is P(class 0)?")
print("A) 0.33  B) 0.50  C) 0.85  D) 0.99")
your_answer = input("\nYour answer (A/B/C/D): ")

# Calculate
logits = np.array([2.0, 1.0, 0.0])
temp = 0.5
scaled = logits / temp
probs = np.exp(scaled) / np.sum(np.exp(scaled))
print(f"\nActual answer: C (0.85)")
print(f"Calculation: {probs.round(3)}")

print("\n" + "=" * 50)
print("QUESTION 2")
print("=" * 50)
print("Why does CLIP use cross-entropy loss instead of MSE?")
print("A) MSE is harder to implement")
print("B) Cross-entropy punishes wrong confident predictions more")
print("C) MSE only works for regression")
print("D) Both B and C")
your_answer = input("\nYour answer (A/B/C/D): ")

print("\nActual answer: D")
print("Cross-entropy is for classification. It heavily penalizes confident wrong answers.")

print("\n" + "=" * 50)
print("QUESTION 3")
print("=" * 50)
print("What does the similarity matrix compute?")
print("A) Difference between embeddings")
print("B) Dot product of every image with every text embedding")
print("C) Loss value")
print("D) Gradient direction")
your_answer = input("\nYour answer (A/B/C/D): ")

print("\nActual answer: B")
print("similarity = image_embeddings @ text_embeddings.T")

print("\n" + "=" * 50)
print("QUESTION 4")
print("=" * 50)
print("A disease affects 0.5% of population. Test is 99% accurate independently of actually having the disease.")
print("If you test positive, what is P(disease|positive)?")
print("A) 99%  B) 50%  C) 33%  D) 10%")
your_answer = input("\nYour answer (A/B/C/D): ")

p_disease = 0.005
p_pos_given_disease = 0.99
p_pos_given_no = 0.01
p_positive = p_pos_given_disease * p_disease + p_pos_given_no * (1 - p_disease)
p_disease_given_pos = (p_pos_given_disease * p_disease) / p_positive
print(f"\nActual answer: C ({p_disease_given_pos*100:.0f}%)")

print("\n" + "=" * 50)
print("QUESTION 5")
print("=" * 50)
print("What does SVD prove about multimodal learning?")
print("A) Images are better than text")
print("B) Any similarity matrix can be factored into embeddings")
print("C) Temperature must be 1.0")
print("D) Cross-entropy is the only loss function")
your_answer = input("\nYour answer (A/B/C/D): ")

print("\nActual answer: B")
print("SVD = U @ S @ Vt. U and V can be image/text embeddings.")

print("\n" + "=" * 50)
print("SCORING")
print("=" * 50)
print("5/5 = Ready for Week 2")
print("4/5 = Review missed concept")
print("3 or fewer = Re-run week1_complete_review.py")