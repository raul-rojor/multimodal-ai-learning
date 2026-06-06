# ============================================
# PART 2: MORNING CONTINUED - CHAIN RULE & TOTAL PROBABILITY (45 min)
# ============================================

print("\n" + "=" * 60)
print("MORNING PART 2: CHAIN RULE & LAW OF TOTAL PROBABILITY")
print("=" * 60)

"""
CHAIN RULE & LAW OF TOTAL PROBABILITY
File: probability_rules.py

TWO CRITICAL RULES FOR MULTIMODAL AI:

1. CHAIN RULE: P(A,B) = P(A|B) * P(B)
   Extends to multiple variables: P(A,B,C) = P(A|B,C) * P(B|C) * P(C)

2. LAW OF TOTAL PROBABILITY: P(A) = sum(P(A|B_i) * P(B_i))
   Used when you don't know B directly
"""

import numpy as np

print("=" * 50)
print("RULE 1: THE CHAIN RULE")
print("=" * 50)

# Scenario: Image classification with two features
# Let's say: P(Cat) = 0.3
#            P(Furry | Cat) = 0.9
#            P(Furry | Not Cat) = 0.2

p_cat = 0.3
p_not_cat = 0.7
p_furry_given_cat = 0.9
p_furry_given_not_cat = 0.2

# Chain rule: P(Cat, Furry) = P(Furry | Cat) * P(Cat)
p_cat_and_furry = p_furry_given_cat * p_cat
p_not_cat_and_furry = p_furry_given_not_cat * p_not_cat

print(f"P(Cat, Furry) = P(Furry|Cat) * P(Cat) = {p_furry_given_cat} * {p_cat} = {p_cat_and_furry}")
print(f"P(Not Cat, Furry) = {p_not_cat_and_furry}")

# Total probability of seeing Furry
p_furry = p_cat_and_furry + p_not_cat_and_furry
print(f"\\nP(Furry) = {p_furry}")

# Now reverse: If we see Furry, what's P(Cat|Furry)?
p_cat_given_furry = p_cat_and_furry / p_furry
print(f"\\nP(Cat | Furry) = {p_cat_given_furry:.3f}")

print("\\n" + "=" * 50)
print("RULE 2: LAW OF TOTAL PROBABILITY")
print("=" * 50)

print("""
Law of Total Probability: P(A) = Σ P(A|B_i) * P(B_i)

This is useful when you know:
  - How A behaves under different conditions (B_i)
  - How likely each condition (B_i) is

Example: Probability of getting a notification on your phone
  B1 = weekday (60% of days), P(notif|weekday) = 0.8
  B2 = weekend (40% of days), P(notif|weekend) = 0.2
  
  P(notif) = 0.8*0.6 + 0.2*0.4 = 0.48 + 0.08 = 0.56
""")

# TODO 1: Apply law of total probability to a medical test
# Disease rate: 1%
# Test positive given disease: 99%
# Test positive given no disease: 5%

p_disease = 0.01
p_no_disease = 0.99
p_pos_given_disease = 0.99
p_pos_given_no = 0.05

# Calculate P(Positive) using law of total probability
p_positive = (p_pos_given_disease * p_disease) + (p_pos_given_no * p_no_disease)

# Calculate P(Disease|Positive) using Bayes
p_disease_given_pos = (p_pos_given_disease * p_disease) / p_positive

print("MEDICAL TEST EXAMPLE (from Monday evening):")
print(f"P(Positive) = {p_positive:.4f}")
print(f"P(Disease|Positive) = {p_disease_given_pos:.4f} (only {p_disease_given_pos*100:.1f}%!)")

# ============================================
# TODO 2: Chain rule for three variables
# ============================================

print("\\n" + "=" * 50)
print("CHAIN RULE FOR 3 VARIABLES")
print("=" * 50)

# Scenario: Image contains (Cat, Dark, Indoor)
# P(Cat) = 0.3
# P(Dark | Cat) = 0.4
# P(Indoor | Cat, Dark) = 0.8

p_cat = 0.3
p_dark_given_cat = 0.4
p_indoor_given_cat_dark = 0.8

# Chain rule: P(Cat, Dark, Indoor) = P(Indoor|Cat,Dark) × P(Dark|Cat) × P(Cat)
p_all = p_indoor_given_cat_dark * p_dark_given_cat * p_cat

print(f"P(Cat, Dark, Indoor) = {p_all:.4f}")
print(f"Interpretation: {p_all*100:.2f}% of all images are cats in dark indoor settings")

print("\\n" + "=" * 50)
print("WHY THIS MATTERS FOR MULTIMODAL AI")
print("=" * 50)
print("""
Chain Rule → Training language models predicts next word given previous words
  P(word5 | word1,word2,word3,word4) * P(word4|word1,word2,word3) * ...

Law of Total Probability → Marginalizing over unknown variables
  When you have P(image|text) but need P(image), sum over all possible texts
  
Bayes' Rule → The foundation of CLIP's contrastive learning
  P(text|image) ∝ P(image|text) * P(text)
  
You're now ready to understand these in papers!
""")