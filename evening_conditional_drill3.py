"""
EVENING PRACTICE: CONDITIONAL PROBABILITY DRILL

Complete these 3 TODOs without peeking at morning code.
"""

import numpy as np

# ============================================
# TODO 1: Given joint table, find conditional
# ============================================

# Joint distribution of (Weather, Temperature)
# Values: P(Sunny,Hot)=0.3, P(Sunny,Cold)=0.2, P(Rainy,Hot)=0.1, P(Rainy,Cold)=0.4

joint = {
    ('Sunny','Hot'): 0.3,
    ('Sunny','Cold'): 0.2,
    ('Rainy','Hot'): 0.1,
    ('Rainy','Cold'): 0.4,
}

# Calculate P(Hot | Sunny)
p_sunny = joint[('Sunny','Hot')] + joint[('Sunny','Cold')]
p_hot_and_sunny = joint[('Sunny','Hot')]
p_hot_given_sunny = p_hot_and_sunny / p_sunny  # REPLACE: p_hot_and_sunny / p_sunny

print("=" * 50)
print("TODO 1: Conditional from Joint")
print(f"P(Hot | Sunny) = {p_hot_given_sunny}")

# ============================================
# TODO 2: Apply Bayes' Rule
# ============================================

# Given:
# P(Disease) = 0.02 (2% have disease)
# P(Positive | Disease) = 0.98
# P(Positive | No Disease) = 0.03

p_disease = 0.02
p_pos_given_disease = 0.98
p_pos_given_no = 0.03

# Calculate P(Disease | Positive)
# Step 1: P(Positive) using law of total probability
p_positive = p_pos_given_disease * p_disease + p_pos_given_no * (1 - p_disease)   # REPLACE

# Step 2: Bayes' Rule
p_disease_given_positive = p_pos_given_disease * p_disease / p_positive  # REPLACE

print("\n" + "=" * 50)
print("TODO 2: Bayes' Rule")
print(f"P(Disease | Positive) = {p_disease_given_positive:.4f}")

# ============================================
# TODO 3: Chain rule for 3 variables
# ============================================

# P(A) = 0.4
# P(B|A) = 0.7
# P(C|A,B) = 0.9

# Calculate P(A,B,C) using chain rule
p_ABC = 0.9 * 0.7 * 0.4  # REPLACE

print("\n" + "=" * 50)
print("TODO 3: Chain Rule")
print(f"P(A,B,C) = {p_ABC}")

# ============================================
# ANSWERS (Check after attempting)
# ============================================

print("\n" + "=" * 50)
print("ANSWERS (Don't peek until you try)")
print("=" * 50)
print("""
TODO 1: P(Hot|Sunny) = 0.3 / (0.3+0.2) = 0.6

TODO 2: P(Positive) = 0.98*0.02 + 0.03*0.98 = 0.0196 + 0.0294 = 0.049
        P(Disease|Positive) = 0.0196 / 0.049 = 0.4

TODO 3: P(A,B,C) = 0.9 × 0.7 × 0.4 = 0.252
""")