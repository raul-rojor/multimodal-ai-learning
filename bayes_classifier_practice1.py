"""
MEDICAL TEST SCENARIO - TONIGHT'S PRACTICE
Complete this script. All imports and functions are provided.
You just fill in the 5 missing lines marked with TODO.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================
# GIVEN DATA (Real medical statistics)
# ============================================

# Prior: P(Disease) = 1% of population has the disease
p_disease = 0.01

# Prior: P(No Disease) = 99% don't have it
p_no_disease = 0.99

# Likelihood: P(Positive | Disease) = Test catches 99% of true cases
p_positive_given_disease = 0.99

# Likelihood: P(Positive | No Disease) = 5% false positive rate
p_positive_given_no_disease = 0.05

# ============================================
# YOUR TASK: Calculate P(Disease | Positive)
# ============================================
# Use Bayes' Rule:
# 
# P(Disease | Positive) = P(Positive | Disease) × P(Disease) 
#                         ───────────────────────────────────
#                              P(Positive)
#
# Where P(Positive) = P(Positive|Disease)×P(Disease) + P(Positive|No Disease)×P(No Disease)

# TODO 1: Calculate numerator = P(Positive|Disease) × P(Disease)
numerator = p_positive_given_disease * p_disease  # <-- REPLACE None with the correct calculation

# TODO 2: Calculate evidence = P(Positive)
# This is the TOTAL probability of testing positive (from both sick and healthy people)
evidence = p_disease * p_positive_given_disease + p_no_disease * p_positive_given_no_disease  # <-- REPLACE None with the correct calculation

# TODO 3: Apply Bayes' Rule
p_disease_given_positive = numerator / evidence  # <-- REPLACE None with numerator / evidence

# ============================================
# VISUALIZATION (Automatically shows your answer)
# ============================================

# TODO 4: Create a bar chart comparing prior vs posterior
# Use plt.bar() with categories ['Prior P(Disease)', 'Posterior P(Disease|Positive)']
# Values should be [p_disease, p_disease_given_positive]
# Add colors: 'gray' for prior, 'blue' for posterior
# Add ylabel 'Probability', title 'Bayes Update', and plt.show()

# Your code here (5-8 lines)
print('TIME TO VISUALIZE ;) \n' \
'==========================')
plt.figure(figsize=(8, 5))
bars = plt.bar(['Prior P(disease)', 'Posterior P(disease|positive)'], [p_disease, p_disease_given_positive], color=['gray', 'blue'])
plt.ylabel('Probability')
plt.title('Bayes Update || YOU ARE STILL PROBABLY NOT DISEASED')
plt.ylim(0, 1)
plt.grid(True, alpha=0.35)
for bar, prob in zip(bars, [p_disease, p_disease_given_positive]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{prob:.3f}', ha='center', va='bottom')
plt.show()

# TODO 5: Print interpretation
print(f"Prior probability of disease: {p_disease:.2%}")
print(f"Posterior probability given positive test: {p_disease_given_positive:.2%}")
print()
print("INTERPRETATION:")
# TODO: Write a one-sentence explanation of why the posterior is not 99%
# (Even though the test is 99% accurate, the disease is rare)
print("Even though the test is 99% accurate, only 16.667% of positive tests actually have the disease because the high " \
"number of people without the disease is so large that even a small false positive rate creates a large number of " \
"false alarms compared to real positive cases.")

# ============================================
# BONUS (Optional - only if you finish early)
# ============================================

# BONUS 1: What if the disease was more common? 
# Change p_disease to 0.10 (10%) and re-run. How does p_disease_given_positive change?
# Write your prediction before running.
# P(disease | positive) will increase by  because the proportion of positive real cases to false positives will be higher.

# BONUS 2: What if the test was more specific?
# Change p_positive_given_no_disease to 0.01 (1% false positive) and re-run with p_disease=0.01
# P(disease | positive) will increase because the proportion of false alarms to real positives will fall.

# ============================================
# ANSWERS (Check yourself here - no peeking until done)
# ============================================

# Expected output when done:
# Prior probability of disease: 1.00%
# Posterior probability given positive test: 16.67%
# 
# Interpretation: Even with a 99% accurate test, only ~17% of positive tests 
# are true positives because the disease is rare (1% prevalence). 
# The 5% false positive rate on the 99% healthy population creates many false alarms.

# Correct code:
# numerator = p_positive_given_disease * p_disease  # 0.99 * 0.01 = 0.0099
# evidence = (p_positive_given_disease * p_disease) + (p_positive_given_no_disease * p_no_disease)  # 0.0099 + 0.0495 = 0.0594
# p_disease_given_positive = numerator / evidence  # 0.0099 / 0.0594 = 0.1667

# Bar chart code:
# plt.bar(['Prior P(Disease)', 'Posterior P(Disease|Positive)'], 
#         [p_disease, p_disease_given_positive], 
#         color=['gray', 'blue'])
# plt.ylabel('Probability')
# plt.title('Bayes Update: How a Positive Test Changes Belief')
# plt.ylim(0, 1)
# plt.show()