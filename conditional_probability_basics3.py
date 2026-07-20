"""
CONDITIONAL PROBABILITY BASICS

WHAT YOU'RE LEARNING:
  P(A|B) = "Probability of A given that B happened"
  
EXAMPLE: 
  P(Rain | Clouds) = probability it rains given you see clouds
  This is NOT the same as P(Clouds | Rain)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ============================================
# TODO 1: Build a joint probability table
# ============================================
# Scenario: Weather and umbrella usage
# Possible weather: Sunny (S), Rainy (R)
# Possible actions: Umbrella (U), No Umbrella (N)

# Joint probabilities P(Weather, Action)
# Rows: Weather, Columns: Action

# P(S, U) = 0.05  (Sunny but brings umbrella - paranoid)
# P(S, N) = 0.45  (Sunny, no umbrella - normal)
# P(R, U) = 0.40  (Rainy, umbrella - smart)
# P(R, N) = 0.10  (Rainy, no umbrella - gets wet)

# These MUST sum to 1.0
joint_table = {
    ('Sunny', 'Umbrella'): 0.05,
    ('Sunny', 'NoUmbrella'): 0.45,
    ('Rainy', 'Umbrella'): 0.40,
    ('Rainy', 'NoUmbrella'): 0.10,
}

print("=" * 50)
print("JOINT PROBABILITY TABLE")
print("=" * 50)
print(f"P(Sunny, Umbrella)     = {joint_table[('Sunny', 'Umbrella')]}")
print(f"P(Sunny, NoUmbrella)   = {joint_table[('Sunny', 'NoUmbrella')]}")
print(f"P(Rainy, Umbrella)     = {joint_table[('Rainy', 'Umbrella')]}")
print(f"P(Rainy, NoUmbrella)   = {joint_table[('Rainy', 'NoUmbrella')]}")
print(f"SUM: {sum(joint_table.values())}")

# TODO 1a: Calculate marginal probabilities P(Weather)
# Marginal = sum over all actions for each weather
p_sunny = joint_table[('Sunny', 'Umbrella')] + joint_table[('Sunny', 'NoUmbrella')]
p_rainy = joint_table[('Rainy', 'Umbrella')] + joint_table[('Rainy', 'NoUmbrella')]

print(f"\nMARGINAL PROBABILITIES:")
print(f"P(Sunny) = {p_sunny}")
print(f"P(Rainy) = {p_rainy}")

# TODO 1b: Calculate marginal probabilities P(Action)
p_umbrella = joint_table[('Sunny', 'Umbrella')] + joint_table[('Rainy', 'Umbrella')]
p_noumbrella = joint_table[('Sunny', 'NoUmbrella')] + joint_table[('Rainy', 'NoUmbrella')]

print(f"\nP(Umbrella) = {p_umbrella}")
print(f"P(NoUmbrella) = {p_noumbrella}")

# ============================================
# TODO 2: Calculate conditional probabilities
# ============================================

# Formula: P(A|B) = P(A,B) / P(B)

# P(Umbrella | Rainy) = P(Umbrella, Rainy) / P(Rainy)
p_umbrella_given_rainy = joint_table[('Rainy', 'Umbrella')] / p_rainy

# P(Umbrella | Sunny) = P(Umbrella, Sunny) / P(Sunny)
p_umbrella_given_sunny = joint_table[('Sunny', 'Umbrella')] / p_sunny

# P(Rainy | Umbrella) = P(Rainy, Umbrella) / P(Umbrella)
p_rainy_given_umbrella = joint_table[('Rainy', 'Umbrella')] / p_umbrella

print("\n" + "=" * 50)
print("CONDITIONAL PROBABILITIES")
print("=" * 50)
print(f"P(Umbrella | Rainy) = {p_umbrella_given_rainy:.3f}")
print(f"P(Umbrella | Sunny) = {p_umbrella_given_sunny:.3f}")
print(f"\nP(Rainy | Umbrella) = {p_rainy_given_umbrella:.3f}")

# Notice: P(Umbrella|Rainy) vs P(Rainy|Umbrella) are VERY different!
print(f"\nNOTE: P(Umbrella|Rainy)={p_umbrella_given_rainy:.3f} ≠ P(Rainy|Umbrella)={p_rainy_given_umbrella:.3f}")

# ============================================
# TODO 3: Verify Bayes' Rule
# ============================================

# Bayes: P(A|B) = P(B|A) × P(A) / P(B)
# Let's verify: P(Rainy|Umbrella) should equal P(Umbrella|Rainy) × P(Rainy) / P(Umbrella)

bayes_result = (p_umbrella_given_rainy * p_rainy) / p_umbrella

print("\n" + "=" * 50)
print("VERIFYING BAYES' RULE")
print("=" * 50)
print(f"Direct calculation P(Rainy|Umbrella) = {p_rainy_given_umbrella:.3f}")
print(f"Bayes formula result = {bayes_result:.3f}")
print(f"Match? {'✓ YES' if abs(p_rainy_given_umbrella - bayes_result) < 0.001 else '✗ NO'}")

# ============================================
# TODO 4: Visualize conditional probabilities
# ============================================

plt.figure(figsize=(10, 5))

# Subplot 1: Joint distribution
plt.subplot(1, 2, 1)
categories = ['Sunny\nUmbrella', 'Sunny\nNoUmbrella', 'Rainy\nUmbrella', 'Rainy\nNoUmbrella']
values = [joint_table[('Sunny', 'Umbrella')], 
          joint_table[('Sunny', 'NoUmbrella')],
          joint_table[('Rainy', 'Umbrella')],
          joint_table[('Rainy', 'NoUmbrella')]]
colors = ['lightgreen', 'green', 'lightcoral', 'red']
plt.bar(categories, values, color=colors)
plt.ylabel('Probability')
plt.title('Joint Distribution P(Weather, Action)')
plt.ylim(0, 0.5)

# Subplot 2: Conditional probabilities
plt.subplot(1, 2, 2)
cond_categories = ['P(Umbrella|Sunny)', 'P(Umbrella|Rainy)', 'P(Rainy|Umbrella)']
cond_values = [p_umbrella_given_sunny, p_umbrella_given_rainy, p_rainy_given_umbrella]
cond_colors = ['green', 'red', 'blue']
plt.bar(cond_categories, cond_values, color=cond_colors)
plt.ylabel('Probability')
plt.title('Conditional Probabilities')
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("KEY INSIGHTS")
print("=" * 50)
print("""
1. Joint probability P(A,B) = probability BOTH happen
2. Marginal probability P(A) = sum over all B of P(A,B)
3. Conditional P(A|B) = P(A,B) / P(B)
4. Bayes' Rule lets you REVERSE the condition
5. P(A|B) is almost NEVER equal to P(B|A)
""")