"""
DISCRETE RANDOM VARIABLES + EXPECTED VALUE - MORNING SCRIPT
Save as: discrete_rv_practice.py
Complete each TODO. Run after each TODO to check your work.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ============================================
# TODO 1: Roll a die 1000 times
# ============================================
# Compare empirical mean vs theoretical E[X] = 3.5

# Roll a fair die (faces 1-6) 1000 times
# Hint: np.random.randint(1, 7, size=1000)

rolls = np.random.randint(1, 7, size=1000)  # REPLACE None with 1000 random dice rolls
print(f"First 10 rolls: {rolls[:10]}")
empirical_mean = np.mean(rolls)  # REPLACE None with np.mean(rolls)
theoretical_mean = 3.5

print("=" * 50)
print("TODO 1: Die Rolls")
print(f"Empirical mean (1000 rolls): {empirical_mean:.3f}")
print(f"Theoretical E[X]: {theoretical_mean}")
print(f"Difference: {abs(empirical_mean - theoretical_mean):.3f}")

# ============================================
# TODO 2: Sum of two dice
# ============================================
# Simulate rolling two dice 10,000 times, plot distribution

# Roll two dice 10,000 times (each die 1-6)
# Hint: np.random.randint(1, 7, size=(10000, 2)) then sum across axis=1

dice1 = np.random.randint(1, 7, size=10000)  # REPLACE: first die rolls
dice2 = np.random.randint(1, 7, size=10000)  # REPLACE: second die rolls
sums = dice1 + dice2   # REPLACE: dice1 + dice2

# Count frequency of each sum (2 through 12)
sum_counts = Counter(sums)
print(f"Sum counts: {sum_counts}")
# Print probabilities
print("\n" + "=" * 50)
print("TODO 2: Sum of Two Dice")
print("Sum : Probability")
for s in range(2, 13):
    prob = sum_counts[s] / 10000
    print(f"  {s}  : {prob:.3f}")

# Plot distribution
plt.figure(figsize=(10, 5))
plt.bar(sum_counts.keys(), [count/10000 for count in sum_counts.values()])
plt.xlabel('Sum of two dice')
plt.ylabel('Probability')
plt.title('Distribution of Sums from Two Dice (10,000 rolls)')
plt.xticks(range(2, 13))
plt.grid(True, alpha=0.3)
plt.show()

# ============================================
# TODO 3: Lottery ticket expected value
# ============================================
# Ticket costs $10
# Prizes: $100 (1% chance), $20 (5% chance), $0 (94% chance)

ticket_cost = 10

# Create arrays of outcomes and probabilities
outcomes = [100, 20, 0]  # REPLACE: [100, 20, 0] (prize amounts)
probabilities = [0.01, 0.05, 0.94]  # REPLACE: [0.01, 0.05, 0.94]

# Calculate expected winnings E[prize]
expected_winnings = sum(outcomes[i] * probabilities[i] for i in range(len(outcomes)))  # REPLACE: sum(outcomes[i] * probabilities[i] for i in range(len(outcomes)))

# Calculate expected profit = expected_winnings - ticket_cost
expected_profit = expected_winnings - ticket_cost  # REPLACE: expected_winnings - ticket_cost

print("\n" + "=" * 50)
print("TODO 3: Lottery Ticket")
print(f"Ticket cost: ${ticket_cost}")
print(f"Expected winnings: ${expected_winnings:.2f}")
print(f"Expected profit: ${expected_profit:.2f}")

if expected_profit < 0:
    print("Conclusion: Don't play. You lose money on average.")
else:
    print("Conclusion: Play! You make money on average.")

# ============================================
# TODO 4: Prove linearity of expectation
# ============================================
# Show that E[aX + b] = aE[X] + b

# Roll a die 100,000 times
X = np.random.randint(1, 7, size=100000)
a, b = 2, 3

# Method 1: Compute E[2X + 3] directly
transformed = a * X + b  # REPLACE: a * X + b
empirical_E_direct = np.mean(transformed)  # REPLACE: np.mean(transformed)

# Method 2: Compute 2*E[X] + 3
empirical_E_X = np.mean(X)  # REPLACE: np.mean(X)
empirical_E_formula = a * empirical_E_X + b  # REPLACE: a * empirical_E_X + b

print("\n" + "=" * 50)
print("TODO 4: Linearity of Expectation")
print(f"E[X] = {empirical_E_X:.3f}")
print(f"Method 1 (direct): E[2X + 3] = {empirical_E_direct:.3f}")
print(f"Method 2 (formula): 2E[X] + 3 = {empirical_E_formula:.3f}")
print(f"Match? {'✓ YES' if abs(empirical_E_direct - empirical_E_formula) < 0.01 else '✗ NO'}")

# ============================================
# BONUS: Probability mass function visualization
# ============================================
# Optional: Plot PMF of a single die

print("\n" + "=" * 50)
print("BONUS: PMF of a Fair Die")
print("Theoretical: P(X=x) = 1/6 ≈ 0.1667 for x=1,...,6")

# Simulate 10,000 die rolls
die_rolls = np.random.randint(1, 7, size=10000)
unique, counts = np.unique(die_rolls, return_counts=True)
empirical_pmf = counts / 10000

plt.figure(figsize=(8, 4))
plt.bar(unique, empirical_pmf, alpha=0.7, label='Empirical')
plt.axhline(y=1/6, color='red', linestyle='--', label='Theoretical (1/6)')
plt.xlabel('Die face')
plt.ylabel('Probability')
plt.title('PMF of a Fair Die')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n✅ Morning complete! Run all TODOs and verify outputs.")