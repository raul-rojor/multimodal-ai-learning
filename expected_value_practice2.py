"""
EVENING PRACTICE: expected_value_practice.py
Complete these 3 TODOs after CodePath.
"""

import numpy as np

# ============================================
# TODO 1: Biased coin
# ============================================
# P(heads)=0.7 (value 1), P(tails)=0.3 (value 0)
# Calculate E[X] by hand, then simulate 10,000 flips

p_heads = 0.7
theoretical_E = p_heads * 1 + (1 - p_heads) * 0  # REPLACE: 1*p_heads + 0*(1-p_heads)

# Simulate 10,000 coin flips
# Hint: np.random.random(10000) < p_heads gives boolean array
flips = (np.random.random(10000) < p_heads).astype(int)  # REPLACE: (np.random.random(10000) < p_heads).astype(int)
empirical_E = np.mean(flips)  # REPLACE: np.mean(flips)

print("=" * 50)
print("EVENING TODO 1: Biased Coin")
print(f"Theoretical E[X] = {theoretical_E}")
print(f"Empirical E[X] (10,000 flips) = {empirical_E:.3f}")

# ============================================
# TODO 2: Dice game expected profit
# ============================================
# Game costs $5 to play
# Roll a die:
#   6 → win $20
#   4-5 → win $10
#   1-3 → win $0

cost = 5
outcomes = [20, 10, 0]
probs = [1/6, 2/6, 3/6]

expected_winnings = sum(o*p for o, p in zip(outcomes, probs))  # REPLACE: sum(o*p for o,p in zip(outcomes, probs))
expected_profit = expected_winnings - cost

print("\n" + "=" * 50)
print("EVENING TODO 2: Dice Game")
print(f"Expected winnings: ${expected_winnings:.2f}")
print(f"Expected profit: ${expected_profit:.2f}")
print(f"Should you play? {'No' if expected_profit < 0 else 'Yes'}")

# ============================================
# TODO 3: Simulate the game
# ============================================
# Play 100,000 times to verify expected profit

def play_one_game():
    roll = np.random.randint(1, 7)
    if roll == 6:
        return 20 - cost
    elif roll >= 4:  # 4 or 5
        return 10 - cost
    else:  # 1,2,3
        return 0 - cost

profits = [play_one_game() for _ in range(100000)]
empirical_avg_profit = np.mean(profits)

print("\n" + "=" * 50)
print("EVENING TODO 3: Simulation Verification")
print(f"Theoretical expected profit: ${expected_profit:.2f}")
print(f"Empirical average profit (100,000 games): ${empirical_avg_profit:.3f}")
print(f"Match? {'✓' if abs(empirical_avg_profit - expected_profit) < 0.05 else '✗'}")

print("\n✅ Evening complete! You're ready for Wednesday.")