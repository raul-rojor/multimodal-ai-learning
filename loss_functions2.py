# ============================================
# INDEPENDENT TOPIC 3: LOSS FUNCTIONS
# ============================================

print("\n" + "=" * 60)
print("TOPIC 3: LOSS FUNCTIONS (30 min)")
print("=" * 60)

print("""
WHAT A LOSS FUNCTION DOES:
Measures how WRONG your prediction is. Lower loss = better prediction.

TWO COMMON LOSS FUNCTIONS FOR MULTIMODAL AI:

1. MEAN SQUARED ERROR (MSE)
   Formula: (prediction - target)²
   Use when: predicting a number (regression)
   Example: Predicting image brightness

2. CROSS-ENTROPY LOSS
   Formula: -target * log(prediction)
   Use when: predicting a probability (classification)
   Example: "Is this a cat or dog?" → cross-entropy

WHY CROSS-ENTROPY FOR MULTIMODAL:
   CLIP uses cross-entropy because it's comparing distributions
   (image should match its correct text with high probability)
""")

# File to create: loss_functions.py
"""
LOSS FUNCTIONS FROM SCRATCH
Save as: loss_functions.py
"""

import numpy as np
import matplotlib.pyplot as plt

def mse_loss(prediction, target):
    """Mean squared error"""
    return (prediction - target) ** 2

def cross_entropy_loss(prediction, target):
    """
    Cross-entropy loss.
    prediction: probability (between 0 and 1)
    target: 0 or 1 (true class)
    """
    # Avoid log(0) by adding tiny epsilon
    epsilon = 1e-8
    prediction = np.clip(prediction, epsilon, 1 - epsilon)
    
    if target == 1:
        return -np.log(prediction)
    else:
        return -np.log(1 - prediction)

# TODO 1: Compare losses for different predictions
print("=" * 50)
print("LOSS COMPARISON (target=1, meaning 'it IS a cat')")
print("-" * 50)

predictions = [0.01, 0.25, 0.5, 0.75, 0.99]
for pred in predictions:
    mse = mse_loss(pred, 1)
    ce = cross_entropy_loss(pred, 1)
    print(f"P(cat)={pred:.2f} → MSE={mse:.4f}, CrossEntropy={ce:.4f}")

print("\nWhen prediction is WRONG (0.01 instead of 1):")
print("  MSE = 0.9801 (moderate penalty)")
print("  CrossEntropy = 4.6052 (HUGE penalty)")

print("\nWhen prediction is CORRECT (0.99 instead of 1):")
print("  MSE = 0.0001 (tiny penalty)")
print("  CrossEntropy = 0.0101 (tiny penalty)")

print("\nKEY: Cross-entropy PUNISHES wrong answers much harder.")
print("      This is why classifiers use cross-entropy, not MSE.")

# TODO 2: Visualize loss functions
predictions = np.linspace(0, 1, 100)
mse_values = [mse_loss(p, 1) for p in predictions]
ce_values = [cross_entropy_loss(p, 1) for p in predictions]

plt.figure(figsize=(10, 5))
plt.plot(predictions, mse_values, label='MSE', linewidth=2)
plt.plot(predictions, ce_values, label='Cross-Entropy', linewidth=2)
plt.xlabel('Predicted P(cat)')
plt.ylabel('Loss')
plt.title('Loss Functions (target=1, meaning it IS a cat)')
plt.axvline(x=1.0, color='red', linestyle='--', label='Correct prediction')
plt.yscale('log')  # Log scale to see cross-entropy spike
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("""
OBSERVATIONS FROM PLOT:
- MSE: symmetric, gentle curve
- Cross-entropy: EXPLODES as prediction → 0 (very wrong)
- Cross-entropy near 0 loss when prediction → 1 (very right)
- This asymmetry makes cross-entropy perfect for classification
""")