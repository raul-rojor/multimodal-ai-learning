"""
GRADIENT DESCENT ON A SIMPLE PROBLEM

Problem: Find weight that makes prediction = target
Target: when input=1, we want output=1
Prediction = weight * input (no bias for simplicity)
Loss = (prediction - target)²  (mean squared error)
"""

print("""
WHAT GRADIENT DESCENT DOES:
Finds the best weight and bias by trial and error, but SMART trial and error.

ANALOGY: Finding the bottom of a valley in FOG
- You can't see the bottom (don't know best weight)
- You can feel the slope at your feet (gradient)
- You step DOWNHILL (opposite direction of gradient)
- Repeat until you reach the bottom (minimum loss)

FORMULA:
    new_weight = old_weight - learning_rate * gradient

WHERE:
    learning_rate = how big your steps are (too big = overshoot, too small = slow)
    gradient = which direction is uphill (negative gradient = downhill)
""")

import numpy as np
import matplotlib.pyplot as plt

# The problem: find weight such that weight * 1 = 1
# Answer should be weight = 1.0

target = 1.0
input_value = 1.0

def predict(weight):
    return weight * input_value

def loss(weight):
    prediction = predict(weight)
    return (prediction - target) ** 2

def gradient(weight):
    # Derivative of loss w.r.t weight: 2 * (weight - target) * input
    # For input=1, this simplifies to: 2 * (weight - target)
    return 2 * (weight - target)

# Gradient descent loop
weight = 3.0  # start far from正确答案 (3 instead of 1)
learning_rate = 0.1
history = []

print("Starting weight:", weight)
print("Target weight: 1.0")
print()

for step in range(20):
    current_loss = loss(weight)
    grad = gradient(weight)
    weight = weight - learning_rate * grad  # step downhill
    history.append(weight)
    print(f"Step {step+1:2d}: weight={weight:.4f}, loss={current_loss:.6f}")

print(f"\nFinal weight: {weight:.4f} (should be close to 1.0)")

# Visualize the path
plt.plot(range(1, 21), history, 'bo-')
plt.axhline(y=1.0, color='red', linestyle='--', label='Target weight=1.0')
plt.xlabel('Gradient descent step')
plt.ylabel('Weight value')
plt.title('Gradient Descending to the Correct Weight')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("""
KEY INSIGHTS:
1. Started at 3.0, ended near 1.0
2. Each step reduced the loss
3. Learning rate controls step size (0.1 worked well)
4. Too large learning rate: would bounce around
5. Too small learning rate: would take forever
""")