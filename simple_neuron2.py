"""
SIMPLE NEURON FROM SCRATCH
"""

print("""
WHAT YOU'RE BUILDING:
A tiny neural network that takes an input (e.g., pixel brightness) 
and outputs a prediction (e.g., is it a cat?).

FORMULA FOR ONE NEURON:
    output = activation(weight * input + bias)

WHERE:
    weight = how strongly input influences output
    bias = baseline output when input is zero
    activation = a function that squashes output (e.g., sigmoid, ReLU)
""")

import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
    
    def sigmoid(self, x):
        """S-shaped activation function. Squashes output between 0 and 1."""
        return 1 / (1 + np.exp(-x))
    
    def forward(self, input_value):
        """Forward pass: output = sigmoid(weight * input + bias)"""
        z = self.weight * input_value + self.bias
        output = self.sigmoid(z)
        return output

# TODO 1: Create a neuron that detects "bright pixel = cat"
# If input brightness is 0 (dark) → output near 0 (not cat)
# If input brightness is 1 (bright) → output near 1 (cat)
# What weight and bias do you need? Try weight=5, bias=-2.5

neuron = Neuron(weight=5, bias=-2.5)

print("Testing neuron:")
for brightness in [0, 0.25, 0.5, 0.75, 1.0]:
    output = neuron.forward(brightness)
    print(f"  brightness={brightness} → P(cat)={output:.3f}")

# MY OWN PRACTICE: THE DARKER THE INPUT, THE MORE LIKELY IT'S A BAT. ONLY LIKELY IF REALLYYYY DARK
bat_neuron = Neuron(weight=-100, bias=2.5)

# TODO 2: Visualize the neuron's decision boundary
inputs = np.linspace(0, 1, 100)
outputs = [neuron.forward(x) for x in inputs]


plt.plot(inputs, outputs)
plt.xlabel('Input brightness')
plt.ylabel('P(cat)')
plt.title('Neuron: Brightness → Cat Probability')
plt.axhline(y=0.5, color='red', linestyle='--', label='Decision threshold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

bat_outputs = [bat_neuron.forward(x) for x in inputs]
plt.plot(inputs, bat_outputs, label='Bat Neuron', color='orange')
plt.xlabel('Input brightness')
plt.ylabel('P(bat)')
plt.title('Neuron: Brightness → Bat Probability')
plt.axhline(y=0.5, color='red', linestyle='--', label='Decision threshold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print("""
YOUR UNDERSTANDING CHECK:
- If weight is larger, the transition from 0→1 becomes sharper
- If bias is more negative, the neuron needs brighter input to activate
- This is exactly how a single neuron "decides" based on input
""")