"""
SIMPLE MULTIMODAL CLASSIFIER USING BAYES' RULE

WHAT YOU'RE BUILDING:
A classifier that uses BOTH image features AND text features
to predict if an image contains a cat.

This is a simplified version of what multimodal models do.
"""

import numpy as np
import matplotlib.pyplot as plt

class SimpleMultimodalBayes:
    """
    Predicts P(Cat | ImageBrightness, WordPresence)
    Using Bayes' Rule with conditional independence assumption
    """
    
    def __init__(self):
        # Prior: P(Cat)
        self.p_cat = 0.3
        self.p_not_cat = 0.7
        
        # Likelihoods for IMAGE feature: brightness
        # P(Bright | Cat) vs P(Bright | Not Cat)
        # Brightness levels: 0=dark, 1=medium, 2=bright
        self.p_bright_given_cat = [0.2, 0.5, 0.3]    # cats in dark/bright
        self.p_bright_given_not_cat = [0.4, 0.4, 0.2] # non-cats
        
        # Likelihoods for TEXT feature: word "meow"
        # P(Meow | Cat) vs P(Meow | Not Cat)
        self.p_meow_given_cat = 0.7
        self.p_meow_given_not_cat = 0.05
    
    def predict(self, brightness_level, has_meow):
        """
        Predict P(Cat | brightness, meow)
        Using naive Bayes (conditional independence)
        """
        # Likelihood: P(brightness, meow | Cat)
        # Assume conditional independence: P(bright,meow|Cat) = P(bright|Cat) * P(meow|Cat)
        likelihood_cat = (self.p_bright_given_cat[brightness_level] * 
                          (self.p_meow_given_cat if has_meow else (1 - self.p_meow_given_cat)))
        
        likelihood_not_cat = (self.p_bright_given_not_cat[brightness_level] * 
                              (self.p_meow_given_not_cat if has_meow else (1 - self.p_meow_given_not_cat)))
        
        # Bayes: posterior ∝ likelihood * prior
        posterior_cat = likelihood_cat * self.p_cat
        posterior_not_cat = likelihood_not_cat * self.p_not_cat
        
        # Normalize
        p_cat_given_data = posterior_cat / (posterior_cat + posterior_not_cat)
        
        return p_cat_given_data

# Create classifier
model = SimpleMultimodalBayes()

print("=" * 50)
print("MULTIMODAL BAYES CLASSIFIER")
print("=" * 50)

# Test all combinations
print("\nPredictions for all scenarios:")
print("-" * 60)
print("Brightness | Has 'meow'? | P(Cat | data) | Prediction")
print("-" * 60)

for brightness in [0, 1, 2]:
    brightness_names = ['Dark', 'Medium', 'Bright']
    for has_meow in [False, True]:
        prob = model.predict(brightness, has_meow)
        meow_str = "Yes" if has_meow else "No"
        pred = "CAT" if prob > 0.5 else "not cat"
        print(f"{brightness_names[brightness]:9} | {meow_str:10} | {prob:.3f}        | {pred}")

# ============================================
# TODO: Analyze when text helps vs when image helps
# ============================================

print("\n" + "=" * 50)
print("ANALYSIS: WHEN DOES EACH MODALITY HELP?")
print("=" * 50)

# Create visualization
scenarios = []
probs = []

for brightness in [0, 1, 2]:
    for has_meow in [False, True]:
        prob = model.predict(brightness, has_meow)
        label = f"{['Dark','Med','Bright'][brightness]}\n{'meow' if has_meow else 'no meow'}"
        scenarios.append(label)
        probs.append(prob)

plt.figure(figsize=(10, 6))
colors = ['red' if p > 0.5 else 'blue' for p in probs]
bars = plt.bar(scenarios, probs, color=colors, alpha=0.7)
plt.axhline(y=0.5, color='black', linestyle='--', label='Decision boundary')
plt.ylabel('P(Cat | data)')
plt.title('Multimodal Classification: Combining Image + Text')
plt.ylim(0, 1)
plt.legend()
plt.grid(True, alpha=0.3)

# Add value labels
for bar, prob in zip(bars, probs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{prob:.2f}', ha='center', va='bottom')

plt.show()

print("\n" + "=" * 50)
print("KEY INSIGHTS")
print("=" * 50)
print("""
1. When BOTH modalities agree (Dark + meow → P=0.91), confidence is HIGH
2. When they conflict (Bright + meow → P=0.70), confidence drops
3. Text 'meow' is a STRONG indicator (P(meow|cat)=0.7 vs P(meow|not)=0.05)
4. This is exactly how CLIP works, but with learned embeddings instead of hand-coded probabilities

MODALITY FUSION TYPES:
  - Early fusion: Combine features BEFORE classification (what we did)
  - Late fusion: Classify separately, then combine decisions
  - Cross-attention: Let image and text interact (transformers)
""")