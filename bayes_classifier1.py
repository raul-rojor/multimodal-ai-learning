"""
BAYES CLASSIFIER FOR SPAM DETECTION
Teaches: Conditional probability, Bayes' Rule, joint distributions
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ============================================
# PART 1: DEFINE OUR PROBABILITY DISTRIBUTIONS
# ============================================

class SpamBayesClassifier:
    def __init__(self):
        """
        We'll learn these probabilities from data.
        For now, we hardcode them for teaching.
        """
        
        # P(Spam) - Prior probability
        # 30% of emails are spam, 70% are not spam
        self.p_spam = 0.30
        self.p_not_spam = 0.70
        
        # LIKELIHOODS: P(Word | Class)
        # These are conditional probabilities
        
        # Probability of seeing "free" given spam
        self.p_free_given_spam = 0.95  # 95% of spam contains "free"
        self.p_free_given_not_spam = 0.10  # 10% of ham contains "free"
        
        # Probability of seeing "urgent" given spam
        self.p_urgent_given_spam = 0.70
        self.p_urgent_given_not_spam = 0.05
        
        # Probability of seeing "meeting" given spam
        self.p_meeting_given_spam = 0.05
        self.p_meeting_given_not_spam = 0.60

        # Probability of seeing 'lottery' given spam
        self.p_lottery_given_spam = 0.90
        self.p_lottery_given_not_spam = 0.01
        
    def bayes_rule_single_word(self, word_probs_given_spam, word_probs_given_not_spam):
        """
        Apply Bayes' Rule:
        
        P(Spam | Word) = P(Word | Spam) * P(Spam) / P(Word)
        
        Where:
        P(Word) = P(Word | Spam)*P(Spam) + P(Word | Not Spam)*P(Not Spam)
        """
        
        # Step 1: Calculate numerator
        numerator = word_probs_given_spam * self.p_spam
        
        # Step 2: Calculate denominator (total probability)
        denominator = (word_probs_given_spam * self.p_spam + 
                      word_probs_given_not_spam * self.p_not_spam)
        
        # Step 3: Apply Bayes' Rule
        p_spam_given_word = numerator / denominator
        
        return p_spam_given_word
    
    def predict_with_one_word(self, word):
        """
        Predict if email is spam given a single word appears
        """
        if word == "free":
            p_given_spam = self.p_free_given_spam
            p_given_not = self.p_free_given_not_spam
        elif word == "urgent":
            p_given_spam = self.p_urgent_given_spam
            p_given_not = self.p_urgent_given_not_spam
        elif word == "meeting":
            p_given_spam = self.p_meeting_given_spam
            p_given_not = self.p_meeting_given_not_spam
        else:
            print(f"Word '{word}' not in vocabulary")
            return None
        
        result = self.bayes_rule_single_word(p_given_spam, p_given_not)
        return result

# ============================================
# PART 2: RUN THE CLASSIFIER
# ============================================

# Create classifier instance
classifier = SpamBayesClassifier()

# Test with single words
print("=" * 50)
print("BAYES CLASSIFIER - SINGLE WORD PREDICTIONS")
print("=" * 50)

for word in ["free", "urgent", "meeting"]:
    prob_spam = classifier.predict_with_one_word(word)
    if prob_spam:
        print(f"\nIf email contains '{word}':")
        print(f"  P(Spam | '{word}') = {prob_spam:.3f}")
        print(f"  P(Not Spam | '{word}') = {1-prob_spam:.3f}")
        
        if prob_spam > 0.5:
            print(f"  → Classify as SPAM")
        else:
            print(f"  → Classify as NOT SPAM")

# ============================================
# PART 3: VISUALIZE THE RESULTS
# ============================================

print("\n" + "=" * 50)
print("VISUALIZATION: Posterior Probabilities")
print("=" * 50)

words = ['free', 'urgent', 'meeting']
spam_probs = [classifier.predict_with_one_word(w) for w in words]

plt.figure(figsize=(8, 5))
bars = plt.bar(words, spam_probs, color=['red', 'orange', 'green'])
plt.axhline(y=0.5, color='black', linestyle='--', label='Decision boundary (0.5)')
plt.ylabel('P(Spam | Word)')
plt.title('Bayesian Posterior Probability by Word')
plt.ylim(0, 1)

# Add value labels on bars
for bar, prob in zip(bars, spam_probs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{prob:.3f}', ha='center', va='bottom')

plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ============================================
# PART 4: MULTI-WORD PREDICTION (Naive Bayes)
# ============================================

print("\n" + "=" * 50)
print("NAIVE BAYES: Multiple Words")
print("=" * 50)

def predict_multiple_words(classifier, words_list):
    """
    Naive Bayes assumes words are INDEPENDENT given the class.
    P(Spam | w1, w2) ∝ P(Spam) * P(w1|Spam) * P(w2|Spam)
    """
    
    # Start with log prior (to avoid underflow)
    log_prob_spam = np.log(classifier.p_spam)
    log_prob_not_spam = np.log(classifier.p_not_spam)
    
    # Multiply likelihoods (add in log space)
    for word in words_list:
        if word == "free":
            log_prob_spam += np.log(classifier.p_free_given_spam)
            log_prob_not_spam += np.log(classifier.p_free_given_not_spam)
        elif word == "urgent":
            log_prob_spam += np.log(classifier.p_urgent_given_spam)
            log_prob_not_spam += np.log(classifier.p_urgent_given_not_spam)
        elif word == "meeting":
            log_prob_spam += np.log(classifier.p_meeting_given_spam)
            log_prob_not_spam += np.log(classifier.p_meeting_given_not_spam)
        elif word == "lottery":
            log_prob_spam += np.log(classifier.p_lottery_given_spam)
            log_prob_not_spam += np.log(classifier.p_lottery_given_not_spam)
    
    # Convert back from log space
    prob_spam = np.exp(log_prob_spam)
    prob_not_spam = np.exp(log_prob_not_spam)
    
    # Normalize
    total = prob_spam + prob_not_spam
    prob_spam_normalized = prob_spam / total
    
    return prob_spam_normalized

# Test different combinations
test_cases = [
    ["free"],
    ["urgent"],
    ["free", "urgent"],
    ["free", "meeting"],
    ["free", "urgent", "meeting"],
    ["free", "lottery"],
]

print("\nMulti-word predictions:")
for words in test_cases:
    prob = predict_multiple_words(classifier, words)
    print(f"\nWords: {words}")
    print(f"  P(Spam | {words}) = {prob:.4f}")
    print(f"  Classification: {'SPAM' if prob > 0.5 else 'HAM'}")

# ============================================
# PART 5: WHAT YOU JUST LEARNED
# ============================================

print("\n" + "=" * 50)
print("KEY PROBABILITY CONCEPTS DEMONSTRATED")
print("=" * 50)

print("""
1. PRIOR PROBABILITY P(Spam) = 0.3
   → Our belief BEFORE seeing evidence

2. LIKELIHOOD P(Word | Spam)
   → How likely to see "free" if email IS spam

3. EVIDENCE / MARGINAL P(Word)  
   → Total probability across both classes

4. POSTERIOR P(Spam | Word)
   → Our updated belief AFTER seeing evidence

5. BAYES' RULE FORMULA:
   P(A|B) = P(B|A) * P(A) / P(B)

6. NAIVE BAYES ASSUMPTION:
   Words are conditionally independent given class
   P(w1,w2|Spam) = P(w1|Spam) * P(w2|Spam)
""")