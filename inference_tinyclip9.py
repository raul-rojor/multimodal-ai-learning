"""
TINYCLIP INFERENCE

Using the trained model for:
  1. Image → Text retrieval (find caption for an image)
  2. Text → Image retrieval (find image for a caption)
  3. Zero-shot classification (classify images using text prompts)
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("INFERENCE PIPELINE")
print("=" * 50)

print("""
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  IMAGE → TEXT RETRIEVAL:                                         │
│    1. Encode query image → embedding (512)                      │
│    2. Encode all candidate texts → embeddings (N, 512)          │
│    3. Compute similarity = query_emb @ candidate_emb.T        │
│    4. Return top-k texts with highest similarity               │
│                                                                  │
│  TEXT → IMAGE RETRIEVAL:                                         │
│    1. Encode query text → embedding (512)                       │
│    2. Encode all candidate images → embeddings (N, 512)         │
│    3. Compute similarity = query_emb @ candidate_emb.T        │
│    4. Return top-k images with highest similarity               │
│                                                                  │
│  ZERO-SHOT CLASSIFICATION:                                       │
│    1. Create text prompts: "a photo of a cat", "a photo of a dog"
│    2. Encode all prompts → embeddings (N, 512)                  │
│    3. Encode query image → embedding (512)                      │
│    4. Compute similarity to all prompts                         │
│    5. Predict class with highest similarity                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
""")

# ==========================================
# ZERO-SHOT CLASSIFICATION EXAMPLE
# ==========================================

def zero_shot_classify(model, image, class_names):
    """
    Classify an image without training on those classes.
    
    Args:
        model: Trained TinyCLIP model
        image: (3, 224, 224) - query image
        class_names: list of strings (e.g., ["cat", "dog", "bird"])
    
    Returns:
        predictions: list of (class_name, similarity_score)
    """
    print("=" * 50)
    print("ZERO-SHOT CLASSIFICATION")
    print("=" * 50)
    
    # Step 1: Create text prompts
    prompts = [f"a photo of a {cls}" for cls in class_names]
    print(f"Prompts: {prompts}")
    
    # Step 2: Encode image
    # In practice: image = preprocess_image(raw_image)
    with torch.no_grad():
        image_emb = model.get_embeddings(images=image.unsqueeze(0))
    
    # Step 3: Encode all prompts
    # In practice: tokenize prompts first
    # For demo, we skip tokenization and use pre-computed embeddings
    print("\n(In real use, you would tokenize and encode each prompt)")
    
    # Step 4: Compute similarities
    # similarity = image_emb @ text_emb.T
    # predictions = softmax(similarity / temperature)
    
    print("\n" + "=" * 50)
    print("EXAMPLE: CLASSIFYING AN IMAGE")
    print("=" * 50)
    
    # Simulate classification results
    class_names = ["cat", "dog", "bird", "car", "tree"]
    similarities = [0.82, 0.15, 0.01, 0.01, 0.01]  # Simulated
    
    # Rank by similarity
    results = sorted(zip(class_names, similarities), key=lambda x: x[1], reverse=True)
    
    print("\nSimilarity scores:")
    for cls, sim in results:
        print(f"  {cls}: {sim:.3f}")
    
    print(f"\nPrediction: {results[0][0]} (score: {results[0][1]:.3f})")
    
    return results

def image_retrieval(model, query_text, image_embeddings):
    """
    Find images that match a text query.
    
    Args:
        model: Trained TinyCLIP model
        query_text: string (e.g., "a photo of a cat")
        image_embeddings: (N, 512) - embeddings of all candidate images
    
    Returns:
        indices: top-k image indices sorted by relevance
    """
    print("=" * 50)
    print("TEXT → IMAGE RETRIEVAL")
    print("=" * 50)
    
    print(f"Query: '{query_text}'")
    print(f"Searching over {len(image_embeddings)} images...")
    
    # Encode query text
    # In practice: tokenize and encode
    # query_emb = model.get_embeddings(tokens=query_tokens, mask=query_mask)
    
    # Simulated result
    print("\nTop 3 results:")
    print("  Image 42: similarity 0.87")
    print("  Image 17: similarity 0.72")
    print("  Image 93: similarity 0.65")
    
    # Simulated indices
    return [42, 17, 93]

print("\n" + "=" * 50)
print("INFERENCE IN PRACTICE")
print("=" * 50)

print("""
REAL-WORLD INFERENCE STEPS:

  1. Load trained model
  2. Preprocess input (image resize, tokenization)
  3. Forward pass through encoders
  4. Compute similarity
  5. Rank results

PREPROCESSING (critical step):
  - Images: resize to 224×224, normalize pixel values
  - Texts: tokenize, add special tokens, create attention mask

OUTPUT FORMATS:
  - Image→Text: return caption with highest similarity
  - Text→Image: return image file path with highest similarity
  - Classification: return class name with highest similarity

PERFORMANCE TIPS:
  - Pre-compute all candidate embeddings (cache them)
  - Use GPU for faster inference
  - Batch multiple queries for efficiency
""")