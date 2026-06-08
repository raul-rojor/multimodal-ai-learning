print("\n" + "=" * 60)
print("MORNING PART 1: MATRIX MULTIPLICATION = SIMILARITY")
print("=" * 60)

"""
MATRIX MULTIPLICATION & SIMILARITY
File: matrix_similarity.py

KEY INSIGHT: 
  Matrix multiplication = computing similarities between vectors
  CLIP does this: image_embeddings @ text_embeddings.T
  That's it. The whole multimodal comparison is ONE matrix multiply.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("DOT PRODUCT = SIMILARITY")
print("=" * 50)

# Two vectors (imagine image features and text features)
image_vec = np.array([0.8, 0.6, 0.4])  # 3-dimensional image embedding
text_vec = np.array([0.7, 0.5, 0.3])   # 3-dimensional text embedding

# Dot product measures similarity
similarity = np.dot(image_vec, text_vec)
print(f"Image vector: {image_vec}")
print(f"Text vector:  {text_vec}")
print(f"Dot product (similarity): {similarity:.3f}")
print("Higher dot product = more similar")

print("\n" + "=" * 50)
print("MATRIX MULTIPLICATION = MANY SIMILARITIES AT ONCE")
print("=" * 50)

# Batch of 3 images, batch of 4 texts
images = np.array([
    [0.8, 0.6, 0.4],  # image 1 features
    [0.2, 0.3, 0.9],  # image 2 features
    [0.5, 0.5, 0.5]   # image 3 features
])

texts = np.array([
    [0.7, 0.5, 0.3],  # text 1: "cat"
    [0.1, 0.2, 0.8],  # text 2: "dog"
    [0.4, 0.6, 0.2],  # text 3: "bird"
    [0.3, 0.3, 0.3]   # text 4: "animal"
])

# One matrix multiply = ALL pairwise similarities
# images @ texts.T means "matrix multiply images by the transpose of texts." (texts.T turns rows into columns)
similarity_matrix = images @ texts.T  # @ is matrix multiply in numpy

print("Similarity matrix (3 images x 4 texts):")
print("Rows = images, Columns = texts")
print("-" * 40)
print("        | cat   dog   bird  animal")
print("--------|-------------------------")
for i in range(3):
    print(f"Image {i+1} | ", end="")
    for j in range(4):
        print(f"{similarity_matrix[i,j]:.2f}  ", end="")
    print()

print("\n" + "=" * 50)
print("THIS IS EXACTLY WHAT CLIP DOES")
print("=" * 50)
print("""
CLIP's core computation:
  1. Image encoder → image_embeddings (shape: batch × 512)
  2. Text encoder → text_embeddings (shape: batch × 512)
  3. similarity = image_embeddings @ text_embeddings.T
  4. Apply softmax with temperature
  5. Cross-entropy loss

That's it. Everything else is details.
""")

# TODO 1: Normalize vectors before dot product
print("\n" + "=" * 50)
print("TODO 1: NORMALIZE VECTORS (Important for CLIP)")
print("=" * 50)

def normalize(vec):
    """Divide by length so vector has magnitude 1"""
    length = np.sqrt(np.sum(vec ** 2))
    return vec / length

# Without normalization
raw_sim = np.dot(image_vec, text_vec)

# With normalization (what CLIP actually does)
norm_image = normalize(image_vec)
norm_text = normalize(text_vec)
norm_sim = np.dot(norm_image, norm_text)

# without normalizing vectors before calculating similarities with dot products,
# then two vectors with large values may get a higher similarity scores relative
# to an actually more similar (directionally aligned) pair of vectors
print(f"Raw similarity: {raw_sim:.3f}")
print(f"Normalized similarity: {norm_sim:.3f}")
print("Normalized similarity is bounded between -1 and 1 (easier to compare)")

# TODO 2: Visualize similarity matrix as heatmap
plt.figure(figsize=(8, 6))
plt.imshow(similarity_matrix, cmap='hot', aspect='auto')
plt.colorbar(label='Similarity')
plt.xlabel('Texts (cat, dog, bird, animal)')
plt.ylabel('Images (1, 2, 3)')
plt.title('Similarity Matrix: Images × Texts')
plt.xticks(range(4), ['cat', 'dog', 'bird', 'animal'])
plt.yticks(range(3), ['Image 1', 'Image 2', 'Image 3'])

# Add value labels
for i in range(3):
    for j in range(4):
        plt.text(j, i, f'{similarity_matrix[i,j]:.2f}', 
                 ha='center', va='center', color='white')

plt.show()

print("""
KEY TAKEAWAYS:
  1. Matrix multiplication = batch similarity computation
  2. CLIP's entire multimodal comparison is ONE line: sim = images @ texts.T
  3. Normalization makes similarities comparable across different vectors
""")