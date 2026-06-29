# ============================================
# PART 1: MORNING - RETRIEVAL METRICS (1 hour)
# ============================================

print("\n" + "=" * 60)
print("MORNING: RETRIEVAL METRICS")
print("=" * 60)

"""
EVALUATION METRICS FOR TINYCLIP
File: evaluate_metrics.py

Compute retrieval metrics:
  - Recall@K: Fraction of times correct item is in top-K predictions
  - Mean Reciprocal Rank (MRR): Average of 1/rank of correct item
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from multimodal_dataset10 import MultimodalDataset
from tinyclip9 import TinyCLIP
import os

print("=" * 50)
print("EVALUATING RETRIEVAL PERFORMANCE")
print("=" * 50)

# Configuration
config = {
    'image_dir': './data/coco/val2014',
    'captions_file': './data/coco/annotations/captions_val2014.json',
    'max_seq_len': 16,
    'batch_size': 32,
    'embedding_dim': 128,
    'vocab_size': 49408,
    'num_heads': 4,
    'num_layers': 1,
    'temperature': 0.07,
    'checkpoint_dir': './checkpoints/',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

device = torch.device(config['device'])
print(f"Using device: {device}")

# Load model
print("\nLoading model...")
model = TinyCLIP(
    image_embedding_dim=config['embedding_dim'],
    text_embedding_dim=config['embedding_dim'],
    vocab_size=config['vocab_size'],
    max_seq_len=config['max_seq_len'],
    num_heads=config['num_heads'],
    num_layers=config['num_layers'],
    temperature=config['temperature']
)

checkpoint_path = os.path.join(config['checkpoint_dir'], 'subset_model.pt')
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Loaded subset_model.pt")
else:
    print("No checkpoint found!")
    exit()

model = model.to(device)
model.eval()

# Load dataset
print("\nLoading validation dataset...")
val_dataset = MultimodalDataset(
    image_dir=config['image_dir'],
    captions_file=config['captions_file'],
    max_seq_len=config['max_seq_len']
)

# Use subset for faster evaluation
val_subset = Subset(val_dataset, range(500))
val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False)

print(f"Evaluation dataset size: {len(val_subset)}")
print(f"Batches: {len(val_loader)}")

# ==========================================
# COMPUTE EMBEDDINGS
# ==========================================

def compute_embeddings(dataloader, model, device):
    """Compute image and text embeddings for all items in dataloader"""
    all_image_embs = []
    all_text_embs = []
    all_labels = []  # Which pairs are correct
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing embeddings")):
            images = batch['image'].to(device)
            token_ids = batch['token_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get embeddings
            image_embs = model.image_encoder(images)
            text_embs = model.text_encoder(token_ids, attention_mask)
            
            all_image_embs.append(image_embs.cpu())
            all_text_embs.append(text_embs.cpu())
            
            # Labels: correct pairs are on diagonal
            batch_size = images.shape[0]
            labels = torch.arange(batch_size) + batch_idx * batch_size
            all_labels.append(labels)
    
    image_embs = torch.cat(all_image_embs, dim=0)
    text_embs = torch.cat(all_text_embs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return image_embs, text_embs, labels

# Compute embeddings
print("\nComputing embeddings...")
image_embs, text_embs, labels = compute_embeddings(val_loader, model, device)

print(f"Image embeddings shape: {image_embs.shape}")
print(f"Text embeddings shape: {text_embs.shape}")

# ==========================================
# COMPUTE SIMILARITY MATRIX
# ==========================================

print("\nComputing similarity matrix...")
similarity = image_embs @ text_embs.T  # (N, N)
print(f"Similarity matrix shape: {similarity.shape}")

# ==========================================
# RECALL@K METRICS
# ==========================================

def recall_at_k(similarity, labels, k_values=[1, 5, 10]):
    """
    Compute Recall@K for image-to-text and text-to-image retrieval.
    
    Args:
        similarity: (N, N) similarity matrix
        labels: (N,) correct pair indices (should be diagonal)
        k_values: list of K values to compute
    
    Returns:
        dict: recall scores for each K
    """
    results = {}
    N = similarity.shape[0]
    
    # Image-to-text: For each image, find top-K texts
    for k in k_values:
        # Get top-K indices for each row
        top_k_indices = similarity.argsort(dim=1, descending=True)[:, :k]
        
        # Check if correct label is in top-K
        correct_in_top_k = (top_k_indices == labels.unsqueeze(1)).any(dim=1)
        recall = correct_in_top_k.float().mean().item()
        
        results[f'Recall@{k} (i2t)'] = recall
    
    # Text-to-image: For each text, find top-K images
    for k in k_values:
        top_k_indices = similarity.T.argsort(dim=1, descending=True)[:, :k]
        correct_in_top_k = (top_k_indices == labels.unsqueeze(1)).any(dim=1)
        recall = correct_in_top_k.float().mean().item()
        
        results[f'Recall@{k} (t2i)'] = recall
    
    return results

# ==========================================
# MEAN RECIPROCAL RANK (MRR)
# ==========================================

def mean_reciprocal_rank(similarity, labels):
    """
    Compute Mean Reciprocal Rank (MRR).
    MRR = average of 1/rank where rank is position of correct item.
    """
    # Get ranks of correct items (1-indexed)
    ranks = []
    for i in range(similarity.shape[0]):
        # Sort by similarity descending
        sorted_indices = similarity[i].argsort(descending=True)
        # Find rank of correct item
        rank = (sorted_indices == labels[i]).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    
    ranks = torch.tensor(ranks)
    mrr = (1.0 / ranks).mean().item()
    return mrr

# ==========================================
# RUN EVALUATION
# ==========================================

print("\n" + "=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)

# Recall@K
recall_results = recall_at_k(similarity, labels, k_values=[1, 5, 10])
for metric, value in recall_results.items():
    print(f"{metric}: {value:.4f}")

# MRR
mrr = mean_reciprocal_rank(similarity, labels)
print(f"MRR: {mrr:.4f}")

# ==========================================
# VISUALIZE RESULTS
# ==========================================

# Plot similarity matrix heatmap (first 50 items)
plt.figure(figsize=(8, 6))
plt.imshow(similarity[:50, :50].numpy(), cmap='hot', aspect='auto')
plt.colorbar(label='Similarity')
plt.xlabel('Texts')
plt.ylabel('Images')
plt.title('Similarity Matrix (First 50 Samples)\nDiagonal should be bright')
plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("INTERPRETING RESULTS")
print("=" * 50)

print("""
Recall@K Interpretation:
  - Random baseline (for K=1): 1/N ≈ 0.2% for 500 samples
  - Good model: Recall@1 > 10%
  - Great model: Recall@1 > 30%
  - Excellent model: Recall@1 > 50%

MRR Interpretation:
  - Random: ~1/N ≈ 0.2%
  - Good: > 0.2
  - Great: > 0.4

Your results show how well the model has learned.
If results are low, the model may need more training or data.
""")