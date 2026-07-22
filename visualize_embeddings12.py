"""
EMBEDDING VISUALIZATION WITH PCA/t-SNE

Visualize the embedding space to see clustering.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from multimodal_dataset10 import MultimodalDataset
from tinyclip9 import TinyCLIP
import os

print("=" * 50)
print("VISUALIZING EMBEDDINGS")
print("=" * 50)

# Configuration
config = {
    'image_dir': './data/coco/val2014',
    'captions_file': './data/coco/annotations/captions_val2014.json',
    'max_seq_len': 16,
    'batch_size': 32,
    'embedding_dim': 128,
    'num_heads': 4,
    'num_layers': 1,
    'checkpoint_dir': './checkpoints/',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

device = torch.device(config['device'])

# Load model
model = TinyCLIP(
    image_embedding_dim=config['embedding_dim'],
    text_embedding_dim=config['embedding_dim'],
    vocab_size=49408,
    max_seq_len=config['max_seq_len'],
    num_heads=config['num_heads'],
    num_layers=config['num_layers']
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
val_dataset = MultimodalDataset(
    image_dir=config['image_dir'],
    captions_file=config['captions_file'],
    max_seq_len=config['max_seq_len']
)

# Use a small subset for visualization
val_subset = Subset(val_dataset, range(200))
val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False)

# ==========================================
# GET EMBEDDINGS
# ==========================================

print("Computing embeddings...")
image_embs = []
text_embs = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Processing"):
        images = batch['image'].to(device)
        token_ids = batch['token_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Get embeddings
        img_emb = model.image_encoder(images)
        txt_emb = model.text_encoder(token_ids, attention_mask)
        
        image_embs.append(img_emb.cpu().numpy())
        text_embs.append(txt_emb.cpu().numpy())

image_embs = np.concatenate(image_embs, axis=0)
text_embs = np.concatenate(text_embs, axis=0)

print(f"Image embeddings shape: {image_embs.shape}")
print(f"Text embeddings shape: {text_embs.shape}")

# ==========================================
# PCA VISUALIZATION
# ==========================================

print("\nPerforming PCA...")
pca = PCA(n_components=2)
all_embs = np.vstack([image_embs, text_embs])
all_embs_2d = pca.fit_transform(all_embs)

# Split back into image and text
image_embs_2d = all_embs_2d[:len(image_embs)]
text_embs_2d = all_embs_2d[len(image_embs):]

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(image_embs_2d[:, 0], image_embs_2d[:, 1], 
           label='Images', alpha=0.5, s=10, c='blue')
plt.scatter(text_embs_2d[:, 0], text_embs_2d[:, 1], 
           label='Texts', alpha=0.5, s=10, c='red')

plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.title('Embedding Space: Images vs Texts')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('custom_embeddings.png')
plt.show()
print("Plot saved to custom_embeddings.png")

print("\n" + "=" * 50)
print("INTERPRETING EMBEDDING VISUALIZATION")
print("=" * 50)

print("""
What to look for:

  1. CLUSTERING: Do embeddings form clusters?
     - Similar concepts should be close together
     - Images and texts of the same concept should overlap

  2. ALIGNMENT: Do images and texts share the same space?
     - Red and blue points should mix (not separate)
     - If separated → model failed to align modalities

  3. SPREAD: Are embeddings well-distributed?
     - Too concentrated → model isn't discriminating
     - Too spread → model may be overfitting

  4. PATTERNS: Do you see semantic categories?
     - Animals, vehicles, people should form subclusters

If your plot shows good alignment, your model works!
""")
