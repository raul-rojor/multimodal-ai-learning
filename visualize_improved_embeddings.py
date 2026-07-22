"""
VISUALIZE IMPROVED MODEL EMBEDDINGS
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from multimodal_dataset10 import MultimodalDataset
from tinyclip_improved15 import TinyCLIPImproved

print("=" * 50)
print("VISUALIZING IMPROVED MODEL EMBEDDINGS")
print("=" * 50)

config = {
    'image_dir': './data/coco/val2014',
    'captions_file': './data/coco/annotations/captions_val2014.json',
    'max_seq_len': 16,
    'batch_size': 32,
    'embedding_dim': 128,
    'num_samples': 200,
    'checkpoint_path': './checkpoints/improved_model.pt',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

device = torch.device(config['device'])
print(f"Using device: {device}")

# Load model
print("Loading model...")
model = TinyCLIPImproved(
    embedding_dim=config['embedding_dim'],
    max_seq_len=config['max_seq_len']
)
model.load_state_dict(torch.load(config['checkpoint_path'], map_location=device))
model = model.to(device)
model.eval()
print("✅ Model loaded")

# Load data
print("Loading data...")
dataset = MultimodalDataset(
    image_dir=config['image_dir'],
    captions_file=config['captions_file'],
    max_seq_len=config['max_seq_len']
)

subset = Subset(dataset, range(min(config['num_samples'], len(dataset))))
dataloader = DataLoader(subset, batch_size=config['batch_size'], shuffle=False)

print(f"Visualizing {len(subset)} samples")

# Get embeddings
image_embs = []
text_embs = []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Computing embeddings"):
        images = batch['image'].to(device)
        token_ids = batch['token_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        image_embs.append(model.image_encoder(images).cpu().numpy())
        text_embs.append(model.text_encoder(token_ids, attention_mask).cpu().numpy())

image_embs = np.concatenate(image_embs)
text_embs = np.concatenate(text_embs)

print(f"Image embeddings shape: {image_embs.shape}")
print(f"Text embeddings shape: {text_embs.shape}")

# PCA
all_embs = np.vstack([image_embs, text_embs])
pca = PCA(n_components=2)
all_embs_2d = pca.fit_transform(all_embs)

image_embs_2d = all_embs_2d[:len(image_embs)]
text_embs_2d = all_embs_2d[len(image_embs):]

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(image_embs_2d[:, 0], image_embs_2d[:, 1], label='Images', alpha=0.5, s=10, c='blue')
plt.scatter(text_embs_2d[:, 0], text_embs_2d[:, 1], label='Texts', alpha=0.5, s=10, c='red')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.title('Improved Model: Images vs Texts')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('improved_embeddings.png')
plt.show()

print("Plot saved to improved_embeddings.png")