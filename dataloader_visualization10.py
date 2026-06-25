"""
DATA LOADER AND VISUALIZATION
File: dataloader_visualization.py
"""

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from multimodal_dataset10 import MultimodalDataset

print("=" * 50)
print("DATA LOADER VISUALIZATION")
print("=" * 50)

# Create dataset 
dataset = MultimodalDataset(
    image_dir='./data/coco/val2014',
    captions_file='./data/coco/annotations/captions_val2014.json',
    max_seq_len=32
)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"Dataset size: {len(dataset)}")
print(f"Number of batches: {len(dataloader)}")

# Get one batch
batch = next(iter(dataloader))
images = batch['image']
token_ids = batch['token_ids']

print(f"Batch images shape: {images.shape}")
print(f"Batch token IDs shape: {token_ids.shape}")

# Denormalize and display images
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
images_display = images * std + mean
images_display = torch.clamp(images_display, 0, 1)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes = axes.flatten()

for i in range(min(4, len(images))):
    img = images_display[i].permute(1, 2, 0).numpy()
    axes[i].imshow(img)
    axes[i].set_title(f'Image {i}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()