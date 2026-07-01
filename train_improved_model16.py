# ============================================
# PART 4: THURSDAY - TRAIN WITH IMPROVED MODEL
# ============================================

print("\n" + "=" * 60)
print("TRAIN IMPROVED MODEL")
print("=" * 60)

"""
TRAIN IMPROVED TINYCLIP

Train with ResNet + DistilBERT on larger subset.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from multimodal_dataset10 import MultimodalDataset
from tinyclip_improved15 import TinyCLIPImproved
from transformers import AutoTokenizer

print("=" * 50)
print("TRAINING IMPROVED TINYCLIP")
print("=" * 50)

# Configuration
config = {
    'image_dir': './data/coco/train2014',
    'captions_file': './data/coco/annotations/captions_train2014.json',
    'max_seq_len': 16,
    'batch_size': 8,
    'embedding_dim': 128,
    'num_samples': 5000,
    'epochs': 5,
    'learning_rate': 1e-4,
    'checkpoint_dir': './checkpoints/',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

device = torch.device(config['device'])
print(f"Using device: {device}")

# Load data
dataset = MultimodalDataset(
    image_dir=config['image_dir'],
    captions_file=config['captions_file'],
    max_seq_len=config['max_seq_len']
)

subset = Subset(dataset, range(min(config['num_samples'], len(dataset))))
dataloader = DataLoader(subset, batch_size=config['batch_size'], shuffle=True, drop_last=True)

print(f"Training on {len(subset)} images")
print(f"Batches: {len(dataloader)}")

# Model
model = TinyCLIPImproved(
    embedding_dim=config['embedding_dim'],
    max_seq_len=config['max_seq_len']
)
model = model.to(device)

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])

# Train
model.train()
losses = []
for epoch in range(config['epochs']):
    epoch_loss = 0
    progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
    
    for batch in progress:
        images = batch['image'].to(device)
        token_ids = batch['token_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        loss, _ = model(images, token_ids, attention_mask)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

# Save
torch.save(model.state_dict(), './checkpoints/improved_model.pt')
print("✅ Improved model saved to ./checkpoints/improved_model.pt")