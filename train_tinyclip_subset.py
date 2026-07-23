"""
TRAIN ON SMALL SUBSET - Mac Friendly

Trains on only 1000 images. Runs in ~30-60 minutes on CPU.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import os
from tqdm import tqdm
from multimodal_dataset10 import MultimodalDataset
from tinyclip9 import TinyCLIP
from transformers import AutoTokenizer

print("=" * 50)
print("TRAINING ON SUBSET (MAC FRIENDLY)")
print("=" * 50)

# ==========================================
# CONFIGURATION (TINY FOR MAC)
# ==========================================

config = {
    'image_dir': './data/coco/val2014',
    'captions_file': './data/coco/annotations/captions_val2014.json',
    'max_seq_len': 16,
    'batch_size': 8,
    'num_workers': 0,
    'embedding_dim': 128,
    'vocab_size': 49408,
    'num_heads': 4,
    'num_layers': 1,
    'temperature': 0.07,
    'epochs': 5,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'num_samples': 5000,
    'checkpoint_dir': './checkpoints/',
    'device': 'cpu'
}

device = torch.device('cpu')
print(f"Using device: {device}")

os.makedirs(config['checkpoint_dir'], exist_ok=True)

# ==========================================
# LOAD TOKENIZER
# ==========================================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# ==========================================
# LOAD DATA (SUBSET)
# ==========================================

print("Loading dataset...")
full_dataset = MultimodalDataset(
    image_dir=config['image_dir'],
    captions_file=config['captions_file'],
    max_seq_len=config['max_seq_len']
)

# Use subset
subset = Subset(full_dataset, range(min(config['num_samples'], len(full_dataset))))
dataloader = DataLoader(subset, batch_size=config['batch_size'], shuffle=True, drop_last=True)

print(f"Training on {len(subset)} images")
print(f"Batches per epoch: {len(dataloader)}")

# ==========================================
# MODEL
# ==========================================

print("Creating model...")
model = TinyCLIP(
    image_embedding_dim=config['embedding_dim'],
    text_embedding_dim=config['embedding_dim'],
    vocab_size=config['vocab_size'],
    max_seq_len=config['max_seq_len'],
    num_heads=config['num_heads'],
    num_layers=config['num_layers'],
    temperature=config['temperature']
)
model = model.to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ==========================================
# OPTIMIZER
# ==========================================

optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

# ==========================================
# TRAINING
# ==========================================

print("\nStarting training...")
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        progress.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")

# ==========================================
# SAVE
# ==========================================

torch.save(model.state_dict(), './checkpoints/custom_model.pt')
print("\nModel saved to ./checkpoints/custom_model.pt")

print("\n✅ Training complete on subset!")