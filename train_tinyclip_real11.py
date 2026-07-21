"""
TRAIN TINYCLIP ON REAL DATA

Complete training pipeline for TinyCLIP using COCO dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Import our modules
from multimodal_dataset10 import MultimodalDataset
from tinyclip9 import TinyCLIP
from transformers import AutoTokenizer

print("=" * 50)
print("TRAINING TINYCLIP ON COCO")
print("=" * 50)

# ==========================================
# CONFIGURATION
# ==========================================

config = {
    # Data
    'image_dir': './data/coco/train2014',  # Path to COCO images
    'val_image_dir': './data/coco/val2014',
    'captions_file': './data/coco/annotations/captions_train2014.json',
    'val_captions_file': './data/coco/annotations/captions_val2014.json',
    'max_seq_len': 16,
    'batch_size': 4,  # Reduce if out of memory
    'num_workers': 0,  # Set to 0 for debugging, >0 for speed
    
    # Model
    'embedding_dim': 512,
    'vocab_size': 49408,  # CLIP tokenizer vocab size
    'num_heads': 4,
    'num_layers': 1,
    'temperature': 0.07,
    
    # Training
    'epochs': 1,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'save_every': 1,  # Save checkpoint every N epochs
    'checkpoint_dir': './checkpoints/',
    'log_dir': './logs/',
    
    # Hardware
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print("Configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")

# ==========================================
# SETUP
# ==========================================

# Create directories
os.makedirs(config['checkpoint_dir'], exist_ok=True)
os.makedirs(config['log_dir'], exist_ok=True)

# Device
device = torch.device(config['device'])
print(f"\nUsing device: {device}")

# Tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# ==========================================
# DATA LOADING
# ==========================================

print("\nLoading datasets...")

# Create training dataset
train_dataset = MultimodalDataset(
    image_dir=config['image_dir'],
    captions_file=config['captions_file'],
    max_seq_len=config['max_seq_len']
)

# Create validation dataset (use subset if full val is too large)
val_dataset = MultimodalDataset(
    image_dir=config['val_image_dir'],
    captions_file=config['val_captions_file'],
    max_seq_len=config['max_seq_len']
)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Val dataset size: {len(val_dataset)}")

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=config['num_workers'],
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=config['num_workers'],
    drop_last=True
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# ==========================================
# MODEL
# ==========================================

print("\nCreating model...")

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

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ==========================================
# OPTIMIZER AND SCHEDULER
# ==========================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=config['learning_rate'],
    weight_decay=config['weight_decay']
)

# Cosine annealing scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config['epochs'] * len(train_loader)
)

# ==========================================
# TRAINING LOOP
# ==========================================

def train_epoch(model, dataloader, optimizer, scheduler, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for batch in progress_bar:
        # Move to device
        images = batch['image'].to(device)
        token_ids = batch['token_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass
        loss, similarity = model(images, token_ids, attention_mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Track loss
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/num_batches:.4f}'
        })
    
    return total_loss / num_batches

def validate(model, dataloader):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch['image'].to(device)
            token_ids = batch['token_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            loss, similarity = model(images, token_ids, attention_mask)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def save_checkpoint(model, optimizer, scheduler, epoch, loss, filename):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': config
    }
    torch.save(checkpoint, os.path.join(config['checkpoint_dir'], filename))
    print(f"Checkpoint saved: {filename}")

# ==========================================
# MAIN TRAINING
# ==========================================

print("\n" + "=" * 50)
print("STARTING TRAINING")
print("=" * 50)

train_losses = []
val_losses = []

for epoch in range(config['epochs']):
    print(f"\nEpoch {epoch+1}/{config['epochs']}")
    print("-" * 30)
    
    # Train
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, epoch)
    train_losses.append(train_loss)
    
    # Validate
    val_loss = validate(model, val_loader)
    val_losses.append(val_loss)
    
    print(f"Train loss: {train_loss:.4f}")
    print(f"Val loss: {val_loss:.4f}")
    
    # Save checkpoint
    if (epoch + 1) % config['save_every'] == 0:
        save_checkpoint(
            model, optimizer, scheduler, epoch, train_loss,
            f'checkpoint_epoch_{epoch+1}.pt'
        )

# ==========================================
# SAVE FINAL MODEL
# ==========================================

save_checkpoint(
    model, optimizer, scheduler, config['epochs']-1, train_loss,
    'final_model.pt'
)

print("\n" + "=" * 50)
print("TRAINING COMPLETE")
print("=" * 50)

# ==========================================
# PLOT LOSS
# ==========================================

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(config['log_dir'], 'loss_plot.png'))
plt.show()

print(f"\nLoss plot saved to: {config['log_dir']}/loss_plot.png")
print(f"Checkpoints saved to: {config['checkpoint_dir']}")