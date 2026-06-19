# ============================================
# PART 2: MORNING - TRAINING LOOP (45 min)
# ============================================

print("\n" + "=" * 60)
print("MORNING PART 2: TRAINING LOOP")
print("=" * 60)

"""
TRAINING TINYCLIP
File: train_tinyclip.py

Complete training loop for TinyCLIP.
Trains the model on (image, text) pairs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

# Import TinyCLIP
from tinyclip9 import TinyCLIP

print("=" * 50)
print("TRAINING LOOP STRUCTURE")
print("=" * 50)

print("""
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FOR EACH EPOCH:                                                 │
│    FOR EACH BATCH:                                               │
│      1. Get batch of (images, texts)                            │
│      2. Forward pass: loss = model(images, texts)              │
│      3. Backward pass: loss.backward()                         │
│      4. Update weights: optimizer.step()                       │
│      5. Log loss                                               │
│                                                                  │
│  AFTER TRAINING:                                                 │
│    Evaluate on validation set                                   │
│    Save model weights                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
""")

# ==========================================
# DUMMY DATASET (Replace with real data)
# ==========================================

class DummyMultimodalDataset(Dataset):
    """
    Dummy dataset for testing the training loop.
    
    In practice, replace this with:
        - COCO dataset (images + captions)
        - Conceptual Captions
        - Custom dataset
    
    REAL DATA SOURCES:
        - torchvision.datasets.CocoCaptions
        - HuggingFace datasets (e.g., "conceptual_captions")
        - Local folder with images + captions files
    """
    
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.vocab_size = 30000
        self.seq_len = 32
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Dummy image: random 224×224×3
        image = torch.randn(3, 224, 224)
        
        # Dummy text: random token IDs
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
        mask = torch.ones(self.seq_len)  # All real tokens
        
        return {
            'image': image,
            'tokens': tokens,
            'mask': mask
        }

def train_tinyclip(model, dataloader, num_epochs=10, lr=1e-4):
    """
    Training loop for TinyCLIP.
    
    Args:
        model: TinyCLIP instance
        dataloader: DataLoader returning (image, tokens, mask)
        num_epochs: Number of training epochs
        lr: Learning rate
    """
    print("=" * 50)
    print("TRAINING TINYCLIP")
    print("=" * 50)
    
    # Setup optimizer
    # AdamW is standard for transformer-based models
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Track loss for plotting
    loss_history = []
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Get batch data
            images = batch['image']
            tokens = batch['tokens']
            mask = batch['mask']
            
            # ==========================================
            # FORWARD PASS
            # ==========================================
            # Compute loss
            loss, similarity = model(images, tokens, mask)
            
            # ==========================================
            # BACKWARD PASS
            # ==========================================
            # Zero gradients from previous step
            optimizer.zero_grad()
            
            # Computes gradients of the scalar loss L w.r.t. all model parameters θ, 
            # i.e. the partial derivative ∂Loss/∂Parameter.
            # Every operation remembers how it was computed (this builds the autograd graph).
            # loss.backward() runs reverse-mode automatic differentiation over that graph.
            # This produces a gradient (direction + magnitude) for every tensor that required gradients.
            # The resulting gradient tensors are written into 
            # each parameter’s (in this case loss') *.grad field and do not change the parameters (in this case loss) 
            # themselves. After this call the optimizer reads those *.grad values and updates parameters when
            # optimizer.step() is called. Remember to call optimizer.zero_grad() before loss.backward() since
            # gradients accumulate, and clip gradients (e.g. clip_grad_norm_) if needed for stability.
            loss.backward()
            
            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            num_batches += 1
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Step scheduler
        scheduler.step()
        
        # Epoch summary
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")
    
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return loss_history

# ==========================================
# RUN TRAINING (with dummy data)
# ==========================================

print("\n" + "=" * 50)
print("RUNNING TRAINING WITH DUMMY DATA")
print("=" * 50)

print("Creating dummy dataset and dataloader...")

# Create dataset
dummy_dataset = DummyMultimodalDataset(num_samples=100)

# Create dataloader
dataloader = DataLoader(
    dummy_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0  # Set to 0 for dummy data
)

print(f"Dataset size: {len(dummy_dataset)}")
print(f"Batch size: 4")
print(f"Number of batches: {len(dataloader)}")

# Create model
model = TinyCLIP(
    vocab_size=30000,
    temperature=0.07
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Train
print("\nStarting training... (this will run on CPU)")
print("(In practice, move to GPU with .cuda())")

# Uncomment to actually train:
# loss_history = train_tinyclip(model, dataloader, num_epochs=5, lr=1e-4)

print("\n" + "=" * 50)
print("WHAT TRAINING DOES")
print("=" * 50)

print("""
During training:

  BEFORE TRAINING:
    - Image and text embeddings are random relative to each other
        (encoders are initialized independently, so the two embedding spaces are not aligned until training)
    - Similarity matrix has random values
    - Matching pairs have random similarity

  AFTER TRAINING:
    - Image embeddings for "cat" are close to text embeddings for "cat"
    - Image embeddings for "cat" are far from text embeddings for "dog"
    - Similarity matrix diagonal is HIGH (matching pairs)
    - Similarity matrix off-diagonal is LOW (non-matching pairs)

  RESULT: The model can match images to texts it has never seen before!
  This is ZERO-SHOT classification.
""")

print("\n" + "=" * 50)
print("REAL DATA SOURCES")
print("=" * 50)

print("""
Replace dummy data with real datasets:

  1. COCO Captions (most common)
     - 118k images, 5 captions per image
     - pip install torchvision
     - torchvision.datasets.CocoCaptions

  2. Conceptual Captions
     - 3.3M image-text pairs from web
     - HuggingFace: datasets.load_dataset("conceptual_captions")

  3. Flickr30k
     - 31k images, 5 captions per image
     - Smaller, good for prototyping

  4. Your own data
     - Folder of images
     - CSV/JSON with captions
     - Custom Dataset class
""")