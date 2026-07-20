"""
SIMPLE CNN FROM SCRATCH IN PYTORCH

This is your first neural network that actually processes images!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 50)
print("BUILDING A CNN CLASS")
print("=" * 50)

class SimpleCNN(nn.Module):
    """
    A simple CNN that outputs image embeddings.
    Input: (batch, 3, 32, 32) - 32×32 color images
    Output: (batch, 512) - embedding vector per image
    """
    
    def __init__(self, embedding_dim=512):
        super(SimpleCNN, self).__init__()
        
        # CONV LAYER 1: 3 channels → 32 channels
        # Input: (batch, 3, 32, 32)
        # Output: (batch, 32, 32, 32) with padding=1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, 
                                kernel_size=3, padding=1)
        
        # CONV LAYER 2: 32 channels → 64 channels
        # Input: (batch, 32, 16, 16) after pooling
        # Output: (batch, 64, 16, 16) with padding=1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, 
                                kernel_size=3, padding=1)
        
        # CONV LAYER 3: 64 channels → 128 channels
        # Input: (batch, 64, 8, 8) after pooling
        # Output: (batch, 128, 8, 8) with padding=1
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, 
                                kernel_size=3, padding=1)
        
        # After 3 pooling layers, 32×32 becomes 4×4
        # 128 channels × 4 × 4 = 2048 features
        self.fc = nn.Linear(128 * 4 * 4, embedding_dim)
    
    def forward(self, x):
        """
        x shape: (batch_size, 3, 32, 32)
        """
        # Conv1 + ReLU + Pool
        x = self.conv1(x)           # (batch, 32, 32, 32)
        x = F.relu(x)               # Apply activation
        x = F.max_pool2d(x, 2)      # (batch, 32, 16, 16)
        
        # Conv2 + ReLU + Pool
        x = self.conv2(x)           # (batch, 64, 16, 16)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      # (batch, 64, 8, 8)
        
        # Conv3 + ReLU + Pool
        x = self.conv3(x)           # (batch, 128, 8, 8)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      # (batch, 128, 4, 4)
        
        # Flatten: (batch, 128, 4, 4) → (batch, 128*4*4)
        x = x.view(x.size(0), -1)   # (batch, 2048)
        
        # Final fully connected layer
        x = self.fc(x)              # (batch, embedding_dim)
        
        return x

# Create the model
model = SimpleCNN(embedding_dim=512)
print(f"Model created: {model.__class__.__name__}")
print(f"Embedding dimension: 512")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# TODO 1: Test the model with random input
print("\n" + "=" * 50)
print("TESTING THE CNN")
print("=" * 50)

# Create dummy batch of images
# Shape: (batch_size=4, channels=3, height=32, width=32)
dummy_images = torch.randn(4, 3, 32, 32)
print(f"Input shape: {dummy_images.shape}")

# Forward pass
with torch.no_grad():  # Don't track gradients for testing
    embeddings = model(dummy_images)

print(f"Output shape: {embeddings.shape}")
print(f"Expected: (4, 512) → {embeddings.shape[0]} images, each {embeddings.shape[1]}-dim embedding")

# TODO 2: Inspect the embeddings
print("\n" + "=" * 50)
print("EMBEDDING PROPERTIES")
print("=" * 50)

print(f"First embedding (first 10 values): {embeddings[0, :10].detach().numpy().round(3)}")
print(f"Embedding mean: {embeddings.mean().item():.3f}")
print(f"Embedding std: {embeddings.std().item():.3f}")

print("\n" + "=" * 50)
print("LAYER BY LAYER BREAKDOWN")
print("=" * 50)

# Create a smaller version for debugging
class DebugCNN(SimpleCNN):
    def forward(self, x):
        print(f"Input: {x.shape}")
        x = self.conv1(x)
        print(f"After conv1: {x.shape}")
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        print(f"After pool1: {x.shape}")
        x = self.conv2(x)
        print(f"After conv2: {x.shape}")
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        print(f"After pool2: {x.shape}")
        x = self.conv3(x)
        print(f"After conv3: {x.shape}")
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        print(f"After pool3: {x.shape}")
        x = x.view(x.size(0), -1)
        print(f"After flatten: {x.shape}")
        x = self.fc(x)
        print(f"After fc: {x.shape}")
        return x

debug_model = DebugCNN()
_ = debug_model(dummy_images)

print("\n" + "=" * 50)
print("WHAT YOU JUST BUILT")
print("=" * 50)

print("""
Your CNN does:

  1. Takes a 32×32 color image (3 channels)
  2. Passes through 3 convolution layers:
     - Layer 1: detects simple patterns (edges, corners)
     - Layer 2: detects textures and simple shapes
     - Layer 3: detects complex patterns (eyes, wheels)
  3. Pooling reduces size after each conv layer
  4. Flatten converts 2D to 1D
  5. Fully connected layer outputs 512 embedding numbers

This embedding vector represents the IMAGE CONTENT.
Similar images → Similar embedding vectors.
""")