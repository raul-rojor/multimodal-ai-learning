"""
COMPLETE IMAGE ENCODER FOR MULTIMODAL AI

This CNN outputs 512-dim embeddings that can be compared with text embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    """
    CNN-based image encoder.
    Input: (batch, 3, 224, 224) - RGB images
    Output: (batch, 512) - image embeddings
    """
    
    def __init__(self, embedding_dim=512):
        super(ImageEncoder, self).__init__()
        
        # CONV BLOCK 1: 3 → 64 channels
        # Input: (batch, 3, 224, 224)
        # Output: (batch, 64, 112, 112)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # CONV BLOCK 2: 64 → 128 channels
        # Input: (batch, 64, 112, 112)
        # Output: (batch, 128, 56, 56)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # CONV BLOCK 3: 128 → 256 channels
        # Input: (batch, 128, 56, 56)
        # Output: (batch, 256, 28, 28)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # CONV BLOCK 4: 256 → 512 channels
        # Input: (batch, 256, 28, 28)
        # Output: (batch, 512, 14, 14)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # After 4 pooling layers: 224 → 112 → 56 → 28 → 14
        # Feature map size: 512 channels × 14 × 14 = 100,352 numbers per image
        
        # Adaptive pooling: ensures same output size regardless of input
        # Instead of calculating sizes, this automatically pools to 7×7
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        # Now: (batch, 512, 7, 7) = 512*7*7 = 25,088 numbers
        
        # Final projection to embedding dimension
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 7 * 7, embedding_dim)
        
        # Normalize embeddings to unit length (important for similarity)
        self.normalize = nn.functional.normalize
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, 3, 224, 224) - normalized pixel values
        
        Returns:
            embeddings: (batch_size, embedding_dim) - L2 normalized
        """
        # Pass through conv blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)  # (batch, 512, 7, 7)
        
        # Flatten and project
        x = self.flatten(x)        # (batch, 512*7*7)
        x = self.fc(x)             # (batch, embedding_dim)
        
        # L2 normalize (makes dot product = cosine similarity)
        x = F.normalize(x, p=2, dim=1)
        
        return x

if __name__ == "__main__":
    # Create model
    model = ImageEncoder(embedding_dim=512)
    print("=" * 50)
    print("IMAGE ENCODER CREATED")
    print("=" * 50)
    print(f"Model: {model.__class__.__name__}")
    print(f"Embedding dimension: 512")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # TODO 1: Test with dummy input
    print("\n" + "=" * 50)
    print("TESTING FORWARD PASS")
    print("=" * 50)

    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    print(f"Input shape: {dummy_images.shape}")

    with torch.no_grad():
        embeddings = model(dummy_images)

    print(f"Output shape: {embeddings.shape}")
    print(f"Expected: ({batch_size}, 512) → got {embeddings.shape}")

    # TODO 2: Check embedding properties
    print("\n" + "=" * 50)
    print("EMBEDDING PROPERTIES")
    print("=" * 50)

    print(f"Embedding values (first 10 of first image): {embeddings[0, :10].cpu().numpy().round(3)}")
    print(f"Embedding norm (should be 1.0): {embeddings[0].norm().item():.3f}")
    print("(L2 normalization ensures dot product = cosine similarity)")

    # TODO 3: Compare similarity between two images
    print("\n" + "=" * 50)
    print("SIMILARITY BETWEEN IMAGES")
    print("=" * 50)

    # Two random images (different)
    img1 = torch.randn(1, 3, 224, 224)
    img2 = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        emb1 = model(img1)
        emb2 = model(img2)

    similarity = (emb1 @ emb2.T).item()
    print(f"Similarity between two RANDOM images: {similarity:.3f}")
print("(Lower is expected for different images)")
print("Similar images should have similarity close to 1.0")

print("\n" + "=" * 50)
print("LAYER-BY-LAYER SHAPE TRACE")
print("=" * 50)

class TraceableImageEncoder(ImageEncoder):
    def forward(self, x):
        print(f"Input: {x.shape}")
        x = self.conv1(x); print(f"After conv1: {x.shape}")
        x = self.conv2(x); print(f"After conv2: {x.shape}")
        x = self.conv3(x); print(f"After conv3: {x.shape}")
        x = self.conv4(x); print(f"After conv4: {x.shape}")
        x = self.adaptive_pool(x); print(f"After adaptive pool: {x.shape}")
        x = self.flatten(x); print(f"After flatten: {x.shape}")
        x = self.fc(x); print(f"After fc: {x.shape}")
        x = F.normalize(x, p=2, dim=1); print(f"After normalize: {x.shape}")
        return x

trace_model = TraceableImageEncoder()
_ = trace_model(torch.randn(1, 3, 224, 224))