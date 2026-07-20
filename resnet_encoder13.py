"""
RESNET IMAGE ENCODER

Replace custom CNN with pre-trained ResNet18.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNetEncoder(nn.Module):
    """
    Pre-trained ResNet18 image encoder.
    Input: (batch, 3, 224, 224)
    Output: (batch, embedding_dim) - L2 normalized
    """
    
    def __init__(self, embedding_dim=128):
        super(ResNetEncoder, self).__init__()
        
        # Load pre-trained ResNet18
        # Trained on 1.2M ImageNet images
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        
        # Remove the final classification layer
        # ResNet18 outputs 512-dim features before classification
        self.resnet.fc = nn.Identity()  # Remove FC layer
        
        # Project to embedding dimension
        self.projection = nn.Linear(512, embedding_dim)
        
    def forward(self, x):
        # Pass through ResNet
        x = self.resnet(x)  # (batch, 512)
        
        # Project to embedding dimension
        x = self.projection(x)  # (batch, embedding_dim)
        
        # L2 normalize (for cosine similarity)
        x = F.normalize(x, p=2, dim=1)
        
        return x

# Test
if __name__ == "__main__":
    model = ResNetEncoder(embedding_dim=128)
    dummy = torch.randn(4, 3, 224, 224)
    output = model(dummy)
    print(f"ResNetEncoder output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")