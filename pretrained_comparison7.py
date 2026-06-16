# ============================================
# PART 3: AFTERNOON - COMPARE WITH PRE-TRAINED (30 min)
# ============================================

print("\n" + "=" * 60)
print("AFTERNOON: COMPARE WITH PRE-TRAINED RESNET")
print("=" * 60)
 
"""
COMPARING WITH PRE-TRAINED RESNET
File: pretrained_comparison.py

Why use pre-trained? They've already learned useful features from millions of images.
"""

import torch
import torch.nn as nn
import torchvision.models as models

print("=" * 50)
print("LOADING PRE-TRAINED RESNET")
print("=" * 50)

# Load ResNet18 pre-trained on ImageNet (1.2 million images, 1000 classes)
resnet = models.resnet18(weights='IMAGENET1K_V1')

print(f"ResNet18 architecture:")
print(f"  - Millions of parameters, 18 layers deep")
print(f"  - Trained on 1.2M images, 1000 classes")
print(f"  - Removed final classification layer for embeddings")

# Remove the final classification layer to get embeddings
resnet_encoder = nn.Sequential(*list(resnet.children())[:-1])
resnet_encoder.eval()

print("\n" + "=" * 50)
print("COMPARING OUR CNN vs RESNET")
print("=" * 50)

print("""
┌──────────────────────┬─────────────────────┬─────────────────────┐
│                      │ OUR Simple CNN      │ ResNet18 (pre-trained)
├──────────────────────┼─────────────────────┼─────────────────────┤
│ Parameters:          │ ~2M (from scratch)  │ ~11M (pre-trained)  │
│ Training data:       │ We will train it    │ ImageNet (1.2M)     │
│ Feature quality:     │ Starts random       │ Already good        │
│ Best for:            │ Learning from zero  │ Transfer learning   │
└──────────────────────┴─────────────────────┴─────────────────────┘

FOR TINYCLIP (this week):
  We'll use our custom CNN to understand how it works.

FOR REAL PROJECTS:
  Use pre-trained ResNet or ViT (faster training, better performance)
""")

# Demo: Extract embedding from ResNet
from image_encoder7 import ImageEncoder

our_model = ImageEncoder()
resnet_model = resnet_encoder

dummy_input = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    our_emb = our_model(dummy_input)
    # ResNet: need to add dimension for pooling
    resnet_emb = resnet_model(dummy_input).flatten().unsqueeze(0)
    resnet_emb = nn.functional.normalize(resnet_emb, p=2, dim=1)

print(f"\nOur CNN embedding dimension: {our_emb.shape[1]}")
print(f"ResNet embedding dimension: {resnet_emb.shape[1]}")
print("(ResNet default is 512 after removing classification layer)")