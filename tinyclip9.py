# ============================================
# PART 1: MORNING - TINYCLIP MODEL (1 hour)
# ============================================

print("\n" + "=" * 60)
print("MORNING: TINYCLIP ARCHITECTURE")
print("=" * 60)

"""
TINYCLIP: COMPLETE MULTIMODAL MODEL
File: tinyclip.py

This is YOUR multimodal model. It combines:
  1. Image encoder (CNN) → image embeddings
  2. Text encoder (Transformer) → text embeddings
  3. Contrastive loss → aligns matching pairs

INPUT: (images, texts) pairs
OUTPUT: Loss value (for training) or similarity matrix (for inference)

HOW IT WORKS:
  1. Encode images → image_embeddings (batch, 512)
  2. Encode texts → text_embeddings (batch, 512)
  3. Compute similarity = image_embeddings @ text_embeddings.T
  4. Apply temperature scaling
  5. Compute contrastive loss (cross-entropy on similarity matrix)
  6. Backpropagate to update both encoders

After training: matching images and texts have HIGH similarity!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import our encoders from previous days
from image_encoder7 import ImageEncoder
from text_encoder8 import TextEncoder

class TinyCLIP(nn.Module):
    """
    Complete multimodal model with contrastive learning.
    
    Architecture:
        Images → ImageEncoder → image_embeddings (512)
        Texts  → TextEncoder  → text_embeddings  (512)
        similarity = image_embeddings @ text_embeddings.T
        loss = contrastive_loss(similarity, temperature)
    
    Input shapes:
        images: (batch, 3, 224, 224)
        token_ids: (batch, seq_len) - tokenized text
        attention_mask: (batch, seq_len) - 1 for real tokens
    
    Output:
        loss: scalar (during training)
        similarity: (batch, batch) (during inference)
    """
    
    def __init__(self, 
                 image_embedding_dim=512,
                 text_embedding_dim=512,
                 vocab_size=30000,
                 max_seq_len=64,
                 num_heads=8,
                 num_layers=4,
                 temperature=0.07):
        super(TinyCLIP, self).__init__()
        
        # ==========================================
        # ENCODERS
        # ==========================================
        # Image encoder: pixels → 512-dim embedding
        self.image_encoder = ImageEncoder(embedding_dim=image_embedding_dim)
        
        # Text encoder: tokens → 512-dim embedding
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=text_embedding_dim,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # ==========================================
        # TEMPERATURE PARAMETER
        # ==========================================
        # Learnable temperature scaling factor.
        # Lower = more confident (sharper softmax)
        # Higher = less confident (softer softmax)
        # CLIP uses 0.07 as starting point.
        self.temperature = nn.Parameter(torch.ones([]) * temperature)
        
        # ==========================================
        # PROJECTION (optional)
        # ==========================================
        # Project both embeddings to same dimension if different.
        # This version assumes both are already 512-dim.
        # (Keep for flexibility)
        self.image_proj = nn.Linear(image_embedding_dim, image_embedding_dim)
        self.text_proj = nn.Linear(text_embedding_dim, text_embedding_dim)
        
    def forward(self, images, token_ids, attention_mask=None):
        """
        Forward pass through TinyCLIP.
        
        Args:
            images: (batch, 3, 224, 224) - normalized RGB images
            token_ids: (batch, seq_len) - tokenized text
            attention_mask: (batch, seq_len) - 1 for real tokens
            
        Returns:
            loss: scalar (if training)
            similarity: (batch, batch) - similarity matrix
        """
        batch_size = images.shape[0]
        
        # ==========================================
        # STEP 1: ENCODE IMAGES
        # ==========================================
        # Pass through CNN to get embeddings
        image_emb = self.image_encoder(images)  # (batch, 512)
        
        # Optional: project to ensure same dimension
        image_emb = self.image_proj(image_emb)
        image_emb = F.normalize(image_emb, p=2, dim=1)  # L2 normalize
        
        # ==========================================
        # STEP 2: ENCODE TEXTS
        # ==========================================
        # Pass through Transformer to get embeddings
        text_emb = self.text_encoder(token_ids, attention_mask)  # (batch, 512)
        
        # Optional: project to ensure same dimension
        text_emb = self.text_proj(text_emb)
        text_emb = F.normalize(text_emb, p=2, dim=1)  # L2 normalize
        
        # ==========================================
        # STEP 3: COMPUTE SIMILARITY MATRIX
        # ==========================================
        # All pairwise similarities between images and texts
        # This is the CORE multimodal comparison
        similarity = image_emb @ text_emb.T  # (batch, batch)
        
        # Apply temperature scaling
        similarity = similarity / self.temperature
        
        # ==========================================
        # STEP 4: CONTRASTIVE LOSS
        # ==========================================
        # This is the cross-entropy loss we learned in Week 1!
        # 
        # For each image i, we want:
        #   - High probability for its matching text (position i)
        #   - Low probability for all other texts (positions j ≠ i)
        #
        # So we apply softmax over rows (image→text) and columns (text→image)
        
        # Image-to-text loss (rows)
        # For each image, which text matches?
        labels = torch.arange(batch_size, device=similarity.device)
        loss_i2t = F.cross_entropy(similarity, labels)
        
        # Text-to-image loss (columns)
        # For each text, which image matches?
        loss_t2i = F.cross_entropy(similarity.T, labels)
        
        # Combined loss (average of both directions)
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss, similarity
    
    def get_embeddings(self, images=None, token_ids=None, attention_mask=None):
        """
        Get embeddings for inference.
        Use this to encode images or texts separately.
        
        Returns:
            image_emb: (batch, 512) if images provided
            text_emb: (batch, 512) if texts provided
        """
        result = {}
        
        if images is not None:
            image_emb = self.image_encoder(images)
            image_emb = self.image_proj(image_emb)
            result['image_emb'] = F.normalize(image_emb, p=2, dim=1)
        
        if token_ids is not None:
            text_emb = self.text_encoder(token_ids, attention_mask)
            text_emb = self.text_proj(text_emb)
            result['text_emb'] = F.normalize(text_emb, p=2, dim=1)
        
        if len(result) == 1:
            return list(result.values())[0]
        else:
            return result
    
    def compute_similarity_from_embeddings(self, image_emb, text_emb):
        """Compute similarity between pre-computed embeddings."""
        image_emb = F.normalize(image_emb, p=2, dim=1)
        text_emb = F.normalize(text_emb, p=2, dim=1)
        return image_emb @ text_emb.T / self.temperature


# ==========================================
# CREATE AND TEST THE MODEL
# ==========================================

print("\n" + "=" * 50)
print("CREATING TINYCLIP")
print("=" * 50)

model = TinyCLIP(
    image_embedding_dim=512,
    text_embedding_dim=512,
    vocab_size=30000,
    max_seq_len=64,
    num_heads=8,
    num_layers=4,
    temperature=0.07
)

print(f"Model: {model.__class__.__name__}")
print(f"Image encoder parameters: {sum(p.numel() for p in model.image_encoder.parameters()):,}")
print(f"Text encoder parameters: {sum(p.numel() for p in model.text_encoder.parameters()):,}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Temperature: {model.temperature.item():.3f}")

print("\n" + "=" * 50)
print("TESTING WITH DUMMY DATA")
print("=" * 50)

# Create dummy batch
batch_size = 4
dummy_images = torch.randn(batch_size, 3, 224, 224)
dummy_tokens = torch.randint(0, 30000, (batch_size, 32))
dummy_mask = torch.ones(batch_size, 32)

print(f"Images shape: {dummy_images.shape}")
print(f"Token IDs shape: {dummy_tokens.shape}")
print(f"Mask shape: {dummy_mask.shape}")

# Forward pass
with torch.no_grad():
    loss, similarity = model(dummy_images, dummy_tokens, dummy_mask)

print(f"\nLoss: {loss.item():.4f}")
print(f"Similarity matrix shape: {similarity.shape}")
print("\nSimilarity matrix (should be random):")
print(similarity.cpu().numpy().round(3))
print("\n(After training, diagonal values will be high!)")