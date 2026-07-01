# ============================================
# PART 3: WEDNESDAY - TINYCLIP WITH PRE-TRAINED ENCODERS
# ============================================

print("\n" + "=" * 60)
print("WEDNESDAY: TINYCLIP WITH PRE-TRAINED ENCODERS")
print("=" * 60)

"""
TINYCLIP WITH PRE-TRAINED ENCODERS

Same contrastive loss, but with ResNet + DistilBERT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet_encoder13 import ResNetEncoder
from distilbert_encoder14 import DistilBERTEncoder

class TinyCLIPImproved(nn.Module):
    """
    TinyCLIP with pre-trained encoders.
    """
    
    def __init__(self, 
                 embedding_dim=128,
                 vocab_size=30522,  # DistilBERT vocab size
                 max_seq_len=16,
                 temperature=0.07):
        super(TinyCLIPImproved, self).__init__()
        
        # Pre-trained encoders
        self.image_encoder = ResNetEncoder(embedding_dim=embedding_dim)
        self.text_encoder = DistilBERTEncoder(
            embedding_dim=embedding_dim,
            max_seq_len=max_seq_len
        )
        
        # Temperature
        self.temperature = nn.Parameter(torch.ones([]) * temperature)
        
    def forward(self, images, token_ids, attention_mask):
        # Get embeddings
        image_emb = self.image_encoder(images)
        text_emb = self.text_encoder(token_ids, attention_mask)
        
        # Similarity matrix
        similarity = image_emb @ text_emb.T
        similarity = similarity / self.temperature
        
        # Contrastive loss
        batch_size = images.shape[0]
        labels = torch.arange(batch_size, device=images.device)
        
        loss_i2t = F.cross_entropy(similarity, labels)
        loss_t2i = F.cross_entropy(similarity.T, labels)
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss, similarity

# Test
if __name__ == "__main__":
    model = TinyCLIPImproved(embedding_dim=128)
    dummy_images = torch.randn(4, 3, 224, 224)
    dummy_tokens = torch.randint(0, 30522, (4, 16))
    dummy_mask = torch.ones(4, 16)
    
    loss, sim = model(dummy_images, dummy_tokens, dummy_mask)
    print(f"Loss: {loss.item():.4f}")
    print(f"Similarity shape: {sim.shape}")