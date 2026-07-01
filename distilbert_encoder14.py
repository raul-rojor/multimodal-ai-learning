# ============================================
# PART 2: TUESDAY - DISTILBERT TEXT ENCODER
# ============================================

print("\n" + "=" * 60)
print("DISTILBERT TEXT ENCODER")
print("=" * 60)

"""
DISTILBERT TEXT ENCODER

Replace custom Transformer with pre-trained DistilBERT.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F

class DistilBERTEncoder(nn.Module):
    """
    Pre-trained DistilBERT text encoder.
    Input: token_ids (batch, seq_len), attention_mask (batch, seq_len)
    Output: (batch, embedding_dim) - L2 normalized
    """
    
    def __init__(self, embedding_dim=128, max_seq_len=16):
        super(DistilBERTEncoder, self).__init__()
        
        # Load pre-trained DistilBERT
        # Trained on 3.3B words (Wikipedia + BooksCorpus)
        self.bert = AutoModel.from_pretrained('distilbert-base-uncased')
        
        # Project to embedding dimension
        self.projection = nn.Linear(768, embedding_dim)
        
    def forward(self, token_ids, attention_mask):
        # Pass through BERT
        # Use [CLS] token output (first token) as sentence embedding
        outputs = self.bert(
            input_ids=token_ids,
            attention_mask=attention_mask
        )
        
        # Take [CLS] token representation
        # Shape: (batch, 768)
        x = outputs.last_hidden_state[:, 0, :]
        
        # Project to embedding dimension
        x = self.projection(x)  # (batch, embedding_dim)
        
        # L2 normalize
        x = F.normalize(x, p=2, dim=1)
        
        return x

# Test
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    model = DistilBERTEncoder(embedding_dim=128)
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    texts = ["a photo of a cat", "a photo of a dog"]
    tokens = tokenizer(texts, padding=True, return_tensors='pt')
    
    output = model(tokens['input_ids'], tokens['attention_mask'])
    print(f"DistilBERTEncoder output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")