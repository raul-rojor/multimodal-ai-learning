"""
SIMPLE TEXT ENCODER IN PYTORCH

This is your text encoder that outputs embeddings for multimodal matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=" * 50)
print("BUILDING A TEXT ENCODER")
print("=" * 50)

class SimpleTextEncoder(nn.Module):
    """
    A simplified Transformer text encoder.
    Input: token IDs (batch, seq_len)
    Output: text embedding (batch, embedding_dim)
    """
    
    def __init__(self, vocab_size, embedding_dim=512, max_seq_len=64, 
                 num_heads=8, num_layers=4, dropout=0.1):
        super(SimpleTextEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        # Token embedding: token ID → vector
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding (fixed, not learned)
        self.register_buffer('positional_encoding', 
                            self._get_positional_encoding(max_seq_len, embedding_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final projection (already embedding_dim, but keeping for clarity)
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        
    def _get_positional_encoding(self, seq_len, d_model):
        """Fixed positional encodings (not learned)"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, seq_len, d_model)
    
    def forward(self, token_ids, attention_mask=None):
        """
        Args:
            token_ids: (batch_size, seq_len) - token IDs
            attention_mask: (batch_size, seq_len) - 1 for real tokens, 0 for padding
        
        Returns:
            text_embeddings: (batch_size, embedding_dim)
        """
        batch_size, seq_len = token_ids.shape
        
        # Step 1: Token embeddings
        x = self.token_embedding(token_ids)  # (batch, seq_len, embedding_dim)
        
        # Step 2: Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Step 3: Apply Transformer (self-attention + feed-forward)
        # Convert attention mask to key_padding_mask format
        key_padding_mask = None
        if attention_mask is not None:
            # attention_mask: 1 for real tokens, 0 for padding
            # key_padding_mask expects True for positions to IGNORE
            key_padding_mask = (attention_mask == 0)
        
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        
        # Step 4: Pool to get single embedding per sequence
        # Option A: Take first token's output (like BERT's [CLS])
        # Option B: Average over all real tokens
        if attention_mask is not None:
            # Average over real tokens only
            mask_expanded = attention_mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            # Average over all tokens
            x = x.mean(dim=1)
        
        # Step 5: Final projection
        x = self.projection(x)
        
        # Normalize embeddings (important for similarity computation)
        x = F.normalize(x, p=2, dim=1)
        
        return x

# Create model instance
vocab_size = 30000  # Typical vocabulary size
model = SimpleTextEncoder(vocab_size=vocab_size, embedding_dim=512)

print(f"Model: {model.__class__.__name__}")
print(f"Vocabulary size: {vocab_size}")
print(f"Embedding dimension: 512")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# TODO 1: Test with dummy input
print("\n" + "=" * 50)
print("TESTING THE TEXT ENCODER")
print("=" * 50)

# Dummy batch of token IDs (4 sentences, each up to 32 tokens)
batch_size = 4
seq_len = 32
dummy_token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
dummy_mask = torch.ones(batch_size, seq_len)  # All real tokens (no padding)

print(f"Input token IDs shape: {dummy_token_ids.shape}")
print(f"Input mask shape: {dummy_mask.shape}")

with torch.no_grad():
    embeddings = model(dummy_token_ids, dummy_mask)

print(f"Output embeddings shape: {embeddings.shape}")
print(f"Expected: ({batch_size}, 512) → {embeddings.shape[0]} texts, each {embeddings.shape[1]}-dim")

# TODO 2: Test with padding (different length sentences)
print("\n" + "=" * 50)
print("TESTING WITH PADDING")
print("=" * 50)

# Sentence 1: "cat" (3 tokens after tokenization)
# Sentence 2: "the fluffy cat sat on the mat" (7 tokens)
# Pad both to length 10
seq_len = 10
sentence1_ids = torch.tensor([[5, 12, 8, 0, 0, 0, 0, 0, 0, 0]])  # 0 = <PAD>
sentence2_ids = torch.tensor([[2, 45, 5, 12, 8, 3, 2, 45, 6, 0]])

mask1 = (sentence1_ids != 0).int()
mask2 = (sentence2_ids != 0).int()

print(f"Sentence 1 token IDs: {sentence1_ids[0].tolist()}")
print(f"Sentence 1 mask: {mask1[0].tolist()} (0 = padding to ignore)")
print(f"Sentence 2 token IDs: {sentence2_ids[0].tolist()}")
print(f"Sentence 2 mask: {mask2[0].tolist()}")

with torch.no_grad():
    emb1 = model(sentence1_ids, mask1)
    emb2 = model(sentence2_ids, mask2)

print(f"\nSentence 1 embedding (first 5 values): {emb1[0, :5].detach().numpy().round(3)}")
print(f"Sentence 2 embedding (first 5 values): {emb2[0, :5].detach().numpy().round(3)}")

# Similarity between the two sentences
similarity = torch.cosine_similarity(emb1, emb2)
print(f"\nCosine similarity between sentences: {similarity.item():.3f}")
print("(Higher = more similar meaning)")

print("\n" + "=" * 50)
print("WHAT YOU JUST BUILT")
print("=" * 50)

print("""
The text encoder:

  1. Converts token IDs to embeddings (token_embedding)
  2. Adds positional information (positional_encoding)
  3. Applies self-attention (Transformer layers)
     - Every word looks at every other word
     - Captures meaning and relationships
  4. Pools to single vector (average over real tokens)
  5. Projects and normalizes to 512-dim embedding

This embedding vector represents the TEXT CONTENT.
Similar meaning → Similar embedding vectors.
""")