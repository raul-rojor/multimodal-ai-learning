# ============================================
# PART 2: MORNING - COMPLETE TEXT ENCODER (1 hour)
# ============================================

print("\n" + "=" * 60)
print("MORNING PART 2: COMPLETE TEXT ENCODER")
print("=" * 60)

"""
COMPLETE TEXT ENCODER FOR MULTIMODAL AI
File: text_encoder.py

This Transformer outputs 512-dim embeddings that can be compared with image embeddings.

INPUT: Token IDs (integers representing words/subwords)
OUTPUT: 512-dim embedding vector per text (L2 normalized)

WHERE INPUT COMES FROM:
  - Raw text (e.g., "a photo of a cat")
  - Tokenizer converts text to token IDs (e.g., [5, 12, 8, 3, 42])
  - Tokenizer also adds special tokens ([CLS], [SEP], [PAD])

WHAT EACH STEP DOES:
  1. Token Embedding: ID → 512-dim vector (learned)
  2. Positional Encoding: Add word order info (fixed formula)
  3. Dropout: Regularization (randomly zero out some values)
  4. Transformer Layers: Self-attention + FFN (×4 layers)
  5. Mean Pooling: Combine word vectors into one sentence vector
  6. Projection: Final linear transformation
  7. L2 Normalize: Make unit vector for cosine similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TextEncoder(nn.Module):
    """
    Transformer-based text encoder.
    Input: token IDs (batch, seq_len)
    Output: (batch, 512) - text embeddings
    """
    
    def __init__(self, vocab_size=30000, embedding_dim=512, max_seq_len=64,
                 num_heads=8, num_layers=4, dropout=0.1):
        super(TextEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        # ==========================================
        # STEP 1: TOKEN EMBEDDING
        # ==========================================
        # Converts token ID (e.g., 42) to a 512-dim vector.
        # This is a LEARNED lookup table.
        # Example: token 5 ("cat") → vector [0.2, -0.5, 0.8, ...]
        # Shape: (vocab_size, embedding_dim)
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # ==========================================
        # STEP 2: POSITIONAL ENCODING (FIXED)
        # ==========================================
        # Adds information about word order.
        # Uses sine/cosine formula (not learned).
        # Added to token embeddings (not concatenated).
        # Shape: (1, max_seq_len, embedding_dim)
        self.register_buffer('positional_encoding', 
                            self._get_positional_encoding(max_seq_len, embedding_dim))
        
        # ==========================================
        # STEP 3: DROPOUT (REGULARIZATION)
        # ==========================================
        # Randomly sets some values to 0 during training.
        # Prevents overfitting (model memorizing training data).
        # Only active during training, disabled during evaluation.
        self.dropout = nn.Dropout(dropout)
        
        # ==========================================
        # STEP 4: TRANSFORMER ENCODER LAYERS
        # ==========================================
        # Each layer contains:
        #   a) Multi-head self-attention - sublayer
        #      - Each word looks at every other word
        #      - Computes relationships (e.g., "it" → "animal")
        #      - Output is weighted sum of values from all words
        #
        #   b) Feed-Forward Network (FFN) - sublayer
        #      - Linear(512→2048) → ReLU → Linear(2048→512)
        #      - Processes each word independently
        #      - Adds non-linearity to learn complex patterns
        #
        #   c) Residual connections + LayerNorm (around both)
        #      - Each sublayer receives the same input and adds the sublayer output back onto it
        #      - Input to the self-attention sublayer is the current token representation `x`
        #      - Self-attention produces `attn_output = Attention(x, x, x)`
        #      - Then the layer computes `x = LayerNorm(x + attn_output)`
        #      - Input to the FFN sublayer is the normalized attention output
        #      - FFN produces `ffn_output = FFN(x)`
        #      - Then the layer computes `x = LayerNorm(x + ffn_output)`
        #      - This preserves the original input signal, stabilizes gradients, and makes deep stacking easier
        #      - Residual + LayerNorm keeps the network trainable and prevents the model from forgetting earlier representations
        #      - Prevents vanishing gradients
        #
        # ×4 layers (num_layers=4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,           # 512
            nhead=num_heads,                 # 8 attention heads
            dim_feedforward=embedding_dim * 4,  # 2048
            dropout=dropout,
            batch_first=True                 # (batch, seq, dim) not (seq, batch, dim)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # ==========================================
        # STEP 6: FINAL PROJECTION
        # ==========================================
        # After pooling, we have 512-dim vector.
        #   - Pooling refers to the previous step, where the model converted a sequence of token vectors into a single sentence vector.
        #   - In this code, the pooling is mean pooling: it averages the token vectors across the sequence so the whole text becomes one fixed-size vector.
        # This is another linear layer that refines the embedding -- a learned transformation.
        # It can:
        #   - Adjust the embedding space
        #   - Mix dimensions
        #   - Help the model learn a better final representation for similarity or downstream tasks.
        # Shape: (512, 512)
        # It computes y = xW^T + b
        # Each output dimension is a weighted sum of all 512 input values
        # Different output dimensions use different learned weights
        # Kept separate from the transformer for clarity.
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        
    def _get_positional_encoding(self, seq_len, d_model):
        """
        FIXED POSITIONAL ENCODINGS (not learned)
        
        Formula:
            PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        Why sine/cosine:
            - Unique pattern for every position
            - Low dimensions change slowly (coarse location)
            - High dimensions change rapidly (fine location)
            - Sin + cos together resolve ambiguity
        """
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))  # (d_model/2,)
        
        # Even indices (0, 2, 4, ...): sine
        pe[:, 0::2] = torch.sin(position * div_term)
        # Odd indices (1, 3, 5, ...): cosine
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # (1, seq_len, d_model)
    
    def forward(self, token_ids, attention_mask=None):
        """
        Forward pass through the text encoder.
        
        Args:
            token_ids: (batch_size, seq_len) - token IDs (integers)
            attention_mask: (batch_size, seq_len) - 1 for real tokens, 0 for padding
        
        Returns:
            text_embeddings: (batch_size, embedding_dim) - L2 normalized
        
        WHERE TOKEN_IDS COME FROM:
            - Raw text → Tokenizer (e.g., HuggingFace tokenizer)
            - Tokenizer adds special tokens ([CLS], [SEP], [PAD])
            - token_ids.shape: (batch, max_seq_len)
            
        WHERE ATTENTION_MASK COMES FROM:
            - Created by tokenizer: 1 for real words, 0 for padding
            - Ensures padding doesn't affect attention
            - Prevents model from "looking at" empty positions
        """
        batch_size, seq_len = token_ids.shape
        
        # ==========================================
        # STEP 1: TOKEN EMBEDDING
        # ==========================================
        # Maps each token ID to a 512-dim vector.
        # Example: token 5 ("cat") → vector [0.2, -0.5, 0.8, ...]
        x = self.token_embedding(token_ids)  # (batch, seq_len, embedding_dim)
        # Now each token has a vector representation.
        # These vectors are LEARNED during training.
        
        # ==========================================
        # STEP 2: ADD POSITIONAL ENCODING
        # ==========================================
        # Adds word order information.
        # Position 0 gets PE[0], position 1 gets PE[1], etc.
        # After this, each token knows BOTH its meaning AND its position.
        x = x + self.positional_encoding[:, :seq_len, :]
        # Shape: (batch, seq_len, embedding_dim)
        
        # ==========================================
        # STEP 3: DROPOUT
        # ==========================================
        # Randomly zeros out some values during training.
        # Helps prevent overfitting.
        x = self.dropout(x)
        
        # ==========================================
        # STEP 4: TRANSFORMER LAYERS
        # ==========================================
        # Convert attention_mask to format PyTorch expects.
        # key_padding_mask = True for positions to IGNORE (padding).
        key_padding_mask = None
        if attention_mask is not None:
            # attention_mask: 1 for real tokens, 0 for padding
            # key_padding_mask: True for padding (to ignore)
            key_padding_mask = (attention_mask == 0)  # True where padding
        
        # Apply all transformer layers
        # Each layer: Self-Attention → Add+Norm → FFN → Add+Norm
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        # Shape: (batch, seq_len, embedding_dim)
        # Now each token's vector is CONTEXT-AWARE:
        #   - "it" contains information from "animal" (coreference resolved)
        #   - "sat" contains information from "cat" (subject-verb agreement)
        #   - Each word knows about all other words
        
        # ==========================================
        # STEP 5: MEAN POOLING (with mask)
        # ==========================================
        # Combine all word vectors into ONE sentence vector.
        # We average only over REAL tokens (not padding).
        if attention_mask is not None:
            # mask_expanded: (batch, seq_len, 1) - same shape as x
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            # Zero out padding contributions
            x = x * mask_expanded  # (batch, seq_len, embedding_dim)
            # Sum over sequence dimension
            x = x.sum(dim=1)  # (batch, embedding_dim)
            # Divide by number of REAL tokens
            x = x / mask_expanded.sum(dim=1)  # (batch, embedding_dim)
        else:
            # No mask: average over all tokens (including padding)
            x = x.mean(dim=1)  # (batch, embedding_dim)
        # Now we have ONE vector per text, not one per word.
        # This vector represents the ENTIRE sentence.
        
        # ==========================================
        # STEP 6: FINAL PROJECTION
        # ==========================================
        # Refine the embedding with one more linear layer.
        # This is a learned transformation.
        x = self.projection(x)  # (batch, embedding_dim)
        
        # ==========================================
        # STEP 7: L2 NORMALIZE
        # ==========================================
        # Divide by the vector's length (Euclidean norm).
        # Makes all embeddings have magnitude = 1.
        # Result: dot product = cosine similarity.
        x = F.normalize(x, p=2, dim=1)  # (batch, embedding_dim)
        # After normalization:
        #   - Similar texts have similar vectors (dot product ≈ 1)
        #   - Different texts have different vectors (dot product ≈ 0)
        #   - This is what we need for multimodal comparison!
        
        return x  # (batch, embedding_dim)


# ==========================================
# EXAMPLE USAGE
# ==========================================

if __name__ == "__main__":
    # Create the model
    model = TextEncoder(vocab_size=30000, embedding_dim=512)
    
    # Create dummy input
    # In practice, this comes from a tokenizer
    batch_size = 4
    seq_len = 32
    # ^ batch of 4 sentences, each with 32 tokens (padded if shorter)
    dummy_tokens = torch.randint(0, 30000, (batch_size, seq_len))
    # ^ creates random token IDs in [0,30000) with shape (4,32);
    #   these are indices into the embedding table (simulates tokenizer output)
    dummy_mask = torch.ones(batch_size, seq_len)  # All real tokens
    # ^ a binary mask (1=real token, 0=padding) with shape (4,32);
    #   used to ignore padding in attention and to compute mean pooling only over real tokens in case of non-existn.
    #   In real use, replace the random tokens and all-ones mask with the tokenizer's input_ids and attention_mask.

    #   Padding: extra token IDs added so every example in the batch has the same length. Padded positions use
    #   a special PAD id (e.g. 0) and are marked with 0 in the attention mask so the model ignores them in attention and pooling.
    
    # Forward pass
    with torch.no_grad():
        embeddings = model(dummy_tokens, dummy_mask)
    
    print(f"Input token IDs shape: {dummy_tokens.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Expected: ({batch_size}, 512)")
    
    # Verify L2 normalization
    norms = torch.norm(embeddings, dim=1)
    print(f"\nEmbedding norms (should all be 1.0): {norms}")