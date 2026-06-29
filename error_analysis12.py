# ============================================
# PART 4: EVENING - ERROR ANALYSIS (30 min)
# ============================================

print("\n" + "=" * 60)
print("EVENING: ERROR ANALYSIS")
print("=" * 60)

"""
ERROR ANALYSIS
File: error_analysis.py

Find where the model fails and why.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from multimodal_dataset10 import MultimodalDataset
from tinyclip9 import TinyCLIP
import os

print("=" * 50)
print("ERROR ANALYSIS")
print("=" * 50)

# Configuration
config = {
    'image_dir': './data/coco/val2014',
    'captions_file': './data/coco/annotations/captions_val2014.json',
    'max_seq_len': 16,
    'batch_size': 32,
    'embedding_dim': 128,
    'num_heads': 4,
    'num_layers': 1,
    'checkpoint_dir': './checkpoints/',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

device = torch.device(config['device'])

# Load model
model = TinyCLIP(
    image_embedding_dim=config['embedding_dim'],
    text_embedding_dim=config['embedding_dim'],
    vocab_size=49408,
    max_seq_len=config['max_seq_len'],
    num_heads=config['num_heads'],
    num_layers=config['num_layers']
)

checkpoint_path = os.path.join(config['checkpoint_dir'], 'subset_model.pt')
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Loaded subset_model.pt")
else:
    print("No checkpoint found!")
    exit()

model = model.to(device)
model.eval()

# Load dataset
val_dataset = MultimodalDataset(
    image_dir=config['image_dir'],
    captions_file=config['captions_file'],
    max_seq_len=config['max_seq_len']
)

val_subset = Subset(val_dataset, range(100))
val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False)

# ==========================================
# FIND FAILURE CASES
# ==========================================

def analyze_failures(model, dataloader, device):
    """Find cases where model retrieved wrong captions"""
    failures = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Analyzing")):
            images = batch['image'].to(device)
            token_ids = batch['token_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get embeddings
            image_embs = model.image_encoder(images)
            text_embs = model.text_encoder(token_ids, attention_mask)
            
            # Compute similarity
            similarity = image_embs @ text_embs.T
            
            # For each image, find best and worst matches
            best_matches = similarity.argmax(dim=1)
            
            # Check if best match is correct (diagonal)
            for i in range(len(images)):
                if best_matches[i] != i:
                    failures.append({
                        'image_idx': batch_idx * config['batch_size'] + i,
                        'correct_idx': i,
                        'wrong_idx': best_matches[i].item(),
                        'similarity': similarity[i, best_matches[i]].item()
                    })
    
    return failures

failures = analyze_failures(model, val_loader, device)

print(f"\\nNumber of failures: {len(failures)} out of {len(val_subset)}")
print(f"Failure rate: {len(failures)/len(val_subset)*100:.1f}%")

# ==========================================
# ANALYZE FAILURE PATTERNS
# ==========================================

if len(failures) > 0:
    print("\n" + "=" * 50)
    print("FAILURE ANALYSIS")
    print("=" * 50)
    
    # Show first few failures
    print("\nFirst 5 failures:")
    for f in failures[:5]:
        print(f"  Image {f['image_idx']} matched text {f['wrong_idx']} (should be {f['correct_idx']})")
        print(f"    Similarity score: {f['similarity']:.3f}")
    
    print("\n" + "=" * 50)
    print("COMMON FAILURE REASONS")
    print("=" * 50)

print("""
Common reasons for failures:

1. AMBIGUOUS IMAGES:
   - Multiple objects in image
   - Unclear subject
   - Image quality issues

2. SIMILAR CAPTIONS:
   - "a photo of a cat" vs "a photo of a kitten"
   - Model can't distinguish fine-grained differences

3. DATA NOISE:
   - Incorrect or irrelevant captions
   - Missing objects in image

4. MODEL LIMITATIONS:
   - Still learning (early in training)
   - Small dataset (not enough examples)
   - Architecture too small (few layers, low capacity)

HOW TO IMPROVE:
   - Train longer
   - Use more data
   - Use pre-trained encoders
   - Fine-tune on specific tasks
""")