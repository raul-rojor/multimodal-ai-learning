# ============================================
# PART 2: MORNING - LOAD CHECKPOINT AND INFERENCE (30 min)
# ============================================

print("\n" + "=" * 60)
print("MORNING PART 2: LOAD CHECKPOINT AND INFERENCE")
print("=" * 60)

"""
LOAD CHECKPOINT AND RUN INFERENCE
File: load_checkpoint.py

Load a trained model and run inference on new images/texts.
"""

import torch
import os
from tinyclip9 import TinyCLIP
from multimodal_dataset10 import MultimodalDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

print("=" * 50)
print("LOADING CHECKPOINT")
print("=" * 50)

# Configuration
config = {
    'embedding_dim': 128,
    'vocab_size': 49408,
    'max_seq_len': 16,
    'num_heads': 4,
    'num_layers': 1,
    'temperature': 0.07,
    'checkpoint_dir': './checkpoints/',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

device = torch.device(config['device'])
print(f"Using device: {device}")

# Create model
model = TinyCLIP(
    image_embedding_dim=config['embedding_dim'],
    text_embedding_dim=config['embedding_dim'],
    vocab_size=config['vocab_size'],
    max_seq_len=config['max_seq_len'],
    num_heads=config['num_heads'],
    num_layers=config['num_layers'],
    temperature=config['temperature']
)

# Load checkpoint
checkpoint_path = os.path.join(config['checkpoint_dir'], 'subset_model.pt')
if os.path.exists(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Loaded subset_model.pt")
else:
    print("No checkpoint found!")
    exit()

model = model.to(device)
model.eval()

# ==========================================
# TEXT → IMAGE RETRIEVAL EXAMPLE
# ==========================================

print("\n" + "=" * 50)
print("TEXT → IMAGE RETRIEVAL")
print("=" * 50)

# Load a small validation set
val_dataset = MultimodalDataset(
    image_dir='./data/coco/val2014',
    captions_file='./data/coco/annotations/captions_val2014.json',
    max_seq_len=config['max_seq_len']
)

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Get one batch
batch = next(iter(val_loader))
images = batch['image'].to(device)
token_ids = batch['token_ids'].to(device)
attention_mask = batch['attention_mask'].to(device)

# Get embeddings
with torch.no_grad():
    image_embs = model.image_encoder(images)
    text_embs = model.text_encoder(token_ids, attention_mask)

# Compute similarity
similarity = image_embs @ text_embs.T

# For each image, find the best matching text
best_matches = similarity.argmax(dim=1)

print(f"Retrieved {len(best_matches)} image-text pairs")
print("Example: Image 0 matches text index {best_matches[0].item()}")

# ==========================================
# ZERO-SHOT CLASSIFICATION EXAMPLE
# ==========================================

print("\n" + "=" * 50)
print("ZERO-SHOT CLASSIFICATION")
print("=" * 50)

# Define class names
class_names = ["cat", "dog", "bird", "car", "tree", "person"]

# Create prompts
prompts = [f"a photo of a {cls}" for cls in class_names]
print(f"Prompts: {prompts}")

# Tokenize prompts
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
prompt_tokens = tokenizer(
    prompts,
    padding='max_length',
    truncation=True,
    max_length=config['max_seq_len'],
    return_tensors='pt'
)
prompt_ids = prompt_tokens['input_ids'].to(device)
prompt_mask = prompt_tokens['attention_mask'].to(device)

# Get embeddings
with torch.no_grad():
    # Get first image embedding
    image_emb = model.image_encoder(images[0:1])
    text_embs = model.text_encoder(prompt_ids, prompt_mask)

# Compute similarities
similarities = image_emb @ text_embs.T

# Get predictions
predicted_class = class_names[similarities.argmax().item()]

print(f"Image 0 predicted class: {predicted_class}")
print(f"Similarities: {similarities.squeeze().tolist()}")