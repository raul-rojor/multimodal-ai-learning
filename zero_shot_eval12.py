# ============================================
# PART 2: MORNING - ZERO-SHOT CLASSIFICATION (30 min)
# ============================================

print("\n" + "=" * 60)
print("MORNING PART 2: ZERO-SHOT CLASSIFICATION")
print("=" * 60)

"""
ZERO-SHOT CLASSIFICATION EVALUATION
File: zero_shot_eval.py

Evaluate zero-shot classification accuracy on COCO categories.
"""

import torch
import os
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoTokenizer
from multimodal_dataset10 import MultimodalDataset
from tinyclip9 import TinyCLIP

print("=" * 50)
print("ZERO-SHOT CLASSIFICATION EVALUATION")
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
print(f"Using device: {device}")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Define COCO categories (common classes)
coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# Create prompts
prompts = [f"a photo of a {cls}" for cls in coco_classes]
print(f"Number of classes: {len(coco_classes)}")

# Tokenize prompts
print("Tokenizing prompts...")
prompt_tokens = tokenizer(
    prompts,
    padding='max_length',
    truncation=True,
    max_length=config['max_seq_len'],
    return_tensors='pt'
)
prompt_ids = prompt_tokens['input_ids'].to(device)
prompt_mask = prompt_tokens['attention_mask'].to(device)

# Load model
print("Loading model...")
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
print("Loading dataset...")
val_dataset = MultimodalDataset(
    image_dir=config['image_dir'],
    captions_file=config['captions_file'],
    max_seq_len=config['max_seq_len']
)

# Use subset
val_subset = Subset(val_dataset, range(200))
val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False)

print(f"Evaluation samples: {len(val_subset)}")

# ==========================================
# ZERO-SHOT CLASSIFICATION
# ==========================================

def zero_shot_classify(model, dataloader, prompt_ids, prompt_mask, device):
    """Classify images using text prompts"""
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        # Get text embeddings once
        text_embs = model.text_encoder(prompt_ids, prompt_mask)
        
        for batch in tqdm(dataloader, desc="Classifying"):
            images = batch['image'].to(device)
            
            # Get image embeddings
            image_embs = model.image_encoder(images)
            
            # Compute similarity with all prompts
            similarities = image_embs @ text_embs.T
            
            # Get predictions (most similar prompt)
            predictions = similarities.argmax(dim=1)
            
            all_predictions.extend(predictions.cpu().tolist())
    
    return all_predictions

print("\\nRunning zero-shot classification...")
predictions = zero_shot_classify(model, val_loader, prompt_ids, prompt_mask, device)

# Since we don't have ground truth class labels for each image,
# we can't compute exact accuracy. This is a demonstration of the pipeline.

print(f"Predictions for first 10 images: {predictions[:10]}")
print(f"Corresponding class names: {[coco_classes[p] for p in predictions[:10]]}")

print("\n" + "=" * 50)
print("ZERO-SHOT CLASSIFICATION OVERVIEW")
print("=" * 50)

print("""
Zero-shot classification is one of CLIP's superpowers!

HOW IT WORKS:
  1. Define text prompts for each class
  2. Encode all prompts (text embeddings)
  3. Encode query image (image embedding)
  4. Find which prompt has highest similarity

ADVANTAGES:
  - No training on those classes needed
  - Can add new classes instantly (just add prompts)
  - Works for thousands of classes

LIMITATIONS:
  - Quality depends on prompt engineering
  - May confuse similar classes (e.g., "dog" vs "wolf")
  - Computationally expensive for many classes

To compute actual accuracy, you would need ground truth class labels.
COCO has these in the annotation files.
""")