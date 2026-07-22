"""
ZERO-SHOT CLASSIFICATION WITH IMPROVED MODEL
"""

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoTokenizer
from multimodal_dataset10 import MultimodalDataset
from tinyclip_improved15 import TinyCLIPImproved

print("=" * 50)
print("ZERO-SHOT CLASSIFICATION (IMPROVED MODEL)")
print("=" * 50)

config = {
    'image_dir': './data/coco/val2014',
    'captions_file': './data/coco/annotations/captions_val2014.json',
    'max_seq_len': 16,
    'batch_size': 32,
    'embedding_dim': 128,
    'num_samples': 100,
    'checkpoint_path': './checkpoints/improved_model.pt',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

device = torch.device(config['device'])
print(f"Using device: {device}")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Define class names (COCO categories)
class_names = [
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
prompts = [f"a photo of a {cls}" for cls in class_names]
print(f"Number of classes: {len(class_names)}")

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
model = TinyCLIPImproved(
    embedding_dim=config['embedding_dim'],
    max_seq_len=config['max_seq_len']
)
model.load_state_dict(torch.load(config['checkpoint_path'], map_location=device))
model = model.to(device)
model.eval()
print("✅ Model loaded")

# Load dataset
print("Loading dataset...")
dataset = MultimodalDataset(
    image_dir=config['image_dir'],
    captions_file=config['captions_file'],
    max_seq_len=config['max_seq_len']
)

subset = Subset(dataset, range(min(config['num_samples'], len(dataset))))
dataloader = DataLoader(subset, batch_size=config['batch_size'], shuffle=False)

print(f"Classifying {len(subset)} images...")

# Run zero-shot classification
all_predictions = []
all_confidences = []

with torch.no_grad():
    # Get text embeddings once
    text_embs = model.text_encoder(prompt_ids, prompt_mask)
    
    for batch in tqdm(dataloader, desc="Classifying"):
        images = batch['image'].to(device)
        
        image_embs = model.image_encoder(images)
        similarities = image_embs @ text_embs.T
        
        # Get top predictions
        top_probs, top_indices = similarities.softmax(dim=1).topk(3, dim=1)
        
        for i in range(len(images)):
            preds = [class_names[idx] for idx in top_indices[i].tolist()]
            confs = [f"{p.item()*100:.1f}%" for p in top_probs[i]]
            all_predictions.append(preds)
            all_confidences.append(confs)

# Print results
print("\n" + "=" * 50)
print("ZERO-SHOT CLASSIFICATION RESULTS")
print("=" * 50)

for i in range(min(5, len(all_predictions))):
    print(f"\nImage {i+1}:")
    for j, (cls, conf) in enumerate(zip(all_predictions[i], all_confidences[i])):
        print(f"  {j+1}. {cls} ({conf})")