"""
EVALUATE SCALED TINYCLIP
"""

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from multimodal_dataset10 import MultimodalDataset
from tinyclip_improved15 import TinyCLIPImproved

print("=" * 50)
print("EVALUATING SCALED MODEL")
print("=" * 50)

config = {
    'image_dir': './data/coco/val2014',
    'captions_file': './data/coco/annotations/captions_val2014.json',
    'max_seq_len': 16,
    'batch_size': 32,
    'embedding_dim': 128,
    'num_samples': 1000,
    'checkpoint_dir': './checkpoints/',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

device = torch.device(config['device'])

# Load dataset
dataset = MultimodalDataset(
    image_dir=config['image_dir'],
    captions_file=config['captions_file'],
    max_seq_len=config['max_seq_len']
)

subset = Subset(dataset, range(min(config['num_samples'], len(dataset))))
dataloader = DataLoader(subset, batch_size=config['batch_size'], shuffle=False)

print(f"Evaluating on {len(subset)} samples")

# Load model
model = TinyCLIPImproved(
    embedding_dim=config['embedding_dim'],
    max_seq_len=config['max_seq_len']
)
model.load_state_dict(torch.load('./checkpoints/scaled_model.pt', map_location=device))
model = model.to(device)
model.eval()

# Compute embeddings
all_image_embs = []
all_text_embs = []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Computing embeddings"):
        images = batch['image'].to(device)
        token_ids = batch['token_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        image_embs = model.image_encoder(images)
        text_embs = model.text_encoder(token_ids, attention_mask)
        
        all_image_embs.append(image_embs.cpu())
        all_text_embs.append(text_embs.cpu())

image_embs = torch.cat(all_image_embs)
text_embs = torch.cat(all_text_embs)

similarity = image_embs @ text_embs.T

def recall_at_k(sim, labels, k):
    top_k = sim.argsort(dim=1, descending=True)[:, :k]
    correct = (top_k == labels.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()

labels = torch.arange(len(image_embs))

print("\n" + "=" * 50)
print("SCALED MODEL RESULTS")
print("=" * 50)

for k in [1, 5, 10]:
    r = recall_at_k(similarity, labels, k)
    print(f"Recall@{k}: {r:.4f}")

print("\nCOMPARISON:")
print(f"  Week 3 (1k images, 3 epochs): Recall@1 = 0.0000")
print(f"  Week 4 (5k images, 5 epochs): Recall@1 = ?")
print(f"  Week 5 (20k images, 10 epochs): Recall@1 = ?")