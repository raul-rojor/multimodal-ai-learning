"""
EVALUATE IMPROVED TINYCLIP

Compare improved model results to Week 3.
"""

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from multimodal_dataset10 import MultimodalDataset
from tinyclip_improved15 import TinyCLIPImproved

print("=" * 50)
print("EVALUATING IMPROVED MODEL")
print("=" * 50)

config = {
    'image_dir': './data/coco/val2014',
    'captions_file': './data/coco/annotations/captions_val2014.json',
    'max_seq_len': 16,
    'batch_size': 32,
    'embedding_dim': 128,
    'num_samples': 500,
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
model.load_state_dict(torch.load('./checkpoints/improved_model.pt', map_location=device))
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

# Recall@K
def recall_at_k(sim, labels, k):
    top_k = sim.argsort(dim=1, descending=True)[:, :k]
    correct = (top_k == labels.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()

labels = torch.arange(len(image_embs))

# MRR calculation
def mean_reciprocal_rank(sim, labels):
    ranks = []
    for i in range(sim.shape[0]):
        sorted_indices = sim[i].argsort(descending=True)
        rank = (sorted_indices == labels[i]).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    return (1.0 / torch.tensor(ranks)).mean().item()

mrr = mean_reciprocal_rank(similarity, labels)

print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)

for k in [1, 5, 10]:
    r = recall_at_k(similarity, labels, k)
    print(f"Recall@{k}: {r:.4f}")

print("\nCOMPARISON TO WEEK 3 (Custom CNN + Transformer):")
print("  Recall@1: 0.0000 → ? (Improved should be higher)")
print("  Recall@5: 0.0120 → ?")
print("  Recall@10: 0.0220 → ?")
print(f"MRR: {mrr:.4f}")