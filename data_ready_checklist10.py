"""
DATA READY CHECKLIST
"""

import os
from torch.utils.data import DataLoader

print("=" * 50)
print("DATA READY CHECKLIST")
print("=" * 50)

def check_data_ready():
    checks = []
    
    try:
        from multimodal_dataset10 import MultimodalDataset
        checks.append(("✅ Dataset class imported", True))
    except ImportError:
        checks.append(("❌ multimodal_dataset.py not found", False))
        return checks
    
    try:
        dataset = MultimodalDataset(
            image_dir='./data/coco/val2014',
            captions_file='./data/coco/annotations/captions_val2014.json',
            max_seq_len=32
        )
        checks.append((f"✅ Dataset created with {len(dataset)} samples", True))
    except Exception as e:
        checks.append((f"❌ Dataset creation failed: {e}", False))
        return checks
    
    try:
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        batch = next(iter(dataloader))
        checks.append((f"✅ Dataloader works. image shape: {batch['image'].shape}", True))
    except Exception as e:
        checks.append((f"❌ Dataloader failed: {e}", False))
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        checks.append(("✅ Tokenizer loaded", True))
    except:
        checks.append(("⚠️ Tokenizer not loaded (run: pip install transformers)", False))
    
    return checks

checks = check_data_ready()
print("\nRESULTS:")
for check, status in checks:
    print(f"  {check}")