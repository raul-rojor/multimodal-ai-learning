"""
MULTIMODAL DATASET CLASS
File: multimodal_dataset.py
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os
import json

class MultimodalDataset(Dataset):
    def __init__(self, image_dir, captions_file, transform=None, max_seq_len=64):
        self.image_dir = image_dir
        self.max_seq_len = max_seq_len
        self.data = self._load_captions(captions_file)
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def _load_captions(self, captions_file):
        if captions_file.endswith('.json'):
            with open(captions_file, 'r') as f:
                data = json.load(f)
            
            # COCO format: data['annotations'] contains captions
            # data['images'] contains image filenames
            if 'annotations' in data:
                # Build mapping from image_id to caption
                captions_by_image = {}
                for ann in data['annotations']:
                    image_id = ann['image_id']
                    if image_id not in captions_by_image:
                        captions_by_image[image_id] = []
                    captions_by_image[image_id].append(ann['caption'])
                
                # Build mapping from image_id to filename
                id_to_filename = {}
                for img in data['images']:
                    id_to_filename[img['id']] = img['file_name']
                
                # Create flat list
                result = []
                for image_id, captions in captions_by_image.items():
                    if image_id in id_to_filename:
                        result.append({
                            'image': id_to_filename[image_id],
                            'caption': captions[0]  # Use first caption
                        })
                return result
            else:
                # Assume it's already a flat list
                return data
        else:
            raise ValueError(f"Unsupported file format: {captions_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        #DEBUG BLOCK
        if idx == 0:
            print(f"Loading image: {self.data[idx]['image']}")
            print(f"Caption: {self.data[idx]['caption']}")
        img_filename = self.data[idx]['image']
        img_path = os.path.join(self.image_dir, img_filename)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        caption = self.data[idx]['caption']
        # REAL TOKENIZATION (NOT RANDOM)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        tokens = tokenizer(
            caption,
           padding='max_length',
           truncation=True,
           max_length=self.max_seq_len,
            return_tensors='pt'
        )
        token_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        
        return {
            'image': image,
            'token_ids': token_ids,
            'attention_mask': attention_mask
        }