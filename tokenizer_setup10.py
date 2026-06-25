"""
TOKENIZER SETUP
File: tokenizer_setup.py
"""

print("=" * 50)
print("TOKENIZER SETUP")
print("=" * 50)

try:
    from transformers import AutoTokenizer
    print("✅ Transformers installed")
    
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    print("✅ Tokenizer loaded")
    
    caption = "a photo of a cat sitting on a mat"
    tokens = tokenizer(
        caption,
        padding='max_length',
        truncation=True,
        max_length=16,
        return_tensors='pt'
    )
    
    print(f"\nCaption: {caption}")
    print(f"Token IDs: {tokens['input_ids'][0].tolist()}")
    print(f"Attention mask: {tokens['attention_mask'][0].tolist()}")
    
except ImportError:
    print("❌ transformers not installed")
    print("Run: pip install transformers")
except Exception as e:
    print(f"❌ Error: {e}")