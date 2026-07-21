"""
DATASET COMPARISON AND DOWNLOAD GUIDE

Choose the right dataset for your TinyCLIP training.
"""

print("=" * 50)
print("DATASET COMPARISON")
print("=" * 50)

print("""
┌─────────────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│ Dataset             │ Images      │ Captions    │ Size        │ Best for    │
├─────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ COCO Captions       │ 118k        │ 5 per image │ ~25 GB      │ Benchmark   │
│ Conceptual Captions │ 3.3M        │ 1 per image │ ~150 GB     │ Large-scale │
│ Flickr30k           │ 31k         │ 5 per image │ ~6 GB       │ Prototyping │
│ SBU Captions        │ 1M          │ 1 per image │ ~20 GB      │ Mid-size    │
└─────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┘

RECOMMENDATION:
  Start with Flickr30k (small, fast to download, enough to test your pipeline).
  Move to COCO after everything works.
""")

print("\n" + "=" * 50)
print("DOWNLOADING FLICKR30K")
print("=" * 50)

print("""
Option 1: Using HuggingFace datasets (easiest)
  pip install datasets
  from datasets import load_dataset
  dataset = load_dataset("nlphuji/flickr30k")

Option 2: Manual download (if HuggingFace doesn't work)
  Go to: http://shannon.cs.illinois.edu/DenotationGraph/
  Download: flickr30k-images.tar.gz
  Download: flickr30k-captions.tar.gz

Option 3: Use a small subset (even faster)
  Download a pre-processed subset from:
  https://github.com/openai/CLIP/tree/main/data
""")

print("\n" + "=" * 50)
print("DOWNLOADING COCO")
print("=" * 50)

print("""
Option 1: Using torchvision (easiest)
  from torchvision.datasets import CocoCaptions
  dataset = CocoCaptions(
      root="data/train2014",
      annFile="data/annotations/captions_train2014.json"
  )

Option 2: Manual download
  Images: http://cocodataset.org/#download
  - 2014 Train images: ~13 GB
  - 2014 Val images: ~6 GB
  - Annotations: ~200 MB

Note: COCO is 25 GB total. Download overnight.
""")

print("\n" + "=" * 50)
print("RECOMMENDED SETUP")
print("=" * 50)

print("""
1. Start with FLICKR30K (or COCO subset)
2. Keep images in: ./data/flickr30k/images/
3. Keep captions in: ./data/flickr30k/captions.txt
4. Use a small subset (e.g., 10,000 images) for initial testing

Your folder structure should look like:
  ./data/
    flickr30k/
      images/
        1000092795.jpg
        1000248817.jpg
        ...
      captions.txt
      train.txt (list of image IDs for train)
      val.txt (list of image IDs for validation)

This keeps everything organized and reproducible.
""")