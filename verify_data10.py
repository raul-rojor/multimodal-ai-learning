"""
VERIFY DATA DOWNLOAD
"""

import os

print("=" * 50)
print("VERIFYING COCO DOWNLOAD")
print("=" * 50)

paths = {
    "train2014": "./data/coco/train2014",
    "val2014": "./data/coco/val2014",
    "annotations": "./data/coco/annotations"
}

for name, path in paths.items():
    if os.path.exists(path):
        if os.path.isdir(path):
            count = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.json'))])
            print(f"✅ {name}: {count} files")
        else:
            print(f"✅ {name}: exists")
    else:
        print(f"❌ {name}: not found")

print("\n" + "=" * 50)
print("STATUS")
print("=" * 50)

if os.path.exists("./data/coco/val2014") and os.path.exists("./data/coco/annotations/captions_val2014.json"):
    print("✅ COCO validation set ready!")
elif os.path.exists("dummy_data"):
    print("⚠️ COCO not ready, but dummy data available")
else:
    print("❌ No data found. Run download_data.py first.")