"""
DOWNLOAD DATA

Download COCO captions dataset manually.
"""

import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(filename, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def setup_coco():
    """Download and setup COCO dataset"""
    
    # Create directories
    os.makedirs("./data/coco/train2014", exist_ok=True)
    os.makedirs("./data/coco/val2014", exist_ok=True)
    os.makedirs("./data/coco/annotations", exist_ok=True)
    
    # URLs for COCO 2014
    urls = {
        "train2014": "http://images.cocodataset.org/zips/train2014.zip",
        "val2014": "http://images.cocodataset.org/zips/val2014.zip",
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    }
    
    print("Downloading COCO 2014...")
    print("This is ~20 GB. It will take a while.")
    
    # Download and extract each file
    for name, url in urls.items():
        zip_path = f"./data/coco/{name}.zip"
        print(f"\nDownloading {name}...")
        download_file(url, zip_path)
        
        print(f"Extracting {name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("./data/coco/")
        
        # Remove zip file to save space
        os.remove(zip_path)
    
    print("\n✅ COCO dataset downloaded successfully!")
    print("Files saved to ./data/coco/")

if __name__ == "__main__":
    setup_coco()