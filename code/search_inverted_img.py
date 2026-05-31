import cv2
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

def is_inverted(img_path):
    """
    List all images which are inverted by checking gradient
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
        
    h, w = img.shape
    
    # border
    by, bx = max(1, int(h * 0.05)), max(1, int(w * 0.05))
    
    top = img[0:by, :]
    bottom = img[h-by:h, :]
    left = img[:, 0:bx]
    right = img[:, w-bx:w]
    
    # cntr
    cy, cx = h // 2, w // 2
    dy, dx = int(h * 0.2), int(w * 0.2)
    center = img[cy-dy:cy+dy, cx-dx:cx+dx]
    
    # avg brightness
    edge_mean = np.mean([
        np.mean(top), 
        np.mean(bottom), 
        np.mean(left), 
        np.mean(right)
    ])
    center_mean = np.mean(center)
    
    return edge_mean > center_mean

def find_inverted_images(directory_path, output_txt_path):
    root_dir = Path(directory_path)
    inverted_files = []
    
    extensions = ['*.png']
    
    print(f"Collect paths in '{directory_path}'...")
    
    all_files = []
    for ext in extensions:
        all_files.extend(list(root_dir.rglob(ext)))
        
    if not all_files:
        print("Not a single image -.-")
        return

    print(f"{len(all_files)} Images found. Lets go...")
    
    for file_path in tqdm(all_files, desc="Check images", unit=" img"):
        if is_inverted(file_path):
            inverted_files.append(str(file_path))
                
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for file in inverted_files:
            f.write(f"{file}\n")
            
    print(f"\nFertig! Es wurden {len(inverted_files)} inverted images found.")
    print(f"Paths were saved in '{output_txt_path}'.")

if __name__ == "__main__":
    DATASET_PATH = sys.argv[1] if len(sys.argv) > 1 else "."
    OUTPUT_FILE = sys.argv[2] if len(sys.argv) > 2 else "data/analysis/inverted_imgs.txt"
    
    find_inverted_images(DATASET_PATH, OUTPUT_FILE)