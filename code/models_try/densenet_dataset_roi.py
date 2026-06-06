import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms

from roi_dynamic import get_roi_mask

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class KneeOADataset(Dataset):
    def __init__(self, csv_file, split='train', root_dir=None):
        """
        Args:
            csv_file (str or Path): .
            split (str): 'train', 'val' or 'test'.
            root_dir (str or Path, optional): image root.
        """
        self.data = pd.read_csv(csv_file)
        self.split = split
        self.root_dir = Path(root_dir) if root_dir else PROJECT_ROOT


        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        img_path_str = str(row['image_path'])
        if Path(img_path_str).is_absolute():
            img_path = Path(img_path_str)
        else:
            img_path = self.root_dir / img_path_str

        label = int(row.get('label', row.get('true_label', 0)))

        image_pil = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image_pil)

        # mask mx
        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise FileNotFoundError(f"Bild nicht gefunden oder defekt: {img_path}")
        img_gray = cv2.resize(img_gray, (224, 224))

        # roi mask
        roi_mask = get_roi_mask(img_gray)
        
        # trigger penalizing
        out_roi_mask = 1 - roi_mask

        # mask border
        h, w = img_gray.shape
        border_frac = 0.08
        by = max(1, int(h * border_frac))
        bx = max(1, int(w * border_frac))

        border_mask = np.zeros((h, w), dtype=np.uint8)
        border_mask[:by, :] = 1
        border_mask[-by:, :] = 1
        border_mask[:, :bx] = 1
        border_mask[:, -bx:] = 1

        # tensor conv
        border_mask_tensor = torch.from_numpy(border_mask).float()
        out_roi_mask_tensor = torch.from_numpy(out_roi_mask).float()
        
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor, border_mask_tensor, out_roi_mask_tensor