import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from roi_dynamic import get_roi_mask

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def apply_blur_background_suppression(image: np.ndarray, roi_mask: np.ndarray, blur_kernel: int = 21) -> np.ndarray:
    """ Keep ROI unchanged and blur everything outside ROI. """
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    roi_mask = (roi_mask > 0).astype(np.float32)
    blurred = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    
    suppressed = (
        image.astype(np.float32) * roi_mask
        + blurred.astype(np.float32) * (1.0 - roi_mask)
    )
    return np.clip(suppressed, 0, 255).astype(np.uint8)

class KneeOADynamicSmoothingBlurDataset(Dataset):
    def __init__(self, csv_path: str | Path, split: str, max_alpha: float = 0.4, noise_std: float = 0.05, suppression_prob: float = 0.5, blur_kernel: int = 21):
        self.csv_path = Path(csv_path)
        self.split = split
        self.max_alpha = max_alpha
        self.noise_std = noise_std
        self.suppression_prob = suppression_prob
        self.blur_kernel = blur_kernel
        self.data = pd.read_csv(self.csv_path)

        if "image_path" not in self.data.columns or "true_label" not in self.data.columns:
            raise ValueError(f"{self.csv_path} must contain image_path and true_label")

        self.alphas = []
        for _, row in self.data.iterrows():
            alpha = 0.0
            if "dynamic_roi_inside_ratio" in row and "border_attention_score" in row:
                roi = float(row["dynamic_roi_inside_ratio"])
                border = float(row["border_attention_score"])
                
                if not np.isnan(roi) and not np.isnan(border):
                    roi_penalty = max(0.0, (0.4 - roi) / 0.4) 
                    border_penalty = max(0.0, (border - 0.2) / 0.8)
                    suspicion = min(1.0, roi_penalty + border_penalty)
                    alpha = suspicion * self.max_alpha
            self.alphas.append(alpha)

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = PROJECT_ROOT / row["image_path"]
        label = int(row["true_label"])
        alpha = float(self.alphas[idx])

        image = Image.open(image_path).convert("L")
        image = image.resize((224, 224), Image.BILINEAR)
        image_np = np.asarray(image).astype(np.uint8)

        # === TRAINING AUGMENTATIONS & BLUR ===
        if self.split == "train":
            # 1. Random Horizontal Flip (50%)
            if random.random() < 0.5:
                image_np = np.fliplr(image_np).copy()
            
            # 2. ROI-guided Background Blur
            if random.random() < self.suppression_prob:
                roi_mask = get_roi_mask(image_np)
                image_np = apply_blur_background_suppression(image_np, roi_mask, self.blur_kernel)

            # 3. Random Inversion (50%) - nach dem Blur, damit Masken-Generierung intakt bleibt
            if random.random() < 0.5:
                image_np = cv2.bitwise_not(image_np)

        image_pil = Image.fromarray(image_np, mode="L")
        image_tensor = self.transform(image_pil)

        # 4. Gaussian Noise auf den Tensor
        if self.split == "train" and self.noise_std > 0.0:
            noise = torch.randn_like(image_tensor) * self.noise_std
            image_tensor = image_tensor + noise

        return image_tensor, label, alpha

class DynamicLabelSmoothingLoss(nn.Module):
    def __init__(self, weight=None, num_classes=5):
        super().__init__()
        self.weight = weight  
        self.num_classes = num_classes

    def forward(self, logits, targets, alphas):
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0)
            alphas = alphas.unsqueeze(1) 
            true_dist = true_dist * (1.0 - alphas) + alphas / self.num_classes

        if self.weight is not None:
            weight_expanded = self.weight.unsqueeze(0) 
            loss = - (weight_expanded * true_dist * log_probs).sum(dim=-1)
            norm = (weight_expanded * true_dist).sum(dim=-1).sum()
            return loss.sum() / norm
        else:
            loss = - (true_dist * log_probs).sum(dim=-1)
            return loss.mean()

def build_densenet201(num_classes: int = 5) -> nn.Module:
    weights = models.DenseNet201_Weights.IMAGENET1K_V1
    model = models.densenet201(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for images, labels, alphas in tqdm(dataloader, desc="Training", leave=False):
        images, labels, alphas = images.to(device), labels.to(device), alphas.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels, alphas)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    return total_loss / len(dataloader.dataset), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro")

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for images, labels, alphas in tqdm(dataloader, desc="Validation", leave=False):
        images, labels, alphas = images.to(device), labels.to(device), alphas.to(device, dtype=torch.float32)

        logits = model(images)
        loss = criterion(logits, labels, alphas)

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    return total_loss / len(dataloader.dataset), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro"), f1_score(all_labels, all_preds, average="weighted")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="outputs/all_cases_dynamic_roi_scores_train.csv")
    parser.add_argument("--val-csv", default="outputs/all_cases_dynamic_roi_scores_val.csv")
    parser.add_argument("--output-dir", default="outputs/densenet_dynamic_smoothing_blur_aug")
    
    # Smoothing & Noise
    parser.add_argument("--max-alpha", type=float, default=0.5)
    parser.add_argument("--noise-std", type=float, default=0.05)
    
    # Blur
    parser.add_argument("--suppression-prob", type=float, default=0.5)
    parser.add_argument("--blur-kernel", type=int, default=21)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = KneeOADynamicSmoothingBlurDataset(PROJECT_ROOT / args.train_csv, split="train", max_alpha=args.max_alpha, noise_std=args.noise_std, suppression_prob=args.suppression_prob, blur_kernel=args.blur_kernel)
    val_dataset = KneeOADynamicSmoothingBlurDataset(PROJECT_ROOT / args.val_csv, split="val", max_alpha=0.0, noise_std=0.0, suppression_prob=0.0)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_densenet201(num_classes=5).to(device)

    class_counts = train_dataset.data["true_label"].value_counts().sort_index()
    class_weights = torch.tensor((len(train_dataset) / (5 * class_counts)).values, dtype=torch.float32, device=device)

    criterion = DynamicLabelSmoothingLoss(weight=class_weights, num_classes=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_macro_f1 = -1.0
    history = []

    config = {
        "architecture": "DenseNet201",
        "loss": "DynamicLabelSmoothingLoss",
        "max_alpha": args.max_alpha,
        "augmentations": ["HorizontalFlip", "Inversion (p=0.5)", f"GaussianNoise (std={args.noise_std})"],
        "intervention": "dynamic ROI-guided background blur",
        "suppression_probability": args.suppression_prob,
        "blur_kernel": args.blur_kernel,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc, train_macro_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_macro_f1, val_weighted_f1 = evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc, "train_macro_f1": train_macro_f1,
            "val_loss": val_loss, "val_accuracy": val_acc, "val_macro_f1": val_macro_f1, "val_weighted_f1": val_weighted_f1
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
        
        print(f"train_loss={train_loss:.4f} train_macro_f1={train_macro_f1:.4f} | val_loss={val_loss:.4f} val_macro_f1={val_macro_f1:.4f}")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            full_checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_macro_f1": best_val_macro_f1,
                "config": config,
            }
            torch.save(full_checkpoint, output_dir / "best_model.pt")
            print(f"Neues bestes Modell gespeichert mit val_macro_f1={best_val_macro_f1:.4f}")

if __name__ == "__main__":
    main()