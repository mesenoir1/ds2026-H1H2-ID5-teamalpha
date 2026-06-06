import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

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

def create_border_mask(h, w, border_frac=0.08):
    mask = np.zeros((h, w), dtype=np.float32)
    by = max(1, int(h * border_frac))
    bx = max(1, int(w * border_frac))
    mask[:by, :] = 1
    mask[-by:, :] = 1
    mask[:, :bx] = 1
    mask[:, -bx:] = 1
    return mask

def apply_blur_background_suppression(image: np.ndarray, roi_mask: np.ndarray, blur_kernel: int = 21) -> np.ndarray:
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    roi_mask_float = (roi_mask > 0).astype(np.float32)
    blurred = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    suppressed = (image.astype(np.float32) * roi_mask_float + blurred.astype(np.float32) * (1.0 - roi_mask_float))
    return np.clip(suppressed, 0, 255).astype(np.uint8)

class KneeOAHybridDataset(Dataset):
    def __init__(self, csv_path: str | Path, split: str, max_alpha: float = 0.5, 
                 lambda_border_base: float = 0.1, lambda_roi_base: float = 0.1,
                 noise_std: float = 0.05, suppression_prob: float = 0.5, blur_kernel: int = 21):
        self.csv_path = Path(csv_path)
        self.split = split
        self.max_alpha = max_alpha
        self.noise_std = noise_std
        self.suppression_prob = suppression_prob
        self.blur_kernel = blur_kernel
        
        self.data = pd.read_csv(self.csv_path)
        if "image_path" not in self.data.columns or "true_label" not in self.data.columns:
            raise ValueError(f"{self.csv_path} must contain image_path and true_label")

        # Berechne individuelle Alphas und Lambdas
        self.alphas, self.l_borders, self.l_rois = [], [], []
        for _, row in self.data.iterrows():
            alpha, l_border, l_roi = 0.0, 0.0, 0.0
            if "dynamic_roi_inside_ratio" in row and "border_attention_score" in row:
                roi = float(row["dynamic_roi_inside_ratio"])
                border = float(row["border_attention_score"])
                
                if not np.isnan(roi) and not np.isnan(border):
                    # Smoothing Alpha
                    roi_penalty = max(0.0, (0.4 - roi) / 0.4) 
                    border_penalty = max(0.0, (border - 0.2) / 0.8)
                    alpha = min(1.0, roi_penalty + border_penalty) * self.max_alpha
                    
                    # Hard Thresholds für die Attention Loss aus der CSV Analyse
                    if roi < 0.40:
                        l_roi = lambda_roi_base
                    if border > 0.20:
                        l_border = lambda_border_base
                        
            self.alphas.append(alpha)
            self.l_borders.append(l_border)
            self.l_rois.append(l_roi)

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
        l_border = float(self.l_borders[idx])
        l_roi = float(self.l_rois[idx])

        image = Image.open(image_path).convert("L").resize((224, 224), Image.BILINEAR)
        image_np = np.asarray(image).astype(np.uint8)

        out_roi_mask = np.zeros_like(image_np, dtype=np.float32)

        # === TRAINING AUGMENTATIONS, BLUR & MASKS ===
        if self.split == "train":
            if random.random() < 0.5:
                image_np = np.fliplr(image_np).copy()
            
            apply_blur = random.random() < self.suppression_prob
            
            # Optimierung: Maske nur berechnen, wenn Blur oder Strafe aktiv ist
            if apply_blur or l_roi > 0.0:
                roi_mask = get_roi_mask(image_np)
                out_roi_mask = (roi_mask == 0).astype(np.float32)
                if apply_blur:
                    image_np = apply_blur_background_suppression(image_np, roi_mask, self.blur_kernel)

            if random.random() < 0.5:
                image_np = cv2.bitwise_not(image_np)
        else:
            # Für Validation brauchen wir die Masken, falls Loss berechnet wird
            if l_roi > 0.0:
                roi_mask = get_roi_mask(image_np)
                out_roi_mask = (roi_mask == 0).astype(np.float32)

        # Statische Border-Maske
        border_mask = create_border_mask(image_np.shape[0], image_np.shape[1])

        image_pil = Image.fromarray(image_np, mode="L")
        image_tensor = self.transform(image_pil)

        if self.split == "train" and self.noise_std > 0.0:
            noise = torch.randn_like(image_tensor) * self.noise_std
            image_tensor = image_tensor + noise

        # Masken zu Tensors
        border_mask_tensor = torch.from_numpy(border_mask).unsqueeze(0)
        out_roi_mask_tensor = torch.from_numpy(out_roi_mask).unsqueeze(0)

        return image_tensor, label, alpha, l_border, l_roi, border_mask_tensor, out_roi_mask_tensor

# DenseNet Architektur (mit Features für Attention Loss)
class DenseNetWithFeatures(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        weights = models.DenseNet201_Weights.IMAGENET1K_V1
        base_model = models.densenet201(weights=weights)
        self.features = base_model.features
        self.classifier = nn.Linear(base_model.classifier.in_features, num_classes)
        
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        logits = self.classifier(out)
        return logits, features

# Hybrid Loss: Label Smoothing + Guided Attention
class DynamicHybridLoss(nn.Module):
    def __init__(self, weight=None, num_classes=5):
        super().__init__()
        self.weight = weight  
        self.num_classes = num_classes

    def forward(self, logits, features, targets, alphas, l_borders, l_rois, border_masks, out_roi_masks):
        # 1. Smoothed Cross Entropy
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0)
            alphas_unsqueeze = alphas.unsqueeze(1) 
            true_dist = true_dist * (1.0 - alphas_unsqueeze) + alphas_unsqueeze / self.num_classes

        if self.weight is not None:
            weight_expanded = self.weight.unsqueeze(0) 
            loss_ce = - (weight_expanded * true_dist * log_probs).sum(dim=-1)
            norm = (weight_expanded * true_dist).sum(dim=-1).sum()
            loss_ce = loss_ce.sum() / norm
        else:
            loss_ce = - (true_dist * log_probs).sum(dim=-1).mean()

        # 2. Guided Attention Loss (Instanz-basiert)
        attention = torch.mean(torch.abs(features), dim=1, keepdim=True)
        border_masks_small = F.adaptive_avg_pool2d(border_masks, attention.shape[2:])
        out_roi_masks_small = F.adaptive_avg_pool2d(out_roi_masks, attention.shape[2:])
        
        attention_sum = torch.sum(attention, dim=(2, 3)) + 1e-8
        
        penalty_border = torch.sum(attention * border_masks_small, dim=(2, 3)) / attention_sum
        penalty_roi = torch.sum(attention * out_roi_masks_small, dim=(2, 3)) / attention_sum
        
        # Multipliziere jeden Penalty mit dem individuellen Lambda des Bildes
        loss_border = torch.mean(penalty_border * l_borders)
        loss_roi = torch.mean(penalty_roi * l_rois)

        return loss_ce + loss_border + loss_roi

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for images, labels, alphas, l_borders, l_rois, border_masks, out_roi_masks in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        alphas, l_borders, l_rois = alphas.to(device), l_borders.to(device), l_rois.to(device)
        border_masks, out_roi_masks = border_masks.to(device), out_roi_masks.to(device)

        optimizer.zero_grad()
        logits, features = model(images)
        loss = criterion(logits, features, labels, alphas, l_borders, l_rois, border_masks, out_roi_masks)
        
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

    for images, labels, alphas, l_borders, l_rois, border_masks, out_roi_masks in tqdm(dataloader, desc="Validation", leave=False):
        images, labels = images.to(device), labels.to(device)
        alphas, l_borders, l_rois = alphas.to(device), l_borders.to(device), l_rois.to(device)
        border_masks, out_roi_masks = border_masks.to(device), out_roi_masks.to(device)

        logits, features = model(images)
        loss = criterion(logits, features, labels, alphas, l_borders, l_rois, border_masks, out_roi_masks)

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    return total_loss / len(dataloader.dataset), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro"), f1_score(all_labels, all_preds, average="weighted")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="outputs/all_cases_dynamic_roi_scores_train.csv")
    parser.add_argument("--val-csv", default="outputs/all_cases_dynamic_roi_scores_val.csv")
    parser.add_argument("--output-dir", default="outputs/densenet_hybrid_dynamic")
    
    # Optional für Finetuning (wenn man ein vorheriges Modell laden will)
    parser.add_argument("--resume-weights", type=str, default="", help="Pfad zum best_model.pt für Finetuning")

    # Hyperparameter
    parser.add_argument("--max-alpha", type=float, default=0.5)
    parser.add_argument("--lambda-border-base", type=float, default=0.1)
    parser.add_argument("--lambda-roi-base", type=float, default=0.1)
    parser.add_argument("--noise-std", type=float, default=0.05)
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

    train_dataset = KneeOAHybridDataset(PROJECT_ROOT / args.train_csv, split="train", 
                                        max_alpha=args.max_alpha, lambda_border_base=args.lambda_border_base, lambda_roi_base=args.lambda_roi_base,
                                        noise_std=args.noise_std, suppression_prob=args.suppression_prob, blur_kernel=args.blur_kernel)
    
    val_dataset = KneeOAHybridDataset(PROJECT_ROOT / args.val_csv, split="val", 
                                      max_alpha=0.0, lambda_border_base=args.lambda_border_base, lambda_roi_base=args.lambda_roi_base,
                                      noise_std=0.0, suppression_prob=0.0)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = DenseNetWithFeatures(num_classes=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Modell laden, falls übergeben (Finetuning)
    if args.resume_weights:
        print(f"Lade Checkpoint aus: {args.resume_weights}")
        checkpoint = torch.load(PROJECT_ROOT / args.resume_weights, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

    class_counts = train_dataset.data["true_label"].value_counts().sort_index()
    class_weights = torch.tensor((len(train_dataset) / (5 * class_counts)).values, dtype=torch.float32, device=device)

    criterion = DynamicHybridLoss(weight=class_weights, num_classes=5)

    best_val_macro_f1 = -1.0
    history = []

    config = {
        "architecture": "DenseNetWithFeatures",
        "loss": "DynamicHybridLoss (Smoothing + Guided Attention)",
        "dynamic_thresholds": "ROI < 0.40, Border > 0.20",
        "lambda_border_base": args.lambda_border_base,
        "lambda_roi_base": args.lambda_roi_base,
        "max_alpha": args.max_alpha,
        "augmentations": ["HorizontalFlip", "Inversion (p=0.5)", f"GaussianNoise (std={args.noise_std})"],
        "intervention": "dynamic ROI-guided background blur",
        "suppression_probability": args.suppression_prob,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "resumed_from": args.resume_weights,
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