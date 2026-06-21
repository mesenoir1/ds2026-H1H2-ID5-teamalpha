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
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')
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

class KneeOAMasterDataset(Dataset):
    def __init__(self, csv_path: str | Path, split: str, max_alpha: float = 0.5, 
                 lambda_border_base: float = 0.1, lambda_roi_base: float = 0.1,
                 roi_threshold: float = 0.40, border_threshold: float = 0.20,
                 noise_std: float = 0.05, suppression_prob: float = 0.5, blur_kernel: int = 21,
                 flip_prob: float = 0.5, invert_prob: float = 0.5):
        self.csv_path = Path(csv_path)
        self.split = split
        self.max_alpha = max_alpha
        self.noise_std = noise_std
        self.suppression_prob = suppression_prob
        self.blur_kernel = blur_kernel
        self.flip_prob = flip_prob
        self.invert_prob = invert_prob
        
        self.roi_threshold = roi_threshold       
        self.border_threshold = border_threshold
        
        self.data = pd.read_csv(self.csv_path)

        # Find Joint Space
        if "image_path" not in self.data.columns:
            if "file_name" in self.data.columns:
                self.data.rename(columns={"file_name": "image_path"}, inplace=True)
            else:
                raise ValueError(f"{self.csv_path} contains neither 'image_path' nor 'file_name'.")

        # Find Proper Lable Column
        if "label" in self.data.columns:
            self.label_col = "label"
        elif "true_label" in self.data.columns:
            self.label_col = "true_label"
        elif "KL_Grade" in self.data.columns:
            self.label_col = "KL_Grade"
        else:
            raise ValueError(f"{self.csv_path} No valid Label Column.")

        # Find Alphas and Lambdas
        self.alphas, self.l_borders, self.l_rois = [], [], []
        
        has_dynamic_scores = ("dynamic_roi_inside_ratio" in self.data.columns and 
                              "border_attention_score" in self.data.columns)

        for _, row in self.data.iterrows():
            alpha, l_border, l_roi = 0.0, 0.0, 0.0
            
            if has_dynamic_scores:
                roi = float(row["dynamic_roi_inside_ratio"])
                border = float(row["border_attention_score"])
                
                if not np.isnan(roi) and not np.isnan(border):

                    if self.roi_threshold > 0.0:
                        roi_penalty = max(0.0, (self.roi_threshold - roi) / self.roi_threshold)
                    else:
                        roi_penalty = 0.0

                    if self.border_threshold < 1.0:
                        border_penalty = max(0.0, (border - self.border_threshold) / (1.0 - self.border_threshold))
                    else:
                        border_penalty = 0.0
                        
                    alpha = min(1.0, roi_penalty + border_penalty) * self.max_alpha
                    
                    if roi < self.roi_threshold: 
                        l_roi = lambda_roi_base
                    if border > self.border_threshold: 
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
        label = int(row[self.label_col]) 
        alpha = float(self.alphas[idx])
        l_border = float(self.l_borders[idx])
        l_roi = float(self.l_rois[idx])

        image = Image.open(image_path).convert("L").resize((224, 224), Image.BILINEAR)
        image_np = np.asarray(image).astype(np.uint8)

        out_roi_mask = np.zeros_like(image_np, dtype=np.float32)

        # === DATA AUGMENTATION & MASKS ===
        if self.split == "train":
            if random.random() < self.flip_prob:
                image_np = np.fliplr(image_np).copy()
            
            apply_blur = random.random() < self.suppression_prob
            if apply_blur or l_roi > 0.0:
                roi_mask = get_roi_mask(image_np)
                out_roi_mask = (roi_mask == 0).astype(np.float32)
                if apply_blur:
                    image_np = apply_blur_background_suppression(image_np, roi_mask, self.blur_kernel)

            if random.random() < self.invert_prob:
                image_np = cv2.bitwise_not(image_np)
        else:
            if l_roi > 0.0:
                roi_mask = get_roi_mask(image_np)
                out_roi_mask = (roi_mask == 0).astype(np.float32)

        border_mask = create_border_mask(image_np.shape[0], image_np.shape[1])

        image_pil = Image.fromarray(image_np, mode="L")
        image_tensor = self.transform(image_pil)

        if self.split == "train" and self.noise_std > 0.0:
            noise = torch.randn_like(image_tensor) * self.noise_std
            image_tensor = image_tensor + noise

        border_mask_tensor = torch.from_numpy(border_mask).unsqueeze(0)
        out_roi_mask_tensor = torch.from_numpy(out_roi_mask).unsqueeze(0)

        return image_tensor, label, alpha, l_border, l_roi, border_mask_tensor, out_roi_mask_tensor

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

class DynamicHybridLoss(nn.Module):
    def __init__(self, weight=None, num_classes=5):
        super().__init__()
        self.weight = weight  
        self.num_classes = num_classes

    def forward(self, logits, features, targets, alphas, l_borders, l_rois, border_masks, out_roi_masks):
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

        # Skip if Lambdas eqaul 0
        if torch.sum(l_borders) == 0 and torch.sum(l_rois) == 0:
            return loss_ce

        attention = torch.mean(torch.abs(features), dim=1, keepdim=True)
        border_masks_small = F.adaptive_avg_pool2d(border_masks, attention.shape[2:])
        out_roi_masks_small = F.adaptive_avg_pool2d(out_roi_masks, attention.shape[2:])
        
        attention_sum = torch.sum(attention, dim=(2, 3)) + 1e-8
        
        penalty_border = torch.sum(attention * border_masks_small, dim=(2, 3)) / attention_sum
        penalty_roi = torch.sum(attention * out_roi_masks_small, dim=(2, 3)) / attention_sum
        
        l_borders = l_borders.unsqueeze(1) # fix (otherwise the loss is weakened)
        l_rois = l_rois.unsqueeze(1)
        
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
    parser.add_argument("--output-dir", required=True, help="Outputfolder")
    parser.add_argument("--resume-weights", type=str, default="")

    # Dynamic Smoothing
    parser.add_argument("--max-alpha", type=float, default=0.5)
    
    # Custom Loss
    parser.add_argument("--lambda-border-base", type=float, default=0.1)
    parser.add_argument("--lambda-roi-base", type=float, default=0.1)
    parser.add_argument("--roi-threshold", type=float, default=0.40, help="ROI < threshold")
    parser.add_argument("--border-threshold", type=float, default=0.20, help="Border > threshold")
    
    # Augmentations (Alle mit 0.0 deaktivierbar)
    parser.add_argument("--flip-prob", type=float, default=0.5, help="horizontal Flip")
    parser.add_argument("--invert-prob", type=float, default=0.5, help="Inversion")
    parser.add_argument("--noise-std", type=float, default=0.05, help="Gaussian Noise (0.0 off)")
    
    # Blur
    parser.add_argument("--suppression-prob", type=float, default=0.5, help="ROI-Blur")
    parser.add_argument("--blur-kernel", type=int, default=21)

    # General
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 Regularisation for AdamW")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = KneeOAMasterDataset(
        PROJECT_ROOT / args.train_csv, split="train", 
        max_alpha=args.max_alpha, lambda_border_base=args.lambda_border_base, lambda_roi_base=args.lambda_roi_base,
        roi_threshold=args.roi_threshold, border_threshold=args.border_threshold, 
        noise_std=args.noise_std, suppression_prob=args.suppression_prob, blur_kernel=args.blur_kernel,
        flip_prob=args.flip_prob, invert_prob=args.invert_prob
    )
    
    val_dataset = KneeOAMasterDataset(
        PROJECT_ROOT / args.val_csv, split="val", 
        max_alpha=0.0, lambda_border_base=args.lambda_border_base, lambda_roi_base=args.lambda_roi_base,
        roi_threshold=args.roi_threshold, border_threshold=args.border_threshold, 
        noise_std=0.0, suppression_prob=0.0, flip_prob=0.0, invert_prob=0.0
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = DenseNetWithFeatures(num_classes=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    if args.resume_weights:
        print(f"Lade Checkpoint aus: {args.resume_weights}")
        checkpoint = torch.load(PROJECT_ROOT / args.resume_weights, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

    class_counts = train_dataset.data[train_dataset.label_col].value_counts().sort_index()
    class_weights = torch.tensor((len(train_dataset) / (5 * class_counts)).values, dtype=torch.float32, device=device)

    criterion = DynamicHybridLoss(weight=class_weights, num_classes=5)

    best_val_macro_f1 = -1.0
    history = []

    config = {
        "architecture": "DenseNetWithFeatures",
        "loss": "DynamicHybridLoss",
        "lambda_border_base": args.lambda_border_base,
        "lambda_roi_base": args.lambda_roi_base,
        "max_alpha": args.max_alpha,
        "roi_threshold": args.roi_threshold,
        "border_threshold": args.border_threshold,
        "augmentations": {
            "flip_prob": args.flip_prob,
            "invert_prob": args.invert_prob,
            "noise_std": args.noise_std
        },
        "intervention_blur": {
            "suppression_probability": args.suppression_prob,
            "blur_kernel": args.blur_kernel
        },
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "resumed_from": args.resume_weights,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        # train
        train_loss, train_acc, train_macro_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # eval val_macro_f1 
        val_loss, val_acc, val_macro_f1, val_weighted_f1 = evaluate(model, val_loader, criterion, device)

        # SCHEDULER
        scheduler.step(val_macro_f1)

        # lr
        current_lr = optimizer.param_groups[0]['lr']

        # history
        row = {
            "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc, "train_macro_f1": train_macro_f1,
            "val_loss": val_loss, "val_accuracy": val_acc, "val_macro_f1": val_macro_f1, "val_weighted_f1": val_weighted_f1,
            "learning_rate": current_lr
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
        
        # terminal
        print(f"LR={current_lr:.2e} | train_loss={train_loss:.4f} train_macro_f1={train_macro_f1:.4f} | val_loss={val_loss:.4f} val_macro_f1={val_macro_f1:.4f}")

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