import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from densenet_dataset_roi import KneeOADataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def build_densenet201(num_classes: int = 5) -> nn.Module:
    return DenseNetWithFeatures(num_classes)

# LOSS
class GuidedAttentionLoss(nn.Module):
    def __init__(self, ce_weights, lambda_border=0.1, lambda_roi=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weights)
        self.lambda_border = lambda_border
        self.lambda_roi = lambda_roi
        
    def forward(self, logits, features, targets, border_masks, out_roi_masks):
        loss_ce = self.ce(logits, targets)
        
        attention = torch.mean(torch.abs(features), dim=1, keepdim=True)
        border_masks_small = F.adaptive_avg_pool2d(border_masks, attention.shape[2:])
        out_roi_masks_small = F.adaptive_avg_pool2d(out_roi_masks, attention.shape[2:])
        
        attention_sum = torch.sum(attention, dim=(2, 3)) + 1e-8
        
        penalty_border = torch.sum(attention * border_masks_small, dim=(2, 3)) / attention_sum
        penalty_roi = torch.sum(attention * out_roi_masks_small, dim=(2, 3)) / attention_sum
        
        loss_border = torch.mean(penalty_border)
        loss_roi = torch.mean(penalty_roi)
        
        total_loss = loss_ce + (self.lambda_border * loss_border) + (self.lambda_roi * loss_roi)
        return total_loss

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for images, labels, border_masks, out_roi_masks in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        if border_masks.ndim == 3: border_masks = border_masks.unsqueeze(1)
        if out_roi_masks.ndim == 3: out_roi_masks = out_roi_masks.unsqueeze(1)
            
        border_masks = border_masks.to(device, dtype=torch.float32)
        out_roi_masks = out_roi_masks.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        logits, features = model(images)
        loss = criterion(logits, features, labels, border_masks, out_roi_masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro")

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for images, labels, border_masks, out_roi_masks in tqdm(dataloader, desc="Validation", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        if border_masks.ndim == 3: border_masks = border_masks.unsqueeze(1)
        if out_roi_masks.ndim == 3: out_roi_masks = out_roi_masks.unsqueeze(1)
            
        border_masks = border_masks.to(device, dtype=torch.float32)
        out_roi_masks = out_roi_masks.to(device, dtype=torch.float32)

        logits, features = model(images)
        loss = criterion(logits, features, labels, border_masks, out_roi_masks)

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro"), f1_score(all_labels, all_preds, average="weighted")

def main():
    parser = argparse.ArgumentParser(description="DenseNet Continuation with Guided Attention Loss")
    parser.add_argument("--resume-weights", required=True, type=str, help="Pfad zum alten best_model.pt")
    parser.add_argument("--train-csv", default="data/splits/train.csv")
    parser.add_argument("--val-csv", default="data/splits/val.csv")
    parser.add_argument("--output-dir", default="outputs/densenet_roi_loss_continued")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5) # Standardmäßig etwas niedriger fürs Finetuning
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-border", type=float, default=0.1)
    parser.add_argument("--lambda-roi", type=float, default=0.1)
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = KneeOADataset(PROJECT_ROOT / args.train_csv, split="train")
    val_dataset = KneeOADataset(PROJECT_ROOT / args.val_csv, split="val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_densenet201(num_classes=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---------------------------------------------------------
    # CHECKPOINT LADEN (Überschreibt nichts!)
    # ---------------------------------------------------------
    print(f"Lade bestehenden Checkpoint aus: {args.resume_weights}")
    checkpoint = torch.load(PROJECT_ROOT / args.resume_weights, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # WICHTIG: Die neue (meist niedrigere) Lernrate für das Fine-Tuning erzwingen
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
    print(f"Modell und Optimierer erfolgreich geladen. Starte mit Lernrate {args.lr}")
    # ---------------------------------------------------------

    class_counts = train_dataset.data["label"].value_counts().sort_index()
    class_weights = torch.tensor((len(train_dataset) / (5 * class_counts)).values, dtype=torch.float32, device=device)

    criterion = GuidedAttentionLoss(ce_weights=class_weights, lambda_border=args.lambda_border, lambda_roi=args.lambda_roi)

    best_val_macro_f1 = -1.0
    history = []

    config = {
        "architecture": "DenseNet201WithFeatures",
        "loss": "GuidedAttentionLoss",
        "resumed_from": args.resume_weights,
        "lambda_border": args.lambda_border,
        "lambda_roi": args.lambda_roi,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
    }

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    for epoch in range(1, args.epochs + 1):
        print(f"\nFortsetzung - Epoche {epoch}/{args.epochs}")

        train_loss, train_acc, train_macro_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_macro_f1, val_weighted_f1 = evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss, "train_accuracy": train_acc, "train_macro_f1": train_macro_f1,
            "val_loss": val_loss, "val_accuracy": val_acc, "val_macro_f1": val_macro_f1, "val_weighted_f1": val_weighted_f1,
            "learning_rate": args.lr,
        }
        history.append(row)

        print(f"train_loss={train_loss:.4f} train_macro_f1={train_macro_f1:.4f} | val_loss={val_loss:.4f} val_macro_f1={val_macro_f1:.4f}")

        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

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
            print(f"Saved new best model with val_macro_f1={best_val_macro_f1:.4f}")

    print("\nTraining finished.")
    print(f"Outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()