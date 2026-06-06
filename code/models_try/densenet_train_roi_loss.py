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

# Importiere deinen angepassten Dataset-Loader
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
        # Die CE-Loss profitiert direkt von Label Smoothing (kannst du hier bei Bedarf aktivieren)
        self.ce = nn.CrossEntropyLoss(weight=ce_weights)
        self.lambda_border = lambda_border
        self.lambda_roi = lambda_roi
        
    def forward(self, logits, features, targets, border_masks, out_roi_masks):

        loss_ce = self.ce(logits, targets)
        
        # Attention Map 
        attention = torch.mean(torch.abs(features), dim=1, keepdim=True)
        # scale map
        border_masks_small = F.adaptive_avg_pool2d(border_masks, attention.shape[2:])
        out_roi_masks_small = F.adaptive_avg_pool2d(out_roi_masks, attention.shape[2:])
        
        # penalty
        attention_sum = torch.sum(attention, dim=(2, 3)) + 1e-8
        
        penalty_border = torch.sum(attention * border_masks_small, dim=(2, 3)) / attention_sum
        penalty_roi = torch.sum(attention * out_roi_masks_small, dim=(2, 3)) / attention_sum
        
        # Batch-mean
        loss_border = torch.mean(penalty_border)
        loss_roi = torch.mean(penalty_roi)
        
        # final loss
        total_loss = loss_ce + (self.lambda_border * loss_border) + (self.lambda_roi * loss_roi)
        
        return total_loss


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels, border_masks, out_roi_masks in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        if border_masks.ndim == 3:
            border_masks = border_masks.unsqueeze(1)
        if out_roi_masks.ndim == 3:
            out_roi_masks = out_roi_masks.unsqueeze(1)
            
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
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, accuracy, macro_f1


# valid
@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels, border_masks, out_roi_masks in tqdm(dataloader, desc="Validation", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        if border_masks.ndim == 3:
            border_masks = border_masks.unsqueeze(1)
        if out_roi_masks.ndim == 3:
            out_roi_masks = out_roi_masks.unsqueeze(1)
            
        border_masks = border_masks.to(device, dtype=torch.float32)
        out_roi_masks = out_roi_masks.to(device, dtype=torch.float32)

        logits, features = model(images)
        loss = criterion(logits, features, labels, border_masks, out_roi_masks)

        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    return avg_loss, accuracy, macro_f1, weighted_f1



def main():
    parser = argparse.ArgumentParser(description="DenseNet Training with Guided Attention Loss")
    parser.add_argument("--train-csv", default="data/splits/train.csv")
    parser.add_argument("--val-csv", default="data/splits/val.csv")
    parser.add_argument("--output-dir", default="outputs/densenet_roi_loss")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    # Hyperparameter for loss penalty -----------------------------------------------------------------------------
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    model = build_densenet201(num_classes=5)
    model = model.to(device)

    class_counts = train_dataset.data["label"].value_counts().sort_index()
    class_weights = len(train_dataset) / (5 * class_counts)
    class_weights = torch.tensor(
        class_weights.values,
        dtype=torch.float32,
        device=device
    )

    print("Class counts:")
    print(class_counts)
    print("Class weights:")
    print(class_weights)

    # ----> Custom Loss Function
    criterion = GuidedAttentionLoss(
        ce_weights=class_weights, 
        lambda_border=args.lambda_border, 
        lambda_roi=args.lambda_roi
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_macro_f1 = -1.0
    history = []

    config = {
        "architecture": "DenseNet201WithFeatures",
        "loss": "GuidedAttentionLoss",
        "weighted_loss": True,
        "lambda_border": args.lambda_border,
        "lambda_roi": args.lambda_roi,
        "num_classes": 5,
        "pretrained": True,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "checkpoint_selection": "best validation macro-F1",
    }

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc, train_macro_f1 = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc, val_macro_f1, val_weighted_f1 = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_macro_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
            "val_weighted_f1": val_weighted_f1,
            "learning_rate": args.lr,
        }

        history.append(row)

        print(
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"train_macro_f1={train_macro_f1:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} "
            f"val_weighted_f1={val_weighted_f1:.4f}"
        )

        pd.DataFrame(history).to_csv(
            output_dir / "training_history.csv",
            index=False,
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_macro_f1": best_val_macro_f1,
                "config": config,
            }

            torch.save(checkpoint, output_dir / "best_model.pt")
            print(f"Saved new best model with val_macro_f1={best_val_macro_f1:.4f}")

    print("\nTraining finished.")
    print(f"Best validation macro-F1: {best_val_macro_f1:.4f}")
    print(f"Outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()