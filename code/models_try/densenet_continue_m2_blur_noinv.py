import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from densenet_m2_dataset import KneeOAM2Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_densenet201(num_classes: int = 5) -> nn.Module:
    weights = models.DenseNet201_Weights.IMAGENET1K_V1
    model = models.densenet201(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

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

    for images, labels in tqdm(dataloader, desc="Validation", leave=False):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro"), f1_score(all_labels, all_preds, average="weighted")

def main():
    parser = argparse.ArgumentParser()
    
    # Neues Argument für das Weitertrainieren
    parser.add_argument("--resume-weights", required=True, type=str, help="Pfad zum alten best_model.pt")

    parser.add_argument("--train-csv", default="data/splits/train.csv")
    parser.add_argument("--val-csv", default="data/splits/val.csv")
    parser.add_argument("--output-dir", default="outputs/m2_noinv_blur_continued")

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5) # Standardmäßig etwas niedriger fürs Finetuning
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--suppression-prob", type=float, default=0.75)
    parser.add_argument("--blur-kernel", type=int, default=21)

    args = parser.parse_args()
    set_seed(args.seed)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = KneeOAM2Dataset(
        csv_path=PROJECT_ROOT / args.train_csv, split="train",
        suppression_prob=args.suppression_prob, use_inversion_for_roi=False,
        blur_kernel=args.blur_kernel, seed=args.seed,
    )

    val_dataset = KneeOAM2Dataset(
        csv_path=PROJECT_ROOT / args.val_csv, split="val",
        suppression_prob=0.0, use_inversion_for_roi=False,
        blur_kernel=args.blur_kernel, seed=args.seed,
    )

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

    # WICHTIG: Die neue Lernrate für das Fine-Tuning erzwingen
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
    print(f"Modell und Optimierer erfolgreich geladen. Starte mit Lernrate {args.lr}")
    # ---------------------------------------------------------

    class_counts = train_dataset.data["label"].value_counts().sort_index()
    class_weights = len(train_dataset) / (5 * class_counts)
    class_weights = torch.tensor(class_weights.values, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_macro_f1 = -1.0
    history = []

    config = {
        "model_id": "M2a_Continued",
        "resumed_from": args.resume_weights,
        "architecture": "DenseNet201",
        "loss": "WeightedCrossEntropyLoss",
        "weighted_loss": True,
        "num_classes": 5,
        "pretrained": True,
        "intervention": "dynamic ROI-guided background suppression",
        "suppression_mode": "blur",
        "roi_inversion_for_mask_generation": False,
        "model_input_inverted": False,
        "suppression_probability_train": args.suppression_prob,
        "blur_kernel": args.blur_kernel,
        "validation_suppression": False,
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
            "suppression_probability": args.suppression_prob,
            "suppression_mode": "blur",
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