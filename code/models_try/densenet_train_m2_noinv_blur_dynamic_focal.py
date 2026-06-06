#!/usr/bin/env python

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from roi_dynamic import get_roi_mask


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_roi_mask_no_inversion(image_np: np.ndarray) -> np.ndarray:
    """
    Generate dynamic ROI mask without internal inversion correction.

    This keeps the M2 no-inversion condition:
    - model input remains original
    - ROI generation does not invert internally
    """

    import roi_dynamic

    original_is_inverted = roi_dynamic.is_inverted
    roi_dynamic.is_inverted = lambda img: False

    try:
        roi_mask = get_roi_mask(image_np)
    finally:
        roi_dynamic.is_inverted = original_is_inverted

    return roi_mask


def apply_blur_background_suppression(
    image: np.ndarray,
    roi_mask: np.ndarray,
    blur_kernel: int = 21,
) -> np.ndarray:
    """
    Keep ROI unchanged and blur outside-ROI background.
    """

    if blur_kernel % 2 == 0:
        blur_kernel += 1

    roi_mask = (roi_mask > 0).astype(np.float32)

    blurred = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)

    suppressed = (
        image.astype(np.float32) * roi_mask
        + blurred.astype(np.float32) * (1.0 - roi_mask)
    )

    return np.clip(suppressed, 0, 255).astype(np.uint8)


def compute_suspicion_score(
    dynamic_roi_inside_ratio: float,
    border_attention_score: float,
    roi_threshold: float = 0.40,
    border_threshold: float = 0.20,
) -> float:
    """
    Convert ROI/border Grad-CAM scores into a bounded suspicion score in [0, 1].

    Higher score means stronger violation of the dynamic ROI / border rule.
    """

    roi_suspicion = max(0.0, roi_threshold - dynamic_roi_inside_ratio) / roi_threshold
    border_suspicion = max(0.0, border_attention_score - border_threshold) / (
        1.0 - border_threshold
    )

    score = max(roi_suspicion, border_suspicion)
    return float(np.clip(score, 0.0, 1.0))


def load_suspicion_scores(
    suspicious_csv: Path,
    roi_threshold: float,
    border_threshold: float,
) -> dict[str, float]:
    """
    Load suspicious training cases and compute dynamic suspicion scores.

    Non-listed cases receive score 0 inside the dataset.
    """

    if not suspicious_csv.exists():
        raise FileNotFoundError(f"Suspicious CSV not found: {suspicious_csv}")

    df = pd.read_csv(suspicious_csv)

    required_cols = {
        "image_path",
        "dynamic_roi_inside_ratio",
        "border_attention_score",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Suspicious CSV missing required columns: {sorted(missing)}"
        )

    scores = {}

    for _, row in df.iterrows():
        image_path = str(row["image_path"])

        score = compute_suspicion_score(
            dynamic_roi_inside_ratio=float(row["dynamic_roi_inside_ratio"]),
            border_attention_score=float(row["border_attention_score"]),
            roi_threshold=roi_threshold,
            border_threshold=border_threshold,
        )

        scores[image_path] = score

    return scores


class KneeOAM2NoInvBlurDynamicFocalDataset(Dataset):
    """
    M2 no-inversion blur dataset with per-sample suspicion score.

    Returns:
    - image tensor
    - label
    - suspicion_score
    """

    def __init__(
        self,
        csv_path: str | Path,
        split: str,
        suspicion_scores: dict[str, float] | None = None,
        suppression_prob: float = 0.5,
        blur_kernel: int = 21,
        seed: int = 42,
    ):
        self.csv_path = Path(csv_path)
        self.split = split
        self.data = pd.read_csv(self.csv_path)

        self.suspicion_scores = suspicion_scores or {}
        self.suppression_prob = suppression_prob
        self.blur_kernel = blur_kernel
        self.rng = random.Random(seed)

        if "image_path" not in self.data.columns or "label" not in self.data.columns:
            raise ValueError(
                f"{self.csv_path} must contain columns: image_path, label"
            )

        self.final_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        relative_image_path = str(row["image_path"])
        image_path = PROJECT_ROOT / relative_image_path
        label = int(row["label"])

        suspicion_score = float(self.suspicion_scores.get(relative_image_path, 0.0))

        image = Image.open(image_path).convert("L")
        image = image.resize((224, 224), Image.BILINEAR)
        image_np = np.asarray(image).astype(np.uint8)

        if self.split == "train" and self.rng.random() < 0.5:
            image_np = np.fliplr(image_np).copy()

        if self.split == "train" and self.rng.random() < self.suppression_prob:
            roi_mask = get_roi_mask_no_inversion(image_np)

            image_np = apply_blur_background_suppression(
                image=image_np,
                roi_mask=roi_mask,
                blur_kernel=self.blur_kernel,
            )

        image_pil = Image.fromarray(image_np, mode="L")
        image_tensor = self.final_transform(image_pil)

        return image_tensor, label, suspicion_score


class DynamicSuspiciousFocalLoss(nn.Module):
    """
    Weighted dynamic focal loss.

    For each sample:
    gamma_i = base_gamma + suspicion_score_i * (max_gamma - base_gamma)

    suspicious_weight_i = 1 + suspicion_score_i * (max_suspicious_weight - 1)

    loss_i = suspicious_weight_i * (1 - pt_i)^gamma_i * weighted_CE_i
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        base_gamma: float = 1.0,
        max_gamma: float = 2.0,
        max_suspicious_weight: float = 1.5,
    ):
        super().__init__()

        if max_gamma < base_gamma:
            raise ValueError("max_gamma must be >= base_gamma")

        if max_suspicious_weight < 1.0:
            raise ValueError("max_suspicious_weight must be >= 1.0")

        self.register_buffer("class_weights", class_weights)
        self.base_gamma = base_gamma
        self.max_gamma = max_gamma
        self.max_suspicious_weight = max_suspicious_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        suspicion_scores: torch.Tensor,
    ) -> torch.Tensor:
        suspicion_scores = suspicion_scores.to(
            device=logits.device,
            dtype=torch.float32,
        ).clamp(0.0, 1.0)

        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            reduction="none",
        )

        pt = torch.exp(-ce_loss)

        gamma = self.base_gamma + suspicion_scores * (
            self.max_gamma - self.base_gamma
        )

        suspicious_weight = 1.0 + suspicion_scores * (
            self.max_suspicious_weight - 1.0
        )

        focal_factor = (1.0 - pt) ** gamma
        loss = suspicious_weight * focal_factor * ce_loss

        return loss.mean()


def build_densenet201(num_classes: int = 5) -> nn.Module:
    weights = models.DenseNet201_Weights.IMAGENET1K_V1
    model = models.densenet201(weights=weights)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels, suspicion_scores in tqdm(
        dataloader,
        desc="Training",
        leave=False,
    ):
        images = images.to(device)
        labels = labels.to(device)
        suspicion_scores = suspicion_scores.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels, suspicion_scores)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, accuracy, macro_f1


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels, suspicion_scores in tqdm(
        dataloader,
        desc="Validation",
        leave=False,
    ):
        images = images.to(device)
        labels = labels.to(device)
        suspicion_scores = suspicion_scores.to(device, dtype=torch.float32)

        logits = model(images)
        loss = criterion(logits, labels, suspicion_scores)

        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return avg_loss, accuracy, macro_f1, weighted_f1


def main():
    parser = argparse.ArgumentParser(
        description="M2 no-inv blur + dynamic suspiciousness-aware focal loss"
    )

    parser.add_argument("--train-csv", default="data/splits/train.csv")
    parser.add_argument("--val-csv", default="data/splits/val.csv")

    parser.add_argument(
        "--suspicious-csv",
        default=(
            "outputs/xai_region_analysis_dynamic/"
            "densenet_weighted_ce/train_denseblock4_predicted/suspicious_cases.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/m2_noinv_blur_dynamic_focal",
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--suppression-prob", type=float, default=0.5)
    parser.add_argument("--blur-kernel", type=int, default=21)

    parser.add_argument("--roi-threshold", type=float, default=0.40)
    parser.add_argument("--border-threshold", type=float, default=0.20)

    parser.add_argument("--base-gamma", type=float, default=1.0)
    parser.add_argument("--max-gamma", type=float, default=2.0)
    parser.add_argument("--max-suspicious-weight", type=float, default=1.5)

    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    suspicious_csv = PROJECT_ROOT / args.suspicious_csv

    print(f"Using device: {device}")
    print(f"Output dir: {output_dir}")
    print(f"Suspicious CSV: {suspicious_csv}")

    suspicion_scores = load_suspicion_scores(
        suspicious_csv=suspicious_csv,
        roi_threshold=args.roi_threshold,
        border_threshold=args.border_threshold,
    )

    print(f"Loaded suspicious scores: {len(suspicion_scores)} cases")

    if suspicion_scores:
        score_values = np.array(list(suspicion_scores.values()), dtype=np.float32)
        print(
            "Suspicion score stats: "
            f"min={score_values.min():.4f}, "
            f"mean={score_values.mean():.4f}, "
            f"max={score_values.max():.4f}"
        )

    train_dataset = KneeOAM2NoInvBlurDynamicFocalDataset(
        csv_path=PROJECT_ROOT / args.train_csv,
        split="train",
        suspicion_scores=suspicion_scores,
        suppression_prob=args.suppression_prob,
        blur_kernel=args.blur_kernel,
        seed=args.seed,
    )

    val_dataset = KneeOAM2NoInvBlurDynamicFocalDataset(
        csv_path=PROJECT_ROOT / args.val_csv,
        split="val",
        suspicion_scores={},
        suppression_prob=0.0,
        blur_kernel=args.blur_kernel,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    model = build_densenet201(num_classes=5).to(device)

    class_counts = train_dataset.data["label"].value_counts().sort_index()
    class_weights = len(train_dataset) / (5 * class_counts)

    class_weights = torch.tensor(
        class_weights.values,
        dtype=torch.float32,
        device=device,
    )

    print("Class counts:")
    print(class_counts)
    print("Class weights:")
    print(class_weights)

    criterion = DynamicSuspiciousFocalLoss(
        class_weights=class_weights,
        base_gamma=args.base_gamma,
        max_gamma=args.max_gamma,
        max_suspicious_weight=args.max_suspicious_weight,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    config = {
        "architecture": "DenseNet201",
        "loss": "DynamicSuspiciousFocalLoss",
        "weighted_loss": True,
        "intervention": "M2 no-inversion dynamic ROI blur + dynamic focal",
        "suppression_prob": args.suppression_prob,
        "blur_kernel": args.blur_kernel,
        "roi_threshold": args.roi_threshold,
        "border_threshold": args.border_threshold,
        "base_gamma": args.base_gamma,
        "max_gamma": args.max_gamma,
        "max_suspicious_weight": args.max_suspicious_weight,
        "suspicious_csv": args.suspicious_csv,
        "num_suspicious_training_cases": len(suspicion_scores),
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

    best_val_macro_f1 = -1.0
    history = []

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
            "suppression_prob": args.suppression_prob,
            "blur_kernel": args.blur_kernel,
            "base_gamma": args.base_gamma,
            "max_gamma": args.max_gamma,
            "max_suspicious_weight": args.max_suspicious_weight,
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