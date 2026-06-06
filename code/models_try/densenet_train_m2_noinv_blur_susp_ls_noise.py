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

IMAGENET_MEAN_TENSOR = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
IMAGENET_STD_TENSOR = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)


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
    Keep ROI unchanged and blur everything outside ROI.
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


def load_suspicious_paths(suspicious_csv: Path) -> set[str]:
    """
    Load suspicious training image paths.

    Only these cases receive label smoothing.
    """

    if not suspicious_csv.exists():
        raise FileNotFoundError(f"Suspicious CSV not found: {suspicious_csv}")

    df = pd.read_csv(suspicious_csv)

    if "image_path" not in df.columns:
        raise ValueError(f"{suspicious_csv} must contain image_path column")

    suspicious_paths = set(df["image_path"].astype(str).tolist())

    return suspicious_paths


class KneeOAM2NoInvBlurSuspLabelSmoothNoiseDataset(Dataset):
    """
    M2 no-inversion blur dataset with suspicious-case label smoothing flag.

    Returns:
    - image tensor
    - label
    - smoothing value

    Training augmentations:
    - horizontal flip p=0.5
    - no-inversion dynamic ROI blur p=suppression_prob
    - Gaussian noise is applied later on tensor batches
    """

    def __init__(
        self,
        csv_path: str | Path,
        split: str,
        suspicious_paths: set[str] | None = None,
        suspicious_label_smoothing: float = 0.10,
        suppression_prob: float = 0.5,
        blur_kernel: int = 21,
        seed: int = 42,
    ):
        self.csv_path = Path(csv_path)
        self.split = split
        self.data = pd.read_csv(self.csv_path)

        self.suspicious_paths = suspicious_paths or set()
        self.suspicious_label_smoothing = suspicious_label_smoothing
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

        smoothing = (
            self.suspicious_label_smoothing
            if relative_image_path in self.suspicious_paths
            else 0.0
        )

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

        return image_tensor, label, float(smoothing)


class WeightedSoftLabelCrossEntropy(nn.Module):
    """
    Weighted cross-entropy with per-sample label smoothing.

    For smoothing eps:
    - true class gets 1 - eps + eps / num_classes
    - all classes get eps / num_classes
    """

    def __init__(self, class_weights: torch.Tensor, num_classes: int = 5):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.num_classes = num_classes

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        smoothing_values: torch.Tensor,
    ) -> torch.Tensor:
        smoothing_values = smoothing_values.to(
            device=logits.device,
            dtype=torch.float32,
        ).clamp(0.0, 1.0)

        log_probs = F.log_softmax(logits, dim=1)

        batch_size = logits.size(0)

        soft_targets = torch.zeros(
            batch_size,
            self.num_classes,
            device=logits.device,
            dtype=torch.float32,
        )

        base = smoothing_values / self.num_classes
        soft_targets += base.unsqueeze(1)

        true_class_values = 1.0 - smoothing_values + base
        soft_targets.scatter_(1, targets.view(-1, 1), true_class_values.view(-1, 1))

        weighted_log_probs = log_probs * self.class_weights.view(1, -1)

        loss = -(soft_targets * weighted_log_probs).sum(dim=1)

        return loss.mean()


def add_gaussian_noise(
    images: torch.Tensor,
    noise_std: float,
    noise_prob: float,
) -> torch.Tensor:
    """
    Add Gaussian noise to normalized image tensors.

    Noise is applied per image with probability noise_prob.
    """

    if noise_std <= 0 or noise_prob <= 0:
        return images

    device = images.device
    batch_size = images.size(0)

    apply_mask = torch.rand(batch_size, device=device) < noise_prob

    if not apply_mask.any():
        return images

    noisy_images = images.clone()
    noise = torch.randn_like(noisy_images[apply_mask]) * noise_std
    noisy_images[apply_mask] = noisy_images[apply_mask] + noise

    mean = IMAGENET_MEAN_TENSOR.to(device)
    std = IMAGENET_STD_TENSOR.to(device)

    min_val = (0.0 - mean) / std
    max_val = (1.0 - mean) / std

    noisy_images = torch.maximum(noisy_images, min_val)
    noisy_images = torch.minimum(noisy_images, max_val)

    return noisy_images


def build_densenet201(num_classes: int = 5) -> nn.Module:
    weights = models.DenseNet201_Weights.IMAGENET1K_V1
    model = models.densenet201(weights=weights)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    noise_std: float,
    noise_prob: float,
):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels, smoothing_values in tqdm(
        dataloader,
        desc="Training",
        leave=False,
    ):
        images = images.to(device)
        labels = labels.to(device)
        smoothing_values = smoothing_values.to(device, dtype=torch.float32)

        images = add_gaussian_noise(
            images=images,
            noise_std=noise_std,
            noise_prob=noise_prob,
        )

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels, smoothing_values)

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

    for images, labels, smoothing_values in tqdm(
        dataloader,
        desc="Validation",
        leave=False,
    ):
        images = images.to(device)
        labels = labels.to(device)
        smoothing_values = smoothing_values.to(device, dtype=torch.float32)

        logits = model(images)
        loss = criterion(logits, labels, smoothing_values)

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
        description=(
            "M2 no-inv blur + suspicious-case label smoothing "
            "+ mild Gaussian noise"
        )
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
        default="outputs/m2_noinv_blur_susp_ls010_noise002",
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--suppression-prob", type=float, default=0.5)
    parser.add_argument("--blur-kernel", type=int, default=21)

    parser.add_argument("--suspicious-label-smoothing", type=float, default=0.10)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--noise-prob", type=float, default=0.5)

    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    suspicious_csv = PROJECT_ROOT / args.suspicious_csv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Output dir: {output_dir}")
    print(f"Suspicious CSV: {suspicious_csv}")

    suspicious_paths = load_suspicious_paths(suspicious_csv)
    print(f"Loaded suspicious training cases: {len(suspicious_paths)}")

    train_dataset = KneeOAM2NoInvBlurSuspLabelSmoothNoiseDataset(
        csv_path=PROJECT_ROOT / args.train_csv,
        split="train",
        suspicious_paths=suspicious_paths,
        suspicious_label_smoothing=args.suspicious_label_smoothing,
        suppression_prob=args.suppression_prob,
        blur_kernel=args.blur_kernel,
        seed=args.seed,
    )

    val_dataset = KneeOAM2NoInvBlurSuspLabelSmoothNoiseDataset(
        csv_path=PROJECT_ROOT / args.val_csv,
        split="val",
        suspicious_paths=set(),
        suspicious_label_smoothing=0.0,
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

    criterion = WeightedSoftLabelCrossEntropy(
        class_weights=class_weights,
        num_classes=5,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    config = {
        "architecture": "DenseNet201",
        "loss": "WeightedSoftLabelCrossEntropy",
        "weighted_loss": True,
        "intervention": (
            "M2 no-inversion dynamic ROI blur + suspicious-case label smoothing "
            "+ Gaussian noise"
        ),
        "suppression_prob": args.suppression_prob,
        "blur_kernel": args.blur_kernel,
        "suspicious_label_smoothing": args.suspicious_label_smoothing,
        "noise_std": args.noise_std,
        "noise_prob": args.noise_prob,
        "suspicious_csv": args.suspicious_csv,
        "num_suspicious_training_cases": len(suspicious_paths),
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
            noise_std=args.noise_std,
            noise_prob=args.noise_prob,
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
            "suspicious_label_smoothing": args.suspicious_label_smoothing,
            "noise_std": args.noise_std,
            "noise_prob": args.noise_prob,
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