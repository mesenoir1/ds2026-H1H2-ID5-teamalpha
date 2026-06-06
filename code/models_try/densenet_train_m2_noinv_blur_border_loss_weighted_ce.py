#!/usr/bin/env python

"""
DenseNet201 training with:
- weighted cross-entropy
- M2 no-inversion ROI-guided blur background suppression
- border attention penalty

Purpose:
This combines the best input-level intervention so far
(no-inversion blur, suppression probability 0.5)
with a lightweight border-only feature penalty.

Important:
- Model input is never inverted.
- ROI mask generation does not use inversion correction.
- Border penalty is applied to internal DenseNet feature activations.
- Validation uses original images: no blur suppression.
"""

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
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from roi_dynamic import find_joint_space_y, suppress_metal_artifacts


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


def get_roi_mask_noinv(image_input) -> np.ndarray:
    """
    Dynamic ROI mask generation without inversion correction.

    This intentionally does not call get_roi_mask(), because get_roi_mask()
    may invert internally. Here we want a strict no-inversion variant.
    """

    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_input}")
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3:
            img = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        else:
            img = image_input.copy()
    else:
        raise TypeError("image_input must be a path or numpy array")

    img = img.astype(np.uint8)
    h, w = img.shape

    img_for_roi = suppress_metal_artifacts(img)
    line_y = find_joint_space_y(img_for_roi)

    y_top = max(0, int(line_y - 0.22 * h))
    y_bottom = min(h, int(line_y + 0.22 * h))
    x_left = int(0.10 * w)
    x_right = int(0.90 * w)

    search_window = np.zeros_like(img_for_roi, dtype=np.uint8)
    search_window[y_top:y_bottom, x_left:x_right] = 1

    processed = cv2.normalize(
        img_for_roi,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    processed = cv2.equalizeHist(processed)

    _, mask = cv2.threshold(processed, 85, 255, cv2.THRESH_BINARY)
    mask = (mask * search_window).astype(np.uint8)

    r = max(1, int(h * 0.04))
    d = 2 * r + 1
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.01 * h * w:
            continue

        x, y, cw, ch = cv2.boundingRect(contour)
        contour_center_y = y + ch / 2

        if y_top <= contour_center_y <= y_bottom:
            valid_contours.append(contour)

    if valid_contours:
        largest_contour = max(valid_contours, key=cv2.contourArea)
        temp_mask = np.zeros_like(mask)
        cv2.drawContours(temp_mask, [largest_contour], -1, 255, -1)
        mask = temp_mask
    else:
        mask = (search_window * 255).astype(np.uint8)

    kw = max(1, int(w * 0.10))
    kh = max(1, int(h * 0.20))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    final_y_top = max(0, int(line_y - 0.30 * h))
    final_y_bottom = min(h, int(line_y + 0.20 * h))

    mask[:final_y_top, :] = 0
    mask[final_y_bottom:, :] = 0

    return np.where(mask > 0, 1, 0).astype(np.uint8)


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


def make_border_mask(h: int, w: int, border_frac: float = 0.08) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)

    by = max(1, int(h * border_frac))
    bx = max(1, int(w * border_frac))

    mask[:by, :] = 1
    mask[-by:, :] = 1
    mask[:, :bx] = 1
    mask[:, -bx:] = 1

    return mask


class KneeOAM2NoInvBlurBorderLossDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        split: str,
        suppression_prob: float = 0.5,
        blur_kernel: int = 21,
        border_frac: float = 0.08,
        seed: int = 42,
    ):
        self.csv_path = Path(csv_path)
        self.split = split
        self.data = pd.read_csv(self.csv_path)

        self.suppression_prob = suppression_prob
        self.blur_kernel = blur_kernel
        self.border_frac = border_frac
        self.rng = random.Random(seed)

        if "image_path" not in self.data.columns or "label" not in self.data.columns:
            raise ValueError(f"{self.csv_path} must contain image_path,label columns")

        self.normalize = transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_path = PROJECT_ROOT / row["image_path"]
        label = int(row["label"])

        image = Image.open(image_path).convert("L")
        image = image.resize((224, 224), Image.BILINEAR)

        image_np = np.asarray(image).astype(np.uint8)

        if self.split == "train" and self.rng.random() < 0.5:
            image_np = np.fliplr(image_np).copy()

        if self.split == "train" and self.rng.random() < self.suppression_prob:
            roi_mask = get_roi_mask_noinv(image_np)

            image_np = apply_blur_background_suppression(
                image=image_np,
                roi_mask=roi_mask,
                blur_kernel=self.blur_kernel,
            )

        h, w = image_np.shape
        border_mask = make_border_mask(h, w, border_frac=self.border_frac)

        image_pil = Image.fromarray(image_np, mode="L")
        image_tensor = TF.to_tensor(image_pil)
        image_tensor = image_tensor.repeat(3, 1, 1)
        image_tensor = self.normalize(image_tensor)

        label_tensor = torch.tensor(label, dtype=torch.long)
        border_mask_tensor = torch.from_numpy(border_mask).float()

        return image_tensor, label_tensor, border_mask_tensor


class DenseNetWithFeatures(nn.Module):
    """
    DenseNet201 that returns logits and final convolutional feature maps.
    """

    def __init__(self, num_classes: int = 5):
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


class WeightedCEWithBorderPenalty(nn.Module):
    """
    Loss = weighted CE + lambda_border * border feature-attention penalty.

    The border penalty is computed from final DenseNet feature activations.
    It is not Grad-CAM. It is a coarse feature-activation regularizer.
    """

    def __init__(self, ce_weights, lambda_border: float = 0.05):
        super().__init__()

        self.ce = nn.CrossEntropyLoss(weight=ce_weights)
        self.lambda_border = lambda_border

    def forward(self, logits, features, targets, border_masks):
        loss_ce = self.ce(logits, targets)

        attention = torch.mean(torch.abs(features), dim=1, keepdim=True)

        if border_masks.ndim == 3:
            border_masks = border_masks.unsqueeze(1)

        border_masks_small = F.adaptive_avg_pool2d(
            border_masks,
            attention.shape[2:],
        )

        attention_sum = torch.sum(attention, dim=(2, 3)) + 1e-8

        penalty_border = (
            torch.sum(attention * border_masks_small, dim=(2, 3))
            / attention_sum
        )

        loss_border = torch.mean(penalty_border)

        total_loss = loss_ce + self.lambda_border * loss_border

        return total_loss, loss_ce.detach(), loss_border.detach()


def build_model(num_classes: int = 5) -> nn.Module:
    return DenseNetWithFeatures(num_classes=num_classes)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_border = 0.0
    all_preds = []
    all_labels = []

    for images, labels, border_masks in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        border_masks = border_masks.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        logits, features = model(images)

        loss, loss_ce, loss_border = criterion(
            logits=logits,
            features=features,
            targets=labels,
            border_masks=border_masks,
        )

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_ce += float(loss_ce.item()) * batch_size
        total_border += float(loss_border.item()) * batch_size

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    n = len(dataloader.dataset)

    avg_loss = total_loss / n
    avg_ce = total_ce / n
    avg_border = total_border / n

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, avg_ce, avg_border, accuracy, macro_f1


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_border = 0.0
    all_preds = []
    all_labels = []

    for images, labels, border_masks in tqdm(dataloader, desc="Validation", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        border_masks = border_masks.to(device, dtype=torch.float32)

        logits, features = model(images)

        loss, loss_ce, loss_border = criterion(
            logits=logits,
            features=features,
            targets=labels,
            border_masks=border_masks,
        )

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_ce += float(loss_ce.item()) * batch_size
        total_border += float(loss_border.item()) * batch_size

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    n = len(dataloader.dataset)

    avg_loss = total_loss / n
    avg_ce = total_ce / n
    avg_border = total_border / n

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    return avg_loss, avg_ce, avg_border, accuracy, macro_f1, weighted_f1


def main():
    parser = argparse.ArgumentParser(
        description="M2 no-inversion blur with border feature penalty"
    )

    parser.add_argument("--train-csv", default="data/splits/train.csv")
    parser.add_argument("--val-csv", default="data/splits/val.csv")
    parser.add_argument(
        "--output-dir",
        default="outputs/m2_noinv_blur_border_loss",
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--suppression-prob", type=float, default=0.5)
    parser.add_argument("--blur-kernel", type=int, default=21)
    parser.add_argument("--border-frac", type=float, default=0.08)
    parser.add_argument("--lambda-border", type=float, default=0.05)

    parser.add_argument(
        "--safe-dataloader",
        action="store_true",
        help="Use num_workers=0 and pin_memory=False for Docker shared-memory issues.",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = KneeOAM2NoInvBlurBorderLossDataset(
        csv_path=PROJECT_ROOT / args.train_csv,
        split="train",
        suppression_prob=args.suppression_prob,
        blur_kernel=args.blur_kernel,
        border_frac=args.border_frac,
        seed=args.seed,
    )

    val_dataset = KneeOAM2NoInvBlurBorderLossDataset(
        csv_path=PROJECT_ROOT / args.val_csv,
        split="val",
        suppression_prob=0.0,
        blur_kernel=args.blur_kernel,
        border_frac=args.border_frac,
        seed=args.seed,
    )

    num_workers = 0 if args.safe_dataloader else args.num_workers
    pin_memory = False if args.safe_dataloader else torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"DataLoader workers: {num_workers}, pin_memory={pin_memory}")

    model = build_model(num_classes=5).to(device)

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

    criterion = WeightedCEWithBorderPenalty(
        ce_weights=class_weights,
        lambda_border=args.lambda_border,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    config = {
        "model_id": "M2_noinv_blur_border_loss",
        "architecture": "DenseNet201WithFeatures",
        "loss": "weighted CE + border feature penalty",
        "weighted_loss": True,
        "intervention": "no-inversion ROI blur suppression plus border feature penalty",
        "suppression_probability_train": args.suppression_prob,
        "blur_kernel": args.blur_kernel,
        "border_frac": args.border_frac,
        "lambda_border": args.lambda_border,
        "roi_inversion_for_mask_generation": False,
        "model_input_inverted": False,
        "validation_suppression": False,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "checkpoint_selection": "best validation macro-F1",
    }

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    best_val_macro_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        (
            train_loss,
            train_ce,
            train_border,
            train_acc,
            train_macro_f1,
        ) = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        (
            val_loss,
            val_ce,
            val_border,
            val_acc,
            val_macro_f1,
            val_weighted_f1,
        ) = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ce_loss": train_ce,
            "train_border_loss": train_border,
            "train_accuracy": train_acc,
            "train_macro_f1": train_macro_f1,
            "val_loss": val_loss,
            "val_ce_loss": val_ce,
            "val_border_loss": val_border,
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
            "val_weighted_f1": val_weighted_f1,
            "learning_rate": args.lr,
            "suppression_probability": args.suppression_prob,
            "blur_kernel": args.blur_kernel,
            "border_frac": args.border_frac,
            "lambda_border": args.lambda_border,
        }

        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

        print(
            f"train_loss={train_loss:.4f} "
            f"train_ce={train_ce:.4f} "
            f"train_border={train_border:.4f} "
            f"train_acc={train_acc:.4f} "
            f"train_macro_f1={train_macro_f1:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_ce={val_ce:.4f} "
            f"val_border={val_border:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} "
            f"val_weighted_f1={val_weighted_f1:.4f}"
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