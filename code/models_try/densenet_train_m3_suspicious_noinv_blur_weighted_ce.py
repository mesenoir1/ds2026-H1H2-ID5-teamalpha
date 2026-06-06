#!/usr/bin/env python

"""
M3: Suspicious-case targeted augmentation.

DenseNet201 + weighted cross-entropy.

Training logic:
- Load normal train.csv.
- Load suspicious_cases.csv from baseline B1 train Grad-CAM analysis.
- Only training images that are in the suspicious-case set receive ROI-guided
  background blur augmentation.
- Non-suspicious training images are kept normal, except standard horizontal flip.
- Validation uses original images only.

Important:
- Use ONLY train suspicious cases for training.
- Do NOT use validation/test suspicious cases for training.
- Model input is never inverted.
- ROI mask generation is no-inversion.
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


def normalize_path_str(path_str: str) -> str:
    """
    Normalize image paths so suspicious-case CSV paths and train.csv paths match.
    """
    path = str(path_str).replace("\\", "/")

    if path.startswith(str(PROJECT_ROOT)):
        try:
            path = str(Path(path).relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            pass

    path = path.lstrip("./")
    return path


def load_suspicious_image_set(suspicious_csv: str | Path) -> set[str]:
    suspicious_csv = Path(suspicious_csv)

    if not suspicious_csv.exists():
        raise FileNotFoundError(f"Suspicious-case CSV not found: {suspicious_csv}")

    df = pd.read_csv(suspicious_csv)

    if "image_path" not in df.columns:
        raise ValueError(f"{suspicious_csv} must contain an image_path column")

    if "suspicious_case" in df.columns:
        df = df[df["suspicious_case"].astype(int) == 1]

    suspicious_paths = {
        normalize_path_str(path)
        for path in df["image_path"].astype(str).tolist()
    }

    return suspicious_paths


def get_roi_mask_noinv(image_input) -> np.ndarray:
    """
    Dynamic ROI mask generation without inversion correction.

    This intentionally does NOT call get_roi_mask(), because get_roi_mask()
    may invert internally. For M3 noinv, ROI generation uses original polarity.
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


class KneeOAM3SuspiciousNoInvBlurDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        split: str,
        suspicious_image_set: set[str] | None = None,
        suspicious_suppression_prob: float = 1.0,
        blur_kernel: int = 21,
        seed: int = 42,
    ):
        self.csv_path = Path(csv_path)
        self.split = split
        self.data = pd.read_csv(self.csv_path)

        self.suspicious_image_set = suspicious_image_set or set()
        self.suspicious_suppression_prob = suspicious_suppression_prob
        self.blur_kernel = blur_kernel
        self.rng = random.Random(seed)

        if "image_path" not in self.data.columns or "label" not in self.data.columns:
            raise ValueError(f"{self.csv_path} must contain image_path,label columns")

        self.normalize = transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

        self.data["_normalized_image_path"] = self.data["image_path"].astype(str).map(
            normalize_path_str
        )

        self.data["_is_suspicious_train_case"] = self.data[
            "_normalized_image_path"
        ].isin(self.suspicious_image_set)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        rel_image_path = normalize_path_str(row["image_path"])
        image_path = PROJECT_ROOT / rel_image_path
        label = int(row["label"])
        is_suspicious = bool(row["_is_suspicious_train_case"])

        image = Image.open(image_path).convert("L")
        image = image.resize((224, 224), Image.BILINEAR)

        image_np = np.asarray(image).astype(np.uint8)

        if self.split == "train" and self.rng.random() < 0.5:
            image_np = np.fliplr(image_np).copy()

        apply_targeted_suppression = (
            self.split == "train"
            and is_suspicious
            and self.rng.random() < self.suspicious_suppression_prob
        )

        if apply_targeted_suppression:
            roi_mask = get_roi_mask_noinv(image_np)

            image_np = apply_blur_background_suppression(
                image=image_np,
                roi_mask=roi_mask,
                blur_kernel=self.blur_kernel,
            )

        image_pil = Image.fromarray(image_np, mode="L")
        image_tensor = TF.to_tensor(image_pil)
        image_tensor = image_tensor.repeat(3, 1, 1)
        image_tensor = self.normalize(image_tensor)

        return image_tensor, label


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

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

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
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, accuracy, macro_f1


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Validation", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

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
    parser = argparse.ArgumentParser(
        description="M3 suspicious-case targeted no-inversion blur augmentation"
    )

    parser.add_argument("--train-csv", default="data/splits/train.csv")
    parser.add_argument("--val-csv", default="data/splits/val.csv")

    parser.add_argument(
        "--suspicious-csv",
        required=True,
        help="Baseline B1 train suspicious_cases.csv. Do not use val/test suspicious cases.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/m3_suspicious_noinv_blur",
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--suspicious-suppression-prob",
        type=float,
        default=1.0,
        help="Probability of applying ROI blur to suspicious training cases.",
    )

    parser.add_argument("--blur-kernel", type=int, default=21)

    parser.add_argument(
        "--safe-dataloader",
        action="store_true",
        help="Use num_workers=0 and pin_memory=False for Docker shared-memory issues.",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    suspicious_image_set = load_suspicious_image_set(
        PROJECT_ROOT / args.suspicious_csv
    )

    print(f"Loaded suspicious training paths: {len(suspicious_image_set)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = KneeOAM3SuspiciousNoInvBlurDataset(
        csv_path=PROJECT_ROOT / args.train_csv,
        split="train",
        suspicious_image_set=suspicious_image_set,
        suspicious_suppression_prob=args.suspicious_suppression_prob,
        blur_kernel=args.blur_kernel,
        seed=args.seed,
    )

    val_dataset = KneeOAM3SuspiciousNoInvBlurDataset(
        csv_path=PROJECT_ROOT / args.val_csv,
        split="val",
        suspicious_image_set=set(),
        suspicious_suppression_prob=0.0,
        blur_kernel=args.blur_kernel,
        seed=args.seed,
    )

    n_suspicious_matched = int(
        train_dataset.data["_is_suspicious_train_case"].sum()
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Suspicious train cases matched in train.csv: {n_suspicious_matched}")

    if n_suspicious_matched == 0:
        raise RuntimeError(
            "No suspicious cases matched train.csv. Check path format and suspicious CSV."
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

    print(f"DataLoader workers: {num_workers}, pin_memory={pin_memory}")

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

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    config = {
        "model_id": "M3_suspicious_noinv_blur",
        "architecture": "DenseNet201",
        "loss": "WeightedCrossEntropyLoss",
        "intervention": "targeted no-inversion ROI blur on B1 suspicious training cases",
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "suspicious_csv": args.suspicious_csv,
        "suspicious_paths_loaded": len(suspicious_image_set),
        "suspicious_train_cases_matched": n_suspicious_matched,
        "suspicious_suppression_probability": args.suspicious_suppression_prob,
        "blur_kernel": args.blur_kernel,
        "roi_inversion_for_mask_generation": False,
        "model_input_inverted": False,
        "validation_suppression": False,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "checkpoint_selection": "best validation macro-F1",
        "data_leakage_rule": "Only B1 train suspicious cases used for training intervention.",
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
            "suspicious_suppression_probability": args.suspicious_suppression_prob,
            "suspicious_train_cases_matched": n_suspicious_matched,
        }

        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

        print(
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"train_macro_f1={train_macro_f1:.4f} | "
            f"val_loss={val_loss:.4f} "
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