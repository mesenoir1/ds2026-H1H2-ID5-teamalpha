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
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


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


def suppress_metal_artifacts(img: np.ndarray, threshold: int = 235) -> np.ndarray:
    metal_mask = img > threshold

    if not np.any(metal_mask):
        return img

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    metal_mask = cv2.dilate(
        metal_mask.astype(np.uint8),
        kernel,
        iterations=2,
    ).astype(bool)

    img_no_metal = img.copy()
    non_metal_pixels = img[~metal_mask]

    replacement_value = (
        np.median(non_metal_pixels)
        if non_metal_pixels.size > 0
        else np.median(img)
    )

    img_no_metal[metal_mask] = replacement_value

    return img_no_metal


def find_joint_space_y(img_gray: np.ndarray) -> int:
    h, w = img_gray.shape

    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    sobel_vertical = cv2.convertScaleAbs(
        cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    )

    x_start, x_end = int(w * 0.10), int(w * 0.90)
    center_roi = sobel_vertical[:, x_start:x_end]

    roi_blurred = cv2.GaussianBlur(center_roi, (5, 5), 0)
    row_means = np.mean(roi_blurred, axis=1)

    search_top = int(h * 0.40)
    search_bottom = int(h * 0.82)

    local_max_index = np.argmax(row_means[search_top:search_bottom])
    line_y = search_top + local_max_index

    return int(line_y)


def get_dynamic_roi_mask_noinv(image: np.ndarray) -> np.ndarray:
    """
    Dynamic ROI mask without inversion correction.
    Used for no-inversion M2 variants.
    """

    img = image.copy()
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

    r = int(h * 0.04)
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

    binary_mask = np.where(mask > 0, 1, 0).astype(np.uint8)

    return binary_mask


def apply_darken_background_suppression(
    image: np.ndarray,
    roi_mask: np.ndarray,
    darken_factor: float = 0.35,
) -> np.ndarray:
    """
    Keep ROI unchanged.
    Outside ROI = original image darkened by darken_factor.
    """

    roi_mask = (roi_mask > 0).astype(np.float32)

    darkened = image.astype(np.float32) * darken_factor

    suppressed = (
        image.astype(np.float32) * roi_mask
        + darkened * (1.0 - roi_mask)
    )

    return np.clip(suppressed, 0, 255).astype(np.uint8)


class KneeOAM2DarkenNoInvDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        split: str,
        suppression_prob: float = 0.5,
        darken_factor: float = 0.35,
        seed: int = 42,
    ):
        self.csv_path = Path(csv_path)
        self.split = split
        self.data = pd.read_csv(self.csv_path)

        self.suppression_prob = suppression_prob
        self.darken_factor = darken_factor
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

        image_path = PROJECT_ROOT / row["image_path"]
        label = int(row["label"])

        image = Image.open(image_path).convert("L")
        image = image.resize((224, 224), Image.BILINEAR)

        image_np = np.asarray(image).astype(np.uint8)

        if self.split == "train" and self.rng.random() < 0.5:
            image_np = np.fliplr(image_np).copy()

        if self.split == "train" and self.rng.random() < self.suppression_prob:
            roi_mask = get_dynamic_roi_mask_noinv(image_np)

            image_np = apply_darken_background_suppression(
                image=image_np,
                roi_mask=roi_mask,
                darken_factor=self.darken_factor,
            )

        image_pil = Image.fromarray(image_np, mode="L")
        image_tensor = self.final_transform(image_pil)

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
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-csv", default="data/splits/train.csv")
    parser.add_argument("--val-csv", default="data/splits/val.csv")
    parser.add_argument("--output-dir", default="outputs/m2_noinv_darken")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--suppression-prob", type=float, default=0.5)
    parser.add_argument("--darken-factor", type=float, default=0.35)

    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = KneeOAM2DarkenNoInvDataset(
        csv_path=PROJECT_ROOT / args.train_csv,
        split="train",
        suppression_prob=args.suppression_prob,
        darken_factor=args.darken_factor,
        seed=args.seed,
    )

    val_dataset = KneeOAM2DarkenNoInvDataset(
        csv_path=PROJECT_ROOT / args.val_csv,
        split="val",
        suppression_prob=0.0,
        darken_factor=args.darken_factor,
        seed=args.seed,
    )

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
        device=device,
    )

    print("Class counts:")
    print(class_counts)

    print("Class weights:")
    print(class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_macro_f1 = -1.0
    history = []

    config = {
        "model_id": "M2_noinv_darken",
        "architecture": "DenseNet201",
        "loss": "WeightedCrossEntropyLoss",
        "weighted_loss": True,
        "num_classes": 5,
        "pretrained": True,
        "intervention": "dynamic ROI-guided background suppression",
        "suppression_mode": "darken",
        "roi_inversion_for_mask_generation": False,
        "model_input_inverted": False,
        "suppression_probability_train": args.suppression_prob,
        "darken_factor": args.darken_factor,
        "validation_suppression": False,
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
            "suppression_probability": args.suppression_prob,
            "suppression_mode": "darken",
            "roi_inversion_for_mask_generation": False,
            "darken_factor": args.darken_factor,
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