
import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from densenet_dataset import KneeOADataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_LABELS = [0, 1, 2, 3, 4]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def build_densenet201(num_classes: int = 5) -> nn.Module:
    model = models.densenet201(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_densenet201(num_classes=5)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self.forward_handle = self.target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, images: torch.Tensor, target_class: Optional[int] = None):
        self.model.zero_grad(set_to_none=True)

        logits = self.model(images)
        pred_classes = torch.argmax(logits, dim=1)

        if target_class is None:
            target_scores = logits.gather(1, pred_classes.view(-1, 1)).sum()
        else:
            target_scores = logits[:, target_class].sum()

        target_scores.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cams = (weights * self.activations).sum(dim=1)
        cams = F.relu(cams)

        cams = cams.unsqueeze(1)
        cams = F.interpolate(
            cams,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        normalized = []
        for cam in cams:
            cam_min = cam.min()
            cam_max = cam.max()
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
            normalized.append(cam)

        heatmaps = torch.stack(normalized, dim=0)
        return heatmaps.detach(), logits.detach(), pred_classes.detach()


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    return image


def make_overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    colored_heatmap = cmap(heatmap)[..., :3]
    overlay = (1 - alpha) * image + alpha * colored_heatmap
    overlay = np.clip(overlay, 0, 1)
    return overlay


def safe_stem(image_path: str, index: int) -> str:
    path = Path(image_path)
    parent = path.parent.name
    stem = path.stem
    return f"{index:06d}_kl{parent}_{stem}"


def save_png(array: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray((array * 255).astype(np.uint8))
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/splits/train.csv")
    parser.add_argument("--checkpoint", default="outputs/densenet_weighted_ce/best_model.pt")
    parser.add_argument("--output-dir", default="outputs/gradcam/densenet_weighted_ce")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument(
        "--target",
        default="predicted",
        choices=["predicted", "true"],
        help="Generate Grad-CAM for predicted class or true class.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit for debugging, e.g. --max-images 20.",
    )
    args = parser.parse_args()

    csv_path = PROJECT_ROOT / args.csv
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    output_dir = PROJECT_ROOT / args.output_dir
    heatmap_dir = output_dir / "heatmaps"
    overlay_dir = output_dir / "overlays"
    original_dir = output_dir / "originals"

    heatmap_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    original_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = KneeOADataset(csv_path=csv_path, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = load_model(checkpoint_path, device)

    target_layer = model.features.denseblock4
    gradcam = GradCAM(model=model, target_layer=target_layer)

    rows = []
    processed = 0

    try:
        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="Generating Grad-CAM")):
            images = images.to(device)
            labels = labels.to(device)

            if args.target == "true":
                if images.size(0) != 1:
                    raise ValueError("Use --batch-size 1 when --target true.")
                target_class = int(labels.item())
            else:
                target_class = None

            heatmaps, logits, pred_classes = gradcam(images, target_class=target_class)
            probabilities = torch.softmax(logits, dim=1)

            batch_size = images.size(0)
            start_idx = batch_idx * args.batch_size

            for i in range(batch_size):
                global_idx = start_idx + i
                if args.max_images is not None and processed >= args.max_images:
                    break

                source_row = dataset.data.iloc[global_idx]
                image_path = source_row["image_path"]
                true_label = int(labels[i].detach().cpu().item())
                pred_label = int(pred_classes[i].detach().cpu().item())
                correct = int(true_label == pred_label)

                file_id = safe_stem(image_path, global_idx)
                heatmap_path = heatmap_dir / f"{file_id}.npy"
                overlay_path = overlay_dir / f"{file_id}.png"
                original_path = original_dir / f"{file_id}.png"

                image_np = denormalize_image(images[i])
                heatmap_np = heatmaps[i].detach().cpu().numpy()
                overlay_np = make_overlay(image_np, heatmap_np, alpha=args.alpha)

                np.save(heatmap_path, heatmap_np.astype(np.float32))
                save_png(image_np, original_path)
                save_png(overlay_np, overlay_path)

                row = {
                    "index": global_idx,
                    "image_path": image_path,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "correct": correct,
                    "target_for_gradcam": args.target,
                    "heatmap_path": str(heatmap_path.relative_to(PROJECT_ROOT)),
                    "overlay_path": str(overlay_path.relative_to(PROJECT_ROOT)),
                    "original_path": str(original_path.relative_to(PROJECT_ROOT)),
                }

                for class_idx in CLASS_LABELS:
                    row[f"prob_{class_idx}"] = float(
                        probabilities[i, class_idx].detach().cpu().item()
                    )

                rows.append(row)
                processed += 1

            if args.max_images is not None and processed >= args.max_images:
                break

    finally:
        gradcam.remove_hooks()

    predictions_path = output_dir / "gradcam_predictions.csv"
    pd.DataFrame(rows).to_csv(predictions_path, index=False)

    config = {
        "csv": args.csv,
        "checkpoint": args.checkpoint,
        "output_dir": args.output_dir,
        "split": args.split,
        "target_layer": "model.features.denseblock4",
        "target_for_gradcam": args.target,
        "num_images": len(rows),
        "alpha": args.alpha,
    }

    with open(output_dir / "gradcam_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved prediction metadata: {predictions_path}")
    print(f"Saved heatmaps to: {heatmap_dir}")
    print(f"Saved overlays to: {overlay_dir}")


if __name__ == "__main__":
    main()