#!/usr/bin/env python

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from roi_dynamic import get_roi_mask


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_str: str) -> Path:
    path = Path(str(path_str))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def make_border_mask(h: int, w: int, border_frac: float = 0.08) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)

    by = max(1, int(h * border_frac))
    bx = max(1, int(w * border_frac))

    mask[:by, :] = 1
    mask[-by:, :] = 1
    mask[:, :bx] = 1
    mask[:, -bx:] = 1

    return mask


def save_roi_debug(image_path: Path, output_path: Path, border_frac: float = 0.08) -> None:
    image = np.asarray(Image.open(image_path).convert("L"))

    roi = get_roi_mask(image)
    border = make_border_mask(image.shape[0], image.shape[1], border_frac=border_frac)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(image, cmap="gray")
    plt.imshow(roi, cmap="Reds", alpha=0.35)
    plt.title(f"Dynamic ROI\narea={roi.mean():.3f}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(image, cmap="gray")
    plt.imshow(border, cmap="Reds", alpha=0.35)
    plt.title(f"Border mask\narea={border.mean():.3f}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image-paths",
        nargs="+",
        required=True,
        help="One or more image paths, relative to project root or absolute.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/roi_debug_specific_cases",
        help="Output directory for ROI debug figures.",
    )

    parser.add_argument(
        "--border-frac",
        type=float,
        default=0.08,
    )

    args = parser.parse_args()

    output_dir = resolve_path(args.output_dir)

    for idx, image_path_str in enumerate(args.image_paths):
        image_path = resolve_path(image_path_str)

        if not image_path.exists():
            print(f"[WARN] Missing image: {image_path}")
            continue

        safe_name = f"{idx:03d}_{image_path.parent.name}_{image_path.stem}.png"

        save_roi_debug(
            image_path=image_path,
            output_path=output_dir / safe_name,
            border_frac=args.border_frac,
        )

        print(f"Saved: {output_dir / safe_name}")


if __name__ == "__main__":
    main()

