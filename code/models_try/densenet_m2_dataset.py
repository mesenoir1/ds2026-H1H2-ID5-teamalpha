from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def is_inverted(img: np.ndarray) -> bool:
    h, w = img.shape

    by, bx = max(1, int(h * 0.05)), max(1, int(w * 0.05))

    top = img[0:by, :]
    bottom = img[h - by:h, :]
    left = img[:, 0:bx]
    right = img[:, w - bx:w]

    cy, cx = h // 2, w // 2
    dy, dx = int(h * 0.2), int(w * 0.2)
    center = img[cy - dy:cy + dy, cx - dx:cx + dx]

    border_mean = np.mean([
        np.mean(top),
        np.mean(bottom),
        np.mean(left),
        np.mean(right),
    ])
    center_mean = np.mean(center)

    return border_mean > center_mean


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


def get_dynamic_roi_mask(
    image: np.ndarray,
    use_inversion_for_roi: bool = False,
) -> np.ndarray:
    """
    Dynamic ROI mask for M2 training.

    use_inversion_for_roi=False for M2a.
    The model input image itself is never inverted.
    """

    img = image.copy()

    h, w = img.shape

    if use_inversion_for_roi and is_inverted(img):
        img = cv2.bitwise_not(img)

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

    suppressed = image.astype(np.float32) * roi_mask + blurred.astype(np.float32) * (1.0 - roi_mask)

    return np.clip(suppressed, 0, 255).astype(np.uint8)


class KneeOAM2Dataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        split: str,
        suppression_prob: float = 0.5,
        use_inversion_for_roi: bool = False,
        blur_kernel: int = 21,
        seed: int = 42,
    ):
        self.csv_path = Path(csv_path)
        self.split = split
        self.data = pd.read_csv(self.csv_path)

        self.suppression_prob = suppression_prob
        self.use_inversion_for_roi = use_inversion_for_roi
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

        image_path = PROJECT_ROOT / row["image_path"]
        label = int(row["label"])

        image = Image.open(image_path).convert("L")
        image = image.resize((224, 224), Image.BILINEAR)

        image_np = np.asarray(image).astype(np.uint8)

        if self.split == "train" and self.rng.random() < 0.5:
            image_np = np.fliplr(image_np).copy()

        if self.split == "train" and self.rng.random() < self.suppression_prob:
            roi_mask = get_dynamic_roi_mask(
                image_np,
                use_inversion_for_roi=self.use_inversion_for_roi,
            )

            image_np = apply_blur_background_suppression(
                image=image_np,
                roi_mask=roi_mask,
                blur_kernel=self.blur_kernel,
            )

        image_pil = Image.fromarray(image_np, mode="L")
        image_tensor = self.final_transform(image_pil)

        return image_tensor, label