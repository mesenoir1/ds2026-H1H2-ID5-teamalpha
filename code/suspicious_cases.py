#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-8


# -------------------------
# Loading / normalization
# -------------------------

def load_grayscale_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image)


def normalize_float01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    min_val = float(x.min())
    max_val = float(x.max())

    if max_val - min_val < EPS:
        return np.zeros_like(x, dtype=np.float32)

    return (x - min_val) / (max_val - min_val)


def normalize_uint8(x: np.ndarray) -> np.ndarray:
    x = normalize_float01(x)
    return (x * 255).astype(np.uint8)


def load_heatmap(path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    heatmap = np.load(path)
    heatmap = np.squeeze(heatmap)

    if heatmap.ndim != 2:
        raise ValueError(f"Expected 2D heatmap after squeeze, got {heatmap.shape}")

    heatmap = heatmap.astype(np.float32)
    heatmap = np.maximum(heatmap, 0)

    h, w = target_shape

    if heatmap.shape != target_shape:
        heatmap = cv2.resize(
            heatmap,
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

    return normalize_float01(heatmap)


def find_heatmap_path(image_path: str, heatmap_dir: Path) -> Path | None:
    """
    Supports heatmaps saved by generate_gradcam.py, e.g.
    000123_kl2_filename.npy
    """

    stem = Path(str(image_path)).stem

    candidates = [
        heatmap_dir / f"{stem}.npy",
        heatmap_dir / f"{stem}_heatmap.npy",
        heatmap_dir / f"{stem}_gradcam.npy",
        heatmap_dir / f"{stem}_cam.npy",
    ]

    candidates.extend(sorted(heatmap_dir.glob(f"*_{stem}.npy")))

    existing = [p for p in candidates if p.exists()]

    if len(existing) == 0:
        return None

    if len(existing) > 1:
        print(f"[WARN] Multiple heatmaps for {image_path}; using {existing[0]}")

    return existing[0]


# -------------------------
# Prediction handling
# -------------------------

def merge_predictions(split_df: pd.DataFrame, pred_csv: Path) -> pd.DataFrame:
    if not pred_csv.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {pred_csv}")

    pred_df = pd.read_csv(pred_csv)

    if "image_path" not in pred_df.columns:
        raise ValueError(f"{pred_csv} must contain image_path column.")

    merged = split_df.merge(
        pred_df,
        on="image_path",
        how="left",
        suffixes=("", "_pred"),
    )

    return merged


def is_correct_row(row: pd.Series) -> bool:
    if "correct" in row.index and pd.notna(row["correct"]):
        try:
            return int(row["correct"]) == 1
        except Exception:
            return str(row["correct"]).lower() in {"true", "yes", "1"}

    if "label" in row.index and "pred_label" in row.index:
        if pd.notna(row["label"]) and pd.notna(row["pred_label"]):
            return int(row["label"]) == int(row["pred_label"])

    if "true_label" in row.index and "pred_label" in row.index:
        if pd.notna(row["true_label"]) and pd.notna(row["pred_label"]):
            return int(row["true_label"]) == int(row["pred_label"])

    if "y_true" in row.index and "y_pred" in row.index:
        if pd.notna(row["y_true"]) and pd.notna(row["y_pred"]):
            return int(row["y_true"]) == int(row["y_pred"])

    return False


# -------------------------
# Final selected masks
# -------------------------

def central_ellipse_mask(
    h: int,
    w: int,
    center_y: float = 0.55,
    center_x: float = 0.50,
    radius_y: float = 0.32,
    radius_x: float = 0.38,
) -> np.ndarray:
    """
    Broad fixed anatomical prior.
    Intended to cover the central knee anatomy region.
    """

    yy, xx = np.ogrid[:h, :w]

    cy = center_y * h
    cx = center_x * w
    ry = radius_y * h
    rx = radius_x * w

    mask = ((yy - cy) ** 2 / (ry ** 2 + EPS)) + (
        (xx - cx) ** 2 / (rx ** 2 + EPS)
    ) <= 1.0

    return mask.astype(np.uint8)


def quantized_central_mask(
    image: np.ndarray,
    grid_size: int = 14,
    percentile: float = 50.0,
) -> np.ndarray:
    """
    Adaptive coarse image-content mask constrained to the central ellipse.
    """

    img = normalize_uint8(image)
    h, w = img.shape

    small = cv2.resize(
        img,
        (grid_size, grid_size),
        interpolation=cv2.INTER_AREA,
    )

    threshold = np.percentile(small, percentile)
    small_mask = (small >= threshold).astype(np.uint8)

    mask = cv2.resize(
        small_mask,
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)

    central = central_ellipse_mask(h, w)

    return (mask * central).astype(np.uint8)


def central_joint_band_mask(
    h: int,
    w: int,
    y1: float = 0.42,
    y2: float = 0.68,
    x1: float = 0.18,
    x2: float = 0.82,
) -> np.ndarray:
    """
    Focused joint-space proxy.
    This is intentionally smaller than central_ellipse.
    """

    mask = np.zeros((h, w), dtype=np.uint8)

    y_start = int(h * y1)
    y_end = int(h * y2)
    x_start = int(w * x1)
    x_end = int(w * x2)

    mask[y_start:y_end, x_start:x_end] = 1

    return mask


def border_mask(
    h: int,
    w: int,
    border_frac: float = 0.08,
) -> np.ndarray:
    """
    Non-diagnostic border region.
    Used only as an additional suspiciousness metric, not as an anatomical mask.
    """

    mask = np.zeros((h, w), dtype=np.uint8)

    by = max(1, int(h * border_frac))
    bx = max(1, int(w * border_frac))

    mask[:by, :] = 1
    mask[-by:, :] = 1
    mask[:, :bx] = 1
    mask[:, -bx:] = 1

    return mask


def build_diagnostic_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    h, w = image.shape

    return {
        "central_ellipse": central_ellipse_mask(h, w),
        "quantized_central": quantized_central_mask(image),
        "central_joint_band": central_joint_band_mask(h, w),
    }


def build_all_masks_for_debug(image: np.ndarray) -> dict[str, np.ndarray]:
    h, w = image.shape

    masks = build_diagnostic_masks(image)
    masks["border"] = border_mask(h, w)

    return masks


# -------------------------
# Metrics
# -------------------------

def compute_region_metrics(
    heatmap: np.ndarray,
    mask: np.ndarray,
    topk_fraction: float = 0.10,
) -> dict[str, float]:
    heatmap = normalize_float01(np.maximum(heatmap.astype(np.float32), 0))
    mask = (mask > 0).astype(np.float32)

    if heatmap.shape != mask.shape:
        mask = cv2.resize(
            mask,
            (heatmap.shape[1], heatmap.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        mask = (mask > 0).astype(np.float32)

    total_activation = float(heatmap.sum()) + EPS
    inside_activation = float((heatmap * mask).sum())

    inside_ratio = inside_activation / total_activation
    outside_ratio = 1.0 - inside_ratio
    mask_area_ratio = float(mask.mean())
    activation_enrichment = inside_ratio / (mask_area_ratio + EPS)

    flat_heatmap = heatmap.reshape(-1)
    flat_mask = mask.reshape(-1)

    k = max(1, int(len(flat_heatmap) * topk_fraction))
    top_idx = np.argpartition(flat_heatmap, -k)[-k:]
    topk_inside_fraction = float(flat_mask[top_idx].mean())

    return {
        "inside_ratio": float(inside_ratio),
        "outside_ratio": float(outside_ratio),
        "mask_area_ratio": float(mask_area_ratio),
        "activation_enrichment": float(activation_enrichment),
        "topk_inside_fraction": float(topk_inside_fraction),
    }


def save_debug_figure(
    output_path: Path,
    image: np.ndarray,
    heatmap: np.ndarray,
    masks: dict[str, np.ndarray],
    ratios: dict[str, float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_cols = 2 + len(masks)
    plt.figure(figsize=(3.1 * n_cols, 3.5))

    plt.subplot(1, n_cols, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, n_cols, 2)
    plt.imshow(image, cmap="gray")
    plt.imshow(heatmap, cmap="jet", alpha=0.45)
    plt.title("Grad-CAM")
    plt.axis("off")

    for idx, (name, mask) in enumerate(masks.items(), start=3):
        plt.subplot(1, n_cols, idx)
        plt.imshow(image, cmap="gray")
        plt.imshow(mask, cmap="Reds", alpha=0.35)
        plt.title(f"{name}\ninside={ratios[name]:.3f}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# -------------------------
# Suspicious-case logic
# -------------------------

def make_wide_scores(scores_df: pd.DataFrame, diagnostic_methods: list[str]) -> pd.DataFrame:
    available = set(scores_df["method"].unique())
    missing = [m for m in diagnostic_methods if m not in available]

    if missing:
        raise ValueError(
            f"Missing selected diagnostic methods: {missing}. "
            f"Available methods: {sorted(available)}"
        )

    metadata_cols = [
        "image_path",
        "heatmap_path",
        "label",
        "true_label",
        "y_true",
        "pred_label",
        "y_pred",
        "prediction",
        "correct",
        "target_for_gradcam",
        "prob_0",
        "prob_1",
        "prob_2",
        "prob_3",
        "prob_4",
    ]

    metadata_cols = [c for c in metadata_cols if c in scores_df.columns]

    metadata = (
        scores_df[metadata_cols]
        .drop_duplicates(subset=["image_path"])
        .copy()
    )

    metrics = [
        "inside_ratio",
        "outside_ratio",
        "mask_area_ratio",
        "activation_enrichment",
        "topk_inside_fraction",
    ]

    wide = metadata.copy()

    for metric in metrics:
        pivot = scores_df.pivot_table(
            index="image_path",
            columns="method",
            values=metric,
            aggfunc="first",
        )

        pivot.columns = [f"{m}_{metric}" for m in pivot.columns]
        wide = wide.merge(pivot.reset_index(), on="image_path", how="left")

    if "border_inside_ratio" in wide.columns:
        wide["border_attention_score"] = wide["border_inside_ratio"]

    return wide


def add_suspicion_flags(
    wide: pd.DataFrame,
    diagnostic_methods: list[str],
    suspicion_quantile: float,
    border_quantile: float,
    min_low_methods: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    thresholds = {}

    for method in diagnostic_methods:
        col = f"{method}_inside_ratio"

        if col not in wide.columns:
            raise ValueError(f"Missing column required for suspicion rule: {col}")

        threshold = wide[col].quantile(suspicion_quantile)

        thresholds[f"{method}_inside_ratio_low_threshold"] = float(threshold)

        wide[f"{method}_threshold"] = threshold
        wide[f"low_{method}"] = wide[col] <= threshold

    low_cols = [f"low_{m}" for m in diagnostic_methods]
    inside_cols = [f"{m}_inside_ratio" for m in diagnostic_methods]
    topk_cols = [f"{m}_topk_inside_fraction" for m in diagnostic_methods]
    enrichment_cols = [f"{m}_activation_enrichment" for m in diagnostic_methods]

    wide["num_low_methods"] = wide[low_cols].sum(axis=1)
    wide["low_region_attention"] = wide["num_low_methods"] >= min_low_methods

    wide["mean_inside_ratio_selected"] = wide[inside_cols].mean(axis=1)
    wide["min_inside_ratio_selected"] = wide[inside_cols].min(axis=1)

    existing_topk_cols = [c for c in topk_cols if c in wide.columns]
    if existing_topk_cols:
        wide["mean_topk_inside_fraction_selected"] = wide[existing_topk_cols].mean(axis=1)

    existing_enrichment_cols = [c for c in enrichment_cols if c in wide.columns]
    if existing_enrichment_cols:
        wide["mean_activation_enrichment_selected"] = wide[existing_enrichment_cols].mean(axis=1)

    if "border_attention_score" in wide.columns:
        border_threshold = wide["border_attention_score"].quantile(border_quantile)
        thresholds["border_attention_high_threshold"] = float(border_threshold)

        wide["border_attention_threshold"] = border_threshold
        wide["high_border_attention"] = wide["border_attention_score"] >= border_threshold
    else:
        thresholds["border_attention_high_threshold"] = None
        wide["high_border_attention"] = False

    wide["suspicious"] = wide["low_region_attention"] | wide["high_border_attention"]

    def reason(row: pd.Series) -> str:
        low_region = bool(row["low_region_attention"])
        high_border = bool(row["high_border_attention"])

        if low_region and high_border:
            return "low_region_and_high_border"
        if low_region:
            return "low_region_attention"
        if high_border:
            return "high_border_attention"
        return "not_suspicious"

    wide["suspicion_reason"] = wide.apply(reason, axis=1)

    return wide, thresholds


# -------------------------
# Main
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post-Grad-CAM region analysis and suspicious correct-for-wrong-reason "
            "case detection using final diagnostic masks plus border attention."
        )
    )

    parser.add_argument(
        "--split-csv",
        required=True,
        type=Path,
        help="Split CSV with image_path and label columns.",
    )

    parser.add_argument(
        "--predictions-csv",
        required=True,
        type=Path,
        help="Grad-CAM predictions CSV with image_path and prediction metadata.",
    )

    parser.add_argument(
        "--heatmap-dir",
        required=True,
        type=Path,
        help="Directory containing raw Grad-CAM .npy heatmaps.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory.",
    )

    parser.add_argument(
        "--only-correct",
        action="store_true",
        help="Only analyze correctly classified cases.",
    )

    parser.add_argument(
        "--diagnostic-methods",
        nargs="+",
        default=["central_ellipse", "quantized_central", "central_joint_band"],
        help="Diagnostic masks used for low-region-attention rule.",
    )

    parser.add_argument(
        "--suspicion-quantile",
        type=float,
        default=0.20,
        help="Bottom quantile for low inside-ratio. Default: 0.20.",
    )

    parser.add_argument(
        "--border-quantile",
        type=float,
        default=0.80,
        help="Top quantile for high border attention. Default: 0.80.",
    )

    parser.add_argument(
        "--min-low-methods",
        type=int,
        default=2,
        help="Minimum number of diagnostic methods that must be low. Default: 2.",
    )

    parser.add_argument(
        "--topk-fraction",
        type=float,
        default=0.10,
        help="Fraction of top heatmap pixels used for top-k inside metric. Default: 0.10.",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional image limit for testing.",
    )

    parser.add_argument(
        "--num-debug-figures",
        type=int,
        default=80,
        help="Number of debug figures to save.",
    )

    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save binary mask PNGs. Can create many files.",
    )

    args = parser.parse_args()

    if not 0 < args.suspicion_quantile < 1:
        raise ValueError("--suspicion-quantile must be between 0 and 1.")

    if not 0 < args.border_quantile < 1:
        raise ValueError("--border-quantile must be between 0 and 1.")

    if not 0 < args.topk_fraction < 1:
        raise ValueError("--topk-fraction must be between 0 and 1.")

    if args.min_low_methods < 1:
        raise ValueError("--min-low-methods must be >= 1.")

    if args.min_low_methods > len(args.diagnostic_methods):
        raise ValueError("--min-low-methods cannot exceed number of diagnostic methods.")

    split_csv = PROJECT_ROOT / args.split_csv
    predictions_csv = PROJECT_ROOT / args.predictions_csv
    heatmap_dir = PROJECT_ROOT / args.heatmap_dir
    output_dir = PROJECT_ROOT / args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = output_dir / "debug_figures"
    masks_dir = output_dir / "masks"

    if not split_csv.exists():
        raise FileNotFoundError(f"Missing split CSV: {split_csv}")

    if not predictions_csv.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {predictions_csv}")

    if not heatmap_dir.exists():
        raise FileNotFoundError(f"Missing heatmap directory: {heatmap_dir}")

    split_df = pd.read_csv(split_csv)

    if "image_path" not in split_df.columns:
        raise ValueError(f"{split_csv} must contain image_path column.")

    df = merge_predictions(split_df, predictions_csv)

    if args.only_correct:
        before = len(df)
        correct_flags = df.apply(is_correct_row, axis=1)
        df = df[correct_flags].copy()
        print(f"Filtered to correct cases: {len(df)}/{before}")

    if args.max_images is not None:
        df = df.head(args.max_images).copy()

    all_score_rows = []
    skipped_rows = []

    debug_count = 0
    missing_images = 0
    missing_heatmaps = 0

    for _, row in df.iterrows():
        rel_image_path = str(row["image_path"])
        image_path = PROJECT_ROOT / rel_image_path

        if not image_path.exists():
            missing_images += 1
            skipped_rows.append({
                "image_path": rel_image_path,
                "reason": "missing_image",
            })
            print(f"[SKIP] Missing image: {rel_image_path}")
            continue

        heatmap_path = find_heatmap_path(rel_image_path, heatmap_dir)

        if heatmap_path is None:
            missing_heatmaps += 1
            skipped_rows.append({
                "image_path": rel_image_path,
                "reason": "missing_heatmap",
            })
            print(f"[SKIP] Missing heatmap for: {rel_image_path}")
            continue

        try:
            image = load_grayscale_image(image_path)
            h, w = image.shape

            heatmap = load_heatmap(heatmap_path, target_shape=(h, w))
            masks = build_all_masks_for_debug(image)

            ratios_for_debug = {}
            image_stem = Path(rel_image_path).stem

            for method_name, mask in masks.items():
                metrics = compute_region_metrics(
                    heatmap=heatmap,
                    mask=mask,
                    topk_fraction=args.topk_fraction,
                )

                ratios_for_debug[method_name] = metrics["inside_ratio"]

                if args.save_masks:
                    mask_out = masks_dir / method_name / f"{image_stem}.png"
                    mask_out.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_out)

                score_row = {
                    "image_path": rel_image_path,
                    "heatmap_path": str(heatmap_path.relative_to(PROJECT_ROOT)),
                    "method": method_name,
                    **metrics,
                }

                for col in [
                    "label",
                    "true_label",
                    "y_true",
                    "pred_label",
                    "y_pred",
                    "prediction",
                    "correct",
                    "target_for_gradcam",
                    "prob_0",
                    "prob_1",
                    "prob_2",
                    "prob_3",
                    "prob_4",
                ]:
                    if col in row.index:
                        score_row[col] = row[col]

                all_score_rows.append(score_row)

            if debug_count < args.num_debug_figures:
                save_debug_figure(
                    output_path=debug_dir / f"{image_stem}_debug.png",
                    image=image,
                    heatmap=heatmap,
                    masks=masks,
                    ratios=ratios_for_debug,
                )
                debug_count += 1

        except Exception as exc:
            skipped_rows.append({
                "image_path": rel_image_path,
                "reason": f"error: {exc}",
            })
            print(f"[SKIP] Error for {rel_image_path}: {exc}")

    scores_df = pd.DataFrame(all_score_rows)

    scores_path = output_dir / "region_scores.csv"
    skipped_path = output_dir / "skipped_rows.csv"

    scores_df.to_csv(scores_path, index=False)
    pd.DataFrame(skipped_rows).to_csv(skipped_path, index=False)

    print()
    print(f"Saved region scores: {scores_path}")
    print(f"Saved skipped rows:  {skipped_path}")
    print(f"Scored rows:         {len(scores_df)}")
    print(f"Missing images:      {missing_images}")
    print(f"Missing heatmaps:    {missing_heatmaps}")

    if scores_df.empty:
        print("No region scores produced. Stop here.")
        return

    method_summary = (
        scores_df
        .groupby("method")
        .agg(
            n=("inside_ratio", "count"),
            mean_inside_ratio=("inside_ratio", "mean"),
            std_inside_ratio=("inside_ratio", "std"),
            median_inside_ratio=("inside_ratio", "median"),
            mean_outside_ratio=("outside_ratio", "mean"),
            mean_mask_area_ratio=("mask_area_ratio", "mean"),
            mean_activation_enrichment=("activation_enrichment", "mean"),
            median_activation_enrichment=("activation_enrichment", "median"),
            mean_topk_inside_fraction=("topk_inside_fraction", "mean"),
            median_topk_inside_fraction=("topk_inside_fraction", "median"),
        )
        .reset_index()
        .sort_values(
            ["mean_activation_enrichment", "mean_inside_ratio"],
            ascending=False,
        )
    )

    method_summary_path = output_dir / "region_method_summary.csv"
    method_summary.to_csv(method_summary_path, index=False)

    wide = make_wide_scores(
        scores_df=scores_df,
        diagnostic_methods=args.diagnostic_methods,
    )

    wide, thresholds = add_suspicion_flags(
        wide=wide,
        diagnostic_methods=args.diagnostic_methods,
        suspicion_quantile=args.suspicion_quantile,
        border_quantile=args.border_quantile,
        min_low_methods=args.min_low_methods,
    )

    suspicious = wide[wide["suspicious"]].copy()

    suspicious = suspicious.sort_values(
        by=[
            "suspicion_reason",
            "num_low_methods",
            "mean_inside_ratio_selected",
            "min_inside_ratio_selected",
            "border_attention_score",
        ],
        ascending=[True, False, True, True, False],
    )

    all_cases_flags_path = output_dir / "all_cases_with_suspicion_flags.csv"
    suspicious_path = output_dir / "suspicious_cases.csv"

    wide.to_csv(all_cases_flags_path, index=False)
    suspicious.to_csv(suspicious_path, index=False)

    suspicious_summary_rows = [
        {
            "metric": "num_cases_evaluated",
            "value": len(wide),
        },
        {
            "metric": "num_suspicious_cases",
            "value": len(suspicious),
        },
        {
            "metric": "suspicious_fraction",
            "value": len(suspicious) / max(len(wide), 1),
        },
        {
            "metric": "num_low_region_attention_cases",
            "value": int(wide["low_region_attention"].sum()),
        },
        {
            "metric": "low_region_attention_fraction",
            "value": float(wide["low_region_attention"].mean()),
        },
        {
            "metric": "num_high_border_attention_cases",
            "value": int(wide["high_border_attention"].sum()),
        },
        {
            "metric": "high_border_attention_fraction",
            "value": float(wide["high_border_attention"].mean()),
        },
        {
            "metric": "suspicion_quantile",
            "value": args.suspicion_quantile,
        },
        {
            "metric": "border_quantile",
            "value": args.border_quantile,
        },
        {
            "metric": "min_low_methods",
            "value": args.min_low_methods,
        },
    ]

    for key, value in thresholds.items():
        suspicious_summary_rows.append({
            "metric": key,
            "value": value,
        })

    for method in args.diagnostic_methods:
        suspicious_summary_rows.append({
            "metric": f"{method}_mean_inside_ratio_all",
            "value": wide[f"{method}_inside_ratio"].mean(),
        })

        suspicious_summary_rows.append({
            "metric": f"{method}_mean_topk_inside_fraction_all",
            "value": wide[f"{method}_topk_inside_fraction"].mean(),
        })

        if len(suspicious) > 0:
            suspicious_summary_rows.append({
                "metric": f"{method}_mean_inside_ratio_suspicious",
                "value": suspicious[f"{method}_inside_ratio"].mean(),
            })

            suspicious_summary_rows.append({
                "metric": f"{method}_mean_topk_inside_fraction_suspicious",
                "value": suspicious[f"{method}_topk_inside_fraction"].mean(),
            })

    if "border_attention_score" in wide.columns:
        suspicious_summary_rows.append({
            "metric": "border_attention_score_mean_all",
            "value": wide["border_attention_score"].mean(),
        })

        if len(suspicious) > 0:
            suspicious_summary_rows.append({
                "metric": "border_attention_score_mean_suspicious",
                "value": suspicious["border_attention_score"].mean(),
            })

    suspicious_summary = pd.DataFrame(suspicious_summary_rows)
    suspicious_summary_path = output_dir / "suspicious_summary.csv"
    suspicious_summary.to_csv(suspicious_summary_path, index=False)

    reason_summary = (
        wide["suspicion_reason"]
        .value_counts(dropna=False)
        .rename_axis("suspicion_reason")
        .reset_index(name="count")
    )

    reason_summary["fraction"] = reason_summary["count"] / max(len(wide), 1)

    reason_summary_path = output_dir / "suspicion_reason_summary.csv"
    reason_summary.to_csv(reason_summary_path, index=False)

    config = {
        "split_csv": str(args.split_csv),
        "predictions_csv": str(args.predictions_csv),
        "heatmap_dir": str(args.heatmap_dir),
        "output_dir": str(args.output_dir),
        "only_correct": args.only_correct,
        "max_images": args.max_images,
        "num_debug_figures": args.num_debug_figures,
        "save_masks": args.save_masks,
        "diagnostic_methods": args.diagnostic_methods,
        "suspicion_quantile": args.suspicion_quantile,
        "border_quantile": args.border_quantile,
        "min_low_methods": args.min_low_methods,
        "topk_fraction": args.topk_fraction,
        "thresholds": thresholds,
        "num_region_score_rows": int(len(scores_df)),
        "num_cases_evaluated": int(len(wide)),
        "num_suspicious_cases": int(len(suspicious)),
        "suspicious_fraction": float(len(suspicious) / max(len(wide), 1)),
        "missing_images": int(missing_images),
        "missing_heatmaps": int(missing_heatmaps),
    }

    config_path = output_dir / "post_gradcam_config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print()
    print("Saved outputs:")
    print(f"  region scores:             {scores_path}")
    print(f"  method summary:            {method_summary_path}")
    print(f"  all cases with flags:      {all_cases_flags_path}")
    print(f"  suspicious cases:          {suspicious_path}")
    print(f"  suspicious summary:        {suspicious_summary_path}")
    print(f"  suspicion reason summary:  {reason_summary_path}")
    print(f"  config:                    {config_path}")

    print()
    print("Suspicion thresholds:")
    for key, value in thresholds.items():
        print(f"  {key}: {value}")

    print()
    print(f"Cases evaluated:  {len(wide)}")
    print(f"Suspicious cases: {len(suspicious)}")
    print(f"Suspicious share: {len(suspicious) / max(len(wide), 1):.3f}")

    print()
    print("Suspicion reason summary:")
    print(reason_summary.to_string(index=False))

    if len(suspicious) > 0:
        preview_cols = [
            "image_path",
            "suspicion_reason",
            "num_low_methods",
            "mean_inside_ratio_selected",
            "min_inside_ratio_selected",
            "border_attention_score",
        ]

        for method in args.diagnostic_methods:
            preview_cols.extend([
                f"{method}_inside_ratio",
                f"{method}_topk_inside_fraction",
                f"low_{method}",
            ])

        preview_cols = [c for c in preview_cols if c in suspicious.columns]

        print()
        print("Top suspicious cases:")
        print(suspicious[preview_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()