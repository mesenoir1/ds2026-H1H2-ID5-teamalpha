#!/usr/bin/env python

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from roi_dynamic import get_roi_mask


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-8


def resolve_path(path_str: str) -> Path:
    path = Path(str(path_str))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_heatmap(path: Path) -> np.ndarray:
    heatmap = np.load(path)
    heatmap = np.squeeze(heatmap).astype(np.float32)

    if heatmap.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got {heatmap.shape}: {path}")

    heatmap = np.maximum(heatmap, 0)

    min_val = float(heatmap.min())
    max_val = float(heatmap.max())

    if max_val - min_val < EPS:
        return np.zeros_like(heatmap, dtype=np.float32)

    return (heatmap - min_val) / (max_val - min_val)


def resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    h, w = target_shape

    if mask.shape == target_shape:
        return (mask > 0).astype(np.uint8)

    mask = cv2.resize(
        mask.astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )

    return (mask > 0).astype(np.uint8)


def border_mask(h: int, w: int, border_frac: float = 0.08) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)

    by = max(1, int(h * border_frac))
    bx = max(1, int(w * border_frac))

    mask[:by, :] = 1
    mask[-by:, :] = 1
    mask[:, :bx] = 1
    mask[:, -bx:] = 1

    return mask


def attention_inside_mask(heatmap: np.ndarray, mask: np.ndarray) -> float:
    mask = resize_mask(mask, heatmap.shape).astype(np.float32)

    total_attention = float(heatmap.sum()) + EPS
    inside_attention = float((heatmap * mask).sum())

    return inside_attention / total_attention


def make_overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image_float = image.astype(np.float32)

    if image_float.max() > 1:
        image_float = image_float / 255.0

    image_rgb = np.stack([image_float, image_float, image_float], axis=-1)
    colored_heatmap = plt.get_cmap("jet")(heatmap)[..., :3]

    overlay = (1 - alpha) * image_rgb + alpha * colored_heatmap
    return np.clip(overlay, 0, 1)


def save_debug_figure(
    output_path: Path,
    image: np.ndarray,
    heatmap: np.ndarray,
    dynamic_roi: np.ndarray,
    border: np.ndarray,
    row: pd.Series,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overlay = make_overlay(image, heatmap)

    title = (
        f"true={row.get('true_label')} pred={row.get('pred_label')} "
        f"correct={row.get('correct')} suspicious={row.get('suspicious_case')}\n"
        f"ROI={row.get('dynamic_roi_inside_ratio'):.3f}, "
        f"Border={row.get('border_attention_score'):.3f}, "
        f"Reason={row.get('suspicion_reason')}"
    )

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(overlay)
    plt.title("Grad-CAM overlay")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(image, cmap="gray")
    plt.imshow(dynamic_roi, cmap="Reds", alpha=0.35)
    plt.title("Dynamic ROI")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(image, cmap="gray")
    plt.imshow(border, cmap="Reds", alpha=0.35)
    plt.title("Border mask")
    plt.axis("off")

    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def compute_case_metrics(row: pd.Series, border_frac: float) -> dict:
    image_path = resolve_path(row["image_path"])
    heatmap_path = resolve_path(row["heatmap_path"])

    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    if not heatmap_path.exists():
        raise FileNotFoundError(f"Missing heatmap: {heatmap_path}")

    image_gray = np.asarray(Image.open(image_path).convert("L"))
    heatmap = load_heatmap(heatmap_path)

    dynamic_roi = get_roi_mask(image_gray)
    dynamic_roi = resize_mask(dynamic_roi, heatmap.shape)

    h, w = heatmap.shape
    border = border_mask(h, w, border_frac=border_frac)

    dynamic_roi_inside_ratio = attention_inside_mask(heatmap, dynamic_roi)
    border_attention_score = attention_inside_mask(heatmap, border)

    return {
        "dynamic_roi_inside_ratio": dynamic_roi_inside_ratio,
        "border_attention_score": border_attention_score,
        "dynamic_roi_area_ratio": float(dynamic_roi.mean()),
        "border_area_ratio": float(border.mean()),
    }


def suspicion_reason(row: pd.Series) -> str:
    reasons = []

    if row["low_dynamic_roi_attention"]:
        reasons.append("low_dynamic_roi_attention")

    if row["high_border_attention"]:
        reasons.append("high_border_attention")

    if not reasons:
        return "not_suspicious"

    return ";".join(reasons)


def safe_filename(image_path: str, idx: int) -> str:
    path = Path(str(image_path))
    parent = path.parent.name
    stem = path.stem
    return f"{idx:05d}_{parent}_{stem}.png"


def save_debug_group(
    df: pd.DataFrame,
    output_dir: Path,
    group_name: str,
    max_figures: int,
    border_frac: float,
) -> None:
    group_dir = output_dir / "debug_figures" / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    subset = df.head(max_figures).copy()

    for local_idx, (_, row) in enumerate(subset.iterrows()):
        try:
            image_path = resolve_path(row["image_path"])
            heatmap_path = resolve_path(row["heatmap_path"])

            image_gray = np.asarray(Image.open(image_path).convert("L"))
            heatmap = load_heatmap(heatmap_path)

            dynamic_roi = get_roi_mask(image_gray)
            dynamic_roi = resize_mask(dynamic_roi, heatmap.shape)

            h, w = heatmap.shape
            border = border_mask(h, w, border_frac=border_frac)

            filename = safe_filename(row["image_path"], local_idx)

            save_debug_figure(
                output_path=group_dir / filename,
                image=image_gray,
                heatmap=heatmap,
                dynamic_roi=dynamic_roi,
                border=border,
                row=row,
            )

        except Exception as exc:
            print(f"[WARN] Failed debug figure for {row.get('image_path')}: {exc}")


def save_all_debug_figures(
    df: pd.DataFrame,
    output_dir: Path,
    max_figures: int,
    border_frac: float,
    random_seed: int,
) -> None:
    if max_figures <= 0:
        return

    suspicious = (
        df[df["suspicious_case"]]
        .sort_values(
            by=["dynamic_roi_inside_ratio", "border_attention_score"],
            ascending=[True, False],
        )
    )

    low_roi = (
        df[df["low_dynamic_roi_attention"]]
        .sort_values(by="dynamic_roi_inside_ratio", ascending=True)
    )

    high_border = (
        df[df["high_border_attention"]]
        .sort_values(by="border_attention_score", ascending=False)
    )

    non_suspicious_correct = (
        df[(df["correct_bool"]) & (~df["suspicious_case"])]
        .sample(
            n=min(max_figures, len(df[(df["correct_bool"]) & (~df["suspicious_case"])])),
            random_state=random_seed,
        )
        if len(df[(df["correct_bool"]) & (~df["suspicious_case"])]) > 0
        else pd.DataFrame()
    )

    if "likely_inverted" in df.columns:
        likely_inverted = df[df["likely_inverted"].fillna(False).astype(bool)].head(max_figures)
    else:
        likely_inverted = pd.DataFrame()

    save_debug_group(
        suspicious,
        output_dir,
        "suspicious",
        max_figures,
        border_frac,
    )

    save_debug_group(
        low_roi,
        output_dir,
        "low_roi",
        max_figures,
        border_frac,
    )

    save_debug_group(
        high_border,
        output_dir,
        "high_border",
        max_figures,
        border_frac,
    )

    save_debug_group(
        non_suspicious_correct,
        output_dir,
        "non_suspicious_correct",
        max_figures,
        border_frac,
    )

    save_debug_group(
        likely_inverted,
        output_dir,
        "likely_inverted",
        max_figures,
        border_frac,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gradcam-csv",
        required=True,
        help="Path to gradcam_predictions.csv",
    )

    parser.add_argument(
        "--inversion-qc-csv",
        default="outputs/data_quality/inversion_qc.csv",
        help="Path to inversion_qc.csv",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for suspicious-case CSVs",
    )

    parser.add_argument(
        "--roi-threshold",
        type=float,
        default=0.40,
        help="Flag correct cases with dynamic ROI attention below this value",
    )

    parser.add_argument(
        "--border-threshold",
        type=float,
        default=0.20,
        help="Flag correct cases with border attention above this value",
    )

    parser.add_argument(
        "--border-frac",
        type=float,
        default=0.08,
        help="Border width as fraction of image height/width",
    )

    parser.add_argument(
        "--num-debug-figures",
        type=int,
        default=25,
        help="Number of debug figures per debug category",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for non-suspicious debug samples",
    )

    args = parser.parse_args()

    gradcam_csv = resolve_path(args.gradcam_csv)
    inversion_qc_csv = resolve_path(args.inversion_qc_csv)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(gradcam_csv)

    required_cols = {
        "image_path",
        "true_label",
        "pred_label",
        "correct",
        "heatmap_path",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in Grad-CAM CSV: {sorted(missing)}")

    if inversion_qc_csv.exists():
        inv_df = pd.read_csv(inversion_qc_csv)

        inversion_cols = [
            "image_path",
            "inversion_score",
            "border_mean",
            "center_mean",
            "border_median",
            "center_median",
            "border_minus_center_mean",
            "border_minus_center_median",
            "center_minus_border_mean",
            "likely_inverted",
        ]

        inversion_cols = [col for col in inversion_cols if col in inv_df.columns]

        df = df.merge(
            inv_df[inversion_cols],
            on="image_path",
            how="left",
        )
    else:
        print(f"[WARN] Inversion QC file not found: {inversion_qc_csv}")

    metric_rows = []

    for idx, row in df.iterrows():
        try:
            metric_rows.append(
                compute_case_metrics(
                    row=row,
                    border_frac=args.border_frac,
                )
            )
        except Exception as exc:
            print(f"[WARN] Failed row {idx}, image={row.get('image_path')}: {exc}")
            metric_rows.append(
                {
                    "dynamic_roi_inside_ratio": np.nan,
                    "border_attention_score": np.nan,
                    "dynamic_roi_area_ratio": np.nan,
                    "border_area_ratio": np.nan,
                }
            )

    metrics_df = pd.DataFrame(metric_rows)
    df = pd.concat([df.reset_index(drop=True), metrics_df], axis=1)

    df["correct_bool"] = df["correct"].astype(int) == 1

    df["low_dynamic_roi_attention"] = (
        df["correct_bool"]
        & (df["dynamic_roi_inside_ratio"] < args.roi_threshold)
    )

    df["high_border_attention"] = (
        df["correct_bool"]
        & (df["border_attention_score"] > args.border_threshold)
    )

    df["suspicious_case"] = (
        df["low_dynamic_roi_attention"]
        | df["high_border_attention"]
    )

    df["suspicion_reason"] = df.apply(suspicion_reason, axis=1)

    suspicious_df = df[df["suspicious_case"]].copy()
    correct_df = df[df["correct_bool"]].copy()

    summary = {
        "n_total": int(len(df)),
        "n_correct": int(df["correct_bool"].sum()),
        "n_suspicious": int(df["suspicious_case"].sum()),
        "suspicious_fraction_total": float(df["suspicious_case"].mean()),
        "suspicious_fraction_correct": float(
            correct_df["suspicious_case"].mean()
            if len(correct_df) > 0
            else np.nan
        ),
        "roi_threshold": float(args.roi_threshold),
        "border_threshold": float(args.border_threshold),
        "mean_dynamic_roi_inside_ratio_correct": float(
            correct_df["dynamic_roi_inside_ratio"].mean()
        ),
        "mean_border_attention_score_correct": float(
            correct_df["border_attention_score"].mean()
        ),
        "mean_dynamic_roi_area_ratio_correct": float(
            correct_df["dynamic_roi_area_ratio"].mean()
        ),
        "mean_border_area_ratio_correct": float(
            correct_df["border_area_ratio"].mean()
        ),
        "num_debug_figures_per_group": int(args.num_debug_figures),
    }

    summary_df = pd.DataFrame([summary])

    reason_summary = (
        df[df["suspicious_case"]]
        .groupby("suspicion_reason")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    class_summary = (
        correct_df
        .groupby("true_label")
        .agg(
            n_correct=("correct_bool", "size"),
            n_suspicious=("suspicious_case", "sum"),
            mean_dynamic_roi_inside_ratio=("dynamic_roi_inside_ratio", "mean"),
            mean_border_attention_score=("border_attention_score", "mean"),
            mean_dynamic_roi_area_ratio=("dynamic_roi_area_ratio", "mean"),
        )
        .reset_index()
    )

    class_summary["suspicious_fraction_correct"] = (
        class_summary["n_suspicious"] / class_summary["n_correct"]
    )

    df.to_csv(output_dir / "all_cases_dynamic_roi_scores.csv", index=False)
    suspicious_df.to_csv(output_dir / "suspicious_cases.csv", index=False)
    summary_df.to_csv(output_dir / "suspicious_summary.csv", index=False)
    reason_summary.to_csv(output_dir / "suspicion_reason_summary.csv", index=False)
    class_summary.to_csv(output_dir / "classwise_suspicious_summary.csv", index=False)

    save_all_debug_figures(
        df=df,
        output_dir=output_dir,
        max_figures=args.num_debug_figures,
        border_frac=args.border_frac,
        random_seed=args.random_seed,
    )

    print(f"Saved outputs to: {output_dir}")
    print(summary_df.to_string(index=False))
    print(f"Saved debug figures to: {output_dir / 'debug_figures'}")


if __name__ == "__main__":
    main()

